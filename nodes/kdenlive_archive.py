"""REST routes for the Kdenlive Archive sidebar tab.

All logic lives in utils/kdenlive_archive.py (also used by
scripts/kdenlive_archive.py); this module only validates requests, runs the
long archive job in a background thread, and reports progress over the
"fbtools.status" websocket event (source="kdenlive_archive").
"""
import asyncio
import os
import threading
import time
import uuid

from aiohttp import web
from folder_paths import get_input_directory, get_output_directory
from server import PromptServer

from ..utils.kdenlive_archive import analyze, archive, strip_metadata
from ..utils.kdenlive_clips import clean_folder
from ..utils.generation_metadata import extract_cast_info, read_embedded_prompt
from ..utils.logging_utils import get_logger
from ..utils.prompt_compositions import list_compositions, load_composition
from .shared import user_data_dir

logger = get_logger(__name__)

routes = PromptServer.instance.routes

_SOURCE = "kdenlive_archive"
_JOBS: dict[str, dict] = {}
_JOB_ORDER: list[str] = []
_JOBS_MAX = 5
_LOCK = threading.Lock()


def _send(text: str, level: str = "info", extra: dict | None = None) -> None:
    # Same payload shape as extension.py's send_status_update (not imported: circular).
    try:
        payload = {"node": _SOURCE, "status": text, "level": level, "source": _SOURCE}
        if extra:
            payload.update(extra)
        PromptServer.instance.send_sync("fbtools.status", payload)
    except Exception as exc:
        logger.debug("kdenlive_archive: status send failed: %s", exc)


def _lines(value) -> list[str]:
    if isinstance(value, str):
        value = value.splitlines()
    return [str(v).strip() for v in (value or []) if str(v).strip()]


def _parse(body: dict, need_dest: bool = False):
    project = str(body.get("project", "")).strip().strip('"')
    if not project:
        raise ValueError("project is required")
    if not project.lower().endswith(".kdenlive"):
        raise ValueError("project must be a .kdenlive file")
    if not os.path.isfile(project):
        raise FileNotFoundError(f"project not found: {project}")
    opts = {"project": project, "path_maps": _lines(body.get("path_maps")), "search_dirs": _lines(body.get("search_dirs"))}
    for d in opts["search_dirs"]:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"search folder not found: {d}")
    if need_dest:
        dest = str(body.get("dest", "")).strip().strip('"')
        if not dest:
            raise ValueError("destination folder is required")
        opts["dest"] = dest
    return opts


async def _body(request: web.Request) -> dict:
    try:
        body = await request.json()
        return body if isinstance(body, dict) else {}
    except Exception as exc:
        raise ValueError(f"Invalid request body: {exc}")


def _error(exc: Exception) -> web.Response:
    if isinstance(exc, FileNotFoundError):
        return web.json_response({"error": str(exc)}, status=404)
    if isinstance(exc, ValueError):
        return web.json_response({"error": str(exc)}, status=400)
    logger.exception("kdenlive_archive route failed")
    return web.json_response({"error": str(exc)}, status=500)


def _folder_base(folder_param: str) -> str:
    """Resolve the "input"/"output" query param to its absolute directory (same validation as
    nodes/media.py's _media_list, which this deliberately doesn't import — see its own note on
    why Kdenlive's browse endpoints return absolute paths instead of that endpoint's relative ones)."""
    folder_param = (folder_param or "input").lower()
    if folder_param == "output":
        return get_output_directory()
    if folder_param == "input":
        return get_input_directory()
    raise ValueError(f"Invalid folder {folder_param!r}. Use input or output.")


@routes.get("/fbtools/kdenlive/browse_files")
async def _kdenlive_browse_files(request: web.Request) -> web.Response:
    """Every .kdenlive project file under input/ or output/ (recursive), as absolute paths —
    for the Project field's file-tree browser. Kept separate from /fbtools/media/list, whose
    relative paths suit its own callers (ComfyUI node widgets) but not Kdenlive's, which take
    plain OS paths with no notion of ComfyUI's input/output roots."""
    try:
        base = _folder_base(request.rel_url.query.get("folder", "input"))
    except ValueError as exc:
        return _error(exc)
    files: list[str] = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for f in filenames:
            if f.lower().endswith(".kdenlive"):
                files.append(os.path.join(dirpath, f))
    return web.json_response({"files": sorted(files)})


@routes.get("/fbtools/kdenlive/browse_dirs")
async def _kdenlive_browse_dirs(request: web.Request) -> web.Response:
    """Every subdirectory under input/ or output/ (recursive, including empty ones — the gap
    /fbtools/media/list can't fill, since it only knows about directories that contain a
    matching file), as absolute paths, for folder-picking fields (destinations, search folders,
    clean-clips source/destination)."""
    try:
        base = _folder_base(request.rel_url.query.get("folder", "input"))
    except ValueError as exc:
        return _error(exc)
    dirs: list[str] = []
    for dirpath, dirnames, _filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for d in dirnames:
            dirs.append(os.path.join(dirpath, d))
    return web.json_response({"root": base, "dirs": sorted(dirs)})


@routes.post("/fbtools/kdenlive/check")
async def _kdenlive_check(request: web.Request) -> web.Response:
    try:
        opts = _parse(await _body(request))
        report = await asyncio.to_thread(analyze, opts["project"], opts["path_maps"], opts["search_dirs"])
        return web.json_response({"ok": True, "report": report})
    except Exception as exc:
        return _error(exc)


@routes.post("/fbtools/kdenlive/strip")
async def _kdenlive_strip(request: web.Request) -> web.Response:
    try:
        body = await _body(request)
        opts = _parse(body)
        output = opts["project"] if body.get("in_place") else None
        report = await asyncio.to_thread(strip_metadata, opts["project"], output)
        return web.json_response({"ok": True, "report": report})
    except Exception as exc:
        return _error(exc)


def _run_job(job: dict, opts: dict, strip: bool, dry_run: bool) -> None:
    def progress(ev: dict) -> None:
        job["progress"] = ev
        phase = ev.get("phase")
        if phase == "copy":
            text = f"Copying {ev['done']}/{ev['total']}: {ev.get('current', '')}"
        elif phase == "resolve":
            text = f"Resolving references {ev['done']}/{ev['total']}"
        elif phase == "index":
            text = "Indexing search folders"
        else:
            text = "Writing project"
        _send(text, extra={"job_id": job["id"], "kind": job["kind"], "progress": ev})

    try:
        report = archive(
            opts["project"], opts["dest"], path_maps=opts["path_maps"], search_dirs=opts["search_dirs"],
            strip_metadata_opt=strip, dry_run=dry_run, cancel=job["cancel"], progress=progress,
        )
        job["report"] = report
        job["state"] = "cancelled" if report.get("cancelled") else "done"
        missing = report["unresolved_count"]
        _send(
            "Archive cancelled" if report.get("cancelled")
            else f"Archive {'checked' if dry_run else 'complete'}" + (f" ({missing} clip(s) missing)" if missing else ""),
            level="warning" if (missing or report.get("cancelled")) else "info",
            extra={"job_id": job["id"], "kind": job["kind"], "finished": True, "state": job["state"]},
        )
    except Exception as exc:
        logger.exception("kdenlive archive job failed")
        job["state"], job["error"] = "error", str(exc)
        _send(f"Archive failed: {exc}", level="error",
              extra={"job_id": job["id"], "kind": job["kind"], "finished": True, "state": "error"})


@routes.post("/fbtools/kdenlive/archive")
async def _kdenlive_archive(request: web.Request) -> web.Response:
    try:
        body = await _body(request)
        opts = _parse(body, need_dest=True)
        if os.path.realpath(opts["dest"]) == os.path.realpath(os.path.dirname(os.path.abspath(opts["project"]))):
            raise ValueError("destination must not be the project's own folder")
    except Exception as exc:
        return _error(exc)

    with _LOCK:
        if any(j["state"] == "running" and j["kind"] == "archive" for j in _JOBS.values()):
            return web.json_response({"error": "An archive job is already running"}, status=409)
        job = {"id": uuid.uuid4().hex[:12], "kind": "archive", "state": "running", "progress": {}, "report": None,
               "error": None, "started": time.time(), "cancel": threading.Event()}
        _JOBS[job["id"]] = job
        _JOB_ORDER.append(job["id"])
        while len(_JOB_ORDER) > _JOBS_MAX:
            _JOBS.pop(_JOB_ORDER.pop(0), None)

    threading.Thread(
        target=_run_job,
        args=(job, opts, bool(body.get("strip_metadata", True)), bool(body.get("dry_run", False))),
        daemon=True,
    ).start()
    return web.json_response({"started": True, "job_id": job["id"]})


def _load_composition_by_name(name: str):
    matched = next((c for c in list_compositions(user_data_dir()) if c["name"] == name), None)
    return load_composition(user_data_dir(), matched["id"]) if matched else None


def _cast_info_cache():
    """Cache one extract_cast_info() result per source path — dest_subdir() and the post-run
    report enrichment below both need it, and ffprobe is not free to call twice per file."""
    cache: dict[str, dict] = {}

    def get(src_path: str) -> dict:
        if src_path not in cache:
            graph = read_embedded_prompt(src_path)
            cache[src_path] = (
                extract_cast_info(graph, load_composition=_load_composition_by_name) if graph
                else {"tags": [], "primary_subject": None, "composition_name": None,
                      "note": "no embedded generation metadata found on this clip"}
            )
        return cache[src_path]

    return get


def _run_clean_job(job: dict, src_dir: str, dest_dir: str, dry_run: bool, organize_by_primary: bool) -> None:
    def progress(ev: dict) -> None:
        job["progress"] = ev
        _send(f"Cleaning {ev['done']}/{ev['total']}: {ev.get('current', '')}",
              extra={"job_id": job["id"], "kind": job["kind"], "progress": ev})

    get_cast_info = _cast_info_cache()
    dest_subdir = (lambda f, src: get_cast_info(src)["primary_subject"]) if organize_by_primary else None

    try:
        report = clean_folder(src_dir, dest_dir, dry_run=dry_run, cancel=job["cancel"], progress=progress,
                              dest_subdir=dest_subdir)
        if organize_by_primary:
            by_file = {os.path.join(src_dir, r["file"]): r for r in report["results"]}
            for src_path, entry in by_file.items():
                info = get_cast_info(src_path)
                entry["tags"] = info["tags"]
                entry["primary_subject"] = info["primary_subject"]
                if info["note"]:
                    entry["note"] = info["note"]
        job["report"] = report
        job["state"] = "cancelled" if report.get("cancelled") else "done"
        n_err = len(report["errors"])
        _send(
            "Clean cancelled" if report.get("cancelled")
            else f"Clean {'checked' if dry_run else 'complete'}" + (f" ({n_err} error(s))" if n_err else ""),
            level="warning" if (n_err or report.get("cancelled")) else "info",
            extra={"job_id": job["id"], "kind": job["kind"], "finished": True, "state": job["state"]},
        )
    except Exception as exc:
        logger.exception("kdenlive clean job failed")
        job["state"], job["error"] = "error", str(exc)
        _send(f"Clean failed: {exc}", level="error",
              extra={"job_id": job["id"], "kind": job["kind"], "finished": True, "state": "error"})


@routes.post("/fbtools/kdenlive/clean")
async def _kdenlive_clean(request: web.Request) -> web.Response:
    try:
        body = await _body(request)
        src_dir = str(body.get("src_dir", "")).strip().strip('"')
        dest_dir = str(body.get("dest_dir", "")).strip().strip('"')
        if not src_dir:
            raise ValueError("source folder is required")
        if not dest_dir:
            raise ValueError("destination folder is required")
        if not os.path.isdir(src_dir):
            raise FileNotFoundError(f"source folder not found: {src_dir}")
        if os.path.realpath(dest_dir) == os.path.realpath(src_dir):
            raise ValueError("destination must be a different folder from the source")
    except Exception as exc:
        return _error(exc)

    with _LOCK:
        if any(j["state"] == "running" and j["kind"] == "clean" for j in _JOBS.values()):
            return web.json_response({"error": "A clean job is already running"}, status=409)
        job = {"id": uuid.uuid4().hex[:12], "kind": "clean", "state": "running", "progress": {}, "report": None,
               "error": None, "started": time.time(), "cancel": threading.Event()}
        _JOBS[job["id"]] = job
        _JOB_ORDER.append(job["id"])
        while len(_JOB_ORDER) > _JOBS_MAX:
            _JOBS.pop(_JOB_ORDER.pop(0), None)

    threading.Thread(
        target=_run_clean_job,
        args=(job, src_dir, dest_dir, bool(body.get("dry_run", False)), bool(body.get("organize_by_primary", False))),
        daemon=True,
    ).start()
    return web.json_response({"started": True, "job_id": job["id"]})


def _public(job: dict) -> dict:
    return {k: job[k] for k in ("id", "kind", "state", "progress", "report", "error", "started")}


@routes.get("/fbtools/kdenlive/status")
async def _kdenlive_status(request: web.Request) -> web.Response:
    job_id = request.query.get("job_id", "").strip()
    kind = request.query.get("kind", "").strip()
    with _LOCK:
        if job_id:
            job = _JOBS.get(job_id)
        else:
            ids = [i for i in reversed(_JOB_ORDER) if not kind or _JOBS[i]["kind"] == kind]
            job = _JOBS[ids[0]] if ids else None
        if job is None:
            return web.json_response({"job": None})
        return web.json_response({"job": _public(job)})


@routes.post("/fbtools/kdenlive/cancel")
async def _kdenlive_cancel(request: web.Request) -> web.Response:
    try:
        body = await _body(request)
    except ValueError as exc:
        return _error(exc)
    job_id = str(body.get("job_id", "")).strip()
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return web.json_response({"error": "Unknown job"}, status=404)
        job["cancel"].set()
    return web.json_response({"ok": True})
