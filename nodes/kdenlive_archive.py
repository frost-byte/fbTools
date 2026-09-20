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
from server import PromptServer

from ..utils.kdenlive_archive import analyze, archive, strip_metadata
from ..utils.logging_utils import get_logger

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
        _send(text, extra={"job_id": job["id"], "progress": ev})

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
            extra={"job_id": job["id"], "finished": True, "state": job["state"]},
        )
    except Exception as exc:
        logger.exception("kdenlive archive job failed")
        job["state"], job["error"] = "error", str(exc)
        _send(f"Archive failed: {exc}", level="error", extra={"job_id": job["id"], "finished": True, "state": "error"})


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
        if any(j["state"] == "running" for j in _JOBS.values()):
            return web.json_response({"error": "An archive job is already running"}, status=409)
        job = {"id": uuid.uuid4().hex[:12], "state": "running", "progress": {}, "report": None,
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


def _public(job: dict) -> dict:
    return {k: job[k] for k in ("id", "state", "progress", "report", "error", "started")}


@routes.get("/fbtools/kdenlive/status")
async def _kdenlive_status(request: web.Request) -> web.Response:
    job_id = request.query.get("job_id", "").strip()
    with _LOCK:
        job = _JOBS.get(job_id) if job_id else (_JOBS[_JOB_ORDER[-1]] if _JOB_ORDER else None)
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
