"""Clean generated clips before adding them to a Kdenlive project's media folder.

Pure stdlib (plus an optional imageio_ffmpeg lookup — no other ComfyUI or sibling
utils imports; see docs/GOTCHAS.md). Used by scripts/kdenlive_archive.py (CLI) and
nodes/kdenlive_archive.py (ComfyUI sidebar tab routes). Companion to
utils/kdenlive_archive.py, which handles the project (.kdenlive) file itself —
this module only ever touches plain media files, never a project.

Operation:
  strip_copy_video()  remux one clip without its embedded metadata (a ComfyUI-saved
                       mp4 carries the full workflow/prompt JSON — often hundreds of
                       KB — as container metadata); the source is never modified
  find_duplicate_files()  group files by content, for the caller to review
  clean_folder()      run strip_copy_video() over every video in a folder into a
                       separate destination folder, skipping ones already done
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi", ".wmv"}

# Same tolerance used when eyeballing this by hand: ffmpeg's stream copy can shift
# a duration by a few container-rounding milliseconds; anything past this suggests
# frames were actually lost/re-encoded rather than just remuxed.
_DURATION_TOLERANCE_SEC = 0.2


def find_ffmpeg() -> str | None:
    """Prefer the ffmpeg imageio_ffmpeg ships (bundled with ComfyUI's own deps);
    fall back to one on PATH. None if neither is available."""
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        return get_ffmpeg_exe()
    except Exception:
        pass
    return shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")


def find_ffprobe() -> str | None:
    """ffprobe usually ships alongside a system ffmpeg but not imageio_ffmpeg's
    bundled binary, so this is checked separately and may be unavailable even
    when find_ffmpeg() succeeds."""
    return shutil.which("ffprobe") or shutil.which("ffprobe.exe")


def probe_duration(ffprobe: str, path: str) -> float | None:
    """Container duration in seconds via ffprobe, or None if it can't be read."""
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=30,
        )
        return float(out.stdout.strip())
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def strip_copy_video(src: str, dest: str) -> dict:
    """Remux src to dest without container metadata (workflow/prompt JSON,
    chapters, encoder tags), copying every stream bit-for-bit — never a re-encode.
    Writes atomically (a .part file renamed on success); src is never touched.

    Returns {"src", "dest", "size_before", "size_after"}. Raises RuntimeError if
    ffmpeg is missing or the run fails, and — when ffprobe is available — if the
    output's duration doesn't match the source's (the .part file is removed
    first, so a bad remux never lands at `dest`).
    """
    if not os.path.isfile(src):
        raise RuntimeError(f"source file not found: {src}")
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found (imageio_ffmpeg not installed and none on PATH)")

    # ffmpeg picks its output container from the destination's extension, so the
    # temp name must keep it (a bare "dest.part" suffix leaves ffmpeg unable to
    # guess the format at all).
    root, ext = os.path.splitext(dest)
    tmp = f"{root}.part{ext}"
    os.makedirs(os.path.dirname(os.path.abspath(dest)) or ".", exist_ok=True)
    proc = subprocess.run(
        [ffmpeg, "-nostdin", "-v", "error", "-y", "-i", src,
         "-map", "0", "-c", "copy", "-map_metadata", "-1", "-map_chapters", "-1",
         "-fflags", "+bitexact", "-flags:v", "+bitexact", "-flags:a", "+bitexact",
         "-movflags", "+faststart", tmp],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not os.path.isfile(tmp):
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise RuntimeError(f"ffmpeg failed on {src}: {proc.stderr.strip()[-500:]}")

    ffprobe = find_ffprobe()
    if ffprobe:
        d_before = probe_duration(ffprobe, src)
        d_after = probe_duration(ffprobe, tmp)
        if d_before is not None and d_after is not None and abs(d_before - d_after) > _DURATION_TOLERANCE_SEC:
            os.remove(tmp)
            raise RuntimeError(
                f"duration mismatch after stripping {src}: {d_before:.2f}s -> {d_after:.2f}s "
                f"(> {_DURATION_TOLERANCE_SEC}s tolerance) — output discarded"
            )

    os.replace(tmp, dest)
    st_before = os.path.getsize(src)
    st_after = os.path.getsize(dest)
    shutil.copystat(src, dest, follow_symlinks=True)
    return {"src": src, "dest": dest, "size_before": st_before, "size_after": st_after}


def find_duplicate_files(paths: list[str]) -> list[list[str]]:
    """Group `paths` by content (sha256). Returns only groups of 2+, each sorted,
    largest groups first — informational only, nothing is removed."""
    by_hash: dict[str, list[str]] = {}
    for p in paths:
        h = hashlib.sha256()
        try:
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                    h.update(chunk)
        except OSError:
            continue
        by_hash.setdefault(h.hexdigest(), []).append(p)
    groups = [sorted(g) for g in by_hash.values() if len(g) > 1]
    groups.sort(key=lambda g: (-len(g), g[0]))
    return groups


def clean_folder(
    src_dir: str,
    dest_dir: str,
    *,
    dry_run: bool = False,
    cancel=None,
    progress=None,
    dest_subdir=None,
) -> dict:
    """strip_copy_video() every video file directly under src_dir into dest_dir
    (not recursive — matches a flat "clips to process" staging folder).

    Existing files in dest_dir are left alone (resume behaviour, like the project
    archiver): re-running only cleans what's missing. Duplicate *sources* are
    reported (by content hash) but every one is still cleaned independently —
    nothing is auto-skipped or deleted on your behalf.

    dest_subdir, if given, is called as dest_subdir(filename, src_path) -> str | None for each
    file; a non-None result nests that file's cleaned copy under dest_dir/<result>/ instead of
    dest_dir/ directly (subfolders are created as needed). This module stays free of any notion of
    *why* a file goes in a particular subfolder — see utils/generation_metadata.py for the
    embedded-metadata lookup a caller can use to decide.
    """
    src_dir = os.path.abspath(src_dir)
    dest_dir = os.path.abspath(dest_dir)
    if not os.path.isdir(src_dir):
        raise RuntimeError(f"source folder not found: {src_dir}")
    if os.path.realpath(dest_dir) == os.path.realpath(src_dir):
        raise RuntimeError("destination must be a different folder from the source")

    files = sorted(
        f for f in os.listdir(src_dir)
        if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS
        and os.path.isfile(os.path.join(src_dir, f))
    )
    src_paths = [os.path.join(src_dir, f) for f in files]
    duplicates = find_duplicate_files(src_paths)

    report = {
        "src_dir": src_dir, "dest_dir": dest_dir, "dry_run": dry_run, "cancelled": False,
        "files_total": len(files), "cleaned": 0, "skipped_existing": 0,
        "bytes_before": 0, "bytes_after": 0,
        "duplicates": duplicates, "errors": [], "results": [],
    }
    def _dest_for(f, src):
        sub = dest_subdir(f, src) if dest_subdir else None
        return (os.path.join(dest_dir, sub, f), sub) if sub else (os.path.join(dest_dir, f), None)

    if dry_run or not files:
        for f, p in zip(files, src_paths):
            dest, sub = _dest_for(f, p)
            entry = {"file": f, "status": "exists" if os.path.exists(dest) else "would_clean",
                     "size_before": os.path.getsize(p)}
            if sub:
                entry["subdir"] = sub
            report["results"].append(entry)
        return report

    os.makedirs(dest_dir, exist_ok=True)
    for i, (f, src) in enumerate(zip(files, src_paths), 1):
        if cancel is not None and cancel.is_set():
            report["cancelled"] = True
            break
        dest, sub = _dest_for(f, src)
        if sub:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.isfile(dest):
            report["skipped_existing"] += 1
            entry = {"file": f, "status": "skipped_existing", "size_before": os.path.getsize(src)}
        else:
            try:
                r = strip_copy_video(src, dest)
                report["cleaned"] += 1
                report["bytes_before"] += r["size_before"]
                report["bytes_after"] += r["size_after"]
                entry = {"file": f, "status": "cleaned",
                         "size_before": r["size_before"], "size_after": r["size_after"]}
            except RuntimeError as exc:
                report["errors"].append({"file": f, "error": str(exc)})
                entry = {"file": f, "status": "error", "error": str(exc)}
        if sub:
            entry["subdir"] = sub
        report["results"].append(entry)
        if progress:
            progress({"phase": "clean", "done": i, "total": len(files), "current": f})

    report["bytes_saved"] = report["bytes_before"] - report["bytes_after"]
    return report
