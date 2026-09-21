"""Media listing/streaming/frame-extraction REST routes (/fbtools/media/*) and the media file-extension helpers.

Moved out of extension.py (pure code motion); extension.py re-imports what the remaining nodes use."""
from __future__ import annotations

from folder_paths import get_input_directory, get_output_directory
import os
import time
import asyncio
import uuid
from aiohttp import web
import numpy as np
from .shared import routes
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".aac", ".m4a", ".opus"}


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


_VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".avi", ".mkv"}


def _audio_get_list() -> list[str]:
    """Return audio filenames from the ComfyUI input directory for combo widgets."""
    try:
        input_dir = get_input_directory()
        files = [
            f for f in os.listdir(input_dir)
            if os.path.splitext(f)[1].lower() in _AUDIO_EXTENSIONS
        ]
        files.sort(key=str.lower)
        return ["None"] + files
    except Exception:
        return ["None"]


_TMP_FRAME_PREFIX = "_fbt_tmp_"


def _purge_old_tmp_frames(input_dir: str, max_age_s: int = 1800) -> None:
    """Delete _fbt_tmp_* files older than max_age_s seconds."""
    now = time.time()
    for f in os.listdir(input_dir):
        if f.startswith(_TMP_FRAME_PREFIX):
            fpath = os.path.join(input_dir, f)
            try:
                if now - os.path.getmtime(fpath) > max_age_s:
                    os.remove(fpath)
            except OSError:
                pass


@routes.post("/fbtools/media/extract_frame")
async def _media_extract_frame(request):
    """Extract a single frame from a video in the ComfyUI input directory.

    Body: {filename: str, frame_index: int}
    Response: {tmp_filename: str, frame_count: int, width: int, height: int}
    The caller is responsible for deleting the temp file via DELETE /fbtools/media/extract_frame.
    """
    try:
        body = await request.json()
        filename = body.get("filename", "").strip()
        frame_index = int(body.get("frame_index", 0))
        dir_hint = (body.get("dir") or "input").strip()
        if not filename:
            return web.json_response({"error": "filename is required"}, status=400)

        base_dir = get_output_directory() if dir_hint == "output" else get_input_directory()
        video_path = os.path.join(base_dir, filename)
        if not os.path.exists(video_path):
            return web.json_response({"error": f"File not found: {filename}"}, status=404)

        # Temp frames always land in the input directory so ComfyUI's /view endpoint
        # can serve them directly (it only serves from input/output roots).
        input_dir = get_input_directory()

        def _extract():
            import cv2
            cap = cv2.VideoCapture(video_path)
            try:
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                idx = max(0, min(frame_index, frame_count - 1))
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"Could not read frame {idx}")
                _purge_old_tmp_frames(input_dir)
                tmp_name = f"{_TMP_FRAME_PREFIX}{uuid.uuid4().hex[:12]}.jpg"
                tmp_path = os.path.join(input_dir, tmp_name)
                cv2.imwrite(tmp_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                return {"tmp_filename": tmp_name, "frame_count": frame_count,
                        "width": width, "height": height, "frame_index": idx}
            finally:
                cap.release()

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _extract)
        return web.json_response(result)
    except Exception as exc:
        logger.error("extract_frame error: %s", exc)
        return web.json_response({"error": str(exc)}, status=500)


@routes.delete("/fbtools/media/extract_frame")
async def _media_delete_tmp_frame(request):
    """Delete a temp frame file created by extract_frame. ?filename=<tmp_filename>"""
    fname = request.rel_url.query.get("filename", "").strip()
    if not fname or not fname.startswith(_TMP_FRAME_PREFIX):
        return web.json_response({"error": "Invalid or missing filename"}, status=400)
    try:
        fpath = os.path.join(get_input_directory(), fname)
        if os.path.exists(fpath):
            os.remove(fpath)
        return web.json_response({"success": True})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.get("/fbtools/media/info")
async def _media_info(request):
    """Return metadata for a video or audio file.

    ?filename=<name>&dir=input|output  (dir defaults to "input")
    Response: {duration, fps, frame_count, width, height}
    """
    filename = request.rel_url.query.get("filename", "").strip()
    src_dir  = request.rel_url.query.get("dir", "input")
    if not filename:
        return web.json_response({"error": "filename required"}, status=400)
    base = get_output_directory() if src_dir == "output" else get_input_directory()
    path = os.path.realpath(os.path.join(base, filename))
    if not path.startswith(os.path.realpath(base)):
        return web.json_response({"error": "Forbidden"}, status=403)
    if not os.path.isfile(path):
        return web.json_response({"error": f"Not found: {filename}"}, status=404)

    def _get_info():
        import cv2
        cap = cv2.VideoCapture(path)
        try:
            fps         = cap.get(cv2.CAP_PROP_FPS) or 0.0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration    = (frame_count / fps) if fps > 0 else 0.0
            fourcc_int  = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec       = "".join(chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)).strip("\x00").strip()
            return {"duration": round(duration, 4), "fps": round(fps, 4),
                    "frame_count": frame_count, "width": width, "height": height,
                    "codec": codec or "unknown"}
        finally:
            cap.release()

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _get_info)
    return web.json_response(result)


@routes.get("/fbtools/media/stream")
async def _media_stream(request):
    """Stream a media file from input or output directory.

    ?filename=<name>&dir=input|output  (dir defaults to "input")
    Supports HTTP Range requests so browsers can seek into video/audio.
    """
    filename = request.rel_url.query.get("filename", "").strip()
    src_dir  = request.rel_url.query.get("dir", "input")
    if not filename:
        return web.Response(status=400, text="filename required")
    base = get_output_directory() if src_dir == "output" else get_input_directory()
    path = os.path.realpath(os.path.join(base, filename))
    if not path.startswith(os.path.realpath(base)):
        return web.Response(status=403, text="Forbidden")
    if not os.path.isfile(path):
        return web.Response(status=404, text="Not found")
    return web.FileResponse(path)


@routes.get("/fbtools/media/list")
async def _media_list(request):
    """Return filenames from the ComfyUI input directory filtered by ?type=.

    type: image | video | audio | all  (default: all)
    Response: {"files": ["fname.mp4", ...]}
    """
    try:
        media_type = request.rel_url.query.get("type", "all").lower()
        if media_type == "image":
            exts = _IMAGE_EXTENSIONS
        elif media_type == "video":
            exts = _VIDEO_EXTENSIONS
        elif media_type == "audio":
            exts = _AUDIO_EXTENSIONS
        elif media_type == "all":
            exts = _IMAGE_EXTENSIONS | _VIDEO_EXTENSIONS | _AUDIO_EXTENSIONS
        else:
            return web.json_response(
                {"error": f"Invalid type {media_type!r}. Use image, video, audio, or all."},
                status=400,
            )
        folder_param = request.rel_url.query.get("folder", "input").lower()
        if folder_param == "output":
            base_dir = get_output_directory()
        elif folder_param == "input":
            base_dir = get_input_directory()
        else:
            return web.json_response({"error": f"Invalid folder {folder_param!r}. Use input or output."}, status=400)

        recursive = request.rel_url.query.get("recursive", "false").lower() == "true"

        if recursive:
            files = []
            for dirpath, dirnames, filenames in os.walk(base_dir):
                # skip hidden dirs (e.g. .cache, .tmp)
                dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
                for f in filenames:
                    if os.path.splitext(f)[1].lower() in exts:
                        rel = os.path.relpath(os.path.join(dirpath, f), base_dir)
                        files.append(rel.replace(os.sep, "/"))
        else:
            files = [
                f for f in os.listdir(base_dir)
                if os.path.splitext(f)[1].lower() in exts
            ]

        files.sort(key=str.lower)
        return web.json_response({"files": files})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/media/sample_frames")
async def _media_sample_frames(request):
    """Extract thumbnail frames from a video clip for the filmstrip UI.

    Body: {filename, dir: "input"|"output", start_time, duration, every_nth, cap}
    Returns: {frames: [{index, timestamp, data_url}], fps, duration, frame_count}

    Thumbnails are 160×90 JPEG at quality 72 — small enough for fast transport.
    """
    try:
        body       = await request.json()
        filename   = body.get("filename", "").strip()
        src_dir    = body.get("dir", "input")
        start_time = float(body.get("start_time", 0.0))
        duration   = float(body.get("duration",   0.0))
        every_nth  = max(1, int(body.get("every_nth", 1)))
        cap        = max(1, min(60, int(body.get("cap", 24))))

        if not filename:
            return web.json_response({"error": "filename required"}, status=400)

        base = get_output_directory() if src_dir == "output" else get_input_directory()
        path = os.path.realpath(os.path.join(base, filename))
        if not path.startswith(os.path.realpath(base)):
            return web.json_response({"error": "Forbidden"}, status=403)
        if not os.path.isfile(path):
            return web.json_response({"error": f"Not found: {filename}"}, status=404)

        def _sample():
            import cv2
            import base64 as _b64
            THUMB_W, THUMB_H = 160, 90
            cap_ = cv2.VideoCapture(path)
            try:
                fps   = cap_.get(cv2.CAP_PROP_FPS) or 24.0
                total = int(cap_.get(cv2.CAP_PROP_FRAME_COUNT))
                native_dur = total / fps

                start_f = int(start_time * fps)
                end_f   = min(total, start_f + int(duration * fps)) if duration > 0 else total
                start_f = max(0, min(start_f, end_f - 1))

                # Build candidate list respecting every_nth, then thin to cap
                candidates = list(range(start_f, end_f, every_nth))
                if len(candidates) > cap:
                    step = len(candidates) / cap
                    candidates = [candidates[int(i * step)] for i in range(cap)]

                frames_out = []
                for idx in candidates:
                    cap_.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ok, bgr = cap_.read()
                    if not ok:
                        continue
                    h, w = bgr.shape[:2]
                    scale = min(THUMB_W / w, THUMB_H / h)
                    nw, nh = int(w * scale), int(h * scale)
                    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
                    canvas = np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8)
                    yo, xo = (THUMB_H - nh) // 2, (THUMB_W - nw) // 2
                    canvas[yo:yo + nh, xo:xo + nw] = resized
                    ok2, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 72])
                    if not ok2:
                        continue
                    b64 = _b64.b64encode(buf.tobytes()).decode()
                    frames_out.append({
                        "index":     idx,
                        "timestamp": round(idx / fps, 2),
                        "data_url":  f"data:image/jpeg;base64,{b64}",
                    })

                return {
                    "frames":      frames_out,
                    "fps":         round(fps, 4),
                    "duration":    round(native_dur, 4),
                    "frame_count": total,
                }
            finally:
                cap_.release()

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _sample)
        return web.json_response(result)
    except Exception as exc:
        logger.error("sample_frames error: %s", exc)
        return web.json_response({"error": str(exc)}, status=500)
