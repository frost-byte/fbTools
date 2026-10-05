"""Reference Bundle REST routes (/fbtools/bundles/*).

Moved out of extension.py (Plan 29, pure code motion). _bundle_proxy_eligible/
_BUNDLE_PROXY_SHORT_EDGE and _read_composition_settings/_h3_load_audio come from
nodes/composition_shared.py -- this file and nodes/compositions.py each need something the
other owns (compositions.py's _resolve_cast_media needs _bundle_proxy_eligible; this file's
preprocess_audio route needs _read_composition_settings and _h3_load_audio), so neither is a
leaf relative to the other. Two function-scope local imports had the wrong dot-depth for this
file's new location (extension.py sat at the repo root; this file is one level deeper) --
fixed here, the same class of bug Plan 28 caught via test_relative_imports_in_nodes_modules_resolve.
"""
from __future__ import annotations

import asyncio
import os

from aiohttp import web
import folder_paths
from folder_paths import get_input_directory, get_output_directory
from comfy_api.latest import io

from .shared import routes, default_bundle_registry_path, user_data_dir, send_status_update, prefixed_node_id
from .composition_shared import _bundle_proxy_eligible, _BUNDLE_PROXY_SHORT_EDGE, _read_composition_settings, _h3_load_audio
from ..utils.reference_bundles import (
    load_registry as _load_bundle_registry,
    save_registry as _save_bundle_registry,
    resolve_bundle_audio_source as _resolve_bundle_audio_source,
    bundle_audio_switch_select as _bundle_audio_switch_select,
)
from ..utils.proxy_cache import ensure_bundle_video_proxy as _ensure_bundle_proxy
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


# ── Reference Bundle routes ───────────────────────────────────────────────────

@routes.get("/fbtools/bundles/list")
async def _bundles_list(request):
    """Return all bundles, optionally filtered by ?subject_id=."""
    try:
        subject_id = request.rel_url.query.get("subject_id") or None
        registry = _load_bundle_registry(default_bundle_registry_path())
        bundles = registry.list_bundles(subject_id=subject_id)
        return web.json_response({"bundles": bundles})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.get("/fbtools/bundles/get")
async def _bundles_get(request):
    """Return a single bundle by ?id=<bundle_id>."""
    bundle_id = request.rel_url.query.get("id", "")
    if not bundle_id:
        return web.json_response({"error": "id parameter required"}, status=400)
    try:
        registry = _load_bundle_registry(default_bundle_registry_path())
        bundle = registry.get(bundle_id)
        if bundle is None:
            return web.json_response({"error": f"Bundle '{bundle_id}' not found"}, status=404)
        return web.json_response(bundle)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


# ── Bundle video proxy cache (Plan 16) ────────────────────────────────────────
# Extends the Source Profile clip proxy system (utils/proxy_cache.py) to reference bundles'
# video reference, so generation never has to re-seek/re-decode the original source file. Three
# trigger points share this eligibility check + fire-and-forget builder: Preview (this file,
# preview_sampled below), Save (_bundles_save above... below), and the generation-time fallback
# inside _resolve_cast_media.

# Bundle proxies bake in 24fps unconditionally, same as Source Profile clips' hardcoded
# "force_rate": 24 (SourceProfileClipPrompt, "H3 requires 24fps reference video") — but a
# bundle's force_rate, unlike a clip's, is a genuinely per-bundle configurable field. Swapping in
# a 24fps-baked proxy is only correct when the bundle also targets 24fps; otherwise
# _h3_load_video_frames would resample an already-24fps file against the wrong target, so this
# guard must be checked at every trigger point before ever calling _ensure_bundle_proxy.
# _BUNDLE_PROXY_SHORT_EDGE / _bundle_proxy_eligible moved to nodes/composition_shared.py (Plan 29)


def _fire_bundle_proxy_build(bundle_id: str, abs_path: str, start_time: float, duration: float,
                              force_rate) -> None:
    """Fire-and-forget: build (or refresh) a bundle's video proxy in a background thread.

    Never awaited by the caller — Preview/Save must stay fast, and a stale/missing proxy is never
    a hard failure (the generation-time fallback in _resolve_cast_media builds one synchronously,
    on the spot, if this hasn't finished or was never triggered). Broadcasts over the same
    fbtools.status/source="proxy_build" channel prebuild_proxies already uses, so
    js/ui/bundle_editor.js's freshness readout picks it up with the same listener pattern already
    used for Source Profile clips (source_profile_editor.js) and SceneCastBuild's clip preview.
    """
    if not bundle_id or not abs_path or not _bundle_proxy_eligible(force_rate, duration):
        return

    def _build():
        label = os.path.basename(abs_path)
        send_status_update("proxy_build", f"Building bundle proxy: {label}", source="proxy_build")
        try:
            result = _ensure_bundle_proxy(
                source_path=abs_path,
                bundle_id=bundle_id,
                start_time=start_time,
                end_time=start_time + duration,
                short_edge=_BUNDLE_PROXY_SHORT_EDGE,
                base_dir=str(user_data_dir()),
            )
            status = "ready" if result else "failed"
        except Exception as exc:
            status = f"error: {exc}"
            logger.warning("bundle proxy build failed for %r (%s): %s", bundle_id, label, exc)
        send_status_update("proxy_build", f"Bundle proxy build complete: {label} ({status})",
                            source="proxy_build")

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _build)


@routes.post("/fbtools/bundles/save")
async def _bundles_save(request):
    """Create or update a bundle.  Body: full bundle dict with 'id'."""
    try:
        data = await request.json()
        bundle_id = (data.get("id") or "").strip()
        if not bundle_id:
            return web.json_response({"error": "Bundle 'id' is required"}, status=400)
        path = default_bundle_registry_path()
        registry = _load_bundle_registry(path)
        registry = registry.upsert(data)
        _save_bundle_registry(registry, path)

        visual = data.get("visual") or {}
        vfile = visual.get("file", "")
        if vfile:
            vdir = visual.get("video_dir", "input")
            abs_vfile = os.path.join(
                get_output_directory() if vdir == "output" else get_input_directory(), vfile,
            )
            _fire_bundle_proxy_build(
                bundle_id, abs_vfile,
                float(visual.get("start_time", 0.0)), float(visual.get("duration", 0.0)),
                visual.get("force_rate", 24),
            )

        return web.json_response({"success": True, "id": bundle_id})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.delete("/fbtools/bundles/delete")
async def _bundles_delete(request):
    """Delete a bundle by ?id=<bundle_id>."""
    bundle_id = request.rel_url.query.get("id", "")
    if not bundle_id:
        return web.json_response({"error": "id parameter required"}, status=400)
    try:
        path = default_bundle_registry_path()
        registry = _load_bundle_registry(path)
        if registry.get(bundle_id) is None:
            return web.json_response({"error": f"Bundle '{bundle_id}' not found"}, status=404)
        registry = registry.delete(bundle_id)
        _save_bundle_registry(registry, path)
        return web.json_response({"success": True})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


# ── Scene Cast routes ─────────────────────────────────────────────────────────









# ── Media file listing ────────────────────────────────────────────────────────

















@routes.post("/fbtools/bundles/preview_sampled")
async def _bundles_preview_sampled(request):
    """Extract sampled video frames and encode as a fragmented MP4 for browser preview.

    Body JSON: {filename, start_time, duration, force_rate, select_every_nth}
    Response: video/mp4 (fragmented) or {"error": "ffmpeg_unavailable"} 503.
    Applies the same time-accumulator resampling as CompositionToH3Conditioning so the
    user sees exactly the frames the model will receive.
    """
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON"}, status=400)

    filename = (data.get("filename") or "").strip()
    if not filename:
        return web.json_response({"error": "filename required"}, status=400)

    src_dir   = (data.get("dir") or "input").strip()
    input_dir = get_output_directory() if src_dir == "output" else get_input_directory()
    path = os.path.realpath(os.path.join(input_dir, filename))
    if not path.startswith(os.path.realpath(input_dir)):
        return web.json_response({"error": "Forbidden"}, status=403)
    if not os.path.isfile(path):
        return web.json_response({"error": f"Not found: {filename}"}, status=404)

    start_time       = float(data.get("start_time",        0.0))
    duration         = float(data.get("duration",          0.0))
    force_rate       = int(data.get("force_rate",          24))  # H3 requires 24fps
    select_every_nth = max(1, int(data.get("select_every_nth", 1)))

    # Plan 16: Preview already pays for equivalent ffmpeg work against these exact settings, so
    # this is the natural moment to also (re)build the bundle's cached generation-time proxy —
    # fire-and-forget, its own independent ffmpeg run, never blocks or affects this response.
    bundle_id = (data.get("bundle_id") or "").strip()
    if bundle_id:
        _fire_bundle_proxy_build(bundle_id, path, start_time, duration, force_rate)

    try:
        import cv2  # noqa: F401
    except ImportError:
        return web.json_response({"error": "ffmpeg_unavailable"}, status=503)

    import shutil as _shutil
    if not _shutil.which("ffmpeg"):
        return web.json_response({"error": "ffmpeg_unavailable"}, status=503)

    def _extract_and_encode():
        import cv2
        import subprocess

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None

        native_fps        = cap.get(cv2.CAP_PROP_FPS) or 24.0
        target_fps        = float(force_rate) if force_rate > 0 else native_fps
        width             = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        base_frame_time   = 1.0 / native_fps
        target_frame_time = 1.0 / target_fps

        if start_time > 0.0:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000.0)

        # Cap at 120 output frames to keep the preview response small
        PREVIEW_CAP = 120
        frame_cap = PREVIEW_CAP
        if duration > 0.0:
            output_approx = max(1, int(duration * target_fps / select_every_nth))
            frame_cap = min(frame_cap, output_approx)

        frames_raw: list = []
        time_offset = target_frame_time   # mirrors VHS cv_frame_generator init
        evaluated   = -1
        sampled     = 0

        ret, current_bgr = cap.read()
        if not ret:
            cap.release()
            return None

        while cap.isOpened():
            if time_offset < target_frame_time:
                ret, bgr = cap.read()
                if not ret:
                    break
                current_bgr = bgr
                time_offset += base_frame_time
            if time_offset < target_frame_time:
                continue
            time_offset -= target_frame_time
            evaluated += 1
            if evaluated % select_every_nth != 0:
                continue
            frames_raw.append(current_bgr.tobytes())
            sampled += 1
            if sampled >= frame_cap:
                break

        cap.release()
        if not frames_raw:
            return None

        effective_fps = target_fps / select_every_nth
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "bgr24",
            "-r", str(effective_fps),
            "-i", "pipe:0",
            "-c:v", "libx264", "-crf", "23", "-preset", "ultrafast",
            "-movflags", "frag_keyframe+empty_moov+faststart",
            "-f", "mp4", "pipe:1",
        ]
        proc = subprocess.run(
            cmd,
            input=b"".join(frames_raw),
            capture_output=True,
            timeout=60,
        )
        if proc.returncode != 0:
            logger.warning(
                "preview_sampled: ffmpeg error: %s",
                proc.stderr.decode(errors="replace")[:500],
            )
            return None
        return proc.stdout

    loop = asyncio.get_event_loop()
    mp4_bytes = await loop.run_in_executor(None, _extract_and_encode)

    if mp4_bytes is None:
        return web.json_response({"error": "extraction failed"}, status=500)

    return web.Response(
        body=mp4_bytes,
        content_type="video/mp4",
        headers={"Content-Disposition": "inline"},
    )


@routes.get("/fbtools/bundles/proxy_status")
async def _bundles_proxy_status(request: web.Request) -> web.Response:
    """Return video-proxy freshness for one bundle (Plan 16).

    Query params:
        bundle_id  str

    Returns: {"fresh": bool, "proxy_path": str|null, "eligible": bool} — "eligible" is false (and
    "fresh" always false) when the bundle has no video reference, no set duration, or a
    force_rate other than 24/0, mirroring _bundle_proxy_eligible()'s own check server-side.
    """
    bundle_id = request.rel_url.query.get("bundle_id", "").strip()
    if not bundle_id:
        return web.json_response({"error": "bundle_id is required"}, status=400)

    try:
        registry = _load_bundle_registry(default_bundle_registry_path())
        bundle   = registry.get(bundle_id)
        if not bundle:
            return web.json_response({"error": f"Bundle '{bundle_id}' not found"}, status=404)

        visual     = bundle.get("visual", {})
        vfile      = visual.get("file", "")
        start_time = float(visual.get("start_time", 0.0))
        duration   = float(visual.get("duration", 0.0))
        force_rate = visual.get("force_rate", 24)
        eligible   = bool(vfile) and _bundle_proxy_eligible(force_rate, duration)
        if not eligible:
            return web.json_response({"fresh": False, "proxy_path": None, "eligible": False})

        vdir      = visual.get("video_dir", "input")
        video_abs = os.path.join(
            get_output_directory() if vdir == "output" else get_input_directory(), vfile,
        )

        from ..utils.proxy_cache import _proxy_dir, _proxy_stem, _is_fresh
        stem       = _proxy_stem(bundle_id, "video", start_time, start_time + duration, _BUNDLE_PROXY_SHORT_EDGE)
        proxy_path = _proxy_dir(str(user_data_dir()), "bundles") / f"{stem}.mp4"
        fresh      = _is_fresh(proxy_path, video_abs) if os.path.exists(video_abs) else False

        return web.json_response({
            "fresh":      fresh,
            "proxy_path": str(proxy_path) if fresh else None,
            "eligible":   True,
        })
    except Exception as exc:
        logger.exception("bundle proxy_status failed for %r", bundle_id)
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/bundles/preprocess_audio")
async def _bundles_preprocess_audio(request):
    """Apply the audio preprocessing pipeline to a bundle's audio source and cache the result.

    Body JSON:
        bundle_id       str      — used for the cache directory name
        filename        str      — audio/video filename
        dir             str      — "input" or "output" (default "input")
        start_time      float    — trim start (seconds); ignored if audio_source == "extract_from_visual"
        duration        float    — trim duration; 0 = to end
        audio_processing dict:
            noise_removal     bool   — spectral denoising (scipy)
            normalize_lufs    bool   — LUFS normalize (pyloudnorm)
            target_lufs       float  — target integrated loudness (default -14)

    Response: {cache_path, duration, lufs_before, lufs_after, fingerprint}
    """
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON"}, status=400)

    bundle_id = (data.get("bundle_id") or "default").strip()
    filename  = (data.get("filename") or "").strip()
    if not filename:
        return web.json_response({"error": "filename required"}, status=400)

    dir_hint = (data.get("dir") or "input").strip()
    base_dir = get_output_directory() if dir_hint == "output" else get_input_directory()
    src_path = os.path.realpath(os.path.join(base_dir, filename))
    if not src_path.startswith(os.path.realpath(base_dir)):
        return web.json_response({"error": "Forbidden"}, status=403)
    if not os.path.isfile(src_path):
        return web.json_response({"error": f"Not found: {filename}"}, status=404)

    start_time = float(data.get("start_time", 0.0))
    duration   = float(data.get("duration",   0.0))
    proc_cfg   = data.get("audio_processing", {})
    noise_removal  = bool(proc_cfg.get("noise_removal",  False))
    normalize_lufs = bool(proc_cfg.get("normalize_lufs", True))
    target_lufs    = float(proc_cfg.get("target_lufs",  -14.0))

    # Resolve MelBand Roformer model path from saved settings
    melband_path: str | None = None
    melband_reason = ""
    if noise_removal:
        settings     = _read_composition_settings()
        melband_raw  = settings.get("melband_model_path", "").strip()
        if melband_raw:
            if os.path.isabs(melband_raw) and os.path.isfile(melband_raw):
                melband_path = melband_raw
            else:
                resolved = folder_paths.get_full_path("diffusion_models", melband_raw)
                if resolved and os.path.isfile(resolved):
                    melband_path = resolved
            if not melband_path:
                melband_reason = f"configured MelBand path {melband_raw!r} not found under diffusion_models"
        else:
            melband_reason = "no MelBand model path configured in Settings"

    # denoise_method reports what actually ran, independent of from_cache — a cache
    # hit is only served when this same (noise_removal, melband_path) combination
    # produced it (see cache_fingerprint below), so it's always accurate here too.
    denoise_method = "melband" if melband_path else ("spectral_fallback" if noise_removal else "none")
    logger.info(
        "preprocess_audio: %s — denoise=%s%s, normalize_lufs=%s, target_lufs=%.1f",
        os.path.basename(src_path), denoise_method,
        f" ({melband_reason})" if melband_reason else "",
        normalize_lufs, target_lufs,
    )

    from ..utils.audio_preprocess import preprocess_audio, cache_fingerprint, measure_lufs

    fp = cache_fingerprint(src_path, start_time, duration, {
        "noise_removal":  noise_removal,
        "normalize_lufs": normalize_lufs,
        "target_lufs":    target_lufs,
        "melband_path":   melband_path or "",
    })

    # Cache dir: user_data_dir/bundles_cache/<bundle_id>/
    safe_bid  = "".join(c if c.isalnum() or c in "-_" else "_" for c in bundle_id)[:64]
    cache_dir = os.path.join(user_data_dir(), "bundles_cache", safe_bid)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"audio_{fp}.wav")

    def _remember_cache_path() -> None:
        """Point the persisted bundle at the exact cache variant just selected."""
        if not bundle_id:
            return
        registry_path = default_bundle_registry_path()
        registry = _load_bundle_registry(registry_path)
        bundle = registry.get(bundle_id)
        if bundle is None:
            logger.warning("preprocess_audio: bundle %r was not found; cache path was not persisted", bundle_id)
            return
        bundle.setdefault("audio", {})["audio_cache"] = cache_path
        _save_bundle_registry(registry.upsert(bundle), registry_path)

    if os.path.isfile(cache_path):
        _remember_cache_path()
        # Re-measure from cached file for the response metrics
        def _measure_cached():
            import torchaudio
            wf, sr = torchaudio.load(cache_path)
            import numpy as np
            audio_np = wf.numpy()
            lufs = measure_lufs(audio_np, sr)
            dur = audio_np.shape[-1] / sr if sr > 0 else 0.0
            return {"duration": round(dur, 2), "lufs_before": None,
                    "lufs_after": round(lufs, 1) if np.isfinite(lufs) else None}
        loop = asyncio.get_event_loop()
        metrics = await loop.run_in_executor(None, _measure_cached)
        return web.json_response({
            "cache_path": cache_path,
            "fingerprint": fp,
            "from_cache": True,
            "denoise_method": denoise_method,
            "denoise_reason": melband_reason,
            **metrics,
        })

    def _run_pipeline():
        raw = _h3_load_audio(src_path, start_time, duration)
        if raw is None:
            return None, None
        wf_out, sr_out, metrics = preprocess_audio(
            raw["waveform"], raw["sample_rate"],
            noise_removal=noise_removal,
            normalize_lufs=normalize_lufs,
            target_lufs=target_lufs,
            melband_model_path=melband_path,
        )
        import torchaudio
        torchaudio.save(cache_path, wf_out.squeeze(0), sr_out)
        _remember_cache_path()
        return metrics, cache_path

    loop = asyncio.get_event_loop()
    try:
        metrics, out_path = await loop.run_in_executor(None, _run_pipeline)
    except Exception as exc:
        logger.error("preprocess_audio: pipeline error: %s", exc)
        return web.json_response({"error": str(exc)}, status=500)

    if out_path is None:
        return web.json_response({"error": "audio load failed — check ffmpeg and filename"}, status=500)

    return web.json_response({
        "cache_path":  cache_path,
        "fingerprint": fp,
        "from_cache":  False,
        "denoise_method": denoise_method,
        "denoise_reason": melband_reason,
        **(metrics or {}),
    })


@routes.get("/fbtools/bundles/audio_cache/stream")
async def _bundles_audio_cache_stream(request):
    """Stream a processed audio cache file.

    ?path=<absolute_path>  — must be inside user_data_dir()/bundles_cache/.
    Supports HTTP Range requests so browsers can seek.
    """
    path = request.rel_url.query.get("path", "").strip()
    if not path:
        return web.Response(status=400, text="path required")
    allowed_root = os.path.realpath(os.path.join(user_data_dir(), "bundles_cache"))
    real_path = os.path.realpath(path)
    if not real_path.startswith(allowed_root + os.sep):
        return web.Response(status=403, text="Forbidden")
    if not os.path.isfile(real_path):
        return web.Response(status=404, text="Not found")
    return web.FileResponse(real_path)


# ── Node: BundleAudioReferenceLoad ──────────────────────────────────────────────

def _bundle_get_ids() -> list[str]:
    """Return available bundle IDs for combo population at schema time, always headed by ""
    (no bundle). Unlike SceneCastLoad/SourceProfileLoad/CompositionLoad's own _get_ids()-style
    helpers -- where picking a real entry is mandatory and "(none)" only ever appears as an
    empty-registry fallback message -- this node is deliberately optional (see
    background_override_id's own "" = no override precedent in nodes/scene_casts.py), and
    OpenShot's own bundle picker submits a literal "" for "no bundle selected" (generate.py's
    _populate_bundle_combo). ComfyUI validates a submitted COMBO value by strict membership in
    this list regardless of the input's optional-ness, so "" must always be a real option here
    or leaving the picker unset fails job submission outright -- the same class of bug fixed
    for composition_name's own empty-value case (SceneCastBuild.execute()'s composition
    wiring)."""
    try:
        registry = _load_bundle_registry(default_bundle_registry_path())
        ids = list(registry.bundles.keys())
    except Exception:
        ids = []
    return [""] + ids


class BundleAudioReferenceLoad(io.ComfyNode):
    """Load a Reference Bundle's own configured audio as a standalone AUDIO output.

    Resolves whatever the bundle's audio.source points to (a standalone file, the same
    clip its own visual reference uses, or a separate audio-only reference video) via
    resolve_bundle_audio_source() (utils/reference_bundles.py) and decodes it with the
    same ffmpeg-based loader CompositionToH3Conditioning uses for every other audio
    reference (_h3_load_audio), so behavior matches the Scene Cast system exactly.

    For wiring a Bundle's voice reference directly into any node with an AUDIO input --
    e.g. MiniMaxH3ReferenceToVideo's ref_video_audios slots in a hand-built template
    (such as OpenShot's Bridge Clips workflow, which has no Subject/Bundle concept of
    its own) -- in place of extracting audio from the footage itself.
    """

    node_id = prefixed_node_id("BundleAudioReferenceLoad")
    display_name = "Bundle Audio Reference Load"
    category = "🧊 frost-byte/Scene"

    @classmethod
    def define_schema(cls):
        bundle_ids = _bundle_get_ids()
        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            category=cls.category,
            inputs=[
                io.Combo.Input(
                    "bundle_id",
                    options=bundle_ids,
                    display_name="Bundle",
                    tooltip="Reference bundle whose own audio.source to load. Press R to refresh after saving a new bundle.",
                ),
            ],
            outputs=[
                io.Audio.Output(
                    "audio",
                    display_name="Audio",
                    tooltip="The bundle's resolved audio reference, or None if it has no audio configured (audio.source == 'none') or loading failed.",
                ),
                io.String.Output(
                    "summary",
                    display_name="Summary",
                    tooltip="Human-readable description of what was loaded (or why nothing was).",
                ),
                io.Int.Output(
                    "switch_select",
                    display_name="Switch Select",
                    tooltip=(
                        "1 when the bundle's audio loaded, 2 otherwise. Wire into an ImpactSwitch "
                        "'select' with this node's Audio on input1 and a fallback audio (e.g. the "
                        "footage's own) on input2, so the bundle wins only when it actually "
                        "provides audio."
                    ),
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, bundle_id: str = "", **_):
        path = default_bundle_registry_path()
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0
        return (path, bundle_id, mtime)

    @classmethod
    def execute(cls, bundle_id: str = "") -> io.NodeOutput:
        no_audio_select = _bundle_audio_switch_select(False)

        if not bundle_id or bundle_id == "(none)":
            logger.warning("BundleAudioReferenceLoad: no bundle_id selected")
            return io.NodeOutput(None, "No bundle selected.", no_audio_select)

        registry = _load_bundle_registry(default_bundle_registry_path())
        bundle = registry.get(bundle_id)
        if bundle is None:
            logger.warning("BundleAudioReferenceLoad: bundle %r not found", bundle_id)
            return io.NodeOutput(None, f"Bundle not found: {bundle_id}", no_audio_select)

        resolved = _resolve_bundle_audio_source(bundle)
        if resolved is None:
            source = bundle.get("audio", {}).get("source", "none")
            summary = f"Bundle {bundle_id!r} has no audio reference configured (source={source!r})."
            logger.info("BundleAudioReferenceLoad: %s", summary)
            return io.NodeOutput(None, summary, no_audio_select)

        base_dir = get_output_directory() if resolved["dir"] == "output" else get_input_directory()
        src_path = os.path.join(base_dir, resolved["file"])
        audio = _h3_load_audio(src_path, resolved["start_time"], resolved["duration"])
        if audio is None:
            summary = f"Failed to load audio for bundle {bundle_id!r} from {resolved['file']!r}."
            logger.warning("BundleAudioReferenceLoad: %s", summary)
            return io.NodeOutput(None, summary, no_audio_select)

        dur = audio["waveform"].shape[-1] / max(audio["sample_rate"], 1)
        source = bundle.get("audio", {}).get("source", "")
        summary = f"Loaded {dur:.1f}s audio reference for bundle {bundle_id!r} (source={source})."
        send_status_update(cls.node_id, summary)
        return io.NodeOutput(audio, summary, _bundle_audio_switch_select(True))
