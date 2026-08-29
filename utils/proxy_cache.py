"""Per-segment video proxy cache.

Source profile clips may come from large videos (100–220 s).  cv2 VideoCapture
seeks by millisecond, but large files still require decoding from the nearest
keyframe, which can be several seconds before the target.  Proxies eliminate
that cost: each segment is trimmed and downscaled once, then reused on every
subsequent run.

Proxy layout under {base_dir}/proxies/source_profiles/:
    {profile_id}__{clip_id}__{start:.2f}-{end:.2f}__h{short_edge}.mp4

A sidecar .json records the source path and mtime so stale proxies are
detected and regenerated when the source file changes.

Invariant: this module has no ComfyUI dependencies.  base_dir is passed in by
the caller (typically user_data_dir() in extension.py).
"""
import json
import os
import subprocess
from pathlib import Path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _proxy_dir(base_dir: str) -> Path:
    d = Path(base_dir) / "proxies" / "source_profiles"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _proxy_stem(
    profile_id: str,
    clip_id: str,
    start: float,
    end: float,
    short_edge: int,
) -> str:
    def _safe(s: str) -> str:
        return s.replace("/", "_").replace("\\", "_").replace(" ", "_")

    return (
        f"{_safe(profile_id)}__{_safe(clip_id)}"
        f"__{start:.2f}-{end:.2f}__h{short_edge}_r32"
    )


def _is_fresh(proxy_path: Path, source_path: str) -> bool:
    """True when the proxy exists and the source file hasn't changed.

    Both the stored source path and the provided source_path are resolved to
    their real paths before comparison so symlink-vs-realpath mismatches
    (common when ComfyUI's output dir is a symlink) don't cause false negatives.
    """
    sidecar = proxy_path.with_suffix(".json")
    if not proxy_path.exists() or not sidecar.exists():
        return False
    try:
        real_source = os.path.realpath(source_path)
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        stored_real = os.path.realpath(meta.get("source", ""))
        return (
            stored_real == real_source
            and abs(meta.get("mtime", -1.0) - os.path.getmtime(source_path)) < 1.0
        )
    except Exception:
        return False


def _write_sidecar(proxy_path: Path, source_path: str) -> None:
    # Store the realpath so sidecars written via different path spellings
    # (symlink vs. resolved) stay comparable.
    real_source = os.path.realpath(source_path)
    sidecar = proxy_path.with_suffix(".json")
    sidecar.write_text(
        json.dumps({"source": real_source, "mtime": os.path.getmtime(source_path)}),
        encoding="utf-8",
    )


def _scale_filter(short_edge: int) -> str:
    """FFmpeg scale filter that sets the shorter dimension to short_edge.

    Both dimensions are rounded to the nearest multiple of 32 so the output
    is compatible with H3 and other models that require 32-divisible dimensions.
    short_edge itself should already be a multiple of 32 (e.g. 480, 768, 1080).
    """
    se = short_edge
    # Commas inside if() must be escaped as \, because ffmpeg's filter-graph
    # parser treats bare commas as filter-chain separators even inside option values.
    w_expr = f"if(gt(iw\\,ih)\\,trunc(iw*{se}/ih/32)*32\\,{se})"
    h_expr = f"if(gt(iw\\,ih)\\,{se}\\,trunc(ih*{se}/iw/32)*32)"
    return f"scale={w_expr}:{h_expr}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ensure_source_profile_proxy(
    source_path: str,
    profile_id: str,
    clip_id: str,
    start_time: float,
    end_time: float,
    short_edge: int,
    base_dir: str,
) -> str | None:
    """Return the path of a trimmed + downscaled proxy for one clip segment.

    Generates the proxy via ffmpeg on first call; subsequent calls with the
    same arguments return the cached path immediately (as long as the source
    file's mtime hasn't changed).

    Returns None if ffmpeg is unavailable, the source is missing, or
    generation fails.  The caller should fall back to the original source.

    Args:
        source_path:  Absolute path to the source video.
        profile_id:   Source profile ID (used to namespace the proxy filename).
        clip_id:      Clip ID within the profile.
        start_time:   Segment start in seconds.
        end_time:     Segment end in seconds.
        short_edge:   Target shorter-dimension pixel count (e.g. 768).
        base_dir:     fbTools user data directory (proxy written under
                      {base_dir}/proxies/source_profiles/).
    """
    if not source_path or not os.path.exists(source_path):
        return None

    duration = max(0.0, end_time - start_time)
    if duration <= 0.0:
        return None

    proxy_dir = _proxy_dir(base_dir)
    stem = _proxy_stem(profile_id, clip_id, start_time, end_time, short_edge)
    proxy_path = proxy_dir / f"{stem}.mp4"

    if _is_fresh(proxy_path, source_path):
        return str(proxy_path)

    # Remove a stale proxy so ffmpeg gets a clean write.
    if proxy_path.exists():
        try:
            proxy_path.unlink()
        except OSError:
            pass

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_time:.3f}",   # fast seek (input side)
        "-t",  f"{duration:.3f}",
        "-i",  source_path,
        "-vf", _scale_filter(short_edge),
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-an",   # no audio needed for H3 reference video
        str(proxy_path),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, timeout=300, check=False
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0 or not proxy_path.exists():
        return None

    _write_sidecar(proxy_path, source_path)
    return str(proxy_path)
