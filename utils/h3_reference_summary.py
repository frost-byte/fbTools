"""Pure logic for describing the references handed to MiniMax H3's ReferenceToVideo node.

No ComfyUI/torch dependencies -- the node (nodes/h3_reference_summary.py) reduces tensors and
audio dicts to the plain dicts used here, so this stays testable in CI.

It reproduces how MiniMaxH3ReferenceToVideo (comfy_extras/nodes_minimax_h3.py) numbers and trims
its references, so the summary describes what the model is actually given:

  * pictures are numbered first, in input order  -> <Picture i>
  * each reference video is numbered in input order -> <Video k>; if it has a paired soundtrack
    (ref_video_audio_N belongs to ref_video_N) that soundtrack gets its own <Audio j> label,
    emitted just before its <Video k>
  * standalone reference audios follow, continuing the <Audio j> count
  * a video is truncated to the generation length, then cut down to the nearest valid clip
    length (17k + 5 frames); fewer than 5 frames is an error in the node
"""
from __future__ import annotations

FPS = 24                  # H3 reference/target frame rate
MIN_VIDEO_FRAMES = 5      # below this the node raises
MIN_RECOMMENDED_S = 2.0   # the node's own tooltip: reference videos "2-15s"
MAX_RECOMMENDED_S = 15.0


def valid_clip_frames(frames: int) -> int:
    """Largest valid H3 clip length <= `frames`: the nearest n with n % 17 == 5. Returns 0 when
    `frames` is below the 5-frame minimum (the node raises in that case)."""
    if frames < MIN_VIDEO_FRAMES:
        return 0
    n = int(frames)
    while n % 17 != 5:
        n -= 1
    return n


def _seconds(frames: int) -> float:
    return frames / float(FPS)


def _audio_desc(audio: dict) -> str:
    secs = float(audio.get("seconds", 0.0) or 0.0)
    rate = int(audio.get("sample_rate", 0) or 0)
    ch = int(audio.get("channels", 0) or 0)
    ch_txt = {1: "mono", 2: "stereo"}.get(ch, f"{ch}ch" if ch else "")
    parts = [f"{secs:.2f}s"]
    if rate:
        parts.append(f"{rate} Hz")
    if ch_txt:
        parts.append(ch_txt)
    return ", ".join(parts)


def describe_references(
    images: list[dict] | None = None,
    videos: list[dict] | None = None,
    video_audios: dict | None = None,
    audios: list[dict] | None = None,
    length: int | None = None,
) -> tuple[str, list[str]]:
    """Return (summary_text, warnings).

    images       [{"name": "ref_image_0", "width": int, "height": int}, ...]   in input order
    videos       [{"name": "ref_video_0", "frames": int, "width": int, "height": int}, ...]
    video_audios {"0": {"seconds", "sample_rate", "channels"}, ...}  keyed by the numeric suffix
                 shared with its video ("ref_video_audio_0" belongs to "ref_video_0")
    audios       [{"name": "ref_audio_0", "seconds", "sample_rate", "channels"}, ...]
    length       the generation length in frames, if known -- longer videos are truncated to it
    """
    images = images or []
    videos = videos or []
    video_audios = video_audios or {}
    audios = audios or []
    warnings: list[str] = []
    lines: list[str] = []

    pic_n = vid_n = aud_n = 0

    for img in images:
        pic_n += 1
        lines.append(f"  <Picture {pic_n}>  {img.get('width')}x{img.get('height')}  ({img.get('name', '')})")

    used_audio_keys: set[str] = set()
    for vid in videos:
        name = str(vid.get("name", ""))
        suffix = name.rsplit("_", 1)[-1]
        frames = int(vid.get("frames", 0) or 0)

        usable = frames
        if length and usable > length:
            usable = int(length)
            warnings.append(f"{name}: {frames} frames exceeds the generation length ({length}); truncated to {usable}.")
        valid = valid_clip_frames(usable)

        soundtrack = video_audios.get(suffix)
        if soundtrack is not None:
            used_audio_keys.add(suffix)
            aud_n += 1
            lines.append(f"  <Audio {aud_n}>  {_audio_desc(soundtrack)}  (soundtrack of {name})")

        vid_n += 1
        size = f"{vid.get('width')}x{vid.get('height')}"
        if valid == 0:
            lines.append(f"  <Video {vid_n}>  {frames} frames, {size}  ({name})  -- TOO SHORT")
            warnings.append(f"{name}: {frames} frames; H3 reference videos need at least {MIN_VIDEO_FRAMES} (the node will fail).")
            continue

        # Rounding down to a 17k+5 length is routine (e.g. 96 -> 90), so it is shown on the line
        # itself rather than raised as a warning.
        used = f"{valid} of {frames} frames used" if valid != frames else f"{frames} frames"
        secs = _seconds(valid)
        lines.append(f"  <Video {vid_n}>  {used}, {secs:.2f}s @ {FPS} fps, {size}  ({name})")
        if secs < MIN_RECOMMENDED_S or secs > MAX_RECOMMENDED_S:
            warnings.append(
                f"{name}: {secs:.2f}s is outside the recommended {MIN_RECOMMENDED_S:g}-{MAX_RECOMMENDED_S:g}s range for reference videos."
            )
        if soundtrack is None:
            lines[-1] += "  [no soundtrack]"

    for key in sorted(set(video_audios) - used_audio_keys):
        warnings.append(f"ref_video_audio_{key} has no matching ref_video_{key}; the node ignores it.")

    for audio in audios:
        aud_n += 1
        lines.append(f"  <Audio {aud_n}>  {_audio_desc(audio)}  (standalone, {audio.get('name', '')})")

    total = len(images) + len(videos) + len(audios) + len(used_audio_keys)
    header = (
        f"H3 references: {pic_n} picture(s), {vid_n} video(s), {aud_n} audio track(s)"
        if total else "H3 references: none"
    )
    text = "\n".join([header, *lines])
    if warnings:
        text += "\n  Warnings:\n" + "\n".join(f"    - {w}" for w in warnings)
    return text, warnings
