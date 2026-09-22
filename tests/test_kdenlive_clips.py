"""Tests for utils/kdenlive_clips.py.

The real-ffmpeg tests generate tiny synthetic clips and are skipped (not failed) when
ffmpeg/ffprobe aren't available, matching the archiver's own precedent for tests that
shell out to real tools.
"""
import json
import os
import subprocess
import threading

import pytest

from conftest import import_test_module

kc = import_test_module("utils/kdenlive_clips.py")

_FFMPEG = kc.find_ffmpeg()
_FFPROBE = kc.find_ffprobe()
needs_ffmpeg = pytest.mark.skipif(not _FFMPEG, reason="ffmpeg not available")
needs_ffprobe = pytest.mark.skipif(not _FFPROBE, reason="ffprobe not available")


def _make_clip(path: str, seconds: float = 1.0, comment: str = "hello from a workflow blob") -> None:
    """A tiny real mp4 carrying a metadata comment, standing in for a ComfyUI-saved clip."""
    subprocess.run(
        [_FFMPEG, "-nostdin", "-v", "error", "-y",
         "-f", "lavfi", "-i", f"color=c=blue:s=64x64:d={seconds}",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
         "-metadata", f"comment={comment}", "-shortest", path],
        check=True, capture_output=True,
    )


def _tags(path: str) -> dict:
    out = subprocess.run(
        [_FFPROBE, "-v", "error", "-show_entries", "format_tags", "-of", "json", path],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout).get("format", {}).get("tags", {})


# ── strip_copy_video ─────────────────────────────────────────────────────────────

@needs_ffmpeg
def test_strip_copy_video_removes_metadata_and_preserves_source(tmp_path):
    src = str(tmp_path / "clip.mp4")
    dest = str(tmp_path / "clean.mp4")
    _make_clip(src)
    src_bytes_before = open(src, "rb").read()

    result = kc.strip_copy_video(src, dest)

    assert os.path.isfile(dest)
    assert open(src, "rb").read() == src_bytes_before  # source untouched
    assert not os.path.exists(os.path.splitext(dest)[0] + ".part.mp4")  # temp file cleaned up
    assert result["size_before"] == os.path.getsize(src)
    assert result["size_after"] == os.path.getsize(dest)


@needs_ffmpeg
@needs_ffprobe
def test_strip_copy_video_drops_the_comment_tag(tmp_path):
    src = str(tmp_path / "clip.mp4")
    dest = str(tmp_path / "clean.mp4")
    _make_clip(src, comment="workflow json goes here")
    assert "comment" in _tags(src)

    kc.strip_copy_video(src, dest)

    assert "comment" not in _tags(dest)


@needs_ffmpeg
@needs_ffprobe
def test_strip_copy_video_keeps_the_same_duration(tmp_path):
    src = str(tmp_path / "clip.mp4")
    dest = str(tmp_path / "clean.mp4")
    _make_clip(src, seconds=2.0)

    kc.strip_copy_video(src, dest)

    d_before = kc.probe_duration(_FFPROBE, src)
    d_after = kc.probe_duration(_FFPROBE, dest)
    assert abs(d_before - d_after) < 0.2


def test_strip_copy_video_missing_source_raises(tmp_path):
    with pytest.raises(RuntimeError, match="source file not found"):
        kc.strip_copy_video(str(tmp_path / "nope.mp4"), str(tmp_path / "out.mp4"))


def test_strip_copy_video_missing_ffmpeg_raises(tmp_path, monkeypatch):
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"not a real video, just needs to exist")
    monkeypatch.setattr(kc, "find_ffmpeg", lambda: None)
    with pytest.raises(RuntimeError, match="ffmpeg not found"):
        kc.strip_copy_video(str(src), str(tmp_path / "out.mp4"))


@needs_ffmpeg
def test_strip_copy_video_bad_input_fails_cleanly(tmp_path):
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"this is not a real video file")
    dest = str(tmp_path / "out.mp4")
    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        kc.strip_copy_video(str(src), dest)
    assert not os.path.exists(dest)
    assert not os.path.exists(os.path.splitext(dest)[0] + ".part.mp4")


# ── find_duplicate_files ─────────────────────────────────────────────────────────

def test_find_duplicate_files_groups_by_content(tmp_path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"  # same content, different name
    c = tmp_path / "c.bin"  # different content
    a.write_bytes(b"same content")
    b.write_bytes(b"same content")
    c.write_bytes(b"different")

    groups = kc.find_duplicate_files([str(a), str(b), str(c)])
    assert groups == [sorted([str(a), str(b)])]


def test_find_duplicate_files_no_duplicates_returns_empty():
    assert kc.find_duplicate_files([]) == []


# ── clean_folder ─────────────────────────────────────────────────────────────────

@needs_ffmpeg
def test_clean_folder_cleans_every_video_and_ignores_other_files(tmp_path):
    src_dir = tmp_path / "src"
    dest_dir = tmp_path / "dest"
    src_dir.mkdir()
    _make_clip(str(src_dir / "a.mp4"))
    _make_clip(str(src_dir / "b.mp4"))
    (src_dir / "notes.txt").write_text("not a video")

    report = kc.clean_folder(str(src_dir), str(dest_dir))

    assert report["files_total"] == 2
    assert report["cleaned"] == 2
    assert (dest_dir / "a.mp4").is_file()
    assert (dest_dir / "b.mp4").is_file()
    assert not (dest_dir / "notes.txt").exists()
    assert report["bytes_saved"] >= 0


@needs_ffmpeg
def test_clean_folder_reports_duplicate_sources_but_cleans_both(tmp_path):
    src_dir = tmp_path / "src"
    dest_dir = tmp_path / "dest"
    src_dir.mkdir()
    _make_clip(str(src_dir / "a.mp4"), comment="same")
    _make_clip(str(src_dir / "a_copy.mp4"), comment="same")  # byte-identical source

    report = kc.clean_folder(str(src_dir), str(dest_dir))

    assert report["cleaned"] == 2
    assert len(report["duplicates"]) == 1
    assert sorted(os.path.basename(p) for p in report["duplicates"][0]) == ["a.mp4", "a_copy.mp4"]


def test_clean_folder_rejects_dest_equal_to_src(tmp_path):
    with pytest.raises(RuntimeError, match="different folder"):
        kc.clean_folder(str(tmp_path), str(tmp_path))


def test_clean_folder_missing_src_raises(tmp_path):
    with pytest.raises(RuntimeError, match="source folder not found"):
        kc.clean_folder(str(tmp_path / "nope"), str(tmp_path / "dest"))


@needs_ffmpeg
def test_clean_folder_dry_run_writes_nothing(tmp_path):
    src_dir = tmp_path / "src"
    dest_dir = tmp_path / "dest"
    src_dir.mkdir()
    _make_clip(str(src_dir / "a.mp4"))

    report = kc.clean_folder(str(src_dir), str(dest_dir), dry_run=True)

    assert report["dry_run"] is True
    assert not dest_dir.exists()
    assert report["results"][0]["status"] == "would_clean"


@needs_ffmpeg
def test_clean_folder_rerun_skips_existing(tmp_path):
    src_dir = tmp_path / "src"
    dest_dir = tmp_path / "dest"
    src_dir.mkdir()
    _make_clip(str(src_dir / "a.mp4"))

    kc.clean_folder(str(src_dir), str(dest_dir))
    report2 = kc.clean_folder(str(src_dir), str(dest_dir))

    assert report2["cleaned"] == 0
    assert report2["skipped_existing"] == 1


@needs_ffmpeg
def test_clean_folder_cancel_stops_partway(tmp_path):
    src_dir = tmp_path / "src"
    dest_dir = tmp_path / "dest"
    src_dir.mkdir()
    for name in ("a.mp4", "b.mp4", "c.mp4"):
        _make_clip(str(src_dir / name))

    cancel = threading.Event()

    def progress(ev):
        if ev["done"] == 1:
            cancel.set()

    report = kc.clean_folder(str(src_dir), str(dest_dir), cancel=cancel, progress=progress)

    assert report["cancelled"] is True
    assert report["cleaned"] == 1


@needs_ffmpeg
def test_clean_folder_reports_ffmpeg_errors_without_stopping(tmp_path):
    src_dir = tmp_path / "src"
    dest_dir = tmp_path / "dest"
    src_dir.mkdir()
    (src_dir / "bad.mp4").write_bytes(b"not a real video")
    _make_clip(str(src_dir / "good.mp4"))

    report = kc.clean_folder(str(src_dir), str(dest_dir))

    assert report["cleaned"] == 1
    assert len(report["errors"]) == 1
    assert report["errors"][0]["file"] == "bad.mp4"
    assert (dest_dir / "good.mp4").is_file()
    assert not (dest_dir / "bad.mp4").exists()
