"""Route-level wiring test for Plan 14: /fbtools/kdenlive/clean's organize_by_primary and
embed_cast_metadata flags sharing one _cast_info_cache() lookup per file.

Exercises _run_clean_job directly (not through the aiohttp handler) with clean_folder faked out,
so this runs without ffmpeg — utils/kdenlive_clips.py's own ffmpeg round-trip (extra_metadata
actually landing on a clip) is covered separately in tests/test_kdenlive_clips.py, and
utils/generation_metadata.py's tag building/reading is covered in tests/test_generation_metadata.py.
Same stub technique as tests/test_kdenlive_browse_routes.py.
"""
import importlib
import json
import sys
import threading
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = "fbt_kdenlive_clean_test_pkg"

_GRAPH = {
    "1": {
        "class_type": "fbt_SceneCastBuild",
        "inputs": {
            "cast_entries_json": json.dumps([
                {"subject_id": "alex", "bundle_id": "alex_amd_norsk_dance_flo"},
                {"subject_id": "sam", "bundle_id": "sam_bundle_3"},
            ]),
            "prompt_composition": ["2", 0],
        },
    },
    "2": {"class_type": "fbt_CompositionLoad", "inputs": {"composition_name": "wide_shot"}},
}


@pytest.fixture()
def mod(tmp_path, monkeypatch):
    fp = types.ModuleType("folder_paths")
    fp.get_input_directory = lambda: str(tmp_path / "input")
    fp.get_output_directory = lambda: str(tmp_path / "output")

    class _Routes:
        def __getattr__(self, name):
            if name in ("get", "post", "delete", "put"):
                return lambda path: (lambda fn: fn)
            raise AttributeError(name)

    class _PS:
        instance = types.SimpleNamespace(routes=_Routes(), send_sync=lambda *a, **k: None)

    srv = types.ModuleType("server")
    srv.PromptServer = _PS

    pkg = types.ModuleType(PKG)
    pkg.__path__ = [str(ROOT)]
    monkeypatch.setitem(sys.modules, "folder_paths", fp)
    monkeypatch.setitem(sys.modules, "server", srv)
    monkeypatch.setitem(sys.modules, PKG, pkg)
    for name in list(sys.modules):
        if name.startswith(PKG + "."):
            monkeypatch.delitem(sys.modules, name)
    return importlib.import_module(f"{PKG}.nodes.kdenlive_archive")


def _fake_clean_folder_factory(clip_name, clip_path):
    def fake_clean_folder(src, dest, *, dry_run, cancel, progress, dest_subdir, extra_metadata):
        sub = dest_subdir(clip_name, clip_path) if dest_subdir else None
        meta = extra_metadata(clip_name, clip_path) if extra_metadata else None
        fake_clean_folder.calls.append({"dest_subdir": sub, "extra_metadata": meta,
                                         "dest_subdir_passed": dest_subdir is not None,
                                         "extra_metadata_passed": extra_metadata is not None})
        return {
            "src_dir": src, "dest_dir": dest, "dry_run": dry_run, "cancelled": False,
            "files_total": 1, "cleaned": 1, "skipped_existing": 0,
            "bytes_before": 10, "bytes_after": 10, "duplicates": [], "errors": [],
            "results": [{"file": clip_name, "status": "cleaned", "size_before": 10, "size_after": 10}],
        }
    fake_clean_folder.calls = []
    return fake_clean_folder


def test_organize_and_embed_together_share_one_embedded_prompt_read(mod, tmp_path, monkeypatch):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    clip = src_dir / "wide_shot_00001-audio.mp4"
    clip.write_bytes(b"x")

    read_calls = []
    monkeypatch.setattr(mod, "read_embedded_prompt", lambda p: (read_calls.append(p), _GRAPH)[1])
    monkeypatch.setattr(mod, "_load_composition_by_name",
                         lambda name: {"subjects": {"A": "alex", "B": "sam"}} if name == "wide_shot" else None)
    fake = _fake_clean_folder_factory(clip.name, str(clip))
    monkeypatch.setattr(mod, "clean_folder", fake)

    job = {"id": "j1", "kind": "clean", "cancel": threading.Event()}
    mod._run_clean_job(job, str(src_dir), str(tmp_path / "dest"), False, True, embed_cast_metadata=True)

    # Both dest_subdir and extra_metadata ran, but read_embedded_prompt (the ffprobe shell-out)
    # only fired once — the shared _cast_info_cache() did its job.
    assert len(read_calls) == 1
    call = fake.calls[0]
    assert call["dest_subdir"] == "alex"

    meta = call["extra_metadata"]
    assert set(meta) == {"fbtools_cast", "creation_time"}
    tag = json.loads(meta["fbtools_cast"])
    assert tag == {
        "composition": "wide_shot", "primary_subject": "alex", "primary_bundle": "alex_amd_norsk_dance_flo",
        "tags": ["alex_amd_norsk_dance_flo", "sam_bundle_3"], "generated_at": meta["creation_time"],
    }

    entry = job["report"]["results"][0]
    assert entry["primary_subject"] == "alex"
    assert entry["primary_bundle"] == "alex_amd_norsk_dance_flo"
    assert entry["tags"] == ["alex_amd_norsk_dance_flo", "sam_bundle_3"]


def test_embed_only_does_not_nest_but_still_enriches_the_report(mod, tmp_path, monkeypatch):
    src_dir = tmp_path / "src2"
    src_dir.mkdir()
    clip = src_dir / "clip.mp4"
    clip.write_bytes(b"x")

    monkeypatch.setattr(mod, "read_embedded_prompt", lambda p: _GRAPH)
    monkeypatch.setattr(mod, "_load_composition_by_name", lambda name: {"subjects": {"A": "alex"}})
    fake = _fake_clean_folder_factory(clip.name, str(clip))
    monkeypatch.setattr(mod, "clean_folder", fake)

    job = {"id": "j2", "kind": "clean", "cancel": threading.Event()}
    mod._run_clean_job(job, str(src_dir), str(tmp_path / "dest2"), False, False, embed_cast_metadata=True)

    assert fake.calls[0]["dest_subdir_passed"] is False
    assert fake.calls[0]["extra_metadata_passed"] is True
    entry = job["report"]["results"][0]
    assert "subdir" not in entry
    assert entry["primary_subject"] == "alex"


def test_organize_only_does_not_write_metadata(mod, tmp_path, monkeypatch):
    src_dir = tmp_path / "src3"
    src_dir.mkdir()
    clip = src_dir / "clip.mp4"
    clip.write_bytes(b"x")

    monkeypatch.setattr(mod, "read_embedded_prompt", lambda p: _GRAPH)
    monkeypatch.setattr(mod, "_load_composition_by_name", lambda name: {"subjects": {"A": "alex"}})
    fake = _fake_clean_folder_factory(clip.name, str(clip))
    monkeypatch.setattr(mod, "clean_folder", fake)

    job = {"id": "j3", "kind": "clean", "cancel": threading.Event()}
    mod._run_clean_job(job, str(src_dir), str(tmp_path / "dest3"), False, True, embed_cast_metadata=False)

    assert fake.calls[0]["dest_subdir_passed"] is True
    assert fake.calls[0]["extra_metadata_passed"] is False


def test_neither_flag_never_reads_embedded_metadata_and_leaves_report_bare(mod, tmp_path, monkeypatch):
    src_dir = tmp_path / "src4"
    src_dir.mkdir()
    clip = src_dir / "clip.mp4"
    clip.write_bytes(b"x")

    def fail_read_embedded_prompt(path):
        raise AssertionError("should not be called when neither flag is set")

    monkeypatch.setattr(mod, "read_embedded_prompt", fail_read_embedded_prompt)
    fake = _fake_clean_folder_factory(clip.name, str(clip))
    monkeypatch.setattr(mod, "clean_folder", fake)

    job = {"id": "j4", "kind": "clean", "cancel": threading.Event()}
    mod._run_clean_job(job, str(src_dir), str(tmp_path / "dest4"), False, False)

    assert fake.calls[0]["dest_subdir_passed"] is False
    assert fake.calls[0]["extra_metadata_passed"] is False
    entry = job["report"]["results"][0]
    assert "primary_subject" not in entry
    assert "tags" not in entry


def test_missing_embedded_prompt_falls_back_cleanly_with_primary_bundle_key_present(mod, tmp_path, monkeypatch):
    """Regression test for a latent KeyError: the no-embedded-metadata fallback dict in
    _cast_info_cache() must carry a primary_bundle key too, since the report-enrichment step reads
    it unconditionally once organize_by_primary or embed_cast_metadata is on."""
    src_dir = tmp_path / "src5"
    src_dir.mkdir()
    clip = src_dir / "no_metadata.mp4"
    clip.write_bytes(b"x")

    monkeypatch.setattr(mod, "read_embedded_prompt", lambda p: None)
    fake = _fake_clean_folder_factory(clip.name, str(clip))
    monkeypatch.setattr(mod, "clean_folder", fake)

    job = {"id": "j5", "kind": "clean", "cancel": threading.Event()}
    mod._run_clean_job(job, str(src_dir), str(tmp_path / "dest5"), False, True, embed_cast_metadata=True)

    entry = job["report"]["results"][0]
    assert entry["primary_subject"] is None
    assert entry["primary_bundle"] is None
    assert entry["tags"] == []
    assert "no embedded generation metadata" in entry["note"]
