"""Tests for utils/proxy_cache.py — pure filename/filter-string construction.

ensure_source_profile_proxy() itself shells out to ffmpeg and isn't covered
here; these tests target the pure helpers that determine the proxy's on-disk
identity and the fps/scale filter chain baked into it.
"""
from conftest import import_test_module

proxy_cache = import_test_module("utils/proxy_cache.py")


def test_proxy_fps_is_24():
    # H3 requires 24fps reference video; baking this into the proxy at build
    # time avoids re-resampling on every generation run.
    assert proxy_cache._PROXY_FPS == 24


def test_video_filter_chain_applies_fps_before_scale():
    chain = proxy_cache._video_filter_chain(768)
    parts = chain.split(",", 1)
    assert parts[0] == "fps=24"
    assert parts[1] == proxy_cache._scale_filter(768)


def test_video_filter_chain_fps_precedes_scale_for_all_edges():
    for short_edge in (480, 768, 1080):
        chain = proxy_cache._video_filter_chain(short_edge)
        assert chain.startswith("fps=24,")
        assert chain.index("fps=") < chain.index("scale=")


def test_proxy_stem_includes_fps_version_tag():
    stem = proxy_cache._proxy_stem("profile1", "clip_5", 1.0, 12.5, 768)
    assert stem.endswith("_f24")


def test_proxy_stem_version_bump_changes_stem():
    # A proxy built before the fps-baking change (no version tag) must not
    # be mistaken for fresh under the new scheme.
    stem = proxy_cache._proxy_stem("profile1", "clip_5", 1.0, 12.5, 768)
    legacy_stem = "profile1__clip_5__1.00-12.50__h768_r32"
    assert stem != legacy_stem
    assert stem == f"{legacy_stem}_{proxy_cache._PROXY_STEM_VERSION}"


def test_proxy_stem_sanitizes_path_like_ids():
    stem = proxy_cache._proxy_stem("a/b", "c d", 0.0, 1.0, 480)
    assert "/" not in stem
    assert " " not in stem
    assert stem.startswith("a_b__c_d__")


# ── _proxy_dir kind separation (Plan 16: bundle video proxies) ────────────────

def test_proxy_dir_defaults_to_source_profiles_unchanged(tmp_path):
    d = proxy_cache._proxy_dir(str(tmp_path))
    assert d == tmp_path / "proxies" / "source_profiles"
    assert d.is_dir()


def test_proxy_dir_bundles_kind_is_a_separate_directory(tmp_path):
    sp_dir = proxy_cache._proxy_dir(str(tmp_path), "source_profiles")
    bundle_dir = proxy_cache._proxy_dir(str(tmp_path), "bundles")
    assert bundle_dir == tmp_path / "proxies" / "bundles"
    assert bundle_dir != sp_dir
    assert bundle_dir.is_dir()


def test_ensure_bundle_video_proxy_and_ensure_source_profile_proxy_never_collide(tmp_path, monkeypatch):
    # Same namespace string used for both a "profile_id"/"clip_id" pair and a bundle id+"video" —
    # must land in different directories, not overwrite each other, even though _proxy_stem alone
    # would produce identical filenames for identical (namespace, key, start, end, short_edge).
    src = tmp_path / "src.mp4"
    src.write_bytes(b"fake")

    calls = []

    def _fake_run(cmd, **kwargs):
        calls.append(cmd)
        # Simulate ffmpeg writing the requested output path.
        out_path = cmd[-1]
        with open(out_path, "wb") as f:
            f.write(b"fake-proxy")

        class _Result:
            returncode = 0
        return _Result()

    monkeypatch.setattr(proxy_cache.subprocess, "run", _fake_run)

    sp_path = proxy_cache.ensure_source_profile_proxy(str(src), "shared", "video", 0.0, 2.0, 768, str(tmp_path))
    bundle_path = proxy_cache.ensure_bundle_video_proxy(str(src), "shared", 0.0, 2.0, 768, str(tmp_path))

    assert sp_path is not None and bundle_path is not None
    assert sp_path != bundle_path
    assert "source_profiles" in sp_path
    assert "bundles" in bundle_path
    assert len(calls) == 2  # both actually invoked ffmpeg (no accidental cache hit across kinds)
