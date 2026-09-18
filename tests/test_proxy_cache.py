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
