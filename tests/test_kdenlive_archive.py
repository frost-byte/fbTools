"""Tests for utils/kdenlive_archive.py — synthetic .kdenlive projects in tmp_path."""
import re
import threading

import pytest
from conftest import import_test_module

ka = import_test_module("utils/kdenlive_archive.py")


def _chain(res, size=None, extra=""):
    size_prop = f'  <property name="kdenlive:file_size">{size}</property>\n' if size is not None else ""
    return (f' <chain id="c">\n  <property name="resource">{res}</property>\n'
            f'  <property name="mlt_service">avformat-novalidate</property>\n{size_prop}{extra} </chain>\n')


def _project(path, root, chains, docprops=""):
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        f'<mlt LC_NUMERIC="en_US.UTF-8" root="{root}">\n{"".join(chains)}{docprops}</mlt>\n',
        encoding="utf-8",
    )
    return str(path)


def _media(base, rel, data=b"x"):
    p = base / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


def _resources(project_path):
    return re.findall(r'name="resource">([^<]*)<', open(project_path, encoding="utf-8").read())


def test_windows_drive_map(tmp_path):
    _media(tmp_path / "share", "output/video/a/one.mp4")
    proj = _project(tmp_path / "p.kdenlive", "Z:/output/video/a", [_chain("Z:/output/video/a/one.mp4")])
    rep = ka.analyze(proj, path_maps=[f"Z:/={tmp_path}/share/"])
    assert rep["resolved"] == 1 and rep["unresolved_count"] == 0
    assert rep["by_method"] == {"mapped": 1}


def test_unc_map_and_backslashes(tmp_path):
    _media(tmp_path / "share", "output/v/two.mp4")
    proj = _project(tmp_path / "p.kdenlive", "", [_chain("//10.0.0.5/comfy/output/v/two.mp4"), _chain("\\\\10.0.0.5\\comfy\\output\\v\\two.mp4")])
    rep = ka.analyze(proj, path_maps=[f"//10.0.0.5/comfy={tmp_path}/share"])
    assert rep["resolved"] == 2 and rep["unresolved_count"] == 0


def test_bare_relative_resolves_against_mapped_root(tmp_path):
    _media(tmp_path / "share", "output/video/proj/seg.mp4")
    proj = _project(tmp_path / "p.kdenlive", "Z:/output/video/proj", [_chain("seg.mp4")])
    rep = ka.analyze(proj, path_maps=[f"Z:/={tmp_path}/share/"])
    assert rep["by_method"] == {"relative": 1}


def test_speed_prefix_preserved_and_non_file_skipped(tmp_path):
    src = _media(tmp_path, "src/a.mp4")
    proj = _project(tmp_path / "p.kdenlive", "", [_chain(f"-1:{src}"), _chain("black"), _chain("0")])
    dest = tmp_path / "out"
    rep = ka.archive(proj, dest, strip_metadata_opt=False)
    vals = _resources(dest / "p_ARCHIVE.kdenlive")
    assert vals == ["-1:media/src/a.mp4", "black", "0"]
    assert rep["unique_files"] == 1


def test_search_dirs_recover_moved_clip(tmp_path):
    _media(tmp_path / "moved" / "deep", "gone.mp4")
    proj = _project(tmp_path / "p.kdenlive", "", [_chain("/old/place/gone.mp4")])
    assert ka.analyze(proj)["unresolved_count"] == 1
    rep = ka.analyze(proj, search_dirs=[str(tmp_path / "moved")])
    assert rep["unresolved_count"] == 0 and rep["by_method"] == {"searched": 1}


def test_search_prefers_more_matching_path_components(tmp_path):
    _media(tmp_path / "s", "a/x/clip.mp4", b"1")
    good = _media(tmp_path / "s", "b/keep/clip.mp4", b"2")
    proj = _project(tmp_path / "p.kdenlive", "", [_chain("/old/keep/clip.mp4")])
    rep = ka.analyze(proj, search_dirs=[str(tmp_path / "s")])
    assert rep["ambiguous"] == []
    dest = tmp_path / "out"
    ka.archive(proj, dest, search_dirs=[str(tmp_path / "s")], strip_metadata_opt=False)
    assert (dest / "media/keep/clip.mp4").read_bytes() == good.read_bytes()


def test_search_tie_broken_by_recorded_file_size(tmp_path):
    _media(tmp_path / "s", "a/clip.mp4", b"12345")
    right = _media(tmp_path / "s", "b/clip.mp4", b"123")
    proj = _project(tmp_path / "p.kdenlive", "", [_chain("/gone/clip.mp4", size=3)])
    rep = ka.analyze(proj, search_dirs=[str(tmp_path / "s")])
    assert rep["ambiguous"] == []
    dest = tmp_path / "out"
    ka.archive(proj, dest, search_dirs=[str(tmp_path / "s")], strip_metadata_opt=False)
    assert (dest / "media/b/clip.mp4").read_bytes() == right.read_bytes()


def test_unresolvable_tie_is_reported_ambiguous(tmp_path):
    _media(tmp_path / "s", "a/clip.mp4", b"1")
    _media(tmp_path / "s", "b/clip.mp4", b"2")
    proj = _project(tmp_path / "p.kdenlive", "", [_chain("/gone/clip.mp4")])
    rep = ka.analyze(proj, search_dirs=[str(tmp_path / "s")])
    assert len(rep["ambiguous"]) == 1 and len(rep["ambiguous"][0]["candidates"]) == 2


def test_key_collisions_get_unique_names(tmp_path):
    a = _media(tmp_path, "one/output/seg.mp4", b"a")
    b = _media(tmp_path, "two/output/seg.mp4", b"b")
    proj = _project(tmp_path / "p.kdenlive", "", [_chain(str(a)), _chain(str(b))])
    dest = tmp_path / "out"
    ka.archive(proj, dest, strip_metadata_opt=False)
    assert (dest / "media/output/seg.mp4").read_bytes() == b"a"
    assert (dest / "media/output/seg_2.mp4").read_bytes() == b"b"
    assert _resources(dest / "p_ARCHIVE.kdenlive") == ["media/output/seg.mp4", "media/output/seg_2.mp4"]


def test_missing_clip_left_untouched_and_reported(tmp_path):
    ok = _media(tmp_path, "src/ok.mp4")
    proj = _project(tmp_path / "p.kdenlive", "", [_chain(str(ok)), _chain("/nowhere/lost.mp4")])
    dest = tmp_path / "out"
    rep = ka.archive(proj, dest, strip_metadata_opt=False)
    assert rep["unresolved"] == ["/nowhere/lost.mp4"] and rep["copied"] == 1
    assert _resources(dest / "p_ARCHIVE.kdenlive") == ["media/src/ok.mp4", "/nowhere/lost.mp4"]


def test_archive_sets_empty_root_and_leaves_source_untouched(tmp_path):
    src = _media(tmp_path, "src/a.mp4")
    proj = _project(tmp_path / "p.kdenlive", "Z:/somewhere", [_chain(str(src))])
    before = open(proj, encoding="utf-8").read()
    dest = tmp_path / "out"
    ka.archive(proj, dest, strip_metadata_opt=False)
    assert 'root=""' in (dest / "p_ARCHIVE.kdenlive").read_text(encoding="utf-8")
    assert open(proj, encoding="utf-8").read() == before


def test_dry_run_writes_nothing(tmp_path):
    src = _media(tmp_path, "src/a.mp4")
    proj = _project(tmp_path / "p.kdenlive", "", [_chain(str(src))])
    dest = tmp_path / "out"
    rep = ka.archive(proj, dest, dry_run=True)
    assert rep["dry_run"] and not dest.exists() and rep["copied"] == 0


def test_rerun_skips_existing_files(tmp_path):
    src = _media(tmp_path, "src/a.mp4", b"abc")
    proj = _project(tmp_path / "p.kdenlive", "", [_chain(str(src))])
    dest = tmp_path / "out"
    assert ka.archive(proj, dest)["copied"] == 1
    again = ka.archive(proj, dest)
    assert again["copied"] == 0 and again["skipped_existing"] == 1


def test_cancel_stops_before_writing_project(tmp_path):
    a = _media(tmp_path, "src/a.mp4")
    proj = _project(tmp_path / "p.kdenlive", "", [_chain(str(a))])
    ev = threading.Event()
    ev.set()
    dest = tmp_path / "out"
    rep = ka.archive(proj, dest, cancel=ev)
    assert rep["cancelled"] and not (dest / "p_ARCHIVE.kdenlive").exists()


def test_dest_cannot_be_project_folder(tmp_path):
    proj = _project(tmp_path / "p.kdenlive", "", [_chain("black")])
    with pytest.raises(ValueError):
        ka.archive(proj, tmp_path)


def test_ampersand_paths_are_escaped(tmp_path):
    src = _media(tmp_path, "a&b/clip.mp4")
    proj = _project(tmp_path / "p.kdenlive", "", [_chain(str(src).replace("&", "&amp;"))])
    dest = tmp_path / "out"
    ka.archive(proj, dest, strip_metadata_opt=False)
    assert _resources(dest / "p_ARCHIVE.kdenlive") == ["media/a&amp;b/clip.mp4"]
    assert (dest / "media/a&b/clip.mp4").exists()


def test_strip_removes_only_target_properties(tmp_path):
    big = "x" * 70000
    meta = (f'  <property name="meta.attr.workflow.markup">{{"a":1}}</property>\n'
            f'  <property name="meta.attr.prompt.markup">{{"b":2}}</property>\n'
            f'  <property name="meta.attr.huge.markup">{big}</property>\n'
            f'  <property name="meta.attr.small.markup">keep</property>\n'
            f'  <property name="meta.media.0.codec.name">h264</property>\n')
    proj = _project(tmp_path / "p.kdenlive", "", [_chain("black", extra=meta)])
    rep = ka.strip_metadata(proj)
    out = open(rep["output"], encoding="utf-8").read()
    assert rep["removed"] == 3 and rep["output"].endswith("p_stripped.kdenlive")
    assert "meta.attr.small.markup" in out and "meta.media.0.codec.name" in out
    assert "workflow" not in out and "huge" not in out
    assert rep["size_after"] < rep["size_before"]


def test_strip_in_place_keeps_backup(tmp_path):
    meta = '  <property name="meta.attr.workflow.markup">{"a":1}</property>\n'
    proj = _project(tmp_path / "p.kdenlive", "", [_chain("black", extra=meta)])
    original = open(proj, encoding="utf-8").read()
    rep = ka.strip_metadata(proj, output=proj)
    assert rep["in_place"] and "workflow" not in open(proj, encoding="utf-8").read()
    assert open(proj + ".bak", encoding="utf-8").read() == original


def test_archive_strips_metadata_by_default(tmp_path):
    src = _media(tmp_path, "src/a.mp4")
    meta = '  <property name="meta.attr.workflow.markup">{"a":1}</property>\n'
    proj = _project(tmp_path / "p.kdenlive", "", [_chain(str(src), extra=meta)])
    dest = tmp_path / "out"
    rep = ka.archive(proj, dest)
    assert rep["stripped"]["removed"] == 1
    assert "workflow" not in (dest / "p_ARCHIVE.kdenlive").read_text(encoding="utf-8")


def test_title_clips_produce_warning(tmp_path):
    chain = ' <producer id="t">\n  <property name="mlt_service">kdenlivetitle</property>\n </producer>\n'
    proj = _project(tmp_path / "p.kdenlive", "", [chain])
    assert any("title" in w for w in ka.analyze(proj)["warnings"])


def test_parse_path_map_validation():
    assert ka.parse_path_map("Z:\\=/mnt/x/") == ("Z:", "/mnt/x")
    with pytest.raises(ValueError):
        ka.parse_path_map("no-equals")
