"""Tests for utils/generation_metadata.py::extract_cast_info (pure graph-walk logic; the
ffprobe-based read_embedded_prompt() is exercised separately, gated on ffprobe availability)."""
import json

from conftest import import_test_module

gm = import_test_module("utils/generation_metadata.py")

extract_cast_info = gm.extract_cast_info
read_embedded_prompt = gm.read_embedded_prompt


def _cast_node(cast_entries, prompt_composition_link=None):
    inputs = {"cast_entries_json": json.dumps(cast_entries), "clip_id": "shot_1"}
    if prompt_composition_link is not None:
        inputs["prompt_composition"] = prompt_composition_link
    return {"class_type": "fbt_SceneCastBuild", "inputs": inputs}


def _comp_load_node(name):
    return {"class_type": "fbt_CompositionLoad", "inputs": {"composition_name": name}}


def _loader_node(composition_name, scene_cast_link=("1", 0)):
    return {"class_type": "fbt_PromptCompositionLoader",
            "inputs": {"composition_name": composition_name, "scene_cast": list(scene_cast_link)}}


def _loader(compositions: dict):
    return lambda name: compositions.get(name)


# ── The real shape confirmed against an actual generated clip ──────────────────────

def test_real_shape_composition_wired_via_composition_load():
    graph = {
        "3918": _cast_node(
            [{"subject_id": "alex", "bundle_id": "alex_amd_norsk_dance_flo", "visual_mode": "both"},
             {"subject_id": "sam", "bundle_id": "sam_bundle_3", "visual_mode": "images"}],
            prompt_composition_link=["3923", 0],
        ),
        "3923": _comp_load_node("wide_shot"),
        "3924": _loader_node("_pops", scene_cast_link=("3918", 0)),  # stale dropdown value
    }
    compositions = {"wide_shot": {"subjects": {"A": "alex", "B": "sam"}}}

    info = extract_cast_info(graph, load_composition=_loader(compositions))

    assert info["tags"] == ["alex_amd_norsk_dance_flo", "sam_bundle_3"]
    assert info["composition_name"] == "wide_shot"  # not the loader's stale "_pops"
    assert info["primary_subject"] == "alex"
    assert info["note"] is None


# ── Composition resolution precedence ───────────────────────────────────────────────

def test_falls_back_to_loader_dropdown_when_no_composition_load_link():
    graph = {
        "1": _cast_node([{"subject_id": "a", "bundle_id": "bun_a"}]),
        "2": _loader_node("direct_pick", scene_cast_link=("1", 0)),
    }
    compositions = {"direct_pick": {"subjects": {"A": "a"}}}
    info = extract_cast_info(graph, load_composition=_loader(compositions))
    assert info["composition_name"] == "direct_pick"
    assert info["primary_subject"] == "a"


def test_wired_to_non_composition_load_node_does_not_guess():
    graph = {
        "1": _cast_node([{"subject_id": "a", "bundle_id": "bun_a"}], prompt_composition_link=["9", 0]),
        "9": {"class_type": "SomethingElse", "inputs": {}},
    }
    info = extract_cast_info(graph, load_composition=_loader({}))
    assert info["composition_name"] is None
    assert info["primary_subject"] is None
    assert info["note"]


def test_no_composition_node_at_all_source_profile_driven():
    graph = {"1": _cast_node([{"subject_id": "a", "bundle_id": "bun_a"}])}
    info = extract_cast_info(graph, load_composition=_loader({}))
    assert info["tags"] == ["bun_a"]
    assert info["composition_name"] is None
    assert info["primary_subject"] is None
    assert "Source-Profile" in info["note"]


# ── Explicit "primary" tag (Scene Cast Build's ★ toggle) ────────────────────────────

def test_explicit_primary_resolves_source_profile_driven_clip():
    # This is exactly the case the docstring used to call "can't be determined yet" —
    # an explicit tag closes it.
    graph = {"1": _cast_node([
        {"subject_id": "a", "bundle_id": "bun_a", "primary": False},
        {"subject_id": "b", "bundle_id": "bun_b", "primary": True},
    ])}
    info = extract_cast_info(graph, load_composition=_loader({}))
    assert info["primary_subject"] == "b"
    assert info["primary_bundle"] == "bun_b"
    assert info["composition_name"] is None
    assert info["note"] is None


def test_explicit_primary_falls_back_to_source_subject_id_when_no_subject_id():
    # Mirrors SceneCastBuild.execute()'s own subject_id-or-source_subject_id resolution
    # for a source-derived entry the UI never assigned an explicit subject_id to.
    graph = {"1": _cast_node([
        {"subject_id": "", "source_subject_id": "sp_char_1", "primary": True},
    ])}
    info = extract_cast_info(graph, load_composition=_loader({}))
    assert info["primary_subject"] == "sp_char_1"


def test_explicit_primary_overrides_composition_slot_order():
    graph = {
        "3918": _cast_node(
            [{"subject_id": "alex", "bundle_id": "alex_bundle", "primary": False},
             {"subject_id": "sam", "bundle_id": "sam_bundle", "primary": True}],
            prompt_composition_link=["3923", 0],
        ),
        "3923": _comp_load_node("wide_shot"),
    }
    compositions = {"wide_shot": {"subjects": {"A": "alex", "B": "sam"}}}
    info = extract_cast_info(graph, load_composition=_loader(compositions))
    # Slot order alone would say "alex" (slot A) — the explicit flag wins instead.
    assert info["primary_subject"] == "sam"
    assert info["primary_bundle"] == "sam_bundle"
    assert info["composition_name"] == "wide_shot"
    assert info["note"] is None


def test_no_primary_tag_falls_through_to_slot_order_unchanged():
    # Regression guard: untagged entries behave exactly as before this feature existed.
    graph = {
        "3918": _cast_node(
            [{"subject_id": "alex", "bundle_id": "alex_bundle"},
             {"subject_id": "sam", "bundle_id": "sam_bundle"}],
            prompt_composition_link=["3923", 0],
        ),
        "3923": _comp_load_node("wide_shot"),
    }
    compositions = {"wide_shot": {"subjects": {"A": "alex", "B": "sam"}}}
    info = extract_cast_info(graph, load_composition=_loader(compositions))
    assert info["primary_subject"] == "alex"


def test_empty_string_primary_is_falsy_not_tagged():
    graph = {"1": _cast_node([{"subject_id": "a", "bundle_id": "bun_a", "primary": ""}])}
    info = extract_cast_info(graph, load_composition=_loader({}))
    assert info["primary_subject"] is None  # no composition either -> untagged source-profile case
    assert "Source-Profile" in info["note"]


# ── Cast / tags ──────────────────────────────────────────────────────────────────

def test_no_scene_cast_build_node():
    info = extract_cast_info({"1": {"class_type": "KSampler", "inputs": {}}}, load_composition=_loader({}))
    assert info == {"tags": [], "primary_subject": None, "primary_bundle": None, "composition_name": None,
                     "note": "no Scene Cast Build node found — not a cast-driven generation"}


def test_duplicate_bundle_ids_deduplicated_in_first_seen_order():
    graph = {"1": _cast_node([
        {"subject_id": "a", "bundle_id": "bun_a"},
        {"subject_id": "b", "bundle_id": "bun_b"},
        {"subject_id": "c", "bundle_id": "bun_a"},
    ])}
    info = extract_cast_info(graph, load_composition=_loader({}))
    assert info["tags"] == ["bun_a", "bun_b"]


def test_entries_without_bundle_id_are_skipped():
    graph = {"1": _cast_node([{"subject_id": "a"}, {"subject_id": "b", "bundle_id": "bun_b"}])}
    info = extract_cast_info(graph, load_composition=_loader({}))
    assert info["tags"] == ["bun_b"]


def test_malformed_cast_entries_json_yields_no_tags():
    graph = {"1": {"class_type": "fbt_SceneCastBuild", "inputs": {"cast_entries_json": "not json"}}}
    info = extract_cast_info(graph, load_composition=_loader({}))
    assert info["tags"] == []


# ── Primary-subject resolution ───────────────────────────────────────────────────

def test_first_slot_uses_insertion_order_not_sorted_order():
    """Regression guard: slot_letter()'s A..Z, AA.. scheme sorts wrong as plain strings past Z
    ("AA" < "Z" lexically) — the ordering bug this session already hit once in the frontend."""
    graph = {
        "1": _cast_node([{"subject_id": "z_subject", "bundle_id": "bun_z"}],
                        prompt_composition_link=["2", 0]),
        "2": _comp_load_node("big_comp"),
    }
    # Insertion order: Z (26th letter) comes before AA (27th) even though "AA" < "Z" as strings.
    compositions = {"big_comp": {"subjects": {"Z": "z_subject", "AA": "aa_subject"}}}
    info = extract_cast_info(graph, load_composition=_loader(compositions))
    assert info["primary_subject"] == "z_subject"


def test_unassigned_first_slot_falls_through_to_next_assigned():
    graph = {
        "1": _cast_node([{"subject_id": "b", "bundle_id": "bun_b"}], prompt_composition_link=["2", 0]),
        "2": _comp_load_node("comp"),
    }
    compositions = {"comp": {"subjects": {"A": "", "B": "b"}}}
    info = extract_cast_info(graph, load_composition=_loader(compositions))
    assert info["primary_subject"] == "b"


def test_no_subjects_assigned_returns_none_with_note():
    graph = {
        "1": _cast_node([], prompt_composition_link=["2", 0]),
        "2": _comp_load_node("comp"),
    }
    compositions = {"comp": {"subjects": {"A": "", "B": ""}}}
    info = extract_cast_info(graph, load_composition=_loader(compositions))
    assert info["primary_subject"] is None
    assert "no subject assigned" in info["note"]


def test_composition_not_found_by_loader():
    graph = {
        "1": _cast_node([{"subject_id": "a", "bundle_id": "bun_a"}], prompt_composition_link=["2", 0]),
        "2": _comp_load_node("missing_comp"),
    }
    info = extract_cast_info(graph, load_composition=_loader({}))
    assert info["primary_subject"] is None
    assert info["composition_name"] == "missing_comp"
    assert "could not be loaded" in info["note"]


def test_no_load_composition_callback_supplied():
    graph = {
        "1": _cast_node([{"subject_id": "a", "bundle_id": "bun_a"}], prompt_composition_link=["2", 0]),
        "2": _comp_load_node("comp"),
    }
    info = extract_cast_info(graph)  # load_composition omitted entirely
    assert info["primary_subject"] is None
    assert info["composition_name"] == "comp"


# ── primary_bundle ───────────────────────────────────────────────────────────────

def test_primary_bundle_is_the_cast_entrys_bundle_for_the_primary_subject():
    graph = {
        "1": _cast_node(
            [{"subject_id": "alex", "bundle_id": "alex_amd_norsk_dance_flo"},
             {"subject_id": "sam", "bundle_id": "sam_bundle_3"}],
            prompt_composition_link=["2", 0],
        ),
        "2": _comp_load_node("comp"),
    }
    compositions = {"comp": {"subjects": {"A": "alex", "B": "sam"}}}
    info = extract_cast_info(graph, load_composition=_loader(compositions))
    assert info["primary_bundle"] == "alex_amd_norsk_dance_flo"


def test_primary_bundle_none_when_primary_subject_has_no_cast_entry():
    graph = {
        "1": _cast_node([{"subject_id": "sam", "bundle_id": "sam_bundle_3"}],
                        prompt_composition_link=["2", 0]),
        "2": _comp_load_node("comp"),
    }
    # "alex" is slot A (primary) but never appears in the cast entries.
    compositions = {"comp": {"subjects": {"A": "alex", "B": "sam"}}}
    info = extract_cast_info(graph, load_composition=_loader(compositions))
    assert info["primary_subject"] == "alex"
    assert info["primary_bundle"] is None
    assert "no bundle" in info["note"]


def test_primary_bundle_none_when_primary_subject_itself_is_none():
    graph = {"1": _cast_node([{"subject_id": "a", "bundle_id": "bun_a"}])}
    info = extract_cast_info(graph, load_composition=_loader({}))
    assert info["primary_subject"] is None
    assert info["primary_bundle"] is None


# ── read_embedded_prompt ──────────────────────────────────────────────────────────

def test_read_embedded_prompt_missing_ffprobe_returns_none():
    assert read_embedded_prompt("/no/such/file.mp4", ffprobe="/definitely/not/a/real/ffprobe") is None


def test_read_embedded_prompt_missing_file_returns_none():
    import shutil
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return  # ffprobe not installed in this environment; nothing to test here
    assert read_embedded_prompt("/no/such/file.mp4", ffprobe=ffprobe) is None


# ── build_cast_summary_tag / read_cast_summary_tag ────────────────────────────────

def test_build_cast_summary_tag_shape():
    info = {"composition_name": "wide_shot", "primary_subject": "alex",
            "primary_bundle": "alex_amd_norsk_dance_flo",
            "tags": ["alex_amd_norsk_dance_flo", "sam_bundle_3"], "note": None}
    raw = gm.build_cast_summary_tag(info, "2026-09-21T14:32:01+00:00")
    parsed = json.loads(raw)
    assert parsed == {
        "composition": "wide_shot", "primary_subject": "alex",
        "primary_bundle": "alex_amd_norsk_dance_flo",
        "tags": ["alex_amd_norsk_dance_flo", "sam_bundle_3"],
        "generated_at": "2026-09-21T14:32:01+00:00",
    }


def test_build_cast_summary_tag_handles_missing_fields():
    raw = gm.build_cast_summary_tag({}, "2026-09-21T14:32:01+00:00")
    parsed = json.loads(raw)
    assert parsed["composition"] is None
    assert parsed["tags"] == []


def test_generated_at_iso_matches_file_mtime(tmp_path):
    import os
    f = tmp_path / "clip.mp4"
    f.write_bytes(b"x")
    os.utime(f, (1700000000, 1700000000))
    iso = gm.generated_at_iso(str(f))
    assert iso.startswith("2023-11-14")  # 1700000000 UTC


def test_read_cast_summary_tag_none_when_ffprobe_missing():
    assert gm.read_cast_summary_tag("/no/such/file.mp4", ffprobe="/not/real/ffprobe") is None


def test_read_cast_summary_tag_none_when_file_missing():
    import shutil as _sh
    ffprobe = _sh.which("ffprobe")
    if not ffprobe:
        return
    assert gm.read_cast_summary_tag("/no/such/file.mp4", ffprobe=ffprobe) is None


def test_cast_summary_tag_round_trips_through_strip_copy_video(tmp_path):
    """End-to-end: write via strip_copy_video's extra_metadata, read back via
    read_cast_summary_tag, get the same structured data build_cast_summary_tag was given."""
    import shutil as _sh
    ffmpeg = _sh.which("ffmpeg")
    ffprobe = _sh.which("ffprobe")
    if not ffmpeg:
        try:
            from imageio_ffmpeg import get_ffmpeg_exe
            ffmpeg = get_ffmpeg_exe()
        except Exception:
            ffmpeg = None
    if not ffmpeg or not ffprobe:
        return  # ffmpeg/ffprobe unavailable in this environment; nothing to verify here

    kc = import_test_module("utils/kdenlive_clips.py")
    src = tmp_path / "clip.mp4"
    dest = tmp_path / "clean.mp4"
    import subprocess as sp
    sp.run([ffmpeg, "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=blue:s=32x32:d=1",
            str(src)], check=True)

    info = {"composition_name": "wide_shot", "primary_subject": "alex",
            "primary_bundle": "alex_amd_norsk_dance_flo", "tags": ["alex_amd_norsk_dance_flo"], "note": None}
    generated_at = gm.generated_at_iso(str(src))
    tag_json = gm.build_cast_summary_tag(info, generated_at)
    kc.strip_copy_video(str(src), str(dest), extra_metadata={
        gm.CAST_SUMMARY_TAG: tag_json, "creation_time": generated_at,
    })

    read_back = gm.read_cast_summary_tag(str(dest), ffprobe=ffprobe)
    assert read_back == {
        "composition": "wide_shot", "primary_subject": "alex",
        "primary_bundle": "alex_amd_norsk_dance_flo", "tags": ["alex_amd_norsk_dance_flo"],
        "generated_at": generated_at,
    }
