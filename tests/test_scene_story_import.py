"""Tests for utils/scene_story_import.py (Scene/Story -> Source Profile plan)."""

from conftest import import_test_module

imp = import_test_module("utils/scene_story_import.py")
scene_prompts = import_test_module("utils/scene_prompts.py")


def _entry(name, order, mask="girl", **kw):
    base = {"scene_name": name, "scene_order": order, "mask_name": mask,
            "prompt_source": "composition", "prompt_key": "img", "custom_prompt": "",
            "video_prompt_source": "auto", "video_prompt_key": "", "video_custom_prompt": ""}
    base.update(kw)
    return base


class Env:
    """Fake scenes on disk: scene -> {exists, base_image, masks, prompts, comps}."""

    def __init__(self, scenes):
        self.scenes = scenes

    def info(self, name):
        s = self.scenes.get(name)
        if s is None:
            return {"exists": False}
        return {"exists": True, "base_image": s.get("base", "base.png"), "masks": s.get("masks", ["girl"])}

    def prompts(self, name):
        s = self.scenes.get(name, {})
        return s.get("prompts", {}), s.get("comps", {})


def _plan(entries, env, **kw):
    return imp.plan_import(
        entries, scene_info=env.info, scene_prompts=env.prompts,
        resolve_video_prompt=scene_prompts.resolve_video_prompt, story_name=kw.pop("story_name", "tale"), **kw,
    )


def _env3():
    return Env({
        "s1": {"comps": {"img": "replace the person"}},
        "s2": {"comps": {"img": "swap again", "vid": "she waves"}},
        "s3": {"comps": {}},
    })


class TestPlan:
    def test_story_becomes_one_image_profile_in_scene_order(self):
        entries = [_entry("s2", 1, video_prompt_source="composition", video_prompt_key="vid"),
                   _entry("s1", 0), _entry("s3", 2)]
        plan = _plan(entries, _env3())
        prof = plan["profile"]
        assert prof["id"] == "sp_story_tale" and prof["media_type"] == "image" and prof["media_dir"] == "output"
        assert [c["label"] for c in prof["clips"]] == ["s1", "s2", "s3"]
        assert [f["filename"] for f in prof["frames"]] == [f"scenes/{s}/base.png" for s in ("s1", "s2", "s3")]
        assert prof["media_filename"] == "scenes/s1/base.png"
        assert prof["origin"] == {"story": "tale"}

    def test_each_clip_selects_its_scene_frame_and_the_mask_subject(self):
        prof = _plan([_entry("s1", 0)], _env3())["profile"]
        clip = prof["clips"][0]
        assert clip["frame_id"] == "frame_s1"
        assert clip["subjects"] == ["subj_girl"]
        assert [s["id"] for s in prof["subjects"]] == ["subj_girl"]
        assert (clip["start_time"], clip["end_time"]) == (0.0, 5.0)

    def test_clips_run_end_to_end_instead_of_overlapping(self):
        clips = _plan([_entry("s1", 0), _entry("s2", 1), _entry("s3", 2)], _env3())["profile"]["clips"]
        assert [(c["start_time"], c["end_time"]) for c in clips] == [(0.0, 5.0), (5.0, 10.0), (10.0, 15.0)]

    def test_reimport_keeps_a_clip_length_the_user_changed(self):
        first = _plan([_entry("s1", 0), _entry("s2", 1)], _env3())["profile"]
        first["clips"][0] = {**first["clips"][0], "end_time": 8.0}
        second = _plan([_entry("s1", 0), _entry("s2", 1)], _env3(), existing_profile=first)["profile"]["clips"]
        assert [(c["start_time"], c["end_time"]) for c in second] == [(0.0, 8.0), (8.0, 13.0)]

    def test_clip_length_option(self):
        clip = _plan([_entry("s1", 0)], _env3(), clip_length=8)["profile"]["clips"][0]
        assert clip["end_time"] == 8.0

    def test_no_bg_mask_is_the_same_subject(self):
        prof = _plan([_entry("s1", 0, mask="girl"), _entry("s2", 1, mask="girl_no_bg")], _env3())["profile"]
        assert [s["id"] for s in prof["subjects"]] == ["subj_girl"]

    def test_action_sources(self):
        entries = [_entry("s1", 0),
                   _entry("s2", 1, video_prompt_source="composition", video_prompt_key="vid"),
                   _entry("s3", 2)]
        plan = _plan(entries, _env3())
        by = {r["scene"]: r["action_source"] for r in plan["scenes"]}
        assert by == {"s1": "image_prompt", "s2": "resolved", "s3": "empty"}
        assert plan["profile"]["clips"][1]["action"] == "she waves"
        assert plan["stats"]["resolved"] == 1 and plan["stats"]["image_prompt"] == 1 and plan["stats"]["empty"] == 1

    def test_loose_scenes_have_no_prompt_and_use_the_first_mask(self):
        env = Env({"a": {"masks": ["boy"]}, "b": {"masks": ["girl_no_bg"]}})
        plan = imp.plan_import(
            imp.loose_scene_entries(["a", "b"]), scene_info=env.info, scene_prompts=env.prompts,
            resolve_video_prompt=scene_prompts.resolve_video_prompt, profile_name="My Set",
        )
        prof = plan["profile"]
        assert prof["id"] == "sp_story_my_set" and prof["name"] == "My Set"
        assert "story" not in prof["origin"]
        assert [c["origin"]["action_source"] for c in prof["clips"]] == ["empty", "empty"]
        assert sorted(s["id"] for s in prof["subjects"]) == ["subj_boy", "subj_girl"]


class TestWarnings:
    def test_missing_scene_and_base_image_are_skipped(self):
        env = Env({"ok": {}, "nobase": {"base": ""}})
        plan = _plan([_entry("ok", 0), _entry("gone", 1), _entry("nobase", 2)], env)
        assert [c["label"] for c in plan["profile"]["clips"]] == ["ok"]
        kinds = {(w["scene"], w["kind"]) for w in plan["warnings"] if w["level"] == "error"}
        assert kinds == {("gone", "scene_missing"), ("nobase", "base_missing")}
        assert plan["stats"]["skipped"] == 2 and plan["stats"]["imported"] == 1

    def test_unresolved_libber_token_warns(self):
        env = Env({"s": {"comps": {"img": "she wears %outfit_x%"}}})
        plan = _plan([_entry("s", 0)], env)
        assert any(w["kind"] == "unresolved_libber" and "%outfit_x%" in w["message"] for w in plan["warnings"])

    def test_duplicate_scene_is_skipped(self):
        plan = _plan([_entry("s1", 0), _entry("s1", 1)], _env3())
        assert len(plan["profile"]["clips"]) == 1
        assert any(w["kind"] == "duplicate_scene" for w in plan["warnings"])

    def test_image_prompt_fallback_and_empty_are_flagged_info(self):
        plan = _plan([_entry("s1", 0), _entry("s3", 1)], _env3())
        kinds = {w["kind"] for w in plan["warnings"] if w["level"] == "info"}
        assert kinds == {"image_prompt", "no_prompt"}


class TestReimport:
    def test_reimport_is_idempotent(self):
        entries = [_entry("s1", 0), _entry("s2", 1)]
        first = _plan(entries, _env3())["profile"]
        second = _plan(entries, _env3(), existing_profile=first)
        assert [c["id"] for c in second["profile"]["clips"]] == [c["id"] for c in first["clips"]]
        assert len(second["profile"]["frames"]) == 2 and len(second["profile"]["subjects"]) == 1
        assert {r["status"] for r in second["scenes"]} == {"updated"}

    def test_manual_action_survives_reimport(self):
        entries = [_entry("s1", 0)]
        first = _plan(entries, _env3())["profile"]
        first["clips"][0]["action"] = "A hand-written action"
        first["clips"][0]["origin"]["action_source"] = "manual"
        plan = _plan(entries, _env3(), existing_profile=first)
        assert plan["profile"]["clips"][0]["action"] == "A hand-written action"
        assert plan["stats"]["kept_manual"] == 1

    def test_unmarked_nonempty_action_is_treated_as_hand_written(self):
        entries = [_entry("s1", 0)]
        first = _plan(entries, _env3())["profile"]
        first["clips"][0]["action"] = "typed in the editor"
        del first["clips"][0]["origin"]["action_source"]
        plan = _plan(entries, _env3(), existing_profile=first)
        assert plan["profile"]["clips"][0]["action"] == "typed in the editor"

    def test_unedited_import_refreshes_from_the_scene(self):
        entries = [_entry("s1", 0)]
        first = _plan(entries, _env3())["profile"]
        env = Env({"s1": {"comps": {"img": "new wording"}}})
        plan = _plan(entries, env, existing_profile=first)
        assert plan["profile"]["clips"][0]["action"] == "new wording"

    def test_other_clips_and_settings_are_left_alone(self):
        existing = {"id": "sp_story_tale", "name": "Renamed", "proxy_short_edge": 576,
                    "clips": [{"id": "mine", "label": "mine", "action": "x"}],
                    "subjects": [{"id": "subj_other", "label": "other"}], "frames": []}
        plan = _plan([_entry("s1", 0)], _env3(), existing_profile=existing)
        prof = plan["profile"]
        assert prof["name"] == "Renamed" and prof["proxy_short_edge"] == 576
        assert [c["id"] for c in prof["clips"]] == ["clip_s1", "mine"]
        assert {s["id"] for s in prof["subjects"]} == {"subj_other", "subj_girl"}

    def test_plan_does_not_mutate_the_existing_profile(self):
        existing = {"id": "sp_story_tale", "clips": [], "subjects": [], "frames": []}
        _plan([_entry("s1", 0)], _env3(), existing_profile=existing)
        assert existing == {"id": "sp_story_tale", "clips": [], "subjects": [], "frames": []}


def test_helpers():
    assert imp.subject_name_from_mask("girl_no_bg") == "girl"
    assert imp.subject_name_from_mask("male") == "male"
    assert imp.profile_id_for("My Tale!") == "sp_story_my_tale"
    assert imp.action_source_for("composition") == "resolved"
    assert imp.action_source_for("image_prompt") == "image_prompt"
    assert imp.action_source_for("empty") == "empty"


# ── disk readers ─────────────────────────────────────────────────────────────────

import json  # noqa: E402


def _write_scene(root, name, base=True, masks=("girl", "girl_no_bg")):
    d = root / name
    d.mkdir(parents=True)
    if base:
        (d / "base.png").write_bytes(b"x")
    (d / "masks.json").write_text(json.dumps({"masks": [{"name": m} for m in masks]}))


def _write_story(root, name, scenes):
    d = root / name
    d.mkdir(parents=True)
    (d / "story.json").write_text(json.dumps({"version": 2, "story_name": name, "story_dir": "/stale/path",
                                              "scenes": scenes}))


def test_make_scene_info_reads_base_and_masks(tmp_path):
    _write_scene(tmp_path, "a")
    _write_scene(tmp_path, "b", base=False, masks=())
    info = imp.make_scene_info(str(tmp_path))
    assert info("a") == {"exists": True, "base_image": "base.png", "masks": ["girl", "girl_no_bg"]}
    assert info("b") == {"exists": True, "base_image": "", "masks": []}
    assert info("nope") == {"exists": False}
    assert info("../a") == {"exists": False}


def test_read_story_entries_orders_by_scene_order_and_ignores_story_dir(tmp_path):
    _write_story(tmp_path, "tale", [{"scene_name": "b", "scene_order": 1}, {"scene_name": "a", "scene_order": 0}])
    assert [e["scene_name"] for e in imp.read_story_entries(str(tmp_path), "tale")] == ["a", "b"]
    assert imp.read_story_entries(str(tmp_path), "missing") == []
    assert imp.read_story_entries(str(tmp_path), "../tale") == []


def test_list_stories_and_loose_scenes(tmp_path):
    scenes, stories = tmp_path / "scenes", tmp_path / "stories"
    scenes.mkdir(), stories.mkdir()
    for n in ("a", "b", "loose1", "loose2"):
        _write_scene(scenes, n, base=(n != "loose2"))
    _write_story(stories, "tale", [{"scene_name": "a", "scene_order": 0}, {"scene_name": "b", "scene_order": 1}])
    (stories / "junk").mkdir()
    listed = imp.list_stories(str(stories))
    assert listed == [{"name": "tale", "scenes": ["a", "b"]}]
    assert imp.list_loose_scenes(str(scenes), listed) == [
        {"name": "loose1", "has_base": True}, {"name": "loose2", "has_base": False}]
    assert imp.list_stories(str(tmp_path / "nowhere")) == []


class TestCleanImageModelPrompt:
    clean = staticmethod(imp.clean_image_model_prompt)

    def test_control_image_notes_are_dropped(self):
        assert self.clean("She waves. image3 provides no meaningful information and should be ignored.") == "She waves."
        assert self.clean("She waves. Picture 3, Picture 4 and Picture 5 provides no meaningful information "
                          "and should be ignored.") == "She waves."

    def test_depth_note_without_a_final_period_is_dropped(self):
        text = ("She waves. image3 provides further depth and orientation for the scene but should not "
                "alter the appearance of the girl from image2")
        assert self.clean(text) == "She waves."

    def test_mask_replace_and_place_instructions_are_dropped(self):
        assert self.clean("A photo-realistic scene. Replace the masked area from image1 with the "
                          "{male|female|girl} from image2. She waves.") == "A photo-realistic scene. She waves."
        assert self.clean("A scene. Place Zed, the girl from Picture 2, in the position and orientation of the "
                          "girl from Picture 1, use the clothing, hair and body of the girl from Picture 2. "
                          "She waves.") == "A scene. She waves."

    def test_named_reference_becomes_slot_a_and_the_trigger_word_follows(self):
        text = "A scene. Zed, the girl from image2, is standing. Later Zed is sitting."
        assert self.clean(text) == "A scene. {A} is standing. Later {A} is sitting."

    def test_reference_at_the_very_start_and_capitalised_picture(self):
        assert self.clean("Zed, the girl from Picture 2, waves.") == "{A} waves."

    def test_plain_reference_without_a_trigger_word(self):
        assert self.clean("She stands behind the woman from image2.") == "She stands behind {A}."

    def test_ordinary_prose_is_untouched(self):
        text = "The girl waves at the camera. Another picture hangs on the wall."
        assert self.clean(text) == text

    def test_only_scaffolding_cleans_to_nothing_useful(self):
        cleaned = self.clean("Replace the masked area from image1 with the girl from image2. Zed, the girl from image2,")
        assert cleaned == "{A}"
        assert not imp.has_action_text(cleaned)
        assert imp.has_action_text("{A} waves")

    def test_leftover_image_references(self):
        assert imp.leftover_image_refs("the man from Picture 3 and image4") == ["image4", "picture3"]
        assert imp.leftover_image_refs("{A} waves") == []


class TestCleaningInThePlan:
    def test_scaffolding_removed_and_counted(self):
        env = Env({"s1": {"comps": {"img": "Zed, the girl from image2, waves. image3 provides no meaningful information "
                                              "and should be ignored."}}})
        res = _plan([_entry("s1", 0)], env)
        assert res["profile"]["clips"][0]["action"] == "{A} waves."
        assert res["stats"]["cleaned"] == 1
        assert res["scenes"][0]["cleaned"] is True

    def test_a_prompt_that_was_all_scaffolding_becomes_empty(self):
        env = Env({"s1": {"comps": {"img": "Replace the masked area from image1 with the girl from image2."}}})
        res = _plan([_entry("s1", 0)], env)
        assert res["profile"]["clips"][0]["action"] == ""
        assert res["scenes"][0]["action_source"] == "empty"
        assert res["stats"]["empty"] == 1

    def test_unhandled_image_reference_is_flagged(self):
        env = Env({"s1": {"comps": {"img": "She waves at the man from image3."}}})
        res = _plan([_entry("s1", 0)], env)
        assert [w["kind"] for w in res["warnings"] if w["kind"] == "image_reference"] == ["image_reference"]


class TestPreamble:
    def test_split_preamble(self):
        assert imp.split_preamble("A photo-realistic scene. {A} waves.") == ("A photo-realistic scene.", "{A} waves.")
        assert imp.split_preamble("a photorealistic scene.   {A} waves.")[1] == "{A} waves."
        assert imp.split_preamble("{A} waves. A photo-realistic scene.") == ("", "{A} waves. A photo-realistic scene.")

    def test_opener_moves_to_the_clip_preamble(self):
        env = Env({"s1": {"comps": {"img": "A photo-realistic scene. Zed, the girl from image2, waves."}}})
        clip = _plan([_entry("s1", 0)], env)["profile"]["clips"][0]
        assert clip["action"] == "{A} waves."
        assert clip["style"] == "A photo-realistic scene."

    def test_no_style_key_without_an_opener(self):
        env = Env({"s1": {"comps": {"img": "She waves."}}})
        assert "style" not in _plan([_entry("s1", 0)], env)["profile"]["clips"][0]

    def test_reimport_keeps_a_preamble_the_user_wrote(self):
        env = Env({"s1": {"comps": {"img": "A photo-realistic scene. She waves."}}})
        first = _plan([_entry("s1", 0)], env)["profile"]
        first["clips"][0]["style"] = "Shot on 35mm film."
        again = _plan([_entry("s1", 0)], env, existing_profile=first)["profile"]["clips"][0]
        assert again["style"] == "Shot on 35mm film."
        assert again["action"] == "She waves."

    def test_scaffolding_only_prompt_leaves_no_preamble(self):
        env = Env({"s1": {"comps": {"img": "A photo-realistic scene. Replace the masked area from image1 with the girl from image2."}}})
        clip = _plan([_entry("s1", 0)], env)["profile"]["clips"][0]
        assert clip["action"] == "" and "style" not in clip
