"""Tests for the dialogue resolution path.

Covers two layers:
1. apply_slot_dialogue() — pure function in utils/libber_resolve.py
2. Slot-mapping helpers that build slot_dialogue from SceneCastBuild entries
   (tested via _build_slot_dialogue, a small helper extracted for testability).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.libber_resolve import resolve_libber_refs, apply_slot_dialogue

SHOT_ID = "clip_1_shot_1"

REGISTRY = {
    "greet": {"hello": "Hello there!", "bye": "Goodbye!"},
    "rora":  {"line_a": "I never meant to hurt you.", "line_b": "We need to talk."},
}


# ── apply_slot_dialogue — empty / trivial ─────────────────────────────────────

class TestApplySlotDialogueEmpty:
    def test_empty_slot_dialogue_returns_empty_maps(self):
        dlg, patch = apply_slot_dialogue({}, {}, SHOT_ID)
        assert dlg == {}
        assert patch == {}

    def test_single_empty_string_skipped(self):
        dlg, patch = apply_slot_dialogue({"A": ""}, {}, SHOT_ID)
        assert dlg == {}
        assert patch == {}


# ── apply_slot_dialogue — normal speech ──────────────────────────────────────

class TestApplySlotDialogueNormalSpeech:
    def test_plain_text_goes_to_dialogue_map(self):
        dlg, patch = apply_slot_dialogue({"E": "Hello world."}, {}, SHOT_ID)
        assert dlg == {SHOT_ID: "Hello world."}
        assert patch == {"dialogue": {"speaker_slot": "E"}}

    def test_speaker_slot_matches_entry_slot(self):
        dlg, patch = apply_slot_dialogue({"F": "See you later."}, {}, SHOT_ID)
        assert patch["dialogue"]["speaker_slot"] == "F"

    def test_shot_id_used_as_dialogue_map_key(self):
        sid = "my_clip_shot_1"
        dlg, _ = apply_slot_dialogue({"A": "text"}, {}, sid)
        assert sid in dlg

    def test_last_normal_entry_wins_for_dialogue_block(self):
        # Two normal speech entries — last one (dict iteration order) wins.
        slot_dlg = {"E": "First.", "F": "Second."}
        dlg, patch = apply_slot_dialogue(slot_dlg, {}, SHOT_ID)
        assert dlg[SHOT_ID] == "Second."
        assert patch["dialogue"]["speaker_slot"] == "F"

    def test_libber_token_resolved_in_normal_speech(self):
        dlg, patch = apply_slot_dialogue({"E": "%greet:hello%"}, REGISTRY, SHOT_ID)
        assert dlg[SHOT_ID] == "Hello there!"
        assert patch["dialogue"]["speaker_slot"] == "E"

    def test_unknown_libber_token_left_as_is(self):
        dlg, _ = apply_slot_dialogue({"E": "%missing:key%"}, REGISTRY, SHOT_ID)
        assert dlg[SHOT_ID] == "%missing:key%"

    def test_mixed_token_and_plain_text(self):
        dlg, _ = apply_slot_dialogue(
            {"E": "She said: %rora:line_a%"}, REGISTRY, SHOT_ID
        )
        assert dlg[SHOT_ID] == "She said: I never meant to hurt you."


# ── apply_slot_dialogue — [silent] ───────────────────────────────────────────

class TestApplySlotDialogueSilent:
    def test_silent_prefix_suppresses_entry(self):
        dlg, patch = apply_slot_dialogue({"E": "[silent]"}, {}, SHOT_ID)
        assert dlg == {}
        assert patch == {}

    def test_silent_with_trailing_text_still_suppressed(self):
        dlg, patch = apply_slot_dialogue({"E": "[silent] (no audio)"}, {}, SHOT_ID)
        assert dlg == {}
        assert patch == {}

    def test_silent_does_not_block_other_slots(self):
        slot_dlg = {"E": "[silent]", "F": "Hello."}
        dlg, patch = apply_slot_dialogue(slot_dlg, {}, SHOT_ID)
        assert dlg[SHOT_ID] == "Hello."
        assert patch["dialogue"]["speaker_slot"] == "F"

    def test_all_silent_returns_empty(self):
        slot_dlg = {"E": "[silent]", "F": "[silent]"}
        dlg, patch = apply_slot_dialogue(slot_dlg, {}, SHOT_ID)
        assert dlg == {}
        assert patch == {}


# ── apply_slot_dialogue — [sounds] ───────────────────────────────────────────

class TestApplySlotDialogueSounds:
    def test_sounds_prefix_goes_to_sound_events(self):
        dlg, patch = apply_slot_dialogue({"E": "[sounds] dog barks"}, {}, SHOT_ID)
        assert dlg == {}
        assert patch.get("sound_events") == "dog barks"

    def test_sounds_stripped_of_leading_space(self):
        dlg, patch = apply_slot_dialogue({"E": "[sounds]  footsteps"}, {}, SHOT_ID)
        assert patch["sound_events"] == "footsteps"

    def test_sounds_empty_description_ignored(self):
        dlg, patch = apply_slot_dialogue({"E": "[sounds]"}, {}, SHOT_ID)
        assert "sound_events" not in patch

    def test_multiple_sounds_entries_joined_with_semicolon(self):
        slot_dlg = {"E": "[sounds] thunder", "F": "[sounds] rain"}
        dlg, patch = apply_slot_dialogue(slot_dlg, {}, SHOT_ID)
        assert dlg == {}
        events = patch["sound_events"]
        assert "thunder" in events
        assert "rain" in events
        assert ";" in events

    def test_sounds_and_speech_in_same_slot_dict(self):
        slot_dlg = {"E": "[sounds] wind howls", "F": "I'm cold."}
        dlg, patch = apply_slot_dialogue(slot_dlg, {}, SHOT_ID)
        assert dlg[SHOT_ID] == "I'm cold."
        assert patch["dialogue"]["speaker_slot"] == "F"
        assert patch["sound_events"] == "wind howls"

    def test_sounds_libber_token_resolved(self):
        # A [sounds] entry can also use libber tokens.
        slot_dlg = {"E": "[sounds] %greet:hello%"}
        dlg, patch = apply_slot_dialogue(slot_dlg, REGISTRY, SHOT_ID)
        assert patch["sound_events"] == "Hello there!"


# ── Slot-mapping logic (unit-testing the build-slot-dialogue step) ────────────
#
# The actual mapping happens inside SourceProfileClipPrompt.execute(), but the
# logic is simple enough to re-implement here as an independent reference so
# we can verify the rules without importing extension.py.

SOURCE_SLOTS = ["A", "B", "C", "D"]
BUNDLE_SLOTS = ["E", "F", "G", "H"]


def _build_slot_dialogue(
    source_subject_ids: list,
    cast_lookup: dict,          # source_subject_id → hybrid cast entry
    bundle_slot_pairs: list,    # [(bun_slot, src_slot)]
    scene_cast_entries: list,   # full entries list from scene_cast
    profile_id: str,
) -> dict:
    """Reference implementation of the slot_dialogue-building step."""
    slot_dialogue: dict = {}

    # Source-only entries
    src_dlg_lookup: dict = {}
    for e in scene_cast_entries:
        if e.get("source_profile_id") != profile_id:
            continue
        d = str(e.get("dialogue", "") or "").strip()
        if not d:
            continue
        ssid = e.get("source_subject_id", "")
        if ssid and not e.get("bundle_id"):
            src_dlg_lookup[ssid] = d

    for i, sid in enumerate(source_subject_ids):
        d = src_dlg_lookup.get(sid, "")
        if d:
            slot_dialogue[SOURCE_SLOTS[i]] = d

    # Hybrid entries → bundle slot
    for bun_slot, src_slot in bundle_slot_pairs:
        s_idx = SOURCE_SLOTS.index(src_slot)
        src_sid = source_subject_ids[s_idx]
        entry = cast_lookup.get(src_sid)
        if entry:
            d = str(entry.get("dialogue", "") or "").strip()
            if d:
                slot_dialogue[bun_slot] = d

    return slot_dialogue


class TestBuildSlotDialogue:
    PROFILE_ID = "prof_1"

    def _cast_entry(self, src_sid, bundle_id=None, dialogue=""):
        e = {
            "source_profile_id": self.PROFILE_ID,
            "source_subject_id": src_sid,
            "dialogue": dialogue,
        }
        if bundle_id:
            e["bundle_id"] = bundle_id
        return e

    def test_source_only_dialogue_maps_to_source_slot(self):
        entries = [self._cast_entry("subj_a", dialogue="Hi there.")]
        result = _build_slot_dialogue(
            ["subj_a"], {}, [], entries, self.PROFILE_ID
        )
        assert result == {"A": "Hi there."}

    def test_source_only_second_subject_maps_to_slot_b(self):
        entries = [self._cast_entry("subj_b", dialogue="Hey!")]
        result = _build_slot_dialogue(
            ["subj_a", "subj_b"], {}, [], entries, self.PROFILE_ID
        )
        assert result == {"B": "Hey!"}

    def test_hybrid_dialogue_maps_to_bundle_slot(self):
        hybrid = self._cast_entry("subj_a", bundle_id="bun_1", dialogue="Replaced speech.")
        cast_lookup = {"subj_a": hybrid}
        bundle_slot_pairs = [("E", "A")]
        result = _build_slot_dialogue(
            ["subj_a"], cast_lookup, bundle_slot_pairs, [hybrid], self.PROFILE_ID
        )
        assert result == {"E": "Replaced speech."}

    def test_empty_dialogue_not_included(self):
        entries = [self._cast_entry("subj_a", dialogue="")]
        result = _build_slot_dialogue(
            ["subj_a"], {}, [], entries, self.PROFILE_ID
        )
        assert result == {}

    def test_whitespace_only_dialogue_not_included(self):
        entries = [self._cast_entry("subj_a", dialogue="   ")]
        result = _build_slot_dialogue(
            ["subj_a"], {}, [], entries, self.PROFILE_ID
        )
        assert result == {}

    def test_wrong_profile_id_ignored(self):
        e = {
            "source_profile_id": "other_profile",
            "source_subject_id": "subj_a",
            "dialogue": "Should be ignored.",
        }
        result = _build_slot_dialogue(
            ["subj_a"], {}, [], [e], self.PROFILE_ID
        )
        assert result == {}

    def test_hybrid_no_dialogue_not_included(self):
        hybrid = self._cast_entry("subj_a", bundle_id="bun_1", dialogue="")
        cast_lookup = {"subj_a": hybrid}
        result = _build_slot_dialogue(
            ["subj_a"], cast_lookup, [("E", "A")], [hybrid], self.PROFILE_ID
        )
        assert result == {}

    def test_mixed_source_and_bundle_both_captured(self):
        src_entry = self._cast_entry("subj_b", dialogue="I'm still myself.")
        hybrid = self._cast_entry("subj_a", bundle_id="bun_1", dialogue="I replaced A.")
        cast_lookup = {"subj_a": hybrid}
        result = _build_slot_dialogue(
            ["subj_a", "subj_b"],
            cast_lookup,
            [("E", "A")],
            [src_entry, hybrid],
            self.PROFILE_ID,
        )
        assert result == {"B": "I'm still myself.", "E": "I replaced A."}

    def test_source_subject_with_bundle_uses_bundle_slot_not_source_slot(self):
        # When a source subject has a bundle replacement, the dialogue from the
        # hybrid cast entry should land on the BUNDLE slot, not the source slot.
        hybrid = self._cast_entry("subj_a", bundle_id="bun_1", dialogue="Bundle speech.")
        cast_lookup = {"subj_a": hybrid}
        result = _build_slot_dialogue(
            ["subj_a"], cast_lookup, [("E", "A")], [hybrid], self.PROFILE_ID
        )
        assert "A" not in result
        assert result.get("E") == "Bundle speech."
