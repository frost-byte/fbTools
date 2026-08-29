"""Test that H3 ref2va subject_definitions correctly describe bundle subjects.

When reference bundles substitute for source profile subjects, the
subject_definitions section must describe the BUNDLE subject's appearance
(from their character sheet images and appearance text), not the source
profile subject's label.

The only reference to source subjects in a bundle subject's line should be in
the motion clause: "Their pose, movement, and screen position in the scene
match those of {source_name} in <Video N>."

Slot convention (mirrors SourceProfileClipPrompt):
  Source profile subjects → slots A, B, C, D  (retention="replaced")
  Bundle replacement subjects → slots E, F, G, H  (retention="attribute_transfer",
                                                     _transfer_to_slot = source slot)
"""
from __future__ import annotations

from conftest import import_test_module

prompt_assembler = import_test_module("utils/prompt_assembler.py")
assemble_prompt = prompt_assembler.assemble_prompt


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_subject_definitions(prompt: str) -> str:
    """Return the raw text of the subject_definitions section."""
    lines = prompt.splitlines()
    in_section = False
    collected: list[str] = []
    for line in lines:
        if line.strip() == "subject_definitions:":
            in_section = True
            continue
        if in_section:
            # Next top-level section key ends the block
            if line and not line.startswith(" ") and line.rstrip().endswith(":"):
                break
            collected.append(line)
    return "\n".join(collected)


def _subject_lines(prompt: str) -> list[str]:
    """Return the <Subject N> lines from the subject_definitions section."""
    section = _extract_subject_definitions(prompt)
    return [l.strip() for l in section.splitlines() if l.strip().startswith("<Subject")]


def _video_lines(prompt: str) -> list[str]:
    """Return the <Video N> lines from the subject_definitions section."""
    section = _extract_subject_definitions(prompt)
    return [l.strip() for l in section.splitlines() if l.strip().startswith("<Video")]


def _make_scene_instance(
    source_A: dict,
    source_B: dict,
    bundle_E: dict,  # replaces A
    bundle_F: dict,  # replaces B
) -> dict:
    """Build a scene_instance that mirrors SourceProfileClipPrompt output.

    Source slots (A, B): retention='replaced', _transfer_to_slot set to E/F.
    Bundle slots (E, F): retention='attribute_transfer', _transfer_to_slot set to A/B.
    """
    return {
        "template_id":   "sp_test_profile",
        "template_name": "Test Profile",
        "task_flags":    ["video editing", "reference generation"],
        "scene_synopsis": "",
        "slot_assignments": {
            # Source subjects — motion donors, no Subject label emitted
            "A": {
                "subject_id":             "sp_subj_alice",
                "name":                   source_A["label"],
                "concept_id":             None,
                "character_sheet_images": [],
                "appearance": {
                    "summary":        source_A["label"],
                    "hair":   "", "face": "", "body": "", "default_outfit": "",
                },
                "voice":            {},
                "_cast_retention":  "replaced",
                "_transfer_to_slot": "E",
                "_pronoun_style":   "neutral",
                "_short_name":      "",
            },
            "B": {
                "subject_id":             "sp_subj_bob",
                "name":                   source_B["label"],
                "concept_id":             None,
                "character_sheet_images": [],
                "appearance": {
                    "summary":        source_B["label"],
                    "hair":   "", "face": "", "body": "", "default_outfit": "",
                },
                "voice":            {},
                "_cast_retention":  "replaced",
                "_transfer_to_slot": "F",
                "_pronoun_style":   "neutral",
                "_short_name":      "",
            },
            # Bundle replacement subjects — get Subject labels
            "E": {
                "subject_id":             "bun_elena",
                "name":                   bundle_E["name"],
                "concept_id":             None,
                "character_sheet_images": bundle_E.get("sheets", []),
                "appearance": {
                    "summary":        bundle_E.get("appearance", ""),
                    "hair":           bundle_E.get("hair", ""),
                    "face":           bundle_E.get("face", ""),
                    "body":           bundle_E.get("body", ""),
                    "default_outfit": bundle_E.get("outfit", ""),
                },
                "voice":            {},
                "_cast_retention":  "attribute_transfer",
                "_transfer_to_slot": "A",
                "_pronoun_style":   "neutral",
                "_short_name":      "",
            },
            "F": {
                "subject_id":             "bun_felix",
                "name":                   bundle_F["name"],
                "concept_id":             None,
                "character_sheet_images": bundle_F.get("sheets", []),
                "appearance": {
                    "summary":        bundle_F.get("appearance", ""),
                    "hair":           bundle_F.get("hair", ""),
                    "face":           bundle_F.get("face", ""),
                    "body":           bundle_F.get("body", ""),
                    "default_outfit": bundle_F.get("outfit", ""),
                },
                "voice":            {},
                "_cast_retention":  "attribute_transfer",
                "_transfer_to_slot": "B",
                "_pronoun_style":   "neutral",
                "_short_name":      "",
            },
        },
        "dialogue":         {},
        "outfit_overrides": {},
        "template": {
            "shots": [{"id": "shot_1", "action": "{A} and {B} interact.", "camera": "", "dialogue": None}],
            "environment":        {},
            "style":              "cinematic",
            "overall_soundscape": "",
            "non_diegetic_music": "",
        },
    }


def _make_video_entries(with_source_video: bool = True) -> list[dict]:
    """Source profile video entry — shared by both source subjects."""
    if not with_source_video:
        return []
    return [{
        "subject_id":       "sp_subj_alice",
        "subject_ids":      ["sp_subj_alice", "sp_subj_bob"],
        "video_file":       "/fake/source_profile.mp4",
        "load_params":      {"start_time": 5.0, "duration": 10.0},
        "audio_source":     "none",
        "audio_path":       "",
        "audio_start_time": 0.0,
        "audio_duration":   0.0,
        "audio_retention":  "timbre",
        "audio_role":       "",
        "audio_cache":      "",
    }]


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestBundleSubjectDefinitions:
    """Bundle subjects must be described by their OWN appearance in subject_definitions."""

    def test_bundle_with_full_appearance_shows_bundle_description(self):
        """A bundle with appearance text should use it, not the source label."""
        source_A = {"label": "alice_profile_subject"}
        source_B = {"label": "bob_profile_subject"}
        bundle_E = {
            "name":       "Elena Reyes",
            "appearance": "a young woman with long dark hair and bronze skin",
            "hair":       "long, dark brown waves",
            "face":       "sharp cheekbones, amber eyes",
            "sheets":     [{"file": "elena_ref.jpg", "role": "character sheet"}],
        }
        bundle_F = {
            "name":       "Felix Crane",
            "appearance": "a lean man in his thirties with close-cropped silver hair",
            "hair":       "close-cropped silver",
            "face":       "angular jaw, pale grey eyes",
            "sheets":     [{"file": "felix_ref.jpg", "role": "character sheet"}],
        }

        scene = _make_scene_instance(source_A, source_B, bundle_E, bundle_F)
        result = assemble_prompt(scene, "h3_ref2va", _make_video_entries())
        prompt = result["prompt"]
        subj_lines = _subject_lines(prompt)

        print("\n--- subject_definitions ---")
        print(_extract_subject_definitions(prompt))

        # Two Subject lines (E and F); A and B are "replaced" — no Subject label.
        assert len(subj_lines) == 2, f"Expected 2 Subject lines, got {len(subj_lines)}: {subj_lines}"

        # Elena's line: identity part must use HER appearance description ("young woman
        # with dark hair"), not her name or the source subject's label.
        elena_line = next((l for l in subj_lines if "young woman" in l or "dark hair" in l), None)
        assert elena_line is not None, (
            f"No line containing Elena's appearance ('young woman'/'dark hair') found.\n"
            f"Subject lines: {subj_lines}"
        )
        assert "alice_profile_subject" not in elena_line.split(".")[0], (
            f"Source label 'alice_profile_subject' must NOT appear in Elena's identity part.\n"
            f"Got: {elena_line}"
        )

        # Felix's line: identity part must use HIS appearance description.
        felix_line = next((l for l in subj_lines if "lean man" in l or "silver hair" in l), None)
        assert felix_line is not None, (
            f"No line containing Felix's appearance ('lean man'/'silver hair') found.\n"
            f"Subject lines: {subj_lines}"
        )
        assert "bob_profile_subject" not in felix_line.split(".")[0], (
            f"Source label 'bob_profile_subject' must NOT appear in Felix's identity part.\n"
            f"Got: {felix_line}"
        )

    def test_bundle_without_appearance_uses_name_not_source_label(self):
        """A bundle with NO appearance text should fall back to the bundle's name,
        NOT the source profile subject's label.  The source label is only appropriate
        in the motion clause ('Their pose matches those of {source_name} in <Video N>').

        This tests the fix for the extension.py bug where bun_appear fell back to
        `label` (the source subject's label) instead of `""`.  When bun_appear is
        empty, the assembler falls back to info["name"] (the bundle's own name).
        """
        source_A = {"label": "alice_profile_subject"}
        source_B = {"label": "bob_profile_subject"}
        bundle_E = {
            "name":       "Elena Reyes",
            "appearance": "",   # no appearance text — should use bundle name, not source label
            "sheets":     [{"file": "elena_ref.jpg", "role": "character sheet"}],
        }
        bundle_F = {
            "name":       "Felix Crane",
            "appearance": "",   # no appearance text
            "sheets":     [],
        }

        scene = _make_scene_instance(source_A, source_B, bundle_E, bundle_F)
        result = assemble_prompt(scene, "h3_ref2va", _make_video_entries())
        prompt = result["prompt"]
        subj_lines = _subject_lines(prompt)

        print("\n--- subject_definitions (no appearance text) ---")
        print(_extract_subject_definitions(prompt))

        assert len(subj_lines) == 2, f"Expected 2 Subject lines, got {len(subj_lines)}: {subj_lines}"

        for line in subj_lines:
            # Source profile labels must NOT appear in the Subject identity part.
            # (They MAY appear in the motion clause, which follows a period.)
            identity_part = line.split(".")[0]  # everything before the first period
            assert "alice_profile_subject" not in identity_part, (
                f"Source label 'alice_profile_subject' leaked into Subject identity: {line}"
            )
            assert "bob_profile_subject" not in identity_part, (
                f"Source label 'bob_profile_subject' leaked into Subject identity: {line}"
            )

        # Bundle names should appear (with preserved capitalisation — no lowercase transform)
        assert any("Elena Reyes" in l for l in subj_lines), (
            f"Expected 'Elena Reyes' (name, preserved case) in one Subject line.\n"
            f"Got: {subj_lines}"
        )
        assert any("Felix Crane" in l for l in subj_lines), (
            f"Expected 'Felix Crane' (name, preserved case) in one Subject line.\n"
            f"Got: {subj_lines}"
        )

    def test_source_label_only_in_motion_clause(self):
        """Source profile subject labels should only appear in the motion clause,
        not in the 'is …' description of the bundle subject.
        """
        source_A = {"label": "alice_profile_subject"}
        source_B = {"label": "bob_profile_subject"}
        bundle_E = {
            "name":       "Elena Reyes",
            "appearance": "a young woman with long dark hair",
            "sheets":     [{"file": "elena.jpg", "role": "character sheet"}],
        }
        bundle_F = {
            "name":       "Felix Crane",
            "appearance": "a lean man with silver hair",
            "sheets":     [],
        }

        scene = _make_scene_instance(source_A, source_B, bundle_E, bundle_F)
        result = assemble_prompt(scene, "h3_ref2va", _make_video_entries())
        prompt = result["prompt"]
        subj_lines = _subject_lines(prompt)

        for line in subj_lines:
            parts = line.split(".")
            # First sentence is the identity "is … whose appearance comes from …"
            identity = parts[0]
            # Remaining sentences include the motion clause
            motion = ".".join(parts[1:]) if len(parts) > 1 else ""

            for src_label in ("alice_profile_subject", "bob_profile_subject"):
                assert src_label not in identity, (
                    f"Source label {src_label!r} must NOT appear in identity part.\n"
                    f"Identity: {identity}"
                )
                # Source labels ARE allowed in the motion clause
                # (e.g. "match those of alice_profile_subject in <Video 1>")

    def test_video_line_references_replacement_subjects(self):
        """The <Video N> role line should name the bundle replacement subjects,
        not the replaced source slots.
        """
        source_A = {"label": "alice_profile_subject"}
        source_B = {"label": "bob_profile_subject"}
        bundle_E = {
            "name":       "Elena Reyes",
            "appearance": "a young woman with long dark hair",
            "sheets":     [{"file": "elena.jpg", "role": "character sheet"}],
        }
        bundle_F = {
            "name":       "Felix Crane",
            "appearance": "a lean man with silver hair",
            "sheets":     [],
        }

        scene = _make_scene_instance(source_A, source_B, bundle_E, bundle_F)
        result = assemble_prompt(scene, "h3_ref2va", _make_video_entries())
        prompt = result["prompt"]
        video_lines = _video_lines(prompt)

        print("\n--- video lines ---")
        print("\n".join(video_lines))

        # Source labels must not appear in the video role lines
        assert len(video_lines) >= 1, f"Expected at least one <Video N> line. Got: {video_lines}"
        for line in video_lines:
            assert "alice_profile_subject" not in line, (
                f"Source label should not appear in video role line: {line}"
            )
            assert "bob_profile_subject" not in line, (
                f"Source label should not appear in video role line: {line}"
            )

    def test_no_bundles_shows_source_subjects_normally(self):
        """Without bundles, retained source subjects get Subject labels from their own info."""
        scene_instance = {
            "template_id":   "sp_test",
            "template_name": "Test",
            "task_flags":    [],
            "scene_synopsis": "",
            "slot_assignments": {
                "A": {
                    "subject_id":             "sp_subj_alice",
                    "name":                   "Alice",
                    "concept_id":             None,
                    "character_sheet_images": [],
                    "appearance": {
                        "summary":        "Alice",
                        "hair": "", "face": "", "body": "", "default_outfit": "",
                    },
                    "voice":            {},
                    "_cast_retention":  "fully_preserved",
                    "_pronoun_style":   "neutral",
                    "_short_name":      "",
                },
            },
            "dialogue":         {},
            "outfit_overrides": {},
            "template": {
                "shots": [{"id": "shot_1", "action": "{A} does something.", "camera": "", "dialogue": None}],
                "environment": {}, "style": "", "overall_soundscape": "", "non_diegetic_music": "",
            },
        }
        video_entries = [{
            "subject_id":   "sp_subj_alice",
            "subject_ids":  ["sp_subj_alice"],
            "video_file":   "/fake/source.mp4",
            "load_params":  {"start_time": 0.0, "duration": 10.0},
            "audio_source": "none",
            "audio_path": "", "audio_start_time": 0.0, "audio_duration": 0.0,
            "audio_retention": "timbre", "audio_role": "", "audio_cache": "",
        }]

        result = assemble_prompt(scene_instance, "h3_ref2va", video_entries)
        subj_lines = _subject_lines(result["prompt"])

        print("\n--- no-bundle subject_definitions ---")
        print(_extract_subject_definitions(result["prompt"]))

        assert len(subj_lines) == 1
        # Source subject label is used as appearance_summary (prose), so it gets
        # lowercased to form "is alice from <Video 1>".
        assert "alice" in subj_lines[0].lower()
        assert "<Video 1>" in subj_lines[0]
