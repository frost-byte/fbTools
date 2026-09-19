"""Tests for apply_cast_to_subjects — cast bundle enrichment of resolved subjects."""

from conftest import import_test_module

pc = import_test_module("utils/prompt_compositions.py")
apply_cast_to_subjects = pc.apply_cast_to_subjects
resolve_subjects       = pc.resolve_subjects


# ── Helpers ────────────────────────────────────────────────────────────────────

def _subject(name="Alice", summary="a tall woman", audio="", sheets=None):
    return {
        "name": name,
        "subject_id": name.lower(),
        "appearance": {"summary": summary},
        "voice": {"audio_reference_file": audio, "description": "", "language": "en-us"},
        "character_sheet_images": list(sheets or []),
        "concept_id": "",
    }


def _composition(slot_subjects):
    """slot_subjects: {slot_key: subject_id}"""
    return {"subjects": slot_subjects}


def _cast(entries):
    return {"entries": entries}


def _entry(subject_id, bundle_id, visual_mode="images", use_audio=False):
    return {
        "subject_id": subject_id,
        "bundle_id": bundle_id,
        "visual_mode": visual_mode,
        "use_audio": use_audio,
    }


class _BundleRegistry:
    """Minimal duck-typed bundle registry for tests."""
    def __init__(self, bundles):
        self._b = bundles

    def get(self, bundle_id):
        return self._b.get(bundle_id)


def _bundle(
    visual_type="images",
    files=None,
    video_file="",
    audio_source="none",
    audio_file="",
    appearance_override="",
):
    return {
        "visual": {
            "type": visual_type,
            "files": list(files or []),
            "file": video_file,
        },
        "audio": {
            "source": audio_source,
            "file": audio_file,
        },
        "appearance_override": appearance_override,
    }


# ── Image-mode files → character_sheet_images ──────────────────────────────────

def test_image_files_appended_to_sheets():
    subj = _subject("Alice")
    comp = _composition({"A": "alice"})
    cast = _cast([_entry("alice", "b1", visual_mode="images")])
    reg  = _BundleRegistry({"b1": _bundle(files=["a1.png", "a2.png"])})
    result = apply_cast_to_subjects({"A": subj}, comp, cast, reg)
    assert result["A"]["character_sheet_images"] == ["a1.png", "a2.png"]


def test_image_files_appended_to_existing_sheets():
    subj = _subject("Alice", sheets=["profile.png"])
    comp = _composition({"A": "alice"})
    cast = _cast([_entry("alice", "b1", visual_mode="images")])
    reg  = _BundleRegistry({"b1": _bundle(files=["ref.png"])})
    result = apply_cast_to_subjects({"A": subj}, comp, cast, reg)
    assert result["A"]["character_sheet_images"] == ["profile.png", "ref.png"]


def test_image_files_deduplicated():
    subj = _subject("Alice", sheets=["same.png"])
    comp = _composition({"A": "alice"})
    cast = _cast([_entry("alice", "b1", visual_mode="images")])
    reg  = _BundleRegistry({"b1": _bundle(files=["same.png", "other.png"])})
    result = apply_cast_to_subjects({"A": subj}, comp, cast, reg)
    assert result["A"]["character_sheet_images"] == ["same.png", "other.png"]


def test_video_mode_entry_does_not_add_sheets():
    subj = _subject("Alice")
    comp = _composition({"A": "alice"})
    cast = _cast([_entry("alice", "b1", visual_mode="video")])
    reg  = _BundleRegistry({"b1": _bundle(video_file="alice.mp4")})
    result = apply_cast_to_subjects({"A": subj}, comp, cast, reg)
    assert result["A"]["character_sheet_images"] == []


# ── Audio → voice.audio_reference_file ────────────────────────────────────────

def test_audio_file_source_sets_reference():
    subj = _subject("Alice")
    comp = _composition({"A": "alice"})
    cast = _cast([_entry("alice", "b1", use_audio=True)])
    reg  = _BundleRegistry({"b1": _bundle(audio_source="file", audio_file="alice.wav")})
    result = apply_cast_to_subjects({"A": subj}, comp, cast, reg)
    assert result["A"]["voice"]["audio_reference_file"] == "alice.wav"


def test_audio_extract_from_visual_does_not_set_voice_file():
    # extract_from_visual is a VIDEO SOUNDTRACK — handled at video-entry level in
    # the refplan; must NOT also appear as a standalone voice.audio_reference_file.
    subj = _subject("Alice")
    comp = _composition({"A": "alice"})
    cast = _cast([_entry("alice", "b1", visual_mode="video", use_audio=True)])
    reg  = _BundleRegistry({"b1": _bundle(
        video_file="alice_clip.mp4",
        audio_source="extract_from_visual",
    )})
    result = apply_cast_to_subjects({"A": subj}, comp, cast, reg)
    assert result["A"]["voice"]["audio_reference_file"] == ""


def test_use_audio_false_does_not_set_reference():
    subj = _subject("Alice")
    comp = _composition({"A": "alice"})
    cast = _cast([_entry("alice", "b1", use_audio=False)])
    reg  = _BundleRegistry({"b1": _bundle(audio_source="file", audio_file="alice.wav")})
    result = apply_cast_to_subjects({"A": subj}, comp, cast, reg)
    assert result["A"]["voice"]["audio_reference_file"] == ""


def test_audio_source_none_does_not_set_reference():
    subj = _subject("Alice")
    comp = _composition({"A": "alice"})
    cast = _cast([_entry("alice", "b1", use_audio=True)])
    reg  = _BundleRegistry({"b1": _bundle(audio_source="none")})
    result = apply_cast_to_subjects({"A": subj}, comp, cast, reg)
    assert result["A"]["voice"]["audio_reference_file"] == ""


def test_audio_file_source_carries_timing_and_role():
    subj = _subject("Alice")
    comp = _composition({"A": "alice"})
    cast = _cast([_entry("alice", "b1", use_audio=True)])
    reg  = _BundleRegistry({"b1": _bundle(
        audio_source="file",
        audio_file="alice.wav",
        # bundle also has retention/role/timing
    )})
    # Manually inject timing fields into the bundle dict for this test
    bundle = reg.get("b1")
    bundle["audio"]["start_time"] = 2.5
    bundle["audio"]["duration"]   = 8.0
    bundle["audio"]["retention"]  = "reuse"
    bundle["audio"]["role"]       = "dialogue track"
    result = apply_cast_to_subjects({"A": subj}, comp, cast, reg)
    v = result["A"]["voice"]
    assert v["audio_reference_file"] == "alice.wav"
    assert v["audio_start_time"]     == 2.5
    assert v["audio_duration"]       == 8.0
    assert v["audio_retention"]      == "reuse"
    assert v["audio_role"]           == "dialogue track"


# ── Appearance override ────────────────────────────────────────────────────────

def test_appearance_override_replaces_summary():
    subj = _subject("Alice", summary="a tall woman")
    comp = _composition({"A": "alice"})
    cast = _cast([_entry("alice", "b1")])
    reg  = _BundleRegistry({"b1": _bundle(appearance_override="the girl from this clip")})
    result = apply_cast_to_subjects({"A": subj}, comp, cast, reg)
    assert result["A"]["appearance"]["summary"] == "the girl from this clip"


def test_empty_appearance_override_leaves_summary_intact():
    subj = _subject("Alice", summary="a tall woman")
    comp = _composition({"A": "alice"})
    cast = _cast([_entry("alice", "b1")])
    reg  = _BundleRegistry({"b1": _bundle(appearance_override="")})
    result = apply_cast_to_subjects({"A": subj}, comp, cast, reg)
    assert result["A"]["appearance"]["summary"] == "a tall woman"


# ── Safety and edge cases ──────────────────────────────────────────────────────

def test_originals_not_mutated():
    subj = _subject("Alice")
    original_sheets = subj["character_sheet_images"]
    comp = _composition({"A": "alice"})
    cast = _cast([_entry("alice", "b1")])
    reg  = _BundleRegistry({"b1": _bundle(files=["new.png"])})
    apply_cast_to_subjects({"A": subj}, comp, cast, reg)
    assert original_sheets == []  # original subject unchanged


def test_missing_bundle_skipped_gracefully():
    subj = _subject("Alice")
    comp = _composition({"A": "alice"})
    cast = _cast([_entry("alice", "no_such_bundle")])
    reg  = _BundleRegistry({})
    result = apply_cast_to_subjects({"A": subj}, comp, cast, reg)
    assert result["A"]["character_sheet_images"] == []


def test_entry_with_unmatched_subject_id_is_ignored():
    # Cast entries bind by subject identity, not array position/order — an
    # entry for "bob" against a composition that only has "alice" in A has
    # nothing to bind to and must be a no-op. This is the fix for the bug
    # where positional matching applied a mismatched entry to whatever slot
    # happened to sort into that array index, regardless of who was there.
    subj = _subject("Alice")
    comp = _composition({"A": "alice"})
    cast = _cast([_entry("bob", "b1")])
    reg  = _BundleRegistry({"b1": _bundle(files=["bob.png"])})
    result = apply_cast_to_subjects({"A": subj}, comp, cast, reg)
    assert result["A"]["character_sheet_images"] == []


def test_blank_entry_is_passthrough():
    # An entry with no subject_id keeps the composition's original subject
    # unchanged; a later entry still binds by subject_id, independent of the
    # blank entry's position in the array.
    alice = _subject("Alice")
    joe   = _subject("Joe")
    comp  = _composition({"A": "alice", "B": "joe"})
    cast  = _cast([
        {"subject_id": "", "bundle_id": "", "visual_mode": "images", "use_audio": False},
        _entry("joe", "b2", visual_mode="images"),
    ])
    breg  = _BundleRegistry({"b2": _bundle(files=["joe_ref.png"])})
    result = apply_cast_to_subjects({"A": alice, "B": joe}, comp, cast, breg)
    # A untouched (blank row)
    assert result["A"]["name"] == "Alice"
    assert result["A"]["character_sheet_images"] == []
    # B enriched — matched by subject_id "joe", not by array position
    assert result["B"]["name"] == "Joe"
    assert result["B"]["character_sheet_images"] == ["joe_ref.png"]


def test_matching_is_order_independent():
    # Entries listed out of composition-slot order still bind to the correct
    # slot, since matching is by subject_id, not by sorted-slot-index.
    alice = _subject("Alice")
    bob   = _subject("Bob")
    comp  = _composition({"A": "alice", "B": "bob"})
    cast  = _cast([
        _entry("bob",   "bj", visual_mode="images"),   # listed first, targets B
        _entry("alice", "ba", visual_mode="images"),   # listed second, targets A
    ])
    breg = _BundleRegistry({
        "ba": _bundle(files=["a.png"]),
        "bj": _bundle(files=["j.png"]),
    })
    result = apply_cast_to_subjects({"A": alice, "B": bob}, comp, cast, breg)
    assert result["A"]["character_sheet_images"] == ["a.png"]
    assert result["B"]["character_sheet_images"] == ["j.png"]


# ── Hybrid entries (bundle replacing a subject that also has a source video) ───

def _hybrid_entry(subject_id, bundle_id, source_profile_id="sp1", source_subject_id="src1",
                   role_description="", retention="partially_preserved"):
    return {
        "subject_id":        subject_id,
        "bundle_id":         bundle_id,
        "source_profile_id": source_profile_id,
        "source_subject_id": source_subject_id,
        "role_description":  role_description,
        "retention":         retention,
        "visual_mode":       "images",
        "use_audio":         False,
    }


def test_hybrid_entry_keeps_bundle_appearance_and_images():
    # Previously the hybrid branch never inspected bundle_id at all, so the
    # bundle's appearance/images were silently dropped from the subject list
    # used for prompt text (even though media wiring elsewhere still used the
    # bundle correctly) — refplan and prompt text ended up describing
    # different subjects for the same entry. Confirm the bundle's own data
    # now survives onto its minted slot.
    comp = _composition({"A": "alice"})
    cast = _cast([_hybrid_entry("alice", "b1", role_description="a woman in the video")])
    breg = _BundleRegistry({"b1": _bundle(files=["bundle_ref.png"], appearance_override="a redheaded woman")})
    result = apply_cast_to_subjects({"A": _subject("Alice")}, comp, cast, breg)
    assert "A_bundle" in result
    assert result["A_bundle"]["character_sheet_images"] == ["bundle_ref.png"]
    assert result["A_bundle"]["appearance"]["summary"] == "a redheaded woman"


def test_hybrid_entry_pairs_donor_and_replacement_retention_markers():
    # The original (source-derived) slot becomes the motion donor, marked
    # "replaced" and pointing at the new bundle slot; the bundle slot is
    # marked "attribute_transfer" and points back — mirroring
    # SourceProfileClipPrompt's SOURCE_SLOTS/BUNDLE_SLOTS pairing so the
    # retention_marker-gated prompt_assembler features fire for compositions
    # the same way they do for Source Profile clips.
    comp = _composition({"A": "alice"})
    cast = _cast([_hybrid_entry("alice", "b1")])
    breg = _BundleRegistry({"b1": _bundle(files=["ref.png"])})
    result = apply_cast_to_subjects({"A": _subject("Alice")}, comp, cast, breg)
    assert result["A"]["_cast_retention"] == "replaced"
    assert result["A"]["_transfer_to_slot"] == "A_bundle"
    assert result["A_bundle"]["_cast_retention"] == "attribute_transfer"
    assert result["A_bundle"]["_transfer_to_slot"] == "A"


def test_source_derived_entry_without_bundle_has_no_transfer_slot():
    # A pure source-derived entry (no bundle_id) has nothing to pair with —
    # it should carry the entry's own retention value and no transfer slot.
    comp = _composition({"A": "alice"})
    cast = _cast([{
        "subject_id": "alice", "bundle_id": "", "source_profile_id": "sp1",
        "source_subject_id": "src1", "role_description": "a woman",
        "retention": "fully_preserved", "visual_mode": "images", "use_audio": False,
    }])
    result = apply_cast_to_subjects({"A": _subject("Alice")}, comp, cast, _BundleRegistry({}))
    assert result["A"]["_cast_retention"] == "fully_preserved"
    assert "_transfer_to_slot" not in result["A"]
    assert "A_bundle" not in result


def test_extra_cast_entries_beyond_slot_count_ignored():
    subj = _subject("Alice")
    comp = _composition({"A": "alice"})
    cast = _cast([
        _entry("angie", "ba"),
        _entry("joe",   "bj"),  # position 1 — no B exists
    ])
    breg = _BundleRegistry({"ba": _bundle(files=["a.png"]), "bj": _bundle(files=["j.png"])})
    result = apply_cast_to_subjects({"A": subj}, comp, cast, breg)
    assert "B" not in result


def test_resolve_subjects_injects_subject_id_from_live_registry():
    """subject_id must be present so _build_ref_map can match video_entries_full."""
    class _Reg:
        def get_subject(self, sid):
            return {"name": sid, "appearance": {"summary": "summary"}, "character_sheet_images": []}
    comp = {"subjects": {"A": "alice", "B": "bob"}, "_subject_snapshots": {}}
    result = resolve_subjects(comp, _Reg())
    assert result["A"]["subject_id"] == "alice"
    assert result["B"]["subject_id"] == "bob"


def test_resolve_subjects_injects_subject_id_from_snapshot():
    """Snapshot path must also carry subject_id for video matching."""
    comp = {
        "subjects": {"A": "alice"},
        "_subject_snapshots": {"A": {"name": "alice", "appearance": {"summary": "s"}}},
    }
    result = resolve_subjects(comp, subject_registry=None)
    assert result["A"]["subject_id"] == "alice"


def test_resolve_subjects_preserves_existing_subject_id_in_snapshot():
    """If snapshot already has subject_id (from SubjectProfileLoad), don't overwrite it."""
    comp = {
        "subjects": {"A": "alice"},
        "_subject_snapshots": {"A": {"name": "alice", "subject_id": "alice_v2",
                                      "appearance": {"summary": "s"}}},
    }
    result = resolve_subjects(comp, subject_registry=None)
    assert result["A"]["subject_id"] == "alice_v2"


def test_multiple_subjects_each_enriched_independently():
    alice = _subject("Alice")
    bob   = _subject("Bob")
    comp  = _composition({"A": "alice", "B": "bob"})
    cast  = _cast([
        _entry("alice", "ba", visual_mode="images", use_audio=True),
        _entry("bob",   "bb", visual_mode="images"),
    ])
    reg = _BundleRegistry({
        "ba": _bundle(files=["a.png"], audio_source="file", audio_file="a.wav"),
        "bb": _bundle(files=["b.png"], appearance_override="the big guy"),
    })
    result = apply_cast_to_subjects({"A": alice, "B": bob}, comp, cast, reg)
    assert result["A"]["character_sheet_images"] == ["a.png"]
    assert result["A"]["voice"]["audio_reference_file"] == "a.wav"
    assert result["B"]["character_sheet_images"] == ["b.png"]
    assert result["B"]["appearance"]["summary"] == "the big guy"
    # Alice's appearance not touched
    assert result["A"]["appearance"]["summary"] == "a tall woman"
