"""Tests for apply_cast_to_subjects — cast bundle enrichment of resolved subjects."""

from conftest import import_test_module

pc = import_test_module("utils/prompt_compositions.py")
apply_cast_to_subjects = pc.apply_cast_to_subjects


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
    comp = _composition({"S1": "alice"})
    cast = _cast([_entry("alice", "b1", visual_mode="images")])
    reg  = _BundleRegistry({"b1": _bundle(files=["a1.png", "a2.png"])})
    result = apply_cast_to_subjects({"S1": subj}, comp, cast, reg)
    assert result["S1"]["character_sheet_images"] == ["a1.png", "a2.png"]


def test_image_files_appended_to_existing_sheets():
    subj = _subject("Alice", sheets=["profile.png"])
    comp = _composition({"S1": "alice"})
    cast = _cast([_entry("alice", "b1", visual_mode="images")])
    reg  = _BundleRegistry({"b1": _bundle(files=["ref.png"])})
    result = apply_cast_to_subjects({"S1": subj}, comp, cast, reg)
    assert result["S1"]["character_sheet_images"] == ["profile.png", "ref.png"]


def test_image_files_deduplicated():
    subj = _subject("Alice", sheets=["same.png"])
    comp = _composition({"S1": "alice"})
    cast = _cast([_entry("alice", "b1", visual_mode="images")])
    reg  = _BundleRegistry({"b1": _bundle(files=["same.png", "other.png"])})
    result = apply_cast_to_subjects({"S1": subj}, comp, cast, reg)
    assert result["S1"]["character_sheet_images"] == ["same.png", "other.png"]


def test_video_mode_entry_does_not_add_sheets():
    subj = _subject("Alice")
    comp = _composition({"S1": "alice"})
    cast = _cast([_entry("alice", "b1", visual_mode="video")])
    reg  = _BundleRegistry({"b1": _bundle(video_file="alice.mp4")})
    result = apply_cast_to_subjects({"S1": subj}, comp, cast, reg)
    assert result["S1"]["character_sheet_images"] == []


# ── Audio → voice.audio_reference_file ────────────────────────────────────────

def test_audio_file_source_sets_reference():
    subj = _subject("Alice")
    comp = _composition({"S1": "alice"})
    cast = _cast([_entry("alice", "b1", use_audio=True)])
    reg  = _BundleRegistry({"b1": _bundle(audio_source="file", audio_file="alice.wav")})
    result = apply_cast_to_subjects({"S1": subj}, comp, cast, reg)
    assert result["S1"]["voice"]["audio_reference_file"] == "alice.wav"


def test_audio_extract_from_visual_uses_video_file():
    subj = _subject("Alice")
    comp = _composition({"S1": "alice"})
    cast = _cast([_entry("alice", "b1", visual_mode="video", use_audio=True)])
    reg  = _BundleRegistry({"b1": _bundle(
        video_file="alice_clip.mp4",
        audio_source="extract_from_visual",
    )})
    result = apply_cast_to_subjects({"S1": subj}, comp, cast, reg)
    assert result["S1"]["voice"]["audio_reference_file"] == "alice_clip.mp4"


def test_use_audio_false_does_not_set_reference():
    subj = _subject("Alice")
    comp = _composition({"S1": "alice"})
    cast = _cast([_entry("alice", "b1", use_audio=False)])
    reg  = _BundleRegistry({"b1": _bundle(audio_source="file", audio_file="alice.wav")})
    result = apply_cast_to_subjects({"S1": subj}, comp, cast, reg)
    assert result["S1"]["voice"]["audio_reference_file"] == ""


def test_audio_source_none_does_not_set_reference():
    subj = _subject("Alice")
    comp = _composition({"S1": "alice"})
    cast = _cast([_entry("alice", "b1", use_audio=True)])
    reg  = _BundleRegistry({"b1": _bundle(audio_source="none")})
    result = apply_cast_to_subjects({"S1": subj}, comp, cast, reg)
    assert result["S1"]["voice"]["audio_reference_file"] == ""


# ── Appearance override ────────────────────────────────────────────────────────

def test_appearance_override_replaces_summary():
    subj = _subject("Alice", summary="a tall woman")
    comp = _composition({"S1": "alice"})
    cast = _cast([_entry("alice", "b1")])
    reg  = _BundleRegistry({"b1": _bundle(appearance_override="the girl from this clip")})
    result = apply_cast_to_subjects({"S1": subj}, comp, cast, reg)
    assert result["S1"]["appearance"]["summary"] == "the girl from this clip"


def test_empty_appearance_override_leaves_summary_intact():
    subj = _subject("Alice", summary="a tall woman")
    comp = _composition({"S1": "alice"})
    cast = _cast([_entry("alice", "b1")])
    reg  = _BundleRegistry({"b1": _bundle(appearance_override="")})
    result = apply_cast_to_subjects({"S1": subj}, comp, cast, reg)
    assert result["S1"]["appearance"]["summary"] == "a tall woman"


# ── Safety and edge cases ──────────────────────────────────────────────────────

def test_originals_not_mutated():
    subj = _subject("Alice")
    original_sheets = subj["character_sheet_images"]
    comp = _composition({"S1": "alice"})
    cast = _cast([_entry("alice", "b1")])
    reg  = _BundleRegistry({"b1": _bundle(files=["new.png"])})
    apply_cast_to_subjects({"S1": subj}, comp, cast, reg)
    assert original_sheets == []  # original subject unchanged


def test_missing_bundle_skipped_gracefully():
    subj = _subject("Alice")
    comp = _composition({"S1": "alice"})
    cast = _cast([_entry("alice", "no_such_bundle")])
    reg  = _BundleRegistry({})
    result = apply_cast_to_subjects({"S1": subj}, comp, cast, reg)
    assert result["S1"]["character_sheet_images"] == []


def test_unmatched_subject_id_skipped():
    subj = _subject("Alice")
    comp = _composition({"S1": "alice"})
    cast = _cast([_entry("bob", "b1")])  # "bob" not in composition
    reg  = _BundleRegistry({"b1": _bundle(files=["bob.png"])})
    result = apply_cast_to_subjects({"S1": subj}, comp, cast, reg)
    assert result["S1"]["character_sheet_images"] == []


def test_multiple_subjects_each_enriched_independently():
    alice = _subject("Alice")
    bob   = _subject("Bob")
    comp  = _composition({"S1": "alice", "S2": "bob"})
    cast  = _cast([
        _entry("alice", "ba", visual_mode="images", use_audio=True),
        _entry("bob",   "bb", visual_mode="images"),
    ])
    reg = _BundleRegistry({
        "ba": _bundle(files=["a.png"], audio_source="file", audio_file="a.wav"),
        "bb": _bundle(files=["b.png"], appearance_override="the big guy"),
    })
    result = apply_cast_to_subjects({"S1": alice, "S2": bob}, comp, cast, reg)
    assert result["S1"]["character_sheet_images"] == ["a.png"]
    assert result["S1"]["voice"]["audio_reference_file"] == "a.wav"
    assert result["S2"]["character_sheet_images"] == ["b.png"]
    assert result["S2"]["appearance"]["summary"] == "the big guy"
    # Alice's appearance not touched
    assert result["S1"]["appearance"]["summary"] == "a tall woman"
