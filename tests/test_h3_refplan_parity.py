"""Parity test: FBTOOLS_H3_REFPLAN reference ordinals must match the label
numbers the native MiniMaxH3ReferenceToVideo tokenizer would assign.

The native node builds ref_items in this order:
  1. All ref_images entries            → type "image"  → <Picture N>
  2. Per ref_video in key order:
       if ref_video_audio_N exists     → type "audio"  → <Audio J>  (before its video)
       ref_video_N                     → type "video"  → <Video K>
  3. All ref_audios entries            → type "audio"  → <Audio J>

Our _build_h3_refplan must produce a references list in the same order so
that both the assembled prompt and the terminal node derive identical ordinals.
"""

from conftest import import_test_module

pa = import_test_module("utils/prompt_assembler.py")
_build_h3_refplan = pa._build_h3_refplan


# ── Helpers ────────────────────────────────────────────────────────────────────

def _subject(name, sheets=None, audio_file="", audio_start=0.0, audio_dur=0.0,
             audio_retention="timbre", audio_role=""):
    return {
        "name": name,
        "subject_id": name.lower(),
        "appearance": {"summary": f"{name} appearance"},
        "voice": {
            "audio_reference_file": audio_file,
            "audio_start_time": audio_start,
            "audio_duration": audio_dur,
            "audio_retention": audio_retention,
            "audio_role": audio_role,
            "description": "clear voice",
            "language": "en-us",
        },
        "character_sheet_images": list(sheets or []),
        "concept_id": "",
    }


def _scene(slot_subjects):
    """slot_subjects: {slot_id: subject_dict}"""
    return {"slot_assignments": slot_subjects, "outfit_overrides": {}}


def _ventry(subject_id, video_file, audio_source="none",
            audio_path="", audio_start=0.0, audio_dur=0.0,
            audio_retention="timbre", audio_role=""):
    return {
        "subject_id": subject_id,
        "video_file": video_file,
        "load_params": {"force_rate": 0, "frame_load_cap": 16,
                        "skip_first_frames": 0, "select_every_nth": 1},
        "audio_source": audio_source,
        "audio_path": audio_path,
        "audio_start_time": audio_start,
        "audio_duration": audio_dur,
        "audio_retention": audio_retention,
        "audio_role": audio_role,
    }


def _simulate_native_ref_items(references):
    """Reproduce the native node ref_items list from our references list.

    Returns a list of ("image"|"video"|"audio") in presentation order.
    This mirrors what MiniMaxH3ReferenceToVideo.execute builds into ref_items
    before calling clip.tokenize(prompt, minimax_ref_items=ref_items).
    """
    items = []
    for ref in references:
        m = ref["modality"]
        if m == "image":
            items.append("image")
        elif m == "soundtrack_audio":
            items.append("audio")   # emitted before its video
        elif m == "video":
            items.append("video")
        elif m == "audio":
            items.append("audio")
    return items


def _assign_ordinals(ref_items):
    """Assign per-type ordinals (1-based) to a ref_items list.

    Returns list of (type, ordinal) in the same order as ref_items.
    """
    counts = {"image": 0, "video": 0, "audio": 0}
    result = []
    for t in ref_items:
        counts[t] += 1
        result.append((t, counts[t]))
    return result


def _check_parity(references):
    """Assert every reference's ordinal matches what the native node would assign.

    Returns a list of (modality, expected_ordinal, actual_ordinal) tuples
    for failures.  Empty list = all match.
    """
    sim_items = _simulate_native_ref_items(references)
    assigned  = _assign_ordinals(sim_items)

    # Walk references and assigned in parallel
    failures = []
    assign_idx = 0
    for ref in references:
        m = ref["modality"]
        if m not in ("image", "soundtrack_audio", "video", "audio"):
            continue
        _type, native_ordinal = assigned[assign_idx]
        assign_idx += 1

        if m == "image":
            got = ref.get("picture_ordinal")
        elif m == "video":
            got = ref.get("video_ordinal")
        elif m in ("soundtrack_audio", "audio"):
            got = ref.get("audio_ordinal")
        else:
            got = None

        if got != native_ordinal:
            failures.append((m, native_ordinal, got))

    return failures


# ── Scenario 1: images only ────────────────────────────────────────────────────

def test_images_only():
    """2 subjects, 2 sheets each → Picture 1-4, no video/audio."""
    scene = _scene({
        "A": _subject("Alice", sheets=["a1.png", "a2.png"]),
        "B": _subject("Bob",   sheets=["b1.png"]),
    })
    plan = _build_h3_refplan(scene)
    refs = plan["references"]

    # All images, no audio/video
    assert all(r["modality"] == "image" for r in refs)
    assert [r["picture_ordinal"] for r in refs] == [1, 2, 3]
    assert _check_parity(refs) == []


# ── Scenario 2: one video + soundtrack ────────────────────────────────────────

def test_one_video_with_soundtrack():
    """1 subject with 1 sheet + a video entry that has extract_from_visual audio.

    Expected ref_items: [image, audio(soundtrack), video]
    → Picture 1, Audio 1, Video 1
    """
    scene = _scene({
        "A": _subject("Alice", sheets=["a1.png"]),
    })
    ventries = [_ventry("alice", "alice.mp4",
                        audio_source="extract_from_visual",
                        audio_path="alice.mp4")]
    plan = _build_h3_refplan(scene, ventries)
    refs = plan["references"]

    modalities = [r["modality"] for r in refs]
    assert modalities == ["image", "soundtrack_audio", "video"]
    assert refs[0]["picture_ordinal"] == 1
    assert refs[1]["audio_ordinal"] == 1
    assert refs[1]["video_ordinal"] == 1   # soundtrack carries its video's ordinal
    assert refs[2]["video_ordinal"] == 1
    assert _check_parity(refs) == []


# ── Scenario 3: two videos, each with a soundtrack ────────────────────────────

def test_two_videos_with_soundtracks():
    """2 subjects, each with a video + soundtrack, no standalone audio.

    Expected ref_items: [audio, video, audio, video]
    → Audio 1, Video 1, Audio 2, Video 2
    """
    scene = _scene({
        "A": _subject("Alice"),
        "B": _subject("Bob"),
    })
    ventries = [
        _ventry("alice", "alice.mp4",
                audio_source="extract_from_visual", audio_path="alice.mp4"),
        _ventry("bob",   "bob.mp4",
                audio_source="extract_from_visual", audio_path="bob.mp4"),
    ]
    plan = _build_h3_refplan(scene, ventries)
    refs = plan["references"]

    modalities = [r["modality"] for r in refs]
    assert modalities == ["soundtrack_audio", "video", "soundtrack_audio", "video"]
    assert refs[0]["audio_ordinal"] == 1
    assert refs[1]["video_ordinal"] == 1
    assert refs[2]["audio_ordinal"] == 2
    assert refs[3]["video_ordinal"] == 2
    assert _check_parity(refs) == []


# ── Scenario 4: one video (no audio) + standalone audio ───────────────────────

def test_video_without_soundtrack_plus_standalone_audio():
    """1 video with no soundtrack + subject with a voice audio file.

    Expected ref_items: [video, audio]
    → Video 1, Audio 1
    """
    scene = _scene({
        "A": _subject("Alice", audio_file="alice_voice.wav"),
    })
    ventries = [_ventry("alice", "alice.mp4", audio_source="none")]
    plan = _build_h3_refplan(scene, ventries)
    refs = plan["references"]

    modalities = [r["modality"] for r in refs]
    assert modalities == ["video", "audio"]
    assert refs[0]["video_ordinal"] == 1
    assert refs[1]["audio_ordinal"] == 1
    assert _check_parity(refs) == []


# ── Scenario 5: mixed — image + video with soundtrack + standalone audio ───────

def test_mixed_image_video_soundtrack_standalone():
    """Picture 1, Audio 1 (soundtrack), Video 1, Audio 2 (standalone).

    Expected ref_items: [image, audio, video, audio]
    """
    scene = _scene({
        "A": _subject("Alice", sheets=["a1.png"]),
        "B": _subject("Bob",   audio_file="bob_voice.wav"),
    })
    ventries = [
        _ventry("alice", "alice.mp4",
                audio_source="extract_from_visual", audio_path="alice.mp4"),
    ]
    plan = _build_h3_refplan(scene, ventries)
    refs = plan["references"]

    modalities = [r["modality"] for r in refs]
    assert modalities == ["image", "soundtrack_audio", "video", "audio"]
    assert refs[0]["picture_ordinal"] == 1
    assert refs[1]["audio_ordinal"] == 1
    assert refs[2]["video_ordinal"] == 1
    assert refs[3]["audio_ordinal"] == 2
    assert _check_parity(refs) == []


# ── Scenario 6: standalone audio only (no images or video) ────────────────────

def test_standalone_audio_only():
    """Subject with a voice audio file, no sheets, no video.

    Expected ref_items: [audio]
    → Audio 1
    """
    scene = _scene({
        "A": _subject("Alice", audio_file="alice_voice.wav",
                      audio_retention="timbre"),
    })
    plan = _build_h3_refplan(scene)
    refs = plan["references"]

    assert len(refs) == 1
    assert refs[0]["modality"] == "audio"
    assert refs[0]["audio_ordinal"] == 1
    assert _check_parity(refs) == []


# ── Scenario 7: two images + two subjects, one with standalone audio ───────────

def test_multiple_sheets_one_standalone_audio():
    """Picture 1, Picture 2, Picture 3, Audio 1."""
    scene = _scene({
        "A": _subject("Alice", sheets=["a1.png", "a2.png"]),
        "B": _subject("Bob",   sheets=["b1.png"], audio_file="bob.wav"),
    })
    plan = _build_h3_refplan(scene)
    refs = plan["references"]

    modalities = [r["modality"] for r in refs]
    assert modalities == ["image", "image", "image", "audio"]
    assert [r["picture_ordinal"] for r in refs if r["modality"] == "image"] == [1, 2, 3]
    assert refs[3]["audio_ordinal"] == 1
    assert _check_parity(refs) == []


# ── Scenario 8: audio retention and role carried through ──────────────────────

def test_audio_retention_and_role_preserved():
    """retention and role fields are carried into the reference dict."""
    scene = _scene({
        "A": _subject("Alice", audio_file="alice.wav",
                      audio_retention="reuse", audio_role="synchronized dialogue track"),
    })
    plan = _build_h3_refplan(scene)
    ref = plan["references"][0]
    assert ref["retention"] == "reuse"
    assert ref["role"] == "synchronized dialogue track"


def test_soundtrack_audio_retention_carried():
    scene = _scene({"A": _subject("Alice")})
    ventries = [_ventry("alice", "alice.mp4",
                        audio_source="extract_from_visual", audio_path="alice.mp4",
                        audio_retention="reuse", audio_role="verbatim soundtrack")]
    plan = _build_h3_refplan(scene, ventries)
    snd = next(r for r in plan["references"] if r["modality"] == "soundtrack_audio")
    assert snd["retention"] == "reuse"
    assert snd["role"] == "verbatim soundtrack"
