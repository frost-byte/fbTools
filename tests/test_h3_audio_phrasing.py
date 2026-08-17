"""Tests for Step 2: audio retention/role phrasing in _assemble_h3_ref2va
and correct <Audio N> ordinal assignment in _build_ref_map.

Two concerns:
  A. Ordinal correctness — <Audio N> labels in the prompt must match what the
     native MiniMaxH3ReferenceToVideo tokenizer assigns.  Soundtracks (from
     extract_from_visual) consume audio ordinals BEFORE standalone audio files
     (matching the native ref_items order: images → [soundtrack+video] → standalone).
  B. Retention phrasing — timbre/reuse/style and custom role produce the correct
     descriptive text in subject_definitions, summary, and retention_analysis.
"""

from conftest import import_test_module

pa = import_test_module("utils/prompt_assembler.py")
_build_ref_map = pa._build_ref_map
assemble_prompt = pa.assemble_prompt


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _subject(name, sheets=None, audio_file="", audio_retention="timbre", audio_role="",
             voice_desc=""):
    return {
        "name": name,
        "subject_id": name.lower(),
        "appearance": {"summary": f"{name} appearance"},
        "voice": {
            "audio_reference_file": audio_file,
            "audio_retention": audio_retention,
            "audio_role": audio_role,
            "description": voice_desc,
            "language": "en-us",
        },
        "character_sheet_images": list(sheets or []),
        "concept_id": "",
    }


def _scene(slot_subjects, shots=None, outfit_overrides=None):
    return {
        "slot_assignments": slot_subjects,
        "outfit_overrides": outfit_overrides or {},
        "template": {
            "shots": shots or [],
            "style": "",
            "environment": {"summary": ""},
            "overall_soundscape": "",
            "non_diegetic_music": "",
        },
        "dialogue": {},
    }


def _ventry(subject_id, video_file, audio_source="none",
            audio_retention="timbre", audio_role=""):
    return {
        "subject_id": subject_id,
        "video_file": video_file,
        "load_params": {},
        "audio_source": audio_source,
        "audio_path": video_file if audio_source == "extract_from_visual" else "",
        "audio_start_time": 0.0,
        "audio_duration": 0.0,
        "audio_retention": audio_retention,
        "audio_role": audio_role,
    }


# ── A. Ordinal correctness ─────────────────────────────────────────────────────

def test_standalone_audio_only_gets_ordinal_1():
    """No video: standalone audio should be <Audio 1>."""
    scene = _scene({"A": _subject("Alice", audio_file="alice.wav")})
    ref_map = _build_ref_map(scene)
    assert ref_map["A"]["audio_num"] == 1
    assert ref_map["A"]["soundtrack_num"] is None


def test_soundtrack_audio_gets_ordinal_1():
    """Video with soundtrack: soundtrack is <Audio 1>, no standalone."""
    scene = _scene({"A": _subject("Alice")})
    ve = [_ventry("alice", "alice.mp4", audio_source="extract_from_visual")]
    ref_map = _build_ref_map(scene, ve)
    assert ref_map["A"]["soundtrack_num"] == 1
    assert ref_map["A"]["audio_num"] is None


def test_soundtrack_gets_ordinal_1_standalone_gets_ordinal_2():
    """Slot A has a video+soundtrack; Slot B has standalone audio.
    Soundtrack must be <Audio 1>, standalone must be <Audio 2>.
    """
    scene = _scene({
        "A": _subject("Alice"),
        "B": _subject("Bob", audio_file="bob.wav"),
    })
    ve = [_ventry("alice", "alice.mp4", audio_source="extract_from_visual")]
    ref_map = _build_ref_map(scene, ve)
    assert ref_map["A"]["soundtrack_num"] == 1
    assert ref_map["A"]["audio_num"] is None
    assert ref_map["B"]["soundtrack_num"] is None
    assert ref_map["B"]["audio_num"] == 2


def test_two_soundtracks_then_standalone():
    """Slot A and B each have a video+soundtrack; Slot C has standalone audio.
    Expected: Audio 1, Audio 2, Audio 3.
    """
    scene = _scene({
        "A": _subject("Alice"),
        "B": _subject("Bob"),
        "C": _subject("Carol", audio_file="carol.wav"),
    })
    ve = [
        _ventry("alice", "alice.mp4", audio_source="extract_from_visual"),
        _ventry("bob",   "bob.mp4",   audio_source="extract_from_visual"),
    ]
    ref_map = _build_ref_map(scene, ve)
    assert ref_map["A"]["soundtrack_num"] == 1
    assert ref_map["B"]["soundtrack_num"] == 2
    assert ref_map["C"]["audio_num"] == 3


def test_video_without_soundtrack_does_not_consume_audio_ordinal():
    """Video with audio_source='none' should not reserve an audio ordinal."""
    scene = _scene({
        "A": _subject("Alice"),
        "B": _subject("Bob", audio_file="bob.wav"),
    })
    ve = [_ventry("alice", "alice.mp4", audio_source="none")]
    ref_map = _build_ref_map(scene, ve)
    assert ref_map["A"]["soundtrack_num"] is None
    assert ref_map["B"]["audio_num"] == 1   # No soundtrack consumed ordinal 1


# ── B. subject_definitions phrasing ───────────────────────────────────────────

def _assembled_ref2va(scene):
    return assemble_prompt(scene, "h3_ref2va")["prompt"]


def _get_section(prompt, section_name):
    """Extract lines under a named section."""
    lines = prompt.splitlines()
    in_section = False
    result = []
    for line in lines:
        if line.strip() == f"{section_name}:":
            in_section = True
            continue
        if in_section:
            if line and not line[0].isspace() and line.rstrip().endswith(":"):
                break  # next section
            result.append(line)
    return "\n".join(result)


def test_timbre_has_nocopy_in_subject_definitions():
    """retention=timbre → 'without copying the original signal' in subject_definitions."""
    scene = _scene({"A": _subject("Alice", audio_file="a.wav", audio_retention="timbre")})
    prompt = _assembled_ref2va(scene)
    sd = _get_section(prompt, "subject_definitions")
    assert "without copying the original signal" in sd


def test_reuse_has_verbatim_in_subject_definitions():
    """retention=reuse → 'verbatim' in subject_definitions."""
    scene = _scene({"A": _subject("Alice", audio_file="a.wav", audio_retention="reuse")})
    prompt = _assembled_ref2va(scene)
    sd = _get_section(prompt, "subject_definitions")
    assert "verbatim" in sd


def test_style_has_audio_style_in_subject_definitions():
    """retention=style → 'audio style' in subject_definitions."""
    scene = _scene({"A": _subject("Alice", audio_file="a.wav", audio_retention="style")})
    prompt = _assembled_ref2va(scene)
    sd = _get_section(prompt, "subject_definitions")
    assert "audio style" in sd.lower()


def test_audio_role_overrides_generic_description():
    """Non-empty audio_role replaces the generic description in subject_definitions."""
    scene = _scene({"A": _subject("Alice", audio_file="a.wav", audio_role="recorded interview track")})
    prompt = _assembled_ref2va(scene)
    sd = _get_section(prompt, "subject_definitions")
    assert "recorded interview track" in sd
    # Generic "voice-timbre reference" should not appear when role is explicit
    assert "voice-timbre reference" not in sd


def test_soundtrack_audio_line_appears_in_subject_definitions():
    """A video+soundtrack entry should produce an <Audio N> line in subject_definitions."""
    scene = _scene({"A": _subject("Alice")})
    ve = [_ventry("alice", "alice.mp4", audio_source="extract_from_visual")]
    prompt = assemble_prompt(scene, "h3_ref2va", video_entries=ve)["prompt"]
    sd = _get_section(prompt, "subject_definitions")
    assert "<Audio 1>" in sd
    assert "<Video 1>" in sd


def test_soundtrack_timbre_has_nocopy_in_subject_definitions():
    """Soundtrack with retention=timbre → 'without copying the original signal'."""
    scene = _scene({"A": _subject("Alice")})
    ve = [_ventry("alice", "alice.mp4", audio_source="extract_from_visual",
                  audio_retention="timbre")]
    prompt = assemble_prompt(scene, "h3_ref2va", video_entries=ve)["prompt"]
    sd = _get_section(prompt, "subject_definitions")
    assert "without copying the original signal" in sd


def test_soundtrack_reuse_has_verbatim():
    """Soundtrack with retention=reuse → 'verbatim' in subject_definitions."""
    scene = _scene({"A": _subject("Alice")})
    ve = [_ventry("alice", "alice.mp4", audio_source="extract_from_visual",
                  audio_retention="reuse")]
    prompt = assemble_prompt(scene, "h3_ref2va", video_entries=ve)["prompt"]
    sd = _get_section(prompt, "subject_definitions")
    assert "verbatim" in sd


# ── C. retention_analysis phrasing ────────────────────────────────────────────

def test_timbre_retention_analysis_line():
    """retention=timbre → correct retention_analysis line (includes nocopy)."""
    scene = _scene({"A": _subject("Alice", audio_file="a.wav", audio_retention="timbre")})
    prompt = _assembled_ref2va(scene)
    ra = _get_section(prompt, "retention_analysis")
    assert "<Audio 1>" in ra
    assert "without copying the original signal" in ra


def test_reuse_retention_analysis_is_fully_copy():
    """retention=reuse → 'fully_copy' in retention_analysis."""
    scene = _scene({"A": _subject("Alice", audio_file="a.wav", audio_retention="reuse")})
    prompt = _assembled_ref2va(scene)
    ra = _get_section(prompt, "retention_analysis")
    assert "<Audio 1>" in ra
    assert "fully_copy" in ra


def test_style_retention_analysis_line():
    """retention=style → tonal style phrasing in retention_analysis."""
    scene = _scene({"A": _subject("Alice", audio_file="a.wav", audio_retention="style")})
    prompt = _assembled_ref2va(scene)
    ra = _get_section(prompt, "retention_analysis")
    assert "<Audio 1>" in ra
    assert "tonal style" in ra.lower()


def test_mixed_ordinals_appear_correctly_in_assembled_prompt():
    """Slot A has video+soundtrack (<Audio 1>), Slot B has standalone (<Audio 2>).
    Both ordinals should appear in the assembled prompt correctly.
    """
    scene = _scene({
        "A": _subject("Alice"),
        "B": _subject("Bob", audio_file="bob.wav", audio_retention="timbre"),
    })
    ve = [_ventry("alice", "alice.mp4", audio_source="extract_from_visual",
                  audio_retention="timbre")]
    prompt = assemble_prompt(scene, "h3_ref2va", video_entries=ve)["prompt"]
    sd = _get_section(prompt, "subject_definitions")
    ra = _get_section(prompt, "retention_analysis")
    assert "<Audio 1>" in sd   # soundtrack
    assert "<Audio 2>" in sd   # standalone
    assert "<Audio 1>" in ra
    assert "<Audio 2>" in ra
    # <Audio 3> should not exist
    assert "<Audio 3>" not in prompt
