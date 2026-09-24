"""Subject Profile nodes (SubjectProfileLoad, SubjectProfileDefine, SubjectProfileList).

Moved out of extension.py (pure code motion, Plan 26). Their REST routes were already
extracted to nodes/registry_api.py in an earlier pass and need no changes here.
"""
from __future__ import annotations

import os

import torch
from comfy_api.latest import io
from folder_paths import get_input_directory, get_output_directory

from .shared import prefixed_node_id, default_subject_profiles_path, reload_counter, send_status_update
from .concepts import _concept_get_ids
from .media import _audio_get_list
from ..utils.subject_profiles import (
    load_registry as _load_subject_registry,
    save_registry as _save_subject_registry,
    SUPPORTED_LANGUAGES as _SUBJECT_LANGUAGES,
)
from ..utils.images import load_image_comfyui, normalize_image_tensor
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def _subject_get_ids() -> list[str]:
    """Read subject_profiles.json and return subject IDs for combo widgets."""
    try:
        reg = _load_subject_registry(default_subject_profiles_path())
        ids = reg.subject_ids()
        return ids if ids else ["(none)"]
    except Exception:
        return ["(none)"]


def _load_subject_images(filenames: "list[str | dict]") -> "torch.Tensor | None":
    """Load character sheet images from the ComfyUI input directory.

    Accepts both legacy list[str] and new list[{file, role}] formats.
    Returns a [N, H, W, 3] float32 tensor (batch), or None if no images load.
    Images that fail to load are silently skipped.
    Images with different sizes are resized to match the first loaded image.
    """
    tensors: list = []
    target_h = target_w = None
    input_dir  = get_input_directory()
    output_dir = get_output_directory()
    for entry in filenames:
        fname = entry["file"] if isinstance(entry, dict) else entry
        if not fname:
            continue
        path = os.path.join(input_dir, fname)
        if not os.path.exists(path):
            alt = os.path.join(output_dir, fname)
            if os.path.exists(alt):
                path = alt
        if not os.path.exists(path):
            logger.debug("Subject image not found: %s", path)
            continue
        try:
            img, _ = load_image_comfyui(path, include_mask=False)  # [1, H, W, 3]
            h, w = img.shape[1], img.shape[2]
            if target_h is None:
                target_h, target_w = h, w
            if h != target_h or w != target_w:
                img = normalize_image_tensor(img, target_h, target_w)
            tensors.append(img)
        except Exception as exc:
            logger.warning("Failed to load subject image %s: %s", fname, exc)
    if not tensors:
        return None
    import torch as _torch
    return _torch.cat(tensors, dim=0)  # [N, H, W, 3]


def _load_subject_audio(filename: str) -> "dict | None":
    """Load an audio reference file from the ComfyUI input directory."""
    if not filename:
        return None
    path = os.path.join(get_input_directory(), filename)
    if not os.path.exists(path):
        logger.debug("Subject audio not found: %s", path)
        return None
    try:
        import torchaudio as _torchaudio
        waveform, sample_rate = _torchaudio.load(path)
        # ComfyUI AUDIO dict: waveform is [batch, channels, samples]
        return {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
    except Exception as exc:
        logger.warning("Failed to load subject audio %s: %s", filename, exc)
        return None


# ── Custom type: SUBJECT_PROFILE ──────────────────────────────────────────────

SUBJECT_PROFILE_TYPE = "SUBJECT_PROFILE"


@io.comfytype(io_type=SUBJECT_PROFILE_TYPE)
class SubjectProfileIOType:
    """Carries a subject profile dict between Load/Define → SceneCompose nodes."""
    Type = object  # dict with name, appearance, voice, character_sheet_images, concept_id

    class Input(io.Input):
        def __init__(self, name: str, **kwargs):
            super().__init__(name, **kwargs)

    class Output(io.Output):
        def __init__(self, name: str = "subject_profile", **kwargs):
            super().__init__(name, **kwargs)


# ── Node: SubjectProfileLoad ──────────────────────────────────────────────────

class SubjectProfileLoad(io.ComfyNode):
    """Load a subject profile from disk and expose its fields as outputs.

    Character sheet images are loaded from the ComfyUI input directory and
    stacked into an IMAGE batch.  Audio reference is loaded if defined.
    Use the Reload button in the REST endpoint to force re-execution after
    editing subject_profiles.json externally.
    """
    node_id = prefixed_node_id("SubjectProfileLoad")
    display_name = "Subject Profile Load"
    category = "🧊 frost-byte/Scene"
    is_output_node = True

    @classmethod
    def define_schema(cls):
        subject_ids = _subject_get_ids()
        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            category=cls.category,
            is_output_node=cls.is_output_node,
            inputs=[
                io.Combo.Input(
                    "subject_id",
                    options=subject_ids,
                    display_name="Subject ID",
                    tooltip="Subject to load. Press R to refresh the list after adding new subjects.",
                ),
            ],
            outputs=[
                SubjectProfileIOType.Output(
                    "subject_profile",
                    display_name="Subject Profile",
                    tooltip="Full subject profile dict for wiring into SceneCompose.",
                ),
                io.String.Output("name", display_name="Name"),
                io.String.Output("appearance_summary", display_name="Appearance Summary"),
                io.Image.Output(
                    "character_sheet_images",
                    display_name="Character Sheet Images",
                    tooltip="Batch of all character sheet images (N×H×W×3). None if no images defined.",
                ),
                io.Audio.Output(
                    "audio_reference",
                    display_name="Audio Reference",
                    tooltip="Voice reference audio, or None if not defined.",
                ),
                io.String.Output("concept_id", display_name="Concept ID"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, subject_id: str = "", **_):
        path = default_subject_profiles_path()
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0
        return (path, subject_id, mtime, reload_counter("subject"))

    @classmethod
    def execute(cls, subject_id: str = "") -> io.NodeOutput:
        path = default_subject_profiles_path()
        registry = _load_subject_registry(path)

        subject = registry.get_subject(subject_id) if subject_id and subject_id != "(none)" else None
        if subject is None:
            logger.warning("SubjectProfileLoad: subject_id %r not found in %s", subject_id, path)
            return io.NodeOutput(None, "", "", None, None, "")

        name = subject.get("name", subject_id)
        appearance = subject.get("appearance", {})
        appearance_summary = appearance.get("summary", "")
        voice = subject.get("voice", {})
        audio_file = voice.get("audio_reference_file", "")
        concept_id = subject.get("concept_id", "")
        sheet_files = subject.get("character_sheet_images", [])

        images = _load_subject_images(sheet_files)
        audio = _load_subject_audio(audio_file)

        # Inject subject_id so downstream nodes (SceneCompose) can reference it
        subject_with_id = dict(subject)
        subject_with_id["subject_id"] = subject_id

        send_status_update(
            cls.node_id,
            f"Loaded: {name} | {len(sheet_files)} sheet images | audio: {'yes' if audio else 'no'}",
        )
        return io.NodeOutput(subject_with_id, name, appearance_summary, images, audio, concept_id)


# ── Node: SubjectProfileDefine ────────────────────────────────────────────────

class SubjectProfileDefine(io.ComfyNode):
    """Create or update a subject profile entry.

    When auto_save is enabled the updated profiles file is written back to
    subject_profiles.json immediately.  Character sheet images are managed
    separately (edit the JSON directly to update the file list).
    """
    node_id = prefixed_node_id("SubjectProfileDefine")
    display_name = "Subject Profile Define"
    category = "🧊 frost-byte/Scene"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            category=cls.category,
            inputs=[
                io.String.Input(
                    "subject_id",
                    display_name="Subject ID",
                    default="",
                    tooltip="Unique snake_case identifier, e.g. char_alice or narrator.",
                    multiline=False,
                ),
                io.String.Input(
                    "name",
                    display_name="Name",
                    default="",
                    multiline=False,
                    tooltip="Full name of the subject as used in prompts and scene summaries.",
                ),
                io.String.Input(
                    "appearance_summary",
                    display_name="Appearance Summary",
                    default="",
                    tooltip="One-sentence description used in compact prompts and as the baseline for detailed descriptions.",
                    multiline=True,
                ),
                io.String.Input(
                    "face",
                    display_name="Face",
                    default="",
                    multiline=True,
                    tooltip="Face shape, eye colour, skin tone, and any distinguishing facial features.",
                ),
                io.String.Input(
                    "hair",
                    display_name="Hair",
                    default="",
                    multiline=False,
                    tooltip="Hair style, length, and colour.",
                ),
                io.String.Input(
                    "body",
                    display_name="Body",
                    default="",
                    multiline=False,
                    tooltip="Height, build, and other physical characteristics.",
                ),
                io.String.Input(
                    "default_outfit",
                    display_name="Default Outfit",
                    default="",
                    multiline=True,
                    tooltip="Default clothing and accessories used when no outfit override is specified in SceneCompose.",
                ),
                io.String.Input(
                    "voice_description",
                    display_name="Voice Description",
                    default="",
                    multiline=True,
                    tooltip="Textual description of vocal quality for use in prompts referencing audio.",
                ),
                io.Combo.Input(
                    "audio_reference_file",
                    display_name="Audio Reference File",
                    options=_audio_get_list(),
                    default="None",
                    tooltip="Voice reference clip from the ComfyUI input directory. Press R to refresh the list after adding new files.",
                ),
                io.Combo.Input(
                    "language",
                    options=_SUBJECT_LANGUAGES,
                    display_name="Language",
                    tooltip="BCP-47 language tag for dialogue tags in H3 prompts.",
                ),
                io.Combo.Input(
                    "concept_id",
                    display_name="Concept ID",
                    options=_concept_get_ids(),
                    default="None",
                    tooltip="Links to a concept registry entry for LoRA resolution. Press R to refresh after adding concepts.",
                ),
                io.Boolean.Input(
                    "auto_save",
                    display_name="Auto Save",
                    default=True,
                    tooltip="Write subject_profiles.json immediately after defining this subject.",
                ),
            ],
            outputs=[
                SubjectProfileIOType.Output(
                    "subject_profile",
                    display_name="Subject Profile",
                    tooltip="Defined subject profile dict.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        subject_id: str,
        name: str = "",
        appearance_summary: str = "",
        face: str = "",
        hair: str = "",
        body: str = "",
        default_outfit: str = "",
        voice_description: str = "",
        audio_reference_file: str = "",
        language: str = "en-us",
        concept_id: str = "",
        auto_save: bool = True,
    ) -> io.NodeOutput:
        if not subject_id.strip():
            raise ValueError("SubjectProfileDefine: subject_id cannot be empty")

        audio_reference_file = "" if audio_reference_file == "None" else audio_reference_file
        concept_id = "" if concept_id == "None" else concept_id

        path = default_subject_profiles_path()
        registry = _load_subject_registry(path)
        registry = registry.define(
            subject_id=subject_id.strip(),
            name=name,
            appearance_summary=appearance_summary,
            face=face,
            hair=hair,
            body=body,
            default_outfit=default_outfit,
            voice_description=voice_description,
            audio_reference_file=audio_reference_file,
            language=language,
            concept_id=concept_id,
        )

        if auto_save:
            _save_subject_registry(registry, path, backup=True)
            logger.info("SubjectProfileDefine: saved %r to %s", subject_id, path)
            send_status_update(cls.node_id, f"Saved subject: {subject_id}")

        sid = subject_id.strip()
        result = dict(registry.get_subject(sid))
        result["subject_id"] = sid
        return io.NodeOutput(result)


# ── Node: SubjectProfileList ──────────────────────────────────────────────────

class SubjectProfileList(io.ComfyNode):
    """Display all defined subject profiles.

    Useful for quickly reviewing what subjects are available without opening
    the JSON file.
    """
    node_id = prefixed_node_id("SubjectProfileList")
    display_name = "Subject Profile List"
    category = "🧊 frost-byte/Scene"
    is_output_node = True

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            category=cls.category,
            is_output_node=cls.is_output_node,
            inputs=[],
            outputs=[
                io.String.Output(
                    "subject_list",
                    display_name="Subject List",
                    tooltip="Formatted list of all defined subjects.",
                ),
                io.Int.Output("subject_count", display_name="Subject Count"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, **_):
        path = default_subject_profiles_path()
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0
        return (path, mtime, reload_counter("subject"))

    @classmethod
    def execute(cls) -> io.NodeOutput:
        registry = _load_subject_registry(default_subject_profiles_path())
        listing = registry.list_subjects()
        count = len(registry.subjects)
        return io.NodeOutput(listing, count, ui={"subject_list": listing, "subject_count": count})
