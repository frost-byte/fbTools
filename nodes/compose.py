"""Scene composition-assembly nodes: SceneCompose and PromptAssemble.

Moved out of extension.py (Plan 29, pure code motion).
"""
from __future__ import annotations

from comfy_api.latest import io

from .shared import prefixed_node_id, send_status_update
from .scene_templates import SceneTemplateIOType
from .subjects import SubjectProfileIOType, _load_subject_images, _load_subject_audio
from .outfits import OutfitRegistryIOType
from .composition_types import SceneInstanceIOType, CastIOType
from .concepts import ConceptRegistryIOType
from .compositions import _resolve_cast_media
from ..utils.scene_templates import SceneTemplate
from ..utils.outfit_registry import OutfitRegistry
from ..utils.scene_compose import (
    compose_scene as _compose_scene,
    validate_scene as _validate_scene,
    format_scene_summary as _format_scene_summary,
)
from ..utils.prompt_assembler import assemble_prompt as _assemble_prompt, MODEL_TYPES as _PROMPT_MODEL_TYPES


# ── Node: SceneCompose ────────────────────────────────────────────────────────

class SceneCompose(io.ComfyNode):
    """Assign subjects to template slots and fill in dialogue.

    Connects subject profiles to a scene template's placeholder slots, maps
    positional dialogue inputs to shots in order, and optionally overrides
    outfit descriptions per slot.  Outputs a SCENE_INSTANCE ready for
    PromptAssemble in Phase 4.
    """
    node_id = prefixed_node_id("SceneCompose")
    display_name = "Scene Compose"
    category = "🧊 frost-byte/Scene"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            category=cls.category,
            inputs=[
                SceneTemplateIOType.Input(
                    "template",
                    display_name="Scene Template",
                    tooltip="Template from SceneTemplateLoad.",
                ),
                SubjectProfileIOType.Input(
                    "slot_A",
                    display_name="Slot A",
                    tooltip="Subject assigned to slot A (typically the primary speaker).",
                    optional=True,
                ),
                SubjectProfileIOType.Input(
                    "slot_B",
                    display_name="Slot B",
                    tooltip="Subject assigned to slot B (optional).",
                    optional=True,
                ),
                SubjectProfileIOType.Input(
                    "slot_C",
                    display_name="Slot C",
                    tooltip="Subject assigned to slot C (optional).",
                    optional=True,
                ),
                SubjectProfileIOType.Input(
                    "slot_D",
                    display_name="Slot D",
                    tooltip="Subject assigned to slot D (optional).",
                    optional=True,
                ),
                io.String.Input(
                    "dialogue_1",
                    display_name="Dialogue 1",
                    default="",
                    multiline=True,
                    tooltip="Fills the first placeholder dialogue slot in shot order.",
                    optional=True,
                ),
                io.String.Input(
                    "dialogue_2",
                    display_name="Dialogue 2",
                    default="",
                    multiline=True,
                    tooltip="Fills the second placeholder dialogue slot in shot order.",
                    optional=True,
                ),
                io.String.Input(
                    "dialogue_3",
                    display_name="Dialogue 3",
                    default="",
                    multiline=True,
                    tooltip="Fills the third placeholder dialogue slot in shot order.",
                    optional=True,
                ),
                io.String.Input(
                    "dialogue_4",
                    display_name="Dialogue 4",
                    default="",
                    multiline=True,
                    tooltip="Fills the fourth placeholder dialogue slot in shot order.",
                    optional=True,
                ),
                io.String.Input(
                    "scene_synopsis",
                    display_name="Scene Synopsis",
                    default="",
                    multiline=True,
                    tooltip=(
                        "Concise scene overview used as the summary body in H3 prompts. "
                        "Use {A}/{B}/… slot references (A–J) — expanded to <Subject N> labels at assemble time. "
                        "Overrides the automatic shot-action summary when non-empty."
                    ),
                    optional=True,
                ),
                io.String.Input(
                    "outfit_override_A",
                    display_name="Outfit Override A",
                    default="",
                    multiline=False,
                    tooltip="Replaces slot A subject's default outfit for this scene only.",
                    optional=True,
                ),
                io.String.Input(
                    "outfit_override_B",
                    display_name="Outfit Override B",
                    default="",
                    multiline=False,
                    tooltip="Replaces slot B subject's default outfit for this scene only.",
                    optional=True,
                ),
                io.String.Input(
                    "outfit_override_C",
                    display_name="Outfit Override C",
                    default="",
                    multiline=False,
                    tooltip="Replaces slot C subject's default outfit for this scene only.",
                    optional=True,
                ),
                io.String.Input(
                    "outfit_override_D",
                    display_name="Outfit Override D",
                    default="",
                    multiline=False,
                    tooltip="Replaces slot D subject's default outfit for this scene only.",
                    optional=True,
                ),
                OutfitRegistryIOType.Input(
                    "outfit_registry",
                    display_name="Outfit Registry",
                    tooltip=(
                        "Optional outfit registry from OutfitRegistryLoad. "
                        "When connected, outfit_*_id inputs resolve to registry descriptions."
                    ),
                    optional=True,
                ),
                io.String.Input(
                    "outfit_A_id",
                    display_name="Outfit A ID",
                    default="",
                    multiline=False,
                    tooltip=(
                        "ID of an outfit in the registry to use for slot A. "
                        "Ignored if outfit_override_A is non-empty."
                    ),
                    optional=True,
                ),
                io.String.Input(
                    "outfit_B_id",
                    display_name="Outfit B ID",
                    default="",
                    multiline=False,
                    tooltip="ID of an outfit in the registry for slot B.",
                    optional=True,
                ),
                io.String.Input(
                    "outfit_C_id",
                    display_name="Outfit C ID",
                    default="",
                    multiline=False,
                    tooltip="ID of an outfit in the registry for slot C.",
                    optional=True,
                ),
                io.String.Input(
                    "outfit_D_id",
                    display_name="Outfit D ID",
                    default="",
                    multiline=False,
                    tooltip="ID of an outfit in the registry for slot D.",
                    optional=True,
                ),
            ],
            outputs=[
                SceneInstanceIOType.Output(
                    "scene_instance",
                    display_name="Scene Instance",
                    tooltip="Composed scene ready for PromptAssemble.",
                ),
                io.String.Output(
                    "scene_summary",
                    display_name="Scene Summary",
                    tooltip="Human-readable summary of the composed scene.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        template=None,
        slot_A=None,
        slot_B=None,
        slot_C=None,
        slot_D=None,
        scene_synopsis: str = "",
        dialogue_1: str = "",
        dialogue_2: str = "",
        dialogue_3: str = "",
        dialogue_4: str = "",
        outfit_override_A: str = "",
        outfit_override_B: str = "",
        outfit_override_C: str = "",
        outfit_override_D: str = "",
        outfit_registry=None,
        outfit_A_id: str = "",
        outfit_B_id: str = "",
        outfit_C_id: str = "",
        outfit_D_id: str = "",
    ) -> io.NodeOutput:
        if template is None:
            return io.NodeOutput(None, "No template connected.")

        template_dict = template.to_dict() if isinstance(template, SceneTemplate) else template

        slot_assignments = {"A": slot_A, "B": slot_B, "C": slot_C, "D": slot_D}
        dialogue = [dialogue_1, dialogue_2, dialogue_3, dialogue_4]

        # Explicit text overrides win; registry lookups fill in where text is empty
        def _resolve_outfit(text_override: str, outfit_id: str) -> str:
            if text_override.strip():
                return text_override
            if outfit_id.strip() and isinstance(outfit_registry, OutfitRegistry):
                return outfit_registry.get_description(outfit_id.strip())
            return text_override

        outfit_overrides = {
            "A": _resolve_outfit(outfit_override_A, outfit_A_id),
            "B": _resolve_outfit(outfit_override_B, outfit_B_id),
            "C": _resolve_outfit(outfit_override_C, outfit_C_id),
            "D": _resolve_outfit(outfit_override_D, outfit_D_id),
        }

        warnings = _validate_scene(template_dict, slot_assignments)
        instance = _compose_scene(template_dict, slot_assignments, dialogue, outfit_overrides)
        if scene_synopsis.strip():
            instance["scene_synopsis"] = scene_synopsis.strip()
        summary = _format_scene_summary(instance, validation_warnings=warnings)

        if warnings:
            send_status_update(cls.node_id, f"Composed with {len(warnings)} warning(s)", level="warn")
        else:
            send_status_update(cls.node_id, f"Composed: {instance['template_name']}")

        return io.NodeOutput(instance, summary, ui={"scene_summary": summary})


# ── Node: PromptAssemble ──────────────────────────────────────────────────────

class PromptAssemble(io.ComfyNode):
    """Assemble a model-specific prompt from a composed scene instance.

    Takes a SCENE_INSTANCE from SceneCompose and generates the complete
    prompt in the format required by the chosen model type.  Also outputs
    reference images and audio in the order expected by the model's
    conditioning nodes, and the concept IDs for downstream LoRA resolution
    via ConceptResolve.
    """
    node_id = prefixed_node_id("PromptAssemble")
    display_name = "Prompt Assemble"
    category = "🧊 frost-byte/Scene"
    is_output_node = True

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            category=cls.category,
            is_output_node=cls.is_output_node,
            inputs=[
                SceneInstanceIOType.Input(
                    "scene_instance",
                    display_name="Scene Instance",
                    tooltip="Composed scene from SceneCompose.",
                ),
                io.Combo.Input(
                    "model_type",
                    options=_PROMPT_MODEL_TYPES,
                    default="h3_ref2va",
                    display_name="Model Type",
                    tooltip="Target model prompt format.",
                ),
                io.String.Input(
                    "task_flags",
                    default="",
                    display_name="Task Flags (H3)",
                    tooltip=(
                        "Comma-separated H3 task type flags for the summary bracket tag. "
                        "Overrides auto-detection when non-empty. "
                        "Official types: reference generation, keyframe completion, "
                        "video editing, video continuation, audio reference, audio reuse. "
                        "Leave blank to auto-detect (reference generation when refs present, "
                        "audio reference when voice refs present)."
                    ),
                    optional=True,
                ),
                CastIOType.Input(
                    "scene_cast",
                    display_name="Scene Cast",
                    tooltip=(
                        "Optional cast from SceneCastLoad or SceneCastBuild. "
                        "When connected, subjects whose bundle has visual_mode='video' "
                        "receive a <Video N> reference label in H3 prompts."
                    ),
                    optional=True,
                ),
                ConceptRegistryIOType.Input(
                    "concept_registry",
                    display_name="Concept Registry",
                    tooltip="Optional registry for trigger word lookup (future use).",
                    optional=True,
                ),
            ],
            outputs=[
                io.String.Output(
                    "prompt",
                    display_name="Prompt",
                    tooltip="Assembled model-specific prompt text.",
                ),
                io.Image.Output(
                    "reference_images",
                    display_name="Reference Images",
                    tooltip="Character sheet images stacked in slot order (Picture 1, 2, …).",
                ),
                io.Audio.Output(
                    "reference_audio",
                    display_name="Reference Audio",
                    tooltip="First subject's voice reference audio (slot A), or None.",
                ),
                io.Audio.Output(
                    "additional_audio",
                    display_name="Additional Audio",
                    tooltip="Second subject's voice reference audio (slot B), or None.",
                ),
                io.String.Output(
                    "concept_ids",
                    display_name="Concept IDs",
                    tooltip="Comma-separated concept IDs for wiring into ConceptResolve.",
                ),
                io.String.Output(
                    "assembly_report",
                    display_name="Assembly Report",
                    tooltip="Human-readable summary of what was assembled.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        scene_instance=None,
        model_type: str = "h3_ref2va",
        task_flags: str = "",
        scene_cast=None,
        concept_registry=None,
    ) -> io.NodeOutput:
        if scene_instance is None:
            return io.NodeOutput("", None, None, None, "", "No scene instance connected.")

        # Thread user-specified task flags into scene_instance for h3_ref2va
        if task_flags and task_flags.strip():
            flags = [f.strip() for f in task_flags.split(",") if f.strip()]
            scene_instance = {**scene_instance, "task_flags": flags}

        # Resolve video_entries from cast so subjects flagged as video get <Video N> labels
        video_entries: list[dict] = []
        if scene_cast:
            cast_media = _resolve_cast_media(scene_cast)
            video_entries = cast_media.get("video_entries_full", [])

        result = _assemble_prompt(scene_instance, model_type, video_entries or None)

        # Load reference images in declared order
        image_order = result["reference_image_order"]
        images = _load_subject_images([fname for _, fname in image_order]) if image_order else None

        # Load audio for first two slots with audio
        audio_slots = result["audio_slots"]
        assignments = scene_instance.get("slot_assignments", {})

        def _get_audio(slot_id: str) -> "dict | None":
            subject = assignments.get(slot_id)
            if subject is None:
                return None
            audio_file = subject.get("voice", {}).get("audio_reference_file", "")
            return _load_subject_audio(audio_file)

        reference_audio = _get_audio(audio_slots[0]) if len(audio_slots) > 0 else None
        additional_audio = _get_audio(audio_slots[1]) if len(audio_slots) > 1 else None

        concept_ids_str = ", ".join(result["concept_ids"])
        report = result["assembly_report"]
        prompt = result["prompt"]

        n_images = len(image_order)
        n_audio = len(audio_slots)
        send_status_update(
            cls.node_id,
            f"Assembled {model_type}: {n_images} image{'s' if n_images != 1 else ''}, "
            f"{n_audio} audio",
        )

        return io.NodeOutput(
            prompt, images, reference_audio, additional_audio,
            concept_ids_str, report,
            ui={"assembly_report": report},
        )


