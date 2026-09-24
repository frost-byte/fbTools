"""Concept Registry nodes (ConceptRegistryLoad, ConceptDefine, ConceptResolve, ConceptList).

Moved out of extension.py (pure code motion, Plan 26). Their REST routes were already
extracted to nodes/registry_api.py in an earlier pass and need no changes here.
"""
from __future__ import annotations

import os

import folder_paths
from comfy_api.latest import io

from .shared import prefixed_node_id, default_registry_path, reload_counter
from .lora_stacks import _lora_get_list
from ..utils.concept_registry import (
    ConceptRegistry,
    load_registry as _load_concept_registry,
    save_registry as _save_concept_registry,
    resolve_concepts as _resolve_concepts,
    assemble_prompt as _assemble_concept_prompt,
    format_resolved_info as _format_resolved_info,
    parse_concept_ids as _parse_concept_ids,
    build_model_entry as _build_model_entry,
    MODEL_PROFILES as _CONCEPT_MODEL_PROFILES,
    MODEL_TYPE_IDS as _CONCEPT_MODEL_TYPE_IDS,
)
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


# ── Custom type: CONCEPT_REGISTRY ────────────────────────────────────────────

CONCEPT_REGISTRY_TYPE = "CONCEPT_REGISTRY"


@io.comfytype(io_type=CONCEPT_REGISTRY_TYPE)
class ConceptRegistryIOType:
    """Carries a ConceptRegistry instance between Load → Define → Resolve nodes."""
    Type = object  # ConceptRegistry

    class Input(io.Input):
        def __init__(self, name: str, **kwargs):
            super().__init__(name, **kwargs)

    class Output(io.Output):
        def __init__(self, name: str = "registry", **kwargs):
            super().__init__(name, **kwargs)


def _concept_get_ids() -> list[str]:
    """Read concept_registry.json and return concept IDs for combo widgets."""
    try:
        registry = _load_concept_registry(default_registry_path())
        ids = sorted(registry.concepts.keys())
        return ["None"] + ids if ids else ["None"]
    except Exception:
        return ["None"]


# ── Node: ConceptRegistryLoad ─────────────────────────────────────────────────

class ConceptRegistryLoad(io.ComfyNode):
    """Load the concept registry from disk.

    Connects to one or more ConceptDefine or ConceptResolve nodes.
    Use the "Reload Registry" button (added by JS) to force re-execution
    after the file has been edited externally.
    """
    node_id = prefixed_node_id("ConceptRegistryLoad")
    display_name = "Concept Registry Load"
    category = "🧊 frost-byte/lora"
    is_output_node = True  # allow standalone execution to preview available concepts

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            category=cls.category,
            is_output_node=cls.is_output_node,
            inputs=[
                io.String.Input(
                    "registry_file",
                    display_name="Registry File (leave empty for default)",
                    default="",
                    tooltip=(
                        "Absolute path to a concept_registry.json file. "
                        "Leave empty to use the default user-data location."
                    ),
                    multiline=False,
                ),
            ],
            outputs=[
                ConceptRegistryIOType.Output(
                    "registry",
                    display_name="Registry",
                    tooltip="Concept registry to wire into ConceptDefine or ConceptResolve nodes.",
                ),
                io.String.Output(
                    "available_concepts",
                    display_name="Available Concepts",
                    tooltip="Human-readable list of all defined concepts.",
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, registry_file: str = "", **_):
        path = registry_file.strip() or default_registry_path()
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0
        return (path, mtime, reload_counter("concept"))

    @classmethod
    def execute(cls, registry_file: str = ""):
        path = registry_file.strip() or default_registry_path()
        registry = _load_concept_registry(path)
        available = registry.list_concepts()
        return io.NodeOutput(registry, available, ui={"available_concepts": available})


# ── Node: ConceptDefine ───────────────────────────────────────────────────────

class ConceptDefine(io.ComfyNode):
    """Define (or update) one concept entry for a specific model type.

    Chain multiple ConceptDefine nodes to build a complete registry before
    passing it to ConceptResolve.  If auto_save is enabled the updated
    registry is written back to its source file (with backup).
    """
    node_id = prefixed_node_id("ConceptDefine")
    display_name = "Concept Define"
    category = "🧊 frost-byte/lora"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            category=cls.category,
            inputs=[
                ConceptRegistryIOType.Input(
                    "registry",
                    display_name="Registry",
                    tooltip="Registry from ConceptRegistryLoad or a previous ConceptDefine.",
                ),
                io.String.Input(
                    "concept_id",
                    display_name="Concept ID",
                    default="",
                    tooltip="Unique snake_case identifier, e.g. char_alice or style_cinematic.",
                    multiline=False,
                ),
                io.String.Input(
                    "name",
                    display_name="Display Name",
                    default="",
                    tooltip="Human-readable label shown in ConceptList.",
                    multiline=False,
                ),
                io.String.Input(
                    "description",
                    display_name="Description",
                    default="",
                    multiline=True,
                    tooltip="Internal notes about this concept — not included in assembled prompts.",
                ),
                io.Combo.Input(
                    "model_type",
                    display_name="Model Type",
                    options=_CONCEPT_MODEL_TYPE_IDS,
                    default=_CONCEPT_MODEL_TYPE_IDS[0],
                    tooltip="Target model family for this LoRA entry.",
                ),
                io.Combo.Input(
                    "lora",
                    display_name="LoRA (or High LoRA for split models)",
                    options=_lora_get_list(),
                    default="None",
                    tooltip="LoRA file. For split models (Wan 2.2, BerniniR) this is the HIGH model LoRA.",
                ),
                io.Combo.Input(
                    "lora_low",
                    display_name="Low LoRA (split models only)",
                    options=_lora_get_list(),
                    default="None",
                    optional=True,
                    tooltip="Low-model LoRA for Wan 2.2 / BerniniR. Hidden by JS for single-model types.",
                ),
                io.Float.Input(
                    "weight",
                    display_name="Weight (or High Weight)",
                    default=1.0,
                    min=0.0,
                    max=3.0,
                    step=0.05,
                    tooltip="LoRA strength. For split models this applies to the HIGH LoRA.",
                ),
                io.Float.Input(
                    "weight_low",
                    display_name="Low Weight (split models only)",
                    default=1.0,
                    min=0.0,
                    max=3.0,
                    step=0.05,
                    optional=True,
                    tooltip="LoRA strength for the LOW model LoRA. Hidden by JS for single-model types.",
                ),
                io.String.Input(
                    "trigger",
                    display_name="Trigger Words",
                    default="",
                    multiline=False,
                    tooltip="Trigger text appended/prepended to the prompt by ConceptResolve.",
                ),
                io.Boolean.Input(
                    "auto_save",
                    display_name="Auto Save",
                    default=False,
                    tooltip="If enabled, persist the updated registry to disk after each execution.",
                ),
            ],
            outputs=[
                ConceptRegistryIOType.Output(
                    "registry",
                    display_name="Registry",
                    tooltip="Updated registry with this concept entry added or merged.",
                ),
            ],
        )

    @classmethod
    def validate_inputs(cls, lora="None", lora_low="None", **kwargs):
        return True

    @classmethod
    def execute(
        cls,
        registry: ConceptRegistry,
        concept_id: str,
        name: str,
        description: str,
        model_type: str,
        lora: str = "None",
        lora_low: str = "None",
        weight: float = 1.0,
        weight_low: float = 1.0,
        trigger: str = "",
        auto_save: bool = False,
    ):
        model_entry = _build_model_entry(model_type, lora, lora_low, weight, weight_low, trigger)
        updated = registry.define(concept_id.strip(), name.strip(), description, model_type, model_entry)
        if auto_save:
            save_path = registry.file_path or default_registry_path()
            try:
                _save_concept_registry(updated, save_path, backup=True)
            except Exception as exc:
                logger.warning("ConceptDefine: auto_save failed: %s", exc)
        return io.NodeOutput(updated)


# ── Node: ConceptResolve ──────────────────────────────────────────────────────

class ConceptResolve(io.ComfyNode):
    """Resolve concept IDs against the registry and apply their LoRAs.

    For split-model types (Wan 2.2, BerniniR) the HIGH LoRA is applied to
    *model* and the LOW LoRA is applied to *model_low*.  For single-model
    types only *model* is used.

    Trigger words from each concept are collected and assembled into the
    output prompt according to the trigger_position setting.
    """
    node_id = prefixed_node_id("ConceptResolve")
    display_name = "Concept Resolve"
    category = "🧊 frost-byte/lora"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            category=cls.category,
            inputs=[
                ConceptRegistryIOType.Input(
                    "registry",
                    display_name="Registry",
                    tooltip="Registry from ConceptRegistryLoad or ConceptDefine chain.",
                ),
                io.String.Input(
                    "concepts",
                    display_name="Concepts",
                    default="",
                    multiline=True,
                    tooltip="Concept IDs to resolve — one per line or comma-separated.",
                ),
                io.Combo.Input(
                    "model_type",
                    display_name="Model Type",
                    options=_CONCEPT_MODEL_TYPE_IDS,
                    default=_CONCEPT_MODEL_TYPE_IDS[0],
                    tooltip="Select the model family so the correct LoRAs are chosen.",
                ),
                io.Model.Input(
                    "model",
                    display_name="Model",
                    tooltip="Primary model (or HIGH model for split types).",
                ),
                io.Model.Input(
                    "model_low",
                    display_name="Model (Low)",
                    optional=True,
                    tooltip="Low model for split types (Wan 2.2 / BerniniR). Leave unconnected for single-model types.",
                ),
                io.Clip.Input(
                    "clip",
                    display_name="CLIP",
                    tooltip="CLIP encoder. Both high and low LoRAs are applied to CLIP for split models.",
                ),
                io.String.Input(
                    "base_prompt",
                    display_name="Base Prompt",
                    default="",
                    multiline=True,
                    tooltip="Starting prompt text. Concept triggers are merged in via trigger_position.",
                ),
                io.Combo.Input(
                    "trigger_position",
                    display_name="Trigger Position",
                    options=["prepend", "append"],
                    default="prepend",
                    tooltip="Where to place concept trigger words relative to base_prompt.",
                ),
            ],
            outputs=[
                io.Model.Output(
                    "model",
                    display_name="Model",
                    tooltip="Primary model with HIGH (or single) LoRAs applied.",
                ),
                io.Model.Output(
                    "model_low",
                    display_name="Model (Low)",
                    tooltip="Low model with LOW LoRAs applied (passthrough for single-model types).",
                ),
                io.Clip.Output(
                    "clip",
                    display_name="CLIP",
                    tooltip="CLIP with all concept LoRAs applied.",
                ),
                io.String.Output(
                    "prompt",
                    display_name="Prompt",
                    tooltip="Base prompt with concept trigger words merged in.",
                ),
                io.String.Output(
                    "resolved_info",
                    display_name="Resolved Info",
                    tooltip="Summary of which concepts were resolved and which LoRAs were applied.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        registry: ConceptRegistry,
        concepts: str,
        model_type: str,
        model,
        clip,
        base_prompt: str = "",
        trigger_position: str = "prepend",
        model_low=None,
    ):
        import comfy.sd as _comfy_sd
        import comfy.utils as _comfy_utils

        concept_ids = _parse_concept_ids(concepts)
        resolved = _resolve_concepts(registry, concept_ids, model_type)
        is_split = _CONCEPT_MODEL_PROFILES.get(model_type, {}).get("split", False)

        cur_model = model
        cur_model_low = model_low
        cur_clip = clip
        triggers: list[str] = []

        for r in resolved:
            if r.error:
                logger.warning("ConceptResolve [%s]: %s", r.concept_id, r.error)
                continue

            if r.trigger:
                triggers.append(r.trigger)

            # Apply high (or single) LoRA to primary model + CLIP
            if r.lora_high:
                path = folder_paths.get_full_path("loras", r.lora_high)
                if path:
                    weights = _comfy_utils.load_torch_file(path, safe_load=True)
                    cur_model, cur_clip = _comfy_sd.load_lora_for_models(
                        cur_model, cur_clip, weights, r.weight_high, r.weight_high
                    )
                else:
                    logger.warning("ConceptResolve: LoRA file not found: %s", r.lora_high)
                    r.error = f"file missing: {r.lora_high}"

            # Apply low LoRA for split model types
            if is_split and r.lora_low:
                if cur_model_low is not None:
                    path = folder_paths.get_full_path("loras", r.lora_low)
                    if path:
                        weights = _comfy_utils.load_torch_file(path, safe_load=True)
                        cur_model_low, cur_clip = _comfy_sd.load_lora_for_models(
                            cur_model_low, cur_clip, weights, r.weight_low, r.weight_low
                        )
                    else:
                        logger.warning("ConceptResolve: LoRA file not found: %s", r.lora_low)
                        r.error = f"file missing: {r.lora_low}"
                else:
                    logger.warning(
                        "ConceptResolve [%s]: model_low not connected for split type %s — "
                        "low LoRA '%s' skipped",
                        r.concept_id, model_type, r.lora_low,
                    )

        prompt = _assemble_concept_prompt(triggers, base_prompt, trigger_position)
        resolved_info = _format_resolved_info(model_type, resolved, prompt)

        return io.NodeOutput(cur_model, cur_model_low, cur_clip, prompt, resolved_info)


# ── Node: ConceptList ─────────────────────────────────────────────────────────

class ConceptList(io.ComfyNode):
    """Display a summary of all concepts in the registry.

    Optionally filter by model_type to show only concepts defined for that
    target.  Useful for quickly reviewing which concepts are available before
    wiring up ConceptResolve.
    """
    node_id = prefixed_node_id("ConceptList")
    display_name = "Concept List"
    category = "🧊 frost-byte/lora"
    is_output_node = True

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            category=cls.category,
            is_output_node=cls.is_output_node,
            inputs=[
                ConceptRegistryIOType.Input(
                    "registry",
                    display_name="Registry",
                    tooltip="Registry to inspect.",
                ),
                io.Combo.Input(
                    "model_type",
                    display_name="Filter by Model Type",
                    options=["all"] + _CONCEPT_MODEL_TYPE_IDS,
                    default="all",
                    tooltip="Show only concepts that have an entry for this model type, or 'all'.",
                ),
            ],
            outputs=[
                io.String.Output(
                    "concept_list",
                    display_name="Concept List",
                    tooltip="Formatted list of matching concepts.",
                ),
                io.Int.Output(
                    "concept_count",
                    display_name="Count",
                    tooltip="Number of matching concepts.",
                ),
            ],
        )

    @classmethod
    def execute(cls, registry: ConceptRegistry, model_type: str = "all"):
        filter_type = None if model_type == "all" else model_type
        listing = registry.list_concepts(filter_type)
        count = len([
            cid for cid, c in registry.concepts.items()
            if filter_type is None or filter_type in c.get("models", {})
        ])
        return io.NodeOutput(listing, count, ui={"concept_list": listing, "concept_count": count})
