"""LoRA stack building/collection/application: LoraEntryDefine, LoraStackCollect, LoraStackApply,
WanVidLoraStack, LoraStackBuilder (plus the dead, unregistered LoraStackView), their shared
LORA_ENTRY/LORA_STACK_DATA wire types, and owned helpers.

Moved out of extension.py (pure code motion; Plan 19 continues Plan 1/11/17/18's extension.py ->
nodes/ package split). NOT moved here: lora *presets* (LoraPresetDefine/LoraPresetSelect/
WanPresetDefine/WanPresetSelect) — they call SceneInfo.load_preview_assets() via
_load_preset_scene_images(), and narrative-scene code (SceneInfo) hasn't been extracted yet.
Also not moved: MultiLoraLoader (a separate, unrelated dead node elsewhere in extension.py — it
uses none of this domain's types/helpers).

Unlike libber/dataset_caption, this domain has zero REST routes but real cross-domain coupling:
LoraStackData (the wire type), LORA_MODEL_TARGETS, and three helper functions
(_lora_get_list, _lora_entries_for_target, _lora_build_wanvid, _lora_json_to_stack) are used by
nine other node classes that stay in extension.py (SceneSelect, SceneLoraStackSave, SceneCreate,
SceneUpdate, SceneOutput, SceneInput, StoryVideoBatch, SourceProfileClipPrompt,
PromptCompositionLoader, ConceptDefine) — extension.py imports those names back from here. This is
one-directional (extension.py depends on this module, never the reverse) and no more implicit
than today: those classes are defined *earlier* in extension.py than this domain was, so they
already only resolve these names at call time (inside define_schema()/execute()), never at class-
definition time — moving the names into an imported module changes nothing about when they
resolve.

Body order preserved as in extension.py: constants/types -> owned helpers -> LoraEntryDefine ->
LoraStackCollect -> LoraStackView (dead) -> LoraStackApply -> its 3 apply helpers ->
WanVidLoraStack -> LoraStackBuilder's inline-input helper -> LoraStackBuilder.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import folder_paths
from comfy_api.latest import io

from .shared import prefixed_node_id

# ── Constants ─────────────────────────────────────────────────────────────────

# Model targets supported by the LoRA scene system.
# Wan2.2 uses two separate diffusion models — one for the high-pass (structure) sampler
# and one for the low-pass (detail) sampler — so each has its own target.
# All other models have a single pass and need no High/Low distinction.
LORA_MODEL_TARGETS = [
    "LTX2.3",
    "Wan2.2-Native-High",   # first-pass (structure) model — outputs LORA_STACK for easy-use
    "Wan2.2-Native-Low",    # second-pass (detail) model — outputs LORA_STACK for easy-use
    "Wan2.2-Wrapper-High",  # first-pass (structure) model — outputs WANVIDLORA for WanVideoWrapper
    "Wan2.2-Wrapper-Low",   # second-pass (detail) model  — outputs WANVIDLORA for WanVideoWrapper
    "Flux2/Klein",
    "Qwen",
    "MiniMaxH3",
    "Z-Image",
]

# Weight key fragments that belong to LTX2.3 audio layers
LORA_AUDIO_KEYWORDS = [
    "audio", "vocoder", "speech", "audio_stream",
    "cross_modal", "video_to_audio", "av_ca",
]

LORA_ENTRY_TYPE      = "LORA_ENTRY"
LORA_STACK_DATA_TYPE = "LORA_STACK_DATA"


@io.comfytype(io_type=LORA_ENTRY_TYPE)
class LoraEntry:
    """
    Carries a single LoRA definition between LoraEntryDefine and LoraStackCollect.
    Internal dict structure:
      {
        "lora":           str,    # filename from loras folder
        "strength_model": float,
        "strength_clip":  float,
        "enabled":        bool,
        "model_target":   str,    # one of LORA_MODEL_TARGETS
        "audio_enabled":  bool,   # LTX2.3 only — include audio weights
      }
    """
    Type = dict

    class Input(io.Input):
        def __init__(self, name: str, **kwargs):
            super().__init__(name, **kwargs)

    class Output(io.Output):
        def __init__(self, name: str = "lora_entry", **kwargs):
            super().__init__(name, **kwargs)


@io.comfytype(io_type=LORA_STACK_DATA_TYPE)
class LoraStackData:
    """
    Carries the collected stack (list of LORA_ENTRY dicts) between nodes.
    Also serialisable to/from JSON for scene persistence.
    """
    Type = list  # list[dict]

    class Input(io.Input):
        def __init__(self, name: str, **kwargs):
            super().__init__(name, **kwargs)

    class Output(io.Output):
        def __init__(self, name: str = "lora_stack_data", **kwargs):
            super().__init__(name, **kwargs)


# ── LoRA scene helpers ────────────────────────────────────────────────────────

def _lora_get_list() -> list[str]:
    return ["None"] + folder_paths.get_filename_list("loras")


# mtime-keyed in-memory cache: {full_path: (mtime, weights_dict, metadata_dict)}
_lora_weight_cache: dict[str, tuple[float, dict, dict]] = {}


def _lora_load_weights(lora_name: str) -> tuple[dict, dict]:
    """Load raw LoRA weights + safetensors metadata from disk (cached by mtime).

    Returns (weights, metadata).  metadata is the safetensors header dict
    (may be empty {}) — pass it to comfy.sd.load_lora_for_models as
    lora_metadata so downstream nodes can inspect which LoRAs are applied.
    """
    path = folder_paths.get_full_path("loras", lora_name)
    if not path:
        raise FileNotFoundError(f"LoRA not found: {lora_name}")
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0.0
    cached = _lora_weight_cache.get(path)
    if cached is not None and cached[0] == mtime:
        return cached[1], cached[2]
    import comfy.utils as _comfy_utils
    weights, metadata = _comfy_utils.load_torch_file(path, safe_load=True, return_metadata=True)
    metadata = metadata or {}
    _lora_weight_cache[path] = (mtime, weights, metadata)
    return weights, metadata


def _lora_entries_for_target(entries: list[dict], model_target: str) -> list[dict]:
    """Filter to enabled entries matching the given model_target."""
    return [
        e for e in entries
        if e.get("enabled", True)
        and e.get("lora", "None") != "None"
        and e.get("model_target") == model_target
    ]


def _lora_stack_to_json(entries: list[dict]) -> str:
    return json.dumps(entries, indent=2, ensure_ascii=False)


def _lora_json_to_stack(json_str: str) -> list[dict]:
    try:
        data = json.loads(json_str)
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, TypeError):
        pass
    return []


# ── Node: LoraEntryDefine ─────────────────────────────────────────────────────

class LoraEntryDefine(io.ComfyNode):
    """
    Define a single LoRA entry for a specific model target.
    Connect one or more of these to LoraStackCollect.

    video/audio/cross-attention strength controls only have effect when
    model_target is LTX2.3.  Set audio=0 and audio_to_video=0 to fully
    mute audio layers (equivalent to the old audio_enabled=False).
    strength_clip is ignored for model targets that have no CLIP encoder.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=prefixed_node_id("LoraEntryDefine"),
            display_name="LoRA Entry Define",
            category="🧊 frost-byte/lora",
            description=(
                "Define one LoRA for a specific model target. "
                "Connect to LoraStackCollect to build a persisted stack."
            ),
            inputs=[
                io.Combo.Input(
                    "lora",
                    display_name="LoRA",
                    options=_lora_get_list(),
                    default="None",
                    tooltip="Select the LoRA file.",
                ),
                io.Combo.Input(
                    "model_target",
                    display_name="Model Target",
                    options=LORA_MODEL_TARGETS,
                    default="LTX2.3",
                    tooltip="Which model pipeline this LoRA applies to.",
                ),
                io.Float.Input(
                    "strength_model",
                    display_name="Strength (Model)",
                    default=1.0,
                    min=-10.0,
                    max=10.0,
                    step=0.0001,
                    tooltip="LoRA strength applied to the UNet/transformer model weights.",
                ),
                io.Float.Input(
                    "strength_clip",
                    display_name="Strength (CLIP)",
                    default=1.0,
                    min=-10.0,
                    max=10.0,
                    step=0.0001,
                    tooltip=(
                        "LoRA strength applied to the text encoder (CLIP). "
                        "Ignored for model targets that have no CLIP component."
                    ),
                ),
                io.Boolean.Input(
                    "enabled",
                    display_name="Enabled",
                    default=True,
                    tooltip="Disable to skip this LoRA without removing it from the stack.",
                ),
                io.Float.Input(
                    "video_strength",
                    display_name="Video Strength",
                    default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip="LTX2.3: multiplier for all video layers (attn, feedforward, video→audio cross-attn). Set to 0 to skip video weights entirely.",
                ),
                io.Float.Input(
                    "audio_strength",
                    display_name="Audio Strength",
                    default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip="LTX2.3: multiplier for all audio layers (attn, feedforward, audio→video cross-attn). Set to 0 to fully mute audio weights.",
                ),
            ],
            outputs=[
                LoraEntry.Output("lora_entry", display_name="LoRA Entry"),
            ],
        )

    @classmethod
    def execute(
        cls,
        lora: str,
        model_target: str,
        strength_model: float,
        strength_clip: float,
        enabled: bool,
        video_strength: float = 1.0,
        audio_strength: float = 1.0,
    ) -> io.NodeOutput:
        entry = {
            "lora":           lora,
            "model_target":   model_target,
            "strength_model": strength_model,
            "strength_clip":  strength_clip,
            "enabled":        enabled,
            "video_strength": video_strength,
            "audio_strength": audio_strength,
        }
        return io.NodeOutput(entry)


# ── Node: LoraStackCollect ────────────────────────────────────────────────────

class LoraStackCollect(io.ComfyNode):
    """
    Collect multiple LoRA entries into a persisted stack.

    Outputs a JSON string suitable for storing in a scene node,
    and a LORA_STACK_DATA object for direct connection to LoraStackApply.

    Merging/override rules (last-write-wins on (lora, model_target) key):
      1. existing_json entries are loaded first (lowest priority).
      2. prev_stack entries override existing_json duplicates.
      3. Connected LoraEntryDefine entries override everything.

    To edit a single entry in an existing scene stack without rebuilding it:
      SceneSelect.lora_stack_data → prev_stack
      LoraEntryDefine (same lora + model_target, new params) → entry_0
      → SceneLoraStackSave
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        autogrow_template = io.Autogrow.TemplatePrefix(
            input=LoraEntry.Input("entry", optional=True),
            prefix="entry",
            min=1,
            max=20,
        )
        return io.Schema(
            node_id=prefixed_node_id("LoraStackCollect"),
            display_name="LoRA Stack Collect",
            category="🧊 frost-byte/lora",
            description=(
                "Collect LoRA entries into a deduplicated stack. "
                "New entries (entry_0, entry_1…) override prev_stack/existing_json "
                "entries with the same (lora, model_target) key."
            ),
            inputs=[
                io.Autogrow.Input("entries", template=autogrow_template),
                LoraStackData.Input(
                    "prev_stack",
                    display_name="Prev Stack",
                    optional=True,
                    tooltip=(
                        "Existing LORA_STACK_DATA to merge into (e.g. from SceneSelect). "
                        "New entries on entry_0/entry_1/… override any duplicate "
                        "(lora, model_target) pairs from this stack."
                    ),
                ),
                io.String.Input(
                    "existing_json",
                    display_name="Existing JSON",
                    default="[]",
                    multiline=False,
                    optional=True,
                    tooltip=(
                        "Existing stack as a JSON string. Used if Prev Stack is not "
                        "connected. New entries override duplicates here too."
                    ),
                ),
            ],
            outputs=[
                LoraStackData.Output("lora_stack_data", display_name="Stack Data"),
                io.String.Output("stack_json",          display_name="Stack JSON"),
                io.Int.Output("entry_count",            display_name="Entry Count"),
                io.Custom("LORA_STACK").Output(
                    "lora_stack",
                    display_name="LoRA Stack",
                    tooltip=(
                        "Easy-use compatible LORA_STACK: list of (lora_name, model_strength, clip_strength) tuples. "
                        "Connect to EasyLoraStack, PowerLoraLoader, or any node that accepts LORA_STACK."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        entries: io.Autogrow.Type,
        prev_stack: Optional[list] = None,
        existing_json: str = "[]",
    ) -> io.NodeOutput:
        # Build ordered base: existing_json first, prev_stack overrides it
        base: list[dict] = _lora_json_to_stack(existing_json)
        if prev_stack:
            base.extend(prev_stack)

        # New entries (from LoraEntryDefine) have highest priority
        new_entries = [v for v in entries.values() if v is not None]
        base.extend(new_entries)

        # Deduplicate: last entry for each (lora, model_target) key wins
        seen: dict[tuple, dict] = {}
        for entry in base:
            key = (entry.get("lora", ""), entry.get("model_target", ""))
            seen[key] = entry
        merged = list(seen.values())

        stack_json = _lora_stack_to_json(merged)
        easy_stack = [
            (e["lora"], e["strength_model"], e["strength_clip"])
            for e in merged
            if e.get("enabled", True) and e.get("lora", "None") != "None"
        ]
        return io.NodeOutput(merged, stack_json, len(merged), easy_stack)


# ── Node: LoraStackView ──────────────────────────────────────────────────────

class LoraStackView(io.ComfyNode):
    """
    Inspect the contents of a LORA_STACK_DATA without modifying it.

    Outputs a human-readable summary string (connect to a Show Text node)
    and the raw stack_json STRING (connect to LoraStackCollect.existing_json
    if you need to extend the stack without the prev_stack input).
    Pass-through lora_stack_data output lets you chain this inline.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=prefixed_node_id("LoraStackView"),
            display_name="LoRA Stack View",
            category="🧊 frost-byte/lora",
            description=(
                "Display the contents of a LoRA stack as readable text. "
                "Also outputs the raw JSON string and a pass-through stack "
                "so it can sit inline between SceneSelect and LoraStackCollect."
            ),
            inputs=[
                LoraStackData.Input(
                    "lora_stack_data",
                    display_name="Stack Data",
                    optional=True,
                    tooltip="Connect from SceneSelect, StoryVideoBatch, or LoraStackCollect.",
                ),
            ],
            outputs=[
                io.String.Output("summary",         display_name="Summary",    tooltip="Human-readable list of LoRA entries."),
                io.String.Output("stack_json",       display_name="Stack JSON", tooltip="Raw JSON — connect to LoraStackCollect.existing_json if needed."),
                LoraStackData.Output("lora_stack_data", display_name="Stack Data", tooltip="Pass-through — same stack, unchanged."),
                io.Int.Output("entry_count",         display_name="Entry Count"),
            ],
        )

    @classmethod
    def execute(
        cls,
        lora_stack_data: Optional[list] = None,
    ) -> io.NodeOutput:
        stack = lora_stack_data or []
        lines: list[str] = [f"LoRA Stack ({len(stack)} entries):"]
        for i, entry in enumerate(stack):
            lora        = entry.get("lora", "?") or "?"
            target      = entry.get("model_target", "?") or "?"
            strength    = entry.get("strength_model", 1.0)
            enabled     = entry.get("enabled", True)
            status      = "" if enabled else "  [DISABLED]"
            lines.append(f"  [{i}] {lora}  |  {target}  |  strength={strength:.4f}{status}")
        summary = "\n".join(lines) if stack else "(empty stack)"
        stack_json = _lora_stack_to_json(stack)
        return io.NodeOutput(summary, stack_json, stack if stack else None, len(stack))


# ── Node: LoraStackApply ──────────────────────────────────────────────────────

class LoraStackApply(io.ComfyNode):
    """
    Apply a persisted LoRA stack to the active model at inference time.

    Set model_target to match the pipeline you are running.
    Only LoRA entries tagged for that target will be loaded and applied.

    Output behaviour by target:
      LTX2.3        → MODEL, CLIP (audio weights filtered per entry flag)
      Wan2.2-Native → MODEL, CLIP, LORA_STACK (easy-use compatible tuple list)
      Wan2.2-Wrapper→ MODEL, CLIP, WANVIDLORA (WanVideoWrapper compatible dict list)
      Flux2/Klein   → MODEL, CLIP (standard load_lora_for_models)
      Qwen          → MODEL, CLIP (standard load_lora_for_models)
      Z-Image       → MODEL, CLIP (standard load_lora_for_models)
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=prefixed_node_id("LoraStackApply"),
            display_name="LoRA Stack Apply",
            category="🧊 frost-byte/lora",
            description=(
                "Apply a persisted LoRA stack to the active model. "
                "Set model_target to match the pipeline you are running. "
                "Only LoRAs tagged for that target are applied."
            ),
            inputs=[
                io.Combo.Input(
                    "model_target",
                    display_name="Model Target",
                    options=LORA_MODEL_TARGETS,
                    default="LTX2.3",
                    tooltip="Which model pipeline to apply LoRAs for.",
                ),
                LoraStackData.Input(
                    "lora_stack_data",
                    display_name="Stack Data",
                    optional=True,
                    tooltip="Connect from LoraStackCollect.",
                ),
                io.String.Input(
                    "stack_json",
                    display_name="Stack JSON",
                    default="[]",
                    multiline=False,
                    optional=True,
                    tooltip=(
                        "JSON string from a scene node. "
                        "Used if Stack Data input is not connected."
                    ),
                ),
                io.Model.Input(
                    "model",
                    display_name="Model",
                    optional=True,
                    tooltip="Connect the MODEL to patch with LoRA weights.",
                ),
                io.Clip.Input(
                    "clip",
                    display_name="CLIP",
                    optional=True,
                    tooltip="Connect the CLIP/text encoder to patch with LoRA weights.",
                ),
                io.Custom("LORA_STACK").Input(
                    "prev_lora_stack",
                    display_name="Prev LoRA Stack",
                    optional=True,
                    tooltip=(
                        "Wan2.2-Native only: chain from another LORA_STACK output "
                        "to prepend existing entries."
                    ),
                ),
                io.Custom("WANVIDLORA").Input(
                    "prev_wanvid_lora",
                    display_name="Prev WanVid LoRA",
                    optional=True,
                    tooltip=(
                        "Wan2.2-Wrapper only: chain from another WANVIDLORA output "
                        "to prepend existing entries."
                    ),
                ),
                io.Boolean.Input(
                    "low_mem_load",
                    display_name="Low VRAM Load",
                    default=False,
                    optional=True,
                    tooltip="Wan2.2-Wrapper only: load LoRA with reduced VRAM usage.",
                ),
                io.Boolean.Input(
                    "merge_loras",
                    display_name="Merge LoRAs",
                    default=True,
                    optional=True,
                    tooltip=(
                        "Wan2.2-Wrapper only: merge LoRAs into model weights. "
                        "Disable for GGUF / scaled fp8 models."
                    ),
                ),
                io.Boolean.Input(
                    "int8_model",
                    display_name="INT8 Model",
                    default=False,
                    optional=True,
                    tooltip=(
                        "Enable when the model is INT8-quantized (ComfyUI-INT8-Fast). "
                        "LoRAs are applied directly to the model via INT8ModelPatcher's "
                        "dequantize→apply→requantize cycle, bypassing Wan2.2 stack outputs."
                    ),
                ),
            ],
            outputs=[
                io.Model.Output("model",        display_name="Model"),
                io.Clip.Output("clip",          display_name="CLIP"),
                io.Custom("LORA_STACK").Output("lora_stack",   display_name="LoRA Stack"),
                io.Custom("WANVIDLORA").Output("wanvid_lora",  display_name="WanVid LoRA"),
                io.Int.Output("applied_count",  display_name="Applied Count"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model_target: str,
        lora_stack_data: Optional[list] = None,
        stack_json: str = "[]",
        model: Optional[object] = None,
        clip: Optional[object] = None,
        prev_lora_stack: Optional[list] = None,
        prev_wanvid_lora: Optional[list] = None,
        low_mem_load: bool = False,
        merge_loras: bool = True,
        int8_model: bool = False,
    ) -> io.NodeOutput:
        all_entries = lora_stack_data if lora_stack_data is not None else _lora_json_to_stack(stack_json)
        target_entries = _lora_entries_for_target(all_entries, model_target)

        if int8_model:
            # INT8 models (ComfyUI-INT8-Fast INT8ModelPatcher) require direct patching.
            # Wan2.2 stack outputs (LORA_STACK / WANVIDLORA) are not usable with INT8 because
            # their downstream consumers have no knowledge of INT8 dequant/requant.
            # For LTX2.3 targets, use per-layer filtering; for all others, use flat apply.
            if model_target == "LTX2.3":
                model, clip, count = _lora_apply_ltx23(model, clip, target_entries)
            else:
                model, clip, count = _lora_apply_standard(model, clip, target_entries)
            return io.NodeOutput(model, clip, None, None, count)

        if model_target == "LTX2.3":
            model, clip, count = _lora_apply_ltx23(model, clip, target_entries)
            return io.NodeOutput(model, clip, None, None, count)
        elif model_target in ("Wan2.2-Native-High", "Wan2.2-Native-Low"):
            lora_stack, count = _lora_build_stack(target_entries, prev_lora_stack)
            return io.NodeOutput(model, clip, lora_stack, None, count)
        elif model_target in ("Wan2.2-Wrapper-High", "Wan2.2-Wrapper-Low"):
            wanvid_lora, count = _lora_build_wanvid(
                target_entries, prev_wanvid_lora, low_mem_load, merge_loras
            )
            return io.NodeOutput(model, clip, None, wanvid_lora, count)
        else:
            # Flux2/Klein, Qwen, Z-Image — standard load_lora_for_models
            model, clip, count = _lora_apply_standard(model, clip, target_entries)
            return io.NodeOutput(model, clip, None, None, count)


# ── LoRA apply implementations ────────────────────────────────────────────────

def _lora_apply_ltx23(model, clip, entries: list[dict]) -> tuple:
    """LTX2.3 apply: per-key strength scaling for audio/video/cross-attn layers.

    New format (video_strength / audio_strength): video_strength scales all video
    and video-side cross-attn keys; audio_strength scales all audio and audio-side
    cross-attn keys.

    Legacy format (video / video_to_audio / audio / audio_to_video / other) and
    the old audio_enabled=False boolean are still accepted for backward compat.
    """
    import comfy.lora as _comfy_lora
    m, c = model, clip
    count = 0
    for entry in entries:
        lora_name = entry.get("lora", "None")
        if not lora_name or lora_name == "None":
            continue

        if "video_strength" in entry or "audio_strength" in entry:
            # New 2-param format
            video_s    = float(entry.get("video_strength", 1.0))
            audio_s    = float(entry.get("audio_strength", 1.0))
            video_to_a = video_s   # video drives video→audio cross-attn
            audio_to_v = audio_s   # audio drives audio→video cross-attn
            other_s    = video_s
        else:
            # Legacy 5-param format (+ audio_enabled boolean compat)
            _old_audio = entry.get("audio_enabled", None)
            if _old_audio is False and "audio" not in entry:
                audio_s    = 0.0
                audio_to_v = 0.0
            else:
                audio_s    = float(entry.get("audio",          1.0))
                audio_to_v = float(entry.get("audio_to_video", 1.0))
            video_s    = float(entry.get("video",          1.0))
            video_to_a = float(entry.get("video_to_audio", 1.0))
            other_s    = float(entry.get("other",          1.0))

        strength_model = entry.get("strength_model", 1.0)
        strength_clip  = entry.get("strength_clip",  1.0)

        try:
            import comfy.lora_convert as _comfy_lora_convert
            raw, _meta = _lora_load_weights(lora_name)
            raw = _comfy_lora_convert.convert_lora(raw)

            key_map = {}
            if m is not None:
                key_map = _comfy_lora.model_lora_keys_unet(m.model, key_map)
            if c is not None:
                key_map = _comfy_lora.model_lora_keys_clip(c.cond_stage_model, key_map)

            loaded = _comfy_lora.load_lora(raw, key_map)

            if not (video_s == 1.0 and video_to_a == 1.0
                    and audio_s == 1.0 and audio_to_v == 1.0 and other_s == 1.0):
                keys_to_delete = []
                for key, value in loaded.items():
                    ks = key if isinstance(key, str) else (key[0] if isinstance(key, tuple) else str(key))
                    if   "video_to_audio_attn" in ks: mult = video_to_a
                    elif "audio_to_video_attn" in ks: mult = audio_to_v
                    elif "audio_attn" in ks or "audio_ff.net" in ks: mult = audio_s
                    elif "attn" in ks or "ff.net" in ks: mult = video_s
                    else: mult = other_s
                    if mult == 0.0:
                        keys_to_delete.append(key)
                    elif mult != 1.0 and hasattr(value, "weights"):
                        wl = list(value.weights)
                        wl[2] = (wl[2] if wl[2] is not None else 1.0) * mult
                        loaded[key].weights = tuple(wl)
                for key in keys_to_delete:
                    loaded.pop(key, None)

            if m is not None:
                new_m = m.clone()
                new_m.add_patches(loaded, strength_model)
                m = new_m
            if c is not None:
                new_c = c.clone()
                new_c.add_patches(loaded, strength_clip)
                c = new_c
            count += 1
        except Exception as e:
            print(f"[LoraStackApply] LTX2.3: failed to load '{lora_name}': {e}")
    return m, c, count


def _lora_apply_standard(model, clip, entries: list[dict]) -> tuple:
    """Standard apply via load_lora_for_models (Flux2/Klein, Qwen, MiniMaxH3, Z-Image)."""
    import comfy.sd as _comfy_sd
    m, c = model, clip
    count = 0
    for entry in entries:
        lora_name = entry.get("lora", "None")
        if not lora_name or lora_name == "None":
            continue
        try:
            weights, metadata = _lora_load_weights(lora_name)
            m, c = _comfy_sd.load_lora_for_models(
                m, c, weights,
                entry.get("strength_model", 1.0),
                entry.get("strength_clip",  1.0),
                lora_metadata=metadata or None,
            )
            count += 1
        except Exception as e:
            print(f"[LoraStackApply] Standard: failed to load '{lora_name}': {e}")
    return m, c, count


def _lora_build_stack(entries: list[dict], prev_lora_stack: Optional[list]) -> tuple:
    """Build a LORA_STACK compatible with easy-use loraStack."""
    stack: list[tuple] = []
    if prev_lora_stack:
        stack.extend([l for l in prev_lora_stack if l[0] != "None"])
    count = 0
    for entry in entries:
        lora_name = entry.get("lora", "None")
        if not lora_name or lora_name == "None":
            continue
        stack.append((lora_name, entry.get("strength_model", 1.0), entry.get("strength_clip", 1.0)))
        count += 1
    return stack if stack else None, count


def _lora_build_wanvid(
    entries: list[dict],
    prev_wanvid_lora: Optional[list],
    low_mem_load: bool,
    merge_loras: bool,
) -> tuple:
    """Build a WANVIDLORA compatible with WanVideoWrapper.

    Per-entry 'low_mem_load', 'merge_loras', 'blocks', and 'layer_filter' fields
    take precedence over the node-level arguments when present (preserved from
    migrated loras.json data or explicitly set by the user).
    """
    loras_list: list[dict] = []
    if prev_wanvid_lora:
        loras_list.extend(list(prev_wanvid_lora))
    count = 0
    for entry in entries:
        lora_name = entry.get("lora", "None")
        if not lora_name or lora_name == "None":
            continue
        # Per-entry values override node-level infrastructure settings when present
        entry_merge    = entry.get("merge_loras",  merge_loras)
        entry_low_mem  = entry.get("low_mem_load", low_mem_load)
        if not entry_merge:
            entry_low_mem = False  # matches WanVideoWrapper behaviour
        try:
            path = folder_paths.get_full_path_or_raise("loras", lora_name)
        except Exception:
            path = folder_paths.get_full_path("loras", lora_name)
            if not path:
                print(f"[LoraStackApply] WanVid: LoRA not found: {lora_name}")
                continue
        loras_list.append({
            "path":         path,
            "strength":     round(entry.get("strength_model", 1.0), 4),
            "name":         os.path.splitext(os.path.basename(lora_name))[0],
            "blocks":       entry.get("blocks", {}),
            "layer_filter": entry.get("layer_filter", ""),
            "low_mem_load": entry_low_mem,
            "merge_loras":  entry_merge,
        })
        count += 1
    return loras_list if loras_list else None, count


# ── Node: WanVidLoraStack ─────────────────────────────────────────────────────

class WanVidLoraStack(io.ComfyNode):
    """
    Build a WANVIDLORA list for WanVideoWrapper directly from LORA_ENTRY inputs.

    Accepts 1–20 LORA_ENTRY inputs (from LoraEntryDefine) and converts them into a
    WANVIDLORA list compatible with WanVideoWrapper's sampler nodes. All enabled,
    non-None entries are included regardless of their model_target tag, making this
    node the Wan-specific alternative to LoraStackApply.

    Use prev_wanvid_lora to chain with upstream WanVideoLoraSelect nodes.
    Per-entry low_mem_load/merge_loras values (from migrated loras.json data) take
    precedence over the node-level settings when present.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        autogrow_template = io.Autogrow.TemplatePrefix(
            input=LoraEntry.Input("entry", optional=True),
            prefix="entry",
            min=1,
            max=20,
        )
        return io.Schema(
            node_id=prefixed_node_id("WanVidLoraStack"),
            display_name="WanVid LoRA Stack",
            category="🧊 frost-byte/lora",
            description=(
                "Build a WANVIDLORA list from LoRA entries for WanVideoWrapper. "
                "Connect LoraEntryDefine nodes to entry_0, entry_1… inputs. "
                "Outputs WANVIDLORA compatible with WanVideoWrapper sampler nodes."
            ),
            inputs=[
                io.Autogrow.Input("entries", template=autogrow_template),
                io.Custom("WANVIDLORA").Input(
                    "prev_wanvid_lora",
                    display_name="Prev WanVid LoRA",
                    optional=True,
                    tooltip="Chain from another WANVIDLORA output to prepend existing entries.",
                ),
                io.Boolean.Input(
                    "low_mem_load",
                    display_name="Low VRAM Load",
                    default=False,
                    optional=True,
                    tooltip=(
                        "Load LoRAs with reduced VRAM usage, at the cost of slower loading. "
                        "No effect when merge_loras is False."
                    ),
                ),
                io.Boolean.Input(
                    "merge_loras",
                    display_name="Merge LoRAs",
                    default=True,
                    optional=True,
                    tooltip=(
                        "Merge LoRAs into model weights before sampling. "
                        "Disable for GGUF or scaled fp8 models."
                    ),
                ),
            ],
            outputs=[
                io.Custom("WANVIDLORA").Output(
                    "wanvid_lora",
                    display_name="WanVid LoRA",
                    tooltip="Connect to WanVideoWrapper sampler nodes.",
                ),
                io.Int.Output("entry_count", display_name="Entry Count"),
            ],
        )

    @classmethod
    def execute(
        cls,
        entries: io.Autogrow.Type,
        prev_wanvid_lora: Optional[list] = None,
        low_mem_load: bool = False,
        merge_loras: bool = True,
    ) -> io.NodeOutput:
        all_entries = [v for v in entries.values() if v is not None]
        enabled = [e for e in all_entries if e.get("enabled", True)]
        wanvid_lora, count = _lora_build_wanvid(enabled, prev_wanvid_lora, low_mem_load, merge_loras)
        return io.NodeOutput(wanvid_lora, count)


# ── Node: LoraStackBuilder ───────────────────────────────────────────────────

_LORA_BUILDER_ROWS = 8


def _lora_builder_inline_inputs(num_rows: int = _LORA_BUILDER_ROWS) -> list:
    """Generate the flat per-row LoRA widget inputs for LoraStackBuilder.

    Order: all 6 per-slot fields for rows 0..N-1 in definition order.
    """
    lora_list = _lora_get_list()
    inputs = []
    for i in range(num_rows):
        inputs.extend([
            io.Combo.Input(
                f"lora_{i}",
                display_name=f"LoRA {i}",
                options=lora_list,
                default="None",
                optional=True,
                tooltip=f"LoRA file for slot {i}. Leave as None to skip.",
            ),
            io.Float.Input(
                f"strength_model_{i}",
                display_name=f"Strength (Model)",
                default=1.0, min=-10.0, max=10.0, step=0.0001,
                optional=True,
                tooltip=f"Model weight strength for slot {i}.",
            ),
            io.Float.Input(
                f"strength_clip_{i}",
                display_name=f"Strength (CLIP)",
                default=1.0, min=-10.0, max=10.0, step=0.0001,
                optional=True,
                tooltip=f"CLIP weight strength for slot {i}.",
            ),
            io.Boolean.Input(
                f"enabled_{i}",
                display_name=f"Enabled",
                default=True,
                optional=True,
                tooltip=f"Uncheck to skip slot {i} without removing it.",
            ),
            io.Float.Input(
                f"video_{i}",
                display_name=f"Video Strength",
                default=1.0, min=0.0, max=1.0, step=0.01,
                optional=True,
                tooltip=f"LTX2.3 only: video-layer multiplier for slot {i}.",
            ),
            io.Float.Input(
                f"audio_{i}",
                display_name=f"Audio Strength",
                default=1.0, min=0.0, max=1.0, step=0.01,
                optional=True,
                tooltip=f"LTX2.3 only: audio-layer multiplier for slot {i}.",
            ),
        ])
    return inputs


class LoraStackBuilder(io.ComfyNode):
    """
    Build a LORA_STACK_DATA from up to 8 inline LoRA rows — no separate
    LoraEntryDefine / LoraStackCollect nodes required.

    Select model_target once at the top; JS hides the video/audio sliders for
    any target that isn't LTX2.3.  An optional autogrow input accepts LORA_ENTRY
    connections from LoraEntryDefine nodes for power-user workflows.
    A prev_stack input lets you merge onto an existing stack (e.g. from a scene).

    Dedup rule: last-write-wins on (lora, model_target) key.
    Priority (lowest → highest): prev_stack → inline rows → autogrow connections.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        autogrow_template = io.Autogrow.TemplatePrefix(
            input=LoraEntry.Input("entry", optional=True),
            prefix="entry",
            min=0,
            max=20,
        )
        return io.Schema(
            node_id=prefixed_node_id("LoraStackBuilder"),
            display_name="LoRA Stack Builder",
            category="🧊 frost-byte/lora",
            description=(
                "Build a LORA_STACK_DATA from inline LoRA rows. "
                "Select model_target to auto-show LTX2.3 video/audio sliders. "
                "Connect prev_stack to merge onto an existing stack."
            ),
            inputs=[
                io.Combo.Input(
                    "model_target",
                    display_name="Model Target",
                    options=LORA_MODEL_TARGETS,
                    default="LTX2.3",
                    tooltip="Which model pipeline these LoRAs apply to.",
                ),
                LoraStackData.Input(
                    "prev_stack",
                    display_name="Prev Stack",
                    optional=True,
                    tooltip="Existing LORA_STACK_DATA to merge into (lowest priority).",
                ),
                io.Autogrow.Input(
                    "entries",
                    template=autogrow_template,
                    optional=True,
                    tooltip="Optional LORA_ENTRY connections from LoraEntryDefine nodes (highest priority).",
                ),
                *_lora_builder_inline_inputs(_LORA_BUILDER_ROWS),
                io.Boolean.Input(
                    "summary_include_prev_stack",
                    display_name="Summary: Include Prev Stack",
                    default=False,
                    tooltip="When off, Enabled Summary only lists LoRAs defined in this node. When on, includes all merged entries from Prev Stack too.",
                ),
            ],
            outputs=[
                LoraStackData.Output("lora_stack_data",   display_name="Stack Data"),
                io.String.Output("stack_json",            display_name="Stack JSON"),
                io.Int.Output("entry_count",              display_name="Entry Count"),
                io.String.Output("enabled_summary",       display_name="Enabled Summary"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model_target: str,
        entries: io.Autogrow.Type,
        prev_stack: Optional[list] = None,
        summary_include_prev_stack: bool = False,
        **kwargs,
    ) -> io.NodeOutput:
        is_ltx = model_target == "LTX2.3"

        # Build inline entries from flat kwargs (rows 0 .. _LORA_BUILDER_ROWS-1)
        inline: list[dict] = []
        for i in range(_LORA_BUILDER_ROWS):
            lora = kwargs.get(f"lora_{i}") or "None"
            if lora == "None":
                continue
            entry: dict = {
                "lora":           lora,
                "model_target":   model_target,
                "strength_model": kwargs.get(f"strength_model_{i}", 1.0),
                "strength_clip":  kwargs.get(f"strength_clip_{i}",  1.0),
                "enabled":        kwargs.get(f"enabled_{i}",        True),
            }
            if is_ltx:
                entry["video_strength"] = kwargs.get(f"video_{i}", 1.0)
                entry["audio_strength"] = kwargs.get(f"audio_{i}", 1.0)
            inline.append(entry)

        # Collect autogrow LORA_ENTRY connections (highest priority)
        connected: list[dict] = [v for v in entries.values() if v is not None]

        # Merge: prev_stack → inline → connected; last-write-wins on (lora, model_target)
        base = list(prev_stack) if prev_stack else []
        seen: dict[tuple, dict] = {}
        for e in base + inline + connected:
            key = (e.get("lora", ""), e.get("model_target", ""))
            seen[key] = e
        merged = list(seen.values())

        stack_json = _lora_stack_to_json(merged)

        def _fv(v: float) -> str:
            return f"{v:.2f}".rstrip("0").rstrip(".")

        summary_source = merged if summary_include_prev_stack else inline + connected
        summary_lines = []
        for e in summary_source:
            if not e.get("enabled", True):
                continue
            name = os.path.splitext(os.path.basename(e.get("lora", "")))[0][:48]
            parts = [_fv(e.get("strength_model", 1.0)), _fv(e.get("strength_clip", 1.0))]
            if "video_strength" in e:
                parts.append(_fv(e["video_strength"]))
            if "audio_strength" in e:
                parts.append(_fv(e["audio_strength"]))
            summary_lines.append(f"{name} {'/'.join(parts)}")
        enabled_summary = "\n".join(summary_lines) + "\n" if summary_lines else ""

        return io.NodeOutput(merged, stack_json, len(merged), enabled_summary)
