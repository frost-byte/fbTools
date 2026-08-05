"""Pure logic for LoraPresetDefine and LoraPresetSelect nodes."""

from __future__ import annotations

# LORA_STACK_DATA is the internal multi-target format: list[dict]
# Each dict has: lora, strength_model, strength_clip, model_target, enabled, …
LoraStackData = list[dict]

# Native LORA_STACK is the ComfyUI easy-use format: list[(name, model_str, clip_str)]
LoraStackNative = list[tuple[str, float, float]]


def _stack_data_to_native(lora_stack: LoraStackData) -> LoraStackNative | None:
    """Convert LORA_STACK_DATA dicts to native (name, model_str, clip_str) tuples."""
    result = [
        (e.get("lora", ""), e.get("strength_model", 1.0), e.get("strength_clip", 1.0))
        for e in lora_stack
        if e.get("enabled", True) and e.get("lora", "None") not in ("None", "", None)
    ]
    return result if result else None


def preset_define(
    name: str,
    lora_stack: LoraStackData | None,
    prompt: str,
    preset_list: list | None = None,
    scene_name: str = "none",
    pose_image_type: str = "open",
    lora_stack_native: LoraStackNative | None = None,
) -> list:
    """Append one preset entry to a copy of preset_list (or start a new list).

    If *lora_stack* (LORA_STACK_DATA) is provided and *lora_stack_native* is not,
    the native LORA_STACK representation is generated automatically so that
    LoraPresetSelect can emit both types without any extra wiring.
    """
    if lora_stack is not None and lora_stack_native is None:
        lora_stack_native = _stack_data_to_native(lora_stack)

    result = list(preset_list) if preset_list else []
    result.append({
        "name": name,
        "lora_stack": lora_stack,
        "lora_stack_native": lora_stack_native,
        "prompt": prompt,
        "scene_name": scene_name,
        "pose_image_type": pose_image_type,
    })
    return result


def preset_select(
    preset_list: list,
    selected_preset: str,
) -> tuple[str, LoraStackData | None, LoraStackNative | None, str, str]:
    """
    Return (name, lora_stack, lora_stack_native, prompt, available_presets).

    Falls back to the first entry if the name is empty or not found.
    """
    if not preset_list:
        return ("", None, None, "", "No presets available.")
    available = "\n".join(f"[{i}] {p.get('name', '')}" for i, p in enumerate(preset_list))
    preset = next(
        (p for p in preset_list if p.get("name") == selected_preset),
        preset_list[0],
    )
    return (
        preset.get("name", ""),
        preset.get("lora_stack"),
        preset.get("lora_stack_native"),
        preset.get("prompt", ""),
        available,
    )
