"""Pure logic for LoraPresetDefine and LoraPresetSelect nodes."""

from __future__ import annotations

# LORA_STACK_DATA is the internal multi-target format: list[dict]
# Each dict has: lora, strength_model, strength_clip, model_target, enabled, …
LoraStackData = list[dict]


def preset_define(
    name: str,
    lora_stack: LoraStackData | None,
    prompt: str,
    preset_list: list | None = None,
) -> list:
    """Append one preset entry to a copy of preset_list (or start a new list)."""
    result = list(preset_list) if preset_list else []
    result.append({"name": name, "lora_stack": lora_stack, "prompt": prompt})
    return result


def preset_select(
    preset_list: list,
    selected_preset: str,
) -> tuple[str, LoraStackData | None, str, str]:
    """
    Return (name, lora_stack, prompt, available_presets) for the entry
    whose name matches *selected_preset*.  Falls back to the first entry if
    the name is empty or not found.
    """
    if not preset_list:
        return ("", None, "", "No presets available.")
    available = "\n".join(f"[{i}] {p.get('name', '')}" for i, p in enumerate(preset_list))
    preset = next(
        (p for p in preset_list if p.get("name") == selected_preset),
        preset_list[0],
    )
    return (
        preset.get("name", ""),
        preset.get("lora_stack"),
        preset.get("prompt", ""),
        available,
    )
