"""Pure logic for WanPresetDefine and WanPresetSelect nodes."""

from __future__ import annotations

# LORA_STACK is the easy-use compatible format: list[tuple[str, float, float]]
# Each tuple is (lora_name, model_strength, clip_strength).
LoraStack = list[tuple[str, float, float]]


def preset_define(
    name: str,
    lora_h: LoraStack | None,
    lora_l: LoraStack | None,
    prompt: str,
    preset_list: list | None = None,
) -> list:
    """Append one preset entry to a copy of preset_list (or start a new list)."""
    result = list(preset_list) if preset_list else []
    result.append({"name": name, "lora_h": lora_h, "lora_l": lora_l, "prompt": prompt})
    return result


def preset_select(preset_list: list, selected_preset: str) -> tuple[str, LoraStack | None, LoraStack | None, str, str]:
    """
    Return (name, lora_h, lora_l, prompt, available_presets) for the entry
    whose name matches *selected_preset*.  Falls back to the first entry if
    the name is empty or not found.
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
        preset.get("lora_h"),
        preset.get("lora_l"),
        preset.get("prompt", ""),
        available,
    )
