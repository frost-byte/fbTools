"""Pure logic for WanPresetDefine and WanPresetSelect nodes."""

from __future__ import annotations


def preset_define(
    name: str,
    lora_h: str,
    lora_l: str,
    prompt: str,
    preset_list: list | None = None,
) -> list:
    """Append one preset entry to a copy of preset_list (or start a new list)."""
    result = list(preset_list) if preset_list else []
    result.append({"name": name, "lora_h": lora_h, "lora_l": lora_l, "prompt": prompt})
    return result


def preset_select(preset_list: list, index: int) -> tuple[str, str, str, str, str]:
    """
    Return (name, lora_h, lora_l, prompt, available_presets) for the entry
    at *index*, clamped to the last valid position.
    """
    if not preset_list:
        return ("", "", "", "", "No presets available.")
    idx = min(index, len(preset_list) - 1)
    preset = preset_list[idx]
    available = "\n".join(f"[{i}] {p.get('name', '')}" for i, p in enumerate(preset_list))
    return (
        preset.get("name", ""),
        preset.get("lora_h", ""),
        preset.get("lora_l", ""),
        preset.get("prompt", ""),
        available,
    )
