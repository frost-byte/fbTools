# Wan Preset Nodes — Implementation Specification

## Overview

Create two custom ComfyUI nodes using the **V3 API** (not the legacy V1 API) for managing reusable video generation presets. These nodes replace a fragile Switch (Any) node pattern that suffers from dynamic input bugs (links disconnecting, index desync).

The architecture follows a **define-chain-select** pattern: individual preset nodes chain sequentially, each appending one entry to a growing list. A selector node picks one preset by index and outputs its fields individually for downstream consumption.

## Data Structure

Each preset is a dictionary with these keys:

```python
{
    "name": str,     # Human-readable preset name (e.g., "Style A")
    "lora_h": str,   # LoRA filename for the high noise model
    "lora_l": str,   # LoRA filename for the low noise model
    "prompt": str,   # Positive prompt text for this preset
}
```

The preset list is a Python `list[dict]` passed between nodes as a custom type called `PRESET_LIST`.

## Node 1: WanPresetDefine

### Purpose

Defines a single preset configuration and appends it to an optional incoming preset list. Multiple instances chain sequentially to build a collection of presets.

### Category

`Licon/Presets`

### Inputs

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `name` | STRING | Yes | Human-readable name for this preset |
| `lora_h` | STRING | Yes | LoRA filename for the high noise model |
| `lora_l` | STRING | Yes | LoRA filename for the low noise model |
| `prompt` | STRING (multiline) | Yes | Positive prompt text |
| `preset_list` | PRESET_LIST | No (optional) | Incoming list from a previous WanPresetDefine node |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `preset_list` | PRESET_LIST | The updated list with this preset appended |

### Behavior

1. If `preset_list` input is connected, clone the incoming list
2. If `preset_list` input is not connected, start with an empty list
3. Append a new dictionary with the current node's `name`, `lora_h`, `lora_l`, and `prompt` values
4. Return the updated list

### Chaining Pattern

The first WanPresetDefine in a chain has no `preset_list` input connected (starts a new list). Each subsequent node receives the previous node's `preset_list` output, appends its own entry, and passes the growing list forward. Adding a new preset means adding a new node at the end of the chain — no existing connections are modified.

## Node 2: WanPresetSelect

### Purpose

Receives a completed preset list and selects one entry by index. Outputs the selected preset's individual fields for use by downstream nodes (LoRA loaders, text encoders, etc.).

### Category

`Licon/Presets`

### Inputs

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `preset_list` | PRESET_LIST | Yes | The complete preset list from the end of a WanPresetDefine chain |
| `index` | INT | Yes | Zero-based index of the preset to select (default: 0, min: 0) |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `name` | STRING | The selected preset's name |
| `lora_h` | STRING | The selected preset's high noise LoRA filename |
| `lora_l` | STRING | The selected preset's low noise LoRA filename |
| `prompt` | STRING | The selected preset's prompt text |
| `available_presets` | STRING | A formatted list of all presets with indices for reference |

### Behavior

1. Clamp `index` to valid range: `index = min(index, len(preset_list) - 1)`
2. Retrieve the dictionary at the clamped index
3. Generate the `available_presets` string by formatting all entries as `[index] name` on separate lines
4. Return all five outputs

### available_presets Output Format

```
[0] Style A
[1] Style B
[2] Style C
```

This output can be wired to a text preview node so the user can see which index corresponds to which preset without inspecting individual define nodes.

## Workflow Wiring

```
[WanPresetDefine #1]          (no preset_list input — starts the chain)
  name: "Style A"
  lora_h: "lora_a_high.safetensors"
  lora_l: "lora_a_low.safetensors"
  prompt: "a woman walking on the beach..."
      ↓ preset_list (1 entry)

[WanPresetDefine #2]          (preset_list input from #1)
  name: "Style B"
  lora_h: "lora_b_high.safetensors"
  lora_l: "lora_b_low.safetensors"
  prompt: "a man sitting in a cafe..."
      ↓ preset_list (2 entries)

[WanPresetDefine #3]          (preset_list input from #2)
  name: "Style C"
  lora_h: "lora_c_high.safetensors"
  lora_l: "lora_c_low.safetensors"
  prompt: "a child playing in a park..."
      ↓ preset_list (3 entries)

[WanPresetSelect]             (preset_list input from #3)
  index: 1                    ← selects "Style B"
      ↓ outputs:
      name → text preview node
      lora_h → high model LoRA loader
      lora_l → low model LoRA loader
      prompt → CLIPTextEncode (positive)
      available_presets → text preview node
```

## Implementation Notes

### V3 API Requirements

- Use the ComfyUI V3 node definition API, not the legacy class-based V1 API with `INPUT_TYPES`, `RETURN_TYPES`, `FUNCTION`, etc.
- Refer to the existing V3 custom nodes in the repository for the correct V3 patterns, decorators, and type annotations used in this codebase.

### Custom Type Registration

Register `PRESET_LIST` as a custom ComfyUI type so the connection system recognizes it and only allows valid connections between preset nodes. Follow whatever pattern the existing codebase uses for custom type registration in V3.

### Edge Cases

- **Empty list on select**: if WanPresetSelect somehow receives an empty list, return empty strings for all fields and an appropriate message for `available_presets`
- **Index out of range**: clamp to the last valid index rather than erroring
- **Special characters in prompts**: the prompt field should support multiline text, commas, quotes, and any Unicode content without escaping issues

### File Location

Place the implementation in the appropriate location within the existing custom node repository structure. Follow the repository's existing conventions for file organization, naming, and module registration.

### Testing

After implementation, verify:

1. A single WanPresetDefine node with no input produces a list with one entry
2. Three chained WanPresetDefine nodes produce a list with three entries in the correct order
3. WanPresetSelect at index 0 returns the first preset's fields
4. WanPresetSelect at index 2 returns the third preset's fields
5. WanPresetSelect at an out-of-range index (e.g., 5 for a 3-entry list) clamps to index 2
6. The `available_presets` output correctly lists all preset names with indices
7. Disconnecting and reconnecting nodes in the chain doesn't break any links (the whole point of this architecture)
8. The prompt field handles multiline text correctly
