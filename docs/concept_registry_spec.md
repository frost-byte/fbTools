# Concept Registry System — Implementation Specification

## Overview

A model-agnostic concept management system for ComfyUI that separates *what* you want (a character, a style, a concept) from *how* it's implemented on each model (which LoRA files, weights, and trigger text). Concepts are defined once and resolved at runtime based on the target model type.

This system eliminates the tedious duplication of LoRA stacks across workflows and model types. A single concept definition covers all supported models. At workflow time, you specify which concepts to use and which model you're targeting — the system resolves the correct LoRA files, applies them to the correct models (handling high/low splits automatically), and assembles the prompt with trigger text.

The system uses JSON files for persistence, consistent with the existing scene system in the `comfyui-fbTools` package.

## Package Context

This system is part of the `comfyui-fbTools` custom node package. All nodes, types, and persistent data belong to this package. The package directory name (`comfyui-fbTools`) is used to derive the persistent data storage path automatically.

## Core Data Model

### Model Types

Each supported model type has a profile that determines its LoRA structure:

| Model Type | ID | Split Model | Has High LoRA | Has Low LoRA | Notes |
|---|---|---|---|---|---|
| Wan 2.2 | `wan22` | Yes | Yes | Yes | High/low noise model split |
| BerniniR | `bernini` | Yes | Yes | Yes | Built on Wan 2.2 architecture |
| LTX 2.3 | `ltx23` | No | No | No | Single model |
| Flux 2 | `flux2` | No | No | No | Single model |
| Krea 2 | `krea2` | No | No | No | Single model |
| Qwen Image | `qwen` | No | No | No | Single model |

Model type profiles should be defined in code as a constant or configuration, making it easy to add new model types in the future without changing the node logic. The key property is `split_model` — when true, the system expects high/low LoRA pairs and routes them to separate models.

### Concept Structure

A concept is a named entity (character, style, effect) with model-specific implementations:

```json
{
    "name": "Character A",
    "description": "A woman with dark shoulder-length hair, light skin, hazel eyes",
    "models": {
        "wan22": {
            "lora_high": "char_a_wan22_high.safetensors",
            "lora_low": "char_a_wan22_low.safetensors",
            "weight_high": 0.8,
            "weight_low": 0.6,
            "trigger": "a woman with dark shoulder-length hair, light skin, hazel eyes"
        },
        "bernini": {
            "lora_high": "char_a_bernini_high.safetensors",
            "lora_low": "char_a_bernini_low.safetensors",
            "weight_high": 0.8,
            "weight_low": 0.6,
            "trigger": "a woman with dark shoulder-length hair"
        },
        "ltx23": {
            "lora": "char_a_ltx23.safetensors",
            "weight": 1.0,
            "trigger": "char_a"
        },
        "flux2": {
            "lora": "char_a_flux.safetensors",
            "weight": 0.8,
            "trigger": "character_a_flux"
        },
        "krea2": {
            "lora": "char_a_krea2.safetensors",
            "weight": 0.7,
            "trigger": "character a"
        },
        "qwen": {
            "lora": "char_a_qwen.safetensors",
            "weight": 0.8,
            "trigger": "character a"
        }
    }
}
```

Not every concept needs definitions for every model type. A concept can have entries for only the models where a LoRA exists. The resolve node handles "concept not available for this model type" gracefully.

### Registry Structure (JSON File)

```json
{
    "version": 1,
    "concepts": {
        "character_a": {
            "name": "Character A",
            "description": "A woman with dark shoulder-length hair, light skin, hazel eyes",
            "models": {
                "wan22": {
                    "lora_high": "char_a_wan22_high.safetensors",
                    "lora_low": "char_a_wan22_low.safetensors",
                    "weight_high": 0.8,
                    "weight_low": 0.6,
                    "trigger": "a woman with dark shoulder-length hair, light skin, hazel eyes"
                },
                "ltx23": {
                    "lora": "char_a_ltx23.safetensors",
                    "weight": 1.0,
                    "trigger": "char_a"
                }
            }
        },
        "style_cinematic": {
            "name": "Cinematic Style",
            "description": "Film-like cinematic lighting and color grading",
            "models": {
                "wan22": {
                    "lora_high": "cinematic_wan22_high.safetensors",
                    "lora_low": "cinematic_wan22_low.safetensors",
                    "weight_high": 1.0,
                    "weight_low": 0.8,
                    "trigger": "cinematic lighting, film grain, shallow depth of field"
                },
                "ltx23": {
                    "lora": "cinematic_ltx.safetensors",
                    "weight": 0.9,
                    "trigger": "cinematic"
                }
            }
        }
    }
}
```

The top-level keys in `concepts` are concept IDs (lowercase, underscored). The `name` field is the human-readable display name. The `description` field is optional documentation.

## Persistence

### File Location

Persistent data is stored in ComfyUI's user directory, under a subdirectory named after the custom node package. The directory name should be derived dynamically from the package's actual directory name on disk, not hardcoded:

```python
import folder_paths
import os

# Derive data directory from the package's own directory name
package_dir = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
user_dir = os.path.join(folder_paths.get_user_directory(), package_dir)
os.makedirs(user_dir, exist_ok=True)
```

This produces the path: `ComfyUI/user/default/comfyui-fbTools/`

Default registry file: `ComfyUI/user/default/comfyui-fbTools/concept_registry.json`

This convention:
- Survives custom node reinstalls (data is outside the package directory)
- Separates persistent data from generated output (not in the `output/` directory)
- Groups all package data together (concept registry, scenes, presets, any future data)
- Is automatically discoverable (directory name matches the custom_nodes directory name)

**Note:** The existing scene system in this package currently stores JSON files in the `output/` directory. Consider migrating scene files to this same `user/default/comfyui-fbTools/` location for consistency. Both locations can be supported during a transition period, with the user directory taking precedence.

The ConceptRegistryLoad node should log the file path at load time so users know where to find the file for manual editing:

```
[INFO] Concept registry loaded from: /path/to/ComfyUI/user/default/comfyui-fbTools/concept_registry.json
```

### Auto-Save Behavior

Changes to the registry (adding, removing, or modifying concepts) should be persisted to disk automatically when modifications are made through the node UI. The registry should be loaded from disk at ComfyUI startup and cached in memory for fast access during workflow execution.

### File Watching (Optional)

If practical, watch the JSON file for external changes (manual edits with a text editor) and reload automatically. This allows advanced users to edit the registry outside ComfyUI. If file watching is too complex, a manual "reload registry" button or node is acceptable.

## Node Definitions

All nodes should use the **V3 API** and follow the conventions established in this codebase. Refer to existing V3 nodes in the repository for patterns, decorators, and type annotations.

### Node 1: ConceptRegistryLoad

#### Purpose

Loads the concept registry from disk and makes it available to downstream nodes.

#### Category

`🧊 frost-byte/Concepts`

#### Inputs

| Name | Type | Required | Description |
|---|---|---|---|
| `registry_file` | STRING | No | Path to JSON file. If empty, uses default location. |

#### Outputs

| Name | Type | Description |
|---|---|---|
| `registry` | CONCEPT_REGISTRY | The loaded concept registry |
| `available_concepts` | STRING | Formatted list of all concept names for reference |

#### Behavior

1. Load JSON from the specified file (or default path)
2. Parse and validate the structure
3. Return the registry object and a formatted string listing all concepts:
   ```
   [character_a] Character A (wan22, ltx23)
   [style_cinematic] Cinematic Style (wan22, ltx23)
   ```
   The parenthetical shows which model types have definitions for each concept.

### Node 2: ConceptDefine

#### Purpose

Defines or updates a concept entry in the registry through the node UI. This is the interactive way to add concepts without editing JSON directly.

#### Category

`fbTools/Concepts`

#### Inputs

| Name | Type | Required | Description |
|---|---|---|---|
| `registry` | CONCEPT_REGISTRY | Yes | Registry to add to (from loader or previous define) |
| `concept_id` | STRING | Yes | Unique identifier (lowercase, underscored) |
| `name` | STRING | Yes | Human-readable display name |
| `description` | STRING (multiline) | No | Optional description of the concept |
| `model_type` | COMBO | Yes | Which model type this definition is for (wan22, bernini, ltx23, flux2, krea2, qwen) |
| `lora` | COMBO (lora file list) | Yes | Primary LoRA file (or high LoRA for split models) |
| `lora_low` | COMBO (lora file list) | No | Low LoRA file (only for split-model types) |
| `weight` | FLOAT (0.0-3.0, default 1.0) | Yes | Primary weight (or high weight for split models) |
| `weight_low` | FLOAT (0.0-3.0, default 1.0) | No | Low weight (only for split-model types) |
| `trigger` | STRING (multiline) | No | Trigger text for this model type |
| `auto_save` | BOOLEAN (default true) | Yes | Whether to persist changes to disk immediately |

#### Outputs

| Name | Type | Description |
|---|---|---|
| `registry` | CONCEPT_REGISTRY | Updated registry with the new/modified concept |

#### Behavior

1. Receive the existing registry
2. Create or update the concept entry for the specified model_type
3. If `auto_save` is true, write the updated registry to disk
4. Return the updated registry

#### UI Behavior

When `model_type` is set to a split-model type (wan22, bernini), the `lora_low` and `weight_low` inputs should be visible/active. When set to a single-model type (ltx23, flux2, krea2, qwen), these inputs should be hidden or ignored. Implementation depends on what the V3 API supports for dynamic input visibility; if not supported, the node always shows all inputs but ignores `lora_low`/`weight_low` for single-model types.

The `lora` COMBO input should be populated from ComfyUI's available LoRA files (scanning all configured LoRA directories). Same for `lora_low`.

#### Chaining

Multiple ConceptDefine nodes can be chained to define multiple concepts or multiple model-type entries for the same concept:

```
[ConceptRegistryLoad] → registry
    ↓
[ConceptDefine: "char_a" on wan22] → registry
    ↓
[ConceptDefine: "char_a" on ltx23] → registry
    ↓
[ConceptDefine: "style_b" on wan22] → registry
    ↓
downstream nodes use the fully-populated registry
```

### Node 3: ConceptResolve

#### Purpose

The primary runtime node. Takes a registry, a list of active concepts, and the target model type. Applies the correct LoRAs to the correct models and assembles the prompt.

#### Category

`fbTools/Concepts`

#### Inputs

| Name | Type | Required | Description |
|---|---|---|---|
| `registry` | CONCEPT_REGISTRY | Yes | The concept registry |
| `concepts` | STRING (multiline) | Yes | Comma-separated concept IDs to activate, or one per line |
| `model_type` | COMBO | Yes | Target model type (wan22, bernini, ltx23, flux2, krea2, qwen) |
| `model` | MODEL | Yes | Primary model (or high model for split types) |
| `model_low` | MODEL | No | Low model (only for split-model types) |
| `clip` | CLIP | Yes | CLIP model for LoRA application |
| `base_prompt` | STRING (multiline) | Yes | Scene description / base prompt text |
| `trigger_position` | COMBO ["prepend", "append"] | Yes | Where trigger text is placed relative to base_prompt |

#### Outputs

| Name | Type | Description |
|---|---|---|
| `model` | MODEL | Primary model with appropriate LoRAs applied |
| `model_low` | MODEL | Low model with low LoRAs applied (passthrough for single-model types) |
| `clip` | CLIP | CLIP with LoRA modifications applied |
| `prompt` | STRING | Assembled prompt (base + triggers) |
| `resolved_info` | STRING | Human-readable summary of what was resolved |

#### Behavior

1. Parse the `concepts` input to get a list of concept IDs
2. For each concept ID:
   a. Look up the concept in the registry
   b. Find the model-type-specific entry for `model_type`
   c. If not found for this model type, log a warning and skip
   d. Load the LoRA file(s) specified in the entry
   e. For split-model types: apply high LoRA to `model` at `weight_high`, apply low LoRA to `model_low` at `weight_low`
   f. For single-model types: apply LoRA to `model` at `weight`, pass `model_low` through unchanged
   g. Apply LoRA CLIP-side weights to `clip`
   h. Collect trigger text
3. Assemble prompt: prepend or append all trigger texts to `base_prompt` based on `trigger_position`
4. Generate `resolved_info` summary:
   ```
   Model type: wan22
   Resolved concepts:
     [character_a] Character A
       high: char_a_wan22_high.safetensors @ 0.8
       low: char_a_wan22_low.safetensors @ 0.6
       trigger: "a woman with dark shoulder-length hair"
     [style_cinematic] Cinematic Style
       high: cinematic_wan22_high.safetensors @ 1.0
       low: cinematic_wan22_low.safetensors @ 0.8
       trigger: "cinematic lighting, film grain"
   Assembled prompt: "a woman with dark shoulder-length hair, cinematic lighting, film grain, walking through a garden at sunset"
   ```
5. Return all outputs

#### Error Handling

- Concept ID not found in registry: log warning, skip, include in `resolved_info` as "[concept_id] NOT FOUND"
- Concept found but no entry for selected model_type: log warning, skip, include in `resolved_info` as "[concept_id] not defined for [model_type]"
- LoRA file not found on disk: log error, skip that LoRA, include in `resolved_info` as "[concept_id] LoRA file missing: [filename]"
- `model_low` not connected for split-model type: log error, apply only high LoRAs, note in `resolved_info`
- `model_low` connected for single-model type: pass through unchanged, no error

### Node 4: ConceptOverride (Optional)

#### Purpose

Allows per-workflow overrides of concept settings without modifying the persistent registry. Useful for experimentation (trying different weights, toggling high/low independently).

#### Category

`fbTools/Concepts`

#### Inputs

| Name | Type | Required | Description |
|---|---|---|---|
| `registry` | CONCEPT_REGISTRY | Yes | Registry to modify |
| `concept_id` | STRING | Yes | Which concept to override |
| `model_type` | COMBO | Yes | Which model-type entry to override |
| `weight_override` | FLOAT | No | Override primary/high weight |
| `weight_low_override` | FLOAT | No | Override low weight |
| `use_high` | BOOLEAN (default true) | Yes | Toggle high LoRA on/off |
| `use_low` | BOOLEAN (default true) | Yes | Toggle low LoRA on/off |
| `trigger_override` | STRING | No | Override trigger text |

#### Outputs

| Name | Type | Description |
|---|---|---|
| `registry` | CONCEPT_REGISTRY | Registry with overrides applied (in-memory only, not persisted) |

#### Behavior

Modifies the in-memory registry with the specified overrides. Does NOT persist to disk — overrides are workflow-specific and temporary. Chain between ConceptRegistryLoad and ConceptResolve:

```
[ConceptRegistryLoad] → registry
    ↓
[ConceptOverride: char_a, wan22, weight=0.5, use_low=false] → registry
    ↓
[ConceptResolve] → models, prompt
```

### Node 5: ConceptList

#### Purpose

Displays all concepts in the registry with their available model types. Utility node for workflow reference — wire the `available_concepts` output to a text preview node.

#### Category

`fbTools/Concepts`

#### Inputs

| Name | Type | Required | Description |
|---|---|---|---|
| `registry` | CONCEPT_REGISTRY | Yes | Registry to list |
| `model_type` | COMBO | No | Filter to show only concepts available for this model type. Empty = show all. |

#### Outputs

| Name | Type | Description |
|---|---|---|
| `concept_list` | STRING | Formatted list of concepts with model availability |
| `concept_count` | INT | Number of concepts (filtered by model_type if specified) |

## Workflow Integration

### Basic Usage (Two Nodes)

For a simple workflow where concepts are already defined in the JSON file:

```
[ConceptRegistryLoad]
  file: concept_registry.json
    ↓ registry

[ConceptResolve]
  registry ← from loader
  concepts: "character_a, style_cinematic"
  model_type: wan22
  model ← high model from checkpoint
  model_low ← low model from checkpoint
  clip ← from checkpoint
  base_prompt: "walking through a garden at sunset"
  trigger_position: prepend
    ↓
  model → high pass sampler
  model_low → low pass sampler
  clip → CLIPTextEncode
  prompt → CLIPTextEncode text input
```

Two nodes replace entire LoRA stack chains for both high and low models.

### Adding New Concepts (Interactive)

```
[ConceptRegistryLoad] → registry
    ↓
[ConceptDefine]
  concept_id: "character_b"
  name: "Character B"
  model_type: wan22
  lora: char_b_high.safetensors
  lora_low: char_b_low.safetensors
  weight: 1.0
  weight_low: 0.8
  trigger: "a man with short blond hair"
  auto_save: true
    ↓ registry (now includes character_b)

[ConceptResolve]
  concepts: "character_a, character_b"
  ...
```

### Per-Workflow Experimentation

```
[ConceptRegistryLoad] → registry
    ↓
[ConceptOverride]
  concept_id: "character_a"
  model_type: wan22
  weight_override: 0.5
  use_low: false  ← disable low LoRA for this run
    ↓ registry (modified in-memory only)

[ConceptResolve]
  concepts: "character_a"
  model_type: wan22
  ...
    ↓
  model (char_a high at 0.5 applied)
  model_low (no LoRA applied — use_low was false)
```

### Integration with Existing Preset System

The WanPresetDefine/Select system designed earlier can reference concept IDs instead of raw LoRA filenames:

```json
{
    "name": "Garden Scene",
    "concepts": ["character_a", "style_cinematic"],
    "prompt": "walking through a garden at sunset",
    "model_type": "wan22"
}
```

A preset selection feeds concept IDs and model_type into ConceptResolve, which handles all LoRA loading and prompt assembly. The preset system handles scene-level configuration; the concept system handles LoRA-level configuration. Clean separation of concerns.

## Implementation Notes

### V3 API

Use the ComfyUI V3 node definition API. Refer to existing V3 nodes in the repository for patterns.

### Custom Type Registration

Register `CONCEPT_REGISTRY` as a custom ComfyUI type for type-safe connections between concept nodes. Follow existing codebase conventions for custom type registration.

### LoRA Loading

Use ComfyUI's built-in LoRA loading utilities (the same ones used by the standard LoraLoader node) to apply LoRAs to model and CLIP. Do not reimplement LoRA application — call the existing utilities. This ensures compatibility with all LoRA formats (safetensors, etc.) and proper weight merging behavior.

The LoRA COMBO dropdowns should be populated by scanning ComfyUI's configured LoRA directories (respecting `extra_model_paths.yaml`).

### CLIP Handling

LoRAs should be applied to CLIP as well as the model (unless the LoRA has no CLIP-side weights). The ConceptResolve node should apply all active concepts' LoRAs to CLIP before the CLIP output is used for text encoding. This ensures trigger words activate properly.

For split-model types, CLIP receives LoRA modifications from both high and low LoRAs (accumulated). The single CLIP output serves both samplers. This matches the standard Wan 2.2 pattern discussed earlier: LoRAs applied to CLIP once, conditioning shared across samplers.

### Model Type Extensibility

Adding a new model type should require:

1. Adding an entry to the model type profiles (is it split or single?)
2. Adding the new type to COMBO dropdown options
3. No changes to core node logic — the split/single distinction drives all behavior

### File Location and Discovery

See the **Persistence** section above for the file location convention. The data directory is derived dynamically from the package directory name (`comfyui-fbTools`) using the pattern:

```python
package_dir = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
user_dir = os.path.join(folder_paths.get_user_directory(), package_dir)
```

The ConceptRegistryLoad node should create the directory and an empty registry file if they don't exist:

```json
{
    "version": 1,
    "concepts": {}
}
```

If other persistent data files exist in this package (scene files, preset files), they should also be stored in this same `user/default/comfyui-fbTools/` directory for consistency.

### Concurrent Access

If multiple ComfyUI instances or browser tabs could access the same registry file, implement basic file locking or last-write-wins semantics. For a single-user setup (which is the expected use case), last-write-wins is sufficient.

## Edge Cases

- **Same concept ID defined twice via ConceptDefine nodes**: later definition overwrites earlier for the same model_type. Different model_types for the same concept_id accumulate.
- **Empty concepts string in ConceptResolve**: no LoRAs applied, prompt is just base_prompt. Not an error.
- **LoRA file in registry but not on disk**: skip that LoRA, log warning, include in resolved_info. Don't crash.
- **Split model type but only high LoRA defined (no low)**: apply only the high LoRA. Log that low is missing. This supports the "use high only" pattern.
- **Split model type but only low LoRA defined (no high)**: apply only the low LoRA. Same pattern.
- **Weight of 0.0**: apply LoRA at zero weight (effectively disabled). This is different from not having a LoRA defined — it's explicitly set to zero.
- **Concept defined for model_type but with empty lora field**: skip LoRA loading, still include trigger text. This supports "trigger word only" concepts that don't have a LoRA.

## Testing

1. **Registry load/save roundtrip**: create registry, save to JSON, reload, verify contents match
2. **Single-model concept resolve**: define concept for ltx23, resolve with ltx23 model type, verify single LoRA applied to model, model_low passed through
3. **Split-model concept resolve**: define concept for wan22 with high+low, resolve, verify high LoRA on model and low LoRA on model_low
4. **Multiple concepts**: resolve 3 concepts simultaneously, verify all LoRAs applied in order
5. **Missing concept for model type**: define concept for wan22 only, resolve with ltx23 model type, verify graceful skip with warning
6. **Override weights**: apply override, resolve, verify overridden weights used instead of registry values
7. **Override use_high/use_low**: disable low via override, verify only high LoRA applied
8. **Prompt assembly**: verify trigger texts are prepended/appended correctly to base_prompt
9. **Auto-save**: verify ConceptDefine with auto_save=true persists to disk
10. **Empty registry**: load empty registry, resolve with concepts, verify graceful handling
11. **LoRA file missing from disk**: define concept pointing to nonexistent file, resolve, verify skip with warning
12. **CLIP application**: verify CLIP is modified by LoRA application, not passed through unchanged
