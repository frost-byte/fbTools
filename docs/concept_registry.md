# Concept Registry Nodes

The concept registry is a named library of LoRA entries, one entry per concept-ID / model-type pair. It lets you define a character, style, or object once and reuse it across workflows — the registry handles selecting the right LoRA file and injecting trigger words for whichever model you're targeting.

**Category:** `🧊 frost-byte/lora`

---

## Nodes

### Concept Registry Load

Loads `concept_registry.json` from disk and passes it downstream.

| Input | Type | Notes |
|---|---|---|
| Registry File | STRING | Absolute path to a custom registry file. Leave empty to use the default location (`ComfyUI/user/default/comfyui-fbTools/concept_registry.json`). |

| Output | Type | Notes |
|---|---|---|
| Registry | CONCEPT_REGISTRY | Wire into Concept Define or Concept Resolve. |
| Available Concepts | STRING | Human-readable list of all defined concepts — displayed in the node UI. |

The node re-executes automatically when the file changes on disk. Use the **Reload Registry** button (added by the extension to the node) to force a re-check after an external edit.

---

### Concept Define

Adds or updates one concept entry in the registry for a specific model type. Chain multiple nodes to build up a full registry before passing it to Concept Resolve.

| Input | Type | Notes |
|---|---|---|
| Registry | CONCEPT_REGISTRY | From Concept Registry Load or a previous Concept Define. |
| Concept ID | STRING | Unique snake_case identifier, e.g. `char_alice` or `style_cinematic`. |
| Display Name | STRING | Human-readable label shown in Concept List. |
| Description | STRING | Internal notes — not included in prompts. |
| Model Type | COMBO | Target model family (`wan22`, `bernini`, `ltx23`, `flux2`, `krea2`, `qwen`, `minimax_h3`). |
| LoRA | COMBO | LoRA file. For split models (Wan 2.2, BerniniR) this is the HIGH model LoRA. |
| Low LoRA | COMBO | LOW model LoRA for Wan 2.2 / BerniniR. Hidden for single-model types. |
| Weight | FLOAT | LoRA strength (0–3). For split models, applies to the HIGH LoRA. |
| Low Weight | FLOAT | Strength for the LOW LoRA. Hidden for single-model types. |
| Trigger Words | STRING | Text appended or prepended to the prompt by Concept Resolve. |
| Auto Save | BOOLEAN | If enabled, writes the registry back to disk immediately. |

| Output | Type | Notes |
|---|---|---|
| Registry | CONCEPT_REGISTRY | Updated registry with this entry added or merged. |

The same concept ID can have entries for multiple model types — they accumulate. The same concept ID + same model type overwrites the previous entry.

---

### Concept Resolve

Looks up concept IDs in the registry, applies their LoRAs to your model and CLIP, and merges trigger words into a prompt.

| Input | Type | Notes |
|---|---|---|
| Registry | CONCEPT_REGISTRY | From Concept Registry Load or a Concept Define chain. |
| Concepts | STRING | Concept IDs to resolve — one per line or comma-separated. |
| Model Type | COMBO | Selects which LoRA entries to use for each concept. |
| Model | MODEL | Primary model (or HIGH model for split types). |
| Model (Low) | MODEL | LOW model for Wan 2.2 / BerniniR. Leave unconnected for single-model types. |
| CLIP | CLIP | Both HIGH and LOW LoRAs are applied to CLIP for split models. |
| Base Prompt | STRING | Starting prompt. Trigger words are merged in via Trigger Position. |
| Trigger Position | COMBO | `prepend` or `append` — where trigger words go relative to the base prompt. |

| Output | Type | Notes |
|---|---|---|
| Model | MODEL | Primary model with HIGH (or single) LoRAs applied. |
| Model (Low) | MODEL | Low model with LOW LoRAs applied. Passthrough if single-model type. |
| CLIP | CLIP | CLIP with all concept LoRAs applied. |
| Prompt | STRING | Base prompt with trigger words merged in. |
| Resolved Info | STRING | Summary of which concepts resolved and which LoRAs were applied. |

---

### Concept List

Displays a summary of all concepts in the registry, optionally filtered by model type.

| Input | Type | Notes |
|---|---|---|
| Registry | CONCEPT_REGISTRY | Registry to inspect. |
| Filter by Model Type | COMBO | `all` or a specific model type — shows only concepts with an entry for that type. |

| Output | Type | Notes |
|---|---|---|
| Concept List | STRING | Formatted list of matching concepts, shown in the node UI. |
| Count | INT | Number of matching concepts. |

---

## Typical Workflow

```
[Concept Registry Load] ──registry──► [Concept Define (char_alice / wan22)]
                                              │ registry
                                              ▼
                                       [Concept Define (style_cinematic / wan22)]
                                              │ registry
                                              ▼
                                       [Concept Resolve]
                                         ◄─ model, clip from your loader
                                         ◄─ base_prompt from your prompt node
                                              │
                              ┌───────────────┼───────────────┐
                           model           CLIP            prompt
                              ▼               ▼               ▼
                         [your sampler]  [CLIP encode]  [text input]
```

For **split models** (Wan 2.2, BerniniR) connect both the high and low model outputs from your loader to `model` and `model_low` respectively, and both LOW-patched outputs come back from Concept Resolve.

---

## Storage

Registry is saved to:
```
ComfyUI/user/default/comfyui-fbTools/concept_registry.json
```
A `.bak` backup is created on every save. Edit the file directly with any text editor; use the **Reload Registry** button or restart ComfyUI to pick up external changes.
