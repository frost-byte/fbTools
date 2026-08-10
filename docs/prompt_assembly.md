# Prompt Assembly Nodes

These nodes are the final step in the Scene Composition Engine. They take a composed scene and produce model-specific prompt text, reference images, audio, and concept IDs — everything needed to drive a generation node.

**Category:** `🧊 frost-byte/Scene`

---

## Nodes

### Prompt Assemble

Takes a SCENE_INSTANCE from Scene Compose and generates the complete prompt in the format required by the chosen model, plus reference images and audio in the correct order.

| Input | Type | Notes |
|---|---|---|
| Scene Instance | SCENE_INSTANCE | Composed scene from Scene Compose. |
| Model Type | COMBO | Target model — determines prompt format (see table below). |
| Concept Registry | CONCEPT_REGISTRY | Optional. Accepted for future trigger word injection; not yet used. |

| Output | Type | Notes |
|---|---|---|
| Prompt | STRING | Fully assembled model-specific prompt text. Wire into your text-conditioning node. |
| Reference Images | IMAGE | Character sheet images stacked in slot order (A's sheets first, then B, …). None if no sheets defined. |
| Reference Audio | AUDIO | First subject's voice reference (slot A), or None. |
| Additional Audio | AUDIO | Second subject's voice reference (slot B), or None. |
| Concept IDs | STRING | Comma-separated concept IDs from all assigned subjects. Wire into Concept Resolve. |
| Assembly Report | STRING | Human-readable summary of what was assembled, shown in the node UI. |

---

## Model Types

| ID | Format |
|---|---|
| `h3_ref2va` | MiniMax H3 structured brief with 6 sections: subject definitions, summary, retention analysis, detailed description, soundscape, music. Subjects referenced as `<Subject N>`, sheets as `<Picture N>`, audio as `<Audio N>`. |
| `h3_fl2va` | MiniMax H3 free-language: shots with `[Shot N]` headers and `<d>[lang] text</d>` dialogue tags. No reference labels. |
| `wan22` | Wan 2.2 production-direction block. |
| `bernini` | BerniniR production-direction block (same structure as `wan22`). |
| `ltx23` | LTX 2.3 simple descriptive. |
| `flux2` | Flux 2 simple descriptive. |
| `krea2` | Krea 2 simple descriptive. |
| `qwen` | Qwen Image simple descriptive. |

For `h3_ref2va`, reference numbering is assigned independently per type in slot order:
- `<Subject N>` — one per assigned slot (A=Subject 1, B=Subject 2, …)
- `<Picture N>` — continuous global numbering across all subjects
- `<Audio N>` — continuous numbering for slots that have audio files

---

### Prompt Composition Loader

Loads a saved composition from the Prompt Compositions editor and assembles it into a prompt in one node, without needing the full Scene Compose chain.

| Input | Type | Notes |
|---|---|---|
| Composition | COMBO | Select a saved composition by name. Populated at startup; refresh the page after saving brand-new ones. |
| Model Type | COMBO | `composition default` uses the model type stored inside the composition; otherwise overrides it. |

| Output | Type | Notes |
|---|---|---|
| Prompt | STRING | Assembled prompt text. |
| Reference Images | IMAGE | Character sheet images in composition subject order. |
| Reference Audio | AUDIO | First subject's voice reference. |
| Additional Audio | AUDIO | Second subject's voice reference. |
| Concept IDs | STRING | Comma-separated concept IDs for Concept Resolve. |
| Assembly Report | STRING | Summary of assembly, shown in the node UI. |

Compositions are created and edited in the **Prompt Compositions** sidebar panel (accessible from the ComfyUI toolbar). After saving a composition, any Prompt Composition Loader nodes on the canvas re-execute automatically to pick up the changes.

---

## Full Pipeline

```
[Concept Registry Load]
       │ registry (optional)
       │
[Scene Template Load]        [Subject Profile Load: char_alice]
       │ template                    │ subject_profile
       │              ┌──────────────┘
       ▼              ▼
  [Scene Compose]   slot_A ◄── char_alice
                    slot_B ◄── char_bob ◄── [Subject Profile Load: char_bob]
                    dialogue_1: "..."
                       │ scene_instance
                       ▼
                 [Prompt Assemble]
                   model_type: h3_ref2va
                       │
          ┌────────────┼──────────────┬──────────────┬──────────────┐
        prompt   ref_images     ref_audio    additional_audio  concept_ids
          │            │              │              │              │
    [text cond]  [model cond]  [audio cond]  [audio cond]  [Concept Resolve]
```

## Wiring Concept IDs

The `concept_ids` output from Prompt Assemble is a comma-separated string of the concept IDs from all subjects in the scene. Wire it into Concept Resolve's `concepts` input alongside your model and CLIP to apply the correct LoRAs automatically:

```
[Prompt Assemble] ──concept_ids──► [Concept Resolve]
[Prompt Assemble] ──prompt──►      [text conditioning]
```

Concept Resolve will select the right LoRA entry for the model type you choose there, so you don't need to manage LoRA files per-scene manually.
