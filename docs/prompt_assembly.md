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
| Composition | COMBO | Select a saved composition by name. Press R to refresh the list after saving new ones. |
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

---

## Backlog — Expanded Subject Definitions (H3 Ref2VA)

The MiniMax H3 `ref2va` spec defines `<Subject N>` as any reusable visible content unit, not only people. The full set of valid Subject types includes:

- People, animals, or objects
- Scenes, backgrounds, or environments
- Clothing, props, interfaces, or visual effects
- Styles, actions, expressions, or poses

One Subject may be defined by multiple reference assets, and one reference asset may provide multiple Subjects.

Our assembler currently maps `<Subject N>` exclusively to character slots. The items below describe the natural extensions that align more closely with the spec, ranked by expected fidelity gain.

---

### B-1 · Outfit as Subject N (dual-path design)

**Current behavior:** Outfit is a text description woven into the character's own Subject line as a detail phrase — `<Subject 1> is …, with red satin minidress, heels.`

**Proposed extension:** When an outfit registry entry carries one or more reference images, the outfit becomes its own Subject with a `<Picture N>` reference, and the character's Subject line links to it:

```
<Subject 1> is the woman in <Picture 1>, with long blonde hair.
<Subject 2> is the outfit in <Picture 2>, a red satin minidress with platform heels, worn by <Subject 1>.
```

**Dual-path principle:** Both paths remain valid. Text-only = fast authoring, good for broad strokes. Media-backed Subject = stronger visual fidelity when outfit consistency matters across shots.

**Scope:**
- `OutfitRegistry` entry: add optional `reference_images: list[str]` field
- `OutfitDefine` node and sidebar editor: image picker for outfit reference images
- `_build_ref_map()` in `prompt_assembler.py`: detect outfit has images → insert extra Subject + Picture slots in sequence
- `_assemble_h3_ref2va()`: emit separate Subject line for outfit, reference it from character's Subject line rather than inlining text
- `PromptAssemble` output: include outfit reference images in the `reference_images` IMAGE batch (after character sheets, before background)

---

### B-2 · Background as Subject N

**Current behavior:** Background description goes into the `summary` section as environment text only. No `<Subject N>` or `<Picture N>` reference is emitted even if the background has a reference image or video.

**Proposed extension:** When `_background_snapshot` carries a reference image or video, emit a dedicated Subject line:

```
<Subject 3> is the environment in <Picture 4>, an outdoor playground at morning, with warm side-lighting.
```

Or for a video-based environment reference:
```
<Subject 3> is the environment from <Video 2>, establishing the outdoor playground setting and lighting.
```

**Scope:**
- Background profile: add optional `reference_images: list[str]` and/or `reference_video: str` field
- Background editor in composition sidebar: image/video picker
- `_build_ref_map()`: detect background has media → insert extra Subject + Picture/Video slot
- `_assemble_h3_ref2va()`: emit Subject line for background; cite it in `retention_analysis`
- `PromptAssemble` output: include background reference images in IMAGE batch (after outfit images)

---

### B-3 · Multi-asset Subject definitions (appearance vs. motion split)

**Current behavior:** When a character has both character sheet images and a video reference (via Scene Cast), all assets are listed on one Subject line without differentiating their roles.

**Proposed extension:** Follow the spec's explicit role-splitting syntax when a character has both image and video references:

```
<Subject 1> is the woman whose appearance comes from <Picture 1> and whose motion comes from <Video 1>.
```

**Scope:**
- `_assemble_h3_ref2va()` in `prompt_assembler.py`: detect when a Subject has both `picture_nums` and `video_num` → emit split-source phrasing instead of the combined form
- Cast entry schema: optional `motion_role` string to customize the motion description (e.g. `"walking gait"`, `"dance motion"`, `"lip-sync pattern"`) rather than defaulting to `"motion"`

---

### Relationship to existing backlog items

**B-4 (already tracked — item 4 in session plan):** Structural video references — `<Video N>` without a Subject, describing editorial/camera relationships. Orthogonal to the above; can land independently.

**B-5 (already tracked — item 5 in session plan):** Per-image role descriptions for `<Picture N>` — changing `character_sheet_images` from `list[str]` to `list[{file, role}]`. This is a prerequisite for B-1 and B-2 if outfit and background images are to carry role labels (e.g. `"outfit reference"`, `"environment reference"`) rather than defaulting to `"character sheet"`.
