# Subject Profile Nodes

Subject profiles store the reusable description of a person or character — their appearance, voice, default outfit, character sheet images, and link to a concept registry entry. Once defined, a profile can be loaded into any scene without re-entering the description every time.

**Category:** `🧊 frost-byte/Scene`

---

## Nodes

### Subject Profile Define

Creates or updates a subject profile entry in `subject_profiles.json`.

| Input | Type | Notes |
|---|---|---|
| Subject ID | STRING | Unique snake_case identifier, e.g. `char_alice` or `narrator`. Required. |
| Name | STRING | Full name as used in prompts and scene summaries. |
| Appearance Summary | STRING | One-sentence description used in compact prompts. |
| Face | STRING | Face shape, eye colour, skin tone, distinguishing features. |
| Hair | STRING | Style, length, and colour. |
| Body | STRING | Height, build, and other physical characteristics. |
| Default Outfit | STRING | Clothing used when no outfit override is specified in Scene Compose. |
| Voice Description | STRING | Textual description of vocal quality for audio-aware prompts. |
| Audio Reference File | COMBO | Voice clip selected from the ComfyUI `input/` directory. Press R to refresh the list after adding new audio files. |
| Language | COMBO | BCP-47 language tag for dialogue tags in H3 prompts (e.g. `en-us`, `ja-jp`). |
| Concept ID | STRING | Links to a concept registry entry for automatic LoRA resolution. |
| Auto Save | BOOLEAN | Write `subject_profiles.json` immediately after defining. Default: on. |

| Output | Type | Notes |
|---|---|---|
| Subject Profile | SUBJECT_PROFILE | Profile dict for wiring into Scene Compose. |

**Tip:** Leave fields empty to inherit from an existing entry — only non-empty fields overwrite the stored record.

---

### Subject Profile Load

Loads a subject from `subject_profiles.json` by ID and exposes its data as outputs.

| Input | Type | Notes |
|---|---|---|
| Subject ID | COMBO | Subject to load. Press R to refresh the list after adding new subjects. |

| Output | Type | Notes |
|---|---|---|
| Subject Profile | SUBJECT_PROFILE | Full profile dict for wiring into Scene Compose. |
| Name | STRING | Subject's display name. |
| Appearance Summary | STRING | One-sentence appearance description. |
| Character Sheet Images | IMAGE | Batch of all character sheet images (N×H×W×3). None if none defined. |
| Audio Reference | AUDIO | Voice reference audio clip, or None if not defined. |
| Concept ID | STRING | Concept registry ID for downstream LoRA resolution. |

Character sheet images are loaded from the ComfyUI `input/` directory by filename. Images with differing sizes are resized to match the first one. The node re-executes automatically when the profiles file changes; use the **Reload** button or `POST /fbtools/subjects/reload` to force a check after external edits.

---

### Subject Profile List

Displays all defined subjects as a formatted summary.

| Input | Type | Notes |
|---|---|---|
| *(none)* | — | No inputs required. |

| Output | Type | Notes |
|---|---|---|
| Subject List | STRING | Formatted list of all subjects, shown in the node UI. |
| Subject Count | INT | Total number of defined subjects. |

---

## Typical Workflow

**Defining a new subject (one-time setup):**

```
[Subject Profile Define]  ──subject_profile──►  (wire into Scene Compose or discard)
  subject_id: char_alice
  name: Alice
  appearance_summary: slender woman, late 20s, auburn hair
  face: oval face, green eyes, light freckles
  hair: shoulder-length wavy auburn
  default_outfit: navy blazer, white shirt
  audio_reference_file: alice_voice.wav
  concept_id: char_alice
  auto_save: true
```

**Loading a subject into a scene:**

```
[Subject Profile Load]           [Subject Profile Load]
  subject_id: char_alice           subject_id: char_bob
       │ subject_profile                │ subject_profile
       └────────────┬──────────────────┘
                    ▼
              [Scene Compose]
                slot_A ◄── char_alice
                slot_B ◄── char_bob
```

---

## Adding Character Sheet Images

1. Copy image files to the ComfyUI `input/` directory (e.g. `alice_front.png`, `alice_side.png`).
2. Edit `subject_profiles.json` directly and add their filenames to the `character_sheet_images` list under the subject entry, or run a Subject Profile Define node with the image filenames listed — the node stores them in the profile.
3. The next time Subject Profile Load executes it will stack all listed images into a single IMAGE batch.

---

## Storage

Profiles are saved to:
```
ComfyUI/user/default/comfyui-fbTools/subject_profiles.json
```
A `.bak` backup is created on every save. Edit the file directly with any text editor; use the **Reload** button or restart ComfyUI to pick up external changes.
