# Scene Composition Engine — Action Plan

## Context

This document describes a scene composition system for the `comfyui-fbTools` custom node package. It builds on top of the existing **Concept Registry** (already implemented), which manages LoRA mappings per concept per model type.

The goal is to enable composable, reusable prompt generation across multiple video generation models (H3 Ref2VA, Wan 2.2, BerniniR, LTX 2.3, etc.), with subject interchangeability, structured scene templates, and automatic prompt assembly — eliminating repetitive structural boilerplate while keeping creative control over per-scene details.

All persistent data uses JSON files stored in `ComfyUI/user/default/comfyui-fbTools/`, consistent with the concept registry. The data directory is derived dynamically from the package directory name:

```python
import folder_paths
import os

package_dir = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
user_dir = os.path.join(folder_paths.get_user_directory(), package_dir)
```

All nodes use the **ComfyUI V3 API**. Refer to existing V3 nodes in the repository for patterns, decorators, and type annotations.

Node category for all nodes in this system: `fbTools/Scene`

---

## Architecture Overview

The system has four layers, each building on the previous:

```
Layer 1: Subject Profiles (persistent JSON)
    Who subjects are — appearance, voice, character sheet images
    Links to Concept Registry for LoRA resolution

Layer 2: Scene Templates (persistent JSON)
    Structural blueprints with placeholder slots
    Shot structure, camera, environment, soundscape
    Model-agnostic — same template works across models

Layer 3: Scene Instance (per-generation, in-workflow)
    Assigns specific subjects to template slots
    Fills in dialogue, per-scene overrides
    Connects subject profiles to template placeholders

Layer 4: Prompt Assembly (automatic, per-model)
    Generates model-specific prompt from the instance
    H3: full 6-section structured brief
    Wan/BerniniR: production-direction format
    Others: model-appropriate format
```

---

## Phase 1: Subject Profiles

### Purpose

A subject profile contains everything needed to describe and reference a subject (person, character, animal) across any scene, independent of which model or workflow is used. It separates "who this subject is" from "which LoRAs implement this subject" (handled by the concept registry).

### Data Schema

File: `ComfyUI/user/default/comfyui-fbTools/subject_profiles.json`

```json
{
    "version": 1,
    "subjects": {
        "character_a": {
            "name": "Sarah",
            "appearance": {
                "summary": "a woman with dark shoulder-length hair, light skin, hazel eyes, athletic build",
                "face": "high cheekbones, slightly asymmetric smile, defined jawline, dark arched eyebrows, small nose with slight upturn",
                "hair": "dark brown shoulder-length with a slight wave, usually parted left",
                "body": "athletic build, medium height",
                "default_outfit": "navy blazer over a white t-shirt, black pants"
            },
            "voice": {
                "description": "clear youthful voice with a warm conversational tone and measured pace",
                "audio_reference_file": "sarah_voice_ref.wav",
                "language": "en-us"
            },
            "character_sheet_images": [
                "sarah_closeup.png",
                "sarah_front.png",
                "sarah_side.png",
                "sarah_back.png"
            ],
            "concept_id": "character_a"
        }
    }
}
```

### Field Descriptions

- **`name`**: human-readable display name for this subject
- **`appearance.summary`**: one-sentence overall description, used for compact prompts (Wan/BerniniR) and as the H3 `subject_definitions` entry baseline
- **`appearance.face`**: facial feature details for identity reinforcement in prompts
- **`appearance.hair`**: hair description, separated because it changes frequently (updos, hats, etc.)
- **`appearance.body`**: build and proportions
- **`appearance.default_outfit`**: default clothing, overrideable per scene instance
- **`voice.description`**: textual description of vocal quality, used in H3 `detailed_description` when citing audio references (e.g., "using the clear youthful voice timbre referenced from `<Audio 1>`")
- **`voice.audio_reference_file`**: filename of the audio reference clip (stored in ComfyUI's input directory or a configurable path). May be empty/null if no voice reference exists for this subject.
- **`voice.language`**: BCP-47 language tag for `<d>` dialogue tags in H3 prompts
- **`character_sheet_images`**: list of image filenames for reference inputs. Order matters — first image is the primary identity reference. Individual images, not composite sheets. Stored in ComfyUI's input directory or a configurable path.
- **`concept_id`**: links to the concept registry entry for LoRA resolution. This is the bridge between "who" (subject profile) and "how" (concept registry LoRAs).

### Nodes

#### SubjectProfileLoad

```
Inputs:
    subject_id: COMBO (populated from subject_profiles.json keys)

Outputs:
    subject_profile: SUBJECT_PROFILE (custom type)
    name: STRING
    appearance_summary: STRING
    character_sheet_images: IMAGE (batch of all character sheet images loaded)
    audio_reference: AUDIO (loaded audio file, or None if not defined)
    concept_id: STRING (for downstream concept registry resolution)
```

Loads a subject profile from the JSON file. Loads character sheet images from disk into an IMAGE batch. Loads audio reference if defined. Outputs individual fields for flexible downstream use as well as the full profile object.

#### SubjectProfileDefine

```
Inputs:
    subject_id: STRING
    name: STRING
    appearance_summary: STRING (multiline)
    face: STRING (multiline)
    hair: STRING
    body: STRING
    default_outfit: STRING (multiline)
    voice_description: STRING (multiline, optional)
    audio_reference_file: STRING (optional, file path)
    language: COMBO ["en-us", "en-gb", "ja", "ko", "zh", "es", "fr", "de", ...]
    concept_id: STRING (optional, links to concept registry)
    auto_save: BOOLEAN (default true)

Outputs:
    subject_profile: SUBJECT_PROFILE
```

Creates or updates a subject profile entry. Character sheet images are managed separately (added via a dedicated node or by editing the JSON directly) since image file management is better handled outside the node graph.

When `auto_save` is true, persists to `subject_profiles.json` immediately.

#### SubjectProfileList

```
Inputs:
    (none required)

Outputs:
    subject_list: STRING (formatted list of all subjects)
    subject_count: INT
```

Utility node for viewing all defined subjects. Output format:

```
[character_a] Sarah - a woman with dark shoulder-length hair...
[character_b] Marcus - a tall man with short cropped hair...
```

---

## Phase 2: Scene Templates

### Purpose

A scene template defines the structural blueprint for a video generation — shot structure, camera work, environment, style, and soundscape — with placeholder slots for subjects. Templates are model-agnostic; the prompt assembler (Phase 4) handles model-specific formatting.

### Data Schema

File: `ComfyUI/user/default/comfyui-fbTools/scene_templates/` (one JSON file per template)

Example: `scene_templates/cafe_conversation_2p.json`

```json
{
    "version": 1,
    "id": "cafe_conversation_2p",
    "name": "Café Conversation (2 speakers)",
    "description": "Two characters having a conversation in a café setting",
    "slots": {
        "A": {
            "role": "primary speaker",
            "needs_voice": true,
            "needs_character_sheet": true
        },
        "B": {
            "role": "secondary speaker",
            "needs_voice": true,
            "needs_character_sheet": true
        }
    },
    "environment": {
        "summary": "a warm, sunlit café with exposed brick walls, wooden tables, and ambient chatter",
        "lighting": "warm natural window light from the left, soft shadows"
    },
    "style": "cinematic with shallow depth of field and warm color grading",
    "shots": [
        {
            "id": "shot_1",
            "timestamp": null,
            "camera": "Medium two-shot from across the table",
            "action": "{A} sits across from {B}, holding a coffee cup. {A} leans forward and speaks.",
            "dialogue": {
                "speaker_slot": "A",
                "placeholder": true,
                "default_text": null
            },
            "sound_events": null
        },
        {
            "id": "shot_2",
            "timestamp": "00:04.000",
            "camera": "Close-up of {B}, shallow depth of field",
            "action": "{B} listens with interest, then responds with a slight smile.",
            "dialogue": {
                "speaker_slot": "B",
                "placeholder": true,
                "default_text": null
            },
            "sound_events": null
        },
        {
            "id": "shot_3",
            "timestamp": "00:08.000",
            "camera": "Medium shot favoring {A}, {B} slightly out of focus",
            "action": "{A} reacts to {B}'s response, laughs warmly, and picks up the coffee cup.",
            "dialogue": null,
            "sound_events": "quiet laughter, coffee cup clink"
        }
    ],
    "overall_soundscape": "Soft café ambience with quiet background chatter, espresso machine hiss, and the occasional clink of cups.",
    "non_diegetic_music": "N/A"
}
```

### Template Placeholder Convention

Subject placeholders use `{SLOT_ID}` syntax in action text and camera descriptions:

- `{A}` — replaced with subject appearance description at assembly time
- `{B}` — replaced with second subject's appearance

Slot IDs are single uppercase letters (A, B, C, ...) for brevity. Mapped to H3's `<Subject N>` labels and `(SN)` speaker IDs during assembly.

### Shot Structure

Each shot contains:

- **`id`**: unique identifier within the template (shot_1, shot_2, ...)
- **`timestamp`**: H3-format timestamp (null for the first shot, `MM:SS.mmm` for subsequent). Used directly in H3 prompts. For non-H3 models, timestamps are informational only.
- **`camera`**: camera description with placeholders. Becomes part of the shot description in all model formats.
- **`action`**: what happens in the shot, with subject placeholders. The core narrative content.
- **`dialogue`**: if present, specifies which slot speaks and whether the text is a placeholder (to be filled at instance time) or has default text. Can be null for non-speaking shots.
- **`sound_events`**: shot-specific sound effects or audio events. Distinct from overall_soundscape.

### Dialogue Structure

Dialogue entries can be:

1. **Placeholder with no default**: `{"speaker_slot": "A", "placeholder": true, "default_text": null}` — must be filled at instance time
2. **Placeholder with default**: `{"speaker_slot": "A", "placeholder": true, "default_text": "Hello there."}` — can be overridden at instance time, falls back to default
3. **Fixed text**: `{"speaker_slot": "A", "placeholder": false, "default_text": "Hello there."}` — always uses this text, not overrideable
4. **Null**: no dialogue in this shot

### Templates for Different Subject Counts

Templates are defined per slot configuration:

- `cafe_conversation_1p.json` — monologue (1 subject)
- `cafe_conversation_2p.json` — dialogue (2 subjects)
- `meeting_room_3p.json` — multi-party (3 subjects)

The slot count is fixed per template. Assigning subjects to a 2-slot template always requires exactly 2 subjects.

### Nodes

#### SceneTemplateLoad

```
Inputs:
    template_id: COMBO (populated by scanning scene_templates/ directory)

Outputs:
    template: SCENE_TEMPLATE (custom type)
    slot_info: STRING (formatted list of slots and their requirements)
```

Loads a scene template from its JSON file. The `slot_info` output shows what each slot needs:

```
Slot A: primary speaker (voice: required, character sheet: required)
Slot B: secondary speaker (voice: required, character sheet: required)
```

#### SceneTemplateList

```
Inputs:
    (none required)

Outputs:
    template_list: STRING (formatted list of all templates)
    template_count: INT
```

Scans the `scene_templates/` directory and lists all available templates with their names, descriptions, and slot counts.

---

## Phase 3: Scene Instance / Composition

### Purpose

A scene instance assigns specific subjects to a template's slots, fills in dialogue, and applies per-scene overrides. This is the per-generation configuration — it connects persistent data (subjects, templates) to the current creative intent.

### Nodes

#### SceneCompose

This is the primary composition node. It takes a template, subject assignments, and dialogue, and produces a complete scene instance ready for prompt assembly.

```
Inputs:
    template: SCENE_TEMPLATE
    slot_A: SUBJECT_PROFILE
    slot_B: SUBJECT_PROFILE (optional — only if template has slot B)
    slot_C: SUBJECT_PROFILE (optional — only if template has slot C)
    dialogue_1: STRING (multiline, optional — fills first dialogue placeholder)
    dialogue_2: STRING (multiline, optional — fills second dialogue placeholder)
    dialogue_3: STRING (multiline, optional — fills third dialogue placeholder)
    outfit_override_A: STRING (optional — overrides subject A's default outfit)
    outfit_override_B: STRING (optional — overrides subject B's default outfit)
    outfit_override_C: STRING (optional — overrides subject C's default outfit)

Outputs:
    scene_instance: SCENE_INSTANCE (custom type)
    scene_summary: STRING (human-readable summary of the composed scene)
```

#### Design Considerations for SceneCompose

**Slot assignment flexibility**: The node has fixed slot inputs (slot_A, slot_B, slot_C) rather than dynamic inputs. Unused slots (for templates with fewer subjects) are left unconnected. The node validates that all required slots are filled and that optional slots match the template's slot count.

**Dialogue assignment**: Dialogue inputs are positional — `dialogue_1` fills the first placeholder dialogue in shot order, `dialogue_2` fills the second, etc. The node maps these to the correct shots based on the template's dialogue placeholder ordering.

**Outfit overrides**: Per-slot outfit overrides replace the subject's `default_outfit` for this scene only. This handles scenarios where the same character wears different clothing across scenes without modifying the subject profile.

**Subject interchangeability**: Swapping subjects is as simple as connecting a different SubjectProfileLoad to a slot input. The same template, same dialogue, different subjects — the prompt assembler handles all the downstream text changes automatically.

**Maximum slot count**: Supporting up to 4 slots (A through D) covers the vast majority of use cases. Templates requiring more than 4 subjects are rare and can be handled by splitting into multiple shots/templates.

---

## Phase 4: Prompt Assembly

### Purpose

Takes a scene instance and generates the complete, model-specific prompt. This is where the structural boilerplate is automated — the assembler knows each model's prompt format and generates the correct structure from the composed scene data.

### Nodes

#### PromptAssemble

```
Inputs:
    scene_instance: SCENE_INSTANCE
    model_type: COMBO [h3_ref2va, h3_fl2va, wan22, bernini, ltx23, flux2, krea2, qwen]
    concept_registry: CONCEPT_REGISTRY (from concept registry loader)

Outputs:
    prompt: STRING (the assembled prompt text)
    reference_images: IMAGE (batch of all character sheet images, ordered by slot)
    reference_audio: AUDIO (first subject's audio reference, or None)
    additional_audio: AUDIO (second subject's audio reference, or None)
    concept_ids: STRING (comma-separated concept IDs for LoRA resolution)
    assembly_report: STRING (detailed report of what was assembled)
```

### Assembly Logic Per Model Type

#### H3 Ref2VA Assembly

Generates the full 6-section structured brief:

**subject_definitions:**
- For each assigned slot, generate a `<Subject N>` entry using the subject's appearance summary
- For each subject with character sheet images, generate `<Picture N>` entries
- For each subject with a voice reference, generate `<Audio N>` entries with voice description and speaker ID mapping
- Speaker IDs (S1, S2, ...) assigned in slot order (A=S1, B=S2, ...)
- Reference labels assigned in order: subjects first, then pictures, then audio

**summary:**
- Auto-detect task types from the instance:
  - Has character sheet images → `reference generation`
  - Has audio references → `+ audio reference`
  - Has video references → `+ video editing` or `+ video continuation`
- Generate summary paragraph using reference labels

**retention_analysis:**
- All assigned subjects: `fully_preserved` with appearance characteristics listed
- Audio references: `reference` (voice timbre only, not `fully_copy`)
- Generate one line per reference label

**detailed_description:**
- Open with style sentence from template
- For each shot:
  - Write `[Shot N]` header with timestamp
  - Replace `{SLOT_ID}` placeholders with full subject descriptions including `<Subject N>` labels and `(SN)` speaker IDs
  - At first appearance of each subject, include full appearance description
  - At subsequent appearances, use label only with brief identifier
  - Insert dialogue with `<d>[language] text</d>` tags
  - Reference voice timbre from `<Audio N>` at first dialogue occurrence
  - Include camera and action descriptions
  - Include shot-specific sound events

**overall_soundscape:**
- Direct from template

**non_diegetic_music:**
- Direct from template

#### H3 FL2VA Assembly

Simpler than Ref2VA — no reference labels, no subject_definitions section. Generates a descriptive prompt with shot structure and dialogue tags but without the reference framework.

#### Wan 2.2 / BerniniR Assembly

Generates a production-direction format prompt:

```
[Task type]: [action description].
The reference image defines [subject appearance from profile].
Preserve [environment and lighting from template].
[Camera descriptions from shots].
Do not [constraints].
```

Trigger texts from the concept registry are appended/prepended based on configuration. No H3-style sections — single block of descriptive text.

#### LTX 2.3 / Flux 2 / Krea 2 / Qwen Assembly

Generates model-appropriate prompt format. Typically simpler descriptive text, with trigger words from concept registry appended. The assembler should be extensible for new model formats.

### Assembly Report

The `assembly_report` output provides a human-readable summary:

```
Scene: Café Conversation (2 speakers)
Model: h3_ref2va
Subjects:
  Slot A → Sarah (character_a) as S1 [voice: yes, sheets: 4 images]
  Slot B → Marcus (character_b) as S2 [voice: yes, sheets: 3 images]
Dialogue:
  Shot 1 (S1): "I've been thinking about what you said..."
  Shot 2 (S2): "Really? I didn't think you were listening."
Reference media:
  Images: 7 total (4 from Sarah, 3 from Marcus)
  Audio: 2 files (sarah_voice_ref.wav, marcus_voice_ref.wav)
Concepts for LoRA resolution:
  character_a (h3_ref2va), character_b (h3_ref2va)
Outfit overrides:
  Slot A: grey sweater, hair pulled back (overrides default navy blazer)
```

---

## Phase 5: Integration with Existing Systems

### Concept Registry Integration

The PromptAssemble node outputs `concept_ids` — a list of concept IDs from all assigned subjects. Feed this into the existing ConceptResolve node to apply the correct LoRAs:

```
[PromptAssemble]
  concept_ids: "character_a, character_b"
    ↓
[ConceptResolve]
  concepts ← concept_ids from assembler
  model_type: h3_ref2va
  model ← checkpoint
  ...
    ↓
  model (with LoRAs applied)
  prompt ← from PromptAssemble (not from ConceptResolve)
```

Note: when using PromptAssemble, the prompt comes from the assembler (which includes trigger words from subject profiles), not from ConceptResolve's trigger text assembly. The concept registry is used only for LoRA resolution in this flow. The ConceptResolve node should support a mode where it applies LoRAs without modifying the prompt, or the assembled prompt should be passed through a separate input that bypasses trigger text injection.

### Reference Media Routing

The PromptAssemble node outputs reference images and audio as ComfyUI types (IMAGE, AUDIO). These connect directly to model-specific conditioning nodes:

**For H3 Ref2VA:**
```
reference_images → MiniMaxH3ReferenceToVideo (image references)
reference_audio → MiniMaxH3ReferenceToVideo (audio reference)
prompt → text input
```

**For BerniniR:**
```
reference_images → BerniniConditioning (reference_image_0, reference_image_1, ...)
prompt → CLIPTextEncode
```

**For Wan 2.2:**
```
reference_images → character sheet conditioning (if applicable)
prompt → CLIPTextEncode
```

The assembler outputs media in the order expected by each model's reference inputs (H3 expects images in the order they're cited in the prompt as `<Picture 1>`, `<Picture 2>`, etc.).

---

## Implementation Order

### Step 1: Subject Profiles (Foundation)

Build the persistent JSON storage and the three subject profile nodes (Load, Define, List). This is independently useful — subject profiles can be used manually in prompts even without templates or auto-assembly.

**Deliverables:**
- `subject_profiles.json` schema and read/write utilities
- `SubjectProfileLoad` node (loads profile, images, audio)
- `SubjectProfileDefine` node (creates/updates profiles)
- `SubjectProfileList` node (utility listing)

**Test by:** Loading a subject profile and manually using its appearance text and character sheet images in an existing workflow.

### Step 2: Scene Templates

Build the template JSON schema and the two template nodes (Load, List). Templates are useful even with manual subject insertion — they provide reusable shot structures.

**Deliverables:**
- `scene_templates/` directory structure
- Template JSON schema
- `SceneTemplateLoad` node
- `SceneTemplateList` node
- 2-3 example templates (monologue, 2-person dialogue, 3-person scene)

**Test by:** Loading a template and reviewing its slot_info output. Creating custom templates by editing JSON directly.

### Step 3: Scene Composition

Build the SceneCompose node that connects subjects to template slots. This produces a complete scene instance that can be inspected via its summary output.

**Deliverables:**
- `SceneCompose` node with slot assignment, dialogue input, outfit overrides
- `SCENE_INSTANCE` custom type
- Validation logic (all required slots filled, dialogue count matches placeholders)

**Test by:** Composing a scene with 2 subjects, verifying the summary output shows correct assignments. Swapping subjects and verifying the summary updates.

### Step 4: Prompt Assembly — H3 Format

Build the PromptAssemble node with H3 Ref2VA as the first output format. H3 is the most complex format and validates that the architecture handles structured multi-section prompts correctly.

**Deliverables:**
- `PromptAssemble` node with H3 Ref2VA assembly
- Full 6-section brief generation
- Reference media output routing
- Assembly report generation

**Test by:** Generating a complete H3 prompt from a composed scene. Running it through the H3 Ref2VA workflow. Comparing output quality against manually-written H3 prompts.

### Step 5: Prompt Assembly — Additional Formats

Add assembly logic for remaining model types. Each format is a separate method in the assembler, selected by the `model_type` input.

**Deliverables:**
- Wan 2.2 / BerniniR production-direction format
- LTX 2.3 / Flux 2 / Krea 2 / Qwen simple descriptive format
- H3 FL2VA format (simpler than Ref2VA)

**Test by:** Generating prompts for the same scene across multiple model types. Verifying each produces model-appropriate output.

### Step 6: Workflow Integration

Connect the full pipeline end-to-end: subject profiles → concept registry → scene template → composition → prompt assembly → model-specific nodes → generation.

**Deliverables:**
- Example workflow JSON files for each supported model type
- Documentation for creating custom templates
- Documentation for defining new subject profiles

---

## File Structure Summary

```
ComfyUI/user/default/comfyui-fbTools/
├── concept_registry.json          (existing — LoRA mappings)
├── subject_profiles.json          (new — subject definitions)
└── scene_templates/               (new — template directory)
    ├── monologue_indoor.json
    ├── cafe_conversation_2p.json
    ├── meeting_room_3p.json
    └── ...
```

## Custom Types Summary

| Type | Purpose | Passed Between |
|---|---|---|
| CONCEPT_REGISTRY | LoRA mappings per concept per model | ConceptRegistryLoad → ConceptResolve |
| SUBJECT_PROFILE | Subject identity, appearance, voice | SubjectProfileLoad → SceneCompose |
| SCENE_TEMPLATE | Shot structure with placeholder slots | SceneTemplateLoad → SceneCompose |
| SCENE_INSTANCE | Composed scene ready for assembly | SceneCompose → PromptAssemble |

## Node Summary

| Node | Category | Phase | Purpose |
|---|---|---|---|
| SubjectProfileLoad | fbTools/Scene | 1 | Load subject from persistent storage |
| SubjectProfileDefine | fbTools/Scene | 1 | Create/update subject profiles |
| SubjectProfileList | fbTools/Scene | 1 | List all defined subjects |
| SceneTemplateLoad | fbTools/Scene | 2 | Load a scene template |
| SceneTemplateList | fbTools/Scene | 2 | List all available templates |
| SceneCompose | fbTools/Scene | 3 | Assign subjects to template slots |
| PromptAssemble | fbTools/Scene | 4 | Generate model-specific prompt |

---

## Design Principles

1. **Separation of concerns**: Who (subject profiles) is separate from how (concept registry) is separate from what happens (scene templates) is separate from this specific generation (scene instance).

2. **Persistence where it matters**: Subjects and templates are persistent (they're reusable assets). Scene instances are ephemeral (they're per-generation configurations stored only in the workflow).

3. **Model agnosticism in data, model specificity in assembly**: Templates and subjects don't know about model formats. Only the PromptAssemble node contains model-specific logic. Adding a new model format means adding one assembly method, not changing the data layer.

4. **Progressive usefulness**: Each phase is independently useful. Subject profiles work without templates. Templates work without auto-assembly. The full stack is the goal, but partial implementation provides value.

5. **Composability over configuration**: Rather than one mega-node with 50 inputs, the system uses small focused nodes that chain together. This fits ComfyUI's visual programming paradigm and lets users build custom composition patterns.

6. **Interchangeability as a first-class feature**: Swapping a subject in a scene is a single connection change, not a prompt rewrite. The system handles all downstream text changes automatically.
