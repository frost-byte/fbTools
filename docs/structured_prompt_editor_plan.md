# Structured Prompt Editor — Implementation Plan

## Overview

A ComfyUI frontend extension for composing structured video generation prompts, built as part of the `comfyui-fbTools` package. The editor provides a rich authoring environment for building prompts with insertable, swappable subject references, inline dropdowns, and automatic structural formatting — all without requiring workflow execution.

The editor is **not a node**. It is a UI panel (sidebar or dialog) that runs independently of the execution graph. Users author prompts while workflows are generating. The output is a completed prompt string (plus associated media references) that feeds into a minimal graph node for generation.

This extension follows ComfyUI's conventions for frontend extensions and Python API endpoints. The existing `comfyui-fbTools` codebase already has patterns for registering API routes and serving frontend assets — follow those conventions throughout.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│ Browser (ComfyUI Frontend)                          │
│                                                     │
│  ┌───────────────────────────────────────────────┐  │
│  │ Structured Prompt Editor (JS/HTML/CSS)         │  │
│  │                                               │  │
│  │  - Subject insertion & dropdowns              │  │
│  │  - Shot block management                      │  │
│  │  - Dialogue formatting                        │  │
│  │  - Auto reference label numbering             │  │
│  │  - Raw prompt preview                         │  │
│  │  - Scene save/load                            │  │
│  └──────────────┬────────────────────────────────┘  │
│                 │ fetch() calls                      │
│                 ▼                                    │
│  ComfyUI Server                                     │
│  ┌───────────────────────────────────────────────┐  │
│  │ Python API Routes (/fbtools/...)              │  │
│  │                                               │  │
│  │  - Subject profile CRUD                       │  │
│  │  - Background/setting CRUD                    │  │
│  │  - Scene save/load                            │  │
│  │  - Prompt assembly & preview                  │  │
│  │  - Camera/sound preset management             │  │
│  └──────────────┬────────────────────────────────┘  │
│                 │                                    │
│                 ▼                                    │
│  Persistent JSON Storage                            │
│  ComfyUI/user/default/comfyui-fbTools/              │
│    subject_profiles.json                            │
│    backgrounds.json                                 │
│    camera_presets.json                              │
│    sound_presets.json                               │
│    scenes/                                          │
│      cafe_conversation_v1.json                      │
│      rainy_street_monologue.json                    │
└─────────────────────────────────────────────────────┘
```

### Relationship to the Execution Graph

The editor produces data. The graph consumes it. One minimal node bridges the two:

```
[SceneLoader]
    scene_id: COMBO (populated from saved scenes)
    model_type: COMBO
    ↓
    prompt: STRING (pre-assembled by the editor)
    reference_images: IMAGE (loaded from subject profiles)
    reference_audio: AUDIO (loaded from subject profiles)
    concept_ids: STRING (for concept registry LoRA resolution)
```

This node performs no assembly logic. It loads the already-composed scene from JSON, loads associated media files from disk, and outputs them for the pipeline. All authoring, previewing, and iterating happens in the UI panel before the workflow is queued.

---

## Backend: Python API Routes

Register all routes following the existing conventions in the `comfyui-fbTools` codebase for API endpoint registration. All routes use the `/fbtools/` prefix.

### Persistent Data Location

All JSON files stored in:

```python
import folder_paths
import os

package_dir = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
data_dir = os.path.join(folder_paths.get_user_directory(), package_dir)
```

Producing: `ComfyUI/user/default/comfyui-fbTools/`

### Route Groups

#### Subjects (`/fbtools/subjects/`)

| Method | Route | Purpose |
|---|---|---|
| GET | `/fbtools/subjects/list` | List all subject profiles |
| GET | `/fbtools/subjects/get?id=<id>` | Get a single subject profile |
| POST | `/fbtools/subjects/save` | Create or update a subject profile |
| DELETE | `/fbtools/subjects/delete?id=<id>` | Delete a subject profile |

Subject profile schema (same as defined in the scene composition action plan):

```json
{
    "id": "character_a",
    "name": "Sarah",
    "type": "character",
    "appearance": {
        "summary": "a woman with dark shoulder-length hair, light skin, hazel eyes",
        "face": "high cheekbones, defined jawline, dark arched eyebrows",
        "hair": "dark brown shoulder-length with a slight wave",
        "body": "athletic build, medium height",
        "default_outfit": "navy blazer over white shirt, black pants"
    },
    "voice": {
        "description": "clear youthful voice with a warm conversational tone",
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
```

#### Backgrounds (`/fbtools/backgrounds/`)

| Method | Route | Purpose |
|---|---|---|
| GET | `/fbtools/backgrounds/list` | List all background/setting definitions |
| GET | `/fbtools/backgrounds/get?id=<id>` | Get a single background |
| POST | `/fbtools/backgrounds/save` | Create or update a background |
| DELETE | `/fbtools/backgrounds/delete?id=<id>` | Delete a background |

Background schema:

```json
{
    "id": "cafe_interior",
    "name": "Café Interior",
    "description": "a warm, sunlit café with exposed brick walls, wooden tables, and soft ambient lighting from large windows",
    "lighting": "warm natural window light from the left, soft diffused shadows",
    "soundscape": "quiet background chatter, espresso machine hiss, occasional clink of cups"
}
```

#### Presets (`/fbtools/presets/`)

| Method | Route | Purpose |
|---|---|---|
| GET | `/fbtools/presets/cameras` | List camera description presets |
| GET | `/fbtools/presets/sounds` | List sound/ambience presets |
| POST | `/fbtools/presets/cameras/save` | Save a camera preset |
| POST | `/fbtools/presets/sounds/save` | Save a sound preset |

Camera preset schema:

```json
{
    "id": "medium_two_shot",
    "name": "Medium Two-Shot",
    "description": "Medium two-shot from across the table, both subjects visible from waist up"
}
```

Sound preset schema:

```json
{
    "id": "cafe_ambience",
    "name": "Café Ambience",
    "description": "Soft café ambience with quiet background chatter, espresso machine hiss, and occasional clink of cups"
}
```

#### Scenes (`/fbtools/scenes/`)

| Method | Route | Purpose |
|---|---|---|
| GET | `/fbtools/scenes/list` | List all saved scenes |
| GET | `/fbtools/scenes/get?id=<id>` | Load a complete scene |
| POST | `/fbtools/scenes/save` | Save a scene |
| DELETE | `/fbtools/scenes/delete?id=<id>` | Delete a scene |
| POST | `/fbtools/scenes/assemble` | Assemble a scene into a model-specific prompt |

Scene schema (what gets saved — the composed scene state):

```json
{
    "id": "cafe_talk_v1",
    "name": "Café Conversation v1",
    "model_type": "h3_ref2va",
    "style": "cinematic with shallow depth of field and warm color grading",
    "subjects": {
        "S1": "character_a",
        "S2": "character_b"
    },
    "outfit_overrides": {
        "S1": "grey sweater, hair pulled back"
    },
    "background": "cafe_interior",
    "shots": [
        {
            "id": "shot_1",
            "timestamp": null,
            "camera": "Medium two-shot from across the table",
            "action": "{S1} sits across from {S2}, holding a coffee cup. {S1} leans forward and speaks.",
            "dialogue": {
                "speaker": "S1",
                "language": "English",
                "text": "I've been thinking about what you said last week."
            },
            "sound_events": null
        },
        {
            "id": "shot_2",
            "timestamp": "00:04.000",
            "camera": "Close-up of {S2}, shallow depth of field",
            "action": "{S2} listens with interest, then responds with a slight smile.",
            "dialogue": {
                "speaker": "S2",
                "language": "English",
                "text": "Really? I didn't think you were even listening."
            },
            "sound_events": null
        }
    ],
    "overall_soundscape": "Soft café ambience with quiet background chatter and espresso machine hiss.",
    "non_diegetic_music": "N/A"
}
```

#### Assembly Route (`/fbtools/scenes/assemble`)

The assembly route takes a scene (either by ID or inline scene data) and a model_type, and returns the fully assembled prompt string with all references resolved.

Request:

```json
{
    "scene_id": "cafe_talk_v1",
    "model_type": "h3_ref2va"
}
```

Response:

```json
{
    "prompt": "subject_definitions:\n<Subject 1> is Sarah...\n...",
    "reference_images": ["sarah_closeup.png", "sarah_front.png", ...],
    "reference_audio": ["sarah_voice_ref.wav", "marcus_voice_ref.wav"],
    "concept_ids": ["character_a", "character_b"],
    "assembly_report": "Scene: Café Conversation v1\nModel: h3_ref2va\n..."
}
```

The assembly logic handles all model-specific formatting:

- **H3 Ref2VA**: full 6-section structured brief (subject_definitions, summary, retention_analysis, detailed_description, overall_soundscape, non_diegetic_music). Auto-generates `<Subject N>`, `<Picture N>`, `<Audio N>` labels. Auto-generates retention_analysis markers. Inserts `<d>` dialogue tags with language codes. Cites voice references inline.
- **H3 FL2VA**: simplified format without reference labels.
- **Wan 2.2 / BerniniR**: production-direction format (task type, reference roles, preservation instructions, camera/motion, constraints).
- **LTX 2.3 / Flux 2 / Krea 2 / Qwen**: simple descriptive prompt with trigger words from concept registry appended.

---

## Frontend: Structured Prompt Editor

The frontend is a ComfyUI extension panel. Follow the existing conventions in the `comfyui-fbTools` codebase for registering frontend extensions and serving static assets (JS, CSS, HTML).

### Panel Registration

Register the editor as a sidebar panel, dialog, or menu-accessible tool within ComfyUI's frontend extension system. The panel should be openable and closable without affecting the workflow graph, and should remain functional while workflows are executing.

### Layout

The editor has three main areas:

```
┌──────────────────┬──────────────────────────────────────┐
│ Resource Sidebar  │ Editor Area                          │
│                  │                                      │
│ (collapsible)    │ (main editing surface)               │
│                  │                                      │
│ Lists of:        │ Section-structured editor with        │
│ - Subjects       │ inline smart elements                │
│ - Backgrounds    │                                      │
│ - Camera presets │                                      │
│ - Sound presets  │                                      │
│ - Saved scenes   │                                      │
│                  ├──────────────────────────────────────┤
│                  │ Action Bar                            │
│                  │ [Preview Raw] [Copy] [Save] [Send]   │
└──────────────────┴──────────────────────────────────────┘
```

### Resource Sidebar

Fetches data from the backend API routes on panel open. Each resource category is a collapsible section showing available items. Items can be:

- **Clicked** to insert at the editor cursor position
- **Dragged** into the editor (if drag-and-drop is implemented)
- **Right-clicked** for edit/delete options

The sidebar also includes a **"New"** button per category for creating new subjects, backgrounds, or presets inline without leaving the editor.

### Editor Area

The editor is a structured text editing surface. It is NOT a plain textarea — it is a rich editor that understands the prompt's section structure and supports inline smart elements.

#### Section Structure

The editor displays the prompt in sections. For H3, these are pre-populated as collapsible/expandable blocks:

```
▾ subject_definitions
  [editable content area]

▾ summary  
  [editable content area]

▾ retention_analysis
  [auto-generated, read-only or editable]

▾ detailed_description
  [primary editing area — shots, action, dialogue]

▾ overall_soundscape
  [editable content area]

▾ non_diegetic_music
  [editable content area]
```

For non-H3 models, the section structure simplifies to a single editing area appropriate to the model's prompt format.

The `retention_analysis` section should be auto-generated from the subjects and audio references currently in use. It updates automatically as subjects are added or removed. It can optionally be manually edited for fine-tuning.

The `summary` section can be auto-generated from the scene configuration (task types, subject roles, reference relationships) with manual override capability.

#### Smart Element: Subject Insertion

**Trigger**: typing `/subject` or `/s` in the editor, or clicking a subject in the sidebar.

**Behavior**: opens a dropdown/autocomplete populated from the backend's subject list. Selecting a subject:

1. Determines the next available `<Subject N>` number by scanning existing references in the document
2. Determines the next available `(SN)` speaker ID
3. Inserts the subject's appearance text at the cursor, formatted with the reference label and speaker ID:
   ```
   <Subject 1> (S1), the woman with dark shoulder-length hair, 
   light skin, hazel eyes, wearing navy blazer over white shirt,
   ```
4. If the subject has a voice reference defined, auto-adds to `subject_definitions`:
   ```
   <Audio 1> is the voice-timbre reference for <Subject 1> (S1).
   ```
5. Auto-adds to `retention_analysis`:
   ```
   <Subject 1> (appears in [Shot N]): fully_preserved - [appearance summary]
   <Audio 1>: reference - vocal timbre guides dialogue delivery without copying the original signal.
   ```
6. Tracks the subject-to-label mapping internally so subsequent references to the same subject use the same labels

**Inline dropdown**: after insertion, the subject reference appears as a styled inline element (highlighted text, small dropdown indicator). Clicking it opens a dropdown to swap the subject. Swapping cascades: updates all references, appearance text, audio entries, and retention_analysis throughout the document.

#### Smart Element: Background Insertion

**Trigger**: typing `/background` or `/bg` in the editor, or clicking a background in the sidebar.

**Behavior**: opens a dropdown of saved backgrounds. Selecting one inserts:
- The background description text at the cursor
- Updates `overall_soundscape` section with the background's soundscape if defined

Background references also appear as styled inline elements with swap capability.

#### Smart Element: Shot Block Insertion

**Trigger**: typing `/shot` in the editor.

**Behavior**: inserts a new shot block:

```
[Shot N] At MM:SS.000, 
```

- Shot number auto-incremented from existing shots in the document
- Timestamp placeholder with cursor positioned for editing (first shot has no timestamp)
- Optionally includes sub-fields for camera, action, dialogue as guided placeholders:

```
[Shot 3] At 00:08.000, 
Camera: [click to set or type]
Action: [describe what happens]
Dialogue: [/dialogue to insert]
Sound: [optional sound events]
```

These sub-fields are editor affordances — in the raw output they're composed into natural prose following the H3 format.

#### Smart Element: Dialogue Insertion

**Trigger**: typing `/dialogue` or `/d` inside a shot block.

**Behavior**: opens a speaker dropdown showing assigned subjects (S1, S2, etc. with names). Selecting a speaker inserts:

```
, using the voice timbre referenced from <Audio 1>, says 
<d>[English] |</d>
```

- Voice reference citation auto-included only if the speaker has a voice reference
- Language tag pulled from the subject's profile
- Cursor positioned inside the `<d>` tags for typing dialogue text
- Speaker dropdown pre-populated from subjects already assigned in the document

If this is the first dialogue for this speaker, the voice reference citation is included. Subsequent dialogue by the same speaker omits the voice reference citation (per H3 convention — cite at first occurrence only).

#### Smart Element: Camera Preset

**Trigger**: typing `/camera` or `/cam` inside a shot block.

**Behavior**: opens a dropdown of saved camera presets. Selecting one inserts the camera description text. The user can then edit the inserted text to customize.

#### Smart Element: Sound Event

**Trigger**: typing `/sound` inside a shot block.

**Behavior**: opens a dropdown of saved sound presets. Selecting one inserts the sound description.

### Command Palette

All smart elements are accessible via a unified command palette triggered by `/` at the start of a line or after whitespace. The palette shows:

```
┌─────────────────────────┐
│ /subject   Insert subject│
│ /bg        Insert setting│
│ /shot      New shot block│
│ /dialogue  Add dialogue  │
│ /camera    Camera preset │
│ /sound     Sound event   │
└─────────────────────────┘
```

Typing further characters filters the list (e.g., `/s` shows `/subject`, `/shot`, `/sound`). Arrow keys navigate, Enter selects.

### Keyboard Shortcuts

| Shortcut | Action |
|---|---|
| `/` | Open command palette |
| `Ctrl+S` | Save current scene |
| `Ctrl+Shift+P` | Preview raw prompt |
| `Ctrl+Shift+C` | Copy assembled prompt to clipboard |
| `Ctrl+Enter` | Send to workflow (set SceneLoader node) |
| `Ctrl+Shift+N` | New shot block |
| `Ctrl+Shift+D` | Insert dialogue at cursor |

### Action Bar

Located below the editor area:

- **Preview Raw**: calls the assembly route and displays the fully resolved prompt in a read-only overlay. This is the exact text that would be sent to the model. All smart elements resolved to plain text, all reference labels numbered, all sections formatted.
- **Copy**: assembles and copies the prompt to clipboard.
- **Save Scene**: saves the current editor state to a scene JSON file via the backend. Prompts for a scene name if saving for the first time.
- **Send to Workflow**: assembles the prompt and pushes it to the SceneLoader node in the active workflow. Sets the node's `scene_id` widget value so the next queue execution uses this scene.

### Model Type Selector

A dropdown at the top of the editor selects the target model type:

```
Target Model: [H3 Ref2VA ▼]
```

Options: H3 Ref2VA, H3 FL2VA, Wan 2.2, BerniniR, LTX 2.3, Flux 2, Krea 2, Qwen

Changing the model type:
- Adjusts the section structure (H3 gets 6 sections; other models get simpler layouts)
- Changes how the Preview Raw output is formatted
- Does NOT change the scene data — subjects, shots, dialogue remain the same
- Allows side-by-side comparison of the same scene across model formats

### Scene Management

The sidebar's "Saved Scenes" section lists all saved scenes. Operations:

- **Load**: replaces the current editor content with the saved scene
- **Duplicate**: creates a copy with a new name (for creating variations)
- **Delete**: removes the scene file
- **Rename**: changes the scene name/ID

Unsaved changes are indicated with a visual marker (dot on the tab, asterisk in the title, etc.). Attempting to load a different scene or close the panel with unsaved changes shows a confirmation prompt.

---

## Graph Node: SceneLoader

A single minimal node that bridges the editor's output into the execution graph.

### V3 API Definition

```python
class SceneLoader(IO.ComfyNode):
    """
    Loads a pre-composed scene from the structured prompt editor.
    All scene authoring happens in the editor UI panel —
    this node only retrieves the result for pipeline consumption.
    """
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SceneLoader_fb",
            display_name="Scene Loader",
            description="Loads a composed scene from the prompt editor. "
                        "Use the Structured Prompt Editor panel to create and edit scenes.",
            category="fbTools/Scene",
            inputs=[
                IO.Combo.Input("scene_id", options=[]),  # populated dynamically
                IO.Combo.Input("model_type", options=[
                    "h3_ref2va", "h3_fl2va", "wan22", "bernini",
                    "ltx23", "flux2", "krea2", "qwen"
                ]),
            ],
            outputs=[
                IO.String.Output("prompt", display_name="Prompt"),
                IO.Image.Output("reference_images", display_name="Reference Images"),
                IO.Audio.Output("reference_audio", display_name="Reference Audio"),
                IO.String.Output("concept_ids", display_name="Concept IDs"),
            ],
        )

    @classmethod
    def execute(cls, scene_id, model_type):
        # 1. Load scene JSON from persistent storage
        # 2. Resolve subject profiles for all assigned subjects
        # 3. Call the assembly logic for the specified model_type
        # 4. Load character sheet images from disk into IMAGE tensor batch
        # 5. Load audio reference files into AUDIO tensors
        # 6. Collect concept_ids from all assigned subjects
        # 7. Return all outputs
        ...
```

The `scene_id` COMBO should be dynamically populated by scanning the scenes directory. This can be done via the node's `define_schema` if V3 supports dynamic options, or via a separate mechanism depending on the codebase's conventions.

---

## Persistent Data Files

All files stored in `ComfyUI/user/default/comfyui-fbTools/`:

```
comfyui-fbTools/
├── concept_registry.json          (existing — LoRA mappings)
├── subject_profiles.json          (subjects with appearance, voice, images)
├── backgrounds.json               (settings/environments)
├── camera_presets.json             (reusable camera descriptions)
├── sound_presets.json              (reusable sound/ambience descriptions)
└── scenes/                        (one JSON per saved scene)
    ├── cafe_conversation_v1.json
    ├── cafe_conversation_v2.json
    ├── rainy_street_monologue.json
    └── ...
```

Scenes are stored as individual files (not in one monolithic JSON) so they can be managed independently — duplicated, shared, version-controlled, or deleted without affecting other scenes.

All other data (subjects, backgrounds, presets) is stored in single JSON files since the total count of each is expected to remain manageable (tens to low hundreds of entries, not thousands).

---

## Prompt Assembly Logic

The assembly logic lives in the Python backend (shared between the API route and the SceneLoader node). It is the core engine that transforms scene data + subject profiles into model-specific prompt text.

### H3 Ref2VA Assembly

Given a scene with subjects, shots, and dialogue, generate all 6 sections:

#### subject_definitions

For each assigned subject:
- Generate `<Subject N>` with appearance text from the subject profile
- If the subject has character sheet images, generate `<Picture N>` entries citing the source
- If the subject has a voice reference, generate `<Audio N>` as voice-timbre reference
- Number labels sequentially: subjects first, then pictures, then audio

Track the mapping between subject IDs (S1, S2) and reference labels (`<Subject 1>`, `<Audio 1>`) for use in subsequent sections.

#### summary

Auto-generate the task-type prefix by analyzing the scene:
- Has character sheet images → `reference generation`
- Has audio references → `+ audio reference`
- Has source video → `+ video editing` or `+ video continuation`

Generate a summary paragraph using reference labels.

#### retention_analysis

For each referenced element:
- Subjects: `fully_preserved` with appearance characteristics
- Audio references: `reference` (voice timbre, not copied)
- Picture references: `fully_preserved` or as appropriate
- One line per reference label

#### detailed_description

- Open with style sentence
- For each shot:
  - Write `[Shot N]` with timestamp (null for first shot)
  - Replace `{S1}`, `{S2}` placeholders with full subject descriptions including `<Subject N>` labels and `(SN)` speaker IDs
  - First appearance: include full appearance description
  - Subsequent appearances: label + brief identifier only
  - Insert dialogue with `<d>[language] text</d>` tags
  - Cite voice reference from `<Audio N>` at first dialogue per speaker
  - Include camera descriptions
  - Include sound events

#### overall_soundscape

From the scene's soundscape field, incorporating the background's soundscape if a background entity is assigned.

#### non_diegetic_music

From the scene's music field, or "N/A" if none.

### Wan 2.2 / BerniniR Assembly

Generate production-direction format:

```
[Task]: [action summary from shots].
The reference image defines [subject appearance from profiles].
Preserve [environment from background, lighting].
[Camera descriptions from shots, continuous motion description].
Do not [constraints — don't change appearance, environment, etc.].
```

Append trigger words from concept registry entries for the active subjects.

### Other Model Assemblies

Simple descriptive prompt format — concatenate appearance descriptions, action text from shots, environment description, and trigger words. Model-specific formatting as appropriate.

---

## Implementation Phases

### Phase 1: Backend API + Data Layer

Build all API routes, JSON persistence, and data schemas. This is the foundation everything else depends on.

**Deliverables:**
- API route registration following existing codebase conventions
- Subject profile CRUD routes with JSON persistence
- Background CRUD routes
- Camera and sound preset CRUD routes
- Scene save/load routes
- Assembly route with H3 Ref2VA formatter

**Test by:** calling routes directly via curl or browser dev tools. Verify JSON files are created and persisted correctly.

### Phase 2: Basic Editor Panel

Build the frontend panel with section-structured editing, resource sidebar, and basic text editing. No smart elements yet — just a structured editor with manual text input.

**Deliverables:**
- Panel registration in ComfyUI's frontend extension system
- Resource sidebar populating from backend API
- Section-structured editor for H3 format
- Model type selector
- Preview Raw button calling assembly route
- Copy to clipboard
- Save/load scenes

**Test by:** manually composing an H3 prompt in the editor, previewing the raw output, and verifying it matches expected H3 format.

### Phase 3: Smart Elements — Subject & Background

Add the `/subject` and `/background` insertion commands with inline dropdowns and swap capability.

**Deliverables:**
- Command palette (triggered by `/`)
- Subject insertion with auto-numbering of reference labels
- Auto-generation of `<Audio N>` entries for voiced subjects
- Auto-generation of `retention_analysis` entries
- Background insertion
- Inline dropdown elements for swapping subjects/backgrounds
- Cascade updates when swapping

**Test by:** inserting two subjects, swapping one, verifying all reference labels, audio entries, and retention_analysis update correctly throughout the document.

### Phase 4: Smart Elements — Shots & Dialogue

Add shot block insertion, dialogue insertion, and camera/sound presets.

**Deliverables:**
- `/shot` insertion with auto-numbering and timestamp placeholders
- `/dialogue` insertion with speaker selection and voice reference citation
- `/camera` preset insertion
- `/sound` preset insertion
- Keyboard shortcuts

**Test by:** composing a complete 3-shot scene using only smart elements and keyboard shortcuts. Preview raw output and verify correct H3 formatting.

### Phase 5: SceneLoader Node

Build the minimal graph node that loads composed scenes into the pipeline.

**Deliverables:**
- SceneLoader V3 API node
- Dynamic scene_id population from saved scenes
- Media file loading (images → IMAGE tensor, audio → AUDIO tensor)
- Concept ID extraction for downstream concept registry resolution
- "Send to Workflow" button in the editor that sets SceneLoader widget values

**Test by:** composing a scene in the editor, sending to workflow, queuing execution, verifying the SceneLoader outputs correct prompt, images, audio, and concept IDs.

### Phase 6: Additional Model Formats

Add assembly logic for non-H3 models. Adjust the editor's section structure per model type.

**Deliverables:**
- Wan 2.2 / BerniniR production-direction assembly
- LTX 2.3 / Flux 2 / Krea 2 / Qwen simple format assembly
- H3 FL2VA simplified format assembly
- Editor section structure adapts when model type changes
- Side-by-side preview of same scene across model formats (optional)

**Test by:** switching model type on the same scene, verifying each format produces model-appropriate output.

### Phase 7: LLM Integration (Optional)

Add optional LLM assistance for per-section prompt generation and refinement.

**Deliverables:**
- Per-section "Assist" button that sends section context to an LLM
- LLM expands/refines action descriptions, dialogue, camera directions
- LLM suggestions appear as editable proposals the user can accept, modify, or reject
- Works with locally-available LLM endpoints (Ollama, etc.) or external APIs

**Test by:** using LLM assist to expand a rough action description into detailed H3-format prose. Verify the output integrates cleanly with the existing smart elements and reference labels.

---

## Design Principles

1. **Editor-first, node-minimal**: all authoring happens in the UI panel. The execution graph receives finished data, not works-in-progress.

2. **Non-blocking**: the editor operates independently of workflow execution. Users compose prompts while generating video.

3. **Composable, not templated**: rather than filling in a rigid form, users compose freely with insertable smart elements. The structure comes from the elements themselves, not from a fixed layout.

4. **Swappable subjects**: changing who appears in a scene is a dropdown selection, not a prompt rewrite. All downstream references update automatically.

5. **Model-agnostic data, model-specific output**: scene data (subjects, shots, dialogue) is the same regardless of target model. Only the assembly step produces model-specific formatting.

6. **Progressive disclosure**: basic usage is typing text with `/` insertions. Advanced features (keyboard shortcuts, presets, LLM assist) are available but not required.

7. **Transparent output**: the "Preview Raw" button always shows exactly what the model will receive. No hidden transformations between what the user sees and what the model gets.

8. **Follow existing conventions**: all backend routes, frontend registration, file paths, and coding patterns follow the conventions already established in the `comfyui-fbTools` codebase. Do not introduce new patterns where existing ones apply.
