# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

`comfyui-fbTools` is a ComfyUI custom nodes extension (Python package) focused on storytelling, scene management, LoRA orchestration, image compositing, and dataset captioning workflows. All nodes appear under the **🧊 frost-byte** category in ComfyUI.

The extension registers via `comfy_entrypoint()` in `__init__.py`, which returns a `FBToolsExtension` (a `ComfyExtension` subclass). The `get_node_list()` method in `extension.py` enumerates every registered node class.

## Commands

### Python Tests

Run tests using the **venv Python**, not the system Python:

```bash
# All tests
/mnt/comfy_ssd/venvs/comfy-preflight/bin/python -m pytest tests/ -v

# Single file
/mnt/comfy_ssd/venvs/comfy-preflight/bin/python -m pytest tests/test_libber.py -v

# With coverage
/mnt/comfy_ssd/venvs/comfy-preflight/bin/python -m pytest tests/ --cov=. --cov-report=html

# With debug logging
FBTOOLS_LOG_LEVEL=DEBUG /mnt/comfy_ssd/venvs/comfy-preflight/bin/python -m pytest tests/ -v
```

### JavaScript Tests

`package.json` and `node_modules` are at the **repo root**. Test files are in `js-tests/`. Run all commands from the repo root:

```bash
npm install          # first time only
npm test
npm run test:watch
npm run test:coverage
```

### Linting

```bash
/mnt/comfy_ssd/venvs/comfy-preflight/bin/pip install -e .[dev]
pre-commit install
pre-commit run --all-files   # runs ruff linter + ruff-format

# Or run ruff directly (what CI does):
ruff check .
ruff format .
```

### Commit Convention

All commits must follow **[Conventional Commits](https://www.conventionalcommits.org/)** format:

```
type(optional-scope)?: description

feat: add LoRA stack export
fix(scene): handle missing mask_name
docs: update CLAUDE.md commit convention
refactor(caption): extract VLM backend base class
```

Valid types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`, `revert`, `ci`, `build`

The `commit-msg` hook in `hooks/commit-msg` enforces this. To install after a fresh clone:

```bash
cp hooks/commit-msg .git/hooks/commit-msg
chmod +x .git/hooks/commit-msg
# or run pre-commit install (picks it up via .pre-commit-config.yaml)
```

### Versioning & Releases

Releases are **fully automatic** via `python-semantic-release` on every push to `main`. Commit messages drive the version bump:

| Commit type | Version bump | Example |
|---|---|---|
| `feat:` | minor (0.**2**.0) | new node or API endpoint |
| `fix:`, `perf:` | patch (0.1.**2**) | bug fix or performance improvement |
| `feat!:` or `BREAKING CHANGE:` footer | major (**2**.0.0) | breaking API change |
| `docs:`, `refactor:`, `test:`, `chore:`, etc. | none | no release created |

The release workflow (`.github/workflows/release.yml`) on each push to `main`:
1. Analyzes commits since the last tag
2. If a release is warranted: bumps `version` in `pyproject.toml` and `__version__` in `__init__.py`, commits, creates a `vX.Y.Z` tag, and creates a GitHub Release with a generated changelog
3. The tag creation automatically triggers `publish_node.yml`, which publishes to the Comfy registry

No manual steps are needed. The version lives in two places kept in sync by `python-semantic-release`:
- `pyproject.toml` → `[project] version`
- `__init__.py` → `__version__`

### Frontend debug flags (browser console)

```javascript
fbToolsDebug.enable('API_CALLS', 'SCENE')   // enable specific flags
fbToolsDebug.all()                           // enable all
fbToolsDebug.clear()                         // reset
fbToolsDebug.list()                          // show current state
```

Flags persist in `localStorage` under `fb_tools_debug_flags`.

## Architecture

### Backend (Python)

All node classes live in `extension.py` and inherit from `io.ComfyNode` (ComfyUI V3 API from `comfy_api.latest`). Business logic is extracted into utility modules so it can be tested independently of ComfyUI.

Key files:

| File | Purpose |
|---|---|
| `extension.py` | All node definitions + REST API routes (~10 000 lines) |
| `prompt_models.py` | `PromptMetadata`, `PromptCollection` (Pydantic v2) |
| `story_models.py` | `SceneInStory`, `StoryInfo` (Pydantic v2) |
| `captioner.py` | VLM captioning backend (Qwen2.5-VL, Qwen2.5-Omni, Gemini) |
| `utils/util.py` | Pose drawing, node graph helpers |
| `utils/io.py` | JSON file I/O helpers |
| `utils/images.py` | Image processing (TailEnhancePro, aspect ratio, SAM, compositing math) |
| `utils/pose.py` | Pose/depth estimation wrappers (DWPose, OpenPose, DepthAnything, etc.) |
| `utils/subject_compositor.py` | Compositing math for SubjectLayerDefine/SubjectCompositor |
| `utils/nlf_pose.py` | NLF 3D pose (optional; gracefully absent if ComfyUI-SCAIL-Pose not installed) |
| `utils/logging_utils.py` | `get_logger()` — configures level from `FBTOOLS_LOG_LEVEL` env var |
| `utils/scene_image_save.py` | Scene image save config/helpers |
| `utils/story_video.py` | Story video generation helpers |
| `utils/concept_registry.py` | Pure concept registry logic (no ComfyUI deps) — models, resolve, persist |
| `utils/subject_profiles.py` | Pure subject profile logic (no ComfyUI deps) — `SubjectRegistry`, load/save/define |
| `utils/scene_templates.py` | Pure scene template logic (no ComfyUI deps) — `SceneTemplate`, scan/load/format |
| `utils/scene_compose.py` | Pure composition logic (no ComfyUI deps) — `compose_scene`, `validate_scene`, `format_scene_summary` |
| `utils/prompt_assembler.py` | Pure prompt assembly logic (no ComfyUI deps) — `assemble_prompt`, per-model-type formatters |
| `scene_templates/` | Bundled example templates (seeded into user_data_dir on first use) |

**Registered nodes** (from `FBToolsExtension.get_node_list()`): SubjectLayerDefine, SubjectCompositor, DatasetCaptioner, DatasetCaptionEditor, DatasetCaptionViewer, DatasetExportSummary, CaptionModelUnloader, FBTextEncodeQwenImageEditPlus (conditioning), SAMPreprocessNHWC, QwenAspectRatio, SubdirLister, MultiLoraLoader, SceneCreate, SceneUpdate, SceneMaskDefinition, SceneSave, SceneInput, SceneOutput, SceneView, SceneSelect, SceneWanVideoLoraMultiSave, SceneLoraStackSave, ScenePromptManager, PromptComposer, StorySceneBatch, StoryScenePick, StoryVideoBatch, StoryCreate, StoryEdit, StoryView, StorySave, StoryLoad, StorySceneImageSave, OpaqueAlpha, MaskProcessor, TailSplit, TailEnhancePro, LibberManager, LibberApply, **LoraStackBuilder** (primary LoRA path), LoraStackApply, LoraEntryDefine (legacy), LoraStackCollect (legacy), WanVidLoraStack, LoraPresetDefine, LoraPresetSelect, WanPresetDefine, WanPresetSelect, AudioFixShape, ConceptRegistryLoad, ConceptDefine, ConceptResolve, ConceptList, **SubjectProfileLoad**, **SubjectProfileDefine**, **SubjectProfileList**, **SceneTemplateLoad**, **SceneTemplateList**, **SceneCompose**, **PromptAssemble**. `LoraStackView` is **defined but not registered** — it will not appear in ComfyUI until added to `get_node_list()`.

**Node categories** — use one of these existing values when adding a new node:

`compositing`, `conditioning`, `Dataset`, `File`, `Image Processing`, `Libber`, `Loaders`, `lora`, `Nodes`, `Preprocessing`, `Scene`, `Story`, `Video`

Full form: `"🧊 frost-byte/<category>"` (e.g., `"🧊 frost-byte/Scene"`).

**REST API endpoints** are registered at the bottom of `extension.py` against `PromptServer.instance.routes` (aiohttp). All routes are prefixed `/fbtools/`. Key groups:
- `/fbtools/prompts/*` — PromptCollection session CRUD
- `/fbtools/libber/*` — Libber template CRUD
- `/fbtools/scene/*` — Scene prompt processing, list, thumbnail
- `/fbtools/story/*` — Story load/save/list/thumbnails
- `/fbtools/dataset_caption/*` — Caption list/edit/save/recaption
- `/fbtools/concepts/reload` (POST) — increment reload counter so ConceptRegistryLoad re-executes
- `/fbtools/concepts/registry` (GET) — return default registry as JSON

**State managers** (`PromptCollectionManager`, `LibberManager` at module level in `extension.py`) hold server-side session state with TTL.

**Node ID convention**: all node IDs are prefixed with `fbt_` via `prefixed_node_id()`. The frontend references them with the constant `EXT_PREFIX = "fbt_"`.

### Frontend (JavaScript)

The `WEB_DIRECTORY = "./js"` tells ComfyUI to serve everything in `js/` as static assets.

| File/Dir | Purpose |
|---|---|
| `js/fb_tools.js` | Main extension registered with `app.registerExtension`; handles node lifecycle hooks |
| `js/index.js` | Re-exports all API clients and utilities |
| `js/api/*.js` | REST API client classes (one per domain: libber, prompt_collection, scene, story, dataset_caption) |
| `js/nodes/*.js` | Node-specific UI handlers imported by `fb_tools.js` |
| `js/utils/api_base.js` | `BaseAPI` class with fetch + error handling |
| `js/utils/debug_config.js` | Bitwise debug flag system (`debugLog`, `DEBUG_FLAGS`) |
| `js/utils/widgets.js` | Widget update helpers (`updateWidgetFromText`, `scheduleNodeRefresh`, `setWidgetVisible`) |
| `js/utils/feedback.js` | Toast notification helpers |

JavaScript tests live in `js-tests/` (not `js/tests/`). `package.json` and `node_modules` are at the repo root, not inside `js/`.

### Testing Approach

Because `extension.py` imports ComfyUI modules unavailable in CI, `tests/conftest.py` mocks all ComfyUI dependencies (`comfy`, `comfy_api`, `folder_paths`, `nodes`, `server`, `torch`) before any test file runs.

**Always import via `import_test_module()`** — never use direct `from prompt_models import ...` in test files:

```python
from conftest import import_test_module

prompt_models = import_test_module("prompt_models.py")
PromptCollection = prompt_models.PromptCollection
```

Keep node classes as thin orchestration wrappers. Put testable logic in `utils/` modules.

### Data Models

- `PromptCollection` (v2): named prompts with metadata (`PromptMetadata`), compositions (ordered prompt key lists), and optional `scene_flags`. Auto-migrates from v1 JSON.
- `SceneInStory`: scene slot in a story, carries `mask_name`, `prompt_source`, `video_prompt_source`, and deprecates `mask_type`/`prompt_type` (v1 fields kept for migration).
- Scene data stored as JSON on disk; migrated lazily on load.

### Custom Types

Node wiring uses custom type strings for type safety:
- `SUBJECT_LAYER` — between `SubjectLayerDefine` → `SubjectCompositor`
- `LORA_ENTRY` — between `LoraEntryDefine` → `LoraStackCollect` / `LoraStackBuilder` (legacy autogrow path)
- `LORA_STACK_DATA` — between `LoraStackBuilder` / `LoraStackCollect` → `LoraStackApply`
- `LORA_PRESET_LIST` — between `LoraPresetDefine` → `LoraPresetSelect` (carries `{ name, lora_stack, prompt }` dicts)
- `PRESET_LIST` — between `WanPresetDefine` → `WanPresetSelect` (carries `{ name, lora_h, lora_l, prompt }` dicts)
- `CONCEPT_REGISTRY` — between `ConceptRegistryLoad` / `ConceptDefine` → `ConceptResolve` / `ConceptList` (carries `ConceptRegistry` instance)
- `SUBJECT_PROFILE` — between `SubjectProfileLoad` / `SubjectProfileDefine` → `SceneCompose` (carries subject dict with name, appearance, voice, character_sheet_images, concept_id)
- `SCENE_TEMPLATE` — between `SceneTemplateLoad` → `SceneCompose` (carries `SceneTemplate` instance with slots, shots, environment, style)
- `SCENE_INSTANCE` — between `SceneCompose` → `PromptAssemble` (carries composed scene dict: template, slot_assignments, dialogue, outfit_overrides)

### Persistence

All package-level data is stored under `user_data_dir()` → `ComfyUI/user/default/comfyui-fbTools/`:
- `concept_registry.json` — concept definitions (with `.bak` auto-backup on save)
- `subject_profiles.json` — subject profile definitions (with `.bak` auto-backup on save)
- `scene_templates/` — user scene template JSON files (seeded from bundled `scene_templates/` on first use)
- `scenes/` — scene directories (new installs); legacy `output/scenes/` is still supported if the new dir is empty
- `libbers/` — libber template JSON files (new installs); legacy `output/libbers/` is still supported

The helper `user_data_dir()` in `extension.py` uses `folder_paths.get_user_directory()` with a fallback chain.

### Concept Registry

The concept system is defined in `utils/concept_registry.py` (no ComfyUI deps) and exposed via four nodes:

| Node | Role |
|---|---|
| `ConceptRegistryLoad` | Load `concept_registry.json` from disk; expose available concepts |
| `ConceptDefine` | Add/update a concept entry for one model type (chainable); auto_save option |
| `ConceptResolve` | Resolve concept IDs → apply LoRAs to model/clip; assemble prompt with trigger words |
| `ConceptList` | Display formatted concept list, optionally filtered by model type |

**Model types** (in `MODEL_PROFILES` in `utils/concept_registry.py`):

| ID | Display | Split model? |
|---|---|---|
| `wan22` | Wan 2.2 | Yes (high + low) |
| `bernini` | BerniniR | Yes (high + low) |
| `ltx23` | LTX 2.3 | No |
| `flux2` | Flux 2 | No |
| `krea2` | Krea 2 | No |
| `qwen` | Qwen Image | No |
| `minimax_h3` | MiniMax H3 | No |

For split models, `ConceptResolve` applies the HIGH LoRA to the primary `model` input and the LOW LoRA to the optional `model_low` input. Both apply to `clip`. For single-model types, only `model` is used.

Same `concept_id` + different `model_type` → entries accumulate (one per model type). Same `concept_id` + same `model_type` → the entry is overwritten. A `.bak` backup is created on every save.

### Subject Profiles (Scene Composition Engine — Phase 1)

The subject profile system is defined in `utils/subject_profiles.py` (no ComfyUI deps) and exposed via three nodes. It is the first layer of the Scene Composition Engine (`docs/scene_composition_action_plan.md`).

| Node | Role |
|---|---|
| `SubjectProfileLoad` | Load a subject from `subject_profiles.json`; outputs IMAGE batch + AUDIO |
| `SubjectProfileDefine` | Create/update a subject profile entry; auto_save option |
| `SubjectProfileList` | Display all defined subjects as formatted text |

**Storage**: `user_data_dir() + "/subject_profiles.json"` with `.bak` backup on save.

**Character sheet images** are loaded from the ComfyUI input directory by filename. Stacked into a single IMAGE batch (N × H × W × 3). Images with different sizes are resized to match the first.

**Audio reference** loaded from the ComfyUI input directory via `torchaudio`. Returns None if file is absent or `torchaudio` unavailable.

**Reload mechanism**: `POST /fbtools/subjects/reload` increments `_subject_reload_counter`, causing `SubjectProfileLoad` and `SubjectProfileList` nodes to re-execute via `fingerprint_inputs`. The `subject_id` combo on `SubjectProfileLoad` is populated at schema load time — a page refresh is needed to see newly-added subject IDs in the dropdown.

**REST endpoints**:
- `POST /fbtools/subjects/reload` — force reload counter increment
- `GET /fbtools/subjects/profiles` — return full subject_profiles.json as JSON

### Scene Templates (Scene Composition Engine — Phase 2)

The scene template system is defined in `utils/scene_templates.py` (no ComfyUI deps) and exposed via two nodes. It is the second layer of the Scene Composition Engine.

| Node | Role |
|---|---|
| `SceneTemplateLoad` | Load a template from `scene_templates/`; outputs SCENE_TEMPLATE + slot_info STRING |
| `SceneTemplateList` | Scan the templates directory and list all available templates |

**Storage**: `user_data_dir() + "/scene_templates/"` — one JSON file per template. Seeded with 3 bundled examples (`monologue_indoor`, `cafe_conversation_2p`, `meeting_room_3p`) on first use.

**Bundled examples** ship in the package's own `scene_templates/` directory and are copied once into the user data dir when that directory is empty.

**Template schema fields**: `id`, `name`, `description`, `slots` (dict of slot_id → `{role, needs_voice, needs_character_sheet}`), `environment` (`{summary, lighting}`), `style`, `shots` (list of `{id, timestamp, camera, action, dialogue, sound_events}`), `overall_soundscape`, `non_diegetic_music`.

**Placeholder convention**: `{A}`, `{B}`, `{C}` in `action` and `camera` fields are replaced at assembly time with subject appearance descriptions.

**Reload mechanism**: `POST /fbtools/scene_templates/reload` increments `_scene_template_reload_counter`. The `template_id` combo on `SceneTemplateLoad` is populated at schema load time — a page refresh is needed to see newly-added templates.

**REST endpoints**:
- `POST /fbtools/scene_templates/reload` — force reload counter increment
- `GET /fbtools/scene_templates/list` — return list of template metadata as JSON

### Scene Composition (Scene Composition Engine — Phase 3)

The composition layer is defined in `utils/scene_compose.py` (no ComfyUI deps) and exposed via one node.

| Node | Role |
|---|---|
| `SceneCompose` | Assign subjects to template slots, fill dialogue, apply outfit overrides; outputs SCENE_INSTANCE |

**Inputs**: `template` (SCENE_TEMPLATE), `slot_A`–`slot_D` (SUBJECT_PROFILE, optional B–D), `dialogue_1`–`dialogue_4` (STRING, positional — fills placeholder shots in shot order), `outfit_override_A`–`outfit_override_D` (STRING, optional).

**Dialogue mapping**: positional — `dialogue_1` fills the first shot with `placeholder: true` dialogue, `dialogue_2` fills the second, etc. Order follows shot order in the template.

**Validation**: warns (via `scene_summary` output and status update) if required slots are unfilled or if voice/character-sheet requirements aren't met. Does not hard-fail — `scene_instance` is still returned so the graph can be inspected.

**`subject_id` injection**: `SubjectProfileLoad` and `SubjectProfileDefine` inject a `subject_id` key into the subject dict they output, so `SceneCompose` and downstream nodes can reference which profile was loaded without needing a separate STRING output.

**SCENE_INSTANCE dict schema**:
```python
{
    "template_id": str,
    "template_name": str,
    "template": {full template dict},
    "slot_assignments": {"A": subject_dict, "B": subject_dict, ...},
    "dialogue": {"shot_1": "line text", "shot_2": "line text", ...},
    "outfit_overrides": {"A": "override text", ...},
}
```

### Prompt Assembly (Scene Composition Engine — Phase 4)

The prompt assembly layer is defined in `utils/prompt_assembler.py` (no ComfyUI deps) and exposed via one node.

| Node | Role |
|---|---|
| `PromptAssemble` | Takes a SCENE_INSTANCE and generates the model-specific prompt, reference image batch, audio outputs, concept IDs, and assembly report |

**Inputs**: `scene_instance` (SCENE_INSTANCE), `model_type` (COMBO), `concept_registry` (CONCEPT_REGISTRY, optional — accepted for future trigger word injection but not yet used).

**Outputs**:
- `prompt` — fully assembled prompt string in the format required by `model_type`
- `reference_images` — IMAGE batch of all character sheets (slot order: slot A's sheets first, then B, …), or None if no sheets
- `reference_audio` — AUDIO dict for first subject's voice reference (slot A), or None
- `additional_audio` — AUDIO dict for second subject's voice reference (slot B), or None
- `concept_ids` — comma-separated concept IDs from all assigned subjects (wire into ConceptResolve)
- `assembly_report` — human-readable summary of what was assembled

**Model types** (prompt formats):

| ID | Format |
|---|---|
| `h3_ref2va` | MiniMax H3 6-section structured brief: `subject_definitions`, `summary`, `retention_analysis`, `detailed_description`, `overall_soundscape`, `non_diegetic_music` |
| `h3_fl2va` | MiniMax H3 free-language: shots with `[Shot N]` headers and `<d>[lang] text</d>` dialogue tags, no reference labels |
| `wan22` | Wan 2.2 production-direction block |
| `bernini` | BerniniR production-direction block (same format as wan22) |
| `ltx23` | LTX 2.3 simple descriptive |
| `flux2` | Flux 2 simple descriptive |
| `krea2` | Krea 2 simple descriptive |
| `qwen` | Qwen Image simple descriptive |

**H3 Ref2VA reference numbering** (independent per type, assigned in slot order):
- `<Subject N>` — one per assigned slot (A=S1/Subject 1, B=S2/Subject 2, …)
- `<Picture N>` — continuous global numbering across all subjects (slot A's sheets first)
- `<Audio N>` — continuous numbering for slots that have audio files

**Placeholder replacement**: `{A}`, `{B}`, `{C}`, `{D}` in template `action` and `camera` fields are replaced with `<Subject N> (Name — appearance_summary)` on first appearance, and `<Subject N> (Name)` on subsequent appearances for H3 formats; with plain subject names for other formats.

**Integration with ConceptResolve**:
```
[PromptAssemble] → concept_ids → [ConceptResolve] applies LoRAs without modifying prompt
[PromptAssemble] → prompt ─────→ [text conditioning node]
[PromptAssemble] → reference_images → [model conditioning]
```

### Optional Dependencies

Gracefully absent:
- `rembg` — background removal in SubjectLayerDefine
- `ComfyUI-SCAIL-Pose` / `taichi` — NLF 3D pose in SceneUpdate
- `ComfyUI-WanVideoWrapper` — WANVIDLORA output in LoraStackApply
- `transformers`, `bitsandbytes`, `google-generativeai` — captioning backends
- `torchaudio` — audio reference loading in SubjectProfileLoad (audio output returns None if absent)

All optional imports are guarded with `try/except` and degrade gracefully.

## Scripts

Developer utilities in `scripts/`:

| Script | Usage |
|---|---|
| `scripts/migrate_lora_stack.py` | Migrate `loras.json` → `lora_stack.json`; clean deprecated `audio_enabled` boolean in LTX2.3 entries. Args: `[SCENES_DIR] [--force] [--dry-run]` |
| `scripts/migrate_masks.py` | Convert legacy `mask_type` scenes to `mask_name` format |
| `scripts/inspect_safetensors.py` | Print metadata and tensor names from a `.safetensors` file |
| `scripts/inspect_gguf.py` | Print metadata and tensor keys from a `.gguf` file |
| `scripts/dataset_caption_edit.fish` | Batch caption find/replace via the `/fbtools/dataset_caption/edit` API |

## Key Conventions

- Node IDs: `fbt_<DisplayName>` (e.g., `fbt_SceneSelect`)
- `send_status_update()` sends real-time feedback to the frontend via websocket event `fbtools.status`
- Masks use `mask_name` (arbitrary string) + `mask_background` (bool). The old `mask_type` field is deprecated; use `scripts/migrate_masks.py` to convert legacy data.
- Libber substitution uses `%key%` delimiters by default with recursive resolution and depth limiting.
