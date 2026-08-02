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

**Registered nodes** (from `FBToolsExtension.get_node_list()`): SubjectLayerDefine, SubjectCompositor, DatasetCaptioner, DatasetCaptionEditor, DatasetCaptionViewer, DatasetExportSummary, CaptionModelUnloader, FBTextEncodeQwenImageEditPlus (conditioning), SAMPreprocessNHWC, QwenAspectRatio, SubdirLister, MultiLoraLoader, SceneCreate, SceneUpdate, SceneMaskDefinition, SceneSave, SceneInput, SceneOutput, SceneView, SceneSelect, SceneWanVideoLoraMultiSave, SceneLoraStackSave, ScenePromptManager, PromptComposer, StorySceneBatch, StoryScenePick, StoryVideoBatch, StoryCreate, StoryEdit, StoryView, StorySave, StoryLoad, StorySceneImageSave, OpaqueAlpha, MaskProcessor, TailSplit, TailEnhancePro, LibberManager, LibberApply, LoraEntryDefine, LoraStackCollect, LoraStackApply, LoraPresetDefine, LoraPresetSelect, WanPresetDefine, WanPresetSelect, ConceptRegistryLoad, ConceptDefine, ConceptResolve, ConceptList. `LoraStackView` is **defined but not registered** — it will not appear in ComfyUI until added to `get_node_list()`.

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
- `LORA_ENTRY` — between `LoraEntryDefine` → `LoraStackCollect`
- `LORA_STACK_DATA` — between `LoraStackCollect` → `LoraStackApply`
- `LORA_PRESET_LIST` — between `LoraPresetDefine` → `LoraPresetSelect` (carries `{ name, lora_stack, prompt }` dicts)
- `PRESET_LIST` — between `WanPresetDefine` → `WanPresetSelect` (carries `{ name, lora_h, lora_l, prompt }` dicts)
- `CONCEPT_REGISTRY` — between `ConceptRegistryLoad` / `ConceptDefine` → `ConceptResolve` / `ConceptList` (carries `ConceptRegistry` instance)

### Persistence

All package-level data is stored under `user_data_dir()` → `ComfyUI/user/default/comfyui-fbTools/`:
- `concept_registry.json` — concept definitions (with `.bak` auto-backup on save)
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

For split models, `ConceptResolve` applies the HIGH LoRA to the primary `model` input and the LOW LoRA to the optional `model_low` input. Both apply to `clip`. For single-model types, only `model` is used.

Same `concept_id` + different `model_type` → entries accumulate (one per model type). Same `concept_id` + same `model_type` → the entry is overwritten. A `.bak` backup is created on every save.

### Optional Dependencies

Gracefully absent:
- `rembg` — background removal in SubjectLayerDefine
- `ComfyUI-SCAIL-Pose` / `taichi` — NLF 3D pose in SceneUpdate
- `ComfyUI-WanVideoWrapper` — WANVIDLORA output in LoraStackApply
- `transformers`, `bitsandbytes`, `google-generativeai` — captioning backends

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
