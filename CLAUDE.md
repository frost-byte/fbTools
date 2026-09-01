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

For the full registered node list, custom wire types, data models, persistence paths, optional dependencies, and developer scripts — invoke the `node-reference` skill. For Scene Composition Engine details (Concept Registry, Subject Profiles, Scene Templates, SceneCompose, PromptAssemble) — invoke the `scene-composition-engine` skill.

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

### Cross-layer naming contracts

Widget names are defined as string literals in `extension.py` (first argument to `io.<Type>.Input("name", ...)` or `id="name"` keyword). JavaScript reads them back at runtime via `node.widgets.find(w => w.name === "name")`. These references are invisible to the type system and break silently.

**Rule:** before renaming a widget in Python, run:

```bash
grep -r '"old_name"' js/
```

…and update every matching JS file first. The same applies when removing a widget.

**Automated check:** `tests/test_widget_name_contracts.py` extracts all widget names defined in Python and all `w.name === "x"` lookups in JS, then fails if any JS-referenced name is missing from the Python schema. Run it after any node schema change:

```bash
/mnt/comfy_ssd/venvs/comfy-preflight/bin/python -m pytest tests/test_widget_name_contracts.py -v
```

If the JS lookup is an intentional backwards-compat fallback for old saved workflows, add it to `ALLOWLIST` in the test file with a reason. Keep the list minimal.

## Key Conventions

- Node IDs: `fbt_<DisplayName>` (e.g., `fbt_SceneSelect`)
- `send_status_update()` sends real-time feedback to the frontend via websocket event `fbtools.status`
- Masks use `mask_name` (arbitrary string) + `mask_background` (bool). The old `mask_type` field is deprecated; use `scripts/migrate_masks.py` to convert legacy data.
- Libber substitution uses `%key%` delimiters by default with recursive resolution and depth limiting.
