<div align="center">
  <img src="logo.png" alt="fbTools Logo" width="200"/>
</div>

# fbTools (frost-byte Tools)

A comprehensive collection of custom nodes for ComfyUI focused on storytelling, scene management, and prompt templating workflows.

## Features

### 🎬 Story & Scene Management
- **Story Nodes**: Create, edit, and manage multi-scene stories with ordered sequences
- **Scene Nodes**: Build scenes with poses, masks, depth maps, and multiple prompt types
- **Scene Selection**: Dynamic scene loading with automatic resource management

### 📝 Prompt System
- **PromptCollection**: Flexible v2 prompt system with metadata (categories, descriptions, tags)
- **Backward Compatibility**: Auto-migration from v1 format with preservation of legacy data
- **Multiple Prompt Types**: Support for character, scene, quality, and custom prompts

### 📚 Libber (Template System)
- **Interactive Table Editor**: Edit key-value templates with inline editing
- **Click-to-Insert**: Click any template key to insert it with delimiters
- **Auto-Save**: Changes automatically saved after modifications
- **Smart Discovery**: Automatically finds and loads libbers from disk
- **Cursor Tracking**: Maintains cursor position across focus changes with native undo/redo

### 🎭 Pose & Depth Processing
- **Multiple Pose Formats**: DWPose, OpenPose, DensePose, and face detection
- **NLF 3D Pose**: Neural Lifting Framework for advanced 3D pose estimation (optional, requires ComfyUI-SCAIL-Pose)
- **Depth Estimation**: Depth Anything v2, MiDaS, Zoe, and more
- **Mask Generation**: Character segmentation with background control

### 🖼️ Image Processing
- **TailEnhancePro**: Advanced frame enhancement with deflicker, color matching, and sharpening
- **Aspect Ratio**: Qwen-specific aspect ratio calculation and layout detection
- **SAM Preprocessing**: Prepare images for Segment Anything Model

### 🖼️ Image Compositing
- **SubjectLayerDefine**: Define a single subject layer — image, optional mask, fractional padding, canvas offset, and background removal model
- **SubjectCompositor**: Composite 1–20 subject layers onto a canvas; output a merged composite, individual per-subject images, or both

### 🎛️ LoRA Scene Management
- **LoraEntryDefine**: Define one LoRA for a specific model target with per-LoRA audio guard (LTX2.3), enable/disable toggle, and separate model/clip strengths
- **LoraStackCollect**: Collect 1–20 LoRA entries into a JSON string for scene persistence; optionally merges with existing stack JSON
- **LoraStackApply**: Apply the right LoRAs at inference time by model target — patching MODEL+CLIP directly (LTX2.3, Flux, Qwen, Z-Image), building a LORA_STACK (Wan2.2-Native), or building a WANVIDLORA (Wan2.2-Wrapper)

### Dataset Captioning
- **Dataset Captioner**: Run a VLM over a directory, write one `.txt` per image
- **Dataset Caption Editor**: Batch edit captions: prepend trigger word, find/replace
- **Dataset Caption Viewer**: Interactive table — view, edit and re-caption images in-graph
- **Dataset Export Summary**: Dataset health check: counts, word stats, missing captions, CSV export
- **Caption Model Unloader**: Release captioner from VRAM before running generation

## Quickstart

1. Install [ComfyUI](https://docs.comfy.org/get_started)
2. Install [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)
3. Install required dependencies (see below)
4. Look up "fb-tools" or "comfyui-fbTools" in ComfyUI-Manager, or manually clone:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/frost-byte/comfyui-fbTools.git
   ```
5. Restart ComfyUI

## Dependencies

### Required
- **[comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux)** - Required for pose detection, depth estimation, and canny edge detection
  - Provides: DWPose, OpenPose, DensePose, face detection
  - Provides: Depth Anything v2, MiDaS, Zoe depth estimation
  - Provides: Canny edge detection
  - Used by: All Scene nodes (SceneCreate, SceneUpdate, etc.)

### Optional
- **[ComfyUI-SCAIL-Pose](https://github.com/kijai/ComfyUI-SCAIL-Pose)** - Required for NLF (Neural Lifting Framework) 3D pose generation
  - Provides: Advanced 3D pose estimation and rendering with torch/taichi backends
  - Used by: SceneUpdate node (update_nlf_pose parameter)
  - Without this: Basic DWPose functionality still works, NLF features gracefully disabled
  - Optional dependency: `taichi` for faster GPU-accelerated rendering

- **[ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper/)** - Required only for LoRA functionality in Scene nodes
  - Provides: WANVIDLORA type for high/low quality LoRA configurations
  - Used by: SceneWanVideoLoraMultiSave node

### Optional LoRA Apply Note

The `LoraStackApply` node requires `comfy.sd.load_lora_for_models` for direct-apply targets (LTX2.3, Flux2/Klein, Qwen, Z-Image) — available in the standard ComfyUI install.
For Wan2.2-Native targets, connect the `lora_stack` output to **easy-use**'s `loraStack` node.
For Wan2.2-Wrapper targets, connect the `wanvid_lora` output to **[ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper/)**.

### Optional Compositing Backend

Install only if you use the Subject Compositor / SubjectLayerDefine nodes and want automatic background removal:

```bash
# BiRefNet via rembg (recommended — GPU-accelerated)
pip install "rembg[gpu]"

# CPU-only fallback
pip install rembg
```

Models (~100–400 MB each) are downloaded on first use and cached in `~/.u2net/` or the rembg cache directory.
If you prefer to supply your own mask, leave `remove_background` off and connect any upstream MASK output to SubjectLayerDefine.

### Optional Captioning Backends

Install one or more of these only if you use Dataset Captioning nodes:

```bash
# Qwen2.5-VL (recommended; image-focused, ~16GB VRAM in bf16)
pip install "transformers>=4.50.0" accelerate qwen-vl-utils

# Qwen2.5-Omni (heavier omni model, ~20GB VRAM)
pip install "transformers>=4.50.0" accelerate qwen-omni-utils

# Gemini Flash (cloud, no local VRAM needed)
pip install google-generativeai
export GEMINI_API_KEY=your_key_here

# Optional: 8-bit quantization (~50% VRAM reduction)
pip install bitsandbytes
```

**Installation via ComfyUI-Manager:**
1. Open ComfyUI-Manager
2. Search for "controlnet aux" and install
3. (Optional) Search for "SCAIL-Pose" or "ComfyUI-SCAIL-Pose" if using NLF pose features
4. (Optional) Search for "WanVideoWrapper" if using LoRA features
5. Restart ComfyUI

## Node Categories

All nodes are organized under the **🧊 frost-byte** category in ComfyUI.

### Story Nodes (`🧊 frost-byte/Story`)
- **StoryCreate**: Create a new story with an initial scene
- **StoryEdit**: Add, remove, reorder, or modify scenes in a story
- **StoryView**: Preview and select scenes with automatic resource loading
- **StorySave**: Persist story configuration to JSON
- **StoryLoad**: Load story from JSON file
- **StorySceneBatch**: Create ordered list of scene descriptors for iteration
- **StoryScenePick**: Select and load a specific scene by index

### Scene Nodes (`🧊 frost-byte/Scene`)
- **SceneCreate**: Create a new scene with all processing options
- **SceneUpdate**: Modify existing scene properties
- **SceneView**: View and preview scene data with images
- **SceneSelect**: Dynamic scene loading with widget updates
- **SceneSave**: Save scene data to disk
- **SceneInput**: Load scene from directory
- **SceneOutput**: Output scene images and data
- **SceneWanVideoLoraMultiSave**: Save video LoRA configurations

### Libber Nodes (`🧊 frost-byte/Libber`)
- **LibberManager**: Interactive table editor for creating and managing templates
  - Inline editing with action buttons (Add, Update, Remove)
  - Sticky controls (Load, Save, Create)
  - Auto-save after modifications
  - Smart auto-loading from memory or disk
- **LibberApply**: Apply template substitutions to text
  - Click-to-insert functionality with delimiter wrapping
  - Always-visible Refresh button
  - Dynamic table display with cursor tracking
  - Native undo/redo support

### Compositing Nodes (`🧊 frost-byte/compositing`)
- **SubjectLayerDefine**: Define one subject layer per image. Specify fractional padding, canvas offset, and optional background removal. Outputs a `SUBJECT_LAYER` token consumed by SubjectCompositor.
- **SubjectCompositor**: Composite 1–20 `SUBJECT_LAYER` inputs onto a canvas. Outputs a merged `composite` image, an `individual_images` batch (one per layer), a `layer_count` int, and the snapped canvas dimensions.

### LoRA Scene Nodes (`🧊 frost-byte/lora`)
- **LoraEntryDefine**: Define one LoRA for a specific model target. Supports `enabled` toggle, audio weight guard (LTX2.3), and separate model/clip strengths.
- **LoraStackCollect**: Collect up to 20 `LORA_ENTRY` inputs into a JSON string for scene persistence. Autogrow inputs. Optionally merges with existing JSON.
- **LoraStackApply**: Apply a persisted LoRA stack at inference time. Filters by `model_target`; routes to direct model patching, LORA_STACK (easy-use), or WANVIDLORA (WanVideoWrapper) output as appropriate.

### Image Processing Nodes
- **TailEnhancePro**: Frame enhancement with deflicker, color matching, and sharpening
- **TailSplit**: Split image batches into main and tail sections
- **OpaqueAlpha**: Create opaque alpha masks for images
- **SAMPreprocessNHWC**: Prepare images for SAM predictor
- **QwenAspectRatio**: Calculate aspect ratios for Qwen models

### Utility Nodes
- **SubdirLister**: List subdirectories with full paths
- **NodeInputSelect**: Select and output node input metadata

### Dataset Captioning Nodes
- **Dataset Captioner**: Run caption generation over a folder of images and write one `.txt` per image
- **Dataset Caption Editor**: Batch edit caption files with prepend/append/find/replace operations
- **Dataset Caption Viewer**: Review images and captions in a table UI with per-image re-caption/clear actions
- **Dataset Export Summary**: Report dataset health and optionally export `dataset_summary.csv`
- **Caption Model Unloader**: Explicitly unload cached caption models from VRAM

## Documentation

### 📖 Core Documentation

#### Node Systems
- **[Libber Nodes](docs/LIBBER_NODES_README.md)**: Template system for reusable text snippets
- **[Story Nodes](docs/STORY_NODES_README.md)**: Multi-scene story building system
- **[Scene Nodes](docs/SCENE_NODES_README.md)**: Scene management with poses, depth, and masks
- **[Dataset Caption Nodes](docs/DATASET_CAPTION_NODES.md)**: Dataset captioning workflow, node parameters, API routes, and troubleshooting
- **[Scene Prompt System](docs/SCENE_PROMPT_SYSTEM.md)**: Scene prompt architecture and usage
- **[Story Video](docs/STORY_VIDEO_README.md)**: Video generation from stories
- **Subject Compositor** (inline below): Multi-subject image compositing with SubjectLayerDefine + SubjectCompositor
- **LoRA Scene Nodes** (inline below): Per-target LoRA persistence and apply with LoraEntryDefine + LoraStackCollect + LoraStackApply

#### Mask System (NEW!)
- **[Mask System Guide](docs/MASK_SYSTEM.md)**: Generic mask system with arbitrary mask names
  - Custom mask definitions (not limited to "girl", "male", "combined")
  - Mask types: transparent and color-keyed
  - Background variant support
  - Migration guide from legacy system

#### Prompt Management
- **[Scene Prompt Usage](docs/SCENE_PROMPT_USAGE.md)**: How to use scene prompts
- **[Scene Prompt Manager Tabs](docs/SCENE_PROMPT_MANAGER_TABS.md)**: UI tabs reference

#### UI Documentation
- **[Video Prompt UI Layout](docs/VIDEO_PROMPT_UI_LAYOUT.md)**: Video prompt interface design
- **[Video Prompt UX Implementation](docs/VIDEO_PROMPT_UX_IMPLEMENTATION.md)**: Video prompt user experience

### 🔧 Development & Debugging
- **[Debugging Guide](docs/DEBUGGING.md)**: Runtime debug flag system and troubleshooting
- **[Development Notes](docs/DEVELOPMENT_NOTES.md)**: Developer notes and implementation details
- **[Implementation Steps](docs/IMPLEMENTATION_STEPS_1_2.md)**: Feature implementation history

### 🧪 Testing Documentation
All testing documentation is in [docs/testing/](docs/testing/):
- **[Testing Strategy](docs/testing/TESTING_STRATEGY.md)**: Overall testing approach
- **[Testing Guide](docs/testing/TESTING_GUIDE.md)**: How to run and write tests
- **[Test Results](docs/testing/TEST_RESULTS.md)**: Test coverage and results
- **[Test Summary](docs/testing/TEST_SUMMARY.md)**: Testing overview
- **[Test Coverage Summary](docs/testing/TEST_COVERAGE_SUMMARY.md)**: Coverage metrics
- **[Story Edit Testing](docs/testing/STORY_EDIT_TESTING_GUIDE.md)**: Story editor testing procedures
- **[Scene Tabs Testing](docs/testing/TESTING_SCENE_TABS.md)**: Scene UI testing procedures

### 💻 Frontend Architecture
- **[JavaScript Overview](js/README.md)**: Frontend modular architecture
- **[Integration Guide](js/INTEGRATION_GUIDE.md)**: How to use API clients
- **[Modular Architecture](js/MODULAR_ARCHITECTURE.md)**: Architecture decisions
- **[Quick Reference](js/QUICK_REFERENCE.md)**: API quick reference

## Key Features Explained

### Libber Template System

The Libber system provides a powerful template/substitution mechanism for reusable text snippets:

**Example:**
```python
# Define templates
libs = {
    "chunky": "incredibly thick, and %yummy%",
    "yummy": "delicious",
    "character": "A %chunky% warrior"
}

# Apply substitutions
"Look at this %character%!"
# Result: "Look at this A incredibly thick, and delicious warrior!"
```

**Features:**
- Recursive substitution with depth limiting
- Custom delimiters (default: `%`)
- Interactive table editor in LibberManager
- Click-to-insert in LibberApply
- File-based persistence

### Story Building Workflow

1. **Create Story**: Use StoryCreate to initialize a story with first scene
2. **Add Scenes**: Use StoryEdit to add more scenes with configurations
3. **Preview**: Use StoryView to preview and select scenes
4. **Batch Process**: Use StorySceneBatch + StoryScenePick for iterative generation
5. **Save**: Use StorySave to persist story configuration

Each scene can have:
- Custom mask type and background settings
- Specific prompt type (character, quality, custom)
- Depth map selection
- Pose image selection

### PromptCollection V2

The new prompt system supports unlimited named prompts with metadata:

```python
collection = PromptCollection()
collection.add_prompt(
    "lighting",
    "soft diffused lighting, golden hour",
    category="scene",
    description="Lighting setup",
    tags=["lighting", "atmosphere"]
)
```

**Features:**
- Automatic v1 → v2 migration with backup
- Metadata: categories, descriptions, tags
- Backward compatible with legacy fields
- REST API for JavaScript integration

### Subject Compositor

Compose two or more subjects onto a single canvas for use with image-conditioning workflows (LTX-Video, Qwen Edit, ReferenceLatent, etc.).

```
[SubjectLayerDefine]  [SubjectLayerDefine]  ...up to 20
        |                    |
        └─────── layer_0 ────┘
                    ↓
           [SubjectCompositor]
                    ↓
     composite / individual_images / layer_count
```

#### SubjectLayerDefine inputs

| Input | Type | Default | Notes |
|---|---|---|---|
| `image` | IMAGE | — | Input image |
| `mask` | MASK | optional | Pre-computed alpha mask. Overrides bg removal if connected. |
| `pad_top/bottom/left/right` | FLOAT | 0.0 | Padding as fraction of longer dimension. 0.2 = 20% padding. |
| `offset_x` | FLOAT | 0.0 | Horizontal offset. 0=center, 1.0=right edge, −1.0=left edge. |
| `offset_y` | FLOAT | 0.0 | Vertical offset. 0=center, 1.0=bottom, −1.0=top. |
| `remove_background` | BOOLEAN | True | Auto background removal via rembg/BiRefNet. |
| `bg_model` | COMBO | BiRefNet-general | Background removal model. |

Output: `SUBJECT_LAYER` (custom type passed to SubjectCompositor)

#### SubjectCompositor inputs

| Input | Type | Default | Notes |
|---|---|---|---|
| `canvas_width` | INT | 1344 | Target width. Snapped to `divisible_by`. |
| `canvas_height` | INT | 768 | Target height. Snapped to `divisible_by`. |
| `canvas_color` | STRING | #222222 | Background color. Accepts hex, named colors, `"transparent"`. |
| `output_mode` | COMBO | both | `composite` / `individual` / `both` |
| `divisible_by` | INT | 32 | Snap dimensions to this multiple. 32 for LTX/video models. |
| `layer_0..N` | SUBJECT_LAYER | — | Autogrow inputs. Connect SubjectLayerDefine outputs. |

Outputs: `composite` (IMAGE), `individual_images` (IMAGE batch `[N, H, W, 3]`), `layer_count` (INT)

#### Padding semantics

Padding is specified as a fraction of the image's **longer dimension**:

```
pad = 0.2,  image = 800×600
longer = 800
pad_pixels = 0.2 × 800 = 160 px added on that side
```

This makes the subject appear smaller relative to the canvas and to other
layers — the primary use case for selectively scaling subjects down.

#### Offset coordinate system

```
(−1, −1) ── (0, −1) ── (1, −1)
    |            |            |
(−1,  0) ── (0,  0) ── (1,  0)   ← center of canvas
    |            |            |
(−1,  1) ── (0,  1) ── (1,  1)
```

The subject's center is placed at the computed canvas position.
Values beyond ±1.0 are allowed and will partially clip the subject at the edge.

#### Output mode guide

| Mode | Use when… |
|---|---|
| `composite` | Feeding a single merged image to `ReferenceLatent` or `TextEncoderQwenImageEditPlus` |
| `individual` | Feeding separate per-subject images to multiple `ReferenceLatent` nodes or a multi-reference conditioning node |
| `both` | You want flexibility without re-running the compositor |

#### Background removal models

| Model | Best for |
|---|---|
| BiRefNet-general | General subjects, objects, scenery |
| BiRefNet-portrait | Human faces and portraits |
| BiRefNet-general-lite | Faster, slightly lower quality |
| u2net | General — good fallback |
| u2net_human_seg | Human silhouettes |
| isnet-general-use | High-detail foreground extraction |

#### Using an external mask instead of auto removal

If your workflow already has a bg-removal node (e.g. RMBG, BiRefNet from ComfyUI-BRIA),
connect its MASK output to the `mask` input of SubjectLayerDefine.
The `remove_background` flag is ignored when a mask is connected.

---

### LoRA Scene Nodes

Persist and apply LoRA settings per model target. Replaces the abandoned `LTX2MasterLoaderLD` and consolidates LoRA management across LTX2.3, Wan2.2 (Native and Wrapper), Flux2/Klein, Qwen, and Z-Image.

```
# Scene definition (save once)
LoraEntryDefine (LTX2.3, my_character.safetensors, audio_enabled=True)
      ↓ lora_entry
LoraEntryDefine (Wan2.2-Native, my_character_wan.safetensors)
      ↓ lora_entry
LoraStackCollect
      ↓ stack_json  →  [Scene Node / String storage]
      ↓ lora_stack_data

# Inference (LTX2.3 pipeline)
[Scene Node] → stack_json
                    ↓
LoraStackApply (model_target=LTX2.3)
  model ← [your model]
  clip  ← [your clip]
      ↓ model  →  [LTX2.3 sampler]
      ↓ clip   →  [text encoder]
```

#### LoraEntryDefine inputs

| Input | Default | Notes |
|---|---|---|
| `lora` | None | File picker from loras folder |
| `model_target` | LTX2.3 | Which pipeline this LoRA applies to |
| `strength_model` | 1.0 | UNet/transformer weight strength |
| `strength_clip` | 1.0 | Text encoder strength (ignored where N/A) |
| `enabled` | True | Toggle off without removing from stack |
| `audio_enabled` | True | LTX2.3 only: include audio weights |

Output: `LORA_ENTRY` (custom type)

#### LoraStackCollect inputs

| Input | Notes |
|---|---|
| `entry_0..N` | Autogrow LORA_ENTRY inputs (up to 20) |
| `existing_json` | Optional — merge with existing scene JSON |

Outputs: `lora_stack_data` (LORA_STACK_DATA), `stack_json` (STRING), `entry_count` (INT)

#### LoraStackApply inputs

| Input | Notes |
|---|---|
| `model_target` | Must match the target you set in LoraEntryDefine |
| `lora_stack_data` | Connect from LoraStackCollect (preferred) |
| `stack_json` | OR a JSON STRING from a scene node |
| `model` | MODEL to patch (optional) |
| `clip` | CLIP to patch (optional) |
| `prev_lora_stack` | Wan2.2-Native: chain from existing LORA_STACK |
| `prev_wanvid_lora` | Wan2.2-Wrapper: chain from existing WANVIDLORA |
| `low_mem_load` | Wan2.2-Wrapper infrastructure setting |
| `merge_loras` | Wan2.2-Wrapper infrastructure setting |

Outputs: `model` (MODEL), `clip` (CLIP), `lora_stack` (LORA_STACK), `wanvid_lora` (WANVIDLORA), `applied_count` (INT)

#### Output behaviour by target

| Target | model | clip | lora_stack | wanvid_lora |
|---|---|---|---|---|
| LTX2.3 | ✓ patched | ✓ patched | — | — |
| Wan2.2-Native | passthrough | passthrough | ✓ built | — |
| Wan2.2-Wrapper | passthrough | passthrough | — | ✓ built |
| Flux2/Klein | ✓ patched | ✓ patched | — | — |
| Qwen | ✓ patched | ✓ patched | — | — |
| Z-Image | ✓ patched | ✓ patched | — | — |

#### LTX2.3 audio guard

The `audio_enabled` flag controls whether audio-related weight keys are included. When `audio_enabled=False`, keys containing these strings are stripped: `audio`, `vocoder`, `speech`, `audio_stream`, `cross_modal`, `video_to_audio`, `av_ca`.

This replicates the behaviour of the abandoned `LTX2MasterLoaderLD` node, now per-LoRA with V3 API.

---

### Dataset Captioning Workflow

Use this flow when preparing LoRA training captions:

```text
[Dataset Captioner]
  |
  v
[Dataset Caption Editor]   <- optional post-processing (trigger word, find/replace)
  |
  v
[Dataset Caption Viewer]   <- review/edit/re-caption individual images
  |
  v
[Dataset Export Summary]   <- verify coverage and caption statistics
```

Then feed your dataset directory into your training configuration.

#### Captioner Inputs

`Dataset Captioner` supports:
- `input_directory`, `output_directory`, `recursive`
- `captioner_type`: `qwen_vl` (recommended), `qwen_omni`, or `gemini_flash`
- `instruction`, `trigger_word`, `clean_caption`
- `device`: `auto`, `cuda`, or `cpu`
- `use_8bit` (requires `bitsandbytes`)
- `override_existing`, `unload_after`, `gemini_api_key`

Outputs: `dataset_path`, `caption_count`, `failed_count`

`Dataset Caption Editor` runs in dry-run mode by default (`dry_run=true`) and only writes changes when disabled.

`Dataset Caption Viewer` provides thumbnail rows, caption editing, per-image re-caption, and clear-caption actions.

The current viewer table viewport is intentionally fixed-height for layout stability; the table scrolls internally.

`Dataset Export Summary` reports total/captioned/missing counts and caption length stats; set `export_csv=true` to write `dataset_summary.csv`.

#### Batch Caption Edits via Fish Script

Use `scripts/dataset_caption_edit.fish` for repeatable multi-pass find/replace edits against the `/fbtools/dataset_caption/edit` API.

```fish
# Dry-run (default)
fish scripts/dataset_caption_edit.fish --dataset rara \
  --pass 'old phrase=>new phrase' \
  --pass 'another old=>another new'

# Apply changes
fish scripts/dataset_caption_edit.fish --dataset rara --apply \
  --pass 'old phrase=>new phrase' \
  --pass 'another old=>another new'
```

Notes:
- Pass pairs are formatted as `find=>replace`.
- Script payload uses `find_text` and `replace_text` fields expected by the API.
- Use `--output <dir>` when captions are stored in a separate output directory.

#### VRAM Guidance

| Model | Precision | Approx VRAM |
|-------|-----------|-------------|
| Qwen2.5-VL-7B | bf16 | ~16 GB |
| Qwen2.5-VL-7B | 8-bit | ~8 GB |
| Qwen2.5-Omni-7B | bf16 | ~20 GB |
| Qwen2.5-Omni-7B | 8-bit | ~11 GB |
| Gemini Flash | cloud | 0 GB |

#### LoRA Captioning Tips

- Set `trigger_word` in `Dataset Captioner` instead of relying on prompt wording for consistency.
- Review captions in `Dataset Caption Viewer` to correct hallucinations before training.
- Aim for moderate caption length (roughly 60-150 words) and use `Dataset Export Summary` to validate.

## Development

### Setup

To install development dependencies and pre-commit hooks:

```bash
cd comfyui-fbTools
pip install -e .[dev]
pre-commit install
```

The `-e` flag installs in "editable" mode, so changes are immediately reflected when ComfyUI restarts.

### Project Structure

```
comfyui-fbTools/
├── extension.py              # Main Python extension with all nodes
├── prompt_models.py          # Data models (PromptMetadata, PromptCollection)
├── utils/                    # Python utilities
│   ├── io.py                # File I/O operations
│   ├── util.py              # General utilities
│   ├── pose.py              # Pose detection utilities
│   └── images.py            # Image processing utilities
├── js/                       # JavaScript frontend code
│   ├── fb_tools.js          # Main extension registration
│   ├── api/                 # REST API clients
│   │   ├── libber.js        # Libber API
│   │   ├── prompt_collection.js  # PromptCollection API
│   │   ├── scene.js         # Scene API
│   │   └── story.js         # Story API
│   ├── nodes/               # Node-specific handlers
│   │   ├── libber.js        # LibberManager & LibberApply
│   │   ├── scene.js         # SceneSelect handler
│   │   └── story.js         # StoryEdit & StoryView handlers
│   └── utils/               # Shared JavaScript utilities
├── tests/                    # Python unit tests
│   ├── test_prompt_collection.py
│   └── test_libber.py
└── js-tests/                # JavaScript unit tests
    ├── prompt_collection_api.test.js
    └── libber_api.test.js
```

### Testing

#### Python Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_libber.py -v

# With coverage report
python -m pytest tests/ --cov=. --cov-report=html
```

**Test Coverage:**
- ✅ 70+ tests across PromptCollection and Libber systems
- ✅ Unit tests for data models
- ✅ Integration tests for workflows
- ✅ Edge case and boundary testing
- ✅ File I/O operations

See [Testing Guide](docs/testing/TESTING_GUIDE.md) for detailed instructions.

#### JavaScript Tests

```bash
cd js/
npm install  # First time only
npm test

# Watch mode
npm run test:watch

# Coverage report
npm run test:coverage
```

**Test Coverage:**
- ✅ 30+ tests for API clients
- ✅ Mock utilities for testing without ComfyUI
- ✅ Integration tests for complete workflows
- ✅ Error handling scenarios

See [TEST_RESULTS.md](TEST_RESULTS.md) for detailed test coverage.

### Code Quality

The project uses:
- **ruff**: Python linting and formatting
- **pre-commit**: Automatic hooks for code quality
- **pytest**: Python testing framework
- **Jest**: JavaScript testing framework

### Architecture

#### Backend (Python)
- **Pydantic Models**: Type-safe data structures
- **REST API**: aiohttp endpoints for frontend integration
- **State Management**: Server-side session management with TTL
- **File I/O**: JSON-based persistence

#### Frontend (JavaScript)
- **Modular Structure**: Separate files for each API/node type
- **API Clients**: Centralized REST client classes
- **Error Handling**: Automatic toast notifications and logging
- **Testability**: Mock-friendly design with dependency injection

See [Frontend Documentation](js/README.md) for architecture details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run tests and linting
5. Submit a pull request

## Publishing to Registry

If you wish to share this extension:

1. Create account on https://registry.comfy.org
2. Add publisher ID to `pyproject.toml`
3. Create API key for publishing
4. Add `REGISTRY_ACCESS_TOKEN` to GitHub Secrets
5. Push to GitHub - action will auto-publish

See [ComfyUI Registry docs](https://docs.comfy.org/registry/publishing) for details.

## License

See [LICENSE](LICENSE) file.

## Support

- **Issues**: [GitHub Issues](https://github.com/frost-byte/comfyui-fbTools/issues)
- **Discord**: [ComfyUI Discord](https://discord.com/invite/comfyorg)
- **Documentation**: See README files in repository

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and release notes.

### Recent Updates

**2026-04-09: LoRA Scene Nodes**
- ✅ `LoraEntryDefine` — define one LoRA for a specific model target with per-LoRA audio guard, enable toggle, and separate model/clip strengths
- ✅ `LoraStackCollect` — collect up to 20 LORA_ENTRY inputs into a persisted JSON stack with optional merge from existing JSON
- ✅ `LoraStackApply` — apply persisted stack at inference time; routes to direct patching (LTX2.3, Flux, Qwen, Z-Image), LORA_STACK (Wan2.2-Native), or WANVIDLORA (Wan2.2-Wrapper)
- ✅ Custom `LORA_ENTRY` and `LORA_STACK_DATA` types for type-safe wiring
- ✅ Replaces abandoned `LTX2MasterLoaderLD` with V3 API per-LoRA granularity

**2026-04-08: Subject Compositor Nodes**
- ✅ `SubjectLayerDefine` — define a subject layer with fractional padding, canvas offset, and optional background removal
- ✅ `SubjectCompositor` — composite 1–20 layers onto a canvas (composite, individual, or both output modes)
- ✅ Custom `SUBJECT_LAYER` type wiring between the two nodes
- ✅ Autogrow inputs (up to 20 layers) on SubjectCompositor
- ✅ Canvas dimension snapping (`divisible_by`, default 32 for LTX/video models)
- ✅ rembg/BiRefNet integration with model selector; external mask support

**2025-01-18: Generic Mask System**
- ✅ Arbitrary mask names (not limited to girl/male/combined)
- ✅ Mask types: TRANSPARENT and COLOR with RGB support
- ✅ Dynamic mask loading via masks.json
- ✅ Migration script for legacy scenes
- ✅ Full backward compatibility
- ✅ Updated all Scene and Story nodes
- ✅ Comprehensive documentation and tests

**2024-12-19: Libber System Overhaul**
- ✅ Interactive table editor in LibberManager
- ✅ Click-to-insert in LibberApply
- ✅ Always-visible Refresh button
- ✅ Auto-save after modifications
- ✅ Smart libber discovery and loading
- ✅ Modular JavaScript architecture
- ✅ Comprehensive test coverage

**Previous Updates:**
- PromptCollection V2 with metadata support
- Story building system with scene management
- Automatic v1→v2 migration with backward compatibility
- REST API for frontend integration
- Modular code organization


