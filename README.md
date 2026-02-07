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

### Image Processing Nodes
- **TailEnhancePro**: Frame enhancement with deflicker, color matching, and sharpening
- **TailSplit**: Split image batches into main and tail sections
- **OpaqueAlpha**: Create opaque alpha masks for images
- **SAMPreprocessNHWC**: Prepare images for SAM predictor
- **QwenAspectRatio**: Calculate aspect ratios for Qwen models

### Utility Nodes
- **SubdirLister**: List subdirectories with full paths
- **NodeInputSelect**: Select and output node input metadata

## Documentation

### 📖 Core Documentation

#### Node Systems
- **[Libber Nodes](docs/LIBBER_NODES_README.md)**: Template system for reusable text snippets
- **[Story Nodes](docs/STORY_NODES_README.md)**: Multi-scene story building system
- **[Scene Nodes](docs/SCENE_NODES_README.md)**: Scene management with poses, depth, and masks
- **[Scene Prompt System](docs/SCENE_PROMPT_SYSTEM.md)**: Scene prompt architecture and usage
- **[Story Video](docs/STORY_VIDEO_README.md)**: Video generation from stories

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

## Develop

To install the dev dependencies and pre-commit (will run the ruff hook), do:

```bash
cd fb_tools
pip install -e .[dev]
pre-commit install
```

The `-e` flag above will result in a "live" install, in the sense that any changes you make to your node extension will automatically be picked up the next time you run ComfyUI.

## Publish to Github

Install Github Desktop or follow these [instructions](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent) for ssh.

1. Create a Github repository that matches the directory name. 
2. Push the files to Git
```
git add .
git commit -m "project scaffolding"
git push
``` 

## Writing custom nodes

An example custom node is located in [node.py](src/fb_tools/nodes.py). To learn more, read the [docs](https://docs.comfy.org/essentials/custom_node_overview).


## Tests

This repo contains unit tests written in Pytest in the `tests/` directory and JavaScript tests in `js-tests/`. 

See [Testing Guide](docs/testing/TESTING_GUIDE.md) for detailed testing procedures and [Test Results](docs/testing/TEST_RESULTS.md) for coverage reports.

- [build-pipeline.yml](.github/workflows/build-pipeline.yml) will run pytest and linter on any open PRs
- [validate.yml](.github/workflows/validate.yml) will run [node-diff](https://github.com/Comfy-Org/node-diff) to check for breaking changes

## Publishing to Registry

If you wish to share this custom node with others in the community, you can publish it to the registry. We've already auto-populated some fields in `pyproject.toml` under `tool.comfy`, but please double-check that they are correct.

You need to make an account on https://registry.comfy.org and create an API key token.

- [ ] Go to the [registry](https://registry.comfy.org). Login and create a publisher id (everything after the `@` sign on your registry profile). 
- [ ] Add the publisher id into the pyproject.toml file.
- [ ] Create an api key on the Registry for publishing from Github. [Instructions](https://docs.comfy.org/registry/publishing#create-an-api-key-for-publishing).
- [ ] Add it to your Github Repository Secrets as `REGISTRY_ACCESS_TOKEN`.

A Github action will run on every git push. You can also run the Github action manually. Full instructions [here](https://docs.comfy.org/registry/publishing). Join our [discord](https://discord.com/invite/comfyorg) if you have any questions!

