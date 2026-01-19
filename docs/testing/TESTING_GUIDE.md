# Testing Guide for comfyui-fbTools

## Quick Start

Run all tests:
```bash
cd /home/beerye/comfyui_env/ComfyUI-0.3.77/custom_nodes/comfyui-fbTools
/home/beerye/comfyui_env/venv-0.3.77/bin/python -m pytest tests/ -v
```

Run specific test file:
```bash
/home/beerye/comfyui_env/venv-0.3.77/bin/python -m pytest tests/test_scene_image_save.py -v
```

## Unified Import Approach

**All test files MUST use the unified import approach defined in `conftest.py`.**

### ✅ Correct Way (Use This Always)

```python
"""
My test file
"""

import pytest
from unittest.mock import Mock, patch

# Use unified import helper from conftest
from conftest import import_test_module

# Import standalone modules (no ComfyUI dependencies)
prompt_models = import_test_module("prompt_models.py")
PromptCollection = prompt_models.PromptCollection
PromptMetadata = prompt_models.PromptMetadata

# Import utility modules
scene_save = import_test_module("utils/scene_image_save.py")
SceneImageSaveConfig = scene_save.SceneImageSaveConfig
ImageSaver = scene_save.ImageSaver


class TestMyFeature:
    def test_something(self):
        # Your test code here
        collection = PromptCollection()
        assert collection is not None
```

### ❌ Wrong Ways (Do NOT Use These)

#### Don't use direct imports:
```python
# ❌ WRONG - will fail with import errors
from prompt_models import PromptCollection
```

#### Don't use custom import code in each test:
```python
# ❌ WRONG - inconsistent, hard to maintain
import importlib.util
spec = importlib.util.spec_from_file_location(...)
module = importlib.util.module_from_spec(spec)
# ... lots of boilerplate
```

#### Don't copy classes into test files:
```python
# ❌ WRONG - tests outdated code, not actual code
class PromptCollection:
    """Test copy of PromptCollection class"""
    # ... duplicated code
```

## Why This Approach?

1. **Consistency**: All tests use the same import mechanism
2. **Maintainability**: Import logic is centralized in `conftest.py`
3. **Clarity**: Test files are clean and focus on testing, not imports
4. **Reliability**: Properly handles ComfyUI dependencies and path issues
5. **Virtual Environment**: Always uses the correct venv Python

## File Organization

```
tests/
├── conftest.py              # Unified import helpers (DON'T MODIFY IMPORTS)
├── test_libber.py           # Libber functionality tests
├── test_prompt_collection.py
├── test_prompt_processing.py
├── test_prompt_composition_integration.py
├── test_scene_image_save.py # Scene image saving tests
└── test_story_edit.py       # Story editing tests
```

## Writing New Tests

### Step 1: Create test file

Create `tests/test_my_feature.py`:

```python
"""
Tests for my new feature
"""

import pytest

# Import using unified approach
from conftest import import_test_module

# Load the module you need to test
my_module = import_test_module("utils/my_feature.py")
MyClass = my_module.MyClass
my_function = my_module.my_function
```

### Step 2: Write testable code

**Extract logic into utility modules:**

Create `utils/my_feature.py`:
```python
"""
Testable utilities for my feature.
Pure functions and classes with minimal dependencies.
"""

class MyClass:
    """Pure data class - easy to test"""
    def __init__(self, value: str):
        self.value = value
    
    def transform(self) -> str:
        """Pure function - easy to test"""
        return self.value.upper()
```

**Use in extension.py:**

```python
# In extension.py
from .utils.my_feature import MyClass

class MyComfyNode(io.ComfyNode):
    @classmethod
    def execute(cls, input_value):
        # Thin orchestration layer
        obj = MyClass(input_value)
        result = obj.transform()
        return io.NodeOutput(result)
```

### Step 3: Write tests

```python
class TestMyFeature:
    def test_transform_basic(self):
        obj = MyClass("hello")
        assert obj.transform() == "HELLO"
    
    def test_transform_empty(self):
        obj = MyClass("")
        assert obj.transform() == ""
```

## Important Notes

### Always Use Venv Python

When running tests, ALWAYS use the full path to the venv Python:

```bash
# ✅ CORRECT
/home/beerye/comfyui_env/venv-0.3.77/bin/python -m pytest tests/

# ❌ WRONG - may use wrong Python version
python -m pytest tests/
pytest tests/
```

The venv path can be obtained programmatically:
```bash
# Get the correct Python path
cd /home/beerye/comfyui_env/ComfyUI-0.3.77/custom_nodes/comfyui-fbTools
# Use: /home/beerye/comfyui_env/venv-0.3.77/bin/python
```

### Test File Naming

- Test files must start with `test_`
- Test classes must start with `Test`
- Test methods must start with `test_`

### Testable Code Principles

1. **Separate concerns**: Business logic separate from I/O
2. **Pure functions**: Same input → same output
3. **Data classes**: Simple classes with no side effects
4. **Inject dependencies**: Pass dependencies as parameters
5. **Thin orchestration**: Keep ComfyUI nodes as thin wrappers

## Common Patterns

### Testing Pure Functions

```python
def test_pure_function():
    # No setup needed
    result = my_function(input_data)
    assert result == expected_output
```

### Testing with Mocks

```python
from unittest.mock import Mock, patch

def test_with_mocks():
    with patch('os.makedirs') as mock_makedirs:
        ImageSaver.ensure_directory("/test/path")
        mock_makedirs.assert_called_once_with("/test/path", exist_ok=True)
```

### Testing Data Classes

```python
def test_data_class():
    config = SceneImageSaveConfig(
        scene_name="test",
        scene_order=1,
        target_dir="/tmp",
        image_format="png"
    )
    assert config.generate_filename() == "001_test.png"
```

## Troubleshooting

### Import Errors

If you see: `ModuleNotFoundError: No module named '...'`

1. Check you're using `import_test_module()` from conftest
2. Verify the file path is correct relative to project root
3. Ensure you're running from the correct directory

### Tests Pass Individually But Fail Together

This usually means:
1. Tests are modifying shared state
2. Use fixtures or setup/teardown to isolate tests

### Python Version Issues

Always check you're using the venv Python:
```bash
/home/beerye/comfyui_env/venv-0.3.77/bin/python --version
# Should show: Python 3.12.3
```

## Additional Resources

- See individual test files for examples
- Check `conftest.py` for implementation details
- Run tests with `-v` flag for verbose output
- Use `--tb=short` for shorter error messages
