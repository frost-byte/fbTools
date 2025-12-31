"""
Pytest configuration and fixtures for comfyui-fbTools tests.

This module handles all the ComfyUI dependency mocking and import setup
so that tests can import from extension.py without issues.

UNIFIED TEST IMPORT APPROACH:
    Use import_test_module() for all test imports to ensure consistency.
    
    Example in test files:
        from conftest import import_test_module
        
        # Import standalone modules (no ComfyUI dependencies)
        prompt_models = import_test_module("prompt_models.py")
        
        # Import utility modules
        scene_save = import_test_module("utils/scene_image_save.py")
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock
import importlib.util

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
comfyui_root = os.path.abspath(os.path.join(project_root, '../..'))

# Add paths to enable both package imports and direct imports
sys.path.insert(0, comfyui_root)  # For custom_nodes.comfyui-fbTools imports
sys.path.insert(0, project_root)  # For direct imports from project root

# Create a mock utils package so prompt_models.py's relative imports work
# when imported directly as a top-level module
if 'utils' not in sys.modules:
    utils_mock = MagicMock()
    logging_utils_mock = MagicMock()
    # Create a real logger that works for tests
    import logging
    def get_logger(name):
        return logging.getLogger(name)
    logging_utils_mock.get_logger = get_logger
    utils_mock.logging_utils = logging_utils_mock
    sys.modules['utils'] = utils_mock
    sys.modules['utils.logging_utils'] = logging_utils_mock

# Mock all ComfyUI modules BEFORE any extension imports
# Core ComfyUI modules
comfy_mock = MagicMock()
comfy_mock.model_management = MagicMock()
comfy_mock.utils = MagicMock()
comfy_mock.utils.common_upscale = MagicMock()
sys.modules['comfy'] = comfy_mock
sys.modules['comfy.model_management'] = comfy_mock.model_management
sys.modules['comfy.utils'] = comfy_mock.utils

# ComfyUI API modules
sys.modules['comfy_api'] = MagicMock()
sys.modules['comfy_api.latest'] = MagicMock()

# ComfyUI helper modules
sys.modules['folder_paths'] = MagicMock()
sys.modules['nodes'] = MagicMock()
sys.modules['node_helpers'] = MagicMock()

# Server module with PromptServer mock
server_mock = MagicMock()
prompt_server_instance = MagicMock()
prompt_server_instance.routes = MagicMock()
prompt_server_instance.routes.post = MagicMock(return_value=lambda f: f)
prompt_server_instance.routes.get = MagicMock(return_value=lambda f: f)
server_mock.PromptServer = MagicMock()
server_mock.PromptServer.instance = prompt_server_instance
sys.modules['server'] = server_mock


# ============================================================================
# UNIFIED TEST IMPORT HELPERS
# ============================================================================

def import_test_module(relative_path: str, module_name: str = None):
    """
    UNIFIED import helper for all tests - use this instead of custom import code.
    
    Import a Python module directly by file path, bypassing package imports.
    This handles all the complexity of importing modules with various dependency
    situations.
    
    Args:
        relative_path: Path relative to project root 
                      Examples: "prompt_models.py", "utils/scene_image_save.py"
        module_name: Optional name for the module (defaults to filename without .py)
    
    Returns:
        The imported module object
    
    Usage in tests:
        from conftest import import_test_module
        
        # Import standalone modules
        prompt_models = import_test_module("prompt_models.py")
        PromptCollection = prompt_models.PromptCollection
        
        # Import utility modules  
        scene_save = import_test_module("utils/scene_image_save.py")
        SceneImageSaveConfig = scene_save.SceneImageSaveConfig
    """
    module_path = Path(project_root) / relative_path
    
    if not module_path.exists():
        raise FileNotFoundError(f"Module file not found: {module_path}")
    
    # Use filename without extension as module name if not provided
    if module_name is None:
        module_name = module_path.stem
    
    # Create a unique module name to avoid conflicts
    unique_name = f"test_import_{module_name}"
    
    spec = importlib.util.spec_from_file_location(unique_name, module_path)
    module = importlib.util.module_from_spec(spec)
    
    # Add to sys.modules so relative imports within the module work
    sys.modules[unique_name] = module
    
    spec.loader.exec_module(module)
    return module


def get_project_root() -> Path:
    """Get the project root directory for tests."""
    return Path(project_root)
