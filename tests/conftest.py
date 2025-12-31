"""
Pytest configuration and fixtures for comfyui-fbTools tests.

This module handles all the ComfyUI dependency mocking and import setup
so that tests can import from extension.py without issues.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

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

# Now we need to handle the relative imports in extension.py
# Create a package context so ".utils" imports work
import importlib.util

# Load extension.py as a module within a package context
extension_path = Path(project_root) / "extension.py"
spec = importlib.util.spec_from_file_location("__main__.extension", extension_path)
if spec and spec.loader:
    # Import utils modules that extension.py needs
    # These are real modules in the project, but we need to import them in a way
    # that doesn't trigger ComfyUI dependencies
    utils_path = Path(project_root) / "utils"
    
    # For now, just make extension importable by temporarily patching the imports
    # The actual implementation: we'll import extension as a regular module
    # and Python will resolve .utils as utils/ in the same directory
    pass
