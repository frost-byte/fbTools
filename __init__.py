"""Top-level package for fbTools."""

WEB_DIRECTORY = "./js"
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

__author__ = """frost-byte"""
__email__ = "brian@frost-byte.net"
__version__ = "0.1.0"

from .extension import FBToolsExtension
async def comfy_entrypoint() -> FBToolsExtension:
    return FBToolsExtension()

