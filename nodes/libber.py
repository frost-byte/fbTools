"""Libber string-templating system: the Libber engine, its two nodes (LibberManager,
LibberApply), the server-side LibberStateManager registry, and the /fbtools/libber/* REST routes.

Moved out of extension.py (pure code motion; Plan 17 continues Plan 1/11's extension.py -> nodes/
package split). Body order here (Libber -> LibberStateManager -> the two io.ComfyNode classes ->
routes) is top-down by dependency for readability; it differs from extension.py's original
non-contiguous layout but changes no behavior, since Python resolves these names at call time,
not definition order.
"""
from __future__ import annotations

import json
import os
import re
from typing import List, Optional

from aiohttp import web
from comfy_api.latest import io

from .shared import routes, prefixed_node_id, default_libber_dir
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class Libber:
    """
    Libber: A string templating system for ComfyUI prompts.

    Allows defining reusable text snippets (libs) that can be referenced
    in other strings using a delimiter (default: %). Supports recursive
    substitution with depth limiting to prevent infinite loops.

    Example:
        libs = {
            "chunky": "incredibly thick, and %yummy%",
            "yummy": "delicious",
            "character": "A %chunky% warrior"
        }
        libber = Libber(libs)
        libber.substitute("Look at this %character%!")
        # Result: "Look at this A incredibly thick, and delicious warrior!"
    """

    def __init__(self, lib_dict=None, delimiter="%", max_depth=10):
        """
        Initialize a Libber instance.

        Args:
            lib_dict: Dictionary of lib_key -> value mappings
            delimiter: Character(s) used to mark lib references (default: "%")
            max_depth: Maximum recursion depth for nested lib substitution
        """
        self.libs = lib_dict or {}
        self.delimiter = delimiter
        self.max_depth = max_depth

    def add_lib(self, key: str, value: str):
        """Add or update a lib entry."""
        # Normalize key to lowercase with underscores
        normalized_key = key.lower().replace(" ", "_").replace("-", "_")
        self.libs[normalized_key] = value

    def remove_lib(self, key: str):
        """Remove a lib entry."""
        normalized_key = key.lower().replace(" ", "_").replace("-", "_")
        if normalized_key in self.libs:
            del self.libs[normalized_key]
            return True
        return False

    def get_lib(self, key: str) -> Optional[str]:
        """Get a lib value by key."""
        normalized_key = key.lower().replace(" ", "_").replace("-", "_")
        return self.libs.get(normalized_key)

    def list_libs(self) -> List[str]:
        """Return a list of all lib keys."""
        return sorted(self.libs.keys())

    def substitute(self, text: str, depth: int = 0) -> str:
        """
        Recursively substitute lib references in text.

        Args:
            text: Input string containing lib references like %lib_name%
            depth: Current recursion depth (used internally)

        Returns:
            String with all lib references substituted
        """
        if depth >= self.max_depth:
            return text

        # Pattern: delimiter + lowercase/underscore words + delimiter
        # e.g., %chunky%, %my_lib%, %test_123%
        pattern = re.escape(self.delimiter) + r'([a-z0-9_]+)' + re.escape(self.delimiter)

        def replacer(match):
            lib_key = match.group(1)
            if lib_key in self.libs:
                # Get the value and recursively substitute
                value = self.libs[lib_key]
                return self.substitute(value, depth + 1)
            # Return unchanged if not found
            return match.group(0)

        return re.sub(pattern, replacer, text)

    def to_dict(self) -> dict:
        """Convert Libber instance to a dictionary for serialization."""
        return {
            "libs": self.libs,
            "delimiter": self.delimiter,
            "max_depth": self.max_depth
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Libber":
        """Create a Libber instance from a dictionary."""
        return cls(
            lib_dict=data.get("libs", {}),
            delimiter=data.get("delimiter", "%"),
            max_depth=data.get("max_depth", 10)
        )

    def save(self, filepath: str):
        """Save Libber configuration to a JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> "Libber":
        """Load Libber configuration from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def __repr__(self):
        return f"Libber(libs={len(self.libs)}, delimiter='{self.delimiter}', max_depth={self.max_depth})"


class LibberStateManager:
    """
    Manages server-side Libber instances for REST API operations.
    Libbers are stored by name and persist until explicitly deleted or server restart.
    """
    _instance = None

    def __init__(self):
        self.libbers = {}  # libber_name -> Libber instance

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def create_libber(self, name: str, delimiter: str = "%", max_depth: int = 10) -> Libber:
        """Create a new Libber instance."""
        libber = Libber(lib_dict={}, delimiter=delimiter, max_depth=max_depth)
        self.libbers[name] = libber
        logger.info("LibberStateManager: Created libber '%s'", name)
        return libber

    def load_libber(self, name: str, filepath: str) -> Libber:
        """Load a Libber from file."""
        libber = Libber.load(filepath)
        self.libbers[name] = libber
        logger.info("LibberStateManager: Loaded libber '%s' from %s", name, filepath)
        return libber

    def get_libber(self, name: str) -> Optional[Libber]:
        """Get a Libber by name."""
        return self.libbers.get(name)

    def ensure_libber(self, name: str, base_dir: Optional[str] = None) -> Optional[Libber]:
        """Get a Libber if loaded; otherwise try loading from disk (base_dir/name.json)."""
        libber = self.get_libber(name)
        if libber:
            return libber
        base_dir = base_dir or default_libber_dir()
        filepath = os.path.join(base_dir, f"{name}.json")
        if os.path.exists(filepath):
            try:
                return self.load_libber(name, filepath)
            except Exception as exc:
                logger.warning(
                    "LibberStateManager: Failed to auto-load libber '%s' from %s: %s",
                    name,
                    filepath,
                    exc,
                )
        else:
            logger.warning(
                "LibberStateManager: Libber '%s' not loaded and file not found at %s",
                name,
                filepath,
            )
        return None

    def save_libber(self, name: str, filepath: str):
        """Save a Libber to file."""
        if name in self.libbers:
            self.libbers[name].save(filepath)
            logger.info("LibberStateManager: Saved libber '%s' to %s", name, filepath)
        else:
            raise ValueError(f"Libber '{name}' not found")

    def list_libbers(self) -> List[str]:
        """List all loaded libber names."""
        return list(self.libbers.keys())

    def delete_libber(self, name: str):
        """Remove a Libber from memory."""
        if name in self.libbers:
            del self.libbers[name]
            logger.info("LibberStateManager: Deleted libber '%s'", name)

    def get_libber_data(self, name: str) -> Optional[dict]:
        """Get Libber data for UI display."""
        if name in self.libbers:
            libber = self.libbers[name]
            return {
                "keys": libber.list_libs(),
                "lib_dict": libber.libs.copy(),
                "delimiter": libber.delimiter,
                "max_depth": libber.max_depth
            }
        return None


class LibberManager(io.ComfyNode):
    """Manage Libber instances - create, load, save, and edit libs with an interactive table."""

    @classmethod
    def define_schema(cls):
        libber_dir = default_libber_dir()

        # Get available libber files (basenames only, no .json extension)
        libber_names = []
        if os.path.isdir(libber_dir):
            for f in os.listdir(libber_dir):
                if f.endswith('.json'):
                    # Remove .json extension for display
                    libber_names.append(f[:-5])

        if not libber_names:
            libber_names = ["none"]

        return io.Schema(
            node_id=prefixed_node_id("LibberManager"),
            display_name="Libber Manager",
            category="🧊 frost-byte/Libber",
            inputs=[
                io.Combo.Input(
                    id="libber_name",
                    display_name="libber_name",
                    options=sorted(libber_names),
                    default=libber_names[0],
                    tooltip="Select an existing libber or create a new one"
                ),
                io.String.Input(
                    id="libber_dir",
                    display_name="libber_dir",
                    default=libber_dir,
                    tooltip="Directory for libber files"
                ),
                io.String.Input(
                    id="delimiter",
                    display_name="delimiter",
                    default="%",
                    tooltip="Delimiter for lib references"
                ),
                io.Int.Input(
                    id="max_depth",
                    display_name="max_depth",
                    default=10,
                    min=1,
                    max=100,
                    tooltip="Maximum substitution depth"
                ),
            ],
            outputs=[
                io.String.Output(id="status", display_name="status", tooltip="Operation status and info"),
                io.String.Output(id="keys_list", display_name="keys_list", tooltip="List of all lib keys"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, libber_name="my_libber",
                libber_dir="", delimiter="%", max_depth=10):

        if not libber_dir:
            libber_dir = default_libber_dir()

        # Skip if no libber selected
        if libber_name == "none":
            return io.NodeOutput("Select or create a libber to begin", "")

        manager = LibberStateManager.instance()

        try:
            # Check if libber file exists and load it to ensure we have the latest data
            libber_filepath = os.path.join(libber_dir, f"{libber_name}.json")
            if os.path.exists(libber_filepath):
                # Reload from file to get latest changes
                libber = manager.load_libber(libber_name, libber_filepath)
                status = f"✓ Reloaded libber '{libber_name}' from file"
            else:
                # Try to get existing in-memory instance or create new one
                libber = manager.get_libber(libber_name)
                if not libber:
                    # Create new libber if it doesn't exist
                    libber = manager.create_libber(libber_name, delimiter, max_depth)
                    status = f"✓ Created new libber '{libber_name}'"
                else:
                    status = f"✓ Libber '{libber_name}' ready (in-memory)"

            keys = libber.list_libs()

            # Format keys list for display
            keys_display = "\n".join(keys) if keys else "(no libs)"

            # Return UI data for dynamic updates
            keys_json = json.dumps(keys)

            # Get libber data for UI display
            libber_data = manager.get_libber_data(libber_name)
            if libber_data:
                lib_dict_json = json.dumps(libber_data["lib_dict"])
            else:
                lib_dict_json = json.dumps({})

            combined_ui = {
                "text": [keys_json, lib_dict_json, status]
            }

            logger.info("LibberManager: %s", status)
            return io.NodeOutput(status, keys_display, ui=combined_ui)

        except Exception as e:
            status = f"✗ Error: {str(e)}"
            logger.error("LibberManager error: %s", status)
            return io.NodeOutput(status, "")


class LibberApply(io.ComfyNode):
    """Apply Libber substitutions to text with libber selection."""

    @classmethod
    def define_schema(cls):
        libber_dir = default_libber_dir()
        manager = LibberStateManager.instance()
        available_libbers = set(manager.list_libbers())

        # Include libbers available on disk (same source behavior as LibberManager)
        if os.path.isdir(libber_dir):
            for f in os.listdir(libber_dir):
                if f.endswith('.json'):
                    available_libbers.add(f[:-5])

        available_libbers = sorted(available_libbers)

        if not available_libbers:
            available_libbers = ["none"]

        return io.Schema(
            node_id=prefixed_node_id("LibberApply"),
            display_name="Libber Apply",
            category="🧊 frost-byte/Libber",
            inputs=[
                io.Combo.Input(
                    id="libber_name",
                    display_name="libber_name",
                    options=available_libbers,
                    default=available_libbers[0],
                    tooltip="Select which Libber to use"
                ),
                io.String.Input(
                    id="text",
                    display_name="text",
                    default="",
                    multiline=True,
                    tooltip="Input text with lib references (e.g., 'A %chunky% character')"
                ),
            ],
            outputs=[
                io.String.Output(id="result", display_name="result", tooltip="Text with all lib references substituted"),
                io.String.Output(id="info", display_name="info", tooltip="Substitution details and available libs"),
            ],
        )

    @classmethod
    def execute(cls, libber_name="my_libber", text=""):
        manager = LibberStateManager.instance()

        if libber_name == "none":
            return io.NodeOutput(text, "Select a libber in LibberManager or create one first.")

        # Try to reload from file to ensure we have the latest data
        libber_dir = default_libber_dir()
        libber_filepath = os.path.join(libber_dir, f"{libber_name}.json")
        if os.path.exists(libber_filepath):
            try:
                libber = manager.load_libber(libber_name, libber_filepath)
            except Exception as e:
                logger.warning("LibberApply: Error reloading from file, using in-memory instance: %s", e)
                libber = manager.get_libber(libber_name)
        else:
            libber = manager.get_libber(libber_name)

        if not libber:
            status = f"✗ Libber '{libber_name}' not found. Create or load it in LibberManager first."
            logger.warning("LibberApply: %s", status)
            return io.NodeOutput(text, status)

        if not text:
            # Display available libs when no text provided
            keys = libber.list_libs()
            info_parts = [f"Libber '{libber_name}' ready ({len(keys)} libs)"]
            if keys:
                info_parts.append("\nAvailable libs:")
                for key in keys[:10]:  # Show first 10
                    value = libber.get_lib(key) or ""
                    preview = value[:40] + "..." if len(value) > 40 else value
                    info_parts.append(f"  {key}: {preview}")
                if len(keys) > 10:
                    info_parts.append(f"  ... and {len(keys) - 10} more")
            else:
                info_parts.append("(no libs defined yet)")

            info = "\n".join(info_parts)
            return io.NodeOutput("", info)

        try:
            result = libber.substitute(text)
            keys = libber.list_libs()
            info = f"✓ Substituted using libber '{libber_name}' ({len(keys)} libs, max_depth={libber.max_depth})"
            logger.info("LibberApply: %s", info)
            logger.debug("LibberApply input preview: %s", text[:100])
            logger.debug("LibberApply output preview: %s", result[:100])

            # Provide UI data showing available libs
            libber_data = manager.get_libber_data(libber_name)
            if libber_data:
                lib_dict_json = json.dumps(libber_data["lib_dict"])
                combined_ui = {"text": [lib_dict_json, info]}
                return io.NodeOutput(result, info, ui=combined_ui)

            return io.NodeOutput(result, info)

        except Exception as e:
            result = text
            info = f"✗ Error during substitution: {e}"
            logger.error("LibberApply: %s", info)
            return io.NodeOutput(result, info)


@routes.post("/fbtools/libber/create")
async def libber_create(request):
    """
    Create a new Libber instance.
    Body: {"name": str, "delimiter": str (optional), "max_depth": int (optional)}
    Returns: {"name": str, "keys": [], "status": "created"}
    """
    try:
        data = await request.json()
        name = data.get("name")
        if not name:
            return web.json_response({"error": "name required"}, status=400)

        delimiter = data.get("delimiter", "%")
        max_depth = data.get("max_depth", 10)

        manager = LibberStateManager.instance()
        libber = manager.create_libber(name, delimiter, max_depth)

        return web.json_response({
            "name": name,
            "keys": libber.list_libs(),
            "delimiter": libber.delimiter,
            "max_depth": libber.max_depth,
            "status": "created"
        })

    except Exception as e:
        logger.exception("Error creating libber")
        return web.json_response({"error": str(e)}, status=500)


@routes.post("/fbtools/libber/load")
async def libber_load(request):
    """
    Load a Libber from file.
    Body: {"name": str, "filepath": str}
    Returns: {"name": str, "keys": [...], "status": "loaded"}
    """
    try:
        data = await request.json()
        name = data.get("name")
        filepath = data.get("filepath")

        if not name or not filepath:
            return web.json_response({"error": "name and filepath required"}, status=400)

        manager = LibberStateManager.instance()
        libber = manager.load_libber(name, filepath)

        return web.json_response({
            "name": name,
            "keys": libber.list_libs(),
            "delimiter": libber.delimiter,
            "max_depth": libber.max_depth,
            "status": "loaded"
        })

    except Exception as e:
        logger.exception("Error loading libber")
        return web.json_response({"error": str(e)}, status=500)


@routes.post("/fbtools/libber/add_lib")
async def libber_add_lib(request):
    """
    Add a lib entry to a Libber.
    Body: {"name": str, "key": str, "value": str}
    Returns: {"name": str, "keys": [...], "status": "added"}
    """
    try:
        data = await request.json()
        name = data.get("name")
        key = data.get("key")
        value = data.get("value")

        if not all([name, key, value is not None]):
            return web.json_response({"error": "name, key, and value required"}, status=400)

        manager = LibberStateManager.instance()
        libber = manager.get_libber(name)

        if not libber:
            return web.json_response({"error": f"Libber '{name}' not found"}, status=404)

        libber.add_lib(key, value)

        return web.json_response({
            "name": name,
            "keys": libber.list_libs(),
            "status": "added",
            "key": key
        })

    except Exception as e:
        logger.exception("Error adding lib")
        return web.json_response({"error": str(e)}, status=500)


@routes.post("/fbtools/libber/remove_lib")
async def libber_remove_lib(request):
    """
    Remove a lib entry from a Libber.
    Body: {"name": str, "key": str}
    Returns: {"name": str, "keys": [...], "status": "removed"}
    """
    try:
        data = await request.json()
        name = data.get("name")
        key = data.get("key")

        if not name or not key:
            return web.json_response({"error": "name and key required"}, status=400)

        manager = LibberStateManager.instance()
        libber = manager.get_libber(name)

        if not libber:
            return web.json_response({"error": f"Libber '{name}' not found"}, status=404)

        libber.remove_lib(key)

        return web.json_response({
            "name": name,
            "keys": libber.list_libs(),
            "status": "removed",
            "key": key
        })

    except Exception as e:
        logger.exception("Error removing lib")
        return web.json_response({"error": str(e)}, status=500)


@routes.post("/fbtools/libber/save")
async def libber_save(request):
    """
    Save a Libber to file.
    Body: {"name": str, "filepath": str}
    Returns: {"name": str, "filepath": str, "status": "saved"}
    """
    try:
        data = await request.json()
        name = data.get("name")
        filepath = data.get("filepath")

        if not name or not filepath:
            return web.json_response({"error": "name and filepath required"}, status=400)

        manager = LibberStateManager.instance()
        manager.save_libber(name, filepath)

        return web.json_response({
            "name": name,
            "filepath": filepath,
            "status": "saved"
        })

    except Exception as e:
        logger.exception("Error saving libber")
        return web.json_response({"error": str(e)}, status=500)


@routes.get("/fbtools/libber/list")
async def libber_list(request):
    """
    List all available libbers in memory and on disk.
    Returns: {"libbers": [...], "files": [...]}
    """
    try:
        manager = LibberStateManager.instance()
        libbers = manager.list_libbers()

        # Also scan default directory for available files
        libber_dir = default_libber_dir()
        files = []
        if os.path.exists(libber_dir):
            files = [f for f in os.listdir(libber_dir) if f.endswith('.json')]

        return web.json_response({
            "libbers": libbers,
            "files": files,
            "libber_dir": libber_dir,
            "count": len(libbers)
        })

    except Exception as e:
        logger.exception("Error listing libbers")
        return web.json_response({"error": str(e)}, status=500)


@routes.get("/fbtools/libber/get_data/{name}")
async def libber_get_data(request):
    """
    Get Libber data for UI display.
    Returns: {"keys": [...], "lib_dict": {...}, "delimiter": str, "max_depth": int}
    """
    try:
        name = request.match_info.get("name")
        if not name:
            return web.json_response({"error": "name required"}, status=400)

        manager = LibberStateManager.instance()
        data = manager.get_libber_data(name)

        if not data:
            return web.json_response({"error": f"Libber '{name}' not found"}, status=404)

        return web.json_response(data)

    except Exception as e:
        logger.exception("Error getting libber data")
        return web.json_response({"error": str(e)}, status=500)


@routes.post("/fbtools/libber/apply")
async def libber_apply(request):
    """
    Apply Libber substitutions to text.
    Body: {"name": str, "text": str}
    Returns: {"result": str, "original": str}
    """
    try:
        data = await request.json()
        name = data.get("name")
        text = data.get("text")

        if not name or text is None:
            return web.json_response({"error": "name and text required"}, status=400)

        manager = LibberStateManager.instance()
        libber = manager.get_libber(name)

        if not libber:
            return web.json_response({"error": f"Libber '{name}' not found"}, status=404)

        result = libber.substitute(text)

        return web.json_response({
            "result": result,
            "original": text,
            "name": name
        })

    except Exception as e:
        logger.exception("Error applying libber")
        return web.json_response({"error": str(e)}, status=500)


@routes.get("/fbtools/libber/scan")
async def libber_scan(request):
    """
    Scan the default libber directory and return metadata for all libbers.
    Returns: {"libbers": [{name, entry_count, delimiter, max_depth, filepath}], "libber_dir": str}
    """
    try:
        manager = LibberStateManager.instance()
        libber_dir = default_libber_dir()
        result = []
        seen = set()

        # Disk files first
        if os.path.exists(libber_dir):
            for fname in sorted(os.listdir(libber_dir)):
                if not fname.endswith(".json"):
                    continue
                name = fname[:-5]
                seen.add(name)
                filepath = os.path.join(libber_dir, fname)
                try:
                    with open(filepath, encoding="utf-8") as fh:
                        data = json.load(fh)
                    result.append({
                        "name": name,
                        "entry_count": len(data.get("libs", {})),
                        "delimiter": data.get("delimiter", "%"),
                        "max_depth": data.get("max_depth", 10),
                        "filepath": filepath,
                    })
                except Exception:
                    result.append({
                        "name": name, "entry_count": 0,
                        "delimiter": "%", "max_depth": 10,
                        "filepath": filepath,
                    })

        # In-memory libbers not yet saved to disk
        for name, libber in manager.libbers.items():
            if name not in seen:
                result.append({
                    "name": name,
                    "entry_count": len(libber.libs),
                    "delimiter": libber.delimiter,
                    "max_depth": libber.max_depth,
                    "filepath": None,
                    "unsaved": True,
                })

        return web.json_response({"libbers": result, "libber_dir": libber_dir})
    except Exception as e:
        logger.exception("Error scanning libbers")
        return web.json_response({"error": str(e)}, status=500)


@routes.post("/fbtools/libber/open")
async def libber_open(request):
    """
    Open a libber for editing — loads from disk if not already in memory.
    Body: {"name": str}
    Returns: {"name": str, "lib_dict": {}, "delimiter": str, "max_depth": int}
    """
    try:
        data = await request.json()
        name = data.get("name")
        if not name:
            return web.json_response({"error": "name required"}, status=400)

        manager = LibberStateManager.instance()
        libber = manager.ensure_libber(name)
        if not libber:
            return web.json_response({"error": f"Libber '{name}' not found"}, status=404)

        return web.json_response({
            "name": name,
            "lib_dict": libber.libs.copy(),
            "delimiter": libber.delimiter,
            "max_depth": libber.max_depth,
        })
    except Exception as e:
        logger.exception("Error opening libber")
        return web.json_response({"error": str(e)}, status=500)


@routes.post("/fbtools/libber/save_full")
async def libber_save_full(request):
    """
    Overwrite a libber's complete data in memory and persist to disk.
    Body: {"name": str, "lib_dict": {}, "delimiter": str, "max_depth": int}
    Returns: {"name": str, "entry_count": int, "filepath": str, "status": "saved"}
    """
    try:
        data = await request.json()
        name = data.get("name", "").strip()
        if not name:
            return web.json_response({"error": "name required"}, status=400)
        # Validate filename safety
        if any(c in name for c in r'/\:*?"<>|'):
            return web.json_response({"error": "name contains invalid characters"}, status=400)

        lib_dict  = data.get("lib_dict", {})
        delimiter = str(data.get("delimiter", "%"))[:1] or "%"
        max_depth = max(1, min(50, int(data.get("max_depth", 10))))

        manager = LibberStateManager.instance()
        libber = Libber(lib_dict=dict(lib_dict), delimiter=delimiter, max_depth=max_depth)
        manager.libbers[name] = libber

        libber_dir = default_libber_dir()
        os.makedirs(libber_dir, exist_ok=True)
        filepath = os.path.join(libber_dir, f"{name}.json")
        libber.save(filepath)

        return web.json_response({
            "name": name,
            "entry_count": len(libber.libs),
            "filepath": filepath,
            "status": "saved",
        })
    except Exception as e:
        logger.exception("Error saving full libber")
        return web.json_response({"error": str(e)}, status=500)


@routes.post("/fbtools/libber/delete")
async def libber_delete(request):
    """
    Delete a libber from memory and from disk.
    Body: {"name": str}
    Returns: {"name": str, "status": "deleted"}
    """
    try:
        data = await request.json()
        name = data.get("name", "").strip()
        if not name:
            return web.json_response({"error": "name required"}, status=400)

        manager = LibberStateManager.instance()
        manager.delete_libber(name)

        libber_dir = default_libber_dir()
        filepath = os.path.join(libber_dir, f"{name}.json")
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info("Deleted libber file: %s", filepath)

        return web.json_response({"name": name, "status": "deleted"})
    except Exception as e:
        logger.exception("Error deleting libber")
        return web.json_response({"error": str(e)}, status=500)


@routes.post("/fbtools/libber/rename")
async def libber_rename(request):
    """
    Rename a libber on disk and in memory.
    Body: {"old_name": str, "new_name": str}
    Returns: {"old_name": str, "new_name": str, "status": "renamed"}
    """
    try:
        data = await request.json()
        old_name = data.get("old_name", "").strip()
        new_name = data.get("new_name", "").strip()
        if not old_name or not new_name:
            return web.json_response({"error": "old_name and new_name required"}, status=400)
        if any(c in new_name for c in r'/\:*?"<>|'):
            return web.json_response({"error": "new_name contains invalid characters"}, status=400)

        manager = LibberStateManager.instance()
        libber_dir = default_libber_dir()
        old_path = os.path.join(libber_dir, f"{old_name}.json")
        new_path = os.path.join(libber_dir, f"{new_name}.json")

        if os.path.exists(new_path) and old_name != new_name:
            return web.json_response(
                {"error": f"A libber named '{new_name}' already exists"}, status=409
            )

        # Load into memory if needed
        libber = manager.ensure_libber(old_name)

        # Save under new name
        if libber:
            os.makedirs(libber_dir, exist_ok=True)
            libber.save(new_path)
            manager.libbers[new_name] = libber
        elif os.path.exists(old_path):
            import shutil
            shutil.copy2(old_path, new_path)

        # Remove old
        if old_name != new_name:
            if os.path.exists(old_path):
                os.remove(old_path)
            manager.delete_libber(old_name)

        return web.json_response({"old_name": old_name, "new_name": new_name, "status": "renamed"})
    except Exception as e:
        logger.exception("Error renaming libber")
        return web.json_response({"error": str(e)}, status=500)
