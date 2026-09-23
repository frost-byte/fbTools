"""Cross-cutting helpers shared across nodes/*.py domain modules and extension.py.

Kept free of imports from sibling nodes/*.py modules and from extension.py so every
domain module can import from here without risk of a circular import.

Contents: the node-id prefix, the aiohttp `routes` singleton, per-user data directories and the
default registry paths, the websocket status helper, and the reload-counter registry that
node `fingerprint_inputs` read and the `/fbtools/<x>/reload` routes bump.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import folder_paths
from folder_paths import get_output_directory
from server import PromptServer

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

# Root of the fbTools package (the directory containing extension.py), NOT this nodes/ folder.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# Extension-wide node prefix to keep node_id globally unique across ComfyUI
EXTENSION_PREFIX = "fbt"


def prefixed_node_id(display_name: str) -> str:
    """Construct a globally-unique node_id using the shared extension prefix."""
    return f"{EXTENSION_PREFIX}_{display_name}"


# aiohttp route table of ComfyUI's PromptServer; every domain module registers its /fbtools/* routes here.
routes = PromptServer.instance.routes


# ── Reload counters ────────────────────────────────────────────────────────────
# POST /fbtools/<x>/reload bumps a counter so nodes that read the matching registry re-execute:
# their fingerprint_inputs include reload_counter("<x>"). Names in use: concept, subject,
# source_profile, scene_template, composition, outfit, cast.

_RELOAD_COUNTERS: dict[str, int] = {}


def reload_counter(name: str) -> int:
    """Current value of the named reload counter (0 until first bumped)."""
    return _RELOAD_COUNTERS.get(name, 0)


def bump_reload(name: str) -> int:
    """Increment the named reload counter and return the new value."""
    _RELOAD_COUNTERS[name] = _RELOAD_COUNTERS.get(name, 0) + 1
    return _RELOAD_COUNTERS[name]



# Status update helper for real-time node feedback
def send_status_update(
    node_id: str,
    status_text: str,
    source: str | None = None,
    level: str = "info",
    extra: dict | None = None,
):
    """Send status update to frontend via websocket.

    `extra` merges additional structured fields into the payload (e.g. progress
    machine-readable fields alongside the human-readable `status_text`) — used
    by long-running multi-step operations like Detect boundaries so the UI can
    render real progress instead of just echoing the latest message string.
    """
    try:
        from server import PromptServer
        server = PromptServer.instance
        payload = {
            "node": node_id,
            "status": status_text,
            "level": level,
        }
        if source:
            payload["source"] = source
        if extra:
            payload.update(extra)
        server.send_sync("fbtools.status", payload)
    except Exception as e:
        logger.debug(f"Failed to send status update: {e}")


def user_data_dir() -> str:
    """Return the package-specific persistent data dir under ComfyUI's user dir.

    Falls back to ComfyUI/user/default/<package> if get_user_directory() is unavailable.
    """
    package_name = os.path.basename(PACKAGE_ROOT)
    try:
        base = folder_paths.get_user_directory()
    except AttributeError:
        try:
            base = os.path.join(folder_paths.base_path, "user", "default")
        except AttributeError:
            base = get_output_directory()
    data_dir = os.path.join(base, package_name)
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def _user_subdir(name: str) -> str:
    """Create and return a named subdirectory under the package user-data dir."""
    path = os.path.join(user_data_dir(), name)
    os.makedirs(path, exist_ok=True)
    return path


def default_registry_path() -> str:
    """Default path for the concept registry JSON file."""
    return os.path.join(user_data_dir(), "concept_registry.json")


def default_subject_profiles_path() -> str:
    """Default path for the subject profiles JSON file."""
    return os.path.join(user_data_dir(), "subject_profiles.json")


def default_source_profiles_path() -> str:
    """Default path for the source profile registry JSON file."""
    return os.path.join(user_data_dir(), "source_profiles.json")


def default_bundle_registry_path() -> str:
    """Default path for the reference bundles JSON file."""
    return os.path.join(user_data_dir(), "reference_bundles.json")


def default_cast_registry_path() -> str:
    """Default path for the scene casts JSON file."""
    return os.path.join(user_data_dir(), "scene_casts.json")


def default_outfit_registry_path() -> str:
    """Default path for the outfit registry JSON file."""
    return os.path.join(user_data_dir(), "outfit_registry.json")


def default_scene_templates_dir() -> str:
    """Return (and create) the user scene_templates directory.

    Seeds bundled example templates on first use when the directory is empty.
    """
    path = _user_subdir("scene_templates")
    if not any(f.endswith(".json") for f in os.listdir(path)):
        _seed_bundled_templates(path)
    return path


def _seed_bundled_templates(dest_dir: str) -> None:
    """Copy bundled example templates into dest_dir (one-time initialisation)."""
    import shutil as _shutil
    src_dir = os.path.join(PACKAGE_ROOT, "scene_templates")
    if not os.path.isdir(src_dir):
        return
    for fname in os.listdir(src_dir):
        if fname.endswith(".json"):
            dst = os.path.join(dest_dir, fname)
            if not os.path.exists(dst):
                _shutil.copy2(os.path.join(src_dir, fname), dst)
    logger.info("Seeded %s with bundled scene templates from %s", dest_dir, src_dir)


def default_libber_dir():
    """Get default directory for storing libber files.

    Prefers user data dir; falls back to legacy output/libbers if that
    directory already has content (non-migrated setups).
    """
    new_dir = os.path.join(user_data_dir(), "libbers")
    if os.path.isdir(new_dir) and any(
        f.endswith(".json") for f in os.listdir(new_dir) if os.path.isfile(os.path.join(new_dir, f))
    ):
        return new_dir
    # Legacy fallback so existing libber files are not lost
    legacy_dir = os.path.join(get_output_directory(), "libbers")
    if not os.path.exists(legacy_dir):
        os.makedirs(legacy_dir, exist_ok=True)
    return legacy_dir


def default_scenes_dir():
    """Scenes directory: prefers user data dir; falls back to legacy output/scenes."""
    new_dir = os.path.join(user_data_dir(), "scenes")
    if os.path.isdir(new_dir) and any(
        os.path.isdir(os.path.join(new_dir, x)) for x in os.listdir(new_dir)
    ):
        return new_dir
    # Legacy location (keeps existing scenes accessible without migration)
    legacy_dir = os.path.join(get_output_directory(), "scenes")
    if not os.path.exists(legacy_dir):
        os.makedirs(legacy_dir, exist_ok=True)
        os.makedirs(os.path.join(legacy_dir, "default_scene"), exist_ok=True)
    return legacy_dir


def default_stories_dir():
    output_dir = get_output_directory()
    default_dir = os.path.join(output_dir, "stories")
    if not os.path.exists(default_dir):
        os.makedirs(default_dir, exist_ok=True)
    return default_dir


# ── Generic directory helpers ──────────────────────────────────────────────────
# Used across Scene, Story, and ScenePromptManager — no domain-specific meaning,
# so they live here rather than in any one domain module (Plan 20).

def get_subdirectories(directory_path: str) -> dict:
    """Return a dictionary mapping subdirectory names to their full paths."""
    subdir_dict = {}

    if not os.path.isdir(directory_path):
        logger.warning("Directory '%s' does not exist or is not a directory.", directory_path)
        return subdir_dict

    with os.scandir(directory_path) as entries:
        for entry in entries:
            if entry.is_dir():
                subdir_dict[entry.name] = entry.path

    return subdir_dict


def _directory_fingerprint(path: Path) -> tuple[str, int, int]:
    """Return a stable fingerprint for a directory tree.

    The hash includes relative paths plus mtime_ns/size for every file and directory,
    so edits, creates, deletes, and renames will invalidate cached nodes.
    """
    if not path.exists() or not path.is_dir():
        return ("missing", 0, 0)

    digest = hashlib.sha1()
    dir_count = 0
    file_count = 0

    for root, dirnames, filenames in os.walk(path):
        dirnames.sort()
        filenames.sort()

        root_path = Path(root)
        rel_root = root_path.relative_to(path).as_posix()
        rel_root = rel_root if rel_root else "."

        try:
            root_stat = root_path.stat()
            root_mtime_ns = int(root_stat.st_mtime_ns)
        except Exception:
            root_mtime_ns = 0

        digest.update(f"D|{rel_root}|{root_mtime_ns}\n".encode("utf-8"))
        dir_count += 1

        for filename in filenames:
            file_path = root_path / filename
            rel_file = file_path.relative_to(path).as_posix()
            try:
                st = file_path.stat()
                file_mtime_ns = int(st.st_mtime_ns)
                file_size = int(st.st_size)
            except Exception:
                file_mtime_ns = 0
                file_size = 0

            digest.update(f"F|{rel_file}|{file_mtime_ns}|{file_size}\n".encode("utf-8"))
            file_count += 1

    return (digest.hexdigest(), dir_count, file_count)
