"""Cross-cutting helpers shared across nodes/*.py domain modules.

Kept minimal and dependency-free (no imports from sibling nodes/*.py modules
or from extension.py) so every domain module can import from here without
risk of a circular import. Extended incrementally as more domains move out
of extension.py.
"""

# Extension-wide node prefix to keep node_id globally unique across ComfyUI
EXTENSION_PREFIX = "fbt"


def prefixed_node_id(display_name: str) -> str:
    """Construct a globally-unique node_id using the shared extension prefix."""
    return f"{EXTENSION_PREFIX}_{display_name}"
