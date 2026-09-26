"""Pure logic for patching an API-format ComfyUI workflow template by node title.

No ComfyUI dependencies — the template is just the JSON dict ComfyUI's own "Export (API)" produces:
{node_id: {"class_type": str, "inputs": {...}, "_meta": {"title": str}}}. Node ids are unstable
across graph edits, so parameter nodes are located by their `_meta.title` instead — the workflow
author sets these titles once in the ComfyUI UI before exporting.
"""
from __future__ import annotations

import copy
import json
import logging

# Pure module (no ComfyUI deps, no sibling-utils imports — see feedback_relative_imports memory
# convention) — plain stdlib logging rather than utils/logging_utils.get_logger().
logger = logging.getLogger(__name__)


def load_template(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def find_node_by_title(template: dict, title: str, *, required: bool = True) -> str | None:
    """Return the node id whose `_meta.title` exactly matches `title`.

    By default (required=True) raises ValueError naming the missing title if no node matches — a
    silent wrong-node patch is worse than a loud failure here. With required=False, a missing title
    returns None instead (used for optional overrides a template may or may not expose) — but a
    genuine duplicate still always raises regardless of `required`, since an ambiguous title is a
    template authoring bug either way, not a "feature not supported" case.
    """
    matches = [
        node_id for node_id, node in template.items()
        if isinstance(node, dict) and node.get("_meta", {}).get("title") == title
    ]
    if not matches:
        if required:
            raise ValueError(f"No node titled {title!r} found in template")
        return None
    if len(matches) > 1:
        raise ValueError(f"Multiple nodes titled {title!r} found in template: {matches}")
    return matches[0]


def patch_prompt(
    template: dict,
    *,
    image: str,
    prompt_text: str,
    seed: int,
    filename_prefix: str,
    overrides: dict[str, dict] | None = None,
) -> dict:
    """Deep-copy `template` and patch its 4 required contract nodes' widget inputs, plus any
    optional overrides the template happens to support.

    Required contract (node titles the workflow author must set before exporting):
      IN:image  — a LoadImage-style node; patches its "image" input.
      IN:prompt — the MiniMax H3 Reference to Video node; patches its "prompt" input.
      IN:seed   — the RandomNoise node; patches its "noise_seed" input.
      OUT:save  — the SaveImage node; patches its "filename_prefix" input.

    `overrides` is an optional {title: {input_name: value, ...}} mapping for anything a template
    may additionally expose (e.g. {"IN:model": {"unet_name": "..."}, "IN:scheduler": {"scheduler":
    "simple", "steps": 8}}). Each title is looked up leniently: if the template doesn't have a node
    with that title, the override is silently skipped (debug-logged) rather than raised — callers
    are expected to only offer a control for a title the template actually supports (see a
    template-capability check), so reaching here with an unsupported title is a caller-side
    inconsistency, not a user-facing error.

    Returns a new dict; `template` itself is never mutated.
    """
    patched = copy.deepcopy(template)

    image_id = find_node_by_title(patched, "IN:image")
    patched[image_id]["inputs"]["image"] = image

    prompt_id = find_node_by_title(patched, "IN:prompt")
    patched[prompt_id]["inputs"]["prompt"] = prompt_text

    seed_id = find_node_by_title(patched, "IN:seed")
    patched[seed_id]["inputs"]["noise_seed"] = seed

    save_id = find_node_by_title(patched, "OUT:save")
    patched[save_id]["inputs"]["filename_prefix"] = filename_prefix

    for title, field_values in (overrides or {}).items():
        node_id = find_node_by_title(patched, title, required=False)
        if node_id is None:
            logger.debug("patch_prompt: template has no node titled %r, skipping override", title)
            continue
        patched[node_id]["inputs"].update(field_values)

    return patched
