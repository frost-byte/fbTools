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


# MiniMax H3 Reference to Video's ref_image_0..ref_image_8 are real IMAGE socket inputs (each fed,
# in the source workflow, by its own chain: a DenoAdvancedImageSourceLoader batch -> a KJNodes
# Set/Get global -> a GetImagesFromBatchIndexed per slot -> an Any Switch (rgthree) per slot) — NOT
# patchable widget values, so IN:refs targets the loader itself, not the Reference-to-Video node.
# DenoAdvancedImageSourceLoader's own "image_paths" field IS a plain multiline-string widget (one
# input/-relative filename per line), confirmed against a real exported h3_character_sheet workflow
# (2026-09-27) — this is the one field patch_character_sheet_prompt actually writes.
MAX_REF_IMAGES = 9


def patch_character_sheet_prompt(
    template: dict,
    *,
    ref_images: list[str],
    mode_select: bool,
    seed: int,
    filename_prefix: str,
    overrides: dict[str, dict] | None = None,
) -> dict:
    """Deep-copy `template` and patch the H3 character/face-sheet template's required contract
    nodes, plus any optional overrides the template happens to support.

    This template's required contract differs from patch_prompt()'s (no single IN:image/IN:prompt —
    up to 9 reference images feed one loader node, and prompt text comes from the template's own
    internal per-mode DictCreate nodes, selected via IN:mode, not patched here):

      IN:refs  — the DenoAdvancedImageSourceLoader node; patches its "image_paths" input to
                 `ref_images` joined with newlines (that node splits the 9 ref_image_N sockets on
                 the Reference-to-Video node back out of this one field internally — see the module
                 comment above).
      IN:mode  — a KJNodes LazySwitchKJ node; patches its "switch" BOOLEAN input.
                 False = Character Sheet Options, True = Face Sheet Options (wired to on_false/
                 on_true respectively — Video mode has no branch on this switch at all, unlike the
                 3-way ImpactSwitch this replaced, since it's out of scope for this feature).
      IN:seed  — the RandomNoise node; patches its "noise_seed" input.
      OUT:save — the SaveImage node; patches its "filename_prefix" input.

    `overrides` behaves exactly as in patch_prompt(): a lenient {title: {field: value}} mapping,
    silently skipped per-title if the template doesn't expose it.

    Returns a new dict; `template` itself is never mutated.
    """
    if not ref_images:
        raise ValueError("ref_images must contain at least one image")
    if len(ref_images) > MAX_REF_IMAGES:
        raise ValueError(f"ref_images supports at most {MAX_REF_IMAGES} images, got {len(ref_images)}")

    patched = copy.deepcopy(template)

    # The 9 ref_image_N sockets are each fed by their own GetImagesFromBatchIndexed(index=N) node
    # reading the SAME shared batch this loader produces — those indices are hard-coded 0-8 in the
    # source workflow with no bounds checking, so a batch smaller than 9 crashes execution with a
    # raw IndexError (confirmed live, 2026-09-27: "index 8 is out of bounds for dimension 0 with
    # size 1" from a single-image request). Padding by cycling the given images back over
    # themselves keeps every slot valid while leaving index 0 — the sole outfit reference, see the
    # module comment above — exactly what the caller actually chose.
    padded = (ref_images * (MAX_REF_IMAGES // len(ref_images) + 1))[:MAX_REF_IMAGES]

    refs_id = find_node_by_title(patched, "IN:refs")
    patched[refs_id]["inputs"]["image_paths"] = "\n".join(padded)

    mode_id = find_node_by_title(patched, "IN:mode")
    patched[mode_id]["inputs"]["switch"] = mode_select

    seed_id = find_node_by_title(patched, "IN:seed")
    patched[seed_id]["inputs"]["noise_seed"] = seed

    save_id = find_node_by_title(patched, "OUT:save")
    patched[save_id]["inputs"]["filename_prefix"] = filename_prefix

    for title, field_values in (overrides or {}).items():
        node_id = find_node_by_title(patched, title, required=False)
        if node_id is None:
            logger.debug(
                "patch_character_sheet_prompt: template has no node titled %r, skipping override", title
            )
            continue
        patched[node_id]["inputs"].update(field_values)

    return patched
