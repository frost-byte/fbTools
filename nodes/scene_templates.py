"""Scene Template nodes (SceneTemplateLoad, SceneTemplateList).

Moved out of extension.py (pure code motion, Plan 26). Their REST routes were already
extracted to nodes/registry_api.py in an earlier pass and need no changes here.
"""
from __future__ import annotations

import os

from comfy_api.latest import io

from .shared import prefixed_node_id, default_scene_templates_dir, reload_counter, send_status_update
from ..utils.scene_templates import (
    load_template as _load_scene_template,
    scan_templates as _scan_scene_templates,
    template_ids as _scene_template_ids,
    format_template_list as _format_template_list,
    dir_fingerprint as _templates_dir_fingerprint,
)
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


# ── Custom type: SCENE_TEMPLATE ──────────────────────────────────────────────

SCENE_TEMPLATE_TYPE = "SCENE_TEMPLATE"


@io.comfytype(io_type=SCENE_TEMPLATE_TYPE)
class SceneTemplateIOType:
    """Carries a SceneTemplate instance between Load → SceneCompose nodes."""
    Type = object  # SceneTemplate

    class Input(io.Input):
        def __init__(self, name: str, **kwargs):
            super().__init__(name, **kwargs)

    class Output(io.Output):
        def __init__(self, name: str = "template", **kwargs):
            super().__init__(name, **kwargs)


# ── Scene Template helpers ─────────────────────────────────────────────────────

def _template_get_ids() -> list[str]:
    """Return available template IDs for combo population at schema time."""
    try:
        ids = _scene_template_ids(default_scene_templates_dir())
        return ids if ids else ["(none)"]
    except Exception:
        return ["(none)"]


# ── Node: SceneTemplateLoad ───────────────────────────────────────────────────

class SceneTemplateLoad(io.ComfyNode):
    """Load a scene template from the scene_templates directory.

    The combo is populated at extension load time from the user's
    scene_templates/ directory (seeded with bundled examples on first use).
    Press R to refresh the list after adding new templates.
    """
    node_id = prefixed_node_id("SceneTemplateLoad")
    display_name = "Scene Template Load"
    category = "🧊 frost-byte/Scene"
    is_output_node = True

    @classmethod
    def define_schema(cls):
        template_id_options = _template_get_ids()
        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            category=cls.category,
            is_output_node=cls.is_output_node,
            inputs=[
                io.Combo.Input(
                    "template_id",
                    options=template_id_options,
                    display_name="Template ID",
                    tooltip="Scene template to load. Press R to refresh the list after adding new templates.",
                ),
            ],
            outputs=[
                SceneTemplateIOType.Output(
                    "template",
                    display_name="Scene Template",
                    tooltip="Template object for wiring into SceneCompose.",
                ),
                io.String.Output(
                    "slot_info",
                    display_name="Slot Info",
                    tooltip="Formatted summary of slot requirements.",
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, template_id: str = "", **_):
        templates_dir = default_scene_templates_dir()
        path = os.path.join(templates_dir, f"{template_id}.json")
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0
        return (path, mtime, reload_counter("scene_template"))

    @classmethod
    def execute(cls, template_id: str = "") -> io.NodeOutput:
        templates_dir = default_scene_templates_dir()
        if not template_id or template_id == "(none)":
            logger.warning("SceneTemplateLoad: no template_id selected")
            return io.NodeOutput(None, "")
        path = os.path.join(templates_dir, f"{template_id}.json")
        if not os.path.exists(path):
            logger.warning("SceneTemplateLoad: template not found: %s", path)
            return io.NodeOutput(None, f"Template not found: {template_id}")
        template = _load_scene_template(path)
        slot_info = template.format_slot_info()
        send_status_update(
            cls.node_id,
            f"Loaded: {template.name} | {template.slot_count} slot(s) | {len(template.shots)} shots",
        )
        return io.NodeOutput(
            template,
            slot_info,
            ui={"slot_info": slot_info},
        )


# ── Node: SceneTemplateList ───────────────────────────────────────────────────

class SceneTemplateList(io.ComfyNode):
    """List all available scene templates.

    Scans the scene_templates/ directory and returns a formatted summary.
    Useful for quickly reviewing available templates.
    """
    node_id = prefixed_node_id("SceneTemplateList")
    display_name = "Scene Template List"
    category = "🧊 frost-byte/Scene"
    is_output_node = True

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            category=cls.category,
            is_output_node=cls.is_output_node,
            inputs=[],
            outputs=[
                io.String.Output(
                    "template_list",
                    display_name="Template List",
                    tooltip="Formatted list of all available templates.",
                ),
                io.Int.Output("template_count", display_name="Template Count"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, **_):
        templates_dir = default_scene_templates_dir()
        return (_templates_dir_fingerprint(templates_dir), reload_counter("scene_template"))

    @classmethod
    def execute(cls) -> io.NodeOutput:
        templates_dir = default_scene_templates_dir()
        listing = _format_template_list(templates_dir)
        count = len(_scan_scene_templates(templates_dir))
        return io.NodeOutput(listing, count, ui={"template_list": listing, "template_count": count})
