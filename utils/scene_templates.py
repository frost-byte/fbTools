"""Pure logic for the Scene Template system.

No ComfyUI dependencies — all I/O helpers that need folder_paths or
torch live in extension.py so this module is fully testable in CI.
"""
from __future__ import annotations

import copy
import json
import os


# ── SceneTemplate ──────────────────────────────────────────────────────────────

class SceneTemplate:
    """Wraps a single scene template JSON dict with typed accessors."""

    def __init__(self, data: dict, file_path: str | None = None) -> None:
        self.data = copy.deepcopy(data)
        self.file_path = file_path

    @property
    def template_id(self) -> str:
        if "id" in self.data:
            return self.data["id"]
        # Fall back to filename stem
        if self.file_path:
            return os.path.splitext(os.path.basename(self.file_path))[0]
        return ""

    @property
    def name(self) -> str:
        return self.data.get("name", self.template_id)

    @property
    def description(self) -> str:
        return self.data.get("description", "")

    @property
    def slots(self) -> dict:
        return self.data.get("slots", {})

    @property
    def slot_ids(self) -> list[str]:
        return sorted(self.slots.keys())

    @property
    def slot_count(self) -> int:
        return len(self.slots)

    @property
    def shots(self) -> list:
        return self.data.get("shots", [])

    @property
    def environment(self) -> dict:
        return self.data.get("environment", {})

    @property
    def style(self) -> str:
        return self.data.get("style", "")

    @property
    def overall_soundscape(self) -> str:
        return self.data.get("overall_soundscape", "")

    @property
    def non_diegetic_music(self) -> str:
        return self.data.get("non_diegetic_music", "")

    def dialogue_placeholders(self) -> list[dict]:
        """Return shots that have a dialogue entry with placeholder=true, in order."""
        return [
            s for s in self.shots
            if isinstance(s.get("dialogue"), dict) and s["dialogue"].get("placeholder")
        ]

    def format_slot_info(self) -> str:
        """Return human-readable slot requirements for the slot_info output."""
        if not self.slots:
            return "No slots defined."
        lines = []
        for slot_id in sorted(self.slots):
            slot = self.slots[slot_id]
            role = slot.get("role", "")
            reqs = []
            if slot.get("needs_voice"):
                reqs.append("voice: required")
            if slot.get("needs_character_sheet"):
                reqs.append("character sheet: required")
            req_str = f" ({', '.join(reqs)})" if reqs else ""
            lines.append(f"Slot {slot_id}: {role}{req_str}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return copy.deepcopy(self.data)


# ── Directory scanning ─────────────────────────────────────────────────────────

def scan_templates(templates_dir: str) -> list[dict]:
    """Scan a directory for template JSON files.

    Returns sorted list of metadata dicts:
        {template_id, name, description, slot_count, file_path}
    Silently skips files that fail to parse.
    """
    if not os.path.isdir(templates_dir):
        return []
    results = []
    for fname in sorted(os.listdir(templates_dir)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(templates_dir, fname)
        template_id = os.path.splitext(fname)[0]
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            name = data.get("name", template_id)
            description = data.get("description", "")
            slot_count = len(data.get("slots", {}))
        except Exception:
            name = template_id
            description = "(error reading file)"
            slot_count = 0
        results.append({
            "template_id": template_id,
            "name": name,
            "description": description,
            "slot_count": slot_count,
            "file_path": path,
        })
    return results


def template_ids(templates_dir: str) -> list[str]:
    """Return sorted list of template IDs available in templates_dir."""
    return [t["template_id"] for t in scan_templates(templates_dir)]


def format_template_list(templates_dir: str) -> str:
    """Return formatted listing of all templates for the SceneTemplateList output."""
    templates = scan_templates(templates_dir)
    if not templates:
        return "No templates found."
    lines = []
    for t in templates:
        count = t["slot_count"]
        slot_str = f"{count} slot{'s' if count != 1 else ''}"
        desc = f" — {t['description']}" if t["description"] else ""
        lines.append(f"[{t['template_id']}] {t['name']} ({slot_str}){desc}")
    return "\n".join(lines)


def dir_fingerprint(templates_dir: str) -> tuple:
    """Return a tuple that changes whenever any template file is added, removed, or modified."""
    if not os.path.isdir(templates_dir):
        return ()
    items = []
    for fname in sorted(os.listdir(templates_dir)):
        if fname.endswith(".json"):
            path = os.path.join(templates_dir, fname)
            try:
                items.append((fname, os.path.getmtime(path)))
            except OSError:
                items.append((fname, 0))
    return tuple(items)


# ── Persistence ────────────────────────────────────────────────────────────────

def load_template(path: str) -> SceneTemplate:
    """Load a scene template from a JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return SceneTemplate(data, file_path=path)
