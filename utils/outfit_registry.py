"""Pure logic for the Outfit Registry system.

No ComfyUI dependencies — all I/O helpers that need folder_paths or
torch live in extension.py so this module is fully testable in CI.
"""
from __future__ import annotations

import copy
import json
import os
import shutil

_EMPTY_DATA: dict = {"version": 1, "outfits": {}}


def _normalize_outfit_image(entry: "str | dict") -> dict:
    if isinstance(entry, str):
        return {"file": entry, "role": "costume detail"}
    return {"file": str(entry.get("file", "")), "role": str(entry.get("role", "costume detail")) or "costume detail"}


def _normalize_outfit_images(entries: list) -> "list[dict]":
    return [_normalize_outfit_image(e) for e in (entries or []) if e]


class OutfitRegistry:
    """In-memory outfit registry.  Wire between Load → Define → (SceneCompose) nodes."""

    def __init__(
        self,
        outfits: dict | None = None,
        file_path: str | None = None,
        version: int = 1,
    ) -> None:
        self.outfits: dict = outfits or {}
        self.file_path: str | None = file_path
        self.version: int = version

    # ── serialisation ──────────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, data: dict, file_path: str | None = None) -> "OutfitRegistry":
        outfits = copy.deepcopy(data.get("outfits", {}))
        for entry in outfits.values():
            if "reference_images" in entry:
                entry["reference_images"] = _normalize_outfit_images(entry["reference_images"])
        return cls(
            outfits=outfits,
            file_path=file_path,
            version=data.get("version", 1),
        )

    def to_dict(self) -> dict:
        return {"version": self.version, "outfits": copy.deepcopy(self.outfits)}

    # ── mutation ───────────────────────────────────────────────────────────────

    def define(
        self,
        outfit_id: str,
        name: str,
        description: str,
        tags: list[str] | None = None,
        reference_images: "list | None" = None,
    ) -> "OutfitRegistry":
        """Return a NEW registry with the outfit entry added or updated.

        reference_images, if provided, replaces the stored list (normalized to
        {file, role} dicts).  Pass None to preserve the existing list.
        """
        new_reg = copy.deepcopy(self)
        entry: dict = new_reg.outfits.get(outfit_id, {})
        entry["name"] = name or outfit_id
        entry["description"] = description
        if tags is not None:
            entry["tags"] = tags
        if reference_images is not None:
            entry["reference_images"] = _normalize_outfit_images(reference_images)
        new_reg.outfits[outfit_id] = entry
        return new_reg

    def remove(self, outfit_id: str) -> "OutfitRegistry":
        """Return a NEW registry with the outfit entry removed."""
        new_reg = copy.deepcopy(self)
        new_reg.outfits.pop(outfit_id, None)
        return new_reg

    # ── queries ────────────────────────────────────────────────────────────────

    def get_outfit(self, outfit_id: str) -> dict | None:
        return self.outfits.get(outfit_id)

    def get_description(self, outfit_id: str) -> str:
        entry = self.outfits.get(outfit_id)
        return entry.get("description", "") if entry else ""

    def outfit_ids(self) -> list[str]:
        return sorted(self.outfits.keys())

    def list_outfits(self, tag_filter: str | None = None) -> str:
        """Return a formatted human-readable outfit list, optionally filtered by tag."""
        if not self.outfits:
            return "No outfits defined."
        lines = []
        for outfit_id in sorted(self.outfits):
            entry = self.outfits[outfit_id]
            tags = entry.get("tags", [])
            if tag_filter and tag_filter not in tags:
                continue
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            desc_preview = entry.get("description", "")
            if len(desc_preview) > 80:
                desc_preview = desc_preview[:77] + "…"
            lines.append(f"[{outfit_id}] {entry.get('name', outfit_id)}{tag_str}")
            if desc_preview:
                lines.append(f"  {desc_preview}")
        if not lines:
            return f"No outfits matching tag '{tag_filter}'."
        return "\n".join(lines)

    def save(self, path: str | None = None, backup: bool = True) -> str:
        target = path or self.file_path
        if not target:
            raise ValueError("No file path specified for outfit registry save")
        save_outfit_registry(self, target, backup=backup)
        return target


# ── Persistence ────────────────────────────────────────────────────────────────

def load_outfit_registry(path: str) -> OutfitRegistry:
    """Load registry from JSON. Returns empty registry if file absent."""
    if not os.path.exists(path):
        return OutfitRegistry(file_path=path)
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return OutfitRegistry.from_dict(data, file_path=path)


def save_outfit_registry(registry: OutfitRegistry, path: str, backup: bool = True) -> None:
    """Write registry to JSON, optionally creating a .bak first."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if backup and os.path.exists(path):
        shutil.copy2(path, path + ".bak")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(registry.to_dict(), fh, indent=2, ensure_ascii=False)
    registry.file_path = path
