#!/usr/bin/env python3
"""Migrate existing loras.json files to lora_stack.json format and clean up legacy fields.

Two operations:
  1. loras.json   → lora_stack.json  (Wan2.2 High/Low entries)
  2. Clean existing lora_stack.json  (remove deprecated audio_enabled boolean;
     for LTX2.3 entries where audio_enabled was False, set audio=0, audio_to_video=0)

Scenes that already have a lora_stack.json are skipped for step 1 unless --force.
Step 2 always runs on lora_stack.json files that contain audio_enabled entries.

Usage:
    python scripts/migrate_lora_stack.py [SCENES_DIR] [--force] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _migrate_loras_json_to_stack(loras_json_path: Path) -> list[dict]:
    """Convert legacy loras.json to a flat lora_stack list.

    high entries → model_target = "Wan2.2-Wrapper-High"
    low  entries → model_target = "Wan2.2-Wrapper-Low"
    """
    try:
        with open(loras_json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  [WARN] cannot read {loras_json_path}: {exc}", file=sys.stderr)
        return []

    if not isinstance(data, dict):
        print(f"  [WARN] unexpected format in {loras_json_path}", file=sys.stderr)
        return []

    entries: list[dict] = []
    for target, key in [("Wan2.2-Wrapper-High", "high"), ("Wan2.2-Wrapper-Low", "low")]:
        for item in data.get(key, []):
            lora_name = item.get("lora_name", "")
            if not lora_name or lora_name.lower() == "none":
                continue
            entries.append({
                "lora":           lora_name,
                "model_target":   target,
                "strength_model": item.get("strength", 1.0),
                "strength_clip":  1.0,
                "enabled":        True,
                "blocks":         item.get("blocks", {}),
                "layer_filter":   item.get("layer_filter", ""),
                "low_mem_load":   item.get("low_mem_load", False),
                "merge_loras":    item.get("merge_loras", False),
            })
    return entries


def _clean_entry(entry: dict) -> tuple[dict, bool]:
    """Remove deprecated audio_enabled from an entry; return (cleaned, was_changed)."""
    if "audio_enabled" not in entry:
        return entry, False
    audio_enabled = entry.pop("audio_enabled")
    # For LTX2.3 entries where audio was explicitly disabled, preserve that intent
    if entry.get("model_target") == "LTX2.3" and audio_enabled is False:
        entry.setdefault("audio",          0.0)
        entry.setdefault("audio_to_video", 0.0)
        entry.setdefault("video",          1.0)
        entry.setdefault("video_to_audio", 1.0)
        entry.setdefault("other",          1.0)
    return entry, True


def clean_existing_stacks(scenes_root: Path, dry_run: bool) -> None:
    """Strip deprecated audio_enabled from all existing lora_stack.json files."""
    stack_files = sorted(scenes_root.rglob("lora_stack.json"))
    if not stack_files:
        return

    cleaned = already_clean = failed = 0
    for stack_path in stack_files:
        try:
            with open(stack_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  [WARN] cannot read {stack_path}: {exc}", file=sys.stderr)
            failed += 1
            continue

        if not isinstance(data, list):
            continue

        changed = False
        new_data = []
        for entry in data:
            entry, entry_changed = _clean_entry(dict(entry))
            changed = changed or entry_changed
            new_data.append(entry)

        if not changed:
            already_clean += 1
            continue

        scene_name = stack_path.parent.name
        print(f"  [CLEAN]   {scene_name}  (removed audio_enabled from {sum(1 for e in new_data if 'audio' in e or True)} entries)")
        if not dry_run:
            try:
                with open(stack_path, "w", encoding="utf-8") as fh:
                    json.dump(new_data, fh, indent=2)
                cleaned += 1
            except OSError as exc:
                print(f"    [ERROR] could not write {stack_path}: {exc}", file=sys.stderr)
                failed += 1
        else:
            cleaned += 1

    if cleaned or failed:
        suffix = "  [DRY-RUN]" if dry_run else ""
        print(f"  Cleaned{suffix}: {cleaned} stacks updated, {already_clean} already clean, {failed} errors")


def migrate_scenes(scenes_root: Path, force: bool, dry_run: bool) -> None:
    loras_files = sorted(scenes_root.rglob("loras.json"))

    if not loras_files:
        print(f"No loras.json files found under {scenes_root}")
        return

    skipped = migrated = failed = 0

    for loras_path in loras_files:
        scene_dir = loras_path.parent
        stack_path = scene_dir / "lora_stack.json"

        if stack_path.exists() and not force:
            print(f"  [SKIP]    {scene_dir.name}  (lora_stack.json already exists)")
            skipped += 1
            continue

        entries = _migrate_loras_json_to_stack(loras_path)

        if not entries:
            print(f"  [EMPTY]   {scene_dir.name}  (no entries after migration, skipping write)")
            skipped += 1
            continue

        print(f"  [MIGRATE] {scene_dir.name}  ({len(entries)} entries)")
        if not dry_run:
            try:
                with open(stack_path, "w", encoding="utf-8") as fh:
                    json.dump(entries, fh, indent=2)
                migrated += 1
            except OSError as exc:
                print(f"    [ERROR] could not write {stack_path}: {exc}", file=sys.stderr)
                failed += 1
        else:
            migrated += 1

    suffix = "  [DRY-RUN]" if dry_run else ""
    print(
        f"\nDone{suffix}: {migrated} migrated, {skipped} skipped, {failed} failed"
        f"  (of {len(loras_files)} total loras.json files)"
    )


def main() -> None:
    default_scenes = "/mnt/comfy_ssd/ComfyUI/output/scenes"

    parser = argparse.ArgumentParser(description="Migrate loras.json → lora_stack.json")
    parser.add_argument(
        "scenes_dir",
        nargs="?",
        default=default_scenes,
        help=f"Root directory to search for loras.json files (default: {default_scenes})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing lora_stack.json files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without writing any files",
    )
    args = parser.parse_args()

    scenes_root = Path(args.scenes_dir)
    if not scenes_root.is_dir():
        print(f"Error: scenes directory not found: {scenes_root}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {scenes_root} ...")
    print("\n--- Step 1: loras.json → lora_stack.json ---")
    migrate_scenes(scenes_root, force=args.force, dry_run=args.dry_run)
    print("\n--- Step 2: clean audio_enabled from existing lora_stack.json ---")
    clean_existing_stacks(scenes_root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
