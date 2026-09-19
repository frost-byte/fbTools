#!/usr/bin/env python3
"""Migrate Prompt Composition subject-slot keys from legacy "S1"/"S2"/...
notation to the letter notation ("A"/"B"/...) Source Profile clips already
use (see utils/slot_letters.py).

Rewrites, per composition JSON file:
  - dict keys: subjects, _subject_snapshots, outfit_overrides, outfit_ids,
    slot_descriptors, appearance_overrides
  - shots[].dialogue.speaker (string value)
  - literal {S1}/{S2}/... placeholders inside shots[].action, shots[].camera,
    and scene_synopsis

A composition whose subject keys are already letter-shaped (or has no
subjects at all) is skipped. A .bak copy is made before any file is
overwritten (skipped if a .bak already exists from a prior run).

DATA_DIR is required (no guessed default) — this machine has more than one
ComfyUI install tree, and this package's own user-data root does not live
under the code root (see docs/GOTCHAS.md); pass the directory that contains
a prompt_compositions/ subdirectory, e.g. the fbTools user-data root such as
.../ComfyUI/user/comfyui-fbTools.

Usage:
    python scripts/migrate_composition_slots.py DATA_DIR [--dry-run] [--force]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.slot_letters import slot_letter  # noqa: E402

_LEGACY_KEY_RE = re.compile(r"^S(\d+)$")

_SLOT_KEYED_DICT_FIELDS = (
    "subjects",
    "_subject_snapshots",
    "outfit_overrides",
    "outfit_ids",
    "slot_descriptors",
    "appearance_overrides",
)


def _build_mapping(subjects: dict) -> dict[str, str]:
    """Extract legacy "S<n>" keys, sorted numerically, mapped to slot_letter(i)."""
    legacy = []
    for key in subjects:
        m = _LEGACY_KEY_RE.match(key)
        if m:
            legacy.append((int(m.group(1)), key))
    legacy.sort(key=lambda pair: pair[0])
    return {old_key: slot_letter(i) for i, (_, old_key) in enumerate(legacy)}


def _rewrite_placeholders(text: str, old_to_new: dict[str, str]) -> str:
    if not text:
        return text
    # Longest old-key first so e.g. "S10" isn't corrupted by an "S1" replacement
    # running first.
    for old_key, new_key in sorted(old_to_new.items(), key=lambda kv: -len(kv[0])):
        text = text.replace(f"{{{old_key}}}", f"{{{new_key}}}")
    return text


def migrate_file(path: Path, dry_run: bool) -> str:
    """Returns a status string: "migrate", "skip", or "warn:<message>"."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            comp = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        return f"warn:cannot read file: {exc}"

    subjects = comp.get("subjects", {})
    old_to_new = _build_mapping(subjects)
    if not old_to_new:
        return "skip"

    warnings: list[str] = []

    for field in _SLOT_KEYED_DICT_FIELDS:
        d = comp.get(field)
        if not isinstance(d, dict):
            continue
        new_d = {}
        for k, v in d.items():
            new_k = old_to_new.get(k)
            if new_k is None:
                if k not in subjects:
                    warnings.append(f"orphaned key '{k}' in {field} — left unmigrated")
                new_k = k  # leave unmapped/orphaned keys untouched, don't drop data
            new_d[new_k] = v
        comp[field] = new_d

    shots_rewritten = 0
    for shot in comp.get("shots", []):
        changed = False
        dlg = shot.get("dialogue")
        if isinstance(dlg, dict) and dlg.get("speaker"):
            spk = dlg["speaker"]
            if spk in old_to_new:
                dlg["speaker"] = old_to_new[spk]
                changed = True
            elif spk not in subjects:
                warnings.append(f"orphaned dialogue speaker '{spk}' in shot '{shot.get('id', '?')}' — left unmigrated")
        for text_field in ("action", "camera"):
            original = shot.get(text_field, "")
            rewritten = _rewrite_placeholders(original, old_to_new)
            if rewritten != original:
                shot[text_field] = rewritten
                changed = True
        if changed:
            shots_rewritten += 1

    original_synopsis = comp.get("scene_synopsis", "")
    rewritten_synopsis = _rewrite_placeholders(original_synopsis, old_to_new)
    if rewritten_synopsis != original_synopsis:
        comp["scene_synopsis"] = rewritten_synopsis

    summary = ", ".join(f"{old}→{new}" for old, new in old_to_new.items())
    warn_suffix = f"; {len(warnings)} warning(s)" if warnings else ""
    print(f"[MIGRATE] {path.name}  ({len(old_to_new)} subjects: {summary}; "
          f"{shots_rewritten} shot(s) rewritten{warn_suffix})")
    for w in warnings:
        print(f"[WARN]    {path.name}  ({w})")

    if not dry_run:
        bak = path.with_suffix(path.suffix + ".bak")
        if bak.exists():
            print(f"[SKIP-BACKUP] {path.name}  (.bak already exists)")
        else:
            shutil.copy2(path, bak)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(comp, fh, indent=2, ensure_ascii=False)

    return "migrate"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate Prompt Composition subject-slot keys from S1/S2 to A/B letters."
    )
    parser.add_argument(
        "data_dir",
        help="fbTools user-data directory containing prompt_compositions/ "
             "(e.g. the ComfyUI install's user/comfyui-fbTools directory — "
             "no guessed default, this machine has more than one ComfyUI tree)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned changes without writing any files.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    compositions_dir = data_dir / "prompt_compositions"
    print(f"Resolved compositions directory: {compositions_dir}")
    if not compositions_dir.is_dir():
        print(f"Error: not a directory: {compositions_dir}", file=sys.stderr)
        sys.exit(1)

    files = sorted(compositions_dir.glob("*.json"))
    if not files:
        print(f"No composition files found under {compositions_dir}")
        return

    migrated = skipped = warned = 0
    for path in files:
        status = migrate_file(path, dry_run=args.dry_run)
        if status == "migrate":
            migrated += 1
        elif status == "skip":
            print(f"[SKIP]    {path.name}  (no legacy-shaped subject keys)")
            skipped += 1
        elif status.startswith("warn:"):
            print(f"[WARN]    {path.name}  ({status[5:]})")
            warned += 1

    suffix = " [DRY-RUN]" if args.dry_run else ""
    print(f"\nDone{suffix}: {migrated} migrated, {skipped} skipped, {warned} warnings "
          f"(of {len(files)} total)")


if __name__ == "__main__":
    main()
