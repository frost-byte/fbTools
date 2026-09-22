#!/usr/bin/env python3
"""Fix composition JSON files whose shots have duplicate ids.

Root cause (fixed in js/ui/composition_editor.js): shot ids used to come from a session-global
counter that never re-synced to a composition's own shots when it was loaded, so the first
"+ Add Shot" click after opening a composition could mint an id already used by one of its
existing shots. Anything keyed by shot id then silently collapses the two — Scene Cast Build's
timeline lookup for a Prompt Composition is the concrete case that surfaced it: switching between
two shots that both ended up "shot_1" always showed the same (last) shot's action text.

This renumbers every shot in an affected file to shot_1, shot_2, ... in array order. Shot ids have
no cross-references elsewhere in a composition file (dialogue speakers use subject slot letters,
not shot ids) — verified by hand against several real composition files before writing this script.

Usage:
    python scripts/fix_duplicate_shot_ids.py DATA_DIR [--dry-run]

DATA_DIR is the fbTools user-data directory containing prompt_compositions/ (no guessed default —
see docs/GOTCHAS.md on this machine having more than one ComfyUI install tree).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

BACKUP_SUFFIX = ".pre-shot-id-fix"


def has_duplicate_shot_ids(shots: list) -> bool:
    ids = [s.get("id") for s in shots if isinstance(s, dict)]
    return len(ids) != len(set(ids))


def renumber_shots(comp: dict) -> None:
    """Mutates comp["shots"] in place, assigning shot_1..shot_N in array order."""
    for i, shot in enumerate(comp.get("shots", []), 1):
        if isinstance(shot, dict):
            shot["id"] = f"shot_{i}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("data_dir", help="fbTools user-data directory containing prompt_compositions/")
    ap.add_argument("--dry-run", action="store_true", help="report what would change; write nothing")
    args = ap.parse_args()

    comps_dir = Path(args.data_dir) / "prompt_compositions"
    print(f"Resolved compositions directory: {comps_dir}")
    if not comps_dir.is_dir():
        print(f"Error: not a directory: {comps_dir}", file=sys.stderr)
        return 1

    files = sorted(comps_dir.glob("*.json"))
    fixed = skipped = warned = 0
    for path in files:
        try:
            comp = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[WARN]  {path.name}  (cannot read: {exc})")
            warned += 1
            continue

        shots = comp.get("shots", [])
        if not has_duplicate_shot_ids(shots):
            skipped += 1
            continue

        before = [s.get("id") for s in shots if isinstance(s, dict)]
        renumber_shots(comp)
        after = [s.get("id") for s in comp.get("shots", []) if isinstance(s, dict)]
        print(f"[FIX]   {path.name}  {before} -> {after}")
        fixed += 1

        if not args.dry_run:
            bak = path.with_name(path.name + BACKUP_SUFFIX)
            if not bak.exists():
                shutil.copy2(path, bak)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(json.dumps(comp, indent=2, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, path)

    tag = " [DRY-RUN]" if args.dry_run else ""
    print(f"\nDone{tag}: {fixed} fixed, {skipped} already unique, {warned} warning(s) (of {len(files)} total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
