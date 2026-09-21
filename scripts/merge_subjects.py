#!/usr/bin/env python3
"""Merge subjects (and/or move individual bundles) into one subject.

A subject in fbTools is mostly a thin identity - a name, a pronoun style, a short name - while the
detail that matters (appearance, media, voice reference) lives in its bundles. This script folds
several subjects into one so a composition slot can use any of their bundles.

Sources (positional):
  SUBJECT          the whole subject: all its bundles move to the target, and every reference to it
                   (composition slots, scene casts, saved workflow cast entries) is rewritten
  SUBJECT:BUNDLE   only that bundle moves to the target; SUBJECT and its references stay

Files updated under DATA_DIR (the fbTools user-data directory):
  subject_profiles.json, reference_bundles.json, scene_casts.json, prompt_compositions/*.json
plus any workflow *.json found under each --workflows directory. Before a file is changed its
original is copied to <name>.pre-subject-merge (kept if one already exists).

The merged profile starts from the existing target subject if there is one, otherwise from
--primary (default: the first whole subject listed). Empty fields, including empty keys inside
appearance/voice, are filled from the other sources; reference sheet images are unioned.

Usage:
    python scripts/merge_subjects.py DATA_DIR --into alex alex_bob alex_jelly alex_amd_norsk \\
        [--primary alex_amd_norsk] [--workflows ~/ComfyUI/user/default/workflows] \\
        [--delete-sources] [--dry-run]
    python scripts/merge_subjects.py DATA_DIR --into alex alex_bob:alex_bob_alex_leaf
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("subject_merge", _HERE.parent / "utils" / "subject_merge.py")
sm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sm)

BACKUP_SUFFIX = ".pre-subject-merge"


def _load(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _save(path: Path, data, dry_run: bool) -> None:
    if dry_run:
        return
    bak = path.with_name(path.name + BACKUP_SUFFIX)
    if not bak.exists():
        shutil.copy2(path, bak)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge subjects / move bundles into one subject.")
    ap.add_argument("data_dir", help="fbTools user-data directory (contains subject_profiles.json)")
    ap.add_argument("sources", nargs="+", help="SUBJECT or SUBJECT:BUNDLE, one or more")
    ap.add_argument("--into", required=True, help="id of the merged (target) subject; created if missing")
    ap.add_argument("--primary", help="whole-subject source whose profile is the base (default: first listed)")
    ap.add_argument("--workflows", action="append", default=[], metavar="DIR",
                    help="also rewrite saved workflow JSON files under DIR (repeatable)")
    ap.add_argument("--delete-sources", action="store_true",
                    help="remove the merged-away subjects from subject_profiles.json")
    ap.add_argument("--dry-run", action="store_true", help="report what would change; write nothing")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    subj_path = data_dir / "subject_profiles.json"
    bun_path = data_dir / "reference_bundles.json"
    for p in (subj_path, bun_path):
        if not p.is_file():
            print(f"Error: {p} not found", file=sys.stderr)
            return 1

    whole: list[str] = []
    bundle_moves: list[tuple[str, str]] = []
    try:
        for spec in args.sources:
            subject, bundle = sm.parse_source_spec(spec)
            if bundle:
                bundle_moves.append((subject, bundle))
            elif subject not in whole:
                whole.append(subject)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    into = args.into.strip()
    if not into or ":" in into:
        print("Error: --into must be a plain subject id", file=sys.stderr)
        return 1
    if into in whole:
        whole.remove(into)

    subj_doc = _load(subj_path)
    subjects = subj_doc.get("subjects", {})
    missing = [s for s in whole if s not in subjects]
    if missing:
        print(f"Error: unknown subject(s): {', '.join(missing)}", file=sys.stderr)
        return 1
    primary = args.primary or (whole[0] if whole else (bundle_moves[0][0] if bundle_moves else None))
    if primary and primary != into and primary not in subjects:
        print(f"Error: primary subject '{primary}' not found", file=sys.stderr)
        return 1

    tag = " [DRY-RUN]" if args.dry_run else ""
    print(f"Merging into '{into}'{tag}; whole subjects: {whole or '-'}; bundle moves: "
          f"{[f'{s}:{b}' for s, b in bundle_moves] or '-'}; primary: {primary or '-'}")

    merged = sm.build_merged_profile(subjects, into, primary, whole + [s for s, _ in bundle_moves])
    created = into not in subjects
    subjects[into] = merged
    print(f"  subject '{into}': {'created' if created else 'updated'}")

    # bundles
    bun_doc = _load(bun_path)
    bundles = bun_doc.get("bundles", {})
    moved, warnings = sm.retarget_bundles(bundles, into, set(whole), bundle_moves)
    print(f"  bundles moved: {len(moved)}")
    for w in warnings:
        print(f"  [WARN] {w}")

    id_map = {s: into for s in whole}

    if args.delete_sources:
        for s in whole:
            subjects.pop(s, None)
        print(f"  removed subjects: {whole}")

    _save(subj_path, subj_doc, args.dry_run)
    _save(bun_path, bun_doc, args.dry_run)

    # compositions, scene casts, workflows only change for whole-subject merges
    if id_map:
        comp_dir = data_dir / "prompt_compositions"
        n_comp = 0
        for path in sorted(comp_dir.glob("*.json")) if comp_dir.is_dir() else []:
            try:
                comp = _load(path)
            except (OSError, ValueError):
                print(f"  [WARN] cannot read {path.name}")
                continue
            if sm.rewrite_composition(comp, id_map, merged):
                n_comp += 1
                _save(path, comp, args.dry_run)
        print(f"  compositions updated: {n_comp}")

        casts_path = data_dir / "scene_casts.json"
        if casts_path.is_file():
            casts_doc = _load(casts_path)
            n = sm.rewrite_scene_casts(casts_doc.get("casts", {}), id_map)
            if n:
                _save(casts_path, casts_doc, args.dry_run)
            print(f"  scene-cast entries updated: {n}")

        n_files = n_entries = 0
        for wf_dir in args.workflows:
            for path in sorted(Path(wf_dir).expanduser().rglob("*.json")):
                try:
                    wf = _load(path)
                except (OSError, ValueError):
                    continue
                n = sm.rewrite_workflow(wf, id_map)
                if n:
                    n_files += 1
                    n_entries += n
                    _save(path, wf, args.dry_run)
        if args.workflows:
            print(f"  workflows updated: {n_files} file(s), {n_entries} cast entr{'y' if n_entries == 1 else 'ies'}")

    print("Done." if not args.dry_run else "Done (dry run - nothing written).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
