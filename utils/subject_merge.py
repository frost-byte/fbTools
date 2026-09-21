"""Pure logic for merging subjects (and moving bundles) into one subject.

Used by scripts/merge_subjects.py, which does the file I/O. Stdlib only, no ComfyUI and no
sibling utils imports (see docs/GOTCHAS.md), so it can be tested with plain dicts.

A merge source is either:
  "subject"          - the whole subject: its bundles move to the target and every reference to
                       it (composition slots, scene-cast entries, saved workflow cast entries)
                       is rewritten to the target id.
  "subject:bundle"   - only that bundle moves to the target subject; the source subject and its
                       references stay as they are.
"""
from __future__ import annotations

import copy
import json


def parse_source_spec(spec: str) -> tuple[str, str | None]:
    """'alex_bob' -> ('alex_bob', None); 'alex_bob:some_bundle' -> ('alex_bob', 'some_bundle')."""
    spec = (spec or "").strip()
    if not spec:
        raise ValueError("empty source")
    subject, sep, bundle = spec.partition(":")
    subject, bundle = subject.strip(), bundle.strip()
    if not subject:
        raise ValueError(f"source {spec!r} has no subject")
    if sep and not bundle:
        raise ValueError(f"source {spec!r} has an empty bundle after ':'")
    return subject, (bundle or None)


def _empty(value) -> bool:
    return value is None or value == "" or value == {} or value == []


def _sheet_key(entry) -> str:
    return entry.get("file", "") if isinstance(entry, dict) else str(entry)


def build_merged_profile(
    subjects: dict,
    into: str,
    primary: str | None,
    order: list[str],
) -> dict:
    """The profile for the merged subject.

    Base is the existing target profile if there is one, else the `primary` source's profile.
    Empty fields (and empty keys inside appearance/voice) are then filled from the primary and the
    remaining sources in `order`; non-empty base values are never overwritten. Reference sheet
    images are the de-duplicated union. `name` is set to the target id.
    """
    donors = []
    for sid in ([primary] if primary else []) + list(order):
        if sid and sid in subjects and sid not in donors and sid != into:
            donors.append(sid)
    if into in subjects:
        base = copy.deepcopy(subjects[into])
    elif donors:
        base = copy.deepcopy(subjects[donors[0]])
    else:
        base = {}
    for sid in donors:
        for key, value in subjects[sid].items():
            if key == "character_sheet_images":
                continue
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                for k2, v2 in value.items():
                    if _empty(base[key].get(k2)) and not _empty(v2):
                        base[key][k2] = copy.deepcopy(v2)
            elif _empty(base.get(key)) and not _empty(value):
                base[key] = copy.deepcopy(value)

    sheets: list = []
    seen: set[str] = set()
    for sid in ([into] if into in subjects else []) + donors:
        for entry in subjects[sid].get("character_sheet_images", []) or []:
            k = _sheet_key(entry)
            if k and k not in seen:
                seen.add(k)
                sheets.append(copy.deepcopy(entry))
    if sheets or "character_sheet_images" in base:
        base["character_sheet_images"] = sheets
    base["name"] = into
    return base


def retarget_bundles(
    bundles: dict,
    into: str,
    whole_subjects: set[str],
    bundle_moves: list[tuple[str, str]],
) -> tuple[list[str], list[str]]:
    """Point bundles at `into` (mutates `bundles`). Returns (moved_ids, warnings).

    Every bundle owned by a whole subject moves; each (subject, bundle) move moves just that
    bundle, provided it exists and is owned by that subject.
    """
    moved: list[str] = []
    warnings: list[str] = []
    for bid, b in bundles.items():
        if b.get("subject_id") in whole_subjects and b.get("subject_id") != into:
            b["subject_id"] = into
            moved.append(bid)
    for subject, bid in bundle_moves:
        b = bundles.get(bid)
        if b is None:
            warnings.append(f"bundle '{bid}' not found")
        elif b.get("subject_id") == into:
            continue
        elif b.get("subject_id") != subject:
            warnings.append(f"bundle '{bid}' belongs to '{b.get('subject_id')}', not '{subject}' - skipped")
        else:
            b["subject_id"] = into
            moved.append(bid)
    return moved, warnings


def rewrite_composition(comp: dict, id_map: dict[str, str], merged_profile: dict) -> bool:
    """Repoint composition slots at the merged subject (mutates). True when anything changed.

    Slot snapshots for remapped slots are replaced with the merged profile so the snapshot
    fallback agrees with the live registry.
    """
    subjects = comp.get("subjects")
    if not isinstance(subjects, dict):
        return False
    changed = False
    for slot, sid in list(subjects.items()):
        if sid in id_map and id_map[sid] != sid:
            subjects[slot] = id_map[sid]
            snaps = comp.get("_subject_snapshots")
            if isinstance(snaps, dict) and slot in snaps:
                snaps[slot] = copy.deepcopy(merged_profile)
            changed = True
    return changed


def rewrite_cast_entries(entries, id_map: dict[str, str]) -> int:
    """Rewrite subject_id in a list of cast-entry dicts (mutates). Returns how many changed."""
    n = 0
    if not isinstance(entries, list):
        return 0
    for e in entries:
        if isinstance(e, dict) and e.get("subject_id") in id_map and id_map[e["subject_id"]] != e["subject_id"]:
            e["subject_id"] = id_map[e["subject_id"]]
            n += 1
    return n


def rewrite_scene_casts(casts: dict, id_map: dict[str, str]) -> int:
    """Rewrite entries of every saved scene cast. Returns the number of entries changed."""
    n = 0
    for cast in (casts or {}).values():
        if isinstance(cast, dict):
            n += rewrite_cast_entries(cast.get("entries"), id_map)
    return n


def rewrite_workflow(obj, id_map: dict[str, str]) -> int:
    """Walk a workflow JSON structure and rewrite cast-entry JSON strings in place.

    Any string value that parses to a list of dicts carrying `subject_id` (the
    `cast_entries_json` widget, in either the widgets_values list or the named/API form) is
    re-serialised with remapped ids. Returns the number of entries changed.
    """
    total = 0

    def fix_string(s: str):
        st = s.strip()
        if not (st.startswith("[") and "subject_id" in st):
            return None
        try:
            parsed = json.loads(st)
        except ValueError:
            return None
        if not (isinstance(parsed, list) and parsed and all(isinstance(x, dict) for x in parsed)):
            return None
        n = rewrite_cast_entries(parsed, id_map)
        return (json.dumps(parsed, separators=(",", ":")), n) if n else None

    def walk(node):
        nonlocal total
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(v, str):
                    fixed = fix_string(v)
                    if fixed:
                        node[k], n = fixed
                        total += n
                else:
                    walk(v)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                if isinstance(v, str):
                    fixed = fix_string(v)
                    if fixed:
                        node[i], n = fixed
                        total += n
                else:
                    walk(v)

    walk(obj)
    return total
