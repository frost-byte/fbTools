#!/usr/bin/env python3
"""
Export each top-level Scenes collection from the Select Scene subgraph as a
standalone custom subgraph file, suitable for drag-and-drop in ComfyUI's
subgraph library (user/default/subgraphs/).

Usage:
  python scripts/export_scene_collections.py \\
      --input  /path/to/workflow.json \\
      --outdir /mnt/comfy_ssd/ComfyUI/user/default/subgraphs/ \\
      [--dry-run]
"""

import json
import argparse
import sys
from pathlib import Path


# ── dependency collector ──────────────────────────────────────────────────────

def collect_deps(root_id, sg_map):
    """Return ordered list of subgraph defs that root_id transitively needs."""
    visited = {}

    def visit(sid):
        if sid in visited or sid not in sg_map:
            return
        sg = sg_map[sid]
        visited[sid] = sg
        for n in sg.get("nodes", []):
            if n.get("type") in sg_map:
                visit(n["type"])

    visit(root_id)
    # root first, then dependencies (topological order doesn't matter for
    # ComfyUI — it resolves by UUID — but root first is tidy)
    ordered = [visited[root_id]]
    for k, v in visited.items():
        if k != root_id:
            ordered.append(v)
    return ordered


# ── instance node builder ─────────────────────────────────────────────────────

def make_instance_node(node_id, sg):
    """
    Build a minimal canvas-level instance node for subgraph `sg`.
    Inputs/outputs mirror the subgraph's declared slots; all links are null.
    """
    def slot_to_input(slot):
        inp = {
            "name": slot["name"],
            "type": slot["type"],
            "link": None,
        }
        if "label" in slot:
            inp["label"] = slot["label"]
        if "localized_name" in slot:
            inp["localized_name"] = slot.get("localized_name", slot["name"])
        # Inputs that carry widget values need a widget hint
        inp["widget"] = {"name": slot["name"]}
        return inp

    def slot_to_output(slot):
        out = {
            "name": slot["name"],
            "type": slot["type"],
            "links": [],
        }
        if "label" in slot:
            out["label"] = slot["label"]
        if "localized_name" in slot:
            out["localized_name"] = slot.get("localized_name", slot["name"])
        return out

    inputs  = [slot_to_input(s)  for s in sg.get("inputs",  [])]
    outputs = [slot_to_output(s) for s in sg.get("outputs", [])]

    # Default widget values: 0 for INT, empty string for STRING, etc.
    type_defaults = {"INT": 0, "FLOAT": 0.0, "BOOLEAN": False, "STRING": ""}
    widgets_values = [
        type_defaults.get(s["type"], 0) for s in sg.get("inputs", [])
    ]

    return {
        "id": node_id,
        "type": sg["id"],
        "title": sg.get("name", ""),
        "pos": [0, 0],
        "size": [210, max(58, 30 + 20 * max(len(inputs), len(outputs)))],
        "flags": {"collapsed": False},
        "order": 0,
        "mode": 0,
        "inputs": inputs,
        "outputs": outputs,
        "properties": {},
        "widgets_values": widgets_values,
    }


# ── file builder ──────────────────────────────────────────────────────────────

def build_subgraph_file(root_sg, all_deps):
    """Return the dict representing a subgraph .json file."""
    instance = make_instance_node(node_id=1, sg=root_sg)
    return {
        "revision": 0,
        "last_node_id": 1,
        "last_link_id": 0,
        "nodes": [instance],
        "links": [],
        "version": 0.4,
        "definitions": {
            "subgraphs": all_deps,
        },
        "extra": {},
    }


# ── naming helper ─────────────────────────────────────────────────────────────

def collection_filename(outer_sg, sg_map):
    """
    Derive a human-readable filename from the scene names inside the collection.
    We look one level deeper (inner Scenes) to find the actual leaf scene names.
    """
    # Find the inner Scenes subgraph (2nd level deep)
    inner_scene_names = []
    for n in outer_sg.get("nodes", []):
        inner = sg_map.get(n.get("type", ""))
        if inner and inner.get("name") == "Scenes":
            for sn in inner.get("nodes", []):
                name = sg_map.get(sn.get("type", ""), {}).get("name", "")
                if name:
                    inner_scene_names.append(name)

    if not inner_scene_names:
        # Fallback: use the outer UUID prefix
        return f"Scenes_{outer_sg['id'][:8]}"

    # Pick 3 most representative / unique names (filter annotation nodes)
    skip = {"Bjornulf_DisplayNote", "Text Multiline", "Scenes", ""}
    unique = [n for n in inner_scene_names if n not in skip]
    tag = "_".join(unique[:4]).replace(" ", "_").replace("/", "-")
    return f"Scenes_{tag}"


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input",  "-i", required=True, type=Path, help="Source workflow JSON (LTX23 output)")
    p.add_argument("--outdir", "-o", required=True, type=Path, help="Destination directory for .json files")
    p.add_argument("--dry-run", action="store_true", help="Print plan only, do not write files")
    args = p.parse_args()

    with open(args.input) as f:
        workflow = json.load(f)

    sg_list = workflow.get("definitions", {}).get("subgraphs", [])
    sg_map  = {sg["id"]: sg for sg in sg_list}

    # Locate the Select Scene subgraph
    select_sg = next(
        (sg for sg in sg_list if sg.get("name") == "Select Scene"), None
    )
    if select_sg is None:
        print("ERROR: 'Select Scene' subgraph not found.", file=sys.stderr)
        sys.exit(1)

    # Find all direct Scenes children of Select Scene
    collections = []
    for n in select_sg.get("nodes", []):
        child = sg_map.get(n.get("type", ""))
        if child and child.get("name") == "Scenes":
            collections.append(child)

    if not collections:
        print("ERROR: no 'Scenes' children found in Select Scene.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(collections)} Scenes collections in Select Scene.")
    print()

    args.outdir.mkdir(parents=True, exist_ok=True)

    for col in collections:
        filename = collection_filename(col, sg_map) + ".json"
        dest = args.outdir / filename
        all_deps = collect_deps(col["id"], sg_map)

        # Count leaf scenes (subgraphs with new-format nodes)
        leaf_count = sum(
            1 for sg in all_deps
            if any(n.get("type") == "fbt_LoraStackCollect" for n in sg.get("nodes", []))
        )

        print(f"  {col['id'][:8]}..  →  {filename}")
        print(f"    deps: {len(all_deps)} subgraph defs  ({leaf_count} converted scene subgraphs)")

        file_data = build_subgraph_file(col, all_deps)

        if args.dry_run:
            continue

        with open(dest, "w") as f:
            json.dump(file_data, f, indent=2, ensure_ascii=False)
        print(f"    written: {dest}")

    if args.dry_run:
        print("\n(dry-run — no files written)")
    else:
        print(f"\nDone. {len(collections)} files written to {args.outdir}")


if __name__ == "__main__":
    main()
