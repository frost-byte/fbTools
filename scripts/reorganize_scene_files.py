#!/usr/bin/env python3
"""
Reorganize Scenes subgraph files to match the Scenes_pris.json pattern:

  OLD: outer-Scenes(SwitchCase + inner-Scenes(scene0 … sceneN)) → pipe output
  NEW: root-Scenes(scene0 … sceneN → CreateList) → list (DICT) output

The two-level Scenes wrapper is replaced by a single flat root that feeds all
scene DICTs into a CreateList node whose output becomes the subgraph's `list`.
Individual scene subgraph definitions are kept unchanged.

Usage:
  python scripts/reorganize_scene_files.py \\
      <file.json> [<file.json> ...] \\
      [--dry-run]
"""

import json
import uuid
import argparse
import sys
from pathlib import Path


# ── helpers ───────────────────────────────────────────────────────────────────

def collect_leaf_scenes(sg_list):
    """
    Return only the scene subgraphs (those NOT named 'Scenes' or 'Select Scene').
    These are the individual scene definitions (Doggy, ffuck, Outro, etc.).
    """
    return [sg for sg in sg_list if sg.get("name") not in ("Scenes", "Select Scene")]


def build_root_subgraph(name, scene_sgs):
    """
    Build a new root Scenes subgraph that:
      - Has no inputs
      - Instantiates each scene subgraph
      - Feeds all scene DICT outputs into CreateList
      - Outputs the resulting list as 'list' (DICT type)
    """
    n_scenes = len(scene_sgs)

    # ── node IDs ──────────────────────────────────────────────────────────────
    # scene instances: 1 … n_scenes
    # CreateList:      n_scenes + 1
    NID_CREATE_LIST = n_scenes + 1

    # ── link IDs ─────────────────────────────────────────────────────────────
    # scene_i output → CreateList input_i  : 1 … n_scenes
    # CreateList output → -20              : n_scenes + 1
    LID_LIST_OUT = n_scenes + 1

    # ── layout ────────────────────────────────────────────────────────────────
    SCENE_W, SCENE_H = 210, 58
    SCENE_GAP_X = 30
    CL_W, CL_H = 140, 26 + 20 * (n_scenes + 1)
    CL_X = (SCENE_W + SCENE_GAP_X) * n_scenes + SCENE_GAP_X

    # ── scene instance nodes ──────────────────────────────────────────────────
    scene_nodes = []
    scene_links = []
    for i, sg in enumerate(scene_sgs):
        node_id = i + 1
        link_id = i + 1
        x = (SCENE_W + SCENE_GAP_X) * i
        scene_nodes.append({
            "id": node_id,
            "type": sg["id"],
            "pos": [x, 0],
            "size": [SCENE_W, SCENE_H],
            "flags": {},
            "order": i,
            "mode": 0,
            "inputs": [],
            "outputs": [{"name": "DICT", "type": "DICT", "links": [link_id]}],
            "properties": {},
        })
        # First slot has no shape; the rest get shape:7 (matching pris)
        inp_entry = {
            "label": f"input{i}",
            "localized_name": f"inputs.input{i}",
            "name": f"inputs.input{i}",
            "type": "*",
            "link": link_id,
        }
        if i > 0:
            inp_entry["shape"] = 7
        scene_links.append({
            "id": link_id,
            "origin_id": node_id,
            "origin_slot": 0,
            "target_id": NID_CREATE_LIST,
            "target_slot": i,
            "type": "DICT",
        })

    # ── CreateList node ───────────────────────────────────────────────────────
    cl_inputs = []
    for i in range(n_scenes):
        entry = {
            "label": f"input{i}",
            "localized_name": f"inputs.input{i}",
            "name": f"inputs.input{i}",
            "type": "*",
            "link": i + 1,
        }
        if i > 0:
            entry["shape"] = 7
        cl_inputs.append(entry)
    # One trailing empty slot (matching pris)
    cl_inputs.append({
        "label": f"input{n_scenes}",
        "localized_name": f"inputs.input{n_scenes}",
        "name": f"inputs.input{n_scenes}",
        "shape": 7,
        "type": "*",
        "link": None,
    })

    create_list_node = {
        "id": NID_CREATE_LIST,
        "type": "CreateList",
        "pos": [CL_X, 0],
        "size": [CL_W, CL_H],
        "flags": {},
        "order": n_scenes,
        "mode": 0,
        "inputs": cl_inputs,
        "outputs": [{
            "localized_name": "list",
            "name": "list",
            "shape": 6,
            "type": "*",
            "links": [LID_LIST_OUT],
        }],
        "properties": {
            "cnr_id": "comfy-core",
            "Node name for S&R": "CreateList",
        },
    }

    # ── CreateList → subgraph output link ─────────────────────────────────────
    out_link = {
        "id": LID_LIST_OUT,
        "origin_id": NID_CREATE_LIST,
        "origin_slot": 0,
        "target_id": -20,
        "target_slot": 0,
        "type": "DICT",
    }

    all_links = scene_links + [out_link]
    out_slot_uuid = str(uuid.uuid4())
    out_x = CL_X + CL_W + 60

    return {
        "id": str(uuid.uuid4()),
        "version": 1,
        "state": {
            "lastGroupId": 0,
            "lastNodeId": NID_CREATE_LIST,
            "lastLinkId": LID_LIST_OUT,
            "lastRerouteId": 0,
        },
        "revision": 0,
        "config": {},
        "name": name,
        "inputNode":  {"id": -10, "bounding": [-150, -20, 120, 80]},
        "outputNode": {"id": -20, "bounding": [out_x, 0, 128, 68]},
        "inputs": [],
        "outputs": [{
            "id": out_slot_uuid,
            "name": "list",
            "type": "DICT",
            "linkIds": [LID_LIST_OUT],
            "pos": [out_x + 24, 34],
        }],
        "widgets": [],
        "nodes": scene_nodes + [create_list_node],
        "groups": [],
        "links": all_links,
        "extra": {"ue_links": [], "links_added_by_ue": [], "workflowRendererVersion": "LG"},
    }


def build_instance_node(root_sg, title):
    """Build the top-level canvas node that instances the root subgraph."""
    return {
        "id": 1,
        "type": root_sg["id"],
        "title": title,
        "pos": [0, 0],
        "size": [372, 100],
        "flags": {"collapsed": False},
        "order": 0,
        "mode": 0,
        "inputs": [],
        "outputs": [{"name": "list", "type": "DICT", "links": []}],
        "properties": {
            "cnr_id": "comfy-core",
            "previewExposures": [],
        },
    }


def reorganize_file(src_path: Path, dry_run: bool = False):
    with open(src_path) as f:
        data = json.load(f)

    sg_list = data.get("definitions", {}).get("subgraphs", [])
    leaf_scenes = collect_leaf_scenes(sg_list)

    # Use the file stem as the collection name
    collection_name = src_path.stem

    print(f"  {src_path.name}")
    print(f"    leaf scenes ({len(leaf_scenes)}): {[sg['name'] for sg in leaf_scenes]}")

    root_sg   = build_root_subgraph(collection_name, leaf_scenes)
    instance  = build_instance_node(root_sg, collection_name)

    new_data = {
        "revision": 0,
        "last_node_id": 1,
        "last_link_id": 0,
        "nodes": [instance],
        "links": [],
        "version": 0.4,
        "definitions": {
            "subgraphs": [root_sg] + leaf_scenes,
        },
        "extra": {},
    }

    if dry_run:
        print(f"    (dry-run) would write {src_path}")
        return

    with open(src_path, "w") as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)
    print(f"    written: {src_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("files", nargs="+", type=Path, help="Scenes .json files to reorganize")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    for f in args.files:
        if not f.exists():
            print(f"SKIP (not found): {f}", file=sys.stderr)
            continue
        reorganize_file(f, dry_run=args.dry_run)

    if args.dry_run:
        print("\n(dry-run — no files written)")


if __name__ == "__main__":
    main()
