#!/usr/bin/env python3
"""
Transform old-style Wan 2.2 scene subgraphs into new LTX-2.3 format.

Source scene subgraph contains:
  - Pipe In 8CH [RvTools]  (slot 1=prompt, slot 5=h_lora_stack, slot 6=l_lora_stack, slot 7=name)
  - StringConstantMultiline  (prompt text connected to Pipe In slot 1)
  - easy loraStack  (H and L stacks connected to Pipe In slots 5/6)
  - easy string  (scene name)

Target scene subgraph contains:
  - PrimitiveStringMultiline  (prompt)
  - fbt_LoraEntryDefine  (one per unique LoRA, title = "LED <name>", strengths 1.0, target LTX2.3)
  - fbt_LoraStackCollect  (aggregates LED outputs)
  - easy string  (scene name)
  - Basic data handling: DictCreate  (outputs {prompt, stack, name})

LoRA names remain as Wan 2.2 placeholders — substitute manually for LTX-2.3 equivalents.

Usage:
  python scripts/transform_scene_subgraphs.py \\
      --input  /path/to/source_workflow.json \\
      --output /path/to/output_workflow.json \\
      [--dry-run]
"""

import json
import uuid
import argparse
import sys
from pathlib import Path


# ── layout constants ──────────────────────────────────────────────────────────
NODE_W_PROMPT  = 473
NODE_H_PROMPT  = 160
NODE_W_LED     = 270
NODE_H_LED     = 290
NODE_W_COLLECT = 270
NODE_H_COLLECT = 158
NODE_W_NAME    = 270
NODE_H_NAME    = 80
NODE_W_DICT    = 270
NODE_H_DICT    = 226

GAP_X = 40
GAP_Y = 60
ORIGIN_X = 0
ORIGIN_Y = 0


# ── extraction helpers ────────────────────────────────────────────────────────

def extract_loras_from_stack(node):
    """Return list of lora names (non-None) for the first num_loras slots."""
    wv = node.get("widgets_values", [])
    if len(wv) < 3:
        return []
    num_loras = int(wv[2])
    loras = []
    for i in range(num_loras):
        base = 3 + i * 4
        if base >= len(wv):
            break
        name = wv[base]
        if name and name != "None":
            loras.append(name)
    return loras


def parse_scene_subgraph(sg):
    """
    Extract (scene_name, prompt, lora_list) from an old-style scene subgraph.

    Returns None if the subgraph doesn't match the expected pattern.
    """
    nodes = sg.get("nodes", [])
    links = sg.get("links", [])

    nmap = {n["id"]: n for n in nodes}

    # Build: target_id → {target_slot: (origin_id, origin_slot)}
    incoming = {}
    for lnk in links:
        tid = lnk["target_id"]
        slot = lnk["target_slot"]
        incoming.setdefault(tid, {})[slot] = (lnk["origin_id"], lnk["origin_slot"])

    # Find Pipe In node
    pipe_in = next(
        (n for n in nodes if n.get("type") == "Pipe In 8CH [RvTools]"), None
    )
    if pipe_in is None:
        return None

    pid = pipe_in["id"]
    slots = incoming.get(pid, {})

    # ── Prompt (slot 1) ──────────────────────────────────────────────────────
    prompt = ""
    if 1 in slots:
        src_id = slots[1][0]
        if src_id in nmap:
            wv = nmap[src_id].get("widgets_values", [])
            prompt = str(wv[0]) if wv else ""

    # ── LoRAs from connected stacks (slots 5 and 6) ──────────────────────────
    raw_loras = []
    for slot in (5, 6):
        if slot in slots:
            src_id = slots[slot][0]
            if src_id in nmap:
                raw_loras.extend(extract_loras_from_stack(nmap[src_id]))

    # Deduplicate preserving order
    seen = set()
    loras = []
    for l in raw_loras:
        if l not in seen:
            seen.add(l)
            loras.append(l)

    # ── Scene name (easy string node) ────────────────────────────────────────
    name_node = next(
        (n for n in nodes if n.get("type") == "easy string"), None
    )
    scene_name = ""
    if name_node:
        wv = name_node.get("widgets_values", [])
        scene_name = str(wv[0]) if wv else ""

    return scene_name, prompt, loras


# ── subgraph builder ──────────────────────────────────────────────────────────

def build_new_scene_subgraph(sg_id, sg_name, scene_name, prompt, loras, old_outputs):
    """
    Construct a replacement subgraph definition in the new LTX-2.3 format.

    old_outputs is the original outputs list so we can preserve slot UUIDs
    (the parent linkIds already reference them; we only change the type).
    """
    n_loras = len(loras)

    # ── assign local node IDs ────────────────────────────────────────────────
    NID_PROMPT  = 1
    NID_COLLECT = 2
    NID_NAME    = 3
    NID_DICT    = 4
    # LED nodes: 5 … 4+n_loras
    NID_LED_BASE = 5

    # ── assign local link IDs ────────────────────────────────────────────────
    # LED_i output → COLLECT input i  :  link 100+i
    # COLLECT slot 0 → DICT value_1   :  link 200
    # PROMPT output  → DICT value_0   :  link 201
    # NAME output    → DICT value_2   :  link 202
    # DICT output    → subgraph out   :  link 203
    LID_LED_BASE     = 100
    LID_COLLECT_DICT = 200
    LID_PROMPT_DICT  = 201
    LID_NAME_DICT    = 202
    LID_DICT_OUT     = 203

    # ── layout positions ─────────────────────────────────────────────────────
    col_led_start = NODE_W_PROMPT + GAP_X
    led_total_w   = max(n_loras, 1) * (NODE_W_LED + GAP_X)

    row0_y = ORIGIN_Y
    row1_y = row0_y + NODE_H_LED + GAP_Y
    row2_y = row1_y + NODE_H_COLLECT + GAP_Y  # noqa: F841 (reserved for future use)

    collect_x = col_led_start + (led_total_w - NODE_W_COLLECT) / 2
    dict_x    = collect_x + NODE_W_COLLECT + GAP_X

    # ── PrimitiveStringMultiline ─────────────────────────────────────────────
    prompt_node = {
        "id": NID_PROMPT,
        "type": "PrimitiveStringMultiline",
        "title": "Prompt",
        "pos": [ORIGIN_X, row0_y],
        "size": [NODE_W_PROMPT, NODE_H_PROMPT],
        "flags": {"collapsed": False},
        "order": 0,
        "mode": 0,
        "inputs": [
            {
                "localized_name": "value",
                "name": "value",
                "type": "STRING",
                "widget": {"name": "value"},
                "link": None,
            }
        ],
        "outputs": [
            {
                "localized_name": "STRING",
                "name": "STRING",
                "type": "STRING",
                "links": [LID_PROMPT_DICT],
            }
        ],
        "properties": {
            "cnr_id": "comfy-core",
            "ver": "0.5.1",
            "Node name for S&R": "PrimitiveStringMultiline",
        },
        "widgets_values": [prompt],
    }

    # ── fbt_LoraEntryDefine nodes (one per LoRA) ─────────────────────────────
    led_nodes = []
    led_links = []
    for i, lora_name in enumerate(loras):
        node_id  = NID_LED_BASE + i
        link_out = LID_LED_BASE + i
        led_nodes.append(
            {
                "id": node_id,
                "type": "fbt_LoraEntryDefine",
                "title": f"LED {lora_name}",
                "pos": [col_led_start + i * (NODE_W_LED + GAP_X), row0_y],
                "size": [NODE_W_LED, NODE_H_LED],
                "flags": {"collapsed": True},
                "order": node_id,
                "mode": 0,
                "inputs": [
                    {"localized_name": "lora",           "name": "lora",           "type": "COMBO",   "widget": {"name": "lora"},           "link": None},
                    {"localized_name": "model_target",   "name": "model_target",   "type": "COMBO",   "widget": {"name": "model_target"},   "link": None},
                    {"localized_name": "strength_model", "name": "strength_model", "type": "FLOAT",   "widget": {"name": "strength_model"}, "link": None},
                    {"localized_name": "strength_clip",  "name": "strength_clip",  "type": "FLOAT",   "widget": {"name": "strength_clip"},  "link": None},
                    {"localized_name": "enabled",        "name": "enabled",        "type": "BOOLEAN", "widget": {"name": "enabled"},        "link": None},
                    {"localized_name": "video",          "name": "video",          "type": "FLOAT",   "widget": {"name": "video"},          "link": None},
                    {"localized_name": "video_to_audio", "name": "video_to_audio", "type": "FLOAT",   "widget": {"name": "video_to_audio"}, "link": None},
                    {"localized_name": "audio",          "name": "audio",          "type": "FLOAT",   "widget": {"name": "audio"},          "link": None},
                    {"localized_name": "audio_to_video", "name": "audio_to_video", "type": "FLOAT",   "widget": {"name": "audio_to_video"}, "link": None},
                    {"localized_name": "other",          "name": "other",          "type": "FLOAT",   "widget": {"name": "other"},          "link": None},
                ],
                "outputs": [
                    {
                        "localized_name": "LoRA Entry",
                        "name": "LoRA Entry",
                        "type": "LORA_ENTRY",
                        "links": [link_out],
                    }
                ],
                "properties": {
                    "aux_id": "frost-byte/fbTools",
                    "Node name for S&R": "fbt_LoraEntryDefine",
                },
                "widgets_values": [lora_name, "LTX2.3", 1.0, 1.0, True, 1, 1, 0, 0, 1, None],
            }
        )
        led_links.append(
            {
                "id": link_out,
                "origin_id": node_id,
                "origin_slot": 0,
                "target_id": NID_COLLECT,
                "target_slot": i,
                "type": "LORA_ENTRY",
            }
        )

    # ── fbt_LoraStackCollect ─────────────────────────────────────────────────
    collect_inputs = []
    for i in range(n_loras):
        collect_inputs.append(
            {
                "label": f"entry{i}",
                "localized_name": f"entries.entry{i}",
                "name": f"entries.entry{i}",
                "shape": 7,
                "type": "LORA_ENTRY",
                "link": LID_LED_BASE + i,
            }
        )
    collect_inputs.append(
        {
            "label": "Prev Stack",
            "localized_name": "prev_stack",
            "name": "prev_stack",
            "shape": 7,
            "type": "LORA_STACK_DATA",
            "link": None,
        }
    )
    collect_inputs.append(
        {
            "localized_name": "existing_json",
            "name": "existing_json",
            "shape": 7,
            "type": "STRING",
            "widget": {"name": "existing_json"},
            "link": None,
        }
    )

    collect_node = {
        "id": NID_COLLECT,
        "type": "fbt_LoraStackCollect",
        "title": "",
        "pos": [collect_x, row1_y],
        "size": [NODE_W_COLLECT, NODE_H_COLLECT],
        "flags": {},
        "order": NID_COLLECT,
        "mode": 0,
        "inputs": collect_inputs,
        "outputs": [
            {"localized_name": "Stack Data",  "name": "Stack Data",  "type": "LORA_STACK_DATA", "links": [LID_COLLECT_DICT]},
            {"localized_name": "Stack JSON",  "name": "Stack JSON",  "type": "STRING",           "links": []},
            {"localized_name": "Entry Count", "name": "Entry Count", "type": "INT",              "links": None},
        ],
        "properties": {
            "aux_id": "frost-byte/fbTools",
            "Node name for S&R": "fbt_LoraStackCollect",
        },
        "widgets_values": ["[]"],
    }

    # ── easy string (scene name) ─────────────────────────────────────────────
    name_node = {
        "id": NID_NAME,
        "type": "easy string",
        "title": "Name",
        "pos": [ORIGIN_X, row1_y],
        "size": [NODE_W_NAME, NODE_H_NAME],
        "flags": {},
        "order": NID_NAME,
        "mode": 0,
        "inputs": [
            {
                "localized_name": "value",
                "name": "value",
                "type": "STRING",
                "widget": {"name": "value"},
                "link": None,
            }
        ],
        "outputs": [
            {"localized_name": "string", "name": "string", "type": "STRING", "links": [LID_NAME_DICT]}
        ],
        "properties": {},
        "widgets_values": [scene_name],
    }

    # ── DictCreate ───────────────────────────────────────────────────────────
    dict_node = {
        "id": NID_DICT,
        "type": "Basic data handling: DictCreate",
        "title": "",
        "pos": [dict_x, row1_y],
        "size": [NODE_W_DICT, NODE_H_DICT],
        "flags": {},
        "order": NID_DICT,
        "mode": 0,
        "inputs": [
            {"localized_name": "key_0",   "name": "key_0",   "shape": 7, "type": "STRING", "widget": {"name": "key_0"},   "link": None},
            {"localized_name": "value_0", "name": "value_0", "shape": 7, "type": "*",      "widget": {"name": "value_0"}, "link": LID_PROMPT_DICT},
            {"name": "key_1",   "shape": 7, "type": "STRING", "widget": {"name": "key_1"},   "link": None},
            {"name": "value_1", "shape": 7, "type": "*",      "widget": {"name": "value_1"}, "link": LID_COLLECT_DICT},
            {"name": "key_2",   "shape": 7, "type": "STRING", "widget": {"name": "key_2"},   "link": None},
            {"name": "value_2", "shape": 7, "type": "*",      "widget": {"name": "value_2"}, "link": LID_NAME_DICT},
            {"name": "key_3",   "shape": 7, "type": "STRING", "widget": {"name": "key_3"},   "link": None},
            {"name": "value_3", "shape": 7, "type": "*",      "widget": {"name": "value_3"}, "link": None},
        ],
        "outputs": [
            {"localized_name": "DICT", "name": "DICT", "type": "DICT", "links": [LID_DICT_OUT]}
        ],
        "properties": {
            "cnr_id": "basic_data_handling",
            "ver": "1.5.0",
            "Node name for S&R": "Basic data handling: DictCreate",
        },
        "widgets_values": ["prompt", "", "stack", "", "name", scene_name, "", ""],
    }

    # ── internal links ───────────────────────────────────────────────────────
    internal_links = led_links + [
        {"id": LID_PROMPT_DICT,  "origin_id": NID_PROMPT,  "origin_slot": 0, "target_id": NID_DICT,    "target_slot": 1, "type": "STRING"},
        {"id": LID_COLLECT_DICT, "origin_id": NID_COLLECT, "origin_slot": 0, "target_id": NID_DICT,    "target_slot": 3, "type": "LORA_STACK_DATA"},
        {"id": LID_NAME_DICT,    "origin_id": NID_NAME,    "origin_slot": 0, "target_id": NID_DICT,    "target_slot": 5, "type": "STRING"},
        {"id": LID_DICT_OUT,     "origin_id": NID_DICT,    "origin_slot": 0, "target_id": -20,          "target_slot": 0, "type": "DICT"},
    ]

    # ── output slot: preserve UUID so parent linkIds stay valid, update type ─
    # linkIds within the subgraph output slot must reference the link that
    # connects to -20 (the subgraph output node), which is LID_DICT_OUT.
    # NOTE: The consuming side (parent Scenes subgraph / Select Scene) must be
    # re-wired to accept DICT instead of pipe — that is out of scope for this
    # script.
    out_x = dict_x + NODE_W_DICT + GAP_X
    out_y = row1_y
    if old_outputs:
        new_outputs = []
        for slot in old_outputs:
            new_slot = dict(slot)
            new_slot["name"] = "DICT"
            new_slot["type"] = "DICT"
            new_slot["linkIds"] = [LID_DICT_OUT]
            new_outputs.append(new_slot)
    else:
        new_outputs = [
            {
                "id": str(uuid.uuid4()),
                "name": "DICT",
                "type": "DICT",
                "linkIds": [LID_DICT_OUT],
                "pos": [out_x, out_y],
            }
        ]

    all_nodes = [prompt_node] + led_nodes + [collect_node, name_node, dict_node]
    max_nid = max(n["id"] for n in all_nodes)
    max_lid = max(lnk["id"] for lnk in internal_links)

    return {
        "id": sg_id,
        "version": 1,
        "state": {
            "lastGroupId": 0,
            "lastNodeId": max_nid,
            "lastLinkId": max_lid,
            "lastRerouteId": 0,
        },
        "revision": 0,
        "config": {},
        "name": sg_name,
        "inputNode":  {"id": -10, "bounding": [ORIGIN_X - 150, ORIGIN_Y - 20, 120, 80]},
        "outputNode": {"id": -20, "bounding": [out_x, out_y, 120, 60]},
        "inputs": [],
        "outputs": new_outputs,
        "widgets": [],
        "nodes": all_nodes,
        "groups": [],
        "links": internal_links,
        "extra": {"ue_links": [], "links_added_by_ue": [], "workflowRendererVersion": "LG"},
    }


# ── main transform ────────────────────────────────────────────────────────────

def is_leaf_scene(sg):
    """True if this subgraph contains the old-style scene structure."""
    types = {n.get("type", "") for n in sg.get("nodes", [])}
    return "easy loraStack" in types


def transform(src_path: Path, dst_path: Path, dry_run: bool = False):
    with open(src_path) as f:
        workflow = json.load(f)

    subgraphs = workflow.get("definitions", {}).get("subgraphs", [])
    if not subgraphs:
        print("ERROR: no subgraph definitions found.", file=sys.stderr)
        sys.exit(1)

    transformed = 0
    skipped = 0
    new_subgraphs = []

    for sg in subgraphs:
        if not is_leaf_scene(sg):
            new_subgraphs.append(sg)
            skipped += 1
            continue

        result = parse_scene_subgraph(sg)
        if result is None:
            print(f"  SKIP {sg['id']} ({sg['name']}): could not parse", file=sys.stderr)
            new_subgraphs.append(sg)
            skipped += 1
            continue

        scene_name, prompt, loras = result
        print(
            f"  TRANSFORM {sg['id']} ({sg['name']!r})"
            f"  scene={scene_name.strip()!r}  loras={len(loras)}  prompt_len={len(prompt)}"
        )
        for lora in loras:
            print(f"    - {lora}")

        new_sg = build_new_scene_subgraph(
            sg_id=sg["id"],
            sg_name=sg["name"],
            scene_name=scene_name,
            prompt=prompt,
            loras=loras,
            old_outputs=sg.get("outputs", []),
        )
        new_subgraphs.append(new_sg)
        transformed += 1

    workflow["definitions"]["subgraphs"] = new_subgraphs

    print(f"\nTransformed: {transformed}  Skipped (non-scene): {skipped}")

    if dry_run:
        print("(dry-run — no file written)")
        return

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_path, "w") as f:
        json.dump(workflow, f, indent=2, ensure_ascii=False)
    print(f"Written: {dst_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input",  "-i", required=True, type=Path, help="Source workflow JSON")
    p.add_argument("--output", "-o", required=True, type=Path, help="Destination workflow JSON")
    p.add_argument("--dry-run", action="store_true", help="Parse and report only, do not write")
    args = p.parse_args()

    print(f"Source:  {args.input}")
    print(f"Output:  {args.output}")
    print()
    transform(args.input, args.output, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
