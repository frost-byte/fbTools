#!/usr/bin/env python3
"""
Scan local ComfyUI workflow JSON files for references to one or more custom node extensions.

Two detection signals (a node is flagged if EITHER matches):
  1. Authoritative: its `type` / `class_type` is in the set of node names whose
     python_module contains the extension name, pulled live from a running ComfyUI's
     /object_info endpoint. This catches nodes whose type name does NOT contain the
     extension name (common with most extensions).
  2. Fallback: the node's properties (or the raw file text) contain the extension
     name, which catches the `cnr_id` / `aux_id` stamp modern ComfyUI writes, and
     works even when ComfyUI isn't running.

Usage:
    python scan_node_usage.py -e mixlab [WORKFLOWS_DIR ...]
    python scan_node_usage.py -e comfyui-mixlab-nodes --url http://192.168.1.10:8188
    python scan_node_usage.py -e rgthree -e comfyui-manager [WORKFLOWS_DIR ...]

If no directory is given, it checks common locations or COMFY_WORKFLOWS_DIR env var.
The ComfyUI URL can be set via --url/-u or the COMFY_URL environment variable
(default: http://127.0.0.1:8188).
"""

import argparse
import glob
import json
import os
import sys
import urllib.request

COMFY_URL_DEFAULT = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")

_ENV_DIR = os.environ.get("COMFY_WORKFLOWS_DIR", "").strip()

# Common places saved workflows live — checked in order, first match wins.
DEFAULT_DIRS = [
    *([_ENV_DIR] if _ENV_DIR else []),
    os.path.expanduser("~/ComfyUI/user/default/workflows"),
    os.path.expanduser("~/comfyui/user/default/workflows"),
    os.path.expanduser("~/.comfyui/user/default/workflows"),
    "./ComfyUI/user/default/workflows",
    "./user/default/workflows",
]


def load_extension_types(url, extension_names):
    """Return a dict mapping node_type_name → matched_extension_name."""
    node_to_ext = {}
    try:
        req = urllib.request.Request(url + "/object_info",
                                     headers={"User-Agent": "comfy-scan"})
        with urllib.request.urlopen(req, timeout=10) as r:
            info = json.load(r)
        for node_name, meta in info.items():
            module = (meta.get("python_module") or "").lower()
            for ext in extension_names:
                if ext.lower() in module:
                    node_to_ext[node_name] = ext
                    break
        print(f"[object_info] {len(node_to_ext)} node type(s) matched across "
              f"{len(extension_names)} extension(s) from {url}")
    except Exception as e:
        print(f"[object_info] unavailable ({e}); relying on text match only")
    return node_to_ext


def iter_nodes(data):
    """Yield (type_name, properties_dict) for both workflow formats."""
    # Litegraph "workflow" format: {"nodes": [{"type": ..., "properties": {...}}]}
    if isinstance(data, dict) and isinstance(data.get("nodes"), list):
        for n in data["nodes"]:
            if isinstance(n, dict) and n.get("type"):
                yield n["type"], (n.get("properties") or {})
    # API "prompt" format: {"<id>": {"class_type": ..., "inputs": {...}}}
    elif isinstance(data, dict):
        for v in data.values():
            if isinstance(v, dict) and v.get("class_type"):
                yield v["class_type"], {}


def scan_file(path, node_to_ext, extension_names):
    """Return sorted list of match labels found in one file, or []."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
    except Exception:
        return []

    raw_lower = raw.lower()

    try:
        data = json.loads(raw)
    except Exception:
        # Not valid JSON — fall back to raw text match.
        return sorted(
            f"(raw text match: '{ext}')"
            for ext in extension_names
            if ext.lower() in raw_lower
        )

    found = {}
    for type_name, props in iter_nodes(data):
        if type_name in node_to_ext:
            found[type_name] = node_to_ext[type_name]
        else:
            props_str = json.dumps(props).lower()
            for ext in extension_names:
                if ext.lower() in props_str:
                    found[f"{type_name}  (via properties/cnr_id)"] = ext
                    break

    # Safety net: cnr stamp or module string the node walk missed.
    if not found:
        for ext in extension_names:
            if ext.lower() in raw_lower:
                found[f"(raw text match: '{ext}')"] = ext

    return sorted(found.keys())


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs="*", metavar="WORKFLOWS_DIR",
                        help="Directories to scan (default: auto-detected)")
    parser.add_argument("--extension", "-e", action="append", dest="extensions",
                        metavar="NAME", required=True,
                        help="Extension name to scan for (substring matched against "
                             "python_module). Repeat to scan multiple extensions.")
    parser.add_argument("--url", "-u", default=COMFY_URL_DEFAULT, metavar="URL",
                        help=f"ComfyUI base URL (default: {COMFY_URL_DEFAULT}; "
                             "or set COMFY_URL env var)")
    args = parser.parse_args()

    dirs = args.dirs or [d for d in DEFAULT_DIRS if os.path.isdir(d)]
    if not dirs:
        print("No workflow directory found. Pass one explicitly:")
        print("    python scan_node_usage.py -e mixlab /path/to/workflows")
        sys.exit(1)

    node_to_ext = load_extension_types(args.url, args.extensions)

    files = []
    for d in dirs:
        files.extend(glob.glob(os.path.join(d, "**", "*.json"), recursive=True))
    files = sorted(set(files))

    ext_label = ", ".join(args.extensions)
    print(f"Scanning {len(files)} workflow file(s) across {len(dirs)} dir(s) "
          f"for: {ext_label}\n")

    hits = {}
    for path in files:
        reasons = scan_file(path, node_to_ext, args.extensions)
        if reasons:
            hits[path] = reasons

    for path, reasons in hits.items():
        print(path)
        for r in reasons:
            print(f"    - {r}")

    print(f"\n{len(hits)} of {len(files)} workflow(s) reference {ext_label} nodes.")
    if hits:
        types_seen = sorted({r for rs in hits.values() for r in rs})
        print("\nDistinct node types / matches seen:")
        for t in types_seen:
            print(f"    {t}")


if __name__ == "__main__":
    main()
