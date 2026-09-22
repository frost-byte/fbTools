#!/usr/bin/env python3
"""Report whether files are referenced anywhere in a .kdenlive project, by filename — a quick
safety check before moving, renaming or deleting a clip that might already be placed in the
project's own bin/timeline. Any of those operations would silently break the project (it stores
plain file paths, not stable ids) unless its XML is updated to match — which this script does not
do; it only tells you whether that risk applies to a given file.

Matching is by basename against every `resource` / `warp_resource` / `kdenlive:originalurl`
property in the project, via utils/kdenlive_archive.py::count_references() — see that function's
docstring for what it does and doesn't resolve (no path maps or search dirs; pure text/XML, no
disk access beyond reading the project itself).

Typical layout (generalized; use your own project/media paths):

  output/video/<project>/<project>.kdenlive
  output/video/<project>/media/<subfolder>/clip_00001-audio.mp4

Usage:
  # Check every video/audio file directly under a folder:
  python scripts/kdenlive_check_resource_usage.py PROJECT.kdenlive --dir media/comps

  # Check specific filenames:
  python scripts/kdenlive_check_resource_usage.py PROJECT.kdenlive clip_00001-audio.mp4 clip_00002-audio.mp4

Exit code: 0 if every checked file is referenced at least once, 1 if any is unreferenced (so this
can gate a following step, e.g. "only move files this prints as unreferenced").
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path


def _load_module(name: str):
    path = Path(__file__).resolve().parent.parent / "utils" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"fbtools_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project", help="path to the .kdenlive project")
    ap.add_argument("files", nargs="*", help="specific filenames to check (basenames, not full paths)")
    ap.add_argument("--dir", metavar="DIR",
                     help="check every video/audio file directly under this folder instead of (or in "
                          "addition to) explicit filenames")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if not args.files and not args.dir:
        ap.error("give one or more filenames, or --dir")

    ka = _load_module("kdenlive_archive")
    clips = _load_module("kdenlive_clips")

    names = list(dict.fromkeys(args.files))  # dedupe, keep order
    if args.dir:
        for f in sorted(os.listdir(args.dir)):
            if os.path.isfile(os.path.join(args.dir, f)) and os.path.splitext(f)[1].lower() in clips.VIDEO_EXTENSIONS:
                if f not in names:
                    names.append(f)

    counts = ka.count_references(args.project)
    rows = [{"file": n, "references": counts.get(n, 0)} for n in names]
    unreferenced = [r["file"] for r in rows if r["references"] == 0]

    if args.json:
        print(json.dumps({"project": args.project, "rows": rows, "unreferenced": unreferenced}, indent=2))
    else:
        width = max((len(r["file"]) for r in rows), default=0)
        for r in rows:
            print(f"{r['references']:>3}  {r['file']:<{width}}")
        print(f"\n{len(unreferenced)} of {len(rows)} file(s) are not referenced anywhere in the project.")
        if unreferenced:
            print("(0-reference files are the only ones safe to move/rename without touching the project's own XML.)")

    return 1 if unreferenced else 0


if __name__ == "__main__":
    sys.exit(main())
