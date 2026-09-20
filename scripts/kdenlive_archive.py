#!/usr/bin/env python3
"""Archive a Kdenlive project into a portable, self-contained folder.

Subcommands:
  check    Resolve every media reference and report what is missing (writes nothing)
  archive  Copy clips into DEST/media/<source_folder>/ and write a project with
           relative paths (root="") that opens on Windows, macOS and Linux
  strip    Remove the embedded ComfyUI workflow/prompt JSON that Kdenlive copies
           from clip metadata into the project (often 95%+ of the file size)

Missing clips: use --map to translate paths from another machine (e.g. a
Windows mapped drive) and --search to look for moved clips by filename. Clips
that still can't be found are left untouched and reported (exit code 2).

Examples:
  python scripts/kdenlive_archive.py check proj.kdenlive --map "Z:/=/mnt/comfy_ssd/ComfyUI/"
  python scripts/kdenlive_archive.py archive proj.kdenlive /data/proj_archive \\
      --map "Z:/=/mnt/comfy_ssd/ComfyUI/" --search /mnt/comfy_ssd/ComfyUI/output/video
  python scripts/kdenlive_archive.py strip proj.kdenlive --in-place

Exit codes: 0 ok, 2 finished with unresolved clips (or cancelled), 1 error.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def _load_lib():
    path = Path(__file__).resolve().parent.parent / "utils" / "kdenlive_archive.py"
    spec = importlib.util.spec_from_file_location("fbtools_kdenlive_archive", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} GB"


def _progress_printer(quiet: bool):
    state = {"last": None}

    def cb(ev):
        if quiet:
            return
        phase = ev.get("phase")
        if phase == "copy":
            line = f"  copying {ev['done']}/{ev['total']}  {_fmt_bytes(ev.get('bytes_done', 0))}"
        elif phase == "resolve" and ev.get("done") == ev.get("total"):
            line = f"  resolved {ev['done']}/{ev['total']} references"
        else:
            return
        if line != state["last"]:
            print(line, file=sys.stderr, flush=True)
            state["last"] = line

    return cb


def _print_report(rep: dict) -> None:
    print(f"Project:        {rep['project']}")
    print(f"References:     {rep['references']} ({rep['resolved']} resolved, {rep['unresolved_count']} unresolved)")
    print(f"Unique files:   {rep['unique_files']} ({_fmt_bytes(rep['total_bytes'])})")
    if rep["by_method"]:
        print("Resolved via:   " + ", ".join(f"{k}={v}" for k, v in sorted(rep["by_method"].items())))
    if "output_project" in rep:
        verb = "Would write" if rep["dry_run"] else "Wrote"
        print(f"{verb}:      {rep['output_project']}")
        if not rep["dry_run"]:
            print(f"Copied:         {rep['copied']} files ({_fmt_bytes(rep['bytes_copied'])}), "
                  f"{rep['skipped_existing']} already present")
        st = rep["stripped"]
        if st["removed"]:
            print(f"Metadata:       stripped {st['removed']} properties, "
                  f"project {_fmt_bytes(rep['size_before'])} -> {_fmt_bytes(rep['size_after'])}")
        if rep.get("cancelled"):
            print("CANCELLED before writing the project file.")
    for a in rep["ambiguous"]:
        print(f"AMBIGUOUS: {a['reference']} -> chose {a['chosen']} (of {len(a['candidates'])} candidates)")
    for u in rep["unresolved"]:
        print(f"MISSING:   {u}")
    if rep["unresolved_count"] > len(rep["unresolved"]):
        print(f"... and {rep['unresolved_count'] - len(rep['unresolved'])} more unresolved")
    for w in rep["warnings"]:
        print(f"WARNING: {w}")
    for n in rep["notes"]:
        print(f"note: {n}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("project", help="path to the .kdenlive project")
        p.add_argument("--map", action="append", default=[], metavar="SRC=DST",
                       help="translate a path prefix, e.g. 'Z:/=/mnt/data/' (repeatable)")
        p.add_argument("--search", action="append", default=[], metavar="DIR",
                       help="folder to search by filename for clips not found at their recorded path (repeatable)")
        p.add_argument("--json", action="store_true", help="print the report as JSON")

    p_check = sub.add_parser("check", help="resolve references and report; writes nothing")
    common(p_check)
    p_arch = sub.add_parser("archive", help="build a portable archive")
    common(p_arch)
    p_arch.add_argument("dest", help="destination folder (created if needed)")
    p_arch.add_argument("--no-strip", action="store_true", help="keep embedded workflow metadata")
    p_arch.add_argument("--dry-run", action="store_true", help="resolve and report, copy nothing")
    p_arch.add_argument("--name", help="output project filename (default <stem>_ARCHIVE.kdenlive)")
    p_strip = sub.add_parser("strip", help="strip embedded ComfyUI metadata from a project")
    p_strip.add_argument("project")
    p_strip.add_argument("--in-place", action="store_true", help="edit the project itself (keeps a .bak)")
    p_strip.add_argument("--output", help="write to this file instead of <stem>_stripped.kdenlive")
    p_strip.add_argument("--json", action="store_true")

    args = ap.parse_args(argv)
    lib = _load_lib()
    try:
        if args.cmd == "strip":
            out = args.project if args.in_place else args.output
            rep = lib.strip_metadata(args.project, output=out)
            if args.json:
                print(json.dumps(rep, indent=2))
            else:
                print(f"Removed {rep['removed']} properties: {_fmt_bytes(rep['size_before'])} -> "
                      f"{_fmt_bytes(rep['size_after'])}\nWrote {rep['output']}")
            return 0
        progress = _progress_printer(args.json)
        if args.cmd == "check":
            rep = lib.analyze(args.project, args.map, args.search, progress=progress)
        else:
            rep = lib.archive(args.project, args.dest, path_maps=args.map, search_dirs=args.search,
                              strip_metadata_opt=not args.no_strip, dry_run=args.dry_run,
                              output_name=args.name, progress=progress)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(rep, indent=2))
    else:
        _print_report(rep)
    return 2 if rep["unresolved_count"] or rep.get("cancelled") else 0


if __name__ == "__main__":
    sys.exit(main())
