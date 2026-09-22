#!/usr/bin/env python3
"""Recover a small cast/composition/time metadata tag (see utils/generation_metadata.py,
CAST_SUMMARY_TAG) for clips that have already had their embedded ComfyUI metadata stripped, by
matching their actual audio/video content against an unstripped copy of the same clip still
sitting elsewhere — e.g. ComfyUI's raw output tree — and embedding the recovered summary in place.

This generalizes a one-off recovery done by hand once: a batch of already-cleaned clips in a
project's media folder had lost their original embedded prompt before
nodes/kdenlive_archive.py's --embed-cast-metadata / utils/generation_metadata.py's fbtools_cast
tag existed, but untouched copies of the very same clips were still sitting in ComfyUI's raw
output folders under the same filenames.

Filenames alone are NOT a safe way to match a cleaned clip back to its raw source: two different
clips can share a filename with genuinely different content (ComfyUI's own numbered-suffix naming
recycles across separate generation batches). This script matches by the actual decoded
audio/video stream content instead (independent of container metadata, so a lossless remux still
matches) — see docs/GOTCHAS.md for the story behind this. Only a byte-for-byte content match is
ever used as a source of truth; a same-named-but-different-content candidate is correctly treated
as "no match", not a false positive.

Typical layout this expects (generalized; point --search at your own raw-output folders):

  output/video/<project>/media/<subdir>/clip_00001-audio.mp4   <- already cleaned, no metadata left
  output/video/clip_00001-audio.mp4                              <- untouched original, still has it
  output/video/<other_project>/clip_00001-audio.mp4              <- another place worth searching

Usage:
  python scripts/kdenlive_recover_cast_metadata.py TARGET_DIR \\
      --search output/video --search output/video/<other_project> \\
      --data-dir /path/to/fbtools/user_data \\
      [--project PROJECT.kdenlive] [--organize-by-primary | --organize-by-bundle] \\
      [--dry-run] [--json]

Safety notes:
  - TARGET_DIR is scanned non-recursively; only files it can positively content-match are touched.
  - A clip already carrying embedded metadata or an existing fbtools_cast tag is left alone
    (nothing to recover) — safe to re-run.
  - Writes are atomic (temp file + os.replace onto the final path), and the file's own original
    mtime is preserved, matching utils/kdenlive_clips.py::strip_copy_video's own guarantees.
  - --organize-by-primary/--organize-by-bundle move the clip into a new subfolder of TARGET_DIR
    (named by its recovered primary subject or bundle). Give --project to make this refuse to
    move a clip that project still references by its current path — moving would silently break
    that reference until the project's own XML is updated, which this script never does (see
    scripts/kdenlive_check_resource_usage.py to check that on its own). Override with
    --force-move-referenced only if you will fix the project's XML yourself.
  - --search folders and the project file are only ever read, never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


def _load_module(name: str):
    path = Path(__file__).resolve().parent.parent / "utils" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"fbtools_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stream_hash(path: str) -> str:
    """Content fingerprint of a clip's decoded audio/video samples, independent of container
    metadata or a lossless remux — two files sharing this are the same recording, whatever their
    filenames or embedded tags say. Requires ffmpeg on PATH (or importable via imageio_ffmpeg,
    same lookup as utils/kdenlive_clips.py::find_ffmpeg — kept independent here since this script
    only ever reads with plain "ffmpeg", never writes with it)."""
    h = hashlib.sha256()
    for stream_map, out_args in (
        ("0:v:0", ["-pix_fmt", "yuv420p", "-f", "rawvideo"]),
        ("0:a:0", ["-f", "s16le", "-ar", "44100", "-ac", "2"]),
    ):
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", path, "-map", stream_map, *out_args, "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        h.update(proc.stdout)
    return h.hexdigest()


def _find_content_match(target: str, search_dirs: list[str]) -> str | None:
    """A same-named file directly under one of search_dirs whose content hash matches target, or
    None. Not recursive — pass each folder worth searching explicitly."""
    name = os.path.basename(target)
    target_hash = _stream_hash(target)
    for d in search_dirs:
        cand = os.path.join(d, name)
        if os.path.isfile(cand) and os.path.realpath(cand) != os.path.realpath(target):
            if _stream_hash(cand) == target_hash:
                return cand
    return None


def _destination_subdir(info: dict, mode: str | None) -> str | None:
    if mode == "primary":
        return info.get("primary_subject")
    if mode == "bundle":
        # primary_bundle can be None even with a resolved primary_subject (e.g. the cast entry's
        # subject_id doesn't exactly match the composition slot's subject id) — fall back to
        # whichever bundle actually appears in this clip's cast, so the file still lands somewhere
        # useful instead of getting silently skipped.
        return info.get("primary_bundle") or next(iter(info.get("tags") or []), None)
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target_dir", help="folder of already-cleaned clips missing cast metadata (not recursive)")
    ap.add_argument("--search", action="append", default=[], metavar="DIR",
                     help="folder to search (top level only) for an unstripped copy by filename (repeatable)")
    ap.add_argument("--data-dir", required=True, metavar="DATA_DIR",
                     help="fbTools user-data directory (containing prompt_compositions/), needed to "
                          "resolve a clip's primary subject from the composition it used")
    ap.add_argument("--project", metavar="PROJECT.kdenlive",
                     help="when organizing (--organize-by-*), skip moving a clip this project still "
                          "references at its current path (see scripts/kdenlive_check_resource_usage.py)")
    ap.add_argument("--force-move-referenced", action="store_true",
                     help="move a clip into a subfolder even if --project says it's still referenced "
                          "at its current path — you will need to update the project's own XML yourself")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--organize-by-primary", action="store_const", dest="organize", const="primary",
                       help="nest each recovered clip under target_dir/<primary subject>/")
    mode.add_argument("--organize-by-bundle", action="store_const", dest="organize", const="bundle",
                       help="nest each recovered clip under target_dir/<primary bundle, or its first "
                            "bundle tag if no single bundle is attributable to the primary subject>/")
    ap.add_argument("--dry-run", action="store_true", help="report what would happen; write nothing")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    clips = _load_module("kdenlive_clips")
    gm = _load_module("generation_metadata")
    pc = _load_module("prompt_compositions")
    ka = _load_module("kdenlive_archive") if args.project else None

    def _load_comp(name):
        matched = next((c for c in pc.list_compositions(args.data_dir) if c["name"] == name), None)
        return pc.load_composition(args.data_dir, matched["id"]) if matched else None

    referenced = ka.count_references(args.project) if ka else {}

    results = []
    for fname in sorted(os.listdir(args.target_dir)):
        target = os.path.join(args.target_dir, fname)
        if not os.path.isfile(target) or os.path.splitext(fname)[1].lower() not in clips.VIDEO_EXTENSIONS:
            continue

        if gm.read_embedded_prompt(target) is not None:
            results.append({"file": fname, "status": "skipped",
                             "reason": "already carries its own embedded metadata (nothing to recover)"})
            continue
        if gm.read_cast_summary_tag(target) is not None:
            results.append({"file": fname, "status": "skipped",
                             "reason": "already carries a fbtools_cast tag"})
            continue

        match = _find_content_match(target, args.search)
        if not match:
            results.append({"file": fname, "status": "no_content_match"})
            continue

        graph = gm.read_embedded_prompt(match)
        info = gm.extract_cast_info(graph, load_composition=_load_comp) if graph else None
        if not info:
            results.append({"file": fname, "status": "matched_source_has_no_cast_info", "matched_source": match})
            continue

        subdir = _destination_subdir(info, args.organize)
        if subdir and referenced.get(fname) and not args.force_move_referenced:
            results.append({"file": fname, "status": "kept_flat_still_referenced", "matched_source": match,
                             "info": info, "would_be_subdir": subdir})
            continue

        dest = os.path.join(args.target_dir, subdir, fname) if subdir else target
        entry = {"file": fname, "status": "would_recover" if args.dry_run else "recovered",
                  "matched_source": match, "info": info, "dest": dest}
        if not args.dry_run:
            generated_at = gm.generated_at_iso(target)  # target's own mtime, not "now" — see strip_copy_video
            meta = {"fbtools_cast": gm.build_cast_summary_tag(info, generated_at), "creation_time": generated_at}
            os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
            tmp = dest + ".recovering" + os.path.splitext(dest)[1]
            clips.strip_copy_video(target, tmp, extra_metadata=meta)  # raises + discards tmp on any mismatch
            os.replace(tmp, dest)  # atomic; safe even when dest == target (in-place recovery)
            if dest != target:
                os.remove(target)
        results.append(entry)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for r in results:
            tail = f" -> {r['dest']}" if r.get("dest") and r["dest"] != os.path.join(args.target_dir, r["file"]) \
                else (f" (would move to {r['would_be_subdir']}/ once {r['matched_source']!r} is confirmed unreferenced)"
                      if r.get("would_be_subdir") else "")
            reason = f" — {r['reason']}" if r.get("reason") else ""
            print(f"{r['file']}: {r['status']}{reason}{tail}")
        n_recovered = sum(1 for r in results if r["status"] in ("recovered", "would_recover"))
        print(f"\n{n_recovered} clip(s) {'would be ' if args.dry_run else ''}recovered, "
              f"{len(results) - n_recovered} skipped/unmatched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
