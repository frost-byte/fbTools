"""Kdenlive project archiver: make a .kdenlive project self-contained and portable.

Pure stdlib, no ComfyUI dependencies (and no imports from sibling utils
modules — see docs/GOTCHAS.md). Used by scripts/kdenlive_archive.py (CLI) and
nodes/kdenlive_archive.py (ComfyUI sidebar tab routes).

Operations (all return JSON-serialisable dicts):
  analyze()         resolve every media reference; writes nothing
  archive()         copy referenced media into <dest>/media/<source_folder>/ and
                    write a copy of the project with relative paths (root="")
  strip_metadata()  remove the huge ComfyUI workflow/prompt JSON that Kdenlive
                    copies from each clip's metadata into the project file

A reference is resolved, in order, by: relative to the project root/folder,
a user path map (e.g. "Z:/=/mnt/data/"), the path as-is, then a search of
extra folders by filename (ties broken by matching trailing path components,
then by the clip's recorded kdenlive:file_size). Unresolved references are
left untouched and reported; the source project is never modified.
"""
from __future__ import annotations

import html
import os
import re
import shutil
import xml.parsers.expat
from collections import defaultdict
from xml.sax.saxutils import escape as _xml_escape

_PROP_RE = re.compile(r'<property name="(resource|warp_resource|kdenlive:originalurl)">([^<]*)</property>')
_BLOCK_RE = re.compile(r"<(chain|producer)\b[^>]*>(.*?)</\1>", re.S)
_SIZE_RE = re.compile(r'name="kdenlive:file_size">(\d+)<')
_SPEED_RE = re.compile(r"^(-?\d+:)(.*)$")
_ROOT_RE = re.compile(r'(<mlt\b[^>]*\broot=")([^"]*)(")')
_STRIP_RE = re.compile(r'[ \t]*<property name="(meta\.attr\.[^"]+\.markup)">([^<]*)</property>[ \t]*\n?')
_ALWAYS_STRIP = {"meta.attr.workflow.markup", "meta.attr.prompt.markup"}
_NON_FILE = {"", "0", "black", "clear"}
_MAX_LISTED = 200


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="surrogateescape") as fh:
        return fh.read()


def _write_atomic(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", errors="surrogateescape") as fh:
        fh.write(text)
    os.replace(tmp, path)


def _check_wellformed(text: str) -> None:
    parser = xml.parsers.expat.ParserCreate()
    parser.Parse(text.encode("utf-8", "surrogateescape"), True)


def _norm(p: str) -> str:
    return p.replace("\\", "/")


def _is_abs(p: str) -> bool:
    return p.startswith("/") or bool(re.match(r"^[A-Za-z]:/", p))


def parse_path_map(spec: str) -> tuple[str, str]:
    """'SRC=DST' -> (src, dst), separators normalised, trailing slashes dropped."""
    if "=" not in spec:
        raise ValueError(f"path map must look like SRC=DST, got {spec!r}")
    src, dst = spec.split("=", 1)
    src, dst = _norm(src.strip()).rstrip("/"), _norm(dst.strip()).rstrip("/")
    if not src or not dst:
        raise ValueError(f"path map needs both sides, got {spec!r}")
    return src, dst


def _prepare_maps(path_maps) -> list[tuple[str, str]]:
    maps = [parse_path_map(m) if isinstance(m, str) else (_norm(m[0]).rstrip("/"), _norm(m[1]).rstrip("/")) for m in path_maps or ()]
    return sorted(maps, key=lambda m: -len(m[0]))


def _apply_maps(p: str, maps) -> str:
    low = p.lower()
    for src, dst in maps:
        s = src.lower()
        if low == s or low.startswith(s + "/"):
            return dst + p[len(src):]
    return p


def _split_speed(value: str) -> tuple[str, str]:
    m = _SPEED_RE.match(value)
    return (m.group(1), m.group(2)) if m else ("", value)


def _build_index(dirs) -> dict[str, list[str]]:
    idx: dict[str, list[str]] = defaultdict(list)
    for d in dirs:
        for dp, _dn, files in os.walk(d):
            for f in files:
                idx[f.lower()].append(os.path.join(dp, f))
    return idx


def _trailing_match(a: list[str], b: list[str]) -> int:
    n = 0
    for x, y in zip(reversed(a), reversed(b)):
        if x != y:
            break
        n += 1
    return n


def _expected_sizes(text: str) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for m in _BLOCK_RE.finditer(text):
        body = m.group(2)
        sm = _SIZE_RE.search(body)
        if not sm:
            continue
        for pm in _PROP_RE.finditer(body):
            sizes[html.unescape(pm.group(2))] = int(sm.group(1))
    return sizes


def _ordered_references(text: str) -> list[str]:
    seen: dict[str, None] = {}
    for m in _PROP_RE.finditer(text):
        seen.setdefault(html.unescape(m.group(2)), None)
    return list(seen)


def _analyze_text(text, project_path, path_maps, search_dirs, progress):
    maps = _prepare_maps(path_maps)
    project_dir = os.path.dirname(os.path.abspath(project_path))
    rm = _ROOT_RE.search(text)
    root = _norm(html.unescape(rm.group(2))) if rm else ""
    bases = []
    if root:
        mapped_root = _apply_maps(root, maps)
        if os.path.isdir(mapped_root):
            bases.append(mapped_root)
    bases.append(project_dir)

    refs = _ordered_references(text)
    if progress:
        progress({"phase": "resolve", "done": 0, "total": len(refs)})

    index = None
    sizes = None
    resolved: dict[str, dict] = {}
    unresolved: list[str] = []
    ambiguous: list[dict] = []
    by_method: dict[str, int] = defaultdict(int)

    for i, raw in enumerate(refs):
        prefix, value = _split_speed(raw)
        value = _norm(value)
        low = value.lower()
        if value in _NON_FILE or low.startswith(("color:", "#", "0x")):
            continue
        found, method = None, None
        if not _is_abs(value):
            for base in bases:
                cand = os.path.join(base, value)
                if os.path.isfile(cand):
                    found, method = cand, "relative"
                    break
        else:
            mapped = _apply_maps(value, maps)
            if mapped != value and os.path.isfile(mapped):
                found, method = mapped, "mapped"
            elif os.path.isfile(value):
                found, method = value, "absolute"
        if not found and search_dirs:
            if index is None:
                if progress:
                    progress({"phase": "index", "done": 0, "total": len(search_dirs)})
                index = _build_index(search_dirs)
            cands = index.get(os.path.basename(value).lower(), [])
            if cands:
                comps = value.lower().split("/")
                scored = sorted(((_trailing_match(comps, c.replace("\\", "/").lower().split("/")), c) for c in cands),
                                key=lambda t: (-t[0], t[1]))
                top = scored[0][0]
                best = [c for s, c in scored if s == top]
                if len(best) > 1:
                    if sizes is None:
                        sizes = _expected_sizes(text)
                    want = sizes.get(raw)
                    if want is not None:
                        by_size = [c for c in best if os.path.getsize(c) == want]
                        if by_size:
                            best = by_size
                found, method = best[0], "searched"
                if len(best) > 1:
                    ambiguous.append({"reference": raw, "chosen": best[0], "candidates": best[:10]})
        if found:
            resolved[raw] = {"prefix": prefix, "local": os.path.abspath(found), "method": method}
            by_method[method] += 1
        else:
            unresolved.append(raw)
        if progress and (i % 25 == 0 or i == len(refs) - 1):
            progress({"phase": "resolve", "done": i + 1, "total": len(refs)})

    warnings: list[str] = []
    n_titles = text.count('name="mlt_service">kdenlivetitle<')
    if n_titles:
        warnings.append(f"{n_titles} title clip(s) found: image/font paths embedded in titles are not rewritten.")
    n_proxy = len(re.findall(r'name="kdenlive:proxy">(?!-<|<)', text))
    if n_proxy:
        warnings.append(f"{n_proxy} proxied clip(s) found: proxy files are not archived.")
    notes = []
    for prop in ("kdenlive:docproperties.renderurl", "kdenlive:docproperties.browserurl", "kdenlive:docproperties.profile"):
        m = re.search(rf'name="{re.escape(prop)}">([^<]*)<', text)
        if m and _is_abs(_norm(html.unescape(m.group(1)))):
            notes.append(f"{prop} still holds an absolute path ({html.unescape(m.group(1))}) (harmless project setting, not a media reference).")

    files = {r["local"] for r in resolved.values()}
    return {
        "project": os.path.abspath(project_path),
        "root": root,
        "references": len(refs),
        "resolved": len(resolved),
        "unique_files": len(files),
        "total_bytes": sum(os.path.getsize(f) for f in files),
        "by_method": dict(by_method),
        "unresolved": unresolved[:_MAX_LISTED],
        "unresolved_count": len(unresolved),
        "ambiguous": ambiguous[:_MAX_LISTED],
        "warnings": warnings,
        "notes": notes,
    }, resolved


def analyze(project, path_maps=(), search_dirs=(), progress=None) -> dict:
    """Resolve every media reference in a project; write nothing."""
    text = _read(project)
    report, _ = _analyze_text(text, project, path_maps, [d for d in search_dirs or () if d], progress)
    return report


def strip_metadata_text(text: str, min_bytes: int = 65536) -> tuple[str, dict]:
    """Remove embedded ComfyUI workflow/prompt JSON (and any other huge
    meta.attr.*.markup property) from project XML text."""
    removed = 0
    removed_bytes = 0

    def _sub(m):
        nonlocal removed, removed_bytes
        if m.group(1) in _ALWAYS_STRIP or len(m.group(2)) >= min_bytes:
            removed += 1
            removed_bytes += len(m.group(0))
            return ""
        return m.group(0)

    new = _STRIP_RE.sub(_sub, text)
    return new, {"removed": removed, "bytes_saved": removed_bytes}


def strip_metadata(project, output=None, backup=True, min_bytes=65536) -> dict:
    """Strip embedded metadata from a project file.

    output=None writes <name>_stripped.kdenlive beside the project; passing the
    project's own path edits it in place (keeping a .bak unless one exists)."""
    project = os.path.abspath(project)
    text = _read(project)
    new, stats = strip_metadata_text(text, min_bytes)
    _check_wellformed(new)
    if output is None:
        stem, ext = os.path.splitext(project)
        output = f"{stem}_stripped{ext}"
    output = os.path.abspath(output)
    in_place = os.path.realpath(output) == os.path.realpath(project)
    if in_place and backup and stats["removed"] and not os.path.exists(project + ".bak"):
        shutil.copy2(project, project + ".bak")
    if stats["removed"] or not in_place:
        _write_atomic(output, new)
    return {"project": project, "output": output, "in_place": in_place,
            "size_before": len(text.encode("utf-8", "surrogateescape")),
            "size_after": len(new.encode("utf-8", "surrogateescape")), **stats}


def _archive_key(local: str, used: dict[str, str], used_lower: set[str]) -> str:
    parent = os.path.basename(os.path.dirname(local)) or "root"
    stem, ext = os.path.splitext(os.path.basename(local))
    n = 1
    while True:
        name = f"{stem}{ext}" if n == 1 else f"{stem}_{n}{ext}"
        key = f"media/{parent}/{name}"
        owner = used.get(key)
        if owner == local:
            return key
        if owner is None and key.lower() not in used_lower:
            used[key] = local
            used_lower.add(key.lower())
            return key
        n += 1


def archive(project, dest, *, path_maps=(), search_dirs=(), strip_metadata_opt=True,
            dry_run=False, cancel=None, progress=None, output_name=None) -> dict:
    """Build a portable archive of a project in `dest`.

    Writes <dest>/media/<source_folder>/<file> for every resolved clip and
    <dest>/<output_name or '<stem>_ARCHIVE.kdenlive'> with relative paths and
    root="". Existing destination files of identical size are skipped, so
    re-running resumes. Unresolved references stay untouched and are reported.
    """
    project = os.path.abspath(project)
    dest = os.path.abspath(dest)
    if os.path.realpath(dest) == os.path.realpath(os.path.dirname(project)):
        raise ValueError("destination must not be the project's own folder")

    text = _read(project)
    size_before = len(text.encode("utf-8", "surrogateescape"))
    strip_stats = {"removed": 0, "bytes_saved": 0}
    if strip_metadata_opt:
        text, strip_stats = strip_metadata_text(text)

    report, resolved = _analyze_text(text, project, path_maps, [d for d in search_dirs or () if d], progress)

    used: dict[str, str] = {}
    used_lower: set[str] = set()
    key_for_local: dict[str, str] = {}
    value_to_new: dict[str, str] = {}
    for raw, info in resolved.items():
        local = info["local"]
        if local not in key_for_local:
            key_for_local[local] = _archive_key(local, used, used_lower)
        value_to_new[raw] = info["prefix"] + key_for_local[local]

    def _sub(m):
        new = value_to_new.get(html.unescape(m.group(2)))
        if new is None:
            return m.group(0)
        return f'<property name="{m.group(1)}">{_xml_escape(new)}</property>'

    new_text = _PROP_RE.sub(_sub, text)
    if _ROOT_RE.search(new_text):
        new_text = _ROOT_RE.sub(r"\1\3", new_text, count=1)
    _check_wellformed(new_text)

    out_name = output_name or f"{os.path.splitext(os.path.basename(project))[0]}_ARCHIVE.kdenlive"
    out_path = os.path.join(dest, out_name)
    report.update({
        "dest": dest, "output_project": out_path, "dry_run": dry_run, "cancelled": False,
        "stripped": strip_stats, "size_before": size_before,
        "size_after": len(new_text.encode("utf-8", "surrogateescape")),
        "copied": 0, "skipped_existing": 0, "bytes_copied": 0,
    })
    if dry_run:
        return report

    os.makedirs(dest, exist_ok=True)
    total = len(key_for_local)
    bytes_total = report["total_bytes"]
    for i, (local, key) in enumerate(key_for_local.items(), 1):
        if cancel is not None and cancel.is_set():
            report["cancelled"] = True
            return report
        target = os.path.join(dest, *key.split("/"))
        size = os.path.getsize(local)
        if os.path.isfile(target) and os.path.getsize(target) == size:
            report["skipped_existing"] += 1
        else:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            tmp = target + ".part"
            shutil.copy2(local, tmp)
            os.replace(tmp, target)
            report["copied"] += 1
            report["bytes_copied"] += size
        if progress:
            progress({"phase": "copy", "done": i, "total": total, "current": key,
                      "bytes_done": report["bytes_copied"], "bytes_total": bytes_total})

    if progress:
        progress({"phase": "write", "done": 0, "total": 1})
    _write_atomic(out_path, new_text)
    return report
