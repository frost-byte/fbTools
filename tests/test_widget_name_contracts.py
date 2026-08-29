"""Cross-layer naming contract: JS widget lookups must match Python widget names.

ComfyUI widget names are defined as string literals in extension.py (the first
argument to io.Combo(), io.String(), io.Int(), etc.).  JavaScript code reads
them back by searching node.widgets with w.name === "some_name".  When a widget
is renamed in Python the JS silently receives undefined and the feature breaks.

This test extracts both sets of names and fails if the JS side references a name
that no longer exists in the Python schema.  Running it after any node schema
change catches stale widget lookups before they reach the user.

ALLOWLIST
---------
Add an entry here ONLY when a JS reference is an intentional backwards-compat
fallback (i.e. it appears alongside the current name in the same find() call).
Include the reason so future readers understand it.
"""
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Widget types that produce node.widgets entries (user-editable controls).
# Wire-input types (Image, Model, Clip, …) are excluded — they don't appear
# in node.widgets so JS never looks them up by w.name.
_WIDGET_IO_TYPES = (
    "Combo", "String", "Int", "Float", "Bool", "Boolean", "Hidden", "Multiline",
)
# Positional first arg: io.Type.Input("name", ...) — name may be on the next line.
_IO_POSITIONAL_RE = re.compile(
    r"\bio\.(?:" + "|".join(_WIDGET_IO_TYPES) + r")\.Input\s*\(\s*[\"']([^\"']+)[\"']"
)
# Keyword id= arg start: we find .Input( then scan forward for id="name".
_IO_INPUT_START_RE = re.compile(
    r"\bio\.(?:" + "|".join(_WIDGET_IO_TYPES) + r")\.Input\s*\("
)
_IO_ID_KW_RE = re.compile(r"\bid\s*=\s*[\"']([^\"']+)[\"']")

# JS pattern: w.name === "x"  or  w.name == "x"
_JS_REF_RE = re.compile(r"w\.name\s*===?\s*[\"']([^\"']+)[\"']")

# Intentional compat fallbacks.  Format:
#   widget_name: "human-readable reason"
# Keep this list minimal.  A name here hides the reference from failure
# reporting — only add it when the JS fallback is deliberate and documented.
ALLOWLIST: dict[str, str] = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _python_widget_names() -> set[str]:
    """Return all widget names defined in extension.py via io.<WidgetType>.Input()."""
    text = (ROOT / "extension.py").read_text(encoding="utf-8")
    names: set[str] = set()
    # Case 1: positional first arg — io.Type.Input("name", ...)
    for m in _IO_POSITIONAL_RE.finditer(text):
        names.add(m.group(1))
    # Case 2: keyword id= arg — io.Type.Input(..., id="name", ...)
    # Scan up to 300 chars past the opening paren for an id= keyword.
    for m in _IO_INPUT_START_RE.finditer(text):
        window = text[m.end() : m.end() + 300]
        id_m = _IO_ID_KW_RE.search(window)
        if id_m:
            names.add(id_m.group(1))
    return names


def _js_widget_refs() -> dict[str, list[str]]:
    """Return {widget_name: ["file:line", ...]} for every w.name lookup in js/."""
    refs: dict[str, list[str]] = {}
    js_dir = ROOT / "js"
    for js_file in sorted(js_dir.rglob("*.js")):
        rel = str(js_file.relative_to(ROOT))
        for lineno, line in enumerate(
            js_file.read_text(encoding="utf-8").splitlines(), 1
        ):
            for m in _JS_REF_RE.finditer(line):
                name = m.group(1)
                refs.setdefault(name, []).append(f"{rel}:{lineno}")
    return refs


# ── Test ──────────────────────────────────────────────────────────────────────

def test_js_widget_names_exist_in_python():
    """Every widget name referenced in JS must exist in the Python node schema.

    If this test fails after a rename:
    1. Update every JS file listed in the failure to use the new name.
    2. OR, if the old name is kept as an intentional compat fallback, add it
       to ALLOWLIST above with a clear reason.
    """
    py_names = _python_widget_names()
    js_refs  = _js_widget_refs()

    failures: list[str] = []
    for name in sorted(js_refs):
        if name in py_names or name in ALLOWLIST:
            continue
        locations = "\n".join(f"    {loc}" for loc in js_refs[name])
        failures.append(f'  "{name}"\n{locations}')

    if failures:
        raise AssertionError(
            "JS widget name lookups with no matching Python widget definition.\n"
            "Rename the widget in JS to match the current Python schema, or add\n"
            "it to ALLOWLIST in this file if the fallback is intentional.\n\n"
            + "\n\n".join(failures)
        )


def test_allowlist_entries_are_still_needed():
    """Allowlist entries whose name now exists in Python should be removed.

    When the Python schema is updated to re-introduce an allowlisted name (or
    a compat period ends), the allowlist entry becomes stale and should be
    cleaned up.
    """
    py_names = _python_widget_names()
    stale = [name for name in ALLOWLIST if name in py_names]
    if stale:
        raise AssertionError(
            "ALLOWLIST entries that now exist in the Python schema and can be removed:\n"
            + "\n".join(f"  {n!r}: {ALLOWLIST[n]}" for n in stale)
        )
