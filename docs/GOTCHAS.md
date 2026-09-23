# Known Gotchas

Short, hard-won lessons about this codebase and the ComfyUI ecosystem it runs
in — patterns that have bitten us more than once, or that took real
investigation to trace the first time. Keep entries terse: symptom → cause →
fix/pointer, a paragraph or two at most. Link out to a `docs/gotchas/<slug>.md`
deep-dive only when an entry genuinely needs one — most won't.

Add an entry here whenever you resolve something that isn't obvious from
reading the code, and that could plausibly bite again in a different context.
Don't log ordinary bug fixes here — only patterns worth recognizing on sight
next time.

---

## ComfyUI custom-node package-name collisions ("model", "nodes", "utils", …)

**Symptom**: importing a class from a sibling `custom_nodes` package fails
with `ImportError` / "class not found" *inside the live ComfyUI server*, even
though the exact same import succeeds fine in an isolated `python -c "..."`
test.

**Cause**: many custom node packs use a generic top-level package name like
`model`, `nodes`, or `utils`. Python caches imports by that bare name in
`sys.modules` — whichever pack claims it first during ComfyUI's startup scan
wins that name for the rest of the process, regardless of `sys.path` order.
A helper that does `sys.path.insert(0, other_pack_dir); from model.x import Y`
silently resolves against whatever *other* installed pack got there first,
the moment two or more packs share that generic name.

**Seen twice**: a SAM/segmentation node integration, and
`utils/audio_preprocess.py`'s MelBandRoFormer loader (fixed 2026-09-15).

**Fix pattern**: never rely on a bare top-level import of a generic name from
outside its own package. Load the target file under a private, synthetic
package name instead, via `importlib.util.spec_from_file_location(...)` +
registering it in `sys.modules` under that synthetic name. See
`utils/util.py`'s `import_virtual_package()` (general-purpose version) or
`utils/audio_preprocess.py`'s `_find_melband_class()` (a minimal, stdlib-only
inline version — that module intentionally has zero ComfyUI dependencies).

---

## PyTorch CUDA OOM "device limit" isn't the real usable ceiling

**Symptom**: `torch.OutOfMemoryError` fires well *under* the reported device
memory limit — e.g. a job failed needing ~17.5 GiB total (8.11 GiB already
allocated + a 9.40 GiB request) against a reported 23.56 GiB device limit,
with `Free (per CUDA)` showing only 245.94 MiB at the moment of failure.

**Cause**: PyTorch's own "Currently allocated" / "device limit" figures don't
account for allocator fragmentation, reserved-but-unallocated segments, or
`--reserve-vram`. On this machine that gap was consistently ~6 GiB. Any
capacity-planning math (e.g. "will this upscale fit?") that compares a
computed requirement against raw device memory will be wrong by that amount
— calibrate against the *device limit* the log reports, not the "currently
allocated" figure, so the gap is baked in rather than silently ignored.

**Where this is used**: `utils/h3_vram_estimator.py`'s `BASE_OVERHEAD_GIB`,
which backs `CompositionToH3Conditioning`'s VRAM estimate / recommended
`MinimaxH3LatentUpscaler3D` scale output (added 2026-09-16, single-incident
calibration — see that module's docstring for the source numbers).

---

## `utils/*.py` modules shouldn't import each other

**Symptom**: a "pure" `utils/*.py` module (one designed to have no ComfyUI
dependencies, loaded via `tests/conftest.py`'s `import_test_module()` by raw
file path) that does `from .other_module import x` can behave inconsistently
between the live ComfyUI server and the test harness, or simply fail to
import, because `utils/` has no `__init__.py` — it isn't a real package, so
relative imports within it rely on implicit namespace-package resolution
that both loading paths handle differently.

**Fix pattern**: for a pure `utils/*.py` module, don't import a sibling
`utils/*.py` module even if the thing you need is tiny and pure. Either
duplicate the small piece of logic locally with a comment cross-referencing
the canonical copy (e.g. `utils/prompt_assembler.py::_slot_letter()` mirrors
`utils/slot_letters.py::slot_letter()` this way), or promote the shared logic
to stdlib-only code both call sites can inline. This rule applies to
`utils/*.py` files specifically — `extension.py` and `scripts/*.py` are real
packages/entry points and import from `utils/*.py` normally via `.utils.x`
(extension.py) or a `sys.path` insert (standalone scripts).

---

## H3 ref plan must come from the assembled scene_instance, not the raw subjects

`assemble_composition()` mints extra slots (background reference, `Fit_N` outfit references,
bundle replacements) that exist only inside the `scene_instance` it builds. The prompt refers to
them as `<Subject N>`, so anything that builds the H3 ref plan from `resolved_subjects` alone
silently drops those images and shifts picture ordinals. `PromptCompositionLoader` therefore builds
the plan from `result["scene_instance"]`. Symptom when it regresses: the prompt mentions a
background/outfit picture but the conditioning node shows no image for it.

---

## Two clips sharing a filename are not necessarily the same clip

A "skip if a file with this name already exists at the destination" check
(`utils/kdenlive_clips.py::clean_folder`'s resume behaviour, and any similar populate-a-folder
script) treats a filename match as "already handled". It isn't safe to assume that: ComfyUI's own
numbered-suffix output naming (`<prefix>_<00001>.<ext>`) recycles across unrelated generation
batches, so a file already at the destination can be genuinely different content from the one
about to be skipped — same name, different clip. This was caught only because a byte-size mismatch
looked suspicious; a same-size coincidence would have hidden it completely.

**Fix pattern**: when it matters whether two same-named files are really the same clip, compare
content, not the filename or even the file size — hash the *decoded* audio/video streams (e.g.
`ffmpeg -i x -map 0:v:0 -f rawvideo - | sha256sum`, and the same for `0:a:0`), not the container
bytes, since a lossless remux (different metadata, same samples) must still count as a match.
`scripts/kdenlive_recover_cast_metadata.py::_stream_hash()` is a reusable example. Never treat a
"file already exists here" check as proof of identical content unless it's backed by this kind of
comparison.

## Check whether a Kdenlive project still references a file before moving it

A `.kdenlive` project stores plain file paths, not stable ids — moving, renaming or deleting a
clip that the project's bin/timeline already references silently breaks that reference until the
project's own XML is updated to match, which none of this repo's Kdenlive tooling does
automatically (see PLAN 5 in the session's working notes for why that's still future work).
`utils/kdenlive_archive.py::count_references(project)` answers "is this filename referenced, and
how many times" from the project's `resource`/`warp_resource`/`kdenlive:originalurl` properties
alone (no disk access, no path resolution) — check it, or run
`scripts/kdenlive_check_resource_usage.py`, before reorganizing any file that might already be
placed in a project you care about. Note a single project can have multiple `.kdenlive` files on
disk (backups, older snapshots with different root-fixup history) that reference the *same* media
folder in inconsistent ways — confirm which file is the one actually being edited before trusting
its reference counts as ground truth.

---

## ComfyUI's `graph.serialize()` doesn't read node draw-order from `graph._nodes`

**Symptom**: two small "pass-through" nodes (e.g. KJNodes' `GetNode`/`SetNode`) end up visually on
top of a larger node they overlap, permanently blocking clicks to it — including the click that
would otherwise bring it back to the front via the frontend's own built-in bring-to-front-on-click.
Reordering `graph._nodes` live (splicing the array, or looping `canvas.sendToBack(node)`) fixes the
*current session's rendering*, but the fix vanishes the moment the workflow is saved and reloaded.

**Cause**: traced through the bundled frontend source
(`comfyui_frontend_package/static/assets/*.js`, minified but not property-mangled, so real method
names like `serialize`/`asSerialisable` are still grep-able). `graph.serialize()` (litegraph's
long-standing public export method, also aliased as `toJSON()`) does not read `graph._nodes` order
at all for its `nodes` array — it rebuilds that array from a separate internal node registry
(`idsByOwner`, a plain JS `Set` keyed by owning-graph id) that only tracks original insertion
order, with no public API to reorder it. Live rendering and the saved-file order are two genuinely
different code paths; fixing one does not touch the other. The workflow JSON's own `order` field on
each node is a red herring here too — it's litegraph's execution/topological-sort order, unrelated
to visual stacking (no correlation with array position or z-order at all).

**Fix pattern**: a *persistent* z-order fix has to patch `graph.serialize()` itself (grab it via
`Object.getPrototypeOf(app.graph)` so the patch covers every graph instance, including subgraphs),
reordering its **output** `nodes` array after calling the original — this is the one point
guaranteed to run on every save regardless of trigger (Ctrl+S, Save menu, autosave) or live canvas
state. For the *live* visual half (so clicks aren't blocked during the current session), call
`app.canvas.sendToBack(node)`/`bringToFront(node)` per node — litegraph's real z-index mechanism —
not a raw `graph._nodes` splice, which doesn't reliably force a repaint (most likely a stale
render-order cache that a plain array reassignment never invalidates). See
`js/fb_tools.js`'s `patchGraphSerializeOrder()` (persistence) and `sendGetSetNodesToBack()` (live)
for a working example of both halves.

---

## ComfyUI's bundled Tailwind CSS is real, but purged to only what ComfyUI itself uses

**Symptom**: a third-party library loaded via CDN (e.g. jsnview, used by the Node Inspector tab)
that assumes a Tailwind utility class works renders with that specific utility silently missing —
not a wholesale "no styling at all" failure, which would be obvious, but one property quietly
absent while its siblings work fine. Concretely: jsnview's collapse/expand toggle used
`absolute -left-4 top-1` to place itself in a row's left gutter; `position:absolute` and `top`
applied correctly but `left` never did, leaving the toggle unplaced (in practice, invisible/
unclickable) while everything else about the row rendered normally.

**Cause**: ComfyUI's frontend (`comfyui_frontend_package`, e.g.
`static/assets/main-DCtjL70R.css`) ships **real, non-fake Tailwind-generated utility CSS** — not
"no Tailwind" as it's easy to assume from a quick look. But it's a production Tailwind build,
purged down to only the exact utility classes ComfyUI's own Vue components reference. That happens
to include common ones like `.absolute`, `.relative`, `.top-1`, `.-rotate-90`, `.pl-7` (ComfyUI's
own code uses those), but not `.-left-4` unscoped (only `.left-4` and a `lg:` responsive variant
survived the purge, because nothing in ComfyUI's own markup uses the bare negative form). There's
no reliable way to know in advance which arbitrary utility a random third-party library needs will
or won't have survived Comfy's purge — it depends entirely on what ComfyUI's own UI happens to use.

**Fix pattern**: don't assume a Tailwind-based CDN library's utility classes either all work or
all fail — check each one that controls layout/positioning specifically, not just colors. For
anything found missing, add an explicit override in this repo's own CSS restoring the exact value
the library intended, same way `js/styles/ui/node_inspector.css` already remaps jsnview's color
utility classes to ComfyUI palette tokens. See that file's `.jsv-toggle` rule (`position`/`left`/
`top` set explicitly) for a working example.

