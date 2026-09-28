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

## `io.NodeOutput(*args)` takes positional values, not a dict — a single dict silently eats the rest

**Symptom**: a node declares 2+ outputs in its schema, but only its first output ever seems to
reach downstream nodes correctly, and it arrives as the *wrong type* — e.g. a whole Python dict
where a plain `IMAGE` tensor was expected. Wiring the node's real output into a real consumer
(`PreviewImage`, `MaskToImage`, etc.) throws something like `KeyError: 0` deep inside ComfyUI's own
`nodes.py` — the consumer tried `images[0]` on what turned out to be a dict, not a tensor/batch.

**Cause**: `comfy_api.latest._io.NodeOutput.__init__(self, *args, ...)` stores whatever positional
arguments you pass as a plain tuple (`self.args = args`), and each declared output slot reads from
that tuple by index — slot 0 is `args[0]`, slot 1 is `args[1]`, and so on. Calling
`io.NodeOutput({"output_image": img, "info": info})` passes **one** argument — a dict — so
`self.args` is a **1-element tuple containing that dict**. Output slot 0 becomes the whole dict
instead of `img`, and slot 1 (`info`) doesn't exist at all, even though the schema promises it.
Nothing validates the argument count against the schema at definition time, so this only surfaces
the moment something downstream actually tries to consume a slot past the first — which can be a
long time after the node was written, if it was never exercised end-to-end before.

**Found**: `SAMPreprocessNHWC`, `TailEnhancePro`, `TailSplit`, `OpaqueAlpha` (`nodes/image_processing.py`)
and `SubdirLister` (`nodes/utility.py`) all had this exact bug, caught only once each was actually
wired into a real consumer while building `example_workflows/` fixtures — not by reading the code,
and not by any existing test (none of these nodes' `execute()` methods are covered by
`tests/`, only their pure `utils/` helpers are). **Not every `io.NodeOutput({...})` call is wrong**:
`AudioFixShape` (`nodes/audio.py`) correctly passes a single dict, because its schema declares
exactly **one** `AUDIO`-typed output, and ComfyUI's own AUDIO convention *is* a
`{"waveform", "sample_rate"}` dict — one argument for one output. Check the schema's output count
before assuming a dict-argument call is broken.

**Fix pattern**: match the schema exactly — `io.NodeOutput(value1, value2, value3)`, one positional
argument per declared output, in the same order as the node's own `outputs=[...]` list. Never pass
a single dict meant to represent multiple outputs.

---

## A tensor "mask" output declared as `io.Image.Output` instead of `io.Mask.Output`

**Symptom**: wiring a node's mask-shaped output into `PreviewImage` throws
`TypeError: Cannot handle this data type: (1, 1, 1), |u1` deep inside ComfyUI's core `save_images`.

**Cause**: `OpaqueAlpha` (`nodes/image_processing.py`) built its "opaque mask" as a genuine 1-channel
tensor (`[B, H, W, 1]`) but declared its output as `io.Image.Output`, which any standard IMAGE
consumer (like `PreviewImage`) expects to be 3- or 4-channel RGB/RGBA. Compare `MaskProcessor` in
the same file, which correctly declares its mask output as `io.Mask.Output` and returns a bare
`[B, H, W]` tensor (no trailing channel dim) — that's this codebase's real MASK convention
throughout, matching ComfyUI's own. A mask-shaped value declared as IMAGE will fail the moment a
real IMAGE consumer touches it; only the earlier `NodeOutput` bug above happened to hide this one
for so long (that bug corrupted the output before it ever reached a consumer).

**Fix pattern**: if a node's output is genuinely a mask, declare it `io.Mask.Output` and squeeze it
to `[B, H, W]` before returning, even if the same tensor needs its `[..., 1]` trailing-channel form
internally for other math (e.g. `torch.cat`/assignment against a 4-channel image) — squeeze only the
value that actually gets returned, not the working copy.

---

## `TailEnhancePro` (and its `utils/images.py` helpers) expect `List[torch.Tensor]`, not a batched tensor

**Symptom**: `RuntimeError: Boolean value of Tensor with more than one value is ambiguous`, or a
`torch.cat` error on a single tensor, inside a node whose own docstring says it takes a
`LIST[IMAGE]`.

**Cause**: ComfyUI's V3 `Input` base class (`comfy_api/latest/_io.py`) has no mechanism to hand a
node a genuine Python list from a single upstream connection — a plain `io.Image.Input(...)` always
receives one batched `[B, H, W, C]` tensor. `TailEnhancePro.execute()` never converts that tensor to
a list before treating it like one (`if not input_frames`, slicing into `head`/`tail`, iterating
`for img in tail`), and its two real helpers, `_compute_ref_stats`/`_pick_ref_image`
(`utils/images.py`), are correctly written for an actual `List[torch.Tensor]` (their own type hints
say so, and `torch.cat(sub, dim=0)` only makes sense concatenating separate list items into a new
batch dim — never called on an already-batched tensor). The node's own internals were never
consistent with each other; it was written years ago for a specific Wan2.1/2.2 clip-stitching
use case (cleaning up flicker/color-mismatch in the last few frames before chaining to the next
clip) and never actually run against real multi-frame input since.

**Fix pattern**: convert once, at the top of the function that needs list semantics, rather than
rewriting already-correct list-based helpers to accept a tensor:
```python
if isinstance(input_frames, torch.Tensor):
    input_frames = [input_frames[i:i + 1] for i in range(input_frames.shape[0])]
```

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

---

## `SubjectLayerDefine`'s mask is inverted relative to core `LoadImageMask(channel="alpha")`

**Symptom**: wire a transparent-background PNG's alpha channel into `SubjectLayerDefine`'s `mask`
input via core `LoadImage` + `LoadImageMask(channel="alpha")`, and every subject renders as a flat
silhouette of the **canvas** color instead of its own colors, while the true background renders
black instead of the configured `canvas_color`. Confirmed by actually running
`example_workflows/subject_compositor_positioning.json` during development — not a hypothetical.

**Cause**: ComfyUI core's `LoadImage`/`LoadImageMask` deliberately inverts alpha on load
(`nodes.py`: `mask = 1. - torch.from_numpy(mask)`), because ComfyUI's MASK convention treats `1` as
"masked-out / inpaint region," the opposite of "area to keep." `SubjectLayerDefine`'s own mask input
(`utils/subject_compositor.py::apply_mask_to_image`) expects the opposite: `1 = keep`, matching its
own docstring. Feeding it straight through silently swaps which pixels are treated as "the subject"
vs. "empty," which is why the *canvas* color shows through the subject shapes and black (the
transparent PNG region's baked-in RGB) shows through everywhere else.

**Fix pattern**: insert a core `InvertMask` node between `LoadImageMask` and `SubjectLayerDefine`'s
`mask` input whenever deriving the mask from an image's own alpha channel this way. If the mask
instead comes from a real segmentation/matting node (SAM, rembg, etc.), check that node's own
convention before assuming either direction — this is a ComfyUI-ecosystem-wide ambiguity, not
something unique to `LoadImageMask`.

---

## Multi-reference-image prompts can assign special meaning to slot ORDER, not just membership

**Symptom**: an H3 character/face-sheet generation renders the subject in a completely different
outfit than any of the reference photos actually show (e.g. a two-piece shirt-and-trousers set
substituted for a single dress) — even though the intended outfit reference genuinely was among the
images sent, and every other identity/generation setting was correct. Confirmed against a real
generation, 2026-09-27 (see `project_h3_character_sheet` session notes).

**Cause**: the workflow's own prompt text (baked into `templates/h3_character_sheet.api.json`'s
`DictCreate` nodes, not anything this repo's code writes) defines `<Outfit 1>` as "the clothing in
`<Picture 1>`" specifically — i.e. whichever reference image lands in the **first** `ref_image_N`
slot — and explicitly instructs the model to ignore clothing in every other reference ("Nothing else
is taken from them: not their clothing or lack of it"). `nodes/h3_character_sheet.py`'s picker UI
(`js/ui/bundle_editor.js::_buildCharSheetPicker`) originally selected images by checkbox and always
sent them in ascending bundle-index order, with no way for the user to control which image landed in
that semantically special first slot — so "Picture 1" ended up being whichever image happened to sit
at the lowest index in the bundle, not necessarily the one that actually showed the intended outfit.

**Fix pattern**: when a template's prompt assigns special meaning to a specific reference-image
*position* (not just "is this image included"), the picker UI must treat position as a first-class,
user-controlled property — not derive it implicitly from an unrelated ordering (array index, alpha
sort, etc.). Fixed here by tracking picks as one unified, insertion-order-preserving list (`picks`)
spanning every source of reference images (the bundle's own saved images AND on-demand video-frame
extracts), with a visible numbered badge on each thumbnail so the user can confirm what's in the
critical first slot *before* generating, rather than only from the DictCreate node's own multi
line prompt text describing the same convention. Before assuming reordering doesn't matter for a
multi-reference-image prompt, read the prompt text itself for `<Picture N>`/positional language.

---

## KJNodes' `GetImagesFromBatchIndexed` has zero bounds checking — a smaller batch crashes, it doesn't clamp

**Symptom**: a ComfyUI workflow that fans one image batch out to several fixed-index consumers
(e.g. N separate `GetImagesFromBatchIndexed` nodes reading indices `0..N-1` from the same upstream
batch) works fine when the batch has exactly N images, then hard-crashes the moment fewer are
supplied: `IndexError: index <k> is out of bounds for dimension 0 with size <actual>` from
`comfyui-kjnodes/nodes/image_nodes.py`'s `indexedimagesfrombatch` (`chosen_images =
images[indices_tensor]`, no `min()`/clamp against the batch's actual size). Confirmed live,
2026-09-27, feeding the H3 character-sheet template (`templates/h3_character_sheet.api.json`) a
single reference image against its 9 hard-coded `GetImagesFromBatchIndexed` consumers.

**Cause**: this node (and likely other plain-indexing KJNodes utilities) assumes the caller already
guarantees a batch at least as large as the largest index requested — it does not degrade
gracefully for a smaller one, unlike nodes explicitly designed for variable-length lists (e.g. ones
using `%` wraparound or `min(idx, len-1)` clamping internally).

**Fix pattern**: when patching a variable number of items into a template that fans a batch out to
fixed-index consumers like this, PAD the supplied list up to the consumers' expected fixed count
before building the batch — don't rely on the workflow to handle a shorter one, and don't assume
"looks optional in the graph" (an unconnected/optional downstream socket) means partial input is
supported either; the crash happens upstream of that, in the indexing node itself. See
`utils/h3_template_runner.py::patch_character_sheet_prompt`'s cyclic-repeat padding for a working
example — pick a padding strategy that preserves whichever position(s) the template's own prompt
treats as semantically special (see the ordering gotcha above) rather than naively repeating the
whole list from the start.

