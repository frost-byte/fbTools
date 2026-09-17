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
