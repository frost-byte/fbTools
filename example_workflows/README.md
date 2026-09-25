# fbTools example workflows

Small, focused workflows demonstrating individual fbTools nodes (or a tightly-related pairing),
one behavior per file — see `~/.claude/plans/test-workflows-and-demo-data.md` for why they're kept
this granular rather than bundled into one large graph.

ComfyUI serves this folder automatically (Workflow → Browse Templates) once loaded.

## Before running one of these

Each workflow references demo images by filename from ComfyUI's own `input/` directory (that's how
`LoadImage` always resolves files — a workflow can't embed the actual image bytes). Copy this
folder's `media/*.png` files into your ComfyUI `input/` directory once before running any of these
workflows:

```bash
cp example_workflows/media/*.png /path/to/ComfyUI/input/
```

All demo images are small, synthetic, generic placeholder art (flat-color shapes) — never real
photos or project-specific content — generated specifically for these workflows.

## Workflows

- **`subject_compositor_positioning.json`** — `SubjectLayerDefine` ×2 → `SubjectCompositor`.
  Demonstrates positioning multiple subject layers (via `offset_x`/`offset_y`) onto one composited
  canvas with a custom background color. Uses `subject_a.png`/`subject_b.png`. **Verified live** —
  ran end-to-end against a real ComfyUI instance.
  **Gotcha this workflow exists partly to document**: deriving a `SubjectLayerDefine` mask from a
  transparent PNG via core `LoadImageMask(channel="alpha")` gives you the *inverse* of what
  `SubjectLayerDefine` expects — ComfyUI's core `LoadImage`/`LoadImageMask` deliberately inverts
  alpha (`mask = 1.0 - alpha`, since ComfyUI's MASK convention treats `1` as "masked-out/inpaint
  region"), while `SubjectLayerDefine`'s own mask input expects `1 = keep`. Feed it straight through
  and every subject renders as a silhouette of the *canvas* color instead of its own — confirmed by
  actually running this workflow during development, not a hypothetical. This workflow wires a core
  `InvertMask` node between `LoadImageMask` and each `SubjectLayerDefine` to correct for it; do the
  same in your own graphs when deriving a mask this way.
- **`mask_processor.json`** — `MaskProcessor` alone: hole-removal, grow, region-smooth, and blur
  applied to a mask derived from `subject_a.png`'s alpha (via `LoadImageMask` + `InvertMask`, same
  gotcha as above), with the optional `image` input wired for the overlay-image output. **Verified
  live.**
- **`qwen_aspect_ratio.json`** — `QwenAspectRatio` alone, feeding its recommended width/height
  straight into a core `ImageScale` node so the result is visible (a dangling numeric output has no
  natural preview otherwise). Uses `background_scene.png`. **Verified live.**
- **`sam_preprocess.json`** — `SAMPreprocessNHWC` alone. Uses `background_scene.png`. **Verified
  live.**
- **`opaque_alpha.json`** — `OpaqueAlpha` alone, previewing both its `image_rgba` and `mask`
  outputs. Uses `background_scene.png`. **Verified live.**
- **`tail_split_and_enhance.json`** — `TailSplit` + `TailEnhancePro` together (the one deliberate
  pairing in this group, besides the compositor demo — both operate on "the tail of a frame batch").
  Builds a 5-frame batch from `frame_1.png`–`frame_5.png` via core `ImageBatch` (deprecated but
  still functional; no non-deprecated core equivalent existed at time of writing). **Verified live.**
  `TailEnhancePro` was originally built to help chain generated Wan2.1/2.2 video clips together —
  cleaning up flicker/color-mismatch in the last few frames of one clip before it hands off to the
  next — but had apparently never been run against real multi-frame input before this workflow
  surfaced its bug (below).

All 6 workflows in this folder are verified live against a real ComfyUI instance.

## Three real bugs this effort caught (all fixed, see `docs/GOTCHAS.md` for the full mechanisms)

Building these workflows and wiring real node outputs into real consumers — not just reading the
code — surfaced three separate pre-existing bugs, none of them introduced by this effort:

1. **`SAMPreprocessNHWC`, `TailEnhancePro`, `TailSplit`, `OpaqueAlpha`** (`nodes/image_processing.py`)
   and **`SubdirLister`** (`nodes/utility.py`) were all calling `io.NodeOutput({...})` with a single
   dict instead of one positional argument per declared output — silently corrupting every output
   past the first.
2. **`OpaqueAlpha`**'s mask output was declared `io.Image.Output` but returned a 1-channel tensor —
   any real IMAGE consumer (like `PreviewImage`) fails on it. Fixed to `io.Mask.Output` with the
   returned tensor squeezed to `[B, H, W]`.
3. **`TailEnhancePro`** treated its `input_frames` tensor as if it were a Python list (per its own
   docstring, `LIST[IMAGE]`) without ever converting it — ComfyUI hands every plain `IMAGE` input a
   batched tensor, never a real list. Fixed by converting once at the top of `execute()`.
