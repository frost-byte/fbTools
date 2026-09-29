# fbTools example workflows

Small, focused workflows demonstrating individual fbTools nodes (or a tightly-related pairing),
one behavior per file — see `~/.claude/plans/test-workflows-and-demo-data.md` for why they're kept
this granular rather than bundled into one large graph.

ComfyUI serves this folder automatically (Workflow → Browse Templates) once loaded.

## Before running one of these

Each workflow references demo media by filename from ComfyUI's own `input/` directory (that's how
`LoadImage`/`VHS_LoadVideo` always resolve files — a workflow can't embed the actual bytes). Copy
this folder's `media/*` files into your ComfyUI `input/` directory once before running any of these
workflows:

```bash
cp example_workflows/media/* /path/to/ComfyUI/input/
```

All demo images are small, synthetic, generic placeholder art (flat-color shapes) — never real
photos or project-specific content — generated specifically for these workflows. The one demo video
(`marker_split_test_80f.mp4`) is likewise synthetic: 80 rendered frames of plain text on a black
background, no real footage.

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

- **`marker_frame_split.json`** — `MarkerFrameSplit` (clip-bridging pipeline support: locates a
  deliberately-inserted marker-color frame segment in a pre-concatenated video and splits it into
  the "before"/"after" clip halves) feeding its boundary frames into `SubjectLayerDefine` ×2 →
  `SubjectCompositor` (positioned side by side, `remove_background=False` on both — no background
  removal needed for this demo) and its three scalar outputs (`clip_a_end_idx`/`clip_b_start_idx`/
  `marker_frame_count`) into `ImageTextOverlay` via `Basic data handling: DictCreateFromInt` +
  `StringFormatMap` (from the `basic_data_handling` pack) — burning the diagnostic numbers onto the
  composited image so a single `PreviewImage` proves both the visual split (frame content reads "A"
  on the left, "B" on the right) and the exact indices at once. Uses `marker_split_test_80f.mp4` (80
  frames: 39 "A" + 1 magenta marker + 40 "B"). **Verified live** — run directly in the ComfyUI
  browser UI; see `docs/GOTCHAS.md` for a real interop gotcha this workflow surfaced with
  `comfy-mcp`/`comfy-cli`'s workflow-conversion path specifically (not a bug in the workflow itself).
  This is also the reference example for the new `ImageTextOverlay` node's intended use: giving a
  scalar/string-output node a real, screenshot-able proof image without a third-party display-node
  dependency (see `~/.claude/plans/test-workflows-and-demo-data.md`).

All 7 workflows in this folder are verified live against a real ComfyUI instance.

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
