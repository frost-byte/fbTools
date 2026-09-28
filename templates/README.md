# Bundled ComfyUI API-format workflow templates

Loaded server-side by `utils/h3_template_runner.py` — never served to the browser, so these are
plain files, not under `WEB_DIRECTORY`.

## `h3_background_plate.api.json`

Powers `POST /fbtools/backgrounds/remove_people` (`nodes/backgrounds_presets.py`), built from
`/mnt/comfy_ssd/ComfyUI/user/default/workflows/h3_fizgig_rem_img_bg.json`. Width/height need no
patching — `GetImageSize` (fed by `AILab_ImageResize`) already derives the canvas size from
whatever image loads, at execution time.

### Required titles (must always be present)

Set via right-click → Title (not the node's display name) — the route finds nodes by title, not by
node id:

- `IN:image` — the `LoadImage` node
- `IN:prompt` — the `MiniMax H3 Reference to Video` node
- `IN:seed` — the `RandomNoise` node
- `OUT:save` — the `SaveImage` node

### Optional titles (Settings → H3 Background Plate overrides)

Each is entirely optional — the feature works with none of them present, falling back to whatever
the template itself specifies. `GET /fbtools/backgrounds/h3_settings_options` reports which of
these the current template exposes, and Settings disables any control whose title is missing so a
user can't configure an override that silently does nothing.

Four are **pure renames** of nodes already in the graph — no rewiring needed:

- `IN:model` — the diffusion-model loader (`UNETLoader`)
- `IN:clip` — the `CLIPLoader` node
- `IN:sampler` — the `KSamplerSelect` node
- `IN:scheduler` — the `BasicScheduler` node

`IN:lora` is different and needs an actual graph edit, not just a rename: add an *active*
`LoRA Stack Builder` (`fbt_LoraStackBuilder`) → `LoRA Stack Apply` (`fbt_LoraStackApply`) pair
(fbTools' own nodes, not core `LoraLoaderModelOnly`), wired between the model loader and the
sampler nodes (`BasicScheduler`/`BasicGuider`'s `model` inputs come from `LoRA Stack Apply`'s
`model` output). Title **`LoRA Stack Builder`** `IN:lora` — that's the node whose row-0 widgets get
patched (`lora_0`, `strength_model_0`, `strength_clip_0`, `enabled_0`); `LoRA Stack Apply` needs no
title, just correct wiring. Set **both** nodes' `model_target` to `MiniMaxH3` — `LoRA Stack Apply`
only applies entries tagged for the target it's set to, and `LoRA Stack Builder` tags whatever it
builds with its own `model_target`, so a mismatch between the two silently applies nothing.

This setup gives a real "off" state, unlike core's loader (which has no None/disable option): set
row 0's `lora_0` to `None`, or `enabled_0` to unchecked, as the template's own default — that's
what "— use template default —" in Settings falls back to when no LoRA override is configured.
Settings only ever overrides row 0 (`lora_0`/`strength_model_0`/`strength_clip_0`/`enabled_0`); the
other 7 rows `LoRA Stack Builder` supports are untouched by this feature.

### Exporting

Workflow menu → **Export (API)** (not the regular Export/Save) → overwrite
`h3_background_plate.api.json` here. Re-export any time the graph changes — the route reads this
file fresh on every call, no restart needed for template-only changes (a restart is only needed
when the Python route code itself changes).

## `h3_character_sheet.api.json`

Powers `POST /fbtools/bundles/generate_character_sheet` (`nodes/h3_character_sheet.py`), built from
`R2V_MK4.2_character_sheet.json`. Takes 1-9 reference images of a subject and produces either a
**character sheet** or a **head/face-shot sheet** — both modes run the same two-pass (base +
upscale) pipeline, differing only in which of the workflow's own `DictCreate` nodes
("Character Sheet Options" / "Face Sheet Options") supplies `prompt`/`duration`/`fps`/`steps` to
the generation. **Video mode is out of scope for this template** — strip the video-reference-input
alternative (each `ref_image_N` slot's `VHS_LoadVideo` branch feeding its `Any Switch (rgthree)`)
and every preview-only node (`PreviewImage` nodes, `MarkdownNote`/`Note`, rgthree `Bookmark`/
`Fast Groups Bypasser`) before exporting. Once the `VHS_LoadVideo` branches are gone, each
`Any Switch (rgthree)` just passes its remaining connected input straight through — safe to leave
in place or delete, your call.

The "🐾 Models" node in the source workflow is a collapsed **subgraph** (UNETLoader + CLIPLoader +
two VAELoaders + a LoRA stack), not a single custom node — ComfyUI's Export (API) flattens it into
its real constituent nodes automatically, so title each of those flattened-out inner nodes exactly
as described below, the same as any other node in the graph.

### Required titles (must always be present)

- `IN:refs` — the `DenoAdvancedImageSourceLoader` node, **not** the Reference-to-Video node. This
  matters: `MiniMax H3 Reference to Video`'s `ref_image_0`..`ref_image_8` are real IMAGE *socket*
  inputs (confirmed against a real exported workflow, 2026-09-27) — each fed by its own chain
  (`DenoAdvancedImageSourceLoader` batch → a KJNodes Set/Get global → a `GetImagesFromBatchIndexed`
  per slot → an `Any Switch (rgthree)` per slot) — so they can't be patched with plain filename
  values the way a widget can. `DenoAdvancedImageSourceLoader`'s own `image_paths` field **is** a
  plain multiline-string widget (one input/-relative filename per line, e.g.
  `subdir/photo1.png\nsubdir/photo2.png`), which is what actually gets patched — `patch_prompt`
  sends it your resolved reference filenames joined with newlines. Keep this node in the graph
  as-is; don't wire images directly into the Reference-to-Video node's sockets yourself.
  **Order matters, not just membership**: the template's own prompt (baked into the
  "Character/Face Sheet Options" `DictCreate` nodes) defines `<Outfit 1>` as "the clothing in
  `<Picture 1>`" — the *first* filename in this list — and instructs the model to take identity
  only, never clothing, from every image after it. `nodes/h3_character_sheet.py` preserves
  whatever order the caller sends; the Bundle editor picker (`js/ui/bundle_editor.js`) is what
  actually lets the user control which pick lands first (see `docs/GOTCHAS.md`'s "Multi-reference-
  image prompts can assign special meaning to slot ORDER" entry — this bit a real generation
  before the picker tracked pick order explicitly).
- `IN:mode` — a KJNodes `LazySwitchKJ` node choosing which `DictCreate` options-dict is active
  (replaces the source workflow's original 3-way `ImpactSwitch`, which also carried a "Video
  Options" branch — out of scope for this feature, so a purpose-built 2-way switch fits better than
  disabling a third of a 3-way one). Wire `DictCreate("Character Sheet Options")`'s `dict` output
  into `LazySwitchKJ`'s `on_false`, and `DictCreate("Face Sheet Options")`'s `dict` output into
  `on_true`; wire `LazySwitchKJ`'s output to the same 4 `DictGet` nodes (prompt/duration/fps/steps)
  the old `ImpactSwitch`'s `selected_value` used to feed. Patches its boolean `switch` input:
  `false` → Character Sheet, `true` → Face Sheet.
- `IN:seed` — the `RandomNoise` node.
- `OUT:save` — the final `SaveImage` node for the two image modes. **Confirmed 2026-09-27** by
  tracing its `images` input backward through `ImageFromBatch` → `VAEDecode` → `SamplerCustomAdvanced`
  — this is real generation output, not raw input. The graph also has a `PreviewImage` fed directly
  from `DenoAdvancedImageSourceLoader`'s `image_list` output (i.e. it just re-displays your own
  input references) — that one is purely cosmetic; don't title it.

Per-mode `prompt`/`duration`/`fps`/`steps` stay inside the template's own "Character Sheet Options"/
"Face Sheet Options" `DictCreate` nodes — edit those directly in ComfyUI like any other template
value; this feature does not override them per-request, only *which mode* runs (`IN:mode`).

### Optional titles (Settings → H3 Character Sheet overrides)

Same graceful-disable pattern as the background-plate template — `GET
/fbtools/bundles/character_sheet_settings_options` reports which of these the current template
exposes, and Settings disables any control whose title is missing:

- `IN:model` — the diffusion-model loader (`UNETLoader`, inside the flattened "🐾 Models" subgraph)
- `IN:clip` — the `CLIPLoader` node (same subgraph)
- `IN:sampler1` — the first-pass `KSamplerSelect` node (`res_multistep` in the source workflow)
- `IN:scheduler1` — the first-pass `BasicScheduler` node — only its `scheduler` field is overridden;
  `steps` for pass 1 comes from the active mode's `DictCreate`, not from Settings
- `IN:sampler2` — the upscale-pass `KSamplerSelect` node (`euler` in the source workflow)
- `IN:upscale_steps_select` — the `ImpactStringSelector` node (`"3,4,5 Steps for Upscaling"`) whose
  `select` input picks one of three preset sigma curves for the upscale pass
- `IN:upscale_factor` — the `MinimaxH3LatentUpscaler3D` node's scale-multiplier field (`1.5` in the
  source workflow). **Confirmed 2026-09-27**: the field is displayed as "scale" but its real
  API-JSON key is the dotted `mode.scale` (a compound widget, same naming convention as
  `ref_images.ref_image_N`) — `nodes/h3_character_sheet.py` writes `{"mode.scale": ...}`.
- `IN:aspect_ratio` — the `ResolutionSelector` node; patches both `aspect_ratio` and `megapixels`
  (defaults `"1:1 (Square)"` / `1` in the source workflow)
- `IN:character_prompt` / `IN:face_prompt` — the `PrimitiveStringMultiline` node holding that
  mode's own prompt text (the one feeding "Character Sheet Options"/"Face Sheet Options"
  `DictCreate` respectively). Entirely optional and independent per mode. To opt in, place the
  literal token `{{OUTFIT_HINT}}` somewhere in that mode's prompt text (wherever you want a
  caller-supplied outfit description to land — e.g. right after your `<Outfit 1>` definition
  paragraph) and title the node. A request's `outfit_hint` replaces that token; with no hint
  supplied, the token is still stripped to `""` so it never leaks into the model's actual prompt as
  literal text. `GET .../character_sheet_settings_options` reports
  `has_character_prompt_override`/`has_face_prompt_override` so the UI only offers the hint field
  for whichever mode currently supports it.

`IN:lora` follows the exact same contract as the background-plate template's `IN:lora` — title the
active `LoRA Stack Builder` (`fbt_LoraStackBuilder`) node, with its paired `LoRA Stack Apply`
correctly wired and both nodes' `model_target` set to `MiniMaxH3`. See the `h3_background_plate`
section above for the full explanation; it is not repeated here.

### Exporting

Same as `h3_background_plate.api.json` above: clean up the graph, title the required + any desired
optional nodes, Workflow menu → **Export (API)** → save as `h3_character_sheet.api.json` here.

### Fewer-than-9-images behavior (resolved 2026-09-27)

The 9 `GetImagesFromBatchIndexed` nodes downstream of `DenoAdvancedImageSourceLoader` hard-code
indices 0-8 against its one shared image batch, with no bounds checking at all — confirmed via a
live single-image request that crashed execution outright: `IndexError: index 8 is out of bounds
for dimension 0 with size 1` (KJNodes `indexedimagesfrombatch`, `images[indices_tensor]`). It does
**not** degrade gracefully; a batch smaller than 9 always errors once execution reaches whichever
index exceeds the actual batch size.

Rather than requiring the graph to support this (or requiring exactly 9 images from the caller,
contradicting the feature's own "1-9 images" design intent), `patch_character_sheet_prompt` pads
the caller's reference list up to exactly 9 by cycling it back over itself before joining into
`image_paths` — e.g. one image becomes that same filename repeated 9 times; three images become
that same three repeated three times. Index 0 (the sole outfit reference — see `IN:refs` above) is
always exactly what the caller supplied first; only the redundant identity-only slots get repeats.
This needs no template changes — it's entirely handled in `utils/h3_template_runner.py`.
