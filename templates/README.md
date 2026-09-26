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
