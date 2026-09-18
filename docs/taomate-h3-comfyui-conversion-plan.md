# Action Plan: Make TaoMate-H3 LoRA ComfyUI-Compatible

## Status: converted, in place, awaiting your live A/B test
Phases 0–4 are done. Converted file is at
`/mnt/NVME/ComfyUI/models/loras/minimax/TaoMate-H3_converted_r128_fl2va.safetensors`
(2.48 GB, FP32 — no dtype cast applied, see Phase 3 note on why). Jump to
**Phase 5** to run it.

## Objective
Determine whether `TaoLiveAIGC/TaoMate-H3` (a rank-128 PEFT-style LoRA adapter
for MiniMax H3) can be loaded in ComfyUI, and if not, convert/repackage it so
it can be.

## Background / known facts (verified via HF web pages, not yet via CLI)
- Repo: https://huggingface.co/TaoLiveAIGC/TaoMate-H3
- Contents: `config.json`, `adapter_config.json`, `adapter_model.safetensors` (2.48 GB, FP32)
- `adapter_config.json` is minimal — only `rank: 128`, `alpha: 128.0`,
  `weight_source: "generator_ema"`, `optimizer_step: 3000`. No `target_modules`,
  no `peft_type`. This suggests it was exported by a custom training loop
  rather than the standard `peft` library's own save routine — see the
  confirmed-format section below for what that means in practice.
- Base model: `MiniMaxAI/MiniMax-H3`, specifically the **FL2VA** variant.
- README states the runtime "materializes BF16 LoRA buffers for inference"
  from this FP32 checkpoint at load time — implying this file is a
  training-time export, not an inference-ready package.

## ✅ Already confirmed from this machine's own install — no network access needed

**Native ComfyUI support exists and its exact key format is known.**
`~/comfyui_env/ComfyUI-torch210/comfy/lora.py` (comfy core, already installed —
no need to clone `comfyanonymous/ComfyUI`):

```python
# comfy/lora.py, inside model_lora_keys_unet()
if isinstance(model, comfy.model_base.MiniMaxH3):
    for k in sdk:
        if k.startswith("diffusion_model.") and k.endswith(".weight"):
            key_lora = k[len("diffusion_model."):-len(".weight")]
            key_map[key_lora] = k
```

This registers **one and only one** accepted key form for MiniMaxH3 LoRAs: the
**bare internal module path** (e.g. `blocks.12.attn.qkv_proj`), no prefix.
Contrast with its neighbor `ACEStep15` a few lines up, which registers *both*
a bare form and a `base_model.model.`-prefixed form — MiniMaxH3 gets no such
fallback. **This is the one thing that can still block a native load.**

The suffix side is generic and already broad. `comfy/weight_adapter/lora.py`
recognizes, for any architecture: standard PEFT `.lora_A.weight` /
`.lora_B.weight`, Kohya `.lora_up` / `.lora_down`, plus a few other variants
(diffusers2, mochi, qwen-default). PEFT-style naming — which is what
TaoMate-H3's minimal, PEFT-shaped `adapter_config.json` implies — is already
handled with zero custom code.

**A working real-world H3 LoRA on this machine confirms the format
empirically.** `custom_nodes/ComfyUI-MiniMax-H3-Turbo` (Larryvrh's fork — this
is what's actually installed; see the corrected repo list below) builds this
exact key map by hand in `_apply_bypass_lora()`, and its own docstring
explains why:

```python
# "The stock model_lora_keys_unet does not recognise the H3 lora naming,
#  so build the key map directly (module -> diffusion_model.<module>.weight)."
key_map = {m: "diffusion_model.{}.weight".format(m) for m in modules}
```

Byte-for-byte the same structure as the native branch above. It derives its
target-module list straight from its own LoRA file's keys —
`modules = sorted({k.rsplit(".lora_", 1)[0] for k in lora})` — which only
works if that file's keys are bare module paths with no PEFT wrapper prefix.
(The docstring's claim that "the stock function doesn't recognise H3" is
likely just stale — this node may predate the native branch above, or its
author simply preferred an explicit local list since they needed one anyway
for the bypass-injection manager.)

**Our own production LoRA nodes go through this same native path — confirmed,
not assumed.** `LoraStackCollect` + `LoraStackApply` (`extension.py`) is what
this project actually uses for LoRA loading, and `"MiniMaxH3"` is a real
selectable `model_target` (`LORA_MODEL_TARGETS`, `extension.py:9772`). For
that target, `LoraStackApply.execute()` falls through to
`_lora_apply_standard()`, which does:

```python
weights, metadata = _lora_load_weights(lora_name)   # plain comfy.utils.load_torch_file
m, c = _comfy_sd.load_lora_for_models(m, c, weights, strength_model, strength_clip, ...)
```

and `comfy/sd.py`'s `load_lora_for_models()` is:

```python
key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)   # ← the MiniMaxH3 branch above
loaded  = comfy.lora.load_lora(lora, key_map)                     # ← the generic suffix matcher above
```

**There is no fbTools-specific H3 logic to account for** — `LoraStackApply` is
a direct passthrough to comfy core for this target. This means: once a
correctly-keyed `.safetensors` file lands in `ComfyUI/models/loras/`, the
existing `LoraStackCollect → LoraStackApply (model_target="MiniMaxH3")`
workflow loads it with **no new node, no code change, and no separate test
harness** — it's the same node already used for every other LoRA target.

## Corrected repo references
The community node packs actually installed on this machine are different
forks than originally assumed here:
- `custom_nodes/ComfyUI-MiniMax-H3-Turbo` → `Larryvrh/ComfyUI-MiniMax-H3-Turbo` (not `A-M-D-R-3-W/comfyui-minimax-h3-turbo`)
- `custom_nodes/ComfyUI-Spectrum-MiniMax-H3` → `xmarre/ComfyUI-Spectrum-MiniMax-H3`
- `custom_nodes/Comfyui_Minimax_h3_latent_Upscaler` → `LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler`

Neither `xiaolibai-sys/ComfyUI-MiniMaxH3` nor `A-M-D-R-3-W/comfyui-minimax-h3-turbo`
are installed. The Larryvrh Turbo LoRA pack already supplied everything Phase
2 needed (see above) — no need to clone anything for that phase.

## Phase 1 — Inspect (done)
Fetched `config.json` / `adapter_config.json` fully (tiny), and the
`adapter_model.safetensors` **header only** via an HTTP range request —
safetensors stores an 8-byte length prefix + JSON tensor index before the
actual weight data, so all 416 key names/shapes/dtypes were readable without
pulling the 2.48GB payload. Full key dump: `~/h3-lora-investigation/adapter_keys.txt`.

Findings:
1. **`config.json` is byte-identical to `adapter_config.json`** — both are
   just `{rank: 128, alpha: 128.0, weight_source: "generator_ema",
   optimizer_step: 3000}`. No `peft_type`, no `target_modules`,
   no `base_model_name_or_path` — confirms this is NOT a standard `peft`
   library save (a real one always includes those fields). It's a bespoke
   training-loop export, as suspected.
2. **416 tensors** = 208 `lora_a`/`lora_b` pairs: 50 main `blocks.N` +
   2 `token_refiner.blocks.N`, each with `attn.qkv_proj`, `attn.out_proj`,
   `mlp.fc1`, `mlp.fc2`. All `F32`.
3. **Prefix: none.** Keys are already bare — `blocks.0.attn.qkv_proj.lora_a`,
   not `base_model.model.blocks.0...`. The one risk flagged earlier (a PEFT
   wrapper prefix ComfyUI's MiniMaxH3 branch doesn't strip) does not apply.
4. **Module-path body: confirmed exact match** against
   `comfy/ldm/minimax/model.py` — `self.out_proj`, `self.fc1`, `self.fc2`,
   `self.blocks` (`MiniMaxH3Model`), `self.token_refiner` (containing its own
   `.blocks`) are the real live attribute names. Not inferred — read directly
   from the class definitions.
5. **Suffix: the actual (only) blocker.** Keys end in `.lora_a` / `.lora_b`
   (lowercase, no `.weight`) — not `.lora_up.weight`/`.lora_down.weight`
   (Kohya), not `.lora_A.weight`/`.lora_B.weight` (PEFT), not any of the other
   forms `comfy/weight_adapter/lora.py`'s `LoRAAdapter.load()` checks. None of
   its seven hardcoded suffix patterns match, so this suffix form is silently
   unrecognized natively.
6. **Shape orientation already matches Comfy's convention**, no transpose
   needed: `lora_a` shape `[rank, in_features]` (down-projection) and
   `lora_b` shape `[out_features, rank]` (up-projection) — exactly Comfy's
   `lora_down`/`lora_up` semantics, just under different names.
7. No per-tensor `.alpha` keys — global `alpha=128, rank=128` in the config
   gives `alpha/rank = 1.0` under Comfy's scale formula, so nothing to encode
   per-tensor.

## Phase 2 — Get ComfyUI's expected key format (done, see prior section above)
Confirmed from local files, no cloning needed: `comfy/lora.py`'s native
`MiniMaxH3` branch + `comfy/weight_adapter/lora.py`'s suffix matcher, cross-
checked against the real, working `ComfyUI-MiniMax-H3-Turbo` LoRA loader.

## Phase 3 — Diagnose (done)
**Conclusion: pure suffix rename, nothing else.** No prefix to strip, no
module-path remap, no dtype cast, no decomposition surgery:
- `<module>.lora_a` → `<module>.lora_down.weight`
- `<module>.lora_b` → `<module>.lora_up.weight`

## Phase 4 — Conversion (done)
`~/h3-lora-investigation/convert_taomate_lora.py` — loads the full
safetensors file, renames all 416 keys per the mapping above (kept FP32; no
cast attempted since Comfy's patch-application path typically casts to the
model's dtype automatically — test the raw FP32 file first, per Phase 3),
writes `taomate_h3_fl2va.safetensors`.

Ran it. Output, self-consistency verified (every `lora_up` has a matching
`lora_down`, zero orphans, zero unrecognized suffixes — checked directly
against the written file):
```
Renamed 416 tensors (208 lora_a/lora_b pairs).
  blocks.0.attn.out_proj.lora_a  ->  blocks.0.attn.out_proj.lora_down.weight
  blocks.0.attn.out_proj.lora_b  ->  blocks.0.attn.out_proj.lora_up.weight
  blocks.0.attn.qkv_proj.lora_a  ->  blocks.0.attn.qkv_proj.lora_down.weight
  blocks.0.attn.qkv_proj.lora_b  ->  blocks.0.attn.qkv_proj.lora_up.weight
Done. taomate_h3_fl2va.safetensors (2.48 GB)
```
Copied to:
`/mnt/NVME/ComfyUI/models/loras/minimax/TaoMate-H3_converted_r128_fl2va.safetensors`

## Phase 5 — Validate (your turn — needs a live GPU generation)
File is already in place (`.../loras/minimax/TaoMate-H3_converted_r128_fl2va.safetensors`)
and self-consistency checked (see Phase 4). What's left needs an actual
generation, which is your call on workflow/prompt/GPU time:
1. Build a minimal graph: `LoraStackCollect` (one entry pointing at
   `TaoMate-H3_converted_r128_fl2va.safetensors`) → `LoraStackApply` with
   `model_target="MiniMaxH3"` → your normal H3 FL2VA generation pipeline.
2. Confirm ComfyUI doesn't silently ignore unmatched keys — check console
   output for "lora key not loaded" (from `comfy/lora.py`'s `load_lora()`),
   which Comfy always prints for anything in the file it couldn't map. Given
   the offline self-consistency check already found zero orphaned/unpaired
   keys, this should come back clean, but it's the one thing that can only be
   confirmed by actually loading it against the real base model.
3. Generate one short test clip with the LoRA on vs. off (toggle via
   `LoraStackApply`'s strength, or remove the stack entry) and confirm a
   visible difference — proves the weights are actually being applied, not
   just parsed.

## Phase 6 — Report back
- Conversion needed: yes — suffix rename only (`.lora_a`/`.lora_b` →
  `.lora_down.weight`/`.lora_up.weight`), no prefix strip, no module remap,
  no dtype cast.
- Script: `~/h3-lora-investigation/convert_taomate_lora.py`.
- Final file: 2.48 GB, FP32, `.../loras/minimax/TaoMate-H3_converted_r128_fl2va.safetensors`.
- Zero unmatched-key warnings: **pending your Phase 5 run** — the static
  self-consistency check (208/208 pairs matched, zero orphans) is done, but
  the real "loaded against the live base model" confirmation needs Phase 5.
- Visual A/B proof: **pending your Phase 5 run.**
- Caveats: FL2VA-only per TaoMate-H3's HF README (untested against Ref2VA
  workflows); FP32 file size is 2x a BF16 equivalent — if VRAM/disk becomes a
  concern, a `.half()`/`.to(torch.bfloat16)` cast in the same conversion
  script would halve it, but wasn't needed for correctness so wasn't applied.
