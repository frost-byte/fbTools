# Action Plan — Model VRAM Profiler & Modal GPU Selector (ComfyUI extension)

**Goal:** an estimator that, for each model the user selects (at its chosen quant / context /
modality), computes an approximate peak VRAM footprint and maps it to the **smallest sufficient**
Modal GPU from the user's available set — replacing any flat per-model profile with a computed,
per-configuration recommendation. Self-corrects over time by caching *measured* peak VRAM.

Companion to the Modal-serving plan (`qwen_modal_claude_code_action_plan.md`) — this profiler's output
is the `gpu="..."` string that deployment consumes.

Marker legend:
- **[INVARIANT]** — must hold.
- **[VERIFY]** — confirm the constant/behavior before relying on it.
- **[DECIDE] / [EXPLORE]** — judgment call.

---

## 0. What already exists vs. what this adds
The extension already knows, per selected model: the **task/modality** it's used for and how it's fed
(text vs image vs video, and the frame-selection budget for video — fps/every-Nth/total-frames from
your existing pipeline). **[INVARIANT]** The profiler *consumes* that; it does not re-derive modality
or frame budget. What it adds is: footprint estimation + GPU mapping + user-facing recommendation +
a measured-peak feedback loop.

---

## 1. Core principle — profile per *configuration*, not per *model*
**[INVARIANT]** The unit that has a stable answer is the tuple, not the model name:
```
config_key = (model_id, quant, context_length, kv_cache_quant, modality, frame_budget)
```
The same model at 4-bit/64k lands on a different card than at FP8/200k. The user's **requantize-to-4bit
option is just a different `quant` value** in this key — the profiler handles it for free, producing a
smaller weight term and possibly a smaller recommended card. A flat "model → GPU" table is wrong the
moment quant or context changes; this key is what makes it adaptive.

---

## 2. The estimator — four VRAM consumers, each on its own axis
Estimate these separately and sum; that separation is the whole point (an 8B and a 27B differ far more
in the KV term than in weights).

**2.1 Weights** = `params × bytes_per_param`
- bytes_per_param by quant: **BF16 = 2.0, FP8/INT8 = 1.0, 4-bit ≈ 0.5–0.6** (include the group-scale
  overhead — 4-bit is not exactly 0.5). **[VERIFY]** the exact 4-bit factor for the quant format used
  (AWQ/GPTQ/GGUF differ slightly).
- **[INVARIANT]** This is the term the requantize option changes — it must be an input, never a stored
  constant per model.

**2.2 Vision tower** (VL models only)
- Stays **full precision regardless of weight quant** — add a roughly fixed **~1–2 GB**. A same-size
  text model profiles smaller than a VL one because of this. Store the real per-model size in metadata
  (§4) where known; use a ~1.5 GB default otherwise.

**2.3 KV cache** = the context-scaling term = `kv_bytes_per_token × context_length`
- This is the term that most separates model tiers. Make `context_length` an input.
- `kv_cache_quant` (FP8/Q8) roughly **halves** it — expose it as an input.
- **[VERIFY]** `kv_bytes_per_token` depends on the architecture (layers, heads, head-dim) — compute it
  from the model config if available, else use a per-model metadata value. Anchor for sanity: a 27B
  dense model runs ~2.1 GB KV per 32k tokens at BF16 cache; the full ~262k window ≈ ~17 GB on its own.
- **[VERIFY / metadata]** Hybrid architectures (e.g. Gated-DeltaNet layers) carry **bounded state** for
  some layers, so their KV term is smaller than naive full-attention math — flag such models in
  metadata so the estimate isn't pessimistic.

**2.4 Activations**
- **Video VL:** the `frames × tokens_per_frame` term — pull frames from the existing frame budget,
  tokens-per-frame from resolution/`max_pixels`. Modest for short-clip description; include it.
- **Text:** small; a flat headroom factor covers it.

**2.5 Total** = (weights + vision + KV + activations) × **safety margin (~1.15–1.20)** for allocator
fragmentation and the serving engine's own buffers. **[VERIFY]** tune the margin against real runs.

---

## 3. Card mapping — smallest sufficient from the user's set
Target set (VRAM), user-configurable but default to the user's Modal options:
```
T4=16GB  L4=24GB  A10=24GB  L40S=48GB  A100(-40GB)=40GB
```
- **[INVARIANT]** Pick the **smallest-VRAM card whose capacity ≥ estimated peak**, walking the set in
  ascending VRAM: T4 → L4 → A10 → L40S → A100-40.
- Two ordering caveats to bake in:
  - **A10 vs L4:** same 24 GB; L4 is cheaper, A10 is faster compute. Default to **L4**; only prefer A10
    if a `throughput_priority` flag is set.
  - **A100-40 vs L40S:** A100-40 has *less* VRAM (40<48) and costs *more* — so for capacity it's
    dominated by L40S. Only surface A100-40 when a `bandwidth_priority` flag is set (it has higher
    memory bandwidth → faster generation), never as the default capacity pick.
- If the estimate **exceeds the largest card** in the set → return a **won't-fit** result, not a card
  (see §5 warnings).

---

## 4. Per-model metadata (only what can't be computed)
Keep a small data file (JSON/YAML) the estimator reads, so the *formula* stays general and only genuine
model-specific constants live as data:
- `params` (or read from model config), `is_moe` (**[INVARIANT]** MoE = **all experts resident** even if
  few active — weight term uses total params, not active), `vision_tower_gb`, `kv_bytes_per_token` (or
  the config fields to compute it), `hybrid_kv` flag, and any known-good measured overrides.
- **[INVARIANT]** Do not hardcode a final GPU per model — only these primitives; the card is always
  computed.

---

## 4a. Dynamic profile derivation for custom HF repos
The metadata file in §4 covers known presets. When the user enters a custom HuggingFace repo ID
the estimator must derive the same primitives without a model download.

**What HF exposes without downloading weights:**

- `config.json` (public models: anonymous HTTPS; private: `HUGGINGFACE_HUB_TOKEN` env var) contains
  every architectural primitive the estimator needs: `num_hidden_layers`, `num_key_value_heads`,
  `num_attention_heads`, `hidden_size`, `head_dim` (explicit or derived as
  `hidden_size / num_attention_heads`), `vision_config` sub-dict (VL signal + tower sizing),
  `quantization_config` (pre-quantized signal), `num_local_experts` / `num_experts_per_tok` (MoE).
- **Hub model-info API** (`huggingface_hub.model_info(repo_id)`) returns `.safetensors.parameters`
  — a dtype-keyed dict of exact parameter counts. Sum the values and divide by 1e9 → `params_b`.
  For sharded safetensors the `model.safetensors.index.json` `metadata.total_size` field is the
  byte-count alternative (divide by 2 for BF16). **[INVARIANT]** Use the measured count, not a
  formula approximation — the API gives the real number for free.

**Architecture detection** — map `config.json`'s `architectures[0]` string to the estimator's
arch tags. Known mappings: `Qwen2_5_VLForConditionalGeneration` → `qwen_vl`,
`Qwen3VLForConditionalGeneration` → `qwen3_vl`, LLaVA-family → `llava`, anything else → `generic`.
Presence of a `vision_config` key is the reliable VL flag independent of arch name.

**Vision tower size** — compute from `vision_config.hidden_size` × `vision_config.num_hidden_layers`
using the same attention+FFN param formula as the language tower, at FP16; this gives ±20% accuracy
which is sufficient for card-tier decisions. Where known encoders appear (Qwen-VL ViT, SigLIP,
InternViT) keep a small lookup of their real sizes in the metadata file (§4) and prefer that over
the formula.

**Pre-quantization detection** — `quantization_config` present in `config.json`, or repo name
contains `-awq`, `-gptq`, `-gguf` (case-insensitive).

**Limitations to document for users:**
- GGUF repos often lack `config.json`; the profiler should surface a "cannot profile — GGUF format"
  message rather than silently estimating wrong.
- Very custom architectures (`architectures` key unknown) fall back to `generic`; warn the user
  that the estimate may be less accurate.
- Private repos require `HUGGINGFACE_HUB_TOKEN` in the server environment.

**Local profile cache** — fetched profiles are written to `user_data_dir()/modal_model_profiles.json`
keyed by `repo_id`, with a `_source: "hf_derived"` field and `_fetched_at` ISO timestamp.
The estimator resolution order is: (1) measured peak from §5 cache, (2) local profile cache,
(3) built-in presets (§4), (4) `fetch_hf_profile()` on cache miss (writes result to cache).
A "Refresh" action re-fetches and overwrites the cached profile for that repo.

**`fetch_hf_profile(repo_id, hf_token=None) → dict`** — the single public function.
No GPU, no model load; completes in under a second for public repos.
Returns the same shape as a §4 preset entry, ready for `estimate_vram()`.

---

## 5. Estimate → measure → cache (the feedback loop that makes it robust)
- The parameter-math estimate is good enough to **choose the card** and to **warn on won't-fit** — don't
  over-invest in formula precision.
- **[INVARIANT]** On the **first real run** of a given `config_key`, capture the container's **measured
  peak VRAM** and store it against that key. Future selections of the same config use the *measured*
  number, not the estimate. This corrects formula drift and makes the profiler more accurate with use —
  same descriptors-plus-measured-on-first-use pattern as the fbTools caching work.
- Cache invalidation: keyed on the full `config_key`; a quant or context change is a new key, so it
  re-estimates (and re-measures) rather than reusing a stale number.

---

## 6. What to surface to the user (this is why it's a node feature, not hidden plumbing)
- Recommended Modal GPU for the current selection + **estimated VRAM and headroom left**.
- **Won't-fit / tight warnings** when the estimate exceeds (or nearly meets) the largest available card:
  e.g. "27B BF16 ≈ 54 GB — exceeds L40S; requantize to 4-bit or cap context."
- **Requantize before/after** when that option is available: "4-bit drops this from L40S → L4" — this is
  what makes the requantize toggle meaningful instead of blind.
- **Cost implication** — once the card is known, show ≈ base $/hr (T4 ~$0.59, L4 ~$0.80, A10 ~$1.10,
  L40S ~$1.95, A100-40 ~$2.10 — **[VERIFY]** current Modal rates). **[INVARIANT]** Note the two
  multipliers so the shown cost isn't misleading: region pinning **×1.5–1.75**, non-preemptible **×3**;
  default config (region-agnostic, preemptible) pays base.

---

## 7. Wrinkles & cautions
- **[INVARIANT] Recommend, don't silently auto-select.** The smallest-sufficient card is cheapest but
  tightest; let the user confirm or bump up for margin/throughput. Surface reasoning, leave the pick.
- **Estimation is approximate** — runtime/allocator overhead is hard to predict; that's *why* §5's
  measure-and-cache matters more than a perfect formula. Bias the margin slightly conservative so a
  first run doesn't OOM before it can be measured.
- **MoE trap** — total params resident, not active params; a "30B-A3B" needs ~30B of weight budget.
- **Cold-start billing** — per-second billing counts model-load seconds; note (from the Modal plan) that
  keeping the container warm across a session changes real cost more than the card choice does.
- **Frame budget is an input, not a constant** — a heavier frame budget raises the activation term and
  can bump the recommended card; the estimator must react to the existing pipeline's frame settings.

---

## 8. Suggested sequence
1. Define the `config_key` and the per-model metadata schema (§1, §4). Populate preset entries for
   the five existing Modal models (`params_b`, `n_layers`, `kv_heads`, `head_dim`, `vision_tower_gb`).
2. Implement `fetch_hf_profile(repo_id)` (§4a): fetch `config.json` + Hub model-info, derive all
   primitives, write to `user_data_dir()/modal_model_profiles.json`. No estimator logic yet — just
   the fetch-and-normalize step. Validate against a known model where ground truth is available.
3. Implement the four-term estimator + margin (§2), reading from presets first, then the local
   profile cache, then calling `fetch_hf_profile` on a cache miss.
4. Implement the ascending-VRAM smallest-sufficient mapping over the user's GPU set, with the A10/L4 and
   A100/L40S tie-break flags (§3).
5. Wire the user-facing surface: recommendation, headroom, won't-fit warning, requantize before/after,
   cost, and a "Profile" button for custom repo IDs that triggers `fetch_hf_profile` and shows the
   derived profile inline before saving (§6, §4a).
6. Add the measure-on-first-run capture + cache keyed on `config_key` (§5).
7. Validate estimates against measured peaks for a small/mid/large model (e.g. 8B-4bit, 30B-MoE,
   27B-dense) and tune the margin.

## 9. Definition of done
- Given a model + quant + context + modality/frame-budget, returns a recommended Modal GPU from the
  user's set (smallest sufficient), with estimated VRAM, headroom, and ≈ cost.
- Requantize/context changes produce a different recommendation without any per-model table edit.
- Won't-fit configs return an actionable warning, not a wrong card.
- First real run of a config caches its measured peak; subsequent runs use it.
- No final GPU is hardcoded per model; only computable primitives live in metadata.
- A custom HuggingFace repo ID can be profiled via `fetch_hf_profile` in under a second (no weight
  download); the derived profile is indistinguishable from a preset entry for estimation purposes.
- GGUF repos and unknown architectures surface a clear warning rather than a silently wrong estimate.
- Cached HF-derived profiles persist across sessions and can be refreshed on demand.
