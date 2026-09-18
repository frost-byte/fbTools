# Reference — MiniMax H3 memory/attention optimization nodes

How the various VRAM-reduction nodes for MiniMax H3 generation actually work,
which ones conflict, and a recommended combination. Written after tracing
each one's actual patch mechanism in the installed custom_nodes and ComfyUI
core source (paths noted below) — not from node descriptions alone, since
several of those undersell or omit the interactions that matter.

**Context:** H3's DiT does one joint QKV self-attention per block over the
full packed sequence (`[text | reference blocks | audio | video]`). Attention
memory scales as `O(tokens²)` in that sequence length; the feedforward (MLP)
memory scales as `O(tokens)`. That distinction is why some of these nodes
matter far more than others for the long/heavy-reference OOMs this project
has been chasing (see `utils/h3_vram_estimator.py` and `docs/GOTCHAS.md`).

---

## The nodes

### `MiniMax H3 Chunk FeedForward` (KJNodes, `comfyui-kjnodes/nodes/minimax_nodes.py`)
- **Patches:** `diffusion_model.blocks.{i}.mlp.forward`, via `add_object_patch`.
- **Mechanism:** splits the packed-token SwiGLU feedforward into `chunks`
  pieces along the token dimension (only when the sequence exceeds
  `seq_threshold`, default 4096). Numerically exact — activations are
  quantized per-token, so chunking doesn't change the result, just the peak
  memory of that one operation.
- **Scaling term addressed:** the linear (`O(tokens)`) FFN term. Does **not**
  touch attention memory at all.
- **Conflicts:** none. Disjoint target (`.mlp.forward`) from every attention
  node below — always safe to combine with any of them.
- **Status in this project:** enabled and confirmed working — 0 OOMs across
  11+ runs since enabling it, including two compositions at 187K-198K total
  tokens, comparable to runs that OOM'd before (see session log / GOTCHAS).

### `MiniMax H3 Low VRAM Attention` (T8, `comfyui-minimax-h3-audio-T8/h3_t8/nodes_h3_memory_advanced.py`)
- **Patches:** `diffusion_model.blocks.{i}.attn.forward`, via `add_object_patch`.
- **Mechanism:** releases the block's normalized input early, then calls
  through to whichever attention backend/function is currently active,
  grouped by `head_chunks` attention heads at a time (default 4) to shrink
  the per-call working set. Same math as stock attention, "只是不同内核形状
  可能产生浮点舍入差异" (different kernel call shapes may cause small
  floating-point rounding differences) — a conservative optimization, not a
  kernel swap.
- **Scaling term addressed:** attention peak memory (the quadratic term),
  via smaller per-call batches rather than a cheaper kernel.
- **Conflicts:** same object-patch target as "MiniMax H3 Mem Eff Sage
  Attention Patch" below — combining the two means whichever is wired later
  in the graph silently wins; the other has no effect. Also conflicts
  (in the partial/confusing sense described below) with "Model Sparse
  Attention".

### `MiniMax H3 Mem Eff Sage Attention Patch` (KJNodes, `comfyui-kjnodes/nodes/ltxv_nodes.py`)
- **Patches:** `diffusion_model.blocks.{i}.attn.forward`, via `add_object_patch`
  — the *same* target as Low VRAM Attention above.
- **Mechanism:** fully replaces the block's attention with a hardcoded call
  to a quantized SageAttention kernel (`_sageattn_int8_fp8_nhd`), including
  its own fused RoPE/RMSNorm and early tensor release. Also supports the same
  head-chunking idea internally (reads `transformer_options["minimax_head_chunks"]`,
  default 1/off) — it borrowed T8's technique rather than needing T8 wired
  in separately.
- **Requires:** a working `sageattention` install and a supported CUDA
  architecture (raises `RuntimeError` at execute time if either check fails
  — cheap to just try).
- **Scaling term addressed:** attention peak memory, more aggressively than
  Low VRAM Attention (genuine kernel replacement, not just smaller batches).
- **Conflicts:** mutually exclusive with T8's Low VRAM Attention (identical
  patch target, last-applied wins). Also partially bypassed by "Model Sparse
  Attention" during its active window — see below.
- **Independent of the global `--use-sage-attention` CLI flag** — it calls
  the sage kernel directly regardless of what ComfyUI's own default backend
  is set to.

### `Patch Sage Attention KJ` (KJNodes, `comfyui-kjnodes/nodes/model_optimization_nodes.py`, class `PathchSageAttentionKJ`)
- **Patches:** sets `model_options["transformer_options"]["optimized_attention_override"]`
  — a model-options flag, *not* an object patch on any method.
- **Mechanism:** generic, works across many architectures by hooking
  ComfyUI's shared `optimized_attention()` dispatch function, which any
  model's *stock* attention module can consult if it threads
  `transformer_options` through (H3's stock `Attention.forward` in
  `comfy/ldm/minimax/model.py` does).
- **Relationship to the MiniMax H3-specific Sage patch: redundant, not
  additive.** The H3-specific patch replaces the block's forward method
  entirely and never calls `optimized_attention()`, so it never consults
  this flag — enabling both means this node's setting is simply inert for
  H3 blocks. Use **one or the other**, not both.
- **When it's actually useful for H3:** as a *fallback* if the H3-specific
  patch's own compatibility checks fail (unsupported CUDA arch, or a
  ComfyUI build without the needed `comfy.quant_ops` support) — this node
  still reaches H3's stock path in that case, just without the H3-specific
  fused RoPE/head-chunking optimizations layered on top.
- **Also independent of the global `--use-sage-attention` CLI flag** — same
  reasoning: it's a per-graph override, evaluated before ComfyUI's global
  default is consulted.

### `Model Sparse Attention` (ComfyUI core, `comfy_extras/nodes_sparse_attention.py`, class `BlockSparseAttention`)
- **Patches:** `set_model_patch_replace(..., "dit", "double_block", i)` — a
  block-level replacement hook, architecturally different from the
  `add_object_patch` calls above.
- **Mechanism:** three selectable `selection` methods — **sol-attn**
  (Sparsifying Online Attention, training-free adaptive per-head threshold —
  a genuinely different technique from SageAttention, not just a naming
  variant), **sla** (fixed keep-percent of key blocks, needs
  SLA-distilled LoRAs), **vsa** (video-cube tiling, needs FastH3 weights).
  Active only within a configurable `start_percent`→`end_percent` window of
  the denoising schedule (plus `dense_blocks`/`min_tokens` exceptions);
  outside that window it falls back to the dense attention configured via
  "Model Attention Backend".
- **Scaling term addressed:** reduces attention *compute* (fewer key blocks
  actually attended to) more than memory per se — the node's own description
  frames it as a speed optimization for long sequences, which matches what
  you observed (no visual/audio quality issues, faster steps).
- **Conflicts with the attention object-patches above (Low VRAM Attention /
  Sage Attention Patch) — confirmed via the actual replacement function,
  `make_h3_block_patch`:** it calls the block's *original* forward with an
  injected `args["attention"]` override only when `h3_eligible(...)` is true
  for that step. When eligible (inside the sparse window), whatever's
  object-patched onto `.attn.forward` is completely bypassed — only the
  sparse implementation runs. When *not* eligible (dense fallback), the
  block's original forward runs its normal internal attention call, which
  *does* resolve to whatever's object-patched. Net effect: combining Sparse
  Attention with either Sage/Low-VRAM patch doesn't crash, but produces
  **time-varying, partial behavior** — the object-patched optimization is
  silently inactive for most of the schedule (however much of it falls
  inside the sparse window) and only contributes during the dense-fallback
  portion. Confusing to reason about; avoid combining unless you specifically
  want both effects and understand the split.

### `Model Attention Backend` (ComfyUI core, `comfy_extras/nodes_model_advanced.py`, class `ModelAttentionBackend`)
- **Patches:** `m.set_model_optimized_attention(attention_function)` — yet
  another distinct mechanism, this one setting the model's own default
  attention function used by its *unpatched* forward path.
- **Purpose (per its own description):** "the dense attention implementation
  for the model. When used with Block Sparse Attention, this backend is used
  whenever sparse attention is inactive or unsupported." I.e. it's the
  dense-fallback selector for Model Sparse Attention specifically.
- **Options:** `"pytorch attention"` (default) or `"comfy kitchen attention"`
  (comfy_kitchen's own INT8-quantized kernel — conceptually similar to
  sageattn but a separate implementation, gated on
  `COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE`).
- **Interaction with the object-patches:** irrelevant wherever `.attn.forward`
  has been object-patched (Low VRAM Attention or Sage Attention Patch) —
  those replace the method outright and never consult this setting. Only
  matters paired with Model Sparse Attention, or used entirely on its own
  with no other attention node in the graph.

### Global `--use-sage-attention` CLI flag (ComfyUI core, service-level)
- Sets ComfyUI's process-wide *default* `optimized_attention` function, used
  only where nothing else overrides it.
- On this machine it's currently **disabled** (commented out in
  `/etc/systemd/system/comfyui_377.service`, with `--use-pytorch-cross-attention`
  active instead) — left that way because Qwen workflows need sage
  disabled while LTX-2 wants it enabled, and a global flag can't satisfy
  both at once.
- Both "Patch Sage Attention KJ" and the MiniMax H3-specific Sage patch
  operate independently of this flag — they're per-graph overrides that
  take effect regardless of the global default, which is exactly the point:
  it lets different workflows on the same service make different choices
  without touching the service config.

---

## Compatibility matrix

| | Chunk FeedForward | Low VRAM Attention (T8) | Sage Attention Patch (H3) | Patch Sage Attention KJ (generic) | Model Sparse Attention |
|---|---|---|---|---|---|
| **Chunk FeedForward** | — | ✅ | ✅ | ✅ | ✅ |
| **Low VRAM Attention (T8)** | ✅ | — | ❌ same patch target, last wins | redundant if T8 applied last | ⚠️ partial/time-varying |
| **Sage Attention Patch (H3)** | ✅ | ❌ same patch target, last wins | — | redundant, KJ's flag unused | ⚠️ partial/time-varying |
| **Patch Sage Attention KJ** | ✅ | redundant if T8 applied last | redundant, KJ's flag unused | — | untested combination |
| **Model Sparse Attention** | ✅ | ⚠️ partial/time-varying | ⚠️ partial/time-varying | untested | — |

✅ = compose cleanly, no interaction. ❌ = mutually exclusive (same patch
target). ⚠️ = doesn't crash, but produces confusing partial/time-varying
behavior — avoid unless you specifically understand and want the split.

---

## Recommended combination

**`MiniMax H3 Chunk FeedForward` + `MiniMax H3 Mem Eff Sage Attention Patch`.**
Disjoint targets (FFN vs. attention), both always-on with no scheduling
windows, and Sage Attention Patch hits the dominant quadratic-scaling term
directly rather than just the linear FFN term. This is the combination
currently in use and validated: 0 confirmed OOMs across 11+ runs since
enabling Chunk FeedForward, including two compositions at 187K-198K total
tokens — a range that reliably OOM'd before.

If the Sage patch's `RuntimeError` checks fail (missing/incompatible
`sageattention`, unsupported CUDA arch), fall back to `Patch Sage Attention KJ`
instead (still independent of Chunk FeedForward) rather than T8's Low VRAM
Attention — both are valid conservative fallbacks, but only one should be
active at a time regardless of which you pick.

Leave `Model Sparse Attention` out of this combination unless you
specifically want its speed characteristics and accept that whichever
attention-memory optimization you're also using will be inactive for
whatever fraction of the denoising schedule falls inside the sparse window.
