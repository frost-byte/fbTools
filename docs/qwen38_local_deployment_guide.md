# Qwen3.8 Local Deployment Guide

*Synthesized from four YouTube reviews · August 2026 · Hardware-assessed for RTX 3090 24GB system*

---

## 01 — Executive Summary

Two models dominate the current local deployment conversation: **Qwen3.8-27B** — a dense model that fits fully in 16 GB of VRAM at Q3 — and **Qwen3.8-Flash-Next**, a 125B mixture-of-experts preview with a revolutionary 51B N-gram lookup table designed to live outside GPU memory.

| Metric | Value |
|---|---|
| 27B @ 16GB VRAM | 10–90 t/s (Q3 KXL, depending on GPU generation) |
| Flash-Next @ 128GB unified | ~20 t/s (Q1–Q4, MTP not yet supported) |
| Flash-Next memory floor | 75 GB (Q1-bit minimum; Q4 needs ~99–112 GB) |
| Best harness (coding) | DeepSeek Harness |

**Key finding:** For most users on single-GPU hardware (16–32 GB VRAM), Qwen3.8 27B at Q3 KXL remains the practical daily driver. Flash-Next is impressive architecture but requires a dedicated machine with 82 GB+ memory for even its 1-bit build.

---

## ✦ This System — Hardware Profile

*Profiled August 2026. "✦ This System" markers throughout show where this hardware lands in each comparison.*

| Component | Spec | Notes |
|---|---|---|
| **GPU** | RTX 3090 · 24 GB GDDR6X | 936 GB/s · CUDA 8.6 (Ampere, GA102) · PCIe 3.0 ×16 · ~15 GB free VRAM currently |
| **CPU** | i5-6600K · 3.5–3.9 GHz | 4 cores / 4 threads (no HT) · Skylake (2015) · **NO AVX-512** · L3 6 MB |
| **RAM** | 64 GB DDR4 | Dual-channel · ~34 GB/s bandwidth · 148 GB swap (HDD-backed — too slow for offload) |
| **NVMe** | Crucial P310 1 TB | PCIe 4.0 (limited to PCIe 3.0 ~3.5 GB/s by CPU) · **Only 91 GB free** |
| **SATA SSD** | Samsung 850 EVO 250 GB | ~550 MB/s · ComfyUI here · 64 GB free · suits all 27B quants up to Q5 |
| **HDD** | WD 750 GB + WD 1 TB | ~120 MB/s — too slow for model loading or offload |

### What This System Can and Cannot Run

| | |
|---|---|
| ✅ 27B at Q4–Q5 fits | 24 GB VRAM holds Q4 (~18 GB) with ~6 GB headroom, or Q5 (~22 GB) tightly |
| ✅ Est. decode speed (Q4) | ~50–60 t/s — 936 GB/s GDDR6X significantly faster than laptop 4090 (16 GB, ~576 GB/s) |
| ❌ Flash-Next (any quant) | Q1 needs 82 GB; system has ~74 GB reachable; severe CPU offload bottleneck |
| ⚠️ CPU offload penalty | ~27× slower (34 GB/s RAM vs 936 GB/s VRAM) — keep all layers on GPU |

> **Critical — llama.cpp must be compiled from source:** The i5-6600K has no AVX-512.
> Pre-built llama.cpp wheels that include AVX-512 will crash with **SIGILL** at startup.
> Always build with `cmake -DGGML_AVX512=OFF` or use a build targeting Skylake (AVX2 only).

> **Storage planning:** The NVMe (91 GB free) fits Q4 (~18 GB) or Q5 (~22 GB) with room, but not Flash-Next. The SATA SSD (64 GB free) can hold any 27B quant up to Q5. Avoid loading models from HDD — 120 MB/s makes cold loads painfully slow.

---

## 02 — Model Landscape

| Model | Params | Active Params | Architecture | Min VRAM | Status |
|---|---|---|---|---|---|
| **Qwen3.8-27B** ★ | 27B | 27B (dense) | Standard transformer | ~12 GB (Q2) | Production |
| **Qwen3.8-27B-Ridge** | 27B | 27B (dense) | Fine-tune of base 27B | ~15 GB (≈Q3) | Community FT |
| **Qwen3.8-Flash-Next** | 125B + 51B table | 6B active | MoE + Gated DeltaNet + QSA | ~75 GB (Q1) | Preview (Qwen4) |
| **Qwen3.6-35B (MoE)** | 35B total | ~3B active | MoE | ~16 GB | Production |

**Name confusion:** "Qwen3.8-Flash-Next" is the open-weight download. "Qwen3.8-Flash" is Alibaba's hosted cloud service (15 cents/M input, 47 cents/M output, 1M token context, built-in tools). The cloud version is intelligence-equivalent but not the same weights.

---

## 03 — Qwen3.8-27B: What to Know

### Behavioral Notes

- **Looping / mid-generation stops** — At Q3 and lower, the model can enter repetitive loops then halt without warning. Often a harness max-output-token limit (32K) being hit silently. Fix: nudge with "continue", raise `max_tokens`, or add `repeat_penalty=1.05, repeat_last_n=512`.
- **Temperature sensitivity** — May loop more at temperature 0.6 on lower quants; some users report better stability at 1.0.
- **Thinking mode** — Three native levels: low, medium, X-high. Pi harness adds its own minimal/high layers. For complex long coding tasks: **low thinking performs best** — higher settings exhaust context before completion.
- **Ridge fine-tune verdict** — Ridge ≈ Q3 KXL in both speed and quality. Reviewer recommendation: use base Q3 KXL over Ridge — same performance, more widely tested.

---

## 04 — Flash-Next Architecture Deep Dive

The config file names the architecture `Qwen4ExpForConditionalGeneration` — this is effectively a Qwen4 preview. Four key engineering decisions separate it from prior releases:

### Layer 1 of 4: Gated DeltaNet
3 of every 4 layers use linear attention — a constant-size "whiteboard" each token edits. Cheap and lossy by design. Holds the gist of everything without the O(n²) context cost.

### Layer 2 of 4: Quick Sparse Attention (QSA)
Every 4th layer runs real attention but scores 4-token micro-blocks instead of individual tokens. Keeps the best 512 blocks → 2048 positions out of 1M tokens (0.2%). Result: **7.6× faster prefill, 4.9× faster decode** vs. full attention.

### Layer 3 of 4: Gated Residual Highway
The residual stream splits into 4 parallel lanes. Without design intent, one lane self-organized into an express highway — 11-layer average jumps vs. 3.5 for the other three. Cross-layer information delivery, emergent.

### Layer 4 of 4: Muon Optimizer
Applied to 2D weight matrices. Produced zero gradient spikes vs. 183 per 10K steps with AdamW. Eliminated batch-size warmup (was wasting 18.8% of optimizer steps). Reduced training cost to ~1/9 of previous flagship.

### The 51B N-gram Table — The Actual Story

Stores ~20 million learned phrases (2–3 token combos). The address (your last 2 tokens) is known ahead of time — no computation needed to find which rows to fetch. The fetch starts early, overlaps with the first decoder layer, and **the table never needs to sit in GPU VRAM**.

**Three independent measurements confirmed zero-degradation offloading:**

| Measurement | Setup | GPU VRAM freed | Throughput change | Output quality |
|---|---|---|---|---|
| SGLang / LMSYS ★ | 4× H200, pinned host memory | −23 GB / card | −0.07% | Bit-identical on 4 prompts |
| Community (King Jones 777) ★ | Strix Halo AMD (GTT page cache) | −46 GB off GPU | Page-cached (OS managed) | 63 GB GPU across all 3 quants |
| SSD offload (community loader) | Custom loader (not upstream) | Full 51B off GPU | ~5% drop | Requires non-stock llama.cpp |

**Benchmark caveats:** Alibaba self-reported 58.7 DeepSqueeze / 62.5 SqueezeBench Pro / 73.9 CoreWorkBench using their own harness. Independent: Artificial Analysis rates it 56 on their intelligence index — strong (level with Gemini 3.7 Flash, above Claude Sonnet 5), but not the open-weights leader (Kimi K3 sits 4 points ahead). The model is chatty: uses ~200M tokens where the median model uses ~110M.

---

## 05 — Quantization Options

### Qwen3.8 27B (Dense)

| Quantization | Approx Size | VRAM Needed | Speed (RTX 2000 Ada) | Speed (RTX 4090 laptop 16GB) | ✦ This System (RTX 3090) | Quality | Notes |
|---|---|---|---|---|---|---|---|
| Q2 KXL | ~12 GB | ~12 GB | 11.4 t/s | ~40+ t/s | ~55 t/s est. | Usable | Good for simple tasks; fine for non-critical coding |
| **Q3 KXL ★** | ~15 GB | ~15 GB | 10.7 t/s | ~40 t/s | ~52 t/s est. | **Recommended** | Best 16GB balance; comparable to Ridge fine-tune |
| **Q4 K_XL ✦** | ~18 GB | ~18 GB | 6.4 t/s | ~40 t/s | **~50 t/s est.** | Good | ✦ Sweet spot on this system — ~6 GB KV headroom |
| **Q5 K_XL ✦** | ~22 GB | ~22 GB | — | — (needs 24GB+) | **~42 t/s est.** | High | ✦ Fits (24 GB); only ~2 GB KV headroom — tight context |
| UD-IQ3_XXS | ~12 GB | ~12 GB | — | 40+ t/s | ~55 t/s est. | High for size | Unsloth ultra-dynamic; maximum context option on this system |
| IVFP4 (Nvidia) | ~16 GB | ~16 GB | — | ~30 t/s | — (Blackwell/Ada only) | High | Not compatible with Ampere (RTX 3090) |
| Ridge ≈ Q3 | ~15 GB | ~15 GB | 10.8 t/s | — | ~52 t/s est. | Comparable to Q3 | Empero AI fine-tune; reviewer recommends base Q3 instead |

### Qwen3.8-Flash-Next (MoE 125B + 51B table)

GPU memory shown is for the MoE core only — the 51B N-gram table stays in host CPU RAM / page cache. Strix Halo GTT measurements show **63 GB GPU regardless of quant** (46 GB off-GPU).

| Quantization | Disk Size | Peak Total Memory | GPU Memory (MoE core) | Speed (Strix Halo 128GB) | Top-1 Accuracy | Status |
|---|---|---|---|---|---|---|
| Q1 (1-bit) | ~48 GB | ~82 GB | ~36 GB | 23.1 t/s | ~80% | Runs (min config) |
| Q2 | ~60 GB | ~90 GB | ~44 GB | ~22 t/s | ~86% | Runs |
| Q3 | ~78 GB | ~95 GB | ~49 GB | ~21 t/s | ~90% | Runs |
| **Q4 IQ4_XS ★** | ~99–112 GB | ~99 GB | ~53 GB | 19.9 t/s | ~93% | **Recommended** |
| Q5+ | >111 GB | >128 GB | — | — (OOM on 128GB) | — | Not runnable |
| **✦ RTX 3090 + 64 GB DDR4** | — | **~74 GB reachable** | — | — | — | **Not runnable** |

> **✦ This System — Flash-Next is not viable:** Q1 (1-bit) build requires ~82 GB minimum; Q4 needs ~99–112 GB. This system has 24 GB VRAM + 64 GB DDR4 = ~74 GB reachable if CPU offload were fully active, which still falls short of the Q1 floor. The 51B N-gram table would add another ~24 GB on top.

---

## 06 — Hardware Requirements & Performance

### Qwen3.8-27B — Single GPU

| Hardware | VRAM | Best Quant | Decode Speed | Max Practical Context | Daily Driver? |
|---|---|---|---|---|---|
| RTX 3060 / 4060 / 5060 (12GB) | 12 GB | Q2 KXL | ~12 t/s est. | ~100K | Constrained |
| RTX 2000 Ada | 16 GB | Q3 KXL | 10.8 t/s | ~130K | Yes |
| RTX 4090 laptop (16GB) ★ | 16 GB | UD-IQ3_XXS | 40+ t/s | ~80K+ | Excellent |
| RTX 5080 (16GB) ★ | 16 GB | Q5 (148K ctx) | 70–90 t/s | ~148K | Excellent |
| RTX 5090 (32GB) ★ | 32 GB | Q5_K_XL + KV Q8 | 95–102 t/s | 131K | Best single GPU |
| Dual RTX 5060 Ti (2 × 16GB) | 32 GB total | IVFP4 | ~30 t/s | — | Good |
| **✦ RTX 3090 (24GB)** | **24 GB** | **Q4_K_XL (sweet spot)** | **~50 t/s est.** | **~100K (Q4) / ~50K (Q5)** | **Yes — Daily Driver** |
| RTX 3090 × 4 (vLLM) | 96 GB total | Various | High (parallel) | 96K+ (32K max-out) | Server use |

### Qwen3.8-Flash-Next — Unified Memory & Multi-GPU

| Hardware | Total Memory | Best Quant | Decode Speed | Context | Practical? |
|---|---|---|---|---|---|
| DDR5 128GB + RTX 5070 Ti | 128 GB total | Q4 IQ4_XS | 22 t/s | — | Marginal (split) |
| Strix Halo 64GB (AMD unified) | 64 GB unified | Q1–Q2 only | ~25+ t/s | — | Q1/Q2 only |
| Strix Halo 128GB (AMD unified) ★ | 128 GB unified | Q4 IQ4_XS | 19.9–32 t/s | 131K+ | Viable (dedicated) |
| Apple M3 Max 128GB ★ | 128 GB unified | eQ4-MTP-MLX | 17+ t/s | 200K | Viable (MLX) |
| Apple M5 Max 128GB ★ | 128 GB unified | Q4 | 40+ t/s | 128K | **Best consumer option** |
| 4 × RTX 5090 (vLLM TP4+EP) | 4 × 32 GB | Production | 2500 t/s (128 streams) | 96K | Enterprise |
| 4 × H200 (SGLang) | 4 × 80 GB | Full weights | Very high | +78% ctx vs baseline | Research/Production |
| **✦ RTX 3090 + 64 GB DDR4** | **~74 GB reachable** | — | — | — | **Not runnable** |

**Flash-Next as daily driver:** On a 128GB Strix Halo used for video production, gaming, and AI, running Flash-Next at Q4 occupies ~99GB leaving little headroom. Treat it as an overnight/dedicated workload machine rather than a general-purpose desktop.

---

## 07 — Harness Comparison

Tested on RTX 5090 with Qwen3.8-27B via Unsloth Desktop. Two benchmarks: a complex 11-page Minecraft clone prompt and a 17-page color palette website prompt.

### DeepSeek Harness — 1st Place

| Attribute | Detail |
|---|---|
| Interface | Browser (localhost) |
| Platform | Windows / Linux |
| First-shot | Excellent (near one-shot) |
| Revisions needed | Few minor tweaks |
| Best thinking | Low |
| Multi-agent | Yes — /goal spawns sub-agents |
| Plugin system | Yes (Creator Mode) |
| Minecraft result | Best — crafting worked, biomes, inventory |
| Website result | Best — closest to professional design |

### Pi Coding Agent — 2nd Place

| Attribute | Detail |
|---|---|
| Interface | Terminal (Claude Code-like) |
| Platform | Win / Mac / Linux |
| First-shot | Good (low), poor (medium/X-high) |
| Revisions needed | Some (texture issues) |
| Best thinking | Low |
| Multi-agent | Limited |
| Plugin system | Yes (extension model) |
| Minecraft result | Good (low) — no crafting |
| Website result | Functional, less polished |

### Hermes Agent — 3rd Place

| Attribute | Detail |
|---|---|
| Interface | Discord / Chat |
| Platform | Any (chat-based) |
| First-shot | Poor — many revisions needed |
| Revisions needed | Most of the three |
| Best thinking | Low |
| Multi-agent | Yes (designed for agents) |
| Plugin system | Partial |
| Minecraft result | Poor — no block break anim, zombies uncollectable |
| Website result | Good visual style, took longest |

**Note:** Hermes's poor coding result reflects using it as a coding harness — its actual design goal is persistent agents, Discord integration, and multi-model routing.

**Thinking level finding:** Across all three harnesses, *low thinking consistently outperformed medium and X-high* on complex, long-running coding tasks. Higher thinking levels exhaust context limits before completion. For shorter, targeted tasks, medium may be more appropriate.

**Not tested but frequently requested:** OpenCode, QwenCode, OMP (Open Model Playground) scaffold, LM Bionic. Community consensus suggests OpenCode and OMP are strong alternatives to DeepSeek Harness. The OMP scaffold with circuit breakers, dependency DAGs, and FSM workflow (Plan → Code → Verify → Commit) reportedly enabled a 64,000-line Minecraft-style voxel engine on a dual 5060Ti setup.

---

## 08 — KV Cache Optimization

KV cache quantization is one of the highest-leverage tuning knobs for extending context on memory-constrained hardware without major quality loss.

| KV Cache Config | Memory Impact | Quality | Best For | Example Use |
|---|---|---|---|---|
| FP16 (default) | Highest | Reference | 32GB+ VRAM, accuracy critical | BF16 model runs |
| **K=Q5_0, V=Q4_0 ★** | ~40% reduction | Near-reference | **Best quality/memory balance** | RTX 4090 16GB, 80K+ ctx |
| **Q8_0 ★** | ~35% reduction | Near-reference | Quality priority, RAM available | RTX 5090 + Q5_K_XL, 131K ctx |
| Q4_0 / Q4_K | ~55% reduction | Good | Maximizing context on 16GB | 16GB card, 200K context attempts |
| IQ4_XS | ~60% reduction | Acceptable | Extreme context, quality secondary | 16GB + Q2 model with 200K ctx |

### Context Window Strategy

- **Hard-cap at 65K, not 200K+** — Community testing shows reliability degrades significantly above 65K context, even with capable hardware. OMP scaffold users reported dramatically reduced looping by capping at 65K.
- **Leverage cached prefill** — At 32K tokens, the RTX 2000 Ada achieves 52,000 cached tokens/sec vs. 227 uncached. Stable system prompts and unchanged codebase sections dramatically reduce effective latency.
- **Flash-Next: N-gram table offloading** — On Strix Halo hardware, the OS page cache handles the 51B N-gram table automatically. On discrete GPU setups, SGLang's host-memory pinning recovers 23GB per H200 card and increases KV context capacity by 78%.

---

## 09 — Other Optimization Strategies

### MTP (Multi-Token Prediction)

- **Enable on 27B** — MTP is well-supported for the 27B dense model. Depth 2 alone yields significant speed gains. Depth 3–4 adds more with minor precision trade-offs.
- **Flash-Next: not yet supported at time of review** — llama.cpp support PR (#27742) was merged August 27, 2026. When added, expect 30–50% speed improvement given the model's 6B active params and high memory bandwidth headroom.
- **Disable for extra context headroom** — On 16GB cards, disabling MTP frees enough VRAM to push context meaningfully higher. The speed improvement on 16GB is reportedly modest enough to consider this trade.

### Anti-Looping Strategies

- **Repeat penalty** — Set `repeat_penalty=1.05` and `repeat_last_n=512` in llama.cpp / Ollama. Effectively eliminates the "model loops silently then stops" pattern.
- **Harness circuit breakers** — OMP scaffold and DSH's /goal mode implement FSMs that detect repetition patterns and force a state transition (Plan → Code → Verify → Commit). This drove the biggest reliability gains in the OMP user's 64K-line project.
- **Dependency DAGs before coding phases** — Generating a dependency graph before major implementation phases reduces mid-session architectural drift and prevents the "rewrite everything" spiral.
- **Raise max_output_tokens** — Many harnesses default to 32K max output tokens as a cost guardrail. For local inference, raising to 64K+ is safe and resolves silent mid-generation stops.

### Inference Framework Selection

| Framework | Best For | Flash-Next Support | Notes |
|---|---|---|---|
| **llama.cpp ★** | Single GPU, CPU, GGUF | Merged (PR #27742) | Primary option for local; MTP coming; SSD offload requires custom build |
| **Unsloth Desktop** | Single NVIDIA GPU | Via llama.cpp backend | GUI wrapper; integrates well with DSH and Pi harness |
| **SGLang** | Multi-GPU, production | Host-memory offload ready | Best KV cache management for Flash-Next; H200 deployment tested |
| **vLLM** | Multi-GPU, high throughput | Implementation guide available | TP4+EP on 4×5090; 2500 t/s at 128 concurrent streams |
| **MLX** | Apple Silicon (M-series) | eQ4-MTP-MLX available | M3 Max: 17 t/s at 200K ctx; M5 Max: 40+ t/s |
| **ROCm + llama.cpp** | AMD GPU / Strix Halo | GTT page cache tested | N-gram table stays in page cache automatically; 10–32 t/s |
| **Ollama** | Ease of use | Limited (version lag) | Fine for 27B Q4+; may lack Flash-Next support initially |

### Thinking Level Reference

At medium thinking setting, Flash-Next spent ~75% of tokens thinking vs. 25% on output. For daily interactive use on 27B, **low is the empirically validated sweet spot** for complex, multi-step tasks.

### License Considerations

**Qwen Community 1.0 license (Flash-Next):** Free for personal, research, and commercial use under 100M monthly active users or $20M/month revenue. If your product's primary value is *reselling model access* (AI coding assistant, office assistant subscription), a separate commercial license from Alibaba/Qwen is required. Internal tools are exempt.

---

## 10 — Recommendations by Use Case

### ✦ This System — RTX 3090 24GB (Daily Driver)
**Quant: Q4_K_XL**

Start here: Q4_K_XL (~18 GB) leaves ~6 GB for KV cache — comfortable 100K context at an estimated ~50 t/s. Q5_K_XL (~22 GB) fits but leaves only ~2 GB KV headroom; use it for highest quality at shorter context. UD-IQ3_XXS (~12 GB) maximizes context window.

**Build llama.cpp from source** with `-DGGML_AVX512=OFF` — pre-built wheels SIGILL on Skylake. Enable MTP (2–4× speed), K=Q5_0/V=Q4_0 KV split. Flash-Next is not viable on this hardware.

### 16 GB VRAM · Daily Driver
**Quant: Q3 KXL + MTP**

Best balance for single 16GB cards. Q3 KXL fits fully with room for context. Add `repeat_penalty=1.05`, cap context at 65K, use DeepSeek Harness or Pi on low thinking. RTX 4090/5080 upgrades bring 4–8× speed gains.

### 16 GB VRAM · Extended Context
**Quant: UD-IQ3_XXS + KV Q4_0**

Unsloth ultra-dynamic quant fits in ~12GB, leaving room for KV cache. On a 4090 16GB: 80K+ context at 40+ t/s. Enable Q5_0/Q4_0 KV split for best quality-to-context ratio.

### 24–32 GB VRAM
**Quant: Q4 or Q5_K_XL + KV Q8**

RTX 3090 through 5090. Q5_K_XL at 131K context with Q8 KV cache on 5090 yields 95–102 t/s — excellent for agentic coding workflows. DSH at low thinking outperforms all other harnesses tested.

### 128 GB Unified Memory
**Flash-Next Q4 (Dedicated Machine)**

Strix Halo or Apple M5 Max. Flash-Next at Q4 IQ4_XS runs at ~20–40 t/s. Not a daily driver alongside other workloads — the N-gram table page-caches automatically on AMD GTT hardware. M5 Max is the strongest consumer option.

### Multi-GPU · Production
**vLLM TP4+EP or SGLang**

4× RTX 5090: 2500 t/s at 128 concurrent streams via vLLM. 4× H200: SGLang with host-memory offload frees 23GB/card and boosts KV context capacity 78%. Both handle Flash-Next in production.

### Cloud Alternative
**Qwen Cloud (hosted Flash)**

15 cents/M input, 47 cents/M output, 1M token context, built-in tools. Not the same weights (hosted production vs. open preview), but intelligence-equivalent and practically the cheapest frontier-tier option available.

---

## Harness Decision Tree

- **Complex coding tasks (new projects, game engines, agentic loops)** — DeepSeek Harness on low thinking. Use /goal for large projects to auto-spawn sub-agents. Install community plugins for sound notifications, dependency DAG generation, and circuit-breaker FSM.
- **Existing codebase work (feature additions, refactors)** — Pi Coding Agent or DeepSeek Harness both work. Pi's terminal interface feels natural for developers. Consider OpenCode (not tested in this review but frequently recommended) as a likely strong alternative.
- **Multi-agent orchestration / persistent memory** — Hermes Agent excels at its intended use case (persistent agents, Discord integration, multi-model routing). Its poor performance in this review reflects using it as a coding harness — that's not its primary design goal.

---

*Sources: Luke's Dev Lab (RTX 2000 Ada / Ridge review), Jose Romero (Strix Halo / Flash-Next), Cloud Codes (architecture deep dive), James Layne (RTX 5090 harness comparison)*
