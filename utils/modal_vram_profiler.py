"""
Modal VRAM profiler — estimates peak VRAM for a given model + configuration
and recommends the smallest-sufficient Modal GPU from the available set.

Architecture (matches modal_gpu_profiler_action_plan.md):
  §1 / §4   PRESET_METADATA  — known primitives for bundled models
  §4a       fetch_hf_profile — derives same primitives from HF config.json + model-info
  §2        estimate_vram    — four-term formula: weights + vision + KV + activations
  §3        recommend_gpu    — ascending-VRAM smallest-sufficient card mapping
  §5        record_measured_peak / load_measured_peaks — feedback loop cache
  Profile resolution order: measured peak > preset > local cache > HF fetch

Public API:
  get_profile(key_or_repo, data_dir, hf_token, auto_fetch) → dict
  fetch_hf_profile(repo_id, hf_token)                       → dict
  estimate_vram(profile, quant, ...)                         → VramEstimate
  recommend_gpu(peak_gb, priority)                           → GpuRecommendation
  get_recommendation(key_or_repo, *, quantize, ...)          → JSON-serializable dict
  record_measured_peak(data_dir, config_key, peak_gb)        → None

No ComfyUI dependencies — pure stdlib + optional huggingface_hub.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ── §3 GPU table ───────────────────────────────────────────────────────────────

@dataclass
class GpuSpec:
    name:         str    # Modal gpu= string
    vram_gb:      float
    cost_per_hr:  float  # approximate Modal base rate (preemptible, region-agnostic)
    label:        str    # human-readable display name

# Ascending by VRAM.  A100-40 has LESS VRAM than L40S (40 < 48) and costs MORE
# — excluded from the default cost walk; only surfaced when priority="bandwidth".
_GPU_TABLE: list[GpuSpec] = [
    GpuSpec("T4",    16.0,  0.59, "NVIDIA T4 (16 GB)"),
    GpuSpec("L4",    24.0,  0.80, "NVIDIA L4 (24 GB)"),
    GpuSpec("A10G",  24.0,  1.10, "NVIDIA A10G (24 GB)"),
    GpuSpec("L40S",  48.0,  1.95, "NVIDIA L40S (48 GB)"),
    GpuSpec("A100",  40.0,  2.10, "NVIDIA A100-40 (40 GB)"),
]

# Default ascending walk: T4 → L4 → L40S.
# A10G same VRAM as L4, higher cost → prefer L4 by default.
# A100-40 dominated by L40S on capacity → excluded from default walk.
_DEFAULT_GPU_WALK: list[GpuSpec] = [g for g in _GPU_TABLE if g.name in ("T4", "L4", "L40S")]

# ── Quant → bytes per parameter ──────────────────────────────────────────────

_BYTES_PER_PARAM: dict[str, float] = {
    "bf16":   2.00,
    "fp16":   2.00,
    "fp32":   4.00,
    "fp8":    1.00,
    "int8":   1.00,
    "nf4":    0.55,   # bitsandbytes NF4: 4-bit + per-group FP16 scales (~10% overhead)
    "awq":    0.52,   # AWQ 4-bit, slightly tighter packing
    "gptq":   0.54,
    "nvfp4":  0.55,   # NVIDIA FP4 (TensorRT-LLM / modelopt): 4-bit + group scales
    "fp4":    0.55,   # generic FP4
    "int4":   0.52,   # INT4 (similar packing to AWQ)
    "gguf":   0.55,   # approximate for Q4_K_M (common default); may vary widely
    "ggml":   0.55,
}
_DEFAULT_BPP = 2.00   # treat unknown quant as BF16 (conservative)

# ── §4 Preset metadata ────────────────────────────────────────────────────────
# Only primitives that cannot be recomputed.  Values are best-effort; they can
# be refreshed for a given preset by running fetch_hf_profile on its repo.
#
# Fields:
#   repo             str    HF repo ID
#   arch             str    internal arch tag (qwen_vl / qwen3_vl / generic)
#   params_b         float  total params in billions (language + vision tower)
#   n_layers         int    transformer hidden layers (language tower)
#   kv_heads         int    number of KV heads (post-GQA)
#   head_dim         int    per-head dimension
#   hidden_size      int    language tower hidden dimension
#   is_vl            bool   has a vision encoder
#   vision_tower_gb  float  FP16 size of vision encoder (always full-precision)
#   native_video     bool   supports frame-sequence video input
#   pre_quantized    str|None  quant format if pre-quantized ("awq", "gptq", None)
#   is_moe           bool   all expert weights resident (use total params, not active)
#   _source          str    "preset" for these; "hf_derived" for fetched profiles

_PRESET_METADATA: dict[str, dict] = {
    "qwen3-vl-8b": {
        "repo":            "Qwen/Qwen3-VL-8B-Instruct",
        "arch":            "qwen3_vl",
        "params_b":        8.03,
        "n_layers":        28,
        "kv_heads":        4,
        "head_dim":        128,
        "hidden_size":     3584,
        "is_vl":           True,
        "vision_tower_gb": 1.2,
        "native_video":    True,
        "pre_quantized":   None,
        "is_moe":          False,
        "_source":         "preset",
    },
    "qwen2.5-vl-7b": {
        "repo":            "Qwen/Qwen2.5-VL-7B-Instruct",
        "arch":            "qwen_vl",
        "params_b":        7.07,
        "n_layers":        28,
        "kv_heads":        4,
        "head_dim":        128,
        "hidden_size":     3584,
        "is_vl":           True,
        "vision_tower_gb": 1.2,
        "native_video":    True,
        "pre_quantized":   None,
        "is_moe":          False,
        "_source":         "preset",
    },
    "qwen2.5-vl-32b-awq": {
        "repo":            "Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
        "arch":            "qwen_vl",
        "params_b":        32.5,
        "n_layers":        64,
        "kv_heads":        8,
        "head_dim":        128,
        "hidden_size":     5120,
        "is_vl":           True,
        "vision_tower_gb": 1.2,
        "native_video":    True,
        "pre_quantized":   "awq",
        "is_moe":          False,
        "_source":         "preset",
    },
    "qwen2.5-vl-3b": {
        "repo":            "Qwen/Qwen2.5-VL-3B-Instruct",
        "arch":            "qwen_vl",
        "params_b":        3.09,
        "n_layers":        28,
        "kv_heads":        2,
        "head_dim":        128,
        "hidden_size":     2048,
        "is_vl":           True,
        "vision_tower_gb": 1.2,
        "native_video":    True,
        "pre_quantized":   None,
        "is_moe":          False,
        "_source":         "preset",
    },
    "qwen2.5-omni-7b": {
        "repo":            "Qwen/Qwen2.5-Omni-7B",
        "arch":            "qwen_vl",
        "params_b":        7.07,
        "n_layers":        28,
        "kv_heads":        4,
        "head_dim":        128,
        "hidden_size":     3584,
        "is_vl":           True,
        "vision_tower_gb": 1.3,    # includes audio encoder
        "native_video":    True,
        "pre_quantized":   None,
        "is_moe":          False,
        "_source":         "preset",
    },
    "gemma3-4b": {
        "repo":            "google/gemma-3-4b-it",
        "arch":            "generic",
        "params_b":        4.30,
        "n_layers":        34,
        "kv_heads":        4,
        "head_dim":        256,
        "hidden_size":     2560,
        "is_vl":           True,    # Gemma 3 handles images via SigLIP encoder
        "vision_tower_gb": 0.40,   # SigLIP ViT-SO400M ≈ 400 MB
        "native_video":    False,
        "pre_quantized":   None,
        "is_moe":          False,
        "_source":         "preset",
    },
}

# ── §2 Four-term VRAM estimator ───────────────────────────────────────────────

_SAFETY_MARGIN        = 1.18    # 18% for allocator fragmentation + serving buffers
_TOKENS_PER_FRAME     = 512     # visual tokens per frame at typical VL resolution
_KV_CACHE_BF16_BYTES  = 2       # bytes per KV element (BF16)
_KV_CACHE_FP8_BYTES   = 1       # bytes per KV element (FP8)


@dataclass
class VramEstimate:
    weights_gb:     float
    vision_gb:      float
    kv_gb:          float
    activation_gb:  float
    subtotal_gb:    float
    peak_gb:        float    # subtotal × safety margin
    margin:         float
    context_length: int
    frame_budget:   int
    quant:          str


def estimate_vram(
    profile: dict,
    quant: str = "nf4",
    context_length: int = 8192,
    kv_cache_quant: str = "bf16",
    modality: str = "image",
    frame_budget: int = 20,
    margin: float = _SAFETY_MARGIN,
) -> VramEstimate:
    """Estimate peak VRAM for the given profile + run configuration.

    profile        — from _PRESET_METADATA or fetch_hf_profile()
    quant          — weight quant: "bf16", "nf4", "awq", "int8", …
    context_length — max tokens the model will process (language + vision)
    kv_cache_quant — "bf16" (default) or "fp8" (halves the KV term)
    modality       — "image", "video", or "text"
    frame_budget   — number of frames/images passed in (video or contact sheet)
    margin         — safety factor for allocator and runtime overhead
    """
    # Pre-quantized models ignore the quant override.
    pre_q = profile.get("pre_quantized")
    if pre_q:
        quant = str(pre_q)

    bpp = _BYTES_PER_PARAM.get(quant, _DEFAULT_BPP)

    # 1. Weights — the term that quantization directly controls.
    params_b   = float(profile.get("params_b", 7.0))
    weights_gb = params_b * bpp

    # 2. Vision tower — always FP16 regardless of weight quant.
    vision_gb = float(profile.get("vision_tower_gb", 0.0)) if profile.get("is_vl") else 0.0

    # 3. KV cache — the context-scaling term.
    #    kv_bytes_per_token = n_layers × 2 (K+V) × kv_heads × head_dim × bytes_per_elem
    n_layers = int(profile.get("n_layers", 32))
    kv_heads = int(profile.get("kv_heads", 8))
    head_dim = int(profile.get("head_dim", 128))
    kv_elem  = _KV_CACHE_FP8_BYTES if kv_cache_quant == "fp8" else _KV_CACHE_BF16_BYTES
    kv_bpt   = n_layers * 2 * kv_heads * head_dim * kv_elem   # bytes per token
    kv_gb    = (kv_bpt * context_length) / (1024 ** 3)

    # 4. Activations — relevant for VL inference; small with flash attention.
    #    Peak activation ≈ one layer's intermediate representation:
    #    frame_budget × tokens_per_frame × hidden_size × 2 bytes (BF16)
    if modality in ("image", "video") and profile.get("is_vl"):
        hidden     = int(profile.get("hidden_size", int(params_b * 450)))
        tpf        = int(profile.get("tokens_per_frame", _TOKENS_PER_FRAME))
        act_tokens = frame_budget * tpf
        act_gb     = (act_tokens * hidden * 2) / (1024 ** 3)
    else:
        act_gb = 0.05   # flat headroom for text-only inference

    subtotal_gb = weights_gb + vision_gb + kv_gb + act_gb
    peak_gb     = subtotal_gb * margin

    return VramEstimate(
        weights_gb=round(weights_gb, 2),
        vision_gb=round(vision_gb, 2),
        kv_gb=round(kv_gb, 3),
        activation_gb=round(act_gb, 3),
        subtotal_gb=round(subtotal_gb, 2),
        peak_gb=round(peak_gb, 2),
        margin=margin,
        context_length=context_length,
        frame_budget=frame_budget,
        quant=quant,
    )


# ── §3 GPU mapper ─────────────────────────────────────────────────────────────

@dataclass
class GpuRecommendation:
    gpu:           GpuSpec | None   # None → won't fit on any card
    peak_gb:       float
    headroom_gb:   float            # negative when won't fit
    cost_per_hr:   float
    warning:       str              # non-empty on tight fit or won't-fit
    alt_gpu:       GpuSpec | None   # next tier up for more headroom


def recommend_gpu(
    peak_gb: float,
    gpu_walk: list[GpuSpec] | None = None,
    priority: str = "cost",
) -> GpuRecommendation:
    """Return the smallest-sufficient GPU for the given peak VRAM.

    priority="cost"       default: T4 → L4 → L40S
    priority="throughput" swaps L4 → A10G at the 24 GB tier (higher compute)
    priority="bandwidth"  adds A100-40 as an option (higher memory bandwidth)
    """
    walk = list(gpu_walk or _DEFAULT_GPU_WALK)

    if priority == "throughput":
        walk = [GpuSpec("A10G", 24.0, 1.10, "NVIDIA A10G (24 GB)") if g.name == "L4" else g
                for g in walk]
    elif priority == "bandwidth":
        walk = sorted(walk + [GpuSpec("A100", 40.0, 2.10, "NVIDIA A100-40 (40 GB)")],
                      key=lambda g: g.vram_gb)

    ordered    = sorted(walk, key=lambda g: g.vram_gb)
    recommended = next((g for g in ordered if g.vram_gb >= peak_gb), None)

    if recommended is None:
        largest = max(ordered, key=lambda g: g.vram_gb)
        return GpuRecommendation(
            gpu=None,
            peak_gb=round(peak_gb, 1),
            headroom_gb=round(largest.vram_gb - peak_gb, 1),
            cost_per_hr=0.0,
            warning=(
                f"Estimated {peak_gb:.1f} GB exceeds the largest available card "
                f"({largest.label}). Reduce context length, switch to 4-bit quant, "
                "or use a larger GPU tier."
            ),
            alt_gpu=None,
        )

    headroom = recommended.vram_gb - peak_gb
    tight    = headroom < (recommended.vram_gb * 0.10)   # < 10% headroom
    idx      = next((i for i, g in enumerate(ordered) if g.name == recommended.name), -1)
    alt_gpu  = ordered[idx + 1] if 0 <= idx < len(ordered) - 1 else None

    return GpuRecommendation(
        gpu=recommended,
        peak_gb=round(peak_gb, 1),
        headroom_gb=round(headroom, 1),
        cost_per_hr=recommended.cost_per_hr,
        warning="Headroom < 10% — consider the next tier for stability." if tight else "",
        alt_gpu=alt_gpu,
    )


# ── §4a HF profile fetcher ────────────────────────────────────────────────────

# architectures[0] (lowercased) → internal arch tag
_ARCH_MAP: dict[str, str] = {
    "qwen2_5_vlforconditionalgeneration":        "qwen_vl",
    "qwen2vlforconditionalgeneration":            "qwen_vl",
    "qwen3vlforconditionalgeneration":            "qwen3_vl",
    # Qwen3-family multimodal models whose class name doesn't follow the *VLFor* pattern
    # (e.g. Qwen3.8 uses Qwen3_5ForConditionalGeneration with model_type="qwen3_5")
    "qwen3_5forconditionalgeneration":            "qwen3_vl",
    "llavaforconditionalgeneration":              "llava",
    "llavaonevisionforconditionalgeneration":     "llava",
    "llavamistralforconditionalgeneration":       "llava",
    "internvlchatmodel":                          "generic",
    "minicpmv":                                   "generic",
    "phi3vforconditionalgeneration":              "generic",
}


def fetch_hf_profile(repo_id: str, hf_token: str | None = None) -> dict:
    """Fetch config.json and Hub model-info from HuggingFace and derive a VRAM profile.

    No model weights are downloaded.  Completes in <1 s for public repos.
    Raises RuntimeError if the repo is inaccessible or uses an unsupported format (GGUF).
    """
    repo_id = repo_id.strip()
    config  = _fetch_config(repo_id, hf_token)

    arch_raw = (config.get("architectures") or [""])[0].lower()
    arch     = _ARCH_MAP.get(arch_raw, "generic")

    # VL: presence of vision_config is the definitive signal
    is_vl = ("vision_config" in config) or any(
        kw in arch_raw for kw in ("vl", "vision", "llava", "visual", "multimodal")
    )

    pre_quantized = _detect_pre_quant(config, repo_id)
    is_moe        = "num_local_experts" in config

    n_layers  = int(config.get("num_hidden_layers", 32))
    num_heads = int(config.get("num_attention_heads", 32))
    kv_heads  = int(config.get("num_key_value_heads", num_heads))
    hidden    = int(config.get("hidden_size", 4096))
    head_dim  = int(config.get("head_dim") or (hidden // max(num_heads, 1)))

    params_b        = _fetch_params_b(repo_id, config, hf_token)
    vision_tower_gb = _estimate_vision_tower_gb(config) if is_vl else 0.0

    return {
        "repo":            repo_id,
        "arch":            arch,
        "params_b":        round(params_b, 3),
        "n_layers":        n_layers,
        "kv_heads":        kv_heads,
        "head_dim":        head_dim,
        "hidden_size":     hidden,
        "is_vl":           is_vl,
        "vision_tower_gb": round(vision_tower_gb, 2),
        "native_video":    arch in ("qwen_vl", "qwen3_vl"),
        "pre_quantized":   pre_quantized,
        "is_moe":          is_moe,
        "_source":         "hf_derived",
        "_fetched_at":     time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def _fetch_config(repo_id: str, hf_token: str | None) -> dict:
    """Download config.json via huggingface_hub (with auth + cache) or plain HTTPS."""
    headers: dict = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    try:
        import huggingface_hub as _hf   # noqa: F401
        path = _hf.hf_hub_download(repo_id, "config.json",
                                    token=hf_token, local_files_only=False)
        with open(path) as f:
            return json.load(f)
    except Exception:
        pass

    # Fallback: anonymous HTTPS (works for public repos without the library)
    try:
        import urllib.request as _req
        url = f"https://huggingface.co/{repo_id}/resolve/main/config.json"
        req = _req.Request(url, headers=headers)
        with _req.urlopen(req, timeout=15) as resp:
            raw = resp.read()
            if b"gguf" in raw[:200].lower():
                raise RuntimeError(
                    f"{repo_id!r} appears to be a GGUF repo — config.json is absent or "
                    "in a non-standard format. GGUF profiling is not supported."
                )
            return json.loads(raw)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Could not fetch config.json for {repo_id!r}: {exc}. "
            "Verify the repo ID is correct and the repo is public "
            "(or set HF_TOKEN / HUGGINGFACE_HUB_TOKEN in the environment)."
        ) from exc


def _fetch_params_b(repo_id: str, config: dict, hf_token: str | None) -> float:
    """Return total parameter count in billions, preferring the Hub model-info API."""
    # 1. Hub API — returns exact counts for safetensors repos
    try:
        import huggingface_hub as _hf
        info   = _hf.model_info(repo_id, token=hf_token)
        sf     = getattr(info, "safetensors", None)
        if sf:
            params = getattr(sf, "parameters", None) or {}
            total  = sum(params.values()) if isinstance(params, dict) else 0
            if total > 0:
                return total / 1e9
    except Exception:
        pass

    # 2. Estimate from architecture config
    n_layers  = int(config.get("num_hidden_layers", 32))
    hidden    = int(config.get("hidden_size", 4096))
    inter     = int(config.get("intermediate_size", hidden * 4))
    num_heads = int(config.get("num_attention_heads", 32))
    kv_heads  = int(config.get("num_key_value_heads", num_heads))
    head_dim  = int(config.get("head_dim") or (hidden // max(num_heads, 1)))
    vocab     = int(config.get("vocab_size", 32000))

    # Attention: Q + K + V + O projections
    kv_dim   = kv_heads * head_dim
    attn_p   = hidden * hidden + hidden * kv_dim + hidden * kv_dim + hidden * hidden
    # FFN: SwiGLU/gated (3 matrices) or standard (2 matrices)
    ffn_p    = (3 if inter > 2 * hidden else 2) * hidden * inter
    embed_p  = vocab * hidden * 2   # embedding + lm_head (may be tied, so conservative)

    total = n_layers * (attn_p + ffn_p) + embed_p
    return total / 1e9


def _detect_pre_quant(config: dict, repo_id: str) -> str | None:
    """Return the pre-quantization format string, or None if not pre-quantized."""
    # 1. Standard quantization_config block (HuggingFace / bitsandbytes / AWQ style)
    qcfg  = config.get("quantization_config") or {}
    qtype = (qcfg.get("quant_type") or qcfg.get("quant_algo") or "").lower()
    if "awq"   in qtype: return "awq"
    if "gptq"  in qtype: return "gptq"
    if "nvfp4" in qtype: return "nvfp4"
    if "fp4"   in qtype: return "fp4"
    if "fp8"   in qtype: return "fp8"
    if "nf4"   in qtype: return "nf4"
    if "bnb"   in qtype: return "nf4"
    if "int4"  in qtype: return "int4"
    if "int8"  in qtype: return "int8"
    if qtype:            return qtype   # pass through any other declared scheme

    # 2. NVIDIA modelopt-style: top-level "quantization" key (TRT-LLM, modelopt)
    modelopt_q = config.get("quantization") or {}
    if isinstance(modelopt_q, dict):
        algo = (modelopt_q.get("quant_algo") or modelopt_q.get("format") or "").lower()
        if algo:
            if "nvfp4" in algo: return "nvfp4"
            if "fp4"   in algo: return "fp4"
            if "fp8"   in algo: return "fp8"
            if "awq"   in algo: return "awq"
            if algo:            return algo

    # 3. Name-based fallback — covers models whose config has no quantization block
    low = repo_id.lower()
    # Ordered longest-first so "nvfp4" matches before "fp4"
    for suffix, result in (
        ("-nvfp4", "nvfp4"), ("-fp4",  "fp4"),  ("-fp8",  "fp8"),
        ("-int4",  "int4"),  ("-int8", "int8"),
        ("-awq",   "awq"),   ("-gptq", "gptq"),
        ("-gguf",  "gguf"),  ("-ggml", "ggml"),
        ("-bnb",   "nf4"),
    ):
        if suffix in low:
            return result

    return None


def _estimate_vision_tower_gb(config: dict) -> float:
    """Estimate the FP16 memory of the vision encoder from vision_config."""
    vcfg = config.get("vision_config") or {}
    if not vcfg:
        return 1.2   # conservative default

    vhidden = int(vcfg.get("hidden_size", 1152))
    vlayers = int(vcfg.get("num_hidden_layers", 27))
    vinter  = int(vcfg.get("intermediate_size", vhidden * 4))
    # Attention + FFN per layer × FP16 (2 bytes/param)
    params  = vlayers * (4 * vhidden * vhidden + 3 * vhidden * vinter)
    return max(0.3, round(params * 2 / (1024 ** 3), 2))


# ── §5 Local caches ────────────────────────────────────────────────────────────

def _profile_cache_path(data_dir: str) -> str:
    return os.path.join(data_dir, "modal_model_profiles.json")


def _measured_peaks_path(data_dir: str) -> str:
    return os.path.join(data_dir, "modal_measured_peaks.json")


def load_profile_cache(data_dir: str) -> dict[str, dict]:
    """Load HF-derived profiles from the local cache file."""
    path = _profile_cache_path(data_dir)
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def save_profile_cache(data_dir: str, cache: dict[str, dict]) -> None:
    path = _profile_cache_path(data_dir)
    try:
        with open(path, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception as exc:
        logger.warning("Could not save modal_model_profiles.json: %s", exc)


def load_measured_peaks(data_dir: str) -> dict[str, float]:
    """Load measured peak VRAM values keyed by serialized config_key tuples."""
    path = _measured_peaks_path(data_dir)
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def record_measured_peak(data_dir: str, config_key: tuple, peak_gb: float) -> None:
    """Store a GPU-measured peak VRAM value after a real inference run.

    config_key = (model_id, quant, context_length, kv_cache_quant, modality, frame_budget)
    """
    peaks = load_measured_peaks(data_dir)
    peaks[str(config_key)] = round(peak_gb, 2)
    path = _measured_peaks_path(data_dir)
    try:
        with open(path, "w") as f:
            json.dump(peaks, f, indent=2)
    except Exception as exc:
        logger.warning("Could not save modal_measured_peaks.json: %s", exc)


# ── Profile resolution chain ──────────────────────────────────────────────────

def get_profile(
    key_or_repo: str,
    data_dir: str | None = None,
    hf_token: str | None = None,
    auto_fetch: bool = True,
) -> dict:
    """Resolve a VRAM profile for the given preset key or HF repo ID.

    Resolution order:
      1. Built-in preset (_PRESET_METADATA)
      2. Local profile cache (modal_model_profiles.json in data_dir)
      3. fetch_hf_profile() on cache miss — result written to cache

    Raises KeyError if not found and auto_fetch=False.
    Raises RuntimeError from fetch_hf_profile on network/format failure.
    """
    key_or_repo = key_or_repo.strip()

    # 1. Preset (by key)
    if key_or_repo in _PRESET_METADATA:
        return _PRESET_METADATA[key_or_repo]

    # Also resolve a full repo ID that matches a known preset's repo field
    for p in _PRESET_METADATA.values():
        if p.get("repo") == key_or_repo:
            return p

    # 2. Local cache
    if data_dir:
        cache = load_profile_cache(data_dir)
        if key_or_repo in cache:
            return cache[key_or_repo]

    # 3. Fetch from HF
    if not auto_fetch:
        raise KeyError(f"No profile found for {key_or_repo!r}")

    profile = fetch_hf_profile(key_or_repo, hf_token=hf_token)
    if data_dir:
        cache = load_profile_cache(data_dir)
        cache[key_or_repo] = profile
        save_profile_cache(data_dir, cache)
    return profile


# ── End-to-end convenience ────────────────────────────────────────────────────

def get_recommendation(
    key_or_repo: str,
    *,
    quantize: bool = True,
    context_length: int = 8192,
    frame_budget: int = 20,
    modality: str = "image",
    data_dir: str | None = None,
    hf_token: str | None = None,
    priority: str = "cost",
) -> dict:
    """Resolve profile → estimate VRAM → recommend GPU.

    Returns a JSON-serializable dict for backend_status() and the UI.
    On any failure returns {"available": False, "error": "..."}.
    """
    try:
        profile = get_profile(key_or_repo, data_dir=data_dir, hf_token=hf_token)
    except Exception as exc:
        return {"available": False, "error": str(exc)}

    pre_q = profile.get("pre_quantized")
    quant = str(pre_q) if pre_q else ("nf4" if quantize else "bf16")

    # Config key for the measured-peak lookup (§5)
    config_key    = (key_or_repo, quant, context_length, "bf16", modality, frame_budget)
    peak_override = None
    if data_dir:
        peaks        = load_measured_peaks(data_dir)
        peak_override = peaks.get(str(config_key))

    est  = estimate_vram(profile, quant=quant, context_length=context_length,
                         modality=modality, frame_budget=frame_budget)
    peak = peak_override if peak_override is not None else est.peak_gb
    rec  = recommend_gpu(peak, priority=priority)

    # Alt-quant comparison (only when not pre-quantized)
    alt_summary: dict | None = None
    if not pre_q:
        alt_quant = "bf16" if quant == "nf4" else "nf4"
        alt_est   = estimate_vram(profile, quant=alt_quant, context_length=context_length,
                                  modality=modality, frame_budget=frame_budget)
        alt_rec   = recommend_gpu(alt_est.peak_gb, priority=priority)
        alt_summary = {
            "quant":    alt_quant,
            "peak_gb":  alt_est.peak_gb,
            "gpu":      alt_rec.gpu.name if alt_rec.gpu else None,
            "cost":     alt_rec.cost_per_hr,
        }

    return {
        "available":      True,
        "profile_source": profile.get("_source", "unknown"),
        "params_b":       profile.get("params_b"),
        "is_vl":          profile.get("is_vl", False),
        "pre_quantized":  pre_q,       # None, or the format string ("nvfp4", "awq", …)
        "quant":          quant,
        "estimate": {
            "weights_gb":    est.weights_gb,
            "vision_gb":     est.vision_gb,
            "kv_gb":         est.kv_gb,
            "activation_gb": est.activation_gb,
            "peak_gb":       round(peak, 2),
            "measured":      peak_override is not None,
        },
        "recommendation": {
            "gpu":         rec.gpu.name if rec.gpu else None,
            "gpu_label":   rec.gpu.label if rec.gpu else None,
            "vram_gb":     rec.gpu.vram_gb if rec.gpu else None,
            "headroom_gb": rec.headroom_gb,
            "cost_per_hr": rec.cost_per_hr,
            "warning":     rec.warning,
            "alt_gpu":     rec.alt_gpu.name if rec.alt_gpu else None,
        },
        "alt_quant":  alt_summary,
        "config_key": list(config_key),
    }
