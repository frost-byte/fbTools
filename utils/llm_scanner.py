"""LLM model scanner for the Composition Editor assistant.

Scans ComfyUI's registered LLMs directories (and extra_model_paths) for
GGUF and HuggingFace-format vision-language models.  No optional
dependencies — pure filesystem reads only.

Capability flags returned per model
------------------------------------
supports_vision : bool
    Model can accept image inputs.
    GGUF: mmproj-*.gguf file present alongside main GGUF.
    HF:   architectures list contains a known vision class, or
          vision_config key present in config.json, or
          preprocessor_config.json exists.

supports_video : bool
    Always True when supports_vision is True (frame sampling is used
    with any image-capable model to analyse video clips).

native_video : bool
    Model understands temporal video sequences natively without frame
    sampling.  Qwen2-VL, Qwen2.5-VL, Qwen2.5-Omni, and LLaVA-Next-Video.

quant_type : str | None
    Quantization format detected from config.json quantization_config:
    "awq", "gptq", "bitsandbytes", "fp8", or None (unquantized / unknown).

quant_requires : str | None
    Python package required to load this quant type, or None if no extra
    package is needed (BnB is already present; unquantized HF needs nothing).

format : "gguf" | "hf"
    GGUF models need llama-cpp-python.
    HF models need transformers.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

# ── HuggingFace architectures that indicate vision capability ─────────────────

_VISION_ARCHS: set[str] = {
    "LlavaForConditionalGeneration",
    "LlavaNextForConditionalGeneration",
    "LlavaNextVideoForConditionalGeneration",
    "Qwen2VLForConditionalGeneration",
    "Qwen2_5_VLForConditionalGeneration",
    "PaliGemmaForConditionalGeneration",
    "Gemma3ForConditionalGeneration",
    "InternVLChatModel",
    "MiniCPMV",
    "MoondreamForConditionalGeneration",
    "Idefics2ForConditionalGeneration",
    "Idefics3ForConditionalGeneration",
    "Phi3VForCausalLM",
    "Phi4VForCausalLM",
    "BakLlavaForConditionalGeneration",
    "ChameleonForConditionalGeneration",
}

# Architectures that process video as a temporal sequence natively
_NATIVE_VIDEO_ARCHS: set[str] = {
    "Qwen2VLForConditionalGeneration",
    "Qwen2_5_VLForConditionalGeneration",
    "Qwen2_5OmniForConditionalGeneration",
    "LlavaNextVideoForConditionalGeneration",
}

# quant_type → (pip package name, import name)
# pip name shown in the warning; import name used to check if it's already installed.
_QUANT_REQUIRES: dict[str, tuple[str, str] | None] = {
    "awq":          ("autoawq",  "awq"),        # pip: autoawq  → import awq
    "gptq":         ("auto-gptq", "auto_gptq"), # pip: BUILD_CUDA_EXT=0 pip install auto-gptq
    "bitsandbytes": None,   # already installed
    "fp8":          None,   # torch-native on Ada/Hopper
}

# model_type values in config.json that imply vision without explicit architecture
_VISION_MODEL_TYPES: set[str] = {
    "llava", "llava_next", "llava_next_video",
    "qwen2_vl", "qwen2_5_vl",
    "paligemma", "paligemma2",
    "internvl_chat",
    "moondream1",
    "idefics", "idefics2", "idefics3",
    "phi3_v", "phi4_v",
    "bakllava",
    "chameleon",
    "gemma3",  # multimodal gemma3
}


# ── Capability labels for the UI ──────────────────────────────────────────────

def _detect_quant(config: dict) -> tuple[str | None, str | None]:
    """Return (quant_type, quant_requires) from a config.json dict.

    quant_requires is the pip package name needed to load this quant format,
    or None if no extra package is needed (either built-in or already installed).
    """
    qc = config.get("quantization_config") or {}
    qt = (
        qc.get("quant_type")
        or qc.get("quantization_type")
        or qc.get("quant_method")
        or (config.get("quantization_method"))
    )
    if qt:
        qt = qt.lower()
    entry = _QUANT_REQUIRES.get(qt) if qt else None
    if entry is None:
        return qt, None
    pip_name, import_name = entry
    try:
        __import__(import_name)
        return qt, None   # already installed — no warning needed
    except ImportError:
        return qt, pip_name


def capability_tags(model: dict) -> list[str]:
    """Return short display tags for a model dict."""
    if not model.get("supports_vision"):
        tags = ["🔤 Text only"]
    else:
        tags = ["📷 Vision"]
        if model.get("native_video"):
            tags.append("🎬 Video (native)")
        else:
            tags.append("🎬 Video (frames)")
    qt = model.get("quant_type")
    if qt:
        req = model.get("quant_requires")
        suffix = f" ⚠️ needs {req}" if req else ""
        tags.append(f"⚡ {qt.upper()}{suffix}")
    return tags


def capability_note(model: dict) -> str:
    """One-line human-readable capability summary."""
    parts = []
    if not model.get("supports_vision"):
        parts.append("Text only — cannot analyse images or video.")
    else:
        parts.append("Accepts images.")
        if model.get("native_video"):
            parts.append("Native video understanding (temporal).")
        else:
            parts.append("Video via frame sampling (keyframes only).")
    qt = model.get("quant_type")
    if qt:
        req = model.get("quant_requires")
        if req:
            parts.append(f"Quantization: {qt.upper()} — requires `pip install {req}`.")
        else:
            parts.append(f"Quantization: {qt.upper()} (ready to load).")
    return " ".join(parts)


# ── GGUF directory scanner ─────────────────────────────────────────────────────

def _scan_gguf_dir(dirpath: str, name: str) -> dict | None:
    """Return a model descriptor if dirpath contains a GGUF model, else None."""
    try:
        entries = os.listdir(dirpath)
    except OSError:
        return None

    gguf_files = [f for f in entries if f.endswith(".gguf") and not _is_mmproj(f)]
    mmproj_files = [f for f in entries if f.endswith(".gguf") and _is_mmproj(f)]

    if not gguf_files:
        return None

    # Pick the largest GGUF as the primary (avoids picking an aux shard)
    main_file = max(gguf_files, key=lambda f: os.path.getsize(os.path.join(dirpath, f)))
    mmproj_file = mmproj_files[0] if mmproj_files else None
    supports_vision = mmproj_file is not None

    main_path = os.path.join(dirpath, main_file)
    size_mb = round(os.path.getsize(main_path) / (1024 * 1024))

    return {
        "id":             _make_id(name),
        "name":           name,
        "path":           dirpath,
        "format":         "gguf",
        "main_file":      main_file,
        "mmproj_file":    mmproj_file,
        "supports_vision": supports_vision,
        "supports_video": supports_vision,
        "native_video":   False,
        "quant_type":     None,   # GGUF quant is internal to the file
        "quant_requires": None,
        "size_mb":        size_mb,
        "vision_handler": _gguf_vision_handler(main_file) if supports_vision else None,
    }


def _is_mmproj(filename: str) -> bool:
    lower = filename.lower()
    return "mmproj" in lower or "vision" in lower and "proj" in lower


def _gguf_vision_handler(main_file: str) -> str:
    """Return the llama-cpp-python chat handler class name for a GGUF vision model.

    Gemma-4 uses the new MTMD backend (Gemma4ChatHandler) instead of the
    legacy LLaVA clip handler.  All other vision models use the old approach
    (clip_model_path kwarg + Llava15ChatHandler-derived handlers).
    """
    lower = main_file.lower()
    if "gemma" in lower:
        return "Gemma4ChatHandler"
    return "Llava15ChatHandler"


# ── HuggingFace directory scanner ─────────────────────────────────────────────

def _scan_hf_dir(dirpath: str, name: str) -> dict | None:
    """Return a model descriptor if dirpath looks like a HF model dir."""
    config_path = os.path.join(dirpath, "config.json")
    if not os.path.exists(config_path):
        return None

    try:
        with open(config_path, encoding="utf-8") as fh:
            config = json.load(fh)
    except Exception:
        return None

    architectures: list[str] = config.get("architectures", [])
    model_type: str = config.get("model_type", "")

    # Vision detection
    has_vision_config = "vision_config" in config
    has_vision_arch = bool(set(architectures) & _VISION_ARCHS)
    has_vision_type = model_type in _VISION_MODEL_TYPES
    has_preprocessor = (
        os.path.exists(os.path.join(dirpath, "preprocessor_config.json"))
        or os.path.exists(os.path.join(dirpath, "image_processor_config.json"))
    )

    supports_vision = has_vision_config or has_vision_arch or has_vision_type or has_preprocessor
    native_video = bool(set(architectures) & _NATIVE_VIDEO_ARCHS) or model_type in {
        "qwen2_vl", "qwen2_5_vl", "qwen2_5_omni", "llava_next_video"
    }

    quant_type, quant_requires = _detect_quant(config)

    # Rough size estimate from total safetensors/bin bytes
    size_mb = _hf_size_mb(dirpath)

    return {
        "id":             _make_id(name),
        "name":           name,
        "path":           dirpath,
        "format":         "hf",
        "main_file":      None,
        "mmproj_file":    None,
        "model_type":     model_type,
        "architectures":  architectures,
        "supports_vision": supports_vision,
        "supports_video": supports_vision,
        "native_video":   native_video,
        "quant_type":     quant_type,
        "quant_requires": quant_requires,
        "size_mb":        size_mb,
    }


def _hf_size_mb(dirpath: str) -> int:
    total = 0
    try:
        for fname in os.listdir(dirpath):
            if fname.endswith((".safetensors", ".bin", ".pt")):
                total += os.path.getsize(os.path.join(dirpath, fname))
    except OSError:
        pass
    return round(total / (1024 * 1024))


# ── Unified directory scanner ─────────────────────────────────────────────────

def _scan_directory(root: str, depth: int = 2) -> list[dict]:
    """Walk root up to `depth` levels deep, collecting model descriptors.

    Iterates children of root rather than root itself — this prevents a
    stray .gguf file in the root (e.g. a text encoder) from causing the
    entire directory to be misidentified as a single GGUF model.
    """
    results: list[dict] = []
    if not os.path.isdir(root):
        return results
    try:
        for entry in sorted(os.listdir(root)):
            if entry.startswith("."):
                continue
            child = os.path.join(root, entry)
            if os.path.isdir(child):
                _walk(child, depth - 1, results)
    except OSError:
        pass
    return results


def _walk(dirpath: str, depth: int, results: list[dict]) -> None:
    name = os.path.basename(dirpath)
    # Try GGUF first (presence of .gguf files)
    try:
        entries = os.listdir(dirpath)
    except OSError:
        return

    has_gguf = any(f.endswith(".gguf") for f in entries)
    has_config = "config.json" in entries

    if has_gguf:
        m = _scan_gguf_dir(dirpath, name)
        if m:
            results.append(m)
        return  # don't recurse into a GGUF model dir

    if has_config:
        m = _scan_hf_dir(dirpath, name)
        if m:
            results.append(m)
        return  # don't recurse into an HF model dir

    if depth > 0:
        for entry in sorted(entries):
            child = os.path.join(dirpath, entry)
            if os.path.isdir(child) and not entry.startswith("."):
                _walk(child, depth - 1, results)


# ── Public API ────────────────────────────────────────────────────────────────

def scan_llm_dirs(extra_dirs: list[str] | None = None) -> list[dict]:
    """Scan all LLM model directories and return capability-annotated descriptors.

    Tries folder_paths.get_folder_paths("LLMs") first (respects
    extra_model_paths.yaml).  Falls back to ComfyUI/models/LLMs/.
    Additional directories can be passed via extra_dirs.

    Each returned dict has the keys documented in the module docstring,
    plus `capability_tags` (list[str]) and `capability_note` (str).
    """
    roots: list[str] = []

    # Honour ComfyUI's registered paths (includes extra_model_paths.yaml entries).
    # "LLM" (singular) is the convention used by ComfyUI-MiniMaxH3-Prompt-Writer
    # and comfyui_llm_party; fall back to "LLMs" in case a node registered that.
    try:
        import folder_paths
        registered = (
            folder_paths.get_folder_paths("LLM")
            or folder_paths.get_folder_paths("LLMs")
        )
        if registered:
            roots.extend(registered)
        else:
            roots.append(os.path.join(folder_paths.base_path, "models", "LLM"))
    except Exception:
        pass

    if extra_dirs:
        roots.extend(extra_dirs)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_roots: list[str] = []
    for r in roots:
        r = os.path.normpath(r)
        if r not in seen:
            seen.add(r)
            unique_roots.append(r)

    models: list[dict] = []
    seen_ids: set[str] = set()
    for root in unique_roots:
        for m in _scan_directory(root, depth=2):
            # Disambiguate duplicate ids from different roots
            mid = m["id"]
            if mid in seen_ids:
                mid = _make_id(os.path.relpath(m["path"], root).replace(os.sep, "_"))
                m["id"] = mid
            seen_ids.add(mid)
            m["capability_tags"] = capability_tags(m)
            m["capability_note"] = capability_note(m)
            models.append(m)

    models.sort(key=lambda m: m["name"].lower())
    return models


def _make_id(name: str) -> str:
    return re.sub(r"[^\w]", "_", name.lower()).strip("_")


# ── Default model recommendation ──────────────────────────────────────────────

DEFAULT_MODEL = {
    "repo_id":     "bartowski/Qwen2.5-VL-3B-Instruct-GGUF",
    "filename":    "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
    "mmproj":      "mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf",
    "name":        "Qwen2.5-VL 3B Instruct (Q4_K_M)",
    "size_hint":   "~2 GB (model) + ~600 MB (vision projector)",
    "description": (
        "Small, fast, vision-capable model. Analyses images and video frames. "
        "Good for drafting shot descriptions, dialogue, and describing characters "
        "from character sheet images. Requires llama-cpp-python."
    ),
    "supports_vision": True,
    "supports_video":  True,
    "native_video":    False,
    "capability_note": (
        "Accepts images. Video via frame sampling (keyframes only). "
        "Native video understanding not available in GGUF format."
    ),
}
