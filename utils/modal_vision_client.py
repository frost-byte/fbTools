"""
Slim client for the Modal-hosted fbtools-vision-llm app.

Public surface mirrors utils/llm_client.py:
    backend_status() -> dict
    generate(...)    -> {success, text, message}

Additional:
    activate(model_key, quantize)  -> {success, message}
    deactivate()                   -> {success, message}
    is_active()                    -> bool

The Modal app ("fbtools-vision-llm" / "VisionLLM") must be deployed via the
fbtools-vision-modal repo before this client can function.  Auth is handled by
~/.modal.toml (workspace frost-byte) — no credentials are sent from the frontend.

Cold starts (~1 min on L40S) are expected if no container is warm.  Callers
should pass status_callback to surface the wait to the user.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ── Preset models ─────────────────────────────────────────────────────────────

PRESET_MODELS: list[dict] = [
    # key          — passed as model_key to VisionLLM
    # label        — display name in the UI
    # native_video — supports video_frames path (Qwen VL family does; Gemma does not)
    # pre_quantized — already quantized (AWQ/GPTQ); do NOT apply NF4 on top
    {"key": "qwen3-vl-8b",        "label": "Qwen3-VL 8B",          "native_video": True,  "pre_quantized": False},
    {"key": "qwen2.5-vl-7b",      "label": "Qwen2.5-VL 7B",        "native_video": True,  "pre_quantized": False},
    {"key": "qwen2.5-vl-32b-awq", "label": "Qwen2.5-VL 32B (AWQ)", "native_video": True,  "pre_quantized": True},
    {"key": "qwen2.5-vl-3b",      "label": "Qwen2.5-VL 3B",        "native_video": True,  "pre_quantized": False},
    {"key": "qwen2.5-omni-7b",    "label": "Qwen2.5-Omni 7B",      "native_video": True,  "pre_quantized": False},
    {"key": "gemma3-4b",          "label": "Gemma 3 4B",            "native_video": False, "pre_quantized": False},
]

_PRESET_NATIVE_VIDEO: dict[str, bool]    = {m["key"]: m["native_video"]    for m in PRESET_MODELS}
_PRESET_PRE_QUANTIZED: dict[str, bool]   = {m["key"]: m["pre_quantized"]   for m in PRESET_MODELS}

# ── Module-level state ────────────────────────────────────────────────────────

_state: dict[str, Any] = {
    "active":         False,
    "model_key":      "qwen3-vl-8b",
    "quantize":       True,
    "native_video":   True,
    "pre_quantized":  False,
}

# ── Availability ──────────────────────────────────────────────────────────────

def _has_modal() -> bool:
    try:
        import modal  # noqa: F401
        return True
    except ImportError:
        return False


# ── Public API ────────────────────────────────────────────────────────────────

def backend_status() -> dict:
    """Return Modal backend state (mirrors llm_client.backend_status shape)."""
    return {
        "active":           _state["active"],
        "model_key":        _state["model_key"],
        "quantize":         _state["quantize"],
        "native_video":     _state["native_video"],
        "pre_quantized":    _state["pre_quantized"],
        "modal_available":  _has_modal(),
    }


def activate(model_key: str = "qwen2.5-vl-7b", quantize: bool = True) -> dict:
    """Mark the Modal backend as active for the given model key.

    Does not spin up a container — the first generate() call triggers that.
    Returns {success, message}.
    """
    if not _has_modal():
        return {
            "success": False,
            "message": (
                "modal package is not installed. "
                "Install with: pip install modal  (or pip install 'fb-Tools[modal]')"
            ),
        }
    pre_quantized = _PRESET_PRE_QUANTIZED.get(model_key, False)
    if quantize and pre_quantized:
        logger.warning(
            "Model %r is already AWQ/GPTQ quantized — disabling NF4 to avoid double-quantization",
            model_key,
        )
        quantize = False

    _state["active"]        = True
    _state["model_key"]     = model_key
    _state["quantize"]      = quantize
    _state["pre_quantized"] = pre_quantized
    # native_video is preset-derived; custom repo IDs default to True (Qwen family)
    _state["native_video"]  = _PRESET_NATIVE_VIDEO.get(model_key, True)
    logger.info("Modal backend activated: model=%s quantize=%s pre_quantized=%s",
                model_key, quantize, pre_quantized)
    return {"success": True, "message": f"Modal activated with model {model_key!r}"}


def deactivate() -> dict:
    """Deactivate the Modal backend.

    Only clears local state — Modal auto-scales to zero on its own.
    Returns {success, message}.
    """
    _state["active"] = False
    logger.info("Modal backend deactivated")
    return {"success": True, "message": "Modal deactivated"}


def is_active() -> bool:
    return bool(_state["active"])


def generate(
    prompt: str,
    *,
    images: list[Any] | None = None,
    video_frames: list[Any] | None = None,
    system_prompt: str = "",
    max_tokens: int = 512,
    temperature: float = 0.7,
    video_meta: dict | None = None,
    status_callback: Callable[[str], None] | None = None,
) -> dict:
    """Call VisionLLM.generate.remote() on the deployed Modal app.

    Signature-compatible with llm_client.generate() — same parameter names
    and return shape {success, text, message}.

    status_callback(msg) is called before the blocking remote call so callers
    can surface the cold-start wait via send_status_update.
    """
    if not _state["active"]:
        return {"success": False, "text": "", "message": "Modal backend is not active."}
    if not _has_modal():
        return {"success": False, "text": "", "message": "modal package is not installed."}

    try:
        import modal

        model_key = _state["model_key"]
        quantize  = _state["quantize"]

        if status_callback:
            status_callback(
                f"Calling Modal ({model_key}) — container may take ~60 s to warm up on first use"
            )

        vlm = modal.Cls.from_name("fbtools-vision-llm", "VisionLLM")()
        result = vlm.generate.remote(
            prompt,
            model_key=model_key,
            quantize=quantize,
            images=images,
            video_frames=video_frames,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            video_meta=video_meta,
        )
        if not isinstance(result, dict):
            return {"success": False, "text": "", "message": f"Unexpected Modal response type: {type(result)}"}
        return result
    except Exception as exc:
        msg = str(exc)
        if "not found in environment" in msg or ("not found" in msg and "fbtools-vision-llm" in msg):
            return {
                "success": False, "text": "",
                "message": (
                    "Modal app 'fbtools-vision-llm' is not deployed. "
                    "Run: modal deploy modal/app.py  from the comfyui-fbTools directory."
                ),
            }
        logger.error("Modal generate failed: %s", exc)
        return {"success": False, "text": "", "message": f"Modal error: {exc}"}
