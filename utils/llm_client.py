"""LLM client for the Composition Editor assistant.

Maintains a single loaded model in memory.  Supports GGUF via
llama-cpp-python and HuggingFace transformers format.

Both backends are optional — the module degrades gracefully when
neither is installed.  Call `backend_status()` to surface what is
available without raising.

Thread safety: ComfyUI runs aiohttp in an event loop; we offload
blocking inference calls to a thread executor to avoid blocking the
loop.  The module-level `_state` dict is not concurrent-write-safe
but in practice only one generate/load request is in flight at a time
due to the server's sequential request dispatch.
"""
from __future__ import annotations

import gc
import os
import traceback
from pathlib import Path
from typing import Any

from utils.logging_utils import get_logger

logger = get_logger(__name__)


# ── Backend availability ───────────────────────────────────────────────────────

def _has_llama_cpp() -> bool:
    try:
        import llama_cpp  # noqa: F401
        return True
    except ImportError:
        return False


def _has_transformers() -> bool:
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def backend_status() -> dict:
    """Return which backends are available and the currently loaded model."""
    return {
        "llama_cpp":    _has_llama_cpp(),
        "transformers": _has_transformers(),
        "torch":        _has_torch(),
        "loaded_model": _state.get("model_name"),
        "loaded_format": _state.get("format"),
        "loaded_path":  _state.get("model_path"),
        "supports_vision": _state.get("supports_vision", False),
    }


# ── Module-level state ─────────────────────────────────────────────────────────

_state: dict[str, Any] = {
    "model":          None,    # llama_cpp.Llama or transformers Pipeline/model
    "processor":      None,    # transformers processor (HF only)
    "model_name":     None,    # display name
    "model_path":     None,    # directory path
    "format":         None,    # "gguf" or "hf"
    "supports_vision": False,
    "native_video":   False,
}


# ── Load ──────────────────────────────────────────────────────────────────────

def load_model(model_info: dict) -> dict:
    """Load a model described by a llm_scanner descriptor dict.

    Returns a status dict with keys: success (bool), message (str).
    Unloads any previously loaded model first.
    """
    fmt = model_info.get("format")
    if fmt == "gguf":
        return _load_gguf(model_info)
    elif fmt == "hf":
        return _load_hf(model_info)
    else:
        return {"success": False, "message": f"Unknown model format: {fmt!r}"}


def _load_gguf(info: dict) -> dict:
    if not _has_llama_cpp():
        return {
            "success": False,
            "message": (
                "llama-cpp-python is not installed. "
                "Install it with: pip install llama-cpp-python"
            ),
        }
    try:
        from llama_cpp import Llama

        unload_model()

        model_path = os.path.join(info["path"], info["main_file"])
        mmproj_path = (
            os.path.join(info["path"], info["mmproj_file"])
            if info.get("mmproj_file")
            else None
        )

        logger.info("Loading GGUF model: %s", model_path)
        kwargs: dict = {
            "model_path":     model_path,
            "n_ctx":          4096,
            "n_gpu_layers":   -1,   # offload everything to GPU
            "verbose":        False,
        }
        if mmproj_path and os.path.exists(mmproj_path):
            kwargs["clip_model_path"] = mmproj_path
            logger.info("Vision projector: %s", mmproj_path)

        model = Llama(**kwargs)

        _state.update({
            "model":           model,
            "processor":       None,
            "model_name":      info.get("name", info.get("id", "unknown")),
            "model_path":      info["path"],
            "format":          "gguf",
            "supports_vision": info.get("supports_vision", False),
            "native_video":    False,
        })
        logger.info("GGUF model loaded: %s", _state["model_name"])
        return {"success": True, "message": f"Loaded: {_state['model_name']}"}

    except Exception as e:
        logger.error("GGUF load failed: %s", e)
        return {"success": False, "message": f"Load failed: {e}"}


def _load_hf(info: dict) -> dict:
    if not _has_transformers():
        return {
            "success": False,
            "message": (
                "transformers is not installed. "
                "Install it with: pip install transformers accelerate"
            ),
        }
    try:
        import transformers
        import torch

        unload_model()

        model_path = info["path"]
        arch = (info.get("architectures") or [""])[0]
        logger.info("Loading HF model from: %s  arch=%s", model_path, arch)

        supports_vision = info.get("supports_vision", False)
        native_video = info.get("native_video", False)

        if supports_vision:
            # Use AutoModelForVision2Seq / processor combo
            processor = transformers.AutoProcessor.from_pretrained(
                model_path, trust_remote_code=True
            )
            model = transformers.AutoModelForVision2Seq.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            processor = transformers.AutoTokenizer.from_pretrained(model_path)
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )

        _state.update({
            "model":           model,
            "processor":       processor,
            "model_name":      info.get("name", info.get("id", "unknown")),
            "model_path":      model_path,
            "format":          "hf",
            "supports_vision": supports_vision,
            "native_video":    native_video,
        })
        logger.info("HF model loaded: %s", _state["model_name"])
        return {"success": True, "message": f"Loaded: {_state['model_name']}"}

    except Exception as e:
        logger.error("HF load failed: %s\n%s", e, traceback.format_exc())
        return {"success": False, "message": f"Load failed: {e}"}


# ── Unload ────────────────────────────────────────────────────────────────────

def unload_model() -> dict:
    """Unload the current model and free VRAM/RAM."""
    if _state["model"] is None:
        return {"success": True, "message": "No model loaded."}

    name = _state.get("model_name", "unknown")
    try:
        _state["model"] = None
        _state["processor"] = None
        _state["model_name"] = None
        _state["model_path"] = None
        _state["format"] = None
        _state["supports_vision"] = False
        _state["native_video"] = False

        gc.collect()

        # Free CUDA memory if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

        logger.info("Model unloaded: %s", name)
        return {"success": True, "message": f"Unloaded: {name}"}

    except Exception as e:
        return {"success": False, "message": f"Unload failed: {e}"}


# ── Generate ──────────────────────────────────────────────────────────────────

def generate(
    prompt: str,
    *,
    images: list[Any] | None = None,   # PIL Images for vision-capable models
    system_prompt: str = "",
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> dict:
    """Generate text.  Returns {success, text, message}."""
    if _state["model"] is None:
        return {"success": False, "text": "", "message": "No model loaded."}

    fmt = _state["format"]
    try:
        if fmt == "gguf":
            return _generate_gguf(prompt, images=images, system_prompt=system_prompt,
                                  max_tokens=max_tokens, temperature=temperature)
        elif fmt == "hf":
            return _generate_hf(prompt, images=images, system_prompt=system_prompt,
                                max_tokens=max_tokens, temperature=temperature)
        else:
            return {"success": False, "text": "", "message": f"Unknown format: {fmt}"}
    except Exception as e:
        logger.error("Generate failed: %s\n%s", e, traceback.format_exc())
        return {"success": False, "text": "", "message": f"Generation error: {e}"}


def _generate_gguf(prompt: str, *, images, system_prompt, max_tokens, temperature) -> dict:
    model = _state["model"]
    supports_vision = _state["supports_vision"]

    if supports_vision and images:
        # llama-cpp-python multimodal path
        try:
            from llama_cpp.llama_chat_format import MoondreamChatHandler
        except ImportError:
            pass

        import base64
        from io import BytesIO

        image_uris: list[str] = []
        for img in images:
            buf = BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            image_uris.append(f"data:image/png;base64,{b64}")

        # Build a chat-format message with embedded images
        content: list[dict] = []
        for uri in image_uris:
            content.append({"type": "image_url", "image_url": {"url": uri}})
        content.append({"type": "text", "text": prompt})

        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        response = model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    else:
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    text = response["choices"][0]["message"]["content"].strip()
    return {"success": True, "text": text, "message": ""}


def _generate_hf(prompt: str, *, images, system_prompt, max_tokens, temperature) -> dict:
    import torch

    model = _state["model"]
    processor = _state["processor"]
    supports_vision = _state["supports_vision"]

    if supports_vision and images and hasattr(processor, "image_processor"):
        # Vision model path
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        content: list[dict] = []
        for _ in images:
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": content})

        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=text_input,
            images=images,
            return_tensors="pt",
        ).to(model.device)
    else:
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        if hasattr(processor, "apply_chat_template"):
            text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=text_input, return_tensors="pt").to(model.device)
        else:
            inputs = processor(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )

    # Strip the prompt tokens
    generated = out[0][inputs["input_ids"].shape[-1]:]
    if hasattr(processor, "decode"):
        text = processor.decode(generated, skip_special_tokens=True).strip()
    else:
        text = processor.tokenizer.decode(generated, skip_special_tokens=True).strip()

    return {"success": True, "text": text, "message": ""}


# ── Task-specific prompt builders ─────────────────────────────────────────────
# Each returns (system_prompt, user_prompt) suitable for generate().

def prompt_for_shot_action(
    shot_number: int,
    subjects: list[str],
    environment: str,
    style: str,
    existing_action: str = "",
) -> tuple[str, str]:
    system = (
        "You are a cinematography assistant helping write scene descriptions "
        "for AI video generation. Be vivid, concrete, and production-focused. "
        "Output only the requested text — no preamble, no labels."
    )
    subj_str = ", ".join(subjects) if subjects else "the character"
    user = (
        f"Write a 1-3 sentence shot action description for Shot {shot_number}.\n"
        f"Subjects: {subj_str}\n"
        f"Environment: {environment or 'unspecified'}\n"
        f"Style: {style or 'cinematic'}\n"
    )
    if existing_action:
        user += f"Existing description (improve or expand): {existing_action}\n"
    user += "\nShot action:"
    return system, user


def prompt_for_shot_dialogue(
    speaker: str,
    context: str,
    tone: str = "",
    language: str = "en-us",
) -> tuple[str, str]:
    system = (
        "You are a screenwriter writing natural, character-specific dialogue "
        f"in {language}. Output only the dialogue line — no speaker label, "
        "no quotes, no stage directions."
    )
    user = (
        f"Write one line of dialogue for {speaker}.\n"
        f"Context: {context or 'general conversation'}\n"
    )
    if tone:
        user += f"Tone: {tone}\n"
    user += "\nDialogue:"
    return system, user


def prompt_for_subject_description(
    name: str,
    role: str = "",
) -> tuple[str, str]:
    system = (
        "You are a character design assistant. Given character sheet images, "
        "write a concise, concrete appearance description suitable for AI image/video "
        "generation. Describe physical features, distinctive clothing, and key visual "
        "traits. Output only the description — no preamble."
    )
    user = (
        f"Describe the appearance of {name}"
        + (f" ({role})" if role else "")
        + " based on the provided character sheet images.\n\nAppearance description:"
    )
    return system, user


def prompt_for_camera_description(
    shot_number: int,
    context: str,
    existing: str = "",
) -> tuple[str, str]:
    system = (
        "You are a cinematographer. Describe a camera setup in 1-2 sentences "
        "using standard film terminology (shot size, angle, movement). "
        "Output only the camera description."
    )
    user = f"Shot {shot_number} camera description.\nContext: {context or 'general scene'}\n"
    if existing:
        user += f"Existing (improve): {existing}\n"
    user += "\nCamera:"
    return system, user


def prompt_for_polish(text: str, context: str = "") -> tuple[str, str]:
    system = (
        "You are an editor. Polish the following text for clarity and vividness "
        "without changing its meaning or adding new content. "
        "Output only the revised text."
    )
    user = f"Original text:\n{text}\n"
    if context:
        user += f"Context: {context}\n"
    user += "\nPolished version:"
    return system, user
