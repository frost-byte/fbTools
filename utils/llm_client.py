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
import logging
import os
import traceback
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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
        "native_video": _state.get("native_video", False),
        "quant_type":   _state.get("quant_type"),
    }


# ── Module-level state ─────────────────────────────────────────────────────────

_state: dict[str, Any] = {
    "model":          None,    # llama_cpp.Llama or transformers Pipeline/model
    "processor":      None,    # transformers processor (HF only)
    "model_name":     None,    # display name
    "model_path":     None,    # directory path
    "format":         None,    # "gguf" or "hf"
    "arch_name":      None,    # primary HF architecture class name (HF only)
    "supports_vision": False,
    "native_video":   False,
    "quant_type":     None,    # "awq", "gptq", "bitsandbytes", or None
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

        chat_handler = None
        if mmproj_path and os.path.exists(mmproj_path):
            logger.info("Vision projector: %s", mmproj_path)
            # vision_handler is set by the scanner; fall back to filename detection
            vision_handler = info.get("vision_handler") or (
                "Gemma4ChatHandler" if "gemma" in info.get("main_file", "").lower() else "Llava15ChatHandler"
            )
            if vision_handler == "Gemma4ChatHandler":
                # Gemma-4 uses the MTMD backend — must be a chat_handler, not clip_model_path
                from llama_cpp.llama_chat_format import Gemma4ChatHandler
                chat_handler = Gemma4ChatHandler(clip_model_path=mmproj_path, verbose=False)
                logger.info("Using Gemma4ChatHandler for vision")
            else:
                # LLaVA-style models: pass clip_model_path directly to Llama
                kwargs["clip_model_path"] = mmproj_path
        if chat_handler is not None:
            kwargs["chat_handler"] = chat_handler

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


def _build_quant_config(info: dict, arch: str = ""):
    """Return a transformers quantization config object, or None for unquantized."""
    qt = info.get("quant_type")
    if not qt:
        return None

    import transformers

    if qt == "awq":
        try:
            from transformers.utils import is_auto_awq_available
            if not is_auto_awq_available():
                raise RuntimeError("autoawq is not installed — run: pip install autoawq")
            return transformers.AwqConfig(bits=4)
        except ImportError:
            raise RuntimeError("autoawq is not installed — run: pip install autoawq")

    if qt == "gptq":
        # disable_exllama=True: exllama v1 is incompatible with desc_act=True
        # checkpoints (e.g. Qwen2.5-Omni-GPTQ-Int4).  optimum.gptq handles
        # loading without it; auto-gptq also works if installed.
        #
        # block_name_to_quantize: optimum.gptq's BLOCK_PATTERNS list does not
        # include "thinker.model.layers" (used by Qwen2.5-Omni).  We must tell
        # it explicitly so GPTQQuantizer.convert_model skips the pattern scan.
        try:
            gptq_kwargs: dict = {"bits": 4, "disable_exllama": True}
            if arch == "Qwen2_5OmniForConditionalGeneration":
                gptq_kwargs["block_name_to_quantize"] = "thinker.model.layers"
                logger.info("GPTQ: setting block_name_to_quantize=thinker.model.layers for Omni arch")
            return transformers.GPTQConfig(**gptq_kwargs)
        except Exception as e:
            logger.warning("GPTQConfig failed (%s) — loading without explicit quant config", e)
            return None

    if qt == "bitsandbytes":
        try:
            return transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=__import__("torch").bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        except Exception as e:
            raise RuntimeError(f"BitsAndBytes config failed: {e}")

    logger.warning("Unknown quant_type %r — loading without quantization config", qt)
    return None


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
        import warnings
        import transformers
        import torch

        unload_model()

        model_path = info["path"]
        arch = (info.get("architectures") or [""])[0]
        qt = info.get("quant_type")
        logger.info("Loading HF model from: %s  arch=%s  quant=%s", model_path, arch, qt or "none")

        supports_vision = info.get("supports_vision", False)
        native_video = info.get("native_video", False)

        quant_config = _build_quant_config(info, arch=arch)

        # accelerate's device_map="auto" reserves 90% of VRAM by default,
        # leaving only 10% for activations/KV cache.  On a shared ComfyUI
        # system this starves diffusion models loaded after the LLM.
        # We cap at 70% GPU / unlimited CPU so the LLM fits but leaves ~30%
        # free for ComfyUI model loading.  CPU offload absorbs any overflow.
        try:
            import torch
            gpu_total = torch.cuda.get_device_properties(0).total_memory
            gpu_cap   = int(gpu_total * 0.70)
            _max_memory = {0: gpu_cap, "cpu": "64GiB"}
        except Exception:
            _max_memory = None

        load_kwargs: dict = {
            "device_map":        "auto",
            "trust_remote_code": True,
        }
        if _max_memory is not None:
            load_kwargs["max_memory"] = _max_memory
            logger.info(
                "LLM max_memory: GPU 70%% (%d MiB) + CPU 64 GiB",
                _max_memory[0] // (1024 * 1024),
            )

        # Qwen2.5-Omni AWQ: the model's embedded quantization_config only lists
        # "visual" in modules_to_not_convert. The talker and token2wav submodels
        # (BigVGAN vocoder, DiT) contain nn.Linear layers with in_features not
        # divisible by group_size=128, which crashes WQLinear_GEMM.__init__.
        #
        # We can't fix this by passing quantization_config= because
        # merge_quantization_configs() uses the model's embedded config as the
        # base and only propagates "loading attributes" from ours
        # (modules_to_not_convert is not a loading attribute, so it's ignored).
        #
        # Instead: pre-load AutoConfig, patch modules_to_not_convert in-place,
        # pass it as config= without a separate quantization_config kwarg.
        if arch == "Qwen2_5OmniForConditionalGeneration" and qt == "awq":
            omni_cfg = transformers.AutoConfig.from_pretrained(
                model_path, trust_remote_code=True
            )
            qc = getattr(omni_cfg, "quantization_config", None)
            _omni_skip = ["visual", "thinker.audio_tower", "talker", "token2wav"]
            if isinstance(qc, dict):
                # AutoConfig leaves quantization_config as a raw dict.
                qc["modules_to_not_convert"] = _omni_skip
                logger.info("Omni AWQ: patched modules_to_not_convert (dict) to skip audio/vocoder layers")
            elif qc is not None and hasattr(qc, "modules_to_not_convert"):
                qc.modules_to_not_convert = _omni_skip
                logger.info("Omni AWQ: patched modules_to_not_convert (object) to skip audio/vocoder layers")
            else:
                logger.warning("Omni AWQ: could not locate quantization_config to patch (type=%s)", type(qc))

            # The AWQ Triton dequantize kernel (awq_dequantize_triton) fails on
            # this Triton version with "invalid operands of type float16" when it
            # tries to bitshift packed int4 weights.  Monkey-patching
            # TRITON_AVAILABLE=False in the awq linear module forces the fallback
            # dequantize_gemm + torch.matmul path for all WQLinear_GEMM forward
            # passes in this process — no site-package edits required.
            try:
                import awq.modules.linear.gemm as _awq_gemm_mod
                if getattr(_awq_gemm_mod, "TRITON_AVAILABLE", False):
                    _awq_gemm_mod.TRITON_AVAILABLE = False
                    logger.info("Omni AWQ: disabled Triton kernel (fallback to dequantize_gemm+matmul)")
            except ImportError:
                pass

            load_kwargs["config"] = omni_cfg
            # Don't pass quantization_config — let the patched config drive it
        elif arch == "Qwen2_5OmniForConditionalGeneration" and qt == "gptq":
            # Same problem as AWQ: merge_quantization_configs() uses the model's
            # embedded config as base and only copies "loading attributes" from
            # ours. block_name_to_quantize is not a loading attribute, so our
            # GPTQConfig value is silently discarded and the pattern scan fails
            # (optimum's BLOCK_PATTERNS has no entry for "thinker.model.layers").
            #
            # Fix: pre-load AutoConfig, patch block_name_to_quantize in the
            # embedded dict, pass config= without a separate quantization_config.
            omni_cfg = transformers.AutoConfig.from_pretrained(
                model_path, trust_remote_code=True
            )
            qc = getattr(omni_cfg, "quantization_config", None)
            if isinstance(qc, dict):
                qc["block_name_to_quantize"] = "thinker.model.layers"
                qc["disable_exllama"] = True
                logger.info("GPTQ: patched block_name_to_quantize=thinker.model.layers + disable_exllama in embedded config")
            else:
                logger.warning("GPTQ: could not patch embedded quantization_config (type=%s) — load may fail", type(qc))
            load_kwargs["config"] = omni_cfg
        elif quant_config is not None:
            load_kwargs["quantization_config"] = quant_config
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16

        if supports_vision:
            processor = transformers.AutoProcessor.from_pretrained(
                model_path, trust_remote_code=True
            )
            # Qwen2.5-Omni is not registered in AutoModelForVision2Seq; load it directly.
            if arch == "Qwen2_5OmniForConditionalGeneration":
                from transformers import Qwen2_5OmniForConditionalGeneration
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=DeprecationWarning, module="awq")
                    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                        model_path, **load_kwargs
                    )
            else:
                model = transformers.AutoModelForVision2Seq.from_pretrained(
                    model_path, **load_kwargs
                )
        else:
            processor = transformers.AutoTokenizer.from_pretrained(model_path)
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_path, **load_kwargs
            )

        _state.update({
            "model":           model,
            "processor":       processor,
            "model_name":      info.get("name", info.get("id", "unknown")),
            "model_path":      model_path,
            "format":          "hf",
            "arch_name":       arch,
            "supports_vision": supports_vision,
            "native_video":    native_video,
            "quant_type":      qt,
        })
        logger.info("HF model loaded: %s (quant=%s)", _state["model_name"], qt or "none")
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
        _state["arch_name"] = None
        _state["supports_vision"] = False
        _state["native_video"] = False
        _state["quant_type"] = None

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
    images: list[Any] | None = None,         # PIL Images for vision models (stills)
    video_frames: list[Any] | None = None,   # PIL frames for native-video models
    system_prompt: str = "",
    max_tokens: int = 512,
    temperature: float = 0.7,
    video_meta: dict | None = None,          # {sample_fps, raw_fps} for temporal RoPE
) -> dict:
    """Generate text.  Returns {success, text, message}.

    video_meta keys (all optional):
        sample_fps  — effective frame rate of the selected frames (frames / clip_duration).
                      Drives temporal RoPE position IDs; defaults to 2.0 if omitted.
        raw_fps     — original video FPS; used to derive total_num_frames metadata.
    """
    if _state["model"] is None:
        return {"success": False, "text": "", "message": "No model loaded."}

    fmt = _state["format"]
    try:
        if fmt == "gguf":
            return _generate_gguf(prompt, images=images, video_frames=video_frames,
                                  system_prompt=system_prompt,
                                  max_tokens=max_tokens, temperature=temperature)
        elif fmt == "hf":
            return _generate_hf(prompt, images=images, video_frames=video_frames,
                                system_prompt=system_prompt,
                                max_tokens=max_tokens, temperature=temperature,
                                video_meta=video_meta)
        else:
            return {"success": False, "text": "", "message": f"Unknown format: {fmt}"}
    except Exception as e:
        logger.error("Generate failed: %s\n%s", e, traceback.format_exc())
        return {"success": False, "text": "", "message": f"Generation error: {e}"}


def _generate_gguf(prompt: str, *, images, video_frames, system_prompt, max_tokens, temperature) -> dict:
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


def _generate_hf(prompt: str, *, images, video_frames, system_prompt, max_tokens, temperature, video_meta=None) -> dict:
    import torch

    model = _state["model"]
    processor = _state["processor"]
    supports_vision = _state["supports_vision"]
    native_video = _state["native_video"]
    arch_name = _state.get("arch_name", "")
    is_omni = "Omni" in (arch_name or "")

    has_images = bool(images)
    has_video = bool(video_frames)
    use_video_path = has_video and native_video
    use_vision = supports_vision and (has_images or has_video) and hasattr(processor, "image_processor")

    # Qwen2.5-Omni was fine-tuned with a specific identity system prompt.
    # Replacing it entirely triggers a warning ("audio output may not work")
    # and degrades quality even for text-only generation.  We prepend it and
    # append our task instructions after it so the model stays in its trained
    # operational mode while still following our directions.
    _OMNI_DEFAULT_SYSTEM = (
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
        "capable of perceiving auditory and visual inputs, as well as generating "
        "text and speech."
    )

    if use_vision:
        messages: list[dict] = []
        effective_system = system_prompt or ""
        if is_omni:
            effective_system = (
                _OMNI_DEFAULT_SYSTEM + ("\n\n" + effective_system if effective_system else "")
            )
        if effective_system:
            # Qwen2.5-Omni's apply_chat_template inspects content[0]["text"] on the
            # system message, so content must be a list of dicts — not a plain string.
            sys_content = ([{"type": "text", "text": effective_system}]
                           if is_omni else effective_system)
            messages.append({"role": "system", "content": sys_content})

        content: list[dict] = []
        if use_video_path:
            # Include temporal metadata so the processor computes correct RoPE
            # position IDs. Without sample_fps the library defaults to 2.0 fps,
            # which misrepresents the actual time span between selected frames.
            vm = video_meta or {}
            video_elem: dict = {"type": "video", "video": video_frames}
            if "sample_fps" in vm:
                video_elem["sample_fps"] = vm["sample_fps"]
            if "raw_fps" in vm:
                video_elem["raw_fps"] = vm["raw_fps"]
            content.append(video_elem)
        else:
            for _ in images:
                content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": content})

        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        if use_video_path and is_omni:
            # Qwen2.5-Omni: qwen_omni_utils resolves video frames → processor tensors
            try:
                from qwen_omni_utils import process_mm_info
                audios, proc_images, proc_videos = process_mm_info(
                    messages, use_audio_in_video=False
                )
                inputs = processor(
                    text=text_input,
                    images=proc_images or None,
                    videos=proc_videos or None,
                    audio=audios or None,
                    return_tensors="pt",
                ).to(model.device)
            except ImportError:
                logger.warning("qwen_omni_utils not installed; treating video frames as images")
                inputs = processor(
                    text=text_input,
                    images=video_frames,
                    return_tensors="pt",
                ).to(model.device)
        elif use_video_path:
            # Qwen2.5-VL / LLaVA-Next-Video: pass frames as a list-of-lists
            inputs = processor(
                text=text_input,
                videos=[video_frames],
                return_tensors="pt",
            ).to(model.device)
        else:
            inputs = processor(
                text=text_input,
                images=images,
                return_tensors="pt",
            ).to(model.device)
    else:
        messages: list[dict] = []
        effective_system_text = system_prompt or ""
        if is_omni and effective_system_text:
            effective_system_text = _OMNI_DEFAULT_SYSTEM + "\n\n" + effective_system_text
        elif is_omni:
            effective_system_text = _OMNI_DEFAULT_SYSTEM
        if effective_system_text:
            sys_content = ([{"type": "text", "text": effective_system_text}]
                           if is_omni else effective_system_text)
            messages.append({"role": "system", "content": sys_content})
        messages.append({"role": "user", "content": prompt})

        if hasattr(processor, "apply_chat_template"):
            text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=text_input, return_tensors="pt").to(model.device)
        else:
            inputs = processor(prompt, return_tensors="pt").to(model.device)

    gen_kwargs: dict = {
        "max_new_tokens": max_tokens,
        "temperature":    temperature,
        "do_sample":      temperature > 0,
    }
    if is_omni:
        # Qwen2.5-Omni's generate() has a custom signature. By default it
        # returns (text_sequences, audio_waveform) when has_talker=True, which
        # breaks the standard out[0][input_len:] decode pattern.
        # return_audio=False skips the talker and returns a plain tensor.
        # thinker_max_new_tokens is the Omni-specific kwarg; max_new_tokens
        # lands in shared_kwargs but thinker_kwargs already has its default,
        # so we must use the prefixed form to override it.
        gen_kwargs["return_audio"] = False
        gen_kwargs["thinker_max_new_tokens"] = max_tokens
        gen_kwargs.pop("max_new_tokens", None)

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    # Omni with return_audio=False → plain [batch, seq_len] tensor.
    # Standard VL models → same shape.
    # Unpack batch dim first so seq slicing always works.
    seq = out[0] if out.dim() == 2 else out
    generated = seq[inputs["input_ids"].shape[-1]:]
    if hasattr(processor, "decode"):
        text = processor.decode(generated, skip_special_tokens=True).strip()
    else:
        text = processor.tokenizer.decode(generated, skip_special_tokens=True).strip()

    logger.debug("_generate_hf: decoded %d tokens → %r", len(generated), text[:120])
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


def prompt_for_video_action(
    shot_number: int,
    subjects: list[str],
    intent: str = "actions",   # "actions" | "expressions" | "appearance"
    environment: str = "",
    style: str = "",
) -> tuple[str, str]:
    """Prompt pair for describing character behaviour in a video clip.

    Pass the extracted frames as ``video_frames`` to ``generate()``.
    ``intent`` controls what the model focuses on:
      - "actions"     → physical movement, gestures, body language
      - "expressions" → facial expressions, emotion, gaze
      - "appearance"  → clothing, visual style, distinguishing features
    """
    system = (
        "You are a film-analysis assistant. Watch the video clip carefully and "
        "describe what you observe with precision. Be concrete and production-focused. "
        "Output only the requested description — no preamble, no labels."
    )
    subj_str = ", ".join(subjects) if subjects else "the character"
    focus_map = {
        "actions":     "physical actions, gestures, and body language",
        "expressions": "facial expressions, emotion, and gaze direction",
        "appearance":  "visible clothing, physical features, and distinguishing details",
    }
    focus = focus_map.get(intent, focus_map["actions"])
    user = (
        f"Describe the {focus} of {subj_str} in Shot {shot_number}.\n"
        f"Environment: {environment or 'unspecified'}\n"
        f"Style: {style or 'cinematic'}\n"
        "\nWrite 1-3 sentences. Be specific about timing and movement.\n\nDescription:"
    )
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
