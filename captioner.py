"""
Standalone image captioning backend.
Supports Qwen2.5-VL (recommended for images) and Qwen2.5-Omni via transformers.
No dependency on ltx_trainer or any Lightricks package.
"""

from __future__ import annotations

import re
import threading
import time
from pathlib import Path
from typing import Literal

import torch
from PIL import Image

# ── Model IDs ────────────────────────────────────────────────────────────────

QWEN_VL_MODEL   = "Qwen/Qwen2.5-VL-7B-Instruct"    # vision-language, best for images
QWEN_OMNI_MODEL = "Qwen/Qwen2.5-Omni-7B"            # omni, heavier but matches LTX trainer
GEMINI_MODEL    = "gemini-2.0-flash"

# VLM boilerplate patterns to strip when clean_caption=True
_BOILERPLATE = re.compile(
    r"^\s*(Sure[,!]?\s*|Certainly[,!]?\s*|Of course[,!]?\s*|"
    r"Here(?:'s| is) (?:a |the )?(?:detailed |description|caption)[^:]*:\s*|"
    r"The image (?:shows|depicts|features|presents)\s*)",
    re.IGNORECASE,
)


# ── Singleton model cache (one model loaded at a time) ────────────────────────

_cache_lock  = threading.Lock()
_cached_type: str | None = None
_cached_model = None
_cached_processor = None


def _captioner_log(message: str):
    print(f"[DatasetCaptioner] {message}")


def _load_qwen_vl(device: str, use_8bit: bool):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    kwargs: dict = {"device_map": device if device != "auto" else "auto"}
    if use_8bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        kwargs["torch_dtype"] = torch.bfloat16

    _captioner_log(
        f"Loading Qwen-VL model ({QWEN_VL_MODEL}) on device={device}, 8bit={use_8bit}. "
        "First run may download model files from Hugging Face."
    )
    start = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_VL_MODEL, **kwargs
    )
    processor = AutoProcessor.from_pretrained(QWEN_VL_MODEL)
    _captioner_log(f"Qwen-VL model ready in {time.time() - start:.1f}s")
    return model, processor


def _load_qwen_omni(device: str, use_8bit: bool):
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    kwargs: dict = {"device_map": device if device != "auto" else "auto"}
    if use_8bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        kwargs["torch_dtype"] = torch.bfloat16

    _captioner_log(
        f"Loading Qwen-Omni model ({QWEN_OMNI_MODEL}) on device={device}, 8bit={use_8bit}. "
        "First run may download model files from Hugging Face."
    )
    start = time.time()
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        QWEN_OMNI_MODEL, **kwargs
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(QWEN_OMNI_MODEL)
    _captioner_log(f"Qwen-Omni model ready in {time.time() - start:.1f}s")
    return model, processor


def get_model(
    captioner_type: Literal["qwen_vl", "qwen_omni", "gemini_flash"],
    device: str = "auto",
    use_8bit: bool = False,
):
    """
    Load and cache the requested model. Evicts the previous model if type changes.
    Returns (model, processor) for local models, (None, None) for gemini_flash.
    """
    global _cached_type, _cached_model, _cached_processor

    if captioner_type == "gemini_flash":
        return None, None

    cache_key = f"{captioner_type}_{device}_{use_8bit}"
    with _cache_lock:
        if _cached_type == cache_key:
            _captioner_log(f"Using cached caption model: {cache_key}")
            return _cached_model, _cached_processor

        # Evict previous model
        if _cached_model is not None:
            _captioner_log("Evicting previous cached caption model")
            del _cached_model
            del _cached_processor
            torch.cuda.empty_cache()

        if captioner_type == "qwen_vl":
            model, processor = _load_qwen_vl(device, use_8bit)
        elif captioner_type == "qwen_omni":
            model, processor = _load_qwen_omni(device, use_8bit)
        else:
            raise ValueError(f"Unknown captioner_type: {captioner_type}")

        _cached_type      = cache_key
        _cached_model     = model
        _cached_processor = processor
        return model, processor


def unload_model():
    """Explicitly release the cached model from VRAM."""
    global _cached_type, _cached_model, _cached_processor
    with _cache_lock:
        if _cached_model is not None:
            del _cached_model
            del _cached_processor
            _cached_model     = None
            _cached_processor = None
            _cached_type      = None
            torch.cuda.empty_cache()


# ── Caption generation ────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    text = _BOILERPLATE.sub("", text).strip()
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def caption_image_qwen_vl(
    image_path: Path,
    model,
    processor,
    instruction: str,
    clean: bool = True,
) -> str:
    from qwen_vl_utils import process_vision_info  # pip install qwen-vl-utils

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": instruction},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=512)

    # Trim the input tokens from the output
    trimmed = [
        out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)
    ]
    result = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return _clean(result) if clean else result.strip()


def caption_image_qwen_omni(
    image_path: Path,
    model,
    processor,
    instruction: str,
    clean: bool = True,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": instruction},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=512, return_audio=False)

    trimmed = [
        out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)
    ]
    result = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return _clean(result) if clean else result.strip()


def caption_image_gemini(
    image_path: Path,
    instruction: str,
    api_key: str,
    clean: bool = True,
) -> str:
    import google.generativeai as genai  # pip install google-generativeai
    genai.configure(api_key=api_key)
    gm = genai.GenerativeModel(GEMINI_MODEL)
    img = Image.open(image_path).convert("RGB")
    response = gm.generate_content([instruction, img])
    result = response.text
    return _clean(result) if clean else result.strip()


def caption_image(
    image_path: Path,
    captioner_type: Literal["qwen_vl", "qwen_omni", "gemini_flash"],
    instruction: str,
    model=None,
    processor=None,
    api_key: str = "",
    clean: bool = True,
) -> str:
    """Unified entry point. Pass pre-loaded model/processor for local models."""
    if captioner_type == "qwen_vl":
        return caption_image_qwen_vl(image_path, model, processor, instruction, clean)
    elif captioner_type == "qwen_omni":
        return caption_image_qwen_omni(image_path, model, processor, instruction, clean)
    elif captioner_type == "gemini_flash":
        return caption_image_gemini(image_path, instruction, api_key, clean)
    else:
        raise ValueError(f"Unknown captioner_type: {captioner_type}")
