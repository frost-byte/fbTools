"""
fbtools-vision-llm — Modal cloud VLM app.

Deploy:
    cd /path/to/comfyui-fbTools
    /mnt/comfy_ssd/venvs/comfy-preflight/bin/modal deploy modal/app.py

The deployed app exposes a VisionLLM class that the fbTools LLM panel connects
to via utils/modal_vision_client.py.  Auth uses ~/.modal.toml (workspace frost-byte).

Supported model keys (choose in the LLM → Modal tab):
    qwen3-vl-8b          Qwen3-VL 8B Instruct        ~16 GB  bf16
    qwen2.5-vl-7b        Qwen2.5-VL 7B Instruct       ~14 GB  bf16
    qwen2.5-vl-32b-awq   Qwen2.5-VL 32B AWQ           ~18 GB  pre-quantized
    qwen2.5-vl-3b        Qwen2.5-VL 3B Instruct        ~6 GB  bf16
    gemma3-4b            Gemma 3 4B Instruct            ~8 GB  bf16

Custom HF repos (standard transformer repos only).

GPU: L40S (48 GB VRAM) fits all presets including 32B AWQ.
Scale-down window: 5 min (scales to zero automatically after idle).
"""

from __future__ import annotations

import modal

# ── Container image ────────────────────────────────────────────────────────────

_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "torchvision",
        "transformers>=4.52.0",  # Qwen3-VL support added in 4.52
        "accelerate>=0.34.0",
        "bitsandbytes>=0.43.0",
        "pillow",
        "qwen-vl-utils>=0.0.8",
    )
)

# ── App ────────────────────────────────────────────────────────────────────────

app = modal.App("fbtools-vision-llm", image=_image)

# ── Persistent model cache (avoids re-downloading on cold start) ───────────────

_vol = modal.Volume.from_name("fbtools-model-cache", create_if_missing=True)
_CACHE = "/models"

# ── Preset registry ────────────────────────────────────────────────────────────

_PRESETS: dict[str, dict] = {
    "qwen3-vl-8b": {
        "repo": "Qwen/Qwen3-VL-8B-Instruct",
        "arch": "qwen3_vl",  # distinct from qwen2.5-vl; uses Qwen3VLForConditionalGeneration
        "pre_quantized": False,
    },
    "qwen2.5-vl-7b": {
        "repo": "Qwen/Qwen2.5-VL-7B-Instruct",
        "arch": "qwen_vl",
        "pre_quantized": False,
    },
    "qwen2.5-vl-32b-awq": {
        "repo": "Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
        "arch": "qwen_vl",
        "pre_quantized": True,
    },
    "qwen2.5-vl-3b": {
        "repo": "Qwen/Qwen2.5-VL-3B-Instruct",
        "arch": "qwen_vl",
        "pre_quantized": False,
    },
    "gemma3-4b": {
        "repo": "google/gemma-3-4b-it",
        "arch": "generic",
        "pre_quantized": False,
    },
}


# ── Shared implementation (no Modal decorators) ───────────────────────────────

class _VisionLLMImpl:
    """Shared implementation — no Modal decorators.  Subclasses apply @app.cls(gpu=...)."""

    def setup(self) -> None:
        self._model      = None
        self._processor  = None
        self._arch       = None
        self._loaded_key = None   # (model_key, quantize) tuple

    def generate(
        self,
        prompt: str,
        *,
        model_key: str = "qwen3-vl-8b",
        quantize: bool = True,
        images: "list | None" = None,
        video_frames: "list | None" = None,
        system_prompt: str = "",
        max_tokens: int = 512,
        temperature: float = 0.7,
        video_meta: "dict | None" = None,
    ) -> dict:
        try:
            cache_key = (model_key, quantize)
            if self._loaded_key != cache_key:
                self._load_model(model_key, quantize)
                self._loaded_key = cache_key

            text = self._run(
                prompt,
                images=images,
                video_frames=video_frames,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return {"success": True, "text": text, "message": ""}
        except Exception as exc:
            return {"success": False, "text": "", "message": str(exc)}

    # ── Model loading ──────────────────────────────────────────────────────────

    def _load_model(self, model_key: str, quantize: bool) -> None:
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig

        preset = _PRESETS.get(model_key)
        if preset:
            repo          = preset["repo"]
            arch          = preset["arch"]
            pre_quantized = preset["pre_quantized"]
        else:
            repo          = model_key   # treat as raw HF repo ID
            arch          = "auto"      # detect from loaded model config
            # Name-based pre-quantization detection — prevents applying NF4 on top of
            # already-quantized weights (AWQ, GPTQ, FP4, etc.) for custom repos.
            _low = repo.lower()
            pre_quantized = any(t in _low for t in
                                ("-awq", "-gptq", "-nvfp4", "-fp4", "-fp8",
                                 "-int4", "-int8", "-gguf", "-ggml"))

        cache_dir = f"{_CACHE}/{model_key.replace('/', '__')}"

        bnb_cfg = None
        if quantize and not pre_quantized:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )

        load_kwargs: dict = dict(
            pretrained_model_name_or_path=repo,
            cache_dir=cache_dir,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        if bnb_cfg:
            load_kwargs["quantization_config"] = bnb_cfg

        if arch == "qwen3_vl":
            try:
                from transformers import Qwen3VLForConditionalGeneration
                self._model = Qwen3VLForConditionalGeneration.from_pretrained(**load_kwargs)
            except (ImportError, AttributeError):
                # transformers < 4.52 — fall back; should not happen with current image
                from transformers import AutoModelForCausalLM
                self._model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
                arch = "qwen_vl"  # keep vision inference path
        elif arch == "qwen_vl":
            from transformers import Qwen2_5_VLForConditionalGeneration
            try:
                self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(**load_kwargs)
            except Exception:
                from transformers import AutoModelForCausalLM
                self._model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
                arch = "generic"
        else:
            # "auto" (custom repo) — read config.json first (cheap, no weights) to pick
            # the right loading class without triggering class-mismatch warnings.
            from transformers import AutoConfig
            pre_cfg    = AutoConfig.from_pretrained(repo, cache_dir=cache_dir,
                                                    trust_remote_code=True)
            model_type = (getattr(pre_cfg, "model_type", "") or "").lower()
            has_vision = hasattr(pre_cfg, "vision_config")

            if "qwen3_vl" in model_type:
                arch = "qwen3_vl"
                from transformers import Qwen3VLForConditionalGeneration
                self._model = Qwen3VLForConditionalGeneration.from_pretrained(**load_kwargs)
            elif any(t in model_type for t in ("qwen2_5_vl", "qwen2_vl", "qwenvl")):
                arch = "qwen_vl"
                from transformers import Qwen2_5_VLForConditionalGeneration
                self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(**load_kwargs)
            elif has_vision:
                # VL model with non-standard model_type (e.g. qwen3_5 = Qwen3.8 multimodal).
                # AutoModelForConditionalGeneration reads architectures[] from config and
                # picks the correct multimodal class — no class-mismatch warning.
                arch = "qwen3_vl" if "qwen3" in model_type else "qwen_vl"
                from transformers import AutoModelForConditionalGeneration
                self._model = AutoModelForConditionalGeneration.from_pretrained(**load_kwargs)
            else:
                arch = "generic"
                from transformers import AutoModelForCausalLM
                self._model = AutoModelForCausalLM.from_pretrained(**load_kwargs)

        self._processor = AutoProcessor.from_pretrained(
            repo, cache_dir=cache_dir, trust_remote_code=True
        )
        self._arch = arch
        self._model.eval()

    # ── Inference ──────────────────────────────────────────────────────────────

    def _run(
        self,
        prompt: str,
        *,
        images: "list | None",
        video_frames: "list | None",
        system_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        if self._arch in ("qwen_vl", "qwen3_vl"):
            return self._run_qwen_vl(
                prompt,
                images=images,
                video_frames=video_frames,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        return self._run_generic(
            prompt,
            images=images or [],
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def _run_qwen_vl(
        self,
        prompt: str,
        *,
        images: "list | None",
        video_frames: "list | None",
        system_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        import torch
        from qwen_vl_utils import process_vision_info

        content: list = []
        if video_frames:
            content.append({"type": "video", "video": video_frames, "fps": 1.0})
        elif images:
            for img in images:
                content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})

        messages: list = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        text_input = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        proc_kwargs: dict = dict(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        # Qwen3-VL skips the per-frame pixel cap by default; enable it to match
        # qwen-vl-utils reference behavior and avoid excessive token consumption.
        if self._arch == "qwen3_vl":
            proc_kwargs["cap_pixels_per_frame"] = True
        inputs = self._processor(**proc_kwargs).to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 1e-6),
                do_sample=temperature > 0.01,
            )
        generated = [
            out[len(inp):]
            for out, inp in zip(output_ids, inputs.input_ids)
        ]
        return self._processor.batch_decode(generated, skip_special_tokens=True)[0]

    def _run_generic(
        self,
        prompt: str,
        *,
        images: list,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        import torch

        content: list = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": prompt})

        messages: list = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 1e-6),
                do_sample=temperature > 0.01,
            )
        prompt_len = inputs["input_ids"].shape[-1]
        return self._processor.batch_decode(
            output_ids[:, prompt_len:], skip_special_tokens=True
        )[0]


# ── GPU-tier subclasses ────────────────────────────────────────────────────────

_CLS_KWARGS = dict(
    timeout=600,
    scaledown_window=300,
    volumes={_CACHE: _vol},
    secrets=[modal.Secret.from_name("huggingface-vision")],
)


@app.cls(gpu="T4", **_CLS_KWARGS)
class VisionLLM_T4(_VisionLLMImpl):
    @modal.enter()
    def setup(self): super().setup()

    @modal.method()
    def generate(self, prompt, *, model_key="qwen3-vl-8b", quantize=True,
                 images=None, video_frames=None, system_prompt="",
                 max_tokens=512, temperature=0.7, video_meta=None):
        return super().generate(
            prompt, model_key=model_key, quantize=quantize,
            images=images, video_frames=video_frames,
            system_prompt=system_prompt, max_tokens=max_tokens,
            temperature=temperature, video_meta=video_meta,
        )


@app.cls(gpu="L4", **_CLS_KWARGS)
class VisionLLM_L4(_VisionLLMImpl):
    @modal.enter()
    def setup(self): super().setup()

    @modal.method()
    def generate(self, prompt, *, model_key="qwen3-vl-8b", quantize=True,
                 images=None, video_frames=None, system_prompt="",
                 max_tokens=512, temperature=0.7, video_meta=None):
        return super().generate(
            prompt, model_key=model_key, quantize=quantize,
            images=images, video_frames=video_frames,
            system_prompt=system_prompt, max_tokens=max_tokens,
            temperature=temperature, video_meta=video_meta,
        )


@app.cls(gpu="L40S", **_CLS_KWARGS)
class VisionLLM_L40S(_VisionLLMImpl):
    @modal.enter()
    def setup(self): super().setup()

    @modal.method()
    def generate(self, prompt, *, model_key="qwen3-vl-8b", quantize=True,
                 images=None, video_frames=None, system_prompt="",
                 max_tokens=512, temperature=0.7, video_meta=None):
        return super().generate(
            prompt, model_key=model_key, quantize=quantize,
            images=images, video_frames=video_frames,
            system_prompt=system_prompt, max_tokens=max_tokens,
            temperature=temperature, video_meta=video_meta,
        )


# Backward-compat alias — old deployments that reference "VisionLLM" still work
VisionLLM = VisionLLM_L40S
