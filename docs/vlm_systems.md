# VLM Systems in fbTools

Two separate VLM (vision-language model) systems exist in this codebase. They were built independently for different purposes and converge in Source Profiles. This doc exists to prevent them from being confused or inadvertently merged.

---

## System A — `captioner.py` (Dataset Captioning)

**Origin:** Built for the `DatasetCaptioner` node. Captions batches of still images for training dataset preparation.

**Backends:**
- `qwen_vl` — `Qwen/Qwen2.5-VL-7B-Instruct` loaded in fp16 (~14 GB). No VRAM cap.
- `qwen_omni` — `Qwen/Qwen2.5-Omni-7B` loaded in fp16. No VRAM cap.
- `gemini_flash` — Google Gemini Flash via API key. No local GPU usage.

**How it loads:** On-demand inside each request handler. A model is loaded, used, and left in memory. There is no explicit load/unload UI for this path.

**When to use:** Dataset captioning only (`DatasetCaptioner` node, `/fbtools/dataset_caption/*` endpoints). Do not use this path for any feature that has a user-facing "load model" step.

**Key file:** `captioner.py` — `get_model()`, `caption_image()`, `unload_model()`

---

## System B — `llm_client.py` (Prompt Composition LLM)

**Origin:** Built for the `PromptComposer` node and the Compose tab's LLM panel. Supports interactive text generation, vision queries, and video description for scene/prompt workflows.

**Backends:**
- GGUF models via llama-cpp-python (requires rebuild on machines without AVX-512 — see `system_llama_cpp_rebuild.md`)
- GPTQ and other quantized formats via transformers
- Any backend supported by the loader configured in the LLM panel

**VRAM cap:** GPU allocation capped at 70% (`gpu_total * 0.70`) to leave headroom for ComfyUI's active model.

**How it loads:** User explicitly selects and loads a model via the Compose tab → LLM panel. The model stays resident until unloaded. Status is visible in the fbTools panel header badge.

**When to use:** Any feature where the user is expected to load a model first — Source Profile analysis, detect segments, describe clip, prompt generation, video description. This is the correct path for all new VLM features.

**Key file:** `utils/llm_client.py` — `backend_status()`, `generate(prompt, *, images=[...])`, `load_model()`, `unload_model()`

---

## Routing Rule

`_run_vision_inference()` in `extension.py` is the single gateway for Source Profile VLM calls. Its contract is:

1. `captioner_type="gemini_flash"` → System A, Gemini path only (no local GPU)
2. All other values (including `"auto"`) → System B, **requires a model to already be loaded in the LLM panel**. Raises `RuntimeError` with a clear message if no model is loaded or the loaded model lacks vision support. Does **not** fall back to captioner.py.

There is intentionally no silent fallback between the two systems. If the user needs a model for Source Profile analysis, they load it in the LLM panel. If they want Gemini, they pass an API key and select it explicitly.

---

## Gemini in the Clips UI

The "Gemini" checkbox in the Source Profile clips section (detect boundaries, describe clip) maps directly to `captioner_type="gemini_flash"` in the request body. It is the only escape hatch to System A from Source Profiles. The analyze section also accepts `"gemini_flash"` via its captioner selector.

---

## What NOT to do

- Do not add a captioner.py fallback to `_run_vision_inference` "for safety". The fp16 load is what caused the 90% VRAM issue in the first place.
- Do not add `qwen_vl` / `qwen_omni` as selectable options in the Source Profile clips UI. Those are System A backends with no VRAM cap; directing users to the LLM panel is correct.
- Do not import `get_model` / `caption_image` from `captioner.py` inside new source profile endpoints. Route through `_run_vision_inference` instead.
