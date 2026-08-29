# Modal Vision Backend — Integration Handoff

Handoff notes for wiring a Modal-cloud-hosted vision LLM into fbTools as a
third VLM backend option, alongside the two systems documented in
[`vlm_systems.md`](vlm_systems.md). **Read that doc first** — this one
assumes it.

Nothing in fbTools has been changed yet. This documents what already
exists on the Modal side and what integrating it would touch.

## What's already built

A separate repo, **https://github.com/frost-byte/fbtools-vision-modal**
(public, GitHub account `frost-byte`), contains a working Modal app:

- **`vision_llm.py`** — Modal app `fbtools-vision-llm`, class `VisionLLM`.
  `VisionLLM.generate()` is **signature-compatible with
  `utils/llm_client.py`'s `generate()`**: `prompt`, `images`,
  `video_frames`, `system_prompt`, `max_tokens`, `temperature`,
  `video_meta` — same keys, same return shape (`{success, text, message}`).
  It was built specifically so it could later stand in for a call to
  `_llm_client.generate(...)`.
- **`subject_analysis.py`** — a verbatim port of this repo's
  `utils/source_profile_analysis.py` (prompt builders + JSON parsers for
  subject-decomposition passes and segment/boundary detection). Two of its
  parsers (`_parse_segments_response`, `_parse_vlm_json_response`) were
  patched there to also accept a bare JSON array response, since
  Qwen2.5-VL sometimes omits the `{"segments": [...]}` / `{"subjects":
  [...]}` envelope despite the schema instruction. Worth porting that same
  leniency fix into this repo's `utils/source_profile_analysis.py` if it
  isn't already fixed here.
- Model registry: `qwen2.5-vl-7b` (default), `qwen2.5-vl-3b`,
  `qwen2.5-omni-7b`, `gemma3-4b` — or any Hugging Face repo id passed
  directly. Optional 4-bit (bitsandbytes NF4) quantization, validated to
  keep vision quality while cutting weight memory roughly 3x. Runs on an
  L40S (48GB) GPU; a T4 (16GB) OOMs outright on a 7B model. Video frames
  are downscaled to a 448px max side before encoding — full-res multi-frame
  video calls OOM'd even on L40S otherwise.
- All of the above validated end-to-end against real fbTools input files
  (image description, native multi-frame video segment detection, 4-bit
  quantized inference, and a gated model load via a Modal secret
  `huggingface-vision` holding `HF_TOKEN`).
- Full setup/usage instructions are in that repo's `README.md` — don't
  duplicate them here, just reference them.

## Critical: this must slot in as a third explicit backend, not a fallback

Per `vlm_systems.md`'s "Routing Rule" and "What NOT to do" sections,
`_run_vision_inference()` in `extension.py` is the single gateway for
Source Profile VLM calls, and the two existing systems are deliberately
**not** allowed to silently fall back into each other — the user
explicitly picks Gemini (`captioner_type="gemini_flash"`) or the LLM-panel
model (any other value, requires a model already loaded, raises if not).

A Modal backend should follow the same pattern: a new explicit
`captioner_type` value (something like `"modal"`), never a default and
never a fallback destination. It also isn't quite System A or System B —
functionally it behaves like System B (an already-loaded, vision-capable,
potentially native-video model with `generate(prompt, images=..., video_frames=...)`)
except "loaded" means "a Modal container is warm" rather than "resident in
local VRAM." Whoever picks this up should decide whether that means:

- adding it as a value `_llm_client.generate()`-compatible caller can pass
  through unchanged (if `_run_vision_inference`/`_run_vision_inference_clip`
  get a thin `utils/modal_vision_client.py` with the same public surface as
  `llm_client.py` — `backend_status()`, `generate()` — swapped in when
  `captioner_type == "modal"`), or
- a fully separate branch alongside the existing `gemini_flash` one.

The former keeps `_run_vision_inference`'s branching shape unchanged; the
latter is more explicit but duplicates a bit more logic. Not resolved here
— the fbTools-side agent should decide, since it owns that codebase's
conventions.

## Exact call sites (line numbers as of this writing — verify before editing)

**Backend (`extension.py`):**
- `_run_vision_inference()` — around line 13109. Single-frame/image gateway.
- `_run_vision_inference_clip()` — around line 13162. Multi-frame video
  gateway; already branches on `st.get("native_video")` to choose between
  passing `video_frames` natively vs. building a contact-sheet fallback —
  a Modal-backed model would want the same branch, using its own
  `native_video` flag (the Modal app already tracks this per model, see
  `VisionLLM.native_video` in `vision_llm.py`).
- `/fbtools/source_profiles/analyze` route — around line 13216.
- `/fbtools/llm/*` routes (`models`, `status`, `load`, `unload`,
  `generate`, `generate/shot_action`, `generate/dialogue`, ...) — around
  line 16444 onward. These back the Compose → LLM panel itself; per
  `vlm_systems.md` this is "the correct path for all new VLM features," so
  it's worth considering whether Modal should be selectable *here* (as a
  loadable "model") rather than only in Source Profiles.

**Frontend (JS):**
- `js/ui/source_profile_editor.js` — `CAPTIONER_TYPES` array at line 38
  (`["auto", "qwen_vl", "qwen_omni", "gemini_flash"]`) drives the Source
  Profile *analyze* dropdown. The *clips* section (segment detection /
  describe clip) instead uses a single Gemini checkbox: `useGeminiForClips`
  state (~line 652), the checkbox element (~line 737-739), and its use when
  building the request body (~line 1244, ~line 1296, as
  `captioner_type: "gemini_flash"`). A Modal option would mirror this
  checkbox pattern for the clips section, and/or add a `"modal"` entry to
  `CAPTIONER_TYPES` for the analyze dropdown.
- `js/api/source_profiles.js` — passes `captioner_type` (and
  `gemini_api_key`) through to the backend routes; would need the same for
  a Modal selection, minus any API key (Modal auth is server-side via the
  token already configured in the ComfyUI Python environment, not something
  the frontend sends).
- `js/api/llm.js` / `js/ui/composition_editor.js` — the Compose → LLM
  panel's frontend, if Modal is exposed there too (see routing note above).

## Calling the deployed Modal app from fbTools' Python process

fbTools runs inside ComfyUI's own Python environment, not the `.venv` in
the `fbtools-vision-modal` repo. To call the deployed app from there:

1. `modal` (the Python client) needs to be a dependency of *this* repo
   (add to `requirements.txt`, following the optional-dependency pattern
   already used for `llama_cpp`/`transformers` in `utils/llm_client.py` —
   guard the import and degrade gracefully if it's missing).
2. The Modal auth token (`~/.modal.toml`, workspace `frost-byte`) needs to
   be readable by whatever user/environment runs ComfyUI — same as any
   other machine calling this app.
3. From Python: `modal.Cls.from_name("fbtools-vision-llm",
   "VisionLLM")(model_key=..., quantize=...).generate.remote(...)`. Note
   `extension.py`'s existing aiohttp route handlers wrap blocking local
   calls in `loop.run_in_executor(None, ...)` (see `_llm_generate` etc.,
   around line 16503) — Modal's client has an async-native
   `.generate.remote.aio(...)` which would fit the same routes more
   naturally than `run_in_executor`, but either works.
4. Cold starts on Modal (container spin-up + model load) can take on the
   order of a minute if no container is warm — worth surfacing via the
   existing `send_status_update(...)` pattern used elsewhere in
   `extension.py` for progress, so it doesn't look hung.

## Not yet explored

- Whether `app.deploy()`'s `min_containers`/keep-warm settings are worth
  paying for here, to avoid cold-start latency on first use per session.
- Cost implications of L40S usage relative to local GPU electricity/time —
  not evaluated, would need real usage patterns.
- Whether the Modal registry's small model set is sufficient, or whether
  fbTools' local `llm_scanner.py`-style capability tagging is worth mirroring
  for Modal's arbitrary-repo-id path (currently just `native_video`
  auto-detected from architecture; no quant-type/size hints surfaced to the
  UI the way local model scanning provides).
