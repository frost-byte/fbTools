# fbTools Backlog

Last updated: 2026-09-02

## Status

H3 refplan fully shipped (v1.13–1.17). 9-item polish backlog 8/9 complete (item 7 deferred).

## Remaining items

### 1. Subject-scoped file organization (deferred — breaking change)
Move files to `user_data_dir()/subjects/<subject_id>/audio/` and `/character_sheets/`.
Requires a migration script before any code lands.

### 2. Structured Prompt Editor (large, not started)
See `docs/structured_prompt_editor_plan.md` and memory `[[project-structured-prompt-editor]]`.
Prerequisite: settle scene JSON schema (ID-reference vs. embedded-data tension).

### 3. End-to-end H3 verification (live ComfyUI required)
`PromptCompositionLoader → CompositionToH3Conditioning → sampler`. Confirmed: mixed case (video char ref + extracted audio + image ref). Still needed:
- images-only
- single video + synchronized soundtrack
- two videos (`<Video 1>`/`<Video 2>` ordinals + ref_image_size scaling)
- video + standalone audio file

### 4. Structural (subject-free) video references in H3 prompts
`<Video N>` for camera movement / editorial rhythm with no subject attached.
Allow cast entries with no `subject_id` and a `role_description`. Scope: `_build_ref_map()`, cast entry schema, UI "structural" entry type.

### 5. Per-image role descriptions for `<Picture N>`
Change `character_sheet_images` from `list[str]` → `list[{file, role}]`.
Roles: `"character sheet"`, `"portrait"`, `"side profile"`, `"full body"`, `"costume detail"`.
Scope: `utils/subject_profiles.py`, SubjectProfileDefine/Load nodes, `_build_ref_map()`, `_assemble_h3_ref2va()`, migration script.
Prerequisite for items 6 and 7.

### 6. Outfit as Subject N (dual-path)
Text-only = fast. Media-backed = emit separate `<Subject N>` with `<Picture N>` references.
Scope: OutfitRegistry schema (add `reference_images`), OutfitDefine + sidebar editor, `_build_ref_map()` + `_assemble_h3_ref2va()`.
Spec: `docs/prompt_assembly.md` § B-1.

### 7. Background as Subject N
When `_background_snapshot` has reference media, emit dedicated Subject line + include background images in IMAGE batch.
Scope: background profile schema (add `reference_images`, `reference_video`), background editor (media picker), `_build_ref_map()` + `_assemble_h3_ref2va()`.
Spec: `docs/prompt_assembly.md` § B-2.

### 8. Audio reference pipeline (partial — D/E/F remaining)
A–C already shipped. Remaining:

**D. CompositionToH3Conditioning validation** (§1 invariants):
- Per-clip: 2–15 s (fail if < 2 or > 15)
- Total audio ≤ 15 s; count ≤ 3 audio refs
- Pairing required (audio must accompany image/video)
- Total mixed files ≤ 12; Turbo LoRA warning when voice audio ref present
- Hard error, not silent truncation

**E. Audio preprocessing pipeline:**
- Bundle schema: `audio_processing: {vocal_isolation, cleanup, mastering, normalize_lufs}` + `audio_cache`
- Pipeline: MelBand Roformer → VRGDG_CleanAudio → MusicTools mastering → LUFS normalize → loop/truncate [2s, 15s]
- Cache: `user_data_dir()/bundles/<id>/audio_proc_<fingerprint8>.wav`
- REST: `POST /fbtools/bundles/preprocess_audio`
- UI: processing toggles, LUFS slider, "Process Audio" button, status badge in Bundle Editor

**F. Global settings** (Prompt Compositions tab):
- chars_per_second (or slow/normal/fast WPM)
- Melband model path selector; default audio processing flags

Observations log: `docs/audio_reference_observations.md`

### 9. MiniMax-H3 Prompt Rewriter LoRA integration (evaluation / comparison)
`lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-Omni` — a PEFT LoRA adapter for Qwen2.5-Omni-7B that rewrites scene descriptions into structured MiniMax-H3 prompts (`integrated_multimodal_description`, `overall_soundscape`, `non_diegetic_music`; Ref2AV expands to six sections including `subject_definitions` and `retention_analysis`).

**Why:** Compare raw assembled prompts from SceneCompose/PromptAssemble against LoRA-rewritten versions as a quality signal before committing H3 generation runs. Ref2AV mode maps directly to our Bundle/Source Profile architecture (`<Subject N>`, `<Video N>`, `<Audio N>` labels).

**Two tiers:**
- Tier 1 (zero new deps): feed `system_prompt_for_task("ref2av")` from their `system_prompt.py` to the existing quantized Qwen2.5-Omni via `llm_client` — structured output without the LoRA.
- Tier 2 (full LoRA): fp16 base model + `peft` library. Requires ~14 GB VRAM; Modal L40S dispatch is the right venue. Needs `pip install peft` and confirming whether fp16 Qwen2.5-Omni weights are on disk (only GPTQ-int4 may be present).

**Entry point:** clone `https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA-Omni`, run `infer.py` standalone before any extension integration.

### 10. LLM scanner / Source Profiles (Phase 7)
See memory `[[project-llm-assistant]]`. model paths on this machine. llama-cpp-python not yet installed.

### 10. Multi-asset Subject definitions (appearance vs. motion split)
When Subject has both `picture_nums` and `video_num`, emit split phrasing:
`<Subject 1> is the woman whose appearance comes from <Picture 1> and whose motion comes from <Video 1>.`
Optional `motion_role` field on cast entries.
Scope: `_assemble_h3_ref2va()`, cast entry schema. Spec: `docs/prompt_assembly.md` § B-3.

### 11. Bundle-owned media resources (copy-on-assign)
Copy images/audio into `user_data_dir()/bundles/<bundle_id>/images/` (or `/audio/`) on assign.
Store canonical path relative to bundle dir. Videos remain as references (too large to copy).
Old entries (no `source: "bundle"`) continue working via existing fallback.
Scope: Bundles REST API (copy endpoint), bundle_editor.js, `/fbtools/bundles/{id}/media/{file}` route, optional migration utility.
