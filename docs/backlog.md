# fbTools Backlog

Last updated: 2026-10-07

## Status

H3 refplan fully shipped (v1.13–1.17). 9-item polish backlog 8/9 complete (item 7 deferred).

## Remaining items

### 1. Subject-scoped file organization (deferred — breaking change)
Move files to `user_data_dir()/subjects/<subject_id>/audio/` and `/character_sheets/`.
Requires a migration script before any code lands.

### 2. Structured Prompt Editor (large, not started)
See `docs/structured_prompt_editor_plan.md` and memory `[[project-structured-prompt-editor]]`.
Prerequisite: settle scene JSON schema (ID-reference vs. embedded-data tension).

### 3. End-to-end H3 verification — done (2026-10-07)
Verified live by Bee across extended use: video, image and audio references all work through
`PromptCompositionLoader → CompositionToH3Conditioning → sampler`. No separate per-combination
checklist remains. Revisit only if a specific combination regresses.

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

### 8. Audio reference pipeline (D done; E/F mostly shipped)
A–C shipped earlier. Status checked 2026-10-07:

**D. CompositionToH3Conditioning validation — done.** All raise in `execute`
(`nodes/compositions.py`), helpers in `utils/prompt_assembler.py`:
- ≤ 3 standalone audio, audio must accompany an image/video, ≤ 12 total refs, `trim_to` ≥ 2 s
  (`validate_h3_refs_pre`)
- per-clip 2–15 s after trim (`validate_h3_audio_clip`), total ≤ 15 s (`validate_h3_audio_total`)
- any reference that fails to load (image, video, soundtrack, standalone audio) is a hard error that
  lists every failure (`validate_h3_load_failures`), instead of being silently skipped
- Turbo LoRA + audio reference → warning (log + status line)

Not enforced (deliberately left out): soundtrack audio is not duration-checked or counted toward the
audio count/total; the Turbo warning does not look at `role`; ≤ 9 images / ≤ 3 videos / video
2–15 s only warn in `H3ReferenceSummary`. The local native node enforces no audio durations.

**E. Audio preprocessing — mostly shipped.** MelBand Roformer vocal extraction (spectral denoise
fallback), LUFS normalize, loop-to-2 s / truncate-to-15 s (silent), cache at
`user_data_dir()/bundles_cache/<bundle_id>/audio_<fp8>.wav`, `POST /fbtools/bundles/preprocess_audio`,
Bundle Editor toggles + "Process Audio" button + status badge. Real schema keys are
`audio_processing: {noise_removal, normalize_lufs, target_lufs}` plus `audio.audio_cache`.
Remaining: VRGDG_CleanAudio and MusicTools mastering steps; separate vocal_isolation/cleanup flags.

**F. Global settings — partly shipped.** `default_speech_pace` (slow/normal/fast),
`melband_model_path` (text field), `default_audio_noise_removal` / `default_audio_normalize_lufs` /
`default_audio_target_lufs` exist. Remaining: numeric `chars_per_second` override (presets are
hard-coded in `utils/prompt_assembler.py`), a MelBand model dropdown, and confirming the
`default_audio_*` values actually seed new bundles.

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
