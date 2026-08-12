# CHANGELOG


## v1.13.0 (2026-08-12)

### Features

- **libber**: Add random wildcard notation for libber key selection
  ([`13139aa`](https://github.com/frost-byte/fbTools/commit/13139aaaf74f0e6a9445432574b88f06bbdefd3c))

Add %*:N% (random from libber N) and %*% (random from combined pool) notation to the composition
  libber substitution system.

Each occurrence draws from a per-libber shuffled deque (sampling without replacement), so no key
  repeats until every key in that libber has been used at least once. When exhausted the queue
  refills with a new shuffle. The combined %*% pool interleaves all attached libbers before
  shuffling.

Pass order: %*% (combined) → %key:N% / %*:N% (indexed) → %key% (chained).

Frontend: completion popup shows random entries in amber italic for each attached libber (and a
  combined "any" entry when multiple are attached). Typing `*` after the delimiter filters to show
  only random entries.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.12.0 (2026-08-12)

### Features

- **scene**: Add Outfit Registry system
  ([`0323e6d`](https://github.com/frost-byte/fbTools/commit/0323e6defe1fa1f4e9513362ecded1112228fb76))

Add utils/outfit_registry.py with OutfitRegistry class (load/save/define/ remove/list), three new
  nodes (OutfitRegistryLoad, OutfitDefine, OutfitList), and OUTFIT_REGISTRY custom type wired into
  SceneCompose.

SceneCompose gains optional outfit_registry + outfit_A_id–outfit_D_id inputs: explicit text
  overrides still win; registry descriptions fill in when no text override is provided.

REST API: GET/POST /fbtools/outfits/registry|save|reload, DELETE /outfits/delete.

Frontend: Outfits sidebar section in the Composition Editor with list/edit/ delete, modal editor
  (id, name, description, tags), and LLM-assisted image analysis button (visible only when a vision
  model is loaded).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.11.0 (2026-08-12)

### Features

- **ui**: Add LoRA association and concept_id to Prompt Compositions
  ([`a339c95`](https://github.com/frost-byte/fbTools/commit/a339c956104718af02ac40fda33e41335be1e0a7))

- Composition schema gets `loras: [{name, weight, target}]` and `concept_id` fields - New LoRAs
  section in editor: Add LoRA button creates rows with name dropdown, weight input, and model_target
  selector; outputs as LORA_STACK_DATA pin on PromptCompositionLoader → wire into LoraStackApply -
  Composition-level concept_id in Info section; merged with per-subject concept IDs on the
  concept_ids output of PromptCompositionLoader - Concept ID now editable on each assigned subject
  slot row (saves to subject_profiles.json via subjects/save merge, no full reload needed) - New GET
  /fbtools/loras/list endpoint returns sorted LoRA filename list

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.10.0 (2026-08-12)

### Features

- **ui**: Add Libber integration to Prompt Composition editor
  ([`5420946`](https://github.com/frost-byte/fbTools/commit/5420946b0f13ba1b9656fcac7c1e1dd0294e1a7f))

- Composition schema gets a `libbers: []` field (attached libber files) - New Libbers section in
  editor form: check/uncheck to attach libbers, attached libbers show their keys as amber monospace
  chips - `%key%` completion in ALL text fields (style, camera, action, dialogue, soundscape,
  music): triggers on delimiter char, shows key + libber name, auto-inserts closing delimiter;
  %key:N% notation for disambiguation when the same key exists in multiple attached libbers (1-based
  index) - Global Settings section at bottom of sidebar: single-char delimiter input (default %),
  persisted to composition_settings.json via REST - PromptCompositionLoader node applies attached
  libbers to the assembled prompt at execute time, honouring the configured delimiter and :N indexed
  references; fingerprint includes settings file mtime so the node re-executes when settings change

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.9.1 (2026-08-11)

### Bug Fixes

- **ui**: Move composition Name+Model into collapsible Info section
  ([`557e7d5`](https://github.com/frost-byte/fbTools/commit/557e7d5a76143742e002e0e52adcb7310a9b7b33))

Replace the standalone top bar with an Info section at the top of the scrollable form, matching the
  Style/Subjects/Shots collapsible pattern. Name and Model each get their own labeled row
  (fbt-ce-info-row + fbt-ce-info-label) so neither field is squished when the model dropdown has a
  long selected value.

Also initialize _newComp() with name: "" instead of "New Composition" to prevent silent data
  corruption when the name field is visually small and a user types into it unaware that a default
  value is already present.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.9.0 (2026-08-11)

### Bug Fixes

- **extension**: Use relative imports for late utils imports
  ([`f0e3297`](https://github.com/frost-byte/fbTools/commit/f0e32972f7a225e6529dae33a033cf0b211180b5))

All utils imports in the Prompt Composition and LLM route blocks were using bare absolute form (from
  utils.x import) which fails when the package is loaded by ComfyUI as a relative package. Changed
  to the same dot-prefix relative form used everywhere else in extension.py.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **llm**: Replace cross-module get_logger with stdlib logging in llm_client
  ([`7bc4afd`](https://github.com/frost-byte/fbTools/commit/7bc4afd4b488d65375b5347540fb63408f7cc00c))

Pure utils modules have no cross-module deps. Using get_logger from logging_utils caused a
  ModuleNotFoundError at ComfyUI load time because utils/ has no __init__.py and the import path was
  absolute. Replace with logging.getLogger(__name__) consistent with other standalone utils modules.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **llm**: Scanner skips root dir to avoid misidentifying stray GGUF files
  ([`4c67a7d`](https://github.com/frost-byte/fbTools/commit/4c67a7d5ff3e8a3626696c4c05ecac9da6e0aa7a))

_scan_directory now iterates root's children rather than treating root itself as a candidate model
  dir. Fixes the case where a loose text-encoder .gguf (e.g. umt5-xxl-encoder-Q8_0.gguf) in the LLM
  root causes the entire directory to be returned as a single model and recursion to stop, hiding
  all nested model subdirectories.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **llm**: Use "LLM" (singular) as the canonical folder_paths key
  ([`e797ade`](https://github.com/frost-byte/fbTools/commit/e797ade17e70924a2d6eaeb73ce158ac02bf3f40))

The ComfyUI convention, established by ComfyUI-MiniMaxH3-Prompt-Writer and comfyui_llm_party, is
  "LLM" not "LLMs". Scanner now checks "LLM" first with "LLMs" as fallback, and defaults to
  models/LLM/ when neither is registered.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **scene**: Fix two bugs in assemble_composition adapter + add tests
  ([`80fdd9a`](https://github.com/frost-byte/fbTools/commit/80fdd9a0954e5f4725eea52474992caeef97059d))

Dialogue map was keyed by positional counter (shot_1, shot_2) but the template shot lookup uses the
  shot's actual id field — so dialogue in shot N with non-dialogue shots before it was never
  emitted. Fix: key dialogue map by shot["id"] directly.

speaker_slot was absent from the template dialogue dict produced by _composition_shots_to_template,
  so h3_ref2va / h3_fl2va always fell back to "en-us" regardless of the subject's configured
  language. Fix: include speaker_slot (remapped S1→A via slot_map) in the dict.

Adds test_assemble_composition.py (42 tests) covering: - S1/S2 → A/B slot remapping - {S1}/{S2}
  placeholder replacement in action/camera text - Dialogue positional mapping by shot ID - Dialogue
  language tag from speaker's voice.language - Background description, lighting, soundscape
  integration - Composition soundscape overrides background soundscape - Style, music, outfit
  overrides, concept IDs - All 8 model types produce non-empty output - {S} placeholders do not leak
  into any model's output - Edge cases: empty subjects, empty shots, 3-subject mapping

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Initialise composition state before building panel
  ([`e3d00b1`](https://github.com/frost-byte/fbTools/commit/e3d00b1b6c7ed220fe83cb9ba62c98d026d3a93b))

_S.composition was null when _buildPanel called _rebuildShots during first render, causing a
  TypeError on .shots. Moving _newComp() before _buildPanel ensures state is ready before any DOM
  callbacks execute.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

### Chores

- **nodes**: Unregister MultiLoraLoader, SceneWanVideoLoraMultiSave, LoraStackView
  ([`bba5915`](https://github.com/frost-byte/fbTools/commit/bba5915dc7e33b72468668143dcb3f43c4283b94))

Workflow audit (338 workflows scanned): - MultiLoraLoader: present in 1 workflow but fully
  disconnected (no inputs or outputs wired) — confirmed never functional -
  SceneWanVideoLoraMultiSave: zero workflow references - LoraStackView: was already absent from
  get_node_list(); made explicit

Class definitions retained in extension.py for reference. LoraEntryDefine and LoraStackCollect kept
  — still active in 11-13 workflows each.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

### Code Style

- **nodes**: Normalize display names to Title Case with spaces
  ([`330870f`](https://github.com/frost-byte/fbTools/commit/330870f21a4376c0f7f4789dd34862df24a45ea1))

All 29 node display_name values that used verbatim CamelCase class names are updated to Title Case
  with spaces. FBTextEncodeQwenImageEditPlus is shortened to "FB Qwen Image Edit Plus" to avoid
  collision with similarly named nodes from other packages. Node IDs are unchanged so existing
  workflows are unaffected.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Unify LoraStackBuilder info icon with ConceptDefine style
  ([`d4488f9`](https://github.com/frost-byte/fbTools/commit/d4488f9de09fb0ed9a81b2e1df8c98f31d0bf7d6))

Remove the explicit circle (arc + stroke) from _lsbDrawIcon and replace with the same approach as
  _cdDrawIcon: bold "i" centered directly in the rounded rect, font size proportional to icon size
  (sz * 0.55). Both icons are now 18px rounded rects with the same visual weight.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

### Documentation

- Add user-facing docs for Scene Composition Engine nodes
  ([`6a6a330`](https://github.com/frost-byte/fbTools/commit/6a6a3304468fa7562cd64cc744ae45f8c43b82aa))

Four new end-user reference docs covering all Phase 1–4 nodes: concept_registry.md,
  subject_profiles.md, scene_composition.md, prompt_assembly.md. Each covers inputs/outputs, typical
  workflow diagrams, and storage locations.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **nodes**: Add missing tooltip strings to SubjectProfileDefine, ConceptDefine, DatasetCaptioner,
  TailEnhancePro
  ([`321f30d`](https://github.com/frost-byte/fbTools/commit/321f30dfafee3262684ac84b0bea8adf6191287b))

SubjectProfileDefine: name, face, hair, body, default_outfit

ConceptDefine: description

DatasetCaptioner: device

TailEnhancePro: all 12 processing parameter inputs (tail_count, ref_window, deflicker, color match,
  unsharp, bilateral filter)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

### Features

- **cast**: Add Reference Bundle and Scene Cast data layer
  ([`7f0e422`](https://github.com/frost-byte/fbTools/commit/7f0e4228f33d3f2cb609ecb37eff1e2e8673db1a))

Pure-utils modules (no ComfyUI deps) for the Reference Bundle & Scene Cast system (spec §1 data
  layer):

- utils/reference_bundles.py — BundleRegistry with upsert/delete/filter-by-subject, validation
  (visual/audio source constraints), JSON persistence with .bak backup - utils/scene_casts.py —
  CastRegistry with upsert/delete, per-entry update (bundle, visual_mode, use_audio), remove_entry,
  resolve_cast_for_subject, validation - tests/test_reference_bundles.py — 29 tests covering CRUD,
  immutability, filtering, serialisation roundtrip, persistence, and all validation rules -
  tests/test_scene_casts.py — 40 tests covering all of the above plus update_entry partial-update
  semantics and append-on-new-subject behaviour

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **cast**: Add Reference Bundle and Scene Cast REST endpoints
  ([`5f30c5c`](https://github.com/frost-byte/fbTools/commit/5f30c5c4f1ab2bfb47e48518a9695b0304145f0c))

Wires the step-1 utils into extension.py via 9 new aiohttp routes:

Reference Bundles (4 routes): GET /fbtools/bundles/list — all bundles, optional ?subject_id= filter
  GET /fbtools/bundles/get — single bundle by ?id= POST /fbtools/bundles/save — create / update
  (upsert) DEL /fbtools/bundles/delete — remove by ?id=

Scene Casts (4 routes): GET /fbtools/casts/list — all casts GET /fbtools/casts/get — single cast by
  ?id= POST /fbtools/casts/save — create / update (upsert) DEL /fbtools/casts/delete — remove by
  ?id=

Media listing (1 route): GET /fbtools/media/list — files from input/ dir filtered by
  ?type=image|video|audio|all

Also adds _IMAGE_EXTENSIONS and _VIDEO_EXTENSIONS constants alongside the existing
  _AUDIO_EXTENSIONS, and path helpers default_bundle_registry_path() and
  default_cast_registry_path() following the same pattern as other registries.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **cast**: Add Reference Bundle Editor sidebar panel
  ([`48fd32c`](https://github.com/frost-byte/fbTools/commit/48fd32c4d16188ee8dccf286c4757cb0cf93c657))

New sidebar tab "Reference Bundles" (pi pi-images icon) for creating and managing reference media
  bundles tied to subject profiles:

js/api/bundles.js: BundlesAPI client covering bundles (list/get/save/delete), casts
  (list/get/save/delete), subjects/list, and media/list — shared by both the Bundle Editor (step 3)
  and the upcoming Cast Editor (step 4)

js/ui/bundle_editor.js: Full panel implementation: - Top bar: subject-filter dropdown, free-text
  search, + New button, ↺ refresh - List view: bundles grouped by subject, each card shows name,
  VIDEO/IMAGES badge, audio indicator (🎙), tag chips, edit + delete actions - Editor form: name,
  auto-generated ID (editable), subject dropdown, appearance override, visual toggle (Images/Video)
  with file pickers, audio 3-way toggle (None/Extract from video/Separate file) with picker, tags,
  save/cancel - Image list: ordered with ↑↓ reorder and × remove; add-image dropdown shows only
  files not yet selected - Extract-from-visual warning when visual mode is Images

js/styles/style.css: All fbt-be-* styles for panel, top bar, card list, group headers, badges, tags,
  toggle buttons, image list rows, and form sections

js/fb_tools.js + js/index.js: Register the new sidebar tab and export renderBundleEditor

js-tests/bundles_api.test.js: 19 tests covering all BundlesAPI methods including URL encoding, query
  param passing, body serialisation, and DELETE error handling

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **cast**: Add Scene Cast system and video/audio reference params
  ([`ee99c55`](https://github.com/frost-byte/fbTools/commit/ee99c5539d1a9dce2698b88d95e5741fa8b0840e))

Reference Bundle & Scene Cast system: - Scene Cast Editor sidebar panel (js/ui/cast_editor.js) with
  two-line entry rows, bundle dropdown filtered by subject, visual mode toggle with amber 'differs'
  highlight, and fire-and-forget cast reload after save/delete - SceneCastLoad node + SCENE_CAST
  custom type; reload counter wired to POST /fbtools/casts/reload - PromptCompositionLoader:
  optional SCENE_CAST input; resolves reference media (video path + image batch) and builds
  video_entries for assembler - BundlesAPI.reloadCasts() client method

Prompt assembler extensions: - Character sheets cited inline inside <Subject N> block ("Character
  sheets: <Picture N> (primary identity)") per official H3 guide; removed standalone picture entries
  from subject_definitions and retention_analysis - <Video N> reference labels in
  subject_definitions (after subject blocks), retention_analysis, and summary - assemble_prompt() /
  assemble_composition() accept video_entries list

Video/audio reference frame-sampling params: - visual block gains force_rate, frame_load_cap,
  skip_first_frames, select_every_nth for the Load Video node - audio extract_from_visual gains its
  own independent set of four frame params (separate Load Video node instance, different segment
  from visual) - audio file source gains start_time and duration (seconds) for Load Audio -
  _resolve_cast_media() returns a 14-key dict covering all params - PromptCompositionLoader grows 12
  new output pins: video frame params, audio_source, audio_file, audio frame params,
  audio_start_time, audio_duration - Bundle editor UI shows frame-sampling grids for video visual
  and extract_from_visual audio, and a Timing grid for file audio

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **editor**: Phase 7 — LLM assistant for Composition Editor
  ([`be1f54c`](https://github.com/frost-byte/fbTools/commit/be1f54c54f7ae92352976b0263a33fb938265bb6))

Add a local-LLM assistant panel to the Prompt Composition Editor sidebar.

Scanner (utils/llm_scanner.py): - Scans ComfyUI/models/LLMs/ plus any paths registered in
  extra_model_paths.yaml - Detects GGUF format (mmproj-*.gguf alongside main = vision capable) -
  Detects HuggingFace format via config.json architectures / model_type / vision_config /
  preprocessor - Returns capability tags (📷 Vision, 🎬 Video (native/frames), 🔤 Text only) -
  Recommends Qwen2.5-VL 3B Instruct (GGUF) as default download

Client (utils/llm_client.py): - GGUF inference via llama-cpp-python (optional, graceful absent) - HF
  transformers path as secondary (optional) - load_model / unload_model with torch.cuda.empty_cache
  on unload - Task-specific prompt builders: shot action, dialogue, camera, polish

REST endpoints in extension.py: - GET /fbtools/llm/models — scan and return model list + default -
  GET /fbtools/llm/status — current loaded model + backend flags - POST /fbtools/llm/load — load
  model by descriptor - POST /fbtools/llm/unload — free VRAM - POST /fbtools/llm/generate — generic
  text/image generation - POST /fbtools/llm/generate/shot_action, /dialogue, /polish - POST
  /fbtools/llm/download/default — download starter GGUF via huggingface_hub

Editor UI (composition_editor.js): - 🤖 LLM Assistant sidebar section with model picker + capability
  tags - Load / Unload buttons; status line; generate buttons per field - Action / Dialogue / Polish
  buttons target the focused shot card - Download prompt when no models found; mentions
  extra_model_paths.yaml

API client (js/api/llm.js): REST client for all LLM endpoints. Tests (tests/test_llm_scanner.py): 30
  tests, all passing.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **nodes**: Replace audio_reference_file text input with file picker combo
  ([`66b6215`](https://github.com/frost-byte/fbTools/commit/66b62154dd5821dfbf69c934dd26a69dc3632aa9))

SubjectProfileDefine now shows a combo of audio files (.wav, .mp3, .flac, .ogg, .aac, .m4a, .opus)
  from the ComfyUI input directory instead of a free-text field. Press R to refresh the list after
  adding new files.

Also corrects all "Refresh the page" tooltip/doc copy to "Press R" across SubjectProfileLoad,
  SceneTemplateLoad, PromptCompositionLoader, and the four user-facing docs — R triggers
  /object_info which re-runs define_schema and refreshes all combo options without a full page
  reload.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **nodes**: Replace concept_id text input with combo in SubjectProfileDefine
  ([`2502138`](https://github.com/frost-byte/fbTools/commit/2502138a081a2142238ebd76c13f8cb8d0214a48))

Adds _concept_get_ids() helper that reads concept_registry.json at schema load time.
  SubjectProfileDefine.concept_id is now a combo picker instead of a free-text field; "None" is
  normalised to "" in execute(). Define nodes (ConceptDefine, SubjectProfileDefine) keep free-text
  subject_id / concept_id inputs since those are used to create new entries.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **scene**: Add PromptCompositionLoader node with reload counter
  ([`3da0d06`](https://github.com/frost-byte/fbTools/commit/3da0d0645672db623088bbb7a8d2c20193a49df2))

- PromptCompositionLoader: selects a saved composition by name from a combo dropdown, assembles it
  with the chosen model type, and outputs prompt + concept_ids (for ConceptResolve) +
  model_type_used + name - model_type combo includes "composition default" as the first option so
  the stored model type is used without requiring a second setting - fingerprint_inputs includes
  compositions dir mtime + _composition_reload_counter so any PromptCompositionLoader node
  re-executes when content changes - POST /fbtools/compositions/reload increments the counter -
  Editor _onSave fires the reload endpoint (fire-and-forget) so canvas nodes pick up the latest
  content immediately after saving

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Phase 4 shot management — reorder, duplicate, preset targeting, shortcuts
  ([`02779a0`](https://github.com/frost-byte/fbTools/commit/02779a0ac7fef2ec51ff81552be19bc59a05b98e))

- Add ↑/↓ reorder buttons and ⧉ duplicate to each shot card header - Track focused shot (focusin
  delegation) so camera/sound presets insert into the correct shot's field rather than copying to
  clipboard - _moveShot / _duplicateShot / _addNewShot helpers keep focus index in sync and scroll
  the target card into view after rebuild - Ctrl+Shift+N: add shot, Ctrl+Shift+P: preview,
  Ctrl+Shift+C: copy - Update sidebar section titles to "click to apply to shot" -
  .fbt-ce-shot-active highlight (blue border) on the focused shot card

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Prompt Composition Editor — Phase 1 + Phase 2
  ([`7a2bb7d`](https://github.com/frost-byte/fbTools/commit/7a2bb7db7f2e0b90dadf6c530c5408fbc599b1bd))

Phase 1 — Backend data layer: - utils/prompt_compositions.py: composition CRUD,
  resolve_subjects/background, validate - utils/composition_resources.py: backgrounds, camera
  presets, sound presets CRUD - utils/prompt_assembler.py: add assemble_composition() and
  _composition_shots_to_template() - extension.py: subject CRUD routes, composition CRUD + assemble
  route, background CRUD routes, camera + sound preset routes (~305 lines)

Phase 2 — Basic editor panel: - js/api/compositions.js: REST client for compositions, subjects,
  backgrounds, presets - js/ui/composition_editor.js: full sidebar panel — resource sidebar,
  structured form editor (subject slots, shot cards, dialogue), Preview Raw modal, Copy, Save/Load,
  keyboard shortcut (Ctrl+S) - js/styles/style.css: composition editor styles (~480 lines) -
  js/fb_tools.js: register sidebar tab via app.extensionManager.registerSidebarTab - js/index.js:
  re-export CompositionsAPI and renderCompositionEditor

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Prompt Composition Editor — Phase 3 smart elements
  ([`7d24ccb`](https://github.com/frost-byte/fbTools/commit/7d24ccbdefd1fe2bceb0c4f8476a40e3b983a7cb))

- {S} slot-reference completion popup in action/camera text fields: type { to trigger, arrow keys to
  navigate, Enter/Tab to insert, Esc to dismiss - Subject slot cards: appearance summary shown below
  each slot dropdown - Background section: auto-fills soundscape when empty, or offers a replace
  button when the soundscape field already has content - Sidebar "New Subject" inline form: name,
  appearance summary, concept ID; saves via POST /fbtools/subjects/save, refreshes dropdowns -
  Sidebar "New Background" inline form: name, description, lighting, soundscape; saves via POST
  /fbtools/backgrounds/save, refreshes editor background dropdown - compositions.js: add
  saveSubject(), deleteSubject(), saveBackground(), deleteBackground() to CompositionsAPI

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.8.0 (2026-08-07)

### Features

- **ui**: Compact canvas rows for LoraStackBuilder and ConceptDefine
  ([`2b787a8`](https://github.com/frost-byte/fbTools/commit/2b787a8b4fd51f746694967878764a0850f87602))

LoraStackBuilder: - Each slot now fits on a single canvas row: toggle, LoRA name, strength spinners
  (Model+CLIP, or Model+Vid+Aud for LTX2.3), ⓘ icon - Row count is dynamic — starts at 1 (or last
  filled slot) and grows via an "+ Add LoRA" button; count persists in node.properties - Backend
  slot widgets hidden with type="converted-widget" so V3 rendering pipeline skips them - ⓘ opens
  Civitai modal (image gallery with hover-prompt overlay, up to 6 example images) - showCivitaiModal
  exported so ConceptDefine can share it

ConceptDefine: - New compact _CdLoraRow canvas widget: LoRA name + weight spinner on one line,
  optional H/L badge for split models - Split models (wan22, bernini): H row + L row; non-split:
  single row - Switching model_type live rebuilds rows immediately - Widget hiding uses
  type="converted-widget" (V3 requirement; "hidden" is ignored by the V3 onDrawForeground pipeline)
  - Both onNodeCreated and onConfigure rebuild via queueMicrotask so onConfigure.apply can assign
  saved widget values before native widgets are converted, preventing misalignment - Weight
  sanitize: coerces false/non-numeric values to 1.0 on load

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.7.0 (2026-08-06)

### Features

- **scene**: Add PromptAssemble node — Phase 4 of Scene Composition Engine
  ([`6187b54`](https://github.com/frost-byte/fbTools/commit/6187b54053d7fed94d200e15caf1ab12532e1bf8))

Implements model-specific prompt generation from a SCENE_INSTANCE: - utils/prompt_assembler.py: pure
  assembly logic for 8 model types - h3_ref2va: full 6-section H3 brief with Subject/Picture/Audio
  reference labels, first-appearance tracking, <d>[lang] text</d> dialogue tags - h3_fl2va:
  shot-structured format with dialogue, no reference labels - wan22/bernini: production-direction
  block with task classification - ltx23/flux2/krea2/qwen: simple descriptive format -
  PromptAssemble node: takes SCENE_INSTANCE + model_type, outputs prompt (STRING), reference_images
  (IMAGE batch), reference_audio (AUDIO), additional_audio (AUDIO), concept_ids (STRING),
  assembly_report (STRING) - 63 new tests covering all model types, reference numbering, placeholder
  replacement, dialogue tags, outfit overrides, image/audio ordering, concept ID extraction, and
  edge cases

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.6.0 (2026-08-06)

### Features

- **scene**: Add SceneCompose node — Phase 3 of Scene Composition Engine
  ([`8ca5c34`](https://github.com/frost-byte/fbTools/commit/8ca5c34fa2f8e1921c2021aaf9ef3167796a51da))

Adds the scene composition layer: assigns subjects to template slots, maps positional dialogue to
  placeholder shots, applies outfit overrides, and validates slot requirements.

New files: - utils/scene_compose.py — pure composition logic, no ComfyUI deps -
  tests/test_scene_compose.py — 25 tests covering compose, validate, summary

New node (🧊 frost-byte/Scene): - SceneCompose — takes SCENE_TEMPLATE + up to 4 SUBJECT_PROFILEs, up
  to 4 dialogue strings, and per-slot outfit overrides; outputs SCENE_INSTANCE + human-readable
  scene_summary with validation warnings

New custom type: SCENE_INSTANCE (dict with template, slot_assignments, dialogue map,
  outfit_overrides)

Also: SubjectProfileLoad and SubjectProfileDefine now inject subject_id into the SUBJECT_PROFILE
  dict they output, so downstream nodes can reference the profile key without a separate STRING
  output.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.5.0 (2026-08-06)

### Features

- **scene**: Add SceneTemplate nodes — Phase 2 of Scene Composition Engine
  ([`804bc45`](https://github.com/frost-byte/fbTools/commit/804bc4546306eeeb37ff129852b2c09f67b8a853))

Adds the scene template layer: JSON blueprints for shot structure, environment, camera, and slot
  placeholders, independent of model format and subject assignment.

New files: - utils/scene_templates.py — pure SceneTemplate logic, no ComfyUI deps -
  tests/test_scene_templates.py — 40 tests covering load, scan, format, fingerprint -
  scene_templates/monologue_indoor.json — 1-slot bundled example -
  scene_templates/cafe_conversation_2p.json — 2-slot bundled example -
  scene_templates/meeting_room_3p.json — 3-slot bundled example

New nodes (🧊 frost-byte/Scene): - SceneTemplateLoad — loads template from scene_templates/ dir;
  outputs SCENE_TEMPLATE + slot_info - SceneTemplateList — scans directory and returns formatted
  template listing

New REST endpoints: - POST /fbtools/scene_templates/reload — force re-execute fingerprint-cached
  nodes - GET /fbtools/scene_templates/list — return template metadata list as JSON

Bundled examples are seeded into user_data_dir/scene_templates/ on first use when the directory is
  empty; user templates live there permanently.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.4.0 (2026-08-06)

### Bug Fixes

- **lora**: Pass lora_metadata and add lora_convert to apply paths
  ([`84f3dbc`](https://github.com/frost-byte/fbTools/commit/84f3dbc327e32f88215a5c95cf585c9387f1e18e))

- _lora_load_weights now loads with return_metadata=True and caches (mtime, weights, metadata) —
  returns (weights, metadata) tuple - _lora_apply_standard: passes safetensors metadata to
  load_lora_for_models(lora_metadata=...) so downstream nodes can inspect which LoRAs are applied to
  a model patcher - _lora_apply_ltx23: adds missing comfy.lora_convert.convert_lora() call before
  load_lora() to handle BFL/Wan-Fun format variants that the standard path converts automatically
  via load_lora_for_models

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

### Features

- **lora**: Add LoraStackBuilder node and refine LTX2.3 params
  ([`94ebcad`](https://github.com/frost-byte/fbTools/commit/94ebcad22c0af3b847c74c1ad53b88efeea036ea))

- New LoraStackBuilder node: 8 inline LoRA rows (combo + sliders) with model_target selector; JS
  hides video/audio strength widgets for non-LTX2.3 targets; optional autogrow LORA_ENTRY input and
  prev_stack merge; outputs LORA_STACK_DATA without requiring LoraEntryDefine/Collect -
  LoraEntryDefine: replace 5 LTX2.3 per-layer params (video, video_to_audio, audio, audio_to_video,
  other) with 2 (video_strength, audio_strength); backward compat preserved in _lora_apply_ltx23 for
  old entries - _lora_apply_ltx23: handle new 2-param format; video_strength scales all
  video/video-side-cross-attn keys, audio_strength scales all audio keys - _lora_load_weights: add
  mtime-keyed in-memory cache; _lora_apply_standard now uses cached loader instead of direct
  load_torch_file calls - get_node_list: LoraStackBuilder listed first; LoraEntryDefine/Collect
  remain registered for backward compatibility

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **scene**: Add SubjectProfile nodes — Phase 1 of Scene Composition Engine
  ([`4a2daaa`](https://github.com/frost-byte/fbTools/commit/4a2daaa40523c5780c3bff9ae7903a452ba19641))

Introduces the subject profile layer: persistent JSON storage for character appearance, voice, and
  character sheet references, linked to the concept registry via concept_id for LoRA resolution.

New files: - utils/subject_profiles.py — pure SubjectRegistry logic, no ComfyUI deps -
  tests/test_subject_profiles.py — 24 tests covering define, persist, list -
  docs/scene_composition_action_plan.md — full 4-phase system spec

New nodes (🧊 frost-byte/Scene): - SubjectProfileLoad — loads subject dict, IMAGE batch, AUDIO from
  disk - SubjectProfileDefine — creates/updates subjects with auto_save - SubjectProfileList — lists
  all defined subjects

New REST endpoints: - POST /fbtools/subjects/reload — force re-execute fingerprint-cached nodes -
  GET /fbtools/subjects/profiles — return subject_profiles.json as JSON

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.3.0 (2026-08-06)

### Features

- **audio**: Add AudioFixShape node to restore batch dimension on audio waveforms
  ([`1e1083d`](https://github.com/frost-byte/fbTools/commit/1e1083d188c83b12f674faa50661df8e075bdee4))

Handles 1-D (samples,) and 2-D (channels, samples) tensors by unsqueezing to the expected (batch,
  channels, samples) layout. Placed under the new 🧊 frost-byte/Audio category.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.2.0 (2026-08-05)

### Features

- **lora**: Add Concept Registry system with ConceptRegistryLoad, ConceptDefine, ConceptResolve,
  ConceptList nodes
  ([`cc05197`](https://github.com/frost-byte/fbTools/commit/cc05197c2d6c748d3f947bb967980c219ab123b0))

- Add utils/concept_registry.py: pure-logic module (no ComfyUI deps) with ConceptRegistry class,
  MODEL_PROFILES for 6 model types (wan22/bernini split, ltx23/flux2/krea2/qwen single), load/save
  with .bak backup, resolve_concepts, assemble_prompt, build_model_entry helpers - Add 4 ComfyUI
  nodes: ConceptRegistryLoad (fingerprint-based reload via REST), ConceptDefine (chainable,
  accumulate-not-overwrite for different model_types, auto_save option), ConceptResolve (applies
  LoRAs via comfy.sd, assembles prompt with trigger words), ConceptList (filter by model type) - Add
  CONCEPT_REGISTRY custom wire type - Add REST endpoints: POST /fbtools/concepts/reload (reload
  counter), GET /fbtools/concepts/registry - Add user_data_dir() + _user_subdir() helpers; update
  default_scenes_dir() and default_libber_dir() to prefer ComfyUI/user/default/comfyui-fbTools/ with
  graceful fallback to legacy output/ directories - Extract setWidgetVisible to js/utils/widgets.js
  (shared); update lora.js to import it; add js/api/concepts.js, js/nodes/concepts.js
  (lora_low/weight_low hidden for single-model types; Reload Registry button on ConceptRegistryLoad)
  - Add 32 tests in tests/test_concept_registry.py; all 330 Python + 83 JS tests pass

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **lora**: Add minimax_h3 to ConceptRegistry MODEL_PROFILES
  ([`fb23e5a`](https://github.com/frost-byte/fbTools/commit/fb23e5ab21ed5f66a3ba36646fa001d5bf163077))

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **lora**: Add native LORA_STACK support to LoraPresetDefine/Select and add MiniMaxH3 target
  ([`b9468d8`](https://github.com/frost-byte/fbTools/commit/b9468d8f2efa894768e9637f80a64510976ef862))

LoraPresetDefine now accepts both LORA_STACK_DATA (from LoraStackCollect's Stack Data output) and a
  native LORA_STACK (easy-use tuple format) as optional inputs, so any LoRA source in the ecosystem
  can be stored in a preset. When LORA_STACK_DATA is provided, the native representation is
  auto-generated so both output types are always populated.

LoraPresetSelect gains a new "LoRA Stack (Native)" output (io.Custom LORA_STACK) appended after the
  existing outputs, preserving backward compatibility for already-wired workflows. The
  LORA_STACK_DATA output is unchanged.

Also adds MiniMaxH3 to LORA_MODEL_TARGETS for use in LoraStackApply (standard
  strength_model/strength_clip path); weight variants can be added later once the LoRA structure is
  known.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **lora**: Add scene/pose image support to preset nodes
  ([`0103882`](https://github.com/frost-byte/fbTools/commit/01038824c1d8f4ce5fda90a3f4ae57517588dc91))

LoraPresetDefine and WanPresetDefine each gain an optional Scene combo (populated at runtime via
  /fbtools/scene/list) and a Pose Image Type combo. The selected scene and pose type are stored in
  the preset dict.

LoraPresetSelect and WanPresetSelect each gain base_image and pose_image outputs. When the active
  preset has a linked scene, those images are loaded from the scene directory and shown as a node
  preview on execution. Placeholder 64x64 images are returned when no scene is set.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.1.0 (2026-07-27)

### Features

- **lora**: Accordion-hide LTX2.3 layer weights in LoraEntryDefine
  ([`435053e`](https://github.com/frost-byte/fbTools/commit/435053e81fb653edf7237203f3c80a1cd38f1ae9))

When model_target is not LTX2.3, the video/audio/cross-attention strength inputs and toggle button
  are hidden entirely. When LTX2.3 is selected, a ▶/▼ caret button between Enabled and the Civitai
  button controls visibility. Accordion defaults to collapsed; onConfigure re-applies visibility
  when a saved graph is loaded.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **lora**: Add dynamic combo and preview to WanPresetSelect
  ([`2dd5e5b`](https://github.com/frost-byte/fbTools/commit/2dd5e5bd20f5b668f926242a5f477a07d6fdce69))

- Replace index INT input with a COMBO widget (selected_preset) that starts with ["none"] and is
  populated with preset names after each execution - Add validate_inputs to accept any string value,
  bypassing static combo option validation so user-selected names are not rejected by the server -
  Add is_output_node=True for standalone preview execution - Execute sends preset names to frontend
  via ui={preset_names:[...]}; JS onExecuted updates widget.options.values and preserves current
  selection - Switch preset lookup from index-based to name-based with first-entry fallback

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

- **lora**: Add LoraPresetDefine and LoraPresetSelect nodes
  ([`be84768`](https://github.com/frost-byte/fbTools/commit/be847688d1922a17a0741870950f3714e8111baf))

Single-stack preset nodes for models without a dual-sampler stage (e.g. Flux2/Klein, Qwen). Uses a
  separate LORA_PRESET_LIST custom type to prevent cross-wiring with Wan preset chains.
  LoraPresetSelect uses the same dynamic combo + validate_inputs pattern as WanPresetSelect.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.0.0 (2026-07-24)

### Bug Fixes

- Add control flags to StorySceneBatch descriptor and fix StoryEdit scene data persistence
  ([`4045691`](https://github.com/frost-byte/fbTools/commit/4045691ea7a538253c16cbf4adf328feb53c49c3))

- Add use_depth, use_mask, use_pose, use_canny flags to batch descriptor - Add logging to
  StorySceneBatch for scene configuration debugging - Add logging to StoryScenePick for pose_type
  resolution tracking - Fix StoryEdit duplicate renderTable call that was overwriting normalized
  scene data - Add fingerprint_inputs to StorySceneBatch for proper cache invalidation

This fixes issues where: 1. Control flags weren't being passed from story config to batch processing
  2. StoryEdit UI changes (pose_type, depth_type) weren't persisting to story.json 3. Cache wasn't
  invalidating when story.json was modified externally

- Complete mask system migration from mask_type to mask_name
  ([`2da731d`](https://github.com/frost-byte/fbTools/commit/2da731d1cf5002293d68cfd2f041d8c99df1714f))

BREAKING CHANGES: - Story persistence now uses mask_name instead of mask_type - Backward
  compatibility maintained for loading old stories

Backend Changes: - story_models.py: Updated save_story() to persist mask_name field (line 142) -
  extension.py: Fixed StorySceneBatch to use mask_name in scene descriptors (lines 5135, 5162) -
  extension.py: Added backward-compatible fallback to mask_type for legacy data - extension.py: All
  mask loading/preview functions now use mask_name consistently

Frontend Changes: - js/nodes/story.js: StoryEdit mask column now uses dropdown instead of text input
  - js/nodes/story.js: Dropdowns populated from scene's available_masks array - js/nodes/story.js:
  Updated prompt_key rendering for conditional dropdown/textarea - js/nodes/story.js: Extended
  populateVideoPromptControls to handle both image and video prompts - js/nodes/story.js: Added
  event listeners for mask-name-select and prompt-key-select

API Changes: - /fbtools/story/load: Returns available_masks array per scene for dropdown population
  - /fbtools/story/save: Accepts mask_name with fallback to mask_type

Migration: - Old stories with mask_type are automatically migrated to mask_name on load -
  SceneInStory.__init__ converts mask_type to mask_name during initialization - All file I/O now
  uses v2 format with mask_name as primary field

Testing: - Verified backward compatibility with v1 story.json files - Verified mask dropdown
  population from masks.json and legacy PNGs - Verified batch system
  (StorySceneBatch/StoryScenePick) uses correct mask field

Closes: Mask persistence bug, prompt_key dropdown regression, batch system migration gap

- Dynamically fetch available libbers when switching to libber type
  ([`a7a3eb5`](https://github.com/frost-byte/fbTools/commit/a7a3eb5de1b8f3dd2e5d3570c56d1098bcbc9b35))

- Import libberAPI in scene.js - When user selects 'libber' type and current value is 'none': *
  Fetch latest libbers from API endpoint * Repopulate dropdown with current libbers * Auto-select
  first available libber * Fallback to existing list if API fails - Prevents showing 'none' when
  libbers exist - Ensures dropdown always shows current state

No service restart needed - just refresh browser (Ctrl+Shift+R)

- Libber nodes now reload from file to prevent stale cache
  ([`643b2bb`](https://github.com/frost-byte/fbTools/commit/643b2bb478109a911b89b7fb5282ac87267462c3))

LibberManager and LibberApply nodes were using in-memory Libber instances that weren't being updated
  when changes were saved via the REST API/web UI.

Changes: - LibberManager: Now reloads from JSON file if it exists on each execution - LibberApply:
  Also reloads from file before applying substitutions - Ensures nodes always use the latest lib
  values from disk - In-memory cache is effectively refreshed on every node execution

This fixes the issue where updating keys in LibberManager wouldn't reflect in LibberApply results
  until server restart.

- Update canny during SceneUpdate
  ([`d43c1f5`](https://github.com/frost-byte/fbTools/commit/d43c1f5107934ac2ff5e79b97b75875d8f443bae))

- **LibberApply**: Improve table display and resize behavior
  ([`692116c`](https://github.com/frost-byte/fbTools/commit/692116c84d5563e764b5e32bd1afe74a06336488))

- Replaced JSONView formatter with clean HTML table layout - Added two-column table format with 🗝️
  Lib and 🪙 Value headers - Implemented scrollable container with overflow-y and overflow-x - Fixed
  table persistence after node execution by storing and reusing updateDisplay function - Added
  dynamic sizing with proper height calculation based on available node space - Implemented resize
  hooks (onResize) to update container height when node is resized - Added height constraints (min:
  150px, max: 600px) to prevent infinite growth - Fixed bottom edge overlap by adding 15px bottom
  margin - Improved widget height computation to account for previous widgets' space - Added HTML
  escaping for safe display of lib values

The table now properly displays libber key-value pairs, persists after execution, and maintains
  reasonable sizing constraints while allowing user resizing.

- **ScenePromptManager**: Fix scene selection, saving, and libber integration
  ([`431be57`](https://github.com/frost-byte/fbTools/commit/431be57a2fb3398afe1b3511ce7a7bad756d1631))

- Fix scene tracking to read widget values at click time instead of cached values - Scene dropdown
  now correctly reloads prompts when changed - Apply Changes now saves to the correct scene
  directory (was using first available scene) - Fix prompt data structure handling (API returns
  array, code expected object) - Add scene save API endpoint POST /fbtools/scene/save_scene_prompts
  - Fix libber_name preservation and libber dropdown population - API now returns libber_name field
  and available libbers list - Add 100ms delay for initial widget value loading to ensure proper
  initialization - Add extensive debug logging for scene tracking and widget values

Resolves issues where: - Changing scene dropdown didn't update the UI - Apply Changes saved to wrong
  scene directory - Libber selections weren't preserved - Prompt keys showed as array indices
  instead of names

### Chores

- Add Conventional Commits hook, semantic release, and CLAUDE.md
  ([`d73123a`](https://github.com/frost-byte/fbTools/commit/d73123a83c720a3b004a52b52596c49d85109dc7))

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

- Add developer utility scripts
  ([`eef9dda`](https://github.com/frost-byte/fbTools/commit/eef9ddaa0ef2a926aa72235bf93cf152f11080ae))

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

### Documentation

- Add comprehensive test coverage summary
  ([`dce7115`](https://github.com/frost-byte/fbTools/commit/dce7115fafca3843c519f56d6eb3e8e966de7d69))

- 90 tests passing (100% pass rate) - Coverage breakdown by test file and category - Real-world
  workflow validation - Backend testing complete - Ready for UI implementation

- Add NLF pose implementation guide and additional tests
  ([`20dd41a`](https://github.com/frost-byte/fbTools/commit/20dd41aa3b8d803087ed9d0ffe3ae61381a2105d))

Complete documentation and test coverage for NLF pose feature:

- NLF_POSE_IMPLEMENTATION.md: Comprehensive implementation guide with: - Step-by-step checklist -
  Format specifications (DWPose, OpenPose, NLFPRED, POSE_KEYPOINT) - Three workflow examples
  (generation, editing, regeneration) - Model requirements and downloads - Configuration options
  reference

- utils/nlf_pose.py: NLF utilities module (476 lines) - load_nlf_model, predict_nlf_pose,
  render_nlf_pose - Format conversion functions - Supports both relative and absolute imports for
  testing

- tests/test_nlf_integration.py: 13 integration tests - Module import validation -
  SceneCreate/SceneUpdate input verification - SceneInfo data model checks - Pose JSON format
  validation - Workflow structure verification

- js-tests/story_nlf_pose.test.js: 11 frontend tests (all passing) - Pose type dropdown includes
  'nlf' - Backend/frontend consistency - Scene serialization with NLF pose - Backward compatibility

Ready for testing in ComfyUI.

- Add Scene Prompt Management System implementation plan
  ([`f6f8e1c`](https://github.com/frost-byte/fbTools/commit/f6f8e1ce95aafe17f303ff2ffde4e2027d933fbd))

- Clarify two workflow approaches - Complete vs Atomic prompts
  ([`33b4311`](https://github.com/frost-byte/fbTools/commit/33b431174fc1009f1b4844a642761132917ae5f7))

- Added 'Two Workflow Approaches' section explaining both strategies - Approach 1: Complete Prompts
  (traditional, recommended for most users) * Each prompt is self-contained and complete * Libber
  handles dynamic parts within single prompt * Example: wan_high with full prompt text - Approach 2:
  Atomic Composition (advanced) * Break into small reusable pieces * Maximum flexibility for
  mixing/matching - Comparison table showing pros/cons of each - Hybrid approach combining both
  strategies - Real-world usage patterns with examples - Migration strategy from legacy prompts -
  Complete examples for both approaches

- Comprehensive documentation and test coverage update
  ([`4426092`](https://github.com/frost-byte/fbTools/commit/44260929ea41429bc2e3ccc7cba0e6709ba3ce80))

Major documentation improvements:

Root README.md: - Complete feature overview with all node categories - Detailed usage examples for
  Libber, Story, and PromptCollection - Development setup and testing instructions - Project
  structure and architecture explanation - Changelog with recent Libber overhaul details

LIBBER_NODES_README.md (NEW): - Complete Libber system documentation - Interactive table editor
  features and workflow - Click-to-insert functionality guide - REST API endpoint reference - Use
  cases and best practices - Troubleshooting guide - JavaScript integration examples

TEST_RESULTS.md: - Updated with Libber test coverage (30 tests) - Summary of all Python tests (70+
  tests total) - Summary of all JavaScript tests (30+ tests) - Execution instructions for both test
  suites

New Tests: - tests/test_libber.py: 30 comprehensive unit tests * Basic operations (create, add,
  remove, list) * Substitution with recursion and depth limiting * Custom delimiters * File
  operations (save/load) * Edge cases (unicode, large values, special chars) * Integration workflows
  - js-tests/libber_api.test.js: 21+ API client tests * CRUD operations * Error handling *
  Integration workflows

All tests passing: - Python: 30/30 Libber tests ✓ - Python: 32/32 PromptCollection tests ✓ -
  JavaScript: API client tests ready

This commit provides complete documentation for users and developers, with comprehensive test
  coverage ensuring reliability.

- Create comprehensive plan for flexible prompt system and workflow improvements
  ([`23f5048`](https://github.com/frost-byte/fbTools/commit/23f5048def5ed131628c2c525337c18019fe94f1))

Add detailed implementation plan (plan-flexibleMultiPromptSystemLibberBugFix.prompt.md) covering:

- PromptCollection data model with v2 format and non-destructive migration - SceneInfo refactoring
  with backward-compatible @property methods - Scene REST API for lightweight metadata operations -
  Dynamic prompt name discovery and selectors - PromptCollectionEdit node with REST backend - Story
  execution-based output organization for two-stage workflows - StoryExecutionInit for execution
  context management - StoryImageNamer/StoryPathResolver for standardized naming -
  StoryImageCollector/StoryVideoNamer for video generation pipeline - Multiple path format outputs
  (abs/rel, with/without extension) - Libber REST API with LibberStateManager for server-side state
  - LibberEdit UI refactoring to fix synchronization bugs

Plan prioritizes Scene/PromptCollection improvements (Steps 1-5), Story workflow enhancements (Step
  6), then Libber bug fixes (Steps 7-8).

Key features: - Non-destructive migration with v1_backup preservation - Execution-aware directory
  structure for multi-run workflows - Support for image generation → video generation pipeline -
  Flexible path outputs for different SaveImage node conventions - Backward compatibility maintained
  throughout

Refs: LibberEdit add operation bug, Story output organization requirements

### Features

- Add compositions support to PromptCollection
  ([`f9e3493`](https://github.com/frost-byte/fbTools/commit/f9e3493c816b2a8d5e3c827ed405e6d24a781fc9))

Backend changes: - Add compositions field to PromptCollection: {output_name: [prompt_keys]} - Add
  composition CRUD methods: add_composition, remove_composition, list_composition_names - Update
  to_dict/from_dict to serialize/deserialize compositions - Update ScenePromptManager to output
  prompt_dict (composed prompts) - Compose prompts automatically when compositions exist - Include
  compositions_list and prompt_dict in UI data

Data structure: - compositions saved in prompts.json alongside prompts - Backward compatible (empty
  dict if no compositions) - compose_prompts() handles libber substitution

ScenePromptManager outputs: - scene_info (updated with prompts + compositions) - prompt_dict
  (Dict[str, str] - composed outputs) - status

Ready for frontend tab implementation

- Add comprehensive testing for PromptCollection with maintainable architecture
  ([`19b5cdc`](https://github.com/frost-byte/fbTools/commit/19b5cdc4b9f0cdd5f972dd792ff129ba1298a882))

Extract data models to standalone module and implement full test coverage for v1→v2 prompt migration
  system.

Changes: - Create prompt_models.py: Pure data models with no ComfyUI dependencies * PromptMetadata:
  Single prompt with metadata fields * PromptCollection: V2 multi-prompt system with migration
  support

- Refactor extension.py: Import from prompt_models instead of inline definitions * Reduces
  extension.py by ~130 lines * Enables independent testing of data models

- Add comprehensive test suite (tests/test_prompt_collection.py): * 32 tests across 8 test classes *
  V1→V2 migration with v1_backup preservation * CRUD operations (add, remove, get, list) *
  Serialization/deserialization roundtrips * Backward compatibility validation * Edge cases
  (unicode, large values, 1000+ prompts) * File I/O operations * Integration workflows * All tests
  passing in 0.19 seconds

- Update test infrastructure: * conftest.py: Mock setup for ComfyUI dependencies * pytest.ini: Clean
  configuration

- Documentation: * TEST_RESULTS.md: Detailed test coverage report * TESTING_STRATEGY.md:
  Architecture decisions and benefits

Benefits: ✓ Single source of truth - no code duplication ✓ Fast, isolated tests - no complex mocking
  needed ✓ Maintainable - updates reflect everywhere automatically ✓ Validates v1→v2 migration
  preserves original data ✓ Ensures backward compatibility

- Add generic mask system and NLF pose generation
  ([`0e65f19`](https://github.com/frost-byte/fbTools/commit/0e65f1934a701ba8e456f47908e956477f8fba33))

Major Features:

1. Generic Mask System (replaces hardcoded masks) - MaskDefinition dataclass with MaskType enum
  (TRANSPARENT/COLOR) - User-definable masks via masks.json (v1 format) - SceneSelect: Dynamic
  mask_name combo loaded from masks.json - SceneInfo: masks dict + mask_images dict (name-keyed) -
  Migration support for legacy 'girl'/'male'/'combined' masks - Tests: test_mask_integration.py (8
  tests), mask_system.test.js (frontend)

2. NLF Pose Generation - utils/nlf_pose.py: Neural Lifting Framework integration - SceneCreate: 7
  NLF inputs for pose generation - SceneUpdate: 9 NLF inputs for pose editing/regeneration -
  SceneInfo: pose_nlf_image field with load/save - default_pose_options: 'nlf' -> 'pose_nlf_image'
  mapping - Story node: 'nlf' added to pose type dropdown - Tests: test_nlf_integration.py (13
  tests), story_nlf_pose.test.js (11 tests)

3. Documentation Reorganization - Moved 23 docs to docs/ folder - Test docs to docs/testing/
  subfolder - Updated README with logo, dependencies, testing links - New docs: MASK_SYSTEM.md,
  PHASE_4_COMPLETE.md

Changes by file: - extension.py: Mask system classes + NLF pose in SceneCreate/SceneUpdate -
  js/nodes/scene.js: Dynamic mask combo via API - js/nodes/story.js: 'nlf' pose type in dropdown -
  story_models.py: mask_name field in StoryScene - dependency.json: Fixed comfyui_controlnet_aux URL
  - tests/conftest.py: Added torch mocking for NLF tests

All tests passing: 213 Python tests, 11 JavaScript tests

- Add modular frontend architecture with API clients and testing framework
  ([`7695ac3`](https://github.com/frost-byte/fbTools/commit/7695ac36714cc8e0bb6b11e6a53e5b9fee4685e0))

Create comprehensive modular JavaScript architecture for fbTools frontend with testable API clients,
  shared utilities, and full Jest testing setup.

New Structure: - js/api/ API client modules for REST endpoints - js/utils/ Shared utility functions
  - js/tests/ Test framework with utilities - js/index.js Main exports file

API Clients Added: - prompt_collection.js: PromptCollection REST API (fully implemented) *
  createSession, addPrompt, removePrompt, listPromptNames, getCollection - scene.js: Scene metadata
  operations (stub ready for backend) - libber.js: Libber placeholder management (stub ready for
  backend) - story.js: Story-level operations (stub ready for backend)

Utilities Added: - api_base.js: BaseAPI class with error handling and fetch wrapper * POST/GET
  methods with automatic error handling * Toast notification helpers (showSuccess, handleError) *
  APIError class for typed error responses - widgets.js: ComfyUI widget update helpers *
  updateWidgetFromText: Update single widget from API response * updateNodeWidgets: Bulk widget
  updates * scheduleNodeRefresh: Node resize/refresh utility

Testing Framework: - test_utils.js: Testing utilities and mocks * mockFetch: Fetch API mocking for
  isolated tests * createMockFn: ES module-compatible mock functions * createMockApp/createMockNode:
  ComfyUI test fixtures * expectToast helpers: Toast assertion utilities -
  prompt_collection_api.test.js: Example tests (9 tests, all passing) - package.json: Jest
  configuration with ES module support - Fixed jest-environment-jsdom dependency for Jest 29 -
  Custom createMockFn() to replace jest.fn() in ES modules

Documentation: - README.md: Architecture overview and usage guide - INTEGRATION_GUIDE.md: Complete
  integration examples - QUICK_REFERENCE.md: Copy-paste code snippets - MODULAR_ARCHITECTURE.md:
  What we built and why - TESTING_SETUP.md: How to run tests and troubleshoot

Benefits: ✓ Testable API clients isolated from ComfyUI dependencies ✓ Centralized error handling
  with automatic user feedback ✓ Reusable utilities across all nodes ✓ Full test coverage capability
  (9 passing tests) ✓ Progressive enhancement - works alongside existing code ✓ Easy to extend with
  new API endpoints

Migration Path: - No breaking changes to existing fb_tools.js - Import and use API clients as needed
  - Gradually refactor nodes to use new architecture - Remove old fetch calls once migrated

Test Results: Test Suites: 1 passed, 1 total Tests: 9 passed, 9 total

Time: 0.526 s

Usage Example: import { promptCollectionAPI } from "./api/prompt_collection.js";

const session = await promptCollectionAPI.createSession(); const result = await
  promptCollectionAPI.addPrompt( session.session_id, "girl_pos", "beautiful woman smiling" );

- Add ScenePromptManager and PromptComposer nodes
  ([`05e9e60`](https://github.com/frost-byte/fbTools/commit/05e9e600769ffc2f13d14aa9e505dddaed601b94))

Implements dictionary-based prompt composition system:

ScenePromptManager: - CRUD operations for scene prompts - Interactive table UI (will add JS in next
  commit) - Manages PromptCollection within SceneInfo - Processing type configuration (raw/libber)

PromptComposer: - Composes multiple output prompts from collection - Flexible output naming (no
  hardcoded prompt_a/b/c) - Returns PROMPT_DICT with user-defined keys - Automatic libber
  substitution during composition - Saves/loads composition maps as JSON

PromptCollection.compose_prompts(): - New method for dynamic composition - Takes composition map:
  {output_name: [prompt_keys]} - Processes libber substitutions inline - Returns dict of composed
  prompt strings

Benefits: - Infinitely extensible outputs (no fixed limit) - Self-documenting (key names describe
  purpose) - Same prompts, different compositions per workflow - Single DICT output type simplifies
  maintenance

- Add ScenePromptManager interactive table UI
  ([`fd5d315`](https://github.com/frost-byte/fbTools/commit/fd5d315f050c04084bec30d891b41b04a8d67804))

- Created setupScenePromptManager() in js/nodes/scene.js - Interactive table similar to
  LibberManager - Columns: Key | Value | Type (raw/libber dropdown) | Libber Name | Category |
  Actions - Add/Remove prompts with visual feedback - Apply button to update collection_json -
  Auto-updates from backend on execution - Type dropdown enables/disables libber name input -
  Registered in fb_tools.js extension system - Toast notifications for user actions

- Add StorySceneBatch job_id input, scene list API, and UI improvements
  ([`325241a`](https://github.com/frost-byte/fbTools/commit/325241a87fb9b9be96592ed0c855068e1b2a6c65))

- Add optional job_id input to StorySceneBatch node for reusable job directories - Add
  /fbtools/scene/list REST API endpoint for available scenes - Improve StoryEdit UI: add scene
  dropdown on new scenes, auto-load scenes - Add stylesheet loading in fb_tools.js init hook -
  Create style.css for prompt textarea styling - Update story.js API client with listScenes method -
  Filter internal flags from story save operations

- Add StoryVideoSave node for video batch workflow
  ([`fc0c215`](https://github.com/frost-byte/fbTools/commit/fc0c2153cd7c3ac557d399f5a374edb16e1f4a52))

- Implement StoryVideoSave node to complete video generation workflow - Takes video output from
  generation nodes + VIDEO_BATCH - Saves to correct path from video descriptor - Automatic directory
  creation - Pass-through video output for chaining - Outputs filename, filepath, scene info

- Node features: - Matches StorySceneImageSave pattern for consistency - Supports string path videos
  (file copy) - Extensible for other video formats - Preview UI shows saved location and scene
  details

- Update STORY_VIDEO_README.md: - Add StoryVideoSave node documentation - Complete workflow examples
  with save step - Show full iteration pattern

Complete video workflow is now: StoryLoad → StoryVideoBatch → [Iterate] → Generate Video →
  StoryVideoSave

This completes the video generation system, providing full parity with the image generation workflow
  (StorySceneBatch → Generate → StorySceneImageSave)

- Add subject compositor utility and tests
  ([`94834a9`](https://github.com/frost-byte/fbTools/commit/94834a90cca4561145fd5ae26ab7250367119dd1))

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

- Add video generation workflow support for story scenes
  ([`9bd33cf`](https://github.com/frost-byte/fbTools/commit/9bd33cf996f98827c8713cd7d9c1640a9eb09ed4))

- Add video prompt fields to SceneInStory model - video_prompt_source:
  'auto'|'prompt'|'composition'|'custom' - video_prompt_key: key for prompt/composition lookup -
  video_custom_prompt: custom video generation prompt

- Create utils/story_video.py with testable video utilities - list_job_ids(): List available jobs
  sorted by modification time - find_scene_image(): Locate scene images by order and name -
  pair_consecutive_scenes(): Create scene transition pairs - generate_video_filename(): Generate
  standardized video filenames - resolve_video_prompt(): Resolve video prompts from scene config -
  build_video_descriptor(): Build complete video generation descriptor

- Implement StoryVideoBatch node - Lists available job IDs from story directory - Iterates through
  scene pairs for video transitions - Outputs VIDEO_BATCH with first/last frame paths, prompts, LoRa
  data - Supports video_prompt_source modes: auto, prompt, composition, custom - Generates
  standardized video filenames (001_to_002_opening_to_battle.mp4)

- Add comprehensive test coverage - 29 new unit tests in tests/test_story_video.py - Tests job
  listing, image finding, scene pairing, prompt resolution - All 150 tests passing (121 existing +
  29 new)

- Create STORY_VIDEO_README.md documentation - Complete workflow guide for video generation - Node
  usage and configuration examples - Video descriptor format specification - Directory structure and
  naming conventions - Integration patterns with video generation nodes

Video generation workflow enables: 1. Load story with StoryLoad 2. Select job ID with
  StoryVideoBatch (lists available jobs) 3. Iterate through video descriptors 4. Generate videos
  between consecutive scenes 5. Use LoRa data and video prompts for consistent style 6. Save to
  job_output_dir with standardized naming

This extends the story building system from image generation to complete video generation workflows,
  maintaining consistency with existing patterns and full test coverage.

- Add websocket image save
  ([`fa764fa`](https://github.com/frost-byte/fbTools/commit/fa764fadb8a21c1be31bdae41993c82b01cc1bcc))

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

- Complete LibberManager and LibberApply UX overhaul with modular architecture
  ([`2224753`](https://github.com/frost-byte/fbTools/commit/2224753eb016ce074ae3fcd5a6338150cf826599))

Major improvements to the Libber system with enhanced user experience:

LibberManager: - Replaced dropdown-based operations with interactive editable table - Inline editing
  with textarea inputs for uniform cell heights (38px) - Per-row action buttons (✏️ Update, ➖
  Remove) matching cell height - Sticky action bar with 📂 Load, 💾 Save, and ➕ Create buttons -
  Inline libber creation with text input field and create button - Auto-save after add/update/remove
  operations - Simplified schema: single libber_name combo (basenames only, no .json extension) -
  Smart auto-loading: checks memory → file → creates new libber

LibberApply: - Click-to-insert functionality with delimiter wrapping - Cursor position tracking
  across focus changes - Native browser undo/redo support using execCommand - Always-visible 🔄
  Refresh button (sticky at top) - Smart libber discovery: scans memory and disk files - Empty state
  messaging with helpful hints - Dynamic table sizing responding to node resize

Code Architecture: - Modularized into separate node modules: * js/nodes/libber.js - LibberManager
  and LibberApply * js/nodes/scene.js - SceneSelect extensions * js/nodes/story.js - StoryEdit and
  StoryView extensions - Main fb_tools.js reduced from ~1400 to ~450 lines - Clean import structure
  with node-type routing

Technical Improvements: - LiteGraph NODE_TITLE_HEIGHT and NODE_WIDGET_HEIGHT for proper sizing - CSS
  variables for theming (--comfy-input-bg, --border-color, --fg-color) - Sticky positioning
  (position: sticky, top: 0, z-index: 10) - Button styling with min-height and flexbox centering -
  Responsive table layout with proper overflow handling

Breaking Changes: - LibberManager schema simplified (removed
  operation/key_selector/lib_key/lib_value widgets) - libber_name and filename merged into single
  libber_name Combo (basenames only) - Execute method auto-creates libber if not exists, skips if
  "none" selected

This commit represents a complete UX transformation from tedious dropdown operations to a modern,
  interactive table-based workflow with significantly improved usability.

- Dynamic job_id dropdown updates when story_name changes in StorySceneBatch
  ([`d455b79`](https://github.com/frost-byte/fbTools/commit/d455b79f33c0a0739542c5ab0fc54e054d670d04))

- Frontend: Added callback to story_name widget to fetch and update job_id options via
  /fbtools/story/job_ids API - Frontend: job_id dropdown now auto-populates on node creation for
  default story - Frontend: job_id options refresh automatically when user changes story selection -
  Backend: Simplified job_id schema to start with empty option only (frontend handles population) -
  Backend: Updated tooltip to clarify dynamic behavior - Improves UX by eliminating need to execute
  node just to update job_id list

- Enhance LibberManager and LibberApply nodes with improved UX
  ([`9595c30`](https://github.com/frost-byte/fbTools/commit/9595c3068cca3acd19dfeaf68351a0cbab527f37))

Backend changes: - Refactored Libber nodes into unified LibberManager node - Fixed get_libber_data
  method to use libber.libs instead of libber.lib_dict - Consolidated LibberCreate, LibberLoad, and
  LibberSave into single manager interface - Added operations: create, load, add_lib, remove_lib,
  save - Implemented LibberStateManager for persistent state management - Added REST API endpoints
  for Libber operations

Frontend changes (LibberManager): - Fixed ComboWidget rendering by using widget.options.values
  pattern - Added auto-save after add_lib and remove_lib operations - Implemented auto-clear of
  lib_key field after successful operations - Added auto-select of newly added key or first
  available after remove - Implemented auto-load of libber data on node creation/page refresh -
  Added key normalization (lowercase, replace spaces/hyphens with underscores)

Frontend changes (LibberApply): - Replaced JSONView formatter with clean HTML table display - Added
  scrollable container with max-height: 250px - Implemented two-column table layout (Key | Value) -
  Added theme-aware styling using CSS variables - Improved dynamic node sizing to fit content -
  Added HTML escaping for safe value display

Testing infrastructure: - Restructured test files from js/tests/ to js-tests/ - Updated package.json
  with Jest configuration - Moved test utilities and test files to new structure

This update significantly improves the Libber workflow by consolidating operations into a single
  manager node, adding automatic persistence, and providing a clean table view for reviewing lib
  definitions.

- Implement PromptCollection v2 system with REST API (Steps 1-2)
  ([`d4a735f`](https://github.com/frost-byte/fbTools/commit/d4a735ff5191b71b25d22bf7e3cd72b2cefea975))

Add flexible multi-prompt system with non-destructive migration:

- PromptCollection data model with PromptMetadata * Supports unlimited named prompts with
  categories/tags * V2 format with v1_backup for rollback capability * Auto-migration from legacy v1
  format

- REST API infrastructure for prompt management * PromptCollectionStateManager with 30min TTL * POST
  /fbtools/prompts/create, add, remove * GET /fbtools/prompts/list_names * Server-side session-based
  state management

- SceneInfo backward compatibility * Added prompts: Optional[PromptCollection] field * Legacy fields
  (girl_pos, male_pos, etc.) still work * save_prompts() auto-migrates to v2 on save *
  load_prompt_json() detects format and auto-migrates

- Non-destructive migration strategy * All v1 data preserved in v1_backup field * Existing code
  continues to work unchanged * Transparent auto-migration on file operations

Refs: plan-flexibleMultiPromptSystemLibberBugFix.prompt.md Steps 1-2

- Implement video prompt configuration with model extraction
  ([`32d0378`](https://github.com/frost-byte/fbTools/commit/32d0378572ecd35a74c1608e3f09d3c82dba8e5e))

Core Changes: - Extract SceneInStory and StoryInfo models to story_models.py * Enables isolated
  testing without ComfyUI dependencies * Follows prompt_models.py architecture pattern * Reduces
  extension.py by ~160 lines

- Fix load_story() to deserialize video prompt fields from JSON * Added video_prompt_source,
  video_prompt_key, video_custom_prompt to load logic * Fields were being saved but not loaded,
  causing defaults on reload * Now properly restores saved video prompt configuration

Frontend (js/nodes/story.js): - Dynamic video prompt UI in StoryEdit Advanced Flags tab *
  Source-based input types: dropdown for prompt/composition, textarea for custom * Auto-populated
  dropdowns with available prompt/composition keys * Live preview textarea showing resolved prompt
  text * Proper event handling for all video prompt controls

Backend (extension.py): - Updated load_story() V2 format parsing to include video fields - API
  endpoints already had video field support via getattr() defaults - All save/load cycles now fully
  support video prompt persistence

Testing: - 6 comprehensive video prompt persistence tests - Tests validate: data structures,
  serialization, deserialization, roundtrip - Full test suite: 156 tests passing (150 existing + 6
  new) - Story models now testable in isolation

Documentation: - VIDEO_PROMPT_UI_LAYOUT.md: Visual reference for UI layout and interactions -
  VIDEO_PROMPT_UX_IMPLEMENTATION.md: Technical implementation details and data flow

Fixes: - Video prompt fields now persist correctly through save/load cycles - Browser reload
  properly restores video prompt configuration - Preview textarea updates dynamically based on
  source and selection

Architecture: - Improved code organization with model extraction - Better separation of concerns
  (data models vs business logic) - Easier testing and maintenance going forward

- Integrate scene_flags into PromptCollection and add overlay feedback utility
  ([`753c1e0`](https://github.com/frost-byte/fbTools/commit/753c1e08c1aa3eea5d000f90f64f27b6f22ec03e))

## Backend Changes - **PromptCollection Model (prompt_models.py)**: - Added scene_flags as
  Optional[dict] field to store per-scene control flags (use_depth, use_mask, use_pose, use_canny) -
  Updated to_dict() to include scene_flags when not None - Updated from_dict() to load scene_flags
  from incoming data - Maintains backward compatibility (scene_flags is optional)

- **Scene Prompts API (extension.py)**: - scene_get_prompts: Now returns scene_flags in response -
  scene_save_prompts: Simplified to use model serialization (scene_flags preserved automatically)

## Frontend Changes - **Reusable Overlay Utility (js/utils/feedback.js)**: - Created showOverlay()
  function for consistent success/error feedback - Replaces hardcoded overlays and toast
  notifications - Supports success (green) and error (red) types with auto-hide

- **Updated Nodes**: - ScenePromptManager: Added 'Save Flags' button with overlay feedback -
  StoryEdit: Migrated to use showOverlay instead of hardcoded overlay HTML

## Test Coverage - **Backend Tests (13 new tests + 7 integration tests)**: -
  test_scene_prompts_api.py: Comprehensive scene_flags testing (serialization, persistence,
  compositions, array formats, migration) - test_prompt_collection.py: Added
  TestSceneFlagsInCollection with 7 integration tests

- **Frontend Tests**: - prompt_collection_api.test.js: Added scene_flags handling tests (3 tests)

All 51 backend tests passing. Scene flags fully integrated through save/load cycle.

- Migrate to structured logging and fix test infrastructure
  ([`ec5c157`](https://github.com/frost-byte/fbTools/commit/ec5c157718e92f69a1eba05eca8ae0d5ae2a104b))

Complete migration from print statements to structured logging with environment-configurable log
  levels via FBTOOLS_LOG_LEVEL.

Backend Changes: - Add utils/logging_utils.py with get_logger() for centralized logging - Replace
  all print statements with logger calls in extension.py - REST managers now use
  logger.info/warning/error/exception - Node execution uses appropriate log levels
  (debug/info/warning) - Exception paths use logger.exception for full tracebacks - Update
  utils/io.py and prompt_models.py to use structured logging - Add try/except fallback in
  prompt_models.py for test compatibility

Test Infrastructure Fixes: - Remove obsolete tests/test_fb_tools.py (referenced non-existent code) -
  Remove tests/__init__.py (caused pytest package resolution issues) - Update tests/conftest.py to
  properly handle package imports - Clean up unused src/fb_tools/ stub files

Frontend Test Fixes: - Add getCalls() method to mockFetch utility for request inspection - Fix
  libber_api.test.js mock setup and response handling - Suppress expected console.error in error
  handling tests - Fix integration test to provide separate mocks per API call

Test Results: - ✅ 99 Python tests passing (pytest) - ✅ 38 JavaScript tests passing (jest) - ✅ 137
  total tests validating no regressions

Log levels available: DEBUG, INFO, WARNING, ERROR, CRITICAL Set via: export FBTOOLS_LOG_LEVEL=DEBUG

- Register compositing and LoRA stack nodes, update docs and deps
  ([`31bc42e`](https://github.com/frost-byte/fbTools/commit/31bc42e81d26a8fa5f1532c85355478dc512e69a))

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

- Replace libber text input with dropdown in ScenePromptManager
  ([`44fd667`](https://github.com/frost-byte/fbTools/commit/44fd667e61aecdc92612d61386a0e42609a15f69))

Backend changes: - Get list of available libbers from LibberStateManager - Include libbers list in
  UI text array (text[3]) - Always include 'none' as first option

Frontend changes: - Replace prompt-libber-input with prompt-libber-select dropdown - Populate
  dropdown with available libbers from backend - When Type='libber': enable dropdown, auto-select
  first libber if 'none' - When Type='raw': disable dropdown, set to 'none' - Updated all event
  handlers to use dropdown value - Apply button handles 'none' correctly (saves as null) - New row
  starts with 'none' selected and disabled

UX improvements: - No more manual libber name entry (prevents typos) - Clear visual indication of
  available libbers - Consistent behavior between raw/libber types - Better defaults (first
  available libber when switching to libber type)

- Storyedit REST API + comprehensive testing
  ([`8e15d67`](https://github.com/frost-byte/fbTools/commit/8e15d67a4aea3641995bb1984baac3f6a69b2113))

Implement complete REST API architecture for StoryEdit node with immediate data loading and full
  test coverage.

## Features

### REST API Implementation - Add GET /fbtools/story/load/{story_name} endpoint - Loads story.json
  with full scene data - Returns JSON with scenes array - Add POST /fbtools/story/save endpoint -
  Saves updated scenes to story.json - Validates story exists before saving - Frontend fetch() calls
  replace execution-based data transfer - Immediate data loading on node initialization

### Frontend Improvements - loadStoryData() - async load via REST API - saveStory() - async save via
  REST API with success feedback - Enhanced error handling and user feedback - Detailed console
  logging for debugging - Table initialization without workflow execution

### Testing - 9 Python unit tests (all passing) - Helper method logic (prompt text, summary,
  metadata) - Scene resolution and reordering - Data structure validation - 12 JavaScript tests (all
  passing) - Node initialization and UI rendering - Scene management logic - Data validation -
  Execution handler - Comprehensive testing documentation - STORY_EDIT_TESTING_GUIDE.md - manual
  test scenarios - STORY_EDIT_TESTING_SUMMARY.md - test overview - STORY_EDIT_TESTING_FINAL.md -
  results summary

### Bug Fixes - Fix jest test compatibility (global.fetch mock) - Fix console.log expectation
  ("Received story data") - Fix create_mask_overlay_image transparency logic - Add pyright
  configuration for type checking

### Configuration - Add nvm.fish persistence (nvm_default_version v20.19.6) - Configure fish shell
  auto-load for Node.js

## Test Results ✅ 9 Python tests passing in 0.02s ✅ 12 JavaScript tests passing in 0.60s ✅ 21 total
  automated tests ✅ All manual test scenarios documented

## Files Changed - extension.py - REST API endpoints + logging - js/nodes/story.js - Complete UI
  redesign with API calls - js-tests/story_edit.test.js - Full test suite - tests/test_story_edit.py
  - Unit tests - pyproject.toml - Add pyright config - utils/images.py - Fix mask overlay
  transparency

## Architecture Changed from execution-based data flow to REST API: - Before: Execute node → backend
  sends data → frontend displays - After: Select story → frontend fetches via API → immediate
  display

Co-authored-by: GitHub Copilot <copilot@github.com>

- **fbtools**: Add MultiLoraLoader and align LibberApply libber discovery/loading
  ([`1e5f02a`](https://github.com/frost-byte/fbTools/commit/1e5f02aa253c748d77216d461f5e8805c0448135))

add MultiLoraLoader node with up to 10 optional LoRA slots and sequential model-only application
  register MultiLoraLoader in extension node list fix LibberApply.define_schema to include libbers
  from both memory and disk (.json scan), like LibberManager handle libber_name == "none" early in
  LibberApply.execute update frontend LibberApply dropdown population to merge/dedupe/sort libbers +
  files from /fbtools/libber/list remove hardcoded frontend load path (libbers) and load using
  backend-provided libber_dir + matching filename extend /fbtools/libber/list response with
  libber_dir for consistent frontend/backend path resolution

- **LibberApply**: Add interactive table with click-to-insert and undo support
  ([`585288e`](https://github.com/frost-byte/fbTools/commit/585288e26329d57065a078e99159dbedaf7c7d50))

Table Display & Sizing: - Fixed table persistence after node execution by storing updateDisplay
  function reference - Implemented dynamic container height that adapts to node size changes - Added
  resize hooks (onResize) to update table when user resizes node - Set height constraints (min:
  150px, max: 600px) to prevent infinite growth - Fixed bottom edge overlap with 15px margin -
  Improved widget height computation accounting for previous widgets

Interactive Features: - Made table rows clickable to insert lib keys into text input - Added cursor
  position tracking with event listeners (click, keyup, select, focus) - Keys are automatically
  wrapped with configured delimiter when inserted - Stores last cursor position to handle focus
  changes when clicking table - Added hover effect to table rows (background color highlight)

Undo/Redo Support: - Implemented browser native undo/redo using document.execCommand('insertText') -
  Users can now press Ctrl+Z/Cmd+Z to undo insertions - Users can press Ctrl+Y/Cmd+Shift+Z to redo -
  Fallback to manual insertion if execCommand not supported - Maintains ComfyUI state
  synchronization after insertions

UX Improvements: - Corrected widget reference from "input_text" to "text" - Added visual feedback
  with row hover states - Automatic focus return to input after insertion - Cursor positioned after
  inserted text for continued editing

Users can now click any lib key in the table to insert it at their cursor position with full
  undo/redo support.

- **lora**: Add LoRA stack API client and node UI
  ([`8465b49`](https://github.com/frost-byte/fbTools/commit/8465b49faf2dd99444c6c8351f78e251764eec76))

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

- **lora**: Add LORA_STACK output to LoraStackCollect and update WanPreset nodes
  ([`c80f400`](https://github.com/frost-byte/fbTools/commit/c80f400e4dc8b683e95c75058fd6ce6c0bb6b6b0))

- LoraStackCollect: add easy-use compatible LORA_STACK output (list of (lora_name, model_strength,
  clip_strength) tuples) for interop with EasyLoraStack, PowerLoraLoader, and other LORA_STACK
  consumers - WanPresetDefine: replace single-lora Combo inputs with optional LORA_STACK inputs for
  lora_h and lora_l, enabling multi-lora stacks per preset slot - WanPresetSelect: change
  lora_h/lora_l outputs from STRING to LORA_STACK for direct connection to downstream loader nodes

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

- **lora**: Add WanPresetDefine and WanPresetSelect nodes
  ([`4ca75b3`](https://github.com/frost-byte/fbTools/commit/4ca75b3ccedb6c659050902a996101df04d4f19d))

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

- **lora**: Register WanPresetDefine and WanPresetSelect in extension
  ([`2caa999`](https://github.com/frost-byte/fbTools/commit/2caa999654d4e57b3d7bbc4b3fe58f6bb1677752))

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

### Refactoring

- Extract testable scene image saving utilities and flatten directory structure
  ([`d692f73`](https://github.com/frost-byte/fbTools/commit/d692f739072dfad246e231244d5da519949f6315))

- Extract scene image save logic to utils/scene_image_save.py - Add SceneImageSaveConfig class for
  pure data handling - Add ImageSaver class with static methods for I/O operations - Add
  select_scene_descriptor() and generate_preview_text() pure functions - Enable comprehensive unit
  testing without ComfyUI dependencies

- Update extension.py to use extracted utilities - Refactor StorySceneBatch to create flat directory
  structure - Change from job_root/{scene_order}_{scene_name}/output/ to job_root/input/ - Update
  StorySceneImageSave to prefer job_input_dir over job_output_dir - Remove inline class definitions
  in favor of imported utilities

- Unify test import strategy across all test files - Create import_test_module() helper in
  conftest.py - Update all 5 test files to use consistent import approach - Resolve module import
  conflicts with built-in utils namespace - Ensure stable imports using importlib.util with unique
  module names

- Add comprehensive test coverage for scene image saving - Create tests/test_scene_image_save.py
  with 22 unit tests - Test filename generation, filepath generation, descriptor parsing - Test
  scene selection, sorting, index clamping - Test preview text generation for different formats -
  Mock I/O operations for isolated unit testing

- Document testing approach - Add TESTING_GUIDE.md with unified import patterns and best practices -
  Add TEST_SUMMARY.md showing 121/121 tests passing - Include examples and troubleshooting guidance

This refactoring improves testability, maintainability, and consistency across the codebase while
  fixing the directory structure to use a flat job-level input/ directory instead of nested
  per-scene subdirectories.

- Make StoryVideoBatch self-contained with story/job combo widgets
  ([`3b41d92`](https://github.com/frost-byte/fbTools/commit/3b41d92ea6cb2f923c38eefc19ee35d18d374121))

- Removed STORY_INFO input requirement - Added story_name combo widget that lists available stories
  - Added job_id combo widget that lists available jobs (auto-populated from first story) - Node now
  loads story internally based on story_name selection - Added story_name output for reference -
  Single execution needed - no need to run twice to populate job_id combo - Default behavior: loads
  first available story and its jobs automatically

- Remove legacy prompt inputs from SceneCreate, add auto-migration
  ([`2651e3c`](https://github.com/frost-byte/fbTools/commit/2651e3c9235a082b038a55ef1f1107b20306320d))

BREAKING CHANGE: SceneCreate no longer has individual prompt inputs.

Changes: - SceneCreate: Removed girl_pos, male_pos, wan_prompt, wan_low_prompt, four_image_prompt
  inputs - SceneCreate: Now creates empty PromptCollection, users add prompts via ScenePromptManager
  - SceneInfo.from_pose_directory(): Auto-migrates legacy prompts.json files * Detects v2 format
  (has 'version' field) → loads as-is * Detects legacy format → calls from_legacy_dict() for
  migration * No prompts.json → creates empty collection - Simplified SceneCreate execute() -
  removed prompt string handling

Migration path for existing scenes: 1. Load scene with SceneSelect or from_pose_directory 2. Legacy
  prompts.json automatically migrated to PromptCollection 3. Edit prompts via ScenePromptManager 4.
  Compose outputs via PromptComposer

This enables clean separation: SceneCreate handles assets, ScenePromptManager handles prompts.

- Simplify PromptMetadata for node-level composition
  ([`80930db`](https://github.com/frost-byte/fbTools/commit/80930db4ca53ae157def29fdd037c9e02b13b9de))

BREAKING CHANGE: Removed output_slot and order from PromptMetadata. Output composition is now
  handled at the node level, not in metadata.

Changes: - PromptMetadata: Removed output_slot and order fields - PromptCollection: Removed
  compose_output() and get_output_slots() - PromptCollection: Added get_prompt_metadata() and
  get_prompts_by_category() - Legacy migration: Simplified to just convert prompts to raw type -
  Tests: Updated to reflect simplified data model

Rationale: Output composition should be workflow-specific, not prompt-specific. Same prompts can be
  composed differently for images vs video workflows. This eliminates prompt duplication and allows
  dynamic composition.

- Simplify StoryVideoBatch to output input folder path, multiline prompts, and aggregated LoRAs
  ([`7f668ee`](https://github.com/frost-byte/fbTools/commit/7f668ee85c1980bdf7fc31376715c33ff0efe38f))

- Changed StoryVideoBatch to output: 1. input_folder_path - Path to job input folder with ordered
  scene images 2. video_prompts - Multiline string with one prompt per transition (with
  libber/composition processing) 3. loras_high - Aggregated high-priority LoRAs (unique by name) 4.
  loras_low - Aggregated low-priority LoRAs (unique by name) - Removed complex VIDEO_BATCH
  descriptor system - Removed StoryVideoSave node (no longer needed) - Video prompts now fully
  processed with libber substitutions and composition support - LoRAs aggregated across all scenes
  so each lora appears only once per output - Simpler workflow: load images from folder, use
  multiline prompts, apply aggregated LoRAs

### Testing

- Add comprehensive integration tests for prompt composition system
  ([`99004f9`](https://github.com/frost-byte/fbTools/commit/99004f95cdc9404a8cc05304dc75dd4f4105ff86))

- TestPromptCollectionCompose: Test compose_prompts() method * Single/multiple outputs * Missing
  keys handling * Libber substitution with/without manager * Mixed raw and libber prompts

- TestPromptCompositionSerialization: Unicode and JSON roundtrip

- TestLegacyPromptMigration: v1->v2 migration and v2 format detection

- TestPromptCollectionFileOperations: Save/load operations

- TestPromptCompositionWorkflows: Real-world scenarios * Image generation workflow * Video high/low
  quality outputs * Multi-image compositions * Libber-enhanced workflows

All 90 tests passing
