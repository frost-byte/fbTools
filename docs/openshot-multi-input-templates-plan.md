# openshot-qt: Declared Multi-Input ComfyUI Templates

## Goal

Generalize the existing one-off `needs_reference_image` mechanism into **declared extra inputs** for ComfyUI templates, so a template can require any number of additional project files (image, video, or audio) *or free-form text fields*, each chosen/entered in the Generate dialog and bound to a named placeholder in the workflow.

Also fix a related bug: files bound to custom (non-core) loader nodes are never uploaded to ComfyUI, so they fail whenever ComfyUI runs on a different machine. This is planned as a **separate PR**, opened after or alongside this one rather than bundled in — smaller, independently reviewable, and not blocking on this one's design settling.

Target: an upstream PR against `OpenShot/openshot-qt` `develop`. Keep it small, generic, and backward compatible. **Nothing H3-specific or bridge-specific goes in this PR.**

**Downstream motivation (not part of the PR itself, context for why this matters to us):** with N declared video inputs + a text field, the same mechanism covers several related fbTools templates without further OpenShot-side work — "bridge" (two video inputs, anchor both ends), "extend" (one video input, anchor only the start, no end anchor), and "prepend" (one video input, anchor only the end) are all just different template JSON files against this one generalized capability, not separate features to build here. The text field is what lets those templates take a per-run prompt detail (e.g. a sidecar-derived scene description) instead of a hardcoded prompt baked into the template.

## Current behavior (verified against `develop` @ b220618, 2026-09-28)

Integration files:

| File | Role |
|---|---|
| `src/classes/comfy_templates.py` | `ComfyTemplateRegistry`: discovers templates (built-in `src/comfyui/` + user `info.COMFYUI_PATH`), parses metadata in `_load_template()`, infers input/output types |
| `src/windows/generate.py` | `GenerateMediaDialog`: prompt, name, and the single `reference_image_combo` |
| `src/classes/generation_service.py` | `_enqueue_generation_for_file()`, `_prepare_template_workflow()` (placeholder substitution), `action_generate_trigger()` |
| `src/classes/comfy_client.py` | `_rewrite_prompt_local_file_inputs()` uploads local files via `upload_input_file()` → `/upload/image` |
| `src/classes/generation_queue.py` | job queue; one active job per source file |

Existing precedent: `src/comfyui/video2video-basic.json` sets `"needs_reference_image": true`; its LoadVideo uses `"__openshot_input__"` and a LoadImage uses `"__openshot_reference_image__"`. The dialog shows `reference_image_combo`, a `QComboBox` populated from `File.filter()` (i.e. files already imported into the current project's Project Files, not an OS file-browse dialog — confirmed directly in `generate.py`, no `QFileDialog` anywhere in it); the service substitutes both placeholders (also `{{openshot_…}}` and `$openshot_…` variants).

Constraints:

1. **Multi-select already means batch.** `action_generate_trigger()` enqueues one job per selected file; `can_open_generate_dialog()` only allows the dialog with ≤1 selected file. Extra inputs must therefore be chosen **in the dialog**, not via multi-selection. Do not change batch semantics.
2. **Upload is class-sniffed.** `_rewrite_prompt_local_file_inputs()` only uploads for `LoadImage`, `LoadVideo`, the VHS video loaders, `LoadAudio` and `VHS_LoadAudioUpload`. The service substitutes placeholders into *any* node's inputs, so custom nodes receive a raw local absolute path, which breaks on remote ComfyUI servers.

## Environment (set up 2026-09-29)

Dev/test environment built and confirmed working, isolated from the production ComfyUI host via Docker (no system-level changes to that host beyond installing Docker Engine itself):

- `openshot-qt` cloned at `develop` to `/home/beerye/dev/openshot-qt` on the ComfyUI Linux box (separate from anything ComfyUI-related).
- `/home/beerye/dev/openshot-qt.Dockerfile` — Ubuntu 24.04 image mirroring `.github/workflows/ci.yml`'s install steps exactly (the `libopenshot-daily` PPA, `python3-openshot`, PyQt5, `QT_QPA_PLATFORM=offscreen`), built as `openshot-qt-dev:latest`. The repo is bind-mounted at `/workspace` at run time, not baked into the image, so edits on the host are picked up without rebuilding.
- Confirmed `import openshot` works inside the container (`OPENSHOT_VERSION_FULL` = `1.0.1`) — no DLL/ABI issues like the ones hit trying to reuse the Windows GUI install's bundled Python for the earlier (abandoned) clip-concatenation-script idea.
- **Baseline test run**: `docker run --rm -v /home/beerye/dev/openshot-qt:/workspace openshot-qt-dev python3 -m unittest discover -s src/tests -t src/tests --quiet` → **877 tests, all passing (1 skipped), clean run.** This is the baseline Phase 0 asks for below.

## Phase 0: Discovery (report before coding)

1. Re-read the five files above on current `develop`; line numbers in this plan will drift, so work from function names.
2. Read `src/tests/test_generation_service.py` and `src/tests/test_project_data.py` to learn test conventions, fixtures, and how the suite is run. Run the existing suite and record the baseline.
3. Check how `_populate_reference_image_combo()` filters files and how `GenerateMediaDialog` validates and toggles visibility (`_needs_reference_image()`, and the required-input check in accept/validation).
4. Check CONTRIBUTING / PR conventions in the repo (branch naming, commit style, changelog expectations).
5. Grep for any other consumer of `needs_reference_image` or `reference_image_file_id` (templates, dialogs, tests) so none are missed.

Deliverable: short findings note, including any place where the design below conflicts with what's actually there.

### Findings (2026-09-29, against `develop` @ `b220618` — confirmed no drift from the citation above)

- **All consumers accounted for** — grepped the whole `src/` tree for `needs_reference_image`, `reference_image_file_id`, `reference_image_combo`, `__openshot_reference_image__`: exactly `generate.py` (12 refs), `comfy_templates.py` (2), `generation_service.py` (3), and the one template `video2video-basic.json`. No other template, dialog, or test file touches this mechanism — nothing hidden to account for.
- **`templates` wrapping resolved (not a bug)**: `generate.py`'s `template_map` does `t.get("template")`, but `_load_template()` never returns a `"template"` key — only `"workflow"`. This isn't a bug: `generation_service.py`'s `templates_for_context()` (the method that actually feeds `GenerateMediaDialog`) wraps each raw template dict as `{"id", "name", "template": t}` first. So `_current_template()` does correctly return the full dict, and a new `extra_inputs` key will flow through this exact same path `needs_reference_image` does today — confirmed by reading the wrapping code, not assumed.
- **The dialog is a fixed 4-tab layout** (`Prompt` / `Reference` / `Tracking` / `Highlight`), not a flat form. The reference-image combo lives inside a dedicated `page_reference` tab (`_build_reference_tab()`) whose body today is just that one combo. Generalizing to N inputs means rebuilding *this tab's contents* (one widget per declared input, per Phase 2), not adding new top-level dialog sections — matches the plan's intent but is more concrete now about exactly which method to rewrite.
- **Tab visibility and validation are a straight-line hardcoded sequence**, not a loop over anything today. `_on_template_changed()` hardcodes visibility for all 4 tabs based on two flags: `_needs_reference_image()` and `_is_track_object_template()` (itself a hardcoded list of exactly 6 SAM2 template IDs — `video-blur-anything-sam2`, `video-mask-anything-sam2`, `video-highlight-anything-sam2`, and the 3 `image-*` equivalents). `_on_generate_clicked()` is similarly a hardcoded `if` chain: name required → reference image required if needed → SAM2 coordinate/prompt required if a track-object template. **Our `extra_inputs` validation loop must be layered in alongside these, not replace them** — the SAM2-specific tabs/checks are a separate, unrelated mechanism that must keep working unchanged. No evidence any SAM2 template also sets `needs_reference_image`, so no known conflict case between the two mechanisms today, but worth a defensive test regardless.
- **`_populate_reference_image_combo()` hardcodes `media_type == "image"`** — confirmed directly. Generalizing it into a type-filtered helper (Phase 2) is a real code change, not just a rename; the same method today has no parameter for which media type to filter on at all.
- **No dialog-level test coverage exists at all.** `src/tests/` has no `test_generate.py` — `GenerateMediaDialog` (tabs, validation, payload building) has zero existing automated tests to extend or mimic. `test_generation_service.py` (877 tests total in the full suite, confirmed passing) covers the service/backend layer only. **Recommendation: keep this PR's new tests scoped to `test_generation_service.py`-style service/schema tests (per Phase 5), and don't introduce new Qt-dialog-widget test infrastructure that has no precedent in this codebase** — consistent with existing practice, keeps the PR's footprint proportionate.
- **Open question worth raising in the PR/issue discussion, not deciding unilaterally**: should the "Reference" tab be renamed (e.g. "Extra Inputs") now that it'll hold more than one image reference? A visible UX change a maintainer may have their own opinion on.

## Phase 1: Template schema (`comfy_templates.py`) — **DONE, 2026-09-29**

Implemented as `_parse_extra_inputs()`, called from `_load_template()`. Verified: full suite still 877 passing/1 skipped; `video2video-basic.json` correctly synthesizes `[{"key": "reference_image", "type": "image", "label": "Reference image", "required": true}]` from its existing `needs_reference_image: true` with no `extra_inputs` key at all (real backward-compat case, not synthetic); a synthetic multi-entry template confirmed every validation path (invalid key rejected, duplicate key rejected, invalid type rejected, auto-derived label from key, explicit `reference_image` entry correctly suppresses the synthesized one).



In `_load_template()`:

- Parse an optional top-level key:
  ```json
  "extra_inputs": [
    { "key": "end_clip", "type": "video", "label": "End clip", "required": true },
    { "key": "scene_note", "type": "text", "label": "Scene detail (optional)", "required": false }
  ]
  ```
  - `key`: `[a-z0-9_]+`, unique within the template; reject/skip invalid entries with a `log.warning`.
  - `type`: one of `image`, `video`, `audio`, `text`.
  - `label`: optional display string; default derived from `key`.
  - `required`: optional, default `true`.
  - `text` inputs have no `File` to resolve — the dialog renders a line/multi-line edit instead of a combo, and the raw string goes straight into the placeholder substitution in Phase 3 (no upload/path handling applies, so Phase 4 never sees these).
- Backward compatibility: if `needs_reference_image` is true and no `extra_inputs` entry has key `reference_image`, synthesize `{ "key": "reference_image", "type": "image", "label": "Reference image", "required": true }`.
- Return the normalized list as `template["extra_inputs"]`. Keep `needs_reference_image` in the dict for any remaining callers.

## Phase 2: Dialog (`windows/generate.py`) — **DONE, 2026-09-29**

Implemented: `_build_reference_tab()` now builds an empty `QFormLayout` container instead of the single hardcoded combo; `_rebuild_extra_inputs()` populates it per-template (combo for image/video/audio via the generalized `_populate_media_combo()`, `QLineEdit` for text); `_first_missing_required_input()` replaces the single hardcoded reference-image check; `_on_template_changed()`/`_on_generate_clicked()`/`get_payload()` all updated accordingly. `_needs_reference_image()` removed (fully superseded, confirmed no other callers). Reference tab intentionally left named "Reference" per your call, pending any maintainer feedback on the issue.

Verified with a throwaway functional smoke test (not kept, per the Phase 0 recommendation against adding new dialog-test scaffolding) against a synthetic two-input template (one required video, one optional text), using the same `get_or_create_app`/`ensure_app_state` pattern `test_add_to_timeline.py` already establishes: tab visibility, correct widget type per declared type, combo correctly filtered by media type (video-only combo excluded fake audio/image files), validation catching the missing required field and clearing once filled, and `get_payload()`'s `input_file_ids`/`input_text_values`/`reference_image_file_id` all matching spec. Full 877-test suite still clean throughout.



- Replace the single `reference_image_combo` with a small container that builds **one input widget per declared extra input**, rebuilt when the selected template changes: a combo for `image`/`video`/`audio` (filtered by type), a `QLineEdit`/`QTextEdit` for `text`.
- Generalize `_populate_reference_image_combo()` into a helper that populates a combo filtered by media type (reuse the existing thumbnail/tooltip behavior). `text` inputs skip this entirely — no file list to populate.
- Validation: every `required` input must have a selection/non-empty value; focus the first missing one (mirrors current reference-image behavior).
- `get_payload()` returns `"input_file_ids": { key: file_id, ... }` for media inputs and `"input_text_values": { key: str, ... }` for text inputs — kept as two separate dicts rather than one polymorphic one, so downstream file-resolution code never has to type-check a value before treating it as a `File` id. Also keep emitting `reference_image_file_id` when a `reference_image` input exists, so nothing downstream breaks mid-refactor.
- Exclude the primary source file from each media combo only if the existing reference combo does so; otherwise leave behavior unchanged.

## Phase 3: Service (`generation_service.py`) — **DONE, 2026-09-29**

Implemented: `_prepare_generation_source_path()`'s img2img-PNG-conversion logic extracted into a reusable `_convert_to_supported_img2img_path()`, now also applied to any image-typed extra input. `_prepare_template_workflow()` gained `extra_input_paths`/`extra_input_texts` dict parameters (with `reference_image_path` still accepted and merged into `extra_input_paths["reference_image"]` for compatibility), a `_named_input_placeholder_key()` matcher for all three named-placeholder syntaxes, and now returns `(workflow, bindings)` instead of just `workflow` (confirmed the only caller before changing the return shape). `_enqueue_generation_for_file()` now loops over the template's declared `extra_inputs`, resolving each against `payload["input_file_ids"]`/`input_text_values` (falling back to the legacy `reference_image_file_id` key), erroring clearly on a missing required input without raising, and threading `bindings` through to the `request` dict for Phase 4.

Verified with a throwaway functional smoke test (deleted after, per the established pattern): all three named-placeholder syntaxes resolve to the correct extra-input value; the legacy `__openshot_input__`/`__openshot_reference_image__` placeholders still work; a text-type substitution applies its value but is confirmed absent from the bindings list; an unresolvable key is left untouched with a warning logged; a missing required input produces a clear error tuple, not an exception; a satisfied required input succeeds through to `generation_queue.enqueue()`. Full 877-test suite stayed clean throughout.



`_enqueue_generation_for_file()`:
- Replace the single `reference_image_file_id` lookup with a loop over `payload["input_file_ids"]` (falling back to `reference_image_file_id` if the new key is absent). Resolve each `File` → path. Error clearly if a required input's file is missing.
- Pass each image path through the same conversion logic `_prepare_generation_source_path()` applies to image sources where appropriate (refactor that conversion into a reusable helper rather than duplicating it). Track temp files the same way.
- Pass `payload["input_text_values"]` straight through into `extra_input_paths` alongside the resolved media paths (Phase 3's substitution step doesn't care whether a value came from a file or a text box — it just replaces a placeholder with a string). Error clearly if a required `text` input is empty.

`_prepare_template_workflow()`:
- Accept `extra_input_paths: dict[str, str]` (keep `reference_image_path` as a compatibility argument mapped into the dict).
- Named placeholder forms, matched case-insensitively like the existing ones:
  - `__openshot_input:<key>__`
  - `{{openshot_input:<key>}}`
  - `$openshot_input:<key>`
- Keep the existing `__openshot_reference_image__` family working (maps to key `reference_image`).
- **Record bindings**: every time a placeholder is replaced with a local file path, append `(node_id, input_key, local_path)` to a list. Include the primary `__openshot_input__` substitutions too. **`text`-type substitutions are never added to this list** — Phase 4's upload fix iterates bindings and calls `upload_input_file()` on each; a plain string handed to that would either error or silently upload garbage, so bindings are file-paths-only by construction (build the list only inside the file-substitution branch, not the text one). Return the bindings list alongside the workflow (or attach it to the request) without changing the workflow JSON sent to ComfyUI.
- Add the bindings list to the `request` dict built in `_enqueue_generation_for_file()`.

## Phase 4: Upload fix (`comfy_client.py` + queue plumbing)

- Extend `queue_prompt()` / `_rewrite_prompt_local_file_inputs()` to accept the bindings list.
- For each binding: if the value is still that local absolute path and exists, upload with `upload_input_file()` and replace the value with the uploaded reference, **regardless of node class**. Normalize the ` [input]` suffix the same way the existing audio/VHS branches do, and document the chosen convention in a comment.
- Keep the existing class-based rewriting as a fallback for templates/paths not covered by bindings, and avoid uploading the same file twice.
- Thread the bindings from the request through `generation_queue.py` `_run_comfy_job()` to the client call. Confirm nothing else in the queue (one-active-job-per-source-file logic, badges, cancel) needs changes; extra inputs should not claim file slots.

## Phase 5: Tests — **DONE, 2026-09-29**

Added `src/tests/test_comfy_templates.py` (new file, 12 tests) covering Phase 1's `_parse_extra_inputs()`: all four types valid, missing/non-list/non-dict entries, invalid key, duplicate key, invalid type, auto-derived label, `required` defaulting/coercion, and both `needs_reference_image` synthesis cases. Added 6 new tests to the existing `test_generation_service.py` (matching its established `GenerationService.__new__(GenerationService)` + `types.SimpleNamespace` + `patch(...)` conventions, no new test infrastructure introduced) covering Phase 3's substitution/bindings/validation.

**A real bug was caught by the new tests, not just confirmed passing**: the substitution loop's original performance guard (`if source_path or reference_image_value or extra_input_paths or extra_input_texts:`) skipped the entire per-node scan whenever none of those were provided — which also silently skipped the unknown-extra-inputs-key warning in that case, meaning a typo'd `__openshot_input:key__` placeholder with no other substitutions present would go completely undetected. Fixed by always running the scan (cheap for a template's node count; correctness here matters more than the micro-optimization). Deliberately **not** testing Phase 4's upload behavior here since that's a separate, not-yet-implemented PR — that test bullet from the original Phase 5 plan belongs in the upload-fix PR's own test suite instead.

Full suite: 895 tests, all passing (877 original + 12 + 6), 1 skipped (pre-existing, unrelated).

## Phase 5 (original plan text, superseded by above)

In `src/tests/test_generation_service.py` (and a new test module for the client if cleaner):

- Template parsing: valid `extra_inputs` across all four types (including `text`); invalid entries skipped with warning; `needs_reference_image` synthesizes a `reference_image` input.
- Substitution: all three named placeholder syntaxes; multiple extra inputs in one workflow (mixing media and `text`); legacy `__openshot_reference_image__` still works; unknown key left untouched (and logged).
- Bindings: recorded for primary and extra *media* inputs, including a custom node class with a non-standard key (e.g. `video_path`); a `text` input's substitution never appears in the bindings list even though the same node also receives a media binding.
- Client: a binding on a non-core node class triggers `upload_input_file()` (mock it) and rewrites the value; core loader behavior unchanged; no double upload.
- Enqueue: missing required extra input produces a clear error, not an exception — for both an unfilled media picker and an empty required `text` field.

All existing tests must still pass.

## Phase 6: Example template and docs — **DONE, 2026-09-29**

New template `src/comfyui/image-blend-multi-input-demo.json` ("Blend + Restyle (Multi-Input Demo)"), using only core nodes already in `KNOWN_NODE_TYPES` (`LoadImage`, `ImageBlend`, `CheckpointLoaderSimple`, `CLIPTextEncode`, `VAEEncode`, `KSampler`, `VAEDecode`, `SaveImage` — zero new entries needed) and the same `sd_xl_base_1.0.safetensors` checkpoint already used by `txt2img-basic`/`img2img-basic`, so it needs no custom node packages and no model beyond what a reviewer testing the other bundled templates would already have. Declares two extra inputs — `overlay_image` (image) blended over the primary source, `style_prompt` (text) feeding `CLIPTextEncode`'s positive prompt directly via the new named-placeholder syntax.

**A real interaction was caught by actually running this through the full pipeline, not just eyeballing the JSON**: an earlier design routed the `text` extra input into `SaveImage`'s `filename_prefix`, which silently never worked — `_prepare_template_workflow()` has a *pre-existing*, unconditional block (`if "filename_prefix" in inputs: ...`) later in the same per-node loop that always overwrites that field with the internal generation name, regardless of what was substituted into it moments earlier. Not a bug (it's how OpenShot keeps generated output filenames predictable/collision-free), but it means `filename_prefix` can never be a target for an `extra_inputs` text substitution. Redesigned to feed `style_prompt` into `CLIPTextEncode` instead — semantically correct anyway, and confirmed working end-to-end (verified the real substituted value lands in the right node, bindings correctly contain only the two media substitutions, `filename_prefix` still gets the generation name as expected).

Docs: added a new "Templates With Multiple Inputs" subsection to `doc/ai.rst` (the real user guide covering `needs_reference_image` etc. had never documented that mechanism *or* the old single-reference-image one — a pre-existing gap, now closed for both), a new entry for the demo template matching the existing per-template format exactly, and a small wording fix to the AI Action Dialog's now-outdated "pick a reference image" bullet. Verified by actually building the docs with Sphinx (matching `.github/workflows/sphinx.yml`'s real `sphinx-build`/dependency versions) — caught and fixed one real "title underline too short" warning before reaching a clean build.

Full suite: 895 tests passing, 1 skipped, unchanged from Phase 5.

## Phase 6 (original plan text, superseded by above)

- Add one built-in template to `src/comfyui/` that uses **only core ComfyUI nodes** and demonstrates two inputs — one media, one `text` (e.g. a first-frame/last-frame image-to-video where the text input feeds a `CLIPTextEncode` prompt, so reviewers see both input kinds exercised in one example). It must be testable by reviewers without custom nodes. Set `menu_category`, `menu_order`, `input_type`, `output_type`, `template_id` like existing templates, and add any new core node types to `KNOWN_NODE_TYPES` so no warning is logged.
- Update the AI user-guide docs (search `doc/` for the ComfyUI/templates section) with the `extra_inputs` key, all four types, and named placeholder syntax.

## Out of scope (follow-up PRs)

- Timeline integration (AI menu on timeline clips in `src/windows/views/timeline.py`; binding trimmed timeline spans as inputs). The AI menu is currently only in `files_listview.py` / `files_treeview.py`.
- Any H3, bridge, or marker-frame logic. That lives in Bee's user templates and the fbTools ComfyUI node package.

## Acceptance criteria

- Existing templates, including `video2video-basic.json`, behave identically.
- A template declaring two extra inputs shows two type-appropriate widgets (picker or text field), blocks submission until required ones are filled, and runs successfully.
- With ComfyUI on a **different machine**, a template whose custom node receives an extra input via a named placeholder runs successfully (file uploaded, not a local path sent).
- Batch generation over multiple selected files is unchanged.
- New and existing tests pass.

## Appendix: draft issue text (post before opening the PR)

> **Proposal: declared multi-input ComfyUI templates (generalizing `needs_reference_image`)**
>
> Templates can currently take one extra input, a reference image, via `needs_reference_image` and `__openshot_reference_image__`. Many current video/image models take several inputs (first + last frame, video + audio for lip-sync, clip + style reference, two clips for a transition), and some need a per-run text detail (a scene description, a style note) that shouldn't have to be hardcoded into the template's own prompt. I'd like to generalize the reference-image mechanism into an `extra_inputs` list in template metadata — `image`/`video`/`audio` pickers plus a `text` field type — with one widget per input in the Generate dialog and named placeholders (`__openshot_input:<key>__`). Existing templates keep working unchanged.
>
> Related bug: `_rewrite_prompt_local_file_inputs()` only uploads files for core loader node classes, so templates using custom loader nodes send local paths to ComfyUI and fail when ComfyUI runs on another machine. I'd like to fix this too, but as a separate PR rather than bundled into this one — happy to open it first, after, or in whichever order is more convenient to review.
>
> Does this general direction seem reasonable before I put together a PR?
