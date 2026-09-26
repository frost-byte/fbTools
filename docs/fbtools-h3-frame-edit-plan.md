# fbTools: Parameterized Background Workflow Runs (H3 Frame Edit)

## Goal

From the fbTools UI extension, let the user pick a video, a frame index, a prompt (from a Prompt Composition or free text) and an output location, then run an H3 single-frame image-edit workflow **in the background** — without loading anything onto the user's canvas.

Mechanism: fetch an API-format workflow template, patch widget values on designated nodes, submit via `api.queuePrompt(0, { output })`, and track completion by `prompt_id`.

Non-goals for this iteration: batch runs across many frames, a standalone CLI, changes to the H3 workflow's internals.

## Background / key constraints

- An API-format prompt is `{ node_id: { class_type, inputs, _meta: { title } } }`. Widget values live in `inputs` by widget name.
- Node IDs are unstable across graph edits; **locate parameter nodes by `_meta.title`**, never by ID.
- `api.queuePrompt` bypasses the canvas. The user's open graph is untouched. The server queue is shared with the user's own runs.
- Frontend "control after generate" (seed randomize/increment) does NOT apply to submitted API JSON. Seeds must be set explicitly on every submission, or ComfyUI will serve cached results.
- `SaveImage.filename_prefix` is relative to ComfyUI's `output/` dir (subfolders allowed); paths escaping it are rejected.
- The frontend highlights executing nodes by ID. If template node IDs collide with IDs in the user's open graph, unrelated nodes may appear to execute. Mitigate by renumbering template node IDs into a high range (e.g. 9001+).

## Phase 0 — Discovery (do this first, report findings before coding)

1. Read the fbTools repo: package layout, `WEB_DIRECTORY`, existing JS extension entry points, node registration style (V1 `NODE_CLASS_MAPPINGS` vs V3 `io.ComfyNode` schema), naming conventions.
2. Find how Source Profiles and Prompt Compositions are stored and loaded (files? JSON? server routes?) and what identifiers the UI uses for them.
3. Inspect the installed ComfyUI and frontend versions (local install: `/home/beerye/comfyui_env/ComfyUI-torch210/`; frontend is the `comfyui_frontend_package` pip package). Confirm, against the actual installed source rather than assumptions:
   - `api.queuePrompt` signature and return shape (`prompt_id`, `number`, `node_errors`), and whether it throws on validation failure.
   - WebSocket event names and `detail` payloads: `execution_start`, `executing`, `progress`, `executed`, `execution_cached`, `execution_success`, `execution_error`, `execution_interrupted`.
   - That files under `WEB_DIRECTORY` are served at `/extensions/<package_dir>/...`.
   - Whether `av` (PyAV) is available in the venv (it is a ComfyUI core dependency in recent versions); whether `cv2` is.
4. Get the H3 single-frame edit workflow from Bee (UI-format JSON). List its required custom nodes and models.

Deliverable: short findings note appended to this plan (or the project's spec file) before Phase 1.

## Phase 1 — Frame input node

Create a node (name per fbTools conventions, e.g. `FBVideoFrame`):

- Inputs: `video_path` (STRING), `frame_index` (INT, min 0).
- Output: `IMAGE` tensor `[1, H, W, 3]`, float32, 0–1 range (ComfyUI convention).
- Implementation: PyAV (preferred, if confirmed available) — seek to nearest keyframe before the target timestamp, decode forward to the exact frame index. Fall back to sequential decode if seeking is unreliable for the container.
- Validate `video_path` against an allowlist of root directories (configurable; default to ComfyUI `input/` plus any roots fbTools already uses for Source Profiles). Reject anything outside with a clear error.
- Error on out-of-range `frame_index` with the video's actual frame count in the message.
- `IS_CHANGED` / fingerprint should include path, mtime and frame index so caching behaves correctly.

Optional (Phase 1b, only if Phase 0 shows it's straightforward): accept a Source Profile identifier as an alternative to a raw path.

Tests: unit tests for frame extraction (first frame, middle frame, last frame, out-of-range, disallowed path) using a small generated test video.

## Phase 2 — Output handling

Default: use core `SaveImage` with `filename_prefix` as a subfolder prefix, e.g. `fbtools/h3_edits/<clip>_f<frame>`.

Only if Bee confirms arbitrary destinations are required: add a small save node (e.g. `FBSaveImageTo`) taking a destination directory + filename, restricted to a configured allowlist of output roots. Never allow unrestricted absolute paths.

## Phase 3 — Workflow template with a parameter contract

1. Starting from Bee's H3 workflow, replace the image source with the Phase 1 node.
2. Set titles on parameter nodes (exact strings are the contract):
   - `IN:video` — the frame node (`video_path`, `frame_index`)
   - `IN:prompt` — the text encode / prompt node (`text` or equivalent widget)
   - `IN:seed` — the node that owns the seed widget
   - `OUT:save` — the save node
3. Export in **API format** and renumber node IDs into the 9001+ range (update all link references). Write a small script for this so it's repeatable when the workflow changes.
4. Store as `web/templates/h3_frame_edit.api.json` (served statically) plus a sidecar `h3_frame_edit.contract.json` describing each parameter: node title, input name, type, required/default.
5. Also save the UI-format version in `example_workflows/` for the template browser and documentation.

## Phase 4 — JS runner module

Create `web/js/` module(s) with two layers:

**Generic runner** (reusable for future templates):

- `loadTemplate(name)` — fetch template + contract, cache in memory.
- `buildPrompt(template, contract, params)` — pure function: deep-clone, find nodes by `_meta.title`, validate required params, set values, throw descriptive errors for missing titles/params. No side effects, so it's unit-testable.
- `runPrompt(prompt)` — call `api.queuePrompt(0, { output: prompt })`; surface `node_errors` / thrown validation errors in a readable form.
- `trackPrompt(promptId)` — returns a promise/handle; subscribes to API events, **filters every event by `prompt_id`**, collects `executed` outputs (image filenames/subfolders), resolves on `execution_success`, rejects on `execution_error` / `execution_interrupted`, exposes progress callbacks. Always removes listeners when done.

**H3-specific wrapper:**

- `runH3FrameEdit({ videoPath, frameIndex, prompt, outPrefix, seed })` — generates a random seed if none given (and returns the seed used), builds, queues, tracks, and resolves with output image URLs (via `/view?filename=...&subfolder=...&type=output`).

## Phase 5 — UI integration

- Add an "Edit frame with H3" action wherever it fits best in the existing extension UI (determine in Phase 0 — likely from the Source Profile view).
- Dialog fields: frame index (with frame count shown), prompt source (Prompt Composition picker or free text), output subfolder/prefix, seed (blank = random).
- While running: show queue position/progress; on success show the resulting image(s) with the seed used; on failure show the error message and failing node.
- Do not call `app.loadGraphData` anywhere in this flow. Optionally add a separate "Open as workflow" button that does, for users who want to inspect the graph.

## Phase 6 — Tests, docs, packaging

- Unit tests: `buildPrompt` (happy path, missing title, missing required param, clone isolation), frame node tests from Phase 1.
- Headless smoke test script `scripts/smoke_h3_frame_edit.py`: loads the same API template, patches params in Python, POSTs to `/prompt`, polls `/history/{prompt_id}`, exits non-zero on failure. Intended for a self-hosted GPU runner or manual runs.
- Node docs: `WEB_DIRECTORY/docs/<NodeName>.md` for each new node.
- README section describing the parameter-contract convention so future templates follow it.
- Check `.comfyignore` keeps `web/templates/`, `web/docs/` and `example_workflows/` in the published package while excluding test fixtures/videos.

## Acceptance criteria

- With an arbitrary unrelated workflow open (and unsaved), running an H3 frame edit from the extension leaves that canvas unchanged and produces the edited image in the chosen output subfolder.
- Running twice with the same inputs but no seed specified produces two different results (seed actually varies); specifying a seed reproduces a result.
- Renaming/moving nodes inside the template without changing titles does not break the runner.
- A template missing a contract title fails with a message naming the missing title.
- A disallowed video path or out-of-range frame index fails before any sampling, with a clear message in the UI.
- Events from the user's own concurrent runs never resolve or pollute the tracking of an extension-launched run.

## Open questions for Bee

1. Location of the H3 single-frame edit workflow JSON and the custom nodes it depends on.
2. Is a subfolder under `output/` sufficient, or are arbitrary (allowlisted) destination directories required?
3. Which roots should the video path allowlist contain?
4. Should the prompt come only from Prompt Compositions, or also allow free text?
