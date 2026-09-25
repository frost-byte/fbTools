# Manual Test Checklist — 2026-09-24

Covers everything landed since the last ComfyUI restart: the Node Inspector UX fixes, the
frontend i18n/tooltip system, the full `extension.py` → `nodes/` package-split refactor (Plans
1-29, pure code motion but worth spot-checking every domain that moved), and Phase 1 of the
component-reference-system plan (background-as-`<Subject N>` reference for Source Profiles).

Check things off as you go. If something fails, note the node/route/console error next to it
rather than deleting the line — that makes it easy to file a fix and re-test just that item.

## 0. Restart sanity (do this first)

- [ ] `journalctl -u comfyui_377 --no-pager -n 200` shows `comfyui-fbTools` loading cleanly, zero
      import errors/tracebacks
- [ ] Browser console is clean on initial page load (no red errors from `fb_tools.js` or any
      `js/ui/*`/`js/nodes/*` file)
- [ ] `GET /object_info` returns successfully and includes `fbt_*` node ids (spot-check a handful
      from each section below rather than all ~76)

## 1. Node Inspector / JSONViewer UX

- [ ] Mousewheel scrolls the JSON viewer's own container, not the page behind it
- [ ] Expand/collapse carets are visible and clickable at every nesting level
- [ ] A toast notification appears on the relevant node-graph interaction (`handleNodes()`)
- [ ] No leftover/dead UI from the old JSONViewer implementation

## 2. Frontend i18n / command tooltips

- [ ] Hovering the "Extract Node JSON" command button shows a tooltip (not blank)
- [ ] Hovering the "Send Get/Set to Back" command button shows a tooltip (not blank)
- [ ] Network tab: `GET /i18n` returns the `commands.json` entries under `en` (or your active
      locale)
- [ ] `docs/frontend_i18n_localization.md` renders/reads correctly as a reference if you need it

## 3. extension.py → nodes/ package split (functional spot-checks)

Pure code motion, but every node's import path changed — spot-check one node per moved domain
rather than assuming the AST diffs caught everything live.

**Scene domain**
- [ ] Scene Create / Scene Update load their schema and run without error
- [ ] `/fbtools/scene/list` and a thumbnail route return real data

**Story domain**
- [ ] Story Load / Story Edit / Story View load an existing story correctly
- [ ] Story Scene Batch / Story Scene Pick run against a real story
- [ ] `/fbtools/story/list`, `/fbtools/story/load/{name}` return real data
- [ ] `/fbtools/story/regenerate_thumbnails` works (exercises the new `SceneInfo` sibling import)

**LoRA presets**
- [ ] LoRA Preset Define/Select and Wan Preset Define/Select load their schema and list presets

**Registry layer**
- [ ] Concept Registry Load / Concept Define / Concept Resolve / Concept List
- [ ] Subject Profile Load / Define / List
- [ ] Scene Template Load / List
- [ ] Outfit Registry Load / Define / List

**Source Profile layer**
- [ ] Source Profile Load / Define / List
- [ ] Source Profile Clip Prompt runs end-to-end against a real profile+clip (exercises
      `_build_h3_refplan`, `asyncio` usage moved into this file)
- [ ] A `/fbtools/source_profiles/*` route round-trip (save, then load back) works

**Composition-assembly layer**
- [ ] Scene Compose runs against a real scene template
- [ ] Prompt Assemble runs — **watch for a known pre-existing bug**: if a `scene_cast` is wired
      in, `_resolve_cast_media(scene_cast)` is called with one arg but the function needs
      `(scene_cast, bundle_registry)` — likely raises `TypeError`. Not yet fixed; confirm whether
      it actually triggers in your workflow.
- [ ] Scene Cast Load / Scene Cast Build
- [ ] Composition Load / Prompt Composition Loader / Composition To H3 Conditioning
- [ ] Bundle routes: list/upload/preprocess-audio round-trip
- [ ] Compositions/settings routes round-trip

**Grab-bag (compositing / image_processing / qwen_conditioning / audio / utility)**
- [ ] Subject Layer Define → Subject Compositor produces a composited image
- [ ] SAM Preprocess NHWC
- [ ] Tail Enhance Pro
- [ ] Tail Split
- [ ] Opaque Alpha
- [ ] Mask Processor
- [ ] FB Text Encode Qwen Image Edit Plus
- [ ] Qwen Aspect Ratio (category still shows "Image Processing" in the node picker — expected)
- [ ] Audio Fix Shape
- [ ] Subdir Lister

**Scene Prompt Management (moved to `nodes/narrative/scene_prompts.py`)**
- [ ] Scene Prompt Manager lists/composes prompts for a real scene
- [ ] Prompt Composer runs against a `SCENE_INFO` input

## 4. Phase 1 — Background-as-`<Subject N>` reference (Source Profiles)

- [ ] "Default background" dropdown appears in a Source Profile's Video settings section,
      populated from the same `backgrounds.json` registry Compositions use
- [ ] Selecting a default background persists after save + reload
- [ ] Per-clip "Background" dropdown appears in each clip card, defaulting to "— none —"
- [ ] Setting a per-clip background persists after save + reload
- [ ] The "→ all" button next to the per-clip background copies it to every other clip
- [ ] Per-clip `background_id` overrides the profile's `default_background_id` when both are set
- [ ] Profile's `default_background_id` is used as fallback when a clip has none set
- [ ] Generated prompt (Source Profile Clip Prompt output) includes a `<Subject N>` reference and
      `character_sheet_images` for the chosen background, when that background has
      `reference_images`
- [ ] A background with no `reference_images` still contributes its `description`/`lighting` text
      to the prompt's `detailed_description:` section, but does **not** mint a `<Subject N>`/
      `<Picture N>` slot
- [ ] Clip Prompt execution doesn't error when `background_id` is empty on both the clip and the
      profile (no background selected at all)

## 5. Not covered by this checklist

Phase 2 (outfit `Fit_N` wearer-disconnection fix) is still blocked on a live H3 generation test
with a flat-lay/unworn outfit reference image — that's a design validation step, not a regression
check, so it isn't in this list. Flag me when you're ready to run that and we'll look at the
result together.
