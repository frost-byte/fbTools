# Manual Test Checklist — 2026-09-28

Covers everything landed today: the Qwen-Image-2.1 photo-restoration feature (template, backend,
Tools tab, Settings section, shared lightbox helper), two real bugs fixed in the Compose system's
cast-enrichment path (`image_selection` and `extract_from_video` were both silently ignored), and
the new `MarkerFrameSplit` node for the clip-bridging pipeline design.

Check things off as you go. If something fails, note the node/route/console error next to it rather
than deleting the line — that makes it easy to file a fix and re-test just that item.

## 0. Restart sanity (do this first)

- [ ] `journalctl -u comfyui_377 --no-pager -n 200` shows `comfyui-fbTools` loading cleanly, zero
      import errors/tracebacks — especially `nodes/qwen21_photo_restore.py` and
      `nodes/marker_frame_split.py` (both new today)
- [ ] Browser console is clean on initial page load
- [ ] `GET /object_info` includes `fbt_MarkerFrameSplit`

## 1. Qwen-Image-2.1 Photo Restoration (Tools tab)

Already live-tested once this session, but worth a clean re-check after this restart:

- [ ] Tools tab renders: file-tree picker (input/output), source preview, hint/prompt-override
      fields, Restore button, Free VRAM button, result panel
- [ ] Selecting a source image shows its preview; clicking the preview opens the click-to-zoom
      lightbox (`js/ui/lightbox.js`) — same for the result preview after a run
- [ ] `POST /fbtools/tools/restore_photo` runs end-to-end and returns a real output file
- [ ] `restore_hint` is respected (substituted into `{{RESTORE_HINT}}` in the default prompt)
- [ ] The advanced full-`prompt` override field replaces the default prompt entirely when set
- [ ] "Use as source" re-selects the result as the new source image (chaining)
- [ ] `GET /fbtools/tools/restore_photo_settings_options` returns correct `has_*_override` flags
      and `template_defaults` matching `templates/qwen21_photo_restore.api.json`
- [ ] Settings → "Qwen Photo Restore" section renders, every override field round-trips
      (save → reload), and disabled fields stay disabled per the `has_*_override` flags

## 2. Compose system — cast-enrichment bug fixes

Both bugs were confirmed live via `CompositionToH3: loading N reference item(s)` log output before
the fix; re-confirm the same way after this restart.

- [ ] A Scene Cast Build entry with `image_selection` set to a subset of a bundle's images produces
      exactly that subset in the H3 refplan — not every image in the bundle
- [ ] A Scene Cast Build entry with `audio.source = "extract_from_video"` (a separate video used
      purely as a voice-timbre source) sets `voice.audio_reference_file` correctly, **even when
      that same entry has no video reference at all** (`visual_mode = "images"`)
- [ ] Removing images from a Reference Bundle correctly reduces the count in the next generation's
      reference log (sanity check that nothing regressed the fix)

## 3. MarkerFrameSplit (new node, not yet live-tested)

- [ ] Node appears in the node picker under `🧊 frost-byte/Video` as "Marker Frame Split"
- [ ] Feed it a real IMAGE batch with a solid-color marker segment spliced in (e.g. via ffmpeg:
      concat a short clip + N frames of solid magenta + another short clip, load with
      `VHS_LoadVideo`) — confirm `clip_a_frames`/`clip_b_frames`/`clip_a_end_idx`/
      `clip_b_start_idx`/`marker_frame_count` all match the real splice point
- [ ] A marker color/tolerance that doesn't match anything in the batch raises a clear `ValueError`
      (not a silent wrong-data pass-through)
- [ ] Sanity-check the `tolerance` and `min_marker_frames` widgets actually change behavior at the
      edges (e.g. a too-tight tolerance misses a slightly-compressed marker; a `min_marker_frames`
      higher than the real marker run's length also fails to find it)

## 4. Not covered by this checklist

- The Qwen-Image-2.1 character/face-sheet workflow (`templates/qwen21_character_sheet.api.json`,
  `patch_qwen21_character_sheet_prompt`) is **shelved**, not shipped — H3 remains the production
  backend for that feature (see memory `project_qwen_character_sheet_shelved`). Don't test it as if
  it were live; there's no route/UI wired to it.
- The rest of the clip-bridging pipeline (`MiniMaxH3AddGuide` chaining, the full bridge-generation
  graph) is still a design sketch, not built yet — `MarkerFrameSplit` is the only piece that exists
  so far.
- OpenShot-ComfyUI's isolated-venv audio fix (`_openshot_audio_python()`, the DeepFilterNet/LavaSR
  subprocess redirect) lives in a separate repo (`OpenShot-ComfyUI`, not `comfyui-fbTools`) and was
  already smoke-tested directly via both runner scripts — not re-listed here since it's not part of
  this repo's own test surface.
