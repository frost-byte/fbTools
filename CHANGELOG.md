# CHANGELOG


## v1.30.0 (2026-10-09)

### Bug Fixes

- **bundles**: Don't send a bundle's voice twice when its video also supplies an audio track
  ([`5ecae05`](https://github.com/frost-byte/fbTools/commit/5ecae051640c5319225486955e1626233567387d))

In Source Profile mode a bundle with a reference video and a separate voice (a file or another
  video) got two audio references: the video entry's soundtrack, which loads the bundle's processed
  audio cache, and the standalone voice. The video entry now carries its own audio only when that
  track is the bundle's voice.

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_016URszCL3deuwf7g3rSPoYu

- **h3**: Fail instead of skipping reference media that cannot be loaded
  ([`f257d0a`](https://github.com/frost-byte/fbTools/commit/f257d0a476a3e5b25a27186ee92763181c38c9bf))

A missing or unreadable image, video, soundtrack or standalone audio was logged and skipped, so the
  prompt kept its <Picture N>/<Video N>/<Audio N> tag with no media behind it.
  CompositionToH3Conditioning now collects every load failure and raises one error listing them all.

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_016URszCL3deuwf7g3rSPoYu

- **h3**: Re-run the conditioning node when a processed audio cache file changes
  ([`c40c727`](https://github.com/frost-byte/fbTools/commit/c40c727ef1966bcb9fe16da274a0eca959e14654))

The cache path already encodes its settings, but a file rewritten in place at the same path was not
  noticed. Track the audio_cache mtime alongside the source path's.

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_016URszCL3deuwf7g3rSPoYu

- **h3**: Size the VRAM estimate from what the reference node actually sends
  ([`5c0eb0f`](https://github.com/frost-byte/fbTools/commit/5c0eb0fee27273e294ee3d52e0ffc7ccd90c29a4))

The estimate priced every reference at the generation canvas, so changing a Source Profile proxy's
  resolution never moved it. Videos are now priced at the native node's own 768-based size (a
  smaller proxy keeps its size), capped to the output length and trimmed to 17k+5, and images follow
  ref_image_size. The Video log line shows the size actually sent.

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_016URszCL3deuwf7g3rSPoYu

- **source-profiles**: List every valid multiple of 32 for the reference proxy short edge
  ([`78b09e7`](https://github.com/frost-byte/fbTools/commit/78b09e7ea47074a607f90ae27f35be38f3112d6a))

The dropdown offered only the editor's five sizes, but any multiple of 32 from 320 to 1088 works
  (608 gave good results). The node now lists them all, so there is nothing to guess.

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_016URszCL3deuwf7g3rSPoYu

- **source-profiles**: Offer the reference proxy short edge as a dropdown
  ([`c0ea093`](https://github.com/frost-byte/fbTools/commit/c0ea093cc967e3286b1a98c6a59fb612095b2bf5))

The free-form number gave no hint of valid values. It now lists the same sizes the Source Profile
  editor offers, plus "Use profile setting".

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_016URszCL3deuwf7g3rSPoYu

- **source-profiles**: Send a bundle's separate voice as its own audio reference in video modes
  ([`451637e`](https://github.com/frost-byte/fbTools/commit/451637e839bc45d748f4ccc0e2ae661a5d3a7f5f))

A bundle whose voice came from another video (extract_from_video) lost its audio when its visual
  mode was video or both: the voice was queued under a made-up subject id no slot owns, so the
  assembler ignored it. The voice now goes to the bundle's slot as a standalone audio reference, the
  same route as a dedicated audio file and the Composition path, in every visual mode and unpaired
  with any reference video.

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_016URszCL3deuwf7g3rSPoYu

### Chores

- **lint**: Remove unused variables and placeholder-less f-strings in prompt assembler
  ([`88ed709`](https://github.com/frost-byte/fbTools/commit/88ed709023a2106a0b92212908f6c7929cc210b1))

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_016URszCL3deuwf7g3rSPoYu

### Documentation

- **backlog**: Mark H3 verification done and bring the audio pipeline item up to date
  ([`610cfc8`](https://github.com/frost-byte/fbTools/commit/610cfc85f446757bc5e5dd4b3fe2722f49f877ce))

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_016URszCL3deuwf7g3rSPoYu

### Features

- **h3**: Add H3 Source Guides node to anchor Source Profile clip frames at their own times
  ([`62ab8a3`](https://github.com/frost-byte/fbTools/commit/62ab8a3ac55778a727d13ac12f7263b77287cfec))

Reads the <Video N> reference of a Source Profile clip prompt, picks one frame (or a short run)
  every N output frames and adds each with MiniMaxH3AddGuide, so a few frames carry the clip's
  progression at real timing instead of a time-compressed reference video. Wire it after Composition
  -> H3 Conditioning, before the guider.

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_016URszCL3deuwf7g3rSPoYu

- **source-profiles**: Override the reference every-nth and frame cap on the Clip Prompt node
  ([`9589ab8`](https://github.com/frost-byte/fbTools/commit/9589ab83e884218509130561c5d8cf2980aa9d0d))

Source Profile Clip Prompt gains Reference Every Nth and Reference Frame Cap inputs (0 = use the
  clip's own value) so the H3 reference video can be thinned per run without editing stored clips.
  The clip summary reports the stride, cap and resulting frame count, and the Source Profile editor
  shows each clip's stored reference sampling.

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_016URszCL3deuwf7g3rSPoYu

- **source-profiles**: Override the reference proxy short edge on the Clip Prompt node
  ([`21b222b`](https://github.com/frost-byte/fbTools/commit/21b222b2601058c11892d953279bec7307c9b8cb))

Reference Proxy Short Edge (0 = use the profile's setting) builds and caches a proxy at a different
  resolution for the run, leaving the profile setting and its existing proxies alone.

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_016URszCL3deuwf7g3rSPoYu


## v1.29.0 (2026-10-07)

### Bug Fixes

- **bundles**: Apply one audio-reference rule to every bundle audio source
  ([`f637107`](https://github.com/frost-byte/fbTools/commit/f637107b70df0bf6311435d535fa4afbacf1f414))

In Source Profile mode a bundle's audio reference is now included only when the cast entry's audio
  checkbox is on AND the clip allows dialogue, for file, extract_from_video and extract_from_visual
  alike (bundle_audio_wanted()). Previously file and extract_from_video ignored the checkbox.

The standalone "file" voice also honors audio.audio_dir, so a voice file in the output folder is
  found by both the Source Profile path and BundleAudioReferenceLoad
  (resolve_bundle_audio_source()).

Adds GOTCHAS entries for VHS's Mapping-typed AUDIO output and for the QwenTTS Basic voice-clone
  node's greedy decoding.

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>

- **bundles**: Log "no bundle selected" at info level
  ([`0e7c7bb`](https://github.com/frost-byte/fbTools/commit/0e7c7bb113a78bb041959ef13d00bcb7bc7438bc))

BundleAudioReferenceLoad's bundle input is optional, so leaving it unset is a normal choice (each
  reference falls back to its own clip's audio), not something to warn about.

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>

- **compose**: Respect image_selection and extract_from_video in cast enrichment
  ([`85ba45e`](https://github.com/frost-byte/fbTools/commit/85ba45edbd1e90beff61d4a334dcfdec70faeeb6))

_enrich_subject_with_bundle() (utils/prompt_compositions.py) was the one call site that never read a
  Cast entry's image_selection, always including every image in a bundle's visual.files regardless
  of what was picked in the Scene Cast Build tab — confirmed live via CompositionToH3's reference
  log showing 10 images instead of the expected 5.

Also fixes a related gap in the same function: extract_from_video (a separate video used purely as
  an audio/voice-timbre source, with no visual reference of its own) was never handled here, only
  source="file" was — so audio pulled from a standalone video silently never reached
  voice.audio_reference_file, even with no video reference present at all.

Both patterns were already correctly implemented in nodes/compositions.py's own separate
  subject-resolution path; this call site was simply never updated to match. 6 new regression tests
  added.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **compositions**: Fall back to plain text for {BG} when the background isn't a reference
  ([`392a2a2`](https://github.com/frost-byte/fbTools/commit/392a2a25ec78f48c07850ff129dea298a63f2370))

{BG} previously only worked when the background was used as a reference subject (checkbox on AND the
  background has reference images); in every other case a literal "{BG}" was left in the prompt. It
  now resolves against the effective background, after any Scene Cast Build override: a reference
  background still expands to <Subject N>, otherwise it becomes the background's description (else
  "the <name>"), and with no background at all, "the setting". Applies to shot action, shot camera
  and the synopsis.

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>

- **compositions**: Join background lighting with a comma, not a sentence break
  ([`bc3db64`](https://github.com/frost-byte/fbTools/commit/bc3db64e5fe4a301020fbd20c25177e833d4a792))

The background text sits mid-sentence in the subject definition and shot prompts, so an inner full
  stop broke it.

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>

- **compositions**: Require the audio checkbox for bundle audio on video entries
  ([`370fce6`](https://github.com/frost-byte/fbTools/commit/370fce68763a85c69364cd8e891f5077550b1d38))

In the Composition path a video-mode cast entry's bundle audio (extract_from_visual,
  extract_from_video, file) is now attached only when the entry's audio checkbox is on, via the same
  bundle_audio_wanted() rule Source Profile mode uses. Compositions have no per-clip
  allows_dialogue, so the checkbox is the only condition. Image-mode entries were already gated in
  _enrich_subject_with_bundle().

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>

- **scene-cast**: Name composition output folders after the composition
  ([`6070816`](https://github.com/frost-byte/fbTools/commit/60708166a673df17df6fd41cfc9b78349695e99f))

filename_prefix's composition-driven branch used the fixed literal "compositions" for every run, so
  every composition sharing a subject+bundle collided onto the same folder/stem, differentiated only
  by ComfyUI's own counter. Now slugifies the composition's own name (falling back to its id),
  matching how the sibling source_profiles branch already names its folder after the source profile.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

### Chores

- **templates**: Add scrubbed H3 composition and source-profile API templates
  ([`b876874`](https://github.com/frost-byte/fbTools/commit/b8768740765ebd84db0498e56be963ebfd9e5fef))

Widget values that held personal subject/bundle IDs, prompt text and composition/profile names are
  reset to the nodes' own defaults (cast_entries_json "[]", composition_overrides_json "{}", empty
  action_preview/clip_id, "(none)" for the loader names); graph structure is unchanged. Personal
  working copies live in templates/*.local.api.json, which is now gitignored.

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>

### Documentation

- **openshot**: Add design note for dropdown inputs whose options come from ComfyUI
  ([`91371c1`](https://github.com/frost-byte/fbTools/commit/91371c1b6d9baf6140a55bad41bec0823757e39d))

Proposes extending the `choice` extra-input type with `options_from` (read a combo's options from a
  workflow node via /object_info) instead of adding one input type per use case such as the fork's
  `bundle`. Records verified facts (ComfyUI re-runs define_schema per object_info request; OpenShot
  already has the combo-option extraction helper), open questions, and a staged plan whose first
  steps are generic and upstream-friendly. Linked from the multi-input templates plan.

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>

- **openshot**: Add OpenShot template pack and install guide
  ([`d8688f5`](https://github.com/frost-byte/fbTools/commit/d8688f5c27a16c59ba4ba6d88237a548c0f3b21d))

Adds openshot/templates (Scene Cast, Scene Cast - Source Profile, and the optional bundle-voice
  bridge) and openshot/README.md describing what each template does, what the ComfyUI server and
  OpenShot build need, and how to install them from OpenShot's user template folder
  (~/.openshot_qt/comfyui).

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>

- **openshot**: Use one bundle voice reference and keep clip A's end in the bridge template
  ([`bcbcb5b`](https://github.com/frost-byte/fbTools/commit/bcbcb5b7a32d97f18d66ca77e54e54e03ef4448a))

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>

- **openshot**: Wire the reference summary into the bundle-voice template
  ([`15cd52d`](https://github.com/frost-byte/fbTools/commit/15cd52d902219e55bf76fac2a37e15b20a1a7cc0))

The bundle-voice bridge template now includes an fbt_H3ReferenceSummary output node fed by the same
  sources as the H3 reference node (audio taken after the bundle switches), so each run logs the
  references the model is actually given. The install guide notes the new requirement and that
  ComfyUI must be restarted after updating fbTools.

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>

- **scene-cast**: Document when a bundle's audio reference is included
  ([`c82027c`](https://github.com/frost-byte/fbTools/commit/c82027cdaf2d077f4e89ecf20281412abf1f5a1e))

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>

### Features

- **backgrounds**: Let the user control VRAM after a "Remove People" run
  ([`d556295`](https://github.com/frost-byte/fbTools/commit/d556295b5f79b36f56b0857c6dd307260c8cc2c8))

The model stays resident after each H3 background-plate generation by default, since chaining
  several passes back-to-back (the existing auto-reselect flow) is the common case and reloading
  each time is slow. Adds a "Free VRAM" button in the Background editor and in Settings for
  on-demand reclaiming, plus a "Unload model after each run" Settings toggle to flip the default for
  anyone who'd rather reclaim memory automatically.

utils/h3_job_runner.py's new free_vram() POSTs to ComfyUI's own core /free endpoint -- the same
  mechanism behind the Manager UI's "Free model and node cache" button. Confirmed the underlying
  flag is processed near-instantly (PromptQueue.set_flag() notifies the same condition variable the
  main loop blocks on) rather than waiting on the next queued prompt, and verified live: the manual
  button actually freed VRAM, and a chained pass without it stayed resident (a ~131s cold-start run
  followed by a ~41s run needing no reload).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **bundles**: Add BundleAudioReferenceLoad node for external-editor voice references
  ([`ea119e0`](https://github.com/frost-byte/fbTools/commit/ea119e0672ecc4c3486d957a79927ec4cee6a01a))

Loads a Reference Bundle's own audio (file / extract_from_visual / extract_from_video, resolved by
  the new resolve_bundle_audio_source()) as a standalone AUDIO output, so hand-built templates such
  as OpenShot's Bridge Clips workflow can use a bundle's voice instead of the footage's audio.

- The bundle combo always offers "" so an unset OpenShot bundle picker doesn't fail ComfyUI's strict
  combo validation at submission. - A third INT output, switch_select, is 1 when the bundle audio
  loaded and 2 otherwise, for driving an ImpactSwitch with the footage audio as fallback
  (ComfyMathExpression can't derive this from a string). - The audio preprocess route now persists
  the selected audio_cache path on the bundle.

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>

- **compositions**: Add inspect_cast_metadata route for external editor pre-fill
  ([`fad2b13`](https://github.com/frost-byte/fbTools/commit/fad2b13f87ae76927ff65d766924181e45265e79))

Lets OpenShot (or any external client) resolve a clip's embedded ComfyUI "prompt" metadata, or the
  lighter fbtools_cast summary tag, back to the Composition/Subject/Bundle that produced it, reusing
  extract_cast_info() and load_composition() instead of duplicating that resolution logic
  externally. Split into resolve_cast_metadata_request() in utils/generation_metadata.py so it stays
  testable without nodes/compositions.py's heavy sibling-node import chain.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **compositions**: Generate H3 character/face sheets from Bundle references
  ([`0d54884`](https://github.com/frost-byte/fbTools/commit/0d54884e4a269974644c11c5530b4d711dc9b1aa))

Adds a Bundle-editor action that runs the H3 character/face-sheet workflow server-to-server against
  up to 9 of a bundle's own reference images and/or on-demand video-reference frames, mirroring the
  existing H3 Background Plate feature's proven pattern (title-based template contract,
  submit-and-poll job runner, Settings-driven overrides).

- nodes/h3_character_sheet.py: new routes to generate a sheet and report which optional template
  overrides are available. - utils/h3_template_runner.py: patch_character_sheet_prompt() pads fewer
  than 9 reference images by cycling them, since the workflow's fixed-index batch consumers crash
  outright on a smaller batch. - Bundle editor: unified, order-preserving pick list across images
  and video frames (the template's prompt treats the first pick as the sole outfit reference), an
  optional outfit-hint text substitution, full-size preview, and a one-click "Add to Bundle" for the
  result. - Settings: new "H3 Character Sheet" section mirroring the Background Plate one, including
  a real aspect-ratio dropdown instead of a free-text field.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **openshot**: Add prompt shorthand expander for H3 bridge template
  ([`6c67a48`](https://github.com/frost-byte/fbTools/commit/6c67a4877986a1600f1c28ff077f67f91225d93e))

Add fbt_PromptShorthandExpander, expanding S1/S2/V1/V2/A1/A2 shorthand into the MiniMax H3 Ref2VA
  native <Subject N>/<Video N>/<Audio N> labels, for use by the OpenShot-qt bridge-two-clips
  template's Prompt field. Confirmed working live against a real 5-subject prompt.

Also documents the shorthand convention and the openshot-qt extra_inputs design this template builds
  on.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **scene-cast**: Add per-run background override for Source Profile mode
  ([`d0e28ec`](https://github.com/frost-byte/fbTools/commit/d0e28ec6c064e9182c0493f1dcf27cbd19e17e9e))

SceneCastBuild gains a background_override_id input, driven by a new Background Override dropdown
  that shows while a Source Profile is connected. SourceProfileClipPrompt resolves the background as
  override > clip background_id > profile default_background_id, and "none" suppresses it for that
  run. Composition mode's composition_overrides_json is unchanged.

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>

- **templates**: Add Qwen 2.1 character sheet runner patch and API templates
  ([`326cf45`](https://github.com/frost-byte/fbTools/commit/326cf45b7a5cded0208d9b6145bda33a3cf6ecb7))

patch_qwen21_character_sheet_prompt() patches the unified restore / character / face Qwen-Image-2.1
  template (IN:refs, IN:mode, IN:seed, OUT:save plus optional overrides). Adds the
  qwen21_character_sheet, qwen21_image_blend and minimax_h3_r2v_bridge_clips API exports.

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>

- **tools**: Add Qwen-Image-2.1 photo restoration
  ([`e8f9aba`](https://github.com/frost-byte/fbTools/commit/e8f9abaeaac30abd1fa3fd1e06cfdbdae005ed74))

MiniMax H3's reference/image-to-video nodes always build an empty starting latent and only ever
  condition on reference images, making genuine img2img restoration of real photo content
  architecturally impossible (confirmed against comfy_extras/nodes_minimax_h3.py). Qwen-Image-2.1's
  dedicated edit-encode node (TextEncodeQwenImage21) genuinely VAE-encodes the source photo instead,
  so this ships restoration on that model.

- utils/h3_template_runner.py: patch_qwen21_photo_restore_prompt(), replacing the earlier H3-based
  patch_photo_restore_prompt (removed, unworkable) - nodes/qwen21_photo_restore.py:
  /fbtools/tools/restore_photo + restore_photo_settings_options routes, with a {{RESTORE_HINT}}
  prompt placeholder for optional per-run hints - new standalone "Tools" tab (js/ui/tools_panel.js)
  and matching "Qwen Photo Restore" Settings section, mirroring the existing H3 feature patterns -
  js/ui/lightbox.js: shared click-to-zoom image preview, wired into the new tab's source/result
  previews (existing CSS for this was previously unused) - templates/qwen21_photo_restore.api.json +
  README.md contract docs

Confirmed working end-to-end live: generation, hint/prompt override, zoom preview, and
  output-as-new-source chaining.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **video**: Add H3 Reference Summary node
  ([`a2ea173`](https://github.com/frost-byte/fbTools/commit/a2ea173f033af237d4172a4d3f623b1b329098f8))

Logs which references MiniMax H3 is given -- <Picture>/<Video>/<Audio> tag numbering, frames used
  after H3's 17k+5 length rule, durations, soundtrack pairing, and warnings -- and returns the same
  text as a STRING for an optional ShowText. Mirrors MiniMaxH3ReferenceToVideo's
  ref_images/ref_videos/ref_video_audios/ref_audios input groups, so the same sources can be wired
  into both, and is an output node so it logs even when its string output is unconnected.

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>

- **video**: Add MarkerFrameSplit and ImageTextOverlay nodes
  ([`dc390cc`](https://github.com/frost-byte/fbTools/commit/dc390cc20e7458a95362bbd35909f627f883a1d0))

MarkerFrameSplit locates a deliberately-inserted marker-color frame segment in a pre-concatenated
  video and splits it into the before/after clip halves - the first building block for a
  clip-bridging pipeline. ImageTextOverlay burns text onto an image batch, giving any
  scalar/string-output node a real, screenshot-able proof image without a third-party display-node
  dependency.

Adds example_workflows/marker_frame_split.json demonstrating both nodes plus
  SubjectLayerDefine/SubjectCompositor, verified live against a real ComfyUI instance. Documents a
  real comfy-mcp/comfy-cli workflow-conversion interop gotcha with dynamic-widget-group nodes, hit
  while generating this workflow's thumbnail.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>


## v1.28.0 (2026-09-27)

### Bug Fixes

- **audio**: Avoid module-name collision loading MelBandRoformer, surface fallback reason
  ([`5cfa755`](https://github.com/frost-byte/fbTools/commit/5cfa7558c30f1b1776efd830cb6c2928cde42847))

_find_melband_class() previously did sys.path.insert(mel_dir) then `import model.mel_band_roformer`
  — "model" is a common top-level package name across node packs (this machine also has one under
  comfyui_llm_party/model/), and Python caches imports by bare module name, so whichever pack claims
  "model" first during ComfyUI startup silently wins every later `import model...` for the rest of
  the process, regardless of sys.path order. Load the file directly via
  importlib.util.spec_from_file_location under a synthetic package name instead, sidestepping the
  collision entirely.

Also thread through *why* preprocessing fell back to spectral denoise instead of MelBand (no model
  configured, configured path not found, or class-not-found with the underlying exception) as
  denoise_method/ denoise_reason in the API response, and surface it as a distinct warning
  toast/status in the bundle editor instead of reporting success indistinguishably from a real
  MelBand run.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **bundles**: Default new bundle visual force_rate to 24fps
  ([`a9aa319`](https://github.com/frost-byte/fbTools/commit/a9aa319d3128931cd8dddcfbe5c53e8afd2d3bed))

Matches every other H3 reference-loading path — a freshly-created bundle's default visual params
  still defaulted to native fps client-side.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **bundles**: Default reference video force_rate to 24fps
  ([`4152683`](https://github.com/frost-byte/fbTools/commit/4152683ab14ada2a2c81d9604767d35049a7992c))

BundleRegistry's visual defaults left force_rate at 0 (native fps) for new bundles, inconsistent
  with every other H3 reference-loading path in this codebase, which requires 24fps.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **canvas**: Patch graph.serialize() so Get/Set node ordering actually persists
  ([`8a0a805`](https://github.com/frost-byte/fbTools/commit/8a0a80510bebd2ed42109490f26839c93a2c585a))

The live graph._nodes reorder from 277afe8 only fixes rendering for the current session -- it never
  survives a save. Traced graph.serialize() (a long-standing public litegraph method, also aliased
  as toJSON(), confirmed via the bundled frontend source to produce the exact {id, revision, nodes,
  links, groups, config, extra, version, ...} shape written to a saved workflow file) and found it
  rebuilds its own `nodes` array every call from a separate internal node registry that tracks pure
  insertion order -- not from graph._nodes -- with no public API to reorder that registry directly.
  So no amount of live array mutation can reliably survive a save.

Patches graph.serialize() itself (via the shared prototype, so it covers any graph instance
  including subgraphs) to reorder its *output* nodes array, moving every GetNode/SetNode to the
  front. This is the one point guaranteed to run on every save (Ctrl+S, the Save menu, autosave)
  regardless of what triggered it or what state the live canvas happens to be in, making the fix
  actually persistent rather than a rendering-only convenience.

sendGetSetNodesToBack (the selection-toolbox button) is kept as-is for the live/visual half of the
  fix during editing; this patch is the independent, automatic, unconditional guarantee that
  whatever gets saved comes out right regardless of whether that button was ever clicked.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **canvas**: Reorder graph._nodes directly instead of using canvas.sendToBack
  ([`277afe8`](https://github.com/frost-byte/fbTools/commit/277afe8d7555b1256c528fd40e7bc2a6760d8390))

canvas.sendToBack()/bringToFront() in the current ComfyUI frontend drive a separate, session-only
  render z-index (allocateZIndex/setNodeZIndex, dispatched as a "layout mutation") that never gets
  written into the saved workflow JSON -- confirmed by checking a real workflow file for a zIndex
  field on any node and finding none. That made the previous implementation of this command a
  visual-only fix: it would look right until the workflow was saved and reloaded, at which point it
  would revert, since nothing persists the change.

What actually controls draw order AND is what gets serialized is the node's position in graph._nodes
  itself (this is what the manual JSON edit earlier in this session actually changed, and it's
  confirmed durable since it's literally the file). Rewritten to move every GetNode/SetNode to the
  front of graph._nodes directly, then call graph.change() (confirmed via the frontend bundle to
  mark the canvas dirty and fire the change hook ComfyUI uses to track unsaved changes) -- this is a
  real, persistent fix, not a session-only convenience.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **canvas**: Surface "Send Get/Set Nodes to Back" as a selection-toolbox button
  ([`257e798`](https://github.com/frost-byte/fbTools/commit/257e798a66b3cd6bffa2d43f9c3a513c17af3b99))

The command added in 7396736 was only reachable via the command palette (Ctrl+K) -- no visible
  button anywhere. This codebase already has a selection-toolbox surface (a floating button bar
  shown when something is selected on canvas) that the sibling "Extract Node as JSON" command uses;
  add this command's id there too so it actually appears as a clickable button, matching the
  existing pattern exactly (unconditional on any selection, same as the sibling command).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **canvas**: Use canvas.sendToBack() for the live half again, not a raw array splice
  ([`99eda67`](https://github.com/frost-byte/fbTools/commit/99eda67484ac8b1643f6a545a378640381d73894))

User confirmed the graph.serialize() patch (8a0a805) genuinely fixes persistence -- it works even
  when nodes are moved to the back manually, independent of this button entirely, since it
  unconditionally reorders serialize()'s output regardless of live state. But the button itself was
  still reporting a node count without reliably repainting: 277afe8's raw graph._nodes splice
  apparently doesn't reliably drive the current renderer (most likely a stale render-order cache
  that a plain array reassignment never invalidates), whereas the user confirmed manually sending
  nodes to back *does* visually work -- i.e. canvas.sendToBack() (litegraph's own mechanism) was the
  right call for the live view all along; 277afe8's move away from it was the actual regression.

Reverts the live half to loop canvas.sendToBack(node) per target, same as the original
  implementation before 277afe8. Persistence no longer depends on this at all now that serialize()
  is patched independently, so this only needs to make the current session's view look right.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **cast**: Discard stale responses when the connected composition/profile changes quickly
  ([`825a5b0`](https://github.com/frost-byte/fbTools/commit/825a5b01dbb07b145a474c62ba955270fb0db4a1))

_refreshCompositionSubjects() and _refreshSourceSubjects() had no guard against overlapping
  refreshes: switching the dropdown again before an in-flight fetch resolved could let an older
  response land after a newer one and silently overwrite it, showing the previous selection's shots
  in the preview. Guarded with a per-refresh sequence number.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **cast**: Drop the invented 'Camera:' label from the composition shot preview
  ([`ac35e2a`](https://github.com/frost-byte/fbTools/commit/ac35e2a5c4e686c8dd8bdc2020fb9ec563ea5590))

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **cast**: Hide the per-entry Dialogue field in composition mode; add SceneCastBuild dev notes
  ([`d30b5b1`](https://github.com/frost-byte/fbTools/commit/d30b5b18d73fa915c081bf56de801515c53ce5ed))

Cast-entry dialogue is only read by SourceProfileClipPrompt; PromptCompositionLoader ignores it and
  composition shots carry their own dialogue, so the field did nothing (and did not override the
  shot's dialogue) with a composition connected. Hide it in that mode and document the decision, the
  two-mode behaviour, ordinal matching and the libber notes in docs/scene_cast_build.md.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **cast**: Put the Background dropdown and both checkboxes on one row
  ([`a342926`](https://github.com/frost-byte/fbTools/commit/a342926b74f8f53c94df73f635bcfba8e1be734a))

The background select stretched full-width on its own row; it's now capped at 150px, sharing a row
  with the image/soundscape checkboxes, which wrap if the node is narrow.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **compose**: Keep the saved-compositions search box in sync after a panel rebuild
  ([`e84d60f`](https://github.com/frost-byte/fbTools/commit/e84d60f2938237e902cf2175f6713a892bf9cb57))

Closing and reopening the fbTools panel rebuilds the Compose list view's DOM from scratch, but its
  filter text lives in module state, which survives that rebuild. The search input was recreated
  empty regardless, so the list stayed filtered while the box that explains why looked blank. Seed
  the rebuilt input's value from the surviving filter.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **compose**: Stop shot ids from colliding across a composition's own shots
  ([`1860944`](https://github.com/frost-byte/fbTools/commit/18609443e53e84b6a95352c211d5541c4178e733))

Shot ids came from a session-global counter that never re-synced to a composition's own shots when
  it was loaded, so the first '+ Add Shot' after opening a composition could mint an id already used
  by one of its existing shots. Confirmed in real data: 'oversize' had two shots both 'shot_1',
  which broke Scene Cast Build's shot-id-keyed timeline lookup (switching between them always showed
  the same, last, shot's action text).

- js/ui/composition_editor.js: shot ids now derive from the composition's own shot list
  (_nextShotId) instead of a counter; removes the whole bug class. -
  scripts/fix_duplicate_shot_ids.py: renumbers an affected composition's shots to shot_1..shot_N;
  found and fixed 8 real files (oversize among them), each backed up to <name>.json.pre-shot-id-fix.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **composition**: Build the H3 ref plan from the assembled scene so background/outfit reference
  images are included
  ([`8fac6bd`](https://github.com/frost-byte/fbTools/commit/8fac6bdd80e40326a5b4d017f79a14335854232b))

The loader built the ref plan from the raw resolved subjects, so slots minted during assembly
  (background reference, outfit references, bundle replacements) never reached the conditioning node
  even though the prompt referenced them. assemble_composition now returns its scene_instance and
  the loader uses it.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **composition**: Default video_force_rate to 24fps in PromptCompositionLoader
  ([`292a6bc`](https://github.com/frost-byte/fbTools/commit/292a6bc25f165f78e5ebbcd48cdf9feee4feba15))

Matches every other H3 reference-loading path in this codebase.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **composition**: Make a background reference actually establish the setting
  ([`a7c2022`](https://github.com/frost-byte/fbTools/commit/a7c20226e1a6f3ddfaa015f1f837741529ce59a8))

With background-as-reference, the prompt never placed the scene in the background subject and, in
  video-editing mode, told the model to preserve the source video's setting and lighting, so the
  render kept the original setting. Now: the first shot is opened with 'The scene takes place in
  {BG}' when no shot names it, the editing retention line says the source setting is replaced by the
  background subject, the background slot is a location (no 'their'), the person-style picture
  retention line is skipped for it, and the 'Set in' sentence no longer ends with '..'.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **composition**: Match cast entries by subject identity, restore retention markers
  ([`9e90911`](https://github.com/frost-byte/fbTools/commit/9e909115dd8954f2a674dfbc1f60efa91a5c51a9))

PromptCompositionLoader's apply_cast_to_subjects() matched SceneCastBuild cast entries to
  composition slots by array position against a lexicographically sorted slot-key list, unlike
  SourceProfileClipPrompt which matches by explicit subject_id. This broke past 9 slots and made
  bindings order-dependent and opaque, per user report of no longer being able to tell how subjects
  mapped between the two nodes.

Also, hybrid cast entries (a bundle replacing a subject that also has a source-video reference)
  never inspected bundle_id at all, silently dropping the bundle's appearance/images/audio from the
  prompt text even though media resolution elsewhere still wired it correctly — refplan and prompt
  text ended up describing different subjects for the same entry.

Rewrite apply_cast_to_subjects to match by subject_id identity, mirroring SourceProfileClipPrompt's
  SOURCE_SLOTS/BUNDLE_SLOTS pairing: a hybrid entry now keeps the source-derived subject as the
  motion donor (marked "replaced") and mints a new "<slot>_bundle" slot for the bundle (marked
  "attribute_transfer"), each pointing at the other via _transfer_to_slot — the same retention
  markers SourceProfileClipPrompt sets, which prompt_assembler.py gates several
  video-continuation-workflow features on (motion-transfer sentence, sharpened discard-identity
  wording, bundle-first <Subject N> ordering) that previously never fired for compositions.

assemble_composition() needed two related fixes to actually surface this: - mint a template letter
  for the new "<slot>_bundle" key, or it silently never appears in the assembled prompt
  (resolved_subjects keys not in the composition's own subject roster were dropped). - remap
  "_transfer_to_slot" values through slot_map: apply_cast_to_subjects stamps composition-level slot
  keys (e.g. "S1_bundle"), but ref_map and slot_assignments are keyed by template letters (e.g.
  "B"), so the donor/replacement pair couldn't find each other without translation.

Updated tests/test_cast_enrichment.py: 4 existing tests encoded the old positional "recast to an
  unrelated subject via array index" behavior, which has no ID-based equivalent (by definition
  there's no identity to match against) and no analogue in SourceProfileClipPrompt. Replaced with
  tests for order-independent matching, unmatched entries being no-ops, and the hybrid
  donor/replacement pairing. Removed an unused _SubjectRegistry test helper.

Added tests/test_prompt_assembler.py::TestHybridCastRetentionInAssembledPrompt as an end-to-end
  check (per the approved plan's verification section) that a hybrid cast entry's retention markers
  actually reach the assembled prompt text — motion-transfer sentence and the bundle's own
  appearance/images — not just the intermediate dict shape.

include_original_subject_tags stays out of scope: compositions have no field for it at all, so
  wiring it in is a schema addition for the deferred SceneCastBuild redesign, not this bug fix.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **composition**: Migration script keeps its own backup beside the editor's .bak
  ([`f4e08c8`](https://github.com/frost-byte/fbTools/commit/f4e08c8a2ba32f83d37b948f96264d54a872c50c))

The script skipped its backup whenever <name>.json.bak existed, but the composition editor already
  writes that file on every save (50 of 70 real compositions had one), so the latest pre-migration
  content was not preserved for most files. Back up to <name>.json.pre-slot-migration instead.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **composition**: Soften audio-reference wording for subjects with no scripted dialogue
  ([`8ad65a8`](https://github.com/frost-byte/fbTools/commit/8ad65a881679dad7f89ae7efd3a1a47f776e7135))

A subject with a voice/audio reference but no dialogue line anywhere in the composition got the same
  'a spoken ... vocal layer' / 'voice timbre and measured delivery' wording as a subject who
  actually speaks, which reads as speech-cadence guidance and could plausibly drive H3 to invent
  dialogue for a subject only meant to keep a consistent voice. Auto-detected per slot from whether
  it's ever a resolved dialogue speaker; drops the speech-implying phrasing while keeping 'without
  copying the original signal' (unrelated to speech). A subject with a scripted line, or an explicit
  audio role override, is unaffected.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **conditioning**: Define the audio-reference list used by the Turbo LoRA warning
  ([`fce68eb`](https://github.com/frost-byte/fbTools/commit/fce68eb77d6857819368ca091a43bbe867a7de6e))

CompositionToH3Conditioning referenced an undefined standalone_audio_refs, raising NameError
  whenever the plan flagged a Turbo LoRA. The warning is about any audio reference (soundtrack or
  standalone), so build the list from both modalities.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **h3**: Duration-to-frame-cap conversion ignored select_every_nth
  ([`c77df2d`](https://github.com/frost-byte/fbTools/commit/c77df2d28f1c90507d3cd824e6166f6643e3d4cb))

_h3_load_video_frames() converted a bundle/clip's `duration` into a frame_load_cap as int(duration *
  target_fps), never dividing by select_every_nth — but frame_load_cap is compared against
  `sampled`, a count taken *after* the select_every_nth filter, so the cap was silently expressed in
  the wrong units. With select_every_nth=2 this read exactly 2x the requested duration's worth of
  source footage before the loader stopped (reported live: 4.7s at fps=24, select_every_nth=2
  produced 112 frames instead of the correct 56, then got ping-pong padded to 124 for not being a
  valid H3 17k+5 count — 56 already is one, so no padding should have fired at all).
  select_every_nth was already being applied correctly to *which* frames survive the filter (line
  17435); only this duration->cap arithmetic missed it.

Fix mirrors the already-correct sibling formula in the /fbtools/bundles/preview_sampled route
  (extension.py:15280): int(duration * target_fps / select_every_nth).

Tests: tests/test_h3_load_video_frames_duration_cap.py (3, AST source-contract style — extension.py
  can't be imported directly in tests, same constraint as test_dataset_caption_api.py). Full suite:
  1268 passed/17 skipped.

Needs a ComfyUI restart to take effect (Python change).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **llm-client**: Robust think-tag stripping, vision image resize, MiniMaxH3 signature
  ([`8d14e1c`](https://github.com/frost-byte/fbTools/commit/8d14e1c37eb7e84e88904f4642a8eb2a143ccaa8))

- Strip thinking blocks unconditionally in both GGUF and HF paths; handle three formats: complete
  <think>…</think> pairs, unclosed <think>, and orphaned </think> (Qwen3 emits thinking as plain
  text ending with </think> when template detection misses) - Downscale vision images exceeding
  1280px before encoding to prevent VRAM exhaustion from large character sheets; returns resized
  flag so frontend can toast the user - Fix bun_subj UnboundLocalError: lookup was placed after its
  first use; moved to immediately after bundle is resolved so Python does not treat it as unbound
  local - Update MiniMaxH3ReferenceToVideo.execute() call to match v0.35.0 signature where vae and
  audio_vae moved from positional to keyword arguments

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **modal**: Add missing modal_vram_profiler.py module
  ([`04aeae8`](https://github.com/frost-byte/fbTools/commit/04aeae845f4b24da9f99617811920c451c995d25))

extension.py has imported this module unconditionally (no try/except) since 65ef76d, but the file
  itself was never committed — every clone of this repo since then has had a hard ImportError on
  startup unless the working tree happened to still have the untracked file locally.

Estimates peak VRAM for a model + configuration (weights + vision + KV + activations) and recommends
  the smallest-sufficient Modal GPU from the available set, with a measured-peak feedback-loop
  cache. Pure stdlib + optional huggingface_hub, no ComfyUI dependencies. Verified working via the
  project's import_test_module() harness against its bundled model presets.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **nodes**: Correct io.NodeOutput argument order and mask/list type mismatches in image-processing
  nodes
  ([`c9e6d77`](https://github.com/frost-byte/fbTools/commit/c9e6d770c7bb3435d217ea32fc5acf51f432d9f4))

io.NodeOutput(*args) takes one positional value per declared output, not a dict — SAMPreprocessNHWC,
  TailEnhancePro, TailSplit, OpaqueAlpha, and SubdirLister were all passing a single dict, silently
  corrupting every output past the first. Also fixes OpaqueAlpha's mask output being declared
  io.Image.Output instead of io.Mask.Output, and TailEnhancePro (plus its utils/images.py helpers)
  treating a batched IMAGE tensor as if it were a real Python list. All three bugs were found by
  actually running the affected nodes rather than reading the code, documented in GOTCHAS.md so they
  don't recur elsewhere.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **nodes**: Repoint relative imports inside moved route handlers
  ([`2b1d30f`](https://github.com/frost-byte/fbTools/commit/2b1d30f486bff386473c0ae8d9422ecf89177043))

Lazy imports inside handlers moved from extension.py kept a single leading dot, so they resolved
  against nodes/ (outfits sam2_status failed with 'No module named ...nodes.utils'; captioner
  imports in the LLM routes would have failed the same way). A new test checks that every relative
  import in nodes/*.py, at any depth, resolves.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **outfit**: Resolve the SAM2 extraction source in input or output subfolders
  ([`5a848c8`](https://github.com/frost-byte/fbTools/commit/5a848c805e9384c95900c76c1a2b800c3820d8ad))

The extract endpoint only looked for the basename in the input dir, so images picked from output/ or
  a subfolder reported 'File not found'. The UI now sends the folder, the server resolves the
  relative path inside it (with a path traversal guard) and always writes the result to the input
  dir.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **prompt-assembler**: Sharpen attribute-transfer wording and combine picture entries
  ([`594ba54`](https://github.com/frost-byte/fbTools/commit/594ba541e213b3d4a383245fd13719d2c731cd79))

Addresses two observed generation failures documented in
  docs/h3_attribute_transfer_assembler_plan.md: the character swap not taking effect from the start
  of the clip, and the original subject's costume/outfit persisting instead of the replacement's.

- "discard visual identity, hair and wardrobe" -> "discard visual identity including head, face,
  body, hair and wardrobe" — more explicit about what the model must actually replace. - "original,
  in <Video N>" -> "originally in <Video N>" — grammar fix. - Picture entries for the same subject's
  multiple reference images now combine into one retention_analysis line ("<Picture N> and <Picture
  M>: fully_preserved - ...") via the existing _join_labels() helper, instead of one redundant line
  per image. - The bundle-replacement edit-description sentence now names the specific source
  video(s) being recreated when resolvable via transfer_to_slot, falling back to the generic
  "photorealistic, seamless identity-replacement edit" wording otherwise.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **run-tracking**: Stop duplicate Run History entries for genuinely-executed tracked nodes
  ([`7131977`](https://github.com/frost-byte/fbTools/commit/7131977758195d57700c3ca2c80d6b428d526f29))

_fbtools_capture_for_node captures a [track: Label]-tagged node's inputs whenever it genuinely
  executes (cache miss), then _fbtools_backfill_cached_tracked_nodes runs on every subsequent
  execute() call to re-emit any tracked node that was pruned from the schedule as a real cache hit.
  The intent was for the "already emitted this prompt_id" marker to be set right after a genuine
  capture so the backfill pass never re-processes it — but the marker line was accidentally nested
  inside the _TRACKED_KWARGS_BY_NODE_ID eviction loop's while body, so it only ran when the kwargs
  store exceeded 100 entries.

In practice this meant a node's own capture never marked itself emitted, so the very next tracked
  node's execute() call — which re-scans every tracked node's cache status, not just its own — found
  this node's output now sitting in caches.outputs (having finished executing a moment earlier) and
  re-emitted the same stashed values a second time, producing duplicate "(extra)" entries in Run
  History for every node that genuinely ran, once per node's own capture and once via the next
  node's backfill sweep.

Confirmed via `git show f14106d^:extension.py` that this bug predates the nodes/ package split —
  it's a pre-existing latent bug, not a refactor regression, just surfaced now by the user's
  restructuring smoke-test.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **scene-cast**: Default force_rate to 24fps in _resolve_cast_media
  ([`2a25616`](https://github.com/frost-byte/fbTools/commit/2a256168758a04152040df94ee382bc5fa30096b))

Matches every other H3 reference-loading path in this codebase (see utils/reference_bundles.py) —
  these three fallback/default video_params dicts were still defaulting to native fps.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **scene-cast**: Include the primary subject's bundle in filename_prefix
  ([`17a3b42`](https://github.com/frost-byte/fbTools/commit/17a3b4269db0d97721cd08bd638d9ee3cbbbfc9c))

build_cast_filename_prefix() combined only the primary subject id and the composition/source-profile
  kind segment, silently dropping the bundle id in between — e.g. video/alex/team_fort/ instead of
  video/alex/alex_salon_eyes/team_fort/. Added resolve_primary_bundle() alongside the existing
  resolve_primary_subject() and threaded the bundle id through. Also renamed the composition-kind
  folder segment from the abbreviated "comps" to "compositions" to match.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **scene-cast**: Resolve ordinal subject matches against the live profile
  ([`72ffe82`](https://github.com/frost-byte/fbTools/commit/72ffe82c6e5faa47187978f0cd4149dcd38a9c21))

Ordinal cast entries (match to "the Nth male/female subject in this clip") stored
  source_subject_id/source_profile_id from whichever Source Profile was connected when the match was
  first made. Swapping the upstream Source Profile for a different one left that source_profile_id
  stale, so the entry kept resolving against a profile that was no longer connected instead of
  re-matching against the live one.

Fix: for ordinal entries, always re-resolve source_subject_id fresh against the currently-connected
  profile/clip on every execute() — there is only ever one source_profile input, so there is nothing
  to disambiguate by caching the old id. A no-match clears source linkage entirely rather than
  silently keeping a stale reference. Also add a diagnostic warning when two or more cast entries
  resolve to the same source subject (an ordinal/explicit assignment conflict), since only one
  bundle can actually replace a given subject.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **scene-cast**: Route status updates through the node's own instance id
  ([`88650f4`](https://github.com/frost-byte/fbTools/commit/88650f40cded2af969dbca920277c75e08c4a9e9))

SceneCastBuild.execute() sent its "Inline cast: N entries" status update tagged with cls.node_id —
  the class-level prefixed node type (e.g. "fbt_SceneCastBuild"), shared by every instance of the
  node. With more than one SceneCastBuild in a graph, status updates from one instance would appear
  to come from all of them. Use cls.hidden.unique_id instead, matching the per-instance pattern
  already used elsewhere in this file.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **source-profiles**: Bump reload counter on all mutating endpoints
  ([`1438a14`](https://github.com/frost-byte/fbTools/commit/1438a1418f245d07d59b2bfae9cb5af83c9d3b03))

auto_partition, set_clips, merge_subjects, upsert_clip, and remove_clip saved the registry but never
  bumped _source_profile_reload_counter, which SourceProfileClipPrompt's fingerprint_inputs() relies
  on to invalidate ComfyUI's execution cache. Edits made through these endpoints could leave a stale
  cached result in place until an unrelated change happened to bump the counter.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **source-profiles**: Honor allows_dialogue across all bundle audio paths
  ([`bf5569f`](https://github.com/frost-byte/fbTools/commit/bf5569fefb64284978005bf58ebb237f233eaf1e))

A clip's allows_dialogue=False already suppressed spoken dialogue text, but a replacement bundle's
  own audio could still leak through via any of extract_from_visual/use_audio (bundle video entry),
  extract_from_video (separate audio file), file (standalone voice reference), or use_audio
  (motion-donor clip extraction) — a clip marked "no audio involvement" would still pull audio in
  through these paths regardless of that setting.

Gate all four on the same clip.get("allows_dialogue", True) check so the clip-level setting is
  actually authoritative over every source of audio for that shot, not just dialogue text.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **source-profiles**: Honor include_original_subject_tags in auto-synopsis
  ([`552e48c`](https://github.com/frost-byte/fbTools/commit/552e48ca8936a9e122b1c06d96d11d65484fd942))

The auto-generated replacement synopsis always described a replaced source subject by its literal
  name ("{b} takes the place of Alice"), even when include_original_subject_tags was set — the flag
  already controlled whether a replaced source subject gets a <Subject N> label elsewhere in the
  assembler (see _pre_subject_nums in prompt_assembler.py), but the synopsis text never branched on
  it.

Also default force_rate to 24fps for this clip's own load_params, matching every other H3
  reference-loading path.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **source-profiles**: Stop clip nav arrows from scrolling the panel to top
  ([`bdddaa6`](https://github.com/frost-byte/fbTools/commit/bdddaa61414ef618adf42bf91b4752d288391d90))

prevBtn/nextBtn's own onclick handler calls redraw(), which rebuilds navEl (innerHTML = "") —
  destroying the very button that still held focus from the click. With focus yanked out from under
  it, it reverts to <body>, and the panel host's focus-tracking scrolls the whole view back to the
  top. Blur the button before triggering the rebuild so focus is released deliberately instead of
  recovered by the browser.

Clicking a clip directly on the timeline never hit this, since the <canvas> element isn't focusable
  by default — nothing gets destroyed out from under a focused element there.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **subjects**: Expose pronoun_style in /fbtools/subjects/list
  ([`55364ab`](https://github.com/frost-byte/fbTools/commit/55364ab86e8b5f684d04523dd34ef3b296c5f1c9))

Required by the ordinal-match frontend (scene_cast_build.js mirrors
  resolve_ordinal_subject()/resolved_pronoun_style() client-side to preview a match before the
  backend runs) — the endpoint previously omitted pronoun_style entirely, so the client-side mirror
  had nothing to resolve against. Also default force_rate to 24fps in _bundles_preview_sampled,
  matching every other H3 reference-loading path.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Confirm Node Inspector destination and add tooltips to toolbox buttons
  ([`46cb7ac`](https://github.com/frost-byte/fbTools/commit/46cb7ac3a1f1e9ad8e519054576892791976ab93))

Extract Node as JSON activated the sidebar's Node Inspector tab silently -- easy to miss if the
  sidebar was closed or on a different tab. Add a toast naming the destination (also copied to
  clipboard, as before).

Neither selection-toolbox icon button (Extract Node as JSON, Send Get/Set Nodes to Back) showed a
  tooltip on hover. Traced live: ComfyUI's own button component resolves its title/tooltip from an
  i18n key (commands.<id>.label) rather than the command's own label field, and we don't ship a
  locale file registering that key, so both aria-label and the tooltip silently resolved to "". Set
  a native title attribute lazily on first hover instead of fighting ComfyUI's i18n plumbing -- a
  delegated pointerover listener, real work only runs once per button.

Also removes the pre-sidebar JSONViewer bottom-panel implementation
  (displayNodesInTab/collapseAll/expandAll/initTab/clearTab/fixPropertyColors, the
  applyPrimeTextVars color-remap helper, and their unused constants/toast objects), confirmed fully
  superseded by js/ui/node_inspector.js and dead via repo-wide grep -- nothing outside this dead
  cluster referenced any of it. Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **ui**: Make the fbTools sidebar panel genuinely scrollable
  ([`d1b0ae4`](https://github.com/frost-byte/fbTools/commit/d1b0ae4bbe9e5ff609b43fdc9def9f3b10d5f4e8))

ComfyUI mounts sidebar tab content inside a plain display:block wrapper with no definite height of
  its own -- only its own ancestor (.sidebar-content-container) has one, from the splitter layout. A
  block parent doesn't pass that height down to a percentage- or flex-sized child, so .fbt-panel's
  height:100%/flex:1 had nothing to resolve against and silently grew to fit its content instead of
  being clamped. Any flex:1;overflow:auto region further down (e.g. Node Inspector's JSON tree)
  never got real internal overflow because of this -- confirmed live, scrollHeight === clientHeight.

That in turn broke mousewheel scrolling wherever overscroll-behavior:contain is set: with zero local
  scrollable range, every wheel attempt is immediately "at the boundary", so contain blocks it from
  chaining to the ancestor that actually scrolls. Dragging that ancestor's own scrollbar directly
  still worked, which is why only mousewheel looked broken.

Fix: escape the unconstrained wrapper via position:absolute sized against the nearest positioned
  ancestor's padding box (.sidebar-content-container, scoped with :has() so no other sidebar tab is
  affected), which resolves correctly regardless of intervening display modes. Verified live: Node
  Inspector's content area now reports real overflow (scrollHeight 6324 vs clientHeight 558) with
  its own native scrollbar, and Compose/Assets tabs render unaffected. Co-Authored-By: Claude Sonnet
  5 <noreply@anthropic.com>

- **ui**: Register selection-toolbox tooltips via ComfyUI's own i18n locale mechanism
  ([`258ff60`](https://github.com/frost-byte/fbTools/commit/258ff6034fd92f74805cbbb882cfaf72fc6e614c))

The previous fix (native title attribute) worked but looked visually inconsistent with every other
  selection-toolbox button, which get a properly styled tooltip via ComfyUI's own v-tooltip
  directive. Traced why: ExtensionCommandButton (the component rendering our two commands) resolves
  its tooltip from an i18n key -- commands.<id-with-dots-as-underscores>.label -- not from the
  command's own label/tooltip field. Confirmed the exact key transform live against ComfyUI's own
  118 built-in command translations (e.g. id "Comfy.3DViewer.Open3DViewer" -> key
  "Comfy_3DViewer_Open3DViewer").

ComfyUI has a real, documented convention for this: a custom node ships locales/<lang>/commands.json
  (see app/custom_node_manager.py's build_translations(), served at GET /i18n and merged into the
  frontend's message catalog on load). Add ours with the two correctly-transformed keys, and add
  'locales' to [tool.comfy] includes in pyproject.toml so it also ships in registry-published
  builds, matching how 'js' is already listed.

This supersedes and removes the native-title-attribute workaround from the previous commit, now that
  the real mechanism is in place.

Requires a ComfyUI *backend* restart, not just a browser refresh -- build_translations() is
  @lru_cache'd for the process lifetime, confirmed live via GET /i18n returning an empty commands
  catalog even after this file was created. Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **ui**: Restore jsnview's toggle icon position lost to ComfyUI's Tailwind purge
  ([`5f35813`](https://github.com/frost-byte/fbTools/commit/5f35813a81b1a19eaf8d1f7cba6372f91b654fe3))

ComfyUI's bundled frontend CSS is real Tailwind output, purged to only the utilities ComfyUI's own
  UI uses. It happens to include .absolute/.relative/ .top-1/.pl-7 (needed elsewhere in its own
  components) but not the unscoped .-left-4 jsnview's toggle relies on, so the toggle got
  position:absolute and top set but no left at all -- effectively unplaced instead of sitting in the
  row's left gutter, making it look like there was no expand/collapse icon.

Set position/left/top explicitly instead of depending on what Comfy's build happens to keep.
  Documented in GOTCHAS.md since the same purge gap could hit any other Tailwind-based utility a
  future CDN library assumes is available. Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **ui**: Restore per-node toggle and show placeholder for blank values in Node Inspector
  ([`7770f2d`](https://github.com/frost-byte/fbTools/commit/7770f2d36fa969bb10d6a1a4bca5d2c0a714230e))

jsnview's own delegated toggle-click listener stops firing reliably once its tree lives inside the
  sidebar tab, so individual expand/collapse silently broke when JSONViewer moved off the bottom
  panel (expand/collapse all still worked since those don't depend on that listener). Add our own
  capturing, propagation-stopping click handler that does the same toggle directly.

Also, jsnview's applyValueStyles() has no case for undefined (or other
  non-string/number/bigint/boolean/null values), so those leaves render as literal blank space with
  no class or text at all. Post-process the rendered tree to give them the same italic/muted
  treatment null already gets. Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

### Code Style

- **libbers**: Narrow key column to 150px, give value column remaining width
  ([`2f4955f`](https://github.com/frost-byte/fbTools/commit/2f4955f1819e92f6d8967fb476e42b875c746ddd))

table-layout: fixed with explicit first-column width stops the key input from claiming half the
  table; value textarea now gets the majority of space. Also aligns the add-row key input to the
  same 150px width.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

### Documentation

- Add GOTCHAS entry for graph.serialize() node-order persistence
  ([`ca28ce1`](https://github.com/frost-byte/fbTools/commit/ca28ce1d58267b70601e8f13a4e60c7b1443d2ed))

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- Add H3 attention mechanisms reference, tracker plan, and session logs
  ([`119ee38`](https://github.com/frost-byte/fbTools/commit/119ee380954d3c185c116e6c9db94d62b9632c33))

- h3_attention_mechanisms_reference.md: compatibility matrix for the MiniMax H3 VRAM/attention
  optimization nodes (Chunk FeedForward, Low VRAM Attention, Sage Attention variants, Model Sparse
  Attention), traced from actual patch mechanisms rather than node descriptions. -
  comfyui-node-output-tracker-plan.md: design notes for the node-output auto-tracker feature. -
  sessions/: dated session logs plus index.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- Add H3 VRAM estimator module, GOTCHAS log, and RefMod integration design
  ([`0513d8c`](https://github.com/frost-byte/fbTools/commit/0513d8cc41342726490bbaeccbd74882f34bf324))

Adds utils/h3_vram_estimator.py (pure token/attention-memory heuristics calibrated against a real
  second-pass OOM incident) with unit tests, a new docs/GOTCHAS.md tracking recurring non-obvious
  patterns (the CUDA "device limit" overhead gap this module's BASE_OVERHEAD_GIB is calibrated
  against), and docs/h3_refmod_integration_design.md scoping a staged, natively implemented
  RefMod-style reference-caching feature for Bundles and Source Profile clips.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- Add manual test checklist for post-refactor verification
  ([`59c508f`](https://github.com/frost-byte/fbTools/commit/59c508fc6982607e98ee47a5e02c1537514d4191))

Covers the Node Inspector UX fixes, frontend i18n/tooltip system, the full extension.py -> nodes/
  package-split refactor (Plans 1-29), and Phase 1 of the background-as-Subject-N reference system,
  since none of these are exercised by the automated test suite (live ComfyUI, browser, and REST
  behavior).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- Add reference notes on ComfyUI's frontend i18n/localization system
  ([`c32c2fa`](https://github.com/frost-byte/fbTools/commit/c32c2fa9d4410b7d2bac473c3884f11cd27f2e57))

Captures what we learned tracing the selection-toolbox tooltip fix: what
  locales/<lang>/{commands,nodeDefs,settings,main}.json each cover (including that widget
  labels/tooltips live under nodeDefs.json's inputs.<name>.* path, since V3 schema unifies widgets
  and sockets), the fragile self-keyed fallback used for sidebar icon tooltips, and that anything
  rendered inside our own type:"custom" sidebar panel is entirely outside the system with no
  supported hook to reach it. Reference only -- not linked from CLAUDE.md, nothing in the extension
  depends on it. Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- Add TaoMate-H3 conversion, Qwen3.8 deployment, and Unsloth handoff notes
  ([`762ac64`](https://github.com/frost-byte/fbTools/commit/762ac64a445c464d7e4577962f94d421230b78c8))

- taomate-h3-comfyui-conversion-plan.md: converting the TaoLiveAIGC/ TaoMate-H3 LoRA adapter for
  ComfyUI compatibility — phases 0-4 done, converted file in place, awaiting a live A/B test. -
  qwen38_local_deployment_guide.md: hardware-assessed local deployment notes for Qwen3.8-27B /
  Qwen3.8-Flash-Next on an RTX 3090 24GB system, synthesized from external review sources. -
  unsloth-comfyui-integration-handoff.md: handoff notes for wiring an Unsloth-on-Modal backend into
  the LLM assistant's existing backend system, distinct in shape from the existing vision_llm.py
  RPC-style integration.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

### Features

- **assets**: Rename Bundles tab to Assets; add Backgrounds, Camera, Sound and Outfits sub-tabs
  ([`966fb8f`](https://github.com/frost-byte/fbTools/commit/966fb8f1183d6a8b9d6993e3535d15ac24292e3b))

Sub-tab lists reuse the extracted background/outfit modals plus a new preset modal; edits fire
  fbt:library-changed so Compose stays in sync, and subject edits now notify Compose too.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **assets**: Thumbnails on background/outfit cards; document the Compose and Assets tabs
  ([`12019dd`](https://github.com/frost-byte/fbTools/commit/12019dd1b57719b4625ad7c19ca6facbf99e8fd4))

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **backgrounds**: H3 "Remove People" generation with configurable model/LoRA/sampler
  ([`bfec86d`](https://github.com/frost-byte/fbTools/commit/bfec86df9124eb01f2619c35d625f251c299815d))

Adds a "Remove People (H3)" action to the Background editor's file browser: given a selected image
  or video frame, it extracts/prepares the source image, submits a MiniMax H3 Reference-to-Video +
  Fizgig H3 Still workflow (templates/h3_background_plate.api.json) to this same ComfyUI server's
  own /prompt + /history endpoints, and adds the clean result as a reference image. No browser
  tab/canvas is involved in the submission, so the user's own open graph is untouched. The result is
  auto-selected on success so a second pass can chain straight off the prior output.

utils/h3_template_runner.py patches a small required contract (image, prompt, seed, filename_prefix)
  by node title, plus a generic optional `overrides` mapping for whatever additional titled nodes a
  template happens to expose. utils/h3_job_runner.py handles the submit-and-poll cycle against
  /prompt and /history.

Also adds a Settings > H3 Background Plate section letting the model, CLIP, sampler,
  scheduler+steps, and an optional LoRA+strength be overridden per-request instead of fixed to
  whatever's baked into the exported template — each control is disabled with an explanatory tooltip
  when the current template doesn't expose that particular optional title, and shows the template's
  own current value as a placeholder so nothing displays a number that looks live but isn't actually
  being applied.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **bundle-editor**: Json appearance analyzer, history pagination, collapsible sections
  ([`02affc7`](https://github.com/frost-byte/fbTools/commit/02affc78baae173e658a411dd04b25d595764672))

- Analyze Appearance LLM now requests structured JSON (summary/hair/face/body/outfit); → Bundle
  button parses JSON and populates trait fields + auto-opens details section; → Subject Profile
  extracts summary from JSON before saving - History section paginated at 8 entries per page with
  prev/next controls - Trait details section now appears before history in form order - Visual
  section is collapsible (<details>) with Images and Video as nested collapsible sub-sections; each
  auto-opens when bundle already has media - Audio section is collapsible; auto-opens when source !=
  none

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **bundle-editor**: Unify appearance-analyzer source into one dropdown
  ([`b1954e9`](https://github.com/frost-byte/fbTools/commit/b1954e90b98097b3bf02e2f2ead24f726b9ed862))

Previously the panel showed either a video-frame extractor OR an image dropdown, chosen once from
  b.visual.type — a bundle in "both" mode (images AND video both configured) could only ever analyze
  from whichever the ternary picked, never its images. Replace with one dropdown listing the video
  reference (if any, marked with a sentinel value) alongside every available image; the
  frame-extraction row shows only when the video entry is selected.

The "restore previous analysis" replay path now skips re-selecting a saved video-frame source (the
  extracted temp file no longer exists), keyed off the analysis's own recorded isVideoFrame flag
  rather than the bundle's current visual.type.

Renames .fbt-be-llm-img-sel -> .fbt-be-llm-source-sel to match.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **bundles**: Add head turnaround image role and video source role dropdown
  ([`05e2792`](https://github.com/frost-byte/fbTools/commit/05e2792e9e81d3a628e0e5cf6d23c3b4d51a3825))

- Add "head turnaround" to SHEET_ROLES and wire descriptions into _SHEET_ROLE_H3 /
  _SHEET_ROLE_INLINE in prompt_assembler.py so prompt assembly correctly describes three-angle
  facial reference images - Define VIDEO_ROLES constant with five predefined options (full body
  turnaround, performance, action, dialogue, expression reference) - Add role: "" field to b.visual
  default data structure - Add _buildVideoRoleSection() that renders a dropdown with predefined
  options plus a "custom…" fallback revealing a free-text input; existing bundles with a
  non-standard role string are automatically migrated to the custom path on load

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **bundles**: Proxy-cache reference bundle video, mirroring Source Profile clips
  ([`98796e7`](https://github.com/frost-byte/fbTools/commit/98796e746bb574dbe8f2052995c87effb3b40ed9))

Reference bundles had no caching layer for their video reference at all — every generation run
  re-seeked/re-decoded the original source file from scratch, unlike Source Profile clips, which
  already get a persistent, pre-trimmed/scaled/24fps-baked proxy (utils/proxy_cache.py). This
  extends that same system to bundles, with three trigger points per the user's own design:

- Preview (POST /fbtools/bundles/preview_sampled): fires a background, fire-and-forget proxy build
  using the in-editor (not-yet-saved) settings — reuses ffmpeg work the user already pays for when
  previewing. - Save (POST /fbtools/bundles/save): fires the same build using the just-saved values,
  idempotent against whatever Preview already built (matching stem -> near-instant no-op). -
  Generation-time fallback (_resolve_cast_media's want_video branch): tries the proxy synchronously,
  swaps the bundle's video_file to its absolute path and resets only load_params["start_time"] (not
  duration/select_every_nth -- proxies are undecimated, decimation still happens at load time,
  exactly mirroring how SourceProfileClipPrompt already uses this system). Any failure is caught and
  logged; generation always falls back to the original file untouched.

A bundle's force_rate is a genuinely per-bundle configurable field (unlike a Source Profile clip's,
  which is hardcoded to 24) -- proxies bake in 24fps unconditionally, so a new
  _bundle_proxy_eligible() guard (duration > 0 and force_rate in (0, 24)) is checked at all three
  trigger points before ever building or using one.

- utils/proxy_cache.py: generalized _proxy_dir/_proxy_stem to take a namespace/kind, extracted the
  shared ffmpeg body into _build_proxy(), added ensure_bundle_video_proxy() writing under
  proxies/bundles/ (kept separate from proxies/source_profiles/) -- ensure_source_profile_proxy's
  public signature/behavior is unchanged. - extension.py: new GET /fbtools/bundles/proxy_status
  route; _fire_bundle_proxy_build() shared by the three trigger points, broadcasting over the
  existing fbtools.status/source="proxy_build" channel so js/ui/bundle_editor.js's new freshness
  readout picks it up via the same listener pattern already used in source_profile_editor.js and
  SceneCastBuild's clip preview. - js/api/bundles.js: proxyStatus(). js/ui/bundle_editor.js:
  freshness readout next to the existing frame-count readout, no manual "Build proxy" button
  (Preview/Save already trigger it, per the user's own design).

Verified end-to-end against real data (scratch temp dir, not the real user-data cache): built a real
  proxy for the actual alex_amd_norsk_dance_flo bundle's murder_dance_001.MP4 (force_rate=24,
  select_every_nth=2, duration=4.7) and confirmed the fixed duration_cap formula against it now
  computes 56 frames, matching the bundle editor exactly.

Tests: tests/test_proxy_cache.py +3, new tests/test_bundle_video_proxy.py (10, AST

source-contract style -- extension.py can't be imported directly in tests). Full suite: 1281
  passed/17 skipped (pytest), 175 passed (npm test).

Needs a ComfyUI restart for the Python changes; JS is hot-loadable. Live verification (Preview ->
  freshness readout -> real generation showing 56 frames) pending restart.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **bundles**: Structured appearance traits + pronoun/short_name on bundles
  ([`245e0b6`](https://github.com/frost-byte/fbTools/commit/245e0b694e7fa351082b6fca915ad92ba484d546))

- Add hair/face/body/default_outfit fields to bundle schema (upsert migrates legacy
  appearance_override → appearance.summary) - Bundle fields override Subject fields with bundle-wins
  merge semantics across all three resolution paths: source-profile+bundle (extension.py), Prompt
  Compositions (prompt_compositions.py), and schema normalisation (reference_bundles.py) -
  retention_analysis uses structured traits for retain clause instead of appearance_summary; falls
  back to short_name's appearance or generic - Fix pronoun resolution reading both _pronoun_style
  and pronoun_style field names - Add short_name field to Source Profile editor (always visible) and
  Bundle editor - Fix appears_clause: video-editing subjects use (appears in [Shot 1]) not (appears
  throughout) - Update scene_cast_build UI and tests accordingly

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **canvas**: Add "Send Get/Set Nodes to Back" command
  ([`7396736`](https://github.com/frost-byte/fbTools/commit/7396736ce88d9fd046e51a017f08695f76b34505))

Small collapsed Get/Set nodes (e.g. from KJNodes) frequently get dropped visually on top of the
  larger node they route a value into/out of. Litegraph's z-order is just array order (later = drawn
  on top = wins hit-testing), so once a Get/Set dot ends up on top it can permanently block clicks
  to whatever is underneath it -- including the click that would otherwise bring that node back to
  the front itself via ComfyUI's own built-in bring-to-front-on-click. This adds a manual escape
  hatch: a command-palette command that sends every GetNode/SetNode instance in the currently-viewed
  graph (root or subgraph) to the back in one shot, via the existing app.canvas.sendToBack() API.

Follows the same commands-array pattern already used for "Extract Node as JSON" in the same file,
  reusing the existing showToast() helper for feedback.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **cast**: Background override on Scene Cast Build when a Prompt Composition is connected
  ([`95294bc`](https://github.com/frost-byte/fbTools/commit/95294bc9ddd43d33d5b3b67056b6b96acb048a4a))

Dropdown (composition default first) plus a use-as-reference checkbox; overrides travel on the cast
  dict and are applied by PromptCompositionLoader before assembly. Run History shows the effective
  background rows.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **cast**: Background section on Scene Cast Build with image and soundscape checkboxes
  ([`bab2475`](https://github.com/frost-byte/fbTools/commit/bab247503575836b4196e2571b8ccac9323c159a))

Adds a soundscape override (use the background's soundscape in place of the composition's) and
  groups the background dropdown with image/soundscape checkboxes under one titled section.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **cast**: Composition Load node and Prompt Composition input on SceneCastBuild
  ([`f0dcefb`](https://github.com/frost-byte/fbTools/commit/f0dcefbecffc3cc8d2d1b63aea54ae705c073f38))

Adds a PROMPT_COMPOSITION wire type and a Composition Load node (combo of saved compositions ->
  composition dict + subject info), wired into a new optional prompt_composition input on
  SceneCastBuild, mirroring how Source Profile Load feeds source_profile.

- SceneCastBuild: the composition's subjects become the cast pool. Ordinal entries ("the Nth subject
  sharing this bundle's pronoun") resolve against the composition's subject roster via
  resolve_ordinal_from_list and stay plain bundle-backed entries, which PromptCompositionLoader
  matches by subject_id. If a Source Profile is also connected it keeps driving the node and the
  composition is ignored with a warning. A new pass-through output is appended at the end so
  existing links keep their slot indexes. - Propagation: composition_load.js wraps the
  composition_name combo callback and re-fires onConnectionsChange on downstream nodes (same
  mechanism as source_profile_load.js), so SceneCastBuild refreshes when the selection changes.
  scene_cast_build.js fetches the composition, restricts the subject select to its roster (labelled
  with slot letters), and shows/resolves the ordinal control for compositions. -
  utils/prompt_compositions.composition_ordinal_roster builds the ordered roster (tested); the
  timeline / action preview for compositions is Phase D.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **cast**: Show a connected composition's shots on SceneCastBuild's timeline and preview
  ([`de97c60`](https://github.com/frost-byte/fbTools/commit/de97c607f54e7a125c5ed06b14b77ec5c163f471))

When a Prompt Composition (and no Source Profile) is wired into SceneCastBuild, its shots appear on
  the existing clip timeline and the action preview resolves {A}/{B}/... placeholders to the cast's
  bundle names, like Source Profile clips.

- js/utils/composition_timeline.js: pure helpers (timestamp parsing, shots -> timeline segments with
  equal-band fallback when timestamps are missing or not strictly increasing, shot display text,
  general slot-placeholder substitution) with Jest tests. - scene_cast_build.js: the composition
  fetch now keeps shots; _refreshClipSelects uses them as segments when no profile is wired
  (duration multiplier hidden); _buildActionPreview has a composition branch using the roster's
  explicit slot letters, ordinal resolution and conflict detection; the profile branch now uses the
  general helper instead of the A..J-only regex; the composition refresh runs before the profile
  refreshes to avoid a race.

Frontend only; no Python change (the clip_id widget stores the shot id).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **compose**: List-then-editor Compose tab; move library palettes to Assets
  ([`be2dc07`](https://github.com/frost-byte/fbTools/commit/be2dc07475f49734df5f6e60cf9bc6da41dd015e))

Compose now opens on a searchable card list of saved compositions (subject/shot counts, background,
  edited date); clicking a card or + New opens the full-width editor with a Back button, like the
  Sources tab. The sidebar (subjects, backgrounds, camera/sound presets, outfits) is gone: subjects
  are added from a picker in the Subjects section, camera/sound presets from pickers on each shot,
  and the assets themselves live in the Assets tab. list_compositions now returns subject_count,
  shot_count and background.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **composition**: Migrate subject slots from S1/S2 to A/B letter notation
  ([`d088c51`](https://github.com/frost-byte/fbTools/commit/d088c510f2b15f274550bda6c2c18d1f2bc64d8f))

Prompt Composition subjects were keyed "S1"/"S2"/... (declared once, composition-wide) with
  {S1}/{S2} placeholders in shot text, while Source Profile clips already use single-letter A/B/...
  notation for the same purpose. assemble_composition() translated one to the other internally via
  an index-based slot_map — exactly the layer where a real bug was found and fixed last session
  (_transfer_to_slot needed remapping through it too).

Store the letters directly instead of translating at assembly time. This is step one of a larger
  SceneCastBuild/Composition unification (see the approved plan): it removes most of the translation
  layer, and sets up Composition subjects to reuse SceneCastBuild's existing action_preview
  mechanism instead of a parallel implementation in a later phase.

- New utils/slot_letters.py: slot_letter(index) generates spreadsheet-column style slots (A, B, ...,
  Z, AA, AB, ...) with no artificial cap — the old inline chr(ord("A")+idx) sites in
  assemble_composition() silently produced garbage past index 25. prompt_assembler.py duplicates the
  same ~10-line algorithm locally as _slot_letter() rather than importing slot_letters, per this
  repo's convention that pure utils/*.py modules don't import each other (utils/ has no __init__.py;
  documented in docs/GOTCHAS.md). - Fixed a real UnboundLocalError in assemble_composition()'s
  text-only-outfit branch: outfit_overrides (the local dict) was read before being defined, raising
  whenever a composition had a text-only outfit override and resolved_outfits was non-empty. -
  Verified and rejected one part of the original sub-plan: renaming the internal "BG"
  background-reference sentinel to avoid a theoretical collision with a 59th generated subject slot
  would have broken the {BG} shortcut for every composition (the literal token users type), not just
  the near-unreachable collision case. Left as a documented, accepted limitation instead. -
  extension.py: PromptCompositionLoader's duplicate sk_to_letter identity map (only used for
  per-slot trim_to durations) is now unnecessary and removed — the dialogue speaker key already is
  the slot key. - js/ui/composition_editor.js: _nextSlotKey()/_renumberSlots() generate the same
  spreadsheet-column letters (removing the old hard cap of 9 slots); _slotKeys() now preserves
  insertion order instead of sorting, since a lexicographic sort would misorder "AA" ahead of "B"
  once slots exceed 26. Also fixed a pre-existing gap: _renumberSlots() rekeyed subjects/
  outfit_overrides/etc. and shot.dialogue.speaker on slot removal, but never rewrote {OLD}
  placeholders hand-typed into shot action/camera text or the scene synopsis — silently leaving a
  stale reference pointing at whatever subject now occupies the renumbered key. Fixed by rewriting
  those placeholders through the same old-key->new-key map. - New
  scripts/migrate_composition_slots.py: one-time migration for existing on-disk compositions,
  following this repo's migrate_masks.py/ migrate_lora_stack.py convention (argparse, --dry-run,
  .bak backup, idempotent). Rewrites subjects/_subject_snapshots/outfit_overrides/
  outfit_ids/slot_descriptors/appearance_overrides keys, shots[].dialogue. speaker, and {S1}/{S2}
  placeholders in shots[].action/camera/ scene_synopsis. Verified manually against a synthetic
  multi-file test directory (S10 sorts correctly after S2, orphaned keys warn and are left
  untouched, re-running is a no-op, .bak is created). - Updated tests/test_assemble_composition.py,
  tests/test_prompt_assembler.py (including last session's
  TestHybridCastRetentionInAssembledPrompt), and tests/test_cast_enrichment.py to use letter-keyed
  fixtures throughout (apply_cast_to_subjects is notation-agnostic, so this is a pure
  fixture-literal substitution with no logic-path changes there).

Full suite: 1092 passed (was 1074 + 18 new slot_letters tests). JS: 104 passed. Static sweep
  confirms zero remaining S1/S2-shaped literals in production code.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **composition**: Prompt Composition input on the loader; accept %*.N% libber typo
  ([`1f0e080`](https://github.com/frost-byte/fbTools/commit/1f0e080500f3b7c0083ac5c22703b7093060385d))

- PromptCompositionLoader gets an optional prompt_composition input (from Composition Load or
  SceneCastBuild's pass-through). When connected it drives the node - subjects, shots, LoRAs,
  libbers, filename prefix - and the Composition dropdown is ignored, so one selector controls both
  nodes instead of two that could drift apart. The dropdown is greyed out in the UI while a
  composition is wired, the fingerprint keys off the wired composition's content, and the Run
  History formatter shows the wired composition and its LoRAs. - The libber resolver accepts %*.N%
  as a spelling of %*:N%; eight saved compositions had the dot form, which never resolved and was
  left visible in the prompt.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **conditioning**: Wire VRAM estimator into CompositionToH3Conditioning
  ([`cd01aad`](https://github.com/frost-byte/fbTools/commit/cd01aad75f21d570e3d65da0703252f52bfc1dfe))

Adds estimate_vram (default on), vram_safety_buffer, and desired_scale inputs, and Recommended Scale
  / VRAM Estimate outputs, using the already-tested utils/h3_vram_estimator.py
  (tokens_for/max_safe_scale) calibrated from real OOM incidents on this machine.

Reference token cost is approximated per the ref_image_size mode: "match" uses the generation
  canvas's own per-frame token cost (since MiniMaxH3ReferenceToVideo rescales every reference to
  that canvas area), "max" uses the reference's actual loaded resolution. Main pass tokens come from
  the requested width/height/length. With no CUDA device available, the estimate is skipped and
  desired_scale (or 1.0) passes through unvalidated rather than failing the node.

desired_scale <= 0 means auto (output the calculated safe maximum); any other value passes through
  unchanged if it fits the estimate, or gets clamped down to the safe maximum with a warning if it
  doesn't.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **history**: Mark tracked nodes with a paw-print title marker instead of a [track:] suffix
  ([`f9d9f78`](https://github.com/frost-byte/fbTools/commit/f9d9f78f4d154f7fae052010ad6838938c1a4027))

A tracked node's title is now '🐾 Label' (label = title minus the marker), which keeps titles short
  and nodes easy to resize. The legacy '[track: Label]' form is still parsed everywhere and is
  rewritten to the marker form when a workflow loads.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **history**: Show start -> end (duration) in Run History headers
  ([`c14dbce`](https://github.com/frost-byte/fbTools/commit/c14dbce8a087f72fb3cf159871867964e15b0aa6))

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **kdenlive**: Archive projects into portable folders (CLI + Archive tab)
  ([`c65927b`](https://github.com/frost-byte/fbTools/commit/c65927b9b8b18b0cbb732b0fa5cb4e9b4fcaf001))

Generalises a one-off script that made a Kdenlive project portable across Windows/macOS/Linux:
  resolve every clip reference, copy the clips into media/<source_folder>/, rewrite the project to
  relative paths with root="", and strip the ComfyUI workflow/prompt JSON Kdenlive copies from each
  clip's metadata into the project (295 MB -> 5.6 MB on the source project).

- utils/kdenlive_archive.py: pure-stdlib engine (analyze / archive / strip_metadata). References
  resolve relative to the project root, via user path maps (Z:/=..., //host/share=...), as-is, or by
  searching extra folders by filename (ties broken by matching trailing path components, then by the
  clip's recorded kdenlive:file_size). Missing clips are left untouched and reported. Colliding
  archive names get unique suffixes, existing identical files are skipped so re-runs resume, the
  source project is never modified. - scripts/kdenlive_archive.py: check / archive / strip
  subcommands; exit 2 when clips are unresolved. - nodes/kdenlive_archive.py + js Archive tab:
  check/strip routes, a background archive job with websocket progress, status polling (survives a
  page reload) and cancel.

Verified against the real project: identical resource list and media tree to the hand-built archive,
  and search-only resolution finds all 445 files.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **kdenlive**: Browse for files/folders in the Archive tab (input/output only)
  ([`a4020d8`](https://github.com/frost-byte/fbTools/commit/a4020d865e2f36d84dd34c6ff64492848a390435))

Every text field in the Archive tab (project, destination folders, search folders, clean-clips
  source/destination) gets a 'Browse...' toggle that reveals an inline file or folder tree, matching
  the browse-or-type UX other editors already give for media. Scope stays input/output-only,
  matching every other picker in this codebase.

- nodes/kdenlive_archive.py: GET /fbtools/kdenlive/browse_files (.kdenlive files, recursive) and
  /browse_dirs (every subdirectory, including empty ones -- the gap /fbtools/media/list can't fill
  since it only knows about dirs containing a matching file). Both return absolute paths, unlike
  /fbtools/media/list's relative ones, since Kdenlive's own functions take plain OS paths. -
  js/ui/folder_tree.js (new): file_tree.js's sibling for picking a folder rather than a file --
  every node is a folder, so a click both selects and expands/collapses it, plus a synthetic '(this
  folder)' root row. - js/ui/kdenlive_archive.js: fetches both lists once, converts each absolute
  path to root-relative for display in file_tree.js/folder_tree.js (both use the given path as-is
  for display and value), and re-absolutizes on selection. Search folders' browser appends a line
  instead of replacing the field, since it's a multi-value textarea.

Verified: 7 new route tests against real temp directories (absolute paths, empty dirs included,
  dot-dirs skipped, invalid folder param rejected) and 10 new folder_tree.js unit tests.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **kdenlive**: Clean generated clips before adding them to a project
  ([`3d34754`](https://github.com/frost-byte/fbTools/commit/3d347543ae6ac3bcd235fa10b785096cc532ee6e))

utils/kdenlive_clips.py: strip_copy_video() remuxes one clip without its embedded metadata (the
  ComfyUI workflow/prompt JSON a saved mp4 carries as container metadata) via ffmpeg stream copy,
  atomic write, source never touched, duration-verified against the source (the temp file is
  discarded if they don't match); find_duplicate_files() groups files by content for review;
  clean_folder() runs it over a folder, skipping files already done, reporting duplicate sources
  without touching them.

scripts/kdenlive_archive.py gains a 'clean' subcommand. nodes/kdenlive_archive.py gains POST
  /fbtools/kdenlive/clean as its own background job kind, sharing the existing status/cancel routes
  (both now carry/accept 'kind' so the Archive tab's two forms - project archive and clean clips -
  can run independently without their status/progress events crossing over the shared websocket
  channel). Archive tab gets a 'Clean clips' section.

Verified against real synthetic clips (metadata dropped, duration preserved, source untouched) and
  end-to-end through the actual route/background-job/status-poll path under a stubbed PromptServer.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **kdenlive**: Embed a curated cast/composition/time metadata tag when cleaning a clip
  ([`11a9519`](https://github.com/frost-byte/fbTools/commit/11a951968c498234d5180903f7e04597bf445b79))

Adds an opt-in "embed_cast_metadata" flag to the clean-clips feature (route, CLI, and Archive tab
  UI), independent of the existing "organize_by_primary" flag but sharing the same per-run cast-info
  cache so turning on both costs one ffprobe read per file, not two.

- utils/kdenlive_clips.py: strip_copy_video()/clean_folder() gain an extra_metadata callback,
  written via ffmpeg's use_metadata_tags muxer option (plain -metadata silently drops custom keys
  outside the mov/mp4 classic whitelist — same mechanism ComfyUI's own workflow/prompt tags rely
  on). Off by default, output byte-identical to before when unset. - utils/generation_metadata.py:
  new CAST_SUMMARY_TAG, generated_at_iso(), build_cast_summary_tag(), read_cast_summary_tag() — a
  compact JSON tag (composition, primary subject/bundle, bundle tags, generated_at) a cleaned clip
  carries even after its original embedded prompt is gone. - nodes/kdenlive_archive.py: /clean
  route's embed_cast_metadata flag; fixed a latent KeyError in _cast_info_cache()'s
  no-embedded-metadata fallback (missing primary_bundle key). - scripts/kdenlive_archive.py:
  --embed-cast-metadata DATA_DIR CLI flag. - js/api/kdenlive.js, js/ui/kdenlive_archive.js: new
  checkbox + its own report section.

Verified against real data: round-tripped the actual wide_shot_00001-audio.mp4 clip through
  build_cast_summary_tag/read_cast_summary_tag, and separately re-ran the full route wiring against
  all 46 real process_me clips into a scratch dir (removed after) — both matched exactly.

Tests: tests/test_kdenlive_clips.py (+5), tests/test_generation_metadata.py (+6), new

tests/test_kdenlive_clean_metadata_route.py (5, incl. the KeyError regression). Full suite: 1257
  passed/17 skipped (pytest), 167 passed (npm test).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **kdenlive**: Generalize this session's ad-hoc cast-metadata recovery/organization scripts
  ([`0425f0d`](https://github.com/frost-byte/fbTools/commit/0425f0d8013042ccf0a6e17c8dff4eb84ef419d0))

Turns two one-off investigations into reusable, documented CLI tools (no project-specific data —
  placeholder names/paths only):

- scripts/kdenlive_recover_cast_metadata.py: for clips that already had their embedded ComfyUI
  metadata stripped before fbtools_cast existed (or lost it some other way), recover a
  composition/subject/bundle summary by content-matching against an unstripped copy of the same clip
  still sitting elsewhere (e.g. the raw output tree), then embed it in place. Matches by decoded
  audio/video stream hash, never by filename or file size — two clips can share a name with
  genuinely different content. Optional --organize-by-primary/--organize-by-bundle nests the
  recovered clip into a subfolder; --project guards against moving a file a Kdenlive project still
  references at its current path (override with --force-move-referenced).

- scripts/kdenlive_check_resource_usage.py: reports whether given files (or every video file in a
  folder) are referenced anywhere in a .kdenlive project, so a file can be confirmed safe to move/
  rename/delete before touching it.

- utils/kdenlive_archive.py: new count_references(project) -> {basename: count}, the pure lookup
  both scripts build on (no path resolution, just what the project's XML says by name).

- docs/GOTCHAS.md: two new entries — same-filename clips aren't necessarily the same content (use
  content hashing, not filename/size, to decide), and check a project's own reference count before
  reorganizing media it might already use.

Purely local, data-specific investigation from this session (the actual real-project findings and
  one-off recovery run) is intentionally not part of this commit — only the generalized, reusable
  technique is.

Tests: tests/test_kdenlive_archive.py +3 (count_references). Full suite: 1260 passed/17 skipped
  (pytest), 167 passed (npm test). Both new scripts smoke-tested end-to-end against synthetic ffmpeg
  clips (dry-run, real recovery, idempotent re-run, in-place vs organize-by-primary/bundle, and the
  --project reference guard with and without --force-move-referenced).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **kdenlive**: Organize cleaned clips by primary subject; report per-clip bundle tags
  ([`fb6dfab`](https://github.com/frost-byte/fbTools/commit/fb6dfabda878c97461b4fac5580bbc47e3d2da0e))

A clip's own embedded prompt metadata (the same JSON we strip for Kdenlive) already carries
  everything needed: fbt_SceneCastBuild.cast_entries_json for the bundles used (tags), and
  fbt_CompositionLoad.composition_name + the composition's first subject slot for the primary
  subject (destination folder). No new tracking needed for composition-driven clips.

- utils/generation_metadata.py (new): read_embedded_prompt() via ffprobe, extract_cast_info() pure
  graph walk. Mirrors PromptCompositionLoader.execute()'s own precedence (a wired prompt_composition
  beats a stale loader dropdown value) and uses composition dict insertion order for 'first slot',
  never a sort (slot_letter()'s A..Z, AA.. scheme sorts wrong past Z as plain strings). -
  utils/kdenlive_clips.py: clean_folder() gains an optional dest_subdir(filename, src_path) callback
  (None = today's flat behaviour, unchanged), keeping the module itself free of any
  ComfyUI/composition concepts. - nodes/kdenlive_archive.py: /clean gains organize_by_primary; wires
  the two together via the existing composition loader, caches one embedded-metadata read per clip,
  enriches each report entry with tags/primary_subject/note. - scripts/kdenlive_archive.py: clean
  gains --organize-by-primary DATA_DIR. - Archive tab: 'Organize into folders by primary subject'
  checkbox; report shows each clip's destination folder and tags, or why it was left unsorted
  (Source-Profile-driven clips have no primary-subject ordering yet -- a gap the user named
  themselves, out of scope here).

Nothing writes into any .kdenlive project -- tag color provisioning respecting Kdenlive's
  one-color-per-tag rule is deferred to Plan 5's bin-insertion work, which this feeds.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **kdenlive**: Resolve the specific bundle used by the primary subject
  ([`9a664f6`](https://github.com/frost-byte/fbTools/commit/9a664f6e2fd707ae298e028705d19d8ef6d935b7))

extract_cast_info() gains primary_bundle: the cast entry's bundle_id for whichever subject is
  primary (not just the subject's raw id), distinct from primary_subject being None outright. Needed
  as a collision fallback for clean-into-a-fixed-folder workflows (e.g. Kdenlive's media/comps):
  when a filename already exists at the flat destination, fall back to a bundle-named subfolder
  instead of silently skipping a same-named-but-different clip or blindly overwriting it — the
  skip-if-name-exists gap in clean_folder that a name collision surfaced in real data. The
  organize_by_primary route/report now also carries primary_bundle alongside primary_subject.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **libbers**: Add Libber Editor tab with full CRUD UI and backend endpoints
  ([`8ec8794`](https://github.com/frost-byte/fbTools/commit/8ec8794912b61d4555be0bad552353430659e1e0))

Backend (extension.py): - GET /fbtools/libber/scan — scan disk + in-memory, return [{name,
  entry_count, delimiter, max_depth}] - POST /fbtools/libber/open — ensure_libber from disk, return
  full {name, lib_dict, ...} - POST /fbtools/libber/save_full — overwrite in-memory libber + persist
  to disk in one call - POST /fbtools/libber/delete — remove from memory and delete disk file - POST
  /fbtools/libber/rename — rename disk file and update memory key - Add libber_max_depth (default
  10) to composition settings schema and POST handler

API client (js/api/libber.js): - Add scan(), open(), saveFull(), deleteFull(), rename() methods

UI (js/ui/libber_editor.js — new file): - List view: search, paged cards showing
  name/entry-count/delimiter/depth, edit and delete buttons - Detail view: back nav, editable name
  (triggers rename on blur), delimiter, max_depth, full entries table - Entries table: key
  (normalized, monospace, editable), value (textarea, auto-saves), delete per row - Add-entry form
  at the bottom of the table; Ctrl+Enter in value field to submit - Debounced auto-save (600 ms) on
  any change; explicit Create button for new libbers

Settings (js/ui/settings_panel.js): - Add Libber max depth row (1–50) in Compose Defaults section

Panel (js/ui/fbt_panel.js): - Register Libbers tab between Sources and LLM

Styles (js/styles/style.css): - Add fbt-lbe-* rules for the new editor

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **libbers**: Add save-state indicator in form header
  ([`c95317b`](https://github.com/frost-byte/fbTools/commit/c95317b6595b301d301807f702d7e7f90de85064))

Three-state inline indicator to the right of the libber name input: - pending — pulsing '···'
  (muted) while debounce timer is running - saving — spinning '↻' + text (accent colour) during
  network call - saved — '✓ saved' (green), fades out after 1.2 s

Timers are cleaned up on Back navigation so detached elements are never mutated after the form is
  torn down.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **llm**: Pre-load VRAM/context-capacity estimate for GGUF models
  ([`3f9df6b`](https://github.com/frost-byte/fbTools/commit/3f9df6b119128c1864b868eb9bf33af7b919c87e))

Adds estimate_context_table() to utils/llm_client.py, reading only the GGUF header via
  gguf.GGUFReader (no weights loaded) so the LLM panel can show a context-capacity guide as soon as
  a model is selected, before Load. Shares its KV-cache-vs-VRAM table logic with the existing
  post-load vram_analysis() via extracted _build_context_table()/
  _arch_meta_from_kv()/_arch_meta_from_mi() helpers.

Handles architectures whose GGUF header omits explicit attention.key_length/value_length
  (Qwen2/2.5-VL, notably) by deriving head_dim from embedding_length/head_count. Headroom accounts
  for the candidate model's own on-disk weight size (+ mmproj + a compute-buffer fudge factor),
  since that VRAM isn't reflected in current usage until the model is actually loaded.

New POST /fbtools/llm/context_estimate endpoint and llmApi. contextEstimate(). The LLM panel's local
  tab now shows this guide on model selection and re-highlights the active context pill on selector
  change without a re-fetch; the existing post-load VRAM card is unchanged in behavior, now sharing
  the same renderer.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **llm**: Warn when queuing a workflow while the local LLM is still loaded
  ([`d490fda`](https://github.com/frost-byte/fbTools/commit/d490fdab616a119b7d5aebfa6f11c543091b589c))

Listens for ComfyUI's promptQueued event (fires the instant Queue is clicked, before the prompt
  reaches the server) and checks GET /fbtools/llm/status fresh each time, so it works even if the
  fbTools panel was never opened this session. Only the local backend is checked — Unsloth/Modal run
  remotely and don't compete for local VRAM. A 15s cooldown avoids nagging during
  auto-queue/batches.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **run-history**: Auto-track node outputs across cache hits
  ([`2b08a71`](https://github.com/frost-byte/fbTools/commit/2b08a716dd533722879070ea9bfba2a1ed2cef3a))

[track: Label] previously only captured a tracked node's literal widget values from the static
  submitted prompt, missing values a node actually computes at execution time — and missed them
  entirely on a cache hit, since a cached intermediate node is pruned from the schedule before
  execute() is ever called (comfy_execution.graph's TopologicalSort/ExecutionList prunes it based on
  is_cached(), so there's no hook to intercept there).

Fix: wrap execution.execute() to capture a tracked node's resolved kwargs on genuine execution
  (cache miss), keyed by node_id so the stash survives into later prompts, then on every prompt
  check caches.outputs directly for any tracked node that didn't execute — a real cache hit means
  its inputs are unchanged, so the stashed values are re-emitted as-is.

RunMetaCapture now shares the same _record_capture/ stringify_capture_values path as the
  auto-tracker instead of duplicating its own capture logic.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **run-history**: Readable cast and LoRA rows for Prompt Composition Loader
  ([`c2176ae`](https://github.com/frost-byte/fbTools/commit/c2176ae0ff07bf7fe8cc33b07b518fcb30688524))

A tracked Prompt Composition Loader showed scene_cast in its "(extra)" History entry as a raw dict
  truncated at 300 chars. Show per-cast-entry rows instead: subject/bundle/mode, reference images
  (honouring image_selection), reference video with start/duration, audio (file, from video, or from
  the reference video), source profile and dialogue, plus a compact LoRA list in the same style as
  LoraStackBuilder's Enabled Summary. The prompt and H3 ref plan stay out.

- utils/composition_track_summary.py: pure summarising functions. - nodes/run_tracking.py: a small
  per-node formatter registry for the auto-tracker; a formatter failure falls back to the generic
  behaviour, and cache-hit backfill re-emits the formatted values unchanged. - extension.py:
  registers the formatter for the Prompt Composition Loader (loras come from the composition, since
  the tracker only sees inputs). - .fbt-rh-val gets white-space: pre-wrap so multi-line rows keep
  their breaks.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **run-history**: Resolve passthrough connections and add SceneCastBuild table
  ([`a3b82b1`](https://github.com/frost-byte/fbTools/commit/a3b82b19dd1665939762dacde7910359268fdfb4))

- extractWidgetValues() now resolves a connection-ref input by walking through
  Primitive/Reroute/Set-Get passthrough chains in nodesDict (bounded depth, no execution) instead of
  dropping every wired input outright — only genuinely computed/ambiguous values still fall through
  to the runtime capture. - Detect a connection ref hiding inside a composite widget's nested value
  (e.g. a V3 DynamicCombo with one sub-field wired) and treat the whole key as statically
  unresolvable, same as a direct ref — previously this showed a stale/partial object and blocked the
  runtime-captured version from replacing it. - Merge runtime captures (node-output auto-tracker +
  RunMetaCapture) into their matching static entry: drop keys the static scan already resolved, keep
  only genuinely new ones, drop the capture entirely once nothing new remains. Drop empty static
  placeholders that a capture already covers under the same label. - Add a dedicated SceneCastBuild
  table renderer (cast entries + resolved action_preview text), mirroring the existing LoRA Builder
  table.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **scene-cast**: Add read-only action_preview widget
  ([`0a2791b`](https://github.com/frost-byte/fbTools/commit/0a2791b8fb1a0da1b2e5005bd5a10f654a3f8ee0))

Read-only string input holding the active clip's action text with {A}/{B}/... placeholders resolved
  to bundle names. Populated by the frontend's on-node preview widget (scene_cast_build.js) so the
  resolved text rides along in the submitted prompt for Run History tracking; execute() itself never
  reads it.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **scene-cast**: Exclude a reference video's own background by default
  ([`169bdf7`](https://github.com/frost-byte/fbTools/commit/169bdf7757a2d668b591c26ecf3e38901a92f42e))

A video wired into a Scene Cast Build subject as an appearance reference was carrying its source
  background/setting into the generated H3 output in some runs. Plain identity-reference video lines
  (not the separate video-editing/continuation flows, which already handle background deliberately)
  now default to telling H3 to ignore the background of <Video N>, with an opt-in "Keep BG" checkbox
  per cast entry when the old behavior is wanted.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **scene-cast**: Tag a primary subject per cast and route it into a unified filename_prefix
  ([`d0cb39b`](https://github.com/frost-byte/fbTools/commit/d0cb39becfd95da4a2010fe1b65dc237eb153640))

Cast entries can now be marked primary (surfaced with a star in the cast editor and summary text),
  and SceneCastBuild exposes a filename_prefix output built from that primary subject so downstream
  nodes (Source Profile Clip Prompt, Prompt Composition Loader) can wire one consistent prefix
  instead of composing their own. generation_metadata now prefers an explicit primary flag over
  composition-slot-order inference.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **scene-cast-build**: Collapsible clip video preview for Source Profile mode
  ([`2e8cf44`](https://github.com/frost-byte/fbTools/commit/2e8cf4482188178dd882aba9f35cab2a92a9c764))

Adds a second "⊙ Clip Preview" collapsible section below the action-text preview, mirroring the
  existing per-entry bundle "⊙ Preview" toggle's DOM/CSS pattern but node-level, showing the
  currently selected Source Profile clip's actual video: its pre-built proxy (silent, already
  trimmed) when fresh, else the full source video seeked to and looped within the clip's
  start_time/end_time. Hidden entirely while a Prompt Composition drives the node.

- extension.py: new GET /fbtools/source_profiles/proxy_stream route, mirroring the existing
  /fbtools/bundles/audio_cache/stream allow-listed-root pattern but scoped to
  user_data_dir()/proxies/source_profiles/. Only ever serves an existing proxy — never generates one
  inline (ensure_source_profile_proxy can run ffmpeg for minutes; that's what the existing
  prebuild_proxies background job is for). - js/utils/clip_preview_source.js (new): pure
  proxy-vs-fallback decision, unit tested. - js/api/source_profiles.js: proxyStreamUrl() URL
  builder. - js/nodes/scene_cast_build.js: the collapsible section itself, wired into
  _updateActionPreview() (visibility + refresh-on-clip-change) and _widgetHeight(); a "Build proxy"
  button when falling back reuses the existing prebuild_proxies job and mirrors
  source_profile_editor.js's own completion-detection pattern (fbtools.status, source="proxy_build",
  /complete/i on the message — no per-job id exists server-side to correlate on more precisely). -
  js/styles/nodes/scene_cast_build.css: small button style; reuses existing preview classes
  (.fbt-scb-preview-toggle/-area/-video/-note) rather than duplicating them.

Tests: js-tests/clip_preview_source.test.js (8, the pure helper), new
  tests/test_source_profile_proxy_stream_route.py (5, AST source-contract style — extension.py can't
  be imported directly in tests since its ~75 io.ComfyNode subclasses need a real base class, not
  conftest.py's bare MagicMock; this mirrors test_dataset_caption_api.py's established pattern for
  that constraint, and is the first test of this allow-listed-root streaming shape in the repo).
  Full suite: 1265 passed/17 skipped (pytest), 175 passed (npm test).

Needs a ComfyUI restart for the new Python route; JS/CSS are hot-loadable. Not yet tried live.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **scripts**: Add YouTube text extractor for LLM research
  ([`43a4478`](https://github.com/frost-byte/fbTools/commit/43a44787dcacabc463488774491cda712e1c97d1))

Pulls metadata, transcript, and top comments from a YouTube video into a single markdown document
  for pasting into an LLM chat session. Depends on the "scripts" optional-dependency group
  (playwright, for --login) already declared in pyproject.toml — that groundwork was committed
  previously but this implementation file was not.

For personal research use only; not wired into any ComfyUI node.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **scripts**: Merge_subjects - fold subjects (or single bundles) into one subject
  ([`b099b52`](https://github.com/frost-byte/fbTools/commit/b099b522d40a028b6402d0fdcc5894334e2fb1f0))

Generic script + pure utils/subject_merge.py: SUBJECT or SUBJECT:BUNDLE sources, an output subject,
  a primary identity with fill-from-others, and rewriting of compositions, scene casts and saved
  workflow cast entries. Dry run and .pre-subject-merge backups.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **source-profiles**: Add live preview and richer progress for Detect boundaries
  ([`070fb4d`](https://github.com/frost-byte/fbTools/commit/070fb4dd46e84fe743a1438f118f24d86e79e540))

- New GET /fbtools/source_profiles/frame_at: a single JPEG frame at an exact timestamp
  (extract_frame_at_time, ffmpeg with cv2 fallback), used for clip-boundary start/end thumbnails
  without a full ffprobe duration lookup on every edit. - New POST
  /fbtools/source_profiles/segment_prompt_preview: returns the exact VLM prompt Detect boundaries
  would send for given flags/override, without running detection — sourced from the same
  build_segment_detection_prompt() the real request uses, so it can never drift from what actually
  gets sent. - batch_window_seconds is now normally omitted and auto-derived as interval_seconds *
  20 (full utilization of the 20-frame-per-call budget) rather than a flat default of 60s;
  interval_seconds default simplified to a flat 3.0s. - send_status_update() gains an `extra` dict
  merged into the websocket payload, used by detect_segments to emit structured per-window progress
  (phase/window_idx/elapsed_s/frames) instead of just a human-readable message string. -
  Subject-inference token budget raised 1024->2048, matching segment detection — 1024 was tight
  enough for thinking-mode models that the <think> block alone could exhaust it before any JSON was
  emitted, silently yielding an empty response with no error surfaced. Added a warning log for that
  case.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **source-profiles**: Add shot_description field and prune history
  ([`3c2b3e0`](https://github.com/frost-byte/fbTools/commit/3c2b3e0f75093987af371f681ca231c9f65a9f91))

- VLM segment-detection and clip-description schemas now request a shot_description field (camera
  framing/angle/POV), deterministically prefixed onto the action text via _combine_shot_and_action()
  rather than asking the model to embed it inline — keeps the join correct regardless of model
  compliance, and degrades gracefully for responses using the older schema with no shot_description
  key. - append_history_entry() now caps the history file at the most recent max_entries (default
  100, global across profiles) so it can't grow unbounded; pass max_entries=0 to disable pruning.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **source-profiles**: Auto-fill soundscape, group segments by setting for reusable backgrounds
  ([`a38755e`](https://github.com/frost-byte/fbTools/commit/a38755e8da2e519b67c60ac5f56f3a8eb6b40b11))

Adds a setting_label field alongside the existing soundscape suggestion from segment detection,
  groups detected segments by that setting in the review step so a background can be created once
  and reused across clips with the same setting, and fixes the profile search field losing focus
  after a single keystroke.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **source-profiles**: Js UI for per-clip and default background picker
  ([`c77a81d`](https://github.com/frost-byte/fbTools/commit/c77a81d3c93de631496162c31d4435ca1785d3bb))

Adds the frontend half of Phase 1 (background-as-Subject reference): a profile-level
  default-background dropdown in Video settings and a per-clip override dropdown in each clip card,
  both backed by the same backgrounds.json registry Compositions already use.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **source-profiles**: Per-clip background-as-Subject reference (Phase 1)
  ([`d4473a4`](https://github.com/frost-byte/fbTools/commit/d4473a4d3d36c66953736bbfbe466b8e9e3ec5d0))

Extends the background-as-<Subject N>-reference feature (already shipped for Compositions via
  assemble_composition()'s background_as_reference flag) to Source Profile clips, addressed per-clip
  rather than per-composition.

Extracted the existing inline background-minting logic in assemble_composition() into a shared
  _build_background_slot() helper (utils/prompt_assembler.py) -- pure refactor, no behavior change
  for Compositions, confirmed by new regression tests plus the existing suite.
  SourceProfileClipPrompt.execute() (nodes/source_profiles.py) already builds the same
  scene_instance/slot_assignments shape assemble_composition() does and calls the same shared
  assembly functions, so no changes were needed downstream of slot-assignment -- confirmed by
  reading the function directly rather than assumed.

Schema: clips gain an optional background_id (utils/source_profiles.py's _normalize_clip), profiles
  gain an optional default_background_id fallback (_normalize_profile + a new define_profile
  parameter). Both default to "" -- fully backward compatible, no existing profile is affected until
  a user opts in. Also fixed a latent bug found while wiring this through: the
  /fbtools/source_profiles/save route never passed default_background_id to define_profile(), which
  would have silently reset it to "" on every save once anything started setting it.

SourceProfileClipPrompt resolves clip.background_id, falling back to the profile's
  default_background_id, loads it from the same backgrounds.json registry Compositions already use,
  and mints the slot into "O" -- the next free letter after this file's existing fixed SOURCE_SLOTS
  (A-J) / BUNDLE_SLOTS (K-N) scheme. No {BG}-shortcut equivalent needed: a clip is always exactly
  one synthetic shot, so retention_analysis's existing "appears throughout" default (for an empty
  appears_list) already reads correctly with no new code.

13 new tests: _build_background_slot() directly (5), assemble_composition's background_as_reference
  path end-to-end as regression coverage for the extraction (3), and the new schema fields (5). Full
  suite: 1296 passed/17 skipped (up from 1283 -- all new tests counted), 175 passed (npm, unaffected
  -- no JS in this change). ruff --select F821,F401 clean on every touched file.

JS editor UI (background picker on the Source Profile clip editor) is follow-up work, not included
  here -- see ~/.claude/plans/component-reference-system.md Phase 1 section for the full design.
  Phase 2 (fixing the outfit Fit_N wearer-disconnection bug) is next, blocked on a live-generation
  test of a flat-lay outfit reference first. Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **source-profiles**: Thumbnail next to each clip's Background dropdown
  ([`b2b25d4`](https://github.com/frost-byte/fbTools/commit/b2b25d4a6cd31021cbaaa958769b801185670111))

Shows the selected Background's first still-image reference (60x34, matching the existing start/end
  boundary thumbnail's aspect ratio) next to the per-clip Background dropdown, below the segment
  subjects. Clicking it opens the full Background editor for that background; hovering shows its
  name.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **source-profiles**: Timeline zoom/paging, proxy-build guard, segment preview UI
  ([`29ce0a4`](https://github.com/frost-byte/fbTools/commit/29ce0a46a7f01e3a591cf91a6d0b71b79c509b35))

- Source Profile editor and SceneCastBuild timelines gain a zoom/paging window (10-clip default)
  with animated pan between windows, replacing unusable click-navigation once a profile has many
  clips. Removes the decorative boundary-marker circles on the timeline. - Fix a proxy-generation
  flood: _refreshProxyStatus() was re-persisting every already-fresh clip on each progress event
  (O(N^2) amplification). Also disable the build-all and per-clip proxy buttons for the duration of
  an active build so overlapping submissions can't trigger it again. - Frontend for the
  Detect-boundaries live preview: start/end clip thumbnails via frameAtUrl(), a live VLM-prompt
  preview via segmentPromptPreview(), and structured per-window progress rendering
  (phase/window_idx/elapsed_s) from the backend's richer status payload.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

### Performance Improvements

- **proxy-cache**: Bake 24fps into source-profile clip proxies
  ([`69d8dc8`](https://github.com/frost-byte/fbTools/commit/69d8dc8862d71d2aab43a53c7a46cb78d6e371d9))

H3 reference video always resamples to 24fps at generation time (force_rate=24), but proxies were
  previously trimmed/scaled at the source's native fps, so every run against a cached proxy paid
  that resample cost anyway. Bake fps=24 into the proxy build itself so generation just decodes an
  already-24fps file. Existing proxies are versioned out of the cache (stem gains an _f24 tag) so
  they regenerate under the new scheme rather than being mistaken for fresh.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

### Refactoring

- Extension.py split — composition-assembly layer (Plan 29) → 5 new nodes/*.py files
  ([`f2d2e87`](https://github.com/frost-byte/fbTools/commit/f2d2e87080ec199c50aa46aead8f58647eef024e))

The last and largest piece of the extension.py -> nodes/ package-split series (Plans 1-29). Moves
  the remaining 7 node classes (SceneCompose, PromptAssemble, SceneCastLoad, SceneCastBuild,
  CompositionLoad, PromptCompositionLoader, CompositionToH3Conditioning) plus 16 REST routes (8
  bundle routes, 8 compositions/settings routes) into 5 new flat files:

- nodes/compose.py — SceneCompose, PromptAssemble - nodes/scene_casts.py — SceneCastLoad,
  SceneCastBuild - nodes/bundles.py — /fbtools/bundles/* routes - nodes/compositions.py —
  CompositionLoad, PromptCompositionLoader, CompositionToH3Conditioning, /fbtools/compositions/*
  routes + settings - nodes/composition_shared.py — small neutral module breaking a genuine two-way
  dependency between bundles.py and compositions.py (bundles.py's preprocess_audio route needs
  compositions.py's _read_composition_settings/_h3_load_audio; compositions.py's _resolve_cast_media
  generation-time fallback needs bundles.py's _bundle_proxy_eligible), the same pattern
  nodes/composition_types.py (Plan 27) already established for this cluster.

Found and fixed the same landmine class Plan 28 caught for _build_h3_refplan: SceneCastBuild used
  composition_ordinal_roster only by extension.py's own top-to-bottom load order (imported ~500
  lines below where SceneCastBuild used to live), not a real top-of-file import — now a genuine
  import in scene_casts.py. Two function-scope local imports (proxy_cache, audio_preprocess) had the
  wrong dot-depth for their new home; fixed the same way Plan 28's did.

Caught and corrected one real extraction mistake before committing (via ruff F821/F401, not assumed
  clean): the old mid-file `.utils.prompt_compositions`/`.utils.prompt_assembler` import block and a
  stale `.utils.composition_resources`/`.utils import unsloth_client, modal_deploy` stretch both sat
  physically inside the copied line range but weren't meant to move — the former was a duplicate of
  imports already re-created correctly in the new file's header (and had the wrong dot-depth, which
  would have broken at runtime despite passing static analysis); the latter is live LLM/Unsloth
  startup config unrelated to composition-assembly, now confirmed still in extension.py where it
  belongs, untouched.

Repointed 2 tests, both by AST-parse path only (no assertion logic changed):
  test_h3_load_video_frames_duration_cap.py -> nodes/compositions.py, and
  test_bundle_video_proxy.py, whose functions are now split across three files -> extended to search
  compositions.py/composition_shared.py/bundles.py instead of one hardcoded path.

extension.py: 3881 -> 605 lines (154 -> 76 net across the whole Plans 17-29 series once every domain
  had a real home). Verified via the same methodology as every prior plan: io.ComfyNode class set
  (7, unchanged, byte-identical against the pre-Plan-29 baseline), route multiset (16, unchanged),
  get_node_list() names (76, byte-identical). ruff --select F821,F401 clean across all 5 new files
  plus extension.py itself (verified per-file, not just diffed). Full suite: 1283 passed/17 skipped
  (pytest, including both repointed tests), 175 passed (npm test, unaffected).

Needs a ComfyUI restart to verify live -- not yet done, queued alongside Plans 26-28's pending
  restart. This closes out the entire extension.py -> nodes/ package-split backlog (Plans 1-29).
  Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- Extension.py split — registry layer (Plan 26) into
  nodes/{concepts,subjects,scene_templates,outfits}.py
  ([`533b226`](https://github.com/frost-byte/fbTools/commit/533b226be75e8112cf94fbdd609dd119832376a8))

Moves the 12 registry-layer node classes (ConceptRegistryLoad/ConceptDefine/
  ConceptResolve/ConceptList, SubjectProfileLoad/SubjectProfileDefine/ SubjectProfileList,
  SceneTemplateLoad/SceneTemplateList, OutfitRegistryLoad/OutfitDefine/OutfitList) plus their custom
  io types and exclusive helpers into 4 flat nodes/*.py files. Adapts the original roadmap sketch (a
  nodes/composition_engine/ subpackage) to flat files instead, matching how every other route module
  in this family already lives (registry_api.py, outfits.py, lora_info.py, etc.) rather than
  introducing a subpackage layout nothing else here uses. Outfit classes join their existing route
  file (nodes/outfits.py) rather than a new one, since concepts/ subjects/scene_templates routes
  already live bundled together in nodes/registry_api.py and don't need touching.

One real cross-file coupling within the layer: SubjectProfileDefine's concept_id combo needs
  concepts.py's _concept_get_ids(), a same-package sibling import. Re-exports the 4 custom io types
  (ConceptRegistryIOType, SubjectProfileIOType, SceneTemplateIOType, OutfitRegistryIOType) and 2
  helpers (_load_subject_images, _load_subject_audio) back into extension.py, since the
  not-yet-moved SceneCompose/PromptAssemble layer still references them as bare names -- standard
  one-directional pattern used throughout this series. Cleaned up 3 imports left genuinely dead by
  the move (verified via ruff diff against the pre-move baseline, not assumed): ConceptRegistry,
  SubjectRegistry, and the unused utils.images import line that only existed for the moved
  subject-image-loading helpers.

extension.py: 7764 -> 6490 lines. Verified via the same methodology as every prior plan:
  io.ComfyNode class set (23 remaining in extension.py + 12 moved = 23 total, unchanged), route
  multiset (35, unchanged -- zero routes move), get_node_list() names (76, byte-identical). ruff
  --select F821,F401 diffed against the pre-move baseline shows zero new findings. Full suite: 1283
  passed/17 skipped (pytest, unchanged aside from needing nodes/media's route-only import restored
  after removing its now-dead _audio_get_list use -- caught by
  test_extension_imports_every_route_module), 175 passed (npm test, unaffected).

Needs a ComfyUI restart to verify live -- not yet done this session. Next in the composition-engine
  roadmap (docs are in the plan file, not checked into the repo): Plan 27 (extract shared custom io
  types to unblock the Source Profile <-> composition-assembly circular dependency), then Plan 28
  (Source Profile layer), then Plan 29 (composition-assembly, the big one). Co-Authored-By: Claude
  Sonnet 5 <noreply@anthropic.com>

- Extension.py split — shared composition-engine io types (Plan 27) → nodes/composition_types.py
  ([`030daf4`](https://github.com/frost-byte/fbTools/commit/030daf4eae7fa45853e76d48a3c5a7d0811b157e))

Moves the 5 custom io types that get cross-referenced between two layers that don't otherwise depend
  on each other: SourceProfileIOType (defined by the not-yet-moved Source Profile layer, consumed by
  SceneCastBuild in the composition-assembly layer) and CastIOType/H3RefplanType/CompositionIOType/
  SceneInstanceIOType (defined by composition-assembly, consumed by SourceProfileClipPrompt).
  Neither layer is a leaf relative to the other, so the one-directional "move the leaf, re-export it
  back" trick every other domain in this refactor series has used doesn't apply here -- this module
  is the neutral home both future layers (Plan 28: Source Profile, Plan 29: composition-assembly)
  can import from without a cycle.

Pure code motion, no behavior change: each type's own IO_TYPE string constant is only ever
  referenced at its own @io.comfytype() decorator site (confirmed via grep), so none needed
  re-exporting -- only the 5 class names themselves, re-exported back into extension.py in one
  import line since every one of them is still used by node classes that haven't moved yet.

extension.py: 6491 -> 6404 lines. Verified via the same methodology as every prior plan:
  io.ComfyNode class set (11, unchanged), route multiset (35, unchanged), get_node_list() names (76,
  byte-identical), and confirmed via a top-level-classdef diff that exactly these 5 classes (and
  nothing else) moved. ruff --select F821,F401 shows zero new findings and zero newly-resolved ones
  (expected -- pure marker classes, no exclusive helpers to leave dead). Full suite: 1283 passed/17
  skipped (pytest); npm test untouched (no JS in this plan).

Needs a ComfyUI restart to verify live -- not yet done. Plan 28 (Source Profile layer:
  SourceProfileLoad/SourceProfileDefine/SourceProfileList/ SourceProfileClipPrompt + 19 routes) is
  next, now unblocked by this plan; Plan 29 (composition-assembly, the big one) after that.
  Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- Extension.py split — Source Profile layer (Plan 28) → nodes/source_profiles.py
  ([`a4c2eb4`](https://github.com/frost-byte/fbTools/commit/a4c2eb4ffcfca9d234b28e0b97c400be94121be4))

Moves the 4 Source Profile node classes (SourceProfileLoad/SourceProfileDefine/
  SourceProfileList/SourceProfileClipPrompt) plus all 19 /fbtools/source_profiles/* REST routes into
  one new file -- the biggest single-domain move in this series so far (extension.py: 6404 -> 3883
  lines).

Two dependencies extension.py had only by virtue of its own top-to-bottom load order, not a real
  import, both confirmed via ruff F821 and fixed with genuine imports in the new file: `asyncio`
  (extension.py has no top-of-file import of it anywhere, only a mid-file one for unrelated LLM
  routes) and `_build_h3_refplan` (extension.py imports it ~2500 lines below where
  SourceProfileClipPrompt used to live). SourceProfileIOType/CastIOType/ H3RefplanType come from
  nodes/composition_types.py (Plan 27), confirming that module's reason for existing:
  SourceProfileClipPrompt needs CastIOType/ H3RefplanType (owned by the not-yet-moved
  composition-assembly layer) while SceneCastBuild (composition-assembly) needs SourceProfileIOType
  right back -- neither is a leaf relative to the other.

Caught two real regressions before committing, both via the existing test suite rather than
  assumption: - Removing nodes/llm_assistant's now-fully-dead names from extension.py's import would
  have silently dropped the only import of that module anywhere in the repo, breaking all 40 of its
  REST routes (same class of bug as nodes/media.py in Plan 26) -- fixed with a route-only `from
  .nodes import llm_assistant as _llm_assistant_routes` import. - A function-scope `from
  .utils.proxy_cache import ...` inside one route handler had the wrong dot-depth for its new home
  (one dot, needed two) -- caught by test_relative_imports_in_nodes_modules_resolve, invisible to
  the module-level dependency analysis since it's a local import inside a function body.

Repointed one test, the first genuine repoint in this whole refactor series (every prior plan needed
  none): test_source_profile_proxy_stream_route.py AST-parsed extension.py by hardcoded path to find
  the route handler's source; now points at nodes/source_profiles.py instead.

Also cleaned up 8 imports left genuinely dead by the move (verified via ruff diff against the
  pre-move baseline): SourceProfileRegistry, ENTITY_TYPES (source_profiles), build_prompt
  (source_profile_analysis), resolve_libber_refs, _route_llm, and -- surfaced only after the move,
  since nothing else in extension.py turned out to call them at module scope -- `math` and `time`.

Verified via the same methodology as every prior plan: io.ComfyNode class set (11 total, split 7
  extension.py / 4 new file, byte-identical combined), route multiset (35 total, split 16/19,
  byte-identical combined), get_node_list() names (76, byte-identical). ruff --select F821,F401
  diffed against the pre-move baseline: zero new findings. Full suite: 1283 passed/17 skipped
  (pytest, including the repointed test); 175 passed (npm test, unaffected -- no JS in this plan).

Needs a ComfyUI restart to verify live -- not yet done. Plan 29 (composition-assembly layer:
  SceneCompose/PromptAssemble/SceneCastLoad/ SceneCastBuild/CompositionLoad/PromptCompositionLoader
  + bundle routes + compositions/casts settings routes, the biggest and last piece) is next.
  Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **cast**: Extract list-based ordinal subject resolver
  ([`c89ccb2`](https://github.com/frost-byte/fbTools/commit/c89ccb29765d2e7f6c7a4a3b70fd73af5cbf1d36))

resolve_ordinal_subject was welded to Source Profile plumbing (profile -> clips -> clip -> subjects)
  around a loop that only needs an ordered list. Extract that loop as
  resolve_ordinal_from_list(subjects, pronoun, ordinal, id_key), leaving resolve_ordinal_subject as
  a thin wrapper with unchanged behaviour, so a Composition's subject roster can use the same
  matching when SceneCastBuild gains a Prompt Composition input. The JS mirror in
  scene_cast_build.js gets the same split.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **compose**: Extract background/outfit editors and a shared library store
  ([`276c49b`](https://github.com/frost-byte/fbTools/commit/276c49bee5e70c8e96a7ac49b00b2a46a9ce64ea))

Move the background and outfit modals out of composition_editor.js into background_editor.js /
  outfit_editor.js, backed by js/ui/library_store.js and an fbt:library-changed event, so the
  upcoming Assets tab can reuse them. Compose behaviour is unchanged.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **compose**: Remove the LLM Assistant section from the Compose sidebar
  ([`f4d1e70`](https://github.com/frost-byte/fbTools/commit/f4d1e70a35000a30c84b5fcaa8d1b0618c37f407))

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **composition**: Trim Prompt Composition Loader outputs to the five in use
  ([`923c7a9`](https://github.com/frost-byte/fbTools/commit/923c7a954e8ef497229e8ee91ecd7672849a4ada))

Keep prompt, composition_name, filename_prefix, lora_stack_data and h3_refplan. Drop concept_ids,
  model_type_used, reference_video, reference_images, audio_source, audio_file, audio_start_time and
  audio_duration: the reference media and audio travel in h3_refplan, and Run History now shows the
  rest - Model Type Used and Concept IDs rows (new summarize_composition_meta) alongside the
  existing per-cast-entry image/video/audio rows and LoRA list.

Removing outputs shifts the remaining output slot indexes, so saved workflows that link from this
  node need those links re-made.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **dataset-caption**: Move the dataset_caption domain out of extension.py into
  nodes/dataset_caption.py
  ([`cce8231`](https://github.com/frost-byte/fbTools/commit/cce8231065210e5488d2123905817cffc2b10a7c))

Continues Plan 1/11/17's extension.py -> nodes/ package split with the next incremental domain move
  (dataset_caption, the second of the remaining "later waves" after Plan 17's libber move), per the
  user's own pacing choice.

Moved, pure code motion, no behavior change: - shared constants (IMAGE_EXTENSIONS,
  DEFAULT_INSTRUCTION, CAPTIONER_OPTIONS, DEVICE_OPTIONS, DATASET_CAPTION_STATUS_ID) + 7 owned
  helpers (_collect_images, _txt_path, _read_caption, _write_caption, _resolve_relative_to,
  _resolve_dataset_input_directory, _resolve_dataset_output_directory) - 5 node classes:
  DatasetCaptioner, DatasetCaptionEditor, DatasetCaptionViewer, DatasetExportSummary,
  CaptionModelUnloader - all 5 /fbtools/dataset_caption/* REST routes

extension.py gains one import resolving the only external reference (get_node_list()'s 5 entries)
  unchanged. _directory_fingerprint, which sits right next to the moved helpers but belongs to
  unrelated scene-composition code, stays in extension.py untouched. One small drive-by cleanup:
  extension.py's top-level `from .captioner import caption_image, get_model, unload_model` dropped
  `unload_model`, now genuinely unused there since its only caller (CaptionModelUnloader) moved —
  the new module has its own `from ..captioner import unload_model`. Also added `from typing import
  Any` to the new module for an annotation that was already relying on `from __future__ import
  annotations` deferring evaluation (dormant pre-existing gap, now actually correct rather than just
  harmless).

Verified via the same AST-diff-against-git-HEAD method established in Plan 17: io.ComfyNode class
  set (76), route decorator multiset (158), and get_node_list()'s name set (72) are byte-identical
  before/after. ruff --select F821,F401 clean on both touched files (confirmed no new dead imports
  beyond the one intentionally dropped above).

tests/test_dataset_caption_api.py repointed at the new file (EXTENSION_PATH) - pure code motion
  means every existing source-text assertion still passes unchanged, verified directly. Learned
  while probe-testing tests/test_route_modules.py: unlike the standalone probe script used for Plan
  17's libber (which wrongly counted a transitive re-registration of llm_assistant's 40 routes
  alongside dataset_caption's own 5), the real test already filters registered routes by
  fn.__module__ before counting - confirmed by reading test_route_modules.py directly rather than
  trusting the simplified probe, then adding "dataset_caption": 5 to EXPECTED and letting the real
  test verify it.

extension.py: 17,338 -> 16,448 lines.

Tests: tests/test_dataset_caption_api.py's 5 tests pass unchanged against the new location;

tests/test_route_modules.py gains "dataset_caption": 5 in EXPECTED. Full suite: 1283 passed/17
  skipped (pytest, +1), 175 passed (npm test, unaffected).

Needs a ComfyUI restart to verify live (queued alongside Plan 17's pending libber restart): all 5
  Dataset Caption nodes still appear and execute, and the Dataset Caption viewer/editor panel still
  works end-to-end (list, save, recaption).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **libber**: Move the libber domain out of extension.py into nodes/libber.py
  ([`13f135b`](https://github.com/frost-byte/fbTools/commit/13f135b94374d4134272e1bd24a7df668496382e))

Continues Plan 1/11's extension.py -> nodes/ package split with the next incremental domain move,
  per the user's own pacing choice (one domain at a time, each its own commit+restart). libber was
  the first of the remaining "later waves" (domains carrying real node classes alongside their
  routes, unlike the route-only modules moved in Plan 11).

Moved, pure code motion, no behavior change: - class Libber (pure templating engine) - class
  LibberManager, class LibberApply (the two registered io.ComfyNode nodes) - class
  LibberStateManager (server-side singleton registry) - all 13 /fbtools/libber/* REST routes

extension.py gains one import (`from .nodes.libber import Libber, LibberManager, LibberApply,
  LibberStateManager`) resolving all 12 external call sites unchanged (SceneInfo, SceneSelect,
  StoryEdit, StorySceneBatch, StoryVideoBatch, ScenePromptManager, PromptComposer, two scene routes,
  SourceProfileClipPrompt, _apply_composition_libbers, PromptCompositionLoader) — none of that
  call-site code needed to change. No circular import: nodes/libber.py only imports from
  nodes/shared.py (default_libber_dir, already there), stdlib, and comfy_api.latest.

Verified, not just asserted: re-derived the current domain map fresh via Explore (Plan 1's old
  20,657-line baseline was stale after Plans 13-16's work; extension.py is 18,225 lines going in).
  AST-diffed the whole repo before/after against git HEAD: io.ComfyNode class set (76), route
  decorator multiset (158), and get_node_list()'s name set (72) are byte-identical. ruff --select
  F821,F401 clean on both touched files. Confirmed empirically (not just by the pattern
  run_tracking.py already sets) that nodes/libber.py imports successfully under
  test_route_modules.py's synthetic harness despite its io.ComfyNode subclasses.

extension.py: 18,225 -> 17,338 lines.

Tests: tests/test_route_modules.py gains "libber": 13 in EXPECTED, verified via a standalone

probe first. Full suite: 1282 passed/17 skipped (pytest, +1), 175 passed (npm test, unaffected).

Needs a ComfyUI restart to verify live: Libber Manager / Libber Apply nodes still appear and
  execute, a composition's %*:N%/%key% tokens still resolve via PromptCompositionLoader, and the
  Libbers sidebar tab still lists/loads/saves/deletes a libber.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **lora-stacks**: Move the lora stacks domain out of extension.py into nodes/lora_stacks.py
  ([`c19a2ce`](https://github.com/frost-byte/fbTools/commit/c19a2cedafd1f86a9e39091346ee49a2887cf758))

Continues Plan 1/11/17/18's extension.py -> nodes/ package split with the next incremental domain
  move (lora stacks, the third of the remaining "later waves"), per the user's own pacing choice.

Moved, pure code motion, no behavior change: - 2 wire types: LoraEntry, LoraStackData (io.comfytype)
  - 6 node classes: LoraEntryDefine, LoraStackCollect, LoraStackView (dead/unregistered, carried
  along inertly per this repo's established policy), LoraStackApply, WanVidLoraStack,
  LoraStackBuilder - shared constants (LORA_MODEL_TARGETS, LORA_AUDIO_KEYWORDS [dead],
  LORA_ENTRY_TYPE, LORA_STACK_DATA_TYPE) and their owned helpers (_lora_get_list,
  _lora_load_weights, _lora_entries_for_target, _lora_stack_to_json, _lora_json_to_stack,
  _lora_apply_ltx23, _lora_apply_standard, _lora_build_stack, _lora_build_wanvid,
  _LORA_BUILDER_ROWS, _lora_builder_inline_inputs)

NOT moved (deliberately, confirmed via exhaustive grep before touching anything): - lora *presets*
  (LoraPresetDefine/LoraPresetSelect/WanPresetDefine/WanPresetSelect) — they call
  SceneInfo.load_preview_assets() via _load_preset_scene_images(), and narrative-scene code hasn't
  been extracted yet; none of the moved stacks code references SceneInfo/story/scene at all,
  confirmed clean. - MultiLoraLoader — a separate, unrelated dead node elsewhere in extension.py;
  uses none of this domain's types/helpers.

Unlike libber/dataset_caption, this domain has zero REST routes but real cross-domain coupling:
  LoraStackData (the wire type), LORA_MODEL_TARGETS, and three helper functions (_lora_get_list,
  _lora_entries_for_target, _lora_build_wanvid, _lora_json_to_stack) are used by nine other node
  classes that stay in extension.py (SceneSelect, SceneLoraStackSave, SceneCreate, SceneUpdate,
  SceneOutput, SceneInput, StoryVideoBatch, SourceProfileClipPrompt, PromptCompositionLoader,
  ConceptDefine) — extension.py imports these 6 names back in one line. One-directional only
  (extension.py depends on the new module, never the reverse) and no more implicit than today: those
  9 classes are already defined earlier in extension.py than this domain was, so they already only
  resolve these names at call time inside define_schema()/execute(), never at class-definition time
  — moving the names into an imported module changes nothing about when they're resolved.

Verified via the same AST-diff-against-git-HEAD method as Plans 17/18: io.ComfyNode class set (76),
  route decorator multiset (158, unchanged since 0 routes moved), and get_node_list()'s name set
  (72) are byte-identical before/after. ruff --select F821,F401 clean on both touched files —
  confirmed none of the 6 re-imported names are flagged unused (i.e. all 9 outside classes really do
  still resolve them). No existing test references any lora-stacks class by name, so no test
  repointing was needed this time (unlike Plan 18's dataset_caption move).

extension.py: 16,448 -> 15,443 lines.

Tests: full suite unaffected structurally (this domain has zero routes, so no
  tests/test_route_modules.py EXPECTED entry needed — that test already skips route-less nodes/*.py
  files). Full suite: 1283 passed/17 skipped (pytest, unchanged), 175 passed (npm test, unaffected).

Needs a ComfyUI restart to verify live (queued alongside Plans 17/18's pending restart): LoRA Entry
  Define / Stack Collect / Stack Apply / WanVid LoRA Stack / LoRA Stack Builder all still appear and
  execute; a scene using a persisted LoRA stack (SceneSelect -> LoraStackCollect -> LoraStackApply,
  or via SceneLoraStackSave) still resolves correctly.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **narrative**: Move LoRA presets domain into nodes/narrative/lora_presets.py
  ([`83f4293`](https://github.com/frost-byte/fbTools/commit/83f4293e83ced9b91b79201908b9552ef4dcd6df))

Completes the next domain in the extension.py -> nodes/ package split (Plan 1). Moves the LoRA
  presets domain (distinct from LoRA stacks, already moved in Plan 19) into a new
  nodes/narrative/lora_presets.py, a third sibling to scene.py/story.py in the nodes/narrative/
  subpackage:

- 4 node classes: LoraPresetDefine, LoraPresetSelect, WanPresetDefine, WanPresetSelect — all 4
  registered in get_node_list(), no dead/unregistered class here - their 2 exclusive helpers:
  _load_preset_scene_images, _preset_scene_ui_and_images - 2 custom io types: LoraPresetList,
  PresetList - zero REST routes (a leftover "Preset routes" section header in extension.py already
  had an empty body before this move — nothing left to move there)

This was the domain nodes/lora_stacks.py's own docstring had flagged as blocked on SceneInfo not
  being extracted yet; that blocker cleared once Plan 20 moved SceneInfo out. Placed in
  nodes/narrative/ rather than alongside nodes/lora_stacks.py since the block's dependency profile
  points there squarely: it needs SceneInfo/default_pose_options (from .scene, single-dot sibling
  import, same pattern story.py established) and only one name from lora_stacks.py (LoraStackData,
  used purely as a wire-type annotation, not for any stack-building logic) — scene.py's own existing
  import block already had every other dependency this code needs.

Also: - fixed 4 local (function-scope) relative imports whose dot-depth needed to change now that
  this code lives one directory deeper: .utils.lora_presets and .utils.wan_presets, both x2
  (Define/Select each), -> ...utils.* - dropped 4 imports in extension.py left dead by the move
  (SceneInfo, default_pose_options, make_empty_image, and the bare `ui` module from comfy_api.latest
  — all verified via grep to have zero remaining callers, not assumed)

extension.py: 9,941 -> 9,494 lines. Verified via the same AST-diff-against-git-HEAD method as every
  prior plan: io.ComfyNode class set (76), route multiset (158, unchanged — zero routes moved),
  get_node_list() names (72) all byte-identical. ruff --select F821,F401 on both files matches the
  pre-existing baseline exactly (diffed). Full suite: 1283 passed/17 skipped (pytest, unchanged),
  175 passed (npm test, unchanged — no JS touched; frontend dispatches on ComfyUI node type-name
  strings, not Python import paths).

Needs a ComfyUI restart to verify live (queued alongside the pending restart verification already
  done for Plans 17-21, with 22 and this plan still pending in the same batch).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **narrative**: Move Scene CRUD node classes + their 5 REST routes into nodes/narrative/scene.py
  ([`9b6055f`](https://github.com/frost-byte/fbTools/commit/9b6055f4a8287c4b60cb8641c1f4e6aa2d21427d))

Completes the Scene domain extraction started in Plan 20. Appends to the same
  nodes/narrative/scene.py file (no longer needs a cross-file import for SceneInfo/MaskType/etc.
  since the CRUD classes are now co-located with them):

- 10 node classes: SceneSelect, SceneWanVideoLoraMultiSave (dead, unregistered in get_node_list —
  carried along inertly, not fixed/deleted), SceneLoraStackSave, SceneCreate, SceneUpdate,
  SceneView, SceneMaskDefinition, SceneOutput, SceneSave, SceneInput - their 3 exclusive
  module-level helpers: save_lora_stack, and the dead load_loras/save_loras (zero live callers, tied
  only to the dead class) - DictType, a small "DICT" wire type used exclusively by SceneSelect - all
  5 /fbtools/scene/* REST routes (process_compositions, get_scene_prompts, save_scene_prompts, list,
  thumbnail) — these were genuinely interleaved with 5 /fbtools/story/* routes in source order, so
  removal was 5 separate small edits rather than one contiguous block; the Story routes stay in
  extension.py untouched

Also: - fixed 4 local (function-scope) relative imports whose dot-depth needed to change now that
  this code lives one directory deeper (nodes/narrative/ vs. the package root) — the one place this
  move wasn't pure text motion - added `from aiohttp import web` to nodes/narrative/scene.py, missed
  on the first pass and caught by ruff (the 5 routes all return web.json_response/ web.FileResponse)
  - dropped 18 imports in extension.py left dead by the move (ImageScaleBy, 3 lora_stacks helpers,
  default_libber_dir, image_resize_ess, load_json_file, save_json_file, the whole utils.pose import
  line, plus MaskType/ MaskDefinition/RGB/DictType/SceneWanVideoLoraMultiSave/
  _migrate_loras_json_to_stack dropped from the re-export list itself, since extension.py has zero
  remaining callers for any of them once their sole consumer moved) - noted (not fixed) a real but
  out-of-scope test-infrastructure gap: unlike every other route module, nodes/narrative/scene.py
  transitively imports utils/images.py, which does `import torchvision...` at module level —
  test_route_modules.py's per-module route-count fixture only mocks folder_paths/server, not
  torch/torchvision, so it can't import this module standalone. Documented with a comment in the
  EXPECTED dict rather than expanding that fixture's mocking, which is beyond a pure code-motion
  plan's scope. The module is still covered by test_relative_imports_in_nodes_modules_resolve and
  test_extension_imports_every_route_module, and by the full test suite.

extension.py: 14,313 -> 11,811 lines. Verified via the same AST-diff-against-git-HEAD method as
  every prior plan: io.ComfyNode class set (76), route multiset (158), get_node_list() names (72)
  all byte-identical — nothing added/removed, only relocated. ruff --select F821,F401 on both files
  matches the pre-existing baseline exactly (diffed, not just counted) after fixing the missing
  `web` import and the 18 dead imports above. Full suite: 1283 passed/17 skipped (pytest,
  unchanged), 175 passed (npm test, unchanged — no JS touched).

Needs a ComfyUI restart to verify live (queued alongside Plans 17-20's pending restart).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **narrative**: Move SceneInfo/MaskType core out of extension.py into nodes/narrative/scene.py
  ([`56626e1`](https://github.com/frost-byte/fbTools/commit/56626e17a2804a44a4cfb8299e5f83d91692cbfc))

Extracts the SceneInfo/MaskType/MaskDefinition core (RGB, MaskType, MaskDefinition, load_masks_json,
  save_masks_json, SceneInfo, _migrate_loras_json_to_stack, load_lora_stack, resolve_mask_key,
  default_depth_options/default_pose_options/ default_mask_options) into nodes/narrative/scene.py —
  the first nodes/ subpackage, per Plan 1's original layout sketch.

Scope deliberately narrower than a full Scene-domain move: the 10 Scene CRUD node classes and their
  5 /fbtools/scene/* routes stay in extension.py for now (deferred to a follow-up plan) and import
  SceneInfo/MaskType/etc. back from the new module, the same one-directional pattern already proven
  for nodes/lora_stacks.py. This also unblocks lora-presets, which only ever needed
  SceneInfo.load_preview_assets and the default_*_options helpers.

Also: - moved the fully generic get_subdirectories/_directory_fingerprint helpers (used across
  Scene, Story, and ScenePromptManager) into nodes/shared.py - dropped 9 imports in extension.py
  left dead by the move (dataclass, Enum, BaseModel, ConfigDict, Dict, Tuple, generate_thumbnail,
  save_image_comfyui, select_text_by_action) — their only user was the code that just moved - fixed
  tests/test_route_modules.py's relative-import-resolution check, which globbed nodes/*.py
  non-recursively and silently never checked the new subdirectory module; now walks nodes/**/*.py
  and computes each file's allowed import depth relative to its own nesting under nodes/, not a
  hardcoded flat nodes/ vs. package-root split - repointed tests/test_mask_integration.py and
  tests/test_nlf_integration.py's SceneInfo/default_pose_options imports to the new module path

extension.py: 15,388 -> 14,229 lines. Verified via AST diff against git HEAD: io.ComfyNode class set
  (76), route multiset (158), and get_node_list() names (72) all byte-identical before/after —
  nothing moving in this plan is a node class or route. ruff --select F821,F401 clean relative to
  the pre-existing baseline (39 errors, unchanged — the 9 newly-dead extension.py imports above are
  fixed, not left as new debt). Full suite: 1283 passed/17 skipped (pytest, unchanged), 175 passed
  (npm test, unchanged — no JS touched).

Needs a ComfyUI restart to verify live (queued alongside Plans 17-19's pending restart).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **narrative**: Move ScenePromptManager/PromptComposer into nodes/narrative/scene_prompts.py
  ([`c1a6ab1`](https://github.com/frost-byte/fbTools/commit/c1a6ab128a1618b95d154a222459b61538f8ab64))

First step of the composition-engine roadmap (see the plan file's ROADMAP section for the full
  multi-agent mapping of the remaining 25-class cluster). ScenePromptManager/PromptComposer were
  discovered during Plan 24's exploration to be neither part of the 13-class "grab-bag" nor the
  composition-engine cluster proper -- a small, fully independent "Scene Prompt Management" pair
  that just happened to sit physically inside the composition-engine's line range. Moving them now
  clears them out of that region ahead of the larger, much more tangled composition-engine work
  still to come.

Confirmed zero dependency on nodes/narrative/scene.py despite the shared "Scene" category and
  physical proximity to Scene/Story code before this move: PromptComposer's scene_info input is a
  duck-typed io.Custom("SCENE_INFO") generic wire type with no backing class to import. Real
  dependencies are PromptCollection (from prompt_models.py, used throughout ScenePromptManager) and
  LibberStateManager (already-moved nodes/libber.py, Plan 17).

Also: - dropped 2 imports in extension.py left dead by the move (default_scenes_dir,
  get_subdirectories from nodes/shared.py), each verified via grep to have zero remaining callers
  before removing - carried along, unfixed, one pre-existing dead import (PromptMetadata --
  confirmed via git history it was already unused in extension.py before this move, imported
  alongside PromptCollection but never itself referenced)

extension.py: 8,095 -> 7,763 lines. Verified via the same AST-diff-against-git-HEAD method as every
  prior plan: io.ComfyNode class set (76), route multiset (158, unchanged -- zero routes moved),
  get_node_list() names (72) all byte-identical. ruff --select F821,F401 on both files matches the
  pre-existing baseline exactly (diffed). Full suite: 1283 passed/17 skipped (pytest, unchanged),
  175 passed (npm test, unchanged -- frontend dispatches on ComfyUI node type-name strings, and no
  test imports either class from extension.py).

Needs a ComfyUI restart to verify live (Plans 21-24 already restart-verified working this session;
  this is the next one queued).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **narrative**: Move Story domain node classes + their 5 REST routes into nodes/narrative/story.py
  ([`617585a`](https://github.com/frost-byte/fbTools/commit/617585a6c0060e015717a0f984eb6ce608d5cda0))

Completes the next domain in the extension.py -> nodes/ package split (Plan 1), following the same
  pattern used for libber, dataset_caption, lora_stacks, and the Scene domain. Moves into a new
  nodes/narrative/story.py, a sibling to nodes/narrative/scene.py in the same subpackage:

- 9 node classes: StoryCreate, StoryEdit, StoryView, StorySceneBatch, StoryScenePick, StorySave,
  StoryLoad, StorySceneImageSave, StoryVideoBatch — all 9 registered in get_node_list(), no
  dead/unregistered Story class exists (unlike every prior domain) - their 2 exclusive module-level
  helpers: get_available_stories, build_positive_prompt - all 5 /fbtools/story/* REST routes (load,
  job_ids, list, regenerate_thumbnails, save) — now cleanly contiguous in the source (the old
  interleaving with Scene routes left only comment-only gaps after Plan 21)

Introduces one new pattern this session's prior moves hadn't needed yet: a same-subpackage sibling
  import (`from .scene import SceneInfo, load_masks_json, resolve_mask_key, default_depth_options,
  default_pose_options, load_lora_stack`) rather than reaching back through extension.py's
  re-export, since story.py and scene.py both live in nodes/narrative/. Two of the moved routes
  (story_load, story_regenerate_thumbnails) already called into Scene-domain code; both become
  same-package calls through this same import.

Also: - fixed 2 local (function-scope) relative imports whose dot-depth needed to change now that
  this code lives one directory deeper (nodes/narrative/ vs. the package root):
  .utils.scene_image_save and .utils.story_video, both -> ...utils.scene_image_save /
  ...utils.story_video - dropped 14 imports in extension.py left dead by the move
  (LORA_MODEL_TARGETS, 4 names re-exported from nodes/narrative/scene.py, default_stories_dir,
  _directory_fingerprint, Path, the whole `from .story_models import SceneInStory, StoryInfo,
  save_story, load_story` line, load_prompt_json, update_ui_widget, uuid) — each verified via grep
  to have zero remaining callers in extension.py before removing, not assumed - carried along,
  unfixed, 4 pre-existing dead re-exports from utils/story_video.py (find_scene_image,
  generate_video_filename, resolve_video_prompt, build_video_descriptor) — confirmed via git history
  these were already unused in extension.py before this move, not something introduced by it -
  extended test_route_modules.py's EXPECTED-dict exclusion comment (from Plan 21) to cover
  narrative.story too: verified directly (not assumed) that it hits the identical torch/torchvision
  transitive-import limitation via its new .scene sibling import, the same real, documented,
  out-of-scope test- fixture gap as narrative.scene

extension.py: 11,811 -> 9,941 lines. Verified via the same AST-diff-against-git-HEAD method as every
  prior plan: io.ComfyNode class set (76), route multiset (158), get_node_list() names (72) all
  byte-identical. ruff --select F821,F401 on both files matches the pre-existing baseline exactly
  (diffed). Full suite: 1283 passed/17 skipped (pytest, unchanged), 175 passed (npm test, unchanged
  — no JS touched).

Needs a ComfyUI restart to verify live (queued alongside the pending restart verification already
  done for Plans 17-21).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **nodes**: Extract run-tracking into nodes/ package (phase 1 of extension.py split)
  ([`f14106d`](https://github.com/frost-byte/fbTools/commit/f14106d194ab739552511df0cc254d08041ee772))

extension.py has grown to ~20,700 lines covering ~15 unrelated domains, making independent commits
  to one feature constantly interleave with unrelated diffs in the same file. This begins a phased
  split into a nodes/ package organized by domain, starting with the most self-contained piece:
  run-tracking.

Moved to nodes/run_tracking.py: RunMetaCapture, JobCompleteNotifier, the node-output auto-tracker
  (execution.execute() monkeypatch + on_prompt handler), and the /fbtools/run_tracker/* routes. This
  was deliberately done first because it's the one piece with import-time side effects (the
  execute() patch and add_on_prompt_handler registration) — validating that pattern works isolated
  in its own module before anything else depends on it.

Also created nodes/shared.py (EXTENSION_PREFIX, prefixed_node_id) to break what would otherwise be a
  circular import: the moved node classes need prefixed_node_id(), which previously lived in
  extension.py itself.

Fixed a real bug this move would otherwise have introduced silently: JobCompleteNotifier's
  notification directory is computed relative to __file__ with a fixed number of ".." segments (no
  test could catch this — it's a filesystem path, not test-covered behavior). Moving the file one
  directory deeper without adjusting the count would have silently redirected notifications to the
  wrong path. Verified the corrected path resolves identically to before the move.

Updated tests/test_widget_name_contracts.py to scan nodes/**/*.py alongside extension.py — required
  now, not just future-proofing, since RunMetaCapture/JobCompleteNotifier's io.String.Input()
  widgets moved out of extension.py's text in this same commit.

Verified via static regression guards (node-class count and @routes.* multiset both match the
  pre-migration baseline exactly: 75 classes, 148 routes) plus the full test suite (1070 passed).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **nodes**: Move backgrounds/presets and media routes to their own modules
  ([`c878882`](https://github.com/frost-byte/fbTools/commit/c8788822a38888de5cbb0dfa7abcbd2a366dc501))

nodes/backgrounds_presets.py (11 handlers) and nodes/media.py (6 handlers plus the media extension
  helpers; _audio_get_list is re-imported by the remaining nodes). Pure code motion.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **nodes**: Move registry, outfit, LoRA-info and prompt-collection routes out of extension.py
  ([`7688ac6`](https://github.com/frost-byte/fbTools/commit/7688ac684f30a0c1ae0f781d476e241443c5e944))

nodes/registry_api.py (concepts, scene templates, subjects, casts), nodes/outfits.py (with SAM2
  extraction), nodes/lora_info.py, nodes/prompt_collections.py. extension.py imports each for its
  route registration, guarded by a test that every route module is imported. Pure code motion.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **nodes**: Move shared path/status/routes helpers and the reload counters to nodes/shared.py
  ([`c1be0ec`](https://github.com/frost-byte/fbTools/commit/c1be0ec5a8f44e72e132e89431e8c7d9f26e486a))

user_data_dir, the default_* registry path helpers, send_status_update, the routes singleton and the
  seven reload counters (now a small registry read via reload_counter()/bumped via bump_reload())
  leave extension.py so upcoming domain modules can import them without a circular dependency. No
  behaviour change; package-root derivation is covered by tests.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **nodes**: Move the LLM assistant routes and inference-routing helpers to nodes/llm_assistant.py
  ([`e38cbbc`](https://github.com/frost-byte/fbTools/commit/e38cbbcbe091dd6b68c20aeabe6f67859ad003fb))

40 llm/modal/unsloth/vlm handlers plus _active_backend, _route_llm and the _run_*_inference helpers
  (re-imported by the remaining nodes and routes). Pure code motion; a stub-based smoke test checks
  the module imports and registers all 40 routes.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **nodes**: Split 13-class extension.py grab-bag into 5 flat nodes/*.py files
  ([`0e9d149`](https://github.com/frost-byte/fbTools/commit/0e9d149bf2c8621a7f98879a41959fa1337a957d))

Completes the "everything else" cleanup queued up after Plans 17-23 finished every domain that was
  blocking on something. Splits 13 classes that were never actually part of the composition-engine
  system (they just happened to be physically interspersed among it) into 5 new flat files, grouped
  by their own existing category="..." declarations rather than inventing new groupings:

- nodes/compositing.py: SubjectLayerDefine, SubjectCompositor (+ their shared SUBJECT_LAYER custom
  wire type and BG_MODELS/OUTPUT_MODES constants) — category "compositing" -
  nodes/image_processing.py: SAMPreprocessNHWC, TailEnhancePro, TailSplit, OpaqueAlpha,
  MaskProcessor — categories "Preprocessing"/"Video"/"Image Processing", all pixel-pipeline work -
  nodes/qwen_conditioning.py: FBTextEncodeQwenImageEditPlus, QwenAspectRatio — paired despite
  differing declared categories ("conditioning" vs. "Image Processing") since both are
  Qwen-model-specific - nodes/audio.py: AudioFixShape — its own "Audio" category - nodes/utility.py:
  SubdirLister (live), plus dead MultiLoraLoader and NodeInputSelect (both unregistered in
  get_node_list() — carried along inertly, not fixed/deleted, matching this session's established
  dead-code policy)

Unlike every prior plan, these 13 classes are NOT contiguous with each other or with any single
  unmoved domain — each sits between other, unrelated composition-engine code that must stay in
  extension.py, so this was 13 separate small extractions (14 counting the shared SUBJECT_LAYER
  type/ constants) rather than a handful of contiguous blocks. Zero REST routes and zero
  composition-engine coupling confirmed for all 13.

Also: - hoisted MaskProcessor's one function-scope local import (`from .utils.images import
  mask_remove_holes, ...`) to module level in its new home, since utils/images.py is already
  imported eagerly there anyway for TailEnhancePro - dropped MultiLoraLoader/NodeInputSelect from
  the re-export line back into extension.py entirely (both dead — nothing there calls them, matching
  how LoraStackView/SceneWanVideoLoraMultiSave were handled in Plans 19/21); they still exist,
  defined, in nodes/utility.py - carried along, unfixed, one pre-existing dead import
  (utils/subject_compositor.tensor_to_pil) — confirmed via git history it was already unused before
  this move - dropped 27 other imports in extension.py left genuinely dead by the move
  (node_helpers, comfy.utils.common_upscale, inspect.cleandoc, torch.nn.functional, typing.Optional,
  the whole utils.images proc_*/_HAS_* set exclusive to TailEnhancePro, utils.subject_compositor's
  whole import block, utils.util's get_workflow_all_nodes/listify_*/node_input_details set,
  utils.images.find_nearest_qwen_aspect_ratio), each verified via grep to have zero remaining
  callers before removing, not assumed

Discovered but explicitly out of scope: ScenePromptManager/PromptComposer form their own small
  "Scene Prompt Management" domain, neither grab-bag nor composition-engine — flagged for a future
  plan, not touched here.

extension.py: 9,494 -> 8,095 lines. Verified via the same AST-diff-against-git-HEAD method as every
  prior plan: io.ComfyNode class set (76), route multiset (158, unchanged — zero routes moved),
  get_node_list() names (72) all byte-identical. ruff --select F821,F401 on all 6 touched/new files
  matches the pre-existing baseline exactly (diffed, not just counted). Full suite: 1283 passed/17
  skipped (pytest, unchanged), 175 passed (npm test, unchanged — frontend dispatches on ComfyUI node
  type-name strings, not Python import paths, and tests/test_subject_compositor.py replicates
  node-level logic rather than importing the classes, confirmed unaffected).

Needs a ComfyUI restart to verify live (queued alongside the pending restart verification already
  done for Plans 17-21, with 22/23/this plan still pending in the same batch).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **scene-cast**: Move duration/multiplier controls above timeline
  ([`1ec29f4`](https://github.com/frost-byte/fbTools/commit/1ec29f4bd8da9eec5fc4ab9f42ea67a2acffbabc))

Duration multiplier buttons and calculated duration label now appear above the canvas rather than
  below the nav row, keeping them away from the '← m/n → Segment m' label they were crowding.

Also bumped the clips section top margin/padding from 6/5px to 8/8px so there is a clearer visual
  gap between the Add entry button and the duration controls, and changed the dur-row's margin from
  margin-top to margin-bottom so the gap sits between it and the canvas.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

### Testing

- Genericize remaining local-content-adjacent placeholder text
  ([`c264151`](https://github.com/frost-byte/fbTools/commit/c264151473dfd36951f958989ba9f54780dbf556))

Replace a UI placeholder example and a leaked local story-content search token in a standalone debug
  script with generic equivalents, following up on the earlier local-name genericization pass.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

- **nodes**: Add example workflows for image-processing grab-bag nodes
  ([`efcf58c`](https://github.com/frost-byte/fbTools/commit/efcf58c1928be11e7e21400163baea372affa973))

Six small, focused workflows (SubjectLayerDefine/SubjectCompositor, MaskProcessor, QwenAspectRatio,
  SAMPreprocessNHWC, OpaqueAlpha, TailSplit+TailEnhancePro), each verified live against a real
  ComfyUI instance with a real-render thumbnail, plus synthetic SFW placeholder demo media. Kept one
  node (or one natural pairing) per file rather than one combined graph, since comfy-action treats a
  workflow file as one pass/fail unit with no per-node assertions — bundling nodes together would
  make one failure obscure which node actually broke. Building these against real output — not just
  reading the code — is what surfaced the three bugs fixed in the previous commit.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>


## v1.27.0 (2026-09-09)

### Bug Fixes

- **llm-client**: Restore thinking mode for Qwen3 GGUF, strip <think> block post-generation
  ([`ce65237`](https://github.com/frost-byte/fbTools/commit/ce65237971c61c4fbba1100f0735ea9f91649ffb))

Disabling thinking degraded output quality significantly. Revert to create_chat_completion() with
  thinking enabled, but triple the token budget when a thinking-mode template is detected so the
  model has room to close </think> before hitting the cap. The </think> strip then removes the
  reasoning preamble and returns only the final answer to callers.

Also clears has_thinking_template / chat_template keys in unload_model().

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **scene-cast**: Remove 10px default DOM widget margin causing right-side gap
  ([`c7fc903`](https://github.com/frost-byte/fbTools/commit/c7fc903f80ebe6e3eee8499fa52637abab88d8e0))

ComfyUI's BaseDOMWidgetImpl applies DEFAULT_MARGIN=10 on each side of every DOM widget, leaving 20px
  of unused space. Passing margin:0 in addDOMWidget options gives the wrap div full node width; the
  existing padding on .fbt-scb-wrap (4px 6px 6px, box-sizing: border-box) provides the visual inset
  instead. Same fix applied to SourceProfileClipPrompt's clip selector widget.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **scene-cast-build**: Center clip nav arrows around label
  ([`3ed1e9b`](https://github.com/frost-byte/fbTools/commit/3ed1e9b79ea1dece5511717cf78d4f7cf46df723))

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **scene-cast-build**: Correct timeline hit-test coordinates at non-100% canvas zoom
  ([`92b0e9f`](https://github.com/frost-byte/fbTools/commit/92b0e9f84df18f9ff9eea07110a267127c4ccf9f))

Mouse coords from getBoundingClientRect() are in screen pixels (scaled by canvas zoom), but hit
  zones were computed using canvas.offsetWidth (layout pixels, unscaled). At 85% zoom this caused
  every segment to register ~15% too far left, highlighting the segment to the left of the one
  moused over.

Fix: use rect.width (from the same getBoundingClientRect call already made in each handler) for hit
  zone boundaries — rect.width is in screen pixels and matches the mouse x coordinate space exactly.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

### Features

- **llm**: Read mmproj metadata for vision handler, add vramAnalysis API
  ([`2552eea`](https://github.com/frost-byte/fbTools/commit/2552eea8627349f5b3323269a86fc12af8cb6c9a))

- llm_scanner: read clip.projector_type from mmproj GGUF metadata instead of filename heuristics;
  maps qwen3vl_merger → MTMDChatHandler, qwen2.5vl_merger/qwen2vl_merger → Qwen25VLChatHandler,
  gemma3 → Gemma4ChatHandler; falls back to filename heuristics when unreadable - js/api/llm.js: add
  vramAnalysis() client method for /llm/vram_analysis - source_profile_analysis: update clip prompt
  wording for placeholder instructions; fix test assertion to match

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **scene-cast**: Add clip duration multiplier (1×–4×) to SceneCastBuild timeline
  ([`0019326`](https://github.com/frost-byte/fbTools/commit/0019326665799a0a96da809eb71a2ef6e2c1e07c))

- SceneCastBuild: add `clip_duration_multiplier` Int input (1–4, hidden, JS-managed) and matching
  Int output (pass-through) so it can be wired to SourceProfileClipPrompt - SourceProfileClipPrompt:
  accept optional `clip_duration_multiplier` input; apply it to clip_duration_frames:
  ceil(duration_s × multiplier × 24)

- Timeline UI: add [1×|2×|3×|4×] button group below nav row with duration display; native duration
  shown as ss.mm format (e.g. 3.42s); when multiplier > 1 shows "3.42s → 6.84s" to indicate scaled
  output length - CSS: add .fbt-scb-dur-row, .fbt-scb-mult-btn, .fbt-scb-dur-label styles

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **scene-cast-build**: Replace clip dropdown with canvas timeline navigator
  ([`8147d67`](https://github.com/frost-byte/fbTools/commit/8147d675028fd2c1deea147c37daccf6a0180c06))

Swap the <select> widget for a canvas-based timeline that mirrors the Source Profile editor's visual
  style:

- Clips drawn as proportional colored bands (by start/end time; equal widths when time data is
  absent) - Click any segment to select it - ← / → nav buttons below the canvas cycle through clips
  with wrap-around - Active segment: brighter fill + triangle indicator above the band - Hover:
  lighter highlight, pointer cursor - Nav label shows "N/total · clip label" for quick orientation -
  ResizeObserver redraws the canvas when the node is resized - _activeClipId replaces the
  clipSel.value reference pattern throughout

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **settings**: Add Settings tab; add H3 max output frames clamp
  ([`cc65ec5`](https://github.com/frost-byte/fbTools/commit/cc65ec51fe46721f158ff53e7925acb9606c5131))

- New Settings tab in sidebar (after Inspect): consolidates all extension preferences in one place.
  Compose-tab settings section removed from composition_editor.js; composer listens for
  `fbt:settings-changed` to keep its in-memory settings object fresh. - Settings: libber delimiter,
  default speech pace, audio processing, vocal isolation (moved from Compose tab) + new H3 model
  section. - H3 max frames (default 360 = 15s × 24fps, 0 = unclamped): persisted server-side via
  /fbtools/compositions/settings and mirrored to localStorage so nodeCreated hooks can read it
  synchronously. - SourceProfileClipPrompt: add optional `max_clip_frames` Int input (default 360);
  clamp applied after duration × multiplier calculation. nodeCreated hook pre-fills the widget from
  the global setting.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **source-profiles**: Combined multi-pass analysis, describe history, tabbed history UI
  ([`70f3fb9`](https://github.com/frost-byte/fbTools/commit/70f3fb99c78adaf0542629e58c7cd4a59ec2db5e))

Focus Pass: - build_multi_prompt() in source_profile_analysis.py generates a single combined VLM
  prompt covering all selected pass types, asking the model to return subjects across all categories
  in one flat JSON array - /analyze endpoint accepts pass_types list (backward-compat with single
  pass_type string); selects multi_prompt vs single-pass prompt accordingly - history entry stores
  pass_types list alongside the joined pass_type label - _runAnalysis in source_profile_editor.js
  sends one request with pass_types instead of looping per type

Describe history: - /describe_clip now calls append_history_entry with pass_type="describe_clip"
  plus clip_start, clip_end, and action extra fields - append_history_entry accepts **extra_fields
  merged into the entry dict

Tabbed history UI: - "Previous runs" section now has Analyze / Describe tabs - Analyze tab: shows
  focus-pass runs (existing behavior), pass_types list rendered as "People + Objects" labels -
  Describe tab: shows describe_clip runs with clip time range, result text, and collapsible
  prompt-used section

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **source-profiles**: Multi-frame describe_clip + Qwen3 thinking suppression
  ([`525983b`](https://github.com/frost-byte/fbTools/commit/525983b1a80ac71de3881a60c07896b854afeb38))

describe_clip now samples up to max_frames (default 5) spread across the clip range via
  _spa_extract_clip_frames / _run_vision_inference_clip, matching the analyze endpoint's
  frame-extraction behaviour. Gemini Flash falls back to a contact sheet; non-video media falls back
  to single midpoint frame.

UI adds Max frames and Every Nth controls to the Describe settings block (stored on the profile as
  describe_max_frames / describe_select_every_nth).

llm_client: detect GGUF models with Qwen3-style thinking templates at load time; render the template
  manually with enable_thinking=False via Jinja2 and call create_completion() directly, bypassing
  the create_chat_completion() path that has no way to pass chat_template_kwargs. The </think> strip
  remains as a fallback for the create_chat_completion() path.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **source-profiles**: Split clip at cursor, multi-pass Focus analysis
  ([`1c89a01`](https://github.com/frost-byte/fbTools/commit/1c89a01fc3d8c7c503571de7b2f46eb187e0112a))

- Add ✂ split button to clip nav bar: splits the current clip at the video player's current time;
  validates the cursor is within the clip's start–end range before splitting (reuses original id for
  the first half, generates a new id for the second half) - Convert Focus Pass single-pill selection
  to multi-select checkboxes so multiple pass types can be chosen and submitted in one batch run -
  _runAnalysis loops over all checked pass types sequentially, merges all candidates with a
  _pass_type tag for display - _renderCandidates shows a faint pass-type badge next to each
  candidate's entity-type badge when results span multiple passes - _updateDefaultPreview renders
  each selected pass type's default prompt separated by dividers when more than one is checked

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

### Refactoring

- **llm**: Unified active-backend routing for all inference calls
  ([`bcb11e0`](https://github.com/frost-byte/fbTools/commit/bcb11e0acc3ca95041dd5ccd484cfce4f9406ad6))

Replace the fragmented _route_vision/_route_text/direct-client pattern with a single
  _active_backend() + _route_llm() system:

- _active_backend(): single source of truth returning 'unsloth', 'modal', or 'local' — eliminates
  the implicit priority chain - _route_llm(): one async function covering text + vision + video for
  all backends; replaces _route_vision and _route_text - _run_vision_inference / _run_text_inference
  / _run_vision_inference_clip: all use _active_backend() for 'auto' captioner_type so
  source-profile calls and bundle-editor calls resolve to the same backend -
  /fbtools/outfits/analyze_media and the background-describe endpoint migrated from direct
  _llm_client.generate() to _route_llm() - Mutual exclusion: activating Unsloth deactivates Modal
  and vice versa, ensuring exactly one backend is active at a time - VRAM guard in _generate_gguf()
  returns a clean error before the C-level SIGSEGV/SIGABRT when vision inference would OOM

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **styles**: Migrate JS-injected CSS to external stylesheets
  ([`4444e0f`](https://github.com/frost-byte/fbTools/commit/4444e0f7a4a620a97534e15af1b9a2a030423cef))

Extract all <style> tag injections from 8 JS files into dedicated CSS files under js/styles/. Add a
  CSS token system (vars.css) that provides a single authoritative set of design tokens mapped onto
  ComfyUI's own --p-* / --comfy-* variables with dark-mode fallbacks.

New structure: js/styles/shared/vars.css — CSS custom properties (colors, radii, z-index)
  js/styles/shared/status.css — semantic .fbt-st-* classes js/styles/nodes/*.css — per-node widget
  styles js/styles/ui/*.css — per-panel editor styles

All component CSS is loaded via @import in the existing style.css, which fb_tools.js already links
  as a <link> element — no loader changes needed.

Hardcoded hex status colors in sceneUpdateStatus.js and dataset_caption_status.js replaced with
  var(--fbt-*) references.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.26.0 (2026-09-06)

### Bug Fixes

- **unsloth**: Add missing top-level httpx import
  ([`9d2a80c`](https://github.com/frost-byte/fbTools/commit/9d2a80c71124976a54e2e2a11e0c5660566c8394))

_post_once() and _probe_warmth() used httpx but the module-level import was missing — only
  _call_with_retry() and health_check() had inline try/import guards. Moved httpx to top-level and
  removed the now-redundant inline imports.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Add user root to nginx; proxy docs-assets; fix SPA 500 errors
  ([`2747172`](https://github.com/frost-byte/fbTools/commit/274717293e235517c272378b5c05ac232fef76f8))

nginx workers default to www-data on Ubuntu/Debian, which cannot read Python package paths — causing
  try_files to return 500 for all SPA routes. Add user root to both placeholder and proxy configs
  since we run in a container.

Also add docs-assets to _API_PREFIXES so Studio's Swagger UI assets are proxied to FastAPI instead
  of falling through to the SPA catch-all.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Add vision/video support for Qwen3.8-27B and Flash-Next
  ([`8d2b5b6`](https://github.com/frost-byte/fbTools/commit/8d2b5b6de66438311c8747ad2294d9d864abdcb2))

Qwen3.8-27B and Flash-Next are native vision-language models; the 8B is text-only. Per-endpoint
  vision/native_video flags gate the image and video_frames paths in generate(). _encode_image() and
  _build_vision_content() produce OpenAI-format image_url content blocks for llama-server.
  extension.py routes _run_vision_inference() and _run_vision_inference_clip() through the unsloth
  client when active and the endpoint supports vision; native_video path sends raw frames, text-only
  endpoints fall back to a contact sheet. llm_panel.js persists
  unslothActive/unslothVision/unslothLabel to localStorage and exports activeBackendSupportsVision()
  for downstream consumers.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Bust app_status cache on deploy/undeploy so UI reflects new state
  ([`992e2f7`](https://github.com/frost-byte/fbTools/commit/992e2f7b7443bf76a5d50cab4981268cc79b4985))

The 60 s cache introduced in the prior commit caused the App checklist row to show stale "not
  deployed" after a successful Deploy App action, since _fetchSetupStatus() was called immediately
  but hit the cached result. Clear _app_status_cache on any deploy() success or undeploy()
  completion so the next setup_status poll sees the real state.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Cache app_status for 60 s to reduce Modal API calls
  ([`c92e877`](https://github.com/frost-byte/fbTools/commit/c92e877f730e56b260b7e83508d3c790a815e36b))

The setup-status poll fires every 15 s while setup is incomplete. Each call previously ran \`modal
  app list\` as a subprocess, generating Modal API traffic every 15 s throughout the bootstrap
  window (up to 30 min = ~120 unnecessary calls).

app_status() now caches its result for 60 s using monotonic time, so the Modal API is hit at most
  once per minute during polling. force=True is available for callers that need a fresh check (not
  currently needed — deploy/undeploy routes don't call app_status).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Cancel warmup thread on deactivate to prevent stale retry requests
  ([`c4b3a47`](https://github.com/frost-byte/fbTools/commit/c4b3a47345ece6f3798cf5becc9770ef1ab89d7f))

deactivate() now sets a _warmup_cancel Event that the _call_with_retry loop checks at the top of
  each iteration. Previously, the warmup thread kept retrying Modal after deactivate, and when the
  user stopped containers and re-activated, the old thread's next retry fired concurrently with the
  new activation — causing Modal to spin up two containers.

_start_warmup() clears the cancel flag before launching the new thread.

Requires a ComfyUI restart to take effect (Python module change).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Cap all serve functions at max_containers=1
  ([`ed148bb`](https://github.com/frost-byte/fbTools/commit/ed148bb1004d354480ddd3f62c25c819cdb9ba48))

Prevents Modal from spinning up a second container when a retry request arrives during cold-start. A
  single-user setup never needs more than one container; max_inputs=4 still allows up to 4
  concurrent in-flight requests to share the same container.

Requires a Modal redeploy to take effect.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Correct vision flags — all endpoints are text-only
  ([`0cd97a9`](https://github.com/frost-byte/fbTools/commit/0cd97a9f4be3a094806d3822e3312a09475c540f))

Confirmed via /api/models/local: all three GGUF models in the HF cache report task=text-generation
  with no mmproj file present. The vision:true flags were set optimistically and were never
  validated.

- utils/unsloth_client.py: vision/native_video → False for 27b and flash_next - js/ui/llm_panel.js:
  same correction in _ENDPOINTS array + updated titles

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Downscale tiles in _tile_frames to 320×180 per cell
  ([`4fe0bd4`](https://github.com/frost-byte/fbTools/commit/4fe0bd4be950cb4aedcb5719694644a25bb17b60))

Raw 1080p frames in an 8-cell grid would be 7680×2160 (~15 MB base64). Resizing each tile to 320×180
  (matching _spa_build_contact_sheet) keeps the sheet at 1280×360 — roughly 150 KB as JPEG, ~200 KB
  base64.

Also downscales single-frame paths so a lone 4K frame doesn't balloon the payload unnecessarily.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Downscale video frames individually, keep native multi-image path
  ([`a6e17a3`](https://github.com/frost-byte/fbTools/commit/a6e17a3db343e57126208f36b5155347b52972df))

The contact-sheet approach gutted Analyze Media's per-frame reasoning. Revert native_video=True on
  both endpoints.

Root cause of the 413: full-resolution 832×832 frames sent as 6 separate image_url entries → ~2 MB,
  exceeding nginx's 1 MB default.

Fix: _downscale_frame(img, max_dim=480) shrinks each PIL Image so its longest side is ≤480 px before
  encoding. 6 frames of 479×479 → ~530 KB total base64, well under the 1 MB nginx default (and under
  the 100 MB cap from the nginx config fix).

A contact-sheet fallback for non-native-video endpoints is kept inline in generate() so the branch
  is explicit.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Drop --mmproj CLI flag; pre-fetch into HF cache instead
  ([`3f24ec1`](https://github.com/frost-byte/fbTools/commit/3f24ec187e333e177f07c2c427d70307a1fbee13))

unsloth studio run does not accept --mmproj, causing the subprocess to exit immediately and port
  8888 to never open (Modal health check failure). Keep the hf_hub_download() call so the file lands
  in the Volume cache for Unsloth Studio's own auto-detection.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Escape f-string brace in nginx config comment
  ([`3bd3c93`](https://github.com/frost-byte/fbTools/commit/3bd3c93aa20ba21389e5cbaadb62602f93f66b6e))

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Fall back to modal CLI for serve mode volume write
  ([`7a0fe65`](https://github.com/frost-byte/fbTools/commit/7a0fe655d040d489bc0ef114dbb66b42d9f544f1))

When modal is not installed in ComfyUI's Python (batch_upload unavailable), fall back to invoking
  write_serve_config via the modal CLI binary. Checks PATH first, then the known preflight venv path
  as a hardcoded fallback.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Fast reconnect after restart + fix Open API link URL
  ([`5513bd1`](https://github.com/frost-byte/fbTools/commit/5513bd13f3377dff10629f981a4e82e62fdad184))

Reconnect: activate() now probes the container with a 5s timeout before starting the warmup thread.
  If it gets 200 (container still warm from a previous session), it sets warmup_status="warm"
  immediately and skips the thread — no more brief "Warming up…" flash on reconnect.

Open API link: _build_url returns /v1/chat/completions (POST-only), causing a 405 Method Not Allowed
  in the browser. Add _build_base_url() and expose endpoint_docs_url ("{base}/docs") from
  backend_status(). The frontend apiLink now uses endpoint_docs_url so it opens the Swagger UI.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Fix container list parser and add 30s auto-refresh
  ([`f1030df`](https://github.com/frost-byte/fbTools/commit/f1030dfa54ca93cd64d39dfae66cb1bc1676818a))

The previous parser tried to parse Rich unicode table output as plain text, silently counting
  box-drawing lines as containers.

Fixes: - Switch to `modal container list --json` for reliable parsing; filter by app_name ==
  "unsloth-studio" from the structured output - JSON key is `start_time` (derived from "Start Time"
  column header via Modal's snake_case conversion) - Add 30s _containerPollTimer so the count stays
  current without manual refresh; stopped on tab cleanup

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Follow Modal 303 Location token URL to prevent second container
  ([`329a90e`](https://github.com/frost-byte/fbTools/commit/329a90edb988c3d258075638974951fe4c86fecd))

Per modal.com/docs/guide/webhook-timeouts: the 303 Location header points to the original URL plus a
  token query parameter. POSTing to *that* URL tells Modal's LB to route the retry to the container
  already being provisioned, rather than treating it as new demand. Previously we ignored Location
  and retried the bare URL, which Modal could interpret as a fresh request and respond by spinning
  up a second container.

Also reverts the 303 sleep from 120 s back to 20 s: with the token URL being followed correctly, the
  sleep is just pacing rather than a workaround.

Requires a ComfyUI restart to take effect.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Increase 303 retry sleep; add --fit to place mmproj on GPU
  ([`4bcb329`](https://github.com/frost-byte/fbTools/commit/4bcb329a008b964058ed1d30d1eb9dfcd913b634))

303 sleep: 20 s → 120 s. After a 303 ("cold start redirect"), Modal's edge proxy may interpret a
  rapid retry as new demand and spin up a second container. 120 s gives the container enough time to
  start its nginx placeholder, so the next retry hits a 503 (from the container itself) rather than
  another 303.

--fit: the 27B model logged "--fit: off", meaning llama-server was not trying to fit the mmproj (0.9
  GB) onto the GPU. At ctx=65536 the VRAM budget is 12.2 + 4.6 + 1.42 = 18.2 GB against 22.5 GB
  free, leaving ~4.3 GB headroom — enough for the mmproj. --fit instructs llama-server to maximise
  GPU layer placement.

The unsloth_studio.py change requires a Modal redeploy to take effect. unsloth_client.py takes
  effect after a ComfyUI restart.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Log serve config at startup; add read/write_serve_config helpers; fix probe_ports
  volume restore
  ([`4985cd5`](https://github.com/frost-byte/fbTools/commit/4985cd56bb6b1631406d0dce2637111c1b0a7f52))

- Log the raw config value and resolved api_only at every container startup so the serve mode
  decision is visible in Modal logs - Add read_serve_config() and write_serve_config() Modal
  functions for manual inspection and override without starting the model - Fix probe_ports to call
  studio_volume.commit() after restoring the config so the volume is actually left in its original
  state

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Move max_containers=1 to @app.function() (correct API)
  ([`4a2a448`](https://github.com/frost-byte/fbTools/commit/4a2a448d82b2594102e7eb590bd74d8ce29e50c5))

@modal.concurrent() does not accept max_containers in Modal 1.5.5; it belongs on @app.function().
  Deploy verified successfully.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Move nginx map directive inside http block; log reload failures
  ([`3a4a10f`](https://github.com/frost-byte/fbTools/commit/3a4a10fea1b2a8511c616d79fe64a9d69fcccff4))

The map directive for WebSocket Connection header handling was placed at the top level of the nginx
  config, outside the http {} block, causing nginx -s reload to fail with [emerg] "map" directive is
  not allowed here. The reload silently failed (check=False), leaving nginx in 503 placeholder mode
  permanently after Studio became ready.

Also surface reload failures explicitly instead of printing success unconditionally.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Poll Studio before starting nginx to prevent 502 on startup
  ([`b1cf638`](https://github.com/frost-byte/fbTools/commit/b1cf638a93e7748c2c79b4863a0d36725a15be19))

nginx opened port 8888 immediately (passing Modal's startup check) while Studio took 60-90s to bind
  to 8889, causing every request to 502. Now _run_unsloth_serve() blocks until /api/health on
  studio_port returns 200 before starting nginx, so port 8888 only opens once the backend is ready.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Pre-cache mmproj for Studio auto-detection instead of passing --mmproj flag
  ([`2b1f6ca`](https://github.com/frost-byte/fbTools/commit/2b1f6ca7513d169e69027030df86d6ff301a16e8))

Unsloth Studio rejects --mmproj as a CLI extra arg: "llama-server flag '--mmproj' is managed by
  Unsloth Studio and cannot be passed as an extra arg". Keep hf_hub_download() to ensure the file is
  in the HF cache so Studio can auto-detect it, but remove the flag from the cmd args.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Prevent duplicate warmup threads and handle 503 placeholder
  ([`5c3399d`](https://github.com/frost-byte/fbTools/commit/5c3399db5ce41e3e90af6dcfbdaef48cfe121313))

Two root causes of multiple containers spinning up:

1. _start_warmup() had no guard — calling activate() while a warmup thread was already running
  (double-click, Restart, ComfyUI reload) launched a second thread that fired another cold POST to
  Modal, causing Modal to queue a second container. Fixed with a global _warmup_thread ref: skip the
  start if the thread is still alive.

2. The nginx placeholder (503 while Studio loads behind it) was not in the retry list.
  _call_with_retry raised immediately on 503, which propagated as a warmup error and could trigger
  another activate(). Fixed: treat 503 the same as 303 — log phase and continue the loop.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Proper WebSocket vs HTTP Connection header in nginx config
  ([`f147884`](https://github.com/frost-byte/fbTools/commit/f14788473a98d51ce0d188d62c0a7bb0959342b5))

Hardcoded 'Connection: upgrade' on all proxied requests broke regular HTTP keepalive, which likely
  caused Studio's thread creation API to fail on every request (producing 'Thread __LOCALID_ not
  found' errors). Use a map to set Connection: upgrade only for actual WebSocket upgrades, close
  otherwise. Also add X-Forwarded-For and X-Forwarded-Proto headers.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Remove pre-warmup probe that caused duplicate Modal containers
  ([`f8d25fb`](https://github.com/frost-byte/fbTools/commit/f8d25fbb8043c67acdb9c9c9f0a3db2a9b653fd4))

_probe_warmth() sent a real POST to the Modal endpoint before _start_warmup() launched the warmup
  thread. Both requests hit Modal while no container was running, causing Modal's auto-scaler to
  start two containers every time Activate was clicked on a cold endpoint.

The warmup thread already handles the "already warm" case: a 200 on the first attempt finishes the
  thread immediately with warmup_status="warm". Removing the pre-probe eliminates the race with no
  functional regression.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Replace --fit with --mmproj-offload for 27B GPU placement
  ([`c5103ab`](https://github.com/frost-byte/fbTools/commit/c5103ab0313ccdae432b3344eb9b8f4b439e21b1))

--fit is not a boolean flag in Unsloth Studio's llama-server; it requires a value. Passing it bare
  caused a 400 error that crashed the container on every cold start.

--mmproj-offload is the correct flag (per Unsloth's own log message) to force the mmproj-F16 (0.9
  GB) onto GPU. Unsloth's auto-detect conservatively places it on CPU even though ~4.3 GB of
  headroom exists at ctx=65536.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Replace frontend health poll with server-side warmup_phase
  ([`285c57e`](https://github.com/frost-byte/fbTools/commit/285c57e34704ef9fe6a4246410bc3d912df984f6))

The 20s health poll was POSTing to /v1/chat/completions on the Modal container every 20 seconds,
  queuing real inference requests during warm-up.

Instead, _call_with_retry() now writes a human-readable warmup_phase into _state at each retry
  outcome (timeout → GPU wait, 303 → container starting, 400 no-model → weight loading, 200 →
  Ready). backend_status() exposes it and the existing 5s /status poll delivers it to the frontend.

Frontend changes: - Remove _fetchHealth, _startHealthPoll, _stopHealthPoll, _healthPollTimer,
  _lastProbeAt, actProbeAge (all health-probe machinery) - Add _startElapsed/_stopElapsed: a 1s tick
  for the elapsed display only - _syncStatus shows st.warmup_phase as the phase message while
  warming - Activity block is shown as soon as active (not gated on warm)

Zero extra Modal traffic from the frontend during warm-up.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Restore --mmproj passthrough; drop reasoning_effort param
  ([`7a89e76`](https://github.com/frost-byte/fbTools/commit/7a89e76fff0427989286b3d663e7058baefcac17))

- modal/unsloth_studio.py: re-add --mmproj <path> to llama-server command; confirmed via local
  `unsloth studio run --help` that unknown flags pass through. File is now cached on Volume from
  prior cold start so download is instant on next cold start. - utils/unsloth_client.py: remove
  reasoning_effort parameter — it is a local CLI shorthand for `unsloth run`, not a llama-server API
  field.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Self-healing setup checklist for dropped bootstrap connections
  ([`c55e7d1`](https://github.com/frost-byte/fbTools/commit/c55e7d192918636f2c42259377797eaa174e3631))

The Bootstrap Key HTTP call can be dropped by the browser or an idle proxy before the server's
  asyncio thread finishes — the key IS stored server-side but the UI never saw the response.

Three changes to recover without a page reload: - _startSetupPoll() / _stopSetupPoll(): poll
  /unsloth/setup_status every 15 s while any check is incomplete; self-terminates once workspace +
  api_key + app_deployed are all green - Poll is started both on tab mount and at the start of every
  bootstrap attempt, so the checklist updates independently of the HTTP call result -
  _sepWithRefresh(): "Setup" section header now has a ↻ button for instant manual refresh (e.g.
  after a known-completed bootstrap that the UI missed)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Separate warmup retry loop from inference path
  ([`279abd3`](https://github.com/frost-byte/fbTools/commit/279abd3c4f73e921f00fb19646704c2836b1ca4e))

generate() was calling _call_with_retry() — the same 20-attempt × 200s cold-start loop used by the
  warmup thread. Concurrent inference calls (e.g. Source Profile analysis) could each block for up
  to 66 minutes if the container was not warm.

Fix: - Add _post_once(): single-shot POST with _GENERATE_TIMEOUT (180s), raises immediately on
  303/400-no-model with a clear "wait for warmup" message rather than retrying for minutes -
  generate() now gates on warmup_status == "warm" and returns an error immediately if the container
  is not ready — the LLM panel shows the warmup phase so users know to wait - Only _run_warmup()
  uses _call_with_retry(); inference never retries through cold starts

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Serve mode button group — active state highlights current mode
  ([`2e4d90b`](https://github.com/frost-byte/fbTools/commit/2e4d90bc79c717fe40b473a1c6573767a907fd04))

Replaces the ambiguous single toggle with two joined buttons (API Only | Full Studio UI). The active
  mode uses the primary style; the inactive one uses ghost. Clicking the inactive button switches
  and saves; clicking the already-active button is a no-op.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Start nginx placeholder immediately to avoid Modal startup_timeout
  ([`2ce5cf6`](https://github.com/frost-byte/fbTools/commit/2ce5cf6136b3b3e6ad2b49296926c7b5bac27c65))

Previously, nginx on port 8888 only started after Studio was ready on 8889 (poll up to 1500s). If
  Studio took longer than Modal's startup_timeout, the container was killed before nginx ever opened
  the port.

New flow: 1. _start_nginx_placeholder() opens port 8888 immediately (returns 503) so Modal's
  startup_timeout check passes within seconds 2. _run_unsloth_serve() polls Studio on 8889 as before
  (up to 1500s) 3. _reload_nginx_proxy() swaps the placeholder with the full proxy config (SPA
  static files + reverse-proxy to Studio) via `nginx -s reload`

Also extracts _nginx_placeholder_conf() / _nginx_proxy_conf() / _reload_nginx_proxy() helpers to
  keep _run_unsloth_serve() readable. client_max_body_size 100m is set in both configs.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Sync _state.unslothActive when server reports inactive
  ([`61e1951`](https://github.com/frost-byte/fbTools/commit/61e1951d5287f8ecbc2c17c85ab3c8d049416a3a))

_state.unslothActive was persisted in localStorage as true across ComfyUI restarts. When _syncStatus
  received active:false from the server it updated the button label but not _state.unslothActive, so
  clicking "Activate" triggered the deactivate() branch instead — doing nothing visible to the user.

Fix: reset _state.unslothActive/Vision/Label and _saveState() in the inactive branch of _syncStatus
  so the handler always uses the correct code path on the first click.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Use --json for app_status to avoid truncated name match
  ([`18a84ea`](https://github.com/frost-byte/fbTools/commit/18a84ea2aa1c78f0c002713eea0584be9c7dc4e6))

modal app list truncates the Description column in Rich table output ("unsloth-stu…") so the
  APP_NAME string check always failed, showing the App row as ✗ not deployed even when the app is
  running.

Switch to --json and match on description == APP_NAME and state == "deployed".

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Use contact sheet for video to avoid multi-image llama-server failures
  ([`f288d68`](https://github.com/frost-byte/fbTools/commit/f288d68946e8974b6902dd45f84943eebe7ebc8d))

llama-server (GGUF/llama.cpp) does not reliably support multiple image_url entries in a single
  request. Sending video frames as separate image_url blocks caused "Failed to load image or audio
  file" 400 errors.

- Set native_video=False on 27b and flash_next endpoint descriptors; _run_vision_inference_clip
  already has a contact-sheet fallback for endpoints where native_video is False - Add
  _tile_frames() to unsloth_client.py: tiles PIL frames into a single contact-sheet image (≤4 per
  row) so generate() sends at most one image_url when native_video=False, covering the
  describe_video → _route_vision path - Add client_max_body_size 100m to the nginx config so large
  vision payloads are not rejected before reaching Studio or llama-server - Update llm_panel.js
  endpoint titles/native_video flags to match

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Use enable_thinking payload param instead of /no_think prefix
  ([`25c0722`](https://github.com/frost-byte/fbTools/commit/25c072254fcebf6f81f86d4c6a87fb8e9fd4720b))

Unsloth Studio supports enable_thinking as a proper request body field (confirmed in docs). Replace
  the /no_think\n message prefix with "enable_thinking": false in the payload — cleaner and works
  correctly with vision content where prepending a prefix to the user message is awkward.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

### Features

- **unsloth**: Add Open API link to endpoint URL in status bar
  ([`cc18ce1`](https://github.com/frost-byte/fbTools/commit/cc18ce11da55bccff0de2050763ec6c48cc31e40))

Shows a small "Open API ↗" link next to the status line when the backend is active. Href is set from
  st.endpoint_url returned by the status poll; hidden when inactive or no URL is available.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Add Open Studio link to LLM panel status row
  ([`04a9a8d`](https://github.com/frost-byte/fbTools/commit/04a9a8d9463b45d33569a0ca39e9215e0db7ad28))

Exposes endpoint_studio_url (base Modal URL) from backend_status() and adds an "Open Studio ↗" link
  next to the existing "Open API ↗" link in the Unsloth tab. Both links are visible only while the
  backend is active.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Add reset_password Modal function to recover from forgotten password
  ([`5eef242`](https://github.com/frost-byte/fbTools/commit/5eef24267c538611f0d888d420f9d7a1d735a55d))

Wipes the Unsloth Studio auth DB from the volume and re-bootstraps with the current
  UNSLOTH_STUDIO_PASSWORD Modal secret value, returning a fresh API key.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Add Unsloth Studio tab to LLM panel
  ([`8d1ebd7`](https://github.com/frost-byte/fbTools/commit/8d1ebd7a8ecc8f003fe77996071365e6a77edff7))

Adds a fourth "Unsloth" tab to the LLM Backend panel alongside Local, Modal, and Gemini. The tab
  provides the full lifecycle UI for the Unsloth Studio Modal backend:

- Status badge with warmup indicator (cold/warming/warm colored dot) - Endpoint selector buttons —
  27B (recommended), 8B, Flash Next — with vision/native-video capability badges that update per
  selection - Activate/Deactivate button; activate triggers server-side warm-up and polls /status
  every 5 s (slows to 30 s once warm) - Setup checklist: workspace, API key, app deployed — each row
  with ✓/✗ check icon fed by /setup_status - Deploy App button (30–90 s) and Undeploy with
  confirmation dialog - Bootstrap Key button with live elapsed-second timer and 35-min fetch timeout
  for the long-running install+key-capture operation - Force-reinstall checkbox for bootstrap -
  Containers section: running count + Stop All with confirmation - Tab indicator dot goes green when
  Unsloth is active - unslothActive/unslothVision/unslothLabel persisted in localStorage

Also adds js/api/unsloth.js (UnslothAPI client) with typed methods for all 10 /fbtools/unsloth/*
  routes, plus a custom bootstrapKey() fetch using AbortController for the 35-min timeout window.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Configurable ctx_size in serve config; default 131072→65536
  ([`7fa7c78`](https://github.com/frost-byte/fbTools/commit/7fa7c78736066c923efdd01e6138815e0b688bda))

Default context for 27B and flash-next endpoints changed from 131072 to 65536. At 131072 the KV
  cache fills VRAM, evicting the mmproj to CPU and making image encoding 5-20× slower. At 65536 the
  mmproj stays resident (~4 GB headroom) and MTP may also re-enable.

The ctx_size is now overridable without redeploying:

modal run modal/unsloth_studio.py::write_serve_config --ctx-size 131072

_run_unsloth_serve reads ctx_size from fbtools_serve_config.json and strips any existing
  -c/--ctx-size from extra_flags before injecting the override, so the volume config is always
  authoritative.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Detect container scale-to-zero and add Restart Warmup button
  ([`8c0faa6`](https://github.com/frost-byte/fbTools/commit/8c0faa61e54d48751ce3673010e3d9d22db4f173))

Problem: when Modal's 10-min idle scaledown fires, the UI still shows "Warm ✓" with no way to
  restart without Deactivate → Activate.

Changes: - unsloth_client: add mark_container_gone() — transitions warmup_status from "warm" to
  "cold" and sets warmup_phase to "Container unavailable — click Restart Warmup"; no-ops if a warmup
  thread is already retrying - extension: _route_text() and _route_vision() call
  mark_container_gone() on any generate() exception so the next /status poll reflects reality -
  llm_panel: add Restart Warmup button (ghost, shown when active but not warm); calls activate()
  directly to kick a new warmup thread without requiring Deactivate → Activate; hidden when warm or
  inactive - llm_panel: update Activate/Deactivate tooltip to explain cold-start timing, idle
  scaledown, and when to use Restart Warmup vs Deactivate

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Enable vision on 27B endpoint via mmproj-F16.gguf
  ([`3eb7bd4`](https://github.com/frost-byte/fbTools/commit/3eb7bd49fb6c92e09a88eaba92138e3d00741592))

- modal/unsloth_studio.py: add mmproj_filename to qwen3.8-27b CONFIGS; _run_unsloth_serve()
  downloads it via hf_hub_download() (cached on Volume after first cold start) and passes --mmproj
  <path> to unsloth studio run - utils/unsloth_client.py: restore vision/native_video=True for 27b -
  js/ui/llm_panel.js: restore vision/native_video=true for 27b endpoint

Flash Next left as text-only (mmproj not yet confirmed for that repo).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Enable vision on Flash Next endpoint via mmproj-F16.gguf
  ([`9513943`](https://github.com/frost-byte/fbTools/commit/9513943d659308fba6ca40858c5ea46d0b2c466f))

Same mmproj pattern as the 27B endpoint — confirmed HF repo has the file.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Expose thinking mode and full sampling params in generate()
  ([`acf9125`](https://github.com/frost-byte/fbTools/commit/acf9125d4b55164e096e71edcd487aa3aab7f8bd))

Adds Unsloth-recommended defaults for Qwen3.8-27B (source: unsloth.ai/docs): thinking mode:
  temp=1.0, top_p=0.95, top_k=20, min_p=0, presence=0.0 instruct mode: temp=0.7, top_p=0.80,
  top_k=20, min_p=0, presence=1.5

New generate() parameters: - thinking (bool, default True): selects mode defaults; instruct mode
  prefixes /no_think to the user message - reasoning_effort ("xhigh"|"medium"|"low"|"none"): passed
  via chat_template_kwargs to control CoT trace depth - temperature, top_p, top_k, min_p,
  presence_penalty, repetition_penalty: all override the mode default when provided - max_tokens
  default raised 512→2048 (thinking traces consume tokens)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Info icons with tooltips on Activate and Stop All
  ([`676003e`](https://github.com/frost-byte/fbTools/commit/676003e9c6984221c4cbf8a5224c27073f3081bd))

Adds a small ⓘ icon (cursor:help, .llmp-iicon) next to the Activate/Deactivate button and the Stop
  All Containers button. Each icon shows a native browser tooltip on hover explaining the
  distinction:

- Activate/Deactivate: local routing control only; container stays running on Modal until the 10-min
  idle scaledown fires - Stop All: kills the GPU container immediately, ends billing, requires a
  fresh cold start on the next request; prefer Deactivate if just switching backends temporarily

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Live activity section with health probe and elapsed timer
  ([`dd369f9`](https://github.com/frost-byte/fbTools/commit/dd369f9ddcafd754e1f99aa135d26f79de34c087))

Adds an Activity section to the Unsloth tab that shows what the container is doing during the
  cold-start window:

- Elapsed timer counting from the moment Activate is clicked (1 s tick) - Health probe every 20 s
  while warming, surfacing the modal-side phase: "No response yet — waiting for an available L4
  GPU." "Container is starting up (GPU worker assigned)." "Container running — loading LLM weights
  into VRAM." "Container is ready." - Probe age ("Last probe: 15s ago") so the user knows the data
  is live - On warm: section updates to "Container ready after Xm Ys." and stops polling; on
  deactivate: section hides - Health poll starts from _syncStatus so it also resumes correctly if
  the tab is opened while already-active-and-warming

Also improves health_check() messages to distinguish GPU queue wait (connection timeout/refused)
  from container boot (HTTP 303) and model load (HTTP 400 "no model loaded").

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Nginx reverse proxy for Full Studio UI access via Modal
  ([`ff0faa7`](https://github.com/frost-byte/fbTools/commit/ff0faa77a3090fa431cfa1396b6cb84681b0f784))

Studio's middleware blocks the SPA for requests that don't arrive via Cloudflare tunnel or its LAN
  listener — Modal's proxy comes in on loopback and is rejected. Fix: in Full Studio UI mode, run
  nginx on port 8888 (what @modal.web_server exposes) to serve studio/frontend/dist/ directly and
  proxy API paths (/api/, /v1/, /docs, etc.) to Studio on port 8889. API-only mode is unchanged
  (Studio stays directly on 8888, no nginx).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Reasoning_effort selector, streaming SSE, retry sleep, job notifier
  ([`acbad2b`](https://github.com/frost-byte/fbTools/commit/acbad2b7116d58b2b8a3ad8744625eec24b7cda7))

**Retry loop** - Sleep 30 s on 503 (nginx placeholder), 20 s on 303/400, 15 s on timeout so warmup
  retries don't burn through _MAX_ATTEMPTS in seconds on a cold 27B start - Bump _MAX_ATTEMPTS 20 →
  40 (covers ~20 min of 503 polling) - Prevent duplicate warmup threads with is_alive() guard

**Reasoning effort** - Add thinking + reasoning_effort to _state; persist across calls - New
  _build_payload() centralises payload construction; injects reasoning_effort and optional stream
  flag - New set_inference_settings() updates _state and validates effort values - generate() reads
  state defaults when caller omits thinking/reasoning_effort - backend_status() surfaces thinking +
  reasoning_effort fields - POST /fbtools/unsloth/inference_settings REST route - LLM panel:
  4-button reasoning mode row (Instruct/Low/Medium/High), _applyReasoningMode(), _syncStatus()
  mirrors server state to buttons - Disable Restart Warmup button while warming to prevent spam
  clicks

**Streaming** - New generate_stream() sync generator with incremental <think> block filter - POST
  /fbtools/llm/generate/stream SSE route: asyncio.Queue bridges sync httpx generator to async
  aiohttp StreamResponse - LlmAPI.generateStream() SSE reader in js/api/llm.js -
  UnslothAPI.inferenceSettings() in js/api/unsloth.js

**Job Complete Notifier** - _NOTIFY_DIR: ComfyUI user-data dir watched by this Claude session -
  JobCompleteNotifier output node writes UUID-named JSON on workflow completion - Enables phone push
  notifications via Claude Code Monitor

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Route llm/generate* and describe_video through Unsloth when active
  ([`0fa5e76`](https://github.com/frost-byte/fbTools/commit/0fa5e76063a33cbeb49eb76026882b10beb07a9e))

Adds _route_text() and _route_vision() async helpers that transparently dispatch to
  _unsloth_client.generate() when Unsloth is active, falling back to _llm_client otherwise. Both
  helpers share the same return shape {success, text, message} so callers need no per-backend logic.

Wired routes: POST /fbtools/llm/generate — vision path when images present, text otherwise POST
  /fbtools/llm/generate/shot_action — text-only POST /fbtools/llm/generate/dialogue — text-only POST
  /fbtools/llm/generate/polish — text-only POST /fbtools/llm/describe_video — video_frames vision
  path

_route_vision() returns a descriptive error when the active Unsloth endpoint is text-only (8B)
  rather than silently stripping images. prompt_for_* builders on _llm_client are still used for
  structured prompt construction regardless of the active backend.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Self-contained Unsloth Studio Modal backend
  ([`c1190b4`](https://github.com/frost-byte/fbTools/commit/c1190b4c0685088a8dc563464907ca4989d910aa))

Bundles the Unsloth Studio Modal app and wires it into the fbTools LLM backend system so a new user
  can deploy and use it entirely from within this extension — no separate project needed.

What's included: - modal/unsloth_studio.py: Modal app definition (3 endpoints: 27B, 8B, Flash-Next;
  install_studio + bootstrap_api_key setup functions; 10-min scaledown window) -
  utils/unsloth_client.py: HTTP client to the OpenAI-compatible endpoints; dynamic URL construction
  from workspace name; cold-start 303 retry loop; background warm-up thread on activate(); Qwen3
  reasoning-block stripping - utils/modal_deploy.py: workspace resolution (MODAL_WORKSPACE env var,
  Modal SDK, ~/.modal.toml parse), deploy/undeploy/app-status via modal subprocess, container
  list/stop, bootstrap_api_key capture + key persistence - extension.py: startup configure() from
  stored key + resolved workspace; _run_text_inference unsloth branch; _run_vision_inference
  text-only guard; REST routes: status, activate, deactivate, health, setup_status, deploy,
  undeploy, containers, containers/stop, bootstrap_key

New user setup flow (all from within ComfyUI after `modal token new`): 1. LLM panel > Unsloth >
  Deploy (modal deploy, ~60s) 2. LLM panel > Unsloth > Setup (install + bootstrap key, ~5-30 min) 3.
  Activate → warm-up starts in background → warm within 2-5 min

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Serve mode toggle — API Only vs Full Studio UI
  ([`f6810a1`](https://github.com/frost-byte/fbTools/commit/f6810a13a61820ad8ededccbad779d993a5d0dfc))

- modal/unsloth_studio.py: _run_unsloth_serve() reads fbtools_serve_config.json from the persistent
  Volume; defaults to api_only=True when absent - utils/modal_deploy.py: add load_serve_config(),
  _save_serve_config(), set_serve_mode() (writes local JSON + Modal Volume); deploy() records
  last_deployed_api_only on success - extension.py: GET/POST /fbtools/unsloth/serve_mode routes -
  js/api/unsloth.js: serveMode() and setServeMode(api_only) client methods - js/ui/llm_panel.js:
  Serve Mode row in the Setup section — toggle button (API Only ↔ Full Studio UI), status line
  showing last-deployed mode and mismatch hint; fetched on mount via _fetchServeMode()

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

### Refactoring

- **unsloth**: Use GET /v1/models for warmup probe instead of POST
  ([`9c667f0`](https://github.com/frost-byte/fbTools/commit/9c667f05d1ee67a7f84d15b1a0ee08f365ecc0c3))

POST /v1/chat/completions with max_tokens=1 generated actual tokens on every warmup cycle. GET
  /v1/models is semantically more appropriate (readiness check, not inference) and generates
  nothing.

Extracted _get_models_with_retry() with the same 303 Location token-following and cancel-event logic
  as _call_with_retry(). _run_warmup() now calls this instead of the POST variant.

Requires a ComfyUI restart to take effect.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.25.0 (2026-09-03)

### Bug Fixes

- **h3**: Drop per-replacement sentences from detailed_description preamble
  ([`5ec0c42`](https://github.com/frost-byte/fbTools/commit/5ec0c4215d1e8ee17b1548d3297ce04e786a2969))

The "The man in gray shirt is completely replaced by <Subject 1>." lines reiterated what
  subject_definitions and retention_analysis already cover. Keep only the single quality directive
  ("The target video is a photorealistic, seamless identity-replacement edit with strong temporal
  consistency.") before the shot descriptions begin.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **h3**: Simplify attribute_transfer replaced-subject prose
  ([`a6bb0bb`](https://github.com/frost-byte/fbTools/commit/a6bb0bb12e4ff1b92fabb27c2e2cda204e8a801c))

Remove the redundant "is NOT copied and" phrase — "is fully replaced by" carries the intent
  unambiguously on its own and avoids double-encoding the same constraint for the model.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **h3**: Use appearance descriptions (not identifiers) in attribute_transfer prose
  ([`10150c4`](https://github.com/frost-byte/fbTools/commit/10150c4d623bbdcf9b9a35cb6bb14778d6b38a82))

Two fixes in prompt_assembler.py:

1. subject_definitions motion clause: `src_info.get("name")` was always truthy so
  `appearance_summary` never fired — flipped priority to prefer appearance_summary over name for the
  "match those of ... in <Video N>" clause.

2. retention_analysis attribute_transfer branch: replaced `info['name']` (bundle identifier like
  "demon_3") and `src_name` (bare source label) with full appearance descriptions for both parties.
  Restructured the sentence from "{name}'s appearance overrides that of {label}" to "The appearance
  of {bun_desc} overrides that of {src_desc}" to avoid possessive-on-description grammar problems.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **h3**: Use role_description and video anchor in retention_analysis for non-person subjects
  ([`d6a2241`](https://github.com/frost-byte/fbTools/commit/d6a2241118bf5540fcf31823d66cebb002f80aa8))

Source profile subjects (objects, locations, animals) were emitting only the bare label in both
  subject_definitions and retention_analysis because appearance.summary was set to just the label
  rather than role_description.

Changes: - extension.py SceneCastBuild: set appearance.summary to role_description when available
  (fallback to label); store entity_type in slot_assignments - prompt_assembler.py _build_ref_map:
  propagate entity_type into ref_map - prompt_assembler.py retention_analysis: append ", as seen in
  <Video N>" for non-person subjects that have a video reference but no picture references

Result: a bed subject with a rich role_description now emits a full description in
  subject_definitions and retention_analysis, plus a <Video N> anchor so H3 knows which source video
  to sample the visual reference from.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Always persist proxy_built_at when server reports a fresh proxy
  ([`745ea67`](https://github.com/frost-byte/fbTools/commit/745ea675ef46132330920a035c319efa561fb5fd))

The !_isProxyDirty() guard in _refreshProxyStatus prevented proxy_built_at from ever being written
  back to disk after a browser reload. Because proxy_built_at was undefined after reload,
  _isProxyDirty returned true, the guard blocked _persistClip, and the badge permanently showed
  "needs rebuild" even for cached proxies. On click, the backend correctly found the proxy fresh and
  skipped ffmpeg — leaving the user with a toast but no machine activity.

Fix: when the server authoritatively says c.fresh = true, always update and persist proxy_built_at
  regardless of the client-side dirty state.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Clip segment QoL — apply-all for LoRA/subjects/soundscape/music, proxy status polling
  ([`5b6152e`](https://github.com/frost-byte/fbTools/commit/5b6152e6e5792a5a396945ec2b6e303e0f6d73d5))

Source profile editor clip section: - Add "→ all" button per LoRA entry: copies LoRA to all other
  segments that lack it; preserves existing weight in segments that already have it - Add "→ all" /
  "✕ all" per subject: bulk-checks or unchecks a subject across every segment in the profile - Add
  "→ all" per soundscape and non-diegetic music field: copies current segment's value to all other
  segments - Single-clip proxy build now shows a toast on success/failure and detects clip_count=0
  (clip not yet persisted on server) - Replace fixed 3s/8s status timeouts with adaptive poll
  (5→10→15→20→30→60s) that stops early once the proxy is marked fresh, fixing stale "needs rebuild"
  badge after a single-clip proxy build completes

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

### Features

- **h3**: Tag Replaced Subjects toggle for original-subject attribute_transfer
  ([`43a53e4`](https://github.com/frost-byte/fbTools/commit/43a53e4fd062d571b7d4308ceecc08802439e608))

Adds an opt-in "Tag Replaced Subjects" boolean input to SourceProfileClipPrompt.

When enabled, each source-profile subject that is being replaced by a SceneCastBuild bundle
  receives:

- A <Subject N> tag in subject_definitions with their role_description (identifiable but no
  structured hair/face/body detail fields, which are empty for source subjects). - A scoped
  attribute_transfer entry in retention_analysis that explicitly states: pose, movement, gestures,
  timing and screen position transfer to <Subject M>; the original's appearance, including face,
  hair, and clothing, is NOT copied and is fully replaced by <Subject M>'s appearance from <Picture
  P>/<Video N>.

The original subject numbers are assigned last (after bundle replacements and retained subjects), so
  existing Subject N ordinals are unaffected.

Gated behind the checkbox (default off) so the user can A/B the scoped-tag version against no-tag at
  a fixed seed before committing. Plan documented in docs/h3_attribute_transfer_assembler_plan.md.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Propagate SourceProfileLoad combo changes to downstream nodes
  ([`22d1dda`](https://github.com/frost-byte/fbTools/commit/22d1dda7818189848575f1ddc92915f732972679))

Hook profile_name widget callback on SourceProfileLoad to fire onConnectionsChange on every node
  connected to its output when the selected profile changes. SceneCastBuild (and
  SourceProfileClipPrompt) now refresh their clip selectors and source subject columns immediately
  on combo change without requiring the user to disconnect and reconnect the wire or execute the
  graph.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.24.0 (2026-09-01)

### Bug Fixes

- **llm-panel**: Clarify Modal activation does not start container
  ([`8387258`](https://github.com/frost-byte/fbTools/commit/838725856fe2734aabccb42ba0f1ed927761d989))

Rename status from "Active" to "Configured" and add "(container starts on first request)" note so
  users understand the dashboard will be empty until the first inference call triggers a cold start.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **llm-panel**: Move model load/unload to Local tab; fix Modal TDZ crash
  ([`aff810f`](https://github.com/frost-byte/fbTools/commit/aff810f73a425b9b5efc1d3cba97e7df2b548ebc))

- Fix ReferenceError: quantCb/quantNotice were accessed in TDZ when _rebuildModelSel() was called
  before their const declarations in _renderModalTab. Fixed by declaring them before the first call.
  - Move full model management (scan, select, load, unload, download) from composition_editor's LLM
  Assistant section to the LLM panel's Local tab. - Compose tab retains status line (read-only) and
  generate buttons; syncs _S.llmLoaded/Vision/NativeVideo via fbt:llm-status custom event dispatched
  by fbt_panel._handleLlmPush after every load/unload. - Remove _llmRefreshModels,
  _populateLlmModelSel, _llmLoadSelected, _llmUnload, _llmDownloadDefault from composition_editor.js
  — all now live in llm_panel.js.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **modal**: Drop modal.parameter() — pass model_key/quantize via generate()
  ([`c18265d`](https://github.com/frost-byte/fbTools/commit/c18265dfeba873c83ee6581456ab5c754727522b))

modal.parameter() doesn't support str type in modal 1.x. Restructured VisionLLM to load the model
  lazily on first generate() call, caching by (model_key, quantize) within a container's lifetime.
  Client updated to pass model_key and quantize as keyword args to generate.remote() instead of the
  class constructor.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **modal**: Rename container_idle_timeout to scaledown_window (modal 1.x)
  ([`db424ff`](https://github.com/frost-byte/fbTools/commit/db424ffc0327b9eb3b17182af06a014fe8f2c3d7))

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **modal**: Use gpu="L40S" string — modal.gpu removed in modal 1.x
  ([`7c08ab2`](https://github.com/frost-byte/fbTools/commit/7c08ab2e04e33890ba11c8b6e9e1ff646a45dab0))

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **prompt-assembler**: Correct H3 video roles, bundle subjects, dialogue verb, speaker IDs
  ([`e6a2ef1`](https://github.com/frost-byte/fbTools/commit/e6a2ef1603b0fd7044ca512ec87e74b27de039b2))

- Video role in subject_definitions and retention_analysis now differentiates motion-donor (source)
  videos from bundle appearance references using _vnum_is_source computed from retention markers -
  Bundle-first subject numbering: attribute_transfer slots get Subject N labels before retained
  source slots; replaced slots get no label - Add _possessive() helper for pronoun-aware possessive
  forms - Dialogue lines now include "says:" verb before quoted text - Speaker IDs (Sx) now assigned
  to dialogue-only slots (no audio file) so subject_definitions shows the (Sx) marker for all
  speaking subjects - Pre-assign subject numbers before the ref_map loop to ensure stable ordering
  independent of slot iteration order

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **source-profile-analysis**: Hoist _SPA_STATUS_ID to module level
  ([`40ec6d7`](https://github.com/frost-byte/fbTools/commit/40ec6d7ed3dda91ee7431ec7945f83dad2aa977d))

_run_vision_inference and _run_vision_inference_clip reference _SPA_STATUS_ID in their Modal
  status_callback but it was only defined as a local variable inside _source_profiles_analyze,
  causing NameError when the Modal path ran. Promoted to module-level constant and removed the
  now-redundant local def.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **source-profile-editor**: Surface server error detail in VLM failure toasts
  ([`e9b9e43`](https://github.com/frost-byte/fbTools/commit/e9b9e4383341da41ea6515c4b2f751dfdce4643a))

APIError.response holds the raw JSON body from the server, which contains the actual reason (e.g.
  "Modal app not deployed", "modal package not installed"). Previously all VLM catch blocks showed
  only err.message ("Internal Server Error"). Added _errMsg() helper that parses err.response for
  the "error" field first, falling back to err.message. Applied to auto-segment, detect, describe,
  and analyze failure toasts.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **source-profiles**: Accept bare JSON array in VLM parser responses
  ([`9764708`](https://github.com/frost-byte/fbTools/commit/976470855fcbc14760fae9c156c987249dd9aa09))

_parse_vlm_json_response and _parse_segments_response now handle both {"subjects":[...]} /
  {"segments":[...]} envelopes and bare [...] arrays, matching the leniency added to the Modal-side
  parsers. Qwen2.5-VL and similar models sometimes omit the wrapping dict.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Output-dir support for extract-frame/audio, source profile card + nav fixes
  ([`a509662`](https://github.com/frost-byte/fbTools/commit/a509662e85917938f0ba4cabadadae3680867670))

Bundle editor: - extractFrame and preprocessAudio now pass the correct dir ("input"/"output") to the
  server — videos placed in output/ were always 404ing - LLM appearance analyzer reads live vision
  status via window._fbtGetLlmStatus and listens to fbt:llm-status so the section appears without
  needing a reload after a model is loaded from the LLM tab

Source profile editor: - Card subject count uses subject_count from the list summary instead of
  subjects.length (which was always 0 before a profile was opened) - Card body shows "N subjects ·
  open to view" when subjects aren't yet fetched - Back button used
  root.parentElement?.parentElement causing DOM drift on each navigation; fixed to render into root
  directly

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

### Documentation

- Add Modal cloud vision backend integration handoff
  ([`efe8ee6`](https://github.com/frost-byte/fbTools/commit/efe8ee616830c7c55b2b6e2aa803b99f415874f5))

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- Update CLAUDE.md with widget contracts; add design docs
  ([`3549a61`](https://github.com/frost-byte/fbTools/commit/3549a61621d66ae880961fd47961e6a141a68fb8))

- CLAUDE.md: document cross-layer widget naming contract and the test_widget_name_contracts.py
  automated check - docs/vlm_systems.md: VLM system architecture overview -
  docs/h3_ref_short_edge_action_plan.md: H3 reference short-edge plan

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

### Features

- Subject inference, multi-GPU modal dispatch, describe-clip refinements
  ([`65ef76d`](https://github.com/frost-byte/fbTools/commit/65ef76d737794dd0a22e130b763288df4f5950af))

Source profile analysis: - build_subject_inference_prompt / parse_inferred_subjects_response for
  LLM-driven subject detection from clip descriptions - detect_segments: batch_window_seconds
  parameter for chunked processing - describe_clip: existing_action and prompt_override parameters -
  New endpoints: set_clips (bulk replace), merge_subjects (dedup upsert) - API client additions in
  js/api/source_profiles.js

Modal / LLM panel: - modal/app.py: per-GPU cls variants (T4, L4, L40S) for flexible dispatch -
  modal_vision_client: activate() accepts gpu parameter - js/api/modal.js: recommend() and
  profileRepo() methods - llm_panel.js: GPU selector UI, Qwen3-VL model list refresh

Other: - prompt_assembler: formatting and correctness fixes - extract_frame and preprocess_audio
  endpoints accept dir="output" - pyproject.toml: dependency/version updates

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **composition**: Add per-slot appearance overrides (slot_descriptors + appearance_overrides)
  ([`7f3152f`](https://github.com/frost-byte/fbTools/commit/7f3152f5a9acea88aaff0cd139e188b01228162b))

- utils/prompt_assembler.py: in assemble_composition(), apply composition.slot_descriptors[Sn] as
  appearance.summary override and composition.appearance_overrides[Sn].{face,hair,body} as granular
  sub-field overrides before passing slot_assignments to assemble_prompt. Deep-copies the affected
  subject dict so original resolved_subjects are never mutated. - js/ui/composition_editor.js: add
  collapsible "Override appearance" section to each slot card with a description textarea
  (slot_descriptors) and face/hair/body field rows (appearance_overrides). Indicator badge (✎) on
  toggle when any override is set. Wired to _markDirty(); both dicts included in _renumberSlots()
  remapping and slot removal cleanup. - js/styles/style.css: add .fbt-ce-slot-override-* rules. -
  tests/test_prompt_assembler.py: 10 new tests for TestSlotDescriptors and TestAppearanceOverrides
  covering override, no-mutation, empty/ whitespace passthrough, unknown key, and combined cases.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **dataset**: Dataset caption status and viewer node improvements
  ([`7c2b613`](https://github.com/frost-byte/fbTools/commit/7c2b613f2ea6c8d7c9dab1dc9f5cceb4168fe507))

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **libber**: Extract libber resolution to stdlib-only utility module
  ([`cd3e734`](https://github.com/frost-byte/fbTools/commit/cd3e73414e8198808ab76a6224f06164b6591143))

Move %libber_name:key% resolution logic out of extension.py into utils/libber_resolve.py so it can
  be imported and tested without any ComfyUI context. Adds resolve_libber_refs() and
  extract_libber_names() with full test coverage.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **modal**: Add Modal cloud VLM backend as third explicit inference path
  ([`7f562ef`](https://github.com/frost-byte/fbTools/commit/7f562efb1d05e55bbe0532e30d91828e25bf551f))

- utils/modal_vision_client.py: slim client wrapping VisionLLM.generate.remote();
  activate/deactivate/is_active/backend_status; PRESET_MODELS list - utils/vlm_activity_log.py:
  rolling 500-entry activity log with record/recent/ model_history/last_activity_ts; drives model
  history in Modal tab and idle tracking - extension.py: third 'modal' branch in
  _run_vision_inference + captioner_type param on _run_vision_inference_clip; activity logged on
  every inference call; new routes GET /fbtools/modal/status, POST /fbtools/modal/activate, POST
  /fbtools/modal/deactivate, GET /fbtools/vlm/activity, GET /fbtools/vlm/model_history; late-imports
  modal_vision_client + vlm_activity_log

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **modal**: Add Qwen3-VL-8B + Qwen2.5-VL-32B-AWQ presets; pre_quantized flag
  ([`e222428`](https://github.com/frost-byte/fbTools/commit/e2224288090340354fdd2576b6fbcce35793c721))

- PRESET_MODELS gains qwen3-vl-8b (new default) and qwen2.5-vl-32b-awq with pre_quantized: True;
  backend_status() exposes pre_quantized; activate() auto- disables NF4 when pre_quantized to
  prevent double-quantization - llm_panel.js: fallback preset list updated to match; _presetMap
  lookup drives _applyPreQuantizedState() which disables and unchecks the NF4 toggle with an amber
  notice when an AWQ/GPTQ preset is selected; re-enables on switch away; quantize label gains
  tooltip explaining bf16-only scope; custom HF ID placeholder now notes standard transformer repos
  only

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **modal**: Add VisionLLM Modal app + better "not deployed" error message
  ([`4ee8445`](https://github.com/frost-byte/fbTools/commit/4ee8445aa6fc2bea6971b38d1ee8fe52e37c959d))

modal/app.py: - Defines fbtools-vision-llm Modal app with a VisionLLM cls - Supports qwen3-vl-8b
  (default), qwen2.5-vl-7b, qwen2.5-vl-32b-awq, qwen2.5-vl-3b, gemma3-4b; custom HF repos via
  model_key parameter - qwen_vl arch: Qwen2_5_VLForConditionalGeneration + qwen-vl-utils for native
  image and video_frames input - generic arch: AutoModelForCausalLM + AutoProcessor chat template
  path (Gemma3 and unknown custom repos) - NF4 quantization via BitsAndBytesConfig (skipped for
  pre-quantized AWQ) - Models cached in modal.Volume "fbtools-model-cache" to avoid re-download -
  GPU: L40S; idle timeout: 5 min; deploy: modal deploy modal/app.py

utils/modal_vision_client.py: - Catch "App not found in environment" from modal.Cls.from_name() and
  return a human-readable deploy instruction instead of the raw SDK error

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **scene**: Extend SceneCastBuild with source profile pool routing
  ([`21f7e75`](https://github.com/frost-byte/fbTools/commit/21f7e75797cbcd71a8c4561a57a4903b3c07bfc4))

Add three optional SOURCE_PROFILE inputs to SceneCastBuild so subjects from SourceProfileLoad nodes
  form a selectable pool alongside existing bundle-backed entries.

- source_profile_1/2/3 optional inputs wire SOURCE_PROFILE → cast pool - fingerprint_inputs()
  includes source_profiles.json mtime + profile IDs - execute() routes source-derived entries
  (source_profile_id + source_subject_id) vs bundle-backed entries (subject_id + bundle_id) - Smart
  retention defaults: fully_preserved for bundle entries, partially_preserved for source-derived -
  Cast output carries source_profiles dict for downstream deduplication - cast_summary shows
  retention mode and entity type per entry

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **scene**: Phase 4 — source-derived subjects in prompt assembly
  ([`75269c5`](https://github.com/frost-byte/fbTools/commit/75269c5a26cc6060959b580250de4a01b3116254))

Teach _resolve_cast_media, apply_cast_to_subjects, _build_ref_map, and the H3 assembler to handle
  source-profile cast entries end-to-end.

_resolve_cast_media (extension.py): - Second pass groups cast entries by source_profile_id; emits
  one video_entries_full entry per unique profile (not per subject), carrying subject_ids: list for
  shared <Video N> assignment downstream.

apply_cast_to_subjects (utils/prompt_compositions.py): - New branch for source-derived entries:
  builds a synthetic subject dict with role_description as appearance.summary and _cast_retention
  set to the entry's retention mode (default: partially_preserved).

_build_ref_map (utils/prompt_assembler.py): - video_lookup handles subject_ids list so co-sourced
  subjects share the same video_entry object. - _video_entry_num dict ensures co-sourced subjects
  get the same <Video N> ordinal (only one video_counter increment per unique source entry). -
  retention_marker falls back to subject._cast_retention when retention_markers dict has no override
  for the slot.

_assemble_h3_ref2va (utils/prompt_assembler.py): - _vnum_to_labels map built after ref_map; drives
  combined subject labels in video role lines ("is the visual identity reference for <Subject 1> and
  <Subject 2>"). - video_sd_emitted / _ra_video_emitted guards prevent duplicate <Video N> lines
  when multiple subjects share one source video. - retention_analysis video lines use plural grammar
  ("their appearance", "the people") when more than one subject shares the video.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **scene**: Phase 7 — LLM subject decomposition for Source Profiles
  ([`cf26241`](https://github.com/frost-byte/fbTools/commit/cf26241a4f30335ace49f70f2505ac03651c6293))

Focused, additive VLM analysis passes identify subjects in source media (people, setting,
  soundscape, objects, animals, or custom) and return structured candidates for review before
  committing to the catalog.

utils/source_profile_analysis.py (new — no ComfyUI deps): - PASS_TYPES + PASS_ENTITY_DEFAULTS define
  the six focus modes - build_prompt(): returns per-pass template with JSON schema instruction
  appended; accepts optional prompt_override replacing the template body -
  _parse_vlm_json_response(): strips markdown code fences, validates schema, fills missing/invalid
  fields with safe defaults, skips entries with no label - extract_video_frame(): pulls one frame at
  10% into clip via ffmpeg (primary) or imageio (fallback) - append_history_entry() / load_history()
  / history_for_profile(): append-only JSON history in source_profile_analysis_history.json with
  .bak backup; newest-first ordering per profile

extension.py: - Import source_profile_analysis helpers at module load - POST
  /fbtools/source_profiles/analyze — resolves media path, extracts frame for video sources, calls
  captioner.py backend, parses response, writes history, returns {candidates, pass_type, prompt} -
  GET /fbtools/source_profiles/analysis_history?profile_id=… — returns history entries for a
  profile, newest first

tests/test_source_profile_analysis.py (new — 35 tests): - prompt template coverage, JSON parser edge
  cases (code fences, leading prose, missing/invalid fields, non-dict items, empty subjects),
  history append/load/filter/ordering, backup creation, mutation safety

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **scene-cast**: Scenecastbuild dialogue and clip prompt UI
  ([`e58a3b5`](https://github.com/frost-byte/fbTools/commit/e58a3b56812e3e52a446494fe2b9f366f2809534))

Extend SceneCastBuild node UI with dialogue entry system and clip prompt support; dynamic slot/cast
  rendering improvements.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **source-profile**: Add SourceProfileLoad/Define/List nodes and REST endpoints
  ([`beb6c73`](https://github.com/frost-byte/fbTools/commit/beb6c73f982fa6795fc64c2f3b01c8c89cbc9bbe))

Registers SOURCE_PROFILE wire type and three nodes (Load, Define, List) following the SubjectProfile
  pattern. Adds five REST endpoints: reload, list, get, save, delete under
  /fbtools/source_profiles/. Nodes are registered in get_node_list() under the Scene category.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **source-profile**: Add SourceProfileRegistry data model and persistence
  ([`8837975`](https://github.com/frost-byte/fbTools/commit/8837975186d2a9296abdfed00fcb894bf0918425))

Pure utility module (no ComfyUI deps) for the media-first subject catalog. Supports
  create/update/remove for profiles and subjects, entity type validation, wire dict generation for
  downstream PromptAssemble rendering, and JSON persistence with .bak backup. 50 tests, all passing.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **source-profiles**: Add video clip segmentation system
  ([`6bf0c5b`](https://github.com/frost-byte/fbTools/commit/6bf0c5b5c2a96b1a671b95e45cb617fd4510599a))

Adds clips array to SourceProfile for time-windowed video segments, VLM-assisted boundary detection
  via contact sheet, and per-clip action description. SceneCastBuild now accepts clip_id_1/2/3 to
  select which segment of each connected source profile to load; _resolve_cast_media looks up clip
  load_params when a clip_id is specified.

- utils/source_profiles.py: _normalize_clip, set_clips, upsert_clip, remove_clip, auto_partition,
  get_clip, clip_load_params, DEFAULT_* constants; define_profile preserves clips and
  default_segment_duration - utils/source_profile_analysis.py: segment detection and clip
  description prompts, _parse_segments_response, parse_clip_description_response - extension.py:
  SceneCastBuild clip_id_1/2/3 inputs + fingerprint; execute stores clip_ids in cast dict;
  _resolve_cast_media uses clip_load_params when clip_id is set; REST endpoints auto_partition,
  upsert_clip, remove_clip, detect_segments, describe_clip - js/api/source_profiles.js:
  detectSegments, describeClip, autoPartition, upsertClip, removeClip API methods - tests: 41 new
  tests for clip CRUD, auto_partition, clip_load_params, _parse_segments_response,
  parse_clip_description_response

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **source-profiles**: Extend backend nodes, clip prompt node, REST API
  ([`a6f9199`](https://github.com/frost-byte/fbTools/commit/a6f9199876531f773943c4f1b48c8a9f424705ba))

- captioner.py: expose clean_caption_text() as public API for callers outside captioner (used by
  source profile analysis pipeline) - extension.py: SourceProfileLoad / SceneCastBuild node updates;
  SourceProfileClipPrompt node for dynamic clip_id selection; new REST endpoints: proxy_status,
  prebuild_proxies, describe_clip, remove_clip; DatasetCaptioner refactor; widget name cleanups

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **source-profiles**: Proxy cache, analysis pipeline, profile data model
  ([`32a7d76`](https://github.com/frost-byte/fbTools/commit/32a7d761f36502fbc46c7f492ad04f9e0a5ae30d))

- Add utils/proxy_cache.py: per-segment ffmpeg proxy builder with sidecar JSON tracking; scale
  filter commas escaped for ffmpeg filter- graph parser; _is_fresh uses os.path.realpath() for
  symlink-safe comparison against ComfyUI's symlinked output directory - source_profile_analysis.py:
  extended segment analysis pipeline - source_profiles.py: profile data model updates -
  reference_bundles.py: reference bundle helpers

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **source-profiles**: Segment editor UI overhaul
  ([`7c39f9a`](https://github.com/frost-byte/fbTools/commit/7c39f9ab702a923cf886ad669518ae385ebfffb3))

- Single-segment panel with ← N/M → nav replacing all-expanded card list - Timeline band click
  selects segment; video preview seeks to clip start - Active segment highlighted in timeline with
  filled triangle indicator - Collapsible Profile Settings and Subjects sections (accordion pattern,
  state persisted in _S.settingsOpen / _S.subjectsOpen) - Slot letters ({A}, {B}, …) shown inline
  after subject labels and updated live on checkbox toggle - Proxy dirty tracking: times_changed_at
  / proxy_built_at ISO timestamps stored on each clip and persisted via _persistClip; dirty clips
  show amber "needs rebuild" badge and amber dot in timeline band; _refreshProxyStatus advances
  proxy_built_at when server confirms fresh

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Add clips timeline panel to Source Profile editor
  ([`d5ad8c7`](https://github.com/frost-byte/fbTools/commit/d5ad8c774ec2e4eaf7a93773c906bb2a044744a2))

Adds a collapsible Clips section to the profile detail view (video profiles only) with:

- Canvas-based timeline rendering colored bands per clip with internal boundary handles that can be
  dragged left/right to adjust split points - Gold dashed markers showing VLM-detected boundary
  suggestions - Time axis ticks and per-clip labels scaled to total video duration - Auto-detect
  video duration via hidden <video> loadedmetadata - "Auto-segment" button — calls REST
  autoPartition, replaces clip list - "Detect boundaries" button — calls detectSegments VLM
  endpoint, renders suggestions on timeline without committing - Per-clip cards: label, start/end
  time inputs, action textarea, subject checkboxes (linked to profile subjects), Describe button
  (describeClip) - Manual "Add clip" appends at end with configured segment duration - All mutations
  sync to profile.clips and trigger onClipsChanged so the Save button picks them up via collectMeta
  spread

Also adds SourceProfilesAPI methods: detectSegments, describeClip, autoPartition, upsertClip,
  removeClip.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Add dedicated LLM tab with Local/Modal/Gemini backend sub-tabs
  ([`d6597d4`](https://github.com/frost-byte/fbTools/commit/d6597d4a8943c12f34f6ad37f7c2cb4ed8110c8e))

- js/api/modal.js: ModalAPI (status/activate/deactivate) + VlmActivityAPI - js/ui/llm_panel.js:
  renderLlmPanel with three sub-tabs; getActiveCaptionerType() returns "modal" | "auto" |
  "gemini_flash" based on active backend; onBackendChange hook so header bar stays in sync; Modal
  tab has preset selector, custom HF ID history, quantize toggle, idle timeout, keep-warm,
  connect/disconnect button - js/ui/fbt_panel.js: add LLM tab to TABS; header bar shows active
  backend label (Modal: blue dot, Local: green, fallback: "No backend — configure in LLM tab"); wire
  onBackendChange to re-sync header on Modal state change - js/ui/source_profile_editor.js: remove
  CAPTIONER_TYPES dropdown and Gemini checkbox; replace with read-only backend badge driven by
  getActiveCaptionerType(); all three VLM request paths (analyze, detect_segments, describe_clip)
  now call getActiveCaptionerType() rather than reading local state

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Add Source Profiles sidebar panel
  ([`2b2eb9e`](https://github.com/frost-byte/fbTools/commit/2b2eb9ef7944021ef187332cfb709e542fcc65e7))

- js/api/source_profiles.js — REST client for source profile CRUD, reload, LLM analyze, and analysis
  history endpoints - js/ui/source_profile_editor.js — full catalog browser panel: profile list with
  search, detail view (meta form + media preview), subject annotation list (add/edit/delete), LLM
  focused-pass analyze section (pass-type pills, captioner selector, prompt override, candidate
  review with add/add-all, history accordion), and auto-save - js/fb_tools.js — import and
  registerSidebarTab for 'fbt.source-profiles'

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Consolidate sidebar into single fbTools panel with lazy tabs
  ([`883152b`](https://github.com/frost-byte/fbTools/commit/883152bd0b56eaf5928d3b36626bfea66c31d1ef))

Replaces five separate sidebar tab registrations with one unified panel:

- js/ui/fbt_panel.js (new): shell with persistent LLM status bar in the header, horizontal tab strip
  (Compose/Bundles/Casts/Sources/History), and lazy mounting — each tab's DOM is created once on
  first activation and kept alive hidden on switch so state is never lost - js/fb_tools.js: single
  registerSidebarTab("fbt.panel") replaces the five individual registrations -
  composition_editor.js: _llmUpdateStatus now pushes state to the panel header via
  window._fbtUpdateLlmStatus (synchronous, no extra fetch) so the LLM badge reflects load/unload
  instantly from any tab

The shared fbtLlm object (exported from fbt_panel.js) will serve as the source of truth for tabs
  that need to check whether a model is loaded before routing inference calls.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

### Refactoring

- **gemini**: Move API key to server-side env var only
  ([`ebda2f2`](https://github.com/frost-byte/fbTools/commit/ebda2f239a20c42b7b68575f8dcace984e873039))

Remove gemini_api_key from all frontend-to-backend paths. The key is now read exclusively from the
  GEMINI_API_KEY environment variable in extension.py. No credentials are accepted from request
  bodies or widget inputs.

- DatasetCaptioner node: remove gemini_api_key widget input and execute() parameter -
  _run_vision_inference(): remove api_key parameter; Gemini path reads
  os.environ.get("GEMINI_API_KEY") internally - _run_vision_inference_clip(): remove unused api_key
  parameter - /analyze, /detect_segments, /describe_clip, /recaption_single endpoints: drop
  body.get("gemini_api_key") fallback

- js/api/source_profiles.js: remove gemini_api_key from analyze(), detectSegments(), describeClip()
  signatures and request bodies - js/nodes/dataset_caption_viewer.js: remove from state, widget
  sync, and recaption request body - Tests updated accordingly

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Panel consolidation, bundle editor, node inspector
  ([`9b4fc00`](https://github.com/frost-byte/fbTools/commit/9b4fc008437204501fa62588fbc00bf7b1ab28c3))

- Consolidate sidebar into unified fbTools panel with lazy tabs (fb_tools.js + fbt_panel.js) -
  Bundle editor updates: pronoun style, short name field, image list improvements, appearance
  analyzer integration - Node inspector tab: collapsible JSON tree for selected node data - Run
  history: capture map extraction, run parsing improvements - File tree: path insertion and
  filtering fixes - story.js: remove stale widget helpers - lora.js: LoraStackBuilder active-row
  display fix - style.css: new panel, clip, and inspector styles

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

### Testing

- Widget name contract checks and dataset caption API tests
  ([`9b13f31`](https://github.com/frost-byte/fbTools/commit/9b13f31a50e9123f6b8b3a20f99f103ae5afc710))

- test_widget_name_contracts.py: cross-layer test that fails when any JS w.name === "x" lookup
  references a widget name not present in the Python node schema; run after any node schema change -
  test_dataset_caption_api.py: updated for refactored captioner API

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.23.0 (2026-08-21)

### Documentation

- Add audio reference rules, brave MCP guide, and generic workflow scanner
  ([`48a8e7f`](https://github.com/frost-byte/fbTools/commit/48a8e7fb912ce1d8d783bdfb86d585643b341e9e))

Add H3 Ref2VA audio reference constraints and community-validated guidance, an empirical audio
  observations log, a generic brave-devtools MCP setup guide (machine-specific config gitignored via
  docs/*.local.md), and a general-purpose scan_node_usage.py script replacing the mixlab-specific
  scanner.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

### Features

- **ui**: Migrate all file pickers to FileTree with input/output tabs
  ([`061e373`](https://github.com/frost-byte/fbTools/commit/061e373271d44ebd496ed94561f1d0de3b2390f0))

Replace flat <select> dropdowns and hand-rolled tree implementations in bundle_editor and
  composition_editor with the shared buildFileTree component.

- bundle_editor: video picker, audio-from-video picker, separate audio picker all use buildFileTree
  with Input/Output tabs; new video_dir and audio_dir fields persist which directory was selected -
  composition_editor: bg editor and outfit editor drop ~160 lines of duplicated
  _insertPath/_renderTreeNode/tab logic in favour of buildFileTree - bundlesApi.streamUrl and
  mediaInfo accept an optional dir parameter - Backend resolves bundle video and audio file paths
  from the saved dir field (visual.video_dir, audio.video_dir, audio.audio_dir) so output-dir files
  load correctly in ComfyUI nodes - preview_sampled endpoint supports dir parameter for output
  videos

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.22.0 (2026-08-20)

### Features

- **llm**: Unified run history across all seven LLM features
  ([`d21bf95`](https://github.com/frost-byte/fbTools/commit/d21bf95a75cba12392a776413f5b3cd3bce42c10))

Single llm_history.json file with kind-discriminated entries replaces the old
  video_describe_history.json. History now covers: video_describe, bg_analyze, outfit_analyze,
  shot_action, dialogue, polish, and appearance_analyze. Shared buildHistorySection() utility
  renders the collapsible accordion for both modals and the sidebar.

- Backend: /fbtools/llm/history (GET ?kind= filter, POST, POST /delete) migrates old flat entries on
  first read; legacy /describe_history routes shim to the new handlers - js/utils/llm_history.js:
  makeEntry() + buildHistorySection() - Sidebar shows shot_action+dialogue+polish together; Restore
  applies result text to the currently focused shot card - BG, Outfit, Appearance modals each get
  their own filtered history accordion with kind-appropriate Restore behaviour

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.21.0 (2026-08-20)

### Features

- **llm**: Add server-side video describe history with REST API
  ([`826b2f0`](https://github.com/frost-byte/fbTools/commit/826b2f09d0e379c03d3057b0191ae5214c0ee32f))

History entries are stored in user_data_dir/video_describe_history.json via GET/POST/DELETE
  endpoints instead of browser localStorage. Tracks composition name, shot, video, extraction
  settings, prompts, and result so runs are restorable across sessions.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.20.0 (2026-08-20)

### Bug Fixes

- **backgrounds**: Use correct modal-overlay CSS class in _openBgEditor
  ([`b660b6d`](https://github.com/frost-byte/fbTools/commit/b660b6d3d28166964ca8f2ef068be2a8971ef2e9))

_openBgEditor was creating the overlay with class fbt-ce-overlay which has no CSS rule, so the
  overlay rendered as an unstyled block element (no position:fixed, no backdrop) instead of a
  fullscreen modal. Each click appended another invisible div, causing the "two forms" symptom. Fix:
  use fbt-ce-modal-overlay to match _openOutfitEditor and the CSS.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **bundle-editor**: Pass audio start_time/duration for extract_from_visual source
  ([`661249e`](https://github.com/frost-byte/fbTools/commit/661249e1205d37e326194a7e734c5c6e551bf6d2))

The Process Audio button was hardcoding start_time=0 and duration=0 when audio source is
  "extract_from_visual", ignoring the Timing section values the user set. All three source modes
  write timing into b.audio.start_time / b.audio.duration via _buildAudioTimeSection, so always use
  those values unconditionally.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **bundle-editor**: Show codec info and ffmpeg conversion command on video error
  ([`1dae58e`](https://github.com/frost-byte/fbTools/commit/1dae58ea9b64343ce751711bbecfa67bcff6e708))

mediaInfo now returns a 'codec' field (cv2 CAP_PROP_FOURCC fourcc string, e.g. HEVC, avc1, xvid).
  The info line shows it alongside duration/fps/dims.

The error handler now provides actionable feedback: - MKV/AVI: tells user these formats aren't
  browser-playable, shows exact ffmpeg command to convert to H.264 MP4 - H.265/HEVC detected from
  codec field: specific re-encode command - MOV/MP4 with unknown codec: suggests re-encode with
  codec detail - All messages include the ready-to-run ffmpeg command with -map_metadata and
  -pix_fmt yuv420p flags

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **bundle-editor**: Use loadedmetadata event + error handler for video preview
  ([`ea5d519`](https://github.com/frost-byte/fbTools/commit/ea5d5193ed378b116df9b94a99f0a746af9f76f1))

Previously the video player showed black/greyed-out controls with no feedback when the stream failed
  or the format wasn't browser-playable (.mkv, .avi, H.265 .mp4). Duration detection also relied
  solely on the mediaInfo endpoint; if that was slow or unavailable the trim slider never appeared.

- Add loadedmetadata listener as primary slider trigger (more reliable than waiting for mediaInfo
  alone) - Add error listener showing a human-readable message per MediaError code (network error,
  unsupported format, not found) - Track _onMeta/_onErr refs in _buildVideoPicker scope so each
  _loadFile call removes stale listeners before registering new ones

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **bundle-editor**: Use setAttribute for input list property in _mk helper
  ([`ba67a03`](https://github.com/frost-byte/fbTools/commit/ba67a0376b8c7085737825fc6fe9bd4f0bc41dce))

HTMLInputElement.list is a read-only getter; assigning it directly via el[k] = v throws "Cannot set
  property list … which has only a getter". Route it through setAttribute("list", v) so datalist
  bindings on the character-sheet and LLM image inputs in the subject form render correctly.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **bundle-editor**: Zero skip_first_frames when trim slider sets start_time
  ([`a052236`](https://github.com/frost-byte/fbTools/commit/a05223690a813b2a6565dfa6097cbc2d6b90b84b))

skip_first_frames is the legacy VHS-style frame-count seek; start_time is the time-based replacement
  introduced with the trim slider. Setting both simultaneously double-offsets the seek position.
  Clear the legacy field whenever the slider or Mark Start button writes start_time.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **conditioning**: Add node_id class attr to CompositionToH3Conditioning
  ([`bc9e4d0`](https://github.com/frost-byte/fbTools/commit/bc9e4d0f361c10da81fff2f058eee6bfac3e21b1))

cls.node_id in execute() raised AttributeError on the ComfyUI-cloned class because node_id was only
  passed to io.Schema(), not stored as a class attribute. Add it explicitly so the clone inherits
  it.

- **conditioning**: Fix H3 video/audio loaders and raise frame_load_cap default
  ([`69e751b`](https://github.com/frost-byte/fbTools/commit/69e751b0429110636437b3d9be2e97aca8cf3068))

Rewrite _h3_load_video_frames to use cv2 exclusively with a VHS-equivalent time-accumulator
  resampling loop — eliminates the broken VHS import path that caused UnboundLocalError and silent
  frame failures.

Replace the torchaudio fallback in _h3_load_audio with a direct ffmpeg subprocess call mirroring VHS
  get_audio: -ss/-t for accurate seeking, -f f32le piped to stdout, SR and channel count parsed from
  ffmpeg stderr. Fixes sample-rate mismatch (rapid playback) and wrong audio segment caused by
  torchaudio loading entire stream before slicing.

Raise frame_load_cap default from 16 to 96 across extension.py, reference_bundles.py, and
  bundle_editor.js so H3's n%17==5 trimming leaves 90 frames (8 Qwen samples) instead of 5 (1
  sample). Add a safety floor in the loader that upgrades any legacy cap < 39.

Also includes SceneCompose scene_synopsis input, PromptCompositionLoader filename_prefix output, and
  media frame extraction REST endpoints (_media_extract_frame, _media_delete_tmp_frame) that were
  developed alongside these fixes and could not be cleanly separated without patch-mode staging.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **conditioning**: Suppress skip_first_frames when start_time is set in video loader
  ([`98389ce`](https://github.com/frost-byte/fbTools/commit/98389cef7f984cc3a75f45fc59d66d908106b6c1))

When both start_time (time-based seek) and skip_first_frames (legacy VHS-style frame-count seek) are
  non-zero, the video loader was applying both, doubling the offset and seeking past the end of the
  video.

The trim slider added in da2018c sets start_time but does not zero skip_first_frames on existing
  bundles, so any bundle configured with the old frame-count approach would seek to (start_time +
  skip/fps) instead of the intended start_time.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **info-section**: Fix label overlap on wrapped multi-word info labels
  ([`58754a6`](https://github.com/frost-byte/fbTools/commit/58754a6d960b310ed4d55981cb15eb14a4288e59))

align-items: center was causing the checkbox to sit at the vertical midpoint of a wrapped 2-line
  label (e.g. DIALOGUE TAGS), making the second line appear to overlap the checkbox.

Switch to flex-start so the checkbox pins to the top of the label. Widen the label from 42px to 56px
  to reduce wrapping on shorter labels. Add padding-top: 2px to keep single-line labels optically
  aligned with their adjacent inputs.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **outfit**: Guard get_folder_paths calls against KeyError
  ([`e756a1b`](https://github.com/frost-byte/fbTools/commit/e756a1bc95af94a58a7aa47310ffc4cce00a427e))

folder_paths.get_folder_paths() raises KeyError for unregistered folder types rather than returning
  an empty list. Wrap both calls in a helper so "sams" is used when registered and "sam2" is
  silently skipped when not.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **outfit**: Use folder_paths.get_folder_paths for SAM2 model discovery
  ([`9e90e24`](https://github.com/frost-byte/fbTools/commit/9e90e2426b945ca2d24ad1c5985cd7b22817bda2))

The SAM2 status endpoint was constructing model search paths manually from folder_paths.models_dir,
  which resolves to the ComfyUI launch directory rather than the paths registered via
  extra_model_paths.yaml.

Switch to folder_paths.get_folder_paths("sams") + get_folder_paths("sam2") so all
  extra_model_paths.yaml-registered sams directories are searched, with the manual construction as
  fallback.

Also re-fetch sam2_status live on every outfit modal open so the UI reflects the current server
  state without requiring a page reload after the model is installed.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **outfit-modal**: Fix label layout and add ref image thumbnails
  ([`c6ddc19`](https://github.com/frost-byte/fbTools/commit/c6ddc19087cb0489a04b2fe13f176277cd493dca))

Wrap ID/Name/Tags/Description fields in .fbt-ce-row so labels sit horizontally beside their inputs
  instead of appearing right-justified in a column. Add a Reference Images section header row for
  consistency.

Add 48x48 thumbnail to each reference image row using _ceViewUrl() so users can visually identify
  added refs. Add CSS for .fbt-ce-outfit-ref-row and .fbt-ce-outfit-ref-thumb.

Pass folder param to all _addFileToRefs() callers (addRefBtn, analyzeBtn, SAM2 addResultBtn) so
  thumbnail URLs resolve correctly.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **outfit-modal**: Populate outfit list after resource load
  ([`b8599af`](https://github.com/frost-byte/fbTools/commit/b8599af21dfcd0e2bd436ad1f6c99ddd85da14b0))

_refreshSidebar() was missing a _rebuildOutfitList() call, so outfits loaded from disk were never
  rendered into the sidebar after page load or panel re-open.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **sam2**: Bypass Hydra global state when loading SAM2 config
  ([`da4e101`](https://github.com/frost-byte/fbTools/commit/da4e1019540309b213bdf635a7c7894f0d3ca910))

Other custom nodes (Comfyui-SecNodes) call GlobalHydra.instance().clear() and re-initialize Hydra
  with their own config module. This breaks the standard build_sam2() path because sam2/__init__.py
  skips its own initialize_config_module() call when Hydra is already initialized, leaving compose()
  searching the wrong config tree.

Fix: load the SAM2 YAML directly from the package directory via OmegaConf.load() +
  hydra.utils.instantiate(), bypassing Hydra's global config resolution entirely. Apply the same
  postprocessing overrides that build_sam2() normally adds via OmegaConf.update(force_add=True).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **sam2**: Search sys.path for config instead of using import sam2
  ([`6f2dabc`](https://github.com/frost-byte/fbTools/commit/6f2dabcb1fdc4a2bdf193773da70611622a4bd35))

import sam2 resolved to ComfyUI-RMBG/models/sam2/ (a bundled copy with no configs directory) instead
  of the installed package in site-packages. Search sys.path directly for a sam2/ directory that
  contains the needed config file so bundled shadow copies are skipped.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **scene**: Restore SceneCastBuild entries on workflow reload
  ([`6292270`](https://github.com/frost-byte/fbTools/commit/629227041c43bf5f02e71e5279cd0abaa03fd518))

onNodeCreated fires before ComfyUI restores widget values from the saved workflow, so the initial
  JSON parse always saw '[]'. Hook onConfigure (which fires after widget values are applied) and
  re-read the widget there to rebuild the table with the saved entries.

- **scene**: Rewrite SceneCastBuild with JSON-backed entries
  ([`d234f0c`](https://github.com/frost-byte/fbTools/commit/d234f0c709dae6c04725fd826c1fcbc8f446498d))

Replace the 16 individual boolean/combo inputs (subject_N, bundle_N, visual_mode_N, use_audio_N)
  with a single io.String.Input("cast_entries_json") storing a JSON array. Eliminates the io.Boolean
  toggle widgets that bypassed setWidgetVisible and leaked into the node UI.

The JS DOM table now manages entries directly in JS state, syncing to the hidden JSON widget via
  _syncToWidget(). The cast editor's _pushToNode() is simplified to a single JSON.stringify write +
  _refreshCastTable() call.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **settings**: Update MelBand model path to expect .safetensors (Kijai builds)
  ([`eba49b1`](https://github.com/frost-byte/fbTools/commit/eba49b19debfb121f04c65ead4b69c1919f84b3f))

Kijai/MelBandRoFormer_comfy provides fp16 (456 MB) and fp32 (913 MB) safetensors checkpoints — not
  .pth. Update placeholder, tooltip, and backend comment to reflect the correct format and source.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Analyzer image pool now stays in sync with bundle's image list
  ([`430373b`](https://github.com/frost-byte/fbTools/commit/430373be901bb66086fdf8fb0fac027aa1b8f7bf))

Previously the Analyze Appearance dropdown was populated once at form-render time. If b.visual.files
  was empty then, it fell back to all images in the input directory and never updated — so adding an
  image to the bundle left the LLM section pointing at the wrong pool and the user had to manually
  pick the right file from a large unsorted list.

Changes: - _buildAppearanceAnalyzer: replace one-shot imagePool capture with a rebuildPool() that
  re-reads b.visual.files on every call; auto-selects and previews the image automatically when the
  bundle has exactly one file; stored as sec._refreshPool for external wiring - _buildImageList:
  accepts an onFilesChange callback; fires it on every add and remove so the analyzer stays in sync
  immediately - _renderForm: holds _llmEl ref (set after analyzer is built) and passes () =>
  _llmEl?._refreshPool?.() into _buildImageList so the two sections are wired together through the
  closure

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Escape double quotes in paceSel.title string
  ([`4610a41`](https://github.com/frost-byte/fbTools/commit/4610a419c2a178c90ec12f276242ffd883896b44))

Unescaped double quotes inside a double-quoted JS string caused a parse error that prevented the
  entire fb_tools.js module chain from loading.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Reload LibberManager table from saved libber_name on workflow load
  ([`751ccea`](https://github.com/frost-byte/fbTools/commit/751cceac854ee367520c180edd5205e00f1fe3b8))

onNodeCreated fires before ComfyUI restores widget values, so refreshTable() was reading the default
  combo value instead of the saved libber_name. Hook onConfigure (which fires after values are
  applied) to re-run refreshTable with the correct name.

- **ui**: Remove CSS import from index.js, loaded by fb_tools.js link tag
  ([`ddd2c22`](https://github.com/frost-byte/fbTools/commit/ddd2c22c5c3e3544ad5be67c1d3a85b82279a386))

Browsers reject CSS loaded via ES module import (strict MIME checking). The stylesheet was already
  injected correctly via a <link> element in fb_tools.js; the duplicate import in index.js broke the
  entire module chain.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Resolve sidebar double-click bug for fbTools panels
  ([`3dcfbd9`](https://github.com/frost-byte/fbTools/commit/3dcfbd9d1c2f49dabbc88df468789464df961488))

ComfyUI calls render(el) while the previous panel's DOM is still present. All four tab guards
  checked for `.fbt-be-panel` — shared by both bundle and cast editors — so switching from one to
  the other silently skipped rendering the new panel, leaving the old one visible.

Fix: stamp each editor's root panel with a unique `data-fbt-editor` attribute ("bundle" / "cast")
  and guard against that specific value. The composition editor gains an equivalent guard on
  `.fbt-ce-panel`. Each render function already calls `el.innerHTML = ""` before building, so the
  old panel is cleared automatically when the guard passes.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Restore widget-dependent UI on workflow reload for Story/Scene nodes
  ([`7a8bba9`](https://github.com/frost-byte/fbTools/commit/7a8bba9db006c16f081b5957f6b2573f06202151))

Add onConfigure hooks to StoryEdit, StorySceneBatch, and SceneSelect so their tables/dropdowns
  reload from the correct saved widget values after a workflow is opened. Each onNodeCreated fires
  before ComfyUI restores widget values, so the initial loads read default values instead of the
  saved ones.

- StoryEdit: store loadStoryData on node._loadStoryData, call from onConfigure - StorySceneBatch:
  re-fetch job_id options for saved story_name on onConfigure - SceneSelect: store updateSceneDir on
  node._updateSceneDir, call from onConfigure

- **ui**: Set widget.hidden and widget.element in setWidgetVisible
  ([`53c6627`](https://github.com/frost-byte/fbTools/commit/53c6627d91e7f5103f2e93ae8cd330b4b8b5c080))

Modern ComfyUI frontend gates visibility on widget.hidden; the old widget.type='hidden' fallback
  only suppresses the LiteGraph canvas draw but leaves the DOM element (widget.element) visible. Add
  both properties and optionally call node.setSize() when a node reference is provided.

### Documentation

- **scene**: Clarify SceneCastBuild works with PromptCompositionLoader
  ([`107e7c2`](https://github.com/frost-byte/fbTools/commit/107e7c26726f8d404812222aa309a028a114a0d2))

Both SceneCastLoad and SceneCastBuild output SCENE_CAST type, so either wires into
  PromptCompositionLoader's scene_cast input. Update tooltip and docstring to reflect this.

### Features

- **audio**: Add MelBand Roformer vocal extraction to preprocessing pipeline
  ([`07c7534`](https://github.com/frost-byte/fbTools/commit/07c75340e207c402b634560704179f333d891f01))

When a MelBand Roformer safetensors checkpoint is configured, noise_removal now runs source
  separation instead of spectral subtraction, producing a clean vocal stem. Falls back to the
  existing spectral-subtraction path when no model path is set. Model is cached in-process after
  first load.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **backgrounds**: Replace inline form with modal + LLM analysis
  ([`3418a83`](https://github.com/frost-byte/fbTools/commit/3418a8360d4c64c8f8b5d11bc88d6c3a4fca7a9b))

Convert background create/edit from an inline sidebar form to a full modal matching the outfit
  editor pattern.

New capabilities: - File browser (Input/Output tabs, tree, image/video preview, frame picker) -
  Reference images list with thumbnails and role dropdown - LLM analysis via POST
  /fbtools/backgrounds/analyze_media — asks the model to return structured JSON and auto-fills
  Description, Lighting, and Soundscape fields in one call - Clicking a reference thumbnail loads it
  into the browser preview - Delete button moved into the modal footer

Backend changes: - New /fbtools/backgrounds/analyze_media endpoint with JSON-structured LLM query
  (description + lighting + soundscape); strips markdown fences and falls back to raw text if JSON
  parse fails; supports folder param so output-dir images work too - list_backgrounds() now returns
  full records (was only id/name/description summaries, so lighting/soundscape were lost on edit) -
  background schema gains reference_images field (persisted as-is by existing save_background
  passthrough) - analyzeBackground() added to CompositionsAPI

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **bundle**: Add audio preprocessing pipeline with caching
  ([`38b0a86`](https://github.com/frost-byte/fbTools/commit/38b0a863c54cedd31ad41d1a1ae9e690cc2160cb))

- utils/audio_preprocess.py: pure numpy pipeline (spectral denoise, LUFS normalize via pyloudnorm,
  loop/truncate) with cache fingerprinting - POST /fbtools/bundles/preprocess_audio endpoint: runs
  pipeline in executor, caches result as WAV under user_data_dir/bundles_cache/ - audio_cache field
  plumbed through prompt_compositions, prompt_assembler, _resolve_cast_media, and
  CompositionToH3Conditioning to bypass raw load - Bundle Editor: _buildAudioProcessingSection()
  with noise/LUFS toggles, target LUFS input, status badge, and Process Audio button -
  tests/test_audio_preprocess.py: 36 tests covering all pipeline steps

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **bundle**: Add sampled preview with frame-count readout
  ([`76fcfbf`](https://github.com/frost-byte/fbTools/commit/76fcfbf5eb21a33700ffd7d503ee59e68d986905))

- Remove frame_load_cap and skip_first_frames from video section UI (start/end is now set via
  slider/mark, cap defaults to 0) - Add live frame-count readout: "~N frames · X fps effective"
  updates when any of FPS override, Every Nth, start, duration, or slider changes - Add POST
  /fbtools/bundles/preview_sampled endpoint: cv2 time-accumulator resampling (mirrors H3
  conditioning logic) → ffmpeg pipe → fragmented MP4 - Add "▶ Preview Sampled" button in the video
  section: plays sampled preview in a separate in-panel video element; falls back to native clip
  loop when ffmpeg unavailable with an explanatory note - Remove frame_load_cap default from 96 to 0
  for new bundles

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **bundle**: Add separate video file as audio source for Reference Bundles
  ([`dc0e53d`](https://github.com/frost-byte/fbTools/commit/dc0e53d25d48937dc0f8e8b9bf5c1107622eeb36))

Users can now excerpt audio from a different video than the visual reference. Adds
  extract_from_video audio source option in bundle_editor.js and handles both the per-entry and
  legacy flat paths in _resolve_cast_media.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **bundle**: Add time-based trim (start_time/duration) to video visual references
  ([`4b3ca7b`](https://github.com/frost-byte/fbTools/commit/4b3ca7b6a2ba264373b06ab7fd063479bfd8c863))

_h3_load_video_frames now seeks to start_time via cap.set(CAP_PROP_POS_MSEC) and caps the output
  frame count from duration * target_fps. Both params are propagated through _resolve_cast_media
  entry_load_params and defaulted in BundleRegistry.upsert(). Bundle editor video visual section
  gains a "Trim" section with Start (s) / Duration (s) number inputs above the existing frame
  sampling controls.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **bundle-editor**: Add processed audio preview player
  ([`1fbde0d`](https://github.com/frost-byte/fbTools/commit/1fbde0de5708f3a99ac8a7a6714870b8413272fe))

After running "Process Audio", an <audio> player appears below the status line so the processed
  (denoised/normalized) result can be listened to without leaving the editor.

Backend: GET /fbtools/bundles/audio_cache/stream?path=<abs_path> serves files from
  user_data_dir()/bundles_cache/ with Range support. Path is validated to stay within bundles_cache/
  root.

Frontend: _buildAudioProcessingSection now creates a hidden <audio> element alongside the status
  line. _updateStatus shows/hides and reloads the player whenever audio_cache changes — on initial
  render (if cache already set), after a successful Process run, and when settings are changed
  (clears cache → hides player).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **bundle-editor**: Add Subjects tab, image tree browser with folder tabs and dual preview
  ([`aea617f`](https://github.com/frost-byte/fbTools/commit/aea617f4699ebc3567c2d2d8b395914d9c239f6f))

- Add Subjects tab to Reference Bundle editor with full profile editor (name, ID, concept ID,
  appearance w/ collapsible subfields, character sheet images with per-image role selects, voice
  settings, LLM analysis) - Replace flat image picker with lazily-rendered collapsible tree browser
  supporting subdirectories - Add Input/Output folder tabs to the tree browser so images from either
  ComfyUI directory can be browsed with identical tree-view behaviour - Split the single preview
  pane into two independent panes: a stable selected-image preview (click any assigned filename to
  change it, tracks reorder/remove) and a dynamic browse-hover preview below the tree - Backend: add
  `folder` param to /fbtools/media/list (input|output); add output-dir fallback for subject image
  loading and LLM image loading

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **cast**: Add media frame extraction endpoints and scene synopsis input
  ([`920ea66`](https://github.com/frost-byte/fbTools/commit/920ea666e6a3be83e1a07be5f8344866780f9437))

Add REST endpoints for extracting a single frame from a video file (_media_extract_frame,
  _media_delete_tmp_frame) with temp file cleanup via _purge_old_tmp_frames. Add scene_synopsis
  string input to SceneCompose, injected into the SCENE_INSTANCE for H3 summary override.

Wire frame extraction into the bundle editor UI (visual frame picker for thumbnail/reference
  selection) and expand SceneCastBuild JS node with bundle media preview and audio picker support.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **cast**: Positional subject recasting in Scene Cast
  ([`b001805`](https://github.com/frost-byte/fbTools/commit/b0018055dff595cb9da12ed2fe4d755c436b6837))

Cast entries now map to composition slots by row order rather than subject_id identity. Entry 0
  targets S1, entry 1 targets S2, etc. A blank entry (no subject_id) is a pass-through that keeps
  the composition's original subject unchanged.

When a subject_registry is available and the entry's subject_id differs from the slot's current
  subject, the slot's subject is fully replaced (name, appearance, voice, concept_id) before bundle
  enrichment applies. This enables recasting: telling the Prompt Composition Loader "use Angie in S1
  and Joe in S2 for this shoot, regardless of who the composition originally assigned."

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **composition**: Add speech_pace to shot dialogue with trim_to estimation
  ([`798977f`](https://github.com/frost-byte/fbTools/commit/798977ffb71e27740ae71b7327d70cd131b03f6b))

Adds a per-shot `speech_pace` field ("slow" / "normal" / "fast") to the composition dialogue schema.
  The field drives two things simultaneously:

- **Prompt injection**: slow/fast paces append a qualifying phrase to the shot's action text
  ("speaking slowly and deliberately" / "speaking quickly") in both h3_ref2va and h3_fl2va formats.
  Normal pace emits nothing. - **Voice reference trim_to**: `PromptCompositionLoader` estimates the
  expected spoken duration of each shot's resolved dialogue (via `estimate_speech_duration`, a new
  public utility) and accumulates it per slot. The per-slot totals are passed to `_build_h3_refplan`
  as `slot_trim_to`, injecting a `trim_to` field into standalone audio reference entries so the
  terminal node can trim the voice clip to match the generated line length.

Pace → chars/sec mapping: slow=10, normal=13, fast=16 (documented in
  `docs/audio_reference_observations.md` for field revision as data arrives).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **composition**: Enhance prompt assembler and add filename_prefix output
  ([`a771e1b`](https://github.com/frost-byte/fbTools/commit/a771e1b8a8a365dd97a2ccfe815ebd4c0b0d52b4))

Expand prompt_assembler.py with additional model-type formatters and assembly logic. Add
  filename_prefix string input/output to PromptCompositionLoader so the composition name can be
  wired directly into a VHS_VideoCombine filename_prefix input.

Update composition editor UI, LLM client/scanner utilities, and extend test_prompt_assembler.py
  coverage. Add prompt_assembly.md docs.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **composition**: Improve Saved Compositions UX and add LoRA search
  ([`cd0a431`](https://github.com/frost-byte/fbTools/commit/cd0a431265d09301d4107b398fd5709ca108f7a5))

- Saved Compositions list: add search field (filters by name/ID) and pagination (10 per page) with
  prev/next controls - Make entire composition row clickable to load; remove redundant ⇩ button;
  delete button uses stopPropagation to avoid double-trigger - Active composition indicator: 3px
  green bottom border on the currently loaded entry in the saved list - LoRA name selector: replace
  plain <select> with searchable combobox; type to filter the full LoRA list, arrow keys to
  navigate, Enter to select; dropdown uses position:fixed to avoid sidebar clipping

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **compositions**: Add background and outfit visual reference subjects for H3 prompts
  ([`a5adc01`](https://github.com/frost-byte/fbTools/commit/a5adc01db97bb1bdd3d3691c567f29305852ced0))

Backgrounds: new "Include as <Subject N>" checkbox on the Background section of the composition
  editor. When checked, the background's reference_images are injected as an extra slot in the
  assembled prompt using the {BG} shortcut in shot action/camera text.

Outfits: the outfit dropdown in each subject slot now stores an outfit ID in outfit_ids[slot_key]
  rather than description text. Each outfit reference image gains a use_as_reference flag
  (per-image, toggled from the outfit modal). Outfits with at least one flagged image are injected
  as their own <Subject N> slot using {Fit_1}/{Fit_2}/… shortcuts. Outfits with no flagged images
  contribute their description text to the subject's appearance phrase (text-only path).

Background and outfit extra slots share a running letter counter so they coexist correctly (subjects
  → BG → Fit_1 → Fit_2 in alphabetical slot order).

Both call sites of assemble_composition() (REST endpoint + PromptCompositionLoader node) now resolve
  and pass resolved_outfits alongside resolved_background.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **conditioning**: Add §1 validation, trim_to, and turbo warning to CompositionToH3Conditioning
  ([`ea92e5e`](https://github.com/frost-byte/fbTools/commit/ea92e5e15eacfe3cdf8f5b7f096acd497ba69465))

Enforces all MiniMax H3 Ref2VA hard limits before delegating to the native node, with explicit
  errors rather than silent truncation:

- Pre-load: ≤ 3 standalone audio refs, audio must pair with a visual, ≤ 12 total reference files,
  trim_to < 2s caught early with an actionable message - Post-load (actual samples): per-clip 2–15s,
  total audio ≤ 15s - trim_to applied to waveform via sample-level slice after ffmpeg load - Turbo
  LoRA warning (logged + status update) when has_turbo_lora is set on the refplan and audio
  references are present

PromptCompositionLoader now sets has_turbo_lora on the refplan dict by checking whether any attached
  LoRA name contains "turbo".

Status line now includes total loaded audio duration.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **history**: Add Run History panel and Run Meta Capture node
  ([`f7bca75`](https://github.com/frost-byte/fbTools/commit/f7bca75f13f60f363da14137e43cccafe25ce4a1))

- RunMetaCapture node: captures runtime string values at execution time, stores by prompt_id;
  autogrow value slots (up to 12); supports partial execution via is_output_node play button; inline
  text preview - [track: Label] tag system: right-click any node to embed a tracking tag in its
  title; green pill bar drawn via onDrawForeground - Run History sidebar panel: merges /history
  widget snapshots with RunMetaCapture captures keyed by prompt_id; shows timestamp, workflow name,
  and short prompt ID chip per run - LoraStackBuilder custom renderer: filters disabled/None LoRA
  slots, suppresses video/audio columns for non-LTX targets, shows model badge

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **llm**: Add LLM video analysis assistant with Qwen2.5-Omni GPTQ support
  ([`cf61034`](https://github.com/frost-byte/fbTools/commit/cf61034deb0be0201d3be942a009031eb95e4c0a))

Adds an LLM assistant to the Composition Editor for analysing video clips and generating shot action
  descriptions.

## Infrastructure - `utils/llm_scanner.py`: scans ComfyUI LLM directories for GGUF and HF models;
  detects vision, native-video, and quantisation capabilities - `utils/llm_client.py`:
  load/unload/generate for GGUF (llama-cpp-python) and HF (transformers) models; task-specific
  prompt builders - REST endpoints: `/fbtools/llm/{models,status,load,unload,generate,
  generate/shot_action,generate/dialogue,generate/polish,
  describe_video,video_prompt,download/default}`

## Qwen2.5-Omni GPTQ workarounds - `block_name_to_quantize`: optimum's BLOCK_PATTERNS list omits
  `thinker.model.layers`; patching the embedded quantisation config dict before `from_pretrained`
  bypasses the pattern scan - `return_audio=False` + `thinker_max_new_tokens`: Omni's `generate()`
  returns `(text, waveform)` by default; standard token-slice decode indexed the batch dim instead
  of seq dim, producing empty text - Omni default system prompt prepended to custom instructions to
  silence the "audio output may not work" warning and stay in the model's trained operational mode -
  AWQ Triton monkey-patch retained for future use: forces `dequantize_gemm+matmul` fallback when the
  Triton bitshift kernel fails on packed int4 float16 weights - `max_memory` capped at 70% GPU / 64
  GiB CPU so LLM and diffusion models can coexist in VRAM

## Temporal RoPE encoding - `_extract_frames` now returns `{sample_fps, raw_fps, duration}` -
  `sample_fps = len(selected_frames) / clip_duration` is passed in the video content element so
  qwen_omni_utils computes correct temporal position IDs (previously defaulted to 2.0 fps regardless
  of actual frame rate)

## Composition Editor UI - "Describe from Video" modal: filmstrip with carousel, frame selection,
  frame budget indicator (tokens/frame from Omni's patch/merge params) - Modal stays open after
  inference; result appears in editable textarea; "Send to Action" pushes text into the focused shot
  card's Action field - Collapsible "Edit Prompt" section pre-populates system + user prompts from
  `/fbtools/llm/video_prompt`; edits are sent as overrides

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **lora**: Add Enabled Summary output to LoraStackBuilder
  ([`deb18ea`](https://github.com/frost-byte/fbTools/commit/deb18ea361a2ca6bbe77e3d842a30ab72036ec0e))

Adds a new string output that lists only enabled LoRAs, one per line, in the format: name
  model/clip[/video/audio]. Name is the basename truncated to 48 chars with extension stripped;
  values use minimal decimal notation (1, 0.5, 0.75). Output always ends with a trailing newline for
  easy concatenation with other string nodes.

A boolean input (Summary: Include Prev Stack, default off) controls whether the summary covers only
  LoRAs defined in this node or all merged entries including Prev Stack.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **outfit**: Replace LLM analyze text input with tree browser + preview
  ([`37c8856`](https://github.com/frost-byte/fbTools/commit/37c885640094936db99214bbbe21bc4b15cdeea9))

- Add Input/Output folder tabs with lazy subdirectory tree showing both images and videos combined
  (same tree pattern as bundle editor) - Add image preview pane (img) and video preview pane (video
  w/controls) that update when a file is selected in the tree - Add frame-time row (hidden for
  images, shown for videos): number input syncs bidirectionally with the video element's
  currentTime; "↺ Use current" button captures the scrubbed position from the video player - Pass
  frame_time in the analyze_media request when a video is selected - Add listMedia() to
  CompositionsAPI; load all four media lists (input/output × image/video) in _loadResources via
  Promise.allSettled

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **outfit-modal**: Click ref thumbnail to load into browser and SAM2
  ([`ea65c2c`](https://github.com/frost-byte/fbTools/commit/ea65c2c5d07ca2bef5d47e3a3cd7117ff0596d4b))

Clicking any reference image thumbnail in the ref list now calls _applySelection(), which updates
  the file browser selection, the preview, and fires _onSelectionForSam2 — so the user can jump
  straight from an existing reference into SAM2 segmentation without re-browsing the tree.

Teal hover glow on thumbnails signals the SAM2 association.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **outfits**: Add reference images and image/video LLM analysis
  ([`dbfe1ad`](https://github.com/frost-byte/fbTools/commit/dbfe1ad66aa1c96fae37a505a217f6f291bdb93a))

OutfitRegistry entries now carry reference_images: [{file, role}]. Legacy plain strings
  auto-normalize to {file, role: "costume detail"}.

Backend: - POST /fbtools/outfits/analyze_media: accepts image or video filename, extracts frame at
  1s for video (saved permanently as _outfit_ref_*.jpg), runs loaded LLM, returns {description,
  frame_file} - POST /fbtools/outfits/save: now accepts reference_images array

Frontend outfit editor: - Image-only input replaced with image/video input; shows hint when video is
  detected explaining frame extraction - After LLM analysis, analyzed file (or extracted frame) is
  appended to a mutable reference list when "Add as reference image" is checked - Reference list
  shows each entry with role dropdown and delete button - Save persists reference_images alongside
  name/description/tags

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **scene**: Add Scene Cast input to PromptAssemble for video references
  ([`3417ece`](https://github.com/frost-byte/fbTools/commit/3417ece37bfcec07752fee7904bbe33210df8e2c))

PromptAssemble was calling assemble_prompt() with video_entries=None, so subjects whose cast bundle
  has visual_mode='video' never received a <Video N> label in the H3 subject_definitions section.

Add an optional SCENE_CAST input; when connected, _resolve_cast_media() extracts video_entries_full
  and passes it to assemble_prompt() so video-referenced subjects are correctly labelled in the
  assembled prompt.

- **settings**: Add global audio + speech pace defaults to composition settings
  ([`34a2964`](https://github.com/frost-byte/fbTools/commit/34a2964841401ad7948c7d84a9057f84e57f52b3))

Backend (extension.py): - _COMPOSITION_SETTINGS_DEFAULTS adds default_speech_pace, default_audio_*,
  and melband_model_path alongside the existing libber_delimiter - POST
  /fbtools/compositions/settings validates and persists all new fields (pace: slow/normal/fast enum;
  LUFS: clamped to −36..−6; others: bool/str)

Composition editor UI (_buildSettingsSection): - "Default speech pace" dropdown (slow/normal/fast
  with WPM hint) - "Default audio processing" group: noise removal checkbox, LUFS normalize
  checkbox, target LUFS numeric input - "Vocal isolation" group: MelBand model path text input
  (reserved) - All controls stored in _dom for post-load sync in _loadResources() - New dialogue
  speech_pace defaults to _S.settings.default_speech_pace

Bundle editor: - Imports compositionsApi; _loadAll() fetches global settings into _S.settings -
  _startNew() reads default_audio_* to pre-fill new bundles' audio_processing

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **subjects**: Add per-image roles to character_sheet_images
  ([`56aea92`](https://github.com/frost-byte/fbTools/commit/56aea926230a605dc9ec2f523fc0a0c9e79a72da))

Changes character_sheet_images from list[str] to list[{file, role}]. Existing plain strings
  auto-migrate to {file, role:"character sheet"}.

Adds <Picture N> role lines to H3 ref2va subject_definitions so the model knows each image is an
  appearance reference and not a scene composition template — prevents spatial bleed from portrait
  framing (where a portrait shot's left-side face anchor was being replicated in the output layout).

Role descriptions always end with "do not use as scene composition". Canonical roles: character
  sheet, portrait, side profile, full body, costume detail, reference. Free-form strings are also
  accepted.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Add reusable FileTree component for input/output file browsing
  ([`da1fd7c`](https://github.com/frost-byte/fbTools/commit/da1fd7cd3e2892f1395b7d27f82a1bd637f19359))

Extracted from the composition editor's video/image file selectors into a standalone
  js/ui/file_tree.js module with Input/Output tabs, lazy folder expansion, and fbt-be-tree-file-cur
  highlight tracking.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Add Subject editor to Reference Bundles panel
  ([`750cb94`](https://github.com/frost-byte/fbTools/commit/750cb946b78e5f31e659610dcb083556623cad30))

Adds a full subject creation/editing UI to the bundle_editor sidebar panel under a new Bundles |
  Subjects tab switcher. Subjects now have a dedicated editor with all profile fields: name, ID,
  concept ID, appearance (summary + face/hair/body/outfit detail fields in a collapsible), voice
  (description, language, audio reference file dropdown), and a character_sheet_images list with
  per-image role dropdowns (character sheet, portrait, side profile, full body, costume detail,
  reference). Optional LLM appearance analysis fills the appearance summary from an image when a
  vision model is loaded.

Also adds getSubject, saveSubject, and deleteSubject methods to BundlesAPI and CSS for the tab
  switcher, subfield grid, and sheet-image rows.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Add video/audio/image previews and dual-handle trim slider to bundle editor
  ([`32a7857`](https://github.com/frost-byte/fbTools/commit/32a7857684a24e3f305bb2c2f8de58d540ed1f20))

Video visual section: - <video controls> player shown when a file is selected (streams via new GET
  /fbtools/media/stream endpoint which supports HTTP Range for seeking) - Info line shows duration,
  fps, resolution, frame count - Dual-handle range slider lets users drag start and end of the trim
  region; the green fill shows the selected segment - "◁ Mark Start" and "Mark End ▷" buttons stamp
  the slider handles at the current video player position (scrub to find the frame, then click) -
  Slider and number inputs stay two-way synced; duration=0 preserved as "to end of file" when the
  right handle is dragged all the way to the end

Image visual section: - Hover any image row to see a full-width thumbnail preview above the list -
  Preview also appears when an image is added from the dropdown

Audio section (file / extract_from_video): - <audio controls> player appears below the file selector
  and updates its src when the selection changes

Backend: - GET /fbtools/media/info — returns duration, fps, width, height, frame_count for any file
  in the input directory (uses cv2) - GET /fbtools/media/stream — streams the file with
  range-request support so the browser can seek without downloading the whole file

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Apply pagination, expanded click area, and active indicator to all list sections
  ([`8c68046`](https://github.com/frost-byte/fbTools/commit/8c68046aec607a7fec4b2acb52c2064fd2524670))

- Composition sidebar: Subjects and Backgrounds lists now paginate (10/page) and highlight currently
  assigned subjects / active background with green underline - Bundle editor: list paginates
  (10/page), full card click opens editor, most-recently-saved bundle underlined on return to list
  view - Cast editor: list paginates (10/page), active indicator on last-saved cast - Filter/search
  changes reset page to 0 in bundle editor - Shared pagination uses existing
  fbt-ce-pg-btn/info/saved-pagination CSS classes - New fbt-be-card-clickable / fbt-be-card-active
  CSS rules for card hover and selection

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Green dot badge on Prompt Composition sidebar tab when LLM is loaded
  ([`24d7161`](https://github.com/frost-byte/fbTools/commit/24d71618819f2456efb57386cbb34be8044262e5))

Uses a CSS ::after pseudo-element on .sidebar-icon-wrapper inside the tab button (targeted via the
  stable data-testid="fbt.composition-editor-tab-button" attribute that ComfyUI derives from our
  registered tab ID). The body class fbt-llm-loaded is toggled by _llmSyncBadge(), called at every
  point _S.llmLoaded changes (initial load, model load success/failure, and unload).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Replace image dropdown with collapsible tree browser
  ([`02197a2`](https://github.com/frost-byte/fbTools/commit/02197a2c1f94e896c1a2d2c1a1194d1fb343eb82))

Adds subdirectory support to the image picker in the bundle visual section: - Backend:
  /fbtools/media/list now accepts ?recursive=true, using os.walk() to return relative paths
  including subdirectories (e.g. portraits/alice.jpg) - Frontend: replaces the flat <select>
  dropdown with a collapsible tree browser showing dirs as lazy-expanded nodes and files as
  clickable leaves - Selected state syncs back to the tree (checkmark + dimmed) when files are added
  or removed from the list - Hover preview now stays visible when moving between the selected-files
  list, the tree browser, and the preview image, by wrapping all three in a single hover container
  rather than attaching handlers to each element independently - Adds _viewUrl() helper that
  correctly splits subfolder/filename for the ComfyUI /view endpoint so subdirectory images preview
  correctly

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

### Refactoring

- **h3**: Extract CompositionToH3Conditioning validation into pure helpers
  ([`2a844c4`](https://github.com/frost-byte/fbTools/commit/2a844c4582d2e1c7e42aaebb5eeb550aa2315ab4))

Move the three §1 invariant checks out of extension.py's execute() method and into testable
  functions in utils/prompt_assembler.py: - validate_h3_refs_pre(references) -> list[str] (pre-load:
  count/pairing/trim_to) - validate_h3_audio_clip(dur, aord, basename) -> str | None (per-clip 2–15
  s) - validate_h3_audio_total(durations) -> str | None (total ≤ 15 s)

extension.py delegates to the imported helpers; behaviour is unchanged. 29 new tests cover every
  rule and boundary value.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **outfit**: Unify outfit modal into single file browser + action flow
  ([`2da5f93`](https://github.com/frost-byte/fbTools/commit/2da5f9358f3a0fd7275711fd978fcec596eb3b92))

Replace the fragmented layout (refs list → LLM section → SAM2 section, each with its own file
  picker) with one coherent structure:

- Single File Browser section (always visible, tree + Input/Output tabs + preview + frame-time row
  for video) at the top - Two action buttons beneath: "+ Add as Reference" (images only, no LLM
  required) and "🔍 Analyze with LLM" (images+video, LLM optional) - LLM query textarea shown only
  when a vision model is loaded - SAM2 section below the browser: no longer has its own srcInput —
  selecting an image from the browser populates the SAM2 click-to-point preview automatically;
  selecting a video disables extraction - Analyze always adds the result file to refs (checkbox
  removed) - Reference Images list moves to bottom as the output of all three paths

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

### Testing

- Fix 4 stale test assertions
  ([`f478051`](https://github.com/frost-byte/fbTools/commit/f4780511927c22faeb3172e17c8e827197773fd5))

- test_s1_maps_to_slot_a / test_two_subjects_remapped_in_order: H3 ref2va uses <Subject N> labels,
  not names; assert appearance summary text ("tall woman", "short man") rather than the name -
  test_dialogue_tags_use_subject_language: <d> wrapping requires use_dialogue_tags=True on the
  composition; set it explicitly - test_style_retention_analysis_line: phrasing is "tonal style",
  not "audio style"; update assertion to match

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.19.1 (2026-08-13)

### Bug Fixes

- **ui**: Suppress boolean widget draw() to prevent toggle leaking through in SceneCastBuild
  ([`9acd42c`](https://github.com/frost-byte/fbTools/commit/9acd42cbe14e88db5d7d21ce6c5c6d9d7a52e098))

ComfyUI V3 toggle widgets have a custom draw() that can bypass the type="hidden" check used by
  setWidgetVisible. Fix by iterating all standard widgets (not by name to avoid any lookup miss) and
  overriding draw() as a belt-and-suspenders guard. Resolves Audio 4 appearing above the slot table.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.19.0 (2026-08-13)

### Features

- **scene**: Add SceneCastBuild node for inline cast configuration
  ([`ef1a227`](https://github.com/frost-byte/fbTools/commit/ef1a2273bb5b90038a2cf7277a3f0ce1484fd646))

New node builds a SCENE_CAST without a saved file. Outputs the same SCENE_CAST type as SceneCastLoad
  — wires into PromptCompositionLoader unchanged.

- Python: SceneCastBuild with 4 slots (subject/bundle/visual_mode/use_audio each), all hidden behind
  a DOM table widget in the frontend. - JS: scene_cast_build.js renders a compact 4-row interactive
  table. Selects are populated from the live subject/bundle registry via API. Changing any field
  syncs back to the hidden widget and marks the canvas dirty. _refreshCastTable() exposed for editor
  integration. - Cast editor: "Send to Workflow" button in the top bar discovers fbt_SceneCastBuild
  nodes on the canvas, shows a picker, and pushes the current editing entries to the selected node.
  "Create new" option adds a fresh node and positions it near the canvas mouse.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.18.0 (2026-08-13)

### Features

- **ui**: Add Send to Workflow button to Composition Editor
  ([`a2e548d`](https://github.com/frost-byte/fbTools/commit/a2e548df8cf25a8ba166601bbd65f9321f5aa461))

Opens a picker listing all PromptCompositionLoader nodes on the canvas by their current
  composition_name widget value. Selecting a node sets its widget to the current composition.
  "Create new" option adds a fresh node and sets its widget via a short timeout after graph.add().

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.17.1 (2026-08-13)

### Bug Fixes

- **node**: Fold composition file mtime into PromptCompositionLoader fingerprint
  ([`e15fa50`](https://github.com/frost-byte/fbTools/commit/e15fa503dc4d9648c1bcf747763e1a3f3e0b3d0e))

The fingerprint previously keyed only off the compositions directory mtime and the reload counter.
  Directory mtime moves when files are added or removed, but NOT when an existing composition file
  is edited in-place (e.g. via a text editor or external tool). Out-of-band JSON edits were
  invisible to the cache until the user manually hit the reload button.

Fix: resolve the matched composition file path (comps_dir/<id>.json) and include its getmtime in the
  fingerprint tuple alongside the directory mtime. The reload counter is still present as an escape
  hatch for cases where mtime is unreliable (network filesystems, etc.).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.17.0 (2026-08-13)

### Features

- **node**: Add CompositionToH3Conditioning terminal node (Steps 5-7)
  ([`ac2b5d5`](https://github.com/frost-byte/fbTools/commit/ac2b5d51659c749400e4d8486a1394b9c76fad86))

Completes the H3 refplan action plan.

Three media-loading helpers added above the node class:

_h3_resolve_path(path) — absolute-or-input-dir path resolution _h3_load_image(path) — PIL →
  [1,H,W,3] float32 tensor _h3_load_video_frames(path, params) — VHS cv_frame_generator adapter →
  [B,H,W,3] float32 tensor _h3_load_audio(path, start, dur) — VHS get_audio (ffmpeg) with torchaudio
  fallback for standalone files

CompositionToH3Conditioning (category: conditioning):

Inputs: h3_refplan (FBTOOLS_H3_REFPLAN), clip, vae, audio_vae, width, height, length, ref_image_size
  (match|max) Outputs: positive (CONDITIONING), LATENT

fingerprint_inputs: md5 hash of bundle JSON + getmtime of every referenced file +
  width/height/length/ref_image_size. Invalidates on any file edit without needing a reload counter.

execute: iterates the bundle's references list in order (images → [soundtrack+video] pairs →
  standalone audio), loads each via the above helpers, assembles
  ref_images/ref_videos/ref_video_audios/ ref_audios dicts with 0-based suffix keys, and delegates
  to MiniMaxH3ReferenceToVideo.execute (lazily imported). Suffix pairing convention: ref_video_N ↔
  ref_video_audio_N (same video_ordinal-1). Standalone audio uses a sequential 0-based counter
  independent of audio_ordinal.

Graceful degradation: missing files are logged and skipped; an empty reference set degrades to
  text-to-video (native node handles it).

Node registered in FBToolsExtension.get_node_list().

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.16.0 (2026-08-13)

### Features

- **assembler**: Retention-aware audio phrasing and fix <Audio N> ordinals
  ([`827cd24`](https://github.com/frost-byte/fbTools/commit/827cd24d5d2eb8b4e01908127d6096736d16cb55))

Step 2 of the H3 refplan action plan.

Bug fixed: _build_ref_map assigned standalone audio ordinals starting at 1 without accounting for
  soundtrack audios (extract_from_visual). When both types were present, the <Audio N> label in the
  assembled prompt would not match what MiniMaxH3ReferenceToVideo assigns at inference time.

Fix: two pre-passes now assign audio ordinals in native ref_items order — soundtracks first (slot
  order), then standalone files (slot order) — exactly mirroring the three-pass algorithm in
  _build_h3_refplan. Ordinal parity is now guaranteed across the prompt assembler and the terminal
  node.

New ref_map fields: audio_retention, audio_role (standalone), soundtrack_num, soundtrack_retention,
  soundtrack_role (for extract_from_visual video entries).

_assemble_h3_ref2va updated in three places:

subject_definitions: - Soundtrack entries now get their own <Audio N> line. - timbre → "…without
  copying the original signal" (matching the h3_prompt libber's %nocopy% convention) - reuse →
  "…reproduced verbatim" - style → "audio style and rhythm reference…" - Non-empty audio_role
  overrides the generic description entirely.

summary: - has_audio now includes soundtrack_num; both audio modalities contribute to the [audio
  reference] task tag and the closing sentence. - Closing sentence uses retention-appropriate phrase
  per audio entry.

retention_analysis: - timbre → "reference - its vocal timbre guides … without copying the original
  signal" (replaces old "reference (voice timbre only, not fully_copy)") - reuse → "fully_copy" -
  style → "reference (audio style, not fully_copy)" - Soundtrack audio entries appear before
  standalone in retention_analysis.

16 new tests in tests/test_h3_audio_phrasing.py cover ordinal correctness (both modality types,
  mixed cases) and retention/role phrasing in all three sections. Updated one existing test whose
  assertion matched old phrasing.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.15.0 (2026-08-13)

### Features

- **cast**: Add FBTOOLS_H3_REFPLAN bundle output to PromptCompositionLoader
  ([`baeddd7`](https://github.com/frost-byte/fbTools/commit/baeddd7c32fcab56b3ccbae631036fc25c7b9742))

Steps 0–4 of the H3 refplan action plan:

- Step 0: Add retention/role fields to bundle audio schema and editor UI (timbre/reuse/style modes;
  free-text role label; shown for both extract_from_visual and file sources in bundle_editor.js)

- Step 1: Implement _build_h3_refplan() in utils/prompt_assembler.py Three-pass algorithm mirrors
  native MiniMaxH3ReferenceToVideo ref_items order: images → [soundtrack_audio + video] pairs →
  standalone audios. Ordinals (picture_ordinal/video_ordinal/audio_ordinal) are assigned so <Picture
  N>/<Video K>/<Audio J> labels in the prompt match what the tokenizer derives from ref_items.
  Parity verified by 9 new tests in tests/test_h3_refplan_parity.py.

- Step 3: Extend _resolve_cast_media to collect video_entries_full — all video-mode cast entry
  descriptors with full audio config (source, path, start_time, duration, retention, role). Existing
  flat outputs unchanged.

- Step 4: Add FBTOOLS_H3_REFPLAN wire type and H3RefplanType class. PromptCompositionLoader gains an
  h3_refplan output; execute() builds the bundle from enriched resolved_subjects +
  video_entries_full and attaches prompt/model_type/ref_image_size before returning.

Also: fix apply_cast_to_subjects so extract_from_visual audio no longer sets
  voice.audio_reference_file (it is a video soundtrack handled by the refplan's soundtrack_audio
  pass, not a standalone voice reference). Verified by updated test_cast_enrichment.py (15 tests,
  all passing).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.14.0 (2026-08-12)

### Bug Fixes

- **assembler**: Align H3 subject_definitions reference phrasing with empirical best practice
  ([`cf05e21`](https://github.com/frost-byte/fbTools/commit/cf05e212ee2a9a64c44002b1884f7b31da90539b))

Matching the manually-crafted prompt format that produces better results:

- Video inline citation: "from <Video N>" instead of "in <Video N>" - Picture inline citation: "from
  the character sheet contained in <Picture N>" (singular) or "from the character sheets contained
  in <Picture N> and <Picture M>" (plural) instead of plain "in <Picture N>" - Add standalone <Video
  N> role lines after subject lines, before audio lines; description is task-flag-aware: video
  continuation → "is the continuation starting point for the target video" video editing → "is the
  source video being edited" default → "is the visual identity reference for <Subject N>" - Hoist
  active_flags computation to top of _assemble_h3_ref2va so it is available in both
  subject_definitions and summary sections - Update 2 existing tests; add 5 new tests

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **assembler**: Align H3 task types with official MiniMax docs
  ([`92ee4e5`](https://github.com/frost-byte/fbTools/commit/92ee4e5859cce3321a379095cf274edf84d465cc))

Per the MiniMax H3 Ref2VA specification, the valid task types are: reference generation, keyframe
  completion, video editing, video continuation, audio reference, audio reuse.

"video reference" is not an official type.

- Auto-detection: pictures AND videos that provide guidance both fall under "reference generation"
  (same bucket per spec); voice timbre files stay as "audio reference" - "video editing" / "video
  continuation" / "keyframe completion" / "audio reuse" cannot be auto-detected and require user
  task_flags - video editing tasks open the summary body with the required sentence: "The target
  video is an edited version of <Video N>." - Composition Editor task flag checkboxes updated to 6
  official types - PromptAssemble tooltip updated with full type list - All affected tests updated;
  5 new tests added

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **assembler**: Rewrite H3 Ref2VA subject_definitions to match official MiniMax format
  ([`2e5071b`](https://github.com/frost-byte/fbTools/commit/2e5071b2204bc90e550223bf6e771640116446a0))

Rewrites the subject_definitions section of _assemble_h3_ref2va to use the official MiniMax H3
  single-line prose format per subject, with picture/video references cited inline and audio
  references as separate bottom entries.

- subject line: "<Subject N> is [summary] in <Pic N> [and <Pic M>], with [details]." - audio line:
  "<Audio N> is the voice-timbre reference for <Subject N> (S1), containing [voice]." - removes old
  multi-line Face:/Hair:/Body:/Outfit: sub-bullets - removes standalone <Video N>: reference lines
  (video now inline in subject line) - fixes auto-detected task tag from "video continuation" to
  "video reference" - adds task_flags user override on PromptAssemble node and assemble_composition
  path - adds task flag checkboxes in Composition Editor Info section (h3_ref2va only) - updates all
  affected tests to match new format; adds 3 new task_flags tests

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **assembler**: Use neutral from <Picture N> phrasing for image references
  ([`6a1b76a`](https://github.com/frost-byte/fbTools/commit/6a1b76a298fc0843e5d5ca88f3d2f225059a9d96))

Drops the "character sheet contained in" qualification since picture references may be any type —
  individual shots, style references, poses, environments, etc. Plain "from <Picture N>" is
  consistent with "from <Video N>" and makes no assumptions about image purpose.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

### Features

- **cast**: Wire Reference Bundle fields into H3 Ref2VA prompt assembly
  ([`2d52031`](https://github.com/frost-byte/fbTools/commit/2d52031f1502818cbda02ab1c43327f71c884fc7))

Enrich resolved subjects from Scene Cast bundle data before prompt assembly: image-mode visual.files
  → character_sheet_images (appended, deduped), use_audio bundles → voice.audio_reference_file, and
  appearance_override → appearance.summary.

- Add apply_cast_to_subjects() to utils/prompt_compositions.py (pure, no ComfyUI deps); deep-copies
  subjects so registries are never mutated - Import and call from PromptCompositionLoader.execute()
  between cast resolution and _assemble_composition() - 14 new tests in
  tests/test_cast_enrichment.py

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Add edit/delete for existing backgrounds in Composition Editor
  ([`b1c93ab`](https://github.com/frost-byte/fbTools/commit/b1c93abb12c97b781163551fffaa27368730d4c7))

Backgrounds in the sidebar now show a pencil (✎) button on hover that opens an inline edit form
  pre-filled with the background's current name, description, lighting, and soundscape fields.

- Refactored _showNewBgForm into _showBgForm(existing) covering both create and edit;
  _showBgForm(null) is the new-background path - Edit form adds a Delete button (danger style,
  confirm dialog) that removes the background and clears the composition's background field if it
  was pointing to the deleted entry - Extracted _refreshBgDropdown() helper that syncs
  _S.backgrounds, rebuilds the sidebar list, and refreshes the editor dropdown in one call - Each
  background row is now wrapped in fbt-ce-sb-item-row flex container; edit button fades in on row
  hover

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.13.0 (2026-08-12)

### Features

- **libber**: Add random wildcard notation for libber key selection
  ([`c096d69`](https://github.com/frost-byte/fbTools/commit/c096d69e09bdd0e6b8e615a05343b3a23825f5ab))

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
  ([`02f0ad1`](https://github.com/frost-byte/fbTools/commit/02f0ad12fb4695fcd35e57f80ff0e3b493374e0b))

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
  ([`72c9fd1`](https://github.com/frost-byte/fbTools/commit/72c9fd1ed859bf67b8bc0dcc1223c16a4a75cba9))

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
  ([`4ae2eef`](https://github.com/frost-byte/fbTools/commit/4ae2eef01be0b75cd664efd4cbe172f3dd1e7bc3))

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
  ([`4120a11`](https://github.com/frost-byte/fbTools/commit/4120a1131c7e6afce2a338e89332a93c58cecc0f))

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
  ([`3c7b036`](https://github.com/frost-byte/fbTools/commit/3c7b036f57e36e213f9df91575d5e80b90263a5d))

All utils imports in the Prompt Composition and LLM route blocks were using bare absolute form (from
  utils.x import) which fails when the package is loaded by ComfyUI as a relative package. Changed
  to the same dot-prefix relative form used everywhere else in extension.py.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **llm**: Replace cross-module get_logger with stdlib logging in llm_client
  ([`15d9dd1`](https://github.com/frost-byte/fbTools/commit/15d9dd10d57d077fa98f8e9194e6b5751dca2c75))

Pure utils modules have no cross-module deps. Using get_logger from logging_utils caused a
  ModuleNotFoundError at ComfyUI load time because utils/ has no __init__.py and the import path was
  absolute. Replace with logging.getLogger(__name__) consistent with other standalone utils modules.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **llm**: Scanner skips root dir to avoid misidentifying stray GGUF files
  ([`e45d83b`](https://github.com/frost-byte/fbTools/commit/e45d83b903e759c62bdf96c53c3e58c34d665bdc))

_scan_directory now iterates root's children rather than treating root itself as a candidate model
  dir. Fixes the case where a loose text-encoder .gguf (e.g. umt5-xxl-encoder-Q8_0.gguf) in the LLM
  root causes the entire directory to be returned as a single model and recursion to stop, hiding
  all nested model subdirectories.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **llm**: Use "LLM" (singular) as the canonical folder_paths key
  ([`d1440b5`](https://github.com/frost-byte/fbTools/commit/d1440b5b7a37ebfb715095bdf54d2a2b8ebf603c))

The ComfyUI convention, established by ComfyUI-MiniMaxH3-Prompt-Writer and comfyui_llm_party, is
  "LLM" not "LLMs". Scanner now checks "LLM" first with "LLMs" as fallback, and defaults to
  models/LLM/ when neither is registered.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **scene**: Fix two bugs in assemble_composition adapter + add tests
  ([`818f8c0`](https://github.com/frost-byte/fbTools/commit/818f8c0d5c1e7d7053fe3b1089e192e2e3e85f11))

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
  ([`c96cc82`](https://github.com/frost-byte/fbTools/commit/c96cc825505d2defecfabb5fe3242c254cf0a2b5))

_S.composition was null when _buildPanel called _rebuildShots during first render, causing a
  TypeError on .shots. Moving _newComp() before _buildPanel ensures state is ready before any DOM
  callbacks execute.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

### Chores

- **nodes**: Unregister MultiLoraLoader, SceneWanVideoLoraMultiSave, LoraStackView
  ([`0a71693`](https://github.com/frost-byte/fbTools/commit/0a716930283be2270ccf75e8f0564f95f6b758ed))

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
  ([`f2763b3`](https://github.com/frost-byte/fbTools/commit/f2763b3ef1c06e80e91721fec8d8bab83899d957))

All 29 node display_name values that used verbatim CamelCase class names are updated to Title Case
  with spaces. FBTextEncodeQwenImageEditPlus is shortened to "FB Qwen Image Edit Plus" to avoid
  collision with similarly named nodes from other packages. Node IDs are unchanged so existing
  workflows are unaffected.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Unify LoraStackBuilder info icon with ConceptDefine style
  ([`e0e7cd9`](https://github.com/frost-byte/fbTools/commit/e0e7cd93bc226be1014664e331cb6be1a41d8994))

Remove the explicit circle (arc + stroke) from _lsbDrawIcon and replace with the same approach as
  _cdDrawIcon: bold "i" centered directly in the rounded rect, font size proportional to icon size
  (sz * 0.55). Both icons are now 18px rounded rects with the same visual weight.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

### Documentation

- Add user-facing docs for Scene Composition Engine nodes
  ([`59d1f2d`](https://github.com/frost-byte/fbTools/commit/59d1f2d88f09c26b5b6c17482f57eaefe57e31ef))

Four new end-user reference docs covering all Phase 1–4 nodes: concept_registry.md,
  subject_profiles.md, scene_composition.md, prompt_assembly.md. Each covers inputs/outputs, typical
  workflow diagrams, and storage locations.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **nodes**: Add missing tooltip strings to SubjectProfileDefine, ConceptDefine, DatasetCaptioner,
  TailEnhancePro
  ([`6e48cdc`](https://github.com/frost-byte/fbTools/commit/6e48cdcaf2cb33c82e122a2b47cabbfd866dc4ba))

SubjectProfileDefine: name, face, hair, body, default_outfit

ConceptDefine: description

DatasetCaptioner: device

TailEnhancePro: all 12 processing parameter inputs (tail_count, ref_window, deflicker, color match,
  unsharp, bilateral filter)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

### Features

- **cast**: Add Reference Bundle and Scene Cast data layer
  ([`0f680de`](https://github.com/frost-byte/fbTools/commit/0f680decf87a5ca766d2bf18344edd0b6bbe2e4f))

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
  ([`aa842a2`](https://github.com/frost-byte/fbTools/commit/aa842a27c2876f567978eedd21325dcab73edf92))

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
  ([`51c6483`](https://github.com/frost-byte/fbTools/commit/51c6483b7302cb93ffa0b35d47189831fed3d444))

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
  ([`b5b82cd`](https://github.com/frost-byte/fbTools/commit/b5b82cdd31376f75a5bde7448919d5f046ffc40a))

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
  ([`f107033`](https://github.com/frost-byte/fbTools/commit/f1070333b717862603d5342446560a49aa1c2798))

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
  ([`a8c95f5`](https://github.com/frost-byte/fbTools/commit/a8c95f5fc9fbc447d0c35d088c0339b5676a8cb9))

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
  ([`e8b90a9`](https://github.com/frost-byte/fbTools/commit/e8b90a96ba8822d3af136204832005bcb5b830f6))

Adds _concept_get_ids() helper that reads concept_registry.json at schema load time.
  SubjectProfileDefine.concept_id is now a combo picker instead of a free-text field; "None" is
  normalised to "" in execute(). Define nodes (ConceptDefine, SubjectProfileDefine) keep free-text
  subject_id / concept_id inputs since those are used to create new entries.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **scene**: Add PromptCompositionLoader node with reload counter
  ([`84931cd`](https://github.com/frost-byte/fbTools/commit/84931cd3bdb01a9b7af95a48672a6dbb6cc3e26b))

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
  ([`f72bfe1`](https://github.com/frost-byte/fbTools/commit/f72bfe1d1fdc15bf1a2d85d9c068512868f6f7fa))

- Add ↑/↓ reorder buttons and ⧉ duplicate to each shot card header - Track focused shot (focusin
  delegation) so camera/sound presets insert into the correct shot's field rather than copying to
  clipboard - _moveShot / _duplicateShot / _addNewShot helpers keep focus index in sync and scroll
  the target card into view after rebuild - Ctrl+Shift+N: add shot, Ctrl+Shift+P: preview,
  Ctrl+Shift+C: copy - Update sidebar section titles to "click to apply to shot" -
  .fbt-ce-shot-active highlight (blue border) on the focused shot card

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Prompt Composition Editor — Phase 1 + Phase 2
  ([`c90509a`](https://github.com/frost-byte/fbTools/commit/c90509a3e71e3d9282edd6c9998e875a65c62b76))

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
  ([`72ad9e4`](https://github.com/frost-byte/fbTools/commit/72ad9e49d1a794b87590da46be1503909a2c3de8))

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
  ([`f97ec73`](https://github.com/frost-byte/fbTools/commit/f97ec735b52ebd58fbd04600267edb51d64c5a0b))

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
  ([`63282d3`](https://github.com/frost-byte/fbTools/commit/63282d3c308c57eb167aab13f397c86854932809))

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
  ([`e7a753e`](https://github.com/frost-byte/fbTools/commit/e7a753e34fc96bea5223c0015b43b252293eaa2b))

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
  ([`1b0837d`](https://github.com/frost-byte/fbTools/commit/1b0837d1794c83661bab1106a5658cbeecbbe24c))

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
  ([`25a83fa`](https://github.com/frost-byte/fbTools/commit/25a83fac25d972eaeec976ba2435cd0b8a388c54))

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
  ([`1639f0b`](https://github.com/frost-byte/fbTools/commit/1639f0b4420ece123d7dd687ce204106d5c6920a))

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
  ([`5564ed3`](https://github.com/frost-byte/fbTools/commit/5564ed342552723703c4451398a0fbaf46962b5d))

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
  ([`a6c9351`](https://github.com/frost-byte/fbTools/commit/a6c9351d66d537d3909dbf83dbb66cd2cdf2771a))

Handles 1-D (samples,) and 2-D (channels, samples) tensors by unsqueezing to the expected (batch,
  channels, samples) layout. Placed under the new 🧊 frost-byte/Audio category.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.2.0 (2026-08-05)

### Features

- **lora**: Add Concept Registry system with ConceptRegistryLoad, ConceptDefine, ConceptResolve,
  ConceptList nodes
  ([`322d84a`](https://github.com/frost-byte/fbTools/commit/322d84a6fe60871ccf19886aabf1b05a3406a2ab))

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
  ([`38904f9`](https://github.com/frost-byte/fbTools/commit/38904f91fa1d44f98b1a23fce5d328aafd1abed1))

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **lora**: Add native LORA_STACK support to LoraPresetDefine/Select and add MiniMaxH3 target
  ([`0d457f4`](https://github.com/frost-byte/fbTools/commit/0d457f4e684f4f7ae690da27e18e8e6018b8b46b))

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
  ([`c5ddfe6`](https://github.com/frost-byte/fbTools/commit/c5ddfe6e14ce6a569b34e8845beb9a33890333f4))

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
  ([`343a3c2`](https://github.com/frost-byte/fbTools/commit/343a3c26cc2d1e3e4072101fa41ed4388bab924e))

When model_target is not LTX2.3, the video/audio/cross-attention strength inputs and toggle button
  are hidden entirely. When LTX2.3 is selected, a ▶/▼ caret button between Enabled and the Civitai
  button controls visibility. Accordion defaults to collapsed; onConfigure re-applies visibility
  when a saved graph is loaded.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **lora**: Add dynamic combo and preview to WanPresetSelect
  ([`0d84353`](https://github.com/frost-byte/fbTools/commit/0d84353674a60183162760e26be25f922049a586))

- Replace index INT input with a COMBO widget (selected_preset) that starts with ["none"] and is
  populated with preset names after each execution - Add validate_inputs to accept any string value,
  bypassing static combo option validation so user-selected names are not rejected by the server -
  Add is_output_node=True for standalone preview execution - Execute sends preset names to frontend
  via ui={preset_names:[...]}; JS onExecuted updates widget.options.values and preserves current
  selection - Switch preset lookup from index-based to name-based with first-entry fallback

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

- **lora**: Add LoraPresetDefine and LoraPresetSelect nodes
  ([`71d2198`](https://github.com/frost-byte/fbTools/commit/71d21987126d8892df78c928aa7fa447daff73ea))

Single-stack preset nodes for models without a dual-sampler stage (e.g. Flux2/Klein, Qwen). Uses a
  separate LORA_PRESET_LIST custom type to prevent cross-wiring with Wan preset chains.
  LoraPresetSelect uses the same dynamic combo + validate_inputs pattern as WanPresetSelect.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.0.0 (2026-07-24)

### Bug Fixes

- Add control flags to StorySceneBatch descriptor and fix StoryEdit scene data persistence
  ([`9752e6e`](https://github.com/frost-byte/fbTools/commit/9752e6e925918dac557a338a4290288d7921c86f))

- Add use_depth, use_mask, use_pose, use_canny flags to batch descriptor - Add logging to
  StorySceneBatch for scene configuration debugging - Add logging to StoryScenePick for pose_type
  resolution tracking - Fix StoryEdit duplicate renderTable call that was overwriting normalized
  scene data - Add fingerprint_inputs to StorySceneBatch for proper cache invalidation

This fixes issues where: 1. Control flags weren't being passed from story config to batch processing
  2. StoryEdit UI changes (pose_type, depth_type) weren't persisting to story.json 3. Cache wasn't
  invalidating when story.json was modified externally

- Complete mask system migration from mask_type to mask_name
  ([`e511448`](https://github.com/frost-byte/fbTools/commit/e51144816eb30492ec622688043f497ecd9485d4))

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
  ([`439feff`](https://github.com/frost-byte/fbTools/commit/439feff79faf4bbb518410f29b9e8aa71e5be37a))

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

- Add developer utility scripts
  ([`f780e74`](https://github.com/frost-byte/fbTools/commit/f780e74c1fed05131d83f64ecc64334143503b15))

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

### Documentation

- Add comprehensive test coverage summary
  ([`dce7115`](https://github.com/frost-byte/fbTools/commit/dce7115fafca3843c519f56d6eb3e8e966de7d69))

- 90 tests passing (100% pass rate) - Coverage breakdown by test file and category - Real-world
  workflow validation - Backend testing complete - Ready for UI implementation

- Add NLF pose implementation guide and additional tests
  ([`e545a83`](https://github.com/frost-byte/fbTools/commit/e545a8382432f2e60a988f40353f5a674359c1ba))

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
  ([`0cf96b0`](https://github.com/frost-byte/fbTools/commit/0cf96b041ac7ab9849f7d46be4fef37a5c2f3c0d))

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
  ([`fe27a6d`](https://github.com/frost-byte/fbTools/commit/fe27a6dd67bd8f9b0cfb683d6ac92d5247794c82))

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
  ([`364f1fe`](https://github.com/frost-byte/fbTools/commit/364f1fe9497d0314cdef228e652134d293d325cb))

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
  ([`2d2fd56`](https://github.com/frost-byte/fbTools/commit/2d2fd56e9791cc788deff8e6680cb1b288573b7a))

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
  ([`515f75f`](https://github.com/frost-byte/fbTools/commit/515f75fd25a9293ee08c5dc4c8515c0228cf702b))

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
  ([`9032be4`](https://github.com/frost-byte/fbTools/commit/9032be43fccb6c37f2cc61a47143e703ac2c78c4))

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
  ([`37a060f`](https://github.com/frost-byte/fbTools/commit/37a060f8419d70abaab893d097d888e6e3b6c62c))

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
  ([`9cc90ef`](https://github.com/frost-byte/fbTools/commit/9cc90efb0148e6a9b15b092511462a836147b036))

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
  ([`457de42`](https://github.com/frost-byte/fbTools/commit/457de42048bf12da7f03a30319a46dd0a7865400))

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

- **lora**: Add LORA_STACK output to LoraStackCollect and update WanPreset nodes
  ([`a4ae9e5`](https://github.com/frost-byte/fbTools/commit/a4ae9e53d59275421572f9a592e46f5494ea12df))

- LoraStackCollect: add easy-use compatible LORA_STACK output (list of (lora_name, model_strength,
  clip_strength) tuples) for interop with EasyLoraStack, PowerLoraLoader, and other LORA_STACK
  consumers - WanPresetDefine: replace single-lora Combo inputs with optional LORA_STACK inputs for
  lora_h and lora_l, enabling multi-lora stacks per preset slot - WanPresetSelect: change
  lora_h/lora_l outputs from STRING to LORA_STACK for direct connection to downstream loader nodes

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

- **lora**: Add WanPresetDefine and WanPresetSelect nodes
  ([`deb4b1e`](https://github.com/frost-byte/fbTools/commit/deb4b1e61849a5439857f09c776fbe070ea02ee4))

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

- **lora**: Register WanPresetDefine and WanPresetSelect in extension
  ([`b3f717a`](https://github.com/frost-byte/fbTools/commit/b3f717aaf165be9d40ec716072242e4b2282803e))

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
