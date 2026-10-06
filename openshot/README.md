# OpenShot templates for fbTools

ComfyUI templates that OpenShot's **Generate** dialog can load and run. They live here, not in
the OpenShot repository, because each one depends on fbTools nodes (Scene Cast, Source
Profiles, Reference Bundles, the LoRA stack, the prompt shorthand expander). Nothing here is
needed by fbTools itself; these are optional integration files.

## The templates

| File (`openshot/templates/`) | Shows in OpenShot as | You provide in the dialog | What it does |
|---|---|---|---|
| `video-scene-cast-generate.api.json` | Generate From Scene Cast (fbTools) | A Composition, optional per-slot Bundle/Subject overrides, optional background override | Builds an H3 reference-to-video generation from a Composition (Scene Cast Build + Composition Load). Takes no input clip; with a clip selected it pre-fills the cast from that clip's embedded metadata when it was generated through fbTools. |
| `video-scene-cast-generate-source-profile.api.json` | Generate From Scene Cast - Source Profile (fbTools) | A Source Profile, one of its clip segments, optional cast entries, optional background override | Same idea, but driven by a Source Profile clip segment (Source Profile Load + Source Profile Clip Prompt). |
| `video-bridge-two-clips-bundle-voice.api.json` | Bridge to Next Clip (Bundle Voice)... | Clip B, a subject description, dialogue, frame counts, and an **optional** Reference Bundle | A two-clip bridge like OpenShot's plain bridge template, plus an optional bundle voice: when a bundle is picked and has audio, that voice is the audio reference for both clips; otherwise each clip's own audio is used. It also logs which references H3 is given (tag numbering, frames used, durations) to the ComfyUI log on every run. |

The two Scene Cast templates use the `scene_cast` input group, which makes OpenShot show an
**Edit Cast...** button (the Scene Cast builder) in place of plain text boxes. The bundle-voice
template uses OpenShot's `bundle` input type for its picker.

Which rule decides whether a bundle's audio is included in Scene Cast runs: the cast entry's
audio checkbox must be on and, for Source Profile clips, the clip segment must allow dialogue.
See `docs/scene_cast_build.md`.

## Requirements

**On the machine running ComfyUI (the one OpenShot is configured to use)**

- fbTools, current `main` (the bundle-voice template needs `BundleAudioReferenceLoad` and
  `H3ReferenceSummary`; the Source Profile template needs `background_override_id`). Restart
  ComfyUI after updating fbTools: a template that references a node the running server doesn't
  have yet fails validation (for example "list index out of range" on a missing output).
- The ComfyUI custom nodes the templates use besides fbTools: VideoHelperSuite, Impact Pack,
  KJNodes, mtb nodes, pysssss custom scripts, and the "Basic data handling" nodes (needed by the
  bundle-voice template), plus the MiniMax H3 / LTXV nodes in your ComfyUI build.
- Models. All three templates reference the same H3 set, by these file names:
  - `10Eros_Max_h3_TURBO-hybrid_beta5_int8.safetensors` (diffusion model)
  - `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` (text encoder)
  - `minimax_h3_video_vae_int8_convrot.safetensors`, `minimax_h3_audio_vae_fp32.safetensors`
  - `minimax_h3_latent_upscaler_3d_bf16.safetensors` (latent upscaler)
  - `taeh3.safetensors` (preview decoder)
  
  These names come from one specific install. If yours differ, change them in the loader nodes
  (open the template in ComfyUI, adjust, re-export as API format) or edit the JSON directly.
  The LoRA stack in each template is present but every LoRA is disabled.
- Content to point at: Compositions, Source Profiles and Reference Bundles are created in
  fbTools' sidebar panel; the templates only reference them by id.

**On the OpenShot side**

- A build with the generalized template inputs (extra inputs of type video/image/audio),
  **plus** the `bundle` input type and the `scene_cast` group with its Cast builder. The last
  two are in the author's fork (`frost-byte/openshot-qt`, branch `feature/scene-cast-generation`),
  not in upstream OpenShot. These templates have not been tried on upstream OpenShot.
- OpenShot must be able to reach the ComfyUI server that has fbTools installed; the bundle and
  Scene Cast pickers are filled from fbTools' `/fbtools/...` routes on that server.

## Installing in OpenShot

OpenShot loads templates from two places: the `src/comfyui` folder of the OpenShot install, and
your **user template folder**. Use the user folder, so nothing is added to an OpenShot checkout.

1. Create the folder if it does not exist:
   - Linux / macOS / WSL: `~/.openshot_qt/comfyui/`
   - Windows: `<your home folder>\.openshot_qt\comfyui\`
2. Copy the templates you want from `openshot/templates/` into it. To keep them in sync with
   this repository instead of copying, symlink them (Linux / macOS / WSL):
   `ln -s /path/to/comfyui-fbTools/openshot/templates/*.api.json ~/.openshot_qt/comfyui/`
3. Open OpenShot. It rescans template folders when their files change, so the entries should
   appear the next time you open the AI menu or the Generate dialog. Restart OpenShot if they
   do not.
4. Templates from the user folder are listed with a **(User)** prefix.

How they appear:

- The Scene Cast templates show under **Create with AI** when no Project File is selected and
  under **Enhance with AI** when one is.
- The bundle-voice bridge is an **Enhance** template. In the *Bridge Clips With AI* dialog,
  choose it from the template list; both the plain and the bundle-voice bridge templates qualify.

**Do not install the same template in both places.** If a template id is found twice, OpenShot
keeps both and renames the second to `<id>__2`, which shows up as a confusing duplicate menu
entry. The bundle-voice file has its own id, so it can sit alongside OpenShot's plain
`video-bridge-two-clips` template.

## Updating

Re-copy the files (or rely on the symlinks). OpenShot picks up changes by file modification time.
If you re-export a template from ComfyUI, keep its `extra_inputs` block and its
`__openshot_input:<key>__` placeholders; those are what connect OpenShot's dialog to the graph.

## Related documentation

- `docs/openshot-bridge-prompt-shorthand.md` -- the `S1`/`V1`/`A1` prompt shorthand used by the bridge templates.
- `docs/scene_cast_build.md` -- the Scene Cast Build node, including audio and background rules.
- `docs/openshot-multi-input-templates-plan.md` -- background on multi-input templates in OpenShot.
