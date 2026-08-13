# Action Plan — Collapse the PromptComposition → H3 ref2va wiring burden

**Repo:** `frost-byte/fbTools` · `extension.py`
**Related upstream:** `Comfy-Org/ComfyUI` · `comfy_extras/nodes_minimax_h3.py`
**Node API:** ComfyUI V3 (`comfy_api.latest.io` — `io.ComfyNode`, `define_schema`, `execute`, `fingerprint_inputs`)

This brief fixes the framework. Where a step says **[EXPLORE]** or **[DECIDE]**, use judgment and inspect the surrounding code before committing — those are genuine decision points, not oversights. Where it says **[INVARIANT]**, do not deviate without flagging it back.

---

## 1. Objective

Today `PromptCompositionLoader` (extension.py ~L13956) resolves a composition into a model-specific prompt plus ~19 flat scalar/path outputs. The user must then hand-wire `VHS_LoadVideo` / audio-loader nodes from those outputs, configure them, and route the loaded media into the native `MiniMaxH3ReferenceToVideo` node — reconciling modality and ordering by hand on every graph.

Move that burden into code by:

1. Adding a **structured, ordered reference bundle** output to the loader (no information loss at the output surface).
2. Adding a **terminal node** that consumes the bundle + models, performs the media loading internally, delegates reference/conditioning assembly to the native H3 node, and returns `(conditioning, latent)` ready for the sampler.

Target common-case graph: `PromptCompositionLoader → CompositionToH3Conditioning → (sampler)`.

**Non-goal:** rewriting the loader's config-resolution logic. It is solid. Only its *output surface* and the *downstream loading/routing* change.

---

## 2. Context the implementer needs

### 2.1 The loader as it stands (keep this working)
- `execute` (L14076+) resolves subjects, background, cast media, builds `video_entries` (drives `<Video N>` labels), assembles the prompt via `_assemble_composition`, applies libbers, merges `concept_ids`, builds `lora_stack_data`.
- `fingerprint_inputs` (L14056) is the V3 change-detection hook and is already well-formed (comps dir mtime + reload counter + cast id/modified + bundle mtime + settings mtime).
- Outputs are flat: `prompt`, `concept_ids`, `model_type_used`, `composition_name`, `reference_video` (**first** video only), `reference_images` (**one concatenated batch**), the `video_*` load params, `audio_source`/`audio_file`/`audio_*` params, and `lora_stack_data`.

### 2.2 The native node we delegate to
`MiniMaxH3ReferenceToVideo.execute(clip, vae, audio_vae, prompt, width, height, length, ref_image_size="match", ref_images=None, ref_videos=None, ref_video_audios=None, ref_audios=None) -> io.NodeOutput(cond, latent)`
- Reference args arrive as **plain dicts** keyed by prefixed names (`ref_image_0`, `ref_video_0`, `ref_video_audio_0`, `ref_audio_0`, …). The Autogrow machinery is frontend-only and irrelevant when calling `execute` directly.
- The node does the **VAE encoding internally** (`vae.encode`, `audio_vae.encode`, with audio resampled to the VAE rate). So we pass **decoded-but-not-encoded** media: images/frames as `[B,H,W,C]`, audio as `{"waveform","sample_rate"}`.
- Per-image slots: it takes `img[:1]` from **each** slot. A batch in one slot uses only the first image.
- Video↔soundtrack pairing is by **trailing numeric suffix**: `ref_video_N` ↔ `ref_video_audio_N`.
- Presentation/label order (this sets the `<Picture i>/<Video k>/<Audio j>` numbering): **all images**, then per video its soundtrack's `<Audio j>` label emitted **immediately before** its `<Video k>`, then **standalone audio**. Audio ordinals count video-soundtracks and standalone audio together in appearance order.
- H3 uses a **basic guider (no CFG)** → there is no negative conditioning; output is `(positive, latent)` only.

---

## 3. Three concrete problems this refactor must fix

These are visible in the current loader outputs and are the reason for the design; treat them as acceptance criteria.

1. **Multi-video flatten-to-first.** `video_entries` is a list and produces N `<Video k>` labels, but `reference_video` surfaces only the first path. A 2-video composition names `<Video 2>` in the prompt with no media behind it. → The bundle must carry **all** videos, in `video_entries` order, each with its own load params and paired audio. **[EXPLORE]** whether `_resolve_cast_media` / `_resolve_cast` already retains all video-mode entries internally (it likely does, via `video_entries`) or must be extended to expose per-entry load params, not just the first path.
2. **Image batch vs discrete slots.** `reference_images` concatenates all image entries into one batch; the native node needs one image per slot. → The terminal node must **split** the batch into `{ref_image_0: t[0:1], ref_image_1: t[1:2], …}`.
3. **`audio_source` branch.** `{extract_from_visual | file | none}` is a runtime branch currently expected to be resolved in the graph. → Internalize it as an if/elif in the terminal node.

---

## 4. Deliverables

### 4.1 A serializable bundle type — `FBTOOLS_H3_REFPLAN`
An **ordered** structure (dataclass or plain dict — must be JSON-serializable, stable identity) that is the **single source of truth for reference ordering**. Suggested shape:

```
{
  "prompt": str,                 # optional convenience copy; prompt still also flows as its own output
  "model_type": str,
  "width": int, "height": int, "length": int,   # see §6 for provenance
  "ref_image_size": "match"|"max",
  "references": [                # in the exact order _assemble_composition numbered the labels
    {"modality": "image",  "ordinal": 1, "subject_id": "...", "path": "...", "load": {...}},
    {"modality": "video",  "ordinal": 1, "subject_id": "...", "path": "...",
       "load": {"force_rate":0,"frame_cap":16,"skip_first":0,"every_nth":1},
       "audio": {"source":"extract_from_visual"|"file"|"none", "path":"", "start":0.0, "duration":0.0,
                 "load": {...}}},
    {"modality": "audio",  "ordinal": 1, "path": "...", "load": {"start":0.0,"duration":0.0}}
  ]
}
```

**[INVARIANT]** Carry **descriptors (paths + params), not decoded tensors**. Rationale in §5. **[DECIDE]** whether the already-eager `reference_images` batch stays eager (small, convenient) or is demoted to paths for consistency — lean toward paths, but images are cheap enough that keeping them eager is an acceptable shortcut if it simplifies the loader.

### 4.2 Loader change (additive, non-breaking)
- Add one `FBTOOLS_H3_REFPLAN` output. Build it from data `execute` **already computes** (`video_entries`, `cast_media`, enriched `resolved_subjects`, bundle registry). Do not recompute.
- **[INVARIANT]** Keep all existing flat outputs unchanged — they are the manual-wiring escape hatch and existing graphs depend on them.
- The bundle's `references` order **must** equal the order `_assemble_composition` used to number `<Picture i>/<Video k>/<Audio j>` (see §7).

### 4.3 Terminal node — `CompositionToH3Conditioning`
Inputs: `refplan` (bundle), `clip`, `vae`, `audio_vae`, and `width`/`height`/`length` (see §6 for whether these come from the bundle or node inputs). Optional `ref_image_size` override.
Behavior in `execute`:
1. Split images into discrete slot dict entries.
2. For each video: load frames with its params (§5); resolve its `audio_source` branch; if a soundtrack exists, add a matching-suffix `ref_video_audio_N` entry.
3. Load standalone audio entries.
4. Assemble the four dicts (`ref_images`, `ref_videos`, `ref_video_audios`, `ref_audios`) with suffixes matching bundle order.
5. Lazily import and call `MiniMaxH3ReferenceToVideo.execute(...)`.
6. Return `(conditioning, latent)`.

**[INVARIANT]** Import the native class **lazily inside `execute`** (`from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo`), never at module top — `comfy_extras` loads dynamically and top-level import risks load-order breakage. Guard with a clear error if the class is missing or its signature has drifted (the V3 API is still moving).

---

## 5. **How this relates to loading the composition's media** (the crux)

Right now the loader emits *paths + load params* whose purpose is to configure downstream `VHS_LoadVideo` / audio-load nodes; the actual decode happens in those hand-wired nodes. **This refactor moves that decode into the terminal node.** Concretely:

- **Loader stays cheap / lazy:** it resolves *descriptors* only (paths + `force_rate`/`frame_cap`/`skip_first`/`every_nth`/`start`/`duration`/`audio_source`). It does **not** decode video or audio. (`reference_images` is the one current exception — see §4.1.)
- **Terminal node decodes late:** immediately before delegating, it turns each video path + params into a frames tensor and each audio descriptor into a waveform. This is exactly the work the user currently wires by hand.
- **Native node encodes:** the terminal node hands **decoded-but-not-VAE-encoded** media to `MiniMaxH3ReferenceToVideo.execute`, which does the VAE/audio-VAE encoding itself.

So the pipeline of responsibility is: **loader = resolve descriptors → terminal = decode media → native execute = VAE-encode + build conditioning.**

**[DECIDE] Video decode: reuse vs reimplement.** The `video_*` params map 1:1 onto `VHS_LoadVideo` semantics (`force_rate`, `frame_load_cap`, `skip_first_frames`, `select_every_nth`).
- *Reuse:* call VideoHelperSuite's load function directly. Pro: battle-tested, exact semantics. Con: hard dependency on VHS internals that can change. **[EXPLORE]** whether VHS is already a declared/installed dependency of fbTools; if so, reuse behind a thin adapter.
- *Reimplement:* decode via `av`/`opencv`/ComfyUI's own video helpers and apply the four params yourself. Pro: no external coupling. Con: you must reproduce `force_rate`/`frame_cap`/`skip`/`nth` behavior precisely (off-by-one and rate-resampling are the usual traps).
- Recommendation: put decoding behind a single internal helper (`_load_video_frames(path, load_params) -> frames[B,H,W,C]`) so the reuse/reimplement choice is swappable and testable in isolation.

**Audio branch inside the terminal node:**
- `extract_from_visual` → pull the audio track from the **same** video file (VHS can emit audio, or decode separately) using the `audio_*` params; attach as that video's `ref_video_audio_N`.
- `file` → load `audio_file` with `start`/`duration` → standalone `ref_audio_N` (or, if it belongs to a video, its `ref_video_audio_N`). **[EXPLORE]** confirm from the composition schema whether `file` audio is ever meant to be a video soundtrack vs always standalone.
- `none` → emit nothing for that entry.

---

## 6. Width / height / length provenance **[DECIDE]**
The native `execute` builds the empty AV latent from `width/height/length` (snapping length to the 17k+5 grid internally — you do **not** replicate that). Decide where the terminal node gets them:
- from the composition/bundle (most self-contained), or
- as node inputs on `CompositionToH3Conditioning` (most flexible), or
- both, with node input overriding bundle default.
Whichever you choose, thread it through the bundle and/or node schema and include it in the fingerprint (§8). Reference videos are also frame-count-clamped/`n%17==5`-snapped by the native node — pass raw frames, let it snap.

---

## 7. **[INVARIANT] Label-order parity — write a test first**
The whole scheme rests on the prompt's `<Picture i>/<Video k>/<Audio j>` numbering matching the order the native node presents references (§2.2). Because we own both `_assemble_composition` (label emission) and the terminal node (dict order), derive **both** from the bundle's single `references` ordering so they cannot drift.

Add a test that, for representative compositions (images only; 1 video + soundtrack; 2 videos; video + standalone audio; mixed), asserts:
- the ordinal of each `<Picture i>/<Video k>/<Audio j>` tag in the assembled prompt equals the ordinal the native `ref_items` construction would assign for the same reference set, **including** the "soundtrack `<Audio j>` label immediately before its `<Video k>`" rule and the shared image/standalone audio counting.

If `_assemble_composition`'s current numbering does **not** already match the native scheme, aligning it is part of this work and is higher priority than the node itself — a perfectly loaded reference set behind a mislabeled tag is a silent quality regression.

---

## 8. Terminal node `fingerprint_inputs` **[INVARIANT]**
Opaque bundles don't hash reliably for change detection. Implement `fingerprint_inputs` on the terminal node to return a stable tuple of:
- a hash of the bundle's serializable content, **plus**
- `os.path.getmtime` (or content hash) of **every referenced media file**, so editing a referenced video/audio invalidates, **plus**
- `width/height/length/ref_image_size`.

Mirror the loader's existing approach. Also **[EXPLORE]** the loader's own fingerprint edge: it keys off `comps_dir` mtime + reload counter; a directory mtime does **not** move on in-place edits to an existing composition file, so out-of-band JSON edits rely solely on the reload counter. If out-of-band editing is a real workflow, fold the matched composition file's own mtime into the loader fingerprint.

---

## 9. Wrinkles & concerns checklist
- [ ] **Multi-video** fully carried (not first-only); `_resolve_cast_media` may need to expose all video entries' params.
- [ ] **Image batch → discrete slots** split; each slot `t[i:i+1]`.
- [ ] **Suffix pairing** `ref_video_N` ↔ `ref_video_audio_N` exact.
- [ ] **`ref_image_size`** ("match" vs "max") — decide home (composition field vs node input); default "match". Note "max" (2048 short edge) is materially slower because ref tokens ride every step.
- [ ] **Decode-late / cache size** — never let the bundle carry decoded video tensors (up to ~15s each); keep it descriptors.
- [ ] **VRAM/RAM peak** — up to 3 videos + 9 images + 3 audio, each decoded then VAE-encoded. The native node encodes per-ref in a loop; ensure the terminal node doesn't hold all decoded tensors longer than needed (decode→hand off→let refs drop). Watch peak on 24 GB.
- [ ] **Empty reference set** — a composition with no refs should degrade gracefully (native node handles empty dicts; result approximates t2va).
- [ ] **Missing / bad file paths** — validate and emit a clear node error rather than a deep stack trace.
- [ ] **Model inputs enter at the terminal node** — `clip`/`vae`/`audio_vae` are not in the loader; the terminal node is where they join. Good separation; keep it.
- [ ] **Independent branches untouched** — `lora_stack_data` → LoraStackApply and `concept_ids` → ConceptResolve stay separate; the terminal node does not touch them.
- [ ] **Native signature drift** — the V3 `execute` kwargs may change; the lazy-import guard should fail loudly with a version hint.
- [ ] **Backward compatibility** — existing graphs using the 19 flat outputs must keep working; the bundle + terminal node are additive.

---

## 10. Suggested sequence
1. Confirm/align **label-order parity** in `_assemble_composition` and land the parity test (§7).
2. Define the `FBTOOLS_H3_REFPLAN` type and populate it in the loader from existing resolved data (§4.1–4.2).
3. Build the `_load_video_frames` / audio-load helpers behind a stable internal interface; settle the VHS reuse-vs-reimplement question (§5).
4. Build `CompositionToH3Conditioning`: dict assembly + lazy delegation to the native `execute` (§4.3).
5. Implement its `fingerprint_inputs` with referenced-file mtimes (§8).
6. Wire the common-case example graph; verify multi-video, mixed-modality, and empty-ref cases end-to-end.
7. Leave flat outputs in place; document the new default path.

## 11. Definition of done
- `Loader → CompositionToH3Conditioning → sampler` produces correct video+audio for: images-only, single video+soundtrack, **two videos**, video+standalone audio, and mixed compositions — with tags resolving to the right media in every case.
- No manual `VHS_LoadVideo` / audio-load wiring required for the common path.
- Existing graphs on the flat outputs still function.
- Parity test and at least one end-to-end smoke test pass.
