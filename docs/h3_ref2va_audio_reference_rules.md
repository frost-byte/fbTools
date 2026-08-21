# MiniMax H3 Ref2VA — Audio Reference Rules & Handling

**Purpose:** constraints and practical guidance for how the fbTools composition system should
**define, validate, load, and route audio references** for MiniMax H3 Ref2VA. Companion to the
`FBTOOLS_H3_REFPLAN` bundle + `CompositionToH3Conditioning` terminal-node plan
(`h3_refplan_action_plan.md`) — audio is one of the reference modalities that plan resolves.

Marker legend (same as the refplan plan):
- **[INVARIANT]** — official MiniMax spec; enforce/validate, do not deviate.
- **[DEFAULT]** — make this the default behavior, allow override.
- **[GUIDANCE]** — soft recommendation / surface as a warning.
- **[DECIDE] / [EXPLORE]** — genuine judgment call; inspect the code and choose.

---

## 1. Hard constraints — official Ref2VA spec — [INVARIANT]

Enforce these at composition-resolution and again at the terminal node before delegating:

- **Per-clip duration: 2–15 seconds.** Minimum is 2 s (there IS a floor), maximum 15 s.
- **Total audio ≤ 15 seconds** aggregate across all audio references.
- **Count ≤ 3** audio references.
- **Pairing required:** audio must accompany at least one image or video reference. Audio cannot be
  the sole media input in local Ref2VA.
- **Combined ceiling ≤ 12** mixed Ref2VA files (images + videos + audio). Related caps: ≤ 9 images;
  ≤ 3 videos, each 2–15 s, ≤ 15 s total.
- **Tagging:** reference as `<Audio N>`. Audio ordinals are numbered **independently** of `<Video N>`.
- **Formats (API):** WAV or MP3. In ComfyUI the terminal node hands the native H3 node a decoded
  waveform dict `{"waveform", "sample_rate"}`, so format handling lives at load time, not in the bundle.

**Validation the loader / terminal node must perform (fail with a clear node error, not silently):**
- audio ref `< 2 s` or `> 15 s`
- total audio `> 15 s`, or `> 3` audio tracks
- an audio ref with **no** accompanying image/video
- total mixed files `> 12`

---

## 2. Voice / dialogue guidance — community-validated — [GUIDANCE / DEFAULT]

The headline use (reference voice → newly generated dialogue matching its timbre) is real but
currently **finicky** in local ComfyUI. Encode these as defaults and warnings, not hard blocks:

- **[DEFAULT] Trim each voice reference to ≈ the expected duration of the spoken line it drives.**
  Known failure mode: the generated voice bleeds into gibberish / keeps speaking past the intended
  line, or replays the source audio. Matching the reference length to the line length is the
  community fix. The *content* of the trimmed audio doesn't matter for timbre — its **length** does.
  → Implication: an audio reference used for voice should carry or derive an
  **expected-line-duration**, and the loader/terminal node should be able to trim to it.
  Add a per-audio `trim_to` field (see §3).

- **[GUIDANCE] Direct pass-through is more reliable than voice-transfer right now.** If a composition
  just needs the reference audio reused as-is, treat that as the robust path and mark true
  voice-transfer (new dialogue in the reference voice) as experimental in any UI/labeling.

- **[DEFAULT] For dialogue/voice compositions, do NOT apply the Turbo LoRA.** Turbo preserves video
  quality but degrades the audio track badly. When a voice / audio-transfer reference is present,
  the system should disable turbo (or warn loudly).

- **[GUIDANCE] Known-workable voice-clone starting preset:** 20 steps, no speedup LoRA, low output
  resolution (~0.2 MP) for the voice pass. Worth exposing as a distinct **"voice/dialogue" preset**
  separate from the visual-quality preset, since their optimal settings diverge (turbo on/off, MP, steps).

---

## 3. Mapping into the refplan bundle + terminal node

- **Audio descriptor in `FBTOOLS_H3_REFPLAN`** should carry: `modality: "audio"`, `path`,
  `role` (voice / sfx / music / passthrough), `duration`, optional `trim_to` (expected line length),
  and **pairing linkage** — which visual it accompanies (needed for the pairing invariant in §1 and
  for the video-soundtrack suffix pairing below).
- **Suffix pairing (from the native node):** a video's soundtrack pairs by trailing numeric suffix,
  `ref_video_N` ↔ `ref_video_audio_N`. Standalone voice/sfx/music audio → `ref_audio_N`. Keep the
  suffixes consistent when the terminal node builds the dicts.
- **Terminal node responsibilities:** load (and, if `trim_to` set, trim) audio →
  `{"waveform", "sample_rate"}`; run the §1 validation; then delegate to
  `MiniMaxH3ReferenceToVideo.execute`. Never pass an out-of-spec audio set to the native node.
- **Ordinal parity (see refplan plan §7):** `<Audio j>` labels in the prompt must match the native
  node's audio ordering — a video-soundtrack `<Audio j>` is emitted immediately before its
  `<Video k>`, then standalone audio, with audio ordinals counting both in appearance order.
  The bundle's ordered `references` list stays the single source of truth for this numbering.

---

## 4. Wrinkles / open decisions

- **[DECIDE] Where trimming happens** — composition-resolution (loader emits an already-trimmed
  descriptor) vs load time in the terminal node. Prefer **terminal-node trim** so the bundle stays
  descriptor-only and cache-cheap (consistent with the decode-late principle in the refplan plan).
- **[EXPLORE] Auto-derive `trim_to` from the prompt's dialogue** — parse the `<d>…</d>` line spans /
  cut timestamps to suggest an expected line duration automatically, vs requiring a manual field.
- **[DECIDE] Over-spec behavior** (total > 15 s, count > 3, files > 12) — hard error vs
  auto-truncate-with-warning. Prefer an **explicit error**; silent truncation hides intent.
- **[GUIDANCE] Preset split** — surface "voice/dialogue" vs "visual-quality" presets, since turbo,
  MP, and step count differ between them.

---

## Sources & freshness
- Official Ref2VA limits (2–15 s each, 15 s total, ≤ 3 audio, ≤ 12 files, audio must accompany a
  visual): MiniMaxAI/MiniMax-H3 model card (Hugging Face), fal.ai/minimax-h3, corroborated by
  secondary guides (morphic, domoai, minimaxh3.run).
- Trim-to-line-length fix, pass-through reliability, turbo-degrades-audio, and the
  20-step / no-turbo / 0.2 MP voice-clone recipe: MiniMaxAI/MiniMax-H3 Hugging Face discussion #64.
- Verified against sources dated ~Aug 2026. Re-check the model card and the official ComfyUI H3
  workflow guide as the model and native nodes evolve — treat the [GUIDANCE] items especially as
  current community findings, not permanent guarantees.
