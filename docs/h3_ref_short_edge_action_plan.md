# Action Plan — Configurable Reference Shortest-Edge (between `match` and `max`) for the H3 Conditioning Node

**Repo:** `frost-byte/fbTools` · the `CompositionToH3Conditioning` terminal node
**Native ref:** `Comfy-Org/ComfyUI` · `comfy_extras/nodes_minimax_h3.py` · `MiniMaxH3ReferenceToVideo.execute`
**Companions:** `h3_refplan_action_plan.md`, `h3_ref2va_audio_reference_rules.md`

Marker legend (same as prior briefs):
- **[INVARIANT]** — must hold; don't deviate without flagging.
- **[VERIFY]** — an assumption in this plan that MUST be confirmed against the actual code before building on it. These are load-bearing.
- **[DECIDE] / [EXPLORE]** — genuine judgment call; inspect and choose.

---

## 1. Objective

Give the terminal node a **configurable reference shortest-edge**, so the reference can be processed at a resolution *between* the native node's two settings — `match` (reference sized to the output canvas) and `max` (reference sized to a large fixed short-edge). The goal is a cost/fidelity dial: `max` is the identity clincher but the dominant per-step time/VRAM cost (reference tokens re-attend every step, and token count scales ~quadratically with short-edge); `match` is cheap but softer on reference fidelity. An intermediate short-edge buys most of `max`'s identity benefit at a fraction of its cost.

Secondary objective: **efficient handling of oversized (e.g. 4K) source video**, since the reference is downscaled internally regardless — feeding 4K wastes decode/RAM on pixels that get thrown away.

**Non-goal:** changing the native node's behavior or the `match`/`max` modes themselves. This is additive to the terminal node.

---

## 2. Assumptions to confirm FIRST — do not build until these are verified

These come from ecosystem working values and general reasoning, **not** from a spec I can see. The whole feature depends on them, so read `nodes_minimax_h3.py` and confirm before writing code.

- **[VERIFY] What `match` actually resolves to.** Assumption: `match` sizes the reference to the output canvas — i.e. shortest edge derived from the node's `width`/`height` inputs (the output video dims). Confirm this is what `execute` does with `ref_image_size="match"`.
- **[VERIFY] What `max` actually resolves to.** Assumption: `max` targets a large fixed short-edge around **2048**. Confirm the real number and whether it's a short-edge cap, a total-pixel cap, or something else. (Look at `adapt_canvas` / `_resize` / the ref-processing loop in `execute`.)
- **[VERIFY] The divisibility grid.** Assumption: reference/output dims must land on a latent-friendly multiple (16, 32, or 64). Confirm the actual required multiple — it determines the legal values for a custom short-edge and shifts the 16:9 pixel rungs below by a few pixels.
- **[VERIFY — THE CRITICAL ONE] Does `execute` re-resize frames you feed it?** The intended implementation (below) is "resize reference frames to the chosen short-edge in our node, then delegate." That only works if the native node does **not** then re-resize them. If `match` downscales to the canvas and `max` upscales to ~2048, then neither mode leaves an arbitrary pre-sized reference untouched, and feeding 1280-px frames + a mode will NOT produce a 1280-px reference. Determine exactly what each mode does to an already-sized reference. **This single fact decides whether the implementation is "pre-resize + pick a passthrough mode" or "replicate/patch the reference-prep path."**

---

## 3. 16:9 pixel dimensions (reference values, pending the divisibility [VERIFY])

Output-canvas options at 16:9 (true 16:9 is 1.777:1; some rungs are near-16:9 for grid friendliness):
- **~0.4 MP (iterate):** 848×480 — 16:9 at /16, ~0.41 MP (official template default).
- **~0.9 MP:** 1280×720 — true 16:9, both /16, 0.92 MP. Clean HD option.
- **~1.0 MP (native target):** 1344×768 — **7:4 (1.75), not true 16:9**; used because /64-friendly and the honest native-quality point. For true 16:9 near 1 MP prefer 1280×720.
- **~2 MP:** 1920×1080 — true 16:9 but at/above the practical local ceiling on a 24 GB card with references.

The **reference** short-edge is a separate number from the output canvas — that's the whole point of this feature. Span to interpolate across: roughly **output-canvas short-edge (match) up to ~2048 (max)** — e.g. 720 → 2048, with 1024/1280/1536 as sensible middles.

---

## 4. Proposed feature

Add an optional input to `CompositionToH3Conditioning`, e.g. `ref_short_edge: INT` (0 or None = "use the selected mode's default," i.e. current behavior). When set, the node resizes reference frames to that shortest edge (snapped to the [VERIFY] divisibility grid), preserving aspect, before delegating.

**Cost model to document in the node's tooltip/help** so the value is meaningful: reference token count scales ~with the square of the short-edge, so e.g. 1280 vs 2048 ≈ (1280/2048)² ≈ 40% of the reference tokens — a large per-step saving for a modest fidelity drop. This is the reason the knob exists.

---

## 5. Implementation approach — branches by what [VERIFY §2] finds

**If the native node passes pre-sized reference frames through unchanged (best case):**
- In the terminal node, resize reference frames to `ref_short_edge` (aspect-preserving, grid-snapped), then call `execute` with `ref_image_size` set to whichever mode is a passthrough for that size. Simple; no native patching.

**If the native node always re-resizes to the mode's target (likely):**
- **[DECIDE]** between: (a) replicate the native reference-prep resize logic in our node using our own target short-edge, then feed frames already at the size the mode will "resize" them to (i.e. pre-empt it so its resize is a no-op); or (b) call the native reference-prep helper directly with a custom size if one is exposed; or (c) as a last resort, monkeypatch/override the size the node uses. Prefer (a) — least coupling, no native modification — but it depends on the mode's resize being idempotent when the input already matches its target.

**[INVARIANT]** Keep `match`/`max` working exactly as before when `ref_short_edge` is unset. Additive only.

---

## 6. The oversized-source / caching optimization (your reuse question)

Two sub-questions to resolve:

- **[VERIFY] Does the resize/decode happen every node call with no caching?** Almost certainly yes — the terminal node decodes and resizes on each execution, and ComfyUI re-runs it when inputs change. Confirm there's no reuse of a prior decode.
- **[DECIDE] Should oversized reference video be pre-downscaled once (a cached proxy) rather than decoded-and-downscaled every run?** For a reused 4K reference, decoding full-res frames every run — only to throw the extra pixels away — is wasted RAM and time. Options:
  - **Cache a downscaled proxy:** on first use, write a downscaled copy of the reference video (or its extracted/selected frames) to a cache keyed on `(source path/mtime, target short-edge, frame-selection params, aspect/crop)`; reuse it on subsequent runs. Biggest win for iterative work on the same reference.
  - **Leave it decode-late** (per the refplan plan's descriptor-only bundle principle) and accept the re-decode cost. Simpler, keeps the bundle cheap, but pays full 4K decode each run.
  - **Note the tension:** the refplan plan says keep the bundle descriptors-only (no decoded tensors) for cache-cheapness. A downscaled *proxy file* is compatible with that — it's still a path/descriptor, just pointing at a smaller asset — so a proxy cache does not violate decode-late; it's a preprocessing step that produces a smaller source. Prefer the proxy-file approach over caching decoded tensors in the bundle.

**Quality guidance to encode:** downscaling source **above** the internal reference target is free (those pixels are discarded anyway); downscaling **below** it softens the reference. So the proxy's short-edge should be `>=` the largest reference short-edge you'll actually use. If you run `max` and want its full fidelity, the proxy short-edge should be `~2048`, not lower — otherwise you're upscaling a downscaled image back toward 2048 (mild softness).

---

## 7. Wrinkles & considerations

- **[VERIFY] divisibility** — snap any custom short-edge to the required multiple; an off-grid value may error or be silently re-adapted (defeating the point).
- **Aspect handling in one pass** — do the 16:9 crop/fit *and* the short-edge resize in the same preprocessing step, not two, and watch that a 9:16→16:9 crop doesn't clip the subject's motion range (recurring issue).
- **`max`-upscale waste** — if a proxy or pre-resize is below `max`'s target and the run uses `max`, you upscale a smaller image toward 2048; acceptable but document it.
- **Fingerprint** — if you add a proxy cache, its key AND the terminal node's `fingerprint_inputs` must include `ref_short_edge` (and the proxy's identity) so changing the short-edge invalidates correctly. Editing `ref_short_edge` must re-run.
- **VRAM/time is the payoff** — validate the square-law saving empirically: same clip/seed, sweep `ref_short_edge` across {match-default, 1024, 1280, 1536, max}, record per-step time, peak VRAM, and identity fidelity. That table is the feature's justification and tells you the sweet spot on a 24 GB card.
- **Interaction with frame count** — reference cost is (frames × tokens-per-frame × steps); this knob lowers tokens-per-frame, independent of the frame-count lever. Both compound.
- **Don't downscale below the mode's target** in the proxy — quality floor.

---

## 8. Suggested sequence
1. Resolve all of §2 [VERIFY] by reading `nodes_minimax_h3.py` `execute` — especially "does it re-resize fed frames" and the real `max` target + divisibility. Nothing else is safe to build until these are known.
2. Pick the §5 implementation branch based on #1.
3. Add `ref_short_edge` input (unset = current behavior); implement resize (aspect + grid) in the terminal node's preprocessing, before delegation.
4. Update `fingerprint_inputs` to include `ref_short_edge`.
5. Run the §7 sweep; confirm the cost/fidelity tradeoff and pick sensible defaults/presets.
6. Only then, if the proxy-cache win is worth it, add the §6 downscaled-proxy cache for oversized reference sources.

## 9. Definition of done
- `ref_short_edge` produces a reference processed at that short-edge (verified by inspecting actual token count or VRAM/time, not just the input value), sitting measurably between `match` and `max` on cost and fidelity.
- `match`/`max` unchanged when `ref_short_edge` is unset.
- Changing `ref_short_edge` invalidates the node (re-runs).
- The sweep table exists (time / VRAM / fidelity across short-edge values) to justify the default.
- (If built) oversized sources are downscaled once and reused, with the proxy keyed so source edits and short-edge changes invalidate it.
