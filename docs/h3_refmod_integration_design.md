# Design Doc — RefMod-style reference caching for H3 (Bundles → Source Profiles)

**Repo:** `frost-byte/fbTools`
**Related:** `utils/h3_vram_estimator.py`, `extension.py`'s `CompositionToH3Conditioning` / `MiniMaxH3ReferenceToVideo` delegation, `SceneCastBuild` ordinal-matching, `SourceProfileClipPrompt`
**External reference (inspiration only, not a dependency):** [`ComfyUI-MiniMaxH3Mod`](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod) — an experimental third-party mod that pre-extracts H3 references into reusable `.safetensors` "RefMod" files.
**Status:** design accepted in principle; no code written yet. This doc records the investigation and the architecture decisions so implementation can start from a stable footing.

---

## 1. Motivation

Long two-pass H3 runs with several reference items (images/videos loaded fresh into `CompositionToH3Conditioning` on every execution) OOM on the second sampler pass. Investigation (see `utils/h3_vram_estimator.py`'s docstring for the calibration incident) established:

- Attention memory scales roughly with `total_tokens²`, where `total_tokens` = main generation tokens + every reference's tokens.
- fbTools' current reference loading is **canvas-matched, unpooled** — a video reference costs full per-frame token weight for its entire length, with no cap.
- A third-party mod (RefMod) demonstrates that references can be **pre-extracted and pooled** to a small, capped token footprint, at the cost of visual fidelity — explicitly *not* recommended by its own author for identity/face references.
- Re-loading + re-VAE-encoding the same reference media on every run is also wasted wall-clock work; a cached, pre-encoded latent loads via a plain tensor deserialization instead.

Two separate benefits are being chased here, and they don't require the same trade-off:

1. **Load-time efficiency** (skip repeated decode+VAE-encode) — free win, no fidelity cost, applies to any reference regardless of pooling.
2. **Token/attention reduction via pooling** — a real fidelity trade-off, only safe where fidelity doesn't matter for the reference's role in the shot (see §3).

---

## 2. Two different "upstreams" — and which one we track

| | Native H3 conditioning contract | ComfyUI-MiniMaxH3Mod's extraction/pooling algorithm |
|---|---|---|
| Source | ComfyUI core (`comfy_extras/nodes_minimax_h3.py`) | Third-party, experimental, explicitly versioned bundle format (v4/v5) |
| What it defines | `minimax_refs` list on the conditioning dict, `[1,24,T,H,W]` latent shape, `T×(H/2)×(W/2)` token math | How to encode + average-pool a reference before appending it |
| Volatility | Low — moves on ComfyUI's own release cadence, same as everything else we build against | High — the repo's own README calls features experimental, flags "known failures" (voice transfer), and its own code prints warnings against certain settings |
| Our relationship | **Hard dependency**, already tracked (`h3_vram_estimator.py`, `MiniMaxH3ReferenceToVideo` delegation) | **Inspiration only.** We reimplement the idea natively; we do not import, vendor, or consume the mod's file format |

**Decision: reimplement natively.** The mod's own extraction nodes are built as `io.ComfyNode.execute()` methods meant to run through ComfyUI's queue executor, not as plain functions — reusing them from a REST endpoint (§5) would mean vendoring code anyway. Native reimplementation also means "upstream changes" to the experimental mod repo can never break us; the only volatility we're exposed to is contract #1, which we already track.

---

## 3. What we learned about the fidelity/overhead trade-off

### 3.1 Masking alone does not reduce token cost
The mod's `_mask_latent` suppresses (blurs) pixels outside a mask within the *same-shape* latent — it doesn't crop or downsample. A "subject-isolated" reference at full canvas resolution costs exactly the same tokens as the unsplit source. Splitting one video reference into N per-subject masked copies **without pooling** multiplies reference token cost by N, and — because attention scales as `tokens²` — compounds into a much larger memory increase than N× once combined with the rest of the sequence. Concretely, against the calibration incident (300 frames @ 832×640, ~39,000 main tokens): one full clip reference ≈ 39,000 reference tokens; three unpooled per-subject copies ≈ 117,000 reference tokens; total sequence tokens roughly double, and attention memory roughly quadruples. **Unmasked-and-unpooled per-subject splitting is a regression, not an optimization.**

### 3.2 Pooling only becomes acceptable when identity isn't the reference's job
The mod's own extraction tool warns against aggressive pooling (`--mode training`) for identity/face references — pooling to a 16×16 (or smaller) latent grid destroys facial detail. That warning matters **only when the reference is the thing determining the output character's appearance.**

Two roles need to be told apart:

- **Reference Bundles** (fbTools' existing per-subject appearance references) — these *are* appearance-determinative. Full-resolution ("encode"-equivalent) should stay the default.
- **Source Profile clip subjects being *replaced* by a bundle** — the source subject's own face/appearance is explicitly discarded; the bundle supplies the visual identity instead. All the source-derived reference needs to convey is **position, motion, and interaction** so the replacement lands in the right place doing the right thing. This maps directly onto the mod's own `attribute_transfer` retention semantics ("keep style/attributes, not identity") and onto the `include_original_subject_tags` behavior already in `SourceProfileClipPrompt` ("face, hair, and clothing are NOT copied" for a replaced subject).

For that second role, aggressive pooling is not a compromise — it's correct, and it's what makes per-subject splitting net-cheaper than the unsplit reference rather than 3-4x more expensive (≈1,000 pooled tokens/subject vs. ≈39,000 for one full-resolution clip reference).

**One caveat that is not about identity:** a subject occupying a small fraction of the frame can lose positional/motion signal at very coarse pooling, independent of face-detail concerns. The pool-grid floor should be driven by the subject's screen-space footprint and motion complexity, not by an identity argument (there isn't one, in this role).

### 3.3 Net design rule
| Reference role | Appearance matters? | Default mode | Rationale |
|---|---|---|---|
| Reference Bundle (subject appearance source) | Yes | Full-resolution ("encode"-equivalent), pooled ("training"-equivalent) as an opt-in toggle | Bundle images are appearance-determinative; the whole point of the bundle is fidelity |
| Source Profile clip, kept subjects | Yes (unchanged) | Left in the base clip reference, untouched | Nothing is replacing them; no reason to alter fidelity |
| Source Profile clip, replaced subjects | No — only position/motion/interaction | Pooled (coarse grid, floor set by screen-space footprint) | Bundle supplies appearance; source-derived reference only needs to place and move it correctly |

---

## 4. Compartmentalization: making our implementation swappable and versionable

Since we're reimplementing rather than depending on the mod repo, we control the whole surface — but should still isolate the parts most likely to need revision (pooling heuristics, VAE loading strategy, on-disk cache schema) behind narrow seams.

```
utils/h3_ref_math.py       # PURE — no comfy deps, unit-tested like the rest of utils/
                            #   pool-grid sizing, aspect-preserving grid math,
                            #   token counting (extends h3_vram_estimator.py, doesn't duplicate it)

utils/h3_ref_backend.py    # comfy-dependent (same tier as utils/pose.py)
                            #   H3RefBackend interface: build(source, mode, pool_grid) -> RefBlock
                            #   RefBlock = {latent, shape, concept_type, retention_default, schema_version}
                            #   NativeH3RefBackend: VAE encode + pool using h3_ref_math, our only
                            #     implementation for now. A second backend (e.g. one that delegates
                            #     to ComfyUI-MiniMaxH3Mod if installed) could implement the same
                            #     interface later without touching anything above it — the same
                            #     graceful-optional-dependency shape utils/nlf_pose.py already uses
                            #     for ComfyUI-SCAIL-Pose.

utils/h3_ref_cache.py       # owns the on-disk cache format
                            #   SCHEMA_VERSION constant, read/write, invalidate-on-mismatch
                            #   (own format — NOT the mod repo's v4/v5 bundle schema)

extension.py                # thin REST handlers calling into backend + cache modules;
                            # "attach RefBlock.latent to minimax_refs" reuses the exact
                            # attachment logic MiniMaxH3ReferenceToVideo already has —
                            # this is contract layer 1 (§2) and must not be duplicated
```

**Why this buys versionability:**
- Changing the pooling algorithm = edit `h3_ref_math.py`, rerun its unit tests, bump `SCHEMA_VERSION` so stale cache entries are treated as invalid and rebuilt. Nothing else in the stack changes. (This is the same reload-counter/invalidation discipline we already learned the hard way from the Source Profile save/cache-invalidation regression — be deliberate about it from the first commit here, not retrofitted later.)
- Adding upstream interop later (if ever wanted) = one new file implementing `H3RefBackend`, gated behind an install-check. It never becomes a hard dependency of the REST/UI/node layer.
- "Keeping in sync with upstream" becomes a manual, opt-in act — periodically checking whether the mod repo's community has found better pool-grid defaults and porting the *idea* into `h3_ref_math.py` — never an automatic coupling that could break when they rev their bundle format.

---

## 5. UI/backend integration shape

Generation-time *consumption* stays node-based (no change to how `CompositionToH3Conditioning` / `SourceProfileClipPrompt` wire into a sampler); only cache *construction* moves off the node graph and onto UI-triggered REST calls, per the requirement that this not be node-based functionality.

**Open technical requirement:** a REST handler has no graph context, so it has no VAE input to reach for. `comfy.sd`'s VAE loading is a plain function, not something requiring the queue executor, so an endpoint can load the H3 VAE directly the way a loader node does — but this needs an explicit strategy (cold-load per request vs. a resident cache, and whether it shares a VAE instance with anything else already resident) before endpoints are implemented. **[DECIDE during implementation.]**

Proposed endpoints (final paths TBD at implementation time, following the existing `/fbtools/*` convention):
- `POST /fbtools/h3_refmod/bundles/<bundle_id>/build` — `mode: encode|training` (default `encode`, per §3.3)
- `POST /fbtools/h3_refmod/source_profiles/<profile>/clips/<clip_id>/build` — `scope: whole_clip|per_subject`, `subject_ids: [...]` when `per_subject`
- `GET /fbtools/h3_refmod/.../status` — cache presence/staleness inspection

Encoding a long clip isn't instant; this should use the existing `send_status_update()` / `fbtools.status` websocket pattern rather than a blocking request, consistent with other async work in this codebase (e.g. dataset recaption).

---

## 6. Staged rollout

### Stage 1 — Reference Bundles
- Cache a VAE-encoded (optionally pooled) latent per bundle, keyed by a content signature (image/video file stat, like the mod's own cheap `os.stat`-based cache-check pattern) so unmodified bundles don't re-encode every run.
- `encode` mode is the default; `training` (pooled) exposed as an explicit user-facing toggle on the bundle editor, per §3.3 — appearance matters here, so the safer setting should require an opt-in, not the reverse.
- Lowest-risk stage: no new segmentation work, no multi-subject ambiguity, pure caching win plus an optional token-cost win for users who accept the fidelity trade explicitly.

### Stage 2 — Source Profile clips, whole-clip scope
- One cached RefMod-equivalent per clip, representing the *entire* clip exactly as `SourceProfileClipPrompt` uses it today (single `<Video 1>` reference, multi-subject disambiguation still resolved via `<Subject N>` prose tags — unchanged).
- Pure caching/perf win (skip repeated decode+encode); does not change accuracy of subject targeting. This is deliberately the same behavior as today, just cached.

### Stage 3 — Source Profile clips, per-subject scope (opt-in)
- Requires a per-subject **spatial segmentation** step that doesn't exist yet — nothing in Source Profile subject metadata currently locates a subject in-frame (metadata today is identity-only: name/id/concept_id). Likely built on the SAM3 segmentation already wired up for `SceneMaskDefinition`, driven by each subject's description (or a tracked bounding box, if the analysis pipeline gains one) across the clip's frame range.
- Toggle: one RefMod per clip (Stage 2 behavior) vs. one per subject.
- Per-subject caching defaults to **pooled** (§3.2/§3.3), since the primary motivating case is subject *replacement* — accurately targeting the (usually 1, sometimes 2-3) subject(s) to swap out, while a bundle supplies the replacement's visuals.
- **Retaining some original subjects while replacing others** is not a new mechanism: kept subjects simply stay in the base clip reference untouched (no split needed for them); only replaced subjects get pulled into their own pooled, position/motion-only reference. This composes directly with the per-clip/per-subject ordinal-matching logic already built for `SceneCastBuild` this session, rather than requiring a parallel resolution system.

Stage 3 is explicitly the highest-effort, highest-uncertainty stage (new segmentation dependency, new UI for per-subject toggling, more cache-invalidation surface if a clip's cut points or subject list change) and should not be started until Stages 1-2 are validated in real use.

---

## 7. Open questions to resolve during implementation

- VAE-loading strategy for REST-triggered encoding (§5) — resident vs. per-request load, and interaction with whatever VAE the active workflow may already have loaded.
- Exact on-disk cache location/naming for bundle and clip caches (likely alongside existing bundle/clip storage under `user_data_dir`, matching how bundle appearance images are already stored).
- Whether pool-grid floor sizing (§3.2) should be a fixed default users can override, or computed automatically from the subject's mask coverage fraction.
- Per-subject spatial segmentation approach for Stage 3 (SAM3 text-prompt grounding vs. extending `describe_clip`/analysis output with tracked bounding boxes) — needs its own scoping pass before Stage 3 starts.
