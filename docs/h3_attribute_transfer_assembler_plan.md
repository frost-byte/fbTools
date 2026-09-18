# Action Plan — Original-Subject Tagging & `attribute_transfer` in the Prompt Assembler (H3 ref2va)

**Goal:** re-introduce a `<Subject N>` tag for the **original subject in the video being edited**,
marked `attribute_transfer`, in the prompt assembler — but scoped *correctly* this time, so it fixes
(rather than causes) the swap problems. This was tried before and removed because it hurt generations;
the likely reason is a scoping mistake, addressed below.

Marker legend:
- **[INVARIANT]** — must hold.
- **[DECIDE] / [EXPLORE]** — judgment call.
- **[CAVEAT]** — empirical / verify by testing.

---

## 1. Why this matters (the two symptoms this targets)
Observed failures the user wants fixed:
1. **The character swap doesn't take from the start** — the output begins showing the original subject
   and only switches over partway through.
2. **Costumes/outfits from the original video aren't replaced** — the reference subject's face/identity
   may come through, but the original's clothing persists instead of the reference subject's appearance.

Both point at the same root cause: without an explicit, correctly-scoped instruction that the original
person is being *overwritten*, the model defaults toward **preserving** the video's content (identity
and clothing), and asserts the replacement weakly/late.

---

## 2. The likely reason it "messed up generations" before — the scoping trap [INVARIANT to avoid]
`attribute_transfer` in the H3 guide means "referenced characteristics are transferred to a different
target subject." The default reading of "characteristics" leans toward **appearance**. So if the
original subject is marked `attribute_transfer` **without explicitly scoping what transfers**, the model
can read it as *"carry the original's appearance attributes onto the output"* — i.e. **keep the original
costume/face** — which is the exact opposite of the intent and matches symptom #2.

**The fix is precise scoping, not just the tag:** state that what transfers from the original is
**motion / pose / gesture / timing / screen-position ONLY**, and that **appearance (face, hair, and
clothing/costume) is NOT copied** — appearance comes entirely from the replacement subject's own
reference image. Get this scoping wrong and the tag hurts; get it right and it's the mechanism that
makes the swap complete.

---

## 3. Correct construction — what the assembler must emit
Four coordinated pieces. **[INVARIANT]** all four, or the tag underperforms.

**3.1 Original subject gets its own `<Subject N>` tag, minimally described.**
- Identify *who* in the video is being replaced (enough for the model to locate the target), but keep the
  description **lean** — do NOT richly describe the original's appearance, since vivid original-appearance
  text reinforces the identity you're trying to overwrite.
- e.g. `<Subject 2> is the person originally in <Video 1> who is being replaced.`

**3.2 `retention_analysis` marks the original `attribute_transfer` with explicit scope.**
- **[INVARIANT]** Spell out the transfer boundary:
  - **Transfers to the replacement:** pose, movement, gestures, timing, screen position, camera framing.
  - **Does NOT transfer / is replaced:** face, hair, and **clothing/costume** — these come from the
    replacement subject's reference image, not the original.
- e.g. `<Subject 2>: attribute_transfer — pose, movement, gestures, timing and screen position transfer
  to <Subject 1>; the original's appearance, including face, hair, and clothing, is NOT copied and is
  fully replaced by <Subject 1>'s appearance from <Picture 1>.`

**3.3 The replacement subject's appearance must be COMPLETE — including the outfit.** (fixes symptom #2)
- **[INVARIANT]** If the replacement (`<Subject 1>`) is described as only a face/hair, the model has
  nothing to put in place of the original costume, so it keeps the original clothing. The replacement's
  description (and its `<Picture 1>`) must convey a **full appearance including clothing/outfit**, so
  there's a complete appearance to swap *in*.
- If the reference image doesn't show the desired outfit, the outfit must be stated explicitly in the
  replacement subject's description so the model has it to render.

**3.4 The replacement identity is asserted from the START and re-cited across the timeline.** (fixes symptom #1)
- **[INVARIANT]** In `detailed_description`, name `<Subject 1>` with its appearance at `[Shot 1]` at the
  very beginning, and re-cite it (appearance-only, no added motion language) at later shot/beat points —
  the same timeline-re-citation pattern proven to stop mid-clip identity decay. This is what makes the
  swap present from frame one rather than switching over partway.

---

## 4. Assembler changes
- Add emission of the **original-subject `<Subject N>`** block (definition + `attribute_transfer` line)
  when the composition is a subject-replacement edit. Derive the tag number consistently with the
  existing subject/ordinal scheme (don't collide with the replacement's tag or the `<Picture>`/`<Video>`
  numbering).
- Generate the `attribute_transfer` line from a **scoped template** (§3.2) — transfers list vs.
  replaced list — rather than free text, so the critical motion-transfers / appearance-replaced boundary
  is always explicit and can't drift into the ambiguous form that caused the earlier regression.
- Ensure the replacement subject's assembled description carries **full appearance incl. outfit** (§3.3);
  if the composition has a separate outfit/clothing field, include it; if not, consider adding one.
- Ensure the replacement's **timeline re-citation** (§3.4) is emitted at `[Shot 1]` and at later beats.

---

## 5. Make it toggle-able for A/B [DECIDE]
Since this was removed once for hurting output, the user needs to compare with/without at a fixed seed.
- **[INVARIANT]** Gate the original-subject-`attribute_transfer` emission behind a flag (per-composition
  or a node toggle), default matching current behavior, so the user can A/B the *correctly-scoped*
  version against no-tag at a fixed seed and judge for themselves.
- This turns "I think it messed things up" into a measurable comparison, rather than a one-way change.

---

## 6. Wrinkles & caveats
- **[CAVEAT]** How strongly the local ComfyUI ref2va path honors `attribute_transfer` semantics is
  empirical — the tag is the format-supported way to express the intent, but the user's own fixed-seed
  tests are the authority. The scoping fixes (§2–3) are what make it *likely* to help; testing confirms.
- **[INVARIANT]** Keep the original subject's description **minimal**; over-describing its appearance
  fights the swap. Spend description words on the *replacement's* appearance, not the original's.
- **Preserved others:** if the composition keeps other people or the background from the video, those
  get their own `<Subject N>` marked `fully_preserved` (not `attribute_transfer`) — separate from the
  replaced subject. Ensure the assembler distinguishes "replace this subject" from "keep this subject."
- **Don't double-describe motion:** §3.4 re-citations are appearance-only; the motion is already carried
  by `<Video 1>` + the `attribute_transfer` scope. Adding motion text competes with the video and
  degrades transfer fidelity (prior finding).
- **Consistency:** the outfit/appearance wording for the replacement must be identical wherever it
  appears (subject definition, retention_analysis, shot re-citations) — contradictions weaken the swap.

---

## 7. Definition of done
- For a subject-replacement composition, the assembler emits: original `<Subject N>` (minimal) +
  scoped `attribute_transfer` (motion transfers, appearance incl. clothing replaced) + a complete
  replacement appearance incl. outfit + `[Shot 1]` and later-beat replacement re-citations.
- The `attribute_transfer` line always uses the scoped template (never the ambiguous "attributes
  transfer" form that reads as keep-the-costume).
- Emission is flag-gated so the user can A/B at a fixed seed.
- Preserved other-subjects/background use `fully_preserved`, distinct from the replaced subject.
