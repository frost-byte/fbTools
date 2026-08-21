# Audio Reference — Observations Log

Running log of empirical findings from live H3 generation runs with voice/audio references.
Revisit and revise default settings and guidance based on entries here.

Companion to: `h3_ref2va_audio_reference_rules.md`

---

## Speech pace mapping (initial defaults — 2026-08-17)

Three-level pace selector per shot dialogue field. Mapped values chosen from typical conversational
English speech rate research; not yet validated against live H3 runs.

| Selector | chars/sec | Prompt injection | Notes |
|----------|-----------|------------------|-------|
| Slow | 10 | "speaking slowly and deliberately" | Dramatic, emotional, deliberate delivery |
| Normal | 13 | *(none)* | Unmarked = default conversational pace |
| Fast | 16 | "speaking quickly" | Excited, nervous, rapid exchanges |

Inter-word pause constant: 0.07 s/word.
Punctuation offsets: comma +0.20 s, sentence-end (.!?) +0.35 s.

**To revisit:** Compare estimated `trim_to` values against the audio duration that actually produces
clean voice output. If the model consistently clips early or runs long, adjust the chars/sec values
and record the corrected values here.

---

## Turbo LoRA + audio (initial guidance — 2026-08-17)

Community finding (HuggingFace discussion #64): Turbo LoRA degrades audio track quality badly.
When a voice/audio reference is present, system warns (or disables) turbo.

**To revisit:** Establish whether there is a turbo strength threshold below which audio quality is
acceptable, or whether it is a binary on/off interaction. Record step count + turbo setting used in
each voice run.

---

## Voice clone starting preset (initial — 2026-08-17)

Community-validated starting point: 20 steps, no turbo LoRA, ~0.2 MP output resolution for the
voice pass. Source: MiniMaxAI/MiniMax-H3 HuggingFace discussion #64.

**To revisit:** Test whether higher step counts improve voice consistency without turbo, and whether
resolution affects voice quality independently of visual quality. Record observations here.

---

## Minimum audio reference duration (open — 2026-08-17)

H3 spec floor is 2 s per clip. Whether clips shorter than some practical threshold (e.g. 3–4 s)
reduce voice fidelity is unknown. The loop/tile feature (copies short clips to meet the floor) is
a mitigation, but the optimal minimum for reliable voice transfer is not yet established.

**To revisit:** Test with clips at 2 s, 3 s, 4 s, 6 s, 8 s. Note whether timbre transfer quality
changes with reference duration. Update the loop_to_target default target accordingly.

---
