# OpenShot Bridge Template — Prompt Shorthand Convention

Status: **shipped** — the `S1`/`V1`/`A1` shorthand is implemented
(`fbt_PromptShorthandExpander`, `utils/prompt_shorthand.py`) and confirmed working live
against a real 5-subject prompt (2026-10-01). The template also pre-fills the Prompt field
with a baseline `default_prompt` the user can edit. Retention shorthand and the short
section-name mapping are not yet built — see "Potential future enhancements" below.

## Why this exists

The `video-bridge-two-clips` OpenShot template (`src/comfyui/video-bridge-two-clips.api.json`
in the `openshot-qt` fork) builds a MiniMax H3 Ref2VA structured prompt — the same six-section
format `utils/prompt_assembler.py` already produces for Bundles/Compositions elsewhere in
fbTools. That format is verbose and easy to get wrong by hand (exact section names, `<Subject
N>`/`<Video N>`/`<Audio N>` labeling, speaker IDs). This doc defines a **short, user-facing
notation** an OpenShot user can type directly into the template's Prompt field, and how it maps
onto the real format before submission.

This is deliberately a *thin* convention, not a reimplementation of `prompt_assembler.py`'s full
`scene_instance`/`slot_assignments` model — the bridge template has exactly two video references
and (today) one subject, so it doesn't need the full Scene Composition Engine's generality.

## Section name mapping

**Not yet built** — see "Potential future enhancements" below. Today, section headers in the
Prompt field must use the model's real section names (`subject_definitions`, `summary`,
`retention_analysis`, `detailed_description`, `overall_soundscape`, `non_diegetic_music`) exactly,
as shown in the template's `default_prompt`.

## Reference shorthand

These are the model's own native labels, not something fbTools invents — used as-is in
user-authored text, and expanded automatically by `fbt_PromptShorthandExpander` before the
prompt reaches the model:

- `V1` / `V2` — the two clip references (Clip A's tail, Clip B's head). Expands to `<Video 1>`/
  `<Video 2>`.
- `S1`, `S2`, `S3`, … — subject labels, one per distinct subject. Expands to `<Subject 1>`,
  `<Subject 2>`, etc. **Implemented and confirmed working** with a real 5-subject prompt
  (e.g. "S1 is Luke Skywalker... S2 is Darth Vader... S3 is Anakin Skywalker... S4 is Obi-Wan
  Kenobi... S5 is R2-D2..."), 2026-10-01. When subjects appear in different source clips but are
  narratively related (e.g. a character at two ages), prefer treating them as separate subjects
  rather than trying to merge them — confirmed easier for the model to work with.
- `A1` / `A2` — the two clips' voice-timbre audio references. Expands to `<Audio 1>`/`<Audio 2>`.
  Rarely referenced directly by the user; mentioned here for completeness since they appear in
  `subject_definitions`.

Dialogue lines follow the same convention `prompt_assembler.py` already uses natively — no
shorthand needed, since it's already short:

```
S1 says: "[English] your line here"
```

which expands to the full `<Subject 1> (S1) says: "[English] your line here"` form.

## Confirmed by trial (2026-09-30)

A flat, general statement worked for suppressing dialogue, with no subject-specific framing
needed:

```
There is no scripted dialogue.
```

This is now the `dialogue` field's default in the template. It's an argument for keeping the
*negative* case simple — the subject/speaker-ID scaffolding likely only matters when there's a
real line to attribute to someone, not for saying nothing is said.

## Potential future enhancements

Not currently planned for near-term work; documented here so they aren't lost.

- **Short section-name mapping** for freeform (non-shorthand) prompt authoring, so a user
  doesn't have to type the model's exact section names: `description` → `detailed_description`,
  `retention` → `retention_analysis`, `summary`/`soundscape` stay as-is, `non_diegetic_music`
  defaults to `N/A`. `subject_definitions` would stay generated/shorthand-driven rather than
  user-authored under its real name, since redefining it by hand risks breaking the reference
  binding the model depends on.
- **Retention shorthand** — `retention_analysis` today is a single fixed sentence; a
  multi-subject or multi-detail retention block would need its own shorthand (e.g. per-subject
  retention clauses keyed to `S1`/`S2`/…).
- **VLM-assisted prompt generation** — use fbTools' existing vision-LLM infrastructure
  (`captioner.py`/`llm_client.py`) to look at both clips and draft a starting
  `detailed_description`/`retention_analysis` block automatically, which the user then edits
  rather than writing from scratch.
