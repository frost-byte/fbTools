# Scene Cast Build — developer notes

`SceneCastBuild` (extension.py) builds a `SCENE_CAST` dict from a JSON list of cast entries
(`cast_entries_json`, edited by the on-node widget in `js/nodes/scene_cast_build.js`). The cast
pool comes from **one** upstream source, either a Source Profile or a Prompt Composition.

## Two modes

| | Source Profile | Prompt Composition |
|---|---|---|
| Input | `source_profile` (from `SourceProfileLoad`) | `prompt_composition` (from `CompositionLoad`) |
| Cast pool | the profile's subjects (per clip) | the composition's subjects (slot letters `A`, `B`, ...) |
| Segments on the timeline | the profile's clips | the composition's shots |
| Consumer | `SourceProfileClipPrompt` | `PromptCompositionLoader` |

If both are wired, **the Source Profile wins** and the composition is ignored (the backend logs a
warning; the widget does the same). Both wire inputs are looked up through `node.inputs`, not
`node.widgets`, so the widget-name contract test is unaffected.

New outputs are always **appended** (the `prompt_composition` pass-through is last) so existing
links keep their slot indexes.

## Refresh when the loader's selection changes

`SourceProfileLoad` and `CompositionLoad` wrap their combo widget's callback
(`js/nodes/source_profile_load.js`, `js/nodes/composition_load.js`) and re-fire
`onConnectionsChange` on every node wired to output slot 0. `SceneCastBuild` reacts in
`onConnectionsChange` for either input name and runs `_refreshProfileDependentState`
(composition first, then the profile refreshes). **Keep the loader's dict output at slot 0.**

## Composition mode details

- **Matching**: a cast entry applies to the composition slot whose `subject_id` equals the entry's
  `subject_id` (`apply_cast_to_subjects`). Array order and slot letters do not matter.
- **Ordinal (`#`)**: "the Nth subject in the composition sharing this bundle's pronoun_style".
  Resolved authoritatively in `SceneCastBuild.execute()` via
  `resolve_ordinal_from_list` (`utils/source_profiles.py`) over `composition_ordinal_roster`
  (`utils/prompt_compositions.py`); the JS mirror is a display aid. An entry that matches nothing is
  skipped with a warning.
- **Timeline / preview**: shots become segments via `shotsToSegments`
  (`js/utils/composition_timeline.js`): proportional bands when every shot has a strictly increasing
  timestamp, equal-width bands otherwise. The selected shot id is stored in the existing `clip_id`
  widget (harmless: only the Source Profile paths read it). There is no duration multiplier. The
  preview only substitutes `{A}`/`{B}`/... with bundle names; it is **not** the assembled prompt
  (punctuation, pacing phrases, dialogue formatting and `<Subject N>` labels come from the assembler
  at run time).

## A stale preview when switching compositions quickly (fixed)

`_refreshCompositionSubjects()` does two sequential `fetch` calls (list, then get) every time it
runs. With no guard, switching the connected Composition Load node's dropdown again before the
first refresh finished could let the OLDER request's response land after the newer one and
overwrite it — the preview then showed the previous composition's shots, one step behind, worst
the faster you switched. Fixed with a per-refresh sequence number: a response is only applied if no
newer refresh has started since. The same fix was applied to `_refreshSourceSubjects()` (same
shape, single fetch, smaller window but the identical bug).

## Composition options: Background section

While a composition drives the node, a **Background** section appears above the cast tabs:

- a dropdown whose first option is `Default: <the composition's background>`, then `(none)` and every
  other background;
- an **image** checkbox: use the background's reference image(s) as a subject the scene is set in
  (the composition's `background_as_reference` is its default; disabled when the background has no
  images);
- a **soundscape** checkbox: use the background's soundscape as the overall soundscape, replacing the
  composition's own (off by default; disabled when the background has no soundscape).

The choices are stored as a small JSON object in the hidden `composition_overrides_json` input, and only
keys that differ from the composition are present (`background`: id or `"none"`, `background_as_reference`:
bool, `background_soundscape`: true), so an empty object means "use the composition as saved". They travel
on the cast dict as `scene_cast["composition_overrides"]` and are applied by `PromptCompositionLoader` right
after the composition is loaded (`apply_composition_overrides` in `utils/prompt_compositions.py`), before the
background is resolved and the prompt assembled. An unknown background id logs a warning and keeps the
composition's own. The section and the overrides are ignored in Source Profile mode. Run History shows the
effective `Background` / `Background as reference` rows (and `Soundscape` when taken from the background),
marked `(override)`.

Without the soundscape checkbox the assembler's rule applies: the composition's Overall Soundscape wins, and
the background's soundscape is only a fallback when the composition's is empty.

## The per-entry Dialogue field is Source-Profile-only

Only `SourceProfileClipPrompt` reads a cast entry's `dialogue`: it matches the entry to a source
subject and uses the text as that slot's line in the clip's single shot. `PromptCompositionLoader`
never reads it, and composition shots already carry their own per-shot dialogue (speaker + text). So
in composition mode the field would be silently ignored, and it does **not** override the shot's
dialogue.

Decision: the field is **hidden** while a composition drives the node (the stored value is left
untouched if you switch back to a Source Profile). If a per-subject line is ever wanted for
compositions ("this subject's line in any shot where they are the speaker", as a default or an
override) it needs a design first: composition dialogue is per shot with a speaker slot, cast
dialogue is per subject, and it must cooperate with libber tokens and the `[silent]` / `[sounds]`
markers.

## Audio references without scripted dialogue (avoiding invented chatter)

`utils/prompt_assembler.py::_assemble_h3_ref2va` auto-detects, per subject slot, whether that slot
is ever the speaker of a resolved shot dialogue line anywhere in the composition. A subject with a
voice/audio reference (bundle audio, standalone or a video's soundtrack) but no scripted line
anywhere gets softer default wording in both `subject_definitions` and `retention_analysis`: it
drops the "a spoken … vocal layer" content descriptor and, in `retention_analysis`, the "and
measured delivery" clause — both read as speech-cadence guidance and were a plausible driver of
the model inventing dialogue for a subject that was only meant to keep a consistent voice. The
"without copying the original signal" clause is kept either way; it is about not reusing the
recording verbatim and is unrelated to whether the subject speaks.

A subject that *does* get a scripted line anywhere keeps the original wording unchanged. An
explicit audio "role" override (set on the bundle) always takes precedence over the auto-detected
wording in `subject_definitions`; `retention_analysis` does not yet read the role override (a known
gap — it always uses the auto-detected/default wording there).

This only softens the *speech-implying* framing; it does not silence the reference. A shot's
`sound_events` field (non-verbal sound — moaning, gasping, etc.) is unrelated machinery, generated
independently of the audio reference and not explicitly tied to `<Audio N>` in the prompt text.

## Libber tokens in composition dialogue

Composition shot text can contain libber tokens (`%*:N%`, `%*%`, `%key:N%`, `%key%`). They are
resolved only when `PromptCompositionLoader` runs (`_apply_composition_libbers`, unseeded random
draws), so the node's preview shows them raw. `%*.N%` is accepted as a spelling of `%*:N%`. A seeded,
preview-accurate scheme has been designed but not built.

## Related: PromptCompositionLoader

It takes the same composition through an optional `prompt_composition` input, which then overrides
its own Composition dropdown (greyed out in the UI), so one selector can drive both nodes. Its
outputs are Prompt, Composition Name, Filename Prefix, LoRA Stack and H3 Ref Plan; everything else it
used to expose is shown in Run History (`utils/composition_track_summary.py`).
