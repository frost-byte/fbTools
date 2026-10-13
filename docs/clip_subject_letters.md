# Clip subject letters: subject order as configuration

A Source Profile clip refers to its subjects with letters in its text: `{A} hugs {B}`. This page
describes how a letter gets its meaning, what can silently change it, and how the tooling keeps the
text and the configuration in step. The same pattern applies anywhere a list's order is used as
configuration, so the reasoning is written to be reusable.

## The rule

> **Among the subjects ticked in a clip, the one the Source Profile lists first is `{A}`, the next
> `{B}`, and so on (up to `{J}`).**

The profile's `subjects` list is the single ordering. Every clip shows that same list, with the
ticked subjects lettered consecutively from the top. The order in which subjects were ticked is
irrelevant, and the order of the ids inside a clip's own `subjects` array is irrelevant too: it is a
set. The same subject can be `{A}` in one clip and `{C}` in another, because the letters count only
the ticked subjects.

Example: a profile lists `girl, boy, dog, cat`. A clip with `dog` and `girl` ticked (in either order)
reads `{A}` = girl, `{B}` = dog. Ticking `boy` makes it girl `{A}`, boy `{B}`, dog `{C}`.

Why order and not a stored letter: a clip only needs to name the subjects *it* uses, with no global
letter allocation, and the one visible list is the whole configuration. The price is that the
letters are implicit. A position is not visible in the data, so anything that changes which subjects
are ticked, or the profile's list, changes what the text means.

## Everything that reads the order must agree

| Consumer | Where |
|---|---|
| Prompt assembly (Clip Prompt node): source slots A–J, `<Subject N>` numbering, bundle replacement pairs | `nodes/source_profiles.py` via `clip_subjects_in_order` |
| Ordinal cast matching ("the Nth subject with my pronoun") | `utils/source_profiles.py::resolve_ordinal_subject` |
| Editor letter labels next to the tick boxes | `js/ui/source_profile_editor.js` via `clipSlotLetters` |
| Describe (the vision model is told which letter is which subject) | same file, `clipSubjectsInOrder` |
| Scene Cast Build action preview and its ordinal preview | `js/nodes/scene_cast_build.js` via `clipSubjectsInOrder` |

Python: `clip_subjects_in_order(profile, clip)`. JavaScript: `clipSubjectsInOrder`, `clipSlotLetters`
(`js/utils/clip_subjects.js`). Never index into `clip["subjects"]` directly to get a letter: that is
the order the ids were stored in, which is not the rule. (This split, with some sites using the
stored order and others the profile order, is exactly the bug of September/October 2026. See
`docs/GOTCHAS.md`.)

Ids that no longer exist in the profile are skipped, repeated ids are ignored, and only the first ten
subjects get a letter. A clip with no subjects ticked falls back to all of the profile's subjects.

## What changes a clip's letters

Letters are ranks, so an edit that changes a rank re-points the text without touching it.

| Edit | Effect |
|---|---|
| Tick a subject | Subjects listed after it in the profile move down one letter. Ticking the last-listed subject moves nothing. |
| Untick a subject | Subjects listed after it move up one letter. |
| Untick, then tick again | Back to the same letter (the click order does not matter). |
| "→ all" / "✕ all" | The tick or untick effect in every clip that lacks or has the subject. |
| Delete a subject from the profile | The untick effect in every clip that had it. |
| Merge with next | The union of both clips' subjects; each clip's text was written against its own letters. |
| Split | Both halves copy the list. No change. |
| Apply suggestions (auto-segment) | New clips get the whole profile's subject list. |
| Reorder the profile's subject list | Moves letters in every clip. (There is no reorder control today; deleting and re-adding is the only way.) |
| Rename a subject, change its pronoun, add a subject to the profile | Adding appends at the end of the list, so no letter moves until it is ticked. |
| Re-import of a story | Existing clips keep their list. No change. |

## Keeping text and order together

When a change would move a letter that the clip's text uses, the editor asks instead of re-pointing
the text silently:

- **Tick or untick** a subject: a dialog shows the new letters and the clip's text before and after,
  with a checkbox to rewrite the letters so each subject keeps its meaning.
- **"→ all", "✕ all" and deleting a subject from the profile**: a confirm dialog lists the affected
  clips; you choose which get their text rewritten.
- **Merge**: both clips' text is rewritten into the merged letters before the two are joined.

"Rewrite" means every `{X}` in the clip's `action`, `style` (preamble), `overall_soundscape` and
`non_diegetic_music` is replaced in a single pass (a swap `{A}`↔`{B}` does not undo itself) so each
subject keeps the letter the text used. A subject that was removed is written out by its label. The
helpers are pure and tested: `letterMoves`, `remapLetters`, `setClipSubjects` in
`js/utils/clip_subjects.js`.

## Ordinal matching depends on the same order

A cast entry in "ordinal" mode means "the Nth subject in this clip that shares my bundle's pronoun
style". It is counted in the same order, so a change to the order also changes which subject "the 2nd
feminine" is. Two further details:

- A subject without an explicit `pronoun_style` resolves to **neutral** (people, animals) or to
  `object` / `location` by entity type. A feminine or masculine bundle never matches a neutral
  subject. The entry then silently becomes bundle-only. Set the pronoun on the source subjects that
  need to be matched.
- An explicit source-subject entry does not depend on pronouns or on order.

## History worth knowing

Until 2026-08-29 the editor showed no letters; from then until 2026-09-07 it lettered subjects in the
order they were ticked (the clip array order). The "refactor(styles)" commit `4444e0f` silently
changed the editor's labels, Describe and the Scene Cast Build preview to profile order while the node
kept using the stored array order, so a clip could show one mapping and assemble another. On
2026-10-12 the rule was settled as profile order and every consumer made to follow it. Clips written
in the click-order era may have text whose letters assume the old order: check them when their letters
do not match their text.

## Guidance

For authors: the profile's subject list is the lettering. Put the subjects you want lettered first at
the top of the list. Expect a prompt when an edit would shift letters your text uses.

For developers: read letters only through the helpers above; if you add a new place that shows or
uses letters, add it to the consumer table. When you write a migration, fix the *text* (or the
profile list) to match what existing text means, and keep a backup of `source_profiles.json` first
(the registry also writes a `.bak` on every save).
