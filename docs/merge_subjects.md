# Merging subjects (scripts/merge_subjects.py)

A subject is mostly a thin identity (name, pronoun style, short name); the detail that shapes a prompt
lives in its **bundles** (appearance, media, audio), and a bundle's appearance, pronoun style and short
name override the subject's. So subjects that are really the same identity in different looks can be
folded into one subject, letting a composition slot use any of those bundles.

```
python scripts/merge_subjects.py DATA_DIR --into TARGET SOURCE [SOURCE ...] \
    [--primary SUBJECT] [--workflows DIR ...] [--delete-sources] [--dry-run]
```

`DATA_DIR` is the fbTools user-data directory (the one containing `subject_profiles.json`).

**Sources**
- `SUBJECT` - the whole subject. Its bundles move to `TARGET`, and every reference is rewritten:
  composition slots (and their saved snapshots), scene-cast entries, and the cast entries saved inside
  workflow files found under each `--workflows` directory.
- `SUBJECT:BUNDLE` - just that bundle moves to `TARGET`; the source subject and its references stay.

**Which identity wins.** The merged profile starts from `TARGET` if it already exists, otherwise from
`--primary` (default: the first whole subject listed). Empty fields, including empty keys inside
`appearance` and `voice`, are filled from the other sources in order; non-empty values are never
overwritten. Reference sheet images are the de-duplicated union.

**Safety.** `--dry-run` reports counts and writes nothing. Before a file is changed its original is copied
to `<name>.pre-subject-merge` (kept if one exists). The run is idempotent, so repeating it changes nothing.
Old subjects are kept unless `--delete-sources` is given.

**Why the workflows matter.** Cast entries bind to a composition slot by subject id. An entry still naming
a merged-away id silently stops applying its bundle, so pass `--workflows` for every directory holding
saved workflows (and re-pick any entries in unsaved graphs).

The logic lives in `utils/subject_merge.py` (pure, tested in `tests/test_subject_merge.py`); the script only
does the file I/O.
