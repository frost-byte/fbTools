# Importing Scenes and Stories as Source Profiles

The legacy Scene/Story system (`output/scenes/<scene>/`, `output/stories/<story>/story.json`) maps onto
an **image-driven Source Profile**: a profile with `media_type: image` whose media is a set of
**frames**, where each **clip** selects one frame. Use **Import…** in the Source Profiles editor
(list view), or the REST endpoints below.

## Convention

| Legacy | Source Profile |
|---|---|
| Story | one profile, id `sp_story_<story>`, `origin.story` set |
| Scenes with no story | one profile you name (`profile_name` required), id `sp_story_<name>` |
| `scene_order` | clip order |
| `base.png` (also `.jpg/.jpeg/.webp`) | a profile **frame** (`dir: output`, `scenes/<scene>/base.png`, referenced in place, never copied) |
| scene mask (`mask_name`, minus `_no_bg`) | a profile **subject** (`subj_<name>`), tagged on the clip |
| resolved video prompt | the clip's **action** |
| `loras.json`, depth/pose/canny/mask images | not migrated; they stay in the scene folder |

The scene image is the clip's `<Picture 1>`: a layout reference (pose, orientation, position), not a
guide frame. Cast bundles replace the scene's subjects at prompt time (Scene Cast Build).

### Where the action comes from

The same resolution `StoryVideoBatch` uses (`utils/scene_prompts.py`): libber substitution, then
compositions, then the story entry's `video_prompt_source`. `auto` reuses the **image** prompt, which is
usually an image-edit instruction rather than an action, so the clip is marked for review.

`origin.action_source` records what the action is:

| value | meaning |
|---|---|
| `resolved` | an explicit video prompt (custom / prompt / composition) |
| `image_prompt` | `auto` fell back to the image prompt; a draft, review it |
| `empty` | the scene had no prompt; use **Describe empty clips** (vision LLM) |
| `manual` | typed or described in the editor; a re-import never overwrites it |

### Image-model instructions are removed

Scene prompts were written for image-edit models that take numbered input images. A Source Profile
clip states that structurally (the frame is `<Picture 1>`, the cast comes from the bundles), so
`clean_image_model_prompt` (`utils/scene_story_import.py`) strips it from every imported action:

- notes about control images (`image3 provides no meaningful information and should be ignored`,
  `Picture 3 provides further depth ... but should not alter ...`);
- `Replace the masked area from image1 with the {male|female|girl} from image2.` and
  `Place <trigger>, the girl from Picture 2, in the position ... of the girl from Picture 1, ...`;
- `<trigger>, the girl from image2,` (and later bare uses of the trigger word, or `the girl from image2`)
  becomes `{A}`: the clip's first subject.

Only the second input image maps to `{A}`. Any other `imageN`/`Picture N` still in the text raises an
`image_reference` warning in the preview, and a prompt that was nothing but scaffolding becomes `empty`.
A leading `A photo-realistic scene.` is a style line, so it moves to the clip's preamble (`style`, which
opens detailed_description before Shot 1) instead of staying in the action. A re-import only fills the
preamble when the clip has none. Other prompt prose is left as written.

### Re-importing

Idempotent. Clips match on `origin.scene`; the import refreshes the frame and (unless the action is
`manual`, or non-empty with no recorded source) the action. Clips, subjects and frames you added
yourself are left alone. `source_profiles.json` is backed up (`.bak`) on every write.

## REST

- `GET  /fbtools/source_profiles/importable` -> `{stories, scenes}`
- `POST /fbtools/source_profiles/import_preview` -> dry run: `{plan: {profile, scenes, warnings, stats}}`
- `POST /fbtools/source_profiles/import_apply` -> writes; same body

Body: `{story}` or `{scenes: [...], profile_name}`, plus optional `profile_id`, `clip_length` (default 5 s).
The story is located by name under the stories folder; the stale `story_dir` inside `story.json` is ignored.

## Code

`utils/scene_story_import.py` (planner + disk readers, pure), `utils/scene_prompts.py` (prompt
resolution, shared with the Story nodes), routes in `nodes/source_profiles.py`, UI in
`js/ui/source_profile_import.js` and the Frames list / clip frame selector in
`js/ui/source_profile_editor.js`.
