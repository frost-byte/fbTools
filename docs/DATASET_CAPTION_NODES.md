# Dataset Caption Nodes

This guide covers the dataset captioning nodes and API endpoints in `comfyui-fbTools`.

## Node Overview

- `Dataset Captioner`: generates one caption `.txt` per image.
- `Dataset Caption Editor`: batch text edits across caption files.
- `Dataset Caption Viewer`: in-node table for review/edit/re-caption.
- `Dataset Export Summary`: reports coverage and word-length stats.
- `Caption Model Unloader`: frees cached caption model VRAM.

## Path Semantics

Path behavior follows Comfy conventions:

- `Dataset Captioner.input_directory` and viewer/export `dataset_path` are resolved relative to Comfy `input/` when not absolute.
- `output_directory`/`caption_directory` values are resolved relative to Comfy `output/` when not absolute.
- `Dataset Caption Editor.dataset_path` (and `/fbtools/dataset_caption/edit` `dataset_path`) resolve relative to Comfy `output/` when not absolute.

Absolute paths are also supported.

## Viewer Behavior

`Dataset Caption Viewer` uses an internally scrollable table. The viewport is fixed-height for stability and to avoid layout feedback loops with LiteGraph DOM widget sizing.

## REST API Endpoints

All routes are mounted under `/fbtools/dataset_caption`:

- `GET /image?path=<abs-image-path>`: serves image thumbnails.
- `GET /list`: paginated rows (`path`, `output_dir`, `page`, `page_size`, `recursive`).
- `POST /save`: save one caption (`txt_path`, `caption`).
- `POST /recaption`: regenerate one caption for a specific image.
- `POST /edit`: batch-edit caption text.

### `/edit` Request Body

```json
{
  "dataset_path": "rara",
  "find_text": "old phrase",
  "replace_text": "new phrase",
  "recursive": false,
  "dry_run": true
}
```

Notes:
- `find_text` is the expected key for search text.
- `dry_run` defaults to `true` if omitted.

## Fish Helper Script

Use `scripts/dataset_caption_edit.fish` for repeatable multi-pass edits.

```fish
# Dry-run (default)
fish scripts/dataset_caption_edit.fish --dataset rara \
  --pass 'old phrase=>new phrase' \
  --pass 'foo=>bar'

# Apply changes
fish scripts/dataset_caption_edit.fish --dataset rara --apply \
  --pass 'old phrase=>new phrase' \
  --pass 'foo=>bar'
```

Script options:
- `--dataset <path>`: caption dataset path.
- `--output <path>`: caption directory override.
- `--server <url>`: Comfy server URL.
- `--apply`: write changes.
- `--dry-run`: force dry-run mode.
- `--pass 'SEARCH=>REPLACE'`: one replacement pass (repeatable).

## Common Troubleshooting

- `edited_count=0` for all passes:
  - Verify phrase exists in target `.txt` files.
  - Confirm you are targeting the correct `dataset_path`/`--output` directory.
  - Confirm request uses `find_text` (not `search_text`).
- Viewer loads but overflows:
  - Ensure frontend is refreshed after updates.
  - Current viewer intentionally uses fixed viewport height with internal scrolling.
