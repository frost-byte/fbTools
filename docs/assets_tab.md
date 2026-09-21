# Panel tabs: Compose and Assets

The fbTools sidebar panel (`js/ui/fbt_panel.js`, `TABS`) hosts the editors. Two of them share data and follow
the same **list first, then editor** pattern as the Sources tab.

## Compose (`js/ui/composition_editor.js`)
- Opens on a searchable list of saved compositions (cards show model type, subject/shot counts, background,
  last edited; the counts come from `list_compositions` in `utils/prompt_compositions.py`).
- Clicking a card or **+ New** opens the full-width editor with a **← Back** button (asks before discarding
  unsaved edits). Save, Preview Raw, Copy and Send to Workflow are in the editor's action bar.
- There is no sidebar. Subjects are added from the **+ Add subject...** picker in the Subjects section, and each
  shot has **Preset...** pickers beside its Camera and Sound fields. Creating or editing those assets happens in
  the Assets tab.

## Assets (`js/ui/bundle_editor.js`, tab id `assets`)
One tab with sub-tabs: **Bundles**, **Subjects**, **Backgrounds**, **Camera**, **Sound**, **Outfits**. Each
sub-tab keeps its own search text. Backgrounds and outfits open the same modal editors Compose used to host
(`background_editor.js`, `outfit_editor.js`); camera/sound presets use a small modal in `asset_lists.js`.
Background and outfit cards show a thumbnail of their first still-image reference.

## Shared data and refresh
- `js/ui/library_store.js` holds the shared library state (`lib`: subjects, backgrounds, outfits, presets, media
  lists, LLM flags, SAM2 status). `asset_lists.js::loadLibrary()` fills it; Compose fills it on open.
- After a save or delete, an editor calls `notifyLibraryChanged(kind, extra)`, which fires the
  `fbt:library-changed` DOM event. Views listen and redraw themselves (Compose refreshes its background dropdown,
  slot pickers and shot preset pickers; the Assets lists redraw). Never mutate another view's DOM directly.
- Small DOM helpers shared by the editors live in `js/ui/library_common.js`.

## Adding another asset kind
1. Add a store field in `library_store.js` and load it in `loadLibrary()` and Compose's `_loadResources`.
2. Add an entry to `ASSET_TABS` and an `_items(kind)` branch in `asset_lists.js` (rows: name, summary, thumb, open,
   remove) plus a modal editor.
3. Call `notifyLibraryChanged(kind)` after saves/deletes and handle the kind in Compose's `_onLibraryChanged` if
   the editor uses it.
