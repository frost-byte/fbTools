# ComfyUI Frontend Localization (i18n) — What It Covers and What It Doesn't

Reference notes from investigating why our two selection-toolbox tooltips (Extract Node as
JSON / Send Get/Set Nodes to Back) showed no hover text at all, then why they looked visually
inconsistent with ComfyUI's own buttons once we patched around it with a native `title`
attribute. The real fix was registering a locale file — this doc is the wider picture that came
out of that investigation, kept as a reference for if we ever want to localize more of fbTools.
Not required reading; nothing in the extension depends on this today.

## The short version

ComfyUI has a real, working per-custom-node localization system covering **commands, node
definitions (including widgets/inputs/outputs), and custom settings**. It does **not** cover
anything a custom node renders itself inside a `type: "custom"` sidebar tab (like fbTools' whole
panel) — that's plain DOM/JS we own, invisible to ComfyUI's i18n layer.

## How it's wired (backend)

`app/custom_node_manager.py::CustomNodeManager.build_translations()` (in the ComfyUI core
codebase, not this repo) scans every installed custom node for a `locales/` folder:

```
custom_nodes/
  <your_custom_node>/
    locales/
      en/
        main.json
        nodeDefs.json
        commands.json
        settings.json
```

Each `<file>.json`'s content is merged into the locale tree under a key matching the filename
(`commands.json`'s content becomes `{"commands": {...}}`, etc.), all merged across every
installed custom node, and served at `GET /i18n`. The frontend fetches that once at startup and
merges it into its message catalog alongside ComfyUI's own bundled translations.

**Caching gotcha**: `build_translations()` is `@lru_cache(maxsize=1)` — a plain browser refresh
will NOT pick up a newly added/edited locale file. A full ComfyUI backend restart is required.

## What each file covers

### `commands.json` — command labels/tooltips

Keyed by the command id with every `.` replaced by `_` (confirmed live against ComfyUI's own 118
built-in commands, e.g. id `Comfy.3DViewer.Open3DViewer` → key `Comfy_3DViewer_Open3DViewer`):

```json
{
  "fb_tools_extract-node-json": { "label": "Extract Node as JSON — ..." }
}
```

This is what `js/fb_tools.js`'s `commands:` array entries need — see `locales/en/commands.json`
in this repo for the real, live example. Without it, ComfyUI's `ExtensionCommandButton` (the
component that renders extension commands in the node selection toolbox) resolves its
aria-label/tooltip via `$t('commands.<id>.label', '')`, which silently falls back to an empty
string — the button still works, it just shows no hover text and no accessible name.

### `nodeDefs.json` — node display names, descriptions, and per-slot text

Keyed by node type name, with dotted field paths. Because ComfyUI's V3 schema (`io.ComfyNode`)
unifies widgets and sockets under one `inputs` list, **widget labels and tooltips live in the
same `inputs.<name>.*` namespace as socket inputs** — there's no separate "widgets" file or key:

```json
{
  "fbt_SceneSelect": {
    "display_name": "Scene Select",
    "description": "...",
    "inputs": {
      "scene_dir": { "name": "Scene Directory", "tooltip": "..." }
    },
    "outputs": {
      "0": { "name": "Scene Info" }
    }
  }
}
```

### `settings.json` — custom setting labels/tooltips

For any setting a custom node registers via `registerExtension({ settings: [...] })`, keyed by
setting id, same shape idea as commands.

### `main.json` — everything else, including category labels

A catch-all merged at the locale root. Notably includes `nodeCategories` and
`settingsCategories`, which relabel the path *segments* shown in the node library tree / settings
panel search — e.g. this could rename how `"🧊 frost-byte/Scene"` displays per-locale without
touching the actual `category=` string declared in Python. (Seen in the wild in
`comfyui-easy-use`'s `locales/en/main.json`, which only uses these two keys.)

## The sidebar icon caveat

The tooltip/title on `app.extensionManager.registerSidebarTab({ tooltip, title, ... })` **does**
pass through `$t()` — but keyed by the literal string itself, with itself as the fallback
(confirmed in the frontend's `SidebarIcon` component: `t = tooltip || label; $t(t, t)`). That's
a self-referential key, not a symbolic id like commands get. It only actually translates if a
locale file happens to define a message keyed by that *exact* English string — otherwise it just
silently displays the literal text we passed, which is indistinguishable from "not localized at
all" unless you go looking for it. Much more fragile than the commands/nodeDefs/settings
mechanisms, and easy to assume works the same way when it doesn't.

## What's entirely outside this system: our own panel content

fbTools' sidebar entry is registered with `type: "custom"` and a raw `render(el)` callback
(`js/ui/fbt_panel.js`). Everything inside that `el` — the COMPOSE/ASSETS/CASTS/.../INSPECT tab
strip, every button label, every list item, all of `js/ui/node_inspector.js`'s "NODE INSPECTOR" /
"Collapse all" / "Expand all" text — is plain DOM built with `document.createElement` +
`textContent`. ComfyUI's i18n system has no visibility into or hook for that subtree at all.

To localize any of it ourselves, we'd have to reach into Vue's i18n composer directly and call
its translate function for every string we render — and that's not a documented/supported
extension API. The only way we found to reach it at all was through Vue app internals:

```js
document.getElementById("vue-app").__vue_app__.config.globalProperties.$t(key, fallback)
```

That's poking at ComfyUI's internal Vue instance, not a stable public surface — fine for one-off
live investigation in devtools, not something to build a real feature on without ComfyUI
exposing it properly first.

## If we ever actually want to localize fbTools

Worth doing, low effort, real payoff:
- `locales/en/commands.json` (already added, real example in this repo)
- `locales/en/nodeDefs.json` for the ~70+ registered node types' display names/descriptions/widget
  tooltips — mechanical to generate from `extension.py`'s existing docstrings/labels, but no
  small amount of content to write for every node.

Not really worth doing without ComfyUI adding a supported hook:
- Anything inside our own custom sidebar panel's DOM. We'd be re-implementing i18n string lookup
  ourselves (a small wrapper around `$t` reached via the same internal path above, or just a
  plain JS dict keyed by our own string constants) rather than plugging into anything ComfyUI
  maintains for us — real work, and it'd need to be re-verified against internal Vue app
  structure on every ComfyUI frontend upgrade since we're relying on an undocumented path to
  reach it.

## Where this was verified

- Backend: `app/custom_node_manager.py` (ComfyUI core repo, not this one) — `build_translations()`
  and its `/i18n` route.
- Frontend: `comfyui_frontend_package`'s bundled `i18n-*.js` chunk (`translateNodeDefText`,
  `resolveNodeDefPath`, `resolveNodeDefSlotText`) and `GraphView-*.js` (`ExtensionCommandButton`,
  `SidebarIcon`) — both minified but not property-mangled, so real function/method names are
  still grep-able. Exact chunk hashes will drift across ComfyUI frontend versions; re-grep rather
  than trusting these filenames to stay put.
- Live: confirmed the `commands.<id>.label` key transform against all 118 of ComfyUI's own
  built-in command translations via `$tm('commands')` in a live browser console, and confirmed
  `GET /i18n`'s `@lru_cache` behavior directly (empty `commands` catalog even after
  `locales/en/commands.json` existed on disk, pre-restart).
