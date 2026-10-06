# Design note: dropdown inputs whose options come from ComfyUI

Status: **proposal, not implemented** (2026-10-06). Written so it can be adapted into an upstream
`OpenShot/openshot-qt` issue/PR description: nothing below is specific to fbTools, Reference
Bundles, or H3.

## Problem

An OpenShot ComfyUI template can declare extra inputs (`extra_inputs`) that the Generate dialog
renders as widgets and substitutes into the workflow. The `choice` type is a dropdown, but its
options are a **static list written in the template**. Plenty of useful dropdowns have options that
only the running ComfyUI server knows: model/LoRA/checkpoint names, sampler names, or the ids of
some custom-node-provided collection.

The fork's workaround for one such case (Reference Bundle ids) was a dedicated `bundle` input type:
a new type name accepted by the template loader, a bundle-specific branch in the dialog that calls
a bundle-specific client, and a place on the service's list of string-valued types. Every new source
of options would need the same four edits. That is a maintenance burden for the fork and not
something an upstream project would want to adopt as-is.

**Goal:** let a template say "this dropdown's options come from the server" in a generic way, so no
OpenShot change is needed per use case.

## What exists today (verified 2026-10-06)

OpenShot (`feature/scene-cast-generation`, which contains upstream PR #6158's generalization):

- `classes/comfy_templates.py`: `EXTRA_INPUT_TYPES` is `{image, video, audio, text, choice}` upstream
  (the fork adds `bundle`). A `choice` entry **requires** a non-empty `choices` list or the entry is
  rejected.
- `windows/generate.py`: builds a widget per type; `choice` becomes a `QComboBox` from the entry's
  `choices`; the fork's `bundle` fills a combo through an fbTools client.
- `classes/generation_service.py`: string-valued types (`text`, `choice`) are collected into the
  substitution map; media types are resolved from Project Files ids to paths. (The fork's `bundle`
  initially fell through to the media branch and was never substituted. The service now carries all
  string-valued types through `_collect_text_extra_inputs`, and substitutes an empty string for a
  blank optional input so a literal `__openshot_input:<key>__` can never reach ComfyUI.)
- `classes/comfy_client.py`: already fetches `/object_info/<node>` for a few hard-coded nodes
  (checkpoints, upscale models, CLIP, RIFE) and has `_extract_combo_options()`, which handles all
  three shapes ComfyUI uses for a combo input (`[[...], {...}]`, `["COMBO", {"options": [...]}]`,
  and a bare list).

ComfyUI:

- `GET /object_info/<class>` returns each input's config, including the full option list for a
  combo input. For V3 nodes it calls `define_schema()` **on every request**
  (`GET_NODE_INFO_V1` -> `GET_SCHEMA` -> `FINALIZE_SCHEMA` -> `define_schema`; `server.py`
  `node_info`), so a list computed inside `define_schema()` (for example "all bundle ids on disk")
  is current each time, not frozen at startup. V1 nodes call `INPUT_TYPES()` per request too,
  although authors may still compute lists at import time.
- Prompt validation checks a submitted combo value against that same list, which is what produced
  `Value not in list: bundle_id: '__openshot_input:bundle_id__' not in (list of length 36)`.

## Proposal

Extend `choice`; add no new type.

### 1. `options_from`: take the options from a node in the workflow

A template can already name the node that consumes an input, because the placeholder
`__openshot_input:<key>__` sits in that node's input. Declare the link explicitly:

```json
{ "key": "bundle_id", "type": "choice", "label": "Voice Reference", "required": false,
  "options_from": { "node": "bundle_audio_load", "input": "bundle_id" } }
```

At dialog open OpenShot:

1. looks up that node's `class_type` in the template's workflow,
2. requests `/object_info/<class_type>` from the configured ComfyUI server,
3. reads the named input's options with the existing `_extract_combo_options()`, and
4. fills the combo (value = option, label = option).

Properties that make this safe and generic:

- **No new endpoint, client, or server code.** Works for any combo in any node pack.
- **The offered values are exactly the ones ComfyUI will accept**, since validation uses the same
  list. This removes the failure class "UI offered something the node rejects".
- Falls back cleanly: if the node/input is missing, is not a combo, or the request fails, use the
  entry's static `choices` if present; otherwise show an empty list with a note and do not block the
  dialog.
- **`choices` is the fallback, `options_from` wins when resolvable.** An older OpenShot build that
  does not know `options_from` still loads the template and shows the static list (for example
  `["" ]`), instead of rejecting it.

### 2. Inference (optional, later)

If a `choice` entry has no `choices` and no `options_from`, find where its placeholder is used. When
it is the **entire value** of exactly one node input, and that input is a combo according to
`/object_info`, use it as if declared. Ambiguous cases (placeholder in several nodes, or embedded in
a longer string) are not inferred and fall back to the normal "needs `choices`" rule. This would make
the declaration unnecessary in the common case, at the cost of some implicitness; it can be left out
if explicit is preferred.

### 3. `options_url` (only if needed)

A combo only carries raw values. If a dropdown needs display names or tooltips distinct from the
value (a bundle's friendly name versus its id), the template can instead point at a JSON endpoint on
the **same server**:

```json
"options_url": { "path": "/fbtools/bundles/list", "items": "bundles", "value": "id", "label": "name" }
```

Restrictions: relative path only (no host), GET only, short timeout, bounded item count and label
length. The cost is that nothing guarantees the endpoint's values match what the consuming node
accepts, so this is the second step, not the first.

## Dialog and service behavior

- **Fetch once per dialog open**, using the existing short timeout (8 s). A failed fetch must not
  block the dialog or the job; an optional input just offers "(none)".
- **Optional input:** include a "(none)" row that submits `""`. This only validates if the node's
  combo lists `""`; if it does not, omit the row (the dialog can check the fetched options).
- **Required input:** no "(none)" row; the dialog's existing required-field check applies.
- **Service:** unchanged beyond what is already done: all non-media types are string-valued, and
  blank optional values substitute `""`. A more robust classification is to treat only `image`,
  `video`, and `audio` as media and everything else as a string, so a future widget type cannot fall
  into the file branch by default.
- **Large lists** (checkpoints can be hundreds): cap the count and keep the combo searchable, or
  decide a limit up front.

## What this would remove from the fork

- The `bundle` entry in `EXTRA_INPUT_TYPES`, the `bundle` branches in `generate.py`
  (`_populate_bundle_combo`, the value collector, the required-field message), and the fork's
  dependency on its fbTools client for this picker.
- Keep `bundle` as a thin alias to `choice` + `options_from` until the templates are migrated.

What stays fork-specific, and is a separate matter: the Scene Cast widget (`group: scene_cast`) and
its builder dialog.

## Security and robustness

- Requests go only to the configured ComfyUI server, GET only, with timeouts and size limits.
- Option values are strings and pass through the same placeholder substitution as typed text, so no
  new injection surface beyond what a free-text input already has.
- Templates are user-installed JSON, but treat labels as untrusted display text (plain text, no
  markup).

## Open questions

- **Display names:** is raw-value-only acceptable for `options_from`, or does upstream want labels
  from day one (which pushes `options_url` earlier)?
- **Threading:** the existing client calls are synchronous. Acceptable for a single fetch at dialog
  open, or should it be asynchronous with a "loading..." state?
- **Naming:** `options_from` versus `source`, and whether `options_url` belongs in the same PR.
- **Dependent dropdowns** (options that depend on another input) and **multi-select** are out of
  scope here.
- **Inference:** on by default, opt-in, or omitted.

## Suggested staging

1. Service: classify by carrying mechanism (media versus string); blank optional inputs substitute
   `""`. Small; the second half is already in the fork.
2. `options_from` with a static-`choices` fallback, plus tests (fetch success, node missing, input
   not a combo, request failure, optional "(none)" handling).
3. Inference (if wanted).
4. `options_url` (only if display names are needed).

Steps 1-2 are generic and upstream-friendly. Nothing in them mentions bundles, fbTools, or H3.

## Verification still needed before building

- Run `options_from` against a real V3 node and a real V1 node (the fbTools bundle node and
  something like `CheckpointLoaderSimple`) and confirm the option lists match what validation accepts.
- Confirm the `""` option behaves as intended through a full job, not just in a unit test.
- Check how the dialog behaves with a very large list.
