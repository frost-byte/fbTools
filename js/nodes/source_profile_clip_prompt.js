/**
 * SourceProfileClipPrompt — dynamic clip_id selector.
 *
 * The clip_id STRING widget is hidden and replaced with a <select> that
 * fetches the connected Source Profile's clip list on each connection change.
 * The selected clip ID is written back to the hidden widget so it round-trips
 * through workflow serialization correctly.
 */

import { setWidgetVisible } from "../utils/widgets.js";
import { LS_H3_MAX } from "../ui/settings_panel.js";

export function setupSourceProfileClipPrompt(nodeType, _nodeData, app) {
    const _origCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        _origCreated?.call(this);
        _buildClipIdWidget(this, app);
        // Pre-fill max_clip_frames from the global Settings value (stored in localStorage).
        // This only runs on fresh node creation, not on workflow restore (onConfigure).
        const maxWidget = this.widgets?.find(w => w.name === "max_clip_frames");
        if (maxWidget) {
            try {
                const stored = localStorage.getItem(LS_H3_MAX);
                if (stored !== null) maxWidget.value = parseInt(stored, 10) || 360;
            } catch {}
        }
    };

    // After workflow load, widget values are restored before onConfigure fires —
    // re-populate the dropdown so the saved clip_id can be re-selected.
    const _origConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (config) {
        _origConfigure?.call(this, config);
        requestAnimationFrame(() => this._refreshClipDropdown?.());
    };

    // Refresh whenever a relevant input is connected/disconnected.
    const _origConnChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (type, index, connected, linkInfo) {
        _origConnChange?.call(this, type, index, connected, linkInfo);
        if (type === LiteGraph?.INPUT) {
            requestAnimationFrame(() => this._refreshClipDropdown?.());
        }
    };
}

function _buildClipIdWidget(node, app) {
    // Find and hide the backing STRING widget — it still serializes the value.
    const clipWidget = node.widgets?.find(w => w.name === "clip_id");
    if (!clipWidget) return;
    setWidgetVisible(clipWidget, false, node);

    // ── DOM ────────────────────────────────────────────────────────────────────
    const wrap = document.createElement("div");
    wrap.className = "fbt-spcp-wrap";

    const lbl = document.createElement("span");
    lbl.className = "fbt-spcp-label";
    lbl.textContent = "Clip ID";
    wrap.appendChild(lbl);

    const sel = document.createElement("select");
    sel.className = "fbt-spcp-sel";
    wrap.appendChild(sel);

    // ── Populate ───────────────────────────────────────────────────────────────

    function _populate(clips) {
        const current = clipWidget.value ?? "";
        sel.innerHTML = "";

        const placeholder = document.createElement("option");
        placeholder.value = "";
        placeholder.textContent = clips.length
            ? "— select clip —"
            : "— connect a Source Profile —";
        if (!current) placeholder.selected = true;
        sel.appendChild(placeholder);

        let matched = !current; // blank value always matches the placeholder
        clips.forEach(c => {
            const o = document.createElement("option");
            o.value = c.id;
            o.textContent = c.label ? `${c.id}  (${c.label})` : c.id;
            if (c.id === current) { o.selected = true; matched = true; }
            sel.appendChild(o);
        });

        // Preserve a value saved in the workflow even if the profile changed.
        if (!matched && current) {
            const o = document.createElement("option");
            o.value = current;
            o.textContent = `${current}  (?)`; // mark as unresolved
            o.selected = true;
            sel.appendChild(o);
        }

        sel.disabled = clips.length === 0;
    }

    sel.addEventListener("change", () => {
        clipWidget.value = sel.value;
        app?.graph?.setDirtyCanvas?.(true, false);
    });

    // ── Refresh ────────────────────────────────────────────────────────────────
    // Walk the connected link → upstream node → profile_id widget → API fetch.

    async function _refreshClipDropdown() {
        // When scene_cast is connected, the backend ignores the widget entirely.
        // Show a disabled placeholder so the user knows the cast controls the clip.
        const hasCast = node.inputs?.some(inp => inp.name === "scene_cast" && inp.link != null);
        if (hasCast) {
            sel.innerHTML = "";
            const o = document.createElement("option");
            o.value = "";
            o.textContent = "— clip controlled by Scene Cast —";
            sel.appendChild(o);
            sel.disabled = true;
            lbl.textContent = "Clip ID  (override: Scene Cast)";
            return;
        }

        // Standalone mode — populate from the connected source profile.
        lbl.textContent = "Clip ID";
        const linkId = node.inputs?.[0]?.link;
        if (!linkId) {
            _populate([]);
            return;
        }
        const linkObj = app.graph.links[linkId];
        const upstream = linkObj ? app.graph.getNodeById(linkObj.origin_id) : null;
        if (!upstream) {
            _populate([]);
            return;
        }
        const profileWidget = upstream.widgets?.find(
            w => w.name === "profile_name" || w.name === "profile_id"
        );
        const profileVal = profileWidget?.value;
        if (!profileVal || profileVal === "(none)") {
            _populate([]);
            return;
        }
        const isName = profileWidget?.name === "profile_name";
        const profileParam = isName
            ? `name=${encodeURIComponent(profileVal)}`
            : `id=${encodeURIComponent(profileVal)}`;
        try {
            const resp = await fetch(`/fbtools/source_profiles/get?${profileParam}`);
            if (!resp.ok) { _populate([]); return; }
            const profile = await resp.json();
            _populate(profile.clips ?? []);
        } catch {
            _populate([]);
        }
    }

    node._refreshClipDropdown = _refreshClipDropdown;

    // ── DOM widget ─────────────────────────────────────────────────────────────
    const domWidget = node.addDOMWidget("_clip_id_select", "preview", wrap, {
        serialize: false,
        hideOnZoom: false,
        margin: 0,
        getValue() { return null; },
        setValue() {},
    });
    domWidget.computeSize = () => [0, 52];

    // First pass — in case we're built inside an already-wired graph.
    requestAnimationFrame(() => _refreshClipDropdown());
}
