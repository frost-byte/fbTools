/**
 * fbTools unified sidebar panel.
 *
 * Single ComfyUI sidebar entry hosting all fbTools tabs:
 *   Compose · Bundles · Casts · Sources · History
 *
 * Features:
 * - Persistent LLM status bar in the header (read-only — model management
 *   stays in the Compositions tab where the full Load/Unload UI lives)
 * - Lazy tab mounting: each tab's DOM is created once on first activation
 *   and kept alive (hidden) on switch, so state is never lost
 * - Synchronous status push: composition_editor calls window._fbtUpdateLlmStatus
 *   after every load/unload so the header badge stays current without polling
 */

import { llmApi }                    from "../api/llm.js";
import { renderCompositionEditor }   from "./composition_editor.js";
import { renderBundleEditor }        from "./bundle_editor.js";
import { renderCastEditor }          from "./cast_editor.js";
import { renderSourceProfileEditor } from "./source_profile_editor.js";
import { renderRunHistory }          from "./run_history.js";
import { renderNodeInspector }       from "./node_inspector.js";

// ── Shared LLM state ───────────────────────────────────────────────────────────
// Any tab can read fbtLlm to see what's currently loaded without its own fetch.

export const fbtLlm = {
    loaded:      null,   // string model name or null
    vision:      false,
    nativeVideo: false,
};

// ── DOM helpers ────────────────────────────────────────────────────────────────

function _mk(tag, props = {}, children = []) {
    const el = document.createElement(tag);
    Object.entries(props).forEach(([k, v]) => {
        if (k === "cls")        el.className = v;
        else if (k === "style") Object.assign(el.style, v);
        else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
        else el[k] = v;
    });
    children.filter(Boolean).forEach(c =>
        el.appendChild(typeof c === "string" ? document.createTextNode(c) : c));
    return el;
}

// ── CSS ────────────────────────────────────────────────────────────────────────

const _CSS = `
.fbt-panel { display:flex; flex-direction:column; height:100%; flex:1; min-height:0; overflow:hidden; font-size:13px; }

/* Header */
.fbt-panel-hdr {
    display:flex; align-items:center; gap:8px; padding:5px 10px;
    background:var(--p-surface-card,#1e1e1e);
    border-bottom:1px solid var(--p-surface-border,#444); flex-shrink:0;
}
.fbt-panel-brand {
    font-weight:700; font-size:13px; letter-spacing:.04em;
    color:var(--p-primary-color,#58a6ff); white-space:nowrap; flex-shrink:0;
}
.fbt-llm-bar {
    flex:1; display:flex; align-items:center; gap:5px; overflow:hidden;
    cursor:pointer; min-width:0;
}
.fbt-llm-dot {
    width:7px; height:7px; border-radius:50%; background:#555;
    flex-shrink:0; transition:background .2s;
}
.fbt-llm-dot.ok   { background:#22c55e; }
.fbt-llm-dot.busy { background:#f59e0b; }
.fbt-llm-lbl {
    font-size:11px; color:var(--p-text-muted-color,#888);
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}
.fbt-llm-lbl.ok { color:var(--p-text-color,#ccc); }
.fbt-llm-refresh {
    font-size:12px; color:var(--p-text-muted-color,#666);
    background:none; border:none; padding:0 2px; cursor:pointer; flex-shrink:0;
}
.fbt-llm-refresh:hover { color:var(--p-text-color,#eee); }

/* Tab strip */
.fbt-tab-strip {
    display:flex; border-bottom:1px solid var(--p-surface-border,#444);
    background:var(--p-surface-section,#252525); flex-shrink:0;
}
.fbt-tab {
    flex:1; display:flex; flex-direction:column; align-items:center; gap:2px;
    padding:7px 2px; cursor:pointer; border:none; background:transparent;
    color:var(--p-text-muted-color,#888);
    border-bottom:2px solid transparent;
    transition:color .12s, border-color .12s;
}
.fbt-tab:hover { color:var(--p-text-color,#eee); }
.fbt-tab.active {
    color:var(--p-primary-color,#58a6ff);
    border-bottom-color:var(--p-primary-color,#58a6ff);
}
.fbt-tab .pi { font-size:14px; }
.fbt-tab-lbl { font-size:9px; text-transform:uppercase; letter-spacing:.04em; line-height:1; }

/* Content panes — pure flex chain, no absolute positioning needed.
   Visibility is controlled via inline display style (not a class) so that
   render functions which set their own inline display can't accidentally
   override the hide logic. */
.fbt-tab-contents { flex:1; min-height:0; overflow:hidden; display:flex; flex-direction:column; }
.fbt-tab-content  { flex:1; min-height:0; display:flex; flex-direction:column; overflow:hidden; }
`;

function _injectCSS() {
    if (document.getElementById("fbt-panel-styles")) return;
    const s = document.createElement("style");
    s.id = "fbt-panel-styles";
    s.textContent = _CSS;
    document.head.appendChild(s);
}

// ── LLM status bar ─────────────────────────────────────────────────────────────

let _dotEl = null;
let _lblEl = null;

function _syncStatusBar() {
    if (!_dotEl || !_lblEl) return;
    const name = fbtLlm.loaded;
    _dotEl.className = "fbt-llm-dot" + (name ? " ok" : "");
    _lblEl.className = "fbt-llm-lbl" + (name ? " ok" : "");
    _lblEl.textContent = name
        ? (name.length > 30 ? name.slice(0, 28) + "…" : name)
        : "No model — load in Compose";
    _lblEl.title = name || "";
}

// Synchronous push — called by composition_editor after every load/unload.
// Avoids an extra network round-trip and keeps the badge instantly in sync.
function _handleLlmPush(loaded, vision, nativeVideo) {
    fbtLlm.loaded      = loaded      ?? null;
    fbtLlm.vision      = vision      ?? false;
    fbtLlm.nativeVideo = nativeVideo ?? false;
    _syncStatusBar();
    // Broadcast to any node handlers listening (e.g. DatasetCaptioner status line)
    document.dispatchEvent(new CustomEvent("fbt:llm-status", { detail: { ...fbtLlm } }));
}

// Async pull — used on panel open and tab switch to reconcile if composition
// editor was used before this panel was open.
async function _fetchLlmStatus() {
    try {
        const st = await llmApi.status();
        _handleLlmPush(st.loaded_model, st.supports_vision, st.native_video);
    } catch (_) {
        _handleLlmPush(null, false, false);
    }
}

// ── Tab definitions ────────────────────────────────────────────────────────────

const TABS = [
    { id: "compositions", label: "Compose",  icon: "pi pi-file-edit", render: renderCompositionEditor },
    { id: "bundles",      label: "Bundles",  icon: "pi pi-images",    render: renderBundleEditor },
    { id: "casts",        label: "Casts",    icon: "pi pi-users",     render: renderCastEditor },
    { id: "sources",      label: "Sources",  icon: "pi pi-video",     render: renderSourceProfileEditor },
    { id: "history",      label: "History",  icon: "pi pi-history",   render: renderRunHistory },
    { id: "inspector",    label: "Inspect",  icon: "pi pi-code",      render: renderNodeInspector },
];

// ── Panel ──────────────────────────────────────────────────────────────────────

export function renderFbtPanel(container) {
    _injectCSS();

    // Guard: only mount once per container
    if (container.querySelector(".fbt-panel")) return;

    const panel = _mk("div", { cls: "fbt-panel" });
    container.appendChild(panel);

    // ── Header ─────────────────────────────────────────────────────────────────
    _dotEl = _mk("span", { cls: "fbt-llm-dot" });
    _lblEl = _mk("span", { cls: "fbt-llm-lbl" }, ["Checking…"]);

    const llmBar = _mk("div", {
        cls: "fbt-llm-bar",
        title: "LLM status — click to refresh",
        onclick: _fetchLlmStatus,
    }, [_dotEl, _lblEl]);

    const refreshBtn = _mk("button", {
        cls: "fbt-llm-refresh", title: "Refresh LLM status",
        onclick: (e) => { e.stopPropagation(); _fetchLlmStatus(); },
    }, ["↻"]);

    panel.appendChild(_mk("div", { cls: "fbt-panel-hdr" }, [
        _mk("span", { cls: "fbt-panel-brand" }, ["🧊 fbTools"]),
        llmBar,
        refreshBtn,
    ]));

    // ── Tab strip + content panes ──────────────────────────────────────────────
    const strip    = _mk("div", { cls: "fbt-tab-strip" });
    const contents = _mk("div", { cls: "fbt-tab-contents" });
    panel.appendChild(strip);
    panel.appendChild(contents);

    const tabBtns    = {};
    const contentEls = {};
    const mounted    = {};
    let   activeId   = TABS[0].id;

    TABS.forEach((tab, i) => {
        const btn = _mk("button", {
            cls: "fbt-tab" + (i === 0 ? " active" : ""),
            title: tab.label,
            onclick: () => activateTab(tab.id),
        }, [
            _mk("i", { cls: tab.icon }),
            _mk("span", { cls: "fbt-tab-lbl" }, [tab.label]),
        ]);
        strip.appendChild(btn);
        tabBtns[tab.id] = btn;

        const pane = _mk("div", { cls: "fbt-tab-content" });
        if (i !== 0) pane.style.display = "none";
        contents.appendChild(pane);
        contentEls[tab.id] = pane;
    });

    function activateTab(id) {
        const alreadyActive = id === activeId && mounted[id];
        // Use inline display to override any inline styles set by render functions
        Object.values(tabBtns).forEach(b => b.classList.remove("active"));
        Object.values(contentEls).forEach(p => { p.style.display = "none"; });
        tabBtns[id].classList.add("active");
        contentEls[id].style.display = "";  // clear inline → CSS flex takes over
        activeId = id;

        // Lazy mount: call render exactly once.
        // Render functions may be async; we don't await but errors surface in console.
        if (!mounted[id]) {
            try {
                const result = TABS.find(t => t.id === id).render(contentEls[id]);
                if (result instanceof Promise) {
                    result.catch(err => console.error(`[fbTools] Tab "${id}" render error:`, err));
                }
            } catch (err) {
                console.error(`[fbTools] Tab "${id}" render error:`, err);
            }
            mounted[id] = true;
        }

        // Refresh status on every tab switch (catches changes made while
        // this panel was closed or another tab was active)
        if (!alreadyActive) _fetchLlmStatus();
    }

    // Eagerly mount and activate the first tab
    activateTab(TABS[0].id);

    // Wire the synchronous push from composition_editor
    window._fbtUpdateLlmStatus = _handleLlmPush;
    // Read-only accessor for other tabs that need to check LLM state
    window._fbtGetLlmStatus = () => ({ ...fbtLlm });
    // Programmatic tab activation — used by node_inspector.js to switch to Inspector tab
    window._fbtActivateTab = activateTab;

    // Initial status fetch
    _fetchLlmStatus();
}
