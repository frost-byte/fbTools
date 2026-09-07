/**
 * fbTools unified sidebar panel.
 *
 * Single ComfyUI sidebar entry hosting all fbTools tabs:
 *   Compose · Bundles · Casts · Sources · LLM · History
 *
 * Features:
 * - Persistent LLM status bar in the header (read-only indicator)
 * - Lazy tab mounting: each tab's DOM is created once on first activation
 *   and kept alive (hidden) on switch, so state is never lost
 * - Synchronous status push: LLM Local tab calls window._fbtUpdateLlmStatus
 *   after every load/unload so the header badge and Compose tab stay current
 */

import { llmApi }                    from "../api/llm.js";
import { renderCompositionEditor }   from "./composition_editor.js";
import { renderBundleEditor }        from "./bundle_editor.js";
import { renderCastEditor }          from "./cast_editor.js";
import { renderSourceProfileEditor } from "./source_profile_editor.js";
import { renderRunHistory }          from "./run_history.js";
import { renderNodeInspector }       from "./node_inspector.js";
import { renderLlmPanel, getActiveBackendLabel, onBackendChange } from "./llm_panel.js";

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

// ── LLM status bar ─────────────────────────────────────────────────────────────

let _dotEl = null;
let _lblEl = null;

function _syncStatusBar() {
    if (!_dotEl || !_lblEl) return;
    const label = getActiveBackendLabel();
    const isModal  = label.startsWith("Modal:");
    const isLocal  = label.startsWith("Local:") && fbtLlm.loaded;
    const hasModel = isModal || isLocal;
    _dotEl.className = "fbt-llm-dot" + (isModal ? " busy" : hasModel ? " ok" : "");
    _lblEl.className = "fbt-llm-lbl" + (hasModel ? " ok" : "");
    const display = label.length > 30 ? label.slice(0, 28) + "…" : label;
    _lblEl.textContent = hasModel ? display : "No backend — configure in LLM tab";
    _lblEl.title = label;
}

// Synchronous push — called by LLM Local tab after load/unload.
// Avoids an extra network round-trip and keeps the badge instantly in sync.
function _handleLlmPush(loaded, vision, nativeVideo) {
    fbtLlm.loaded      = loaded      ?? null;
    fbtLlm.vision      = vision      ?? false;
    fbtLlm.nativeVideo = nativeVideo ?? false;
    _syncStatusBar();
    // Broadcast to any node handlers listening (e.g. DatasetCaptioner status line)
    document.dispatchEvent(new CustomEvent("fbt:llm-status", { detail: { ...fbtLlm } }));
}

// Async pull — used on panel open and tab switch to reconcile with server state.
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
    { id: "llm",          label: "LLM",      icon: "pi pi-microchip", render: renderLlmPanel },
    { id: "history",      label: "History",  icon: "pi pi-history",   render: renderRunHistory },
    { id: "inspector",    label: "Inspect",  icon: "pi pi-code",      render: renderNodeInspector },
];

// ── Panel ──────────────────────────────────────────────────────────────────────

export function renderFbtPanel(container) {
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

    // Synchronous push — called by LLM Local tab (and anywhere that loads/unloads a model)
    window._fbtUpdateLlmStatus = _handleLlmPush;
    // Read-only accessor for other tabs that need to check LLM state
    window._fbtGetLlmStatus = () => ({ ...fbtLlm });
    // Programmatic tab activation — used by node_inspector.js and llm_panel.js
    window._fbtActivateTab = activateTab;
    // Re-sync header when Modal backend state changes
    onBackendChange(_syncStatusBar);

    // Initial status fetch
    _fetchLlmStatus();
}
