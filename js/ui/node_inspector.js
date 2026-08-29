/**
 * Node Inspector tab — renders serialized node data as a collapsible JSON tree.
 *
 * Uses the same jsnview library that the old bottom-panel tab used.
 * Call updateNodeInspector(nodeData) from fb_tools.js to push new data here.
 * The tab is activated automatically each time new data arrives.
 */

// ── jsnview loader (mirrors the loader in fb_tools.js) ────────────────────────

const _jsonViewLoader = {
    _promise: null,
    async ensureLoaded() {
        if (this._promise) return this._promise;
        if (window.jsnview) { this._promise = Promise.resolve(window.jsnview); return this._promise; }
        this._promise = new Promise((resolve, reject) => {
            const s = document.createElement("script");
            s.src = "https://unpkg.com/jsnview@3.0.0/dist/index.js";
            s.onload  = () => resolve(window.jsnview);
            s.onerror = reject;
            document.head.appendChild(s);
        });
        return this._promise;
    },
    get lib() { return window.jsnview; },
};

// ── Module state ──────────────────────────────────────────────────────────────

let _container = null;   // the .fbt-ni-content div, set on first render
let _pending   = null;   // node data that arrived before the tab was mounted

// ── CSS ───────────────────────────────────────────────────────────────────────

const _CSS = `
.fbt-ni-panel {
    display:flex; flex-direction:column; height:100%; overflow:hidden;
    font-size:13px; font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;
}
.fbt-ni-toolbar {
    display:flex; align-items:center; gap:6px; padding:6px 10px;
    background:var(--p-surface-card,#1e1e1e);
    border-bottom:1px solid var(--p-surface-border,#444); flex-shrink:0;
}
.fbt-ni-title {
    font-size:11px; font-weight:600; letter-spacing:.05em;
    text-transform:uppercase; color:var(--p-text-muted-color,#888);
    flex:1;
}
.fbt-ni-btn {
    font-size:11px; padding:2px 8px; border-radius:4px; cursor:pointer;
    border:1px solid var(--p-surface-border,#444);
    background:var(--p-surface-section,#252525);
    color:var(--p-text-color,#ccc);
}
.fbt-ni-btn:hover { background:var(--p-surface-hover,#333); }
.fbt-ni-empty {
    padding:16px; color:var(--p-text-muted-color,#888);
    font-size:12px; font-style:italic;
}
.fbt-ni-content {
    flex:1; min-height:0; overflow:auto; padding:8px 10px;
    overscroll-behavior:contain;
}
/* jsnview dark-theme overrides — jsnview@3.0.0 ships Tailwind classes; we
   remap every color + the white root background to ComfyUI palette tokens. */
.fbt-ni-content .jsv {
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
    color: var(--p-text-color, #ccc) !important;
}
.fbt-ni-content .jsv-content {
    border-color: var(--p-surface-border, #444) !important;
}
.fbt-ni-content .jsv-toggle {
    color: var(--p-text-muted-color, #888) !important;
}
/* object keys */
.fbt-ni-content .text-amber-800 { color: var(--p-amber-400,  #fbbf24) !important; }
/* property names */
.fbt-ni-content .text-gray-600  { color: var(--p-surface-200, #e5e5e5) !important; }
/* type/length hints */
.fbt-ni-content .text-gray-500  { color: var(--p-text-muted-color, #888) !important; }
/* string values */
.fbt-ni-content .text-green-700 { color: var(--p-green-400,  #4ade80) !important; }
/* number values */
.fbt-ni-content .text-blue-700  { color: var(--p-blue-400,   #60a5fa) !important; }
/* boolean / null values (jsnview uses -700, not -400) */
.fbt-ni-content .text-rose-700  { color: var(--p-red-400,    #f87171) !important; }
/* type label (italic) */
.fbt-ni-content .text-stone-700 { color: var(--p-stone-400,  #a8a29e) !important; }
`;

function _injectCSS() {
    if (document.getElementById("fbt-ni-styles")) return;
    const s = document.createElement("style");
    s.id = "fbt-ni-styles";
    s.textContent = _CSS;
    document.head.appendChild(s);
}

// ── Tree helpers ──────────────────────────────────────────────────────────────

function _collapseAll() {
    _container?.querySelectorAll(".jsv-toggle").forEach(t => {
        const content = t.parentElement?.querySelector(".jsv-content");
        if (content && !content.classList.contains("hidden")) {
            t.classList.add("-rotate-90");
            content.classList.add("hidden");
        }
    });
}

function _expandAll() {
    _container?.querySelectorAll(".jsv-toggle").forEach(t => {
        const content = t.parentElement?.querySelector(".jsv-content");
        if (content?.classList.contains("hidden")) {
            t.classList.remove("-rotate-90");
            content.classList.remove("hidden");
        }
    });
}

// ── Render helpers ────────────────────────────────────────────────────────────

function _showEmpty() {
    if (!_container) return;
    _container.innerHTML = "";
    const p = document.createElement("div");
    p.className = "fbt-ni-empty";
    p.textContent = "Select a node and click the inspect button (or right-click → Extract Node as JSON).";
    _container.appendChild(p);
}

async function _renderData(nodeData) {
    if (!_container) return;
    _container.innerHTML = "";

    const lib = await _jsonViewLoader.ensureLoaded().catch(() => null);
    if (!lib) {
        // Fallback: plain <pre>
        const pre = document.createElement("pre");
        pre.style.cssText = "white-space:pre-wrap;word-break:break-all;margin:0;font-size:12px;";
        pre.textContent = JSON.stringify(nodeData, null, 2);
        _container.appendChild(pre);
        return;
    }

    const viewerEl = document.createElement("div");
    const formatter = new lib(nodeData, {
        element: viewerEl,
        collapsed: false,
        showLen: true,
        showType: false,
        showFoldmarker: true,
        maxDepth: 4,
    });
    _container.appendChild(formatter.getElement());
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Called by fb_tools.js when the user inspects a node.
 * Activates the Inspector tab and renders the data.
 */
export function updateNodeInspector(nodeData) {
    if (!_container) {
        _pending = nodeData;
        // Try to open the tab so it mounts and picks up _pending
        window._fbtActivateTab?.("inspector");
        return;
    }
    _pending = null;
    window._fbtActivateTab?.("inspector");
    _renderData(nodeData);
}

/**
 * Called by fbt_panel.js when the Inspector tab is first activated.
 */
export function renderNodeInspector(rootEl) {
    _injectCSS();
    rootEl.innerHTML = "";

    const panel = document.createElement("div");
    panel.className = "fbt-ni-panel";

    // Toolbar
    const toolbar = document.createElement("div");
    toolbar.className = "fbt-ni-toolbar";
    const title = document.createElement("span");
    title.className = "fbt-ni-title";
    title.textContent = "Node Inspector";
    const collapseBtn = document.createElement("button");
    collapseBtn.className = "fbt-ni-btn";
    collapseBtn.textContent = "Collapse all";
    collapseBtn.onclick = _collapseAll;
    const expandBtn = document.createElement("button");
    expandBtn.className = "fbt-ni-btn";
    expandBtn.textContent = "Expand all";
    expandBtn.onclick = _expandAll;
    toolbar.append(title, collapseBtn, expandBtn);

    // Content area
    const content = document.createElement("div");
    content.className = "fbt-ni-content";
    _container = content;

    panel.append(toolbar, content);
    rootEl.appendChild(panel);

    // Render pending data if we got a push before the tab mounted
    if (_pending !== null) {
        _renderData(_pending);
        _pending = null;
    } else {
        _showEmpty();
    }
}
