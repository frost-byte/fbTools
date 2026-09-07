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
