/**
 * LLM Backend panel — rendered in the dedicated "LLM" sidebar tab.
 *
 * Three sub-tabs:
 *   Local  — shows current local model status; model management stays in Compose tab.
 *   Modal  — activate/deactivate Modal cloud VLM; model selector + timeout config.
 *   Gemini — setup info for the GEMINI_API_KEY environment variable.
 *
 * Exports:
 *   renderLlmPanel(container)
 *   getActiveCaptionerType()  → "modal" | "gemini_flash" | "auto"
 *   getActiveBackendLabel()   → human-readable string for status display
 */

import { llmApi }           from "../api/llm.js";
import { modalApi, vlmActivityApi } from "../api/modal.js";

// ── Shared backend state (read by source_profile_editor and fbt_panel header) ─

const _state = {
    modalActive:  false,
    modalModel:   "qwen3-vl-8b",
    modalQuant:   true,
    modalNativeV: true,
    idleMinutes:  10,
    keepWarm:     false,
    presets:      [],
    modelHistory: [],   // custom HF repo IDs from activity log
};

const STORAGE_KEY = "fbt_llm_panel_v1";

function _loadState() {
    try {
        const s = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
        if (s.modalModel)  _state.modalModel  = s.modalModel;
        if (s.modalQuant   !== undefined) _state.modalQuant   = !!s.modalQuant;
        if (s.idleMinutes) _state.idleMinutes = s.idleMinutes;
        if (s.keepWarm     !== undefined) _state.keepWarm     = !!s.keepWarm;
    } catch (_) {}
}

function _saveState() {
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify({
            modalModel:  _state.modalModel,
            modalQuant:  _state.modalQuant,
            idleMinutes: _state.idleMinutes,
            keepWarm:    _state.keepWarm,
        }));
    } catch (_) {}
}

_loadState();

// ── Public API ────────────────────────────────────────────────────────────────

/** Returns the captioner_type string to use for VLM inference requests. */
export function getActiveCaptionerType() {
    if (_state.modalActive) return "modal";
    const llm = window._fbtGetLlmStatus?.() || {};
    if (llm.loaded && llm.vision) return "auto";
    return "gemini_flash";
}

/** Human-readable active backend description for status bars. */
export function getActiveBackendLabel() {
    if (_state.modalActive) return `Modal: ${_state.modalModel}`;
    const llm = window._fbtGetLlmStatus?.() || {};
    if (llm.loaded && llm.vision) return `Local: ${llm.loaded}`;
    return "Gemini";
}

// Internal change listeners (used by fbt_panel header)
const _changeListeners = new Set();
export function onBackendChange(fn) { _changeListeners.add(fn); }
function _notify() { _changeListeners.forEach(fn => { try { fn(); } catch (_) {} }); }

// ── DOM helpers ───────────────────────────────────────────────────────────────

function _mk(tag, props = {}, children = []) {
    const el = document.createElement(tag);
    Object.entries(props).forEach(([k, v]) => {
        if (k === "cls")        el.className = v;
        else if (k === "style") Object.assign(el.style, v);
        else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
        else el[k] = v;
    });
    (children || []).filter(Boolean).forEach(c =>
        el.appendChild(typeof c === "string" ? document.createTextNode(c) : c));
    return el;
}

function _txt(el, text) { el.textContent = text; }

// ── CSS ───────────────────────────────────────────────────────────────────────

const _CSS = `
.llmp-wrap { display:flex; flex-direction:column; height:100%; overflow:hidden; padding:8px; box-sizing:border-box; gap:8px; }

/* Sub-tab strip */
.llmp-tabs { display:flex; gap:4px; border-bottom:1px solid var(--p-surface-border,#444); padding-bottom:6px; flex-shrink:0; }
.llmp-tab {
    flex:1; padding:5px 4px; border:none; background:transparent;
    color:var(--p-text-muted-color,#888); font-size:11px; font-weight:600;
    text-transform:uppercase; letter-spacing:.05em; cursor:pointer;
    border-bottom:2px solid transparent; transition:color .12s, border-color .12s;
}
.llmp-tab:hover { color:var(--p-text-color,#ccc); }
.llmp-tab.active { color:var(--p-primary-color,#58a6ff); border-bottom-color:var(--p-primary-color,#58a6ff); }
.llmp-tab .llmp-tab-dot {
    display:inline-block; width:6px; height:6px; border-radius:50%;
    background:#555; margin-right:5px; vertical-align:middle;
}
.llmp-tab .llmp-tab-dot.ok  { background:#22c55e; }
.llmp-tab .llmp-tab-dot.blue { background:#60a5fa; }

/* Tab panes */
.llmp-pane { display:none; flex-direction:column; gap:10px; overflow-y:auto; flex:1; }
.llmp-pane.active { display:flex; }

/* Rows */
.llmp-row { display:flex; align-items:center; gap:8px; }
.llmp-label { font-size:11px; color:var(--p-text-muted-color,#888); flex-shrink:0; min-width:68px; }
.llmp-value { font-size:12px; color:var(--p-text-color,#ccc); font-weight:500; }

/* Status badge */
.llmp-status {
    display:flex; align-items:center; gap:6px; padding:7px 10px;
    border-radius:6px; background:var(--p-surface-section,#252525);
    border:1px solid var(--p-surface-border,#444);
}
.llmp-status-dot { width:8px; height:8px; border-radius:50%; background:#555; flex-shrink:0; }
.llmp-status-dot.ok   { background:#22c55e; }
.llmp-status-dot.blue { background:#60a5fa; }
.llmp-status-dot.warn { background:#f59e0b; }
.llmp-status-text { font-size:12px; flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }

/* Buttons */
.llmp-btn {
    padding:5px 12px; border-radius:4px; border:none; cursor:pointer; font-size:12px;
    font-weight:500; transition:opacity .12s;
}
.llmp-btn:hover { opacity:.85; }
.llmp-btn.primary { background:var(--p-primary-color,#58a6ff); color:#000; }
.llmp-btn.danger  { background:#ef4444; color:#fff; }
.llmp-btn.ghost   { background:transparent; color:var(--p-text-color,#ccc);
                    border:1px solid var(--p-surface-border,#444); }
.llmp-btn:disabled { opacity:.4; cursor:default; }

/* Select / input */
.llmp-select, .llmp-input {
    flex:1; padding:4px 6px; border-radius:4px; font-size:12px;
    background:var(--p-surface-ground,#1a1a1a); color:var(--p-text-color,#eee);
    border:1px solid var(--p-surface-border,#444); min-width:0;
}
.llmp-input[type=number] { width:60px; flex:none; }

/* Info block */
.llmp-info {
    font-size:11px; color:var(--p-text-muted-color,#888); line-height:1.5;
    padding:8px 10px; background:var(--p-surface-section,#252525);
    border-radius:6px; border-left:3px solid var(--p-primary-color,#58a6ff);
}
.llmp-info code { font-family:monospace; background:rgba(255,255,255,.08); padding:1px 4px; border-radius:3px; }

/* Idle indicator */
.llmp-idle { font-size:11px; color:var(--p-text-muted-color,#888); text-align:right; }
.llmp-idle.warn { color:#f59e0b; }

/* Custom model input row */
.llmp-custom-row { display:flex; gap:6px; align-items:center; }
.llmp-history-list { display:flex; flex-direction:column; gap:3px; }
.llmp-history-item {
    display:flex; align-items:center; gap:6px; padding:3px 6px;
    border-radius:3px; cursor:pointer; font-size:11px;
    color:var(--p-text-muted-color,#888);
}
.llmp-history-item:hover { background:var(--p-surface-section,#252525); color:var(--p-text-color,#ccc); }
`;

function _injectCSS() {
    if (document.getElementById("llmp-styles")) return;
    const s = document.createElement("style");
    s.id = "llmp-styles";
    s.textContent = _CSS;
    document.head.appendChild(s);
}

// ── Local sub-tab ─────────────────────────────────────────────────────────────

function _renderLocalTab(pane) {
    const _ls = { models: [], busy: false };

    // ── Status badge ───────────────────────────────────────────────────────
    const dot    = _mk("div", { cls: "llmp-status-dot" });
    const lbl    = _mk("span", { cls: "llmp-status-text" }, ["Checking…"]);
    const status = _mk("div", { cls: "llmp-status" }, [dot, lbl]);
    const badgeRow = _mk("div", { cls: "llmp-row", style: { gap: "6px", flexWrap: "wrap" } });

    // ── Model selector ─────────────────────────────────────────────────────
    const modelSel   = _mk("select", { cls: "llmp-select" });
    const refreshBtn = _mk("button", { cls: "llmp-btn ghost", title: "Re-scan model directories" }, ["↻ Scan"]);
    const capNote    = _mk("div", { cls: "llmp-info", style: { display: "none", padding: "4px 8px", marginTop: "0" } });

    // ── Load / Unload ──────────────────────────────────────────────────────
    const loadBtn   = _mk("button", { cls: "llmp-btn primary" }, ["Load"]);
    const unloadBtn = _mk("button", { cls: "llmp-btn danger",   style: { display: "none" } }, ["Unload"]);

    // ── Download prompt ────────────────────────────────────────────────────
    const downloadRow = _mk("div", {});
    const dlInfo = _mk("div", { cls: "llmp-info", style: { borderLeftColor: "#f59e0b" } });
    dlInfo.innerHTML = (
        "<b>No models found.</b> Place GGUF models in <code>ComfyUI/models/LLMs/&lt;name&gt;/</code>, " +
        "one subdirectory per model. Vision models need an <code>mmproj-*.gguf</code> alongside the main " +
        "<code>.gguf</code>.<br>Or download the recommended starter:"
    );
    const dlBtn = _mk("button", { cls: "llmp-btn ghost", style: { marginTop: "4px" } }, ["Download Qwen2.5-VL 3B"]);
    downloadRow.append(dlInfo, dlBtn);

    function _setStatus(text, ok = false) {
        lbl.textContent = text;
        dot.className   = "llmp-status-dot" + (ok ? " ok" : "");
    }

    function _setBusy(busy) {
        _ls.busy           = busy;
        loadBtn.disabled   = busy || _ls.models.length === 0;
        unloadBtn.disabled = busy;
        refreshBtn.disabled = busy;
    }

    function _syncFromGlobal() {
        const llm = window._fbtGetLlmStatus?.() || {};
        if (llm.loaded) {
            _setStatus(llm.loaded, true);
            loadBtn.textContent     = "Reload";
            unloadBtn.style.display = "";
            badgeRow.innerHTML      = "";
            if (llm.vision)      badgeRow.appendChild(_mk("span", { cls: "llmp-value", style: { fontSize: "11px", color: "#22c55e" } }, ["✓ vision"]));
            if (llm.nativeVideo) badgeRow.appendChild(_mk("span", { cls: "llmp-value", style: { fontSize: "11px", color: "#60a5fa" } }, ["✓ native-video"]));
        } else {
            _setStatus("No model loaded");
            loadBtn.textContent     = "Load";
            unloadBtn.style.display = "none";
            badgeRow.innerHTML      = "";
        }
    }

    function _populateSel() {
        modelSel.innerHTML = "";
        const noModels = _ls.models.length === 0;
        if (noModels) {
            modelSel.appendChild(_mk("option", { value: "" }, ["— no models found —"]));
        } else {
            _ls.models.forEach(m => {
                const o = _mk("option", { value: m.id });
                o.textContent = `${m.name}  ${(m.capability_tags || []).join(" ")}`;
                o.title       = m.capability_note || "";
                modelSel.appendChild(o);
            });
        }
        loadBtn.disabled          = noModels;
        downloadRow.style.display = noModels ? "" : "none";
    }

    async function _scanModels() {
        try {
            const data = await llmApi.listModels();
            _ls.models = data.models || [];
            _populateSel();
        } catch (_) { /* silent */ }
    }

    modelSel.onchange = () => {
        const m = _ls.models.find(x => x.id === modelSel.value);
        if (m?.capability_note) {
            capNote.textContent   = m.capability_note;
            capNote.style.display = "";
        } else {
            capNote.style.display = "none";
        }
    };

    loadBtn.onclick = async () => {
        if (_ls.busy) return;
        const modelInfo = _ls.models.find(m => m.id === modelSel.value);
        if (!modelInfo) return;
        _setBusy(true);
        _setStatus(`Loading ${modelInfo.name}…`);
        try {
            const r = await llmApi.loadModel(modelInfo);
            if (r.success) {
                window._fbtUpdateLlmStatus?.(
                    modelInfo.name,
                    modelInfo.supports_vision,
                    modelInfo.native_video ?? false,
                );
            } else {
                _setStatus(`Load failed: ${r.message || "unknown error"}`);
            }
        } catch (e) {
            _setStatus(`Load error: ${e.message}`);
        }
        _setBusy(false);
        _syncFromGlobal();
    };

    unloadBtn.onclick = async () => {
        if (_ls.busy) return;
        _setBusy(true);
        _setStatus("Unloading…");
        try {
            await llmApi.unloadModel();
            window._fbtUpdateLlmStatus?.(null, false, false);
        } catch (e) {
            _setStatus(`Unload error: ${e.message}`);
        }
        _setBusy(false);
        _syncFromGlobal();
    };

    refreshBtn.onclick = _scanModels;

    dlBtn.onclick = async () => {
        dlBtn.disabled    = true;
        dlBtn.textContent = "Downloading…";
        try {
            const r = await llmApi.downloadDefault();
            if (r.success) {
                dlBtn.textContent = "Done — click ↻ Scan";
            } else {
                dlBtn.disabled    = false;
                dlBtn.textContent = "Retry Download";
            }
        } catch (_) {
            dlBtn.disabled    = false;
            dlBtn.textContent = "Retry Download";
        }
    };

    document.addEventListener("fbt:llm-status", _syncFromGlobal);

    _syncFromGlobal();
    _scanModels();

    pane.append(
        status,
        badgeRow,
        _mk("div", { cls: "llmp-row" }, [modelSel, refreshBtn]),
        capNote,
        _mk("div", { cls: "llmp-row" }, [loadBtn, unloadBtn]),
        downloadRow,
    );
}

// ── Modal sub-tab ─────────────────────────────────────────────────────────────

function _renderModalTab(pane) {
    // Status bar
    const dot   = _mk("div", { cls: "llmp-status-dot" + (_state.modalActive ? " blue" : "") });
    const label = _mk("span", { cls: "llmp-status-text" },
        [_state.modalActive ? `Active — ${_state.modalModel}` : "Inactive"]);
    const statusEl = _mk("div", { cls: "llmp-status" }, [dot, label]);

    const idleEl = _mk("div", { cls: "llmp-idle" }, [""]);

    // Model selector
    const modelSel = _mk("select", { cls: "llmp-select" });

    const _FALLBACK_PRESETS = [
        { key: "qwen3-vl-8b",        label: "Qwen3-VL 8B",          pre_quantized: false },
        { key: "qwen2.5-vl-7b",      label: "Qwen2.5-VL 7B",        pre_quantized: false },
        { key: "qwen2.5-vl-32b-awq", label: "Qwen2.5-VL 32B (AWQ)", pre_quantized: true  },
        { key: "qwen2.5-vl-3b",      label: "Qwen2.5-VL 3B",        pre_quantized: false },
        { key: "qwen2.5-omni-7b",    label: "Qwen2.5-Omni 7B",      pre_quantized: false },
        { key: "gemma3-4b",          label: "Gemma 3 4B",            pre_quantized: false },
    ];

    // Lookup map: key → preset object (pre_quantized flag etc.)
    let _presetMap = Object.fromEntries(_FALLBACK_PRESETS.map(p => [p.key, p]));

    function _rebuildModelSel() {
        modelSel.innerHTML = "";
        const presets = _state.presets.length ? _state.presets : _FALLBACK_PRESETS;
        _presetMap = Object.fromEntries(presets.map(p => [p.key, p]));
        presets.forEach(p => {
            const suffix = p.pre_quantized ? " ·AWQ" : "";
            const o = _mk("option", { value: p.key }, [(p.label || p.key) + suffix]);
            if (p.key === _state.modalModel) o.selected = true;
            modelSel.appendChild(o);
        });
        // Custom history entries not already in presets
        const presetKeys = new Set(presets.map(p => p.key));
        _state.modelHistory.filter(k => !presetKeys.has(k)).forEach(k => {
            const o = _mk("option", { value: k }, [k]);
            if (k === _state.modalModel) o.selected = true;
            modelSel.appendChild(o);
        });
        // "Custom…" sentinel
        modelSel.appendChild(_mk("option", { value: "__custom__" }, ["Custom HF repo ID…"]));
        _applyPreQuantizedState();
    }

    // Declare quantCb/quantNotice BEFORE calling _rebuildModelSel — it calls
    // _applyPreQuantizedState which references both (TDZ guard).
    const quantCb     = _mk("input", { type: "checkbox", id: "llmp-quant-cb" });
    const quantNotice = _mk("span", {
        style: { fontSize: "11px", color: "#f59e0b", display: "none" },
    }, [" (disabled — model is pre-quantized)"]);

    quantCb.checked  = _state.modalQuant;
    quantCb.onchange = () => { _state.modalQuant = quantCb.checked; _saveState(); };

    _rebuildModelSel();

    function _applyPreQuantizedState() {
        const key = modelSel.value === "__custom__" ? "" : modelSel.value;
        const preset = _presetMap[key];
        const isPreQ = preset?.pre_quantized ?? false;
        if (isPreQ) {
            quantCb.checked  = false;
            quantCb.disabled = true;
            quantNotice.style.display = "";
        } else {
            quantCb.disabled = false;
            quantNotice.style.display = "none";
            // Restore saved preference when switching away from pre-quantized
            quantCb.checked = _state.modalQuant;
        }
    }

    // Custom input row (shown when "Custom…" is selected)
    const customInput = _mk("input", {
        cls: "llmp-select", type: "text",
        placeholder: "org/model-name — standard HF transformer repo only",
        style: { display: "none" },
    });

    modelSel.onchange = () => {
        if (modelSel.value === "__custom__") {
            customInput.style.display = "";
            customInput.focus();
            quantCb.disabled = false;
            quantNotice.style.display = "none";
            quantCb.checked = _state.modalQuant;
        } else {
            customInput.style.display = "none";
            _state.modalModel = modelSel.value;
            _saveState();
            _applyPreQuantizedState();
        }
    };
    customInput.onchange = () => {
        const v = customInput.value.trim();
        if (v) { _state.modalModel = v; _saveState(); }
    };

    // Timeout
    const timeoutInput = _mk("input", {
        cls: "llmp-input", type: "number", min: "1", max: "120", step: "1",
    });
    timeoutInput.value = _state.idleMinutes;
    timeoutInput.onchange = () => {
        const v = parseInt(timeoutInput.value, 10);
        if (v > 0) { _state.idleMinutes = v; _saveState(); }
    };

    // Keep-warm toggle
    const warmCb = _mk("input", { type: "checkbox", id: "llmp-warm-cb" });
    warmCb.checked = _state.keepWarm;
    warmCb.onchange = () => { _state.keepWarm = warmCb.checked; _saveState(); };

    // Activate / Disconnect button
    const actionBtn = _mk("button", {
        cls: "llmp-btn " + (_state.modalActive ? "danger" : "primary"),
    }, [_state.modalActive ? "Disconnect" : "Activate"]);

    let _activating = false;

    function _syncUI() {
        dot.className = "llmp-status-dot" + (_state.modalActive ? " blue" : "");
        label.textContent = _state.modalActive ? `Active — ${_state.modalModel}` : "Inactive";
        actionBtn.className = "llmp-btn " + (_state.modalActive ? "danger" : "primary");
        actionBtn.textContent = _state.modalActive ? "Disconnect" : "Activate";
        actionBtn.disabled = _activating;
    }

    actionBtn.onclick = async () => {
        if (_activating) return;
        _activating = true;
        actionBtn.disabled = true;
        try {
            if (_state.modalActive) {
                await modalApi.deactivate();
                _state.modalActive = false;
            } else {
                const modelKey = modelSel.value === "__custom__"
                    ? (customInput.value.trim() || "qwen2.5-vl-7b")
                    : modelSel.value;
                _state.modalModel = modelKey;
                _saveState();
                const res = await modalApi.activate(modelKey, _state.modalQuant);
                if (!res.success) {
                    alert(`Modal activation failed: ${res.message}`);
                    return;
                }
                _state.modalActive = true;
                // Refresh model history after activation
                _refreshHistory();
            }
        } catch (err) {
            alert(`Modal error: ${err.message}`);
        } finally {
            _activating = false;
            _syncUI();
            _notify();
        }
    };

    // Keep-warm interval (fires keep-alive pings before the idle timeout)
    let _warmInterval = null;
    function _startWarm() {
        if (_warmInterval) clearInterval(_warmInterval);
        _warmInterval = setInterval(() => {
            if (!_state.modalActive || !_state.keepWarm) return;
            // A status fetch is enough to show liveness; actual keep-warm is Modal-side
            modalApi.status().catch(() => {});
        }, Math.max(1, _state.idleMinutes - 1) * 60 * 1000);
    }

    // Idle indicator — updates every 30s
    async function _updateIdle() {
        try {
            const res = await vlmActivityApi.recent({ n: 100, backend: "modal" });
            const entries = res.entries || [];
            if (!entries.length) { idleEl.textContent = ""; return; }
            const last = new Date(entries[0].ts);
            const mins = Math.round((Date.now() - last.getTime()) / 60000);
            const overIdle = _state.modalActive && mins >= _state.idleMinutes;
            idleEl.className = "llmp-idle" + (overIdle ? " warn" : "");
            idleEl.textContent = `Last used: ${mins < 1 ? "< 1 min" : mins + " min"} ago${overIdle ? " — may be cold" : ""}`;
        } catch (_) {}
    }

    async function _refreshHistory() {
        try {
            const res = await vlmActivityApi.modelHistory("modal");
            _state.modelHistory = res.model_ids || [];
            _rebuildModelSel();
        } catch (_) {}
    }

    async function _fetchStatus() {
        try {
            const res = await modalApi.status();
            _state.presets = res.presets || [];
            _state.modalActive = !!res.active;
            if (res.active) {
                _state.modalModel = res.model_key;
                _state.modalQuant = !!res.quantize;
                quantCb.checked   = _state.modalQuant;
            }
            _rebuildModelSel();  // also calls _applyPreQuantizedState
            _syncUI();
            _notify();
        } catch (_) {}
    }

    _fetchStatus();
    _refreshHistory();
    _updateIdle();
    const _idleTimer = setInterval(_updateIdle, 30_000);
    _startWarm();

    // Layout
    pane.append(
        statusEl,
        idleEl,
        _mk("div", { cls: "llmp-row" }, [
            _mk("span", { cls: "llmp-label" }, ["Model"]),
            modelSel,
        ]),
        customInput,
        _mk("div", { cls: "llmp-row", style: { flexWrap: "wrap" } }, [
            _mk("span", { cls: "llmp-label" }, ["Quantize (NF4)"]),
            quantCb,
            _mk("label", {
                htmlFor: "llmp-quant-cb",
                style: { fontSize: "12px" },
                title: "Applies NF4 4-bit quantization at load time to bf16 models (~3× VRAM reduction). Disable for pre-quantized repos (AWQ/GPTQ) — applying NF4 on top degrades quality.",
            }, [" bf16 → 4-bit, ~3× VRAM reduction"]),
            quantNotice,
        ]),
        actionBtn,
        _mk("div", { cls: "llmp-row" }, [
            _mk("span", { cls: "llmp-label" }, ["Idle timeout"]),
            timeoutInput,
            _mk("span", { cls: "llmp-label", style: { minWidth: "auto" } }, ["min"]),
        ]),
        _mk("div", { cls: "llmp-row" }, [
            _mk("span", { cls: "llmp-label" }, ["Keep-warm"]),
            warmCb,
            _mk("label", { htmlFor: "llmp-warm-cb", style: { fontSize: "12px" } }, [" Ping before idle timeout"]),
        ]),
        _mk("div", { cls: "llmp-info" }, [
            "Cold starts take ~60 s on first use. Auth uses ",
            _mk("code", {}, ["~/.modal.toml"]),
            " (workspace ",
            _mk("code", {}, ["frost-byte"]),
            "). Run ",
            _mk("code", {}, ["modal token new"]),
            " to authenticate.",
        ]),
    );

    // Cleanup
    pane._llmpCleanup = () => {
        clearInterval(_idleTimer);
        if (_warmInterval) clearInterval(_warmInterval);
        document.removeEventListener("fbt:llm-status", _syncUI);
    };
}

// ── Gemini sub-tab ────────────────────────────────────────────────────────────

function _renderGeminiTab(pane) {
    const dot   = _mk("div", { cls: "llmp-status-dot warn" });
    const label = _mk("span", { cls: "llmp-status-text" }, ["Status unknown — check server environment"]);
    const status = _mk("div", { cls: "llmp-status" }, [dot, label]);

    // Check activity log for recent Gemini usage as a proxy for "configured"
    vlmActivityApi.recent({ n: 5, backend: "gemini" }).then(res => {
        if ((res.entries || []).length > 0) {
            dot.className   = "llmp-status-dot ok";
            label.textContent = "Key appears configured (recent activity found)";
        }
    }).catch(() => {});

    pane.append(
        status,
        _mk("div", { cls: "llmp-info" }, [
            _mk("strong", {}, ["Setup:"]),
            " Set the ",
            _mk("code", {}, ["GEMINI_API_KEY"]),
            " environment variable before starting ComfyUI, then restart if already running.",
            _mk("br", {}),
            _mk("br", {}),
            "On Linux/Mac: add ",
            _mk("code", {}, ["export GEMINI_API_KEY=your-key"]),
            " to your shell profile (",
            _mk("code", {}, ["~/.bashrc"]),
            " or ",
            _mk("code", {}, ["~/.zshrc"]),
            "). On Windows: set it in System → Environment Variables.",
            _mk("br", {}),
            _mk("br", {}),
            "The key is read exclusively server-side — it is never sent from or stored in the browser.",
        ]),
        _mk("div", { cls: "llmp-info", style: { borderLeftColor: "#f59e0b" } }, [
            _mk("strong", {}, ["Model:"]),
            " gemini-2.0-flash-lite (via the Gemini API, free tier available).",
            " Gemini is the last-resort backend — used only when Modal is inactive and no local vision model is loaded.",
        ]),
    );
}

// ── Panel entry point ─────────────────────────────────────────────────────────

export function renderLlmPanel(container) {
    _injectCSS();

    const wrap = _mk("div", { cls: "llmp-wrap" });
    container.appendChild(wrap);

    const TABS = [
        { id: "local",  label: "Local",  dot: "ok",   render: _renderLocalTab },
        { id: "modal",  label: "Modal",  dot: "blue", render: _renderModalTab },
        { id: "gemini", label: "Gemini", dot: "warn", render: _renderGeminiTab },
    ];

    const strip = _mk("div", { cls: "llmp-tabs" });
    const panes = {};
    const btns  = {};
    let   activeId = "local";

    TABS.forEach((tab, i) => {
        const dotEl = _mk("span", { cls: `llmp-tab-dot` });
        const btn   = _mk("button", {
            cls: "llmp-tab" + (i === 0 ? " active" : ""),
            onclick: () => activate(tab.id),
        }, [dotEl, tab.label]);
        strip.appendChild(btn);
        btns[tab.id] = { btn, dot: dotEl };

        const pane = _mk("div", { cls: "llmp-pane" + (i === 0 ? " active" : "") });
        wrap.appendChild(pane);
        panes[tab.id] = pane;
    });

    function _refreshTabDots() {
        const llm = window._fbtGetLlmStatus?.() || {};
        btns.local.dot.className  = "llmp-tab-dot" + (llm.loaded && llm.vision ? " ok" : "");
        btns.modal.dot.className  = "llmp-tab-dot" + (_state.modalActive ? " blue" : "");
        btns.gemini.dot.className = "llmp-tab-dot";  // no good way to check without an endpoint
    }

    _refreshTabDots();
    document.addEventListener("fbt:llm-status", _refreshTabDots);
    onBackendChange(_refreshTabDots);

    const mounted = {};
    function activate(id) {
        Object.values(btns).forEach(({ btn }) => btn.classList.remove("active"));
        Object.values(panes).forEach(p => p.classList.remove("active"));
        btns[id].btn.classList.add("active");
        panes[id].classList.add("active");
        activeId = id;
        if (!mounted[id]) {
            TABS.find(t => t.id === id).render(panes[id]);
            mounted[id] = true;
        }
    }

    activate("local");
    wrap.prepend(strip);
}
