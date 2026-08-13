/**
 * SceneCastBuild node — interactive slot table.
 *
 * All 16 standard widgets (subject_N, bundle_N, visual_mode_N, use_audio_N)
 * are hidden via setWidgetVisible so the node presents only a compact 4-row
 * table. The table's selects/toggles/checkbox write back to those hidden
 * widgets so ComfyUI serialises them with the workflow.
 *
 * node._refreshCastTable(entries?) is exposed for the cast editor's
 * "Send to Workflow" button to call after pushing widget values.
 */

import { setWidgetVisible } from "../utils/widgets.js";
import { bundlesApi }       from "../api/bundles.js";

const SLOTS       = 4;
const TARGET_TYPE = "fbt_SceneCastBuild";

// ── CSS (injected once) ────────────────────────────────────────────────────────

let _cssInjected = false;
function _injectCss() {
    if (_cssInjected) return;
    _cssInjected = true;
    const s = document.createElement("style");
    s.textContent = `
.fbt-scb-wrap {
    width: 100%;
    padding: 4px 6px 6px;
    box-sizing: border-box;
}
.fbt-scb-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 11px;
    table-layout: fixed;
}
.fbt-scb-table thead th {
    padding: 2px 4px;
    color: var(--p-surface-400, #888);
    font-weight: 500;
    text-align: left;
    white-space: nowrap;
}
.fbt-scb-table thead th.fbt-scb-center {
    text-align: center;
}
.fbt-scb-table tbody tr {
    border-top: 1px solid var(--border-color, #333);
    transition: opacity 0.15s;
}
.fbt-scb-table tbody tr.fbt-scb-empty {
    opacity: 0.38;
}
.fbt-scb-table tbody td {
    padding: 3px 4px;
    vertical-align: middle;
}
.fbt-scb-table tbody td.fbt-scb-center {
    text-align: center;
}
.fbt-scb-slot-num {
    color: var(--p-surface-400, #888);
    font-size: 10px;
    width: 16px;
}
.fbt-scb-sel {
    width: 100%;
    background: var(--comfy-input-bg, #222);
    border: 1px solid var(--border-color, #444);
    border-radius: 3px;
    color: inherit;
    font-size: 11px;
    padding: 2px 3px;
    box-sizing: border-box;
    cursor: pointer;
}
.fbt-scb-sel:focus {
    outline: none;
    border-color: var(--p-blue-400, #60a5fa);
}
.fbt-scb-mode {
    display: inline-flex;
    border-radius: 3px;
    overflow: hidden;
    border: 1px solid var(--border-color, #444);
}
.fbt-scb-mode-btn {
    padding: 2px 5px;
    font-size: 10px;
    background: var(--comfy-input-bg, #222);
    border: none;
    color: var(--p-surface-400, #888);
    cursor: pointer;
    line-height: 1.4;
}
.fbt-scb-mode-btn.active {
    background: var(--p-blue-700, #1d4ed8);
    color: #fff;
}
.fbt-scb-audio {
    width: 14px;
    height: 14px;
    cursor: pointer;
    accent-color: var(--p-blue-400, #60a5fa);
}
`;
    document.head.appendChild(s);
}

// ── Node setup ────────────────────────────────────────────────────────────────

export function setupSceneCastBuild(nodeType) {
    _injectCss();

    const _orig = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        _orig?.call(this);
        _buildCastBuildUI(this);
    };
}

function _buildCastBuildUI(node) {
    // ── 1. Hide all standard widgets ──────────────────────────────────────────
    const _w = (name) => node.widgets?.find(w => w.name === name);
    for (let i = 1; i <= SLOTS; i++) {
        setWidgetVisible(_w(`subject_${i}`),     false);
        setWidgetVisible(_w(`bundle_${i}`),      false);
        setWidgetVisible(_w(`visual_mode_${i}`), false);
        setWidgetVisible(_w(`use_audio_${i}`),   false);
    }

    // ── 2. State for subject / bundle lists ───────────────────────────────────
    let _subjects = [];   // [{id, name}]
    let _bundles  = [];   // [{id, name, subject_id}]

    // Per-slot DOM references for refresh
    const _rows = [];   // [{subjSel, bundSel, imgBtn, vidBtn, audCb}]

    // ── 3. Build DOM widget ───────────────────────────────────────────────────
    const wrap = document.createElement("div");
    wrap.className = "fbt-scb-wrap";

    const table = document.createElement("table");
    table.className = "fbt-scb-table";

    // Header
    const thead = document.createElement("thead");
    thead.innerHTML = `<tr>
        <th class="fbt-scb-slot-num"></th>
        <th style="width:30%">Subject</th>
        <th style="width:34%">Bundle</th>
        <th class="fbt-scb-center" style="width:52px">Mode</th>
        <th class="fbt-scb-center" style="width:28px">Aud</th>
    </tr>`;
    table.appendChild(thead);

    const tbody = document.createElement("tbody");
    table.appendChild(tbody);
    wrap.appendChild(table);

    // ── 4. Build row helpers ──────────────────────────────────────────────────

    function _fillSubjectSel(sel, currentId) {
        sel.innerHTML = "";
        const blank = document.createElement("option");
        blank.value = "";
        blank.textContent = "— subject —";
        if (!currentId) blank.selected = true;
        sel.appendChild(blank);
        _subjects.forEach(s => {
            const o = document.createElement("option");
            o.value = s.id;
            o.textContent = s.name || s.id;
            if (s.id === currentId) o.selected = true;
            sel.appendChild(o);
        });
    }

    function _fillBundleSel(sel, subjectId, currentBundleId) {
        sel.innerHTML = "";
        const blank = document.createElement("option");
        blank.value = "";
        blank.textContent = "— bundle —";
        if (!currentBundleId) blank.selected = true;
        sel.appendChild(blank);
        const available = subjectId
            ? _bundles.filter(b => !b.subject_id || b.subject_id === subjectId)
            : _bundles;
        available.forEach(b => {
            const o = document.createElement("option");
            o.value = b.id;
            o.textContent = b.name || b.id;
            if (b.id === currentBundleId) o.selected = true;
            sel.appendChild(o);
        });
    }

    function _buildRow(slotIdx) {
        const i = slotIdx + 1;  // 1-based for widget names

        const tr = document.createElement("tr");

        // Slot number cell
        const numTd = document.createElement("td");
        numTd.className = "fbt-scb-slot-num";
        numTd.textContent = String(i);
        tr.appendChild(numTd);

        // Subject select
        const subjTd = document.createElement("td");
        const subjSel = document.createElement("select");
        subjSel.className = "fbt-scb-sel";
        _fillSubjectSel(subjSel, _w(`subject_${i}`)?.value || "");
        subjTd.appendChild(subjSel);
        tr.appendChild(subjTd);

        // Bundle select
        const bundTd = document.createElement("td");
        const bundSel = document.createElement("select");
        bundSel.className = "fbt-scb-sel";
        _fillBundleSel(bundSel, subjSel.value, _w(`bundle_${i}`)?.value || "");
        bundTd.appendChild(bundSel);
        tr.appendChild(bundTd);

        // Mode toggle
        const modeTd = document.createElement("td");
        modeTd.className = "fbt-scb-center";
        const modeWrap = document.createElement("div");
        modeWrap.className = "fbt-scb-mode";
        const imgBtn = document.createElement("button");
        imgBtn.className = "fbt-scb-mode-btn";
        imgBtn.textContent = "Img";
        imgBtn.title = "Images mode";
        const vidBtn = document.createElement("button");
        vidBtn.className = "fbt-scb-mode-btn";
        vidBtn.textContent = "Vid";
        vidBtn.title = "Video mode";
        const curMode = _w(`visual_mode_${i}`)?.value || "images";
        imgBtn.classList.toggle("active", curMode === "images");
        vidBtn.classList.toggle("active", curMode === "video");
        modeWrap.appendChild(imgBtn);
        modeWrap.appendChild(vidBtn);
        modeTd.appendChild(modeWrap);
        tr.appendChild(modeTd);

        // Audio checkbox
        const audTd = document.createElement("td");
        audTd.className = "fbt-scb-center";
        const audCb = document.createElement("input");
        audCb.type = "checkbox";
        audCb.className = "fbt-scb-audio";
        audCb.title = "Use audio reference";
        audCb.checked = !!(_w(`use_audio_${i}`)?.value);
        audTd.appendChild(audCb);
        tr.appendChild(audTd);

        // ── Wire events ───────────────────────────────────────────────────────

        const _syncEmpty = () => {
            const active = subjSel.value && bundSel.value;
            tr.classList.toggle("fbt-scb-empty", !active);
        };

        subjSel.addEventListener("change", () => {
            const sw = _w(`subject_${i}`);
            if (sw) sw.value = subjSel.value;
            // Refill bundles filtered to new subject
            _fillBundleSel(bundSel, subjSel.value, _w(`bundle_${i}`)?.value || "");
            _syncEmpty();
            app?.graph?.setDirtyCanvas(true, false);
        });

        bundSel.addEventListener("change", () => {
            const bw = _w(`bundle_${i}`);
            if (bw) bw.value = bundSel.value;
            _syncEmpty();
            app?.graph?.setDirtyCanvas(true, false);
        });

        imgBtn.addEventListener("click", () => {
            const mw = _w(`visual_mode_${i}`);
            if (mw) mw.value = "images";
            imgBtn.classList.add("active");
            vidBtn.classList.remove("active");
            app?.graph?.setDirtyCanvas(true, false);
        });

        vidBtn.addEventListener("click", () => {
            const mw = _w(`visual_mode_${i}`);
            if (mw) mw.value = "video";
            vidBtn.classList.add("active");
            imgBtn.classList.remove("active");
            app?.graph?.setDirtyCanvas(true, false);
        });

        audCb.addEventListener("change", () => {
            const aw = _w(`use_audio_${i}`);
            if (aw) aw.value = audCb.checked;
            app?.graph?.setDirtyCanvas(true, false);
        });

        _syncEmpty();
        tbody.appendChild(tr);
        _rows[slotIdx] = { subjSel, bundSel, imgBtn, vidBtn, audCb };
    }

    for (let s = 0; s < SLOTS; s++) _buildRow(s);

    // ── 5. Add DOM widget ─────────────────────────────────────────────────────
    const displayWidget = node.addDOMWidget("cast_build_table", "preview", wrap, {
        serialize: false,
        hideOnZoom: false,
        getValue() { return null; },
        setValue() {},
    });
    displayWidget.computeSize = () => [0, 110];

    // ── 6. Refresh function (called by cast editor "Send to Workflow") ────────

    node._refreshCastTable = function (entries) {
        // entries: optional [{subject_id, bundle_id, visual_mode, use_audio}]
        // If omitted, re-reads from hidden widget values.
        for (let s = 0; s < SLOTS; s++) {
            const i = s + 1;
            const r = _rows[s];
            if (!r) continue;

            let subjectId  = _w(`subject_${i}`)?.value || "";
            let bundleId   = _w(`bundle_${i}`)?.value  || "";
            let visualMode = _w(`visual_mode_${i}`)?.value || "images";
            let useAudio   = !!(_w(`use_audio_${i}`)?.value);

            if (entries && s < entries.length) {
                subjectId  = entries[s].subject_id  || "";
                bundleId   = entries[s].bundle_id   || "";
                visualMode = entries[s].visual_mode || "images";
                useAudio   = !!entries[s].use_audio;
            }

            _fillSubjectSel(r.subjSel, subjectId);
            _fillBundleSel(r.bundSel, subjectId, bundleId);
            r.imgBtn.classList.toggle("active", visualMode === "images");
            r.vidBtn.classList.toggle("active", visualMode === "video");
            r.audCb.checked = useAudio;

            const active = subjectId && bundleId;
            r.subjSel.closest("tr").classList.toggle("fbt-scb-empty", !active);
        }
    };

    // ── 7. Load subject + bundle lists from API ───────────────────────────────
    Promise.allSettled([
        bundlesApi.listSubjects(),
        bundlesApi.listBundles(),
    ]).then(([subjRes, bundRes]) => {
        _subjects = subjRes.value?.subjects ?? [];
        _bundles  = bundRes.value?.bundles  ?? [];
        // Re-render selects with live data, preserving current widget values
        node._refreshCastTable();
    }).catch(() => {});

    // ── 8. Minimum node size ──────────────────────────────────────────────────
    node.size[0] = Math.max(node.size[0], 360);
    node.size[1] = Math.max(node.size[1], 160);
}
