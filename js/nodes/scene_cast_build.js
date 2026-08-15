/**
 * SceneCastBuild node — JSON-backed interactive entry table.
 *
 * One hidden STRING widget (cast_entries_json) serialises all entries as a
 * JSON array.  The DOM widget renders a table with +/− row controls so the
 * user can add or remove entries freely (no fixed-slot limit).
 *
 * node._refreshCastTable(entries?) is exposed for the cast editor's
 * "Send to Workflow" button to call after pushing new entries.
 */

import { setWidgetVisible } from "../utils/widgets.js";
import { bundlesApi }       from "../api/bundles.js";

const JSON_WIDGET = "cast_entries_json";
const MAX_ENTRIES = 8;

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
.fbt-scb-table thead th.fbt-scb-c { text-align: center; }
.fbt-scb-table tbody tr {
    border-top: 1px solid var(--border-color, #333);
}
.fbt-scb-table tbody tr.fbt-scb-empty { opacity: 0.38; }
.fbt-scb-table tbody td {
    padding: 3px 4px;
    vertical-align: middle;
}
.fbt-scb-table tbody td.fbt-scb-c { text-align: center; }
.fbt-scb-row-num {
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
.fbt-scb-rm-btn {
    background: transparent;
    border: none;
    color: var(--p-red-400, #f87171);
    cursor: pointer;
    font-size: 12px;
    padding: 0 2px;
    line-height: 1;
}
.fbt-scb-rm-btn:hover { color: var(--p-red-300, #fca5a5); }
.fbt-scb-preview-btn {
    background: transparent;
    border: none;
    color: var(--p-surface-400, #888);
    cursor: pointer;
    font-size: 12px;
    padding: 0 2px;
    line-height: 1;
}
.fbt-scb-preview-btn:hover { color: var(--p-blue-400, #60a5fa); }
.fbt-scb-preview-btn.active { color: var(--p-blue-400, #60a5fa); }
.fbt-scb-preview-row td {
    padding: 4px 6px 8px 22px;
    background: var(--comfy-menu-bg, #111);
}
.fbt-scb-preview-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-items: flex-start;
}
.fbt-scb-thumb {
    width: 72px;
    height: 72px;
    object-fit: cover;
    border-radius: 3px;
    border: 1px solid var(--border-color, #333);
    background: #000;
    cursor: zoom-in;
}
.fbt-scb-preview-video {
    max-width: 100%;
    max-height: 120px;
    border-radius: 3px;
    border: 1px solid var(--border-color, #333);
    background: #000;
    flex-basis: 100%;
    margin-top: 4px;
}
.fbt-scb-preview-audio {
    width: 100%;
    height: 28px;
    margin-top: 4px;
    flex-basis: 100%;
}
.fbt-scb-preview-note {
    color: var(--p-surface-400, #888);
    font-size: 10px;
    font-style: italic;
}
.fbt-scb-add-btn {
    margin-top: 5px;
    width: 100%;
    padding: 3px 0;
    background: transparent;
    border: 1px dashed var(--border-color, #444);
    border-radius: 3px;
    color: var(--p-blue-400, #60a5fa);
    font-size: 11px;
    cursor: pointer;
}
.fbt-scb-add-btn:hover { border-color: var(--p-blue-400, #60a5fa); }
`;
    document.head.appendChild(s);
}

// ── Node setup ────────────────────────────────────────────────────────────────

export function setupSceneCastBuild(nodeType, _nodeData, app) {
    _injectCss();

    const _origCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        _origCreated?.call(this);
        _buildCastBuildUI(this, app);
    };

    // onConfigure fires after widget values are restored from the saved workflow.
    // Rebuild the table so saved entries are visible on reload.
    const _origConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (config) {
        _origConfigure?.call(this, config);
        this._refreshCastTable?.();
    };
}

function _buildCastBuildUI(node, app) {
    // ── 1. Find and hide the single JSON backing widget ───────────────────────
    // We use a regular STRING widget (not boolean) so setWidgetVisible works
    // reliably regardless of ComfyUI version quirks.
    const jsonWidget = node.widgets?.find(w => w.name === JSON_WIDGET);
    if (jsonWidget) setWidgetVisible(jsonWidget, false, node);

    // ── 2. Internal state ─────────────────────────────────────────────────────
    let _subjects = [];
    let _bundles  = [];

    // Parse initial entries from the widget value (populated from saved workflow)
    let _entries = [];
    try {
        const parsed = JSON.parse(jsonWidget?.value || "[]");
        if (Array.isArray(parsed)) _entries = parsed;
    } catch { /* leave empty */ }

    // tmp frame filenames keyed by bundle_id — cleaned up on collapse/rebuild
    const _tmpFrames = new Map();

    // ── 3. Build DOM structure ────────────────────────────────────────────────
    const wrap = document.createElement("div");
    wrap.className = "fbt-scb-wrap";

    const table = document.createElement("table");
    table.className = "fbt-scb-table";

    const thead = document.createElement("thead");
    thead.innerHTML = `<tr>
        <th class="fbt-scb-row-num"></th>
        <th style="width:27%">Subject</th>
        <th style="width:30%">Bundle</th>
        <th class="fbt-scb-c" style="width:50px">Mode</th>
        <th class="fbt-scb-c" style="width:26px">Aud</th>
        <th style="width:34px"></th>
    </tr>`;
    table.appendChild(thead);

    const tbody = document.createElement("tbody");
    table.appendChild(tbody);

    const addBtn = document.createElement("button");
    addBtn.className = "fbt-scb-add-btn";
    addBtn.textContent = "+ Add entry";
    addBtn.addEventListener("click", () => {
        if (_entries.length >= MAX_ENTRIES) return;
        _entries.push({ subject_id: "", bundle_id: "", visual_mode: "images", use_audio: false });
        _rebuildTable();
        _syncWidget();
    });

    wrap.appendChild(table);
    wrap.appendChild(addBtn);

    // ── 4. Select helpers ─────────────────────────────────────────────────────

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

    // ── 5. Preview helpers ────────────────────────────────────────────────────

    function _buildPreviewRowEl() {
        const tr = document.createElement("tr");
        tr.className = "fbt-scb-preview-row";
        tr.style.display = "none";
        const td = document.createElement("td");
        td.colSpan = 6;
        tr.appendChild(td);
        return tr;
    }

    async function _loadPreviewContent(td, entry) {
        if (!entry.bundle_id) {
            td.innerHTML = '<span class="fbt-scb-preview-note">No bundle selected.</span>';
            return;
        }
        td.innerHTML = '<span class="fbt-scb-preview-note">Loading…</span>';
        try {
            const bundle = await bundlesApi.getBundle(entry.bundle_id);
            const strip = document.createElement("div");
            strip.className = "fbt-scb-preview-strip";

            if (entry.visual_mode === "video") {
                const file = bundle.visual?.file || "";
                if (file) {
                    try {
                        const frameData = await bundlesApi.extractFrame(file, 0);
                        const prev = _tmpFrames.get(entry.bundle_id);
                        if (prev) bundlesApi.deleteTmpFrame(prev);
                        _tmpFrames.set(entry.bundle_id, frameData.tmp_filename);
                        const img = document.createElement("img");
                        img.className = "fbt-scb-thumb";
                        img.src = `/view?filename=${encodeURIComponent(frameData.tmp_filename)}&type=input`;
                        img.title = `${file}\n${frameData.width}×${frameData.height}, ${frameData.frame_count} frames`;
                        strip.appendChild(img);
                        const note = document.createElement("span");
                        note.className = "fbt-scb-preview-note";
                        note.style.alignSelf = "center";
                        note.textContent = `${file} — ${frameData.width}×${frameData.height}, ${frameData.frame_count} frames`;
                        strip.appendChild(note);
                    } catch {
                        const note = document.createElement("span");
                        note.className = "fbt-scb-preview-note";
                        note.textContent = `${file} (frame extract unavailable)`;
                        strip.appendChild(note);
                    }
                } else {
                    strip.innerHTML = '<span class="fbt-scb-preview-note">No video file in bundle.</span>';
                }
            } else {
                const files = bundle.visual?.files || [];
                if (files.length) {
                    files.forEach(f => {
                        const img = document.createElement("img");
                        img.className = "fbt-scb-thumb";
                        img.src = `/view?filename=${encodeURIComponent(f)}&type=input`;
                        img.title = f;
                        img.loading = "lazy";
                        strip.appendChild(img);
                    });
                } else {
                    strip.innerHTML = '<span class="fbt-scb-preview-note">No images in bundle.</span>';
                }
            }

            // Audio player — show regardless of use_audio so user can verify the file
            const audioSrc = bundle.audio?.source;
            const audioFile = bundle.audio?.file || "";
            if (audioFile && audioSrc !== "none" && audioSrc !== "extract_from_visual") {
                const aud = document.createElement("audio");
                aud.className = "fbt-scb-preview-audio";
                aud.src = `/view?filename=${encodeURIComponent(audioFile)}&type=input`;
                aud.controls = true;
                aud.preload = "none";
                aud.title = audioFile;
                strip.appendChild(aud);
            }

            // When audio is extracted from the video itself, add a video player so the user can hear it
            if (audioSrc === "extract_from_visual" && entry.visual_mode === "video") {
                const videoFile = bundle.visual?.file || "";
                if (videoFile) {
                    const vid = document.createElement("video");
                    vid.className = "fbt-scb-preview-video";
                    vid.src = `/view?filename=${encodeURIComponent(videoFile)}&type=input`;
                    vid.controls = true;
                    vid.preload = "none";
                    vid.title = `${videoFile} — play to hear extracted audio`;
                    strip.appendChild(vid);
                }
            }

            td.innerHTML = "";
            td.appendChild(strip);
        } catch (err) {
            td.innerHTML = `<span class="fbt-scb-preview-note">Error: ${err.message}</span>`;
        }
    }

    async function _togglePreview(idx, previewRow, previewBtn) {
        const entry = _entries[idx];
        const td = previewRow.querySelector("td");
        const isOpen = previewRow.style.display !== "none";
        if (isOpen) {
            previewRow.style.display = "none";
            previewBtn.classList.remove("active");
            const tmp = _tmpFrames.get(entry.bundle_id);
            if (tmp) { bundlesApi.deleteTmpFrame(tmp); _tmpFrames.delete(entry.bundle_id); }
        } else {
            previewRow.style.display = "";
            previewBtn.classList.add("active");
            await _loadPreviewContent(td, entry);
        }
    }

    // ── 6. Build a single table row ───────────────────────────────────────────

    function _buildRow(idx, previewRow) {
        const entry = _entries[idx];
        const tr = document.createElement("tr");
        tr.classList.toggle("fbt-scb-empty", !entry.subject_id || !entry.bundle_id);

        // Row number
        const numTd = document.createElement("td");
        numTd.className = "fbt-scb-row-num";
        numTd.textContent = String(idx + 1);
        tr.appendChild(numTd);

        // Subject select
        const subjTd = document.createElement("td");
        const subjSel = document.createElement("select");
        subjSel.className = "fbt-scb-sel";
        _fillSubjectSel(subjSel, entry.subject_id);
        subjTd.appendChild(subjSel);
        tr.appendChild(subjTd);

        // Bundle select
        const bundTd = document.createElement("td");
        const bundSel = document.createElement("select");
        bundSel.className = "fbt-scb-sel";
        _fillBundleSel(bundSel, entry.subject_id, entry.bundle_id);
        bundTd.appendChild(bundSel);
        tr.appendChild(bundTd);

        // Mode toggle
        const modeTd = document.createElement("td");
        modeTd.className = "fbt-scb-c";
        const modeWrap = document.createElement("div");
        modeWrap.className = "fbt-scb-mode";
        const imgBtn = document.createElement("button");
        imgBtn.className = "fbt-scb-mode-btn" + (entry.visual_mode !== "video" ? " active" : "");
        imgBtn.textContent = "Img";
        imgBtn.title = "Images";
        const vidBtn = document.createElement("button");
        vidBtn.className = "fbt-scb-mode-btn" + (entry.visual_mode === "video" ? " active" : "");
        vidBtn.textContent = "Vid";
        vidBtn.title = "Video";
        modeWrap.appendChild(imgBtn);
        modeWrap.appendChild(vidBtn);
        modeTd.appendChild(modeWrap);
        tr.appendChild(modeTd);

        // Audio checkbox
        const audTd = document.createElement("td");
        audTd.className = "fbt-scb-c";
        const audCb = document.createElement("input");
        audCb.type = "checkbox";
        audCb.className = "fbt-scb-audio";
        audCb.title = "Use audio reference";
        audCb.checked = !!entry.use_audio;
        audTd.appendChild(audCb);
        tr.appendChild(audTd);

        // Preview + Remove buttons
        const rmTd = document.createElement("td");
        rmTd.style.whiteSpace = "nowrap";

        const previewBtn = document.createElement("button");
        previewBtn.className = "fbt-scb-preview-btn";
        previewBtn.textContent = "⊙";
        previewBtn.title = "Preview media";
        previewBtn.addEventListener("click", () => _togglePreview(idx, previewRow, previewBtn));
        rmTd.appendChild(previewBtn);

        const rmBtn = document.createElement("button");
        rmBtn.className = "fbt-scb-rm-btn";
        rmBtn.textContent = "✕";
        rmBtn.title = "Remove entry";
        rmBtn.addEventListener("click", () => {
            // Clean up any open tmp frame for this entry
            const tmp = _tmpFrames.get(entry.bundle_id);
            if (tmp) { bundlesApi.deleteTmpFrame(tmp); _tmpFrames.delete(entry.bundle_id); }
            _entries.splice(idx, 1);
            _rebuildTable();
            _syncWidget();
        });
        rmTd.appendChild(rmBtn);
        tr.appendChild(rmTd);

        // ── Events ────────────────────────────────────────────────────────────

        subjSel.addEventListener("change", () => {
            entry.subject_id = subjSel.value;
            // Clear bundle if it belongs to a different subject
            const cur = _bundles.find(b => b.id === entry.bundle_id);
            if (cur?.subject_id && cur.subject_id !== entry.subject_id) {
                entry.bundle_id = "";
            }
            _fillBundleSel(bundSel, entry.subject_id, entry.bundle_id);
            tr.classList.toggle("fbt-scb-empty", !entry.subject_id || !entry.bundle_id);
            _syncWidget();
        });

        bundSel.addEventListener("change", () => {
            entry.bundle_id = bundSel.value;
            // Auto-inherit bundle's visual type
            const bun = _bundles.find(b => b.id === entry.bundle_id);
            if (bun?.visual?.type) {
                entry.visual_mode = bun.visual.type;
                imgBtn.classList.toggle("active", entry.visual_mode !== "video");
                vidBtn.classList.toggle("active", entry.visual_mode === "video");
            }
            tr.classList.toggle("fbt-scb-empty", !entry.subject_id || !entry.bundle_id);
            _syncWidget();
        });

        imgBtn.addEventListener("click", () => {
            entry.visual_mode = "images";
            imgBtn.classList.add("active");
            vidBtn.classList.remove("active");
            _syncWidget();
        });

        vidBtn.addEventListener("click", () => {
            entry.visual_mode = "video";
            vidBtn.classList.add("active");
            imgBtn.classList.remove("active");
            _syncWidget();
        });

        audCb.addEventListener("change", () => {
            entry.use_audio = audCb.checked;
            _syncWidget();
        });

        return tr;
    }

    // ── 7. Rebuild the whole tbody ────────────────────────────────────────────

    function _rebuildTable() {
        // Clean up any extracted video frames still open
        _tmpFrames.forEach(tmp => bundlesApi.deleteTmpFrame(tmp));
        _tmpFrames.clear();

        tbody.innerHTML = "";
        _entries.forEach((_, i) => {
            const previewRow = _buildPreviewRowEl();
            const entryRow = _buildRow(i, previewRow);
            tbody.appendChild(entryRow);
            tbody.appendChild(previewRow);
        });
        addBtn.style.display = _entries.length >= MAX_ENTRIES ? "none" : "";
        // Update DOM widget height
        if (displayWidget) {
            displayWidget.computeSize = () => [0, _tableHeight()];
            node.setSize?.([node.size[0], node.size[1]]);
        }
    }

    function _tableHeight() {
        return Math.max(70, 28 + _entries.length * 28 + 26);
    }

    // ── 8. Sync entries → hidden widget ───────────────────────────────────────

    function _syncWidget() {
        if (jsonWidget) jsonWidget.value = JSON.stringify(_entries);
        app?.graph?.setDirtyCanvas?.(true, false);
    }

    // ── 9. Add DOM widget ─────────────────────────────────────────────────────
    let displayWidget = null;
    displayWidget = node.addDOMWidget("cast_build_table", "preview", wrap, {
        serialize: false,
        hideOnZoom: false,
        getValue() { return null; },
        setValue() {},
    });
    displayWidget.computeSize = () => [0, _tableHeight()];

    // ── 10. Public refresh (called by cast editor "Send to Workflow") ─────────

    node._refreshCastTable = function (newEntries) {
        if (Array.isArray(newEntries)) {
            // Called by cast editor "Send to Workflow" — use the provided entries
            _entries = newEntries.map(e => ({ ...e }));
            _syncWidget();
        } else {
            // Called after workflow load (onConfigure) — re-read from widget
            try {
                const parsed = JSON.parse(jsonWidget?.value || "[]");
                if (Array.isArray(parsed)) _entries = parsed;
            } catch { /* leave as-is */ }
        }
        _rebuildTable();
    };

    // ── 11. Load subject + bundle lists, then render ─────────────────────────
    Promise.allSettled([
        bundlesApi.listSubjects(),
        bundlesApi.listBundles(),
    ]).then(([subjRes, bundRes]) => {
        _subjects = subjRes.value?.subjects ?? [];
        _bundles  = bundRes.value?.bundles  ?? [];
        _rebuildTable();
    }).catch(() => {
        _rebuildTable();
    });

    node.size[0] = Math.max(node.size[0], 370);
    node.size[1] = Math.max(node.size[1], _tableHeight() + 60);
}
