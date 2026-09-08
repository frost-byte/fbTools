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


// ── Node setup ────────────────────────────────────────────────────────────────

export function setupSceneCastBuild(nodeType, _nodeData, app) {
    const _origCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        _origCreated?.call(this);
        _buildCastBuildUI(this, app);
    };

    // onConfigure fires after widget values are restored from the saved workflow.
    const _origConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (config) {
        _origConfigure?.call(this, config);
        this._refreshCastTable?.();
        requestAnimationFrame(() => {
            this._refreshClipSelects?.();
            this._refreshSourceSubjects?.();
        });
    };

    // Refresh clip dropdowns and source subject column whenever a source_profile input changes.
    const _origConnChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (type, index, connected, linkInfo) {
        _origConnChange?.call(this, type, index, connected, linkInfo);
        if (type === LiteGraph?.INPUT) {
            const inp = this.inputs?.[index];
            if (inp?.name === "source_profile") {
                requestAnimationFrame(() => {
                    this._refreshClipSelects?.();
                    this._refreshSourceSubjects?.();
                });
            }
        }
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
    // [{pid, label, subjects:[{id,label}]}] — populated by _refreshSourceSubjects
    let _connectedSPSubjects = [];
    // true unless the selected clip explicitly disallows dialogue
    let _clipAllowsDialogue = true;
    // ID of the currently selected clip (mirrors clipWidget.value)
    let _activeClipId = "";

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
        <th style="width:18%">Subject</th>
        <th style="width:20%">Bundle</th>
        <th class="fbt-scb-src-th" style="width:18%;display:none">Source</th>
        <th class="fbt-scb-c" style="width:80px">Mode</th>
        <th class="fbt-scb-c" style="width:26px">Aud</th>
        <th style="width:22%" title="Dialogue text. Prefixes: [silent] = no audio; [sounds] desc = sound event. Use %libber:key% for libber lookup.">Dlg</th>
        <th style="width:34px"></th>
    </tr>`;
    table.appendChild(thead);
    const srcTh = thead.querySelector(".fbt-scb-src-th");

    const tbody = document.createElement("tbody");
    table.appendChild(tbody);

    const addBtn = document.createElement("button");
    addBtn.className = "fbt-scb-add-btn";
    addBtn.textContent = "+ Add entry";
    addBtn.addEventListener("click", () => {
        if (_entries.length >= MAX_ENTRIES) return;
        _entries.push({ subject_id: "", bundle_id: "", visual_mode: "images", use_audio: false, dialogue: "" });
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

    function _fillSourceSubjectSel(sel, currentProfileId, currentSubjectId) {
        sel.innerHTML = "";
        const blank = document.createElement("option");
        blank.value = "";
        blank.textContent = "— (none) —";
        if (!currentProfileId || !currentSubjectId) blank.selected = true;
        sel.appendChild(blank);

        // If a clip is selected and has tagged subjects, restrict to those only.
        const selectedClipId = _activeClipId;
        const clip = selectedClipId ? _clipMap?.get(selectedClipId) : null;
        const clipSubjectIds = (clip && Array.isArray(clip.subjects) && clip.subjects.length)
            ? new Set(clip.subjects)
            : null;  // null = no filter (no clip selected, or clip has no tagged subjects)

        _connectedSPSubjects.forEach((sp, i) => {
            const visibleSubjects = clipSubjectIds
                ? sp.subjects.filter(s => clipSubjectIds.has(s.id))
                : sp.subjects;
            if (!visibleSubjects.length) return;
            const grp = document.createElement("optgroup");
            grp.label = sp.label || `SP${i + 1}`;
            visibleSubjects.forEach(s => {
                const o = document.createElement("option");
                o.value = `${sp.pid}::${s.id}`;
                o.textContent = s.label || s.id;
                if (sp.pid === currentProfileId && s.id === currentSubjectId) o.selected = true;
                grp.appendChild(o);
            });
            sel.appendChild(grp);
        });
    }

    // ── 5. Preview helpers ────────────────────────────────────────────────────

    function _buildPreviewRowEl() {
        const tr = document.createElement("tr");
        tr.className = "fbt-scb-preview-row";
        tr.style.display = "none";
        const td = document.createElement("td");
        td.colSpan = 99;
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
                const imageFiles = files
                    .map(f => (typeof f === "object" ? (f.file || "") : (f || "")))
                    .filter(Boolean);
                if (imageFiles.length) {
                    imageFiles.forEach(file => {
                        const img = document.createElement("img");
                        img.className = "fbt-scb-thumb";
                        const slashIdx = file.lastIndexOf("/");
                        const fname = slashIdx >= 0 ? file.slice(slashIdx + 1) : file;
                        const sfolder = slashIdx >= 0 ? file.slice(0, slashIdx) : "";
                        const _mkUrl = type => `/view?filename=${encodeURIComponent(fname)}${sfolder ? `&subfolder=${encodeURIComponent(sfolder)}` : ""}&type=${type}`;
                        img.src = _mkUrl("input");
                        img.onerror = () => { img.onerror = null; img.src = _mkUrl("output"); };
                        img.title = file;
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

        // Source Profile subject select (hidden when no profiles are connected)
        const srcTd = document.createElement("td");
        srcTd.className = "fbt-scb-src-td";
        srcTd.style.display = _connectedSPSubjects.length ? "" : "none";
        const srcSel = document.createElement("select");
        srcSel.className = "fbt-scb-sel fbt-scb-src-sel";
        _fillSourceSubjectSel(srcSel, entry.source_profile_id || "", entry.source_subject_id || "");
        srcTd.appendChild(srcSel);
        tr.appendChild(srcTd);

        // Mode toggle
        const modeTd = document.createElement("td");
        modeTd.className = "fbt-scb-c";
        const modeWrap = document.createElement("div");
        modeWrap.className = "fbt-scb-mode";
        const bun0 = _bundles.find(b => b.id === entry.bundle_id);
        const _hasImages = b => Boolean(b?.visual?.files?.length);
        const _hasVideo  = b => Boolean(b?.visual?.file);
        const _imgCount  = b => b?.visual?.files?.length ?? 0;

        if (!("image_selection" in entry)) entry.image_selection = null;

        // Img button — "use all images"
        const imgBtn = document.createElement("button");
        imgBtn.className = "fbt-scb-mode-btn";
        imgBtn.textContent = "Img";

        // Numbered sub-buttons (dynamic; always at least 2 slots rendered)
        const imgSubWrap = document.createElement("div");
        imgSubWrap.className = "fbt-scb-mode-num-wrap";

        // Vid button
        const vidBtn = document.createElement("button");
        vidBtn.className = "fbt-scb-mode-btn fbt-scb-mode-btn-vid";
        vidBtn.textContent = "Vid";

        modeWrap.appendChild(imgBtn);
        modeWrap.appendChild(imgSubWrap);
        modeWrap.appendChild(vidBtn);
        modeTd.appendChild(modeWrap);
        tr.appendChild(modeTd);

        // Sync active/disabled states across the whole button group
        const _syncModeActive = () => {
            const isVid = entry.visual_mode === "video";
            const sel   = entry.image_selection;
            imgBtn.classList.toggle("active", !isVid && sel == null);
            vidBtn.classList.toggle("active", isVid);
            imgSubWrap.querySelectorAll(".fbt-scb-mode-btn-num").forEach((btn, i) => {
                btn.classList.toggle("active", !isVid && sel === i);
            });
        };

        // Build (or rebuild) numbered image sub-buttons for the given bundle.
        // Always renders at least 2 slots; extras are disabled if the bundle has fewer images.
        const _buildNumBtns = bun => {
            imgSubWrap.innerHTML = "";
            const fileCount = _imgCount(bun);
            const numSlots  = Math.max(2, fileCount);
            for (let i = 0; i < numSlots; i++) {
                const btn = document.createElement("button");
                btn.className = "fbt-scb-mode-btn fbt-scb-mode-btn-num";
                btn.textContent = String(i + 1);
                const hasSlot = i < fileCount;
                const file    = bun?.visual?.files?.[i];
                const role    = (typeof file === "object" ? file?.role : null) || "character sheet";
                btn.title     = hasSlot ? `Image ${i + 1}: ${role}` : "No image at this slot";
                btn.disabled  = !hasSlot;
                btn.addEventListener("click", () => {
                    entry.visual_mode    = "images";
                    entry.image_selection = i;
                    _syncModeActive();
                    _syncWidget();
                });
                imgSubWrap.appendChild(btn);
            }
            const hasImg = fileCount > 0;
            imgBtn.disabled = bun && !hasImg;
            imgBtn.title    = hasImg ? "Use all image references" : "No images in bundle";
            vidBtn.disabled = bun && !_hasVideo(bun);
            vidBtn.title    = _hasVideo(bun) ? "Use video reference" : "No video in bundle";
            _syncModeActive();
        };

        _buildNumBtns(bun0);

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

        // Dialogue text input
        const dlgTd = document.createElement("td");
        const dlgInput = document.createElement("input");
        dlgInput.type = "text";
        dlgInput.className = "fbt-scb-dlg";
        dlgInput.value = entry.dialogue || "";
        dlgInput.placeholder = "text or %libber:key%";
        dlgInput.title = "Dialogue for this cast entry.\n[silent] = no audio contribution\n[sounds] desc = sound event\n%libber:key% or %libber:*% = libber lookup (% delimiters required)";
        dlgTd.appendChild(dlgInput);
        tr.appendChild(dlgTd);

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
            const bun = _bundles.find(b => b.id === entry.bundle_id);
            const hasImg = _hasImages(bun);
            const hasVid = _hasVideo(bun);
            // Default to the bundle's preferred mode; fall back if not available.
            if (bun?.visual?.type) {
                entry.visual_mode = bun.visual.type;
                if (entry.visual_mode === "video" && !hasVid && hasImg) entry.visual_mode = "images";
                if (entry.visual_mode === "images" && !hasImg && hasVid) entry.visual_mode = "video";
            }
            // If the previously-selected image index no longer exists, reset to "all".
            const fc = _imgCount(bun);
            if (entry.image_selection != null && entry.image_selection >= fc) {
                entry.image_selection = null;
            }
            _buildNumBtns(bun);
            tr.classList.toggle("fbt-scb-empty", !entry.subject_id || !entry.bundle_id);
            _syncWidget();
        });

        srcSel.addEventListener("change", () => {
            const val = srcSel.value;
            if (val) {
                const sep = val.indexOf("::");
                entry.source_profile_id = sep >= 0 ? val.slice(0, sep) : "";
                entry.source_subject_id  = sep >= 0 ? val.slice(sep + 2) : val;
            } else {
                delete entry.source_profile_id;
                delete entry.source_subject_id;
            }
            _syncWidget();
        });

        imgBtn.addEventListener("click", () => {
            entry.visual_mode    = "images";
            entry.image_selection = null;
            _syncModeActive();
            _syncWidget();
        });

        vidBtn.addEventListener("click", () => {
            entry.visual_mode = "video";
            _syncModeActive();
            _syncWidget();
        });

        audCb.addEventListener("change", () => {
            entry.use_audio = audCb.checked;
            _syncWidget();
        });

        dlgInput.addEventListener("input", () => {
            entry.dialogue = dlgInput.value;
            _syncWidget();
        });

        return tr;
    }

    // ── 7. Rebuild the whole tbody ────────────────────────────────────────────

    function _applyDlgState() {
        const disabled = !_clipAllowsDialogue;
        tbody.querySelectorAll(".fbt-scb-dlg").forEach(inp => {
            inp.disabled = disabled;
            inp.title = disabled
                ? "Dialogue disabled — this clip has 'Allows dialogue' turned off"
                : "Dialogue for this cast entry.\n[silent] = no audio contribution\n[sounds] desc = sound event\n%libber:key% = libber lookup";
            inp.style.opacity = disabled ? "0.35" : "";
            inp.style.cursor  = disabled ? "not-allowed" : "";
        });
        const dlgTh = thead.querySelector("th[style*='22%']");
        if (dlgTh) dlgTh.style.opacity = disabled ? "0.45" : "";
    }

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
        _applyDlgState();
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
        _updateActionPreview();
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

    // ── 10. Source subject column refresh ─────────────────────────────────────

    function _updateSourceColVisibility() {
        const show = _connectedSPSubjects.length > 0;
        srcTh.style.display = show ? "" : "none";
        tbody.querySelectorAll(".fbt-scb-src-td").forEach(td => {
            td.style.display = show ? "" : "none";
        });
    }

    async function _refreshSourceSubjects() {
        const results = [];
        const inp = node.inputs?.find(i => i.name === "source_profile");
        const linkId = inp?.link;
        if (linkId) {
            const linkObj = app.graph.links[linkId];
            const upstream = linkObj ? app.graph.getNodeById(linkObj.origin_id) : null;
            // Widget was renamed from "profile_id" to "profile_name"; support both.
            const profileWidget = upstream?.widgets?.find(
                w => w.name === "profile_name" || w.name === "profile_id"
            );
            const profileVal = profileWidget?.value;
            const isName = profileWidget?.name === "profile_name";
            if (profileVal && profileVal !== "(none)") {
                try {
                    const param = isName
                        ? `name=${encodeURIComponent(profileVal)}`
                        : `id=${encodeURIComponent(profileVal)}`;
                    const resp = await fetch(`/fbtools/source_profiles/get?${param}`);
                    if (resp.ok) {
                        const profile = await resp.json();
                        results.push({
                            pid: profile.id,
                            label: profile.name || profile.id || profileVal,
                            subjects: (profile.subjects ?? []).map(s => ({
                                id: s.id,
                                label: s.label || s.role_description || s.id,
                            })),
                        });
                    }
                } catch { /* skip */ }
            }
        }
        _connectedSPSubjects = results;

        // Update each existing source select in-place (no full rebuild needed)
        const srcSels = [...tbody.querySelectorAll(".fbt-scb-src-sel")];
        srcSels.forEach((sel, i) => {
            const entry = _entries[i];
            if (entry) {
                _fillSourceSubjectSel(sel, entry.source_profile_id || "", entry.source_subject_id || "");
            }
        });

        _updateSourceColVisibility();
        _updateActionPreview();
    }

    node._refreshSourceSubjects = _refreshSourceSubjects;
    requestAnimationFrame(() => _refreshSourceSubjects());

    // ── 12. Clip timeline (single source_profile input) ──────────────────────
    // Hide the backing STRING widget; replace with a canvas timeline that lets
    // the user click a segment or use ← / → arrows (with wrap-around) to pick
    // the active clip. Purely navigational — no boundary dragging.

    const clipWidget = node.widgets?.find(w => w.name === "clip_id");
    if (clipWidget) setWidgetVisible(clipWidget, false, node);

    const TIMELINE_COLORS = ["#3b82f6","#10b981","#f59e0b","#ef4444","#8b5cf6","#06b6d4","#f97316","#ec4899"];

    const clipsSection = document.createElement("div");
    clipsSection.className = "fbt-scb-clips";
    clipsSection.style.display = "none"; // hidden until a profile is connected

    const canvas = document.createElement("canvas");
    canvas.className = "fbt-scb-timeline";
    canvas.height = 52;
    clipsSection.appendChild(canvas);

    const navRow = document.createElement("div");
    navRow.className = "fbt-scb-clip-nav";
    const prevBtn = document.createElement("button");
    prevBtn.className = "fbt-scb-clip-nav-btn";
    prevBtn.textContent = "←";
    const navLabel = document.createElement("span");
    navLabel.className = "fbt-scb-clip-nav-label";
    const nextBtn = document.createElement("button");
    nextBtn.className = "fbt-scb-clip-nav-btn";
    nextBtn.textContent = "→";
    navRow.append(prevBtn, navLabel, nextBtn);
    clipsSection.appendChild(navRow);

    wrap.appendChild(clipsSection);

    // Timeline state
    let _clips    = [];   // ordered clip objects from the connected profile
    let _activeIdx = -1;  // index into _clips; -1 = none
    let _hoverIdx  = -1;

    // When clip time data is absent or flat, synthesise equal-width slots.
    function _displayClips() {
        const hasTimes = _clips.some(c => c.end_time > c.start_time);
        if (hasTimes) return _clips;
        return _clips.map((c, i) => ({ ...c, start_time: i, end_time: i + 1 }));
    }

    function _drawTimeline() {
        const dc = _displayClips();
        const totalDur = dc.length ? Math.max(...dc.map(c => c.end_time)) : 1;
        const W = canvas.width = canvas.offsetWidth || 300;
        const H = canvas.height;
        const ctx = canvas.getContext("2d");
        const toX = t => (t / totalDur) * W;
        const BAND_TOP = 8, BAND_BOT = H - 4;

        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = "#111";
        ctx.fillRect(0, 0, W, H);

        if (!dc.length) {
            ctx.fillStyle = "#555";
            ctx.font = "10px sans-serif";
            ctx.textAlign = "center";
            ctx.fillText("No clips", W / 2, H / 2 + 4);
            return;
        }

        dc.forEach((clip, i) => {
            const x1 = toX(clip.start_time);
            const x2 = toX(clip.end_time);
            const clipW = Math.max(x2 - x1, 1);
            const col   = TIMELINE_COLORS[i % TIMELINE_COLORS.length];
            const isActive  = i === _activeIdx;
            const isHovered = i === _hoverIdx && !isActive;

            ctx.fillStyle   = isActive ? col + "55" : isHovered ? col + "44" : col + "22";
            ctx.fillRect(x1, BAND_TOP, clipW, BAND_BOT - BAND_TOP);
            ctx.strokeStyle = col;
            ctx.lineWidth   = isActive ? 2.5 : isHovered ? 1.5 : 1;
            ctx.strokeRect(x1 + 0.5, BAND_TOP + 0.5, clipW - 1, BAND_BOT - BAND_TOP - 1);

            if (isActive) {
                const mid = (x1 + x2) / 2;
                ctx.fillStyle = col;
                ctx.beginPath();
                ctx.moveTo(mid - 5, BAND_TOP - 1);
                ctx.lineTo(mid + 5, BAND_TOP - 1);
                ctx.lineTo(mid, BAND_TOP + 6);
                ctx.closePath();
                ctx.fill();
            }

            const label = clip.label || `Clip ${i + 1}`;
            const mid   = (x1 + x2) / 2;
            ctx.fillStyle = isActive ? "#fff" : isHovered ? "#eee" : "#aaa";
            ctx.font      = isActive ? "bold 10px sans-serif" : "10px sans-serif";
            ctx.textAlign = "center";
            if (ctx.measureText(label).width <= clipW - 6) {
                ctx.fillText(label, mid, (BAND_TOP + BAND_BOT) / 2 + 4);
            }
        });
    }

    function _updateNavRow() {
        const n = _clips.length;
        if (!n || _activeIdx < 0) {
            navLabel.textContent = n ? "—" : "No clips";
            prevBtn.disabled = nextBtn.disabled = true;
            return;
        }
        const clip = _clips[_activeIdx];
        navLabel.textContent = clip.label
            ? `${_activeIdx + 1}/${n}  ·  ${clip.label}`
            : `${_activeIdx + 1} of ${n}`;
        prevBtn.disabled = nextBtn.disabled = false;
    }

    function _selectClipIdx(idx) {
        if (!_clips.length) return;
        _activeIdx    = ((idx % _clips.length) + _clips.length) % _clips.length;
        _activeClipId = _clips[_activeIdx]?.id ?? "";
        if (clipWidget) {
            clipWidget.value = _activeClipId;
            app?.graph?.setDirtyCanvas?.(true, false);
        }
        _drawTimeline();
        _updateNavRow();
        _updateDlgFromClip();
        [...tbody.querySelectorAll(".fbt-scb-src-sel")].forEach((sel, i) => {
            const entry = _entries[i];
            if (entry) _fillSourceSubjectSel(sel, entry.source_profile_id || "", entry.source_subject_id || "");
        });
        _updateActionPreview();
    }

    prevBtn.onclick = () => _selectClipIdx(_activeIdx - 1);
    nextBtn.onclick = () => _selectClipIdx(_activeIdx + 1);

    canvas.addEventListener("mousemove", e => {
        const rect = canvas.getBoundingClientRect();
        const x    = e.clientX - rect.left;
        const dc   = _displayClips();
        const totalDur = dc.length ? Math.max(...dc.map(c => c.end_time)) : 1;
        let newHover = -1;
        for (let i = 0; i < dc.length; i++) {
            const x1 = (dc[i].start_time / totalDur) * canvas.offsetWidth;
            const x2 = (dc[i].end_time   / totalDur) * canvas.offsetWidth;
            if (x >= x1 && x <= x2) { newHover = i; break; }
        }
        if (newHover !== _hoverIdx) { _hoverIdx = newHover; _drawTimeline(); }
        canvas.style.cursor = newHover >= 0 ? "pointer" : "default";
    });

    canvas.addEventListener("mouseleave", () => {
        if (_hoverIdx !== -1) { _hoverIdx = -1; _drawTimeline(); }
        canvas.style.cursor = "default";
    });

    canvas.addEventListener("click", e => {
        const rect = canvas.getBoundingClientRect();
        const x    = e.clientX - rect.left;
        const dc   = _displayClips();
        const totalDur = dc.length ? Math.max(...dc.map(c => c.end_time)) : 1;
        for (let i = 0; i < dc.length; i++) {
            const x1 = (dc[i].start_time / totalDur) * canvas.offsetWidth;
            const x2 = (dc[i].end_time   / totalDur) * canvas.offsetWidth;
            if (x >= x1 && x <= x2) { _selectClipIdx(i); break; }
        }
    });

    new ResizeObserver(() => _drawTimeline()).observe(canvas);

    // ── Action preview ────────────────────────────────────────────────────────
    // Shows the clip's action text with {A}/{B}/… substituted by cast/source labels.
    const actionPreviewEl = document.createElement("div");
    actionPreviewEl.className = "fbt-scb-action-preview";
    actionPreviewEl.style.display = "none";
    wrap.appendChild(actionPreviewEl);

    function _buildActionPreview() {
        const clipId = _activeClipId;
        if (!clipId) return null;
        const clip = _clipMap.get(clipId);
        const action = clip?.action;
        if (!action) return null;

        // Ordered source subjects for this clip (same ordering as slot label fix).
        const clipSubjectIds = new Set(clip.subjects ?? []);
        const spData = _connectedSPSubjects[0];
        const orderedSubjects = spData
            ? spData.subjects.filter(s => clipSubjectIds.has(s.id))
            : [];

        const SLOTS = ["A","B","C","D","E","F","G","H","I","J"];
        const slotLabel = {};
        orderedSubjects.forEach((subj, idx) => {
            if (idx >= SLOTS.length) return;
            const castEntry = _entries.find(e => e.source_subject_id === subj.id);
            if (castEntry?.bundle_id) {
                const bun = _bundles.find(b => b.id === castEntry.bundle_id);
                slotLabel[SLOTS[idx]] = bun?.name || castEntry.bundle_id;
            } else {
                slotLabel[SLOTS[idx]] = subj.label || subj.id;
            }
        });

        return action.replace(/\{([A-J])\}/g, (match, letter) =>
            slotLabel[letter] != null ? `[${slotLabel[letter]}]` : match
        );
    }

    function _updateActionPreview() {
        const text = _buildActionPreview();
        if (text == null) {
            actionPreviewEl.style.display = "none";
            actionPreviewEl.textContent = "";
        } else {
            actionPreviewEl.style.display = "";
            actionPreviewEl.textContent = text;
        }
    }

    // keyed by clip id — populated in _refreshClipSelects
    let _clipMap = new Map();

    function _updateDlgFromClip() {
        const clip = _clipMap.get(_activeClipId);
        _clipAllowsDialogue = !clip || clip.allows_dialogue !== false;
        _applyDlgState();
    }

    async function _refreshClipSelects() {
        const savedVal = clipWidget?.value ?? "";
        const inp = node.inputs?.find(i => i.name === "source_profile");
        const linkId = inp?.link;

        if (!linkId) {
            clipsSection.style.display = "none";
            _clips = [];
            _clipMap.clear();
            _activeIdx = -1;
            _activeClipId = "";
            _clipAllowsDialogue = true;
            _applyDlgState();
            return;
        }
        const linkObj = app.graph.links[linkId];
        const upstream = linkObj ? app.graph.getNodeById(linkObj.origin_id) : null;
        const profileWidget = upstream?.widgets?.find(
            w => w.name === "profile_name" || w.name === "profile_id"
        );
        const profileVal = profileWidget?.value;
        const isName = profileWidget?.name === "profile_name";

        if (!profileVal || profileVal === "(none)") {
            clipsSection.style.display = "none";
            _clips = [];
            _clipMap.clear();
            _activeIdx = -1;
            _activeClipId = "";
            _clipAllowsDialogue = true;
            _applyDlgState();
            return;
        }

        clipsSection.style.display = "";

        let clips = [];
        try {
            const param = isName
                ? `name=${encodeURIComponent(profileVal)}`
                : `id=${encodeURIComponent(profileVal)}`;
            const resp = await fetch(`/fbtools/source_profiles/get?${param}`);
            if (resp.ok) clips = (await resp.json()).clips ?? [];
        } catch { /* leave empty */ }

        _clips  = clips;
        _clipMap = new Map(clips.map(c => [c.id, c]));

        // Restore saved selection; fall back to first clip.
        const savedIdx = savedVal ? clips.findIndex(c => c.id === savedVal) : -1;
        if (savedIdx >= 0) {
            _selectClipIdx(savedIdx);
        } else if (clips.length) {
            _selectClipIdx(0);
        } else {
            _activeIdx = -1;
            _activeClipId = "";
            if (clipWidget) clipWidget.value = "";
            _drawTimeline();
            _updateNavRow();
            _updateDlgFromClip();
        }

        if (displayWidget) {
            displayWidget.computeSize = () => [0, _tableHeight() + 28];
            node.setSize?.([node.size[0], node.size[1]]);
        }
    }

    node._refreshClipSelects = _refreshClipSelects;
    requestAnimationFrame(() => _refreshClipSelects());

    // ── 13. Public refresh (called by cast editor "Send to Workflow") ─────────

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

    // ── 14. Load subject + bundle lists, then render ─────────────────────────
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
