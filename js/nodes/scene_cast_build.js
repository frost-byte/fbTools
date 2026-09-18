/**
 * SceneCastBuild node — JSON-backed interactive tab UI.
 *
 * One hidden STRING widget (cast_entries_json) serialises all entries as a
 * JSON array.  The DOM widget renders a tab strip (one tab per entry, plus a
 * "+" add button) so each entry's controls have room to breathe.
 *
 * node._refreshCastTable(entries?) is exposed for the cast editor's
 * "Send to Workflow" button to call after pushing new entries.
 */

import { setWidgetVisible } from "../utils/widgets.js";
import { bundlesApi }       from "../api/bundles.js";
import { app }               from "../../../scripts/app.js";
import { api }               from "../../../scripts/api.js";

const JSON_WIDGET = "cast_entries_json";
const MAX_ENTRIES = 8;


// ── Node setup ────────────────────────────────────────────────────────────────

export function setupSceneCastBuild(nodeType, _nodeData, app) {
    const _origCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        _origCreated?.call(this);
        _buildCastBuildUI(this, app);
    };

    // _refreshClipSelects() and _refreshSourceSubjects() each fetch the same
    // profile independently and race — _buildActionPreview() needs BOTH
    // (_clipMap/_activeClipId from the former, _connectedSPSubjects from the
    // latter) to be from the *same* profile, or ordinal resolution mixes old
    // clip data with new subject data (or vice versa) and the preview shows
    // a plausible-looking but wrong result. Await both, then do one final
    // authoritative preview update — the two functions' own internal preview
    // calls may still run first with inconsistent intermediate state, but
    // this always corrects it once both have actually settled.
    async function _refreshProfileDependentState(node) {
        await Promise.all([
            node._refreshClipSelects?.(),
            node._refreshSourceSubjects?.(),
        ]);
        node._updateActionPreview?.();
    }

    const _origConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (config) {
        _origConfigure?.call(this, config);
        this._refreshCastTable?.();
        const node = this;
        requestAnimationFrame(() => _refreshProfileDependentState(node));
    };

    const _origConnChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (type, index, connected, linkInfo) {
        _origConnChange?.call(this, type, index, connected, linkInfo);
        if (type === LiteGraph?.INPUT) {
            const inp = this.inputs?.[index];
            if (inp?.name === "source_profile") {
                const node = this;
                requestAnimationFrame(() => _refreshProfileDependentState(node));
            }
        }
    };

}

// The action preview only recomputes on user interaction (selecting a
// different clip/segment) — nothing previously refreshed it after the graph
// actually ran, so it could sit showing a stale substitution once execution
// changed something upstream (e.g. a swapped bundle) without the user
// re-clicking a segment. SceneCastBuild's execute() has no `ui` output, so
// core ComfyUI never sends an "executed" websocket message for it (see
// execution.py: that message only fires when output_ui is non-empty) —
// onExecuted would simply never be called. It already emits its own
// fbtools.status event on completion instead; listen for that.
api.addEventListener("fbtools.status", (event) => {
    const nodeId = event?.detail?.node;
    if (nodeId == null) return;
    const node = app.graph?._nodes?.find(n => n.id == nodeId);
    if (node?._updateActionPreview) node._updateActionPreview();
});

function _buildCastBuildUI(node, app) {
    // ── 1. Find and hide the single JSON backing widget ───────────────────────
    const jsonWidget = node.widgets?.find(w => w.name === JSON_WIDGET);
    if (jsonWidget) setWidgetVisible(jsonWidget, false, node);

    // Backing widget for the action preview text below — hidden the same way
    // as jsonWidget. execute() never reads it (absorbed by its **_ catch-all);
    // it exists purely so the frontend-computed preview rides along in the
    // submitted prompt as a literal value, making it visible to Run History's
    // [track:] scan without any runtime-capture plumbing.
    const actionPreviewWidget = node.widgets?.find(w => w.name === "action_preview");
    if (actionPreviewWidget) setWidgetVisible(actionPreviewWidget, false, node);

    // ── 2. Internal state ─────────────────────────────────────────────────────
    let _subjects  = [];
    let _bundles   = [];
    let _connectedSPSubjects = [];
    let _clipAllowsDialogue  = true;
    let _activeClipId        = "";
    let _multiplier          = 1;
    let _activeTabIdx        = 0;
    let _previewOpen         = false;
    let _clipMap             = new Map();

    let _entries = [];
    try {
        const parsed = JSON.parse(jsonWidget?.value || "[]");
        if (Array.isArray(parsed)) _entries = parsed;
    } catch { /* leave empty */ }
    if (!_entries.length) {
        _entries.push({ subject_id: "", bundle_id: "", visual_mode: "images", use_audio: false, dialogue: "" });
    }
    const _tmpFrames = new Map();

    // ── 3. Build DOM structure ────────────────────────────────────────────────
    const wrap = document.createElement("div");
    wrap.className = "fbt-scb-wrap";

    const tabStrip = document.createElement("div");
    tabStrip.className = "fbt-scb-tab-strip";

    // Inner scroller holds just the tabs — scrolls when node is narrow
    const tabScroller = document.createElement("div");
    tabScroller.className = "fbt-scb-tab-scroller";

    // "+" button lives outside the scroller so it's always visible
    const addTabBtn = document.createElement("button");
    addTabBtn.className = "fbt-scb-tab-add";
    addTabBtn.textContent = "+";
    addTabBtn.title = `Add entry (max ${MAX_ENTRIES})`;
    addTabBtn.addEventListener("click", () => {
        if (_entries.length >= MAX_ENTRIES) return;
        _entries.push({ subject_id: "", bundle_id: "", visual_mode: "images", use_audio: false, dialogue: "" });
        _activeTabIdx = _entries.length - 1;
        _previewOpen  = false;
        _rebuildTabs();
        _syncWidget();
    });

    tabStrip.appendChild(tabScroller);
    tabStrip.appendChild(addTabBtn);

    const tabContent = document.createElement("div");
    tabContent.className = "fbt-scb-tab-content";

    wrap.appendChild(tabStrip);
    wrap.appendChild(tabContent);

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
        blank.textContent = "— source —";
        if (!currentProfileId || !currentSubjectId) blank.selected = true;
        sel.appendChild(blank);

        const selectedClipId = _activeClipId;
        const clip = selectedClipId ? _clipMap?.get(selectedClipId) : null;
        const clipSubjectIds = (clip && Array.isArray(clip.subjects) && clip.subjects.length)
            ? new Set(clip.subjects) : null;

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

    // ── 5. Tab label ──────────────────────────────────────────────────────────

    function _tabLabel(idx) {
        const e = _entries[idx];
        if (e?.bundle_id) {
            const bun = _bundles.find(b => b.id === e.bundle_id);
            if (bun?.name) {
                const n = bun.name;
                return n.length > 12 ? n.slice(0, 11) + "…" : n;
            }
        }
        if (e?.subject_id) {
            const s = _subjects.find(s => s.id === e.subject_id);
            if (s?.name) {
                const n = s.name;
                return n.length > 12 ? n.slice(0, 11) + "…" : n;
            }
        }
        return `#${idx + 1}`;
    }

    // ── 6. Build tab strip ────────────────────────────────────────────────────

    function _buildTabStrip() {
        tabScroller.innerHTML = "";
        _entries.forEach((_, idx) => {
            const tab = document.createElement("button");
            tab.className = "fbt-scb-tab" + (idx === _activeTabIdx ? " active" : "");

            const lbl = document.createElement("span");
            lbl.className = "fbt-scb-tab-lbl";
            lbl.textContent = _tabLabel(idx);
            lbl.title = _entries[idx]?.bundle_id || _entries[idx]?.subject_id || `Entry ${idx + 1}`;
            tab.appendChild(lbl);

            tab.addEventListener("click", () => {
                if (_activeTabIdx === idx) return;
                _activeTabIdx = idx;
                _buildTabStrip();
                _renderActiveTab();
            });

            if (_entries.length > 1) {
                const closeBtn = document.createElement("button");
                closeBtn.className = "fbt-scb-tab-close";
                closeBtn.textContent = "×";
                closeBtn.title = "Remove entry";
                closeBtn.addEventListener("click", ev => {
                    ev.stopPropagation();
                    const bid = _entries[idx]?.bundle_id;
                    if (bid) { const t = _tmpFrames.get(bid); if (t) { bundlesApi.deleteTmpFrame(t); _tmpFrames.delete(bid); } }
                    _entries.splice(idx, 1);
                    _activeTabIdx = Math.min(_activeTabIdx, _entries.length - 1);
                    _rebuildTabs();
                    _syncWidget();
                });
                tab.appendChild(closeBtn);
            }

            tabScroller.appendChild(tab);
        });

        addTabBtn.disabled = _entries.length >= MAX_ENTRIES;
    }

    // ── 7. Per-reference preview ──────────────────────────────────────────────

    async function _buildRefList(entry) {
        if (!entry.bundle_id) return [];
        let bundle;
        try { bundle = await bundlesApi.getBundle(entry.bundle_id); } catch { return []; }
        const refs = [];
        const mode = entry.visual_mode || "images";

        if (mode === "video" || mode === "both") {
            const vfile = bundle.visual?.file || "";
            if (vfile) {
                refs.push({
                    type: "video",
                    file: vfile,
                    dir:  bundle.visual?.video_dir || "input",
                    role: bundle.visual?.role || "reference footage",
                });
            }
        }

        if (mode !== "video") {
            let files = bundle.visual?.files || [];
            const sel = entry.image_selection;
            if (sel != null) {
                const indices = Array.isArray(sel) ? sel : [parseInt(sel, 10)];
                files = indices.filter(i => i >= 0 && i < files.length).map(i => files[i]);
            }
            files.forEach(f => {
                const file = typeof f === "object" ? (f.file || "") : (f || "");
                const role = typeof f === "object" ? (f.role || "character sheet") : "character sheet";
                if (file) refs.push({ type: "image", file, role });
            });
        }

        // Audio preview entry
        const aSrc  = bundle.audio?.source;
        const aFile = bundle.audio?.file || "";
        if (aFile && aSrc !== "none" && aSrc !== "extract_from_visual") {
            refs.push({ type: "audio", file: aFile, role: bundle.audio?.role || "voice reference" });
        }

        return refs;
    }

    async function _loadRefPreview(area, entry) {
        area.innerHTML = "";
        if (!entry.bundle_id) {
            area.innerHTML = '<span class="fbt-scb-preview-note">No bundle selected.</span>';
            return;
        }
        area.innerHTML = '<span class="fbt-scb-preview-note">Loading…</span>';

        const refs = await _buildRefList(entry);
        area.innerHTML = "";

        if (!refs.length) {
            area.innerHTML = '<span class="fbt-scb-preview-note">No media references in bundle.</span>';
            return;
        }

        let refIdx = 0;

        // Navigator row — arrows inline with label, centered as a group
        const nav = document.createElement("div");
        nav.className = "fbt-scb-ref-nav";
        const prevRef = document.createElement("button");
        prevRef.className = "fbt-scb-clip-nav-btn";
        prevRef.textContent = "←";
        const refLabel = document.createElement("span");
        refLabel.className = "fbt-scb-ref-label";
        const nextRef = document.createElement("button");
        nextRef.className = "fbt-scb-clip-nav-btn";
        nextRef.textContent = "→";
        nav.append(prevRef, refLabel, nextRef);

        const panel = document.createElement("div");
        panel.className = "fbt-scb-ref-panel";

        area.appendChild(nav);
        area.appendChild(panel);

        const _fname = path => path.split("/").pop();

        async function _showRef(i) {
            refIdx = ((i % refs.length) + refs.length) % refs.length;
            const ref = refs[refIdx];
            refLabel.textContent = `${refIdx + 1}/${refs.length} · ${ref.role}`;
            prevRef.disabled = refs.length <= 1;
            nextRef.disabled = refs.length <= 1;
            panel.innerHTML = "";

            const _addMeta = text => {
                const m = document.createElement("div");
                m.className = "fbt-scb-ref-meta";
                m.textContent = text;
                panel.appendChild(m);
            };

            if (ref.type === "image") {
                const slashIdx = ref.file.lastIndexOf("/");
                const fname  = slashIdx >= 0 ? ref.file.slice(slashIdx + 1) : ref.file;
                const sfold  = slashIdx >= 0 ? ref.file.slice(0, slashIdx) : "";
                const mkUrl  = type => `/view?filename=${encodeURIComponent(fname)}${sfold ? `&subfolder=${encodeURIComponent(sfold)}` : ""}&type=${type}`;
                const img    = document.createElement("img");
                img.className = "fbt-scb-ref-img";
                img.src       = mkUrl("input");
                img.onerror   = () => { img.onerror = null; img.src = mkUrl("output"); };
                img.title     = ref.file;
                img.addEventListener("click", () => window.open(img.src, "_blank"));
                panel.appendChild(img);
                _addMeta(_fname(ref.file));
            } else if (ref.type === "video") {
                const note = document.createElement("div");
                note.className = "fbt-scb-ref-meta";
                note.textContent = `${_fname(ref.file)} — extracting frame…`;
                panel.appendChild(note);
                try {
                    const frameData = await bundlesApi.extractFrame(ref.file, 0, ref.dir);
                    const prev = _tmpFrames.get(entry.bundle_id);
                    if (prev) bundlesApi.deleteTmpFrame(prev);
                    _tmpFrames.set(entry.bundle_id, frameData.tmp_filename);
                    panel.innerHTML = "";
                    const img = document.createElement("img");
                    img.className = "fbt-scb-ref-img";
                    img.src   = `/view?filename=${encodeURIComponent(frameData.tmp_filename)}&type=input`;
                    img.title = ref.file;
                    img.addEventListener("click", () => window.open(img.src, "_blank"));
                    panel.appendChild(img);
                    _addMeta(`${_fname(ref.file)} · ${frameData.width}×${frameData.height} · ${frameData.frame_count} frames`);
                } catch {
                    note.textContent = `${_fname(ref.file)} (frame extract unavailable)`;
                }
            } else if (ref.type === "audio") {
                const aud = document.createElement("audio");
                aud.className = "fbt-scb-preview-audio";
                aud.src      = `/view?filename=${encodeURIComponent(ref.file)}&type=input`;
                aud.controls  = true;
                aud.preload   = "none";
                aud.title     = ref.file;
                panel.appendChild(aud);
                _addMeta(_fname(ref.file));
            }
        }

        prevRef.addEventListener("click", () => _showRef(refIdx - 1));
        nextRef.addEventListener("click", () => _showRef(refIdx + 1));
        _showRef(0);
    }

    // ── 8. Render active tab content ──────────────────────────────────────────

    function _renderActiveTab() {
        tabContent.innerHTML = "";
        if (!_entries.length) return;

        const entry = _entries[_activeTabIdx];
        const idx   = _activeTabIdx;

        const _lbl = text => {
            const s = document.createElement("span");
            s.className = "fbt-scb-field-label";
            s.textContent = text;
            return s;
        };

        // ── Row 1: Subject → Bundle ──────────────────────────────────────────────
        const row1 = document.createElement("div");
        row1.className = "fbt-scb-form-row";

        const subjSel = document.createElement("select");
        subjSel.className = "fbt-scb-sel fbt-scb-sel-narrow";
        _fillSubjectSel(subjSel, entry.subject_id || "");

        const arrow = document.createElement("span");
        arrow.className = "fbt-scb-arrow";
        arrow.textContent = "→";

        const bundSel = document.createElement("select");
        bundSel.className = "fbt-scb-sel fbt-scb-sel-narrow";
        _fillBundleSel(bundSel, entry.subject_id || "", entry.bundle_id || "");

        row1.append(_lbl("Subject"), subjSel, arrow, _lbl("Bundle"), bundSel);

        // ── Row 2: Source + Mode + Audio ─────────────────────────────────────────
        const row2 = document.createElement("div");
        row2.className = "fbt-scb-form-row";

        const srcSel = document.createElement("select");
        srcSel.className = "fbt-scb-sel fbt-scb-sel-narrow fbt-scb-src-sel";
        srcSel.style.display = _connectedSPSubjects.length ? "" : "none";
        _fillSourceSubjectSel(srcSel, entry.source_profile_id || "", entry.source_subject_id || "");

        // Mode buttons
        const modeWrap = document.createElement("div");
        modeWrap.className = "fbt-scb-mode";

        const bun0 = _bundles.find(b => b.id === entry.bundle_id);
        const _hasImages = b => Boolean(b?.visual?.files?.length);
        const _hasVideo  = b => Boolean(b?.visual?.file);
        const _imgCount  = b => b?.visual?.files?.length ?? 0;

        if (!("image_selection" in entry)) entry.image_selection = null;

        const imgBtn = document.createElement("button");
        imgBtn.className = "fbt-scb-mode-btn";
        imgBtn.textContent = "Img";

        const imgSubWrap = document.createElement("div");
        imgSubWrap.className = "fbt-scb-mode-num-wrap";

        const vidBtn = document.createElement("button");
        vidBtn.className = "fbt-scb-mode-btn fbt-scb-mode-btn-vid";
        vidBtn.textContent = "Vid";

        const bothBtn = document.createElement("button");
        bothBtn.className = "fbt-scb-mode-btn fbt-scb-mode-btn-both";
        bothBtn.textContent = "Both";

        modeWrap.append(imgBtn, imgSubWrap, vidBtn, bothBtn);

        const _selArr  = () => Array.isArray(entry.image_selection) ? entry.image_selection : [];
        const _selHas  = i  => _selArr().includes(i);
        const _selNone = ()  => entry.image_selection == null || (Array.isArray(entry.image_selection) && !entry.image_selection.length);

        const _syncModeActive = () => {
            const mode   = entry.visual_mode;
            const isVid  = mode === "video";
            const isBoth = mode === "both";
            const hasImg = mode === "images" || isBoth;
            // "Img" lights up when in images/both with no specific selection (all images)
            imgBtn.classList.toggle("active", hasImg && _selNone());
            vidBtn.classList.toggle("active", isVid);
            bothBtn.classList.toggle("active", isBoth);
            // Numbered buttons highlight when their index is in the selection array
            imgSubWrap.querySelectorAll(".fbt-scb-mode-btn-num").forEach((btn, i) => {
                btn.classList.toggle("active", hasImg && _selHas(i));
            });
        };

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
                    // Stay in "both" if already there; otherwise switch to "images"
                    if (entry.visual_mode !== "both") entry.visual_mode = "images";
                    // Toggle this index in/out of the selection array
                    const arr = _selArr();
                    const pos = arr.indexOf(i);
                    if (pos === -1) arr.push(i); else arr.splice(pos, 1);
                    arr.sort((a, b) => a - b);
                    // Empty array → null (all images); otherwise store sorted array
                    entry.image_selection = arr.length ? arr : null;
                    _syncModeActive();
                    _syncWidget();
                    if (_previewOpen) _loadRefPreview(previewArea, entry);
                });
                imgSubWrap.appendChild(btn);
            }
            const hasImg = fileCount > 0;
            const hasVid = _hasVideo(bun);
            imgBtn.disabled  = bun && !hasImg;
            imgBtn.title     = hasImg ? "Use all image references" : "No images in bundle";
            vidBtn.disabled  = bun && !hasVid;
            vidBtn.title     = hasVid ? "Use video reference" : "No video in bundle";
            bothBtn.disabled = bun && !(hasImg && hasVid);
            bothBtn.title    = (hasImg && hasVid) ? "Use both images and video" : "Bundle needs both images and a video";
            _syncModeActive();
        };

        _buildNumBtns(bun0);

        const audCb = document.createElement("input");
        audCb.type      = "checkbox";
        audCb.className = "fbt-scb-audio";
        audCb.title     = "Use audio reference";
        audCb.checked   = !!entry.use_audio;

        // Ordinal match: instead of picking one specific source subject, match
        // the Nth clip subject (in that clip's own order) whose resolved
        // pronoun_style equals this bundle's own — resolved fresh per clip at
        // execution time, so the same cast entry tracks "the 1st feminine
        // subject" across every clip in the profile without re-picking per clip.
        const ordToggleBtn = document.createElement("button");
        ordToggleBtn.className = "fbt-scb-ord-toggle";
        ordToggleBtn.textContent = "#";
        ordToggleBtn.title =
            "Match by ordinal position instead of a specific subject — e.g. the " +
            "1st subject in each clip sharing this bundle's pronoun/category, " +
            "resolved fresh per clip. No match in a given clip → falls back to " +
            "bundle-only (no source reference) for that clip.";

        const ordInput = document.createElement("input");
        ordInput.type      = "number";
        ordInput.min       = "1";
        ordInput.step      = "1";
        ordInput.className = "fbt-scb-ord-input";
        ordInput.value     = entry.ordinal || 1;
        ordInput.title     = "1st, 2nd, 3rd, … matching subject in the active clip";

        const _isOrdinal = () => entry.match_mode === "ordinal";
        const _syncOrdinalVisibility = () => {
            const on = _isOrdinal();
            ordToggleBtn.classList.toggle("active", on);
            srcSel.style.display   = (_connectedSPSubjects.length && !on) ? "" : "none";
            ordInput.style.display = (_connectedSPSubjects.length && on)  ? "" : "none";
        };
        _syncOrdinalVisibility();

        ordToggleBtn.addEventListener("click", () => {
            if (_isOrdinal()) {
                delete entry.match_mode;
                delete entry.ordinal;
            } else {
                entry.match_mode = "ordinal";
                entry.ordinal    = entry.ordinal || 1;
                delete entry.source_subject_id;
                // Still need to know which connected profile to match within.
                if (!entry.source_profile_id && _connectedSPSubjects.length === 1) {
                    entry.source_profile_id = _connectedSPSubjects[0].pid;
                }
            }
            _syncOrdinalVisibility();
            _syncWidget();
        });

        ordInput.addEventListener("change", () => {
            const n = parseInt(ordInput.value, 10);
            entry.ordinal = Number.isFinite(n) && n >= 1 ? n : 1;
            ordInput.value = entry.ordinal;
            _syncWidget();
        });

        // Source label hides when no source profile connected
        const srcLabel = _lbl("Source");
        srcLabel.style.display = _connectedSPSubjects.length ? "" : "none";
        const audLabel = _lbl("Audio");
        row2.append(srcLabel, srcSel, ordToggleBtn, ordInput, _lbl("Mode"), modeWrap, audLabel, audCb);

        // ── Row 3: Dialogue ──────────────────────────────────────────────────────
        const row3 = document.createElement("div");
        row3.className = "fbt-scb-form-row";

        const dlgInput = document.createElement("input");
        dlgInput.type        = "text";
        dlgInput.className   = "fbt-scb-dlg";
        dlgInput.value       = entry.dialogue || "";
        dlgInput.placeholder = "dialogue / %libber:key%";
        dlgInput.title       = "Dialogue for this cast entry.\n[silent] = no audio contribution\n[sounds] desc = sound event\n%libber:key% or %libber:*% = libber lookup";
        row3.append(_lbl("Dialogue"), dlgInput);
        _applyDlgToInput(dlgInput);

        // ── Preview section ──────────────────────────────────────────────────────
        const previewToggle = document.createElement("button");
        previewToggle.className = "fbt-scb-preview-btn fbt-scb-preview-toggle";
        previewToggle.textContent = "⊙ Preview";
        previewToggle.title = "Preview media references for this entry";

        const previewArea = document.createElement("div");
        previewArea.className = "fbt-scb-preview-area";

        if (_previewOpen) {
            previewToggle.classList.add("active");
            previewArea.style.display = "";
            _loadRefPreview(previewArea, entry);
        } else {
            previewArea.style.display = "none";
        }

        previewToggle.addEventListener("click", () => {
            _previewOpen = !_previewOpen;
            previewToggle.classList.toggle("active", _previewOpen);
            if (_previewOpen) {
                previewArea.style.display = "";
                _loadRefPreview(previewArea, entry);
            } else {
                previewArea.style.display = "none";
                const tmp = _tmpFrames.get(entry.bundle_id);
                if (tmp) { bundlesApi.deleteTmpFrame(tmp); _tmpFrames.delete(entry.bundle_id); }
            }
            _updateHeight();
        });

        tabContent.append(row1, row2, row3, previewToggle, previewArea);

        // ── Events ───────────────────────────────────────────────────────────────

        subjSel.addEventListener("change", () => {
            entry.subject_id = subjSel.value;
            const cur = _bundles.find(b => b.id === entry.bundle_id);
            if (cur?.subject_id && cur.subject_id !== entry.subject_id) entry.bundle_id = "";
            _fillBundleSel(bundSel, entry.subject_id, entry.bundle_id);
            _buildTabStrip();
            _syncWidget();
        });

        bundSel.addEventListener("change", () => {
            entry.bundle_id = bundSel.value;
            const bun = _bundles.find(b => b.id === entry.bundle_id);
            const hasImg = _hasImages(bun);
            const hasVid = _hasVideo(bun);
            if (bun?.visual?.type) {
                entry.visual_mode = bun.visual.type;
                if (entry.visual_mode === "video"  && !hasVid && hasImg) entry.visual_mode = "images";
                if (entry.visual_mode === "images" && !hasImg && hasVid) entry.visual_mode = "video";
                if (entry.visual_mode === "both"   && !hasImg)           entry.visual_mode = hasVid ? "video" : "images";
                if (entry.visual_mode === "both"   && !hasVid)           entry.visual_mode = "images";
            }
            const fc = _imgCount(bun);
            if (Array.isArray(entry.image_selection)) {
                const filtered = entry.image_selection.filter(i => i < fc);
                entry.image_selection = filtered.length ? filtered : null;
            } else if (entry.image_selection != null && entry.image_selection >= fc) {
                entry.image_selection = null;
            }
            _buildNumBtns(bun);
            _buildTabStrip();
            if (_previewOpen) _loadRefPreview(previewArea, entry);
            _syncWidget();
        });

        srcSel.addEventListener("change", () => {
            const val = srcSel.value;
            if (val) {
                const sep = val.indexOf("::");
                entry.source_profile_id  = sep >= 0 ? val.slice(0, sep) : "";
                entry.source_subject_id  = sep >= 0 ? val.slice(sep + 2) : val;
            } else {
                delete entry.source_profile_id;
                delete entry.source_subject_id;
            }
            _syncWidget();
        });

        imgBtn.addEventListener("click", () => {
            // Stay in "both" if already there (clearing to all images within both)
            if (entry.visual_mode !== "both") entry.visual_mode = "images";
            entry.image_selection = null;
            _syncModeActive();
            _syncWidget();
            if (_previewOpen) _loadRefPreview(previewArea, entry);
        });

        vidBtn.addEventListener("click", () => {
            entry.visual_mode = "video";
            _syncModeActive();
            _syncWidget();
            if (_previewOpen) _loadRefPreview(previewArea, entry);
        });

        bothBtn.addEventListener("click", () => {
            entry.visual_mode = "both";
            _syncModeActive();
            _syncWidget();
            if (_previewOpen) _loadRefPreview(previewArea, entry);
        });

        audCb.addEventListener("change", () => {
            entry.use_audio = audCb.checked;
            _syncWidget();
        });

        dlgInput.addEventListener("input", () => {
            entry.dialogue = dlgInput.value;
            _syncWidget();
        });
    }

    // ── 9. Dialogue state ─────────────────────────────────────────────────────

    function _applyDlgToInput(inp) {
        const disabled = !_clipAllowsDialogue;
        inp.disabled      = disabled;
        inp.title         = disabled
            ? "Dialogue disabled — this clip has 'Allows dialogue' turned off"
            : inp.title;
        inp.style.opacity = disabled ? "0.35" : "";
        inp.style.cursor  = disabled ? "not-allowed" : "";
    }

    function _applyDlgState() {
        const inp = tabContent.querySelector(".fbt-scb-dlg");
        if (inp) _applyDlgToInput(inp);
    }

    // ── 10. Rebuild all tabs ──────────────────────────────────────────────────

    function _rebuildTabs() {
        // Clean up extracted frames no longer in any entry
        const activeIds = new Set(_entries.map(e => e.bundle_id).filter(Boolean));
        _tmpFrames.forEach((tmp, bid) => {
            if (!activeIds.has(bid)) { bundlesApi.deleteTmpFrame(tmp); _tmpFrames.delete(bid); }
        });

        _buildTabStrip();
        _renderActiveTab();
        _updateHeight();
    }

    // ── 11. Widget height ─────────────────────────────────────────────────────

    function _widgetHeight() {
        // Tab strip + 3 form rows (subject/mode/dlg) + preview toggle + preview area
        const base = 30 + 26 + 26 + 26 + 24;  // ≈ 132px
        return base + (_previewOpen ? 140 : 0);
    }

    function _updateHeight() {
        if (displayWidget) {
            displayWidget.computeSize = () => [0, _widgetHeight()];
            node.setSize?.([node.size[0], node.size[1]]);
        }
    }

    // ── 12. Sync entries → hidden widget ─────────────────────────────────────

    function _syncWidget() {
        if (jsonWidget) jsonWidget.value = JSON.stringify(_entries);
        app?.graph?.setDirtyCanvas?.(true, false);
        _updateActionPreview();
    }

    // ── 13. Add DOM widget ────────────────────────────────────────────────────
    let displayWidget = null;
    displayWidget = node.addDOMWidget("cast_build_table", "preview", wrap, {
        serialize:   false,
        hideOnZoom:  false,
        margin:      0,
        getValue()   { return null; },
        setValue()   {},
    });
    displayWidget.computeSize = () => [0, _widgetHeight()];

    // ── 14. Source subject refresh ────────────────────────────────────────────

    async function _refreshSourceSubjects() {
        const results = [];
        const inp     = node.inputs?.find(i => i.name === "source_profile");
        const linkId  = inp?.link;
        if (linkId) {
            const linkObj  = app.graph.links[linkId];
            const upstream = linkObj ? app.graph.getNodeById(linkObj.origin_id) : null;
            const profileWidget = upstream?.widgets?.find(
                w => w.name === "profile_name" || w.name === "profile_id"
            );
            const profileVal = profileWidget?.value;
            const isName     = profileWidget?.name === "profile_name";
            if (profileVal && profileVal !== "(none)") {
                try {
                    const param = isName
                        ? `name=${encodeURIComponent(profileVal)}`
                        : `id=${encodeURIComponent(profileVal)}`;
                    const resp = await fetch(`/fbtools/source_profiles/get?${param}`);
                    if (resp.ok) {
                        const profile = await resp.json();
                        results.push({
                            pid:      profile.id,
                            label:    profile.name || profile.id || profileVal,
                            subjects: (profile.subjects ?? []).map(s => ({
                                id:            s.id,
                                label:         s.label || s.role_description || s.id,
                                entity_type:   s.entity_type || "person",
                                pronoun_style: s.pronoun_style || "",
                            })),
                        });
                    }
                } catch { /* skip */ }
            }
        }
        _connectedSPSubjects = results;

        // Update source select in the currently visible tab in-place
        const srcSel = tabContent.querySelector(".fbt-scb-src-sel");
        if (srcSel) {
            const entry = _entries[_activeTabIdx];
            if (entry) {
                _fillSourceSubjectSel(srcSel, entry.source_profile_id || "", entry.source_subject_id || "");
                const show = _connectedSPSubjects.length ? "" : "none";
                srcSel.style.display = show;
                // Also toggle the sibling Source label (previous sibling)
                if (srcSel.previousElementSibling?.classList.contains("fbt-scb-field-label")) {
                    srcSel.previousElementSibling.style.display = show;
                }
            }
        }

        _updateActionPreview();
    }

    node._refreshSourceSubjects = _refreshSourceSubjects;
    requestAnimationFrame(() => _refreshSourceSubjects());

    // ── 15. Clip timeline ─────────────────────────────────────────────────────

    const clipWidget = node.widgets?.find(w => w.name === "clip_id");
    if (clipWidget) setWidgetVisible(clipWidget, false, node);

    const multWidget = node.widgets?.find(w => w.name === "clip_duration_multiplier");
    if (multWidget) {
        setWidgetVisible(multWidget, false, node);
        _multiplier = Math.max(1, Math.min(4, parseInt(multWidget.value, 10) || 1));
    }

    const TIMELINE_COLORS = ["#3b82f6","#10b981","#f59e0b","#ef4444","#8b5cf6","#06b6d4","#f97316","#ec4899"];

    const clipsSection = document.createElement("div");
    clipsSection.className = "fbt-scb-clips";
    clipsSection.style.display = "none";

    // Timeline row: [prev arrow] [canvas] [next arrow] — arrows get their own
    // fixed-width space rather than overlaying the canvas, mirroring the
    // Source Profile editor's timeline zoom/paging behavior.
    const timelineRow = document.createElement("div");
    timelineRow.className = "fbt-scb-timeline-row";

    const canvas = document.createElement("canvas");
    canvas.className = "fbt-scb-timeline";
    canvas.height = 52;

    const zoomPrevBtn = document.createElement("button");
    zoomPrevBtn.className = "fbt-scb-clip-nav-btn fbt-scb-timeline-zoom-btn";
    zoomPrevBtn.textContent = "←";
    zoomPrevBtn.title = "Show previous clips";
    const zoomNextBtn = document.createElement("button");
    zoomNextBtn.className = "fbt-scb-clip-nav-btn fbt-scb-timeline-zoom-btn";
    zoomNextBtn.textContent = "→";
    zoomNextBtn.title = "Show next clips";

    const ZOOM_WINDOW_SIZE = 10;
    let _zoomStart = 0;

    // Paging jumps _zoomStart by a whole window, which used to redraw
    // instantly — jarring since the visible clips change completely in one
    // frame. Animate the pan/rescale between the old and new visible range.
    function _animateZoomPan(fromRange, toRange) {
        if (!fromRange || !toRange) { _drawTimeline(); return; }
        const DURATION_MS = 180;
        const t0 = performance.now();
        function step(now) {
            const t = Math.min(1, (now - t0) / DURATION_MS);
            const eased = 1 - Math.pow(1 - t, 3);  // ease-out cubic
            const range = [
                fromRange[0] + (toRange[0] - fromRange[0]) * eased,
                fromRange[1] + (toRange[1] - fromRange[1]) * eased,
            ];
            _drawTimeline(range);
            if (t < 1) requestAnimationFrame(step);
            else _drawTimeline();  // final pass to fully resync arrow disabled state
        }
        requestAnimationFrame(step);
    }

    zoomPrevBtn.onclick = () => {
        const fromRange = _visibleClipRange();
        _zoomStart = Math.max(0, _zoomStart - ZOOM_WINDOW_SIZE);
        _animateZoomPan(fromRange, _visibleClipRange());
    };
    zoomNextBtn.onclick = () => {
        const fromRange = _visibleClipRange();
        _zoomStart = Math.min(Math.max(0, _clips.length - ZOOM_WINDOW_SIZE), _zoomStart + ZOOM_WINDOW_SIZE);
        _animateZoomPan(fromRange, _visibleClipRange());
    };

    timelineRow.append(zoomPrevBtn, canvas, zoomNextBtn);

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

    const durRow = document.createElement("div");
    durRow.className = "fbt-scb-dur-row";
    const multGroup = document.createElement("div");
    multGroup.className = "fbt-scb-mode";
    const multBtns = [1, 2, 3, 4].map(m => {
        const btn = document.createElement("button");
        btn.className = "fbt-scb-mode-btn fbt-scb-mult-btn";
        btn.textContent = `${m}×`;
        btn.dataset.mult = m;
        btn.addEventListener("click", () => _setMultiplier(m));
        return btn;
    });
    multBtns.forEach(b => multGroup.appendChild(b));
    const durLabel = document.createElement("span");
    durLabel.className = "fbt-scb-dur-label";
    durRow.append(multGroup, durLabel);

    clipsSection.appendChild(durRow);
    clipsSection.appendChild(timelineRow);
    clipsSection.appendChild(navRow);

    wrap.appendChild(clipsSection);

    let _clips     = [];
    let _activeIdx = -1;
    let _hoverIdx  = -1;

    function _displayClips() {
        const hasTimes = _clips.some(c => c.end_time > c.start_time);
        if (hasTimes) return _clips;
        return _clips.map((c, i) => ({ ...c, start_time: i, end_time: i + 1 }));
    }

    // Windowing: with many clips, individual bands become too thin to click
    // reliably. Above ZOOM_WINDOW_SIZE clips, the timeline only renders that
    // many at a time, scaled to fill the canvas, with arrows to page _zoomStart.
    function _visibleClipRange() {
        const dc = _displayClips();
        if (dc.length <= ZOOM_WINDOW_SIZE) return null;  // fits in one view
        _zoomStart = Math.max(0, Math.min(_zoomStart, dc.length - ZOOM_WINDOW_SIZE));
        const lastIdx = Math.min(_zoomStart + ZOOM_WINDOW_SIZE, dc.length) - 1;
        return [dc[_zoomStart].start_time, dc[lastIdx].end_time];
    }

    function _updateZoomArrows() {
        const zoomed = _clips.length > ZOOM_WINDOW_SIZE;
        zoomPrevBtn.style.display = zoomed ? "" : "none";
        zoomNextBtn.style.display = zoomed ? "" : "none";
        if (!zoomed) return;
        zoomPrevBtn.disabled = _zoomStart <= 0;
        zoomNextBtn.disabled = _zoomStart + ZOOM_WINDOW_SIZE >= _clips.length;
    }

    function _timeToX(t, range, totalDur, W) {
        if (!range) return (t / totalDur) * W;
        return ((t - range[0]) / Math.max(range[1] - range[0], 0.001)) * W;
    }

    function _drawTimeline(overrideRange) {
        _updateZoomArrows();
        const dc       = _displayClips();
        const totalDur = dc.length ? Math.max(...dc.map(c => c.end_time)) : 1;
        const range    = overrideRange !== undefined ? overrideRange : _visibleClipRange();
        const W = canvas.width = canvas.offsetWidth || 300;
        const H = canvas.height;
        const ctx = canvas.getContext("2d");
        const toX = t => _timeToX(t, range, totalDur, W);
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
            if (range && (clip.end_time <= range[0] || clip.start_time >= range[1])) return;  // outside window
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

    function _fmtDur(s) {
        if (!isFinite(s) || s <= 0) return "—";
        const secs = Math.floor(s);
        const hund = Math.round((s - secs) * 100);
        return `${secs}.${String(hund).padStart(2, "0")}s`;
    }

    function _updateDurRow() {
        const clip    = _clips[_activeIdx];
        const hasClip = clip && clip.end_time > clip.start_time;
        const native  = hasClip ? clip.end_time - clip.start_time : 0;
        const scaled  = native * _multiplier;
        if (hasClip && _multiplier > 1) {
            durLabel.textContent = `${_fmtDur(native)} → ${_fmtDur(scaled)}`;
        } else if (hasClip) {
            durLabel.textContent = _fmtDur(native);
        } else {
            durLabel.textContent = "";
        }
        multBtns.forEach(btn => {
            btn.classList.toggle("active", parseInt(btn.dataset.mult, 10) === _multiplier);
        });
    }

    function _setMultiplier(m) {
        _multiplier = m;
        if (multWidget) {
            multWidget.value = m;
            app?.graph?.setDirtyCanvas?.(true, false);
        }
        _updateDurRow();
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
        _updateDurRow();
        _updateDlgFromClip();
        // Refresh source select for the visible tab only
        const srcSel = tabContent.querySelector(".fbt-scb-src-sel");
        if (srcSel) {
            const entry = _entries[_activeTabIdx];
            if (entry) _fillSourceSubjectSel(srcSel, entry.source_profile_id || "", entry.source_subject_id || "");
        }
        _updateActionPreview();
    }

    prevBtn.onclick = () => _selectClipIdx(_activeIdx - 1);
    nextBtn.onclick = () => _selectClipIdx(_activeIdx + 1);

    canvas.addEventListener("mousemove", e => {
        const rect = canvas.getBoundingClientRect();
        const x    = e.clientX - rect.left;
        const W    = rect.width;
        const dc   = _displayClips();
        const totalDur = dc.length ? Math.max(...dc.map(c => c.end_time)) : 1;
        const range = _visibleClipRange();
        let newHover = -1;
        for (let i = 0; i < dc.length; i++) {
            const x1 = _timeToX(dc[i].start_time, range, totalDur, W);
            const x2 = _timeToX(dc[i].end_time, range, totalDur, W);
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
        const W    = rect.width;
        const dc   = _displayClips();
        const totalDur = dc.length ? Math.max(...dc.map(c => c.end_time)) : 1;
        const range = _visibleClipRange();
        for (let i = 0; i < dc.length; i++) {
            const x1 = _timeToX(dc[i].start_time, range, totalDur, W);
            const x2 = _timeToX(dc[i].end_time, range, totalDur, W);
            if (x >= x1 && x <= x2) { _selectClipIdx(i); break; }
        }
    });

    new ResizeObserver(() => _drawTimeline()).observe(canvas);

    // ── Action preview ────────────────────────────────────────────────────────

    const actionPreviewEl = document.createElement("div");
    actionPreviewEl.className = "fbt-scb-action-preview";
    actionPreviewEl.style.display = "none";
    wrap.appendChild(actionPreviewEl);

    // Shown when two or more cast entries (explicit or ordinal) resolve to the
    // same source subject in the active clip — never intentional, since only
    // one bundle can actually replace a given subject.
    const conflictWarningEl = document.createElement("div");
    conflictWarningEl.className = "fbt-scb-conflict-warning";
    conflictWarningEl.style.display = "none";
    wrap.appendChild(conflictWarningEl);

    // Mirrors utils/source_profiles.py's resolved_pronoun_style() /
    // resolve_ordinal_subject() so the preview reflects ordinal-match entries
    // too, not just explicit ones. This is a display aid — the backend
    // resolution at execution time is the authoritative one.
    const _ENTITY_PRONOUN_DEFAULTS = { location: "location", object: "object", soundscape: "object" };

    function _resolvedPronounStyle(entityType, explicit) {
        return explicit || _ENTITY_PRONOUN_DEFAULTS[(entityType || "person").toLowerCase()] || "neutral";
    }

    function _bundlePronounStyle(bundleId) {
        const bun = _bundles.find(b => b.id === bundleId);
        if (!bun) return "";
        // A bundle's Pronoun dropdown defaults to "— inherit from subject —"
        // (stored as ""), matching SceneCastBuild.execute()'s own precedence:
        // the owning Subject Profile's pronoun_style wins when set, the
        // bundle's own explicit value is next, entity-type default last.
        const subj = _subjects.find(s => s.id === bun.subject_id);
        const explicit = (subj?.pronoun_style) || bun.pronoun_style || "";
        return _resolvedPronounStyle(bun.entity_type, explicit);
    }

    function _resolveOrdinalSubjectId(clip, subjects, wantPronoun, ordinal) {
        if (!clip || ordinal < 1) return "";
        const byId = new Map(subjects.map(s => [s.id, s]));
        let matches = 0;
        for (const sid of (clip.subjects || [])) {
            const subj = byId.get(sid);
            if (!subj) continue;
            if (_resolvedPronounStyle(subj.entity_type, subj.pronoun_style) === wantPronoun) {
                matches++;
                if (matches === ordinal) return sid;
            }
        }
        return "";
    }

    function _buildActionPreview() {
        const clipId = _activeClipId;
        if (!clipId) return { text: null, conflicts: [] };
        const clip   = _clipMap.get(clipId);
        const action = clip?.action;
        if (!action) return { text: null, conflicts: [] };

        const clipSubjectIds  = new Set(clip.subjects ?? []);
        const spData          = _connectedSPSubjects[0];
        const orderedSubjects = spData
            ? spData.subjects.filter(s => clipSubjectIds.has(s.id))
            : [];

        // Resolve every cast entry (explicit or ordinal) to the subject id it
        // currently applies to in this clip. Two entries landing on the same
        // subject is never intentional (only one bundle can replace a given
        // subject) — track it as a conflict rather than silently letting the
        // later entry overwrite the earlier one in the lookup.
        const resolvedIdToEntry = new Map();
        const conflicts = [];
        for (const e of _entries) {
            if (!e.bundle_id) continue;
            let sid = "";
            if (e.match_mode === "ordinal") {
                const pronoun = _bundlePronounStyle(e.bundle_id);
                sid = _resolveOrdinalSubjectId(
                    clip, spData ? spData.subjects : [], pronoun, parseInt(e.ordinal, 10) || 0,
                );
            } else if (e.source_subject_id) {
                sid = e.source_subject_id;
            }
            if (!sid) continue;
            const prior = resolvedIdToEntry.get(sid);
            if (prior) {
                const subj = spData?.subjects.find(s => s.id === sid);
                conflicts.push({
                    subjectLabel: subj?.label || sid,
                    bundleNames: [prior, e].map(x => _bundles.find(b => b.id === x.bundle_id)?.name || x.bundle_id),
                });
            }
            resolvedIdToEntry.set(sid, e);
        }

        const SLOTS = ["A","B","C","D","E","F","G","H","I","J"];
        const slotLabel = {};
        orderedSubjects.forEach((subj, sidx) => {
            if (sidx >= SLOTS.length) return;
            const castEntry = resolvedIdToEntry.get(subj.id);
            if (castEntry?.bundle_id) {
                const bun = _bundles.find(b => b.id === castEntry.bundle_id);
                slotLabel[SLOTS[sidx]] = bun?.name || castEntry.bundle_id;
            } else {
                slotLabel[SLOTS[sidx]] = subj.label || subj.id;
            }
        });

        const text = action.replace(/\{([A-J])\}/g, (match, letter) =>
            slotLabel[letter] != null ? `[${slotLabel[letter]}]` : match
        );
        return { text, conflicts };
    }

    function _updateActionPreview() {
        const { text, conflicts } = _buildActionPreview();
        if (text == null) {
            actionPreviewEl.style.display = "none";
            actionPreviewEl.textContent   = "";
        } else {
            actionPreviewEl.style.display = "";
            actionPreviewEl.textContent   = text;
        }
        if (conflicts.length) {
            conflictWarningEl.style.display = "";
            conflictWarningEl.textContent = conflicts
                .map(c => `⚠ ${c.bundleNames.join(" + ")} both resolve to "${c.subjectLabel}" — only one will actually apply.`)
                .join("\n");
        } else {
            conflictWarningEl.style.display = "none";
            conflictWarningEl.textContent = "";
        }
        // Mirror into the hidden backing widget so this value is part of the
        // submitted prompt, not just the on-canvas display.
        if (actionPreviewWidget) actionPreviewWidget.value = text ?? "";
    }

    function _updateDlgFromClip() {
        const clip = _clipMap.get(_activeClipId);
        _clipAllowsDialogue = !clip || clip.allows_dialogue !== false;
        _applyDlgState();
    }

    async function _refreshClipSelects() {
        const savedVal = clipWidget?.value ?? "";
        const inp      = node.inputs?.find(i => i.name === "source_profile");
        const linkId   = inp?.link;

        if (!linkId) {
            clipsSection.style.display = "none";
            _clips = [];
            _clipMap.clear();
            _activeIdx    = -1;
            _activeClipId = "";
            _clipAllowsDialogue = true;
            _applyDlgState();
            return;
        }
        const linkObj   = app.graph.links[linkId];
        const upstream  = linkObj ? app.graph.getNodeById(linkObj.origin_id) : null;
        const profileWidget = upstream?.widgets?.find(
            w => w.name === "profile_name" || w.name === "profile_id"
        );
        const profileVal = profileWidget?.value;
        const isName     = profileWidget?.name === "profile_name";

        if (!profileVal || profileVal === "(none)") {
            clipsSection.style.display = "none";
            _clips = [];
            _clipMap.clear();
            _activeIdx    = -1;
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

        _clips     = clips;
        _zoomStart = 0;
        _clipMap = new Map(clips.map(c => [c.id, c]));

        const savedIdx = savedVal ? clips.findIndex(c => c.id === savedVal) : -1;
        if (savedIdx >= 0) {
            _selectClipIdx(savedIdx);
        } else if (clips.length) {
            _selectClipIdx(0);
        } else {
            _activeIdx    = -1;
            _activeClipId = "";
            if (clipWidget) clipWidget.value = "";
            _drawTimeline();
            _updateNavRow();
            _updateDurRow();
            _updateDlgFromClip();
        }

        if (displayWidget) {
            displayWidget.computeSize = () => [0, _widgetHeight() + 80];
            node.setSize?.([node.size[0], node.size[1]]);
        }
    }

    node._refreshClipSelects = _refreshClipSelects;
    requestAnimationFrame(() => _refreshClipSelects());
    node._updateActionPreview = _updateActionPreview;

    // ── 16. Public refresh ────────────────────────────────────────────────────

    node._refreshCastTable = function (newEntries) {
        if (Array.isArray(newEntries)) {
            _entries = newEntries.map(e => ({ ...e }));
            if (!_entries.length) _entries.push({ subject_id: "", bundle_id: "", visual_mode: "images", use_audio: false, dialogue: "" });
            _syncWidget();
        } else {
            try {
                const parsed = JSON.parse(jsonWidget?.value || "[]");
                if (Array.isArray(parsed)) _entries = parsed;
            } catch { /* leave as-is */ }
            if (!_entries.length) _entries.push({ subject_id: "", bundle_id: "", visual_mode: "images", use_audio: false, dialogue: "" });
        }
        _activeTabIdx = Math.min(_activeTabIdx, _entries.length - 1);
        _rebuildTabs();
    };

    // ── 17. Load subject + bundle lists, then render ──────────────────────────
    Promise.allSettled([
        bundlesApi.listSubjects(),
        bundlesApi.listBundles(),
    ]).then(([subjRes, bundRes]) => {
        _subjects = subjRes.value?.subjects ?? [];
        _bundles  = bundRes.value?.bundles  ?? [];
        _rebuildTabs();
    }).catch(() => {
        _rebuildTabs();
    });

    node.size[0] = Math.max(node.size[0], 320);
    node.size[1] = Math.max(node.size[1], _widgetHeight() + 60);
}
