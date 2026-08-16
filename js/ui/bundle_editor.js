/**
 * Reference Bundle Editor sidebar panel.
 *
 * Lets users create and manage Reference Bundles — named pools of visual
 * (video or images) and audio media files scoped to a subject profile.
 *
 * Registered as a ComfyUI sidebar tab via app.extensionManager.registerSidebarTab.
 */

import { bundlesApi } from "../api/bundles.js";
import { llmApi }     from "../api/llm.js";

// ── Module state ───────────────────────────────────────────────────────────────

const _S = {
    bundles:      [],   // [{id, name, subject_id, visual, audio, tags, ...}]
    subjects:     [],   // [{id, name, appearance_summary}]
    mediaImages:  [],   // filenames from /fbtools/media/list?type=image
    mediaVideos:  [],   // filenames from /fbtools/media/list?type=video
    mediaAudio:   [],   // filenames from /fbtools/media/list?type=audio
    filterSubject: "",
    filterText:    "",
    editing:       null,  // bundle object being edited; null = list view
    isNew:         false,
    llmVision:    false,  // true when a vision-capable model is loaded
};

// Key DOM refs
const _dom = {};

// ── Helpers ────────────────────────────────────────────────────────────────────

function _mk(tag, props = {}, children = []) {
    const el = document.createElement(tag);
    Object.entries(props).forEach(([k, v]) => {
        if (k === "cls") el.className = v;
        else if (k === "style") Object.assign(el.style, v);
        else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
        else el[k] = v;
    });
    children.forEach(c => c && el.appendChild(c));
    return el;
}

function _toast(msg, severity = "info") {
    try {
        const app = window._fbtApp;
        if (app?.extensionManager?.toast) {
            app.extensionManager.toast.add({ severity, summary: msg, life: 2500 });
        }
    } catch (_) {}
}

function _subjectName(id) {
    const s = _S.subjects.find(s => s.id === id);
    return s ? (s.name || s.id) : (id || "No subject");
}

function _slugify(str) {
    return (str || "").toLowerCase().replace(/\s+/g, "_").replace(/[^\w]/g, "").slice(0, 48);
}

function _autoId(name, subjectId) {
    const base = _slugify(name);
    const sub  = _slugify(subjectId);
    if (!base) return "";
    return sub ? `${sub}_${base}` : base;
}

function _filteredBundles() {
    let list = _S.bundles;
    if (_S.filterSubject) list = list.filter(b => b.subject_id === _S.filterSubject);
    if (_S.filterText) {
        const q = _S.filterText.toLowerCase();
        list = list.filter(b =>
            (b.name || "").toLowerCase().includes(q) ||
            (b.id || "").toLowerCase().includes(q) ||
            (b.tags || []).some(t => t.toLowerCase().includes(q))
        );
    }
    return list;
}

// ── Data load ─────────────────────────────────────────────────────────────────

async function _loadAll() {
    const [bundles, subjects, imgs, vids, aud, llmSt] = await Promise.allSettled([
        bundlesApi.listBundles(),
        bundlesApi.listSubjects(),
        bundlesApi.listMedia("image"),
        bundlesApi.listMedia("video"),
        bundlesApi.listMedia("audio"),
        llmApi.status(),
    ]);
    _S.bundles     = bundles.value?.bundles   ?? [];
    _S.subjects    = subjects.value?.subjects ?? [];
    _S.mediaImages = imgs.value?.files        ?? [];
    _S.mediaVideos = vids.value?.files        ?? [];
    _S.mediaAudio  = aud.value?.files         ?? [];
    _S.llmVision   = llmSt.value?.supports_vision ?? false;
}

// ── List view ─────────────────────────────────────────────────────────────────

function _renderList() {
    const c = _dom.content;
    if (!c) return;
    c.innerHTML = "";

    const filtered = _filteredBundles();
    if (!filtered.length) {
        c.appendChild(_mk("div", { cls: "fbt-be-empty", textContent: "No bundles found. Click + New to create one." }));
        return;
    }

    // Group by subject_id in subject-list order, unknowns last
    const subjectOrder = _S.subjects.map(s => s.id);
    const grouped = new Map();
    filtered.forEach(b => {
        const k = b.subject_id || "";
        if (!grouped.has(k)) grouped.set(k, []);
        grouped.get(k).push(b);
    });

    const sortedKeys = [...grouped.keys()].sort((a, b) => {
        const ia = subjectOrder.indexOf(a);
        const ib = subjectOrder.indexOf(b);
        if (ia === -1 && ib === -1) return a.localeCompare(b);
        if (ia === -1) return 1;
        if (ib === -1) return -1;
        return ia - ib;
    });

    sortedKeys.forEach(subjectId => {
        c.appendChild(_mk("div", {
            cls: "fbt-be-group-header",
            textContent: _subjectName(subjectId),
        }));
        grouped.get(subjectId).forEach(b => c.appendChild(_buildCard(b)));
    });
}

function _buildCard(b) {
    const card = _mk("div", { cls: "fbt-be-card" });

    const top = _mk("div", { cls: "fbt-be-card-top" });
    top.appendChild(_mk("span", { cls: "fbt-be-card-name", textContent: b.name || b.id }));

    const badges = _mk("span", { cls: "fbt-be-badges" });
    const vt = b.visual?.type || "images";
    badges.appendChild(_mk("span", {
        cls: `fbt-be-badge fbt-be-badge-${vt}`,
        textContent: vt === "video" ? "VIDEO" : "IMAGES",
    }));
    if (b.audio?.source && b.audio.source !== "none") {
        badges.appendChild(_mk("span", { cls: "fbt-be-badge fbt-be-badge-audio", title: "Has audio", textContent: "🎙" }));
    }
    top.appendChild(badges);
    card.appendChild(top);

    if (b.tags?.length) {
        const tagsEl = _mk("div", { cls: "fbt-be-tags" });
        b.tags.forEach(t => tagsEl.appendChild(_mk("span", { cls: "fbt-be-tag", textContent: t })));
        card.appendChild(tagsEl);
    }

    const actions = _mk("div", { cls: "fbt-be-card-actions" });
    actions.appendChild(_mk("button", {
        cls: "fbt-ce-icon-btn", title: "Edit", textContent: "✎",
        onclick: () => _startEdit(b),
    }));
    actions.appendChild(_mk("button", {
        cls: "fbt-ce-icon-btn fbt-ce-danger", title: "Delete", textContent: "✕",
        onclick: () => _onDelete(b.id, b.name),
    }));
    card.appendChild(actions);
    return card;
}

async function _onDelete(id, name) {
    if (!confirm(`Delete bundle "${name || id}"?`)) return;
    try {
        await bundlesApi.deleteBundle(id);
        _S.bundles = _S.bundles.filter(b => b.id !== id);
        _renderList();
        _toast(`Deleted "${name || id}"`, "success");
    } catch (e) {
        _toast("Delete failed: " + e.message, "error");
    }
}

// ── Editor form ───────────────────────────────────────────────────────────────

function _startEdit(bundle) {
    _S.editing = structuredClone(bundle);
    _S.isNew   = false;
    _renderForm();
}

function _startNew(subjectId = "") {
    _S.editing = {
        id:                  "",
        name:                "",
        subject_id:          subjectId || _S.filterSubject || "",
        visual:              { type: "images", file: "", files: [], start_time: 0.0, duration: 0.0, force_rate: 0, frame_load_cap: 96, skip_first_frames: 0, select_every_nth: 1 },
        audio:               { source: "none", file: "", video_file: "", force_rate: 0, frame_load_cap: 0, skip_first_frames: 0, select_every_nth: 1, start_time: 0.0, duration: 0.0, retention: "timbre", role: "" },
        appearance_override: "",
        tags:                [],
    };
    _S.isNew = true;
    _renderForm();
}

function _cancelEdit() {
    _S.editing = null;
    _S.isNew   = false;
    _renderList();
}

function _renderForm() {
    const c = _dom.content;
    if (!c) return;
    c.innerHTML = "";

    const b = _S.editing;
    let idManuallyEdited = !_S.isNew;

    // Name
    const nameEl = _mk("input", {
        cls: "fbt-ce-input", type: "text",
        placeholder: "Bundle name *", value: b.name || "",
    });
    nameEl.addEventListener("input", () => {
        b.name = nameEl.value;
        if (!idManuallyEdited) {
            idEl.value = _autoId(b.name, b.subject_id);
            b.id = idEl.value;
        }
    });

    // ID
    const idEl = _mk("input", {
        cls: "fbt-ce-input", type: "text",
        placeholder: "bundle_id (auto-generated)", value: b.id || "",
    });
    idEl.addEventListener("input", () => {
        b.id = idEl.value.trim();
        idManuallyEdited = true;
    });

    // Subject
    const subjectEl = document.createElement("select");
    subjectEl.className = "fbt-ce-select";
    const blankOpt = document.createElement("option");
    blankOpt.value = "";
    blankOpt.textContent = "— select subject —";
    if (!b.subject_id) blankOpt.selected = true;
    subjectEl.appendChild(blankOpt);
    _S.subjects.forEach(s => {
        const o = document.createElement("option");
        o.value = s.id;
        o.textContent = s.name || s.id;
        if (s.id === b.subject_id) o.selected = true;
        subjectEl.appendChild(o);
    });
    subjectEl.addEventListener("change", () => {
        b.subject_id = subjectEl.value;
        if (!idManuallyEdited) {
            idEl.value = _autoId(b.name, b.subject_id);
            b.id = idEl.value;
        }
    });

    // Appearance override
    const appearEl = _mk("textarea", {
        cls: "fbt-ce-textarea", rows: 2,
        placeholder: "Leave empty to use subject profile appearance",
        value: b.appearance_override || "",
    });
    appearEl.addEventListener("input", () => { b.appearance_override = appearEl.value; });

    // ── Visual ─────────────────────────────────────────────────────────────────

    const visualPickerWrap = _mk("div", { cls: "fbt-be-picker-wrap" });

    const visualToggle = _buildToggle(
        ["images", "video"],
        ["Images", "Video"],
        b.visual.type,
        val => {
            b.visual.type = val;
            _rebuildVisualPicker();
            if (val === "images" && b.audio.source === "extract_from_visual") {
                b.audio.source = "none";
                audioSourceEl.value = "none";
                _rebuildAudioPicker();
            }
        }
    );

    const _rebuildVisualPicker = () => {
        visualPickerWrap.innerHTML = "";
        if (b.visual.type === "video") {
            _buildVideoPicker(visualPickerWrap, b);
        } else {
            _buildImageList(visualPickerWrap, b);
        }
    };
    _rebuildVisualPicker();

    const visualSec = _mk("div", { cls: "fbt-be-section" }, [
        _mk("div", { cls: "fbt-be-sec-label", textContent: "Visual" }),
        visualToggle,
        visualPickerWrap,
    ]);

    // ── Audio ──────────────────────────────────────────────────────────────────

    const audioPickerWrap = _mk("div", { cls: "fbt-be-picker-wrap" });

    const audioSourceEl = document.createElement("select");
    audioSourceEl.className = "fbt-ce-select";
    [
        { id: "none",                label: "None" },
        { id: "extract_from_visual", label: "Extract from video" },
        { id: "extract_from_video",  label: "Separate video file" },
        { id: "file",                label: "Separate audio file" },
    ].forEach(({ id, label }) => {
        const o = document.createElement("option");
        o.value = id;
        o.textContent = label;
        if (id === b.audio.source) o.selected = true;
        audioSourceEl.appendChild(o);
    });

    const _rebuildAudioPicker = () => {
        audioPickerWrap.innerHTML = "";
        if (b.audio.source === "extract_from_visual" && b.visual.type !== "video") {
            audioPickerWrap.appendChild(_mk("div", {
                cls: "fbt-be-warn",
                textContent: "Switch visual to Video first, or choose a separate audio file.",
            }));
        } else if (b.audio.source === "extract_from_visual") {
            _buildFrameParamSection(audioPickerWrap, b.audio, "Frame sampling (legacy VHS path)");
            _buildAudioTimeSection(audioPickerWrap, b.audio);
            _buildAudioRoleSection(audioPickerWrap, b.audio);
        } else if (b.audio.source === "extract_from_video") {
            _buildAudioVideoPicker(audioPickerWrap, b);
            _buildFrameParamSection(audioPickerWrap, b.audio, "Frame sampling");
            _buildAudioTimeSection(audioPickerWrap, b.audio);
            _buildAudioRoleSection(audioPickerWrap, b.audio);
        } else if (b.audio.source === "file") {
            _buildAudioPicker(audioPickerWrap, b);
            _buildAudioRoleSection(audioPickerWrap, b.audio);
        }
    };
    audioSourceEl.addEventListener("change", () => {
        b.audio.source = audioSourceEl.value;
        _rebuildAudioPicker();
    });
    _rebuildAudioPicker();

    const audioSec = _mk("div", { cls: "fbt-be-section" }, [
        _mk("div", { cls: "fbt-be-sec-label", textContent: "Audio" }),
        audioSourceEl,
        audioPickerWrap,
    ]);

    // Tags
    const tagsEl = _mk("input", {
        cls: "fbt-ce-input", type: "text",
        placeholder: "Tags (comma-separated)",
        value: (b.tags || []).join(", "),
    });
    tagsEl.addEventListener("input", () => {
        b.tags = tagsEl.value.split(",").map(t => t.trim()).filter(Boolean);
    });

    // Warnings
    const warnEl = _mk("div", { cls: "fbt-be-warn", style: { display: "none" } });

    // Buttons
    const btnRow = _mk("div", { cls: "fbt-be-btn-row" });
    btnRow.appendChild(_mk("button", {
        cls: "fbt-ce-btn fbt-ce-btn-primary", textContent: "Save",
        onclick: () => _onSave(b, warnEl),
    }));
    btnRow.appendChild(_mk("button", {
        cls: "fbt-ce-btn", textContent: "Cancel",
        onclick: () => _cancelEdit(),
    }));

    const form = _mk("div", { cls: "fbt-be-form" });
    form.appendChild(_formRow("Name",   nameEl));
    form.appendChild(_formRow("ID",     idEl));
    form.appendChild(_formRow("Subject", subjectEl));
    form.appendChild(_formRow("Appear.", appearEl));
    if (_S.llmVision) form.appendChild(_buildAppearanceAnalyzer(b, appearEl));
    form.appendChild(visualSec);
    form.appendChild(audioSec);
    form.appendChild(_formRow("Tags",   tagsEl));
    form.appendChild(warnEl);
    form.appendChild(btnRow);

    c.appendChild(form);
    nameEl.focus();
}

function _formRow(label, fieldEl) {
    return _mk("div", { cls: "fbt-ce-row" }, [
        _mk("div", { cls: "fbt-ce-label", textContent: label }),
        _mk("div", { cls: "fbt-ce-input-wrap" }, [fieldEl]),
    ]);
}

function _buildParamCell(labelText, obj, key, { isFloat = false, hint = "" } = {}) {
    const cell = _mk("div", { cls: "fbt-be-param-cell" });
    const lbl = _mk("span", { cls: "fbt-be-param-label", textContent: labelText });
    if (hint) lbl.title = hint;
    const inp = _mk("input", {
        cls: "fbt-be-param-input",
        type: "number", min: 0,
        step: isFloat ? 0.1 : 1,
        value: (obj[key] ?? (isFloat ? 0.0 : 0)),
    });
    inp.addEventListener("input", () => {
        obj[key] = isFloat ? parseFloat(inp.value) || 0 : parseInt(inp.value) || 0;
    });
    cell.appendChild(lbl);
    cell.appendChild(inp);
    return cell;
}

function _buildFrameParamSection(wrap, obj, title = "Frame sampling") {
    wrap.appendChild(_mk("div", { cls: "fbt-be-param-section-label", textContent: title }));
    const grid = _mk("div", { cls: "fbt-be-param-grid" });
    grid.appendChild(_buildParamCell("FPS override", obj, "force_rate",       { hint: "0 = use native fps" }));
    grid.appendChild(_buildParamCell("Frame cap",    obj, "frame_load_cap",   { hint: "0 = no cap" }));
    grid.appendChild(_buildParamCell("Skip first",   obj, "skip_first_frames"));
    grid.appendChild(_buildParamCell("Every Nth",    obj, "select_every_nth", { hint: "1 = every frame" }));
    wrap.appendChild(grid);
}

function _buildAudioTimeSection(wrap, obj) {
    wrap.appendChild(_mk("div", { cls: "fbt-be-param-section-label", textContent: "Timing" }));
    const grid = _mk("div", { cls: "fbt-be-param-grid" });
    grid.appendChild(_buildParamCell("Start (s)", obj, "start_time", { isFloat: true, hint: "Start time in seconds" }));
    grid.appendChild(_buildParamCell("Duration (s)", obj, "duration", { isFloat: true, hint: "0 = to end of file" }));
    wrap.appendChild(grid);
}

function _buildAudioRoleSection(wrap, audio) {
    wrap.appendChild(_mk("div", { cls: "fbt-be-param-section-label", textContent: "Prompt role" }));

    const retSel = document.createElement("select");
    retSel.className = "fbt-ce-select";
    [
        { id: "timbre", label: "Timbre reference — do not copy signal" },
        { id: "reuse",  label: "Audio reuse — reproduce verbatim" },
        { id: "style",  label: "Style / rhythm reference" },
    ].forEach(({ id, label }) => {
        const o = document.createElement("option");
        o.value = id;
        o.textContent = label;
        if (id === (audio.retention || "timbre")) o.selected = true;
        retSel.appendChild(o);
    });
    retSel.addEventListener("change", () => { audio.retention = retSel.value; });

    const roleEl = _mk("input", {
        cls: "fbt-ce-input", type: "text",
        placeholder: "Role description (auto-generated if empty)",
        value: audio.role || "",
    });
    roleEl.addEventListener("input", () => { audio.role = roleEl.value; });

    wrap.appendChild(retSel);
    wrap.appendChild(roleEl);
}

function _buildToggle(values, labels, current, onChange) {
    const wrap = _mk("div", { cls: "fbt-be-toggle-row" });
    values.forEach((val, i) => {
        const btn = _mk("button", {
            cls: "fbt-be-toggle-btn" + (val === current ? " active" : ""),
            textContent: labels[i],
        });
        btn.addEventListener("click", () => {
            wrap.querySelectorAll(".fbt-be-toggle-btn").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            onChange(val);
        });
        wrap.appendChild(btn);
    });
    return wrap;
}

function _buildVideoPicker(wrap, b) {
    if (!_S.mediaVideos.length) {
        wrap.appendChild(_mk("div", { cls: "fbt-be-media-empty", textContent: "No video files in input directory" }));
    } else {
        const sel = document.createElement("select");
        sel.className = "fbt-ce-select";
        const blank = document.createElement("option");
        blank.value = "";
        blank.textContent = "— select video file —";
        if (!b.visual.file) blank.selected = true;
        sel.appendChild(blank);
        _S.mediaVideos.forEach(f => {
            const o = document.createElement("option");
            o.value = f;
            o.textContent = f;
            if (f === b.visual.file) o.selected = true;
            sel.appendChild(o);
        });
        sel.addEventListener("change", () => { b.visual.file = sel.value; });
        wrap.appendChild(sel);
    }
    wrap.appendChild(_mk("div", { cls: "fbt-be-param-section-label", textContent: "Trim" }));
    const trimGrid = _mk("div", { cls: "fbt-be-param-grid" });
    trimGrid.appendChild(_buildParamCell("Start (s)",    b.visual, "start_time", { isFloat: true, hint: "Start time in seconds" }));
    trimGrid.appendChild(_buildParamCell("Duration (s)", b.visual, "duration",   { isFloat: true, hint: "0 = to end of file" }));
    wrap.appendChild(trimGrid);
    _buildFrameParamSection(wrap, b.visual);
}

function _buildImageList(wrap, b) {
    const rebuildList = () => {
        listEl.innerHTML = "";
        if (!b.visual.files.length) {
            listEl.appendChild(_mk("div", { cls: "fbt-be-media-empty", textContent: "No images selected" }));
        } else {
            b.visual.files.forEach((f, i) => {
                const row = _mk("div", { cls: "fbt-be-img-row" });
                row.appendChild(_mk("span", { cls: "fbt-be-img-name", textContent: f, title: f }));
                const btns = _mk("span", { cls: "fbt-be-img-btns" });
                if (i > 0) {
                    btns.appendChild(_mk("button", {
                        cls: "fbt-ce-icon-btn", textContent: "↑", title: "Move up",
                        onclick: () => {
                            [b.visual.files[i - 1], b.visual.files[i]] = [b.visual.files[i], b.visual.files[i - 1]];
                            rebuildList();
                        },
                    }));
                }
                if (i < b.visual.files.length - 1) {
                    btns.appendChild(_mk("button", {
                        cls: "fbt-ce-icon-btn", textContent: "↓", title: "Move down",
                        onclick: () => {
                            [b.visual.files[i], b.visual.files[i + 1]] = [b.visual.files[i + 1], b.visual.files[i]];
                            rebuildList();
                        },
                    }));
                }
                btns.appendChild(_mk("button", {
                    cls: "fbt-ce-icon-btn fbt-ce-danger", textContent: "✕", title: "Remove",
                    onclick: () => {
                        b.visual.files.splice(i, 1);
                        rebuildList();
                        rebuildAddSel();
                    },
                }));
                row.appendChild(btns);
                listEl.appendChild(row);
            });
        }
    };

    const rebuildAddSel = () => {
        addSel.innerHTML = "";
        const available = _S.mediaImages.filter(f => !b.visual.files.includes(f));
        const placeholder = document.createElement("option");
        placeholder.value = "";
        placeholder.textContent = available.length ? "Add image…" : "All available images selected";
        addSel.appendChild(placeholder);
        available.forEach(f => {
            const o = document.createElement("option");
            o.value = f;
            o.textContent = f;
            addSel.appendChild(o);
        });
        addSel.style.display = _S.mediaImages.length ? "" : "none";
    };

    const listEl = _mk("div", { cls: "fbt-be-img-list" });
    const addSel = document.createElement("select");
    addSel.className = "fbt-ce-select";
    addSel.style.marginTop = "4px";
    addSel.addEventListener("change", () => {
        if (!addSel.value) return;
        if (!b.visual.files.includes(addSel.value)) {
            b.visual.files.push(addSel.value);
            rebuildList();
            rebuildAddSel();
        }
        addSel.value = "";
    });

    if (!_S.mediaImages.length) {
        wrap.appendChild(_mk("div", { cls: "fbt-be-media-empty", textContent: "No image files in input directory" }));
        return;
    }

    rebuildList();
    rebuildAddSel();
    wrap.appendChild(listEl);
    wrap.appendChild(addSel);
}

function _buildAudioVideoPicker(wrap, b) {
    if (!_S.mediaVideos.length) {
        wrap.appendChild(_mk("div", { cls: "fbt-be-media-empty", textContent: "No video files in input directory" }));
        return;
    }
    const sel = document.createElement("select");
    sel.className = "fbt-ce-select";
    const blank = document.createElement("option");
    blank.value = "";
    blank.textContent = "— select video file —";
    if (!b.audio.video_file) blank.selected = true;
    sel.appendChild(blank);
    _S.mediaVideos.forEach(f => {
        const o = document.createElement("option");
        o.value = f;
        o.textContent = f;
        if (f === b.audio.video_file) o.selected = true;
        sel.appendChild(o);
    });
    sel.addEventListener("change", () => { b.audio.video_file = sel.value; });
    wrap.appendChild(sel);
}

function _buildAudioPicker(wrap, b) {
    if (!_S.mediaAudio.length) {
        wrap.appendChild(_mk("div", { cls: "fbt-be-media-empty", textContent: "No audio files in input directory" }));
    } else {
        const sel = document.createElement("select");
        sel.className = "fbt-ce-select";
        const blank = document.createElement("option");
        blank.value = "";
        blank.textContent = "— select audio file —";
        if (!b.audio.file) blank.selected = true;
        sel.appendChild(blank);
        _S.mediaAudio.forEach(f => {
            const o = document.createElement("option");
            o.value = f;
            o.textContent = f;
            if (f === b.audio.file) o.selected = true;
            sel.appendChild(o);
        });
        sel.addEventListener("change", () => { b.audio.file = sel.value; });
        wrap.appendChild(sel);
    }
    _buildAudioTimeSection(wrap, b.audio);
}

const _DEFAULT_APPEARANCE_QUERY =
    "Describe this person's physical appearance for a video generation prompt. " +
    "Include hair color and style, eye color (if visible), skin tone, facial structure, " +
    "build and approximate height, age range, and any distinctive features. " +
    "Write two to four plain English sentences. Do not use markdown, bullet points, or headings.";

function _buildAppearanceAnalyzer(b, appearEl) {
    const isVideoMode = b.visual.type === "video" && !!b.visual.file;
    let _currentTmpFrame = null;  // temp filename on server; replaced on each extraction

    const sec = _mk("div", { cls: "fbt-be-llm-section" });

    // ── Header ────────────────────────────────────────────────────────────────
    const hdrHint = isVideoMode ? "extract a frame from the video reference"
        : (b.visual.files?.length ? "" : "no images in bundle — using full media pool");
    sec.appendChild(_mk("div", { cls: "fbt-be-llm-header" }, [
        _mk("span", { cls: "fbt-be-llm-title", textContent: "Analyze Appearance" }),
        ...(hdrHint ? [_mk("span", { cls: "fbt-be-llm-hint", textContent: `(${hdrHint})` })] : []),
    ]));

    // ── Preview image (shared) ─────────────────────────────────────────────────
    const previewImg = _mk("img", { cls: "fbt-be-llm-preview", style: { display: "none" } });
    previewImg.alt = "";

    // ── Source section ────────────────────────────────────────────────────────
    let getSourceImage;   // () => filename string | null

    if (isVideoMode) {
        // Video mode: frame extractor
        const frameInput = _mk("input", {
            cls: "fbt-ce-input fbt-be-llm-frame-input",
            type: "number", value: "0", min: "0", step: "1",
            title: "Frame index (0 = first frame)",
        });
        const frameCountEl = _mk("span", { cls: "fbt-be-llm-frame-count", textContent: "" });

        const extractBtn = _mk("button", {
            cls: "fbt-ce-btn",
            textContent: "Extract Frame",
            onclick: async () => {
                // Replace previous temp frame
                if (_currentTmpFrame) {
                    bundlesApi.deleteTmpFrame(_currentTmpFrame).catch(() => {});
                    _currentTmpFrame = null;
                }
                extractBtn.disabled = true;
                extractBtn.textContent = "Extracting…";
                previewImg.style.display = "none";
                try {
                    const r = await bundlesApi.extractFrame(
                        b.visual.file, parseInt(frameInput.value, 10) || 0);
                    _currentTmpFrame = r.tmp_filename;
                    frameInput.max = r.frame_count - 1;
                    frameCountEl.textContent = `of ${r.frame_count} frames  (${r.width}×${r.height})`;
                    previewImg.src = `/view?filename=${encodeURIComponent(r.tmp_filename)}&type=input&subfolder=`;
                    previewImg.style.display = "";
                } catch (e) {
                    _toast("Frame extraction failed: " + e.message, "error");
                } finally {
                    extractBtn.disabled = false;
                    extractBtn.textContent = "Extract Frame";
                }
            },
        });

        getSourceImage = () => _currentTmpFrame;

        sec.appendChild(_mk("div", { cls: "fbt-be-llm-video-row" }, [
            _mk("span", { cls: "fbt-be-llm-video-name", textContent: b.visual.file }),
        ]));
        sec.appendChild(_mk("div", { cls: "fbt-be-llm-top-row" }, [
            frameInput, frameCountEl, extractBtn,
        ]));

    } else {
        // Image mode: dropdown of bundle images or full media pool
        const imagePool = b.visual.files?.length ? b.visual.files : _S.mediaImages;
        const imgSel = document.createElement("select");
        imgSel.className = "fbt-ce-select fbt-be-llm-img-sel";
        const blankOpt = document.createElement("option");
        blankOpt.value = "";
        blankOpt.textContent = "— select image —";
        imgSel.appendChild(blankOpt);
        imagePool.forEach(f => {
            const o = document.createElement("option");
            o.value = f; o.textContent = f;
            imgSel.appendChild(o);
        });
        if (!imagePool.length) { imgSel.disabled = true; imgSel.title = "No images available"; }
        imgSel.addEventListener("change", () => {
            const f = imgSel.value;
            previewImg.src = f ? `/view?filename=${encodeURIComponent(f)}&type=input&subfolder=` : "";
            previewImg.style.display = f ? "" : "none";
        });

        getSourceImage = () => imgSel.value || null;

        sec.appendChild(_mk("div", { cls: "fbt-be-llm-top-row" }, [imgSel]));
    }

    sec.appendChild(previewImg);

    // ── Query ─────────────────────────────────────────────────────────────────
    sec.appendChild(_mk("textarea", {
        cls: "fbt-ce-textarea fbt-be-llm-query",
        rows: 3,
        value: _DEFAULT_APPEARANCE_QUERY,
    }));

    // ── Result ────────────────────────────────────────────────────────────────
    const resultEl = _mk("textarea", {
        cls: "fbt-ce-textarea fbt-be-llm-result",
        rows: 3,
        placeholder: "Generated description will appear here…",
        style: { display: "none" },
    });
    sec.appendChild(resultEl);

    // ── Apply buttons ─────────────────────────────────────────────────────────
    const applyRow = _mk("div", { cls: "fbt-be-llm-apply-row", style: { display: "none" } });
    applyRow.appendChild(_mk("button", {
        cls: "fbt-ce-btn",
        textContent: "→ Bundle",
        title: "Copy to appearance override for this bundle",
        onclick: () => {
            appearEl.value = resultEl.value;
            b.appearance_override = resultEl.value;
            _toast("Applied to bundle appearance override", "success");
        },
    }));
    applyRow.appendChild(_mk("button", {
        cls: "fbt-ce-btn",
        textContent: "→ Subject Profile",
        title: "Save as appearance summary on the subject profile",
        onclick: async () => {
            const sid = b.subject_id;
            if (!sid) { _toast("No subject selected on this bundle", "warn"); return; }
            try {
                await bundlesApi.saveSubjectAppearance(sid, resultEl.value);
                _toast("Saved to subject profile", "success");
                const res = await bundlesApi.listSubjects();
                _S.subjects = res.subjects ?? [];
            } catch (e) {
                _toast("Save failed: " + e.message, "error");
            }
        },
    }));
    sec.appendChild(applyRow);

    // ── Analyze button ────────────────────────────────────────────────────────
    const queryEl = sec.querySelector("textarea.fbt-be-llm-query");
    const analyzeBtn = _mk("button", {
        cls: "fbt-ce-btn fbt-be-llm-analyze-btn",
        textContent: "🔍 Analyze",
        onclick: async () => {
            const img = getSourceImage();
            if (!img) {
                _toast(isVideoMode ? "Extract a frame first" : "Select an image to analyze", "warn");
                return;
            }
            analyzeBtn.disabled = true;
            analyzeBtn.textContent = "Analyzing…";
            try {
                const query = sec.querySelector("textarea.fbt-be-llm-query")?.value.trim()
                    || _DEFAULT_APPEARANCE_QUERY;
                const r = await llmApi.generate(query, { images: [img], max_tokens: 400 });
                if (r?.text) {
                    resultEl.value = r.text.trim();
                    resultEl.style.display = "";
                    applyRow.style.display = "";
                }
            } catch (e) {
                _toast("LLM error: " + e.message, "error");
            } finally {
                analyzeBtn.disabled = false;
                analyzeBtn.textContent = "🔍 Analyze";
            }
        },
    });
    // Insert analyze button after the query textarea, before the result
    sec.insertBefore(analyzeBtn, resultEl);

    return sec;
}

async function _onSave(b, warnEl) {
    const name = (b.name || "").trim();
    const id   = (b.id   || "").trim();
    if (!name) { _toast("Bundle name is required", "warn"); return; }
    if (!id)   { _toast("Bundle ID is required", "warn"); return; }

    warnEl.style.display = "none";
    try {
        await bundlesApi.saveBundle({ ...b, id, name });
        const res = await bundlesApi.listBundles();
        _S.bundles = res.bundles ?? [];
        _S.editing = null;
        _S.isNew   = false;
        _renderList();
        _toast(`Saved "${name}"`, "success");
    } catch (e) {
        warnEl.textContent = e.message;
        warnEl.style.display = "";
        _toast("Save failed", "error");
    }
}

// ── Top bar ───────────────────────────────────────────────────────────────────

function _buildTopBar() {
    const bar = _mk("div", { cls: "fbt-be-top-bar" });

    // Subject filter
    const subjSel = document.createElement("select");
    subjSel.className = "fbt-ce-select fbt-be-filter-sel";
    const allOpt = document.createElement("option");
    allOpt.value = "";
    allOpt.textContent = "All subjects";
    subjSel.appendChild(allOpt);
    _S.subjects.forEach(s => {
        const o = document.createElement("option");
        o.value = s.id;
        o.textContent = s.name || s.id;
        if (s.id === _S.filterSubject) o.selected = true;
        subjSel.appendChild(o);
    });
    subjSel.addEventListener("change", () => {
        _S.filterSubject = subjSel.value;
        if (!_S.editing) _renderList();
    });
    _dom.subjSel = subjSel;

    // Search
    const searchEl = _mk("input", {
        cls: "fbt-ce-input fbt-be-search",
        type: "text", placeholder: "Search…",
        value: _S.filterText,
    });
    searchEl.addEventListener("input", () => {
        _S.filterText = searchEl.value;
        if (!_S.editing) _renderList();
    });

    // New bundle
    const newBtn = _mk("button", {
        cls: "fbt-ce-btn fbt-ce-btn-primary fbt-be-new-btn",
        textContent: "+ New",
        onclick: () => _startNew(_S.filterSubject),
    });

    // Refresh
    const refreshBtn = _mk("button", {
        cls: "fbt-ce-icon-btn fbt-be-refresh-btn",
        textContent: "↺", title: "Refresh media and bundle list",
        onclick: async () => {
            try {
                await _loadAll();
                _repopulateSubjectFilter();
                if (!_S.editing) _renderList();
                _toast("Refreshed", "success");
            } catch (e) {
                _toast("Refresh failed: " + e.message, "error");
            }
        },
    });

    bar.appendChild(subjSel);
    bar.appendChild(searchEl);
    bar.appendChild(newBtn);
    bar.appendChild(refreshBtn);
    return bar;
}

function _repopulateSubjectFilter() {
    const sel = _dom.subjSel;
    if (!sel) return;
    const current = sel.value;
    sel.innerHTML = "";
    const all = document.createElement("option");
    all.value = "";
    all.textContent = "All subjects";
    sel.appendChild(all);
    _S.subjects.forEach(s => {
        const o = document.createElement("option");
        o.value = s.id;
        o.textContent = s.name || s.id;
        if (s.id === current) o.selected = true;
        sel.appendChild(o);
    });
}

// ── Main render ───────────────────────────────────────────────────────────────

export async function renderBundleEditor(el) {
    el.innerHTML = "";

    const panel = _mk("div", { cls: "fbt-be-panel" });
    _dom.content = _mk("div", { cls: "fbt-be-content" });

    // Show loading state while fetching
    _dom.content.appendChild(_mk("div", { cls: "fbt-be-empty", textContent: "Loading…" }));

    panel.appendChild(_buildTopBar());
    panel.appendChild(_dom.content);
    el.appendChild(panel);

    try {
        await _loadAll();
    } catch (e) {
        console.error("fbt BundleEditor: load error", e);
    }

    _repopulateSubjectFilter();
    _renderList();
}
