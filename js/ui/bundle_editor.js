/**
 * Reference Bundle Editor sidebar panel.
 *
 * Lets users create and manage Reference Bundles — named pools of visual
 * (video or images) and audio media files scoped to a subject profile.
 *
 * Registered as a ComfyUI sidebar tab via app.extensionManager.registerSidebarTab.
 */

import { bundlesApi }    from "../api/bundles.js";
import { llmApi }        from "../api/llm.js";
import { compositionsApi } from "../api/compositions.js";

const BUNDLE_PAGE_SIZE = 10;

const SHEET_ROLES = ["character sheet", "portrait", "side profile", "full body", "costume detail", "reference"];

// ── Module state ───────────────────────────────────────────────────────────────

const _S = {
    bundles:      [],   // [{id, name, subject_id, visual, audio, tags, ...}]
    subjects:     [],   // [{id, name, appearance_summary}]
    mediaImages:  [],   // filenames from /fbtools/media/list?type=image
    mediaVideos:  [],   // filenames from /fbtools/media/list?type=video
    mediaAudio:   [],   // filenames from /fbtools/media/list?type=audio
    settings:     {},   // global composition settings (audio processing defaults, etc.)
    filterSubject:  "",
    filterText:     "",
    listPage:       0,
    lastId:         "",
    editing:        null,  // bundle object being edited; null = list view
    isNew:          false,
    llmVision:      false,  // true when a vision-capable model is loaded
    // Subject editor state
    viewMode:       "bundles",  // "bundles" | "subjects"
    subjectEditing: null,       // full subject dict being edited; null = list view
    subjectIsNew:   false,
    subjectFilter:  "",
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
    const [bundles, subjects, imgs, vids, aud, llmSt, settingsRes] = await Promise.allSettled([
        bundlesApi.listBundles(),
        bundlesApi.listSubjects(),
        bundlesApi.listMedia("image", true),  // recursive — includes subdirectories
        bundlesApi.listMedia("video"),
        bundlesApi.listMedia("audio"),
        llmApi.status(),
        compositionsApi.getSettings(),
    ]);
    _S.bundles     = bundles.value?.bundles   ?? [];
    _S.subjects    = subjects.value?.subjects ?? [];
    _S.mediaImages = imgs.value?.files        ?? [];
    _S.mediaVideos = vids.value?.files        ?? [];
    _S.mediaAudio  = aud.value?.files         ?? [];
    _S.llmVision   = llmSt.value?.supports_vision ?? false;
    if (settingsRes.value) _S.settings = settingsRes.value;
}

// ── List view ─────────────────────────────────────────────────────────────────

function _renderList() {
    const c = _dom.content;
    if (!c) return;
    c.innerHTML = "";

    const filtered = _filteredBundles();

    // Build flat ordered list, preserving subject grouping order
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
    const flat = [];
    sortedKeys.forEach(sid => grouped.get(sid).forEach(b => flat.push({ sid, b })));

    const total      = flat.length;
    const totalPages = Math.max(1, Math.ceil(total / BUNDLE_PAGE_SIZE));
    _S.listPage      = Math.max(0, Math.min(_S.listPage, totalPages - 1));
    const start      = _S.listPage * BUNDLE_PAGE_SIZE;
    const pageItems  = flat.slice(start, start + BUNDLE_PAGE_SIZE);

    if (!pageItems.length) {
        c.appendChild(_mk("div", { cls: "fbt-be-empty", textContent: "No bundles found. Click + New to create one." }));
    } else {
        let lastSid = null;
        pageItems.forEach(({ sid, b }) => {
            if (sid !== lastSid) {
                c.appendChild(_mk("div", { cls: "fbt-be-group-header", textContent: _subjectName(sid) }));
                lastSid = sid;
            }
            c.appendChild(_buildCard(b));
        });
    }

    if (_dom.pagination) {
        _dom.pagination.innerHTML = "";
        if (totalPages > 1) {
            const prevBtn = _mk("button", {
                cls: "fbt-ce-pg-btn", textContent: "‹", title: "Previous page",
                onclick: () => { _S.listPage--; _renderList(); },
            });
            prevBtn.disabled = _S.listPage === 0;
            const info = _mk("span", {
                cls: "fbt-ce-pg-info",
                textContent: `${_S.listPage + 1} / ${totalPages}`,
            });
            const nextBtn = _mk("button", {
                cls: "fbt-ce-pg-btn", textContent: "›", title: "Next page",
                onclick: () => { _S.listPage++; _renderList(); },
            });
            nextBtn.disabled = _S.listPage >= totalPages - 1;
            _dom.pagination.appendChild(prevBtn);
            _dom.pagination.appendChild(info);
            _dom.pagination.appendChild(nextBtn);
        }
    }
}

function _buildCard(b) {
    const isActive = b.id && b.id === _S.lastId;
    const card = _mk("div", {
        cls: "fbt-be-card fbt-be-card-clickable" + (isActive ? " fbt-be-card-active" : ""),
    });
    card.addEventListener("click", () => _startEdit(b));

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
        onclick: (e) => { e.stopPropagation(); _startEdit(b); },
    }));
    actions.appendChild(_mk("button", {
        cls: "fbt-ce-icon-btn fbt-ce-danger", title: "Delete", textContent: "✕",
        onclick: (e) => { e.stopPropagation(); _onDelete(b.id, b.name); },
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
        visual:              { type: "images", file: "", files: [], start_time: 0.0, duration: 0.0, force_rate: 0, frame_load_cap: 0, skip_first_frames: 0, select_every_nth: 1 },
        audio:               { source: "none", file: "", video_file: "", force_rate: 0, frame_load_cap: 0, skip_first_frames: 0, select_every_nth: 1, start_time: 0.0, duration: 0.0, retention: "timbre", role: "", audio_processing: { noise_removal: !!(_S.settings.default_audio_noise_removal), normalize_lufs: _S.settings.default_audio_normalize_lufs !== false, target_lufs: _S.settings.default_audio_target_lufs ?? -14.0 }, audio_cache: "" },
        appearance_override: "",
        tags:                [],
    };
    _S.isNew = true;
    _renderForm();
}

function _cancelEdit() {
    _S.lastId  = _S.editing?.id || _S.lastId;
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

    // _llmEl is set below after the analyzer is built; the closure captures it by reference
    // so _rebuildVisualPicker can pass the refresh callback even though _llmEl is null now.
    let _llmEl = null;

    const _rebuildVisualPicker = () => {
        visualPickerWrap.innerHTML = "";
        if (b.visual.type === "video") {
            _buildVideoPicker(visualPickerWrap, b);
        } else {
            _buildImageList(visualPickerWrap, b, () => _llmEl?._refreshPool?.());
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
            _buildFrameParamSection(audioPickerWrap, b.audio, { title: "Frame sampling (legacy VHS path)" });
            _buildAudioTimeSection(audioPickerWrap, b.audio);
            _buildAudioRoleSection(audioPickerWrap, b.audio);
            _buildAudioProcessingSection(audioPickerWrap, b, null);
        } else if (b.audio.source === "extract_from_video") {
            const srcPlayerA = _buildAudioVideoPicker(audioPickerWrap, b);
            _buildFrameParamSection(audioPickerWrap, b.audio, { title: "Frame sampling" });
            _buildAudioTimeSection(audioPickerWrap, b.audio);
            _buildAudioRoleSection(audioPickerWrap, b.audio);
            _buildAudioProcessingSection(audioPickerWrap, b, srcPlayerA);
        } else if (b.audio.source === "file") {
            const srcPlayerB = _buildAudioPicker(audioPickerWrap, b);
            _buildAudioRoleSection(audioPickerWrap, b.audio);
            _buildAudioProcessingSection(audioPickerWrap, b, srcPlayerB);
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
    if (_S.llmVision) {
        _llmEl = _buildAppearanceAnalyzer(b, appearEl);
        form.appendChild(_llmEl);
    }
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

function _buildParamCell(labelText, obj, key, { isFloat = false, hint = "", onchange } = {}) {
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
        onchange?.();
    });
    cell.appendChild(lbl);
    cell.appendChild(inp);
    return cell;
}

function _buildFrameParamSection(wrap, obj, { title = "Frame sampling", onchange } = {}) {
    wrap.appendChild(_mk("div", { cls: "fbt-be-param-section-label", textContent: title }));
    const grid = _mk("div", { cls: "fbt-be-param-grid" });
    grid.appendChild(_buildParamCell("FPS override", obj, "force_rate",       { hint: "0 = use native fps", onchange }));
    grid.appendChild(_buildParamCell("Every Nth",    obj, "select_every_nth", { hint: "1 = every frame, 2 = half frames (saves VRAM)", onchange }));
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

function _buildRangeSlider(minVal, maxVal, loVal, hiVal, { step = 0.1, onchange } = {}) {
    loVal = Math.max(minVal, Math.min(loVal, maxVal));
    hiVal = Math.max(loVal,  Math.min(hiVal, maxVal));

    const wrap = _mk("div", { cls: "fbt-range-wrap" });
    const track = _mk("div", { cls: "fbt-range-track" });
    const fill  = _mk("div", { cls: "fbt-range-fill" });
    track.appendChild(fill);
    wrap.appendChild(track);

    const lo = _mk("input", { cls: "fbt-range-input fbt-range-lo",
        type: "range", min: minVal, max: maxVal, step, value: loVal });
    const hi = _mk("input", { cls: "fbt-range-input fbt-range-hi",
        type: "range", min: minVal, max: maxVal, step, value: hiVal });

    const loLabel = _mk("span", { cls: "fbt-range-lo-label" });
    const hiLabel = _mk("span", { cls: "fbt-range-hi-label" });
    wrap.appendChild(lo);
    wrap.appendChild(hi);
    wrap.appendChild(_mk("div", { cls: "fbt-range-labels" }, [loLabel, hiLabel]));

    const range = maxVal - minVal;
    const fmt = v => {
        const m = Math.floor(v / 60);
        const s = (v % 60).toFixed(1);
        return m > 0 ? `${m}:${s.padStart(4, "0")}` : `${s}s`;
    };

    const updateFill = () => {
        if (range <= 0) return;
        const loV = parseFloat(lo.value);
        const hiV = parseFloat(hi.value);
        const loP = ((loV - minVal) / range) * 100;
        const hiP = ((hiV - minVal) / range) * 100;
        fill.style.left  = loP + "%";
        fill.style.width = (hiP - loP) + "%";
        loLabel.textContent = fmt(loV);
        hiLabel.textContent = fmt(hiV);
        lo.style.zIndex = loV >= hiV - range * 0.02 ? 5 : 2;
    };

    lo.addEventListener("input", () => {
        if (parseFloat(lo.value) > parseFloat(hi.value)) lo.value = hi.value;
        updateFill();
        onchange && onchange(parseFloat(lo.value), parseFloat(hi.value));
    });
    hi.addEventListener("input", () => {
        if (parseFloat(hi.value) < parseFloat(lo.value)) hi.value = lo.value;
        updateFill();
        onchange && onchange(parseFloat(lo.value), parseFloat(hi.value));
    });

    updateFill();

    return {
        el: wrap, lo, hi, updateFill,
        setValues(newLo, newHi) {
            lo.value = Math.max(minVal, Math.min(newLo, maxVal));
            hi.value = Math.max(parseFloat(lo.value), Math.min(newHi, maxVal));
            updateFill();
        },
    };
}

function _buildVideoPicker(wrap, b) {
    if (!_S.mediaVideos.length) {
        wrap.appendChild(_mk("div", { cls: "fbt-be-media-empty", textContent: "No video files in input directory" }));
        _buildFrameParamSection(wrap, b.visual);
        return;
    }

    // ── File selector ──────────────────────────────────────────────────────────
    const sel = document.createElement("select");
    sel.className = "fbt-ce-select";
    const blank = document.createElement("option");
    blank.value = ""; blank.textContent = "— select video file —";
    if (!b.visual.file) blank.selected = true;
    sel.appendChild(blank);
    _S.mediaVideos.forEach(f => {
        const o = document.createElement("option");
        o.value = f; o.textContent = f;
        if (f === b.visual.file) o.selected = true;
        sel.appendChild(o);
    });

    // ── Preview ────────────────────────────────────────────────────────────────
    const videoEl = document.createElement("video");
    videoEl.className = "fbt-be-video-preview";
    videoEl.controls = true;
    videoEl.preload = "metadata";
    videoEl.style.display = "none";

    const infoEl = _mk("div", { cls: "fbt-be-video-info" });
    infoEl.style.display = "none";

    // ── Slider container ───────────────────────────────────────────────────────
    const sliderWrap = _mk("div");
    sliderWrap.style.display = "none";

    // ── Mark start/end buttons ─────────────────────────────────────────────────
    const markRow = _mk("div", { cls: "fbt-be-mark-row" });
    markRow.style.display = "none";

    // ── Trim number inputs (always visible; synced with slider) ───────────────
    wrap.appendChild(sel);
    wrap.appendChild(videoEl);
    wrap.appendChild(infoEl);
    wrap.appendChild(sliderWrap);
    wrap.appendChild(markRow);
    wrap.appendChild(_mk("div", { cls: "fbt-be-param-section-label", textContent: "Trim" }));

    const trimGrid = _mk("div", { cls: "fbt-be-param-grid" });

    // Declare early so trim input listeners can reference these via closure
    let _updateFrameCount = () => {};
    const frameCountReadout = _mk("div", { cls: "fbt-be-frame-count-readout" });
    let _previewBlobUrl  = null;
    let _previewVideoEl  = null;
    let _previewWrap     = null;

    // Build cells manually so we hold the <input> refs for two-way sync
    const _makeCell = (label, key, hint) => {
        const cell = _mk("div", { cls: "fbt-be-param-cell" });
        const lbl  = _mk("span", { cls: "fbt-be-param-label", textContent: label });
        if (hint) lbl.title = hint;
        const inp  = _mk("input", {
            cls: "fbt-be-param-input", type: "number", min: 0, step: 0.1,
            value: (b.visual[key] ?? 0.0),
        });
        inp.addEventListener("input", () => {
            b.visual[key] = parseFloat(inp.value) || 0;
            _syncSlider();
            _updateFrameCount();
        });
        cell.appendChild(lbl); cell.appendChild(inp);
        return { cell, inp };
    };
    const { cell: startCell, inp: startInp } = _makeCell("Start (s)",    "start_time", "Start time in seconds");
    const { cell: durCell,   inp: durInp   } = _makeCell("Duration (s)", "duration",   "0 = to end of file");
    trimGrid.appendChild(startCell);
    trimGrid.appendChild(durCell);
    wrap.appendChild(trimGrid);

    // ── Slider / mark logic ────────────────────────────────────────────────────
    let _slider    = null;
    let _vidDur    = 0;
    let _onMeta    = null;
    let _onErr     = null;
    let _vidInfo   = null;  // last mediaInfo response; available to error handler

    // Frame-count readout — updated whenever clip range or sampling params change
    _updateFrameCount = () => {
        const dur = _vidDur || 0;
        if (!dur) { frameCountReadout.textContent = ""; return; }
        const nativeFps = _vidInfo?.fps || 24;
        const targetFps = b.visual.force_rate > 0 ? b.visual.force_rate : nativeFps;
        const every     = Math.max(1, b.visual.select_every_nth || 1);
        const startT    = b.visual.start_time || 0;
        const clipDur   = b.visual.duration > 0 ? b.visual.duration : Math.max(0, dur - startT);
        const outFrames = Math.floor(Math.floor(clipDur * targetFps) / every);
        const effFps    = (targetFps / every).toFixed(1);
        frameCountReadout.textContent = `~${outFrames} frames · ${effFps} fps effective`;
    };

    const _syncSlider = () => {
        if (!_slider || _vidDur <= 0) return;
        const lo  = Math.max(0, parseFloat(b.visual.start_time) || 0);
        const dur = parseFloat(b.visual.duration) || 0;
        const hi  = dur > 0 ? lo + dur : _vidDur;
        _slider.setValues(lo, Math.min(hi, _vidDur));
    };

    const _buildSlider = () => {
        sliderWrap.innerHTML = "";
        markRow.innerHTML    = "";
        if (_vidDur <= 0) { sliderWrap.style.display = "none"; markRow.style.display = "none"; return; }

        const step = Math.max(0.05, _vidDur / 2000);
        const lo0  = parseFloat(b.visual.start_time) || 0;
        const dur0 = parseFloat(b.visual.duration) || 0;
        const hi0  = dur0 > 0 ? lo0 + dur0 : _vidDur;

        _slider = _buildRangeSlider(0, _vidDur, lo0, Math.min(hi0, _vidDur), {
            step,
            onchange: (lo, hi) => {
                b.visual.start_time        = parseFloat(lo.toFixed(2));
                b.visual.duration          = hi >= _vidDur - step ? 0 : parseFloat((hi - lo).toFixed(2));
                b.visual.skip_first_frames = 0;
                startInp.value = b.visual.start_time;
                durInp.value   = b.visual.duration;
                _updateFrameCount();
            },
        });
        sliderWrap.appendChild(_slider.el);
        sliderWrap.style.display = "";

        markRow.appendChild(_mk("button", {
            cls: "fbt-ce-btn", textContent: "◁ Mark Start",
            title: "Set start time to current video position",
            onclick: () => {
                if (!_slider) return;
                const t = Math.min(videoEl.currentTime, parseFloat(_slider.hi.value));
                _slider.setValues(t, parseFloat(_slider.hi.value));
                b.visual.start_time        = parseFloat(t.toFixed(2));
                b.visual.skip_first_frames = 0;
                startInp.value = b.visual.start_time;
                _updateFrameCount();
            },
        }));
        markRow.appendChild(_mk("button", {
            cls: "fbt-ce-btn", textContent: "Mark End ▷",
            title: "Set end time to current video position",
            onclick: () => {
                if (!_slider) return;
                const t   = Math.max(videoEl.currentTime, parseFloat(_slider.lo.value));
                const dur = t >= _vidDur - step ? 0 : parseFloat((t - parseFloat(_slider.lo.value)).toFixed(2));
                _slider.setValues(parseFloat(_slider.lo.value), t);
                b.visual.duration = dur;
                durInp.value = dur;
                _updateFrameCount();
            },
        }));
        markRow.style.display = "";
    };

    const errorEl = _mk("div", { cls: "fbt-be-warn" });
    errorEl.style.display = "none";
    wrap.insertBefore(errorEl, sliderWrap);

    const _loadFile = async filename => {
        // Remove stale listeners from a previous load
        if (_onMeta) { videoEl.removeEventListener("loadedmetadata", _onMeta); _onMeta = null; }
        if (_onErr)  { videoEl.removeEventListener("error",           _onErr);  _onErr  = null; }

        sliderWrap.innerHTML    = "";
        markRow.innerHTML       = "";
        sliderWrap.style.display = "none";
        markRow.style.display    = "none";
        infoEl.style.display     = "none";
        errorEl.style.display    = "none";
        _slider = null; _vidDur = 0; _vidInfo = null;

        // Clear any sampled preview
        if (_previewBlobUrl) { URL.revokeObjectURL(_previewBlobUrl); _previewBlobUrl = null; }
        if (_previewVideoEl) { _previewVideoEl.src = ""; }
        if (_previewWrap)    { _previewWrap.style.display = "none"; }
        frameCountReadout.textContent = "";

        if (!filename) { videoEl.style.display = "none"; videoEl.src = ""; return; }

        videoEl.src = bundlesApi.streamUrl(filename);
        videoEl.style.display = "";

        // loadedmetadata fires when the browser can decode the video header —
        // use it as the primary slider trigger so it works even if mediaInfo is slow.
        _onMeta = () => {
            _onMeta = null;
            if (!_vidDur && isFinite(videoEl.duration) && videoEl.duration > 0) {
                _vidDur = videoEl.duration;
                _buildSlider();
                _updateFrameCount();
            }
        };
        videoEl.addEventListener("loadedmetadata", _onMeta, { once: true });

        _onErr = () => {
            _onErr = null;
            const code  = videoEl.error?.code;
            const ext   = filename.split(".").pop().toLowerCase();
            const codec = _vidInfo?.codec ?? "";
            // Map known codec strings to friendly names
            const codecLabel = { avc1: "H.264", h264: "H.264", "h.264": "H.264",
                hev1: "H.265/HEVC", hevc: "H.265/HEVC", "h.265": "H.265/HEVC",
                hvc1: "H.265/HEVC", av01: "AV1", vp09: "VP9", vp08: "VP8",
                xvid: "XVID", divx: "DIVX", mp4v: "MPEG-4 Part 2",
                theo: "Theora", "": "" }[codec.toLowerCase()] ?? codec;
            const codecStr = codecLabel ? ` (codec: ${codecLabel})` : (codec ? ` (codec: ${codec})` : "");
            let msg = "";
            if (code === 2) {
                msg = "Network error loading video — check the server log.";
            } else if (code === 1) {
                msg = "Video load aborted.";
            } else if (["mkv", "avi"].includes(ext)) {
                msg = `${ext.toUpperCase()} files cannot be played in the browser${codecStr}. Convert to H.264 MP4:\n` +
                    `ffmpeg -i "${filename}" -c:v libx264 -c:a aac -map_metadata 0 -pix_fmt yuv420p output.mp4`;
            } else if (codec && ["hev1","hevc","hvc1","h.265"].includes(codec.toLowerCase())) {
                msg = `H.265/HEVC is not supported in Chrome. Re-encode as H.264:\n` +
                    `ffmpeg -i "${filename}" -c:v libx264 -c:a aac -map_metadata 0 -pix_fmt yuv420p output.mp4`;
            } else if (["mov","mp4"].includes(ext)) {
                msg = `Video could not be played${codecStr}. If encoded with H.265/HEVC or ProRes, re-encode:\n` +
                    `ffmpeg -i "${filename}" -c:v libx264 -c:a aac -map_metadata 0 -pix_fmt yuv420p output.mp4`;
            } else {
                msg = `Video could not be played${codecStr}. Try converting to H.264 MP4:\n` +
                    `ffmpeg -i "${filename}" -c:v libx264 -c:a aac -map_metadata 0 -pix_fmt yuv420p output.mp4`;
            }
            errorEl.textContent = msg;
            errorEl.style.whiteSpace = "pre-wrap";
            errorEl.style.display = "";
            videoEl.style.display = "none";
        };
        videoEl.addEventListener("error", _onErr, { once: true });

        try {
            const info = await bundlesApi.mediaInfo(filename);
            _vidInfo = info;
            if (info.duration > 0) {
                _vidDur = info.duration;
                const m = Math.floor(_vidDur / 60);
                const s = (_vidDur % 60).toFixed(2);
                const durStr = m > 0 ? `${m}m ${s}s` : `${s}s`;
                const codecPart = info.codec && info.codec !== "unknown" ? `  •  ${info.codec}` : "";
                infoEl.textContent = `${durStr}  •  ${info.fps.toFixed(2)} fps  •  ${info.width}×${info.height}  •  ${info.frame_count} frames${codecPart}`;
                infoEl.style.display = "";
                _buildSlider();
                _updateFrameCount();
            }
        } catch (_) { /* loadedmetadata will build the slider if mediaInfo is unavailable */ }
    };

    sel.addEventListener("change", () => { b.visual.file = sel.value; _loadFile(sel.value); });
    if (b.visual.file) _loadFile(b.visual.file);

    _buildFrameParamSection(wrap, b.visual, { onchange: _updateFrameCount });
    wrap.appendChild(frameCountReadout);

    // ── Sampled preview ────────────────────────────────────────────────────────
    _previewWrap = _mk("div", { cls: "fbt-be-preview-wrap", style: { display: "none" } });
    _previewVideoEl = document.createElement("video");
    _previewVideoEl.className = "fbt-be-video-preview fbt-be-preview-video";
    _previewVideoEl.controls = true;
    _previewVideoEl.loop = true;
    _previewWrap.appendChild(_mk("div", { cls: "fbt-be-preview-header" }, [
        _mk("span", { cls: "fbt-be-param-section-label", style: { padding: "0" }, textContent: "Sampled Preview" }),
        _mk("button", {
            cls: "fbt-ce-icon-btn", textContent: "✕", title: "Close preview",
            onclick: () => {
                if (_previewBlobUrl) { URL.revokeObjectURL(_previewBlobUrl); _previewBlobUrl = null; }
                _previewVideoEl.src = "";
                _previewWrap.style.display = "none";
            },
        }),
    ]));
    _previewWrap.appendChild(_previewVideoEl);
    wrap.appendChild(_previewWrap);

    const previewFallback = _mk("div", { cls: "fbt-be-warn", style: { display: "none" } });
    wrap.appendChild(previewFallback);

    wrap.appendChild(_mk("button", {
        cls: "fbt-ce-btn fbt-be-preview-btn",
        textContent: "▶ Preview Sampled",
        title: "Preview the exact frames the model will see, played at the effective sampling rate",
        onclick: async (e) => {
            if (!b.visual.file) { _toast("Select a video file first", "warn"); return; }
            const btn = e.currentTarget;
            btn.disabled = true;
            btn.textContent = "Extracting…";
            previewFallback.style.display = "none";

            // Revoke any existing preview
            if (_previewBlobUrl) { URL.revokeObjectURL(_previewBlobUrl); _previewBlobUrl = null; }
            _previewVideoEl.src = "";
            _previewWrap.style.display = "none";

            try {
                const resp = await fetch("/fbtools/bundles/preview_sampled", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        filename:         b.visual.file,
                        start_time:       b.visual.start_time       || 0,
                        duration:         b.visual.duration         || 0,
                        force_rate:       b.visual.force_rate       || 0,
                        select_every_nth: b.visual.select_every_nth || 1,
                    }),
                });

                if (!resp.ok) {
                    const data = await resp.json().catch(() => ({}));
                    if (data.error === "ffmpeg_unavailable") {
                        previewFallback.textContent =
                            "ffmpeg not found — the main player will loop between your marked points at the native rate. " +
                            "Install ffmpeg to enable sampled frame preview.";
                        previewFallback.style.display = "";
                        const lo = b.visual.start_time || 0;
                        const hi = b.visual.duration > 0 ? lo + b.visual.duration : _vidDur;
                        if (_vidDur > 0 && hi > lo) {
                            videoEl.currentTime = lo;
                            const _loopFn = () => {
                                if (videoEl.currentTime >= hi - 0.15) videoEl.currentTime = lo;
                            };
                            videoEl.addEventListener("timeupdate", _loopFn);
                            videoEl.play().catch(() => {});
                        }
                        return;
                    }
                    _toast("Preview failed: " + (data.error || resp.statusText), "error");
                    return;
                }

                const blob = await resp.blob();
                _previewBlobUrl = URL.createObjectURL(blob);
                _previewVideoEl.src = _previewBlobUrl;
                _previewWrap.style.display = "";
                _previewVideoEl.play().catch(() => {});
            } catch (err) {
                _toast("Preview failed: " + err.message, "error");
            } finally {
                btn.disabled = false;
                btn.textContent = "▶ Preview Sampled";
            }
        },
    }));
}

function _viewUrl(relPath) {
    const slash = relPath.lastIndexOf("/");
    const name  = slash === -1 ? relPath        : relPath.slice(slash + 1);
    const sub   = slash === -1 ? ""             : relPath.slice(0, slash);
    return `/view?filename=${encodeURIComponent(name)}&type=input&subfolder=${encodeURIComponent(sub)}`;
}

function _buildImageList(wrap, b, onFilesChange = null) {
    if (!_S.mediaImages.length) {
        wrap.appendChild(_mk("div", { cls: "fbt-be-media-empty", textContent: "No image files in input directory" }));
        return;
    }

    // ── Preview (shown on hover; hides only when mouse leaves the whole section) ─
    const previewImg = _mk("img", { cls: "fbt-be-img-preview", alt: "" });
    previewImg.style.display = "none";
    let _hideTimer = null;

    const _showPreview = path => {
        clearTimeout(_hideTimer);
        previewImg.src = _viewUrl(path);
        previewImg.style.display = "";
    };
    const _scheduleHide = () => {
        _hideTimer = setTimeout(() => { previewImg.style.display = "none"; }, 300);
    };

    // ── Tree builder ───────────────────────────────────────────────────────────
    const treeEl = _mk("div", { cls: "fbt-be-tree" });

    const _insertPath = (node, parts, fullPath) => {
        if (parts.length === 1) {
            node.files.push({ name: parts[0], path: fullPath });
        } else {
            const dir = parts[0];
            if (!node.dirs.has(dir)) node.dirs.set(dir, { dirs: new Map(), files: [] });
            _insertPath(node.dirs.get(dir), parts.slice(1), fullPath);
        }
    };

    const rootNode = { dirs: new Map(), files: [] };
    _S.mediaImages.forEach(p => _insertPath(rootNode, p.split("/"), p));

    const _syncTree = () => {
        treeEl.querySelectorAll("[data-path]").forEach(el => {
            el.classList.toggle("fbt-be-tree-file-sel", b.visual.files.includes(el.dataset.path));
        });
    };

    const renderNode = (node, depth) => {
        const el = _mk("div", { cls: "fbt-be-tree-node" });
        const indent = depth * 14;

        // Directories (lazy — children rendered on first expand)
        [...node.dirs.entries()].sort(([a], [b]) => a.localeCompare(b)).forEach(([dirName, child]) => {
            const childWrap = _mk("div", { cls: "fbt-be-tree-children" });
            childWrap.style.display = "none";

            const arrow  = _mk("span", { cls: "fbt-be-tree-arrow", textContent: "▶" });
            const toggle = _mk("button", { cls: "fbt-be-tree-toggle" });
            toggle.style.paddingLeft = (indent + 2) + "px";
            toggle.appendChild(arrow);
            toggle.appendChild(document.createTextNode(" " + dirName + "/"));

            toggle.addEventListener("click", () => {
                const opening = childWrap.style.display === "none";
                childWrap.style.display = opening ? "" : "none";
                arrow.textContent = opening ? "▼" : "▶";
                if (opening && !childWrap.firstChild) {
                    childWrap.appendChild(renderNode(child, depth + 1));
                }
            });

            const dirWrap = _mk("div", {}, [toggle, childWrap]);
            el.appendChild(dirWrap);
        });

        // Files
        [...node.files].sort((a, z) => a.name.localeCompare(z.name)).forEach(({ name, path }) => {
            const fileEl = _mk("div", { cls: "fbt-be-tree-file" });
            fileEl.dataset.path = path;
            fileEl.style.paddingLeft = (indent + 16) + "px";
            fileEl.classList.toggle("fbt-be-tree-file-sel", b.visual.files.includes(path));
            fileEl.appendChild(_mk("span", { cls: "fbt-be-tree-file-name", textContent: name, title: path }));

            fileEl.addEventListener("mouseenter", () => _showPreview(path));
            fileEl.addEventListener("click", () => {
                if (!b.visual.files.includes(path)) {
                    b.visual.files.push(path);
                    rebuildList();
                    _showPreview(path);
                    onFilesChange?.();
                }
            });
            el.appendChild(fileEl);
        });

        return el;
    };

    treeEl.appendChild(renderNode(rootNode, 0));

    // ── Selected-files list ────────────────────────────────────────────────────
    const listEl = _mk("div", { cls: "fbt-be-img-list" });

    const rebuildList = () => {
        listEl.innerHTML = "";
        if (!b.visual.files.length) {
            listEl.appendChild(_mk("div", { cls: "fbt-be-media-empty", textContent: "No images selected" }));
        } else {
            b.visual.files.forEach((f, i) => {
                const row = _mk("div", { cls: "fbt-be-img-row" });
                row.addEventListener("mouseenter", () => _showPreview(f));

                const namePart = f.split("/").pop();
                row.appendChild(_mk("span", { cls: "fbt-be-img-name", textContent: namePart, title: f }));

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
                        onFilesChange?.();
                    },
                }));
                row.appendChild(btns);
                listEl.appendChild(row);
            });
        }
        _syncTree();
    };

    rebuildList();

    // ── Hover container — wraps list + tree + preview so mouse can move freely ─
    const hoverWrap = _mk("div", { cls: "fbt-be-img-hover-wrap" });
    hoverWrap.addEventListener("mouseenter", () => clearTimeout(_hideTimer));
    hoverWrap.addEventListener("mouseleave", _scheduleHide);

    const treeLabel = _mk("div", { cls: "fbt-be-param-section-label", textContent: "Browse library" });
    hoverWrap.appendChild(listEl);
    hoverWrap.appendChild(treeLabel);
    hoverWrap.appendChild(treeEl);
    hoverWrap.appendChild(previewImg);
    wrap.appendChild(hoverWrap);
}

function _buildAudioProcessingSection(wrap, b, sourceAudioEl = null) {
    // Ensure schema fields exist on older bundles
    if (!b.audio.audio_processing) {
        b.audio.audio_processing = { noise_removal: false, normalize_lufs: true, target_lufs: -14.0 };
    }
    const proc = b.audio.audio_processing;
    if (proc.target_lufs == null) proc.target_lufs = -14.0;
    if (!("audio_cache" in b.audio)) b.audio.audio_cache = "";

    let _updateStatus = () => {};

    const sec = _mk("div", { cls: "fbt-be-proc-section" });
    sec.appendChild(_mk("div", { cls: "fbt-be-param-section-label", textContent: "Processing" }));

    const _checkRow = (label, key, hint, extras = []) => {
        const row = _mk("div", { cls: "fbt-be-proc-row" });
        const cb  = _mk("input", { type: "checkbox" });
        cb.className = "fbt-be-proc-cb";
        cb.checked   = !!proc[key];
        if (hint) cb.title = hint;
        cb.addEventListener("change", () => {
            proc[key] = cb.checked;
            b.audio.audio_cache = "";
            _updateStatus();
        });
        row.appendChild(cb);
        row.appendChild(_mk("span", { cls: "fbt-be-proc-label", textContent: label }));
        extras.forEach(el => row.appendChild(el));
        return { row, cb };
    };

    const _hasMelband = !!(_S.settings?.melband_model_path?.trim());
    const _noiseLabel = _hasMelband ? "Vocal extraction" : "Noise reduction";
    const _noiseHint  = _hasMelband
        ? "Isolate the vocal stem using MelBand Roformer source separation"
        : "Spectral subtraction — removes background hiss. For better results, set a MelBand Roformer model path in Settings.";
    const { row: noiseRow } = _checkRow(_noiseLabel, "noise_removal", _noiseHint);
    sec.appendChild(noiseRow);

    // LUFS normalize row with inline target input
    const lufsInp = _mk("input", {
        cls: "fbt-be-param-input", type: "number",
        min: "-36", max: "-6", step: "0.5",
        value: proc.target_lufs ?? -14.0,
        title: "Target integrated loudness — streaming standard is −14 LUFS",
    });
    const lufsUnit = _mk("span", { cls: "fbt-be-proc-unit", textContent: "LUFS" });
    lufsInp.addEventListener("input", () => {
        proc.target_lufs = parseFloat(lufsInp.value) || -14.0;
        b.audio.audio_cache = "";
        _updateStatus();
    });
    const { row: lufsRow, cb: lfsCb } = _checkRow("LUFS normalize", "normalize_lufs",
        "Normalize to target integrated loudness (ITU BS.1770 / EBU R128)", [lufsInp, lufsUnit]);
    const _syncLufsVisibility = () => {
        lufsInp.style.display  = lfsCb.checked ? "" : "none";
        lufsUnit.style.display = lfsCb.checked ? "" : "none";
    };
    lfsCb.addEventListener("change", _syncLufsVisibility);
    _syncLufsVisibility();
    sec.appendChild(lufsRow);

    // Cache status + preview player
    const statusEl  = _mk("div", { cls: "fbt-be-proc-status" });
    const previewEl = document.createElement("audio");
    previewEl.className = "fbt-be-audio-preview fbt-be-proc-preview";
    previewEl.controls  = true;
    previewEl.preload   = "none";
    previewEl.style.display = "none";

    _updateStatus = (overrideText) => {
        const cache = b.audio.audio_cache;
        if (overrideText !== undefined) {
            statusEl.textContent = overrideText;
            statusEl.className   = "fbt-be-proc-status fbt-be-proc-ok";
        } else if (cache) {
            const base = cache.split(/[/\\]/).pop() || cache;
            statusEl.textContent = `✓ Cached · ${base}`;
            statusEl.className   = "fbt-be-proc-status fbt-be-proc-ok";
        } else {
            statusEl.textContent = "Not processed";
            statusEl.className   = "fbt-be-proc-status";
        }
        if (cache) {
            const url = `/fbtools/bundles/audio_cache/stream?path=${encodeURIComponent(cache)}`;
            if (previewEl.dataset.cachePath !== cache) {
                previewEl.src = url;
                previewEl.load();
                previewEl.dataset.cachePath = cache;
            }
            previewEl.style.display = "";
            // Hide the original source player — processed audio is now the one to hear
            if (sourceAudioEl) sourceAudioEl.style.display = "none";
        } else {
            previewEl.src = "";
            previewEl.style.display = "none";
            delete previewEl.dataset.cachePath;
            // Restore the original source player when cache is cleared
            if (sourceAudioEl) {
                sourceAudioEl.style.display = sourceAudioEl.src ? "" : "none";
            }
        }
    };
    _updateStatus();
    sec.appendChild(statusEl);
    sec.appendChild(previewEl);

    sec.appendChild(_mk("button", {
        cls: "fbt-ce-btn fbt-be-proc-btn",
        textContent: "⚙ Process Audio",
        title: "Run denoising / LUFS normalization and cache the result on disk",
        onclick: async (e) => {
            const btn = e.currentTarget;
            const src = b.audio.source;
            let filename = "";
            if (src === "file")                filename = b.audio.file;
            else if (src === "extract_from_video") filename = b.audio.video_file;
            else if (src === "extract_from_visual") filename = b.visual.file;
            if (!filename) { _toast("No audio source selected", "warn"); return; }

            btn.disabled     = true;
            btn.textContent  = "Processing…";
            statusEl.textContent = "Processing…";
            statusEl.className   = "fbt-be-proc-status";

            try {
                const result = await bundlesApi.preprocessAudio({
                    bundle_id:   b.id || "default",
                    filename,
                    start_time:  b.audio.start_time || 0,
                    duration:    b.audio.duration   || 0,
                    audio_processing: {
                        noise_removal:  !!proc.noise_removal,
                        normalize_lufs: proc.normalize_lufs !== false,
                        target_lufs:    proc.target_lufs ?? -14.0,
                    },
                });
                b.audio.audio_cache = result.cache_path;
                const durStr  = result.duration  != null ? `${result.duration}s`    : "";
                const lufsStr = result.lufs_after != null ? ` · ${result.lufs_after} LUFS` : "";
                const label   = result.from_cache ? "Cached" : "Processed";
                _updateStatus(`✓ ${label} · ${durStr}${lufsStr}`);
                _toast("Audio processed — click Save to persist the cache reference", "success");
            } catch (err) {
                statusEl.textContent = "Failed: " + err.message;
                statusEl.className   = "fbt-be-proc-status fbt-be-proc-err";
                previewEl.style.display = "none";
                _toast("Processing failed: " + err.message, "error");
            } finally {
                btn.disabled    = false;
                btn.textContent = "⚙ Process Audio";
            }
        },
    }));

    wrap.appendChild(sec);
}

function _buildAudioPlayer(filename) {
    const el = document.createElement("audio");
    el.className = "fbt-be-audio-preview";
    el.controls = true;
    el.preload = "none";
    if (filename) el.src = bundlesApi.streamUrl(filename);
    else el.style.display = "none";
    return el;
}

function _buildAudioVideoPicker(wrap, b) {
    if (!_S.mediaVideos.length) {
        wrap.appendChild(_mk("div", { cls: "fbt-be-media-empty", textContent: "No video files in input directory" }));
        return null;
    }
    const sel = document.createElement("select");
    sel.className = "fbt-ce-select";
    const blank = document.createElement("option");
    blank.value = ""; blank.textContent = "— select video file —";
    if (!b.audio.video_file) blank.selected = true;
    sel.appendChild(blank);
    _S.mediaVideos.forEach(f => {
        const o = document.createElement("option");
        o.value = f; o.textContent = f;
        if (f === b.audio.video_file) o.selected = true;
        sel.appendChild(o);
    });

    const audioEl = _buildAudioPlayer(b.audio.video_file);
    audioEl.title = "Original source audio";
    sel.addEventListener("change", () => {
        b.audio.video_file = sel.value;
        if (sel.value) { audioEl.src = bundlesApi.streamUrl(sel.value); audioEl.style.display = ""; }
        else { audioEl.src = ""; audioEl.style.display = "none"; }
    });

    wrap.appendChild(sel);
    wrap.appendChild(audioEl);
    return audioEl;
}

function _buildAudioPicker(wrap, b) {
    let audioEl = null;
    if (!_S.mediaAudio.length) {
        wrap.appendChild(_mk("div", { cls: "fbt-be-media-empty", textContent: "No audio files in input directory" }));
    } else {
        const sel = document.createElement("select");
        sel.className = "fbt-ce-select";
        const blank = document.createElement("option");
        blank.value = ""; blank.textContent = "— select audio file —";
        if (!b.audio.file) blank.selected = true;
        sel.appendChild(blank);
        _S.mediaAudio.forEach(f => {
            const o = document.createElement("option");
            o.value = f; o.textContent = f;
            if (f === b.audio.file) o.selected = true;
            sel.appendChild(o);
        });

        audioEl = _buildAudioPlayer(b.audio.file);
        audioEl.title = "Original source audio";
        sel.addEventListener("change", () => {
            b.audio.file = sel.value;
            if (sel.value) { audioEl.src = bundlesApi.streamUrl(sel.value); audioEl.style.display = ""; }
            else { audioEl.src = ""; audioEl.style.display = "none"; }
        });

        wrap.appendChild(sel);
        wrap.appendChild(audioEl);
    }
    _buildAudioTimeSection(wrap, b.audio);
    return audioEl;
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
        // Image mode: dropdown rebuilt dynamically from bundle's file list (falls back to full media pool)
        const imgSel = document.createElement("select");
        imgSel.className = "fbt-ce-select fbt-be-llm-img-sel";

        const _setPreview = f => {
            previewImg.src = f ? _viewUrl(f) : "";
            previewImg.style.display = f ? "" : "none";
        };

        const rebuildPool = () => {
            const pool = b.visual.files?.length ? b.visual.files : _S.mediaImages;
            const prev = imgSel.value;
            imgSel.innerHTML = "";
            const blank = document.createElement("option");
            blank.value = "";
            blank.textContent = pool.length ? "— select image —" : "No images available";
            imgSel.appendChild(blank);
            pool.forEach(f => {
                const o = document.createElement("option");
                o.value = f; o.textContent = f;
                imgSel.appendChild(o);
            });
            imgSel.disabled = !pool.length;
            // Restore previous selection or auto-select when there's exactly one option
            if (pool.includes(prev)) imgSel.value = prev;
            else if (pool.length === 1) imgSel.value = pool[0];
            _setPreview(imgSel.value);
        };

        imgSel.addEventListener("change", () => _setPreview(imgSel.value));
        getSourceImage = () => imgSel.value || null;

        rebuildPool();
        sec.appendChild(_mk("div", { cls: "fbt-be-llm-top-row" }, [imgSel]));
        // Expose for external refresh (called when bundle images change)
        sec._refreshPool = rebuildPool;
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
        _S.lastId  = id;
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

// ── Subject management ────────────────────────────────────────────────────────

function _startNewSubject() {
    _S.subjectEditing = {
        id: "", name: "",
        appearance: { summary: "", face: "", hair: "", body: "", default_outfit: "" },
        voice: { description: "", language: "English", audio_reference_file: "" },
        character_sheet_images: [],
        concept_id: "",
    };
    _S.subjectIsNew = true;
    _renderSubjectForm();
}

async function _startEditSubject(id) {
    try {
        const data = await bundlesApi.getSubject(id);
        _S.subjectEditing = data;
        _S.subjectEditing.id = id;
        if (!_S.subjectEditing.appearance) _S.subjectEditing.appearance = {};
        if (!_S.subjectEditing.voice)      _S.subjectEditing.voice = {};
        if (!_S.subjectEditing.character_sheet_images) _S.subjectEditing.character_sheet_images = [];
        _S.subjectIsNew = false;
        _renderSubjectForm();
    } catch (e) {
        _toast("Failed to load subject: " + e.message, "error");
    }
}

function _cancelSubjectEdit() {
    _S.subjectEditing = null;
    _S.subjectIsNew = false;
    _renderSubjectList();
}

async function _onDeleteSubject(id, name) {
    if (!confirm(`Delete subject "${name || id}"? This cannot be undone.`)) return;
    try {
        await bundlesApi.deleteSubject(id);
        _S.subjects = _S.subjects.filter(s => s.id !== id);
        _repopulateSubjectFilter();
        _S.subjectEditing = null;
        _S.subjectIsNew = false;
        _renderSubjectList();
        _toast(`Deleted "${name || id}"`, "success");
    } catch (e) {
        _toast("Delete failed: " + e.message, "error");
    }
}

async function _onSaveSubject(s, warnEl) {
    const name = (s.name || "").trim();
    const id   = (s.id   || "").trim();
    if (!name) { _toast("Subject name is required", "warn"); return; }
    if (!id)   { _toast("Subject ID is required", "warn"); return; }

    warnEl.style.display = "none";
    try {
        await bundlesApi.saveSubject({ ...s, id, name });
        const res = await bundlesApi.listSubjects();
        _S.subjects = res.subjects ?? [];
        _repopulateSubjectFilter();
        _S.subjectEditing = null;
        _S.subjectIsNew = false;
        _renderSubjectList();
        _toast(`Saved "${name}"`, "success");
    } catch (e) {
        warnEl.textContent = e.message;
        warnEl.style.display = "";
        _toast("Save failed", "error");
    }
}

function _renderSubjectList() {
    const c = _dom.content;
    if (!c) return;
    c.innerHTML = "";

    let items = _S.subjects;
    if (_S.subjectFilter) {
        const q = _S.subjectFilter.toLowerCase();
        items = items.filter(s =>
            (s.name || "").toLowerCase().includes(q) ||
            (s.id   || "").toLowerCase().includes(q)
        );
    }

    if (!items.length) {
        c.appendChild(_mk("div", { cls: "fbt-be-empty",
            textContent: _S.subjects.length
                ? "No subjects match the search."
                : "No subjects defined. Click + New Subject to create one." }));
        return;
    }

    items.forEach(s => {
        const card = _mk("div", { cls: "fbt-be-card fbt-be-card-clickable" });
        const top  = _mk("div", { cls: "fbt-be-card-top" });
        top.appendChild(_mk("span", { cls: "fbt-be-card-name", textContent: s.name || s.id }));
        if (s.concept_id) {
            top.appendChild(_mk("span", { cls: "fbt-be-card-meta", textContent: s.concept_id, title: "Concept ID" }));
        }
        card.appendChild(top);

        if (s.appearance_summary) {
            card.appendChild(_mk("div", { cls: "fbt-be-card-summary", textContent: s.appearance_summary }));
        }

        const actions = _mk("div", { cls: "fbt-be-card-actions" });
        actions.appendChild(_mk("button", {
            cls: "fbt-ce-icon-btn", title: "Edit", textContent: "✎",
            onclick: e => { e.stopPropagation(); _startEditSubject(s.id); },
        }));
        actions.appendChild(_mk("button", {
            cls: "fbt-ce-icon-btn fbt-ce-danger", title: "Delete", textContent: "✕",
            onclick: e => { e.stopPropagation(); _onDeleteSubject(s.id, s.name); },
        }));
        card.appendChild(actions);
        card.addEventListener("click", () => _startEditSubject(s.id));
        c.appendChild(card);
    });
}

function _renderSubjectForm() {
    const c = _dom.content;
    if (!c) return;
    c.innerHTML = "";

    const s = _S.subjectEditing;
    let idManuallyEdited = !_S.subjectIsNew;

    const form = _mk("div", { cls: "fbt-be-form" });

    // ── Basic fields ───────────────────────────────────────────────────────────
    const nameEl = _mk("input", { cls: "fbt-ce-input", type: "text",
        placeholder: "Subject name *", value: s.name || "" });
    const idEl   = _mk("input", { cls: "fbt-ce-input", type: "text",
        placeholder: "subject_id (auto-generated)", value: s.id || "" });

    nameEl.addEventListener("input", () => {
        s.name = nameEl.value;
        if (!idManuallyEdited) { idEl.value = _slugify(s.name); s.id = idEl.value; }
    });
    idEl.addEventListener("input", () => { s.id = idEl.value.trim(); idManuallyEdited = true; });

    const conceptEl = _mk("input", { cls: "fbt-ce-input", type: "text",
        placeholder: "Concept ID (optional)", value: s.concept_id || "" });
    conceptEl.addEventListener("input", () => { s.concept_id = conceptEl.value.trim(); });

    form.appendChild(_formRow("Name",    nameEl));
    form.appendChild(_formRow("ID",      idEl));
    form.appendChild(_formRow("Concept", conceptEl));

    // ── Appearance section ─────────────────────────────────────────────────────
    const appearSec = _mk("div", { cls: "fbt-be-section" });
    appearSec.appendChild(_mk("div", { cls: "fbt-be-sec-label", textContent: "Appearance" }));

    if (!s.appearance) s.appearance = {};
    const summaryEl = _mk("textarea", { cls: "fbt-ce-textarea", rows: 2,
        placeholder: "Appearance summary used in prompts…",
        value: s.appearance.summary || "" });
    summaryEl.addEventListener("input", () => { s.appearance.summary = summaryEl.value; });
    appearSec.appendChild(_formRow("Summary", summaryEl));

    const detailsEl = document.createElement("details");
    detailsEl.className = "fbt-be-details";
    const detailsSummary = document.createElement("summary");
    detailsSummary.className = "fbt-be-details-summary";
    detailsSummary.textContent = "Details (face, hair, body, outfit)";
    detailsEl.appendChild(detailsSummary);

    const subGrid = _mk("div", { cls: "fbt-be-subfield-grid" });
    [["face", "Face", "Facial features, eyes, nose, expression…"],
     ["hair", "Hair", "Hair color, style, length…"],
     ["body", "Body", "Body type, height, build…"],
     ["default_outfit", "Outfit", "Default outfit description…"]].forEach(([key, label, ph]) => {
        const inp = _mk("input", { cls: "fbt-ce-input", type: "text",
            placeholder: ph, value: s.appearance[key] || "" });
        inp.addEventListener("input", () => { s.appearance[key] = inp.value; });
        subGrid.appendChild(_mk("div", { cls: "fbt-be-subfield-cell" }, [
            _mk("span", { cls: "fbt-be-subfield-label", textContent: label }),
            inp,
        ]));
    });
    detailsEl.appendChild(subGrid);
    appearSec.appendChild(detailsEl);
    form.appendChild(appearSec);

    // ── Character Sheet Images section ─────────────────────────────────────────
    const sheetSec = _mk("div", { cls: "fbt-be-section" });
    sheetSec.appendChild(_mk("div", { cls: "fbt-be-sec-label", textContent: "Character Sheet Images" }));

    if (!s.character_sheet_images) s.character_sheet_images = [];
    const sheetListEl = _mk("div", { cls: "fbt-be-sheet-list" });

    const _renderSheetList = () => {
        sheetListEl.innerHTML = "";
        if (!s.character_sheet_images.length) {
            sheetListEl.appendChild(_mk("div", { cls: "fbt-be-empty",
                textContent: "No images added yet." }));
            return;
        }
        s.character_sheet_images.forEach((img, i) => {
            const row     = _mk("div", { cls: "fbt-be-sheet-row" });
            const nameEl2 = _mk("span", { cls: "fbt-be-sheet-name",
                textContent: img.file, title: img.file });
            const roleSel = document.createElement("select");
            roleSel.className = "fbt-ce-select fbt-be-sheet-role-sel";
            SHEET_ROLES.forEach(r => {
                const o = document.createElement("option");
                o.value = r; o.textContent = r;
                if (r === (img.role || "character sheet")) o.selected = true;
                roleSel.appendChild(o);
            });
            roleSel.onchange = () => { s.character_sheet_images[i].role = roleSel.value; };
            const delBtn = _mk("button", { cls: "fbt-ce-icon-btn fbt-ce-danger", textContent: "✕",
                onclick: () => { s.character_sheet_images.splice(i, 1); _renderSheetList(); } });
            row.append(nameEl2, roleSel, delBtn);
            sheetListEl.appendChild(row);
        });
    };
    _renderSheetList();

    // Add image row
    const sheetUid   = Math.random().toString(36).slice(2, 8);
    const addInput   = _mk("input", { cls: "fbt-ce-input fbt-be-sheet-add-input", type: "text",
        placeholder: "Image filename…", list: `fbt-sht-dl-${sheetUid}` });
    const addDl      = _mk("datalist", { id: `fbt-sht-dl-${sheetUid}` });
    _S.mediaImages.forEach(f => { const o = document.createElement("option"); o.value = f; addDl.appendChild(o); });

    const addRoleSel = document.createElement("select");
    addRoleSel.className = "fbt-ce-select fbt-be-sheet-role-sel";
    SHEET_ROLES.forEach(r => {
        const o = document.createElement("option"); o.value = r; o.textContent = r; addRoleSel.appendChild(o);
    });
    const addBtn = _mk("button", { cls: "fbt-ce-btn fbt-ce-btn-sm", textContent: "+ Add",
        onclick: () => {
            const fname = addInput.value.trim();
            if (!fname) return;
            s.character_sheet_images.push({ file: fname, role: addRoleSel.value });
            addInput.value = "";
            _renderSheetList();
        },
    });
    sheetSec.appendChild(sheetListEl);
    sheetSec.appendChild(_mk("div", { cls: "fbt-be-sheet-add-row" }, [addInput, addDl, addRoleSel, addBtn]));
    form.appendChild(sheetSec);

    // ── Voice section ──────────────────────────────────────────────────────────
    const voiceSec = _mk("div", { cls: "fbt-be-section" });
    voiceSec.appendChild(_mk("div", { cls: "fbt-be-sec-label", textContent: "Voice" }));

    if (!s.voice) s.voice = {};
    const voiceDescEl = _mk("textarea", { cls: "fbt-ce-textarea", rows: 2,
        placeholder: "Tone, accent, style…", value: s.voice.description || "" });
    voiceDescEl.addEventListener("input", () => { s.voice.description = voiceDescEl.value; });

    const voiceLangEl = _mk("input", { cls: "fbt-ce-input", type: "text",
        placeholder: "Language (e.g. English)", value: s.voice.language || "" });
    voiceLangEl.addEventListener("input", () => { s.voice.language = voiceLangEl.value.trim(); });

    const voiceAudioSel = document.createElement("select");
    voiceAudioSel.className = "fbt-ce-select";
    [{ v: "", l: "— no audio reference —" }, ..._S.mediaAudio.map(f => ({ v: f, l: f }))].forEach(({ v, l }) => {
        const o = document.createElement("option");
        o.value = v; o.textContent = l;
        if (v === (s.voice.audio_reference_file || "")) o.selected = true;
        voiceAudioSel.appendChild(o);
    });
    voiceAudioSel.addEventListener("change", () => { s.voice.audio_reference_file = voiceAudioSel.value; });

    voiceSec.appendChild(_formRow("Description", voiceDescEl));
    voiceSec.appendChild(_formRow("Language",    voiceLangEl));
    voiceSec.appendChild(_formRow("Audio ref.",  voiceAudioSel));
    form.appendChild(voiceSec);

    // ── LLM Appearance Analysis (optional) ─────────────────────────────────────
    if (_S.llmVision) {
        const llmSec = _mk("div", { cls: "fbt-be-section" });
        llmSec.appendChild(_mk("div", { cls: "fbt-be-sec-label", textContent: "LLM Appearance Analysis" }));

        const llmUid   = Math.random().toString(36).slice(2, 8);
        const imgInput = _mk("input", { cls: "fbt-ce-input", type: "text",
            placeholder: "Image filename to analyze…", list: `fbt-sht-llm-${llmUid}` });
        const imgDl    = _mk("datalist", { id: `fbt-sht-llm-${llmUid}` });
        _S.mediaImages.forEach(f => { const o = document.createElement("option"); o.value = f; imgDl.appendChild(o); });

        const queryEl  = _mk("textarea", { cls: "fbt-ce-textarea", rows: 2,
            value: "Describe this character's appearance in detail: face, hair, body type, and clothing." });
        const resultEl = _mk("textarea", { cls: "fbt-ce-textarea", rows: 2,
            placeholder: "Analysis result…", style: { display: "none" } });
        const applyBtn = _mk("button", { cls: "fbt-ce-btn fbt-ce-btn-sm",
            textContent: "→ Appearance Summary", style: { display: "none" },
            onclick: () => {
                summaryEl.value = resultEl.value.trim();
                s.appearance.summary = summaryEl.value;
            },
        });
        const analyzeBtn = _mk("button", { cls: "fbt-ce-btn", textContent: "🔍 Analyze",
            onclick: async () => {
                const fname = imgInput.value.trim();
                if (!fname) { _toast("Enter an image filename first", "warn"); return; }
                analyzeBtn.disabled = true;
                analyzeBtn.textContent = "Analyzing…";
                try {
                    const r = await llmApi.generate(queryEl.value.trim(), { images: [fname], max_tokens: 400 });
                    if (r?.text) {
                        resultEl.value = r.text.trim();
                        resultEl.style.display = "";
                        applyBtn.style.display = "";
                    }
                } catch (e) { _toast("LLM error: " + e.message, "error"); }
                finally {
                    analyzeBtn.disabled = false;
                    analyzeBtn.textContent = "🔍 Analyze";
                }
            },
        });
        llmSec.append(imgInput, imgDl, queryEl, analyzeBtn, resultEl, applyBtn);
        form.appendChild(llmSec);
    }

    // ── Buttons ────────────────────────────────────────────────────────────────
    const warnEl = _mk("div", { cls: "fbt-be-warn", style: { display: "none" } });
    const btnRow = _mk("div", { cls: "fbt-be-btn-row" });
    btnRow.appendChild(_mk("button", { cls: "fbt-ce-btn fbt-ce-btn-primary", textContent: "Save",
        onclick: () => _onSaveSubject(s, warnEl) }));
    btnRow.appendChild(_mk("button", { cls: "fbt-ce-btn", textContent: "Cancel",
        onclick: () => _cancelSubjectEdit() }));
    if (!_S.subjectIsNew) {
        btnRow.appendChild(_mk("button", { cls: "fbt-ce-btn fbt-ce-danger", textContent: "Delete",
            onclick: () => _onDeleteSubject(s.id, s.name) }));
    }
    form.appendChild(warnEl);
    form.appendChild(btnRow);
    c.appendChild(form);
    nameEl.focus();
}

// ── Top bar ───────────────────────────────────────────────────────────────────

function _switchView(mode) {
    _S.viewMode = mode;
    if (_dom.tabBundles)  _dom.tabBundles.classList.toggle("active",  mode === "bundles");
    if (_dom.tabSubjects) _dom.tabSubjects.classList.toggle("active", mode === "subjects");
    if (_dom.subjSel) _dom.subjSel.style.display = mode === "bundles" ? "" : "none";
    if (_dom.newBtn)  _dom.newBtn.textContent = mode === "bundles" ? "+ New Bundle" : "+ New Subject";
    if (_dom.searchEl) {
        _dom.searchEl.value       = mode === "bundles" ? _S.filterText : _S.subjectFilter;
        _dom.searchEl.placeholder = mode === "bundles" ? "Search bundles…" : "Search subjects…";
    }
    _render();
}

function _render() {
    if (_S.viewMode === "subjects") {
        if (_S.subjectEditing !== null) _renderSubjectForm();
        else                            _renderSubjectList();
    } else {
        if (_S.editing !== null) _renderForm();
        else                     _renderList();
    }
}

function _buildTopBar() {
    const bar = _mk("div", { cls: "fbt-be-top-bar" });

    // Tab switcher
    const tabBundles  = _mk("button", { cls: "fbt-be-tab active", textContent: "Bundles",
        onclick: () => _switchView("bundles") });
    const tabSubjects = _mk("button", { cls: "fbt-be-tab", textContent: "Subjects",
        onclick: () => _switchView("subjects") });
    _dom.tabBundles  = tabBundles;
    _dom.tabSubjects = tabSubjects;
    bar.appendChild(_mk("div", { cls: "fbt-be-tab-row" }, [tabBundles, tabSubjects]));

    // Subject filter (bundles view only)
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
        _S.listPage = 0;
        if (!_S.editing) _renderList();
    });
    _dom.subjSel = subjSel;

    // Search (serves both modes)
    const searchEl = _mk("input", {
        cls: "fbt-ce-input fbt-be-search",
        type: "text", placeholder: "Search bundles…",
        value: _S.filterText,
    });
    searchEl.addEventListener("input", () => {
        if (_S.viewMode === "subjects") {
            _S.subjectFilter = searchEl.value;
        } else {
            _S.filterText = searchEl.value;
            _S.listPage = 0;
        }
        if (!_S.editing && !_S.subjectEditing) _render();
    });
    _dom.searchEl = searchEl;

    // New button (mode-aware)
    const newBtn = _mk("button", {
        cls: "fbt-ce-btn fbt-ce-btn-primary fbt-be-new-btn",
        textContent: "+ New Bundle",
        onclick: () => {
            if (_S.viewMode === "subjects") _startNewSubject();
            else _startNew(_S.filterSubject);
        },
    });
    _dom.newBtn = newBtn;

    // Refresh
    const refreshBtn = _mk("button", {
        cls: "fbt-ce-icon-btn fbt-be-refresh-btn",
        textContent: "↺", title: "Refresh",
        onclick: async () => {
            try {
                await _loadAll();
                _repopulateSubjectFilter();
                if (!_S.editing && !_S.subjectEditing) _render();
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
    _dom.content    = _mk("div", { cls: "fbt-be-content" });
    _dom.pagination = _mk("div", { cls: "fbt-ce-saved-pagination" });

    // Show loading state while fetching
    _dom.content.appendChild(_mk("div", { cls: "fbt-be-empty", textContent: "Loading…" }));

    panel.appendChild(_buildTopBar());
    panel.appendChild(_dom.content);
    panel.appendChild(_dom.pagination);
    el.appendChild(panel);

    try {
        await _loadAll();
    } catch (e) {
        console.error("fbt BundleEditor: load error", e);
    }

    _repopulateSubjectFilter();
    _render();
}
