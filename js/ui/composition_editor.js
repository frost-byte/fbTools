/**
 * Prompt Composition Editor — Phase 2
 *
 * Structured editor for composing video-generation prompts.
 * Registered as a ComfyUI sidebar tab via app.extensionManager.registerSidebarTab.
 *
 * Stores compositions in the Prompt Composition JSON schema:
 *   {id, name, model_type, style, subjects, outfit_overrides, background,
 *    shots, overall_soundscape, non_diegetic_music}
 *
 * Phase 3 additions: {S} slot-reference completion in action/camera fields,
 *   subject slot appearance display, background soundscape auto-fill,
 *   inline New Subject / New Background creation forms in the sidebar.
 */

import { compositionsApi } from "../api/compositions.js";

// ── Constants ──────────────────────────────────────────────────────────────────

const MODEL_TYPES = [
    { id: "h3_ref2va", label: "MiniMax H3 Ref2VA" },
    { id: "h3_fl2va",  label: "MiniMax H3 FL2VA" },
    { id: "wan22",     label: "Wan 2.2" },
    { id: "bernini",   label: "BerniniR" },
    { id: "ltx23",     label: "LTX 2.3" },
    { id: "flux2",     label: "Flux 2" },
    { id: "krea2",     label: "Krea 2" },
    { id: "qwen",      label: "Qwen Image" },
];

const LANGUAGES = ["English", "Japanese", "Chinese", "Korean", "Spanish", "French", "German", "Other"];

// ── Module state ───────────────────────────────────────────────────────────────

const _S = {
    composition:    null,
    subjects:       [],
    backgrounds:    [],
    cameraPresets:  [],
    soundPresets:   [],
    savedComps:     [],
    dirty:          false,
    shotSeq:        0,
};

// Key DOM refs rebuilt on each panel render
const _dom = {};

// Slot-reference completion popup element (singleton)
let _completionEl = null;

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

function _sel(options, val) {
    const el = document.createElement("select");
    options.forEach(({ id, label }) => {
        const o = document.createElement("option");
        o.value = id;
        o.textContent = label;
        if (id === val) o.selected = true;
        el.appendChild(o);
    });
    return el;
}

function _subjectOptions(includeNone = true) {
    const opts = includeNone ? [{ id: "", label: "— none —" }] : [];
    _S.subjects.forEach(s => opts.push({ id: s.id, label: s.name || s.id }));
    return opts;
}

function _bgOptions() {
    return [
        { id: "", label: "— none —" },
        ..._S.backgrounds.map(b => ({ id: b.id, label: b.name || b.id })),
    ];
}

function _newComp() {
    return {
        id: "", name: "New Composition",
        model_type: "h3_ref2va", style: "",
        subjects: {}, outfit_overrides: {},
        background: "", shots: [],
        overall_soundscape: "", non_diegetic_music: "",
    };
}

function _newShot() {
    return {
        id: `shot_${++_S.shotSeq}`,
        timestamp: null, camera: "", action: "",
        dialogue: null, sound_events: null,
    };
}

function _slotKeys() {
    return Object.keys(_S.composition?.subjects || {}).sort();
}

function _nextSlotKey() {
    const keys = _slotKeys();
    for (let i = 1; i <= 9; i++) {
        const k = `S${i}`;
        if (!keys.includes(k)) return k;
    }
    return null;
}

function _markDirty() {
    _S.dirty = true;
    if (_dom.dirtyDot) _dom.dirtyDot.style.display = "inline";
}

function _markClean() {
    _S.dirty = false;
    if (_dom.dirtyDot) _dom.dirtyDot.style.display = "none";
}

function _toast(msg, severity = "info") {
    try {
        const app = window._fbtApp;
        if (app?.extensionManager?.toast) {
            app.extensionManager.toast.add({ severity, summary: msg, life: 2500 });
        }
    } catch (_) {}
}

function _sectionToggle(headerEl, bodyEl) {
    let open = true;
    const icon = _mk("span", { cls: "fbt-ce-chevron", textContent: "▾" });
    headerEl.prepend(icon);
    headerEl.style.cursor = "pointer";
    headerEl.addEventListener("click", () => {
        open = !open;
        bodyEl.style.display = open ? "" : "none";
        icon.textContent = open ? "▾" : "▸";
    });
}

// ── Slot-reference completion ({S1}, {S2} …) ──────────────────────────────────

function _dismissCompletion() {
    if (_completionEl) { _completionEl.remove(); _completionEl = null; }
}

function _showCompletion(textEl, matches, bracePos) {
    _dismissCompletion();
    if (!matches.length) return;

    const popup = _mk("div", { cls: "fbt-ce-completion" });

    matches.forEach((key, i) => {
        const subId  = _S.composition?.subjects?.[key] || "";
        const subName = _S.subjects.find(s => s.id === subId)?.name || "";
        const item = document.createElement("div");
        item.className = "fbt-ce-comp-item" + (i === 0 ? " active" : "");
        item.innerHTML = `<strong>${key}</strong>${subName ? ` <span class="fbt-ce-comp-name">${subName}</span>` : ""}`;
        item.addEventListener("mousedown", e => {
            e.preventDefault(); // keep focus on textEl
            const curPos = textEl.selectionStart;
            const before = textEl.value.substring(0, bracePos);
            const after  = textEl.value.substring(curPos);
            const insert = `{${key}}`;
            textEl.value = before + insert + after;
            const newPos = bracePos + insert.length;
            textEl.selectionStart = textEl.selectionEnd = newPos;
            textEl.dispatchEvent(new Event("input", { bubbles: true }));
            _dismissCompletion();
            textEl.focus();
        });
        popup.appendChild(item);
    });

    const rect = textEl.getBoundingClientRect();
    Object.assign(popup.style, {
        position:  "fixed",
        left:      rect.left + "px",
        top:       Math.min(rect.bottom + 2, window.innerHeight - 140) + "px",
        minWidth:  Math.min(200, rect.width) + "px",
        zIndex:    "9999",
    });
    document.body.appendChild(popup);
    _completionEl = popup;

    // Dismiss on blur (mousedown handler above prevents blur on item click)
    const onBlur = () => _dismissCompletion();
    textEl.addEventListener("blur", onBlur, { once: true });
    // Dismiss on outside click (defer one tick so this event doesn't self-dismiss)
    setTimeout(() => {
        document.addEventListener("mousedown", function outsideClick(e) {
            if (!_completionEl?.contains(e.target)) { _dismissCompletion(); }
            document.removeEventListener("mousedown", outsideClick);
        });
    }, 0);
}

/** Attaches slot-reference completion to a text input or textarea. */
function _attachCompletion(el) {
    el.addEventListener("input", () => {
        const pos    = el.selectionStart;
        const before = el.value.substring(0, pos);
        const last   = before.lastIndexOf("{");
        if (last === -1) { _dismissCompletion(); return; }
        const fragment = before.substring(last + 1);
        // Only complete short, slot-key-shaped fragments: "", "S", "S1" …
        if (fragment.length > 3 || !/^S?\d*$/.test(fragment)) { _dismissCompletion(); return; }
        const matches = _slotKeys().filter(k => k.toLowerCase().startsWith(fragment.toLowerCase()));
        if (!matches.length) { _dismissCompletion(); return; }
        _showCompletion(el, matches, last);
    });

    el.addEventListener("keydown", e => {
        if (!_completionEl) return;
        if (e.key === "Escape") { e.stopPropagation(); _dismissCompletion(); return; }
        if (e.key === "ArrowDown" || e.key === "ArrowUp") {
            e.preventDefault();
            const items = Array.from(_completionEl.querySelectorAll(".fbt-ce-comp-item"));
            let idx = items.findIndex(i => i.classList.contains("active"));
            items[idx]?.classList.remove("active");
            idx = e.key === "ArrowDown" ? (idx + 1) % items.length : (idx - 1 + items.length) % items.length;
            items[idx]?.classList.add("active");
        }
        if (e.key === "Enter" || e.key === "Tab") {
            const active = _completionEl?.querySelector(".fbt-ce-comp-item.active");
            if (active) { e.preventDefault(); active.dispatchEvent(new MouseEvent("mousedown")); }
        }
    });
}

// ── API load ───────────────────────────────────────────────────────────────────

async function _loadResources() {
    try {
        const [subj, bg, cam, snd, comps] = await Promise.allSettled([
            compositionsApi.listSubjects(),
            compositionsApi.listBackgrounds(),
            compositionsApi.listCameraPresets(),
            compositionsApi.listSoundPresets(),
            compositionsApi.listCompositions(),
        ]);
        _S.subjects      = subj.value?.subjects      ?? [];
        _S.backgrounds   = bg.value?.backgrounds     ?? [];
        _S.cameraPresets = cam.value?.camera_presets ?? [];
        _S.soundPresets  = snd.value?.sound_presets  ?? [];
        _S.savedComps    = comps.value?.compositions ?? [];
    } catch (e) {
        console.error("fbt CompositionEditor: resource load error", e);
    }
}

// ── Sidebar ────────────────────────────────────────────────────────────────────

function _buildSidebarSection(title, listEl) {
    const header = _mk("div", { cls: "fbt-ce-sb-header", textContent: title });
    const body   = _mk("div", { cls: "fbt-ce-sb-body" }, [listEl]);
    _sectionToggle(header, body);
    return _mk("div", { cls: "fbt-ce-sb-section" }, [header, body]);
}

function _populateSavedList() {
    const list = _dom.savedList;
    if (!list) return;
    list.innerHTML = "";
    if (!_S.savedComps.length) {
        list.appendChild(_mk("div", { cls: "fbt-ce-empty", textContent: "No saved compositions" }));
        return;
    }
    _S.savedComps.forEach(comp => {
        const row = _mk("div", { cls: "fbt-ce-sb-item" });
        row.appendChild(_mk("span", { cls: "fbt-ce-sb-name", textContent: comp.name || comp.id }));
        const actions = _mk("span", { cls: "fbt-ce-sb-actions" });
        actions.appendChild(_mk("button", {
            cls: "fbt-ce-icon-btn", title: "Load",
            textContent: "⇩",
            onclick: () => _onLoad(comp.id),
        }));
        actions.appendChild(_mk("button", {
            cls: "fbt-ce-icon-btn fbt-ce-danger", title: "Delete",
            textContent: "✕",
            onclick: () => _onDeleteComp(comp.id, comp.name),
        }));
        row.appendChild(actions);
        list.appendChild(row);
    });
}

function _populateSubjectList() {
    const list = _dom.subjectList;
    if (!list) return;
    list.innerHTML = "";

    // "New Subject" quick-add button
    list.appendChild(_mk("div", {
        cls: "fbt-ce-sb-item fbt-ce-sb-new",
        textContent: "+ New Subject…",
        onclick: () => _showNewSubjectForm(),
    }));

    if (!_S.subjects.length) {
        list.appendChild(_mk("div", { cls: "fbt-ce-empty", textContent: "No subjects defined" }));
        return;
    }
    _S.subjects.forEach(s => {
        const item = _mk("div", { cls: "fbt-ce-sb-item fbt-ce-clickable" });
        const nameEl = _mk("span", { cls: "fbt-ce-sb-name", textContent: s.name || s.id });
        const hint   = _mk("span", {
            cls: "fbt-ce-sb-hint",
            title: s.appearance_summary || "",
            textContent: s.appearance_summary ? "…" : "",
        });
        item.title = s.appearance_summary || "";
        item.appendChild(nameEl);
        item.appendChild(hint);
        item.addEventListener("click", () => _assignNextSlot(s.id));
        list.appendChild(item);
    });
}

function _showNewSubjectForm() {
    const list = _dom.subjectList;
    if (!list) return;
    list.innerHTML = "";

    const nameEl    = _mk("input",    { cls: "fbt-ce-input", type: "text", placeholder: "Name*" });
    const summaryEl = _mk("textarea", { cls: "fbt-ce-textarea", placeholder: "Appearance summary…", rows: 2 });
    const conceptEl = _mk("input",    { cls: "fbt-ce-input", type: "text", placeholder: "Concept ID (optional)" });

    const form = _mk("div", { cls: "fbt-ce-inline-form" }, [
        _mk("div", { cls: "fbt-ce-form-label", textContent: "New Subject" }),
        nameEl, summaryEl, conceptEl,
    ]);

    const btnRow = _mk("div", { cls: "fbt-ce-form-btns" });
    btnRow.appendChild(_mk("button", {
        cls: "fbt-ce-btn fbt-ce-btn-primary",
        textContent: "Add",
        onclick: async () => {
            const name = nameEl.value.trim();
            if (!name) { nameEl.focus(); return; }
            const id = name.toLowerCase().replace(/\s+/g, "_").replace(/[^\w]/g, "");
            try {
                await compositionsApi.saveSubject({
                    id,
                    name,
                    appearance: { summary: summaryEl.value.trim() },
                    concept_id: conceptEl.value.trim(),
                });
                const res = await compositionsApi.listSubjects();
                _S.subjects = res.subjects ?? [];
                _populateSubjectList();
                _rebuildSlots();
                _toast(`Subject "${name}" added`, "success");
            } catch (e) {
                _toast("Failed: " + e.message, "error");
            }
        },
    }));
    btnRow.appendChild(_mk("button", {
        cls: "fbt-ce-btn",
        textContent: "Cancel",
        onclick: () => _populateSubjectList(),
    }));
    form.appendChild(btnRow);

    list.appendChild(form);
    nameEl.focus();
}

function _populateBgList() {
    const list = _dom.bgList;
    if (!list) return;
    list.innerHTML = "";

    // "New Background" quick-add button
    list.appendChild(_mk("div", {
        cls: "fbt-ce-sb-item fbt-ce-sb-new",
        textContent: "+ New Background…",
        onclick: () => _showNewBgForm(),
    }));

    if (!_S.backgrounds.length) {
        list.appendChild(_mk("div", { cls: "fbt-ce-empty", textContent: "No backgrounds defined" }));
        return;
    }
    _S.backgrounds.forEach(b => {
        const item = _mk("div", {
            cls: "fbt-ce-sb-item fbt-ce-clickable",
            title: b.description || "",
            onclick: () => _assignBg(b.id),
        });
        item.appendChild(_mk("span", { cls: "fbt-ce-sb-name", textContent: b.name || b.id }));
        list.appendChild(item);
    });
}

function _showNewBgForm() {
    const list = _dom.bgList;
    if (!list) return;
    list.innerHTML = "";

    const nameEl  = _mk("input",    { cls: "fbt-ce-input", type: "text", placeholder: "Name*" });
    const descEl  = _mk("textarea", { cls: "fbt-ce-textarea", placeholder: "Environment description…", rows: 2 });
    const lightEl = _mk("input",    { cls: "fbt-ce-input", type: "text", placeholder: "Lighting…" });
    const sndEl   = _mk("input",    { cls: "fbt-ce-input", type: "text", placeholder: "Default soundscape…" });

    const form = _mk("div", { cls: "fbt-ce-inline-form" }, [
        _mk("div", { cls: "fbt-ce-form-label", textContent: "New Background" }),
        nameEl, descEl, lightEl, sndEl,
    ]);

    const btnRow = _mk("div", { cls: "fbt-ce-form-btns" });
    btnRow.appendChild(_mk("button", {
        cls: "fbt-ce-btn fbt-ce-btn-primary",
        textContent: "Add",
        onclick: async () => {
            const name = nameEl.value.trim();
            if (!name) { nameEl.focus(); return; }
            try {
                const saved = await compositionsApi.saveBackground({
                    name,
                    description: descEl.value.trim(),
                    lighting:    lightEl.value.trim(),
                    soundscape:  sndEl.value.trim(),
                });
                const res = await compositionsApi.listBackgrounds();
                _S.backgrounds = res.backgrounds ?? [];
                _populateBgList();
                // Refresh the background dropdown in the editor
                if (_dom.bgSel) {
                    const cur = _dom.bgSel.value;
                    _dom.bgSel.innerHTML = "";
                    _bgOptions().forEach(o => {
                        const opt = document.createElement("option");
                        opt.value = o.id; opt.textContent = o.label;
                        if (o.id === cur) opt.selected = true;
                        _dom.bgSel.appendChild(opt);
                    });
                }
                _toast(`Background "${name}" added`, "success");
            } catch (e) {
                _toast("Failed: " + e.message, "error");
            }
        },
    }));
    btnRow.appendChild(_mk("button", {
        cls: "fbt-ce-btn",
        textContent: "Cancel",
        onclick: () => _populateBgList(),
    }));
    form.appendChild(btnRow);

    list.appendChild(form);
    nameEl.focus();
}

function _populatePresetList(list, presets, field) {
    if (!list) return;
    list.innerHTML = "";
    if (!presets.length) {
        list.appendChild(_mk("div", { cls: "fbt-ce-empty", textContent: "No presets defined" }));
        return;
    }
    presets.forEach(p => {
        const item = _mk("div", {
            cls: "fbt-ce-sb-item fbt-ce-clickable",
            title: `Click to copy: ${p.description || ""}`,
            onclick: () => {
                navigator.clipboard?.writeText(p.description || "").catch(() => {});
                _toast(`Copied: ${p.name}`, "success");
            },
        });
        item.appendChild(_mk("span", { cls: "fbt-ce-sb-name", textContent: p.name || p.id }));
        list.appendChild(item);
    });
}

function _refreshSidebar() {
    _populateSavedList();
    _populateSubjectList();
    _populateBgList();
    _populatePresetList(_dom.camList, _S.cameraPresets, "description");
    _populatePresetList(_dom.sndList, _S.soundPresets, "description");
}

function _buildSidebar(parent) {
    const sidebar = _mk("div", { cls: "fbt-ce-sidebar" });

    _dom.savedList  = _mk("div", { cls: "fbt-ce-sb-list" });
    _dom.subjectList = _mk("div", { cls: "fbt-ce-sb-list" });
    _dom.bgList     = _mk("div", { cls: "fbt-ce-sb-list" });
    _dom.camList    = _mk("div", { cls: "fbt-ce-sb-list" });
    _dom.sndList    = _mk("div", { cls: "fbt-ce-sb-list" });

    sidebar.appendChild(_buildSidebarSection("Saved Compositions", _dom.savedList));
    sidebar.appendChild(_buildSidebarSection("Subjects (click to assign)", _dom.subjectList));
    sidebar.appendChild(_buildSidebarSection("Backgrounds (click to assign)", _dom.bgList));
    sidebar.appendChild(_buildSidebarSection("Camera Presets (click to copy)", _dom.camList));
    sidebar.appendChild(_buildSidebarSection("Sound Presets (click to copy)", _dom.sndList));

    parent.appendChild(sidebar);
}

// ── Editor sections ────────────────────────────────────────────────────────────

function _labeledRow(label, inputEl, hint = "") {
    const row = _mk("div", { cls: "fbt-ce-row" });
    row.appendChild(_mk("label", { cls: "fbt-ce-label", textContent: label }));
    const wrap = _mk("div", { cls: "fbt-ce-input-wrap" });
    wrap.appendChild(inputEl);
    if (hint) wrap.appendChild(_mk("div", { cls: "fbt-ce-hint", textContent: hint }));
    row.appendChild(wrap);
    return row;
}

function _editorSection(title, buildFn) {
    const header = _mk("div", { cls: "fbt-ce-sec-header", textContent: title });
    const body   = _mk("div", { cls: "fbt-ce-sec-body" });
    _sectionToggle(header, body);
    buildFn(body);
    return _mk("div", { cls: "fbt-ce-section" }, [header, body]);
}

// Subject slots section
function _buildSubjectSlotsSection(parent) {
    _dom.slotsContainer = _mk("div", { cls: "fbt-ce-slots" });
    const addBtn = _mk("button", {
        cls: "fbt-ce-add-btn",
        textContent: "+ Add Subject Slot",
        onclick: () => {
            const key = _nextSlotKey();
            if (!key) return _toast("Maximum 9 subject slots", "warn");
            _S.composition.subjects[key] = "";
            _rebuildSlots();
            _markDirty();
        },
    });
    parent.appendChild(_dom.slotsContainer);
    parent.appendChild(addBtn);
    _rebuildSlots();
}

function _rebuildSlots() {
    const container = _dom.slotsContainer;
    if (!container) return;
    container.innerHTML = "";
    const comp = _S.composition;
    const slots = _slotKeys();
    slots.forEach(key => {
        const card  = _mk("div", { cls: "fbt-ce-slot-card" });
        const row   = _mk("div", { cls: "fbt-ce-slot-row" });
        const label = _mk("span", { cls: "fbt-ce-slot-label", textContent: key });

        // Subject dropdown
        const subSel = _sel(_subjectOptions(true), comp.subjects[key] || "");
        subSel.className = "fbt-ce-select";

        // Appearance info line below the row
        const infoEl = _mk("div", { cls: "fbt-ce-slot-info" });
        const updateInfo = (sid) => {
            const s = _S.subjects.find(x => x.id === sid);
            infoEl.textContent = s?.appearance_summary || "";
        };
        updateInfo(comp.subjects[key] || "");

        subSel.addEventListener("change", () => {
            comp.subjects[key] = subSel.value;
            updateInfo(subSel.value);
            _markDirty();
        });

        // Outfit override
        const outfit = _mk("input", {
            cls: "fbt-ce-input fbt-ce-outfit",
            type: "text",
            placeholder: "Outfit override…",
            value: comp.outfit_overrides?.[key] || "",
        });
        outfit.addEventListener("input", () => {
            if (!comp.outfit_overrides) comp.outfit_overrides = {};
            comp.outfit_overrides[key] = outfit.value;
            _markDirty();
        });

        // Remove slot
        const removeBtn = _mk("button", {
            cls: "fbt-ce-icon-btn fbt-ce-danger",
            title: "Remove slot",
            textContent: "✕",
            onclick: () => {
                delete comp.subjects[key];
                delete comp.outfit_overrides?.[key];
                _renumberSlots();
                _rebuildSlots();
                _refreshShotDialogueSpeakers();
                _markDirty();
            },
        });

        row.appendChild(label);
        row.appendChild(subSel);
        row.appendChild(outfit);
        row.appendChild(removeBtn);
        card.appendChild(row);
        card.appendChild(infoEl);
        container.appendChild(card);
    });
    if (!slots.length) {
        container.appendChild(_mk("div", { cls: "fbt-ce-empty", textContent: "No subject slots. Click subjects in the sidebar to assign." }));
    }
}

function _renumberSlots() {
    const comp = _S.composition;
    const oldKeys = Object.keys(comp.subjects || {}).sort();
    const newSubjects = {};
    const newOutfits = {};
    oldKeys.forEach((k, i) => {
        const newKey = `S${i + 1}`;
        newSubjects[newKey] = comp.subjects[k];
        if (comp.outfit_overrides?.[k]) newOutfits[newKey] = comp.outfit_overrides[k];
    });
    comp.subjects = newSubjects;
    comp.outfit_overrides = newOutfits;
    // Update shot dialogue speaker keys
    (comp.shots || []).forEach(shot => {
        if (shot.dialogue?.speaker) {
            const oldIdx = oldKeys.indexOf(shot.dialogue.speaker);
            if (oldIdx >= 0) shot.dialogue.speaker = `S${oldIdx + 1}`;
        }
    });
}

// Shots section
function _buildShotsSection(parent) {
    _dom.shotsContainer = _mk("div", { cls: "fbt-ce-shots" });
    const addBtn = _mk("button", {
        cls: "fbt-ce-add-btn",
        textContent: "+ Add Shot",
        onclick: () => {
            _S.composition.shots.push(_newShot());
            _rebuildShots();
            _markDirty();
        },
    });
    parent.appendChild(_dom.shotsContainer);
    parent.appendChild(addBtn);
    _rebuildShots();
}

function _buildShotCard(shot, index) {
    const card = _mk("div", { cls: "fbt-ce-shot-card" });

    // Header
    const hdr = _mk("div", { cls: "fbt-ce-shot-header" });
    hdr.appendChild(_mk("span", { cls: "fbt-ce-shot-num", textContent: `Shot ${index + 1}` }));
    hdr.appendChild(_mk("button", {
        cls: "fbt-ce-icon-btn fbt-ce-danger",
        title: "Remove shot",
        textContent: "✕",
        onclick: () => {
            _S.composition.shots.splice(index, 1);
            _rebuildShots();
            _markDirty();
        },
    }));
    card.appendChild(hdr);

    // Timestamp
    const ts = _mk("input", {
        cls: "fbt-ce-input fbt-ce-ts",
        type: "text",
        placeholder: "MM:SS.mmm (optional)",
        value: shot.timestamp || "",
    });
    ts.addEventListener("input", () => { shot.timestamp = ts.value.trim() || null; _markDirty(); });
    card.appendChild(_labeledRow("Timestamp", ts));

    // Camera — {S} completion enabled
    const cam = _mk("input", {
        cls: "fbt-ce-input",
        type: "text",
        placeholder: "Camera direction… (type { for slot reference)",
        value: shot.camera || "",
    });
    cam.addEventListener("input", () => { shot.camera = cam.value; _markDirty(); });
    _attachCompletion(cam);
    card.appendChild(_labeledRow("Camera", cam));

    // Action — {S} completion enabled
    const action = _mk("textarea", {
        cls: "fbt-ce-textarea fbt-ce-action",
        placeholder: "Describe what happens. Type { to insert a subject reference ({S1}, {S2} …).",
        value: shot.action || "",
        rows: 3,
    });
    action.addEventListener("input", () => { shot.action = action.value; _markDirty(); });
    _attachCompletion(action);
    card.appendChild(_labeledRow("Action", action));

    // Dialogue
    const dlgWrap = _mk("div", { cls: "fbt-ce-dialogue-wrap" });

    const hasDialogue = !!shot.dialogue;
    const dlgToggle = _mk("label", { cls: "fbt-ce-toggle-label" });
    const dlgCheck = _mk("input", { type: "checkbox" });
    dlgCheck.checked = hasDialogue;
    dlgToggle.appendChild(dlgCheck);
    dlgToggle.appendChild(document.createTextNode(" Has dialogue"));

    const dlgFields = _mk("div", { cls: "fbt-ce-dlg-fields", style: { display: hasDialogue ? "" : "none" } });

    if (!hasDialogue) shot.dialogue = null;

    const buildDlgFields = (dlg) => {
        dlgFields.innerHTML = "";
        const slots = _slotKeys();
        const speakerOpts = slots.length
            ? slots.map(k => ({
                id: k,
                label: `${k} — ${_S.subjects.find(s => s.id === _S.composition.subjects[k])?.name || _S.composition.subjects[k] || k}`,
              }))
            : [{ id: "S1", label: "S1" }];

        const spkSel = _sel(speakerOpts, dlg?.speaker || speakerOpts[0].id);
        spkSel.className = "fbt-ce-select";
        spkSel.addEventListener("change", () => { if (shot.dialogue) shot.dialogue.speaker = spkSel.value; _markDirty(); });

        const langSel = _sel(LANGUAGES.map(l => ({ id: l, label: l })), dlg?.language || "English");
        langSel.className = "fbt-ce-select";
        langSel.addEventListener("change", () => { if (shot.dialogue) shot.dialogue.language = langSel.value; _markDirty(); });

        const dlgText = _mk("textarea", {
            cls: "fbt-ce-textarea",
            placeholder: "Dialogue text…",
            value: dlg?.text || "",
            rows: 2,
        });
        dlgText.addEventListener("input", () => { if (shot.dialogue) shot.dialogue.text = dlgText.value; _markDirty(); });

        dlgFields.appendChild(_labeledRow("Speaker", spkSel));
        dlgFields.appendChild(_labeledRow("Language", langSel));
        dlgFields.appendChild(_labeledRow("Text", dlgText));
    };

    dlgCheck.addEventListener("change", () => {
        if (dlgCheck.checked) {
            shot.dialogue = { speaker: _slotKeys()[0] || "S1", language: "English", text: "" };
            buildDlgFields(shot.dialogue);
            dlgFields.style.display = "";
        } else {
            shot.dialogue = null;
            dlgFields.innerHTML = "";
            dlgFields.style.display = "none";
        }
        _markDirty();
    });

    if (hasDialogue) buildDlgFields(shot.dialogue);
    // Store rebuilder for when subject slots change
    dlgWrap._rebuildDlgFields = () => {
        if (dlgCheck.checked && shot.dialogue) buildDlgFields(shot.dialogue);
    };

    dlgWrap.appendChild(dlgToggle);
    dlgWrap.appendChild(dlgFields);
    card.appendChild(_labeledRow("Dialogue", dlgWrap));

    // Sound events
    const snd = _mk("input", {
        cls: "fbt-ce-input",
        type: "text",
        placeholder: "Sound events (optional)…",
        value: shot.sound_events || "",
    });
    snd.addEventListener("input", () => { shot.sound_events = snd.value.trim() || null; _markDirty(); });
    card.appendChild(_labeledRow("Sound", snd));

    return card;
}

function _rebuildShots() {
    const container = _dom.shotsContainer;
    if (!container) return;
    container.innerHTML = "";
    const shots = _S.composition.shots || [];
    if (!shots.length) {
        container.appendChild(_mk("div", { cls: "fbt-ce-empty", textContent: "No shots yet." }));
        return;
    }
    shots.forEach((shot, i) => container.appendChild(_buildShotCard(shot, i)));
}

function _refreshShotDialogueSpeakers() {
    // Rebuild all shot cards so speaker dropdowns reflect current slots
    _rebuildShots();
}

// ── Main editor area ───────────────────────────────────────────────────────────

function _buildEditor(parent) {
    const editorWrap = _mk("div", { cls: "fbt-ce-editor" });

    // ─── Top bar ────────────────────────────────────────────────────────────────
    const topBar = _mk("div", { cls: "fbt-ce-top-bar" });

    _dom.nameInput = _mk("input", {
        cls: "fbt-ce-name-input",
        type: "text",
        placeholder: "Composition name…",
    });
    _dom.nameInput.addEventListener("input", () => {
        _S.composition.name = _dom.nameInput.value;
        _markDirty();
    });

    _dom.dirtyDot = _mk("span", {
        cls: "fbt-ce-dirty",
        textContent: "●",
        title: "Unsaved changes",
        style: { display: "none" },
    });

    _dom.modelSel = _sel(MODEL_TYPES, "h3_ref2va");
    _dom.modelSel.className = "fbt-ce-select fbt-ce-model-sel";
    _dom.modelSel.addEventListener("change", () => {
        _S.composition.model_type = _dom.modelSel.value;
        _markDirty();
    });

    topBar.appendChild(_dom.nameInput);
    topBar.appendChild(_dom.dirtyDot);
    topBar.appendChild(_dom.modelSel);
    editorWrap.appendChild(topBar);

    // ─── Scrollable form ────────────────────────────────────────────────────────
    const form = _mk("div", { cls: "fbt-ce-form" });

    // Style
    form.appendChild(_editorSection("Style", body => {
        _dom.styleInput = _mk("input", {
            cls: "fbt-ce-input",
            type: "text",
            placeholder: "Visual style, e.g. cinematic with shallow depth of field…",
        });
        _dom.styleInput.addEventListener("input", () => { _S.composition.style = _dom.styleInput.value; _markDirty(); });
        body.appendChild(_dom.styleInput);
    }));

    // Subjects
    form.appendChild(_editorSection("Subjects", _buildSubjectSlotsSection));

    // Background — changing it offers to auto-fill the soundscape field
    form.appendChild(_editorSection("Background", body => {
        _dom.bgSel = _sel(_bgOptions(), "");
        _dom.bgSel.className = "fbt-ce-select";

        _dom.bgSoundscapeHint = _mk("div", { cls: "fbt-ce-bg-hint", style: { display: "none" } });

        _dom.bgSel.addEventListener("change", () => {
            const bgId = _dom.bgSel.value;
            _S.composition.background = bgId;
            _markDirty();

            // Auto-fill or offer to fill soundscape from background
            const bg = _S.backgrounds.find(b => b.id === bgId);
            if (bg?.soundscape) {
                if (!_S.composition.overall_soundscape?.trim()) {
                    // Field is empty — auto-fill silently
                    _S.composition.overall_soundscape = bg.soundscape;
                    if (_dom.soundscapeArea) _dom.soundscapeArea.value = bg.soundscape;
                    _dom.bgSoundscapeHint.style.display = "none";
                } else {
                    // Field already has content — show a replace button
                    _dom.bgSoundscapeHint.innerHTML = "";
                    _dom.bgSoundscapeHint.appendChild(_mk("button", {
                        cls: "fbt-ce-hint-btn",
                        textContent: "↙ Use background soundscape",
                        title: bg.soundscape,
                        onclick: () => {
                            _S.composition.overall_soundscape = bg.soundscape;
                            if (_dom.soundscapeArea) _dom.soundscapeArea.value = bg.soundscape;
                            _markDirty();
                            _dom.bgSoundscapeHint.style.display = "none";
                        },
                    }));
                    _dom.bgSoundscapeHint.style.display = "";
                }
            } else {
                _dom.bgSoundscapeHint.style.display = "none";
            }
        });

        body.appendChild(_dom.bgSel);
        body.appendChild(_dom.bgSoundscapeHint);
    }));

    // Shots
    form.appendChild(_editorSection("Shots", _buildShotsSection));

    // Soundscape
    form.appendChild(_editorSection("Overall Soundscape", body => {
        _dom.soundscapeArea = _mk("textarea", {
            cls: "fbt-ce-textarea",
            placeholder: "Describe the ambient sound environment…",
            rows: 3,
        });
        _dom.soundscapeArea.addEventListener("input", () => { _S.composition.overall_soundscape = _dom.soundscapeArea.value; _markDirty(); });
        body.appendChild(_dom.soundscapeArea);
    }));

    // Music
    form.appendChild(_editorSection("Non-Diegetic Music", body => {
        _dom.musicArea = _mk("textarea", {
            cls: "fbt-ce-textarea",
            placeholder: "Background music description, or 'N/A'…",
            rows: 2,
        });
        _dom.musicArea.addEventListener("input", () => { _S.composition.non_diegetic_music = _dom.musicArea.value; _markDirty(); });
        body.appendChild(_dom.musicArea);
    }));

    editorWrap.appendChild(form);

    // ─── Action bar ─────────────────────────────────────────────────────────────
    const actionBar = _mk("div", { cls: "fbt-ce-action-bar" });
    _dom.statusEl = _mk("span", { cls: "fbt-ce-status" });

    actionBar.appendChild(_mk("button", {
        cls: "fbt-ce-btn",
        textContent: "Preview Raw",
        title: "Assemble and preview the prompt for the selected model type",
        onclick: _onPreview,
    }));
    actionBar.appendChild(_mk("button", {
        cls: "fbt-ce-btn",
        textContent: "Copy",
        title: "Assemble and copy prompt to clipboard",
        onclick: _onCopy,
    }));
    actionBar.appendChild(_mk("button", {
        cls: "fbt-ce-btn fbt-ce-btn-primary",
        textContent: "Save",
        title: "Save composition (Ctrl+S)",
        onclick: _onSave,
    }));
    actionBar.appendChild(_mk("button", {
        cls: "fbt-ce-btn",
        textContent: "New",
        title: "Start a new composition",
        onclick: _onNew,
    }));
    actionBar.appendChild(_dom.statusEl);
    editorWrap.appendChild(actionBar);

    parent.appendChild(editorWrap);
}

// ── Preview modal ──────────────────────────────────────────────────────────────

function _showPreviewModal(result) {
    const existing = document.getElementById("fbt-ce-preview-modal");
    if (existing) existing.remove();

    const modal = _mk("div", {
        cls: "fbt-ce-preview-modal",
        id: "fbt-ce-preview-modal",
    });

    const hdr = _mk("div", { cls: "fbt-ce-preview-hdr" });
    hdr.appendChild(_mk("span", { textContent: "Preview — " + (_S.composition.model_type || "") }));
    const closeBtn = _mk("button", {
        cls: "fbt-ce-icon-btn",
        textContent: "✕",
        onclick: () => modal.remove(),
    });
    hdr.appendChild(closeBtn);
    modal.appendChild(hdr);

    if (result.warnings?.length) {
        const warn = _mk("div", { cls: "fbt-ce-preview-warn" });
        result.warnings.forEach(w => warn.appendChild(_mk("div", { textContent: "⚠ " + w })));
        modal.appendChild(warn);
    }

    const pre = _mk("pre", { cls: "fbt-ce-preview-text", textContent: result.prompt || "(empty)" });
    modal.appendChild(pre);

    const footer = _mk("div", { cls: "fbt-ce-preview-footer" });
    footer.appendChild(_mk("button", {
        cls: "fbt-ce-btn",
        textContent: "Copy to Clipboard",
        onclick: () => {
            navigator.clipboard?.writeText(result.prompt || "").then(
                () => _toast("Copied to clipboard", "success"),
                () => _toast("Clipboard unavailable", "warn"),
            );
        },
    }));
    footer.appendChild(_mk("button", {
        cls: "fbt-ce-btn",
        textContent: "Close",
        onclick: () => modal.remove(),
    }));

    if (result.assembly_report) {
        const rpt = _mk("details", { cls: "fbt-ce-preview-report" });
        rpt.appendChild(_mk("summary", { textContent: "Assembly report" }));
        rpt.appendChild(_mk("pre", { textContent: result.assembly_report }));
        footer.appendChild(rpt);
    }

    modal.appendChild(footer);

    // Mount inside the editor panel so it scrolls with it
    const panel = _dom.panel;
    if (panel) {
        panel.style.position = "relative";
        panel.appendChild(modal);
    } else {
        document.body.appendChild(modal);
    }
}

// ── Actions ────────────────────────────────────────────────────────────────────

function _setStatus(msg, error = false) {
    if (!_dom.statusEl) return;
    _dom.statusEl.textContent = msg;
    _dom.statusEl.style.color = error ? "var(--p-red-400, #f87171)" : "var(--p-green-400, #4ade80)";
    if (msg) setTimeout(() => { if (_dom.statusEl) _dom.statusEl.textContent = ""; }, 3000);
}

async function _onPreview() {
    const comp = _S.composition;
    const mt = comp.model_type || "h3_ref2va";
    _setStatus("Assembling…");
    try {
        const result = await compositionsApi.assembleComposition(comp, mt);
        _showPreviewModal(result);
        _setStatus("");
    } catch (e) {
        _setStatus("Assembly failed: " + e.message, true);
    }
}

async function _onCopy() {
    const comp = _S.composition;
    const mt = comp.model_type || "h3_ref2va";
    _setStatus("Assembling…");
    try {
        const result = await compositionsApi.assembleComposition(comp, mt);
        await navigator.clipboard?.writeText(result.prompt || "");
        _setStatus("Copied!");
        _toast("Prompt copied to clipboard", "success");
    } catch (e) {
        _setStatus("Error: " + e.message, true);
    }
}

async function _onSave() {
    const comp = _S.composition;
    if (!comp.name?.trim()) {
        _setStatus("Enter a composition name first", true);
        return;
    }
    _setStatus("Saving…");
    try {
        const saved = await compositionsApi.saveComposition(comp);
        _S.composition.id = saved.id || comp.id;
        _markClean();
        // Refresh saved list
        const list = await compositionsApi.listCompositions();
        _S.savedComps = list.compositions ?? [];
        _populateSavedList();
        _setStatus("Saved ✓");
    } catch (e) {
        _setStatus("Save failed: " + e.message, true);
    }
}

async function _onLoad(id) {
    if (_S.dirty) {
        if (!confirm("Discard unsaved changes?")) return;
    }
    try {
        const comp = await compositionsApi.getComposition(id);
        _S.composition = comp;
        _populateEditor();
        _markClean();
    } catch (e) {
        _setStatus("Load failed: " + e.message, true);
    }
}

async function _onDeleteComp(id, name) {
    if (!confirm(`Delete "${name}"?`)) return;
    try {
        await compositionsApi.deleteComposition(id);
        const list = await compositionsApi.listCompositions();
        _S.savedComps = list.compositions ?? [];
        _populateSavedList();
        _toast(`Deleted "${name}"`, "success");
    } catch (e) {
        _setStatus("Delete failed: " + e.message, true);
    }
}

function _onNew() {
    if (_S.dirty && !confirm("Discard unsaved changes?")) return;
    _S.composition = _newComp();
    _populateEditor();
    _markClean();
}

// ── Sidebar click actions ──────────────────────────────────────────────────────

function _assignNextSlot(subjectId) {
    const slots = _slotKeys();
    // Find first empty slot or add a new one
    const emptyKey = slots.find(k => !_S.composition.subjects[k]);
    if (emptyKey) {
        _S.composition.subjects[emptyKey] = subjectId;
    } else {
        const key = _nextSlotKey();
        if (!key) { _toast("Maximum 9 slots reached", "warn"); return; }
        _S.composition.subjects[key] = subjectId;
    }
    _rebuildSlots();
    _markDirty();
    const name = _S.subjects.find(s => s.id === subjectId)?.name || subjectId;
    _toast(`Assigned ${name}`, "success");
}

function _assignBg(bgId) {
    _S.composition.background = bgId;
    if (_dom.bgSel) {
        // rebuild background select options to reflect current list
        _dom.bgSel.innerHTML = "";
        _bgOptions().forEach(o => {
            const opt = document.createElement("option");
            opt.value = o.id;
            opt.textContent = o.label;
            if (o.id === bgId) opt.selected = true;
            _dom.bgSel.appendChild(opt);
        });
    }
    _markDirty();
    const name = _S.backgrounds.find(b => b.id === bgId)?.name || bgId;
    _toast(`Background: ${name}`, "success");
}

// ── Populate editor from state ─────────────────────────────────────────────────

function _populateEditor() {
    const comp = _S.composition;
    if (!comp) return;
    if (_dom.nameInput) _dom.nameInput.value = comp.name || "";
    if (_dom.modelSel) _dom.modelSel.value = comp.model_type || "h3_ref2va";
    if (_dom.styleInput) _dom.styleInput.value = comp.style || "";
    if (_dom.soundscapeArea) _dom.soundscapeArea.value = comp.overall_soundscape || "";
    if (_dom.musicArea) _dom.musicArea.value = comp.non_diegetic_music || "";

    // Rebuild dynamic sections
    _rebuildSlots();
    _rebuildShots();

    // Update background dropdown
    if (_dom.bgSel) {
        _dom.bgSel.innerHTML = "";
        _bgOptions().forEach(o => {
            const opt = document.createElement("option");
            opt.value = o.id;
            opt.textContent = o.label;
            if (o.id === (comp.background || "")) opt.selected = true;
            _dom.bgSel.appendChild(opt);
        });
    }
}

// ── Panel construction ─────────────────────────────────────────────────────────

function _buildPanel(el) {
    el.innerHTML = "";
    _dom.panel = _mk("div", { cls: "fbt-ce-panel" });
    el.appendChild(_dom.panel);

    const body = _mk("div", { cls: "fbt-ce-body" });
    _buildSidebar(body);
    _buildEditor(body);
    _dom.panel.appendChild(body);

    // Ctrl+S to save — stopPropagation prevents ComfyUI's document-level handler from also firing
    el.addEventListener("keydown", e => {
        if (e.ctrlKey && e.key === "s") { e.preventDefault(); e.stopPropagation(); _onSave(); }
    });
}

// ── Public entry point ─────────────────────────────────────────────────────────

export async function renderCompositionEditor(el) {
    if (el.dataset.fbtceBuilt) {
        // Re-shown: refresh resource lists only
        await _loadResources();
        _refreshSidebar();
        return;
    }
    el.dataset.fbtceBuilt = "1";
    Object.assign(el.style, {
        display: "flex",
        flexDirection: "column",
        height: "100%",
        overflow: "hidden",
    });

    _buildPanel(el);
    _S.composition = _newComp();
    await _loadResources();
    _refreshSidebar();
    _populateEditor();
    _markClean();
}
