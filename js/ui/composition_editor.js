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
import { llmApi } from "../api/llm.js";
import { libberAPI } from "../api/libber.js";

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

const LORA_MODEL_TARGETS = [
    "LTX2.3", "Wan2.2-Native-High", "Wan2.2-Native-Low",
    "Wan2.2-Wrapper-High", "Wan2.2-Wrapper-Low",
    "Flux2/Klein", "Qwen", "MiniMaxH3", "Z-Image",
];

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
    // Libber state
    libbers:        [],    // available libber filenames from /fbtools/libber/list
    libberData:     {},    // filename → {keys, lib_dict} cache
    settings:       { libber_delimiter: "%" },
    // LoRA state
    lorasList:      [],    // LoRA filenames from /fbtools/loras/list
    // Outfit registry state
    outfits:        {},    // id → {name, description, tags} from /fbtools/outfits/registry
    // LLM assistant state
    llmModels:      [],    // descriptors from /fbtools/llm/models
    llmDefault:     null,  // recommended default model info
    llmLoaded:      null,  // currently loaded model name (string) or null
    llmVision:      false, // does loaded model support vision?
    llmBusy:        false, // in-flight load or generate
};

// Key DOM refs rebuilt on each panel render
const _dom = {};

// Slot-reference completion popup element (singleton)
let _completionEl = null;

// Index of the shot card most recently focused — used to target preset insertions
let _focusedShotIdx = -1;

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
        id: "", name: "",
        model_type: "h3_ref2va", style: "",
        concept_id: "",
        task_flags: [],
        use_dialogue_tags: false,
        subjects: {}, outfit_overrides: {},
        background: "", scene_synopsis: "", shots: [],
        overall_soundscape: "", non_diegetic_music: "",
        libbers: [],
        loras: [],
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

// ── Libber key completion (%key%) ──────────────────────────────────────────────

/** Build flat list of {displayKey, insertKey, libberName, isRandom} from attached libbers. */
function _libberCompletionKeys() {
    const attached = _S.composition?.libbers ?? [];
    if (!attached.length) return [];

    // Count occurrences of each key across libbers to detect duplicates
    const counts = {};
    attached.forEach(name => {
        const keys = _S.libberData[name]?.keys ?? [];
        keys.forEach(k => { counts[k] = (counts[k] || 0) + 1; });
    });

    const result = [];

    // Random wildcard entries — one per attached libber, plus one combined
    if (attached.length === 1) {
        result.push({ displayKey: "* random", insertKey: "*:1", libberName: attached[0], isRandom: true });
    } else {
        result.push({ displayKey: "* random (any)", insertKey: "*", libberName: "all libbers", isRandom: true });
        attached.forEach((name, i) => {
            result.push({ displayKey: `* random :${i + 1}`, insertKey: `*:${i + 1}`, libberName: name, isRandom: true });
        });
    }

    // Named key entries
    attached.forEach((name, i) => {
        const keys = _S.libberData[name]?.keys ?? [];
        keys.forEach(k => {
            const isDup = counts[k] > 1;
            result.push({
                displayKey: isDup ? `${k}:${i + 1}` : k,
                insertKey:  isDup ? `${k}:${i + 1}` : k,
                libberName: name,
                isRandom:   false,
            });
        });
    });
    return result;
}

/** Show libber key completion popup at the position of the opening delimiter. */
function _showLibberCompletion(textEl, matches, delimPos) {
    _dismissCompletion();
    if (!matches.length) return;
    const d = _S.settings?.libber_delimiter ?? "%";

    const popup = _mk("div", { cls: "fbt-ce-completion" });
    matches.forEach(({ displayKey, insertKey, libberName, isRandom }, i) => {
        const item = document.createElement("div");
        item.className = "fbt-ce-comp-item" + (i === 0 ? " active" : "") + (isRandom ? " fbt-ce-comp-random" : "");
        item.innerHTML = isRandom
            ? `<span class="fbt-ce-comp-random-key">${d}${insertKey}${d}</span><span class="fbt-ce-comp-name fbt-ce-comp-name-random">${libberName}</span>`
            : `<strong>${d}${displayKey}${d}</strong><span class="fbt-ce-comp-name">${libberName}</span>`;
        item.addEventListener("mousedown", e => {
            e.preventDefault();
            const curPos = textEl.selectionStart;
            const before = textEl.value.substring(0, delimPos);
            const after  = textEl.value.substring(curPos);
            const insert = `${d}${insertKey}${d}`;
            textEl.value = before + insert + after;
            const newPos = delimPos + insert.length;
            textEl.selectionStart = textEl.selectionEnd = newPos;
            textEl.dispatchEvent(new Event("input", { bubbles: true }));
            _dismissCompletion();
            textEl.focus();
        });
        popup.appendChild(item);
    });

    const rect = textEl.getBoundingClientRect();
    Object.assign(popup.style, {
        position: "fixed",
        left:     rect.left + "px",
        top:      Math.min(rect.bottom + 2, window.innerHeight - 140) + "px",
        minWidth: Math.min(200, rect.width) + "px",
        zIndex:   "9999",
    });
    document.body.appendChild(popup);
    _completionEl = popup;

    textEl.addEventListener("blur", _dismissCompletion, { once: true });
    setTimeout(() => {
        document.addEventListener("mousedown", function outsideClick(e) {
            if (!_completionEl?.contains(e.target)) _dismissCompletion();
            document.removeEventListener("mousedown", outsideClick);
        });
    }, 0);
}

/** Attaches libber %key% completion to a text input or textarea. */
function _attachLibberCompletion(el) {
    el.addEventListener("input", () => {
        const d = _S.settings?.libber_delimiter ?? "%";
        const pos    = el.selectionStart;
        const before = el.value.substring(0, pos);
        const last   = before.lastIndexOf(d);
        if (last === -1) return;

        // Only trigger when the delimiter is not preceded by an alphanumeric (e.g. "100%" → skip)
        const charBefore = last > 0 ? before[last - 1] : "";
        if (/[a-z0-9_]/i.test(charBefore)) return;

        const fragment = before.substring(last + d.length);
        // Fragment must be alphanumeric/underscore/colon/asterisk (covers %key:2%, %*:1%, %*%)
        if (!/^[a-z0-9_:*]*$/i.test(fragment)) return;

        const all = _libberCompletionKeys();
        if (!all.length) return;
        const frag = fragment.toLowerCase();
        // Typing "*" shows only random entries; anything else filters named keys (and hides random)
        const matches = frag === "*" || frag === ""
            ? all.filter(({ displayKey, isRandom }) => isRandom || displayKey.toLowerCase().startsWith(frag))
            : all.filter(({ displayKey, isRandom }) => !isRandom && displayKey.toLowerCase().startsWith(frag));
        if (!matches.length) { _dismissCompletion(); return; }
        _showLibberCompletion(el, matches, last);
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
        const [subj, bg, cam, snd, comps, llmData, llmStatus, settingsRes, libbersRes, lorasRes, outfitsRes] = await Promise.allSettled([
            compositionsApi.listSubjects(),
            compositionsApi.listBackgrounds(),
            compositionsApi.listCameraPresets(),
            compositionsApi.listSoundPresets(),
            compositionsApi.listCompositions(),
            llmApi.listModels(),
            llmApi.status(),
            compositionsApi.getSettings(),
            libberAPI.listLibbers(),
            compositionsApi.listLoras(),
            compositionsApi.getOutfitRegistry(),
        ]);
        _S.subjects      = subj.value?.subjects      ?? [];
        _S.backgrounds   = bg.value?.backgrounds     ?? [];
        _S.cameraPresets = cam.value?.camera_presets ?? [];
        _S.soundPresets  = snd.value?.sound_presets  ?? [];
        _S.savedComps    = comps.value?.compositions ?? [];
        _S.llmModels     = llmData.value?.models     ?? [];
        _S.llmDefault    = llmData.value?.default_model ?? null;
        const st = llmStatus.value;
        _S.llmLoaded     = st?.loaded_model  ?? null;
        _S.llmVision     = st?.supports_vision ?? false;
        if (settingsRes.value) {
            _S.settings = settingsRes.value;
            if (_dom.delimInput) _dom.delimInput.value = _S.settings.libber_delimiter ?? "%";
        }
        _S.libbers    = libbersRes.value?.files    ?? [];
        _S.lorasList  = lorasRes.value?.loras     ?? [];
        _S.outfits    = outfitsRes.value?.outfits ?? {};
    } catch (e) {
        console.error("fbt CompositionEditor: resource load error", e);
    }
    _populateLlmModelSel();
    _llmUpdateStatus();
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

async function _refreshBgDropdown(selectId) {
    const res = await compositionsApi.listBackgrounds();
    _S.backgrounds = res.backgrounds ?? [];
    _populateBgList();
    if (_dom.bgSel) {
        _dom.bgSel.innerHTML = "";
        _bgOptions().forEach(o => {
            const opt = document.createElement("option");
            opt.value = o.id; opt.textContent = o.label;
            if (o.id === selectId) opt.selected = true;
            _dom.bgSel.appendChild(opt);
        });
    }
}

function _populateBgList() {
    const list = _dom.bgList;
    if (!list) return;
    list.innerHTML = "";

    // "New Background" quick-add button
    list.appendChild(_mk("div", {
        cls: "fbt-ce-sb-item fbt-ce-sb-new",
        textContent: "+ New Background…",
        onclick: () => _showBgForm(null),
    }));

    if (!_S.backgrounds.length) {
        list.appendChild(_mk("div", { cls: "fbt-ce-empty", textContent: "No backgrounds defined" }));
        return;
    }
    _S.backgrounds.forEach(b => {
        const row = _mk("div", { cls: "fbt-ce-sb-item-row" });

        const nameEl = _mk("div", {
            cls: "fbt-ce-sb-item fbt-ce-clickable fbt-ce-sb-item-flex",
            title: b.description || "",
            onclick: () => _assignBg(b.id),
        });
        nameEl.appendChild(_mk("span", { cls: "fbt-ce-sb-name", textContent: b.name || b.id }));

        const editBtn = _mk("button", {
            cls: "fbt-ce-sb-edit-btn",
            textContent: "✎",
            title: "Edit background",
        });
        editBtn.addEventListener("click", e => { e.stopPropagation(); _showBgForm(b); });

        row.appendChild(nameEl);
        row.appendChild(editBtn);
        list.appendChild(row);
    });
}

function _showBgForm(existing) {
    const list = _dom.bgList;
    if (!list) return;
    list.innerHTML = "";

    const isEdit  = !!existing;
    const nameEl  = _mk("input",    { cls: "fbt-ce-input", type: "text", placeholder: "Name*" });
    const descEl  = _mk("textarea", { cls: "fbt-ce-textarea", placeholder: "Environment description…", rows: 2 });
    const lightEl = _mk("input",    { cls: "fbt-ce-input", type: "text", placeholder: "Lighting…" });
    const sndEl   = _mk("input",    { cls: "fbt-ce-input", type: "text", placeholder: "Default soundscape…" });

    if (isEdit) {
        nameEl.value  = existing.name        || "";
        descEl.value  = existing.description  || "";
        lightEl.value = existing.lighting     || "";
        sndEl.value   = existing.soundscape   || "";
    }

    const form = _mk("div", { cls: "fbt-ce-inline-form" }, [
        _mk("div", { cls: "fbt-ce-form-label", textContent: isEdit ? "Edit Background" : "New Background" }),
        nameEl, descEl, lightEl, sndEl,
    ]);

    const btnRow = _mk("div", { cls: "fbt-ce-form-btns" });
    btnRow.appendChild(_mk("button", {
        cls: "fbt-ce-btn fbt-ce-btn-primary",
        textContent: isEdit ? "Save" : "Add",
        onclick: async () => {
            const name = nameEl.value.trim();
            if (!name) { nameEl.focus(); return; }
            try {
                const bg = {
                    name,
                    description: descEl.value.trim(),
                    lighting:    lightEl.value.trim(),
                    soundscape:  sndEl.value.trim(),
                };
                if (isEdit) bg.id = existing.id;
                const saved = await compositionsApi.saveBackground(bg);
                await _refreshBgDropdown(_dom.bgSel?.value || "");
                _toast(`Background "${name}" ${isEdit ? "updated" : "added"}`, "success");
            } catch (e) {
                _toast("Failed: " + e.message, "error");
            }
        },
    }));

    if (isEdit) {
        btnRow.appendChild(_mk("button", {
            cls: "fbt-ce-btn fbt-ce-btn-danger",
            textContent: "Delete",
            onclick: async () => {
                if (!confirm(`Delete background "${existing.name}"?`)) return;
                try {
                    await compositionsApi.deleteBackground(existing.id);
                    // Clear the composition's background field if it pointed here
                    if (_S.composition.background === existing.id) {
                        _S.composition.background = "";
                        _markDirty();
                    }
                    await _refreshBgDropdown(_S.composition.background || "");
                    _toast(`Deleted "${existing.name}"`, "success");
                } catch (e) {
                    _toast("Failed to delete: " + e.message, "error");
                }
            },
        }));
    }

    btnRow.appendChild(_mk("button", {
        cls: "fbt-ce-btn",
        textContent: "Cancel",
        onclick: () => _populateBgList(),
    }));
    form.appendChild(btnRow);
    list.appendChild(form);
    nameEl.focus();
}

function _populatePresetList(list, presets, insertFn) {
    if (!list) return;
    list.innerHTML = "";
    if (!presets.length) {
        list.appendChild(_mk("div", { cls: "fbt-ce-empty", textContent: "No presets defined" }));
        return;
    }
    presets.forEach(p => {
        const item = _mk("div", {
            cls: "fbt-ce-sb-item fbt-ce-clickable",
            title: p.description || "",
            onclick: () => {
                if (insertFn) {
                    insertFn(p.description || "");
                } else {
                    navigator.clipboard?.writeText(p.description || "").catch(() => {});
                    _toast(`Copied: ${p.name}`, "success");
                }
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
    _rebuildPresetList("camera");
    _rebuildPresetList("sound");
}

// ── Camera / Sound preset sidebar sections ────────────────────────────────────

function _rebuildPresetList(kind) {
    const listEl  = kind === "camera" ? _dom.camList : _dom.sndList;
    const presets = kind === "camera" ? _S.cameraPresets : _S.soundPresets;
    const insertFn = kind === "camera" ? _insertCameraPreset : _insertSoundPreset;
    if (!listEl) return;
    listEl.innerHTML = "";
    if (!presets.length) {
        listEl.appendChild(_mk("div", { cls: "fbt-ce-empty", textContent: "No presets saved" }));
        return;
    }
    presets.forEach(p => {
        const row     = _mk("div", { cls: "fbt-ce-sb-item" });
        const nameEl  = _mk("span", { cls: "fbt-ce-sb-name fbt-ce-clickable",
            textContent: p.name || p.id, title: p.description || "",
            onclick: () => insertFn(p.description || "") });
        const actions = _mk("span", { cls: "fbt-ce-sb-actions" });
        const delBtn  = _mk("button", { cls: "fbt-ce-icon-btn fbt-ce-danger", title: "Delete preset", textContent: "✕",
            onclick: async () => {
                if (!confirm(`Delete preset "${p.name || p.id}"?`)) return;
                try {
                    if (kind === "camera") await compositionsApi.deleteCameraPreset(p.id);
                    else                  await compositionsApi.deleteSoundPreset(p.id);
                    const r = kind === "camera"
                        ? await compositionsApi.listCameraPresets()
                        : await compositionsApi.listSoundPresets();
                    if (kind === "camera") _S.cameraPresets = r.camera_presets ?? [];
                    else                   _S.soundPresets  = r.sound_presets  ?? [];
                    _rebuildPresetList(kind);
                    _toast("Preset deleted", "success");
                } catch (e) { _toast(`Delete failed: ${e.message}`, "error"); }
            }});
        actions.appendChild(delBtn);
        row.appendChild(nameEl);
        row.appendChild(actions);
        listEl.appendChild(row);
    });
}

function _buildPresetSection(parent, kind) {
    const title   = kind === "camera" ? "Camera Presets (click to apply)" : "Sound Presets (click to apply)";
    const body    = _mk("div", { cls: "fbt-ce-sb-body" });
    const listEl  = _mk("div", { cls: "fbt-ce-sb-list" });
    if (kind === "camera") _dom.camList = listEl;
    else                   _dom.sndList = listEl;

    const nameIn  = _mk("input",    { cls: "fbt-ce-input",    placeholder: "Preset name" });
    const textEl  = _mk("textarea", { cls: "fbt-ce-textarea", rows: 3,
        placeholder: kind === "camera" ? "Camera movement / framing…" : "Sound / ambience description…" });
    const saveBtn = _mk("button",   { cls: "fbt-ce-btn fbt-ce-btn-sm", textContent: "Save",
        onclick: async () => {
            const name        = nameIn.value.trim();
            const description = textEl.value.trim();
            if (!name || !description) { _toast("Name and text are required", "warn"); return; }
            try {
                if (kind === "camera") await compositionsApi.saveCameraPreset({ name, description });
                else                   await compositionsApi.saveSoundPreset({ name, description });
                const r = kind === "camera"
                    ? await compositionsApi.listCameraPresets()
                    : await compositionsApi.listSoundPresets();
                if (kind === "camera") _S.cameraPresets = r.camera_presets ?? [];
                else                   _S.soundPresets  = r.sound_presets  ?? [];
                _rebuildPresetList(kind);
                nameIn.value = "";
                textEl.value = "";
                _toast("Preset saved", "success");
            } catch (e) { _toast(`Save failed: ${e.message}`, "error"); }
        }});

    const form = _mk("div", { cls: "fbt-ce-preset-form" }, [nameIn, textEl, saveBtn]);
    body.appendChild(listEl);
    body.appendChild(form);
    _rebuildPresetList(kind);
    parent.appendChild(_buildSidebarSection(title, body));
}

// ── Outfit Registry sidebar section ───────────────────────────────────────────

const DEFAULT_OUTFIT_QUERY =
    "Describe this outfit concisely for video generation prompts. " +
    "Focus on garment types, colors, materials, and notable accessories.";

function _outfitIdFromRow(row) { return row?.dataset.outfitId ?? ""; }

function _rebuildOutfitList() {
    const list = _dom.outfitList;
    if (!list) return;
    list.innerHTML = "";
    const entries = Object.entries(_S.outfits).sort(([, a], [, b]) =>
        (a.name || "").localeCompare(b.name || ""));
    if (!entries.length) {
        list.appendChild(_mk("div", { cls: "fbt-ce-empty", textContent: "No outfits defined." }));
        return;
    }
    entries.forEach(([id, entry]) => {
        const row = _mk("div", { cls: "fbt-ce-outfit-row" });
        row.dataset.outfitId = id;
        const nameEl = _mk("div", { cls: "fbt-ce-outfit-name", textContent: entry.name || id });
        const idEl   = _mk("div", { cls: "fbt-ce-outfit-id",   textContent: id });
        const editBtn = _mk("button", { cls: "fbt-ce-btn fbt-ce-btn-sm", textContent: "Edit",
            onclick: () => _openOutfitEditor(id) });
        const delBtn = _mk("button", { cls: "fbt-ce-btn fbt-ce-btn-sm fbt-ce-btn-danger", textContent: "✕",
            onclick: async () => {
                if (!confirm(`Delete outfit '${id}'?`)) return;
                try {
                    await compositionsApi.deleteOutfit(id);
                    delete _S.outfits[id];
                    _rebuildOutfitList();
                    _rebuildSlots();
                } catch (e) { alert(`Delete failed: ${e.message}`); }
            }});
        const ctrl = _mk("div", { cls: "fbt-ce-outfit-ctrl" }, [editBtn, delBtn]);
        row.appendChild(_mk("div", { cls: "fbt-ce-outfit-info" }, [nameEl, idEl]));
        row.appendChild(ctrl);
        list.appendChild(row);
    });
}

function _openOutfitEditor(existingId) {
    const isNew = !existingId;
    const entry = existingId ? (_S.outfits[existingId] || {}) : {};

    // Modal overlay
    const overlay = _mk("div", { cls: "fbt-ce-modal-overlay",
        onclick: e => { if (e.target === overlay) overlay.remove(); } });

    const modal = _mk("div", { cls: "fbt-ce-modal" });
    overlay.appendChild(modal);

    modal.appendChild(_mk("div", { cls: "fbt-ce-modal-title",
        textContent: isNew ? "New Outfit" : `Edit Outfit: ${existingId}` }));

    const idInput   = _mk("input", { cls: "fbt-ce-input", placeholder: "outfit_id (snake_case)",
        value: existingId || "", disabled: !isNew });
    const nameInput = _mk("input", { cls: "fbt-ce-input", placeholder: "Display name",
        value: entry.name || "" });
    const tagsInput = _mk("input", { cls: "fbt-ce-input", placeholder: "Tags (comma-separated, optional)",
        value: (entry.tags || []).join(", ") });
    const descArea  = _mk("textarea", { cls: "fbt-ce-textarea fbt-ce-outfit-desc",
        placeholder: "Outfit description…", value: entry.description || "" });

    // LLM analyze section (shown only when vision model loaded)
    const analyzeSection = _mk("div", { cls: "fbt-ce-outfit-analyze" });
    if (_S.llmVision) {
        const imgInput   = _mk("input", { cls: "fbt-ce-input", placeholder: "Image filename (from ComfyUI input dir)" });
        const queryArea  = _mk("textarea", { cls: "fbt-ce-textarea fbt-ce-outfit-query",
            placeholder: "Describe what you want from the analysis…",
            value: DEFAULT_OUTFIT_QUERY });
        const analyzeBtn = _mk("button", { cls: "fbt-ce-btn",
            textContent: "🔍 Analyze with LLM",
            onclick: async () => {
                const imgName = imgInput.value.trim();
                if (!imgName) { alert("Enter an image filename first."); return; }
                analyzeBtn.disabled = true;
                analyzeBtn.textContent = "Analyzing…";
                try {
                    const r = await llmApi.generate(queryArea.value.trim() || DEFAULT_OUTFIT_QUERY,
                        { images: [imgName], max_tokens: 400 });
                    if (r?.text) descArea.value = r.text.trim();
                } catch (e) { alert(`LLM error: ${e.message}`); }
                finally {
                    analyzeBtn.disabled = false;
                    analyzeBtn.textContent = "🔍 Analyze with LLM";
                }
            }});
        analyzeSection.appendChild(_mk("div", { cls: "fbt-ce-hint", textContent: "LLM Image Analysis" }));
        analyzeSection.appendChild(imgInput);
        analyzeSection.appendChild(queryArea);
        analyzeSection.appendChild(analyzeBtn);
    }

    const saveBtn = _mk("button", { cls: "fbt-ce-btn fbt-ce-btn-primary", textContent: "Save",
        onclick: async () => {
            const id = isNew ? idInput.value.trim() : existingId;
            if (!id) { alert("Outfit ID is required."); return; }
            if (!/^[a-z0-9_]+$/.test(id)) { alert("ID must be lowercase letters, digits, or underscores."); return; }
            const outfit = {
                id,
                name:        nameInput.value.trim(),
                description: descArea.value.trim(),
                tags:        tagsInput.value.split(",").map(t => t.trim()).filter(Boolean),
            };
            try {
                await compositionsApi.saveOutfit(outfit);
                _S.outfits[id] = { name: outfit.name, description: outfit.description, tags: outfit.tags };
                _rebuildOutfitList();
                _rebuildSlots();
                overlay.remove();
            } catch (e) { alert(`Save failed: ${e.message}`); }
        }});
    const cancelBtn = _mk("button", { cls: "fbt-ce-btn", textContent: "Cancel",
        onclick: () => overlay.remove() });

    const btnRow = _mk("div", { cls: "fbt-ce-modal-btns" }, [cancelBtn, saveBtn]);

    modal.appendChild(_mk("label", { cls: "fbt-ce-label", textContent: "ID" }));
    modal.appendChild(idInput);
    modal.appendChild(_mk("label", { cls: "fbt-ce-label", textContent: "Name" }));
    modal.appendChild(nameInput);
    modal.appendChild(_mk("label", { cls: "fbt-ce-label", textContent: "Tags" }));
    modal.appendChild(tagsInput);
    modal.appendChild(_mk("label", { cls: "fbt-ce-label", textContent: "Description" }));
    modal.appendChild(descArea);
    if (_S.llmVision) modal.appendChild(analyzeSection);
    modal.appendChild(btnRow);

    document.body.appendChild(overlay);
    (isNew ? idInput : nameInput).focus();
}

function _buildOutfitsSection(parent) {
    const body = _mk("div", { cls: "fbt-ce-sb-body" });
    _dom.outfitList = _mk("div", { cls: "fbt-ce-sb-list" });

    const addBtn = _mk("button", { cls: "fbt-ce-btn fbt-ce-btn-sm",
        textContent: "+ New Outfit",
        onclick: () => _openOutfitEditor(null) });
    const reloadBtn = _mk("button", { cls: "fbt-ce-btn fbt-ce-btn-sm",
        textContent: "↺",
        title: "Reload outfit registry from disk",
        onclick: async () => {
            try {
                const r = await compositionsApi.getOutfitRegistry();
                _S.outfits = r?.outfits ?? {};
                _rebuildOutfitList();
            } catch (e) { console.error("Outfit reload error", e); }
        }});

    const ctrl = _mk("div", { cls: "fbt-ce-outfit-section-ctrl" }, [addBtn, reloadBtn]);
    body.appendChild(ctrl);
    body.appendChild(_dom.outfitList);
    _rebuildOutfitList();
    parent.appendChild(_buildSidebarSection("Outfits", body));
}

// ── LLM assistant sidebar section ─────────────────────────────────────────────

function _llmGetFocusedCard() {
    const cards = _dom.shotsContainer?.querySelectorAll(".fbt-ce-shot-card");
    return (_focusedShotIdx >= 0 && cards?.[_focusedShotIdx]) ? cards[_focusedShotIdx] : null;
}

function _llmSetBusy(busy) {
    _S.llmBusy = busy;
    if (_dom.llmGenSection) {
        _dom.llmGenSection.querySelectorAll("button").forEach(b => { b.disabled = busy; });
    }
    if (_dom.llmLoadBtn) _dom.llmLoadBtn.disabled = busy;
    if (_dom.llmUnloadBtn) _dom.llmUnloadBtn.disabled = busy;
}

function _llmUpdateStatus() {
    if (!_dom.llmStatusEl) return;
    if (_S.llmLoaded) {
        _dom.llmStatusEl.textContent = `Loaded: ${_S.llmLoaded}`;
        _dom.llmStatusEl.className = "fbt-ce-llm-status fbt-ce-llm-ok";
        if (_dom.llmUnloadBtn) _dom.llmUnloadBtn.style.display = "";
        if (_dom.llmLoadBtn)   _dom.llmLoadBtn.textContent = "Reload";
        if (_dom.llmGenSection) _dom.llmGenSection.style.display = "";
    } else {
        _dom.llmStatusEl.textContent = "No model loaded.";
        _dom.llmStatusEl.className = "fbt-ce-llm-status";
        if (_dom.llmUnloadBtn) _dom.llmUnloadBtn.style.display = "none";
        if (_dom.llmLoadBtn)   _dom.llmLoadBtn.textContent = "Load";
        if (_dom.llmGenSection) _dom.llmGenSection.style.display = "none";
    }
}

async function _llmLoadSelected() {
    const sel = _dom.llmModelSel;
    if (!sel) return;
    const modelInfo = _S.llmModels.find(m => m.id === sel.value);
    if (!modelInfo) { _toast("Select a model first", "warn"); return; }
    _llmSetBusy(true);
    _dom.llmStatusEl.textContent = `Loading ${modelInfo.name}…`;
    try {
        const r = await llmApi.loadModel(modelInfo);
        if (r.success) {
            _S.llmLoaded = modelInfo.name;
            _S.llmVision = modelInfo.supports_vision;
            _toast(`Loaded: ${modelInfo.name}`, "ok");
        } else {
            _toast(r.message || "Load failed", "error");
            _S.llmLoaded = null;
        }
    } catch (e) {
        _toast("Load error: " + e.message, "error");
        _S.llmLoaded = null;
    }
    _llmSetBusy(false);
    _llmUpdateStatus();
}

async function _llmUnload() {
    _llmSetBusy(true);
    try {
        await llmApi.unloadModel();
        _S.llmLoaded = null;
        _S.llmVision = false;
        _toast("Model unloaded", "ok");
    } catch (e) {
        _toast("Unload error: " + e.message, "error");
    }
    _llmSetBusy(false);
    _llmUpdateStatus();
}

async function _llmGenAction() {
    const card = _llmGetFocusedCard();
    if (!card) { _toast("Click inside a shot first", "warn"); return; }
    const shot = _S.composition.shots[_focusedShotIdx];
    if (!shot) return;

    const subjects = Object.entries(_S.composition.subjects || {})
        .map(([k]) => {
            const subj = _S.subjects.find(s => s.id === _S.composition.subjects[k]);
            return subj?.name || k;
        });
    const bg = _S.backgrounds.find(b => b.id === _S.composition.background);
    const environment = bg?.description || "";

    _llmSetBusy(true);
    _dom.llmStatusEl.textContent = "Generating action…";
    try {
        const r = await llmApi.generateShotAction({
            shotNumber:  _focusedShotIdx + 1,
            subjects,
            environment,
            style:    _S.composition.style || "cinematic",
            existing: shot.action || "",
        });
        if (r.success && r.text) {
            card._actInput.value = r.text;
            card._actInput.dispatchEvent(new Event("input", { bubbles: true }));
        } else {
            _toast(r.message || "Generation failed", "error");
        }
    } catch (e) {
        _toast("Error: " + e.message, "error");
    }
    _llmSetBusy(false);
    _llmUpdateStatus();
}

async function _llmGenDialogue() {
    const card = _llmGetFocusedCard();
    if (!card) { _toast("Click inside a shot first", "warn"); return; }
    const shot = _S.composition.shots[_focusedShotIdx];
    if (!shot) return;

    const speakerKey = shot.dialogue?.speaker || Object.keys(_S.composition.subjects || {})[0] || "S1";
    const subjectId  = _S.composition.subjects?.[speakerKey];
    const subject    = _S.subjects.find(s => s.id === subjectId);
    const speaker    = subject?.name || speakerKey;
    const lang       = shot.dialogue?.language || "English";
    const bg         = _S.backgrounds.find(b => b.id === _S.composition.background);
    const context    = (shot.action || "") + (bg?.description ? ` Setting: ${bg.description}` : "");

    // Ensure dialogue is toggled on
    if (!shot.dialogue) {
        shot.dialogue = { speaker: speakerKey, language: lang, text: "" };
        _rebuildShots();
    }

    _llmSetBusy(true);
    _dom.llmStatusEl.textContent = "Generating dialogue…";
    try {
        const r = await llmApi.generateDialogue({ speaker, context, language: lang });
        if (r.success && r.text) {
            // Find the dialogue text textarea in the rebuilt card and set it
            const newCard = _dom.shotsContainer?.querySelectorAll(".fbt-ce-shot-card")[_focusedShotIdx];
            const dlgTextarea = newCard?.querySelector(".fbt-ce-dlg-fields textarea");
            if (dlgTextarea) {
                dlgTextarea.value = r.text;
                dlgTextarea.dispatchEvent(new Event("input", { bubbles: true }));
            }
            if (shot.dialogue) shot.dialogue.text = r.text;
            _markDirty();
        } else {
            _toast(r.message || "Generation failed", "error");
        }
    } catch (e) {
        _toast("Error: " + e.message, "error");
    }
    _llmSetBusy(false);
    _llmUpdateStatus();
}

async function _llmPolishAction() {
    const card = _llmGetFocusedCard();
    if (!card?._actInput) { _toast("Click inside a shot first", "warn"); return; }
    const text = card._actInput.value.trim();
    if (!text) { _toast("Nothing to polish", "warn"); return; }
    _llmSetBusy(true);
    _dom.llmStatusEl.textContent = "Polishing…";
    try {
        const r = await llmApi.polish({ text, context: `Scene style: ${_S.composition.style || "cinematic"}` });
        if (r.success && r.text) {
            card._actInput.value = r.text;
            card._actInput.dispatchEvent(new Event("input", { bubbles: true }));
        } else {
            _toast(r.message || "Polish failed", "error");
        }
    } catch (e) {
        _toast("Error: " + e.message, "error");
    }
    _llmSetBusy(false);
    _llmUpdateStatus();
}

async function _llmDownloadDefault() {
    if (!_dom.llmDownloadBtn) return;
    _dom.llmDownloadBtn.disabled = true;
    _dom.llmDownloadBtn.textContent = "Downloading…";
    try {
        const r = await llmApi.downloadDefault();
        if (r.success) {
            _toast(`Downloaded to: ${r.model_dir}`, "ok");
            // Refresh model list
            _dom.llmDownloadBtn.textContent = "Done — refresh models";
            _dom.llmDownloadBtn.onclick = () => _llmRefreshModels();
        } else {
            _toast(r.error || "Download failed", "error");
            _dom.llmDownloadBtn.disabled = false;
            _dom.llmDownloadBtn.textContent = "Download";
        }
    } catch (e) {
        _toast("Download error: " + e.message, "error");
        _dom.llmDownloadBtn.disabled = false;
        _dom.llmDownloadBtn.textContent = "Download";
    }
}

async function _llmRefreshModels() {
    try {
        const data = await llmApi.listModels();
        _S.llmModels = data.models || [];
        _S.llmDefault = data.default_model || null;
        _populateLlmModelSel();
    } catch (_) { /* silent */ }
}

function _populateLlmModelSel() {
    const sel = _dom.llmModelSel;
    if (!sel) return;
    sel.innerHTML = "";
    const noModels = _S.llmModels.length === 0;
    if (noModels) {
        const o = document.createElement("option");
        o.value = "";
        o.textContent = "— no models found —";
        sel.appendChild(o);
    } else {
        _S.llmModels.forEach(m => {
            const o = document.createElement("option");
            o.value = m.id;
            const tags = (m.capability_tags || []).join(" ");
            o.textContent = `${m.name}  ${tags}`;
            o.title = m.capability_note || "";
            sel.appendChild(o);
        });
    }
    if (_dom.llmLoadBtn)     _dom.llmLoadBtn.disabled = noModels;
    if (_dom.llmDownloadRow) _dom.llmDownloadRow.style.display = noModels ? "" : "none";
}

function _buildLlmSection(parent) {
    const body = _mk("div", { cls: "fbt-ce-sb-list fbt-ce-llm-body" });

    // Model selector row
    _dom.llmModelSel = _mk("select", { cls: "fbt-ce-select fbt-ce-llm-sel" });
    const selRow = _mk("div", { cls: "fbt-ce-llm-row" }, [_dom.llmModelSel]);

    // Refresh button
    const refreshBtn = _mk("button", {
        cls: "fbt-ce-icon-btn", title: "Re-scan model directories", textContent: "↻",
        onclick: _llmRefreshModels,
    });
    selRow.appendChild(refreshBtn);
    body.appendChild(selRow);

    // Status line
    _dom.llmStatusEl = _mk("div", { cls: "fbt-ce-llm-status", textContent: "No model loaded." });
    body.appendChild(_dom.llmStatusEl);

    // Capability note (updates when selection changes)
    const capNote = _mk("div", { cls: "fbt-ce-llm-cap-note" });
    body.appendChild(capNote);
    _dom.llmModelSel.addEventListener("change", () => {
        const m = _S.llmModels.find(x => x.id === _dom.llmModelSel.value);
        capNote.textContent = m?.capability_note || "";
    });

    // Load / Unload buttons
    const btnRow = _mk("div", { cls: "fbt-ce-llm-row" });
    _dom.llmLoadBtn = _mk("button", {
        cls: "fbt-ce-btn", textContent: "Load",
        onclick: _llmLoadSelected,
    });
    _dom.llmUnloadBtn = _mk("button", {
        cls: "fbt-ce-btn fbt-ce-btn-secondary", textContent: "Unload",
        style: { display: "none" },
        onclick: _llmUnload,
    });
    btnRow.appendChild(_dom.llmLoadBtn);
    btnRow.appendChild(_dom.llmUnloadBtn);
    body.appendChild(btnRow);

    // Generate buttons (hidden when no model loaded)
    _dom.llmGenSection = _mk("div", {
        cls: "fbt-ce-llm-gen",
        style: { display: "none" },
    });
    const genLabel = _mk("div", { cls: "fbt-ce-llm-gen-label", textContent: "Generate for focused shot:" });
    _dom.llmGenSection.appendChild(genLabel);

    const genBtnRow = _mk("div", { cls: "fbt-ce-llm-row fbt-ce-llm-gen-btns" });
    genBtnRow.appendChild(_mk("button", {
        cls: "fbt-ce-btn fbt-ce-btn-sm", textContent: "Action",
        title: "Draft / improve the shot action description",
        onclick: _llmGenAction,
    }));
    genBtnRow.appendChild(_mk("button", {
        cls: "fbt-ce-btn fbt-ce-btn-sm", textContent: "Dialogue",
        title: "Generate a dialogue line for the focused shot",
        onclick: _llmGenDialogue,
    }));
    genBtnRow.appendChild(_mk("button", {
        cls: "fbt-ce-btn fbt-ce-btn-sm", textContent: "Polish",
        title: "Polish / improve the existing action text",
        onclick: _llmPolishAction,
    }));
    _dom.llmGenSection.appendChild(genBtnRow);
    body.appendChild(_dom.llmGenSection);

    // Download prompt (hidden when models exist)
    _dom.llmDownloadRow = _mk("div", { cls: "fbt-ce-llm-download" });
    const d = _mk("div", { cls: "fbt-ce-llm-download-info" });
    d.innerHTML = (
        "<b>No LLMs found.</b> Place GGUF models in <code>ComfyUI/models/LLMs/&lt;name&gt;/</code>, " +
        "one subdirectory per model. Vision-capable GGUF models must include an <code>mmproj-*.gguf</code> file " +
        "alongside the main <code>.gguf</code>.<br><br>" +
        "Models stored elsewhere (e.g. a shared drive) can be registered by adding an <code>LLMs:</code> " +
        "entry to your <code>extra_model_paths.yaml</code> — ComfyUI will scan those paths too.<br><br>" +
        "Or download the recommended starter model:"
    );
    _dom.llmDownloadRow.appendChild(d);

    const defInfo = _mk("div", { cls: "fbt-ce-llm-default-info" });
    defInfo.innerHTML = (
        "<b>Qwen2.5-VL 3B Instruct</b> — ~2.6 GB total<br>" +
        "<span class='fbt-ce-llm-cap-note'>Accepts images. Video via frame sampling.</span><br>" +
        "<i>Requires: llama-cpp-python</i>"
    );
    _dom.llmDownloadRow.appendChild(defInfo);

    _dom.llmDownloadBtn = _mk("button", {
        cls: "fbt-ce-btn", textContent: "Download",
        onclick: _llmDownloadDefault,
    });
    _dom.llmDownloadRow.appendChild(_dom.llmDownloadBtn);
    body.appendChild(_dom.llmDownloadRow);

    parent.appendChild(_buildSidebarSection("🤖 LLM Assistant", body));
}

function _buildSettingsSection(parent) {
    const body = _mk("div", { cls: "fbt-ce-sb-body" });

    const row = _mk("div", { cls: "fbt-ce-settings-row" });
    row.appendChild(_mk("span", { cls: "fbt-ce-settings-label", textContent: "Libber delimiter" }));

    _dom.delimInput = _mk("input", {
        cls: "fbt-ce-delimiter-input",
        type: "text",
        maxLength: 1,
        value: _S.settings?.libber_delimiter ?? "%",
        title: "Single character used to wrap libber keys (e.g. %key%)",
    });
    _dom.delimInput.addEventListener("change", async () => {
        const d = _dom.delimInput.value;
        if (!d.length) { _dom.delimInput.value = _S.settings.libber_delimiter; return; }
        _S.settings.libber_delimiter = d;
        try { await compositionsApi.saveSettings(_S.settings); } catch (_) {}
        _rebuildLibbers(); // refresh key chips to show updated delimiter
    });
    row.appendChild(_dom.delimInput);
    body.appendChild(row);

    parent.appendChild(_buildSidebarSection("⚙ Settings", body));
}

function _buildSidebar(parent) {
    const sidebar = _mk("div", { cls: "fbt-ce-sidebar" });

    _dom.savedList  = _mk("div", { cls: "fbt-ce-sb-list" });
    _dom.subjectList = _mk("div", { cls: "fbt-ce-sb-list" });
    _dom.bgList     = _mk("div", { cls: "fbt-ce-sb-list" });
    sidebar.appendChild(_buildSidebarSection("Saved Compositions", _dom.savedList));
    sidebar.appendChild(_buildSidebarSection("Subjects (click to assign)", _dom.subjectList));
    sidebar.appendChild(_buildSidebarSection("Backgrounds (click to assign)", _dom.bgList));
    _buildPresetSection(sidebar, "camera");
    _buildPresetSection(sidebar, "sound");
    _buildOutfitsSection(sidebar);
    _buildLlmSection(sidebar);
    _buildSettingsSection(sidebar);

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

        // Concept ID — editable inline, saves to subject profile on change
        const conceptRow = _mk("div", { cls: "fbt-ce-slot-concept-row" });
        conceptRow.appendChild(_mk("span", { cls: "fbt-ce-slot-concept-label", textContent: "Concept" }));
        const conceptInput = _mk("input", {
            cls: "fbt-ce-input fbt-ce-slot-concept-input",
            type: "text",
            placeholder: "concept ID…",
            value: _S.subjects.find(s => s.id === (comp.subjects[key] || ""))?.concept_id || "",
        });
        const updateConceptId = (sid) => {
            conceptInput.value = _S.subjects.find(s => s.id === sid)?.concept_id || "";
        };
        conceptInput.addEventListener("change", async () => {
            const sid = comp.subjects[key];
            if (!sid) return;
            const newCid = conceptInput.value.trim();
            try {
                await compositionsApi.saveSubject({ id: sid, concept_id: newCid });
                const sub = _S.subjects.find(s => s.id === sid);
                if (sub) sub.concept_id = newCid;
            } catch (_) { _toast("Failed to update concept ID", "error"); }
        });
        conceptRow.appendChild(conceptInput);

        subSel.addEventListener("change", () => {
            comp.subjects[key] = subSel.value;
            updateInfo(subSel.value);
            updateConceptId(subSel.value);
            _markDirty();
        });

        // Outfit override — dropdown from registry
        const outfitSel = document.createElement("select");
        outfitSel.className = "fbt-ce-select fbt-ce-outfit";
        const noneOpt = document.createElement("option");
        noneOpt.value = "";
        noneOpt.textContent = "— default outfit —";
        outfitSel.appendChild(noneOpt);
        const currentOutfitText = comp.outfit_overrides?.[key] || "";
        Object.entries(_S.outfits)
            .sort(([, a], [, b]) => (a.name || "").localeCompare(b.name || ""))
            .forEach(([id, entry]) => {
                const opt = document.createElement("option");
                opt.value = entry.description || "";
                opt.textContent = entry.name || id;
                opt.selected = (entry.description || "") === currentOutfitText && currentOutfitText !== "";
                outfitSel.appendChild(opt);
            });
        outfitSel.addEventListener("change", () => {
            if (!comp.outfit_overrides) comp.outfit_overrides = {};
            comp.outfit_overrides[key] = outfitSel.value;
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
        row.appendChild(outfitSel);
        row.appendChild(removeBtn);
        card.appendChild(row);
        card.appendChild(infoEl);
        card.appendChild(conceptRow);
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
        title: "Add a new shot (Ctrl+Shift+N)",
        onclick: () => _addNewShot(),
    });
    parent.appendChild(_dom.shotsContainer);
    parent.appendChild(addBtn);
    _rebuildShots();
}

function _buildShotCard(shot, index) {
    const card = _mk("div", { cls: "fbt-ce-shot-card" });

    // Track focused shot index so sidebar presets know where to insert
    card.addEventListener("focusin", () => {
        _focusedShotIdx = index;
        _updateShotActive();
    });

    // Header
    const hdr = _mk("div", { cls: "fbt-ce-shot-header" });
    hdr.appendChild(_mk("span", { cls: "fbt-ce-shot-num", textContent: `Shot ${index + 1}` }));

    const hdrBtns = _mk("span", { cls: "fbt-ce-shot-hdr-btns" });
    if (index > 0) {
        hdrBtns.appendChild(_mk("button", {
            cls: "fbt-ce-icon-btn", title: "Move up", textContent: "↑",
            onclick: () => _moveShot(index, -1),
        }));
    }
    if (index < _S.composition.shots.length - 1) {
        hdrBtns.appendChild(_mk("button", {
            cls: "fbt-ce-icon-btn", title: "Move down", textContent: "↓",
            onclick: () => _moveShot(index, +1),
        }));
    }
    hdrBtns.appendChild(_mk("button", {
        cls: "fbt-ce-icon-btn", title: "Duplicate shot", textContent: "⧉",
        onclick: () => _duplicateShot(index),
    }));
    hdrBtns.appendChild(_mk("button", {
        cls: "fbt-ce-icon-btn fbt-ce-danger", title: "Remove shot", textContent: "✕",
        onclick: () => {
            _S.composition.shots.splice(index, 1);
            _focusedShotIdx = Math.min(_focusedShotIdx, _S.composition.shots.length - 1);
            _rebuildShots();
            _updateShotActive();
            _markDirty();
        },
    }));
    hdr.appendChild(hdrBtns);
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
    _attachLibberCompletion(cam);
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
    _attachLibberCompletion(action);
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
        _attachLibberCompletion(dlgText);

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

    // Store refs so sidebar preset/LLM handlers can insert into the right fields
    card._camInput = cam;
    card._sndInput = snd;
    card._actInput = action;

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

// ── Shot management helpers ────────────────────────────────────────────────────

function _updateShotActive() {
    const cards = _dom.shotsContainer?.querySelectorAll(".fbt-ce-shot-card");
    if (!cards) return;
    Array.from(cards).forEach((c, i) => c.classList.toggle("fbt-ce-shot-active", i === _focusedShotIdx));
}

function _addNewShot() {
    _S.composition.shots.push(_newShot());
    const newIdx = _S.composition.shots.length - 1;
    _focusedShotIdx = newIdx;
    _rebuildShots();
    _updateShotActive();
    _markDirty();
    const cards = _dom.shotsContainer?.querySelectorAll(".fbt-ce-shot-card");
    cards?.[newIdx]?.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function _moveShot(fromIdx, direction) {
    const shots = _S.composition.shots;
    const toIdx = fromIdx + direction;
    if (toIdx < 0 || toIdx >= shots.length) return;
    [shots[fromIdx], shots[toIdx]] = [shots[toIdx], shots[fromIdx]];
    _focusedShotIdx = toIdx;
    _rebuildShots();
    _updateShotActive();
    _markDirty();
}

function _duplicateShot(idx) {
    const dupe = JSON.parse(JSON.stringify(_S.composition.shots[idx]));
    dupe.id = `shot_${++_S.shotSeq}`;
    _S.composition.shots.splice(idx + 1, 0, dupe);
    _focusedShotIdx = idx + 1;
    _rebuildShots();
    _updateShotActive();
    _markDirty();
    const cards = _dom.shotsContainer?.querySelectorAll(".fbt-ce-shot-card");
    cards?.[_focusedShotIdx]?.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function _insertCameraPreset(text) {
    if (_focusedShotIdx < 0) { _toast("Click inside a shot first", "warn"); return; }
    const cards = Array.from(_dom.shotsContainer?.querySelectorAll(".fbt-ce-shot-card") || []);
    const card = cards[_focusedShotIdx];
    if (!card?._camInput) { _toast("Click inside a shot first", "warn"); return; }
    card._camInput.value = text;
    card._camInput.dispatchEvent(new Event("input", { bubbles: true }));
    _toast("Camera preset applied", "success");
}

function _insertSoundPreset(text) {
    if (_focusedShotIdx < 0) { _toast("Click inside a shot first", "warn"); return; }
    const cards = Array.from(_dom.shotsContainer?.querySelectorAll(".fbt-ce-shot-card") || []);
    const card = cards[_focusedShotIdx];
    if (!card?._sndInput) { _toast("Click inside a shot first", "warn"); return; }
    card._sndInput.value = text;
    card._sndInput.dispatchEvent(new Event("input", { bubbles: true }));
    _toast("Sound preset applied", "success");
}

// ── LoRAs section ─────────────────────────────────────────────────────────────

function _rebuildLoras() {
    const c = _dom.lorasContainer;
    if (!c) return;
    c.innerHTML = "";
    const loras = _S.composition?.loras ?? [];

    loras.forEach((entry, i) => {
        const row = _mk("div", { cls: "fbt-ce-lora-row" });

        // Name dropdown
        const nameSel = document.createElement("select");
        nameSel.className = "fbt-ce-select fbt-ce-lora-name";
        [{ id: "", label: "— select LoRA —" }, ..._S.lorasList.map(n => ({ id: n, label: n }))].forEach(({ id, label }) => {
            const o = document.createElement("option");
            o.value = id; o.textContent = label;
            if (id === entry.name) o.selected = true;
            nameSel.appendChild(o);
        });
        nameSel.addEventListener("change", () => { entry.name = nameSel.value; _markDirty(); });

        // Weight
        const weightInput = _mk("input", {
            cls: "fbt-ce-lora-weight",
            type: "number", min: 0, max: 2, step: 0.05,
            value: entry.weight ?? 1.0,
            title: "LoRA weight (strength_model and strength_clip)",
        });
        weightInput.addEventListener("input", () => { entry.weight = parseFloat(weightInput.value) || 1.0; _markDirty(); });

        // Target
        const targetSel = document.createElement("select");
        targetSel.className = "fbt-ce-select fbt-ce-lora-target";
        LORA_MODEL_TARGETS.forEach(t => {
            const o = document.createElement("option");
            o.value = t; o.textContent = t;
            if (t === (entry.target ?? "MiniMaxH3")) o.selected = true;
            targetSel.appendChild(o);
        });
        targetSel.addEventListener("change", () => { entry.target = targetSel.value; _markDirty(); });

        // Remove
        const removeBtn = _mk("button", {
            cls: "fbt-ce-icon-btn fbt-ce-danger", title: "Remove LoRA", textContent: "✕",
            onclick: () => { _S.composition.loras.splice(i, 1); _rebuildLoras(); _markDirty(); },
        });

        row.appendChild(nameSel);
        row.appendChild(weightInput);
        row.appendChild(targetSel);
        row.appendChild(removeBtn);
        c.appendChild(row);
    });

    if (!loras.length) {
        c.appendChild(_mk("div", { cls: "fbt-ce-empty", textContent: "No LoRAs attached." }));
    }
}

// ── Libbers section ───────────────────────────────────────────────────────────

async function _rebuildLibbers() {
    const c = _dom.libbersContainer;
    if (!c) return;
    c.innerHTML = "";

    if (!_S.libbers.length) {
        c.appendChild(_mk("div", { cls: "fbt-ce-empty", textContent: "No libber files found in libbers directory." }));
        return;
    }

    const attached = new Set(_S.composition?.libbers ?? []);
    const d = _S.settings?.libber_delimiter ?? "%";

    for (const name of _S.libbers) {
        const isAttached = attached.has(name);

        const row = _mk("div", { cls: "fbt-ce-libber-row" });
        const cb = _mk("input", { type: "checkbox" });
        cb.checked = isAttached;
        const label = _mk("span", { cls: "fbt-ce-libber-name", textContent: name });

        cb.addEventListener("change", async () => {
            if (!_S.composition.libbers) _S.composition.libbers = [];
            if (cb.checked) {
                if (!_S.composition.libbers.includes(name)) _S.composition.libbers.push(name);
                if (!_S.libberData[name]) {
                    try {
                        const key = name.replace(/\.json$/i, "");
                        _S.libberData[name] = await libberAPI.getLibberData(key);
                    } catch (_) {}
                }
            } else {
                _S.composition.libbers = _S.composition.libbers.filter(n => n !== name);
            }
            _markDirty();
            _rebuildLibbers();
        });

        row.appendChild(cb);
        row.appendChild(label);
        c.appendChild(row);

        // Show key chips when attached and data is loaded
        if (isAttached) {
            if (!_S.libberData[name]) {
                try {
                    const key = name.replace(/\.json$/i, "");
                    _S.libberData[name] = await libberAPI.getLibberData(key);
                } catch (_) {}
            }
            const keys = _S.libberData[name]?.keys ?? [];
            if (keys.length) {
                const chipsEl = _mk("div", { cls: "fbt-ce-libber-keys" });
                keys.forEach(k => {
                    chipsEl.appendChild(_mk("span", { cls: "fbt-ce-libber-key-chip", textContent: `${d}${k}${d}` }));
                });
                c.appendChild(chipsEl);
            }
        }
    }
}

// ── Main editor area ───────────────────────────────────────────────────────────

function _buildEditor(parent) {
    const editorWrap = _mk("div", { cls: "fbt-ce-editor" });

    // ─── Scrollable form ────────────────────────────────────────────────────────
    const form = _mk("div", { cls: "fbt-ce-form" });

    // Info (name + model)
    form.appendChild(_editorSection("Info", body => {
        _dom.nameInput = _mk("input", {
            cls: "fbt-ce-input",
            type: "text",
            placeholder: "Untitled…",
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
        _dom.modelSel.className = "fbt-ce-select";

        // Task flags row — only visible for h3_ref2va.
        // Official MiniMax types; "video reference" is NOT valid per spec.
        const H3_TASK_FLAGS = [
            "reference generation",
            "keyframe completion",
            "video editing",
            "video continuation",
            "audio reference",
            "audio reuse",
        ];
        _dom.taskFlagsRow = _mk("div", { cls: "fbt-ce-info-row fbt-ce-task-flags-row" });
        _dom.taskFlagsRow.appendChild(_mk("span", { cls: "fbt-ce-info-label", textContent: "Task flags" }));
        const flagsWrap = _mk("div", { cls: "fbt-ce-task-flags-wrap" });
        _dom.taskFlagBoxes = {};
        for (const flag of H3_TASK_FLAGS) {
            const lbl = _mk("label", { cls: "fbt-ce-task-flag-label" });
            const cb = _mk("input", { type: "checkbox" });
            cb.addEventListener("change", () => {
                const active = H3_TASK_FLAGS.filter(f => _dom.taskFlagBoxes[f]?.checked);
                _S.composition.task_flags = active;
                _markDirty();
            });
            _dom.taskFlagBoxes[flag] = cb;
            lbl.appendChild(cb);
            lbl.appendChild(document.createTextNode(" " + flag));
            flagsWrap.appendChild(lbl);
        }
        _dom.taskFlagsRow.appendChild(flagsWrap);

        function _updateTaskFlagsVisibility() {
            const show = (_dom.modelSel.value === "h3_ref2va");
            _dom.taskFlagsRow.style.display = show ? "" : "none";
        }

        _dom.modelSel.addEventListener("change", () => {
            _S.composition.model_type = _dom.modelSel.value;
            _updateTaskFlagsVisibility();
            _markDirty();
        });

        body.appendChild(_mk("div", { cls: "fbt-ce-info-row" }, [
            _mk("span", { cls: "fbt-ce-info-label", textContent: "Name" }),
            _dom.nameInput,
            _dom.dirtyDot,
        ]));
        body.appendChild(_mk("div", { cls: "fbt-ce-info-row" }, [
            _mk("span", { cls: "fbt-ce-info-label", textContent: "Model" }),
            _dom.modelSel,
        ]));
        body.appendChild(_dom.taskFlagsRow);
        _updateTaskFlagsVisibility();

        // Dialogue tags toggle — only relevant for H3 formats
        _dom.dialogueTagsCb = _mk("input", { type: "checkbox" });
        _dom.dialogueTagsCb.addEventListener("change", () => {
            _S.composition.use_dialogue_tags = _dom.dialogueTagsCb.checked;
            _markDirty();
        });
        _dom.dialogueTagsRow = _mk("div", { cls: "fbt-ce-info-row" }, [
            _mk("span", { cls: "fbt-ce-info-label", textContent: "Dialogue tags" }),
            _mk("label", { cls: "fbt-ce-task-flag-label" }, [
                _dom.dialogueTagsCb,
                document.createTextNode(" use <d></d> tags (off = quoted text)"),
            ]),
        ]);
        body.appendChild(_dom.dialogueTagsRow);

        _dom.compConceptInput = _mk("input", {
            cls: "fbt-ce-input",
            type: "text",
            placeholder: "Scene-level concept ID (optional)…",
        });
        _dom.compConceptInput.addEventListener("input", () => {
            _S.composition.concept_id = _dom.compConceptInput.value;
            _markDirty();
        });
        body.appendChild(_mk("div", { cls: "fbt-ce-info-row" }, [
            _mk("span", { cls: "fbt-ce-info-label", textContent: "Concept" }),
            _dom.compConceptInput,
        ]));
    }));

    // Style
    form.appendChild(_editorSection("Style", body => {
        _dom.styleInput = _mk("input", {
            cls: "fbt-ce-input",
            type: "text",
            placeholder: "Visual style, e.g. cinematic with shallow depth of field…",
        });
        _dom.styleInput.addEventListener("input", () => { _S.composition.style = _dom.styleInput.value; _markDirty(); });
        _attachLibberCompletion(_dom.styleInput);
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

    // Synopsis — concise scene overview used as the summary body in H3 prompts.
    // Supports {S1}/{S2}/… shorthand; expanded to <Subject N> labels at assemble time.
    form.appendChild(_editorSection("Synopsis", body => {
        _dom.synopsisArea = _mk("textarea", {
            cls: "fbt-ce-textarea",
            placeholder: "{S1} eating a cookie in the café. {S2} enters with a dog, which lunges toward the cookie.",
            rows: 3,
        });
        _dom.synopsisArea.addEventListener("input", () => {
            _S.composition.scene_synopsis = _dom.synopsisArea.value;
            _markDirty();
        });
        _attachCompletion(_dom.synopsisArea);
        body.appendChild(_dom.synopsisArea);
    }));

    // Shots
    form.appendChild(_editorSection("Shots", _buildShotsSection));

    // LoRAs
    form.appendChild(_editorSection("LoRAs", body => {
        _dom.lorasContainer = _mk("div", { cls: "fbt-ce-loras-list" });
        body.appendChild(_dom.lorasContainer);
        body.appendChild(_mk("button", {
            cls: "fbt-ce-add-btn",
            textContent: "+ Add LoRA",
            onclick: () => {
                if (!_S.composition.loras) _S.composition.loras = [];
                _S.composition.loras.push({ name: "", weight: 1.0, target: "MiniMaxH3" });
                _rebuildLoras();
                _markDirty();
            },
        }));
        _rebuildLoras();
    }));

    // Libbers
    form.appendChild(_editorSection("Libbers", body => {
        _dom.libbersContainer = _mk("div", { cls: "fbt-ce-libbers-list" });
        body.appendChild(_dom.libbersContainer);
        _rebuildLibbers();
    }));

    // Soundscape
    form.appendChild(_editorSection("Overall Soundscape", body => {
        _dom.soundscapeArea = _mk("textarea", {
            cls: "fbt-ce-textarea",
            placeholder: "Describe the ambient sound environment…",
            rows: 3,
        });
        _dom.soundscapeArea.addEventListener("input", () => { _S.composition.overall_soundscape = _dom.soundscapeArea.value; _markDirty(); });
        _attachLibberCompletion(_dom.soundscapeArea);
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
        _attachLibberCompletion(_dom.musicArea);
        body.appendChild(_dom.musicArea);
    }));

    editorWrap.appendChild(form);

    // ─── Action bar ─────────────────────────────────────────────────────────────
    const actionBar = _mk("div", { cls: "fbt-ce-action-bar" });
    _dom.statusEl = _mk("span", { cls: "fbt-ce-status" });

    actionBar.appendChild(_mk("button", {
        cls: "fbt-ce-btn",
        textContent: "Preview Raw",
        title: "Assemble and preview the prompt for the selected model type (Ctrl+Shift+P)",
        onclick: _onPreview,
    }));
    actionBar.appendChild(_mk("button", {
        cls: "fbt-ce-btn",
        textContent: "Copy",
        title: "Assemble and copy prompt to clipboard (Ctrl+Shift+C)",
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
    const _stwWrap = _mk("div", { cls: "fbt-ce-stw-wrap" });
    _stwWrap.appendChild(_mk("button", {
        cls: "fbt-ce-btn",
        textContent: "Send to Workflow",
        title: "Set a Prompt Composition Loader node on the canvas to load this composition",
        onclick: () => _onSendToWorkflow(_stwWrap),
    }));
    actionBar.appendChild(_stwWrap);
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
        // Increment server counter so PromptCompositionLoader nodes re-execute
        compositionsApi.reloadCompositions().catch(() => {});
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

function _onSendToWorkflow(wrapEl) {
    // Toggle: close existing picker if already open
    const existing = wrapEl.querySelector(".fbt-ce-stw-picker");
    if (existing) { existing.remove(); return; }

    const compName = _S.composition?.name;
    if (!compName) {
        _setStatus("Save the composition first.", true);
        return;
    }

    const TARGET_TYPE = "fbt_PromptCompositionLoader";
    const graphNodes = (typeof app !== "undefined" && app?.graph?._nodes) || [];
    const loaderNodes = graphNodes.filter(n => n.type === TARGET_TYPE);

    const picker = _mk("div", { cls: "fbt-ce-stw-picker" });

    if (loaderNodes.length === 0) {
        picker.appendChild(_mk("div", {
            cls: "fbt-ce-stw-empty",
            textContent: "No Prompt Composition Loader nodes on canvas",
        }));
    } else {
        loaderNodes.forEach(node => {
            const widget = node.widgets?.find(w => w.name === "composition_name");
            const curVal = widget?.value || "(empty)";
            const isCurrent = curVal === compName;
            const row = _mk("div", {
                cls: "fbt-ce-stw-row",
                title: `Set node #${node.id} to: ${compName}`,
            });
            row.appendChild(_mk("span", {
                cls: "fbt-ce-stw-row-label",
                textContent: (isCurrent ? "✓ " : "") + curVal,
            }));
            row.appendChild(_mk("span", { cls: "fbt-ce-stw-row-idx", textContent: `#${node.id}` }));
            row.addEventListener("click", () => {
                if (widget) {
                    widget.value = compName;
                    app.graph.setDirtyCanvas(true, false);
                    compositionsApi.reload().catch(() => {});
                }
                picker.remove();
                document.removeEventListener("pointerdown", _closeOnOutside, true);
                _setStatus(`Sent to node #${node.id}`);
            });
            picker.appendChild(row);
        });
    }

    // "Create new" row
    const newRow = _mk("div", {
        cls: "fbt-ce-stw-row fbt-ce-stw-new",
        textContent: "＋ New Prompt Composition Loader",
        title: "Add a new Prompt Composition Loader node to the canvas",
    });
    newRow.addEventListener("click", () => {
        picker.remove();
        document.removeEventListener("pointerdown", _closeOnOutside, true);
        if (typeof LiteGraph === "undefined" || !app?.graph) {
            _setStatus("Canvas not available", true);
            return;
        }
        const node = LiteGraph.createNode(TARGET_TYPE);
        if (!node) { _setStatus("Could not create node", true); return; }
        const mouse = app.canvas?.canvas_mouse;
        node.pos = mouse ? [mouse[0] + 20, mouse[1] + 20] : [200, 200];
        app.graph.add(node);
        // Widget may not exist until the node is fully initialized
        setTimeout(() => {
            const w = node.widgets?.find(w => w.name === "composition_name");
            if (w) {
                w.value = compName;
                app.graph.setDirtyCanvas(true, false);
                compositionsApi.reload().catch(() => {});
            }
        }, 80);
        _setStatus("Created Prompt Composition Loader");
    });
    picker.appendChild(newRow);

    // Click-away listener
    const _closeOnOutside = (e) => {
        if (!wrapEl.contains(e.target)) {
            picker.remove();
            document.removeEventListener("pointerdown", _closeOnOutside, true);
        }
    };
    document.addEventListener("pointerdown", _closeOnOutside, true);

    wrapEl.appendChild(picker);
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
    if (_dom.modelSel) {
        _dom.modelSel.value = comp.model_type || "h3_ref2va";
        if (_dom.taskFlagsRow) {
            _dom.taskFlagsRow.style.display = (_dom.modelSel.value === "h3_ref2va") ? "" : "none";
        }
    }
    if (_dom.compConceptInput) _dom.compConceptInput.value = comp.concept_id || "";
    if (_dom.dialogueTagsCb) _dom.dialogueTagsCb.checked = !!(comp.use_dialogue_tags);
    if (_dom.styleInput) _dom.styleInput.value = comp.style || "";
    if (_dom.synopsisArea) _dom.synopsisArea.value = comp.scene_synopsis || "";
    if (_dom.soundscapeArea) _dom.soundscapeArea.value = comp.overall_soundscape || "";
    if (_dom.musicArea) _dom.musicArea.value = comp.non_diegetic_music || "";

    // Populate task flags checkboxes
    if (_dom.taskFlagBoxes) {
        const activeFlags = comp.task_flags || [];
        for (const [flag, cb] of Object.entries(_dom.taskFlagBoxes)) {
            cb.checked = activeFlags.includes(flag);
        }
    }

    // Rebuild dynamic sections
    _rebuildSlots();
    _rebuildShots();
    _rebuildLoras();
    _rebuildLibbers();

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

    // Keyboard shortcuts — stopPropagation prevents ComfyUI's document-level handlers from also firing
    el.addEventListener("keydown", e => {
        if (e.ctrlKey && !e.shiftKey && e.key === "s") { e.preventDefault(); e.stopPropagation(); _onSave(); }
        if (e.ctrlKey && e.shiftKey && e.key === "N") { e.preventDefault(); e.stopPropagation(); _addNewShot(); }
        if (e.ctrlKey && e.shiftKey && e.key === "P") { e.preventDefault(); e.stopPropagation(); _onPreview(); }
        if (e.ctrlKey && e.shiftKey && e.key === "C") { e.preventDefault(); e.stopPropagation(); _onCopy(); }
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

    _S.composition = _newComp();
    _buildPanel(el);
    await _loadResources();
    _refreshSidebar();
    _populateEditor();
    _markClean();
}
