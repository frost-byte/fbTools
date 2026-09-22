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
import { buildFileTree } from "./file_tree.js";
import { makeEntry, buildHistorySection } from "../utils/llm_history.js";
import { lib, LIBRARY_CHANGED } from "./library_store.js";
import { mk as _mk, toast as _toast } from "./library_common.js";

// ── Constants ──────────────────────────────────────────────────────────────────

const SAVED_PAGE_SIZE   = 10;

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

// ── Media helpers (video filmstrip) ───────────────────────────────────────────

/** Dual-handle range slider — same logic as _buildRangeSlider in bundle_editor.js */
function _buildRangeSlider(minVal, maxVal, loVal, hiVal, { step = 0.1, onchange } = {}) {
    loVal = Math.max(minVal, Math.min(loVal, maxVal));
    hiVal = Math.max(loVal,  Math.min(hiVal, maxVal));
    const wrap  = _mk("div", { cls: "fbt-range-wrap" });
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
        const m = Math.floor(v / 60), s = (v % 60).toFixed(1);
        return m > 0 ? `${m}:${s.padStart(4, "0")}` : `${s}s`;
    };
    const updateFill = () => {
        if (range <= 0) return;
        const loV = parseFloat(lo.value), hiV = parseFloat(hi.value);
        const loP = ((loV - minVal) / range) * 100, hiP = ((hiV - minVal) / range) * 100;
        fill.style.left  = loP + "%";
        fill.style.width = (hiP - loP) + "%";
        loLabel.textContent = fmt(loV);
        hiLabel.textContent = fmt(hiV);
        lo.style.zIndex = loV >= hiV - range * 0.02 ? 5 : 2;
    };
    lo.addEventListener("input", () => {
        if (parseFloat(lo.value) > parseFloat(hi.value)) lo.value = hi.value;
        updateFill();
        onchange?.(parseFloat(lo.value), parseFloat(hi.value));
    });
    hi.addEventListener("input", () => {
        if (parseFloat(hi.value) < parseFloat(lo.value)) hi.value = lo.value;
        updateFill();
        onchange?.(parseFloat(lo.value), parseFloat(hi.value));
    });
    updateFill();
    return { el: wrap, lo, hi, updateFill,
        setValues(newLo, newHi) {
            lo.value = Math.max(minVal, Math.min(newLo, maxVal));
            hi.value = Math.max(parseFloat(lo.value), Math.min(newHi, maxVal));
            updateFill();
        },
    };
}

/** Call POST /fbtools/media/sample_frames — returns {frames, fps, duration, frame_count} */
async function _mediaSampleFrames({ filename, dir, startTime, duration, everyNth, cap }) {
    const res = await fetch("/fbtools/media/sample_frames", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            filename,
            dir:        dir       ?? "input",
            start_time: startTime ?? 0,
            duration:   duration  ?? 0,
            every_nth:  everyNth  ?? 1,
            cap:        cap       ?? 24,
        }),
    });
    if (!res.ok) { const t = await res.text(); throw new Error(t); }
    return res.json();
}

const LORA_MODEL_TARGETS = [
    "LTX2.3", "Wan2.2-Native-High", "Wan2.2-Native-Low",
    "Wan2.2-Wrapper-High", "Wan2.2-Wrapper-Low",
    "Flux2/Klein", "Qwen", "MiniMaxH3", "Z-Image",
];

// ── Module state ───────────────────────────────────────────────────────────────

const _S = {
    composition:    null,
    savedComps:     [],
    savedPage:      0,
    savedQuery:     "",
    view:           "list",   // "list" (saved compositions) | "editor"
    dirty:          false,
    // Libber state
    libbers:        [],    // available libber filenames from /fbtools/libber/list
    libberData:     {},    // filename → {keys, lib_dict} cache
    settings:       { libber_delimiter: "%" },
    // LoRA state
    lorasList:      [],    // LoRA filenames from /fbtools/loras/list
    // Outfit registry state
    // LLM assistant state
    llmBusy:        false, // in-flight load or generate
    // SAM2 segmentation state
    // Media file lists for outfit LLM browser
};

// Key DOM refs rebuilt on each panel render
const _dom = {};

// Slot-reference completion popup element (singleton)
let _completionEl = null;

// Index of the shot card most recently focused — used to target preset insertions
let _focusedShotIdx = -1;

// ── Helpers ────────────────────────────────────────────────────────────────────

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
    lib.subjects.forEach(s => opts.push({ id: s.id, label: s.name || s.id }));
    return opts;
}

function _bgOptions() {
    return [
        { id: "", label: "— none —" },
        ...lib.backgrounds.map(b => ({ id: b.id, label: b.name || b.id })),
    ];
}

function _newComp() {
    return {
        id: "", name: "",
        model_type: "h3_ref2va", style: "",
        concept_id: "",
        task_flags: [],
        use_dialogue_tags: false,
        subjects: {}, outfit_overrides: {}, outfit_ids: {},
        slot_descriptors: {}, appearance_overrides: {},
        background: "", scene_synopsis: "", shots: [],
        overall_soundscape: "", non_diegetic_music: "",
        libbers: [],
        loras: [],
    };
}

/**
 * Next unique shot id for `shots`, derived from its own contents rather than a session
 * counter — a counter that started fresh on page load (or on loading a composition whose
 * own shots already used higher numbers) could mint an id that collided with an existing
 * shot, silently merging the two in anything keyed by shot id (e.g. Scene Cast Build's
 * timeline lookup — a real composition hit this: two shots both ended up "shot_1", and
 * switching between them in the timeline always showed the same (last) shot's action text).
 */
export function _nextShotId(shots) {
    let max = 0;
    for (const s of shots || []) {
        const m = /^shot_(\d+)$/.exec(s?.id || "");
        if (m) max = Math.max(max, parseInt(m[1], 10));
    }
    return `shot_${max + 1}`;
}

function _newShot(shots) {
    return {
        id: _nextShotId(shots),
        timestamp: null, camera: "", action: "",
        dialogue: null, sound_events: null,
    };
}

// Spreadsheet-column-style slot letters: A, B, ..., Z, AA, AB, ...
// Mirrors utils/slot_letters.py::slot_letter() — keep in lockstep.
function _slotLetterForIndex(index) {
    let n = index + 1;
    let letters = "";
    while (n > 0) {
        const rem = (n - 1) % 26;
        letters = String.fromCharCode(65 + rem) + letters;
        n = Math.floor((n - 1) / 26);
    }
    return letters;
}

function _slotKeys() {
    // Composition subject-slot keys are letters (A, B, ..., AA, ...) — plain
    // insertion order is always correct (JS never reorders non-integer-index
    // string keys), and is required once slots can mix single/double letters
    // (a lexicographic sort would put "AA" before "B").
    return Object.keys(_S.composition?.subjects || {});
}

function _nextSlotKey() {
    const keys = new Set(_slotKeys());
    const MAX_SLOTS_SAFETY_CAP = 500; // guard against runaway loops, not a real product cap
    for (let i = 0; i < MAX_SLOTS_SAFETY_CAP; i++) {
        const k = _slotLetterForIndex(i);
        if (!keys.has(k)) return k;
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

// ── Slot-reference completion ({A}, {B} …) ─────────────────────────────────────

function _dismissCompletion() {
    if (_completionEl) { _completionEl.remove(); _completionEl = null; }
}

function _showCompletion(textEl, matches, bracePos) {
    _dismissCompletion();
    if (!matches.length) return;

    const popup = _mk("div", { cls: "fbt-ce-completion" });

    matches.forEach((key, i) => {
        const subId  = _S.composition?.subjects?.[key] || "";
        const subName = lib.subjects.find(s => s.id === subId)?.name || "";
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
        // Only complete short, slot-key-shaped fragments: "", "A", "AB" …
        if (fragment.length > 3 || !/^[A-Za-z]{0,3}$/.test(fragment)) { _dismissCompletion(); return; }
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
        const [subj, bg, cam, snd, comps, llmStatus, settingsRes, libbersRes, lorasRes, outfitsRes, sam2Res,
               mediaInImg, mediaOutImg, mediaInVid, mediaOutVid] = await Promise.allSettled([
            compositionsApi.listSubjects(),
            compositionsApi.listBackgrounds(),
            compositionsApi.listCameraPresets(),
            compositionsApi.listSoundPresets(),
            compositionsApi.listCompositions(),
            llmApi.status(),
            compositionsApi.getSettings(),
            libberAPI.listLibbers(),
            compositionsApi.listLoras(),
            compositionsApi.getOutfitRegistry(),
            fetch("/fbtools/outfits/sam2_status").then(r => r.json()),
            compositionsApi.listMedia("image", true),
            compositionsApi.listMedia("image", true, "output"),
            compositionsApi.listMedia("video", true),
            compositionsApi.listMedia("video", true, "output"),
        ]);
        lib.subjects      = subj.value?.subjects      ?? [];
        lib.backgrounds   = bg.value?.backgrounds     ?? [];
        lib.cameraPresets = cam.value?.camera_presets ?? [];
        lib.soundPresets  = snd.value?.sound_presets  ?? [];
        _S.savedComps    = comps.value?.compositions ?? [];
        const st = llmStatus.value;
        lib.llmLoaded      = st?.loaded_model   ?? null;
        lib.llmVision      = st?.supports_vision ?? false;
        lib.llmNativeVideo = st?.native_video    ?? false;
        _llmSyncBadge();
        if (settingsRes.value) {
            _S.settings = settingsRes.value;
            const st = _S.settings;
        }
        _S.libbers    = libbersRes.value?.files    ?? [];
        _S.lorasList  = lorasRes.value?.loras     ?? [];
        lib.outfits    = outfitsRes.value?.outfits ?? {};
        lib.sam2          = sam2Res.status === "fulfilled" ? sam2Res.value : null;
        lib.mediaInImages  = mediaInImg.value?.files  ?? [];
        lib.mediaOutImages = mediaOutImg.value?.files ?? [];
        lib.mediaInVideos  = mediaInVid.value?.files  ?? [];
        lib.mediaOutVideos = mediaOutVid.value?.files ?? [];
    } catch (e) {
        console.error("fbt CompositionEditor: resource load error", e);
    }
}

// ── Sidebar ────────────────────────────────────────────────────────────────────

function _syncBgDropdown(selectId) {
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

/** React to asset edits made in a modal here or in the Assets tab. */
function _onLibraryChanged(e) {
    const { kind, deletedId } = e.detail ?? {};
    if (kind === "backgrounds") {
        if (deletedId && _S.composition?.background === deletedId) {
            _S.composition.background = "";
            _markDirty();
        }
        _syncBgDropdown(_S.composition?.background || "");
    } else if (kind === "outfits" || kind === "subjects") {
        _dom.fillAddSubject?.();
        _rebuildSlots();
    } else if (kind === "cameraPresets" || kind === "soundPresets") {
        _rebuildShots();   // preset pickers live in the shot cards
    }
}

// ── Camera / Sound preset sidebar sections ────────────────────────────────────

// ── Outfit media helpers ───────────────────────────────────────────────────────

// ── Outfit Registry sidebar section ───────────────────────────────────────────

// ── LLM status (the Compose sidebar no longer has an LLM section) ──────────────

function _llmSyncBadge() {
    document.body.classList.toggle("fbt-llm-loaded", !!lib.llmLoaded);
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
            if (!key) return _toast("Could not allocate a new slot", "warn");
            _S.composition.subjects[key] = "";
            _rebuildSlots();
            _markDirty();
        },
    });
    // Add a subject straight from the library: fills the first empty slot, else adds a new one.
    const addSubjSel = _mk("select", { cls: "fbt-ce-select fbt-ce-add-subject-sel",
        title: "Add a subject from the library (manage subjects in the Assets tab)" });
    const fillAddSubject = () => {
        addSubjSel.innerHTML = "";
        addSubjSel.appendChild(_mk("option", { value: "", textContent: "+ Add subject…" }));
        lib.subjects.forEach(s => addSubjSel.appendChild(
            _mk("option", { value: s.id, textContent: s.name || s.id })));
    };
    fillAddSubject();
    _dom.fillAddSubject = fillAddSubject;
    addSubjSel.addEventListener("change", () => {
        const sid = addSubjSel.value;
        addSubjSel.value = "";
        if (!sid) return;
        const emptyKey = _slotKeys().find(k => !_S.composition.subjects[k]);
        const key = emptyKey || _nextSlotKey();
        if (!key) return _toast("Could not allocate a new slot", "warn");
        _S.composition.subjects[key] = sid;
        _rebuildSlots();
        _markDirty();
        _toast(`Assigned ${lib.subjects.find(s => s.id === sid)?.name || sid} to ${key}`, "success");
    });
    parent.appendChild(_dom.slotsContainer);
    parent.appendChild(_mk("div", { cls: "fbt-ce-slot-add-row" }, [addBtn, addSubjSel]));
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
            const s = lib.subjects.find(x => x.id === sid);
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
            value: lib.subjects.find(s => s.id === (comp.subjects[key] || ""))?.concept_id || "",
        });
        const updateConceptId = (sid) => {
            conceptInput.value = lib.subjects.find(s => s.id === sid)?.concept_id || "";
        };
        conceptInput.addEventListener("change", async () => {
            const sid = comp.subjects[key];
            if (!sid) return;
            const newCid = conceptInput.value.trim();
            try {
                await compositionsApi.saveSubject({ id: sid, concept_id: newCid });
                const sub = lib.subjects.find(s => s.id === sid);
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

        // Outfit — dropdown from registry, stores outfit ID in outfit_ids[key]
        const outfitSel = document.createElement("select");
        outfitSel.className = "fbt-ce-select fbt-ce-outfit";
        const noneOpt = document.createElement("option");
        noneOpt.value = "";
        noneOpt.textContent = "— no outfit —";
        outfitSel.appendChild(noneOpt);
        const currentOutfitId = comp.outfit_ids?.[key] || "";
        Object.entries(lib.outfits)
            .sort(([, a], [, b]) => (a.name || "").localeCompare(b.name || ""))
            .forEach(([id, entry]) => {
                const opt = document.createElement("option");
                opt.value = id;
                opt.textContent = entry.name || id;
                opt.selected = id === currentOutfitId;
                outfitSel.appendChild(opt);
            });
        // Info line: shows outfit description + reference image count
        const outfitInfoEl = _mk("div", { cls: "fbt-ce-slot-outfit-info" });
        const _updateOutfitInfo = (oid) => {
            const entry = lib.outfits[oid];
            if (!entry) { outfitInfoEl.textContent = ""; return; }
            const refCount = (entry.reference_images || []).filter(r => r?.use_as_reference).length;
            const hint = refCount ? ` · ${refCount} ref${refCount > 1 ? "s" : ""} → {Fit_N}` : "";
            outfitInfoEl.textContent = (entry.description || "").slice(0, 80) + hint;
        };
        _updateOutfitInfo(currentOutfitId);
        outfitSel.addEventListener("change", () => {
            if (!comp.outfit_ids) comp.outfit_ids = {};
            comp.outfit_ids[key] = outfitSel.value;
            _updateOutfitInfo(outfitSel.value);
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
                delete comp.outfit_ids?.[key];
                delete comp.slot_descriptors?.[key];
                delete comp.appearance_overrides?.[key];
                _renumberSlots();
                _rebuildSlots();
                _refreshShotDialogueSpeakers();
                _markDirty();
            },
        });

        // ── Appearance overrides ─────────────────────────────────────────────────
        // slot_descriptors[key]   — replaces appearance.summary for this composition
        // appearance_overrides[key] — granular sub-field overrides (face, hair, body)
        const overrideWrap = _mk("div", { cls: "fbt-ce-slot-override-wrap" });
        const overrideToggle = _mk("button", {
            cls: "fbt-ce-slot-override-toggle",
            title: "Override appearance fields for this composition",
        });
        const _hasOverrides = () => {
            const d = comp.slot_descriptors?.[key];
            const f = comp.appearance_overrides?.[key];
            return (d && d.trim()) || (f && Object.values(f).some(v => v?.trim()));
        };
        const _updateToggleLabel = () => {
            overrideToggle.textContent = "Override appearance" + (_hasOverrides() ? " ✎" : "");
        };
        _updateToggleLabel();
        const overrideBody = _mk("div", { cls: "fbt-ce-slot-override-body", style: { display: "none" } });
        overrideToggle.onclick = () => {
            const open = overrideBody.style.display === "none";
            overrideBody.style.display = open ? "" : "none";
        };

        // Description override
        const descLabel = _mk("label", { cls: "fbt-ce-slot-override-label",
            textContent: "Description (replaces profile summary)" });
        const descEl = _mk("textarea", {
            cls: "fbt-ce-textarea fbt-ce-slot-override-desc",
            placeholder: "Leave empty to use the profile's appearance summary",
            rows: 2,
        });
        descEl.value = comp.slot_descriptors?.[key] || "";
        descEl.addEventListener("input", () => {
            if (!comp.slot_descriptors) comp.slot_descriptors = {};
            comp.slot_descriptors[key] = descEl.value;
            _updateToggleLabel();
            _markDirty();
        });
        overrideBody.appendChild(descLabel);
        overrideBody.appendChild(descEl);

        // Per-field overrides
        const fieldLabel = _mk("label", { cls: "fbt-ce-slot-override-label",
            textContent: "Field overrides (face / hair / body)" });
        overrideBody.appendChild(fieldLabel);
        const fields = ["face", "hair", "body"];
        fields.forEach(field => {
            const row2 = _mk("div", { cls: "fbt-ce-slot-override-field-row" });
            row2.appendChild(_mk("span", { cls: "fbt-ce-slot-override-field-key", textContent: field }));
            const inp = _mk("input", {
                type: "text",
                cls: "fbt-ce-input fbt-ce-slot-override-field-inp",
                placeholder: `override ${field}…`,
                value: comp.appearance_overrides?.[key]?.[field] || "",
            });
            inp.addEventListener("input", () => {
                if (!comp.appearance_overrides) comp.appearance_overrides = {};
                if (!comp.appearance_overrides[key]) comp.appearance_overrides[key] = {};
                comp.appearance_overrides[key][field] = inp.value;
                _updateToggleLabel();
                _markDirty();
            });
            row2.appendChild(inp);
            overrideBody.appendChild(row2);
        });

        overrideWrap.appendChild(overrideToggle);
        overrideWrap.appendChild(overrideBody);

        row.appendChild(label);
        row.appendChild(subSel);
        row.appendChild(outfitSel);
        row.appendChild(removeBtn);
        card.appendChild(row);
        card.appendChild(infoEl);
        card.appendChild(outfitInfoEl);
        card.appendChild(conceptRow);
        card.appendChild(overrideWrap);
        container.appendChild(card);
    });
    if (!slots.length) {
        container.appendChild(_mk("div", { cls: "fbt-ce-empty", textContent: "No subject slots. Click subjects in the sidebar to assign." }));
    }
}

// Rewrites {OLD} -> {NEW} literal placeholder occurrences in free text using
// an old-key -> new-key map, longest-key-first so e.g. a 10th slot's 2-char
// key can't get partially clobbered by a shorter key's replacement running first.
function _rewritePlaceholders(text, oldToNew) {
    if (!text) return text;
    const entries = [...oldToNew.entries()].sort((a, b) => b[0].length - a[0].length);
    for (const [oldKey, newKey] of entries) {
        text = text.split(`{${oldKey}}`).join(`{${newKey}}`);
    }
    return text;
}

function _renumberSlots() {
    const comp = _S.composition;
    // Insertion order, not .sort() — slot keys are letters (A, B, ..., AA, ...)
    // and a lexicographic sort would misorder "AA" ahead of "B".
    const oldKeys = Object.keys(comp.subjects || {});
    const oldToNew = new Map(oldKeys.map((k, i) => [k, _slotLetterForIndex(i)]));
    const newSubjects = {};
    const newOutfits = {};
    const newOutfitIds = {};
    const newDescriptors = {};
    const newAppOverrides = {};
    oldKeys.forEach(k => {
        const newKey = oldToNew.get(k);
        newSubjects[newKey] = comp.subjects[k];
        if (comp.outfit_overrides?.[k])     newOutfits[newKey]      = comp.outfit_overrides[k];
        if (comp.outfit_ids?.[k])           newOutfitIds[newKey]    = comp.outfit_ids[k];
        if (comp.slot_descriptors?.[k])     newDescriptors[newKey]  = comp.slot_descriptors[k];
        if (comp.appearance_overrides?.[k]) newAppOverrides[newKey] = comp.appearance_overrides[k];
    });
    comp.subjects = newSubjects;
    comp.outfit_overrides = newOutfits;
    comp.outfit_ids = newOutfitIds;
    comp.slot_descriptors = newDescriptors;
    comp.appearance_overrides = newAppOverrides;
    // Update shot dialogue speaker keys, and rewrite any hand-typed {OLD}
    // placeholders in shot action/camera text so a removed early slot doesn't
    // silently leave a stale reference pointing at a different subject now
    // occupying that renumbered key.
    (comp.shots || []).forEach(shot => {
        if (shot.dialogue?.speaker && oldToNew.has(shot.dialogue.speaker)) {
            shot.dialogue.speaker = oldToNew.get(shot.dialogue.speaker);
        }
        shot.action = _rewritePlaceholders(shot.action, oldToNew);
        shot.camera = _rewritePlaceholders(shot.camera, oldToNew);
    });
    comp.scene_synopsis = _rewritePlaceholders(comp.scene_synopsis, oldToNew);
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

    // Camera — {A}/{B} completion enabled
    const cam = _mk("input", {
        cls: "fbt-ce-input",
        type: "text",
        placeholder: "Camera direction… (type { for slot reference)",
        value: shot.camera || "",
    });
    cam.addEventListener("input", () => { shot.camera = cam.value; _markDirty(); });
    _attachCompletion(cam);
    _attachLibberCompletion(cam);
    card.appendChild(_labeledRow("Camera", _withPresetPicker(cam, "camera")));

    // Action — {A}/{B} completion enabled
    const action = _mk("textarea", {
        cls: "fbt-ce-textarea fbt-ce-action",
        placeholder: "Describe what happens. Type { to insert a subject reference ({A}, {B} …).",
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
                label: `${k} — ${lib.subjects.find(s => s.id === _S.composition.subjects[k])?.name || _S.composition.subjects[k] || k}`,
              }))
            : [{ id: "A", label: "A" }];

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

        const paceSel = _sel(
            [
                { id: "normal", label: "Normal" },
                { id: "slow",   label: "Slow" },
                { id: "fast",   label: "Fast" },
            ],
            dlg?.speech_pace || "normal"
        );
        paceSel.className = "fbt-ce-select";
        paceSel.title = "Slow (~2 words/sec): adds “speaking slowly and deliberately” to the prompt\nNormal (~2.5 words/sec): default conversational pace\nFast (~3 words/sec): adds “speaking quickly” to the prompt\n\nAlso controls how much voice reference audio is used (trim_to).";
        paceSel.addEventListener("change", () => { if (shot.dialogue) shot.dialogue.speech_pace = paceSel.value; _markDirty(); });

        dlgFields.appendChild(_labeledRow("Speaker", spkSel));
        dlgFields.appendChild(_labeledRow("Language", langSel));
        dlgFields.appendChild(_labeledRow("Text", dlgText));
        dlgFields.appendChild(_labeledRow("Pace", paceSel, "Affects prompt phrasing and voice reference trim length"));
    };

    dlgCheck.addEventListener("change", () => {
        if (dlgCheck.checked) {
            shot.dialogue = { speaker: _slotKeys()[0] || "A", language: "English", text: "", speech_pace: _S.settings?.default_speech_pace ?? "normal" };
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
    card.appendChild(_labeledRow("Sound", _withPresetPicker(snd, "sound")));

    // Store refs so sidebar preset/LLM handlers can insert into the right fields
    card._camInput = cam;
    card._sndInput = snd;
    card._actInput = action;

    return card;
}

/**
 * Wrap a shot's camera/sound input with an "Insert preset" dropdown. Picking a preset writes its
 * text into that input (replacing the current value), like clicking a preset in the old sidebar.
 * Presets are managed in the Assets tab.
 */
function _withPresetPicker(inputEl, kind) {
    const presets = kind === "camera" ? lib.cameraPresets : lib.soundPresets;
    const sel = _mk("select", { cls: "fbt-ce-select fbt-ce-preset-sel",
        title: `Insert a ${kind} preset (manage presets in the Assets tab)` });
    sel.appendChild(_mk("option", { value: "", textContent: "Preset…" }));
    presets.forEach((p, i) => sel.appendChild(_mk("option", { value: String(i), textContent: p.name || p.id })));
    sel.disabled = !presets.length;
    sel.addEventListener("change", () => {
        const p = presets[parseInt(sel.value, 10)];
        sel.value = "";
        if (!p) return;
        inputEl.value = p.description || "";
        inputEl.dispatchEvent(new Event("input", { bubbles: true }));
        _toast(`${kind === "camera" ? "Camera" : "Sound"} preset applied`, "success");
    });
    return _mk("div", { cls: "fbt-ce-inline" }, [inputEl, sel]);
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
    _S.composition.shots.push(_newShot(_S.composition.shots));
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
    dupe.id = _nextShotId(_S.composition.shots);
    _S.composition.shots.splice(idx + 1, 0, dupe);
    _focusedShotIdx = idx + 1;
    _rebuildShots();
    _updateShotActive();
    _markDirty();
    const cards = _dom.shotsContainer?.querySelectorAll(".fbt-ce-shot-card");
    cards?.[_focusedShotIdx]?.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ── LoRAs section ─────────────────────────────────────────────────────────────

/** Basename without extension — what the user sees in the combobox. */
function _loraDisplayName(val) {
    return val ? val.replace(/\.[^.]+$/, "").split(/[\\/]/).pop() : "";
}

/**
 * Searchable combobox for LoRA selection.
 * Returns a wrapper div styled to fill the same flex slot as the old <select>.
 * The dropdown appends to document.body (position:fixed) so it's never clipped
 * by the sidebar's overflow.
 */
function _makeLoraCombobox(currentValue, onSelect) {
    let selectedValue = currentValue || "";
    let dropdownEl    = null;

    const wrap  = _mk("div",   { cls: "fbt-ce-lora-combo" });
    const input = _mk("input", {
        cls:          "fbt-ce-input fbt-ce-lora-combo-input",
        type:         "text",
        placeholder:  "— select LoRA —",
        value:        _loraDisplayName(selectedValue),
        autocomplete: "off",
    });
    input.setAttribute("spellcheck", "false");

    function _close() {
        if (dropdownEl) { dropdownEl.remove(); dropdownEl = null; }
        input.value = _loraDisplayName(selectedValue);
    }

    function _open(query) {
        if (dropdownEl) dropdownEl.remove();

        const q       = query.trim().toLowerCase();
        const matches = q
            ? _S.lorasList.filter(n => n.toLowerCase().includes(q))
            : _S.lorasList;

        const list = _mk("div", { cls: "fbt-ce-lora-dropdown" });

        // "— none —" item only shown when not filtering
        if (!q) {
            const noneItem = _mk("div", {
                cls: "fbt-ce-lora-dd-item" + (!selectedValue ? " fbt-ce-lora-dd-active" : ""),
                textContent: "— none —",
            });
            noneItem.addEventListener("mousedown", e => {
                e.preventDefault();
                selectedValue = "";
                _close();
                onSelect("");
            });
            list.appendChild(noneItem);
        }

        if (!matches.length) {
            list.appendChild(_mk("div", { cls: "fbt-ce-lora-dd-empty", textContent: "No matches" }));
        } else {
            matches.forEach(n => {
                const item = _mk("div", {
                    cls:   "fbt-ce-lora-dd-item" + (n === selectedValue ? " fbt-ce-lora-dd-active" : ""),
                    title: n,
                    textContent: _loraDisplayName(n),
                });
                item.addEventListener("mousedown", e => {
                    e.preventDefault();
                    selectedValue = n;
                    _close();
                    onSelect(n);
                });
                list.appendChild(item);
            });
        }

        const rect = input.getBoundingClientRect();
        Object.assign(list.style, {
            position: "fixed",
            left:     `${rect.left}px`,
            top:      `${rect.bottom + 2}px`,
            width:    `${Math.max(rect.width, 200)}px`,
            zIndex:   "9999",
        });
        document.body.appendChild(list);
        dropdownEl = list;

        const active = list.querySelector(".fbt-ce-lora-dd-active");
        if (active) active.scrollIntoView({ block: "nearest" });
    }

    input.addEventListener("focus", ()  => _open(""));
    input.addEventListener("input", ()  => _open(input.value));
    input.addEventListener("blur",  ()  => setTimeout(_close, 150));
    input.addEventListener("keydown", e => {
        if (!dropdownEl) return;
        const items = Array.from(dropdownEl.querySelectorAll(".fbt-ce-lora-dd-item"));
        let idx = items.findIndex(el => el.classList.contains("fbt-ce-lora-dd-active"));
        if (e.key === "ArrowDown") {
            e.preventDefault();
            items[idx]?.classList.remove("fbt-ce-lora-dd-active");
            items[Math.min(idx + 1, items.length - 1)]?.classList.add("fbt-ce-lora-dd-active");
            dropdownEl.querySelector(".fbt-ce-lora-dd-active")?.scrollIntoView({ block: "nearest" });
        } else if (e.key === "ArrowUp") {
            e.preventDefault();
            items[idx]?.classList.remove("fbt-ce-lora-dd-active");
            items[Math.max(idx - 1, 0)]?.classList.add("fbt-ce-lora-dd-active");
            dropdownEl.querySelector(".fbt-ce-lora-dd-active")?.scrollIntoView({ block: "nearest" });
        } else if (e.key === "Enter") {
            e.preventDefault();
            dropdownEl.querySelector(".fbt-ce-lora-dd-active")?.dispatchEvent(new MouseEvent("mousedown"));
        } else if (e.key === "Escape") {
            e.stopPropagation();
            _close();
        }
    });

    wrap.appendChild(input);
    return wrap;
}

function _rebuildLoras() {
    const c = _dom.lorasContainer;
    if (!c) return;
    c.innerHTML = "";
    const loras = _S.composition?.loras ?? [];

    loras.forEach((entry, i) => {
        const row = _mk("div", { cls: "fbt-ce-lora-row" });

        // Searchable name combobox
        const comboWrap = _makeLoraCombobox(entry.name || "", val => { entry.name = val; _markDirty(); });

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

        row.appendChild(comboWrap);
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

    // Header: ← Back to the list, and the composition's name (mirrors the Sources tab)
    _dom.editorTitle = _mk("h3", { cls: "fbt-ce-editor-title", textContent: "New composition" });
    editorWrap.appendChild(_mk("div", { cls: "fbt-ce-editor-header" }, [
        _mk("button", { cls: "fbt-ce-btn", textContent: "← Back", title: "Back to saved compositions", onclick: _onBack }),
        _dom.editorTitle,
    ]));

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
            if (_dom.editorTitle) _dom.editorTitle.textContent = _dom.nameInput.value || "New composition";
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
            const bg = lib.backgrounds.find(b => b.id === bgId);
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

        // "Include as <Subject N>" — background reference images become a visual reference slot
        const bgRefRow = _mk("div", { cls: "fbt-ce-info-row", style: { marginTop: "8px" } });
        _dom.bgAsRefCb = _mk("input", { type: "checkbox" });
        _dom.bgAsRefCb.id = "fbt-bg-as-ref";
        _dom.bgAsRefCb.addEventListener("change", () => {
            _S.composition.background_as_reference = _dom.bgAsRefCb.checked;
            _markDirty();
        });
        const bgRefLabel = _mk("label", {
            cls: "fbt-ce-info-label",
            htmlFor: "fbt-bg-as-ref",
            title: "Treat this background's reference images as a <Subject N> visual reference in H3 Ref2VA prompts. Use {BG} in shot action/camera fields to reference it.",
        });
        bgRefLabel.textContent = "Include as <Subject N>";
        bgRefRow.appendChild(_dom.bgAsRefCb);
        bgRefRow.appendChild(bgRefLabel);
        body.appendChild(bgRefRow);
    }));

    // Synopsis — concise scene overview used as the summary body in H3 prompts.
    // Supports {A}/{B}/… shorthand; expanded to <Subject N> labels at assemble time.
    form.appendChild(_editorSection("Synopsis", body => {
        _dom.synopsisArea = _mk("textarea", {
            cls: "fbt-ce-textarea",
            placeholder: "{A} eating a cookie in the café. {B} enters with a dog, which lunges toward the cookie.",
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
        await _refreshSavedComps();
        // Increment server counter so PromptCompositionLoader nodes re-execute
        compositionsApi.reloadCompositions().catch(() => {});
        _setStatus("Saved ✓");
    } catch (e) {
        _setStatus("Save failed: " + e.message, true);
    }
}

async function _onLoad(id) {
    if (_S.dirty && _S.composition?.id !== id) {
        if (!confirm("Discard unsaved changes?")) return;
    }
    try {
        const comp = await compositionsApi.getComposition(id);
        _S.composition = comp;
        _populateEditor();
        _markClean();
        _showView("editor");
    } catch (e) {
        _toast("Load failed: " + e.message, "error");
    }
}

async function _onDeleteComp(id, name) {
    if (!confirm(`Delete "${name}"?`)) return;
    try {
        await compositionsApi.deleteComposition(id);
        await _refreshSavedComps();
        _toast(`Deleted "${name}"`, "success");
    } catch (e) {
        _toast("Delete failed: " + e.message, "error");
    }
}

function _onNew() {
    if (_S.dirty && !confirm("Discard unsaved changes?")) return;
    _S.composition = _newComp();
    _populateEditor();
    _markClean();
    _showView("editor");
}

/** Back from the editor to the saved-compositions list (asks first when there are unsaved edits). */
function _onBack() {
    if (_S.dirty && !confirm("Discard unsaved changes?")) return;
    _markClean();
    _showView("list");
}

function _showView(view) {
    _S.view = view;
    if (_dom.listView)   _dom.listView.style.display   = view === "list" ? "" : "none";
    if (_dom.editorView) _dom.editorView.style.display = view === "editor" ? "" : "none";
    if (view === "list") _populateSavedList();
    else if (_dom.editorTitle) _dom.editorTitle.textContent = _S.composition?.name || "New composition";
}

// ── Saved compositions list (the default view) ─────────────────────────────────

function _fmtUpdated(iso) {
    if (!iso) return "";
    const d = new Date(iso);
    return Number.isNaN(d.getTime()) ? "" : d.toLocaleDateString();
}

function _bgName(id) {
    return id ? (lib.backgrounds.find(b => b.id === id)?.name || id) : "";
}

function _populateSavedList() {
    const list = _dom.savedList;
    const pagination = _dom.savedPagination;
    if (!list) return;

    const query = _S.savedQuery.toLowerCase().trim();
    const filtered = query
        ? _S.savedComps.filter(c =>
              (c.name || "").toLowerCase().includes(query) || (c.id || "").toLowerCase().includes(query))
        : _S.savedComps;
    const total      = filtered.length;
    const totalPages = Math.max(1, Math.ceil(total / SAVED_PAGE_SIZE));
    _S.savedPage     = Math.max(0, Math.min(_S.savedPage, totalPages - 1));
    const pageItems  = filtered.slice(_S.savedPage * SAVED_PAGE_SIZE, (_S.savedPage + 1) * SAVED_PAGE_SIZE);

    list.innerHTML = "";
    if (!pageItems.length) {
        list.appendChild(_mk("div", { cls: "fbt-ce-empty",
            textContent: query ? "No compositions match the search."
                               : "No saved compositions yet. Click + New to create one." }));
    }
    pageItems.forEach(comp => {
        const meta = [
            comp.model_type,
            comp.subject_count != null ? `${comp.subject_count} subj` : "",
            comp.shot_count != null ? `${comp.shot_count} shot${comp.shot_count === 1 ? "" : "s"}` : "",
            _bgName(comp.background),
            _fmtUpdated(comp.updated_at) ? `edited ${_fmtUpdated(comp.updated_at)}` : "",
        ].filter(Boolean).join(" · ");
        const card = _mk("div", { cls: "fbt-ce-list-card", title: "Open composition",
            onclick: () => _onLoad(comp.id) }, [
            _mk("div", { cls: "fbt-ce-list-card-top" }, [
                _mk("i", { cls: "pi pi-file-edit" }),
                _mk("span", { cls: "fbt-ce-list-card-name", textContent: comp.name || comp.id }),
                _mk("button", { cls: "fbt-ce-icon-btn fbt-ce-danger", title: "Delete", textContent: "✕",
                    onclick: e => { e.stopPropagation(); _onDeleteComp(comp.id, comp.name); } }),
            ]),
            _mk("div", { cls: "fbt-ce-list-card-meta", textContent: meta }),
        ]);
        list.appendChild(card);
    });

    if (pagination) {
        pagination.innerHTML = "";
        if (totalPages > 1) {
            const prev = _mk("button", { cls: "fbt-ce-pg-btn", textContent: "‹", title: "Previous page",
                onclick: () => { _S.savedPage--; _populateSavedList(); } });
            prev.disabled = _S.savedPage === 0;
            const next = _mk("button", { cls: "fbt-ce-pg-btn", textContent: "›", title: "Next page",
                onclick: () => { _S.savedPage++; _populateSavedList(); } });
            next.disabled = _S.savedPage >= totalPages - 1;
            pagination.append(prev,
                _mk("span", { cls: "fbt-ce-pg-info", textContent: `${_S.savedPage + 1} / ${totalPages}` }), next);
        }
    }
}

async function _refreshSavedComps() {
    try {
        _S.savedComps = (await compositionsApi.listCompositions()).compositions ?? [];
    } catch (_) { /* keep the previous list */ }
    _populateSavedList();
}

function _buildListView() {
    const view = _mk("div", { cls: "fbt-ce-list-view" });
    _dom.savedSearchInput = _mk("input", { cls: "fbt-ce-input fbt-ce-list-search", type: "text",
        placeholder: "Search compositions…" });
    _dom.savedSearchInput.addEventListener("input", () => {
        _S.savedQuery = _dom.savedSearchInput.value;
        _S.savedPage = 0;
        _populateSavedList();
    });
    const newBtn = _mk("button", { cls: "fbt-ce-btn fbt-ce-btn-primary", textContent: "+ New", onclick: _onNew });
    const refreshBtn = _mk("button", { cls: "fbt-ce-icon-btn", textContent: "↺", title: "Refresh",
        onclick: async () => { await _loadResources(); await _refreshSavedComps(); } });
    view.appendChild(_mk("div", { cls: "fbt-ce-list-toolbar" }, [_dom.savedSearchInput, newBtn, refreshBtn]));
    _dom.savedList       = _mk("div", { cls: "fbt-ce-list-cards" });
    _dom.savedPagination = _mk("div", { cls: "fbt-ce-saved-pagination" });
    view.append(_dom.savedList, _dom.savedPagination);
    return view;
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

// ── Populate editor from state ─────────────────────────────────────────────────

function _populateEditor() {
    const comp = _S.composition;
    if (!comp) return;
    if (_dom.nameInput) _dom.nameInput.value = comp.name || "";
    if (_dom.editorTitle) _dom.editorTitle.textContent = comp.name || "New composition";
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
    if (_dom.bgAsRefCb) _dom.bgAsRefCb.checked = !!(comp.background_as_reference);
}

// ── Panel construction ─────────────────────────────────────────────────────────

function _buildPanel(el) {
    el.innerHTML = "";
    _dom.panel = _mk("div", { cls: "fbt-ce-panel" });
    el.appendChild(_dom.panel);

    const body = _mk("div", { cls: "fbt-ce-body" });
    _dom.listView = _buildListView();
    _dom.editorView = _mk("div", { cls: "fbt-ce-editor-view" });
    _buildEditor(_dom.editorView);
    body.append(_dom.listView, _dom.editorView);
    _dom.panel.appendChild(body);
    _showView("list");

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
        _dom.fillAddSubject?.();
        await _refreshSavedComps();
        if (_S.view === "editor") { _rebuildSlots(); _rebuildShots(); }
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
    lib.getCompositionName = () => _S.composition?.name || "";
    document.addEventListener(LIBRARY_CHANGED, _onLibraryChanged);
    _buildPanel(el);
    await _loadResources();
    _dom.fillAddSubject?.();
    _populateSavedList();
    _populateEditor();
    _markClean();

    // Keep _S.settings fresh when the global Settings tab changes a value.
    document.addEventListener("fbt:settings-changed", (e) => {
        _S.settings = { ...(_S.settings ?? {}), ...e.detail };
    });
}
