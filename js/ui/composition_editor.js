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
import { openBackgroundEditor } from "./background_editor.js";
import { openOutfitEditor } from "./outfit_editor.js";

// ── Constants ──────────────────────────────────────────────────────────────────

const SAVED_PAGE_SIZE   = 10;
const SUBJECT_PAGE_SIZE = 10;
const BG_PAGE_SIZE      = 10;

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
    subjectPage:    0,
    bgPage:         0,
    dirty:          false,
    shotSeq:        0,
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

function _newShot() {
    return {
        id: `shot_${++_S.shotSeq}`,
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
            if (_dom.delimInput)              _dom.delimInput.value            = st.libber_delimiter           ?? "%";
            if (_dom.settingsPaceSel)         _dom.settingsPaceSel.value       = st.default_speech_pace        ?? "normal";
            if (_dom.settingsNoiseRemovalCb)  _dom.settingsNoiseRemovalCb.checked  = !!st.default_audio_noise_removal;
            if (_dom.settingsNormalizeLufsCb) _dom.settingsNormalizeLufsCb.checked = st.default_audio_normalize_lufs !== false;
            if (_dom.settingsTargetLufsInp)   _dom.settingsTargetLufsInp.value = st.default_audio_target_lufs  ?? -14.0;
            if (_dom.settingsMelbandInp)      _dom.settingsMelbandInp.value    = st.melband_model_path         ?? "";
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

function _buildSidebarSection(title, listEl) {
    const header = _mk("div", { cls: "fbt-ce-sb-header", textContent: title });
    const body   = _mk("div", { cls: "fbt-ce-sb-body" }, [listEl]);
    _sectionToggle(header, body);
    return _mk("div", { cls: "fbt-ce-sb-section" }, [header, body]);
}

function _populateSavedList() {
    const list       = _dom.savedList;
    const pagination = _dom.savedPagination;
    if (!list) return;

    const query    = _S.savedQuery.toLowerCase().trim();
    const filtered = query
        ? _S.savedComps.filter(c =>
              (c.name || "").toLowerCase().includes(query) ||
              (c.id   || "").toLowerCase().includes(query))
        : _S.savedComps;

    const total      = filtered.length;
    const totalPages = Math.max(1, Math.ceil(total / SAVED_PAGE_SIZE));
    _S.savedPage     = Math.max(0, Math.min(_S.savedPage, totalPages - 1));

    const start     = _S.savedPage * SAVED_PAGE_SIZE;
    const pageItems = filtered.slice(start, start + SAVED_PAGE_SIZE);

    list.innerHTML = "";
    if (!filtered.length) {
        list.appendChild(_mk("div", {
            cls: "fbt-ce-empty",
            textContent: query ? "No matches" : "No saved compositions",
        }));
    } else {
        const activeId = _S.composition?.id || "";
        pageItems.forEach(comp => {
            const isActive = comp.id && comp.id === activeId;
            const row = _mk("div", {
                cls: "fbt-ce-sb-item fbt-ce-clickable" + (isActive ? " fbt-ce-sb-item-active" : ""),
                title: "Load composition",
                onclick: () => _onLoad(comp.id),
            });
            row.appendChild(_mk("span", { cls: "fbt-ce-sb-name", textContent: comp.name || comp.id }));
            const actions = _mk("span", { cls: "fbt-ce-sb-actions" });
            actions.appendChild(_mk("button", {
                cls: "fbt-ce-icon-btn fbt-ce-danger", title: "Delete",
                textContent: "✕",
                onclick: (e) => { e.stopPropagation(); _onDeleteComp(comp.id, comp.name); },
            }));
            row.appendChild(actions);
            list.appendChild(row);
        });
    }

    // Pagination controls — only shown when more than one page exists
    if (pagination) {
        pagination.innerHTML = "";
        if (totalPages > 1) {
            const prevBtn = _mk("button", {
                cls: "fbt-ce-pg-btn",
                textContent: "‹",
                title: "Previous page",
                onclick: () => { _S.savedPage--; _populateSavedList(); },
            });
            prevBtn.disabled = _S.savedPage === 0;

            const info = _mk("span", {
                cls: "fbt-ce-pg-info",
                textContent: `${_S.savedPage + 1} / ${totalPages}`,
            });

            const nextBtn = _mk("button", {
                cls: "fbt-ce-pg-btn",
                textContent: "›",
                title: "Next page",
                onclick: () => { _S.savedPage++; _populateSavedList(); },
            });
            nextBtn.disabled = _S.savedPage >= totalPages - 1;

            pagination.appendChild(prevBtn);
            pagination.appendChild(info);
            pagination.appendChild(nextBtn);
        }
    }
}

function _populateSubjectList() {
    const list       = _dom.subjectList;
    const pagination = _dom.subjectPagination;
    if (!list) return;
    list.innerHTML = "";

    // "New Subject" quick-add button — always shown, not paginated
    list.appendChild(_mk("div", {
        cls: "fbt-ce-sb-item fbt-ce-sb-new",
        textContent: "+ New Subject…",
        onclick: () => _showNewSubjectForm(),
    }));

    if (!lib.subjects.length) {
        list.appendChild(_mk("div", { cls: "fbt-ce-empty", textContent: "No subjects defined" }));
        if (pagination) pagination.innerHTML = "";
        return;
    }

    const assignedIds = new Set(Object.values(_S.composition?.subjects || {}));
    const total       = lib.subjects.length;
    const totalPages  = Math.max(1, Math.ceil(total / SUBJECT_PAGE_SIZE));
    _S.subjectPage    = Math.max(0, Math.min(_S.subjectPage, totalPages - 1));
    const start       = _S.subjectPage * SUBJECT_PAGE_SIZE;
    const pageItems   = lib.subjects.slice(start, start + SUBJECT_PAGE_SIZE);

    pageItems.forEach(s => {
        const isActive = assignedIds.has(s.id);
        const item = _mk("div", {
            cls: "fbt-ce-sb-item fbt-ce-clickable" + (isActive ? " fbt-ce-sb-item-active" : ""),
        });
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

    if (pagination) {
        pagination.innerHTML = "";
        if (totalPages > 1) {
            const prevBtn = _mk("button", {
                cls: "fbt-ce-pg-btn", textContent: "‹", title: "Previous page",
                onclick: () => { _S.subjectPage--; _populateSubjectList(); },
            });
            prevBtn.disabled = _S.subjectPage === 0;
            const info = _mk("span", {
                cls: "fbt-ce-pg-info",
                textContent: `${_S.subjectPage + 1} / ${totalPages}`,
            });
            const nextBtn = _mk("button", {
                cls: "fbt-ce-pg-btn", textContent: "›", title: "Next page",
                onclick: () => { _S.subjectPage++; _populateSubjectList(); },
            });
            nextBtn.disabled = _S.subjectPage >= totalPages - 1;
            pagination.appendChild(prevBtn);
            pagination.appendChild(info);
            pagination.appendChild(nextBtn);
        }
    }
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
                lib.subjects = res.subjects ?? [];
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

function _syncBgDropdown(selectId) {
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

/** React to asset edits made in a modal here or in the Assets tab. */
function _onLibraryChanged(e) {
    const { kind, deletedId } = e.detail ?? {};
    if (kind === "backgrounds") {
        if (deletedId && _S.composition?.background === deletedId) {
            _S.composition.background = "";
            _markDirty();
        }
        _syncBgDropdown(_S.composition?.background || "");
    } else if (kind === "outfits") {
        _rebuildOutfitList();
        _rebuildSlots();
    } else if (kind === "subjects") {
        _populateSubjectList();
        _rebuildSlots();
    } else if (kind === "cameraPresets") {
        _rebuildPresetList("camera");
    } else if (kind === "soundPresets") {
        _rebuildPresetList("sound");
    }
}

function _populateBgList() {
    const list       = _dom.bgList;
    const pagination = _dom.bgPagination;
    if (!list) return;
    list.innerHTML = "";

    // "New Background" quick-add button — always shown, not paginated
    list.appendChild(_mk("div", {
        cls: "fbt-ce-sb-item fbt-ce-sb-new",
        textContent: "+ New Background…",
        onclick: () => openBackgroundEditor(null),
    }));

    if (!lib.backgrounds.length) {
        list.appendChild(_mk("div", { cls: "fbt-ce-empty", textContent: "No backgrounds defined" }));
        if (pagination) pagination.innerHTML = "";
        return;
    }

    const activeBgId = _S.composition?.background || "";
    const total      = lib.backgrounds.length;
    const totalPages = Math.max(1, Math.ceil(total / BG_PAGE_SIZE));
    _S.bgPage        = Math.max(0, Math.min(_S.bgPage, totalPages - 1));
    const start      = _S.bgPage * BG_PAGE_SIZE;
    const pageItems  = lib.backgrounds.slice(start, start + BG_PAGE_SIZE);

    pageItems.forEach(b => {
        const isActive = b.id === activeBgId;
        const row = _mk("div", { cls: "fbt-ce-sb-item-row" });

        const nameEl = _mk("div", {
            cls: "fbt-ce-sb-item fbt-ce-clickable fbt-ce-sb-item-flex" + (isActive ? " fbt-ce-sb-item-active" : ""),
            title: b.description || "",
            onclick: () => _assignBg(b.id),
        });
        nameEl.appendChild(_mk("span", { cls: "fbt-ce-sb-name", textContent: b.name || b.id }));

        const editBtn = _mk("button", {
            cls: "fbt-ce-sb-edit-btn",
            textContent: "✎",
            title: "Edit background",
        });
        editBtn.addEventListener("click", e => { e.stopPropagation(); openBackgroundEditor(b); });

        row.appendChild(nameEl);
        row.appendChild(editBtn);
        list.appendChild(row);
    });

    if (pagination) {
        pagination.innerHTML = "";
        if (totalPages > 1) {
            const prevBtn = _mk("button", {
                cls: "fbt-ce-pg-btn", textContent: "‹", title: "Previous page",
                onclick: () => { _S.bgPage--; _populateBgList(); },
            });
            prevBtn.disabled = _S.bgPage === 0;
            const info = _mk("span", {
                cls: "fbt-ce-pg-info",
                textContent: `${_S.bgPage + 1} / ${totalPages}`,
            });
            const nextBtn = _mk("button", {
                cls: "fbt-ce-pg-btn", textContent: "›", title: "Next page",
                onclick: () => { _S.bgPage++; _populateBgList(); },
            });
            nextBtn.disabled = _S.bgPage >= totalPages - 1;
            pagination.appendChild(prevBtn);
            pagination.appendChild(info);
            pagination.appendChild(nextBtn);
        }
    }
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
    _rebuildOutfitList();
}

// ── Camera / Sound preset sidebar sections ────────────────────────────────────

function _rebuildPresetList(kind) {
    const listEl  = kind === "camera" ? _dom.camList : _dom.sndList;
    const presets = kind === "camera" ? lib.cameraPresets : lib.soundPresets;
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
                    if (kind === "camera") lib.cameraPresets = r.camera_presets ?? [];
                    else                   lib.soundPresets  = r.sound_presets  ?? [];
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
                if (kind === "camera") lib.cameraPresets = r.camera_presets ?? [];
                else                   lib.soundPresets  = r.sound_presets  ?? [];
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

// ── Outfit media helpers ───────────────────────────────────────────────────────

// ── Outfit Registry sidebar section ───────────────────────────────────────────

function _outfitIdFromRow(row) { return row?.dataset.outfitId ?? ""; }

function _rebuildOutfitList() {
    const list = _dom.outfitList;
    if (!list) return;
    list.innerHTML = "";
    const entries = Object.entries(lib.outfits).sort(([, a], [, b]) =>
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
            onclick: () => openOutfitEditor(id) });
        const delBtn = _mk("button", { cls: "fbt-ce-btn fbt-ce-btn-sm fbt-ce-btn-danger", textContent: "✕",
            onclick: async () => {
                if (!confirm(`Delete outfit '${id}'?`)) return;
                try {
                    await compositionsApi.deleteOutfit(id);
                    delete lib.outfits[id];
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

function _buildOutfitsSection(parent) {
    const body = _mk("div", { cls: "fbt-ce-sb-body" });
    _dom.outfitList = _mk("div", { cls: "fbt-ce-sb-list" });

    const addBtn = _mk("button", { cls: "fbt-ce-btn fbt-ce-btn-sm",
        textContent: "+ New Outfit",
        onclick: () => openOutfitEditor(null) });
    const reloadBtn = _mk("button", { cls: "fbt-ce-btn fbt-ce-btn-sm",
        textContent: "↺",
        title: "Reload outfit registry from disk",
        onclick: async () => {
            try {
                const r = await compositionsApi.getOutfitRegistry();
                lib.outfits = r?.outfits ?? {};
                _rebuildOutfitList();
            } catch (e) { console.error("Outfit reload error", e); }
        }});

    const ctrl = _mk("div", { cls: "fbt-ce-outfit-section-ctrl" }, [addBtn, reloadBtn]);
    body.appendChild(ctrl);
    body.appendChild(_dom.outfitList);
    _rebuildOutfitList();
    parent.appendChild(_buildSidebarSection("Outfits", body));
}

// ── LLM status (the Compose sidebar no longer has an LLM section) ──────────────

function _llmSyncBadge() {
    document.body.classList.toggle("fbt-llm-loaded", !!lib.llmLoaded);
}

function _buildSettingsSection(parent) {
    const body = _mk("div", { cls: "fbt-ce-sb-body" });
    const s    = _S.settings ?? {};

    const _saveSettings = async () => {
        try { await compositionsApi.saveSettings(_S.settings); } catch (_) {}
    };

    // ── Libber delimiter ─────────────────────────────────────────────────────
    const delimRow = _mk("div", { cls: "fbt-ce-settings-row" });
    delimRow.appendChild(_mk("span", { cls: "fbt-ce-settings-label", textContent: "Libber delimiter" }));
    _dom.delimInput = _mk("input", {
        cls: "fbt-ce-delimiter-input",
        type: "text",
        maxLength: 1,
        value: s.libber_delimiter ?? "%",
        title: "Single character used to wrap libber keys (e.g. %key%)",
    });
    _dom.delimInput.addEventListener("change", async () => {
        const d = _dom.delimInput.value;
        if (!d.length) { _dom.delimInput.value = _S.settings.libber_delimiter; return; }
        _S.settings.libber_delimiter = d;
        await _saveSettings();
        _rebuildLibbers();
    });
    delimRow.appendChild(_dom.delimInput);
    body.appendChild(delimRow);

    // ── Default speech pace ──────────────────────────────────────────────────
    const paceRow = _mk("div", { cls: "fbt-ce-settings-row" });
    paceRow.appendChild(_mk("span", {
        cls: "fbt-ce-settings-label",
        textContent: "Default speech pace",
        title: "Pre-selected pace for new shots' dialogue. Affects how much voice audio is trimmed.",
    }));
    _dom.settingsPaceSel = document.createElement("select");
    _dom.settingsPaceSel.className = "fbt-ce-select fbt-ce-settings-sel";
    [
        { id: "slow",   label: "Slow (~2 words/sec)" },
        { id: "normal", label: "Normal (~2.5 words/sec)" },
        { id: "fast",   label: "Fast (~3 words/sec)" },
    ].forEach(({ id, label }) => {
        const o = document.createElement("option");
        o.value = id; o.textContent = label;
        if (id === (s.default_speech_pace ?? "normal")) o.selected = true;
        _dom.settingsPaceSel.appendChild(o);
    });
    _dom.settingsPaceSel.addEventListener("change", async () => {
        _S.settings.default_speech_pace = _dom.settingsPaceSel.value;
        await _saveSettings();
    });
    paceRow.appendChild(_dom.settingsPaceSel);
    body.appendChild(paceRow);

    // ── Default audio processing ─────────────────────────────────────────────
    body.appendChild(_mk("div", { cls: "fbt-ce-settings-group-label", textContent: "Default audio processing" }));

    const _cb = (label, domKey, settingsKey, hint) => {
        const row = _mk("div", { cls: "fbt-ce-settings-row" });
        row.appendChild(_mk("span", { cls: "fbt-ce-settings-label", textContent: label, title: hint || "" }));
        const cb = _mk("input", { type: "checkbox" });
        cb.checked = !!(s[settingsKey] ?? false);
        cb.addEventListener("change", async () => {
            _S.settings[settingsKey] = cb.checked;
            await _saveSettings();
        });
        _dom[domKey] = cb;
        row.appendChild(cb);
        return row;
    };

    body.appendChild(_cb("Noise removal", "settingsNoiseRemovalCb", "default_audio_noise_removal",
        "Spectral subtraction applied to new bundles by default"));
    body.appendChild(_cb("LUFS normalize", "settingsNormalizeLufsCb", "default_audio_normalize_lufs",
        "Normalize to target loudness for new bundles by default"));

    const lufsRow = _mk("div", { cls: "fbt-ce-settings-row" });
    lufsRow.appendChild(_mk("span", { cls: "fbt-ce-settings-label", textContent: "Target LUFS" }));
    _dom.settingsTargetLufsInp = _mk("input", {
        cls: "fbt-ce-input fbt-ce-settings-lufs",
        type: "number", min: "-36", max: "-6", step: "0.5",
        value: s.default_audio_target_lufs ?? -14.0,
        title: "Target integrated loudness in LUFS (−14 = streaming standard)",
    });
    _dom.settingsTargetLufsInp.addEventListener("change", async () => {
        const v = parseFloat(_dom.settingsTargetLufsInp.value);
        if (!isNaN(v)) {
            _S.settings.default_audio_target_lufs = Math.max(-36, Math.min(-6, v));
            _dom.settingsTargetLufsInp.value = _S.settings.default_audio_target_lufs;
            await _saveSettings();
        }
    });
    lufsRow.appendChild(_dom.settingsTargetLufsInp);
    lufsRow.appendChild(_mk("span", { cls: "fbt-be-proc-unit", textContent: "LUFS" }));
    body.appendChild(lufsRow);

    // ── Melband model path ───────────────────────────────────────────────────
    body.appendChild(_mk("div", { cls: "fbt-ce-settings-group-label", textContent: "Vocal isolation" }));
    const mbRow = _mk("div", { cls: "fbt-ce-settings-row fbt-ce-settings-row--wide" });
    mbRow.appendChild(_mk("span", {
        cls: "fbt-ce-settings-label",
        textContent: "MelBand model path",
        title: "Path to a MelBand Roformer .safetensors checkpoint — used for vocal isolation (future feature). Kijai/MelBandRoFormer_comfy on HuggingFace has fp16 (456 MB) and fp32 (913 MB) builds.",
    }));
    _dom.settingsMelbandInp = _mk("input", {
        cls: "fbt-ce-input",
        type: "text",
        placeholder: "MelBandRoformer_fp16.safetensors",
        value: s.melband_model_path ?? "",
        title: "Filename or path to a MelBand Roformer .safetensors checkpoint (Kijai/MelBandRoFormer_comfy — fp16 or fp32)",
    });
    _dom.settingsMelbandInp.addEventListener("change", async () => {
        _S.settings.melband_model_path = _dom.settingsMelbandInp.value.trim();
        await _saveSettings();
    });
    mbRow.appendChild(_dom.settingsMelbandInp);
    body.appendChild(mbRow);

    parent.appendChild(_buildSidebarSection("⚙ Settings", body));
}

function _buildSavedSection(parent) {
    const body = _mk("div", { cls: "fbt-ce-sb-body" });

    // Search row
    const searchRow = _mk("div", { cls: "fbt-ce-saved-search-row" });
    _dom.savedSearchInput = _mk("input", {
        cls: "fbt-ce-input fbt-ce-saved-search",
        type: "text",
        placeholder: "Search…",
    });
    _dom.savedSearchInput.addEventListener("input", () => {
        _S.savedQuery = _dom.savedSearchInput.value;
        _S.savedPage  = 0;
        _populateSavedList();
    });
    const clearBtn = _mk("button", {
        cls: "fbt-ce-icon-btn fbt-ce-saved-clear",
        textContent: "✕",
        title: "Clear search",
        onclick: () => {
            _dom.savedSearchInput.value = "";
            _S.savedQuery = "";
            _S.savedPage  = 0;
            _populateSavedList();
            _dom.savedSearchInput.focus();
        },
    });
    searchRow.appendChild(_dom.savedSearchInput);
    searchRow.appendChild(clearBtn);
    body.appendChild(searchRow);

    _dom.savedList       = _mk("div", { cls: "fbt-ce-sb-list" });
    _dom.savedPagination = _mk("div", { cls: "fbt-ce-saved-pagination" });
    body.appendChild(_dom.savedList);
    body.appendChild(_dom.savedPagination);

    parent.appendChild(_buildSidebarSection("Saved Compositions", body));
}

function _buildSidebar(parent) {
    const sidebar = _mk("div", { cls: "fbt-ce-sidebar" });

    _dom.subjectList       = _mk("div", { cls: "fbt-ce-sb-list" });
    _dom.subjectPagination = _mk("div", { cls: "fbt-ce-saved-pagination" });
    const subjectBody = _mk("div", { cls: "fbt-ce-sb-body" });
    subjectBody.appendChild(_dom.subjectList);
    subjectBody.appendChild(_dom.subjectPagination);

    _dom.bgList       = _mk("div", { cls: "fbt-ce-sb-list" });
    _dom.bgPagination = _mk("div", { cls: "fbt-ce-saved-pagination" });
    const bgBody = _mk("div", { cls: "fbt-ce-sb-body" });
    bgBody.appendChild(_dom.bgList);
    bgBody.appendChild(_dom.bgPagination);

    _buildSavedSection(sidebar);
    sidebar.appendChild(_buildSidebarSection("Subjects (click to assign)", subjectBody));
    sidebar.appendChild(_buildSidebarSection("Backgrounds (click to assign)", bgBody));
    _buildPresetSection(sidebar, "camera");
    _buildPresetSection(sidebar, "sound");
    _buildOutfitsSection(sidebar);

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
            if (!key) return _toast("Could not allocate a new slot", "warn");
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
    card.appendChild(_labeledRow("Camera", cam));

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
        _populateSavedList(); // refresh active indicator
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
    const name = lib.subjects.find(s => s.id === subjectId)?.name || subjectId;
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
    const name = lib.backgrounds.find(b => b.id === bgId)?.name || bgId;
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
    if (_dom.bgAsRefCb) _dom.bgAsRefCb.checked = !!(comp.background_as_reference);
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
    lib.getCompositionName = () => _S.composition?.name || "";
    document.addEventListener(LIBRARY_CHANGED, _onLibraryChanged);
    _buildPanel(el);
    await _loadResources();
    _refreshSidebar();
    _populateEditor();
    _markClean();

    // Keep _S.settings fresh when the global Settings tab changes a value.
    document.addEventListener("fbt:settings-changed", (e) => {
        _S.settings = { ...(_S.settings ?? {}), ...e.detail };
    });
}
