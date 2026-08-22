/**
 * Source Profile Editor — sidebar panel.
 *
 * Catalog browser for media-first subject profiles. Each profile points at
 * one source video or image and annotates the identifiable subjects inside it
 * (people, objects, locations, animals, soundscapes).
 *
 * Supports manual annotation and LLM-assisted focused-pass analysis.
 */

import { sourceProfilesApi } from "../api/source_profiles.js";
import { bundlesApi }        from "../api/bundles.js";

// ── Constants ──────────────────────────────────────────────────────────────────

const ENTITY_TYPES  = ["person", "object", "location", "animal", "soundscape"];
const MEDIA_TYPES   = ["video", "image"];
const MEDIA_DIRS    = ["input", "output"];
const PASS_TYPES    = ["people", "setting", "soundscape", "objects", "animals", "custom"];

const ENTITY_ICONS = {
    person:     "pi pi-user",
    object:     "pi pi-box",
    location:   "pi pi-map-marker",
    animal:     "pi pi-star",
    soundscape: "pi pi-volume-up",
};

const PASS_LABELS = {
    people:     "People",
    setting:    "Setting",
    soundscape: "Soundscape",
    objects:    "Objects / Props",
    animals:    "Animals",
    custom:     "Custom",
};

const CAPTIONER_TYPES = ["qwen_vl", "qwen_omni", "gemini_flash"];

const CLIP_COLORS = ["#3b82f6","#10b981","#f59e0b","#ef4444","#8b5cf6","#06b6d4","#f97316","#ec4899"];

// ── Module state ───────────────────────────────────────────────────────────────

const _S = {
    profiles:       [],     // [{id, name, media_filename, media_dir, media_type, subjects:[]}]
    mediaVideos:    [],     // filenames from input dir
    mediaImages:    [],
    mediaVideosOut: [],
    mediaImagesOut: [],
    filterText:     "",
    selected:       null,   // profile id currently open in detail view
    editingSubject: null,   // {idx, data} or null (new = idx === -1)
    analyzeOpen:    false,
    analyzePassType: "people",
    analyzePromptOverride: "",
    analyzeCaptioner: "qwen_vl",
    analyzeRunning:  false,
    analyzeCandidates: [],
    analyzeHistory:  [],
    historyOpen:     false,
};

const _dom = {};

// ── Helpers ────────────────────────────────────────────────────────────────────

function _mk(tag, props = {}, children = []) {
    const el = document.createElement(tag);
    Object.entries(props).forEach(([k, v]) => {
        if (k === "cls")   el.className = v;
        else if (k === "style")  Object.assign(el.style, v);
        else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
        else el[k] = v;
    });
    children.forEach(c => c && el.appendChild(typeof c === "string" ? document.createTextNode(c) : c));
    return el;
}

function _toast(msg, severity = "info") {
    try { window._fbtApp?.extensionManager?.toast?.add({ severity, summary: msg, life: 2500 }); }
    catch (_) {}
}

function _slugify(str) {
    return (str || "").toLowerCase().replace(/\s+/g, "_").replace(/[^\w]/g, "").slice(0, 48);
}

function _genId(name) {
    const base = _slugify(name);
    const ts   = Date.now().toString(36).slice(-4);
    return base ? `sp_${base}_${ts}` : `sp_${ts}`;
}

function _genSubjId(label) {
    const base = _slugify(label);
    const ts   = Date.now().toString(36).slice(-4);
    return base ? `subj_${base}_${ts}` : `subj_${ts}`;
}

function _profile() {
    return _S.profiles.find(p => p.id === _S.selected) || null;
}

function _allMedia(type, dir) {
    if (type === "video") return dir === "output" ? _S.mediaVideosOut : _S.mediaVideos;
    return dir === "output" ? _S.mediaImagesOut : _S.mediaImages;
}

// ── Data load ─────────────────────────────────────────────────────────────────

async function _loadAll() {
    const [pr, vIn, vOut, iIn, iOut] = await Promise.allSettled([
        sourceProfilesApi.list(),
        bundlesApi.listMedia("video", false, "input"),
        bundlesApi.listMedia("video", false, "output"),
        bundlesApi.listMedia("image", false, "input"),
        bundlesApi.listMedia("image", false, "output"),
    ]);
    _S.profiles       = pr.value?.profiles    ?? [];
    _S.mediaVideos    = vIn.value?.files      ?? [];
    _S.mediaVideosOut = vOut.value?.files     ?? [];
    _S.mediaImages    = iIn.value?.files      ?? [];
    _S.mediaImagesOut = iOut.value?.files     ?? [];
}

async function _loadHistory(profileId) {
    try {
        const res = await sourceProfilesApi.analysisHistory(profileId);
        _S.analyzeHistory = res.entries ?? [];
    } catch (_) {
        _S.analyzeHistory = [];
    }
}

// ── CSS injection ─────────────────────────────────────────────────────────────

const _CSS = `
.spe-panel { display:flex; flex-direction:column; height:100%; font-size:13px; }

/* Toolbar */
.spe-toolbar { display:flex; gap:6px; align-items:center; padding:8px 10px;
    border-bottom:1px solid var(--p-surface-border,#444); flex-shrink:0; }
.spe-toolbar input { flex:1; padding:4px 8px; border-radius:4px;
    border:1px solid var(--p-surface-border,#555);
    background:var(--p-surface-ground,#1a1a1a);
    color:var(--p-text-color,#eee); font-size:12px; }
.spe-btn { padding:4px 10px; border-radius:4px; border:none; cursor:pointer; font-size:12px;
    background:var(--p-surface-d,#333); color:var(--p-text-color,#eee); }
.spe-btn:hover { background:var(--p-surface-hover,#444); }
.spe-btn.primary { background:var(--p-primary-color,#58a6ff); color:#000; }
.spe-btn.primary:hover { opacity:.88; }
.spe-btn.danger  { background:#f85149; color:#fff; }
.spe-btn.danger:hover  { opacity:.88; }
.spe-btn.sm { padding:2px 8px; font-size:11px; }
.spe-btn.ghost { background:transparent; border:1px solid var(--p-surface-border,#555); }

/* List */
.spe-list { flex:1; overflow-y:auto; padding:6px 8px; }
.spe-profile-card { border:1px solid var(--p-surface-border,#444);
    border-radius:6px; margin-bottom:6px; overflow:hidden;
    background:var(--p-surface-card,#1e1e1e); cursor:pointer; }
.spe-profile-card:hover { border-color:var(--p-primary-color,#58a6ff); }
.spe-profile-card.active { border-color:var(--p-primary-color,#58a6ff); }
.spe-card-header { display:flex; align-items:center; gap:8px; padding:8px 10px;
    background:var(--p-surface-section,#252525); }
.spe-card-name { font-weight:600; flex:1; }
.spe-card-meta { font-size:11px; color:var(--p-text-muted-color,#888); }
.spe-card-body { padding:8px 10px; }
.spe-subject-chip { display:inline-flex; align-items:center; gap:4px;
    background:var(--p-surface-d,#2d2d2d); border-radius:12px;
    padding:2px 8px; margin:2px; font-size:11px; }

/* Detail panel */
.spe-detail { flex:1; overflow-y:auto; display:flex; flex-direction:column; }
.spe-detail-header { display:flex; align-items:center; gap:8px; padding:8px 10px;
    border-bottom:1px solid var(--p-surface-border,#444); flex-shrink:0;
    background:var(--p-surface-section,#252525); }
.spe-detail-header h3 { flex:1; margin:0; font-size:14px; }
.spe-detail-body { flex:1; overflow-y:auto; padding:10px; }

/* Media preview */
.spe-media-wrap { background:#000; border-radius:6px; overflow:hidden; margin-bottom:10px;
    max-height:160px; display:flex; align-items:center; justify-content:center; }
.spe-media-wrap video, .spe-media-wrap img { max-width:100%; max-height:160px; object-fit:contain; }

/* Form */
.spe-form-row { margin-bottom:8px; }
.spe-form-row label { display:block; font-size:11px; color:var(--p-text-muted-color,#888);
    margin-bottom:3px; text-transform:uppercase; letter-spacing:.04em; }
.spe-form-row input, .spe-form-row select, .spe-form-row textarea {
    width:100%; padding:5px 8px; border-radius:4px;
    border:1px solid var(--p-surface-border,#555);
    background:var(--p-surface-ground,#1a1a1a);
    color:var(--p-text-color,#eee); font-size:12px; box-sizing:border-box; }
.spe-form-row textarea { min-height:54px; resize:vertical; }
.spe-form-actions { display:flex; gap:6px; justify-content:flex-end; margin-top:10px; }

/* Subject list */
.spe-subj-row { display:flex; align-items:flex-start; gap:6px; padding:6px 8px;
    border:1px solid var(--p-surface-border,#444); border-radius:5px; margin-bottom:5px;
    background:var(--p-surface-ground,#1a1a1a); }
.spe-subj-icon { font-size:14px; padding-top:1px; color:var(--p-text-muted-color,#888); flex-shrink:0; }
.spe-subj-body { flex:1; min-width:0; }
.spe-subj-label { font-weight:600; font-size:12px; }
.spe-subj-role  { font-size:11px; color:var(--p-text-muted-color,#888); white-space:nowrap;
    overflow:hidden; text-overflow:ellipsis; }
.spe-subj-actions { display:flex; gap:4px; flex-shrink:0; }

/* Analyze panel */
.spe-analyze { border-top:1px solid var(--p-surface-border,#444); padding:10px;
    background:var(--p-surface-section,#252525); flex-shrink:0; }
.spe-analyze-title { font-weight:600; font-size:12px; margin-bottom:8px; display:flex;
    align-items:center; gap:6px; cursor:pointer; }
.spe-pass-pills { display:flex; flex-wrap:wrap; gap:4px; margin-bottom:8px; }
.spe-pass-pill { padding:3px 10px; border-radius:12px; font-size:11px; cursor:pointer;
    border:1px solid var(--p-surface-border,#555); background:transparent;
    color:var(--p-text-color,#ddd); }
.spe-pass-pill.active { background:var(--p-primary-color,#58a6ff); color:#000; border-color:transparent; }
.spe-candidate-row { display:flex; align-items:flex-start; gap:6px; padding:5px 6px;
    border:1px solid var(--p-surface-border,#444); border-radius:4px; margin-bottom:4px; }
.spe-candidate-row.dup { border-color:#d29922; }
.spe-cand-body { flex:1; min-width:0; }
.spe-cand-label { font-size:12px; font-weight:600; }
.spe-cand-role  { font-size:11px; color:var(--p-text-muted-color,#888); }
.spe-badge { font-size:10px; padding:1px 5px; border-radius:3px; letter-spacing:.04em;
    background:var(--p-surface-d,#2d2d2d); color:var(--p-text-muted-color,#888); }
.spe-badge.dup { background:rgba(210,153,34,.18); color:#d29922; }

/* Collapse toggle */
.spe-collapse-toggle { cursor:pointer; user-select:none; }
.spe-collapsible { overflow:hidden; }
.spe-collapsible.collapsed { display:none; }

/* Divider */
.spe-divider { height:1px; background:var(--p-surface-border,#444); margin:8px 0; }

/* Section heading */
.spe-section-head { font-size:11px; text-transform:uppercase; letter-spacing:.06em;
    color:var(--p-text-muted-color,#888); margin:10px 0 5px; }

/* Clips section */
.spe-clips { border-top:1px solid var(--p-surface-border,#444); padding:10px;
    background:var(--p-surface-section,#252525); flex-shrink:0; }
.spe-clips-title { font-weight:600; font-size:12px; margin-bottom:6px; display:flex;
    align-items:center; gap:6px; cursor:pointer; }
.spe-clips-toolbar { display:flex; gap:6px; align-items:center; flex-wrap:wrap; margin-bottom:6px; }
.spe-clips-toolbar label { font-size:11px; color:var(--p-text-muted-color,#888); white-space:nowrap; }
.spe-clips-toolbar input[type=number] { width:68px; padding:3px 6px; border-radius:4px;
    border:1px solid var(--p-surface-border,#555);
    background:var(--p-surface-ground,#1a1a1a); color:var(--p-text-color,#eee); font-size:12px; }
.spe-clips-toolbar select { padding:3px 6px; border-radius:4px;
    border:1px solid var(--p-surface-border,#555);
    background:var(--p-surface-ground,#1a1a1a); color:var(--p-text-color,#eee); font-size:12px; }
.spe-timeline-wrap { margin-bottom:8px; overflow:hidden; border-radius:4px; }
.spe-timeline { display:block; width:100%; height:72px; cursor:default; }
.spe-clip-card { border:1px solid var(--p-surface-border,#444); border-radius:5px;
    margin-bottom:5px; overflow:hidden; background:var(--p-surface-ground,#1a1a1a); }
.spe-clip-card-head { display:flex; align-items:center; gap:6px; padding:5px 8px;
    background:var(--p-surface-section,#252525); cursor:pointer; user-select:none; }
.spe-clip-color-dot { width:10px; height:10px; border-radius:50%; flex-shrink:0; }
.spe-clip-card-body { padding:8px; }
.spe-clip-card-body.collapsed { display:none; }
.spe-clip-row { display:flex; gap:6px; align-items:center; margin-bottom:6px; font-size:11px;
    color:var(--p-text-muted-color,#888); }
.spe-clip-inp { width:68px; padding:3px 5px; border-radius:3px;
    border:1px solid var(--p-surface-border,#555);
    background:var(--p-surface-ground,#1a1a1a); color:var(--p-text-color,#eee); font-size:11px; }
.spe-clip-inp.wide { width:100%; box-sizing:border-box; }
.spe-clip-ta { width:100%; padding:4px 6px; border-radius:3px; box-sizing:border-box;
    border:1px solid var(--p-surface-border,#555);
    background:var(--p-surface-ground,#1a1a1a); color:var(--p-text-color,#eee); font-size:12px;
    resize:vertical; margin-bottom:6px; }
.spe-clip-subj-list { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:6px; }
.spe-clip-subj-check { display:flex; align-items:center; gap:4px; font-size:11px;
    cursor:pointer; color:var(--p-text-color,#eee); }
.spe-clips-det-note { font-size:11px; color:#fbbf24; margin-top:4px; }
`;

function _injectCSS() {
    if (document.getElementById("spe-styles")) return;
    const s = document.createElement("style");
    s.id = "spe-styles";
    s.textContent = _CSS;
    document.head.appendChild(s);
}

// ── Render helpers ─────────────────────────────────────────────────────────────

function _mediaUrl(filename, dir) {
    if (!filename) return "";
    return bundlesApi.streamUrl(filename, dir);
}

function _genClipId() {
    return "clip_" + Date.now().toString(36).slice(-5);
}

function _fmtT(sec) {
    return (Math.round(sec * 10) / 10).toFixed(1) + "s";
}

function _drawTimeline(canvas, clips, suggestions, totalDuration) {
    const W = canvas.width;
    const H = canvas.height;
    const ctx = canvas.getContext("2d");
    const dur = totalDuration > 0 ? totalDuration : 1;
    const toX = t => (t / dur) * W;

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#111";
    ctx.fillRect(0, 0, W, H);

    const BAND_TOP = 10, BAND_BOT = H - 20;

    // Clip bands
    clips.forEach((clip, i) => {
        const x1 = toX(clip.start_time);
        const x2 = toX(clip.end_time);
        const col = CLIP_COLORS[i % CLIP_COLORS.length];
        ctx.fillStyle = col + "33";
        ctx.fillRect(x1, BAND_TOP, x2 - x1, BAND_BOT - BAND_TOP);
        ctx.strokeStyle = col;
        ctx.lineWidth = 1.5;
        ctx.strokeRect(x1 + 0.5, BAND_TOP + 0.5, x2 - x1 - 1, BAND_BOT - BAND_TOP - 1);
        // Label
        const label = clip.label || `Clip ${i + 1}`;
        ctx.fillStyle = "#ddd";
        ctx.font = "10px sans-serif";
        ctx.textAlign = "center";
        const mid = (x1 + x2) / 2;
        const maxW = x2 - x1 - 6;
        const text = ctx.measureText(label).width > maxW ? "" : label;
        if (text) ctx.fillText(text, mid, (BAND_TOP + BAND_BOT) / 2 + 4);
    });

    // Internal boundary handles
    clips.forEach((clip, i) => {
        if (i === 0) return;
        const x = toX(clip.start_time);
        ctx.strokeStyle = "#ffffffcc";
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(x, BAND_TOP); ctx.lineTo(x, BAND_BOT); ctx.stroke();
        ctx.fillStyle = "#fff";
        ctx.beginPath(); ctx.arc(x, (BAND_TOP + BAND_BOT) / 2, 5, 0, Math.PI * 2); ctx.fill();
    });

    // Suggestion markers (dashed gold)
    ctx.setLineDash([4, 3]);
    ctx.strokeStyle = "#fbbf24";
    ctx.lineWidth = 1.5;
    suggestions.forEach(seg => {
        const x = toX(seg.start_time);
        ctx.beginPath(); ctx.moveTo(x, 2); ctx.lineTo(x, H - 2); ctx.stroke();
    });
    ctx.setLineDash([]);

    // Time axis ticks
    const tickStep = dur <= 20 ? 2 : dur <= 60 ? 5 : dur <= 180 ? 15 : 30;
    ctx.fillStyle = "#555"; ctx.font = "9px monospace"; ctx.textAlign = "center";
    for (let t = 0; t <= dur + 0.001; t += tickStep) {
        const x = Math.round(toX(t));
        ctx.fillStyle = "#444";
        ctx.fillRect(x, BAND_BOT, 1, 4);
        ctx.fillStyle = "#666";
        ctx.fillText(`${t}`, x, H - 2);
    }

    // Start / end labels under first / last clip
    if (clips.length) {
        ctx.font = "9px monospace"; ctx.fillStyle = "#888";
        ctx.textAlign = "left";
        ctx.fillText(_fmtT(clips[0].start_time), toX(clips[0].start_time) + 2, BAND_BOT + 4);
        ctx.textAlign = "right";
        const last = clips[clips.length - 1];
        ctx.fillText(_fmtT(last.end_time), toX(last.end_time) - 2, BAND_BOT + 4);
    }
}

function _renderClipsSection(container, profile, onClipsChanged) {
    const wrap = _mk("div", { cls: "spe-clips" });
    container.appendChild(wrap);

    let clipsOpen = false;
    let suggestions = [];
    let detecting = false;
    let autoSegRunning = false;
    let clipsCaptioner = "qwen_vl";
    let videoDuration = 0;
    let hiddenVideo = null;
    let clips = [...(profile.clips || [])];

    // ── Title row (toggle) ────────────────────────────────────────────────────
    const chevron = _mk("i", { cls: "pi pi-chevron-down", style: { marginLeft: "auto" } });
    const titleRow = _mk("div", { cls: "spe-clips-title" }, [
        _mk("i", { cls: "pi pi-clock" }), "  Clips ", chevron,
    ]);
    wrap.appendChild(titleRow);

    const body = _mk("div", { cls: "spe-collapsible collapsed" });
    wrap.appendChild(body);

    // ── Duration probe ────────────────────────────────────────────────────────
    let durationInput, segDurInput;

    const probeDuration = () => {
        if (profile.media_type !== "video" || !profile.media_filename || hiddenVideo) return;
        hiddenVideo = _mk("video", { preload: "metadata" });
        hiddenVideo.style.display = "none";
        hiddenVideo.src = _mediaUrl(profile.media_filename, profile.media_dir);
        hiddenVideo.addEventListener("loadedmetadata", () => {
            videoDuration = hiddenVideo.duration || 0;
            if (durationInput && videoDuration > 0) durationInput.value = videoDuration.toFixed(1);
            redraw();
        });
        document.body.appendChild(hiddenVideo);
    };

    const getTotalDuration = () => {
        if (videoDuration > 0) return videoDuration;
        const v = parseFloat(durationInput?.value) || 0;
        if (v > 0) return v;
        if (!clips.length) return 60;
        return Math.max(...clips.map(c => c.end_time), 1);
    };

    // ── Toolbar ───────────────────────────────────────────────────────────────
    durationInput = _mk("input", { type: "number", placeholder: "auto", min: 0, step: 0.5 });
    durationInput.onchange = () => { videoDuration = parseFloat(durationInput.value) || 0; redraw(); };

    segDurInput = _mk("input", { type: "number", placeholder: "10", min: 1, step: 0.5,
        value: (profile.default_segment_duration ?? 10).toFixed(1) });

    const autoSegBtn = _mk("button", { cls: "spe-btn sm", onclick: runAutoSegment }, ["Auto-segment"]);

    body.appendChild(_mk("div", { cls: "spe-clips-toolbar" }, [
        _mk("label", {}, ["Duration (s):"]), durationInput,
        _mk("label", {}, ["Seg (s):"]), segDurInput,
        autoSegBtn,
    ]));

    // Detect row
    const captEl = _mk("select");
    CAPTIONER_TYPES.forEach(c => {
        const o = _mk("option", { value: c }, [c]);
        if (c === clipsCaptioner) o.selected = true;
        captEl.appendChild(o);
    });
    captEl.onchange = () => { clipsCaptioner = captEl.value; };

    const detectBtn = _mk("button", { cls: "spe-btn sm ghost", onclick: runDetect }, ["Detect boundaries"]);
    const detectSpinner = _mk("span", { style: { fontSize: "11px", color: "#888", display: "none" } }, [" Detecting…"]);
    let detNoteEl = null;

    body.appendChild(_mk("div", { cls: "spe-clips-toolbar" }, [
        _mk("label", {}, ["Captioner:"]), captEl, detectBtn, detectSpinner,
    ]));

    // ── Timeline canvas ───────────────────────────────────────────────────────
    const canvasWrap = _mk("div", { cls: "spe-timeline-wrap" });
    const canvas = document.createElement("canvas");
    canvas.className = "spe-timeline";
    canvas.height = 72;
    canvasWrap.appendChild(canvas);
    body.appendChild(canvasWrap);

    // ── Clip list ─────────────────────────────────────────────────────────────
    const listEl = _mk("div");
    body.appendChild(listEl);

    const addClipBtn = _mk("button", { cls: "spe-btn sm", style: { marginTop: "4px" },
        onclick: addNewClip }, ["+ Add clip"]);
    body.appendChild(addClipBtn);

    // ── Toggle ────────────────────────────────────────────────────────────────
    titleRow.onclick = () => {
        clipsOpen = !clipsOpen;
        body.classList.toggle("collapsed", !clipsOpen);
        chevron.className = "pi pi-chevron-" + (clipsOpen ? "up" : "down");
        if (clipsOpen) {
            probeDuration();
            requestAnimationFrame(redraw);
        } else {
            if (hiddenVideo) { hiddenVideo.remove(); hiddenVideo = null; }
        }
    };

    // ── Redraw ────────────────────────────────────────────────────────────────
    function redraw() {
        canvas.width = canvas.offsetWidth || 380;
        _drawTimeline(canvas, clips, suggestions, getTotalDuration());
        renderClipList();
    }

    // ── Timeline drag ─────────────────────────────────────────────────────────
    const MIN_DUR = 1.0;
    let _drag = null;

    canvas.addEventListener("mousedown", e => {
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const totalDur = getTotalDuration() || 1;
        for (let i = 1; i < clips.length; i++) {
            const hx = (clips[i].start_time / totalDur) * canvas.offsetWidth;
            if (Math.abs(x - hx) <= 8) { _drag = { i }; e.preventDefault(); break; }
        }
    });

    canvas.addEventListener("mousemove", e => {
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const totalDur = getTotalDuration() || 1;
        let over = false;
        for (let i = 1; i < clips.length; i++) {
            const hx = (clips[i].start_time / totalDur) * canvas.offsetWidth;
            if (Math.abs(x - hx) <= 8) { over = true; break; }
        }
        canvas.style.cursor = over ? "col-resize" : "default";
        if (!_drag) return;
        const i = _drag.i;
        const t = Math.round(((x / canvas.offsetWidth) * totalDur) * 10) / 10;
        const clamped = Math.max(clips[i - 1].start_time + MIN_DUR,
                                 Math.min(clips[i].end_time - MIN_DUR, t));
        clips = clips.map((c, j) =>
            j === i - 1 ? { ...c, end_time: clamped }
          : j === i     ? { ...c, start_time: clamped }
          : c);
        canvas.width = canvas.offsetWidth;
        _drawTimeline(canvas, clips, suggestions, totalDur);
    });

    const endDrag = () => {
        if (!_drag) return;
        _drag = null;
        profile.clips = clips;
        onClipsChanged(clips);
        renderClipList();
    };
    canvas.addEventListener("mouseup", endDrag);
    canvas.addEventListener("mouseleave", endDrag);

    // ── Clip list render ──────────────────────────────────────────────────────
    function renderClipList() {
        listEl.innerHTML = "";
        clips.forEach((clip, i) => {
            const color = CLIP_COLORS[i % CLIP_COLORS.length];
            const subjects = profile.subjects || [];

            const labelInp = _mk("input", { type: "text", cls: "spe-clip-inp wide",
                value: clip.label || "", placeholder: "Segment label" });
            labelInp.onchange = () => { clips[i] = { ...clips[i], label: labelInp.value }; commitClip(i); };

            const startEl = _mk("input", { type: "number", cls: "spe-clip-inp",
                value: clip.start_time.toFixed(1), min: 0, step: 0.1 });
            const endEl   = _mk("input", { type: "number", cls: "spe-clip-inp",
                value: clip.end_time.toFixed(1),   min: 0, step: 0.1 });
            const applyTimes = () => {
                const s = parseFloat(startEl.value) || 0;
                const e = parseFloat(endEl.value)   || 0;
                if (e > s) { clips[i] = { ...clips[i], start_time: s, end_time: e }; commitClip(i); }
            };
            startEl.onchange = applyTimes; endEl.onchange = applyTimes;

            const durSpan = _mk("span", { style: { color: "#888" } },
                [`(${(clip.end_time - clip.start_time).toFixed(1)}s)`]);

            const actionEl = _mk("textarea", { cls: "spe-clip-ta", rows: 2,
                placeholder: "Action description — edit manually or click Describe" });
            actionEl.value = clip.action || "";
            actionEl.onchange = () => { clips[i] = { ...clips[i], action: actionEl.value }; commitClip(i); };

            const describeBtn = _mk("button", { cls: "spe-btn sm ghost",
                onclick: () => describeClipAction(i, actionEl, describeBtn) }, ["Describe"]);

            const subjWrap = _mk("div", { cls: "spe-clip-subj-list" });
            subjects.forEach(s => {
                const cb = _mk("input", { type: "checkbox" });
                cb.checked = (clip.subjects || []).includes(s.id);
                cb.onchange = () => {
                    const cur = new Set(clips[i].subjects || []);
                    if (cb.checked) cur.add(s.id); else cur.delete(s.id);
                    clips[i] = { ...clips[i], subjects: [...cur] };
                    commitClip(i);
                };
                subjWrap.appendChild(_mk("label", { cls: "spe-clip-subj-check" },
                    [cb, " " + (s.label || s.id)]));
            });

            const clipBody = _mk("div", { cls: "spe-clip-card-body" }, [
                _mk("div", { cls: "spe-clip-row", style: { marginBottom: "6px" } }, [labelInp]),
                _mk("div", { cls: "spe-clip-row" }, [
                    "Start:", startEl, "End:", endEl, durSpan,
                ]),
                actionEl,
                subjects.length ? subjWrap : null,
                _mk("div", { style: { display: "flex", gap: "4px" } }, [describeBtn]),
            ]);

            let cardOpen = true;
            const head = _mk("div", { cls: "spe-clip-card-head" }, [
                _mk("span", { cls: "spe-clip-color-dot", style: { background: color } }),
                _mk("span", { style: { flex: "1", fontWeight: "600", fontSize: "12px" } },
                    [clip.label || `Clip ${i + 1}`]),
                _mk("span", { style: { color: "#888", fontSize: "11px" } },
                    [`${clip.start_time.toFixed(1)}–${clip.end_time.toFixed(1)}s`]),
                _mk("button", { cls: "spe-btn sm danger",
                    onclick: e => { e.stopPropagation(); removeClip(i); } }, ["×"]),
            ]);
            head.addEventListener("click", e => {
                if (e.target.closest(".spe-btn")) return;
                cardOpen = !cardOpen;
                clipBody.classList.toggle("collapsed", !cardOpen);
            });

            listEl.appendChild(_mk("div", { cls: "spe-clip-card" }, [head, clipBody]));
        });
    }

    // ── Operations ────────────────────────────────────────────────────────────
    function commitClip(i) {
        profile.clips = clips;
        onClipsChanged(clips);
        canvas.width = canvas.offsetWidth;
        _drawTimeline(canvas, clips, suggestions, getTotalDuration());
    }

    function removeClip(i) {
        clips = clips.filter((_, j) => j !== i);
        profile.clips = clips;
        onClipsChanged(clips);
        redraw();
    }

    function addNewClip() {
        const lastEnd = clips.length ? clips[clips.length - 1].end_time : 0;
        const segDur  = parseFloat(segDurInput.value) || (profile.default_segment_duration ?? 10);
        clips.push({
            id: _genClipId(), label: `Clip ${clips.length + 1}`,
            start_time: lastEnd, end_time: lastEnd + segDur,
            select_every_nth: 2, frame_load_cap: 120, subjects: [], action: "",
        });
        profile.clips = clips;
        onClipsChanged(clips);
        redraw();
    }

    async function runAutoSegment() {
        const totalDur = getTotalDuration();
        if (totalDur <= 0) { _toast("Set video duration first", "warn"); return; }
        if (autoSegRunning) return;
        autoSegRunning = true;
        autoSegBtn.disabled = true;
        autoSegBtn.textContent = "…";
        try {
            const segDur = parseFloat(segDurInput.value) || (profile.default_segment_duration ?? 10);
            const res = await sourceProfilesApi.autoPartition({
                profile_id: profile.id, video_duration: totalDur, segment_duration: segDur,
            });
            clips = res.profile?.clips || [];
            profile.clips = clips;
            onClipsChanged(clips);
            redraw();
            _toast(`Created ${clips.length} clip(s)`, "success");
        } catch (err) {
            _toast(`Auto-segment failed: ${err.message}`, "error");
        } finally {
            autoSegRunning = false;
            autoSegBtn.disabled = false;
            autoSegBtn.textContent = "Auto-segment";
        }
    }

    async function runDetect() {
        const totalDur = getTotalDuration();
        if (totalDur <= 0) { _toast("Set video duration first", "warn"); return; }
        if (!profile.media_filename) { _toast("Profile has no media file set", "warn"); return; }
        if (detecting) return;
        detecting = true;
        detectBtn.disabled = true;
        detectSpinner.style.display = "inline";
        try {
            const res = await sourceProfilesApi.detectSegments({
                profile_id: profile.id, video_duration: totalDur, captioner_type: clipsCaptioner,
            });
            suggestions = res.segments || [];
            if (detNoteEl) detNoteEl.remove();
            if (suggestions.length) {
                detNoteEl = _mk("div", { cls: "spe-clips-det-note" },
                    [`${suggestions.length} suggested boundary(s) shown as gold markers. Auto-segment to apply.`]);
                canvasWrap.after(detNoteEl);
            }
            redraw();
            _toast(`${suggestions.length} suggested boundary(s)`, "info");
        } catch (err) {
            _toast(`Detection failed: ${err.message}`, "error");
        } finally {
            detecting = false;
            detectBtn.disabled = false;
            detectSpinner.style.display = "none";
        }
    }

    async function describeClipAction(i, actionEl, btn) {
        const clip = clips[i];
        const prev = btn.textContent;
        btn.disabled = true;
        btn.textContent = "…";
        try {
            const res = await sourceProfilesApi.describeClip({
                profile_id: profile.id, start_time: clip.start_time, end_time: clip.end_time,
                captioner_type: clipsCaptioner,
            });
            if (res.action) {
                clips[i] = { ...clips[i], action: res.action };
                actionEl.value = res.action;
                commitClip(i);
                _toast("Action filled", "success");
            }
        } catch (err) {
            _toast(`Describe failed: ${err.message}`, "error");
        } finally {
            btn.disabled = false;
            btn.textContent = prev;
        }
    }
}

function _renderMediaPreview(profile) {
    const { media_filename: fn, media_dir: dir, media_type: type } = profile;
    if (!fn) return _mk("div", { cls: "spe-media-wrap", style: { color: "#666", fontSize: "11px" } }, ["No media file"]);
    const wrap = _mk("div", { cls: "spe-media-wrap" });
    if (type === "video") {
        const v = _mk("video", { controls: true, preload: "metadata" });
        v.src = _mediaUrl(fn, dir);
        wrap.appendChild(v);
    } else {
        const img = _mk("img", { alt: fn });
        img.src = _mediaUrl(fn, dir);
        wrap.appendChild(img);
    }
    return wrap;
}

// ── Subject form ───────────────────────────────────────────────────────────────

function _renderSubjectForm(container, initial = {}, onSave, onCancel) {
    container.innerHTML = "";
    const labelEl = _mk("input", { type: "text", placeholder: "woman on left", value: initial.label || "" });
    const roleEl  = _mk("textarea", { placeholder: "the woman originally in the video", rows: 2 });
    roleEl.value  = initial.role_description || "";
    const typeEl  = _mk("select");
    ENTITY_TYPES.forEach(t => {
        const o = _mk("option", { value: t }, [t]);
        if (t === (initial.entity_type || "person")) o.selected = true;
        typeEl.appendChild(o);
    });
    const notesEl = _mk("textarea", { placeholder: "Internal notes (not used in prompt)", rows: 2 });
    notesEl.value = initial.notes || "";

    container.appendChild(_mk("div", { cls: "spe-form-row" }, [
        _mk("label", {}, ["Label (brief identifier)"]), labelEl,
    ]));
    container.appendChild(_mk("div", { cls: "spe-form-row" }, [
        _mk("label", {}, ["Role description (appears in prompt)"]), roleEl,
    ]));
    container.appendChild(_mk("div", { cls: "spe-form-row" }, [
        _mk("label", {}, ["Entity type"]), typeEl,
    ]));
    container.appendChild(_mk("div", { cls: "spe-form-row" }, [
        _mk("label", {}, ["Notes (internal)"]), notesEl,
    ]));

    container.appendChild(_mk("div", { cls: "spe-form-actions" }, [
        _mk("button", { cls: "spe-btn sm ghost", onclick: onCancel }, ["Cancel"]),
        _mk("button", { cls: "spe-btn sm primary", onclick: () => {
            onSave({
                label:            labelEl.value.trim(),
                role_description: roleEl.value.trim(),
                entity_type:      typeEl.value,
                notes:            notesEl.value.trim(),
            });
        }}, ["Save subject"]),
    ]));
}

// ── Detail view ────────────────────────────────────────────────────────────────

function _renderDetail(root) {
    root.innerHTML = "";
    const profile = _profile();
    if (!profile) return;

    // ── Header
    const header = _mk("div", { cls: "spe-detail-header" }, [
        _mk("button", { cls: "spe-btn sm ghost", onclick: () => { _S.selected = null; _render(root.parentElement?.parentElement || root); } }, ["← Back"]),
        _mk("h3", {}, [profile.name || profile.id]),
        _mk("button", { cls: "spe-btn sm primary", onclick: () => _saveProfile(root, profile) }, ["Save"]),
    ]);
    root.appendChild(header);

    const body = _mk("div", { cls: "spe-detail-body" });
    root.appendChild(body);

    // ── Media preview
    body.appendChild(_renderMediaPreview(profile));

    // ── Profile meta form
    body.appendChild(_mk("div", { cls: "spe-section-head" }, ["Profile settings"]));

    const nameEl = _mk("input", { type: "text", value: profile.name || "" });
    body.appendChild(_mk("div", { cls: "spe-form-row" }, [_mk("label", {}, ["Name"]), nameEl]));

    const allMedia = _allMedia(profile.media_type, profile.media_dir);
    const fileEl = _mk("select");
    ["", ...allMedia].forEach(fn => {
        const o = _mk("option", { value: fn }, [fn || "(none)"]);
        if (fn === (profile.media_filename || "")) o.selected = true;
        fileEl.appendChild(o);
    });

    const dirEl = _mk("select");
    MEDIA_DIRS.forEach(d => {
        const o = _mk("option", { value: d }, [d]);
        if (d === (profile.media_dir || "input")) o.selected = true;
        dirEl.appendChild(o);
    });

    const typeEl = _mk("select");
    MEDIA_TYPES.forEach(t => {
        const o = _mk("option", { value: t }, [t]);
        if (t === (profile.media_type || "video")) o.selected = true;
        typeEl.appendChild(o);
    });

    // Refresh file list when dir/type change
    const refreshFiles = () => {
        const files = _allMedia(typeEl.value, dirEl.value);
        const cur = fileEl.value;
        fileEl.innerHTML = "";
        ["", ...files].forEach(fn => {
            const o = _mk("option", { value: fn }, [fn || "(none)"]);
            if (fn === cur) o.selected = true;
            fileEl.appendChild(o);
        });
    };
    dirEl.addEventListener("change", refreshFiles);
    typeEl.addEventListener("change", refreshFiles);

    body.appendChild(_mk("div", { cls: "spe-form-row" }, [_mk("label", {}, ["Media type"]), typeEl]));
    body.appendChild(_mk("div", { cls: "spe-form-row" }, [_mk("label", {}, ["Media dir"]), dirEl]));
    body.appendChild(_mk("div", { cls: "spe-form-row" }, [_mk("label", {}, ["Media file"]), fileEl]));

    // Wire save
    const collectMeta = () => ({
        ...profile,
        name:           nameEl.value.trim() || profile.name,
        media_filename: fileEl.value,
        media_dir:      dirEl.value,
        media_type:     typeEl.value,
    });

    // ── Subjects section
    body.appendChild(_mk("div", { cls: "spe-divider" }));
    body.appendChild(_mk("div", { cls: "spe-section-head" }, ["Subjects"]));

    const subjListEl = _mk("div");
    body.appendChild(subjListEl);

    const subjFormEl = _mk("div");
    body.appendChild(subjFormEl);

    const renderSubjList = () => {
        subjListEl.innerHTML = "";
        const subjects = profile.subjects || [];
        if (!subjects.length) {
            subjListEl.appendChild(_mk("div", { style: { color: "#666", fontSize: "11px", margin: "6px 0" } }, ["No subjects yet. Add one below or run an Analyze pass."]));
        }
        subjects.forEach((s, idx) => {
            const icon = _mk("i", { cls: ENTITY_ICONS[s.entity_type] || "pi pi-circle", cls2: "spe-subj-icon" });
            icon.className = (ENTITY_ICONS[s.entity_type] || "pi pi-circle") + " spe-subj-icon";
            const row = _mk("div", { cls: "spe-subj-row" }, [
                icon,
                _mk("div", { cls: "spe-subj-body" }, [
                    _mk("div", { cls: "spe-subj-label" }, [s.label || s.id || ""]),
                    _mk("div", { cls: "spe-subj-role" }, [s.role_description || ""]),
                ]),
                _mk("div", { cls: "spe-subj-actions" }, [
                    _mk("button", { cls: "spe-btn sm ghost", onclick: () => {
                        _S.editingSubject = { idx, data: { ...s } };
                        _renderSubjectForm(subjFormEl, s,
                            (updated) => {
                                profile.subjects[idx] = { ...s, ...updated };
                                _S.editingSubject = null;
                                subjFormEl.innerHTML = "";
                                renderSubjList();
                            },
                            () => { _S.editingSubject = null; subjFormEl.innerHTML = ""; }
                        );
                    }}, ["Edit"]),
                    _mk("button", { cls: "spe-btn sm danger", onclick: () => {
                        profile.subjects.splice(idx, 1);
                        renderSubjList();
                    }}, ["×"]),
                ]),
            ]);
            subjListEl.appendChild(row);
        });

        // Add button
        if (!_S.editingSubject) {
            subjListEl.appendChild(_mk("button", { cls: "spe-btn sm", style: { marginTop: "4px" }, onclick: () => {
                _S.editingSubject = { idx: -1, data: {} };
                _renderSubjectForm(subjFormEl, {},
                    (data) => {
                        if (!data.label) { _toast("Label is required", "warn"); return; }
                        profile.subjects = profile.subjects || [];
                        profile.subjects.push({ id: _genSubjId(data.label), ...data });
                        _S.editingSubject = null;
                        subjFormEl.innerHTML = "";
                        renderSubjList();
                    },
                    () => { _S.editingSubject = null; subjFormEl.innerHTML = ""; }
                );
            }}, ["+ Add subject"]));
        }
    };
    renderSubjList();

    // ── Clips section (video only)
    if ((profile.media_type || "video") === "video") {
        body.appendChild(_mk("div", { cls: "spe-divider" }));
        _renderClipsSection(body, profile, (_clips) => {
            profile.clips = _clips;
        });
    }

    // ── Analyze section
    _renderAnalyzeSection(body, profile, root, renderSubjList);

    // Bind save button
    const saveBtn = header.querySelector("button.primary");
    saveBtn.onclick = async () => {
        const updated = collectMeta();
        await _saveProfile(root, updated);
    };
}

// ── Analyze section ────────────────────────────────────────────────────────────

function _renderAnalyzeSection(container, profile, rootEl, onSubjectsChanged) {
    const wrap = _mk("div", { cls: "spe-analyze" });
    container.appendChild(wrap);

    const titleRow = _mk("div", { cls: "spe-analyze-title spe-collapse-toggle" }, [
        _mk("i", { cls: "pi pi-magic" }),
        "  Analyze Media",
        _mk("i", { cls: "pi pi-chevron-" + (_S.analyzeOpen ? "up" : "down"), style: { marginLeft: "auto" } }),
    ]);
    wrap.appendChild(titleRow);

    const body = _mk("div", { cls: "spe-collapsible" + (_S.analyzeOpen ? "" : " collapsed") });
    wrap.appendChild(body);

    titleRow.onclick = () => {
        _S.analyzeOpen = !_S.analyzeOpen;
        body.classList.toggle("collapsed", !_S.analyzeOpen);
        titleRow.querySelector(".pi-chevron-up, .pi-chevron-down").className =
            "pi pi-chevron-" + (_S.analyzeOpen ? "up" : "down");
        if (_S.analyzeOpen) _loadHistory(profile.id);
    };

    // Pass type pills
    body.appendChild(_mk("div", { cls: "spe-section-head" }, ["Focus pass"]));
    const pillsWrap = _mk("div", { cls: "spe-pass-pills" });
    PASS_TYPES.forEach(pt => {
        const pill = _mk("button", {
            cls: "spe-pass-pill" + (pt === _S.analyzePassType ? " active" : ""),
            onclick: () => {
                _S.analyzePassType = pt;
                pillsWrap.querySelectorAll(".spe-pass-pill").forEach(p => p.classList.remove("active"));
                pill.classList.add("active");
            },
        }, [PASS_LABELS[pt]]);
        pillsWrap.appendChild(pill);
    });
    body.appendChild(pillsWrap);

    // Captioner selector
    body.appendChild(_mk("div", { cls: "spe-form-row" }, [
        _mk("label", {}, ["Captioner"]),
        (() => {
            const s = _mk("select");
            CAPTIONER_TYPES.forEach(c => {
                const o = _mk("option", { value: c }, [c]);
                if (c === _S.analyzeCaptioner) o.selected = true;
                s.appendChild(o);
            });
            s.onchange = () => { _S.analyzeCaptioner = s.value; };
            return s;
        })(),
    ]));

    // Prompt override
    const overrideToggle = _mk("div", { cls: "spe-section-head spe-collapse-toggle",
        style: { cursor: "pointer" } }, ["▶ Edit prompt"]);
    const overrideWrap = _mk("div", { cls: "spe-collapsible collapsed" });
    const overrideEl = _mk("textarea", { rows: 4, placeholder: "Leave empty to use the built-in template for the selected pass type." });
    overrideEl.value = _S.analyzePromptOverride;
    overrideEl.onchange = () => { _S.analyzePromptOverride = overrideEl.value; };
    overrideWrap.appendChild(overrideEl);
    overrideToggle.onclick = () => {
        const open = overrideWrap.classList.toggle("collapsed") === false;
        overrideToggle.textContent = (open ? "▼" : "▶") + " Edit prompt";
    };
    body.appendChild(overrideToggle);
    body.appendChild(overrideWrap);

    // Run button + spinner
    const runBtn = _mk("button", { cls: "spe-btn primary", style: { width: "100%", marginTop: "8px" },
        onclick: () => _runAnalysis(profile, body, onSubjectsChanged),
    }, ["Run analysis"]);
    body.appendChild(runBtn);

    const spinnerEl = _mk("div", { style: { textAlign: "center", fontSize: "11px", color: "#888", marginTop: "4px", display: "none" } },
        ["Running…"]);
    body.appendChild(spinnerEl);

    // Candidates area
    const candidatesEl = _mk("div");
    body.appendChild(candidatesEl);

    // History section
    const histToggle = _mk("div", { cls: "spe-section-head spe-collapse-toggle",
        style: { cursor: "pointer", marginTop: "8px" } }, ["▶ Previous runs"]);
    const histWrap = _mk("div", { cls: "spe-collapsible collapsed" });
    histToggle.onclick = async () => {
        const open = histWrap.classList.toggle("collapsed") === false;
        histToggle.textContent = (open ? "▼" : "▶") + " Previous runs";
        if (open) {
            await _loadHistory(profile.id);
            _renderHistory(histWrap, profile, onSubjectsChanged);
        }
    };
    body.appendChild(histToggle);
    body.appendChild(histWrap);

    // Stash refs so _runAnalysis can update them
    body._runBtn      = runBtn;
    body._spinnerEl   = spinnerEl;
    body._candidatesEl = candidatesEl;
    body._histWrap    = histWrap;
    body._histToggle  = histToggle;
    body._profile     = profile;
    body._onChanged   = onSubjectsChanged;
}

async function _runAnalysis(profile, analyzeBody, onSubjectsChanged) {
    if (_S.analyzeRunning) return;
    _S.analyzeRunning = true;
    analyzeBody._runBtn.disabled = true;
    analyzeBody._spinnerEl.style.display = "block";
    analyzeBody._candidatesEl.innerHTML  = "";

    try {
        const res = await sourceProfilesApi.analyze({
            profile_id:       profile.id,
            pass_type:        _S.analyzePassType,
            prompt_override:  _S.analyzePromptOverride,
            captioner_type:   _S.analyzeCaptioner,
        });

        _S.analyzeCandidates = res.candidates ?? [];
        _renderCandidates(analyzeBody._candidatesEl, profile, onSubjectsChanged);

        // Refresh history
        await _loadHistory(profile.id);
        if (!analyzeBody._histWrap.classList.contains("collapsed")) {
            _renderHistory(analyzeBody._histWrap, profile, onSubjectsChanged);
        }

        _toast(`Found ${_S.analyzeCandidates.length} candidate(s)`, "success");
    } catch (err) {
        _toast(`Analysis failed: ${err.message}`, "error");
    } finally {
        _S.analyzeRunning = false;
        analyzeBody._runBtn.disabled = false;
        analyzeBody._spinnerEl.style.display = "none";
    }
}

function _renderCandidates(container, profile, onSubjectsChanged) {
    container.innerHTML = "";
    const candidates = _S.analyzeCandidates;
    if (!candidates.length) {
        container.appendChild(_mk("div", { style: { color: "#888", fontSize: "11px", padding: "6px 0" } }, ["No candidates returned."]));
        return;
    }

    const existingLabels = new Set((profile.subjects || []).map(s => s.label.toLowerCase()));

    container.appendChild(_mk("div", { cls: "spe-section-head" }, [
        `${candidates.length} candidate(s) — `,
        _mk("a", { href: "#", style: { color: "var(--p-primary-color,#58a6ff)", fontSize: "11px" },
            onclick: (e) => { e.preventDefault(); _acceptAll(candidates, profile, onSubjectsChanged, container); }
        }, ["Add all"]),
    ]));

    candidates.forEach((c, i) => {
        const isDup = existingLabels.has(c.label.toLowerCase());
        const icon = _mk("i", {});
        icon.className = (ENTITY_ICONS[c.entity_type] || "pi pi-circle") + " spe-subj-icon";

        const addBtn = _mk("button", { cls: "spe-btn sm primary", onclick: () => {
            _acceptOne(c, profile, onSubjectsChanged);
            addBtn.disabled = true;
            addBtn.textContent = "Added";
        }}, ["Add"]);

        const row = _mk("div", { cls: "spe-candidate-row" + (isDup ? " dup" : "") }, [
            icon,
            _mk("div", { cls: "spe-cand-body" }, [
                _mk("div", { cls: "spe-cand-label" }, [
                    c.label,
                    " ",
                    _mk("span", { cls: "spe-badge" + (isDup ? " dup" : "") }, [isDup ? "duplicate" : c.entity_type]),
                ]),
                _mk("div", { cls: "spe-cand-role" }, [c.role_description || ""]),
            ]),
            addBtn,
        ]);
        container.appendChild(row);
    });
}

function _renderHistory(container, profile, onSubjectsChanged) {
    container.innerHTML = "";
    const entries = _S.analyzeHistory;
    if (!entries.length) {
        container.appendChild(_mk("div", { style: { color: "#888", fontSize: "11px", padding: "4px 0" } }, ["No analysis runs yet."]));
        return;
    }
    entries.forEach(entry => {
        const ts   = entry.timestamp?.replace("T", " ").slice(0, 16) ?? "";
        const n    = entry.candidates?.length ?? 0;
        const head = _mk("div", { style: { display: "flex", justifyContent: "space-between", alignItems: "center",
            padding: "4px 0", borderBottom: "1px solid var(--p-surface-border,#444)", cursor: "pointer",
            fontSize: "11px" } }, [
            _mk("span", {}, [`${PASS_LABELS[entry.pass_type] ?? entry.pass_type} — ${n} candidate(s)`]),
            _mk("span", { style: { color: "#888" } }, [ts]),
        ]);
        const cands = _mk("div", { cls: "spe-collapsible collapsed" });
        head.onclick = () => {
            cands.classList.toggle("collapsed");
            if (!cands.classList.contains("collapsed") && !cands.childElementCount) {
                _S.analyzeCandidates = entry.candidates ?? [];
                _renderCandidates(cands, profile, onSubjectsChanged);
            }
        };
        container.appendChild(head);
        container.appendChild(cands);
    });
}

function _acceptOne(candidate, profile, onSubjectsChanged) {
    profile.subjects = profile.subjects || [];
    profile.subjects.push({
        id:               _genSubjId(candidate.label),
        label:            candidate.label,
        role_description: candidate.role_description || "",
        entity_type:      candidate.entity_type || "object",
        notes:            candidate.notes || "",
    });
    onSubjectsChanged();
}

function _acceptAll(candidates, profile, onSubjectsChanged, container) {
    const existingLabels = new Set((profile.subjects || []).map(s => s.label.toLowerCase()));
    candidates.forEach(c => {
        if (!existingLabels.has(c.label.toLowerCase())) {
            _acceptOne(c, profile, () => {});
        }
    });
    onSubjectsChanged();
    _toast(`Added ${candidates.length} subjects`, "success");
    container.querySelectorAll(".spe-btn.primary").forEach(b => {
        b.disabled = true;
        b.textContent = "Added";
    });
}

// ── Save ───────────────────────────────────────────────────────────────────────

async function _saveProfile(rootEl, updated) {
    try {
        await sourceProfilesApi.save(updated);
        await sourceProfilesApi.reload();
        // Update local state
        const idx = _S.profiles.findIndex(p => p.id === updated.id);
        if (idx >= 0) _S.profiles[idx] = updated;
        else _S.profiles.push(updated);
        _toast("Profile saved", "success");
    } catch (err) {
        _toast(`Save failed: ${err.message}`, "error");
    }
}

// ── List view ──────────────────────────────────────────────────────────────────

function _renderList(root) {
    root.innerHTML = "";

    // Toolbar
    const searchEl = _mk("input", { type: "text", placeholder: "Search profiles…", value: _S.filterText });
    searchEl.oninput = () => { _S.filterText = searchEl.value; _renderList(root); };

    const newBtn = _mk("button", { cls: "spe-btn primary", onclick: () => _createNewProfile(root) }, ["+ New"]);
    root.appendChild(_mk("div", { cls: "spe-toolbar" }, [searchEl, newBtn]));

    // List
    const listEl = _mk("div", { cls: "spe-list" });
    root.appendChild(listEl);

    const q = _S.filterText.toLowerCase();
    const filtered = _S.profiles.filter(p =>
        !q || (p.name || p.id || "").toLowerCase().includes(q)
    );

    if (!filtered.length) {
        listEl.appendChild(_mk("div", { style: { padding: "16px", color: "#666", fontSize: "12px" } },
            [_S.profiles.length ? "No profiles match." : "No source profiles yet. Click + New to create one."]));
    }

    filtered.forEach(p => {
        const subjects = p.subjects || [];
        const chips = subjects.slice(0, 4).map(s =>
            _mk("span", { cls: "spe-subject-chip" }, [
                (() => { const i = _mk("i", {}); i.className = (ENTITY_ICONS[s.entity_type] || "pi pi-circle") + " spe-subj-icon"; i.style.fontSize = "10px"; return i; })(),
                s.label || s.id,
            ])
        );
        if (subjects.length > 4) chips.push(_mk("span", { cls: "spe-subject-chip" }, [`+${subjects.length - 4} more`]));

        const card = _mk("div", { cls: "spe-profile-card" + (p.id === _S.selected ? " active" : ""),
            onclick: () => { _S.selected = p.id; _S.editingSubject = null; _S.analyzeCandidates = []; _renderDetail(root); }
        }, [
            _mk("div", { cls: "spe-card-header" }, [
                _mk("i", { cls: "pi pi-film" }),
                _mk("span", { cls: "spe-card-name" }, [p.name || p.id]),
                _mk("span", { cls: "spe-card-meta" }, [`${p.media_type || "video"} · ${subjects.length} subj`]),
            ]),
            _mk("div", { cls: "spe-card-body" }, chips.length ? chips : [
                _mk("span", { style: { color: "#666", fontSize: "11px" } }, ["No subjects annotated yet"]),
            ]),
        ]);
        listEl.appendChild(card);
    });
}

function _createNewProfile(root) {
    const name = "New Source Profile";
    const id   = _genId(name);
    const profile = {
        id, name, media_filename: "", media_dir: "input", media_type: "video",
        subjects: [], clips: [], default_segment_duration: null,
    };
    _S.profiles.push(profile);
    _S.selected = id;
    _S.editingSubject = null;
    _S.analyzeCandidates = [];
    _renderDetail(root);
}

// ── Main render ────────────────────────────────────────────────────────────────

function _render(root) {
    if (_S.selected) {
        _renderDetail(root);
    } else {
        _renderList(root);
    }
}

// ── Public entry point ─────────────────────────────────────────────────────────

export async function renderSourceProfileEditor(container) {
    _injectCSS();
    container.innerHTML = "";

    const panel = _mk("div", { cls: "spe-panel", "data-fbt-editor": "source-profiles" });
    container.appendChild(panel);

    // Show loading state
    panel.appendChild(_mk("div", { style: { padding: "16px", color: "#888", fontSize: "12px" } },
        ["Loading source profiles…"]));

    await _loadAll();

    panel.innerHTML = "";
    _render(panel);
}
