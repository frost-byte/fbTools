/**
 * Source Profile Editor — sidebar panel.
 *
 * Catalog browser for media-first subject profiles. Each profile points at
 * one source video or image and annotates the identifiable subjects inside it
 * (people, objects, locations, animals, soundscapes).
 *
 * Supports manual annotation and LLM-assisted focused-pass analysis.
 */

import { sourceProfilesApi }        from "../api/source_profiles.js";
import { bundlesApi }               from "../api/bundles.js";
import { getActiveCaptionerType }   from "./llm_panel.js";
import { api }                      from "../../../scripts/api.js";

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

// captioner_type is determined by the globally active VLM backend (LLM tab)

const PASS_DEFAULT_PROMPTS = {
    people:
        "Examine this image carefully. Identify every distinct person visible.\n\n" +
        "For each person:\n" +
        "- Disambiguate them by position, clothing colour, or prominent feature " +
        "(e.g. 'woman in blue top, stage left', 'man seated at table, background right').\n" +
        "- role_description: 1-2 sentences describing their appearance and position in the scene.\n" +
        "- entity_type must be 'person'.",
    setting:
        "Examine the environment and setting of this image. Identify the location and its " +
        "distinct visual elements — room type, architecture, furniture layout, lighting quality, " +
        "colour palette, time of day, weather if visible.\n\n" +
        "Treat the overall setting as one subject labelled by the most specific location name " +
        "you can give ('corner booth in a dimly-lit diner', not just 'restaurant').\n" +
        "If distinct sub-zones are visible (foreground vs background, stage vs audience), " +
        "list them as separate subjects.\n" +
        "entity_type must be 'location'.",
    soundscape:
        "This image is a frame from a video. Based on the visual cues visible — " +
        "instruments, speakers, crowds, machinery, natural environment, signage — " +
        "infer what audio layers are likely present in the original video.\n\n" +
        "List each distinct audio layer as a separate subject " +
        "(e.g. 'ambient cafe chatter', 'acoustic guitar performance', 'traffic from open window').\n" +
        "entity_type must be 'soundscape'.",
    objects:
        "Examine this image for significant objects and props. Focus on items that are " +
        "visually prominent, narratively important, or that a director would specifically " +
        "reference when describing the scene (e.g. 'the red suitcase', 'the vintage typewriter', " +
        "'the chess board on the table').\n\n" +
        "Exclude generic furniture unless it is a featured prop. " +
        "entity_type must be 'object'.",
    animals:
        "Examine this image for any animals. Identify each distinct animal visible, " +
        "including pets, wildlife, birds, fish in tanks, insects if prominent.\n\n" +
        "entity_type must be 'animal'.",
    custom: "(No default — enter your full prompt above.)",
};

const CLIP_COLORS = ["#3b82f6","#10b981","#f59e0b","#ef4444","#8b5cf6","#06b6d4","#f97316","#ec4899"];

// ── Module state ───────────────────────────────────────────────────────────────

const _S = {
    profiles:       [],     // [{id, name, media_filename, media_dir, media_type, subjects:[]}]
    mediaVideos:    [],     // filenames from input dir
    mediaImages:    [],
    mediaVideosOut: [],
    mediaImagesOut: [],
    lorasList:      [],     // LoRA filenames from /fbtools/loras/list
    filterText:     "",
    selected:       null,   // profile id currently open in detail view
    editingSubject: null,   // {idx, data} or null (new = idx === -1)
    analyzeOpen:    false,
    analyzePassType: "people",
    analyzePromptOverride: "",
    analyzeCaptioner: "auto",  // kept for potential legacy; overridden by getActiveCaptionerType()
    analyzeRunning:  false,
    analyzeCandidates: [],
    analyzeHistory:  [],
    analyzeClipIdx:  null,   // index into profile.clips, or null for single-frame mode
    analyzeMaxFrames: 20,
    analyzeSelectNth: 1,
    historyOpen:     false,
    settingsOpen:    true,
    subjectsOpen:    true,
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

// Extract the most useful error string from an APIError or generic Error.
// APIError.response is the raw server response body (may be JSON with an "error" key).
function _errMsg(err) {
    try {
        if (err?.response) {
            const body = typeof err.response === "string" ? JSON.parse(err.response) : err.response;
            if (body?.error) return body.error;
        }
    } catch (_) {}
    return err?.message || String(err);
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
    const [pr, vIn, vOut, iIn, iOut, lorasRes] = await Promise.allSettled([
        sourceProfilesApi.list(),
        bundlesApi.listMedia("video", true, "input"),
        bundlesApi.listMedia("video", true, "output"),
        bundlesApi.listMedia("image", true, "input"),
        bundlesApi.listMedia("image", true, "output"),
        fetch("/fbtools/loras/list").then(r => r.json()),
    ]);
    _S.profiles       = pr.value?.profiles    ?? [];
    _S.mediaVideos    = vIn.value?.files      ?? [];
    _S.mediaVideosOut = vOut.value?.files     ?? [];
    _S.mediaImages    = iIn.value?.files      ?? [];
    _S.mediaImagesOut = iOut.value?.files     ?? [];
    _S.lorasList      = lorasRes.value?.loras ?? [];
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
.spe-clip-field-label { font-size:11px; color:var(--p-text-muted-color,#888); margin-bottom:2px; }
.spe-clip-subj-list { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:6px; }
.spe-clip-subj-check { display:flex; align-items:center; gap:4px; font-size:11px;
    cursor:pointer; color:var(--p-text-color,#eee); }
.spe-clips-det-note { font-size:11px; color:#fbbf24; margin-top:4px; }
.spe-clips-llm-note { font-size:11px; color:var(--p-text-muted-color,#888); flex:1; }
.spe-clip-nav { display:flex; align-items:center; justify-content:center; gap:8px; margin-bottom:6px; }
.spe-clip-nav-label { font-size:12px; color:var(--p-text-muted-color,#888); min-width:60px; text-align:center; }

/* Clip LoRA editor */
.spe-lora-section { margin-top:6px; border-top:1px solid var(--p-surface-border,#333); padding-top:6px; }
.spe-lora-header { display:flex; align-items:center; justify-content:space-between;
    font-size:11px; color:var(--p-text-muted-color,#888); margin-bottom:4px; }
.spe-lora-header span { font-weight:600; }
.spe-lora-row { display:flex; gap:4px; align-items:center; margin-bottom:3px; }
.spe-lora-name { flex:1; padding:3px 5px; border-radius:3px; font-size:11px;
    border:1px solid var(--p-surface-border,#555);
    background:var(--p-surface-ground,#1a1a1a); color:var(--p-text-color,#eee); }
.spe-lora-strength { width:56px; padding:3px 5px; border-radius:3px; font-size:11px;
    border:1px solid var(--p-surface-border,#555);
    background:var(--p-surface-ground,#1a1a1a); color:var(--p-text-color,#eee); }
.spe-lora-rm { background:transparent; border:none; color:var(--p-red-400,#f87171);
    cursor:pointer; font-size:13px; padding:0 3px; line-height:1; }
.spe-lora-rm:hover { color:var(--p-red-300,#fca5a5); }
.spe-lora-empty { font-size:11px; color:var(--p-text-muted-color,#666);
    font-style:italic; margin-bottom:3px; }
.spe-lora-add { background:transparent; border:1px dashed var(--p-surface-border,#444);
    border-radius:3px; width:100%; padding:2px 0; font-size:11px;
    color:var(--p-blue-400,#60a5fa); cursor:pointer; margin-top:2px; }
.spe-lora-add:hover { border-color:var(--p-blue-400,#60a5fa); }

/* LoRA name dropdown (fixed-position, appended to body) */
.spe-lora-dropdown { position:fixed; background:var(--p-surface-d,#2a2a2a);
    border:1px solid var(--p-surface-border,#555); border-radius:4px;
    max-height:200px; overflow-y:auto; z-index:9999; box-shadow:0 4px 12px rgba(0,0,0,.4); }
.spe-lora-dd-item { padding:4px 8px; font-size:11px; cursor:pointer;
    color:var(--p-text-color,#eee); white-space:nowrap; overflow:hidden;
    text-overflow:ellipsis; }
.spe-lora-dd-item:hover, .spe-lora-dd-item.active { background:var(--p-primary-color,#3b82f6); color:#fff; }
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

function _mergeLoras(lorasA, lorasB) {
    const merged = lorasA.map(l => ({ ...l }));
    for (const lb of lorasB) {
        const ex = merged.find(la => la.name === lb.name);
        if (ex) {
            ex.strength_model = Math.max(ex.strength_model ?? 1, lb.strength_model ?? 1);
            ex.strength_clip  = Math.max(ex.strength_clip  ?? 1, lb.strength_clip  ?? 1);
        } else {
            merged.push({ ...lb });
        }
    }
    return merged;
}

function _drawTimeline(canvas, clips, suggestions, totalDuration, activeIdx = -1, dirtyIds = null, hoverIdx = -1) {
    const W = canvas.width;
    const H = canvas.height;
    const ctx = canvas.getContext("2d");
    const dur = totalDuration > 0 ? totalDuration : 1;
    const toX = t => (t / dur) * W;

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#111";
    ctx.fillRect(0, 0, W, H);

    const BAND_TOP = 10, BAND_BOT = H - 20;
    const HOVER_RISE = 6;  // px the hovered band extends above BAND_TOP

    // Clip bands
    clips.forEach((clip, i) => {
        const x1 = toX(clip.start_time);
        const x2 = toX(clip.end_time);
        const clipW = x2 - x1;
        const col = CLIP_COLORS[i % CLIP_COLORS.length];
        const isActive  = i === activeIdx;
        const isHovered = i === hoverIdx && !isActive;
        const bandTop   = isHovered ? BAND_TOP - HOVER_RISE : BAND_TOP;

        ctx.fillStyle = isActive ? col + "55" : isHovered ? col + "44" : col + "22";
        ctx.fillRect(x1, bandTop, clipW, BAND_BOT - bandTop);
        ctx.strokeStyle = col;
        ctx.lineWidth = isActive ? 2.5 : isHovered ? 1.5 : 1;
        ctx.strokeRect(x1 + 0.5, bandTop + 0.5, clipW - 1, BAND_BOT - bandTop - 1);

        // Active indicator: small filled triangle above the band
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
        // Dirty indicator: small amber dot inside band bottom-right
        if (dirtyIds?.has(clip.id)) {
            ctx.fillStyle = "#f59e0b";
            ctx.beginPath();
            ctx.arc(x2 - 6, BAND_BOT - 5, 3, 0, Math.PI * 2);
            ctx.fill();
        }
        // Label
        const label = clip.label || `Clip ${i + 1}`;
        const mid = (x1 + x2) / 2;
        ctx.fillStyle = isActive ? "#fff" : isHovered ? "#eee" : "#aaa";
        ctx.font = (isActive || isHovered) ? "bold 10px sans-serif" : "10px sans-serif";
        ctx.textAlign = "center";
        const textW = ctx.measureText(label).width;

        if (isHovered && clipW < 32) {
            // Narrow hovered clip: draw a flag-pole + label pill above the band
            const poleX = Math.max(x1 + 1, Math.min(x2 - 1, mid));
            ctx.strokeStyle = col + "bb";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(poleX, bandTop - 1);
            ctx.lineTo(poleX, 2);
            ctx.stroke();
            const pillPad = 3, pillH = 12;
            const pillW = textW + pillPad * 2;
            const pillX = Math.max(1, Math.min(W - pillW - 1, poleX - pillW / 2));
            ctx.fillStyle = col + "dd";
            ctx.beginPath();
            ctx.roundRect?.(pillX, 1, pillW, pillH, 3) || ctx.rect(pillX, 1, pillW, pillH);
            ctx.fill();
            ctx.fillStyle = "#fff";
            ctx.font = "bold 9px sans-serif";
            ctx.fillText(label, pillX + pillPad + textW / 2, 10);
        } else if (textW <= clipW - 6) {
            ctx.fillText(label, mid, (BAND_TOP + BAND_BOT) / 2 + 4);
        }
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

// ── Clip LoRA editor ──────────────────────────────────────────────────────────

function _loraDisplayName(val) {
    return val ? val.replace(/\.[^.]+$/, "").split(/[\\/]/).pop() : "";
}

function _makeLoraNameInput(currentValue, onChange) {
    let selected = currentValue || "";
    let dropdown = null;

    const wrap  = _mk("div", { style: { flex: "1", position: "relative" } });
    const input = _mk("input", {
        cls: "spe-lora-name",
        type: "text",
        placeholder: "— select LoRA —",
        value: _loraDisplayName(selected),
        autocomplete: "off",
    });
    input.setAttribute("spellcheck", "false");
    wrap.appendChild(input);

    function _close() {
        dropdown?.remove();
        dropdown = null;
        input.value = _loraDisplayName(selected);
    }

    function _open(query) {
        dropdown?.remove();
        const q = query.trim().toLowerCase();
        const matches = q
            ? _S.lorasList.filter(n => n.toLowerCase().includes(q))
            : _S.lorasList;

        const list = _mk("div", { cls: "spe-lora-dropdown" });
        if (!q) {
            const none = _mk("div", { cls: "spe-lora-dd-item" + (!selected ? " active" : ""),
                textContent: "— none —" });
            none.addEventListener("mousedown", e => {
                e.preventDefault();
                selected = "";
                _close();
                onChange("");
            });
            list.appendChild(none);
        }
        if (!matches.length && q) {
            list.appendChild(_mk("div", { cls: "spe-lora-dd-item", textContent: "No matches" }));
        }
        matches.forEach(n => {
            const item = _mk("div", {
                cls: "spe-lora-dd-item" + (n === selected ? " active" : ""),
                title: n,
                textContent: _loraDisplayName(n),
            });
            item.addEventListener("mousedown", e => {
                e.preventDefault();
                selected = n;
                _close();
                onChange(n);
            });
            list.appendChild(item);
        });

        const rect = input.getBoundingClientRect();
        Object.assign(list.style, {
            left:  `${rect.left}px`,
            top:   `${rect.bottom + 2}px`,
            width: `${Math.max(rect.width, 180)}px`,
        });
        document.body.appendChild(list);
        dropdown = list;
        list.querySelector(".active")?.scrollIntoView({ block: "nearest" });
    }

    input.addEventListener("focus",  ()  => _open(""));
    input.addEventListener("input",  ()  => _open(input.value));
    input.addEventListener("blur",   ()  => setTimeout(_close, 150));
    input.addEventListener("keydown", e => {
        if (!dropdown) return;
        const items = Array.from(dropdown.querySelectorAll(".spe-lora-dd-item"));
        let idx = items.findIndex(el => el.classList.contains("active"));
        if (e.key === "ArrowDown") {
            e.preventDefault();
            items[idx]?.classList.remove("active");
            items[Math.min(idx + 1, items.length - 1)]?.classList.add("active");
            dropdown.querySelector(".active")?.scrollIntoView({ block: "nearest" });
        } else if (e.key === "ArrowUp") {
            e.preventDefault();
            items[idx]?.classList.remove("active");
            items[Math.max(idx - 1, 0)]?.classList.add("active");
            dropdown.querySelector(".active")?.scrollIntoView({ block: "nearest" });
        } else if (e.key === "Enter") {
            e.preventDefault();
            dropdown.querySelector(".active")?.dispatchEvent(new MouseEvent("mousedown"));
        } else if (e.key === "Escape") {
            e.stopPropagation();
            _close();
        }
    });

    return wrap;
}

function _buildClipLoraSection(clip, onCommit) {
    const section = _mk("div", { cls: "spe-lora-section" });
    const header  = _mk("div", { cls: "spe-lora-header" }, [
        _mk("span", {}, ["LoRAs"]),
    ]);

    const listEl = _mk("div");
    section.appendChild(header);
    section.appendChild(listEl);

    function _rebuild() {
        listEl.innerHTML = "";
        const loras = clip.loras || [];
        if (!loras.length) {
            listEl.appendChild(_mk("div", { cls: "spe-lora-empty" }, ["No LoRAs attached."]));
        }
        loras.forEach((entry, i) => {
            const nameWrap = _makeLoraNameInput(entry.name, val => {
                loras[i] = { ...loras[i], name: val };
                clip.loras = loras;
                onCommit();
            });

            const strengthInp = _mk("input", {
                cls: "spe-lora-strength",
                type: "number", min: 0, max: 2, step: 0.05,
                value: entry.strength_model ?? 1.0,
                title: "Strength (applied to both model and clip)",
            });
            strengthInp.addEventListener("change", () => {
                const v = parseFloat(strengthInp.value) || 1.0;
                loras[i] = { ...loras[i], strength_model: v, strength_clip: v };
                clip.loras = loras;
                onCommit();
            });

            const rmBtn = _mk("button", { cls: "spe-lora-rm", title: "Remove", textContent: "✕" });
            rmBtn.addEventListener("click", () => {
                loras.splice(i, 1);
                clip.loras = loras;
                onCommit();
                _rebuild();
            });

            listEl.appendChild(_mk("div", { cls: "spe-lora-row" }, [nameWrap, strengthInp, rmBtn]));
        });

        const addBtn = _mk("button", { cls: "spe-lora-add", textContent: "+ Add LoRA" });
        addBtn.addEventListener("click", () => {
            loras.push({ name: "", strength_model: 1.0, strength_clip: 1.0 });
            clip.loras = loras;
            onCommit();
            _rebuild();
        });
        listEl.appendChild(addBtn);
    }

    _rebuild();
    return section;
}

function _renderClipsSection(container, profile, onClipsChanged, onEnsureSaved, onSelect = null) {
    const wrap = _mk("div", { cls: "spe-clips" });
    container.appendChild(wrap);

    let clipsOpen = false;
    let suggestions = [];
    let inferredSubjects = [];
    let detecting = false;
    let autoSegRunning = false;
    // captioner_type for clip requests comes from the global active backend (LLM tab)
    let videoDuration = 0;
    let activeClipIdx = 0;
    let hoverClipIdx  = -1;
    let detectFlags = { camera_cuts: true, subject_changes: false, lower_threshold: false };
    let detectPromptOverride = "";
    let lastRawResponse = "";
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

    // Proxy short-edge selector — must be a multiple of 32 for H3 compatibility
    const _PROXY_EDGES = [480, 576, 640, 768, 1080];
    const proxyEdgeSel = _mk("select", { cls: "spe-clip-sel",
        title: "Proxy video resolution (shorter edge). Must be a multiple of 32. Lower = faster; 768 recommended for H3 reference." });
    _PROXY_EDGES.forEach(v => {
        const o = _mk("option", { value: String(v) }, [String(v) + "px"]);
        if (v === (profile.proxy_short_edge ?? 768)) o.selected = true;
        proxyEdgeSel.appendChild(o);
    });
    proxyEdgeSel.addEventListener("change", () => {
        profile.proxy_short_edge = parseInt(proxyEdgeSel.value, 10);
        // Persist immediately so the next proxy build uses the new value
        sourceProfilesApi.save(profile).catch(err => _toast(`Save failed: ${err.message}`, "error"));
    });

    const autoSegBtn = _mk("button", { cls: "spe-btn sm", onclick: runAutoSegment }, ["Auto-segment"]);

    body.appendChild(_mk("div", { cls: "spe-clips-toolbar" }, [
        _mk("label", {}, ["Duration (s):"]), durationInput,
        _mk("label", {}, ["Seg (s):"]), segDurInput,
        autoSegBtn,
    ]));
    body.appendChild(_mk("div", { cls: "spe-clips-toolbar", style: { marginTop: "4px" } }, [
        _mk("label", { title: "Proxy resolution — shorter edge in pixels (must be ÷32)" }, ["Proxy edge:"]),
        proxyEdgeSel,
    ]));

    // Detect row — active backend badge + detect button
    const backendNote = _mk("span", {
        cls: "spe-clips-llm-note",
        style: { fontSize: "11px", cursor: "pointer", textDecoration: "underline dotted" },
        title: "Configure in LLM tab",
        onclick: () => window._fbtActivateTab?.("llm"),
    }, [getActiveCaptionerType()]);

    const detectBtn = _mk("button", { cls: "spe-btn sm ghost", onclick: runDetect }, ["Detect boundaries"]);
    const detectSpinner = _mk("span", { style: { fontSize: "11px", color: "#888", display: "none" } }, [" Detecting…"]);
    let detNoteEl = null;

    const applySugBtn = _mk("button", {
        cls: "spe-btn sm",
        onclick: runApplySuggestions,
        disabled: true,
        title: "Create clips from detected segment boundaries, pre-filling action descriptions",
    }, ["Apply suggestions"]);

    body.appendChild(_mk("div", { cls: "spe-clips-toolbar" }, [
        backendNote, detectBtn, detectSpinner, applySugBtn,
    ]));

    // Detect flags row
    const _mkFlagCb = (key, label, defaultOn) => {
        const cb  = _mk("input", { type: "checkbox", id: `spe-flag-${key}-${profile.id || "new"}` });
        cb.checked = defaultOn;
        cb.onchange = () => { detectFlags[key] = cb.checked; };
        const lbl = _mk("label", { htmlFor: cb.id, style: { fontSize: "11px" } }, [label]);
        return [cb, lbl];
    };
    const [camCb, camLbl]     = _mkFlagCb("camera_cuts",    "Camera cuts",     true);
    const [subjCb, subjLbl]   = _mkFlagCb("subject_changes","Subject changes",  false);
    const [lowCb, lowLbl]     = _mkFlagCb("lower_threshold","More boundaries",  false);
    body.appendChild(_mk("div", { cls: "spe-clips-toolbar", style: { flexWrap: "wrap", gap: "6px" } }, [
        _mk("span", { style: { fontSize: "11px", color: "#888" } }, ["Flags:"]),
        camCb, camLbl, subjCb, subjLbl, lowCb, lowLbl,
    ]));

    // Prompt override textarea (collapsible)
    const promptToggle = _mk("button", { cls: "spe-btn sm ghost", style: { fontSize: "11px" } }, ["▸ Prompt override"]);
    const promptWrap   = _mk("div", { style: { display: "none", marginTop: "4px" } });
    const promptTa     = _mk("textarea", { placeholder: "Leave empty to use flags above…",
        rows: 4, style: { width: "100%", fontSize: "11px", resize: "vertical",
                          background: "var(--bg2)", color: "var(--fg)", border: "1px solid var(--border)",
                          borderRadius: "4px", padding: "4px", boxSizing: "border-box" } });
    promptTa.oninput = () => { detectPromptOverride = promptTa.value; };
    promptWrap.appendChild(promptTa);
    promptToggle.onclick = () => {
        const open = promptWrap.style.display === "none";
        promptWrap.style.display = open ? "block" : "none";
        promptToggle.textContent = (open ? "▾ " : "▸ ") + "Prompt override";
    };
    body.appendChild(_mk("div", { style: { padding: "2px 0" } }, [promptToggle, promptWrap]));

    // Raw response section (shown after detection)
    const rawWrap = _mk("div", { style: { display: "none", marginTop: "6px" } });
    const rawToggle = _mk("button", { cls: "spe-btn sm ghost", style: { fontSize: "11px" } }, ["▸ Raw VLM response"]);
    const rawPre    = _mk("pre", { style: { fontSize: "10px", whiteSpace: "pre-wrap", wordBreak: "break-all",
        maxHeight: "180px", overflowY: "auto", background: "var(--bg2)", padding: "6px",
        border: "1px solid var(--border)", borderRadius: "4px", marginTop: "4px", display: "none" } });
    rawToggle.onclick = () => {
        const open = rawPre.style.display === "none";
        rawPre.style.display = open ? "block" : "none";
        rawToggle.textContent = (open ? "▾ " : "▸ ") + "Raw VLM response";
    };
    rawWrap.appendChild(rawToggle);
    rawWrap.appendChild(rawPre);
    body.appendChild(rawWrap);

    // ── Timeline canvas ───────────────────────────────────────────────────────
    const canvasWrap = _mk("div", { cls: "spe-timeline-wrap", style: { position: "relative" } });
    const canvas = document.createElement("canvas");
    canvas.className = "spe-timeline";
    canvas.height = 72;
    canvasWrap.appendChild(canvas);

    // Floating tooltip for narrow clips that are hard to identify on hover
    const hoverTooltipEl = _mk("div", { style: {
        position: "absolute", top: "-26px", left: "0",
        background: "#1a1a2e", border: "1px solid #555", borderRadius: "4px",
        padding: "3px 8px", fontSize: "11px", color: "#ddd",
        pointerEvents: "none", whiteSpace: "nowrap",
        boxShadow: "0 2px 8px rgba(0,0,0,.5)", display: "none", zIndex: "10",
    }});
    canvasWrap.appendChild(hoverTooltipEl);

    body.appendChild(canvasWrap);

    // ── Describe prompt override ──────────────────────────────────────────────
    const DEFAULT_DESCRIBE_PROMPT =
        "Examine this video frame carefully.  Describe in 1-2 sentences " +
        "what action or situation is visually depicted — focus on what the " +
        "subjects are doing or what is occurring in the scene at this moment.";

    const promptOverrideWrap = _mk("div", { style: { marginBottom: "6px" } });
    const promptOverrideLabel = _mk("label", {
        style: { fontSize: "11px", color: "#888", display: "block", marginBottom: "2px" },
        title: "Replaces the opening instruction sent to the VLM when you click Describe. Leave blank to use the default. The JSON schema is always appended after this text.",
    }, ["Describe instruction (leave blank for default)"]);
    const promptOverrideTa = _mk("textarea", {
        cls: "spe-clip-ta",
        rows: 3,
        placeholder: DEFAULT_DESCRIBE_PROMPT,
        title: "Replaces the opening instruction sent to the VLM when you click Describe. Leave blank to use the default. The JSON schema is always appended after this text.",
    });
    promptOverrideTa.value = profile.describe_prompt_override || "";

    let _promptSaveTimer = null;
    promptOverrideTa.onchange = () => {
        profile.describe_prompt_override = promptOverrideTa.value.trim();
        clearTimeout(_promptSaveTimer);
        _promptSaveTimer = setTimeout(() => {
            sourceProfilesApi.save(profile)
                .catch(err => _toast(`Prompt override save failed: ${err.message}`, "error"));
        }, 800);
    };
    promptOverrideWrap.append(promptOverrideLabel, promptOverrideTa);
    body.appendChild(promptOverrideWrap);

    // ── Clip list ─────────────────────────────────────────────────────────────
    const listEl = _mk("div");
    body.appendChild(listEl);

    const addClipBtn = _mk("button", { cls: "spe-btn sm", style: { marginTop: "4px" },
        onclick: addNewClip }, ["+ Add clip"]);
    body.appendChild(addClipBtn);

    // ── Toggle ────────────────────────────────────────────────────────────────
    // ── Build-all-proxies button (sits in the title row) ─────────────────────
    const buildAllBtn = _mk("button", { cls: "spe-btn sm ghost",
        title: "Pre-build proxies for all clips in this profile",
        style: { marginRight: "4px" },
        onclick: async e => {
            e.stopPropagation();
            buildAllBtn.disabled = true;
            buildAllBtn.textContent = "building…";
            try {
                const res = await sourceProfilesApi.prebuildProxies({ profile_id: profile.id });
                const n = res.clip_count || 0;
                _toast(`Building ${n} proxy clip${n !== 1 ? "s" : ""} in background`, "info");
                // Refresh status after estimated completion time (rough: 5s/clip)
                const delay = Math.max(4000, n * 5000);
                setTimeout(() => _refreshProxyStatus(), delay);
            } catch (err) {
                _toast(`Proxy build failed: ${err.message}`, "error");
            } finally {
                buildAllBtn.disabled = false;
                buildAllBtn.textContent = "build proxies";
            }
        } }, ["build proxies"]);
    titleRow.insertBefore(buildAllBtn, chevron);

    titleRow.onclick = () => {
        clipsOpen = !clipsOpen;
        body.classList.toggle("collapsed", !clipsOpen);
        chevron.className = "pi pi-chevron-" + (clipsOpen ? "up" : "down");
        if (clipsOpen) {
            probeDuration();
            requestAnimationFrame(redraw);
            _refreshProxyStatus();
        } else {
            if (hiddenVideo) { hiddenVideo.remove(); hiddenVideo = null; }
        }
    };

    // ── Redraw ────────────────────────────────────────────────────────────────
    function redraw() {
        canvas.width = canvas.offsetWidth || 380;
        _drawTimeline(canvas, clips, suggestions, getTotalDuration(), activeClipIdx,
            new Set(clips.filter(_isProxyDirty).map(c => c.id)), hoverClipIdx);
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

    function _updateHoverTooltip(e, totalDur) {
        if (hoverClipIdx < 0) { hoverTooltipEl.style.display = "none"; return; }
        const clip = clips[hoverClipIdx];
        const x1 = (clip.start_time / totalDur) * canvas.offsetWidth;
        const x2 = (clip.end_time   / totalDur) * canvas.offsetWidth;
        if (x2 - x1 >= 40) { hoverTooltipEl.style.display = "none"; return; }
        const dur = (clip.end_time - clip.start_time).toFixed(1);
        hoverTooltipEl.textContent =
            `${clip.label || `Clip ${hoverClipIdx + 1}`}  ${clip.start_time.toFixed(1)}–${clip.end_time.toFixed(1)}s  (${dur}s)`;
        const wrapRect = canvasWrap.getBoundingClientRect();
        const tipWidth = hoverTooltipEl.offsetWidth || 160;
        const mouseX   = e.clientX - wrapRect.left;
        hoverTooltipEl.style.left = Math.max(0, Math.min(canvas.offsetWidth - tipWidth, mouseX - tipWidth / 2)) + "px";
        hoverTooltipEl.style.display = "block";
    }

    canvas.addEventListener("mousemove", e => {
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const totalDur = getTotalDuration() || 1;
        const BAND_TOP = 10, BAND_BOT = canvas.height - 20;
        const y = e.clientY - rect.top;

        // Handle hover tracking (skip during active drag)
        if (!_drag) {
            let newHoverIdx = -1;
            if (y >= BAND_TOP - 6 && y <= BAND_BOT) {
                for (let i = 0; i < clips.length; i++) {
                    const x1 = (clips[i].start_time / totalDur) * canvas.offsetWidth;
                    const x2 = (clips[i].end_time   / totalDur) * canvas.offsetWidth;
                    if (x >= x1 && x <= x2) { newHoverIdx = i; break; }
                }
            }
            if (newHoverIdx !== hoverClipIdx) {
                hoverClipIdx = newHoverIdx;
                canvas.width = canvas.offsetWidth;
                _drawTimeline(canvas, clips, suggestions, totalDur, activeClipIdx,
                    new Set(clips.filter(_isProxyDirty).map(c => c.id)), hoverClipIdx);
            }
            _updateHoverTooltip(e, totalDur);
        }

        // Cursor: col-resize for boundary handles, but only when adjacent clips are wide enough to click
        let overHandle = false;
        for (let i = 1; i < clips.length; i++) {
            const hx = (clips[i].start_time / totalDur) * canvas.offsetWidth;
            if (Math.abs(x - hx) <= 8) {
                const leftW  = hx - (clips[i - 1].start_time / totalDur) * canvas.offsetWidth;
                const rightW = (clips[i].end_time / totalDur) * canvas.offsetWidth - hx;
                if (leftW >= 16 && rightW >= 16) overHandle = true;
                break;
            }
        }
        if (overHandle) {
            canvas.style.cursor = "col-resize";
        } else {
            let overBand = false;
            if (y >= BAND_TOP && y <= BAND_BOT) {
                for (let i = 0; i < clips.length; i++) {
                    const x1 = (clips[i].start_time / totalDur) * canvas.offsetWidth;
                    const x2 = (clips[i].end_time   / totalDur) * canvas.offsetWidth;
                    if (x >= x1 && x <= x2) { overBand = true; break; }
                }
            }
            canvas.style.cursor = overBand ? "pointer" : "default";
        }

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
        _drawTimeline(canvas, clips, suggestions, totalDur, activeClipIdx,
            new Set(clips.filter(_isProxyDirty).map(c => c.id)), hoverClipIdx);
    });

    canvas.addEventListener("mouseup", e => {
        if (_drag) {
            // End boundary drag
            const dragI = _drag.i;
            _drag = null;
            profile.clips = clips;
            onClipsChanged(clips);
            [dragI - 1, dragI].forEach(j => { if (clips[j]) _markTimesDirty(j); });
            renderClipList();
            return;
        }
        // No drag — treat as a click to select a clip band
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const totalDur = getTotalDuration() || 1;
        for (let i = 0; i < clips.length; i++) {
            const x1 = (clips[i].start_time / totalDur) * canvas.offsetWidth;
            const x2 = (clips[i].end_time / totalDur) * canvas.offsetWidth;
            if (x >= x1 && x <= x2) {
                activeClipIdx = i;
                redraw();
                onSelect?.(clips[i].start_time);
                break;
            }
        }
    });
    canvas.addEventListener("mouseleave", () => {
        hoverClipIdx = -1;
        hoverTooltipEl.style.display = "none";
        if (!_drag) return;
        const dragI = _drag.i;
        _drag = null;
        profile.clips = clips;
        onClipsChanged(clips);
        [dragI - 1, dragI].forEach(j => { if (clips[j]) _markTimesDirty(j); });
        renderClipList();
    });

    // ── Proxy status helpers ──────────────────────────────────────────────────
    let _proxyStatusMap = {}; // clip_id → { fresh: bool }

    // A clip is dirty when its timing was changed after the proxy was last built.
    // Both timestamps are ISO strings stored on the clip object itself and
    // persisted via _persistClip, so the state survives panel close/reopen.
    function _isProxyDirty(clip) {
        if (!clip.times_changed_at) return false;
        if (!clip.proxy_built_at)   return true;
        return clip.times_changed_at > clip.proxy_built_at;
    }

    // Mark a clip's timing as changed and persist the timestamp immediately.
    function _markTimesDirty(idx) {
        const now = new Date().toISOString();
        clips[idx] = { ...clips[idx], times_changed_at: now };
        profile.clips = clips;
        _persistClip(clips[idx]);
    }

    async function _refreshProxyStatus() {
        if (!profile.id) return;
        try {
            const res = await sourceProfilesApi.proxyStatus(profile.id);
            _proxyStatusMap = {};
            const now = new Date().toISOString();
            (res.clips || []).forEach(c => {
                _proxyStatusMap[c.clip_id] = c;
                if (c.fresh) {
                    const idx = clips.findIndex(cl => cl.id === c.clip_id);
                    if (idx >= 0 && !_isProxyDirty(clips[idx])) {
                        clips[idx] = { ...clips[idx], proxy_built_at: now };
                        profile.clips = clips;
                        _persistClip(clips[idx]);
                    }
                }
            });
            renderClipList();
        } catch { /* silent — proxy status is informational */ }
    }

    function _isProxyDirtyById(clip_id) {
        const clip = clips.find(c => c.id === clip_id);
        return clip ? _isProxyDirty(clip) : false;
    }

    function _proxyBadgeEl(clip_id) {
        const dirty = _isProxyDirtyById(clip_id);
        const info  = _proxyStatusMap[clip_id];
        const span  = _mk("span", { style: {
            fontSize: "9px", padding: "1px 5px", borderRadius: "3px",
            fontWeight: "600", letterSpacing: "0.04em",
            background: dirty          ? "var(--p-amber-800,#78350f)"
                       : info == null  ? "transparent"
                       : info.fresh    ? "var(--p-green-800,#166534)"
                                       : "var(--p-surface-600,#555)",
            color:      dirty          ? "var(--p-amber-300,#fcd34d)"
                       : info == null  ? "transparent"
                       : info.fresh    ? "var(--p-green-300,#86efac)"
                                       : "var(--p-surface-200,#ccc)",
        }}, [dirty ? "needs rebuild" : info == null ? "" : info.fresh ? "proxy ready" : "no proxy"]);
        return span;
    }

    // ── Clip list render — shows one clip at a time ───────────────────────────
    function renderClipList() {
        listEl.innerHTML = "";
        if (!clips.length) return;

        // Clamp in case clips were removed
        activeClipIdx = Math.max(0, Math.min(activeClipIdx, clips.length - 1));
        const i = activeClipIdx;

        // Nav bar: ← Segment N / M →
        const prevBtn  = _mk("button", { cls: "spe-btn sm ghost" }, ["←"]);
        const nextBtn  = _mk("button", { cls: "spe-btn sm ghost" }, ["→"]);
        const mergeBtn = _mk("button", { cls: "spe-btn sm ghost",
            title: "Merge with next clip — combines timing, subjects, action, dialogue flag, and LoRAs",
            style: { fontSize: "10px" } }, ["⊕ merge →"]);
        if (i === 0) prevBtn.disabled = true;
        if (i === clips.length - 1) { nextBtn.disabled = true; mergeBtn.disabled = true; }
        const navLabel = _mk("span", { cls: "spe-clip-nav-label" },
            [`${clips[i].label || "Segment " + (i + 1)}  (${i + 1}/${clips.length})`]);
        prevBtn.onclick  = () => { activeClipIdx = Math.max(0, i - 1); redraw(); onSelect?.(clips[activeClipIdx].start_time); };
        nextBtn.onclick  = () => { activeClipIdx = Math.min(clips.length - 1, i + 1); redraw(); onSelect?.(clips[activeClipIdx].start_time); };
        mergeBtn.onclick = () => mergeWithNext(i);
        listEl.appendChild(_mk("div", { cls: "spe-clip-nav" }, [prevBtn, navLabel, nextBtn, mergeBtn]));

        {   // single-clip block (braces preserve the original forEach-scoped variable names)
            const clip = clips[i];
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
                if (e > s) {
                    clips[i] = { ...clips[i], start_time: s, end_time: e };
                    _markTimesDirty(i);
                    commitClip(i);
                    renderClipList();
                }
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

            const soundscapeEl = _mk("textarea", { cls: "spe-clip-ta", rows: 2,
                placeholder: "Overall soundscape (ambient audio, room tone, environment sounds…)" });
            soundscapeEl.value = clip.overall_soundscape || "";
            soundscapeEl.onchange = () => { clips[i] = { ...clips[i], overall_soundscape: soundscapeEl.value }; commitClip(i); };

            const musicEl = _mk("textarea", { cls: "spe-clip-ta", rows: 2,
                placeholder: "Non-diegetic music (score, background music not heard by characters…)" });
            musicEl.value = clip.non_diegetic_music || "";
            musicEl.onchange = () => { clips[i] = { ...clips[i], non_diegetic_music: musicEl.value }; commitClip(i); };

            const subjWrap = _mk("div", { cls: "spe-clip-subj-list" });
            const CLIP_SLOTS = ["A","B","C","D","E","F","G","H","I","J"];
            const slotSpans = {};  // subjectId → span element showing "{A}" etc.

            const updateSlotLabels = () => {
                const tagged = clips[i].subjects || [];
                subjects.forEach(s => {
                    const span = slotSpans[s.id];
                    if (!span) return;
                    const idx = tagged.indexOf(s.id);
                    span.textContent = (idx >= 0 && idx < CLIP_SLOTS.length) ? ` {${CLIP_SLOTS[idx]}}` : "";
                });
            };

            subjects.forEach(s => {
                const cb = _mk("input", { type: "checkbox" });
                cb.checked = (clip.subjects || []).includes(s.id);
                const slotSpan = _mk("span", { style: { opacity: "0.6", fontFamily: "monospace", fontSize: "10px" } });
                slotSpans[s.id] = slotSpan;
                cb.onchange = () => {
                    const cur = new Set(clips[i].subjects || []);
                    if (cb.checked) cur.add(s.id); else cur.delete(s.id);
                    clips[i] = { ...clips[i], subjects: [...cur] };
                    commitClip(i);
                    updateSlotLabels();
                };
                subjWrap.appendChild(_mk("label", { cls: "spe-clip-subj-check" },
                    [cb, " " + (s.label || s.id), slotSpan]));
            });
            updateSlotLabels();

            const loraSection = _buildClipLoraSection(clips[i], () => commitClip(i));

            const dlgCb = _mk("input", { type: "checkbox" });
            dlgCb.checked = clip.allows_dialogue !== false;
            dlgCb.onchange = () => { clips[i] = { ...clips[i], allows_dialogue: dlgCb.checked }; commitClip(i); };

            const clipBody = _mk("div", { cls: "spe-clip-card-body" }, [
                _mk("div", { cls: "spe-clip-row", style: { marginBottom: "6px" } }, [labelInp]),
                _mk("div", { cls: "spe-clip-row" }, [
                    "Start:", startEl, "End:", endEl, durSpan,
                ]),
                actionEl,
                subjects.length ? subjWrap : null,
                _mk("div", { style: { display: "flex", gap: "4px" } }, [describeBtn]),
                _mk("div", { cls: "spe-clip-field-label", textContent: "Overall soundscape" }),
                soundscapeEl,
                _mk("div", { cls: "spe-clip-field-label", textContent: "Non-diegetic music" }),
                musicEl,
                _mk("label", { cls: "spe-clip-subj-check", style: { marginTop: "4px" }, title: "When off, dialogue from cast entries is ignored for this segment" }, [dlgCb, " Allows dialogue"]),
                loraSection,
            ]);

            const proxyBadge = _proxyBadgeEl(clip.id);
            const buildProxyBtn = _mk("button", { cls: "spe-btn sm ghost",
                title: "Pre-build proxy for this clip",
                onclick: async e => {
                    e.stopPropagation();
                    buildProxyBtn.disabled = true;
                    buildProxyBtn.textContent = "…";
                    try {
                        await sourceProfilesApi.prebuildProxies({ profile_id: profile.id, clip_id: clip.id });
                        setTimeout(() => _refreshProxyStatus(), 3000);
                        setTimeout(() => _refreshProxyStatus(), 8000);
                    } catch (err) {
                        _toast(`Proxy build failed: ${err.message}`, "error");
                    } finally {
                        buildProxyBtn.disabled = false;
                        buildProxyBtn.textContent = "proxy";
                    }
                } }, ["proxy"]);
            const head = _mk("div", { cls: "spe-clip-card-head" }, [
                _mk("span", { cls: "spe-clip-color-dot", style: { background: color } }),
                _mk("span", { style: { flex: "1", fontWeight: "600", fontSize: "12px" } },
                    [clip.label || `Clip ${i + 1}`]),
                _mk("span", { style: { color: "#888", fontSize: "11px" } },
                    [`${clip.start_time.toFixed(1)}–${clip.end_time.toFixed(1)}s`]),
                proxyBadge,
                buildProxyBtn,
                _mk("button", { cls: "spe-btn sm danger",
                    onclick: e => { e.stopPropagation(); removeClip(i); } }, ["×"]),
            ]);

            listEl.appendChild(_mk("div", { cls: "spe-clip-card" }, [head, clipBody]));
        }
    }

    // ── Operations ────────────────────────────────────────────────────────────
    function _persistClip(clip) {
        sourceProfilesApi.upsertClip({ profile_id: profile.id, clip })
            .catch(err => _toast(`Clip save failed: ${err.message}`, "error"));
    }

    function commitClip(i) {
        profile.clips = clips;
        onClipsChanged(clips);
        canvas.width = canvas.offsetWidth;
        _drawTimeline(canvas, clips, suggestions, getTotalDuration(), activeClipIdx,
            new Set(clips.filter(_isProxyDirty).map(c => c.id)), hoverClipIdx);
        _persistClip(clips[i]);
    }

    function removeClip(i) {
        const clipId = clips[i].id;
        clips = clips.filter((_, j) => j !== i);
        profile.clips = clips;
        onClipsChanged(clips);
        redraw();
        sourceProfilesApi.removeClip({ profile_id: profile.id, clip_id: clipId })
            .catch(err => _toast(`Clip remove failed: ${err.message}`, "error"));
    }

    function mergeWithNext(i) {
        if (i >= clips.length - 1) return;
        const a = clips[i];
        const b = clips[i + 1];
        const aSubjects = a.subjects || [];
        const merged = {
            ...a,
            end_time:           b.end_time,
            action:             [a.action, b.action].filter(Boolean).join("\n\n"),
            subjects:           [...aSubjects, ...(b.subjects || []).filter(id => !aSubjects.includes(id))],
            allows_dialogue:    (a.allows_dialogue !== false) || (b.allows_dialogue !== false),
            overall_soundscape: [a.overall_soundscape, b.overall_soundscape].filter(Boolean).join("\n\n") || undefined,
            non_diegetic_music: [a.non_diegetic_music, b.non_diegetic_music].filter(Boolean).join("\n\n") || undefined,
            loras:              _mergeLoras(a.loras || [], b.loras || []),
        };
        const removedId = b.id;
        clips = [...clips.slice(0, i), merged, ...clips.slice(i + 2)];
        profile.clips = clips;
        onClipsChanged(clips);
        _markTimesDirty(i);
        commitClip(i);
        sourceProfilesApi.removeClip({ profile_id: profile.id, clip_id: removedId })
            .catch(err => _toast(`Merge cleanup failed: ${err.message}`, "error"));
        _toast(`Merged → ${_fmtT(merged.start_time)}–${_fmtT(merged.end_time)} (${(merged.end_time - merged.start_time).toFixed(1)}s)`, "success");
    }

    function addNewClip() {
        const lastEnd = clips.length ? clips[clips.length - 1].end_time : 0;
        const segDur  = parseFloat(segDurInput.value) || (profile.default_segment_duration ?? 10);
        const newClip = {
            id: _genClipId(), label: `Clip ${clips.length + 1}`,
            start_time: lastEnd, end_time: lastEnd + segDur,
            select_every_nth: 2, frame_load_cap: 120, subjects: [], action: "",
        };
        clips.push(newClip);
        activeClipIdx = clips.length - 1;  // jump to the new clip
        profile.clips = clips;
        onClipsChanged(clips);
        redraw();
        _persistClip(newClip);
    }

    async function runAutoSegment() {
        const totalDur = getTotalDuration();
        if (totalDur <= 0) { _toast("Set video duration first", "warn"); return; }
        if (autoSegRunning) return;
        autoSegRunning = true;
        autoSegBtn.disabled = true;
        autoSegBtn.textContent = "…";
        try {
            await onEnsureSaved?.();
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
            _toast(`Auto-segment failed: ${_errMsg(err)}`, "error");
        } finally {
            autoSegRunning = false;
            autoSegBtn.disabled = false;
            autoSegBtn.textContent = "Auto-segment";
        }
    }

    async function runApplySuggestions() {
        if (!suggestions.length) return;
        const subjectIds = (profile.subjects || []).map(s => s.id || s).filter(Boolean);
        const defNth     = profile.default_select_every_nth ?? 1;
        const defCap     = profile.default_frame_load_cap   ?? 0;
        const newClips   = suggestions.map((seg, idx) => ({
            id:               `clip_${idx + 1}`,
            label:            seg.label  || `Segment ${idx + 1}`,
            start_time:       seg.start_time,
            end_time:         seg.end_time,
            action:           seg.action || "",
            subjects:         subjectIds,
            select_every_nth: defNth,
            frame_load_cap:   defCap,
        }));
        applySugBtn.disabled = true;
        applySugBtn.textContent = "…";
        try {
            await onEnsureSaved?.();
            // Apply clips
            const clipsRes = await sourceProfilesApi.setClips({ profile_id: profile.id, clips: newClips });
            clips = clipsRes.profile?.clips || [];
            profile.clips = clips;

            // Merge inferred subjects (dedup by label server-side)
            let addedSubjects = 0;
            if (inferredSubjects.length) {
                try {
                    const subjRes = await sourceProfilesApi.mergeSubjects({
                        profile_id: profile.id,
                        subjects:   inferredSubjects,
                    });
                    addedSubjects = subjRes.added ?? 0;
                    if (subjRes.profile) {
                        profile.subjects = subjRes.profile.subjects || [];
                    }
                } catch (subjErr) {
                    console.warn("mergeSubjects failed:", subjErr);
                }
            }

            suggestions      = [];
            inferredSubjects = [];
            if (detNoteEl) { detNoteEl.remove(); detNoteEl = null; }
            onClipsChanged(clips);
            redraw();
            const subjNote = addedSubjects ? `, ${addedSubjects} subject(s) added` : "";
            _toast(`Applied ${newClips.length} clip(s)${subjNote}`, "success");
        } catch (err) {
            _toast(`Apply failed: ${_errMsg(err)}`, "error");
            applySugBtn.disabled = false;
        } finally {
            applySugBtn.textContent = "Apply suggestions";
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
        detectSpinner.textContent = " Detecting…";

        // Live status updates from the backend — update spinner text while active
        const _onStatus = (event) => {
            const d = event?.detail || {};
            if (d.source !== "source_profile_analysis") return;
            const msg = String(d.status || "").trim();
            if (msg && detecting) detectSpinner.textContent = ` ${msg}`;
        };
        api.addEventListener("fbtools.status", _onStatus);

        try {
            await onEnsureSaved?.();
            const res = await sourceProfilesApi.detectSegments({
                profile_id:      profile.id,
                video_duration:  totalDur,
                prompt_override: detectPromptOverride.trim(),
                flags:           detectPromptOverride.trim() ? null : { ...detectFlags },
                captioner_type:  getActiveCaptionerType(),
            });
            suggestions      = res.segments          || [];
            inferredSubjects = res.inferred_subjects || [];
            lastRawResponse  = res.raw_response      || "";
            applySugBtn.disabled = suggestions.length === 0;
            if (detNoteEl) detNoteEl.remove();
            if (suggestions.length) {
                const subjNote = inferredSubjects.length
                    ? ` and ${inferredSubjects.length} subject(s)` : "";
                detNoteEl = _mk("div", { cls: "spe-clips-det-note" },
                    [`${suggestions.length} suggested boundary(s)${subjNote} — click "Apply suggestions" to create clips.`]);
                canvasWrap.after(detNoteEl);
            }
            // Show raw response section
            rawPre.textContent = lastRawResponse;
            rawWrap.style.display = "block";
            redraw();
            const subjHint = inferredSubjects.length ? `, ${inferredSubjects.length} subject(s)` : "";
            _toast(`${suggestions.length} suggested boundary(s)${subjHint}`, "info");
        } catch (err) {
            _toast(`Detection failed: ${_errMsg(err)}`, "error");
        } finally {
            api.removeEventListener("fbtools.status", _onStatus);
            detecting = false;
            detectBtn.disabled = false;
            detectSpinner.style.display = "none";
            detectSpinner.textContent = " Detecting…";
        }
    }

    async function describeClipAction(i, actionEl, btn) {
        const clip = clips[i];
        const prev = btn.textContent;
        btn.disabled = true;
        btn.textContent = "…";

        // Map the subjects tagged in this clip to slot letters (A–J).
        // The VLM will use {A}, {B} etc. instead of repeating appearance inline.
        const SLOTS = ["A","B","C","D","E","F","G","H","I","J"];
        const taggedIds = clip.subjects || [];
        const allSubjects = profile.subjects || [];
        const subjects = taggedIds
            .map(id => allSubjects.find(s => s.id === id))
            .filter(Boolean)
            .slice(0, SLOTS.length)
            .map((s, idx) => ({
                slot:       SLOTS[idx],
                name:       s.label || s.id,
                appearance: s.role_description || "",
            }));

        try {
            await onEnsureSaved?.();
            const res = await sourceProfilesApi.describeClip({
                profile_id:      profile.id,
                start_time:      clip.start_time,
                end_time:        clip.end_time,
                subjects,
                existing_action: clip.action || "",
                captioner_type:  getActiveCaptionerType(),
                prompt_override: profile.describe_prompt_override || "",
            });
            if (res.action) {
                clips[i] = { ...clips[i], action: res.action };
                actionEl.value = res.action;
                commitClip(i);
                _toast("Action filled", "success");
            }
        } catch (err) {
            _toast(`Describe failed: ${_errMsg(err)}`, "error");
        } finally {
            btn.disabled = false;
            btn.textContent = prev;
        }
    }

    return { redraw };
}

function _renderMediaPreview(profile) {
    const { media_filename: fn, media_dir: dir, media_type: type } = profile;
    if (!fn) return _mk("div", { cls: "spe-media-wrap", style: { color: "#666", fontSize: "11px" } }, ["No media file"]);
    const wrap = _mk("div", { cls: "spe-media-wrap" });
    if (type === "video") {
        const v = _mk("video", { controls: true, preload: "metadata" });
        v.src = _mediaUrl(fn, dir);
        const timeEl = _mk("div", { style: {
            fontFamily: "monospace", fontSize: "12px", textAlign: "center",
            color: "var(--fg)", opacity: "0.7", marginTop: "2px",
        }}, ["0:00.00"]);
        v.addEventListener("timeupdate", () => {
            const t = v.currentTime;
            const m = Math.floor(t / 60);
            const s = (t % 60).toFixed(2).padStart(5, "0");
            timeEl.textContent = `${m}:${s}`;
        });
        wrap.appendChild(v);
        wrap.appendChild(timeEl);
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

    const PRONOUN_STYLES = [
        ["neutral",   "they/their (default, non-binary or unknown)"],
        ["feminine",  "she/her"],
        ["masculine", "he/his"],
        ["object",    "it/its (props, objects)"],
        ["location",  "the [name]'s (rooms, environments)"],
    ];
    const pronounEl = document.createElement("select");
    pronounEl.className = "spe-select";
    PRONOUN_STYLES.forEach(([val, lbl]) => {
        const o = document.createElement("option");
        o.value = val; o.textContent = lbl;
        if (val === (initial.pronoun_style || "neutral")) o.selected = true;
        pronounEl.appendChild(o);
    });
    const shortNameEl = _mk("input", {
        type: "text", placeholder: "e.g. room, hallway, courtyard",
        value: initial.short_name || "",
    });
    const shortNameRow = _mk("div", { cls: "spe-form-row" }, [
        _mk("label", {}, ["Short name (for 'the …' reference)"]), shortNameEl,
    ]);
    shortNameRow.style.display = pronounEl.value === "location" ? "" : "none";
    pronounEl.addEventListener("change", () => {
        shortNameRow.style.display = pronounEl.value === "location" ? "" : "none";
    });

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
        _mk("label", {}, ["Pronoun style"]), pronounEl,
    ]));
    container.appendChild(shortNameRow);
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
                pronoun_style:    pronounEl.value,
                short_name:       shortNameEl.value.trim(),
                notes:            notesEl.value.trim(),
            });
        }}, ["Save subject"]),
    ]));
}

// ── Detail view ────────────────────────────────────────────────────────────────

async function _renderDetail(root) {
    root.innerHTML = "";
    let profile = _profile();
    if (!profile) return;

    // List endpoint returns summary only (no subjects/clips arrays).
    // Fetch the full record on first open and cache it in _S.profiles.
    if (!Array.isArray(profile.subjects)) {
        root.innerHTML = `<div style="padding:16px;color:var(--p-text-muted-color,#888)">Loading…</div>`;
        try {
            const full = await sourceProfilesApi.getProfile(profile.id);
            const idx = _S.profiles.findIndex(p => p.id === profile.id);
            if (idx >= 0) _S.profiles[idx] = full;
            profile = full;
        } catch (err) {
            root.innerHTML = `<div style="padding:16px;color:#f85149;">Failed to load profile: ${err.message}</div>`;
            return;
        }
        root.innerHTML = "";
    }

    // ── Header
    const header = _mk("div", { cls: "spe-detail-header" }, [
        _mk("button", { cls: "spe-btn sm ghost", onclick: () => { _S.selected = null; _render(root); } }, ["← Back"]),
        _mk("h3", {}, [profile.name || profile.id]),
        _mk("button", { cls: "spe-btn sm primary", onclick: () => _saveProfile(root, profile) }, ["Save"]),
    ]);
    root.appendChild(header);

    const body = _mk("div", { cls: "spe-detail-body" });
    root.appendChild(body);

    // ── Media preview (updated live when file/dir/type changes)
    const previewWrap = _mk("div");
    const _refreshPreview = () => {
        previewWrap.innerHTML = "";
        previewWrap.appendChild(_renderMediaPreview(profile));
    };
    _refreshPreview();
    body.appendChild(previewWrap);

    // ── Profile meta form (collapsible)
    const settingsTitleRow = _mk("div", { cls: "spe-section-head spe-collapse-toggle" }, [
        "Profile settings",
        _mk("i", { cls: "pi pi-chevron-" + (_S.settingsOpen ? "up" : "down"), style: { marginLeft: "auto" } }),
    ]);
    body.appendChild(settingsTitleRow);
    const settingsBody = _mk("div", { cls: "spe-collapsible" + (_S.settingsOpen ? "" : " collapsed") });
    body.appendChild(settingsBody);
    settingsTitleRow.onclick = () => {
        _S.settingsOpen = !_S.settingsOpen;
        settingsBody.classList.toggle("collapsed", !_S.settingsOpen);
        settingsTitleRow.querySelector("[class*=pi-chevron]").className =
            "pi pi-chevron-" + (_S.settingsOpen ? "up" : "down");
    };

    const nameEl = _mk("input", { type: "text", value: profile.name || "" });
    settingsBody.appendChild(_mk("div", { cls: "spe-form-row" }, [_mk("label", {}, ["Name"]), nameEl]));

    const fileUid = Math.random().toString(36).slice(2, 8);
    const fileDl  = _mk("datalist", { id: `spe-media-dl-${fileUid}` });
    const fileEl  = _mk("input", { type: "text",
        placeholder: "filename or subdir/filename...",
        value: profile.media_filename || "" });
    fileEl.setAttribute("list", `spe-media-dl-${fileUid}`);

    const _rebuildDl = (type, dir) => {
        fileDl.innerHTML = "";
        _allMedia(type, dir).forEach(fn => {
            const o = document.createElement("option"); o.value = fn; fileDl.appendChild(o);
        });
    };
    _rebuildDl(profile.media_type || "video", profile.media_dir || "input");

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

    fileEl.addEventListener("input",  () => { profile.media_filename = fileEl.value.trim(); _refreshPreview(); });
    dirEl.addEventListener("change",  () => { profile.media_dir  = dirEl.value;  _rebuildDl(typeEl.value, dirEl.value); _refreshPreview(); });
    typeEl.addEventListener("change", () => { profile.media_type = typeEl.value; _rebuildDl(typeEl.value, dirEl.value); _refreshPreview(); });

    settingsBody.appendChild(_mk("div", { cls: "spe-form-row" }, [_mk("label", {}, ["Media type"]), typeEl]));
    settingsBody.appendChild(_mk("div", { cls: "spe-form-row" }, [_mk("label", {}, ["Media dir"]), dirEl]));
    settingsBody.appendChild(_mk("div", { cls: "spe-form-row" }, [_mk("label", {}, ["Media file"]), fileEl, fileDl]));

    // Wire save
    const collectMeta = () => ({
        ...profile,
        name:           nameEl.value.trim() || profile.name,
        media_filename: fileEl.value,
        media_dir:      dirEl.value,
        media_type:     typeEl.value,
    });

    // ── Subjects section (collapsible)
    body.appendChild(_mk("div", { cls: "spe-divider" }));
    const subjectsTitleRow = _mk("div", { cls: "spe-section-head spe-collapse-toggle" }, [
        "Subjects",
        _mk("i", { cls: "pi pi-chevron-" + (_S.subjectsOpen ? "up" : "down"), style: { marginLeft: "auto" } }),
    ]);
    body.appendChild(subjectsTitleRow);
    const subjectsBody = _mk("div", { cls: "spe-collapsible" + (_S.subjectsOpen ? "" : " collapsed") });
    body.appendChild(subjectsBody);
    subjectsTitleRow.onclick = () => {
        _S.subjectsOpen = !_S.subjectsOpen;
        subjectsBody.classList.toggle("collapsed", !_S.subjectsOpen);
        subjectsTitleRow.querySelector("[class*=pi-chevron]").className =
            "pi pi-chevron-" + (_S.subjectsOpen ? "up" : "down");
    };

    const subjListEl = _mk("div");
    subjectsBody.appendChild(subjListEl);

    const subjFormEl = _mk("div");
    subjectsBody.appendChild(subjFormEl);

    let clipsSection = null;

    const renderSubjList = () => {
        clipsSection?.redraw();
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
        clipsSection = _renderClipsSection(body, profile, (_clips) => {
            profile.clips = _clips;
        }, async () => {
            // Auto-save the profile (with current form values) before
            // backend calls that require the profile to exist on disk.
            const current = collectMeta();
            await _saveProfile(root, current);
            // Keep the in-memory profile in sync so further calls get the updated id.
            Object.assign(profile, current);
        }, (startTime) => {
            const vid = previewWrap.querySelector("video");
            if (vid) vid.currentTime = startTime;
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

    let _rebuildClipSel = null;

    titleRow.onclick = () => {
        _S.analyzeOpen = !_S.analyzeOpen;
        body.classList.toggle("collapsed", !_S.analyzeOpen);
        titleRow.querySelector(".pi-chevron-up, .pi-chevron-down").className =
            "pi pi-chevron-" + (_S.analyzeOpen ? "up" : "down");
        if (_S.analyzeOpen) { _loadHistory(profile.id); _rebuildClipSel?.(); }
    };

    // Pass type pills
    body.appendChild(_mk("div", { cls: "spe-section-head" }, ["Focus pass"]));
    const pillsWrap = _mk("div", { cls: "spe-pass-pills" });
    let _updateDefaultPreview;
    PASS_TYPES.forEach(pt => {
        const pill = _mk("button", {
            cls: "spe-pass-pill" + (pt === _S.analyzePassType ? " active" : ""),
            onclick: () => {
                _S.analyzePassType = pt;
                pillsWrap.querySelectorAll(".spe-pass-pill").forEach(p => p.classList.remove("active"));
                pill.classList.add("active");
                _updateDefaultPreview?.();
            },
        }, [PASS_LABELS[pt]]);
        pillsWrap.appendChild(pill);
    });
    body.appendChild(pillsWrap);

    // Clip selector
    body.appendChild(_mk("div", { cls: "spe-section-head" }, ["Clip / frame source"]));

    const clipSel = _mk("select");
    const maxFramesInp = _mk("input", { type: "number", min: 1, max: 120, step: 1,
        value: _S.analyzeMaxFrames ?? 20,
        style: { width: "56px" },
        title: "Max frames sent to the VLM (native-video models) or tiled in the contact sheet" });
    maxFramesInp.onchange = () => { _S.analyzeMaxFrames = parseInt(maxFramesInp.value) || 20; };

    const nthInp = _mk("input", { type: "number", min: 1, max: 60, step: 1,
        value: _S.analyzeSelectNth ?? 1,
        style: { width: "48px" },
        title: "Use every Nth frame of the clip (1 = every frame at effective fps)" });
    nthInp.onchange = () => { _S.analyzeSelectNth = parseInt(nthInp.value) || 1; };

    const frameControls = _mk("div", { cls: "spe-form-row", style: { display: "none" } }, [
        _mk("label", {}, ["Max frames"]), maxFramesInp,
        _mk("label", {}, ["Every Nth"]), nthInp,
    ]);

    _rebuildClipSel = () => {
        clipSel.innerHTML = "";
        const none = _mk("option", { value: "" }, ["Whole video (single frame @ 10%)"]);
        clipSel.appendChild(none);
        (profile.clips || []).forEach((c, i) => {
            const lbl = c.label || `Clip ${i + 1}`;
            const o = _mk("option", { value: i }, [`${lbl}  (${c.start_time.toFixed(1)}–${c.end_time.toFixed(1)}s)`]);
            clipSel.appendChild(o);
        });
        if (_S.analyzeClipIdx != null) clipSel.value = _S.analyzeClipIdx;
    };
    _rebuildClipSel();

    clipSel.onchange = () => {
        _S.analyzeClipIdx = clipSel.value === "" ? null : parseInt(clipSel.value);
        const hasClip = _S.analyzeClipIdx != null;
        frameControls.style.display = hasClip ? "" : "none";
        if (hasClip) {
            const clip = (profile.clips || [])[_S.analyzeClipIdx];
            if (clip?.select_every_nth) {
                nthInp.value = clip.select_every_nth;
                _S.analyzeSelectNth = clip.select_every_nth;
            }
        }
    };

    body.appendChild(_mk("div", { cls: "spe-form-row" }, [
        _mk("label", {}, ["Clip"]), clipSel,
    ]));
    body.appendChild(frameControls);

    // Active backend indicator (read-only — configured in LLM tab)
    const backendBadge = _mk("span", {
        style: { fontSize: "11px", color: "var(--p-text-muted-color,#888)",
                 padding: "2px 6px", borderRadius: "3px",
                 background: "var(--p-surface-section,#252525)" },
    }, [getActiveCaptionerType()]);
    body.appendChild(_mk("div", { cls: "spe-form-row" }, [
        _mk("label", {}, ["Backend"]),
        backendBadge,
        _mk("span", { style: { fontSize: "10px", color: "#666", marginLeft: "4px" } }, ["← LLM tab"]),
    ]));

    // Prompt override
    const overrideToggle = _mk("div", { cls: "spe-section-head spe-collapse-toggle",
        style: { cursor: "pointer" } }, ["▶ Edit prompt"]);
    const overrideWrap = _mk("div", { cls: "spe-collapsible collapsed" });

    const defaultLbl = _mk("div", { cls: "spe-clip-field-label", style: { marginTop: "6px" } },
        ["Default prompt (leave override empty to use this):"] );
    const defaultPreviewEl = _mk("textarea", { rows: 5, readOnly: true,
        style: { opacity: "0.6", resize: "vertical", width: "100%", boxSizing: "border-box",
                 fontFamily: "monospace", fontSize: "11px" } });
    _updateDefaultPreview = () => {
        defaultPreviewEl.value = PASS_DEFAULT_PROMPTS[_S.analyzePassType] ?? "";
    };
    _updateDefaultPreview();

    const overrideEl = _mk("textarea", { rows: 4,
        placeholder: "Leave empty to use the default prompt shown below." });
    overrideEl.value = _S.analyzePromptOverride;
    overrideEl.onchange = () => { _S.analyzePromptOverride = overrideEl.value; };

    const clearBtn = _mk("button", { cls: "spe-btn sm ghost", style: { marginTop: "4px" },
        onclick: () => { overrideEl.value = ""; _S.analyzePromptOverride = ""; } }, ["Clear override"]);

    overrideWrap.append(overrideEl, clearBtn, defaultLbl, defaultPreviewEl);
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
        const clip = (_S.analyzeClipIdx != null)
            ? (profile.clips || [])[_S.analyzeClipIdx]
            : null;

        const res = await sourceProfilesApi.analyze({
            profile_id:       profile.id,
            pass_type:        _S.analyzePassType,
            prompt_override:  _S.analyzePromptOverride,
            captioner_type:   getActiveCaptionerType(),
            start_time:       clip ? clip.start_time : null,
            end_time:         clip ? clip.end_time   : null,
            select_every_nth: _S.analyzeSelectNth ?? 1,
            max_frames:       _S.analyzeMaxFrames  ?? 20,
            // Whole-video mode: pass duration so the backend can window the video
            // instead of falling back to a single representative frame.
            video_duration:   clip ? 0 : getTotalDuration(),
        });

        _S.analyzeCandidates = res.candidates ?? [];
        _renderCandidates(analyzeBody._candidatesEl, profile, onSubjectsChanged);

        // Refresh history
        await _loadHistory(profile.id);
        if (!analyzeBody._histWrap.classList.contains("collapsed")) {
            _renderHistory(analyzeBody._histWrap, profile, onSubjectsChanged);
        }

        const modeNote = res.frame_mode === "windowed_multi"
            ? ` · ${res.frame_count} frames across full video`
            : res.frame_count > 1 ? ` · ${res.frame_count} frames` : "";
        _toast(`Found ${_S.analyzeCandidates.length} candidate(s)${modeNote}`, "success");
    } catch (err) {
        _toast(`Analysis failed: ${_errMsg(err)}`, "error");
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
        // subjects array is only present after the profile is opened (full fetch).
        // subject_count comes from the list endpoint summary and is always available.
        const subjects = Array.isArray(p.subjects) ? p.subjects : [];
        const subjectCount = p.subject_count ?? subjects.length;
        const chips = subjects.slice(0, 4).map(s =>
            _mk("span", { cls: "spe-subject-chip" }, [
                (() => { const i = _mk("i", {}); i.className = (ENTITY_ICONS[s.entity_type] || "pi pi-circle") + " spe-subj-icon"; i.style.fontSize = "10px"; return i; })(),
                s.label || s.id,
            ])
        );
        if (subjects.length > 4) chips.push(_mk("span", { cls: "spe-subject-chip" }, [`+${subjects.length - 4} more`]));

        let bodyContent;
        if (chips.length) {
            bodyContent = chips;
        } else if (subjectCount > 0) {
            bodyContent = [_mk("span", { style: { color: "#888", fontSize: "11px" } }, [`${subjectCount} subject${subjectCount !== 1 ? "s" : ""} · open to view`])];
        } else {
            bodyContent = [_mk("span", { style: { color: "#666", fontSize: "11px" } }, ["No subjects annotated yet"])];
        }

        const card = _mk("div", { cls: "spe-profile-card" + (p.id === _S.selected ? " active" : ""),
            onclick: () => { _S.selected = p.id; _S.editingSubject = null; _S.analyzeCandidates = []; _renderDetail(root); }
        }, [
            _mk("div", { cls: "spe-card-header" }, [
                _mk("i", { cls: "pi pi-film" }),
                _mk("span", { cls: "spe-card-name" }, [p.name || p.id]),
                _mk("span", { cls: "spe-card-meta" }, [`${p.media_type || "video"} · ${subjectCount} subj`]),
            ]),
            _mk("div", { cls: "spe-card-body" }, bodyContent),
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
    panel.addEventListener("keydown",  e => e.stopPropagation());
    panel.addEventListener("keyup",    e => e.stopPropagation());
    panel.addEventListener("keypress", e => e.stopPropagation());
    container.appendChild(panel);

    // Show loading state
    panel.appendChild(_mk("div", { style: { padding: "16px", color: "#888", fontSize: "12px" } },
        ["Loading source profiles…"]));

    await _loadAll();

    panel.innerHTML = "";
    _render(panel);
}
