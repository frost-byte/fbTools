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
    analyzePassTypes: new Set(["people"]),
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
    clipsSettingsOpen: true,  // whole-video Detect/segmentation settings, inside Clips section
    clipsDescribeSettingsOpen: false,  // Describe instruction / Max frames / Every Nth
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

function _drawTimeline(canvas, clips, suggestions, totalDuration, activeIdx = -1, dirtyIds = null, hoverIdx = -1, viewRange = null) {
    const W = canvas.width;
    const H = canvas.height;
    const ctx = canvas.getContext("2d");
    const [rangeStart, rangeEnd] = viewRange || [0, totalDuration > 0 ? totalDuration : 1];
    const dur = Math.max(rangeEnd - rangeStart, 0.001);
    const toX = t => ((t - rangeStart) / dur) * W;

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#111";
    ctx.fillRect(0, 0, W, H);

    const BAND_TOP = 10, BAND_BOT = H - 20;
    const HOVER_RISE = 6;  // px the hovered band extends above BAND_TOP

    let firstVisible = null, lastVisible = null;

    // Clip bands
    clips.forEach((clip, i) => {
        if (clip.end_time <= rangeStart || clip.start_time >= rangeEnd) return;  // outside the current window
        if (firstVisible === null) firstVisible = clip;
        lastVisible = clip;

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

    // Internal boundary lines (no handle — resizing happens via the start/end
    // inputs; the old drag-handle circle is just visual noise now)
    clips.forEach((clip, i) => {
        if (i === 0 || clip.start_time <= rangeStart) return;
        const x = toX(clip.start_time);
        ctx.strokeStyle = "#ffffffcc";
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(x, BAND_TOP); ctx.lineTo(x, BAND_BOT); ctx.stroke();
    });

    // Suggestion markers (dashed gold)
    ctx.setLineDash([4, 3]);
    ctx.strokeStyle = "#fbbf24";
    ctx.lineWidth = 1.5;
    suggestions.forEach(seg => {
        if (seg.start_time <= rangeStart || seg.start_time >= rangeEnd) return;
        const x = toX(seg.start_time);
        ctx.beginPath(); ctx.moveTo(x, 2); ctx.lineTo(x, H - 2); ctx.stroke();
    });
    ctx.setLineDash([]);

    // Time axis ticks
    const tickStep = dur <= 20 ? 2 : dur <= 60 ? 5 : dur <= 180 ? 15 : 30;
    ctx.fillStyle = "#555"; ctx.font = "9px monospace"; ctx.textAlign = "center";
    const tickStart = Math.ceil(rangeStart / tickStep) * tickStep;
    for (let t = tickStart; t <= rangeEnd + 0.001; t += tickStep) {
        const x = Math.round(toX(t));
        ctx.fillStyle = "#444";
        ctx.fillRect(x, BAND_BOT, 1, 4);
        ctx.fillStyle = "#666";
        ctx.fillText(`${Math.round(t)}`, x, H - 2);
    }

    // Start / end labels under first / last VISIBLE clip
    if (firstVisible) {
        ctx.font = "9px monospace"; ctx.fillStyle = "#888";
        ctx.textAlign = "left";
        ctx.fillText(_fmtT(firstVisible.start_time), toX(firstVisible.start_time) + 2, BAND_BOT + 4);
        ctx.textAlign = "right";
        ctx.fillText(_fmtT(lastVisible.end_time), toX(lastVisible.end_time) - 2, BAND_BOT + 4);
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

function _buildClipLoraSection(clip, onCommit, onApplyToAll = null) {
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

            const applyAllBtn = _mk("button", {
                cls: "spe-lora-apply-all",
                title: "Apply to all segments (skips any that already have this LoRA; preserves their weight)",
                textContent: "→ all",
                disabled: !entry.name,
            });
            applyAllBtn.addEventListener("click", () => {
                if (entry.name && onApplyToAll) onApplyToAll({ ...loras[i] });
            });

            const rmBtn = _mk("button", { cls: "spe-lora-rm", title: "Remove", textContent: "✕" });
            rmBtn.addEventListener("click", () => {
                loras.splice(i, 1);
                clip.loras = loras;
                onCommit();
                _rebuild();
            });

            listEl.appendChild(_mk("div", { cls: "spe-lora-row" }, [nameWrap, strengthInp, applyAllBtn, rmBtn]));
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

function _renderClipsSection(container, profile, onClipsChanged, onEnsureSaved, onSelect = null, getVideoTime = null) {
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
    // Timeline windowing: with many clips, individual bands become too thin to
    // click reliably. Once clip count exceeds zoomWindowSize, the timeline only
    // renders that many clips at a time, scaled to fill the canvas width, with
    // arrow buttons to page zoomStart left/right. clips.length <= zoomWindowSize
    // means "not zoomed" — the timeline shows everything, as before.
    const zoomWindowSize = 10;
    let zoomStart = 0;
    let detectFlags = { camera_cuts: true, subject_changes: false, lower_threshold: false };
    let detectPromptOverride = "";
    let detectProgress = null;  // { windows:[[s,e]], status:[], segs:[], elapsed:[], frames:[] }
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

    // ── Video settings (collapsible) ─────────────────────────────────────────
    // Everything here applies to the whole source video and is primarily used
    // once, when first segmenting it — collapse it out of the way once you're
    // done and want more room for adjusting clip boundaries/descriptions below.
    const vsChevron = _mk("i", { cls: "pi pi-chevron-" + (_S.clipsSettingsOpen ? "up" : "down") });
    const videoSettingsTitle = _mk("div", {
        cls: "spe-clips-subtitle",
        style: { cursor: "pointer", display: "flex", alignItems: "center", gap: "4px",
                 fontSize: "11px", color: "#888", margin: "4px 0 2px", userSelect: "none" },
        title: "Whole-video settings used when first segmenting this source — collapse once done.",
    }, ["▸ Video settings (duration, proxy, Detect boundaries)", vsChevron]);
    const videoSettingsBody = _mk("div", { cls: "spe-collapsible" + (_S.clipsSettingsOpen ? "" : " collapsed") });
    videoSettingsTitle.onclick = () => {
        _S.clipsSettingsOpen = !_S.clipsSettingsOpen;
        videoSettingsBody.classList.toggle("collapsed", !_S.clipsSettingsOpen);
        vsChevron.className = "pi pi-chevron-" + (_S.clipsSettingsOpen ? "up" : "down");
    };
    body.append(videoSettingsTitle, videoSettingsBody);

    videoSettingsBody.appendChild(_mk("div", { cls: "spe-clips-toolbar" }, [
        _mk("label", {}, ["Duration (s):"]), durationInput,
        _mk("label", {}, ["Seg (s):"]), segDurInput,
        autoSegBtn,
    ]));
    videoSettingsBody.appendChild(_mk("div", { cls: "spe-clips-toolbar", style: { marginTop: "4px" } }, [
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

    videoSettingsBody.appendChild(_mk("div", { cls: "spe-clips-toolbar" }, [
        backendNote, detectBtn, detectSpinner, applySugBtn,
    ]));

    // Detect flags row — tooltips explain actual VLM-prompt impact, not just the label
    const _FLAG_TOOLTIPS = {
        camera_cuts: (
            "Adds: \"Hard camera cuts, lens angle changes, and scene edits are transition " +
            "boundaries even when the subject or setting remains the same.\"\n" +
            "A boundary CRITERION — recognizes cuts/angle changes as boundaries even with no other change."
        ),
        subject_changes: (
            "Adds: \"A key subject entering or leaving the frame counts as a transition when it " +
            "represents a notable shift in the scene's cast.\"\n" +
            "A boundary CRITERION — recognizes people entering/leaving frame as boundaries."
        ),
        lower_threshold: (
            "Adds: \"When in doubt, err on the side of marking more boundaries rather than fewer.\"\n" +
            "NOT a new boundary type — a SENSITIVITY dial. Lowers the bar for how confident the " +
            "VLM needs to be before calling something a boundary; use when detection is under-splitting."
        ),
    };
    const _mkFlagCb = (key, label, defaultOn) => {
        const cb  = _mk("input", { type: "checkbox", id: `spe-flag-${key}-${profile.id || "new"}`,
            title: _FLAG_TOOLTIPS[key] });
        cb.checked = defaultOn;
        cb.onchange = () => { detectFlags[key] = cb.checked; _refreshPromptPreview(); };
        const lbl = _mk("label", { htmlFor: cb.id, style: { fontSize: "11px" }, title: _FLAG_TOOLTIPS[key] }, [label]);
        return [cb, lbl];
    };
    const [camCb, camLbl]     = _mkFlagCb("camera_cuts",    "Camera cuts",     true);
    const [subjCb, subjLbl]   = _mkFlagCb("subject_changes","Subject changes",  false);
    const [lowCb, lowLbl]     = _mkFlagCb("lower_threshold","More boundaries",  false);
    videoSettingsBody.appendChild(_mk("div", { cls: "spe-clips-toolbar", style: { flexWrap: "wrap", gap: "6px" } }, [
        _mk("span", { style: { fontSize: "11px", color: "#888" } }, ["Flags:"]),
        camCb, camLbl, subjCb, subjLbl, lowCb, lowLbl,
    ]));

    // ── Precision slider ─────────────────────────────────────────────────────
    // Single knob: seconds between sampled frames. The per-call time window is
    // ALWAYS derived as interval * 20 so every VLM call uses the full 20-frame
    // budget evenly — no wasted capacity, no silently-dropped tail frames.
    const DETECT_FRAMES_PER_CALL = 20;
    let detectIntervalSeconds = 3.0;
    const precisionSlider = _mk("input", {
        type: "range", min: "1", max: "8", step: "0.5", value: String(detectIntervalSeconds),
        style: { verticalAlign: "middle" },
        title: (
            "Seconds between analyzed frames — the only sampling control. Lower = catches " +
            "quick cuts and short beats but costs more VLM calls; higher = coarser and faster, " +
            "may miss brief transitions.\n\n" +
            `The time window sent per VLM call is always ${DETECT_FRAMES_PER_CALL} × this value, ` +
            "so every call fully uses its frame budget regardless of the setting."
        ),
    });
    const precisionReadout = _mk("span", { style: { fontSize: "11px", color: "#888", marginLeft: "6px" } });
    function _updatePrecisionReadout() {
        const interval = detectIntervalSeconds;
        const windowSec = interval * DETECT_FRAMES_PER_CALL;
        const dur = getTotalDuration();
        const nWindows = Math.max(1, Math.ceil(dur / windowSec));
        precisionReadout.textContent =
            `1 frame / ${interval.toFixed(1)}s  ·  ${Math.round(windowSec)}s per VLM call  ·  ` +
            `~${nWindows} call${nWindows !== 1 ? "s" : ""} for this ${dur.toFixed(0)}s video`;
    }
    precisionSlider.oninput = () => {
        detectIntervalSeconds = parseFloat(precisionSlider.value);
        _updatePrecisionReadout();
    };
    videoSettingsBody.appendChild(_mk("div", { cls: "spe-clips-toolbar" }, [
        _mk("label", { style: { fontSize: "11px", color: "#888" }, title: precisionSlider.title },
            ["Precision:"]),
        precisionSlider, precisionReadout,
    ]));
    _updatePrecisionReadout();

    // Prompt override textarea (collapsible)
    const promptToggle = _mk("button", { cls: "spe-btn sm ghost", style: { fontSize: "11px" },
        title: "Replaces the entire boundary-detection prompt below — flags above are ignored " +
               "when this is non-empty. The JSON output schema is still appended automatically." },
        ["▸ Prompt override"]);
    const promptWrap   = _mk("div", { style: { display: "none", marginTop: "4px" } });
    const promptTa     = _mk("textarea", { placeholder: "Leave empty to use flags above…",
        rows: 4, style: { width: "100%", fontSize: "11px", resize: "vertical",
                          background: "var(--bg2)", color: "var(--fg)", border: "1px solid var(--border)",
                          borderRadius: "4px", padding: "4px", boxSizing: "border-box" } });
    let _previewDebounce = null;
    promptTa.oninput = () => {
        detectPromptOverride = promptTa.value;
        clearTimeout(_previewDebounce);
        _previewDebounce = setTimeout(_refreshPromptPreview, 300);
    };
    promptWrap.appendChild(promptTa);
    promptToggle.onclick = () => {
        const open = promptWrap.style.display === "none";
        promptWrap.style.display = open ? "block" : "none";
        promptToggle.textContent = (open ? "▾ " : "▸ ") + "Prompt override";
    };
    videoSettingsBody.appendChild(_mk("div", { style: { padding: "2px 0" } }, [promptToggle, promptWrap]));

    // Prompt preview (collapsible) — shows the EXACT prompt Detect boundaries will
    // send, live-updated from the flags/override above via the backend builder
    // itself (never hand-duplicated in JS, so it can't drift from the real prompt).
    const previewToggle = _mk("button", { cls: "spe-btn sm ghost", style: { fontSize: "11px" },
        title: "Preview the exact instruction text sent to the VLM, built from the flags/override " +
               "above. Only fetched while this section is open." },
        ["▸ Prompt preview"]);
    const previewWrap = _mk("div", { style: { display: "none", marginTop: "4px" } });
    const previewNote = _mk("div", { style: { fontSize: "10px", color: "#888", marginBottom: "4px" } },
        ["This base prompt is reused per time-window; each actual call additionally appends " +
         "a timestamp-range note pinning that window's absolute start/end seconds."]);
    const previewPre = _mk("pre", { style: { fontSize: "10px", whiteSpace: "pre-wrap", wordBreak: "break-word",
        maxHeight: "260px", overflowY: "auto", background: "var(--bg2)", padding: "6px",
        border: "1px solid var(--border)", borderRadius: "4px", margin: 0 } }, ["Loading…"]);
    previewWrap.append(previewNote, previewPre);
    let _previewOpen = false;
    async function _refreshPromptPreview() {
        if (!_previewOpen) return;
        try {
            const r = await sourceProfilesApi.segmentPromptPreview({
                prompt_override: detectPromptOverride.trim(),
                flags:           detectPromptOverride.trim() ? null : { ...detectFlags },
            });
            previewPre.textContent = r.prompt || "";
        } catch (err) {
            previewPre.textContent = `(preview failed: ${_errMsg(err)})`;
        }
    }
    previewToggle.onclick = () => {
        _previewOpen = previewWrap.style.display === "none";
        previewWrap.style.display = _previewOpen ? "block" : "none";
        previewToggle.textContent = (_previewOpen ? "▾ " : "▸ ") + "Prompt preview";
        if (_previewOpen) _refreshPromptPreview();
    };
    videoSettingsBody.appendChild(_mk("div", { style: { padding: "2px 0" } }, [previewToggle, previewWrap]));

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
    videoSettingsBody.appendChild(rawWrap);

    // ── Detect-boundaries progress strip ────────────────────────────────────
    // One block per time-window, sized proportional to its duration, colored by
    // status (pending/active/done). Shown only while a Detect run is active or
    // just finished; hidden otherwise.
    const _DETECT_BLOCK_COLOR = { pending: "#3a3a3a", active: "#f0ad4e", done: "#22c55e" };
    const detectProgressWrap = _mk("div", { style: { display: "none", margin: "4px 0" } });
    const detectProgressBar  = _mk("div", { style: {
        display: "flex", gap: "1px", height: "14px", borderRadius: "2px",
        overflow: "hidden", border: "1px solid var(--border)",
    } });
    const detectProgressText = _mk("div", { style: { fontSize: "10px", color: "#888", marginTop: "2px" } });
    detectProgressWrap.append(detectProgressBar, detectProgressText);
    body.appendChild(detectProgressWrap);

    function _renderDetectProgress() {
        if (!detectProgress) { detectProgressWrap.style.display = "none"; return; }
        detectProgressWrap.style.display = "block";
        detectProgressBar.innerHTML = "";
        detectProgress.windows.forEach(([s, e], i) => {
            const status = detectProgress.status[i] || "pending";
            const blk = _mk("div", { style: {
                flexGrow: String(Math.max(0.01, e - s)), flexBasis: "0",
                background: _DETECT_BLOCK_COLOR[status],
                transition: "background-color 0.2s",
            } });
            let tip = `${s.toFixed(0)}s – ${e.toFixed(0)}s`;
            if (status === "done") {
                tip += `  ·  ${detectProgress.segs[i] ?? 0} segment(s)  ·  ${(detectProgress.elapsed[i] ?? 0).toFixed(1)}s`;
            } else if (status === "active") {
                tip += "  ·  processing…";
            } else {
                tip += "  ·  pending";
            }
            blk.title = tip;
            detectProgressBar.appendChild(blk);
        });

        const doneIdx    = detectProgress.status.map((s, i) => s === "done" ? i : -1).filter(i => i >= 0);
        const totalSegs  = doneIdx.reduce((a, i) => a + (detectProgress.segs[i] || 0), 0);
        const elapsedSum = doneIdx.reduce((a, i) => a + (detectProgress.elapsed[i] || 0), 0);
        const framesSum  = doneIdx.reduce((a, i) => a + (detectProgress.frames[i] || 0), 0);
        let text = `${doneIdx.length}/${detectProgress.windows.length} window(s) processed`;
        if (doneIdx.length > 0) {
            const secPerWindow = elapsedSum / doneIdx.length;
            const framesPerSec = elapsedSum > 0 ? framesSum / elapsedSum : 0;
            text += `  ·  ${totalSegs} segment(s) found so far  ·  ` +
                    `${secPerWindow.toFixed(1)}s/window avg  ·  ~${framesPerSec.toFixed(1)} frames/s`;
        }
        detectProgressText.textContent = text;
    }

    // ── Timeline canvas ───────────────────────────────────────────────────────
    // canvasWrap is a flex row: [prev arrow] [canvasInner (canvas + tooltip)]
    // [next arrow]. The arrows take their own fixed width rather than
    // overlaying the canvas, so they never sit on top of a clip segment —
    // canvasInner (flex:1) shrinks to make room for them instead.
    const canvasWrap = _mk("div", { cls: "spe-timeline-wrap" });
    const canvasInner = _mk("div", { cls: "spe-timeline-inner", style: { position: "relative" } });
    const canvas = document.createElement("canvas");
    canvas.className = "spe-timeline";
    canvas.height = 72;
    canvasInner.appendChild(canvas);

    // Zoom-window paging arrows — only shown once clip count exceeds
    // zoomWindowSize, since below that the full timeline already fits.
    const zoomPrevBtn = _mk("button", {
        cls: "spe-btn sm ghost spe-timeline-zoom-btn spe-timeline-zoom-prev",
        title: "Show previous clips",
    }, ["←"]);
    const zoomNextBtn = _mk("button", {
        cls: "spe-btn sm ghost spe-timeline-zoom-btn spe-timeline-zoom-next",
        title: "Show next clips",
    }, ["→"]);
    // Paging jumps zoomStart by a whole window, which used to redraw instantly
    // — jarring since the visible clips change completely in one frame.
    // Animate the pan/rescale between the old and new visible range instead.
    function _animateZoomPan(fromRange, toRange) {
        if (!fromRange || !toRange) { redraw(); return; }
        const DURATION_MS = 180;
        const t0 = performance.now();
        const dirty = new Set(clips.filter(_isProxyDirty).map(c => c.id));
        function step(now) {
            const t = Math.min(1, (now - t0) / DURATION_MS);
            const eased = 1 - Math.pow(1 - t, 3);  // ease-out cubic
            const range = [
                fromRange[0] + (toRange[0] - fromRange[0]) * eased,
                fromRange[1] + (toRange[1] - fromRange[1]) * eased,
            ];
            canvas.width = canvas.offsetWidth || 380;
            _drawTimeline(canvas, clips, suggestions, getTotalDuration(), activeClipIdx, dirty, hoverClipIdx, range);
            if (t < 1) requestAnimationFrame(step);
            else redraw();  // final pass to fully resync (nav list, arrow disabled state, etc.)
        }
        requestAnimationFrame(step);
    }

    zoomPrevBtn.onclick = () => {
        const fromRange = _visibleClipRange();
        zoomStart = Math.max(0, zoomStart - zoomWindowSize);
        _animateZoomPan(fromRange, _visibleClipRange());
    };
    zoomNextBtn.onclick = () => {
        const fromRange = _visibleClipRange();
        zoomStart = Math.min(Math.max(0, clips.length - zoomWindowSize), zoomStart + zoomWindowSize);
        _animateZoomPan(fromRange, _visibleClipRange());
    };

    function _visibleClipRange() {
        if (clips.length <= zoomWindowSize) return null;  // fits in one view — show everything
        zoomStart = Math.max(0, Math.min(zoomStart, clips.length - zoomWindowSize));
        const lastIdx = Math.min(zoomStart + zoomWindowSize, clips.length) - 1;
        return [clips[zoomStart].start_time, clips[lastIdx].end_time];
    }

    function _updateZoomArrows() {
        const zoomed = clips.length > zoomWindowSize;
        zoomPrevBtn.style.display = zoomed ? "" : "none";
        zoomNextBtn.style.display = zoomed ? "" : "none";
        if (!zoomed) return;
        zoomPrevBtn.disabled = zoomStart <= 0;
        zoomNextBtn.disabled = zoomStart + zoomWindowSize >= clips.length;
    }

    // Floating tooltip for narrow clips that are hard to identify on hover
    const hoverTooltipEl = _mk("div", { style: {
        position: "absolute", top: "-26px", left: "0",
        background: "#1a1a2e", border: "1px solid #555", borderRadius: "4px",
        padding: "3px 8px", fontSize: "11px", color: "#ddd",
        pointerEvents: "none", whiteSpace: "nowrap",
        boxShadow: "0 2px 8px rgba(0,0,0,.5)", display: "none", zIndex: "10",
    }});
    canvasInner.appendChild(hoverTooltipEl);

    canvasWrap.append(zoomPrevBtn, canvasInner, zoomNextBtn);
    body.appendChild(canvasWrap);

    // ── Clip navigation (directly below the timeline) ────────────────────────
    // Nav row (←/label/→) and Merge/Split are rendered by renderClipList() into
    // separate rows so Merge isn't a misclick away from the → arrow.
    const navEl = _mk("div");
    body.appendChild(navEl);

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

    // ── Describe frame controls ───────────────────────────────────────────────
    const descFrameRow = _mk("div", { style: { display: "flex", gap: "10px", alignItems: "center", marginBottom: "8px", fontSize: "11px" } });

    const descMaxFrInp = _mk("input", { type: "number", min: 1, max: 30, step: 1,
        value: profile.describe_max_frames ?? 5,
        style: { width: "48px" },
        title: "Maximum number of frames to sample from the clip for action description",
    });
    descMaxFrInp.onchange = () => {
        profile.describe_max_frames = parseInt(descMaxFrInp.value) || 5;
        sourceProfilesApi.save(profile).catch(() => {});
    };

    const descNthInp = _mk("input", { type: "number", min: 1, max: 30, step: 1,
        value: profile.describe_select_every_nth ?? 1,
        style: { width: "48px" },
        title: "Sample every Nth frame from the clip (1 = every frame up to Max frames)",
    });
    descNthInp.onchange = () => {
        profile.describe_select_every_nth = parseInt(descNthInp.value) || 1;
        sourceProfilesApi.save(profile).catch(() => {});
    };

    descFrameRow.append(
        _mk("label", {}, ["Max frames"]), descMaxFrInp,
        _mk("label", { style: { marginLeft: "6px" } }, ["Every Nth"]), descNthInp,
    );

    // Collapsible "Describe settings" — the instruction override and frame-sampling
    // controls used by the per-clip Describe button below. Collapsed by default:
    // these are set-once defaults, not something touched while adjusting clips.
    const dsChevron = _mk("i", { cls: "pi pi-chevron-" + (_S.clipsDescribeSettingsOpen ? "up" : "down") });
    const describeSettingsTitle = _mk("div", {
        cls: "spe-clips-subtitle",
        style: { cursor: "pointer", display: "flex", alignItems: "center", gap: "4px",
                 fontSize: "11px", color: "#888", margin: "6px 0 2px", userSelect: "none" },
        title: "Instruction and frame-sampling settings used by the Describe button on the clip below.",
    }, ["▸ Describe settings (instruction, max frames, every Nth)", dsChevron]);
    const describeSettingsBody = _mk("div",
        { cls: "spe-collapsible" + (_S.clipsDescribeSettingsOpen ? "" : " collapsed") });
    describeSettingsTitle.onclick = () => {
        _S.clipsDescribeSettingsOpen = !_S.clipsDescribeSettingsOpen;
        describeSettingsBody.classList.toggle("collapsed", !_S.clipsDescribeSettingsOpen);
        dsChevron.className = "pi pi-chevron-" + (_S.clipsDescribeSettingsOpen ? "up" : "down");
    };
    describeSettingsBody.append(promptOverrideWrap, descFrameRow);
    body.append(describeSettingsTitle, describeSettingsBody);

    // ── Clip list ─────────────────────────────────────────────────────────────
    const listEl = _mk("div");
    body.appendChild(listEl);

    const addClipBtn = _mk("button", { cls: "spe-btn sm", style: { marginTop: "4px" },
        onclick: addNewClip }, ["+ Add clip"]);
    body.appendChild(addClipBtn);

    // ── Toggle ────────────────────────────────────────────────────────────────
    // ── Build-all-proxies button (sits in the title row) ─────────────────────
    const proxyBuildStatusEl = _mk("span", {
        cls: "spe-clips-llm-note",
        style: { fontSize: "11px", color: "#888", marginRight: "6px", display: "none" },
    }, [""]);
    const buildAllBtn = _mk("button", { cls: "spe-btn sm ghost",
        title: "Pre-build proxies for all clips in this profile",
        style: { marginRight: "4px" },
        onclick: async e => {
            e.stopPropagation();
            if (_proxyBuildActive) return;  // a build (batch or single-clip) is already running
            buildAllBtn.disabled = true;
            buildAllBtn.textContent = "building…";
            try {
                const res = await sourceProfilesApi.prebuildProxies({ profile_id: profile.id });
                const n = res.clip_count || 0;
                _toast(`Building ${n} proxy clip${n !== 1 ? "s" : ""} in background`, "info");
                proxyBuildStatusEl.textContent = "Starting…";
                proxyBuildStatusEl.style.display = "";
                // Stays disabled until the "complete" status arrives — see
                // _onProxyBuildStatus, not reset here since the background job
                // keeps running well after this request itself returns.
                _proxyBuildActive = true;
            } catch (err) {
                _toast(`Proxy build failed: ${err.message}`, "error");
                buildAllBtn.disabled = false;
                buildAllBtn.textContent = "build proxies";
            }
        } }, ["build proxies"]);
    titleRow.insertBefore(proxyBuildStatusEl, chevron);
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
        _updateZoomArrows();
        _drawTimeline(canvas, clips, suggestions, getTotalDuration(), activeClipIdx,
            new Set(clips.filter(_isProxyDirty).map(c => c.id)), hoverClipIdx, _visibleClipRange());
        renderClipList();
        _updatePrecisionReadout();
    }

    // ── Timeline: hover + click-to-select ────────────────────────────────────
    //
    // Boundary-handle drag-to-resize used to live here, but with many short
    // clips the 8px handle hitbox was too easy to grab by accident while just
    // trying to select a clip. Resizing now happens exclusively through the
    // clip's start/end inputs (also generally easier to line up precisely);
    // the timeline is click-to-select only.
    //
    // _timeToX mirrors _drawTimeline's own coordinate mapping so hover/click
    // hit-testing lines up with whatever's actually on screen — the full
    // timeline, or (once zoomed) just the current window of clips.
    function _timeToX(t, range) {
        if (!range) return (t / (getTotalDuration() || 1)) * canvas.offsetWidth;
        return ((t - range[0]) / Math.max(range[1] - range[0], 0.001)) * canvas.offsetWidth;
    }

    function _updateHoverTooltip(e, range) {
        if (hoverClipIdx < 0) { hoverTooltipEl.style.display = "none"; return; }
        const clip = clips[hoverClipIdx];
        const x1 = _timeToX(clip.start_time, range);
        const x2 = _timeToX(clip.end_time, range);
        if (x2 - x1 >= 40) { hoverTooltipEl.style.display = "none"; return; }
        const dur = (clip.end_time - clip.start_time).toFixed(1);
        hoverTooltipEl.textContent =
            `${clip.label || `Clip ${hoverClipIdx + 1}`}  ${clip.start_time.toFixed(1)}–${clip.end_time.toFixed(1)}s  (${dur}s)`;
        const wrapRect = canvasInner.getBoundingClientRect();
        const tipWidth = hoverTooltipEl.offsetWidth || 160;
        const mouseX   = e.clientX - wrapRect.left;
        hoverTooltipEl.style.left = Math.max(0, Math.min(canvas.offsetWidth - tipWidth, mouseX - tipWidth / 2)) + "px";
        hoverTooltipEl.style.display = "block";
    }

    canvas.addEventListener("mousemove", e => {
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const range = _visibleClipRange();
        const BAND_TOP = 10, BAND_BOT = canvas.height - 20;
        const y = e.clientY - rect.top;

        // Handle hover tracking
        let newHoverIdx = -1;
        if (y >= BAND_TOP - 6 && y <= BAND_BOT) {
            for (let i = 0; i < clips.length; i++) {
                const x1 = _timeToX(clips[i].start_time, range);
                const x2 = _timeToX(clips[i].end_time, range);
                if (x >= x1 && x <= x2) { newHoverIdx = i; break; }
            }
        }
        if (newHoverIdx !== hoverClipIdx) {
            hoverClipIdx = newHoverIdx;
            canvas.width = canvas.offsetWidth;
            _drawTimeline(canvas, clips, suggestions, getTotalDuration(), activeClipIdx,
                new Set(clips.filter(_isProxyDirty).map(c => c.id)), hoverClipIdx, range);
        }
        _updateHoverTooltip(e, range);

        // Cursor: pointer over a clip band (click to select), default otherwise.
        let overBand = false;
        if (y >= BAND_TOP && y <= BAND_BOT) {
            for (let i = 0; i < clips.length; i++) {
                const x1 = _timeToX(clips[i].start_time, range);
                const x2 = _timeToX(clips[i].end_time, range);
                if (x >= x1 && x <= x2) { overBand = true; break; }
            }
        }
        canvas.style.cursor = overBand ? "pointer" : "default";
    });

    canvas.addEventListener("mouseup", e => {
        // Click to select a clip band
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const range = _visibleClipRange();
        for (let i = 0; i < clips.length; i++) {
            const x1 = _timeToX(clips[i].start_time, range);
            const x2 = _timeToX(clips[i].end_time, range);
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
    });

    // ── Proxy status helpers ──────────────────────────────────────────────────
    let _proxyStatusMap = {}; // clip_id → { fresh: bool }
    // True from the moment any proxy build (batch or single-clip) is kicked
    // off until its "complete" status arrives — guards against overlapping
    // submissions, which is exactly what was flooding upsert_clip: every
    // progress event during a build re-triggers _refreshProxyStatus(), and a
    // second build starting mid-flight multiplies that further.
    let _proxyBuildActive = false;

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
                    // Only persist once per clip: every progress event during a
                    // build calls this for *all* clips, not just the one that
                    // just finished — without this guard, a profile with many
                    // clips fires roughly N² upsert_clip calls before the batch
                    // is done (this is what was flooding the server).
                    if (idx >= 0 && (!clips[idx].proxy_built_at || _isProxyDirty(clips[idx]))) {
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

    // Live proxy-build progress. The backend already broadcasts per-clip status
    // via send_status_update(source="proxy_build") during prebuild_proxies — this
    // just wires up a listener so it actually reaches the UI instead of going
    // unheard, and refreshes badges the moment each clip's build finishes rather
    // than waiting on a guessed timeout or a slow poll loop.
    let _proxyBuildStatusHideTimer = null;
    function _onProxyBuildStatus(event) {
        const d = event?.detail || {};
        if (d.source !== "proxy_build") return;
        const msg = String(d.status || "").trim();
        if (msg) {
            proxyBuildStatusEl.textContent = msg;
            proxyBuildStatusEl.style.display = "";
            clearTimeout(_proxyBuildStatusHideTimer);
            if (/complete/i.test(msg)) {
                _proxyBuildActive = false;
                buildAllBtn.disabled = false;
                buildAllBtn.textContent = "build proxies";
                _proxyBuildStatusHideTimer = setTimeout(() => {
                    proxyBuildStatusEl.style.display = "none";
                }, 4000);
            }
        }
        _refreshProxyStatus();  // also re-renders the clip list, picking up _proxyBuildActive
    }
    api.addEventListener("fbtools.status", _onProxyBuildStatus);

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
        navEl.innerHTML = "";
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
        const splitBtn = _mk("button", { cls: "spe-btn sm ghost",
            title: getVideoTime
                ? "Split this clip at the current video playback position (seek into clip range first)"
                : "Video not available for split",
            style: { fontSize: "10px" },
            disabled: !getVideoTime,
        }, ["✂ split"]);
        if (i === 0) prevBtn.disabled = true;
        if (i === clips.length - 1) { nextBtn.disabled = true; mergeBtn.disabled = true; }
        const navLabel = _mk("span", { cls: "spe-clip-nav-label" },
            [`${clips[i].label || "Segment " + (i + 1)}  (${i + 1}/${clips.length})`]);
        // Blur before redraw(): redraw() rebuilds navEl (innerHTML = ""), destroying
        // this very button while it still holds focus. Left unblurred, the browser's
        // default focus-recovery (focus reverts to <body>) triggers the panel host to
        // scroll the whole view back to the top — clicking the timeline directly
        // doesn't hit this because <canvas> isn't focusable, so nothing is destroyed
        // out from under the focused element there.
        prevBtn.onclick  = () => { prevBtn.blur(); activeClipIdx = Math.max(0, i - 1); redraw(); onSelect?.(clips[activeClipIdx].start_time); };
        nextBtn.onclick  = () => { nextBtn.blur(); activeClipIdx = Math.min(clips.length - 1, i + 1); redraw(); onSelect?.(clips[activeClipIdx].start_time); };
        mergeBtn.onclick = () => mergeWithNext(i);
        splitBtn.onclick = () => splitClip(i);
        navEl.appendChild(_mk("div", { cls: "spe-clip-nav" }, [prevBtn, navLabel, nextBtn]));
        // Merge/split on their own centered row — kept away from the ←/→ arrows
        // so a misaimed click can't accidentally merge two clips together.
        navEl.appendChild(_mk("div", {
            style: { display: "flex", justifyContent: "center", gap: "8px", margin: "4px 0" },
        }, [mergeBtn, splitBtn]));

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
                    renderClipList();  // rebuilds the card, so boundary thumbnails
                                       // below refresh at the newly committed times
                }
            };
            // change fires for both stepper-arrow clicks and blur-after-typing;
            // Enter alone doesn't blur a bare <input>, so handle it explicitly too.
            startEl.onchange = applyTimes; endEl.onchange = applyTimes;
            const _commitOnEnter = e => { if (e.key === "Enter") applyTimes(); };
            startEl.addEventListener("keydown", _commitOnEnter);
            endEl.addEventListener("keydown", _commitOnEnter);

            const durSpan = _mk("span", { style: { color: "#888" } },
                [`(${(clip.end_time - clip.start_time).toFixed(1)}s)`]);

            // ── Boundary thumbnails ────────────────────────────────────────────
            const canPreview = profile.media_type === "video" && !!profile.media_filename && !!profile.id;
            const _mkBoundThumb = () => canPreview
                ? _mk("img", { cls: "spe-bound-thumb" })
                : _mk("div", { cls: "spe-bound-thumb spe-bound-thumb-empty" },
                      [profile.id ? "no preview" : "save profile for preview"]);
            const startThumb = _mkBoundThumb();
            const endThumb   = _mkBoundThumb();
            if (canPreview) {
                startThumb.src = sourceProfilesApi.frameAtUrl(profile.id, clip.start_time, 160);
                endThumb.src   = sourceProfilesApi.frameAtUrl(profile.id, clip.end_time, 160);
                startThumb.onerror = () => { startThumb.style.visibility = "hidden"; };
                endThumb.onerror   = () => { endThumb.style.visibility   = "hidden"; };
            }
            const boundsRow = _mk("div", { cls: "spe-bounds-row" }, [
                _mk("div", { cls: "spe-bound-group start" }, [
                    startThumb,
                    _mk("div", { cls: "spe-bound-ctl" }, [_mk("label", {}, ["Start"]), startEl]),
                ]),
                _mk("div", { cls: "spe-bound-mid" }, [durSpan]),
                _mk("div", { cls: "spe-bound-group end" }, [
                    endThumb,
                    _mk("div", { cls: "spe-bound-ctl" }, [_mk("label", {}, ["End"]), endEl]),
                ]),
            ]);

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
                const tagged = new Set(clips[i].subjects || []);
                let slotIdx = 0;
                subjects.forEach(s => {
                    const span = slotSpans[s.id];
                    if (!span) return;
                    if (tagged.has(s.id) && slotIdx < CLIP_SLOTS.length) {
                        span.textContent = ` {${CLIP_SLOTS[slotIdx++]}}`;
                    } else {
                        span.textContent = "";
                    }
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

                const addAllBtn = _mk("button", {
                    cls: "spe-btn sm ghost",
                    title: `Check "${s.label || s.id}" in all segments`,
                    textContent: "→ all",
                    style: { padding: "1px 5px", fontSize: "10px", opacity: "0.6" },
                });
                addAllBtn.addEventListener("click", e => {
                    e.preventDefault();
                    clips.forEach((c, j) => {
                        const cur = new Set(c.subjects || []);
                        if (cur.has(s.id)) return;
                        cur.add(s.id);
                        clips[j] = { ...clips[j], subjects: [...cur] };
                        commitClip(j);
                    });
                    renderClipList();
                    _toast(`"${s.label || s.id}" added to all segments`, "success");
                });

                const rmAllBtn = _mk("button", {
                    cls: "spe-btn sm ghost",
                    title: `Uncheck "${s.label || s.id}" in all segments`,
                    textContent: "✕ all",
                    style: { padding: "1px 5px", fontSize: "10px", opacity: "0.6", color: "var(--p-red-400,#f87171)" },
                });
                rmAllBtn.addEventListener("click", e => {
                    e.preventDefault();
                    clips.forEach((c, j) => {
                        const cur = new Set(c.subjects || []);
                        if (!cur.has(s.id)) return;
                        cur.delete(s.id);
                        clips[j] = { ...clips[j], subjects: [...cur] };
                        commitClip(j);
                    });
                    renderClipList();
                    _toast(`"${s.label || s.id}" removed from all segments`, "success");
                });

                subjWrap.appendChild(
                    _mk("div", { style: { display: "flex", alignItems: "center", gap: "4px", marginBottom: "2px" } }, [
                        _mk("label", { cls: "spe-clip-subj-check", style: { flex: "1", marginBottom: "0" } },
                            [cb, " " + (s.label || s.id), slotSpan]),
                        addAllBtn,
                        rmAllBtn,
                    ])
                );
            });
            updateSlotLabels();

            const applyLoraToAll = (loraEntry) => {
                if (!loraEntry.name) return;
                let applied = 0;
                clips.forEach((c, j) => {
                    if (j === i) return;
                    if ((c.loras || []).some(l => l.name === loraEntry.name)) return;
                    clips[j] = { ...clips[j], loras: [...(c.loras || []), { ...loraEntry }] };
                    commitClip(j);
                    applied++;
                });
                _toast(
                    applied ? `LoRA added to ${applied} segment${applied !== 1 ? "s" : ""}` : "All segments already have this LoRA",
                    applied ? "success" : "info"
                );
            };

            const applySoundscapeAllBtn = _mk("button", {
                cls: "spe-btn sm ghost",
                title: "Copy this soundscape to all segments",
                textContent: "→ all",
                style: { padding: "1px 5px", fontSize: "10px", opacity: "0.65" },
            });
            applySoundscapeAllBtn.addEventListener("click", () => {
                const val = soundscapeEl.value;
                clips.forEach((c, j) => {
                    if (j === i) return;
                    clips[j] = { ...clips[j], overall_soundscape: val };
                    commitClip(j);
                });
                _toast("Soundscape applied to all segments", "success");
            });

            const applyMusicAllBtn = _mk("button", {
                cls: "spe-btn sm ghost",
                title: "Copy this music to all segments",
                textContent: "→ all",
                style: { padding: "1px 5px", fontSize: "10px", opacity: "0.65" },
            });
            applyMusicAllBtn.addEventListener("click", () => {
                const val = musicEl.value;
                clips.forEach((c, j) => {
                    if (j === i) return;
                    clips[j] = { ...clips[j], non_diegetic_music: val };
                    commitClip(j);
                });
                _toast("Music applied to all segments", "success");
            });

            const loraSection = _buildClipLoraSection(clips[i], () => commitClip(i), applyLoraToAll);

            const dlgCb = _mk("input", { type: "checkbox" });
            dlgCb.checked = clip.allows_dialogue !== false;
            dlgCb.onchange = () => { clips[i] = { ...clips[i], allows_dialogue: dlgCb.checked }; commitClip(i); };

            const clipBody = _mk("div", { cls: "spe-clip-card-body" }, [
                _mk("div", { cls: "spe-clip-row", style: { marginBottom: "6px" } }, [labelInp]),
                boundsRow,
                actionEl,
                subjects.length ? subjWrap : null,
                _mk("div", { style: { display: "flex", gap: "4px" } }, [describeBtn]),
                _mk("div", { style: { display: "flex", alignItems: "center", gap: "6px", marginBottom: "2px" } }, [
                    _mk("span", { cls: "spe-clip-field-label", style: { flex: "1", marginBottom: "0" } }, ["Overall soundscape"]),
                    applySoundscapeAllBtn,
                ]),
                soundscapeEl,
                _mk("div", { style: { display: "flex", alignItems: "center", gap: "6px", marginBottom: "2px" } }, [
                    _mk("span", { cls: "spe-clip-field-label", style: { flex: "1", marginBottom: "0" } }, ["Non-diegetic music"]),
                    applyMusicAllBtn,
                ]),
                musicEl,
                _mk("label", { cls: "spe-clip-subj-check", style: { marginTop: "4px" }, title: "When off, dialogue text from cast entries is ignored for this segment, and Scene Cast \"Audio\" (voice-timbre extraction) is skipped too — no audio involvement for this clip at all" }, [dlgCb, " Allows dialogue"]),
                loraSection,
            ]);

            const proxyBadge = _proxyBadgeEl(clip.id);
            const buildProxyBtn = _mk("button", { cls: "spe-btn sm ghost",
                title: _proxyBuildActive
                    ? "A proxy build is already running"
                    : "Pre-build proxy for this clip",
                disabled: _proxyBuildActive,
                onclick: async e => {
                    e.stopPropagation();
                    if (_proxyBuildActive) return;  // another build (batch or single-clip) is running
                    buildProxyBtn.disabled = true;
                    buildProxyBtn.textContent = "…";
                    try {
                        const res = await sourceProfilesApi.prebuildProxies({ profile_id: profile.id, clip_id: clip.id });
                        if ((res.clip_count ?? 1) === 0) {
                            _toast("Clip not found on server — try saving the profile first", "warn");
                            buildProxyBtn.disabled = false;
                            buildProxyBtn.textContent = "proxy";
                        } else {
                            _toast(`Building proxy for "${clip.label || `Clip ${i + 1}`}" in background`, "info");
                            proxyBuildStatusEl.textContent = "Starting…";
                            proxyBuildStatusEl.style.display = "";
                            // Stays disabled until the "complete" status arrives — see
                            // _onProxyBuildStatus. The 90s timeout below is only a safety
                            // net in case that event gets missed (e.g. a dropped websocket
                            // message), so this clip's badge doesn't get stuck stale forever.
                            _proxyBuildActive = true;
                            setTimeout(() => {
                                if (!_proxyStatusMap[clip.id]?.fresh) _refreshProxyStatus();
                            }, 90000);
                        }
                    } catch (err) {
                        _toast(`Proxy build failed: ${err.message}`, "error");
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
        _updateZoomArrows();
        _drawTimeline(canvas, clips, suggestions, getTotalDuration(), activeClipIdx,
            new Set(clips.filter(_isProxyDirty).map(c => c.id)), hoverClipIdx, _visibleClipRange());
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

    function splitClip(i) {
        if (!getVideoTime) { _toast("No video time source available", "warn"); return; }
        const t = getVideoTime();
        const clip = clips[i];
        if (t <= clip.start_time || t >= clip.end_time) {
            _toast(
                `Current time (${t.toFixed(1)}s) is outside clip range ` +
                `(${clip.start_time.toFixed(1)}–${clip.end_time.toFixed(1)}s)`,
                "warn"
            );
            return;
        }
        const baseLabel = clip.label || `Clip ${i + 1}`;
        // Reuse original id for first half (upsert updates it), generate new id for second half
        const clipA = { ...clip, end_time: t, label: baseLabel + " A" };
        const clipB = { ...clip, id: _genClipId(), start_time: t, label: baseLabel + " B" };
        clips = [...clips.slice(0, i), clipA, clipB, ...clips.slice(i + 1)];
        profile.clips = clips;
        onClipsChanged(clips);
        _markTimesDirty(i);
        _markTimesDirty(i + 1);
        _persistClip(clipA);
        _persistClip(clipB);
        redraw();
        _toast(`Split at ${t.toFixed(1)}s → "${clipA.label}" + "${clipB.label}"`, "success");
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
        detectProgress = null;
        _renderDetectProgress();
        // Timeline stays visible for orientation but edits are disabled while
        // boundaries are being detected — see the mousedown guard below.
        canvasWrap.style.opacity = "0.6";
        canvasWrap.style.pointerEvents = "none";

        // Live status updates from the backend — update spinner text and the
        // per-window progress strip as each window is processed.
        const _onStatus = (event) => {
            const d = event?.detail || {};
            if (d.source !== "source_profile_analysis") return;
            const msg = String(d.status || "").trim();
            if (msg && detecting) detectSpinner.textContent = ` ${msg}`;
            if (d.profile_id && d.profile_id !== profile.id) return;
            if (d.phase === "start") {
                detectProgress = {
                    windows: d.windows || [],
                    status:  (d.windows || []).map(() => "pending"),
                    segs:    (d.windows || []).map(() => 0),
                    elapsed: (d.windows || []).map(() => 0),
                    frames:  (d.windows || []).map(() => 0),
                };
                _renderDetectProgress();
            } else if (d.phase === "window_start" && detectProgress) {
                detectProgress.status[d.window_idx] = "active";
                _renderDetectProgress();
            } else if (d.phase === "window_done" && detectProgress) {
                detectProgress.status[d.window_idx]  = "done";
                detectProgress.segs[d.window_idx]    = d.segments_found ?? 0;
                detectProgress.elapsed[d.window_idx] = d.elapsed_s ?? 0;
                detectProgress.frames[d.window_idx]  = d.frames ?? 0;
                _renderDetectProgress();
            }
        };
        api.addEventListener("fbtools.status", _onStatus);

        try {
            await onEnsureSaved?.();
            const res = await sourceProfilesApi.detectSegments({
                profile_id:       profile.id,
                video_duration:   totalDur,
                interval_seconds: detectIntervalSeconds,
                prompt_override:  detectPromptOverride.trim(),
                flags:            detectPromptOverride.trim() ? null : { ...detectFlags },
                captioner_type:   getActiveCaptionerType(),
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
            canvasWrap.style.opacity = "";
            canvasWrap.style.pointerEvents = "";
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
        const taggedSet = new Set(clip.subjects || []);
        const allSubjects = profile.subjects || [];
        const subjects = allSubjects
            .filter(s => taggedSet.has(s.id))
            .slice(0, SLOTS.length)
            .map((s, idx) => ({
                slot:       SLOTS[idx],
                name:       s.label || s.id,
                appearance: s.role_description || "",
            }));

        try {
            await onEnsureSaved?.();
            const res = await sourceProfilesApi.describeClip({
                profile_id:       profile.id,
                start_time:       clip.start_time,
                end_time:         clip.end_time,
                subjects,
                existing_action:  clip.action || "",
                captioner_type:   getActiveCaptionerType(),
                prompt_override:  profile.describe_prompt_override || "",
                max_frames:       profile.describe_max_frames       ?? 5,
                select_every_nth: profile.describe_select_every_nth ?? 1,
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
        type: "text", placeholder: "e.g. young woman, demonic warrior, stone hallway",
        value: initial.short_name || "",
    });
    const shortNameRow = _mk("div", { cls: "spe-form-row" }, [
        _mk("label", {}, ["Short name (compact descriptor)"]), shortNameEl,
    ]);
    pronounEl.addEventListener("change", () => {});

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
        }, () => {
            const vid = previewWrap.querySelector("video");
            return vid ? vid.currentTime : 0;
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

    // Pass type checkboxes (multi-select)
    body.appendChild(_mk("div", { cls: "spe-section-head" }, ["Focus passes (select one or more)"]));
    const pillsWrap = _mk("div", { cls: "spe-pass-pills", style: { flexWrap: "wrap", gap: "8px" } });
    let _updateDefaultPreview;
    PASS_TYPES.forEach(pt => {
        const uid = `spe-pass-cb-${pt}-${profile.id || "new"}`;
        const cb  = _mk("input", { type: "checkbox", id: uid });
        cb.checked = _S.analyzePassTypes.has(pt);
        cb.onchange = () => {
            if (cb.checked) {
                _S.analyzePassTypes.add(pt);
            } else {
                _S.analyzePassTypes.delete(pt);
                if (_S.analyzePassTypes.size === 0) {
                    _S.analyzePassTypes.add(pt);
                    cb.checked = true;
                }
            }
            _updateDefaultPreview?.();
        };
        const lbl = _mk("label", {
            htmlFor: uid,
            cls: "spe-clip-subj-check",
            style: { cursor: "pointer", gap: "4px", marginBottom: "0", fontSize: "12px" },
        }, [cb, " " + PASS_LABELS[pt]]);
        pillsWrap.appendChild(lbl);
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
        const types = [..._S.analyzePassTypes];
        if (types.length === 1) {
            defaultPreviewEl.value = PASS_DEFAULT_PROMPTS[types[0]] ?? "";
        } else {
            defaultPreviewEl.value = types
                .map(pt => `[${PASS_LABELS[pt]}]\n${PASS_DEFAULT_PROMPTS[pt] ?? ""}`)
                .join("\n\n─────\n\n");
        }
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

    // History section — tabbed: Analyze runs | Describe runs
    const histToggle = _mk("div", { cls: "spe-section-head spe-collapse-toggle",
        style: { cursor: "pointer", marginTop: "8px" } }, ["▶ Previous runs"]);
    const histWrap = _mk("div", { cls: "spe-collapsible collapsed" });

    // Tab bar
    const histTabBar  = _mk("div", { style: { display: "flex", gap: "4px", marginBottom: "6px" } });
    const histTabAnalyze  = _mk("button", { cls: "spe-btn sm", style: { fontWeight: "600" } }, ["Analyze"]);
    const histTabDescribe = _mk("button", { cls: "spe-btn sm ghost" }, ["Describe"]);
    histTabBar.append(histTabAnalyze, histTabDescribe);
    const histBodyAnalyze  = _mk("div");
    const histBodyDescribe = _mk("div", { style: { display: "none" } });
    histWrap.append(histTabBar, histBodyAnalyze, histBodyDescribe);

    let _activeHistTab = "analyze";
    const _switchHistTab = (tab) => {
        _activeHistTab = tab;
        histTabAnalyze.className  = "spe-btn sm" + (tab === "analyze"  ? "" : " ghost");
        histTabDescribe.className = "spe-btn sm" + (tab === "describe" ? "" : " ghost");
        histBodyAnalyze.style.display  = tab === "analyze"  ? "" : "none";
        histBodyDescribe.style.display = tab === "describe" ? "" : "none";
    };
    histTabAnalyze.onclick  = () => _switchHistTab("analyze");
    histTabDescribe.onclick = () => _switchHistTab("describe");

    histToggle.onclick = async () => {
        const open = histWrap.classList.toggle("collapsed") === false;
        histToggle.textContent = (open ? "▼" : "▶") + " Previous runs";
        if (open) {
            await _loadHistory(profile.id);
            _renderHistory(histBodyAnalyze, profile, onSubjectsChanged);
            _renderDescribeHistory(histBodyDescribe, profile);
        }
    };
    body.appendChild(histToggle);
    body.appendChild(histWrap);

    // Stash refs so _runAnalysis can update them
    body._runBtn          = runBtn;
    body._spinnerEl       = spinnerEl;
    body._candidatesEl    = candidatesEl;
    body._histWrap        = histWrap;
    body._histBodyAnalyze = histBodyAnalyze;
    body._histBodyDescribe= histBodyDescribe;
    body._profile         = profile;
    body._onChanged       = onSubjectsChanged;
}

async function _runAnalysis(profile, analyzeBody, onSubjectsChanged) {
    if (_S.analyzeRunning) return;
    _S.analyzeRunning = true;
    analyzeBody._runBtn.disabled = true;
    analyzeBody._spinnerEl.style.display = "block";
    analyzeBody._candidatesEl.innerHTML  = "";

    const passTypes = [..._S.analyzePassTypes];

    try {
        const clip = (_S.analyzeClipIdx != null)
            ? (profile.clips || [])[_S.analyzeClipIdx]
            : null;

        const res = await sourceProfilesApi.analyze({
            profile_id:       profile.id,
            pass_types:       passTypes,
            prompt_override:  _S.analyzePromptOverride,
            captioner_type:   getActiveCaptionerType(),
            start_time:       clip ? clip.start_time : null,
            end_time:         clip ? clip.end_time   : null,
            select_every_nth: _S.analyzeSelectNth ?? 1,
            max_frames:       _S.analyzeMaxFrames  ?? 20,
            video_duration:   clip ? 0 : 0,
        });

        _S.analyzeCandidates = res.candidates ?? [];
        _renderCandidates(analyzeBody._candidatesEl, profile, onSubjectsChanged);

        // Refresh history
        await _loadHistory(profile.id);
        _refreshHistoryTabs(analyzeBody);

        const passNote = passTypes.length > 1 ? ` (${passTypes.length} categories)` : "";
        _toast(`Found ${_S.analyzeCandidates.length} candidate(s)${passNote}`, "success");
    } catch (err) {
        _toast(`Analysis failed: ${_errMsg(err)}`, "error");
    } finally {
        _S.analyzeRunning = false;
        analyzeBody._runBtn.disabled = false;
        analyzeBody._spinnerEl.style.display = "none";
        analyzeBody._spinnerEl.textContent = "Running…";
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

function _refreshHistoryTabs(analyzeBody) {
    if (analyzeBody._histWrap.classList.contains("collapsed")) return;
    _renderHistory(analyzeBody._histBodyAnalyze, analyzeBody._profile, analyzeBody._onChanged);
    _renderDescribeHistory(analyzeBody._histBodyDescribe, analyzeBody._profile);
}

function _renderHistory(container, profile, onSubjectsChanged) {
    container.innerHTML = "";
    const entries = _S.analyzeHistory.filter(e => e.pass_type !== "describe_clip");
    if (!entries.length) {
        container.appendChild(_mk("div", { style: { color: "#888", fontSize: "11px", padding: "4px 0" } }, ["No analysis runs yet."]));
        return;
    }
    entries.forEach(entry => {
        const ts      = entry.timestamp?.replace("T", " ").slice(0, 16) ?? "";
        const n       = entry.candidates?.length ?? 0;
        const ptLabel = entry.pass_types
            ? entry.pass_types.map(pt => PASS_LABELS[pt] ?? pt).join(" + ")
            : (PASS_LABELS[entry.pass_type] ?? entry.pass_type);
        const head = _mk("div", { style: { display: "flex", justifyContent: "space-between", alignItems: "center",
            padding: "4px 0", borderBottom: "1px solid var(--p-surface-border,#444)", cursor: "pointer",
            fontSize: "11px" } }, [
            _mk("span", {}, [`${ptLabel} — ${n} candidate(s)`]),
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

function _renderDescribeHistory(container, profile) {
    container.innerHTML = "";
    const entries = _S.analyzeHistory.filter(e => e.pass_type === "describe_clip");
    if (!entries.length) {
        container.appendChild(_mk("div", { style: { color: "#888", fontSize: "11px", padding: "4px 0" } }, ["No Describe runs yet."]));
        return;
    }
    entries.forEach(entry => {
        const ts    = entry.timestamp?.replace("T", " ").slice(0, 16) ?? "";
        const start = typeof entry.clip_start === "number" ? entry.clip_start.toFixed(1) + "s" : "?";
        const end   = typeof entry.clip_end   === "number" ? entry.clip_end.toFixed(1)   + "s" : "?";
        const action = entry.action || "(no result)";

        const head = _mk("div", { style: { display: "flex", justifyContent: "space-between",
            alignItems: "center", padding: "4px 0",
            borderBottom: "1px solid var(--p-surface-border,#444)", cursor: "pointer",
            fontSize: "11px" } }, [
            _mk("span", {}, [`${start}–${end} clip`]),
            _mk("span", { style: { color: "#888" } }, [ts]),
        ]);
        const detail = _mk("div", { cls: "spe-collapsible collapsed",
            style: { padding: "6px 0", fontSize: "11px" } });
        head.onclick = () => { detail.classList.toggle("collapsed"); };

        // Action result
        detail.appendChild(_mk("div", { cls: "spe-clip-field-label" }, ["Result"]));
        const actionEl = _mk("div", {
            style: { background: "var(--p-surface-ground,#1a1a1a)", border: "1px solid var(--p-surface-border,#444)",
                     borderRadius: "3px", padding: "5px 8px", marginBottom: "6px",
                     whiteSpace: "pre-wrap", wordBreak: "break-word", lineHeight: "1.5" },
        }, [action]);
        detail.appendChild(actionEl);

        // Prompt (collapsible)
        const promptToggle = _mk("div", { style: { color: "var(--p-text-muted-color,#888)", cursor: "pointer",
            fontSize: "10px", marginBottom: "2px" } }, ["▸ Prompt used"]);
        const promptEl = _mk("pre", { style: { fontSize: "10px", whiteSpace: "pre-wrap", wordBreak: "break-all",
            maxHeight: "120px", overflowY: "auto", background: "var(--p-surface-ground,#1a1a1a)",
            border: "1px solid var(--p-surface-border,#444)", borderRadius: "3px",
            padding: "4px 6px", display: "none" } }, [entry.prompt || "(no prompt stored)"]);
        promptToggle.onclick = () => {
            const open = promptEl.style.display === "none";
            promptEl.style.display = open ? "block" : "none";
            promptToggle.textContent = (open ? "▾" : "▸") + " Prompt used";
        };
        detail.append(promptToggle, promptEl);

        container.appendChild(head);
        container.appendChild(detail);
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
