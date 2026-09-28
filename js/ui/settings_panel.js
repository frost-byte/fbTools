/**
 * fbTools global Settings tab.
 *
 * Consolidates all extension-wide preferences in one place:
 *   Compose defaults  — libber delimiter, speech pace, audio processing, vocal isolation
 *   H3 model          — max output frames per clip
 *
 * Settings backed by the server (/fbtools/compositions/settings) persist across
 * browser sessions and machines. The h3_max_frames value is also mirrored to
 * localStorage so nodeCreated hooks can read it synchronously without a fetch.
 */

import { compositionsApi } from "../api/compositions.js";
import { bundlesApi } from "../api/bundles.js";
import { toolsApi } from "../api/tools.js";
import { toast as _toast } from "./library_common.js";

export const LS_H3_MAX = "fbt_h3_max_frames";

// ── DOM helper ─────────────────────────────────────────────────────────────────

function _mk(tag, props = {}, children = []) {
    const el = document.createElement(tag);
    Object.entries(props).forEach(([k, v]) => {
        if (k === "cls")        el.className = v;
        else if (k === "style") Object.assign(el.style, v);
        else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
        else el[k] = v;
    });
    children.filter(Boolean).forEach(c =>
        el.appendChild(typeof c === "string" ? document.createTextNode(c) : c));
    return el;
}

// ── Settings panel ─────────────────────────────────────────────────────────────

export async function renderSettingsPanel(parent) {
    const wrap = _mk("div", { cls: "fbt-settings-wrap" });
    parent.appendChild(wrap);

    // ── Loading state ──────────────────────────────────────────────────────────
    const statusEl = _mk("p", { cls: "fbt-settings-status", textContent: "Loading…" });
    wrap.appendChild(statusEl);

    let settings = {};
    try {
        settings = await compositionsApi.getSettings();
        statusEl.remove();
    } catch {
        statusEl.textContent = "Failed to load settings.";
        return;
    }

    const _save = async (patch) => {
        Object.assign(settings, patch);
        try {
            const updated = await compositionsApi.saveSettings(patch);
            Object.assign(settings, updated);
            if ("h3_max_frames" in patch) {
                try { localStorage.setItem(LS_H3_MAX, String(settings.h3_max_frames)); } catch {}
            }
            document.dispatchEvent(new CustomEvent("fbt:settings-changed", { detail: { ...settings } }));
        } catch {
            console.warn("[fbTools] Settings save failed");
        }
    };

    // Seed localStorage with server value on first load
    try { localStorage.setItem(LS_H3_MAX, String(settings.h3_max_frames ?? 360)); } catch {}

    // ── Section builder ────────────────────────────────────────────────────────

    const _section = (title) => {
        const sec = _mk("div", { cls: "fbt-settings-section" });
        sec.appendChild(_mk("div", { cls: "fbt-settings-section-title", textContent: title }));
        return sec;
    };

    const _row = (label, control, hint) => {
        const row = _mk("div", { cls: "fbt-ce-settings-row" });
        const lbl = _mk("span", { cls: "fbt-ce-settings-label", textContent: label });
        if (hint) lbl.title = hint;
        row.appendChild(lbl);
        row.appendChild(control);
        return row;
    };

    const _groupLabel = (text) =>
        _mk("div", { cls: "fbt-ce-settings-group-label", textContent: text });

    // ── Compose defaults ───────────────────────────────────────────────────────

    const composeSec = _section("Compose Defaults");

    // Libber delimiter
    const delimInp = _mk("input", {
        cls: "fbt-ce-delimiter-input",
        type: "text",
        maxLength: 1,
        value: settings.libber_delimiter ?? "%",
        title: "Single character that wraps libber keys (e.g. %key%)",
    });
    delimInp.addEventListener("change", () => {
        const d = delimInp.value;
        if (!d.length) { delimInp.value = settings.libber_delimiter; return; }
        _save({ libber_delimiter: d });
    });
    composeSec.appendChild(_row("Libber delimiter", delimInp,
        "Single character used to wrap libber keys, e.g. %key%"));

    // Libber max depth
    const depthInp = _mk("input", {
        cls: "fbt-ce-input fbt-ce-settings-lufs",
        type: "number", min: "1", max: "50", step: "1",
        value: settings.libber_max_depth ?? 10,
        title: "Maximum substitution depth for nested libber keys (default 10)",
    });
    depthInp.addEventListener("change", () => {
        const v = parseInt(depthInp.value, 10);
        if (!isNaN(v)) {
            const clamped = Math.max(1, Math.min(50, v));
            depthInp.value = clamped;
            _save({ libber_max_depth: clamped });
        }
    });
    composeSec.appendChild(_row("Libber max depth", depthInp,
        "How many levels deep recursive %key% substitutions are resolved (1–50)"));

    // Default speech pace
    const paceSel = _mk("select", { cls: "fbt-ce-select fbt-ce-settings-sel" });
    [
        { id: "slow",   label: "Slow (~2 words/sec)" },
        { id: "normal", label: "Normal (~2.5 words/sec)" },
        { id: "fast",   label: "Fast (~3 words/sec)" },
    ].forEach(({ id, label }) => {
        const o = _mk("option", { value: id, textContent: label });
        if (id === (settings.default_speech_pace ?? "normal")) o.selected = true;
        paceSel.appendChild(o);
    });
    paceSel.addEventListener("change", () => _save({ default_speech_pace: paceSel.value }));
    composeSec.appendChild(_row("Default speech pace", paceSel,
        "Pre-selected pace for new shots' dialogue"));

    composeSec.appendChild(_groupLabel("Default audio processing"));

    const _cb = (label, key, hint) => {
        const cb = _mk("input", { type: "checkbox" });
        cb.checked = !!(settings[key]);
        cb.addEventListener("change", () => _save({ [key]: cb.checked }));
        return _row(label, cb, hint);
    };

    composeSec.appendChild(_cb("Noise removal", "default_audio_noise_removal",
        "Spectral subtraction applied to new bundles by default"));
    composeSec.appendChild(_cb("LUFS normalize", "default_audio_normalize_lufs",
        "Normalize to target loudness for new bundles by default"));

    const lufsInp = _mk("input", {
        cls: "fbt-ce-input fbt-ce-settings-lufs",
        type: "number", min: "-36", max: "-6", step: "0.5",
        value: settings.default_audio_target_lufs ?? -14.0,
        title: "Target integrated loudness in LUFS (−14 = streaming standard)",
    });
    lufsInp.addEventListener("change", () => {
        const v = parseFloat(lufsInp.value);
        if (!isNaN(v)) {
            const clamped = Math.max(-36, Math.min(-6, v));
            lufsInp.value = clamped;
            _save({ default_audio_target_lufs: clamped });
        }
    });
    const lufsRow = _row("Target LUFS", lufsInp);
    lufsRow.appendChild(_mk("span", { cls: "fbt-be-proc-unit", textContent: "LUFS" }));
    composeSec.appendChild(lufsRow);

    composeSec.appendChild(_groupLabel("Vocal isolation"));

    const mbInp = _mk("input", {
        cls: "fbt-ce-input",
        type: "text",
        style: { flex: "1", minWidth: "0" },
        placeholder: "MelBandRoformer_fp16.safetensors",
        value: settings.melband_model_path ?? "",
        title: "Filename or path to a MelBand Roformer .safetensors checkpoint",
    });
    mbInp.addEventListener("change", () => _save({ melband_model_path: mbInp.value.trim() }));
    const mbRow = _mk("div", { cls: "fbt-ce-settings-row fbt-ce-settings-row--wide" });
    mbRow.appendChild(_mk("span", {
        cls: "fbt-ce-settings-label",
        textContent: "MelBand model path",
        title: "Kijai/MelBandRoFormer_comfy on HuggingFace (fp16 456 MB / fp32 913 MB)",
    }));
    mbRow.appendChild(mbInp);
    composeSec.appendChild(mbRow);

    wrap.appendChild(composeSec);

    // ── H3 model ───────────────────────────────────────────────────────────────

    const h3Sec = _section("H3 Model");

    const h3MaxInp = _mk("input", {
        cls: "fbt-ce-input fbt-settings-h3-max",
        type: "number", min: "0", max: "9999", step: "1",
        value: settings.h3_max_frames ?? 360,
        title: "Hard ceiling on clip_duration_frames output. 0 = unclamped.",
    });
    h3MaxInp.addEventListener("change", () => {
        const v = parseInt(h3MaxInp.value, 10);
        if (!isNaN(v)) {
            const clamped = Math.max(0, Math.min(9999, v));
            h3MaxInp.value = clamped;
            _save({ h3_max_frames: clamped });
        }
    });
    const h3Row = _row("Max output frames", h3MaxInp,
        "Applied as the default ceiling for new SourceProfileClipPrompt nodes. 0 = unclamped.");
    h3Row.appendChild(_mk("span", {
        cls: "fbt-be-proc-unit fbt-settings-h3-hint",
        textContent: `= ${((settings.h3_max_frames ?? 360) / 24).toFixed(1)}s`,
        title: "Approximate seconds at 24 fps",
    }));

    // Update the seconds hint as the user types
    h3MaxInp.addEventListener("input", () => {
        const v = parseInt(h3MaxInp.value, 10);
        const hintEl = h3Row.querySelector(".fbt-settings-h3-hint");
        if (hintEl) hintEl.textContent = isNaN(v) || v === 0 ? "= unclamped" : `= ${(v / 24).toFixed(1)}s`;
    });

    h3Sec.appendChild(h3Row);
    h3Sec.appendChild(_mk("p", {
        cls: "fbt-settings-note",
        textContent: "This is the default applied when a new SourceProfileClipPrompt node is added. "
            + "Each node also has its own Max Clip Frames widget that you can override per-node.",
    }));
    wrap.appendChild(h3Sec);

    // ── H3 Background Plate ────────────────────────────────────────────────────
    // Optional overrides for the "Remove People (H3)" generation (js/ui/background_editor.js).
    // Each control maps to a titled node the exported templates/h3_background_plate.api.json may
    // or may not currently expose (see nodes/backgrounds_presets.py::_backgrounds_h3_settings_options)
    // — unsupported ones render disabled rather than silently doing nothing when set.
    const bgPlateSec = _section("H3 Background Plate");
    const DEFAULT_OPT = "— use template default —";

    let bgOptions = { models: [], clips: [], samplers: [], schedulers: [],
        has_model_override: false, has_clip_override: false, has_lora_override: false,
        has_sampler_override: false, has_scheduler_override: false, template_defaults: {} };
    let loraList = [];
    let optionsError = "";
    try {
        [bgOptions, loraList] = await Promise.all([
            compositionsApi.getH3SettingsOptions(),
            compositionsApi.listLoras().then(r => r.loras ?? []),
        ]);
    } catch (e) {
        optionsError = e.message || "Failed to load options";
    }

    const tplDefaults = bgOptions.template_defaults || {};

    /** A "— use template default (X) —"-first select, disabled with a tooltip when unsupported. */
    const _overrideSelect = (values, currentValue, supported, templateDefault, onChange) => {
        const sel = _mk("select", { cls: "fbt-ce-select fbt-ce-settings-sel",
            style: { flex: "1", minWidth: "0", maxWidth: "260px" } });
        const label = templateDefault ? `${DEFAULT_OPT} (${templateDefault})` : DEFAULT_OPT;
        sel.appendChild(_mk("option", { value: "", textContent: label }));
        values.forEach(v => {
            const o = _mk("option", { value: v, textContent: v });
            if (v === currentValue) o.selected = true;
            sel.appendChild(o);
        });
        sel.disabled = !supported;
        sel.title = supported ? "" : "The current template doesn't expose this override yet — "
            + "see templates/README.md";
        sel.addEventListener("change", onChange);
        return sel;
    };

    const modelSel = _overrideSelect(bgOptions.models, settings.h3_bg_plate_model,
        bgOptions.has_model_override, tplDefaults.model, () => _save({ h3_bg_plate_model: modelSel.value }));
    bgPlateSec.appendChild(_row("Model", modelSel, "Diffusion model checkpoint override"));

    const clipSel = _overrideSelect(bgOptions.clips, settings.h3_bg_plate_clip,
        bgOptions.has_clip_override, tplDefaults.clip, () => _save({ h3_bg_plate_clip: clipSel.value }));
    bgPlateSec.appendChild(_row("Clip", clipSel, "Text encoder / CLIP model override"));

    const samplerSel = _overrideSelect(bgOptions.samplers, settings.h3_bg_plate_sampler,
        bgOptions.has_sampler_override, tplDefaults.sampler, () => _save({ h3_bg_plate_sampler: samplerSel.value }));
    bgPlateSec.appendChild(_row("Sampler", samplerSel, "Sampler algorithm override"));

    const schedulerSel = _overrideSelect(bgOptions.schedulers, settings.h3_bg_plate_scheduler,
        bgOptions.has_scheduler_override, tplDefaults.scheduler,
        () => _save({ h3_bg_plate_scheduler: schedulerSel.value }));
    const stepsInp = _mk("input", {
        cls: "fbt-ce-input fbt-ce-settings-lufs",
        type: "number", min: "0", max: "10000", step: "1",
        value: settings.h3_bg_plate_steps || "",
        placeholder: tplDefaults.steps != null ? String(tplDefaults.steps) : "default",
        title: `Sampler steps override. Blank = use the template's own value`
            + (tplDefaults.steps != null ? ` (currently ${tplDefaults.steps}).` : "."),
    });
    stepsInp.disabled = !bgOptions.has_scheduler_override;
    stepsInp.addEventListener("change", () => {
        const v = parseInt(stepsInp.value, 10);
        const clamped = isNaN(v) || v <= 0 ? 0 : Math.max(1, Math.min(10000, v));
        stepsInp.value = clamped || "";
        _save({ h3_bg_plate_steps: clamped });
    });
    stepsInp.style.flex     = "0 0 auto";
    stepsInp.style.width    = "72px";
    const schedulerGroup = _mk("div", { style: { display: "flex", alignItems: "center", gap: "6px", flex: "1", minWidth: "0" } },
        [schedulerSel, _mk("span", { cls: "fbt-be-proc-unit", textContent: "Steps" }), stepsInp]);
    bgPlateSec.appendChild(_row("Scheduler", schedulerGroup, "Scheduler algorithm override, plus a steps override"));

    const loraSel = _overrideSelect(loraList, settings.h3_bg_plate_lora,
        bgOptions.has_lora_override, tplDefaults.lora, () => _save({ h3_bg_plate_lora: loraSel.value }));
    // Only show a real number here when an override is actually active (h3_bg_plate_lora set) —
    // otherwise leave it blank with the template's own current strength as the placeholder, so
    // this never displays a number that looks live but isn't actually being applied.
    const loraStrengthPlaceholder = tplDefaults.lora_strength != null ? String(tplDefaults.lora_strength) : "default";
    const loraStrengthInp = _mk("input", {
        cls: "fbt-ce-input fbt-ce-settings-lufs",
        type: "number", min: "0", max: "2", step: "0.05",
        value: settings.h3_bg_plate_lora ? (settings.h3_bg_plate_lora_strength ?? "") : "",
        placeholder: loraStrengthPlaceholder,
        title: `LoRA strength (applied to both model and CLIP) — only used while a LoRA above is `
            + `selected. Blank = use the template's own value (currently ${loraStrengthPlaceholder}).`,
    });
    loraStrengthInp.disabled = !bgOptions.has_lora_override;
    loraStrengthInp.addEventListener("change", () => {
        const v = parseFloat(loraStrengthInp.value);
        if (!isNaN(v)) {
            const clamped = Math.max(0, Math.min(2, v));
            loraStrengthInp.value = clamped;
            _save({ h3_bg_plate_lora_strength: clamped });
        }
    });
    loraStrengthInp.style.flex  = "0 0 auto";
    loraStrengthInp.style.width = "64px";
    const loraStrengthLabel = _mk("span", { cls: "fbt-be-proc-unit", textContent: "Weight" });
    const loraGroup = _mk("div", { style: { display: "flex", alignItems: "center", gap: "6px", flex: "1", minWidth: "0" } },
        [loraSel, loraStrengthLabel, loraStrengthInp]);
    bgPlateSec.appendChild(_row("LoRA", loraGroup,
        "Optional turbo/style LoRA override — the Weight value is applied to both the LoRA's "
        + "model and CLIP strength"));

    bgPlateSec.appendChild(_groupLabel("Memory"));
    bgPlateSec.appendChild(_cb("Unload model after each run", "h3_bg_plate_unload_after_run",
        "Off (default) leaves the model resident so chaining several \"Remove People\" passes back "
        + "to back stays fast. On unloads it after every run, trading that speed for freed VRAM."));

    const freeVramBtn = _mk("button", { cls: "fbt-ce-btn sm", textContent: "Free VRAM now",
        title: "Unload resident models immediately — same effect as Manager's own "
            + "\"Free model and node cache\" button.",
        onclick: async () => {
            freeVramBtn.disabled    = true;
            freeVramBtn.textContent = "Freeing…";
            try {
                await compositionsApi.freeH3Vram();
                _toast("VRAM freed", "success");
            } catch (e) {
                alert(`Free VRAM failed: ${e.message}`);
            } finally {
                freeVramBtn.disabled    = false;
                freeVramBtn.textContent = "Free VRAM now";
            }
        } });
    bgPlateSec.appendChild(_row("", freeVramBtn));

    if (optionsError) {
        bgPlateSec.appendChild(_mk("p", { cls: "fbt-settings-note",
            textContent: `Couldn't load model/LoRA options (${optionsError}) — all overrides above `
                + "are disabled until this loads. Reopen Settings to retry." }));
    } else {
        bgPlateSec.appendChild(_mk("p", { cls: "fbt-settings-note",
            textContent: "Every field above is optional and only takes effect once the exported "
                + "template exposes the matching titled node (disabled fields don't yet — see "
                + "templates/README.md). A turbo LoRA is typically tuned for a low step count "
                + "(often as low as 8); pairing one with a mismatched step count, or a non-turbo "
                + "model with too few steps, can produce poor results — tune Steps to match your "
                + "Model/LoRA choice." }));
    }
    wrap.appendChild(bgPlateSec);

    // ── H3 Character Sheet ─────────────────────────────────────────────────────
    // Optional overrides for the Bundle editor's "Generate Character/Face Sheet" action. Same
    // disabled-until-supported pattern as H3 Background Plate above — see
    // nodes/h3_character_sheet.py::_bundles_character_sheet_settings_options and
    // templates/README.md's "h3_character_sheet.api.json" section. VRAM is shared machine-wide
    // state, so "Unload model after each run" above (h3_bg_plate_unload_after_run) also governs
    // this feature — there's no separate checkbox here, just a matching Free VRAM button.
    const charSheetSec = _section("H3 Character Sheet");

    let csOptions = { models: [], clips: [], samplers: [], schedulers: [], aspect_ratios: [],
        has_model_override: false, has_clip_override: false, has_lora_override: false,
        has_sampler1_override: false, has_scheduler1_override: false, has_sampler2_override: false,
        has_upscale_steps_select_override: false, has_upscale_factor_override: false,
        has_aspect_ratio_override: false, template_defaults: {} };
    let csLoraList = [];
    let csOptionsError = "";
    try {
        [csOptions, csLoraList] = await Promise.all([
            bundlesApi.getCharSheetSettingsOptions(),
            compositionsApi.listLoras().then(r => r.loras ?? []),
        ]);
    } catch (e) {
        csOptionsError = e.message || "Failed to load options";
    }

    const csDefaults = csOptions.template_defaults || {};

    const csModelSel = _overrideSelect(csOptions.models, settings.h3_char_sheet_model,
        csOptions.has_model_override, csDefaults.model, () => _save({ h3_char_sheet_model: csModelSel.value }));
    charSheetSec.appendChild(_row("Model", csModelSel, "Diffusion model checkpoint override"));

    const csClipSel = _overrideSelect(csOptions.clips, settings.h3_char_sheet_clip,
        csOptions.has_clip_override, csDefaults.clip, () => _save({ h3_char_sheet_clip: csClipSel.value }));
    charSheetSec.appendChild(_row("Clip", csClipSel, "Text encoder / CLIP model override"));

    const csSampler1Sel = _overrideSelect(csOptions.samplers, settings.h3_char_sheet_sampler1,
        csOptions.has_sampler1_override, csDefaults.sampler1,
        () => _save({ h3_char_sheet_sampler1: csSampler1Sel.value }));
    charSheetSec.appendChild(_row("Sampler (1st pass)", csSampler1Sel, "Base-pass sampler algorithm override"));

    const csScheduler1Sel = _overrideSelect(csOptions.schedulers, settings.h3_char_sheet_scheduler1,
        csOptions.has_scheduler1_override, csDefaults.scheduler1,
        () => _save({ h3_char_sheet_scheduler1: csScheduler1Sel.value }));
    charSheetSec.appendChild(_row("Scheduler (1st pass)", csScheduler1Sel,
        "Base-pass scheduler override — steps for this pass come from the selected mode "
        + "(Character/Face Sheet), not from Settings"));

    const csSampler2Sel = _overrideSelect(csOptions.samplers, settings.h3_char_sheet_sampler2,
        csOptions.has_sampler2_override, csDefaults.sampler2,
        () => _save({ h3_char_sheet_sampler2: csSampler2Sel.value }));
    charSheetSec.appendChild(_row("Sampler (upscale pass)", csSampler2Sel, "Upscale-pass sampler algorithm override"));

    const csUpscaleStepsInp = _mk("input", {
        cls: "fbt-ce-input fbt-ce-settings-lufs",
        type: "number", min: "1", max: "10", step: "1",
        value: settings.h3_char_sheet_upscale_steps_select || "",
        placeholder: csDefaults.upscale_steps_select != null ? String(csDefaults.upscale_steps_select) : "default",
        title: "Upscale-pass sigma preset select (the workflow's 3/4/5-step curve choice). "
            + "Blank = use the template's own value"
            + (csDefaults.upscale_steps_select != null ? ` (currently ${csDefaults.upscale_steps_select}).` : "."),
    });
    csUpscaleStepsInp.disabled = !csOptions.has_upscale_steps_select_override;
    csUpscaleStepsInp.addEventListener("change", () => {
        const v = parseInt(csUpscaleStepsInp.value, 10);
        const clamped = isNaN(v) || v <= 0 ? 0 : Math.max(1, Math.min(10, v));
        csUpscaleStepsInp.value = clamped || "";
        _save({ h3_char_sheet_upscale_steps_select: clamped });
    });
    charSheetSec.appendChild(_row("Upscale Steps Preset", csUpscaleStepsInp,
        "Which preset sigma curve the upscale pass uses"));

    const csUpscaleFactorInp = _mk("input", {
        cls: "fbt-ce-input fbt-ce-settings-lufs",
        type: "number", min: "0.1", max: "8", step: "0.1",
        value: settings.h3_char_sheet_upscale_factor || "",
        placeholder: csDefaults.upscale_factor != null ? String(csDefaults.upscale_factor) : "default",
        title: "Upscale scale multiplier. Blank = use the template's own value"
            + (csDefaults.upscale_factor != null ? ` (currently ${csDefaults.upscale_factor}).` : "."),
    });
    csUpscaleFactorInp.disabled = !csOptions.has_upscale_factor_override;
    csUpscaleFactorInp.addEventListener("change", () => {
        const v = parseFloat(csUpscaleFactorInp.value);
        const clamped = isNaN(v) || v <= 0 ? 0 : Math.max(0.1, Math.min(8, v));
        csUpscaleFactorInp.value = clamped || "";
        _save({ h3_char_sheet_upscale_factor: clamped });
    });
    charSheetSec.appendChild(_row("Upscale Factor", csUpscaleFactorInp, "Scale multiplier for the upscale pass"));

    const csAspectSel = _overrideSelect(csOptions.aspect_ratios || [], settings.h3_char_sheet_aspect_ratio,
        csOptions.has_aspect_ratio_override, csDefaults.aspect_ratio,
        () => _save({ h3_char_sheet_aspect_ratio: csAspectSel.value }));
    const csMegapixelsInp = _mk("input", {
        cls: "fbt-ce-input fbt-ce-settings-lufs",
        type: "number", min: "0.1", max: "8", step: "0.1",
        value: settings.h3_char_sheet_megapixels || "",
        placeholder: csDefaults.megapixels != null ? String(csDefaults.megapixels) : "1",
        title: "Target megapixels override. Blank = use the template's own value"
            + (csDefaults.megapixels != null ? ` (currently ${csDefaults.megapixels}).` : "."),
    });
    csMegapixelsInp.disabled = !csOptions.has_aspect_ratio_override;
    csMegapixelsInp.addEventListener("change", () => {
        const v = parseFloat(csMegapixelsInp.value);
        const clamped = isNaN(v) || v <= 0 ? 0 : Math.max(0.1, Math.min(8, v));
        csMegapixelsInp.value = clamped || "";
        _save({ h3_char_sheet_megapixels: clamped });
    });
    csMegapixelsInp.style.flex  = "0 0 auto";
    csMegapixelsInp.style.width = "64px";
    const csAspectGroup = _mk("div", { style: { display: "flex", alignItems: "center", gap: "6px", flex: "1", minWidth: "0" } },
        [csAspectSel, _mk("span", { cls: "fbt-be-proc-unit", textContent: "MP" }), csMegapixelsInp]);
    charSheetSec.appendChild(_row("Aspect Ratio", csAspectGroup, "Output aspect ratio preset, plus a megapixels override"));

    const csLoraSel = _overrideSelect(csLoraList, settings.h3_char_sheet_lora,
        csOptions.has_lora_override, csDefaults.lora, () => _save({ h3_char_sheet_lora: csLoraSel.value }));
    const csLoraStrengthPlaceholder = csDefaults.lora_strength != null ? String(csDefaults.lora_strength) : "default";
    const csLoraStrengthInp = _mk("input", {
        cls: "fbt-ce-input fbt-ce-settings-lufs",
        type: "number", min: "0", max: "2", step: "0.05",
        value: settings.h3_char_sheet_lora ? (settings.h3_char_sheet_lora_strength ?? "") : "",
        placeholder: csLoraStrengthPlaceholder,
        title: `LoRA strength (applied to both model and CLIP) — only used while a LoRA above is `
            + `selected. Blank = use the template's own value (currently ${csLoraStrengthPlaceholder}).`,
    });
    csLoraStrengthInp.disabled = !csOptions.has_lora_override;
    csLoraStrengthInp.addEventListener("change", () => {
        const v = parseFloat(csLoraStrengthInp.value);
        if (!isNaN(v)) {
            const clamped = Math.max(0, Math.min(2, v));
            csLoraStrengthInp.value = clamped;
            _save({ h3_char_sheet_lora_strength: clamped });
        }
    });
    csLoraStrengthInp.style.flex  = "0 0 auto";
    csLoraStrengthInp.style.width = "64px";
    const csLoraGroup = _mk("div", { style: { display: "flex", alignItems: "center", gap: "6px", flex: "1", minWidth: "0" } },
        [csLoraSel, _mk("span", { cls: "fbt-be-proc-unit", textContent: "Weight" }), csLoraStrengthInp]);
    charSheetSec.appendChild(_row("LoRA", csLoraGroup,
        "Optional turbo/style LoRA override — the Weight value is applied to both the LoRA's "
        + "model and CLIP strength"));

    const csFreeVramBtn = _mk("button", { cls: "fbt-ce-btn sm", textContent: "Free VRAM now",
        title: "Unload resident models immediately — same effect as Manager's own "
            + "\"Free model and node cache\" button.",
        onclick: async () => {
            csFreeVramBtn.disabled    = true;
            csFreeVramBtn.textContent = "Freeing…";
            try {
                await compositionsApi.freeH3Vram();
                _toast("VRAM freed", "success");
            } catch (e) {
                alert(`Free VRAM failed: ${e.message}`);
            } finally {
                csFreeVramBtn.disabled    = false;
                csFreeVramBtn.textContent = "Free VRAM now";
            }
        } });
    charSheetSec.appendChild(_row("", csFreeVramBtn,
        "Shares VRAM with H3 Background Plate — see \"Unload model after each run\" above"));

    if (csOptionsError) {
        charSheetSec.appendChild(_mk("p", { cls: "fbt-settings-note",
            textContent: `Couldn't load model/LoRA options (${csOptionsError}) — all overrides above `
                + "are disabled until this loads. Reopen Settings to retry." }));
    } else {
        charSheetSec.appendChild(_mk("p", { cls: "fbt-settings-note",
            textContent: "Every field above is optional and only takes effect once "
                + "templates/h3_character_sheet.api.json exposes the matching titled node (disabled "
                + "fields don't yet — see templates/README.md). Prompt/duration/fps/steps for each "
                + "mode live inside the template's own Character/Face Sheet Options nodes, not here." }));
    }
    wrap.appendChild(charSheetSec);

    // ── Qwen Photo Restore ─────────────────────────────────────────────────────
    // Optional overrides for the Tools tab's "Restore Photo" action (js/ui/tools_panel.js). Same
    // disabled-until-supported pattern as the two H3 sections above — see
    // nodes/qwen21_photo_restore.py::_tools_restore_photo_settings_options and
    // templates/README.md's "qwen21_photo_restore.api.json" section. A genuinely different model
    // (Qwen-Image-2.1, not MiniMax H3) but VRAM is still shared machine-wide state, so "Unload
    // model after each run" above also governs this feature.
    const restoreSec = _section("Qwen Photo Restore");

    let prOptions = { models: [], clips: [], vaes: [], samplers: [], schedulers: [],
        has_model_override: false, has_clip_override: false, has_vae_override: false,
        has_sampler_override: false, has_negative_prompt_override: false, template_defaults: {} };
    let prOptionsError = "";
    try {
        prOptions = await toolsApi.getRestorePhotoSettingsOptions();
    } catch (e) {
        prOptionsError = e.message || "Failed to load options";
    }

    const prDefaults = prOptions.template_defaults || {};

    const prModelSel = _overrideSelect(prOptions.models, settings.qwen21_photo_restore_model,
        prOptions.has_model_override, prDefaults.model, () => _save({ qwen21_photo_restore_model: prModelSel.value }));
    restoreSec.appendChild(_row("Model", prModelSel, "Diffusion model checkpoint override"));

    const prClipSel = _overrideSelect(prOptions.clips, settings.qwen21_photo_restore_clip,
        prOptions.has_clip_override, prDefaults.clip, () => _save({ qwen21_photo_restore_clip: prClipSel.value }));
    restoreSec.appendChild(_row("Clip", prClipSel, "Text encoder / CLIP model override"));

    const prVaeSel = _overrideSelect(prOptions.vaes, settings.qwen21_photo_restore_vae,
        prOptions.has_vae_override, prDefaults.vae, () => _save({ qwen21_photo_restore_vae: prVaeSel.value }));
    restoreSec.appendChild(_row("VAE", prVaeSel, "VAE override"));

    const prSamplerSel = _overrideSelect(prOptions.samplers, settings.qwen21_photo_restore_sampler,
        prOptions.has_sampler_override, prDefaults.sampler,
        () => _save({ qwen21_photo_restore_sampler: prSamplerSel.value }));
    const prSchedulerSel = _overrideSelect(prOptions.schedulers, settings.qwen21_photo_restore_scheduler,
        prOptions.has_sampler_override, prDefaults.scheduler,
        () => _save({ qwen21_photo_restore_scheduler: prSchedulerSel.value }));
    const prStepsInp = _mk("input", {
        cls: "fbt-ce-input fbt-ce-settings-lufs",
        type: "number", min: "0", max: "10000", step: "1",
        value: settings.qwen21_photo_restore_steps || "",
        placeholder: prDefaults.steps != null ? String(prDefaults.steps) : "default",
        title: "Sampler steps override. Blank = use the template's own value"
            + (prDefaults.steps != null ? ` (currently ${prDefaults.steps}).` : "."),
    });
    prStepsInp.disabled = !prOptions.has_sampler_override;
    prStepsInp.addEventListener("change", () => {
        const v = parseInt(prStepsInp.value, 10);
        const clamped = isNaN(v) || v <= 0 ? 0 : Math.max(1, Math.min(10000, v));
        prStepsInp.value = clamped || "";
        _save({ qwen21_photo_restore_steps: clamped });
    });
    prStepsInp.style.flex  = "0 0 auto";
    prStepsInp.style.width = "64px";
    const prSamplerGroup = _mk("div", { style: { display: "flex", alignItems: "center", gap: "6px", flex: "1", minWidth: "0" } },
        [prSamplerSel, prSchedulerSel, _mk("span", { cls: "fbt-be-proc-unit", textContent: "Steps" }), prStepsInp]);
    restoreSec.appendChild(_row("Sampler", prSamplerGroup, "Sampler/scheduler algorithm override, plus a steps override"));

    const prCfgInp = _mk("input", {
        cls: "fbt-ce-input fbt-ce-settings-lufs",
        type: "number", min: "0.1", max: "30", step: "0.1",
        value: settings.qwen21_photo_restore_cfg || "",
        placeholder: prDefaults.cfg != null ? String(prDefaults.cfg) : "1",
        title: "CFG override. Qwen-Image-2.1's official path keeps this at 1 (Negative Prompt below "
            + "is inert below cfg 1). Blank = use the template's own value"
            + (prDefaults.cfg != null ? ` (currently ${prDefaults.cfg}).` : "."),
    });
    prCfgInp.disabled = !prOptions.has_sampler_override;
    prCfgInp.addEventListener("change", () => {
        const v = parseFloat(prCfgInp.value);
        const clamped = isNaN(v) || v <= 0 ? 0 : Math.max(0.1, Math.min(30, v));
        prCfgInp.value = clamped || "";
        _save({ qwen21_photo_restore_cfg: clamped });
    });
    const prDenoiseInp = _mk("input", {
        cls: "fbt-ce-input fbt-ce-settings-lufs",
        type: "number", min: "0.01", max: "1", step: "0.01",
        value: settings.qwen21_photo_restore_denoise || "",
        placeholder: prDefaults.denoise != null ? String(prDefaults.denoise) : "1",
        title: "Denoise override. Blank = use the template's own value"
            + (prDefaults.denoise != null ? ` (currently ${prDefaults.denoise}).` : "."),
    });
    prDenoiseInp.disabled = !prOptions.has_sampler_override;
    prDenoiseInp.addEventListener("change", () => {
        const v = parseFloat(prDenoiseInp.value);
        const clamped = isNaN(v) || v <= 0 ? 0 : Math.max(0.01, Math.min(1, v));
        prDenoiseInp.value = clamped || "";
        _save({ qwen21_photo_restore_denoise: clamped });
    });
    const prCfgDenoiseGroup = _mk("div", { style: { display: "flex", alignItems: "center", gap: "6px", flex: "1", minWidth: "0" } },
        [_mk("span", { cls: "fbt-be-proc-unit", textContent: "CFG" }), prCfgInp,
         _mk("span", { cls: "fbt-be-proc-unit", textContent: "Denoise" }), prDenoiseInp]);
    restoreSec.appendChild(_row("CFG / Denoise", prCfgDenoiseGroup, "Same IN:seed node as Sampler above"));

    const prNegPromptInp = _mk("textarea", {
        cls: "fbt-ce-textarea", rows: 2,
        value: settings.qwen21_photo_restore_negative_prompt || "",
        placeholder: prDefaults.negative_prompt || "default (blank)",
        title: "Negative prompt override — inert unless CFG above is raised past 1.",
    });
    prNegPromptInp.disabled = !prOptions.has_negative_prompt_override;
    prNegPromptInp.addEventListener("change", () =>
        _save({ qwen21_photo_restore_negative_prompt: prNegPromptInp.value.trim() }));
    restoreSec.appendChild(_row("Negative Prompt", prNegPromptInp,
        "Only takes effect once CFG above is raised past 1"));

    const prFreeVramBtn = _mk("button", { cls: "fbt-ce-btn sm", textContent: "Free VRAM now",
        title: "Unload resident models immediately — same effect as Manager's own "
            + "\"Free model and node cache\" button.",
        onclick: async () => {
            prFreeVramBtn.disabled    = true;
            prFreeVramBtn.textContent = "Freeing…";
            try {
                await compositionsApi.freeH3Vram();
                _toast("VRAM freed", "success");
            } catch (e) {
                alert(`Free VRAM failed: ${e.message}`);
            } finally {
                prFreeVramBtn.disabled    = false;
                prFreeVramBtn.textContent = "Free VRAM now";
            }
        } });
    restoreSec.appendChild(_row("", prFreeVramBtn,
        "Shares VRAM with H3 Background Plate — see \"Unload model after each run\" above"));

    if (prOptionsError) {
        restoreSec.appendChild(_mk("p", { cls: "fbt-settings-note",
            textContent: `Couldn't load model options (${prOptionsError}) — all overrides above `
                + "are disabled until this loads. Reopen Settings to retry." }));
    } else {
        restoreSec.appendChild(_mk("p", { cls: "fbt-settings-note",
            textContent: "Every field above is optional and only takes effect once "
                + "templates/qwen21_photo_restore.api.json exposes the matching titled node (disabled "
                + "fields don't yet — see templates/README.md)." }));
    }
    wrap.appendChild(restoreSec);
}
