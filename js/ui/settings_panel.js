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
}
