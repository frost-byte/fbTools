/**
 * Pure helpers for showing a Prompt Composition's shots on SceneCastBuild's
 * clip timeline and in its action preview. No DOM, no ComfyUI imports.
 */

/** "MM:SS.mmm" (or "HH:MM:SS.mmm" / "SS.mmm") -> seconds, or null if absent/invalid. */
export function parseShotTimestamp(ts) {
    if (ts == null) return null;
    const s = String(ts).trim();
    if (!s) return null;
    const parts = s.split(":");
    if (parts.length > 3) return null;
    let total = 0;
    for (const p of parts) {
        if (!/^\d+(\.\d+)?$/.test(p)) return null;
        total = total * 60 + parseFloat(p);
    }
    return total;
}

function _median(values) {
    if (!values.length) return 0;
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function _firstWords(text, n = 5) {
    const words = String(text || "").replace(/\{[^}]*\}/g, "").trim().split(/\s+/).filter(Boolean);
    const out = words.slice(0, n).join(" ");
    return words.length > n ? `${out}…` : out;
}

/**
 * Text shown for a shot: camera, action and dialogue, still containing the
 * {A}/{B} placeholders (substituteSlotPlaceholders resolves them).
 */
export function shotDisplayText(shot) {
    const lines = [];
    if (shot.camera && String(shot.camera).trim()) lines.push(String(shot.camera).trim());
    if (shot.action && String(shot.action).trim()) lines.push(String(shot.action).trim());
    const dlg = shot.dialogue;
    if (dlg && dlg.text && String(dlg.text).trim()) {
        lines.push(`${dlg.speaker ? `{${dlg.speaker}}` : "Dialogue"}: "${String(dlg.text).trim()}"`);
    }
    return lines.join("\n");
}

/**
 * Composition shots -> SceneCastBuild timeline segments
 * [{id, label, action, allows_dialogue, start_time?, end_time?}].
 *
 * Shots only carry a start timestamp. When every shot has one and they strictly
 * increase, each segment ends where the next begins (the last gets the median
 * gap, or 1s). Otherwise start/end are left unset so the timeline falls back to
 * equal-width bands; a mix of null and real timestamps counts as "otherwise".
 */
export function shotsToSegments(shots) {
    const list = Array.isArray(shots) ? shots : [];
    const starts = list.map(s => parseShotTimestamp(s?.timestamp));
    const usable = list.length > 0 && starts.every(v => v != null)
        && starts.every((v, i) => i === 0 || v > starts[i - 1]);

    let lastGap = 1;
    if (usable && list.length > 1) {
        const gaps = starts.slice(1).map((v, i) => v - starts[i]);
        lastGap = _median(gaps) || 1;
    }

    return list.map((shot, i) => {
        const seg = {
            id: shot?.id ?? `shot_${i + 1}`,
            label: `${i + 1} · ${_firstWords(shot?.action) || "shot"}`,
            action: shotDisplayText(shot || {}),
            allows_dialogue: true,
        };
        if (usable) {
            seg.start_time = starts[i];
            seg.end_time = i + 1 < starts.length ? starts[i + 1] : starts[i] + lastGap;
        }
        return seg;
    });
}

/**
 * Replace {A}, {B}, {AA}, ... with `[label]` when slotLabel (an object or Map
 * keyed by slot letters) knows the slot; unknown placeholders are left as-is.
 */
export function substituteSlotPlaceholders(text, slotLabel) {
    const get = slotLabel instanceof Map ? (k => slotLabel.get(k)) : (k => slotLabel?.[k]);
    return String(text ?? "").replace(/\{([A-Z]+)\}/g, (m, letter) => {
        const label = get(letter);
        return label ? `[${label}]` : m;
    });
}
