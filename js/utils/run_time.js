/**
 * Run Time — start/end/duration helpers for the Run History header.
 * ComfyUI history messages carry millisecond timestamps.
 */

const END_MESSAGES = new Set(["execution_success", "execution_error", "execution_interrupted"]);

/** Timestamp (ms) of the message that ended a run, or null while it is still running. */
export function findRunEndTs(messages) {
    for (const m of messages || []) {
        if (Array.isArray(m) && END_MESSAGES.has(m[0]) && m[1]?.timestamp) return m[1].timestamp;
    }
    return null;
}

/** 4s -> "4s", 606s -> "10m 6s", 3725s -> "1h 2m 5s". */
export function formatDuration(ms) {
    if (!Number.isFinite(ms) || ms < 0) return "";
    const total = Math.round(ms / 1000);
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const s = total % 60;
    if (h) return `${h}h ${m}m ${s}s`;
    if (m) return `${m}m ${s}s`;
    return `${s}s`;
}

/**
 * "9/20/2026, 11:42:04 PM → 11:52:10 PM (10m 6s)". The end drops the date when it is the
 * same calendar day. With no end time only the start is returned.
 */
export function formatRunTime(startMs, endMs, locale) {
    if (!startMs) return "";
    const start = new Date(startMs);
    const startStr = start.toLocaleString(locale);
    if (!endMs || endMs < startMs) return startStr;
    const end = new Date(endMs);
    const sameDay = start.toDateString() === end.toDateString();
    const endStr = sameDay ? end.toLocaleTimeString(locale) : end.toLocaleString(locale);
    return `${startStr} → ${endStr} (${formatDuration(endMs - startMs)})`;
}
