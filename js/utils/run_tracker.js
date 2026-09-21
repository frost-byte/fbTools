/**
 * Run Tracker — tag parsing, visual indicator, and context menu integration.
 *
 * A tracked node's title is "<marker> <label>" (e.g. "🐾 Video Shift"); the label is
 * the title minus the leading marker. The legacy "[track: Label]" form is still
 * understood and is rewritten to the marker form when a workflow loads.
 * A tracked node gets a green bar drawn at the top via onDrawForeground.
 *
 * Keep TRACK_MARKER in sync with utils/node_output_tracker.py and ui/run_history.js.
 */

export const TRACK_MARKER = "\u{1F43E}"; // 🐾
const LEGACY_TRACK_RE = /\s*\[track:\s*([^\]]*)\]/;

const BAR_H = 3;
const BAR_R = 1.5;

/** Label for a title ("🐾 Foo" or legacy "X [track: Foo]"), or null when not tracked. */
export function parseTrackTitle(title) {
    const s = (title || "").trim();
    if (s.startsWith(TRACK_MARKER)) return s.slice(TRACK_MARKER.length).trim() || null;
    const m = s.match(LEGACY_TRACK_RE);
    return m ? m[1].trim() : null;
}

export function formatTrackTitle(label) {
    return `${TRACK_MARKER} ${label}`;
}

/**
 * New title for a node still carrying the legacy tag, or null if the title needs no change.
 * An empty legacy label falls back to the rest of the title.
 */
export function migrateLegacyTrackTitle(title) {
    const s = title || "";
    if (s.trim().startsWith(TRACK_MARKER)) return null;
    const m = s.match(LEGACY_TRACK_RE);
    if (!m) return null;
    const label = m[1].trim() || s.replace(LEGACY_TRACK_RE, "").trim() || "node";
    return formatTrackTitle(label);
}

export function getTrackTag(node) {
    return parseTrackTitle(node.title);
}

/** Set the tracked label; a falsy label untracks (title becomes the plain label). */
export function setTrackTag(node, label) {
    if (label) {
        node.title = formatTrackTitle(label);
    } else {
        node.title = parseTrackTitle(node.title) ?? (node.title || "");
    }
}

function _migrateNodeTitle(node) {
    const next = migrateLegacyTrackTitle(node.title);
    if (next) node.title = next;
}

export function patchNodeForTracking(node, app) {
    // Titles are restored from the saved workflow during configure(), after nodeCreated.
    _migrateNodeTitle(node);
    const origConfigure = node.onConfigure?.bind(node);
    node.onConfigure = function(info) {
        const r = origConfigure?.(info);
        _migrateNodeTitle(this);
        return r;
    };

    const origDraw = node.onDrawForeground?.bind(node);
    node.onDrawForeground = function(ctx) {
        origDraw?.(ctx);
        if (!getTrackTag(this)) return;
        const scale = app.canvas?.ds?.scale ?? 1;
        if (scale < 0.2) return;
        // Centered horizontally at the top of the node body, clear of pips on either edge
        const barW = Math.max(20, this.size[0] * 0.4);
        const barX = (this.size[0] - barW) / 2;
        const barY = 1;
        ctx.save();
        ctx.fillStyle = "#4ade80";
        ctx.beginPath();
        ctx.roundRect(barX, barY, barW, BAR_H, BAR_R);
        ctx.fill();
        ctx.restore();
    };

    const origMenu = node.getExtraMenuOptions?.bind(node);
    node.getExtraMenuOptions = function(canvas, options) {
        const r = origMenu ? origMenu(canvas, options) : options;
        _addTrackMenuItems(this, options, app);
        return r;
    };
}

function _addTrackMenuItems(node, options, app) {
    const tag = getTrackTag(node);
    options.push(null);
    if (tag !== null) {
        options.push({ content: `● Tracked: "${tag}"`, disabled: true });
        options.push({
            content: "Edit track label…",
            callback: () => {
                const next = window.prompt("Track label:", tag);
                if (next === null) return;
                setTrackTag(node, next.trim() || tag);
                app.graph.setDirtyCanvas(true, false);
            },
        });
        options.push({
            content: "Remove track tag",
            callback: () => {
                setTrackTag(node, null);
                app.graph.setDirtyCanvas(true, false);
            },
        });
    } else {
        options.push({
            content: "Track this node…",
            callback: () => {
                const base = (node.title || "").trim();
                const label = window.prompt("Track label (shown in Run History):", base || "");
                if (label === null) return;
                setTrackTag(node, label.trim() || base || "node");
                app.graph.setDirtyCanvas(true, false);
            },
        });
    }
}
