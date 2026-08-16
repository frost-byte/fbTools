/**
 * Run Tracker — tag parsing, visual indicator, and context menu integration.
 *
 * Tags are embedded in node.title as: [track: Label]
 * The rest of the title is preserved. A node with a track tag gets a green dot
 * drawn in its top-right corner via onDrawForeground (works on any node).
 */

const TRACK_RE = /\s*\[track:\s*([^\]]*)\]/;

const BAR_H = 3;
const BAR_R = 1.5;

export function getTrackTag(node) {
    const m = (node.title || "").match(TRACK_RE);
    return m ? m[1].trim() : null;
}

export function setTrackTag(node, label) {
    const base = (node.title || "").replace(TRACK_RE, "").trim();
    node.title = label ? `${base} [track: ${label}]` : base;
}

export function patchNodeForTracking(node, app) {
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
                const base = (node.title || "").replace(TRACK_RE, "").trim();
                const label = window.prompt("Track label (shown in Run History):", base || "");
                if (label === null) return;
                setTrackTag(node, label.trim() || base || "node");
                app.graph.setDirtyCanvas(true, false);
            },
        });
    }
}
