/**
 * Shared LLM run-history utilities.
 *
 * Provides makeEntry() to build a unified history record and
 * buildHistorySection() to render a collapsible history accordion
 * that works in both modals (bg, outfit, video, bundle) and the
 * sidebar (shot_action, dialogue, polish).
 */

import { llmApi } from "../api/llm.js";

// Human-readable labels per kind
const KIND_LABELS = {
    video_describe:     "Video → Action",
    bg_analyze:         "Background",
    outfit_analyze:     "Outfit",
    appearance_analyze: "Appearance",
    shot_action:        "Shot Action",
    dialogue:           "Dialogue",
    polish:             "Polish",
};

// ── Helpers ────────────────────────────────────────────────────────────────────

function _mk(tag, props = {}, children = []) {
    const el = document.createElement(tag);
    const { cls, textContent, style = {}, ...attrs } = props;
    if (cls) el.className = cls;
    if (textContent !== undefined) el.textContent = textContent;
    Object.assign(el.style, style);
    for (const [k, v] of Object.entries(attrs)) {
        if (k === "onclick") el.addEventListener("click", v);
        else el.setAttribute(k, v);
    }
    children.forEach(c => c && el.appendChild(c));
    return el;
}

function _timeAgo(iso) {
    const m = Math.floor((Date.now() - new Date(iso).getTime()) / 60000);
    if (m < 1)  return "just now";
    if (m < 60) return `${m}m ago`;
    const h = Math.floor(m / 60);
    if (h < 24) return `${h}h ago`;
    return `${Math.floor(h / 24)}d ago`;
}

function _resultSnippet(entry) {
    const r = entry.result || {};
    const text = r.text || r.description || "";
    return text.length > 90 ? text.slice(0, 87) + "…" : text;
}

function _resultFull(entry) {
    const r = entry.result || {};
    const parts = [];
    if (r.text)        parts.push(r.text);
    if (r.description) parts.push(r.description);
    if (r.lighting)    parts.push(`Lighting: ${r.lighting}`);
    if (r.soundscape)  parts.push(`Soundscape: ${r.soundscape}`);
    return parts.join("\n\n");
}

function _paramsMeta(entry) {
    const p = entry.params || {};
    const parts = [];
    if (p.videoPath)   parts.push(p.videoPath.split("/").pop());
    if (p.filename)    parts.push(p.filename.split("/").pop());
    if (p.sourceImage) parts.push(p.sourceImage.split("/").pop());
    if (p.intent)      parts.push(p.intent);
    if (p.speaker)     parts.push(`speaker: ${p.speaker}`);
    if (p.language && p.language !== "English") parts.push(p.language);
    if (Array.isArray(p.frameIndices) && p.frameIndices.length)
        parts.push(`${p.frameIndices.length} frames`);
    if (entry.result?.frameFile)
        parts.push(`→ ${entry.result.frameFile}`);
    return parts.join(" · ");
}

// ── Public API ─────────────────────────────────────────────────────────────────

/**
 * Build a unified history entry object.
 *
 * @param {object} opts
 * @param {string}  opts.kind            - Discriminator (e.g. "shot_action")
 * @param {string}  [opts.compositionName]
 * @param {number}  [opts.shotNumber]    - 1-based shot index, if applicable
 * @param {string}  [opts.modelId]       - Display name of the LLM used
 * @param {object}  [opts.params]        - Kind-specific input parameters
 * @param {object}  [opts.result]        - Kind-specific outputs
 */
export function makeEntry({ kind, compositionName = "", shotNumber = null, modelId = "",
                            params = {}, result = {} }) {
    const entry = {
        id: Date.now(),
        ts: new Date().toISOString(),
        kind,
        params,
        result,
    };
    if (compositionName) entry.compositionName = compositionName;
    if (shotNumber != null) entry.shotNumber = shotNumber;
    if (modelId) entry.modelId = modelId;
    return entry;
}

/**
 * Build a collapsible history section element.
 *
 * @param {object}          opts
 * @param {string|string[]} opts.kind      - Kind tag(s) to show; multiple = comma filter
 * @param {Function}        opts.onRestore - Called with the full entry dict on Restore click
 * @param {boolean}         [opts.multiKind] - Force showing the kind label even for single kind
 * @returns {{ el: HTMLElement, refresh: () => Promise<void> }}
 */
export function buildHistorySection({ kind, onRestore, multiKind = false }) {
    const kinds     = Array.isArray(kind) ? kind : [kind];
    const kindParam = kinds.join(",");
    const showKind  = multiKind || kinds.length > 1;

    const section = _mk("div", { cls: "fbt-hist-section" });
    const toggle  = _mk("button", { cls: "fbt-hist-toggle", textContent: "▶ History" });
    const body    = _mk("div",    { cls: "fbt-hist-body" });
    body.style.display = "none";

    toggle.addEventListener("click", () => {
        const open = body.style.display !== "none";
        body.style.display = open ? "none" : "";
        toggle.textContent = (open ? "▶" : "▼") + " History";
    });

    section.append(toggle, body);

    async function refresh() {
        body.innerHTML = "";
        let entries = [];
        try { entries = await llmApi.historyList({ kind: kindParam }); }
        catch { /* server unavailable */ }

        if (!entries.length) {
            body.appendChild(_mk("div", { cls: "fbt-hist-empty", textContent: "No runs yet." }));
            return;
        }

        entries.forEach(entry => {
            const item    = _mk("div", { cls: "fbt-hist-item" });
            const summary = _mk("div", { cls: "fbt-hist-summary" });
            const detail  = _mk("div", { cls: "fbt-hist-detail" });
            detail.style.display = "none";

            // ── Summary row ──────────────────────────────────────────────────
            summary.appendChild(_mk("span", { cls: "fbt-hist-age",
                textContent: _timeAgo(entry.ts) }));
            if (showKind)
                summary.appendChild(_mk("span", { cls: "fbt-hist-tag fbt-hist-kind",
                    textContent: KIND_LABELS[entry.kind] || entry.kind }));
            if (entry.compositionName)
                summary.appendChild(_mk("span", { cls: "fbt-hist-tag",
                    textContent: entry.compositionName }));
            if (entry.shotNumber != null)
                summary.appendChild(_mk("span", { cls: "fbt-hist-tag",
                    textContent: `Shot ${entry.shotNumber}` }));
            if (entry.modelId)
                summary.appendChild(_mk("span", { cls: "fbt-hist-tag fbt-hist-model",
                    textContent: entry.modelId }));
            const snippet = _resultSnippet(entry);
            if (snippet)
                summary.appendChild(_mk("span", { cls: "fbt-hist-snippet",
                    textContent: snippet }));

            summary.addEventListener("click", () => {
                const open = detail.style.display !== "none";
                detail.style.display = open ? "none" : "";
            });

            // ── Detail body ──────────────────────────────────────────────────
            const fullText = _resultFull(entry);
            if (fullText)
                detail.appendChild(_mk("div", { cls: "fbt-hist-result",
                    textContent: fullText }));

            const meta = _paramsMeta(entry);
            if (meta)
                detail.appendChild(_mk("div", { cls: "fbt-hist-meta", textContent: meta }));

            const btnRow = _mk("div", { cls: "fbt-hist-btnrow" });
            if (onRestore) {
                btnRow.appendChild(_mk("button", {
                    cls: "fbt-ce-btn fbt-ce-btn-primary", textContent: "Restore",
                    onclick: () => onRestore(entry),
                }));
            }
            btnRow.appendChild(_mk("button", {
                cls: "fbt-ce-btn fbt-ce-btn-secondary", textContent: "Delete",
                onclick: async () => {
                    await llmApi.historyDelete(entry.id).catch(() => {});
                    refresh();
                },
            }));
            detail.appendChild(btnRow);

            item.append(summary, detail);
            body.appendChild(item);
        });
    }

    // Initial fetch: auto-expand when there are entries
    refresh().then(() => {
        if (!body.querySelector(".fbt-hist-empty")) {
            body.style.display = "";
            toggle.textContent = "▼ History";
        }
    });

    return { el: section, refresh };
}
