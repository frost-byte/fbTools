/**
 * DatasetCaptioner live status widget.
 * Shows current status text, processed/remaining counts, and active filename.
 */

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const DATASET_STATUS_SOURCE = "dataset_captioner";
const DATASET_NODE_ID = "fbt_DatasetCaptioner";

function createStatusContainer() {
    const container = document.createElement("div");
    container.style.cssText = [
        "width:100%",
        "padding:8px 10px",
        "box-sizing:border-box",
        "border:1px solid #3c3c3c",
        "border-radius:6px",
        "background:#171717",
        "color:#cfcfcf",
        "font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
        "font-size:12px",
        "line-height:1.35",
        "display:grid",
        "gap:3px",
        "min-height:80px",
    ].join(";");

    const llmLine = document.createElement("div");
    llmLine.style.cssText = "border-bottom:1px solid #2a2a2a;padding-bottom:4px;margin-bottom:1px;color:#888;";
    llmLine.textContent = "Model: checking…";

    const statusLine = document.createElement("div");
    const progressLine = document.createElement("div");
    const metricsLine = document.createElement("div");
    const fileLine = document.createElement("div");

    statusLine.textContent = "Status: Ready";
    progressLine.textContent = "Progress: 0/0 (Remaining: 0)";
    metricsLine.textContent = "Speed: 0.00 it/s | Elapsed: 00:00 | ETA: --:--";
    fileLine.textContent = "Active File: -";

    container.appendChild(llmLine);
    container.appendChild(statusLine);
    container.appendChild(progressLine);
    container.appendChild(metricsLine);
    container.appendChild(fileLine);

    return { container, llmLine, statusLine, progressLine, metricsLine, fileLine };
}

function formatDuration(totalSeconds) {
    const sec = Math.max(0, Math.floor(Number(totalSeconds) || 0));
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    const s = sec % 60;
    if (h > 0) {
        return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
    }
    return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function parseStatus(statusText, currentState) {
    const nowMs = Date.now();
    const next = {
        status: statusText || "Ready",
        processed: currentState?.processed ?? 0,
        total: currentState?.total ?? 0,
        activeFile: currentState?.activeFile ?? "-",
        startedAtMs: currentState?.startedAtMs ?? null,
    };

    if (/loading\s+.+model/i.test(statusText || "")) {
        next.processed = 0;
        next.total = 0;
        next.activeFile = "-";
        next.startedAtMs = null;
    }

    const processingMatch = /processing\s+(\d+)\/(\d+)(?:\s*\((.+)\))?/i.exec(statusText || "");
    if (processingMatch) {
        const newProcessed = Number(processingMatch[1]);
        if (next.startedAtMs == null || newProcessed < (currentState?.processed ?? 0)) {
            next.startedAtMs = nowMs;
        }
        next.processed = newProcessed;
        next.total = Number(processingMatch[2]);
        next.activeFile = processingMatch[3]?.trim() || next.activeFile;
    }

    const modelReadyMatch = /model ready,\s*captioning\s*(\d+)\s*image/i.exec(statusText || "");
    if (modelReadyMatch) {
        next.total = Number(modelReadyMatch[1]);
        next.processed = 0;
        next.activeFile = "-";
        next.startedAtMs = nowMs;
    }

    const errorFileMatch = /error on\s+(.+)$/i.exec(statusText || "");
    if (errorFileMatch) {
        next.activeFile = errorFileMatch[1].trim();
    }

    const completedMatch = /completed\s*\((\d+)\s*ok,\s*(\d+)\s*failed\)/i.exec(statusText || "");
    if (completedMatch) {
        const ok = Number(completedMatch[1]);
        const failed = Number(completedMatch[2]);
        next.processed = ok + failed;
        if (next.total < next.processed) {
            next.total = next.processed;
        }
        next.activeFile = "-";
    }

    if (/no images to process/i.test(statusText || "")) {
        next.processed = 0;
        next.total = 0;
        next.activeFile = "-";
        next.startedAtMs = null;
    }

    return next;
}

function applyStateToNode(node, state, level) {
    if (!node?._dcsUi) return;

    const { container, statusLine, progressLine, metricsLine, fileLine } = node._dcsUi;
    const remaining = Math.max(0, (state.total || 0) - (state.processed || 0));
    const nowMs = Date.now();
    const elapsedSeconds = state.startedAtMs ? Math.max(0, (nowMs - state.startedAtMs) / 1000) : 0;
    const rate = elapsedSeconds > 0 ? (state.processed || 0) / elapsedSeconds : 0;
    const etaSeconds = rate > 0 && remaining > 0 ? remaining / rate : null;

    statusLine.textContent = `Status: ${state.status}`;
    progressLine.textContent = `Progress: ${state.processed}/${state.total} (Remaining: ${remaining})`;
    metricsLine.textContent = `Speed: ${rate.toFixed(2)} it/s | Elapsed: ${formatDuration(elapsedSeconds)} | ETA: ${etaSeconds == null ? "--:--" : formatDuration(etaSeconds)}`;
    fileLine.textContent = `Active File: ${state.activeFile || "-"}`;

    const lvl = String(level || "").toLowerCase();
    if (lvl === "error" || /error|failed/i.test(state.status)) {
        container.style.borderColor = "var(--fbt-error-border)";
        container.style.color = "var(--fbt-error-text)";
    } else if (lvl === "success" || /completed|complete|ready/i.test(state.status)) {
        container.style.borderColor = "var(--fbt-ok-border)";
        container.style.color = "var(--fbt-ok-text)";
    } else if (lvl === "warn" || /warn/i.test(state.status)) {
        container.style.borderColor = "var(--fbt-warn-border)";
        container.style.color = "var(--fbt-warn-text)";
    } else {
        container.style.borderColor = "var(--fbt-idle-border)";
        container.style.color = "var(--fbt-idle-text)";
    }
}

function getDatasetCaptionerNodes() {
    return (app?.graph?._nodes || []).filter((n) => n?._dcsUi);
}

// ── LLM status badge ──────────────────────────────────────────────────────────

let _lastLlmStatus = { loaded: null, vision: false };

function _getCaptionerType(node) {
    return node.widgets?.find((w) => w.name === "captioner_type")?.value || "llm_client";
}

function updateLlmBadge(node) {
    if (!node?._dcsUi?.llmLine) return;
    const { llmLine } = node._dcsUi;
    const captType = _getCaptionerType(node);

    if (captType === "gemini_flash") {
        llmLine.textContent = "Model: Gemini Flash (API)";
        llmLine.style.color = "var(--fbt-blue)";
        return;
    }

    const { loaded, vision } = _lastLlmStatus;
    if (loaded && vision) {
        const short = loaded.length > 36 ? loaded.slice(0, 34) + "…" : loaded;
        llmLine.textContent = `Model: ${short} ✓`;
        llmLine.style.color = "var(--fbt-ok)";
    } else if (loaded && !vision) {
        const short = loaded.length > 28 ? loaded.slice(0, 26) + "…" : loaded;
        llmLine.textContent = `Model: ${short} ⚠ no vision`;
        llmLine.style.color = "var(--fbt-warn)";
    } else {
        llmLine.textContent = "Model: ⚠ none — load in fbTools Compose";
        llmLine.style.color = "var(--fbt-error)";
    }
}

// Listen for status broadcasts from fbt_panel
document.addEventListener("fbt:llm-status", ({ detail }) => {
    _lastLlmStatus = { loaded: detail.loaded ?? null, vision: detail.vision ?? false };
    for (const node of getDatasetCaptionerNodes()) {
        updateLlmBadge(node);
    }
    app?.graph?.setDirtyCanvas(true, false);
});

export function setupDatasetCaptionerStatus(nodeType) {
    const onNodeCreated = nodeType.prototype.onNodeCreated;

    nodeType.prototype.onNodeCreated = function onNodeCreatedWithDatasetStatus() {
        const result = onNodeCreated?.apply(this, arguments);

        if (this._dcsUi) {
            return result;
        }

        const ui = createStatusContainer();
        const statusWidget = this.addDOMWidget("dataset_caption_status", "status", ui.container, {
            serialize: false,
            hideOnZoom: false,
            getValue() {
                return ui.statusLine.textContent;
            },
            setValue() {
                // Live widget; values are sourced from websocket events.
            },
        });

        this._dcsUi = ui;
        this._dcsState = {
            status: "Ready",
            processed: 0,
            total: 0,
            activeFile: "-",
            startedAtMs: null,
        };
        this._dcsStatusWidget = statusWidget;
        statusWidget.parentNode = this;
        statusWidget.computeSize = function computeSize(width) {
            return [width, 126];
        };

        // Set initial badge from last known state (or trigger a fetch if panel isn't open yet)
        const initialStatus = window._fbtGetLlmStatus?.();
        if (initialStatus) {
            _lastLlmStatus = { loaded: initialStatus.loaded, vision: initialStatus.vision };
        }
        updateLlmBadge(this);

        // Update badge immediately when the user switches captioner_type
        const captWidget = this.widgets?.find((w) => w.name === "captioner_type");
        if (captWidget) {
            const origCallback = captWidget.callback;
            const self = this;
            captWidget.callback = function (value) {
                origCallback?.call(captWidget, value);
                updateLlmBadge(self);
            };
        }

        return result;
    };
}

api.addEventListener("fbtools.status", (event) => {
    try {
        const detail = event?.detail || {};
        if (detail.source !== DATASET_STATUS_SOURCE) return;
        if (detail.node && detail.node !== DATASET_NODE_ID) return;

        const statusText = String(detail.status || "").trim();
        if (!statusText) return;

        const nodes = getDatasetCaptionerNodes();
        for (const node of nodes) {
            const nextState = parseStatus(statusText, node._dcsState);
            node._dcsState = nextState;
            applyStateToNode(node, nextState, detail.level);
        }

        app?.graph?.setDirtyCanvas(true, false);
    } catch (err) {
        console.error("fb_tools -> DatasetCaptioner status widget error", err);
    }
});
