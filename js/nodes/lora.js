/**
 * LoRA node frontend extensions.
 *
 * Adds a "Civitai Info" button to LoraEntryDefine nodes that fetches and
 * displays model metadata (trained words, base model, description) from civitai.
 */

import { loraAPI } from "../api/lora.js";
import { sceneAPI } from "../api/scene.js";
import { setWidgetVisible } from "../utils/widgets.js";

// ── Modal ─────────────────────────────────────────────────────────────────────

function stripHtml(html) {
    const tmp = document.createElement("div");
    tmp.innerHTML = html;
    return tmp.textContent || tmp.innerText || "";
}

function showCivitaiModal(data) {
    document.getElementById("fbt-civitai-modal")?.remove();

    const overlay = document.createElement("div");
    overlay.id = "fbt-civitai-modal";
    overlay.style.cssText = [
        "position:fixed", "inset:0", "z-index:9999",
        "background:rgba(0,0,0,0.72)",
        "display:flex", "align-items:center", "justify-content:center",
    ].join(";");

    const panel = document.createElement("div");
    panel.style.cssText = [
        "background:var(--comfy-menu-bg)",
        "border:1px solid var(--border-color)",
        "border-radius:8px",
        "padding:20px",
        "max-width:520px",
        "width:90%",
        "max-height:80vh",
        "overflow-y:auto",
        "color:var(--input-text)",
        "font-size:13px",
        "line-height:1.5",
    ].join(";");

    const modelName  = data.model?.name  || "Unknown model";
    const versionName = data.name        || "";
    const baseModel  = data.baseModel    || "Unknown";
    const civitaiUrl = `https://civitai.com/models/${data.modelId}`;

    const trainedWords = Array.isArray(data.trainedWords) && data.trainedWords.length
        ? data.trainedWords.join(", ")
        : "None listed";

    const rawDesc   = data.description ? stripHtml(data.description) : "";
    const shortDesc = rawDesc.length > 400 ? rawDesc.substring(0, 400) + "…" : rawDesc || "No description";

    // Build image thumbnail if civitai provides one
    const thumbUrl  = data.images?.[0]?.url || "";
    const thumbHtml = thumbUrl
        ? `<img src="${thumbUrl}" alt="preview"
               style="width:100%;max-height:200px;object-fit:cover;border-radius:4px;margin-bottom:12px;">`
        : "";

    panel.innerHTML = `
        <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px;">
            <div>
                <div style="font-weight:700;font-size:1.05em;">${modelName}</div>
                ${versionName ? `<div style="opacity:0.6;font-size:0.9em;">${versionName}</div>` : ""}
            </div>
            <button id="fbt-civitai-close"
                style="background:none;border:none;color:var(--input-text);cursor:pointer;font-size:1.3em;padding:0 0 0 12px;flex-shrink:0;">✕</button>
        </div>
        ${thumbHtml}
        <table style="width:100%;border-collapse:collapse;">
            <tr>
                <td style="padding:4px 10px 4px 0;opacity:0.6;white-space:nowrap;vertical-align:top;">Base model</td>
                <td style="padding:4px 0;">${baseModel}</td>
            </tr>
            <tr>
                <td style="padding:4px 10px 4px 0;opacity:0.6;white-space:nowrap;vertical-align:top;">Trigger words</td>
                <td style="padding:4px 0;word-break:break-word;">${trainedWords}</td>
            </tr>
            <tr>
                <td style="padding:4px 10px 4px 0;opacity:0.6;white-space:nowrap;vertical-align:top;">Description</td>
                <td style="padding:4px 0;">${shortDesc}</td>
            </tr>
        </table>
        <div style="margin-top:14px;">
            <a href="${civitaiUrl}" target="_blank"
               style="color:var(--p-blue-400,#60a5fa);text-decoration:none;">
                View on Civitai ↗
            </a>
        </div>
    `;

    overlay.appendChild(panel);
    document.body.appendChild(overlay);

    overlay.addEventListener("click", (e) => { if (e.target === overlay) overlay.remove(); });
    panel.querySelector("#fbt-civitai-close").addEventListener("click", () => overlay.remove());
}

// ── Node handler: LoraPresetSelect / WanPresetSelect ──────────────────────────

function setupDynamicPresetCombo(nodeType) {
    // Shared logic: update "selected_preset" combo options from ui.preset_names
    // sent back by the execute() method of any *PresetSelect node.
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
        if (onExecuted) onExecuted.apply(this, arguments);

        const names = message?.ui?.preset_names || message?.preset_names;
        if (!Array.isArray(names) || !names.length) return;

        const widget = this.widgets?.find(w => w.name === "selected_preset");
        if (!widget) return;

        widget.options.values = names;

        // Keep current selection if still valid; otherwise default to first
        if (!names.includes(widget.value)) {
            widget.value = names[0];
        }
    };
}

export function setupLoraPresetSelect(nodeType, nodeData, app) {
    setupDynamicPresetCombo(nodeType);
}

export function setupWanPresetSelect(nodeType, nodeData, app) {
    setupDynamicPresetCombo(nodeType);
}

// ── Node handler: LoraPresetDefine / WanPresetDefine ──────────────────────────

async function populateSceneCombo(node) {
    try {
        const data = await sceneAPI.list();
        const scenes = data?.scenes ?? [];
        const widget = node.widgets?.find(w => w.name === "scene_name");
        if (!widget) return;
        const options = ["none", ...scenes];
        widget.options.values = options;
        if (!options.includes(widget.value)) widget.value = "none";
    } catch (e) {
        console.warn("[fbTools] PresetDefine: could not load scene list", e);
    }
}

function setupPresetDefineSceneCombo(nodeType) {
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        if (onNodeCreated) onNodeCreated.apply(this, arguments);
        populateSceneCombo(this);
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (data) {
        if (onConfigure) onConfigure.apply(this, arguments);
        // Re-populate after load so the saved scene name is a valid option
        populateSceneCombo(this);
    };
}

export function setupLoraPresetDefine(nodeType, nodeData, app) {
    setupPresetDefineSceneCombo(nodeType);
}

export function setupWanPresetDefine(nodeType, nodeData, app) {
    setupPresetDefineSceneCombo(nodeType);
}

// ── Node handler: LoraEntryDefine ─────────────────────────────────────────────

const LTX_WIDGET_NAMES = ["video_strength", "audio_strength"];
const LORA_BUILDER_ROWS = 8;

function applyLoraEntryTarget(node, target) {
    const isLtx = target === "LTX2.3";
    setWidgetVisible(node._ltxToggleBtn, isLtx);
    const ltxWidgets = LTX_WIDGET_NAMES.map(n => node.widgets?.find(w => w.name === n)).filter(Boolean);
    for (const w of ltxWidgets) setWidgetVisible(w, isLtx && !!node._ltxAccordionOpen);
    node.setSize(node.computeSize());
    node.setDirtyCanvas(true, true);
}

export function setupLoraEntryDefine(nodeType, nodeData, app) {
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        if (onNodeCreated) onNodeCreated.apply(this, arguments);

        const self = this;
        self._ltxAccordionOpen = false;

        // ── LTX2.3 accordion toggle ──
        const toggleBtn = this.addWidget("button", "▶ LTX2.3 Layer Weights", null, function () {
            self._ltxAccordionOpen = !self._ltxAccordionOpen;
            toggleBtn.name = self._ltxAccordionOpen
                ? "▼ LTX2.3 Layer Weights"
                : "▶ LTX2.3 Layer Weights";
            const ltxWidgets = LTX_WIDGET_NAMES
                .map(n => self.widgets?.find(w => w.name === n))
                .filter(Boolean);
            for (const w of ltxWidgets) setWidgetVisible(w, self._ltxAccordionOpen);
            self.setSize(self.computeSize());
            self.setDirtyCanvas(true, true);
        });
        self._ltxToggleBtn = toggleBtn;

        // Move toggle button to sit between "enabled" and "video_strength"
        const enabledIdx = this.widgets.findIndex(w => w.name === "enabled");
        const btnIdx = this.widgets.indexOf(toggleBtn);
        if (enabledIdx >= 0 && btnIdx > enabledIdx + 1) {
            this.widgets.splice(btnIdx, 1);
            this.widgets.splice(enabledIdx + 1, 0, toggleBtn);
        }

        // Watch model_target for changes
        const modelTargetWidget = this.widgets?.find(w => w.name === "model_target");
        if (modelTargetWidget) {
            const origCallback = modelTargetWidget.callback;
            modelTargetWidget.callback = function (value) {
                if (origCallback) origCallback.apply(this, arguments);
                applyLoraEntryTarget(self, value);
            };
            applyLoraEntryTarget(self, modelTargetWidget.value);
        }

        // ── Civitai Info button ──
        this.addWidget("button", "ℹ Civitai Info", null, async function () {
            const loraWidget = self.widgets?.find(w => w.name === "lora");
            const loraName   = loraWidget?.value;

            if (!loraName || loraName === "None") {
                app.extensionManager?.toast?.add({
                    severity: "warn",
                    summary: "No LoRA selected",
                    detail: "Choose a LoRA file before fetching info.",
                    life: 3000,
                });
                return;
            }

            app.extensionManager?.toast?.add({
                severity: "info",
                summary: "Fetching Civitai info…",
                detail: loraName,
                life: 2000,
            });

            try {
                const data = await loraAPI.getCivitaiInfo(loraName);
                showCivitaiModal(data);
            } catch (err) {
                const notFound = err.statusCode === 404;
                app.extensionManager?.toast?.add({
                    severity: "error",
                    summary: "Civitai lookup failed",
                    detail: notFound
                        ? "This LoRA was not found on Civitai."
                        : (err.message || "Request failed"),
                    life: 5000,
                });
            }
        });
    };

    // Re-apply visibility after a saved graph restores widget values
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (data) {
        if (onConfigure) onConfigure.apply(this, arguments);
        const modelTargetWidget = this.widgets?.find(w => w.name === "model_target");
        if (modelTargetWidget) applyLoraEntryTarget(this, modelTargetWidget.value);
    };
}

// ── Node handler: LoraStackBuilder ────────────────────────────────────────────

function applyLoraBuilderTarget(node, target) {
    const isLtx = target === "LTX2.3";
    for (let i = 0; i < LORA_BUILDER_ROWS; i++) {
        const videoW = node.widgets?.find(w => w.name === `video_${i}`);
        const audioW = node.widgets?.find(w => w.name === `audio_${i}`);
        setWidgetVisible(videoW, isLtx);
        setWidgetVisible(audioW, isLtx);
    }
    node.setSize(node.computeSize());
    node.setDirtyCanvas(true, true);
}

export function setupLoraStackBuilder(nodeType, nodeData, app) {
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        if (onNodeCreated) onNodeCreated.apply(this, arguments);
        const self = this;
        const modelTargetWidget = this.widgets?.find(w => w.name === "model_target");
        if (modelTargetWidget) {
            const origCallback = modelTargetWidget.callback;
            modelTargetWidget.callback = function (value) {
                if (origCallback) origCallback.apply(this, arguments);
                applyLoraBuilderTarget(self, value);
            };
            applyLoraBuilderTarget(self, modelTargetWidget.value);
        }
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (data) {
        if (onConfigure) onConfigure.apply(this, arguments);
        const modelTargetWidget = this.widgets?.find(w => w.name === "model_target");
        if (modelTargetWidget) applyLoraBuilderTarget(this, modelTargetWidget.value);
    };
}
