/**
 * Concept Registry node frontend extensions.
 *
 * ConceptRegistryLoad — adds a "Reload Registry" button that calls
 *   POST /fbtools/concepts/reload so the node re-executes on next queue.
 *
 * ConceptDefine — hides lora_low / weight_low widgets for single-model types.
 */

import { conceptsAPI } from "../api/concepts.js";
import { setWidgetVisible } from "../utils/widgets.js";

// Model types that use a split high/low model
const SPLIT_MODEL_TYPES = new Set(["wan22", "bernini"]);

// ── ConceptDefine ─────────────────────────────────────────────────────────────

function applyConceptDefineModelType(node, modelType) {
    const isSplit = SPLIT_MODEL_TYPES.has(modelType);
    for (const name of ["lora_low", "weight_low"]) {
        const widget = node.widgets?.find(w => w.name === name);
        setWidgetVisible(widget, isSplit);
    }
    node.setSize(node.computeSize());
    node.setDirtyCanvas(true, true);
}

export function setupConceptDefine(nodeType, nodeData, app) {
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        if (onNodeCreated) onNodeCreated.apply(this, arguments);
        const self = this;
        const modelTypeWidget = this.widgets?.find(w => w.name === "model_type");
        if (modelTypeWidget) {
            const origCallback = modelTypeWidget.callback;
            modelTypeWidget.callback = function (value) {
                if (origCallback) origCallback.apply(this, arguments);
                applyConceptDefineModelType(self, value);
            };
            applyConceptDefineModelType(self, modelTypeWidget.value);
        }
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (data) {
        if (onConfigure) onConfigure.apply(this, arguments);
        const modelTypeWidget = this.widgets?.find(w => w.name === "model_type");
        if (modelTypeWidget) applyConceptDefineModelType(this, modelTypeWidget.value);
    };
}

// ── ConceptRegistryLoad ───────────────────────────────────────────────────────

export function setupConceptRegistryLoad(nodeType, nodeData, app) {
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        if (onNodeCreated) onNodeCreated.apply(this, arguments);
        const self = this;

        this.addWidget("button", "↺ Reload Registry", null, async function () {
            try {
                await conceptsAPI.reload();
                app.extensionManager?.toast?.add({
                    severity: "success",
                    summary: "Registry reload triggered",
                    detail: "Re-queue the workflow to load the updated concepts.",
                    life: 3000,
                });
            } catch (e) {
                app.extensionManager?.toast?.add({
                    severity: "error",
                    summary: "Reload failed",
                    detail: e.message || "Unknown error",
                    life: 5000,
                });
            }
        });
    };
}
