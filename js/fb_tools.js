import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const EXT_PREFIX = "fbt_";

const FROST_CATEGORY = "frost-byte";

// Map node name -> array of widget updates to apply
// Each entry: { widget_index: number, widget_name: string }
const NODE_WIDGET_MAP = {
    [`${EXT_PREFIX}SceneSelect`]: [
        { widget_index: 0, widget_name: "girl_pos_in" },
        { widget_index: 1, widget_name: "male_pos_in" },
        { widget_index: 2, widget_name: "loras_high_in" },
        { widget_index: 3, widget_name: "loras_low_in" },
        { widget_index: 4, widget_name: "wan_prompt_in" },
        { widget_index: 5, widget_name: "wan_low_prompt_in" },
        { widget_index: 6, widget_name: "four_image_prompt_in" },
    ],
    [`${EXT_PREFIX}StoryEdit`]: [
        { widget_index: 1, widget_name: "selected_prompt_in" },
    ],
    [`${EXT_PREFIX}LibberManager`]: [
        // text[0]=keys_json, text[1]=lib_dict_json, text[2]=status
        { widget_index: 0, widget_name: "key_selector", widget_type: "combo" },
    ],
};

function addMenuHandler(nodeType, cb) {
    const menuOptions = nodeType.prototype.getExtraMenuOptions;
    nodeType.prototype.getExtraMenuOptions = function () {
        const r = menuOptions.apply(this, arguments);
        cb.apply(this, arguments);
        return r;
    };
}

function showToast(options) {
    app.extensionManager.toast.add(options);
}

const DATASET_STATUS_SOURCE = "dataset_captioner";
const datasetStatusToastState = {
    lastStatus: "",
    lastTs: 0,
};

function statusToSeverity(level, statusText = "") {
    const normalized = String(level || "").toLowerCase();
    if (normalized === "error" || /error|failed/i.test(statusText)) return "error";
    if (normalized === "warn" || /warn/i.test(statusText)) return "warn";
    if (normalized === "success" || /complete|ready|loaded|unloaded|ok/i.test(statusText)) return "success";
    return "info";
}

function isDatasetToastMilestone(statusText = "") {
    const text = String(statusText).toLowerCase();

    // Start milestones
    const isStart =
        text.includes("loading") && text.includes("model")
        || text.includes("generating caption");

    // End milestones
    const isEnd =
        text.includes("completed")
        || text.includes("complete")
        || text.includes("failed")
        || text.includes("error")
        || text.includes("no images to process");

    return isStart || isEnd;
}

setupLlmQueueWarning(app, api);

api.addEventListener("fbtools.status", (event) => {
    try {
        const detail = event?.detail || {};
        if (detail.source !== DATASET_STATUS_SOURCE) return;

        const status = String(detail.status || "").trim();
        if (!status) return;
        if (!isDatasetToastMilestone(status)) return;

        const now = Date.now();
        const isDuplicate = status === datasetStatusToastState.lastStatus && (now - datasetStatusToastState.lastTs) < 1200;
        if (isDuplicate) return;

        datasetStatusToastState.lastStatus = status;
        datasetStatusToastState.lastTs = now;

        app.extensionManager?.toast?.add({
            severity: statusToSeverity(detail.level, status),
            summary: "Dataset Captioner",
            detail: status,
            life: 2600,
        });
    } catch (err) {
        console.error("fb_tools -> DatasetCaptioner status toast handler error", err);
    }
});

function serializedNodes() {
    const nodes = app.canvas?.selected_nodes || {};
    if (!Object.keys(nodes).length > 0) {
        return [];
    }
    return Object.values(nodes)[0].serialize();
}

// Populates the sidebar's Node Inspector tab (js/ui/node_inspector.js) rather than rendering
// anything itself. The tab activates automatically, but that's easy to miss if the sidebar was
// closed or on a different tab, so surface a toast pointing at it explicitly.
function handleNodes() {
    const nodeData = serializedNodes();
    if (!nodeData || (Array.isArray(nodeData) && !nodeData.length)) return;
    updateNodeInspector(nodeData);
    if (navigator.clipboard) {
        navigator.clipboard.writeText(JSON.stringify(nodeData, null, 2));
    }
    showToast({
        severity: "success",
        summary: "Node JSON Extracted",
        detail: "View it in the fbTools sidebar's Node Inspector tab (also copied to clipboard).",
        life: 3000,
    });
}

// Small collapsed Get/Set nodes (e.g. from KJNodes) are frequently dropped visually on top of the
// larger node they route a value into/out of, which can permanently block clicks to whatever's
// underneath. This is the LIVE half of the fix, for the current editing session. Confirmed by
// testing: canvas.sendToBack(node) (litegraph's own z-index-based mechanism) is what actually
// drives the live renderer correctly -- an earlier version of this function instead spliced
// graph._nodes directly, which reliably reported the right count but did not reliably repaint
// (most likely a stale render-order cache that a raw array reassignment never invalidates). Since
// persistence is independently guaranteed by patchGraphSerializeOrder() below regardless of live
// z-index/array state, this only needs to get the *current view* looking right.
function sendGetSetNodesToBack() {
    const graph = app.canvas?.graph || app.graph;
    const canvas = app.canvas;
    if (!graph || !Array.isArray(graph._nodes) || typeof canvas?.sendToBack !== "function") return;

    const targets = graph._nodes.filter((n) => n.type === "GetNode" || n.type === "SetNode");
    if (!targets.length) {
        showToast({
            severity: "info",
            summary: "No Get/Set Nodes",
            detail: "No GetNode/SetNode instances found in this graph.",
            life: 2500,
        });
        return;
    }

    for (const node of targets) {
        canvas.sendToBack(node);
    }
    graph.setDirtyCanvas(true, true);

    showToast({
        severity: "success",
        summary: "Sent to Back",
        detail: `Moved ${targets.length} Get/Set node${targets.length === 1 ? "" : "s"} to the back of the draw order.`,
        life: 2500,
    });
}

// The actual persistence half of the Get/Set-node z-order fix. graph.serialize() (litegraph's
// long-standing public export method -- also aliased as toJSON() -- confirmed via the bundled
// frontend source to be what produces the exact {id, revision, nodes, links, groups, config,
// extra, version, ...} shape written to a saved workflow file) rebuilds its own `nodes` array
// every time it runs, from an internal registry that tracks insertion order, not from whatever
// order graph._nodes happens to be in live. That means no amount of live array reordering (see
// sendGetSetNodesToBack above) can reliably survive a save on its own. Patching serialize() itself
// to reorder its *output* is the only point that's guaranteed to run on every save (Ctrl+S, the
// Save menu, autosave) regardless of what triggered it or what state the live canvas is in.
function patchGraphSerializeOrder() {
    const graph = app.graph;
    const proto = graph && Object.getPrototypeOf(graph);
    if (!proto || typeof proto.serialize !== "function" || proto.__fbToolsGetSetOrderPatched) return;

    const isGetSet = (n) => n?.type === "GetNode" || n?.type === "SetNode";
    const originalSerialize = proto.serialize;
    proto.serialize = function (...args) {
        const result = originalSerialize.apply(this, args);
        if (result && Array.isArray(result.nodes) && result.nodes.some(isGetSet)) {
            const targets = result.nodes.filter(isGetSet);
            const rest = result.nodes.filter((n) => !isGetSet(n));
            result.nodes = [...targets, ...rest];
        }
        return result;
    };
    proto.__fbToolsGetSetOrderPatched = true;
}

/**
 * Update a widget's value from message.text array
 * @param {object} node - The node instance
 * @param {Array} textArray - The message.text array
 * @param {number} index - Index into the text array
 * @param {string} widgetName - Name of the widget to update
 * @param {string} widgetType - Type of the widget (default: "text")
 * @param {string} logPrefix - Prefix for console log messages
 */
function updateWidgetFromText(node, textArray, index, widgetName, widgetType = "text", logPrefix = "fbTools") {
    if (textArray && textArray[index]) {
        const widget = node.widgets.find((w) => w.name === widgetName);
        if (widget) {

            if (widgetType === "text" ) {
                widget.value = textArray[index];
                if (widget.inputEl) {
                    widget.inputEl.value = textArray[index];
                }
                console.log(`${logPrefix}: ${widgetName} updated from text[${index}]`);
            }

            else if (widgetType === "combo") {
                const newOptions = textArray[index];
                const options = (widget.options && widget.options.values) || widget.options || [];
                if (options.includes(textArray[index])) {
                    widget.value = textArray[index];
                    if (widget.inputEl) {
                        widget.inputEl.value = textArray[index];
                    }
                    console.log(`${logPrefix}: ${widgetName} updated from text[${index}]`);
                } else {
                    console.warn(`${logPrefix}: ${widgetName} - value from text[${index}] not in options, skipping update`);
                }
            }
            return true;
        }
    }
    return false;
}

// Bulk update widgets for a node based on NODE_WIDGET_MAP
function updateNodeInputs(node, textArray, nodeName) {
    const entries = NODE_WIDGET_MAP[nodeName];
    if (!entries || !entries.length) return;
    entries.forEach(({ widget_index, widget_name, widget_type }) => {
        updateWidgetFromText(node, textArray, widget_index, widget_name, widget_type, `fbTools -> ${nodeName}`);
    });
}

// Normalize node resize/refresh after widget updates
function scheduleNodeRefresh(node, app) {
    requestAnimationFrame(() => {
        const sz = node.computeSize();
        if (sz[0] < node.size[0]) sz[0] = node.size[0];
        if (sz[1] < node.size[1]) sz[1] = node.size[1];
        node.onResize?.(sz);
        app.graph.setDirtyCanvas(true, false);
    });
}

// Import node-specific modules
import { setupSceneSelect, setupScenePromptManager, setupSceneView } from "./nodes/scene.js";
import { setupSceneUpdateStatus } from "./nodes/sceneUpdateStatus.js";
import { setupStoryEdit, setupStoryView, setupStorySceneBatch } from "./nodes/story.js";
import { setupLibberManager, setupLibberApply } from "./nodes/libber.js";
import { setupDatasetCaptionViewer } from "./nodes/dataset_caption_viewer.js";
import { setupDatasetCaptionerStatus } from "./nodes/dataset_caption_status.js";
import { setupLoraEntryDefine, setupLoraPresetDefine, setupLoraPresetSelect, setupWanPresetDefine, setupWanPresetSelect, setupLoraStackBuilder } from "./nodes/lora.js";
import { setupConceptDefine, setupConceptRegistryLoad } from "./nodes/concepts.js";
import { setupSceneCastBuild } from "./nodes/scene_cast_build.js";
import { setupSourceProfileLoad } from "./nodes/source_profile_load.js";
import { setupCompositionLoad } from "./nodes/composition_load.js";
import { setupPromptCompositionLoader } from "./nodes/prompt_composition_loader.js";
import { setupSourceProfileClipPrompt } from "./nodes/source_profile_clip_prompt.js";
import { renderFbtPanel } from "./ui/fbt_panel.js";
import { patchNodeForTracking } from "./utils/run_tracker.js";
import { setupLlmQueueWarning } from "./utils/llm_queue_warning.js";
import { updateNodeInspector } from "./ui/node_inspector.js";

// Single sidebar entry — hosts Compose, Assets, Casts, Sources, History tabs
// with a persistent LLM status bar and lazy tab mounting.
app.extensionManager.registerSidebarTab({
    id: "fbt.panel",
    icon: "pi pi-box",
    title: "fbTools",
    tooltip: "Prompt Compositions · Assets (bundles, subjects, backgrounds…) · Scene Casts · Source Profiles · Run History",
    type: "custom",
    render: (el) => {
        window._fbtApp = app;
        renderFbtPanel(el);
    },
});

// Add context menu entry for extracting a node as json
app.registerExtension({
    name: "FBToolsContextMenu",
    async init() {
        const styleTagId = 'fb_tools-stylesheet';
        let styleTag = document.getElementById(styleTagId);
        if (styleTag) {
            return;
        }

        document.head.appendChild(Object.assign(document.createElement('link'), {
            id: styleTagId,
            rel: 'stylesheet',
            type: 'text/css',
            href: 'extensions/comfyui-fbTools/styles/style.css'
        }));
    },
    setup() {
        patchGraphSerializeOrder();
    },
    commands: [{
        id: "fb_tools.extract-node-json",
        label: "Extract Node as JSON",
        icon: "pi pi-file-arrow-up",
        function: handleNodes,
    }, {
        id: "fb_tools.send-get-set-to-back",
        label: "Send Get/Set Nodes to Back",
        icon: "pi pi-angle-double-down",
        function: sendGetSetNodesToBack,
    }],
    getSelectionToolboxCommands: (selectedItem) => {
        return ["fb_tools.extract-node-json", "fb_tools.send-get-set-to-back"];
    },
    nodeCreated(node) {
        patchNodeForTracking(node, app);
    },
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        const isNode = (baseName) => nodeData.name === `${EXT_PREFIX}${baseName}` || nodeData.name === baseName;
        const isFrostCategory = nodeData?.category?.indexOf(FROST_CATEGORY) >= 0;

        // Most node handlers are frost-byte category scoped, but DatasetCaptionViewer
        // currently lives under Dataset/Caption and still needs frontend integration.
        if (!isFrostCategory && !isNode("DatasetCaptionViewer")) {
            return;
        }
        
        // Scene nodes
        if (isNode("SceneSelect")) {
            setupSceneSelect(nodeType, nodeData, app);
        }
        else if (isNode("SceneUpdate")) {
            setupSceneUpdateStatus(nodeType, nodeData, app);
        }
        else if (isNode("SceneView")) {
            setupSceneView(nodeType, nodeData, app);
        }
        else if (isNode("ScenePromptManager")) {
            setupScenePromptManager(nodeType, nodeData, app);
        }
        
        // Story nodes
        else if (isNode("StoryEdit")) {
            setupStoryEdit(nodeType, nodeData, app);
        }
        else if (isNode("StoryView")) {
            setupStoryView(nodeType, nodeData, app);
        }
        else if (isNode("StorySceneBatch")) {
            setupStorySceneBatch(nodeType, nodeData, app);
        }
        
        // Libber nodes
        else if (isNode("LibberManager")) {
            setupLibberManager(nodeType, nodeData, app);
        }
        else if (isNode("LibberApply")) {
            setupLibberApply(nodeType, nodeData, app);
        }

        // Dataset captioning nodes
        else if (isNode("DatasetCaptioner")) {
            setupDatasetCaptionerStatus(nodeType, nodeData, app);
        }
        else if (isNode("DatasetCaptionViewer")) {
            setupDatasetCaptionViewer(nodeType, nodeData, app);
        }

        // LoRA nodes
        else if (isNode("LoraStackBuilder")) {
            setupLoraStackBuilder(nodeType, nodeData, app);
        }
        else if (isNode("LoraEntryDefine")) {
            setupLoraEntryDefine(nodeType, nodeData, app);
        }
        else if (isNode("LoraPresetDefine")) {
            setupLoraPresetDefine(nodeType, nodeData, app);
        }
        else if (isNode("LoraPresetSelect")) {
            setupLoraPresetSelect(nodeType, nodeData, app);
        }
        else if (isNode("WanPresetDefine")) {
            setupWanPresetDefine(nodeType, nodeData, app);
        }
        else if (isNode("WanPresetSelect")) {
            setupWanPresetSelect(nodeType, nodeData, app);
        }

        // Concept Registry nodes
        else if (isNode("ConceptRegistryLoad")) {
            setupConceptRegistryLoad(nodeType, nodeData, app);
        }
        else if (isNode("ConceptDefine")) {
            setupConceptDefine(nodeType, nodeData, app);
        }

        // Scene Cast nodes
        else if (isNode("SceneCastBuild")) {
            setupSceneCastBuild(nodeType, nodeData, app);
        }
        else if (isNode("SourceProfileLoad")) {
            setupSourceProfileLoad(nodeType, nodeData, app);
        }
        else if (isNode("CompositionLoad")) {
            setupCompositionLoad(nodeType, nodeData, app);
        }
        else if (isNode("PromptCompositionLoader")) {
            setupPromptCompositionLoader(nodeType, nodeData, app);
        }
        else if (isNode("SourceProfileClipPrompt")) {
            setupSourceProfileClipPrompt(nodeType, nodeData, app);
        }

        // Add context menu for frost-byte nodes only.
        if (isFrostCategory) {
            addMenuHandler(nodeType, function (_, options) {
                options.push({
                    content: "Extract Node as JSON",
                    callback: handleNodes,
                });
            });
        }
    },
});