import { app } from "../../scripts/app.js";

const EXT_PREFIX = "fbt_";

const PRIME_SWAP = {
    amber: "amber",
    blue: "blue",
    gray: "gray",
    green: "green",
    rose: "red",
    stone: "stone",
};

const ALLOWED_WEIGHTS = ["100", "200", "300", "400", "500", "600", "700", "800", "900"];
const ALLOWED_NUMS = ALLOWED_WEIGHTS.map(w => Number(w)).sort((a, b) => a-b);

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
        { widget_index: 2, widget_name: "story_json_in" },
        { widget_index: 4, widget_name: "story_scene_selector", widget_type: "combo" },
    ],
    [`${EXT_PREFIX}LibberManager`]: [
        // text[0]=keys_json, text[1]=lib_dict_json, text[2]=status
        { widget_index: 0, widget_name: "key_selector", widget_type: "combo" },
    ],
};

function nearestAllowedWeight(weight) {
    const w = Number(weight);
    if (Number.isNaN(w) || !ALLOWED_NUMS.length) return ALLOWED_WEIGHTS[0];
    if (w <= ALLOWED_NUMS[0]) return String(ALLOWED_WEIGHTS[0]);
    if (w >= ALLOWED_NUMS[ALLOWED_NUMS.length - 1]) return String(ALLOWED_WEIGHTS[ALLOWED_NUMS.length - 1]);
    let lo = 0;
    let hi = ALLOWED_NUMS.length - 1;
    while (lo <= hi) {
        const mid = lo + hi >> 1;
        const mv = ALLOWED_NUMS[mid];
        if (mv === w) return String(mv);
        if (mv < w) {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    const lower = ALLOWED_NUMS[hi];
    const upper = ALLOWED_NUMS[lo];
    return String(Math.abs(w - lower) <= Math.abs(upper - w) ? lower : upper);
}

const TEXT_CLASS_REGEX = /\btext-([a-z]+)-(\d{2,3})\b/;

/**
 * Applies Prime Vue color variables to elements with Tailwind CSS text color classes.
 * ComfyUI uses Prime Vue for theming, so this function helps ensure that existing text colors are applied.
 * @param {HTMLElement} root
 * @param {object} param1 
 */
function applyPrimeTextVars(root = document, { removeClass = false } = {}) {
    // Find all child elements of root with class matching text-
    const textElements = root.querySelectorAll("[class*='text-']");
 
    textElements.forEach((elm) => {
        const classList = Array.from(elm.classList);
        classList.forEach((cls) => {
            const match = cls.match(TEXT_CLASS_REGEX);

            if (!match) return;
            // matched text-{color}-{weight}
            const [, twColor, weight] = match;

            // If the color is in the swap list, replace it
            if (twColor in PRIME_SWAP) {
                const newColor = PRIME_SWAP[twColor] || twColor;
                const newWeight = nearestAllowedWeight(weight);
                const newColorStyle = `var(--p-${newColor}-${newWeight})`;
                elm.style.color = newColorStyle;

                // remove the tailwind class if specified
                if (removeClass) {
                    elm.classList.remove(`text-${twColor}-${weight}`);
                }
            }
        });
    });
}

// https://unpkg.com/jsnview@3.0.0/dist/index.js
const JSONView = {
    _promise: null,
    async ensureLoaded() {
        if (this._promise) return this._promise;
        this._promise = new Promise((resolve, reject) => {
            const script = document.createElement("script");
            script.src = "https://unpkg.com/jsnview@3.0.0/dist/index.js";
            script.onload = () => {
                resolve(window.jsnview);
            };
            script.onerror = (e) => {
                reject(e);
            };
            document.head.appendChild(script);
        });
        return this._promise;
    },
    get JSONView() {
        return window.jsnview;
    },
}
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
const successToast = {
    severity: "success",
    summary: "Copied",
    detail: "Node JSON copied to Bottom Panel and Clipboard (if supported)",
    life: 2000,
}
const clipboardErrorToast = {
    severity: "error",
    summary: "Clipboard Error",
    detail: "Failed to copy node JSON to clipboard, requires HTTPS connection",
    life: 4000,
}

function serializedNodes() {
    const nodes = app.canvas?.selected_nodes || {};
    if (!Object.keys(nodes).length > 0) {
        return [];
    }
    return Object.values(nodes)[0].serialize();
}

function displayNodesInTab(nodeData) {
    clearTab();
    const tabContainer = document.getElementById(tabContainerId);
    if (tabContainer && nodeData) {
        if (!JSONView.JSONView) {
            tabContainer.innerHTML = `<pre style="white-space: pre-wrap; word-break: break-all; max-height: 400px; overflow: auto; padding: 1rem;">${nodeData}</pre>`;
        } else {
            const viewerElement = document.createElement("div");
            viewerElement.id = "fb_tools_json_viewer";
            const formatter = new JSONView.JSONView(nodeData, {
                element: viewerElement,
                collapsed: false,
                showLen: true,
                showType: false,
                showFoldmarker: true,
                maxDepth: 4,
            });
            initTab(formatter);
        }
        //showToast(successToast);
    }
}
function fixPropertyColors() {
    const tabContainer = document.getElementById(tabContainerId);
    if (!tabContainer) return;
    tabContainer.querySelectorAll(".text-amber-800").forEach((elm => {
        elm.classList.remove("text-amber-800");
        elm.style.color = 'var(--p-amber-500)';
    }));
    tabContainer.querySelectorAll(".text-rose-400").forEach((elm => {
        elm.classList.remove("text-rose-400");
        elm.style.color = 'var(--p-red-400)';
    }));
    tabContainer.querySelectorAll(".text-stone-700").forEach((elm => {
        elm.classList.remove("text-stone-700");
        elm.style.color = 'var(--p-stone-600)';
    }));

}
function collapseAll() { 
    const root = document.querySelector(`#${tabContainerId} .jsv`);
    root?.querySelectorAll(".jsv-toggle").forEach(
        (t) => {
            const s = t.parentElement?.querySelector(".jsv-content");

            if (!s.classList.contains("hidden")) {
                t.classList.add("-rotate-90");
                s.classList.add("hidden");
            }
        }
    )
}
function expandAll() { 
    const root = document.querySelector(`#${tabContainerId} .jsv`);
    root?.querySelectorAll(".jsv-toggle").forEach(
        (t) => { 
            const s = t.parentElement?.querySelector(".jsv-content");
            if (s?.classList?.contains("hidden")) {
                t.classList.remove("-rotate-90");
                s.classList.remove("hidden");
            }
        }
    )
}
function initTab(formatter) {
    const tabContainer = document.getElementById(tabContainerId);
    if (!tabContainer) return;
    tabContainer.style.overflow = "auto";
    const formatterElement = formatter.getElement();
    const buttonContainer = document.createElement("div");
    buttonContainer.style.marginBottom = "1rem";
    const collapseButton = document.createElement("button");
    collapseButton.innerText = "Collapse All";
    collapseButton.style.marginRight = "1rem";
    collapseButton.onclick = () => {
        collapseAll();
    };
    const expandButton = document.createElement("button");
    expandButton.innerText = "Expand All";
    expandButton.onclick = () => {
        expandAll();
    };
    buttonContainer.appendChild(collapseButton);
    buttonContainer.appendChild(expandButton);
    tabContainer.appendChild(buttonContainer);
    tabContainer.appendChild(formatterElement);
    applyPrimeTextVars(tabContainer, { removeClass: true });
}

function clearTab() {
    const tabContainer = document.getElementById(tabContainerId);
    if (tabContainer) {
        tabContainer.innerHTML = `<div style="padding: 1rem;">FB Tools Extension Loaded</div>`;
    }
}

function handleNodes() {
    const nodeData = serializedNodes();
    displayNodesInTab(nodeData);
    if (navigator.clipboard) {
        navigator.clipboard.writeText(JSON.stringify(nodeData, null, 2));
    }
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

// Derive scene selector options from the story JSON or an explicit options JSON payload, then update the combo widget.
function updateStorySceneSelector(node, storyJsonText, optionsJsonText) {
    let options = null;

    // Priority 1: explicit options payload (text[3] from StoryEdit)
    const optionsJson = optionsJsonText || node.widgets?.find((w) => w.name === "story_scene_selector_options")?.value;
    if (!options && optionsJson && typeof optionsJson === "string") {
        try {
            const parsed = JSON.parse(optionsJson);
            if (Array.isArray(parsed) && parsed.length) {
                options = parsed;
            }
        } catch (err) {
            console.warn("fbTools -> StoryEdit: failed to parse selector options JSON", err);
        }
    }

    // Priority 2: derive from story JSON when options not provided
    if (!options || !options.length) {
        const storyJson = storyJsonText || node.widgets?.find((w) => w.name === "story_json_in")?.value;
        if (storyJson && typeof storyJson === "string") {
            try {
                const data = JSON.parse(storyJson);
                if (Array.isArray(data?.scenes)) {
                    const scenes = [...data.scenes].sort((a, b) => (a?.scene_order ?? 0) - (b?.scene_order ?? 0));
                    options = scenes.map((scene, idx) => `${idx}: ${scene?.scene_name || "scene"}`);
                }
            } catch (err) {
                console.warn("fbTools -> StoryEdit: failed to parse story_json for selector options", err);
            }
        }
    }

    if (!options || !options.length) return;

    const widget = node.widgets?.find((w) => w.name === "story_scene_selector");
    if (!widget) return;

    const currentOptions = (widget.options && widget.options.values) || widget.options || [];
    const sameOptions =
        Array.isArray(currentOptions) &&
        currentOptions.length === options.length &&
        currentOptions.every((val, idx) => val === options[idx]);

    const nextValue = options.includes(widget.value) ? widget.value : options[0];

    if (!sameOptions) {
        if (widget.options && typeof widget.options === "object") {
            widget.options.values = options;
        } else {
            widget.options = { values: options };
        }
        widget.options_values = options;

        if (widget.inputEl && widget.inputEl.tagName === "SELECT") {
            widget.inputEl.innerHTML = "";
            options.forEach((opt) => {
                const optionEl = document.createElement("option");
                optionEl.value = opt;
                optionEl.textContent = opt;
                widget.inputEl.appendChild(optionEl);
            });
        }
    }

    if (widget.value !== nextValue) {
        widget.value = nextValue;
        if (widget.inputEl) {
            widget.inputEl.value = nextValue;
        }
    }
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

const tabContainerId = "fb_tools_container";
let _isLoaded = false;

// Import node-specific modules
import { setupSceneSelect, setupScenePromptManager } from "./nodes/scene.js";
import { setupStoryEdit, setupStoryView, setupStorySceneBatch } from "./nodes/story.js";
import { setupLibberManager, setupLibberApply } from "./nodes/libber.js";

// Add context menu entry for extracting a node as json
app.registerExtension({
    name: "FBToolsContextMenu",
    setup() {
        JSONView.ensureLoaded().then(() => {
            console.log("fb_tools -> JSONView loaded:", JSONView.JSONView);
            _isLoaded = true;
        }).catch((e) => {
            console.error("fb_tools -> Failed to load JSONView:", e);
        });
    },
    bottomPanelTabs: [
        {
            id: "fb_tools",
            title: "FB Tools",
            icon: "pi pi-wrench",
            type: "custom",
            render: (element) => {
                Object.assign(element.style, {
                    display: "flex",
                    flexDirection: "column",
                    height: "100%",
                    minHeight: "0",
                    overflow: "hidden",
                });
                const container = document.createElement("div");
                container.id = tabContainerId;
                container.innerHTML = `<div style="padding: 1rem;"><i>Select a node and click the 'view json' button to display its json here. (and copy it to the clipboard)</i></div>`;
                Object.assign(container.style, {
                    flex: "1 1 auto",
                    minHeight: "0",
                    padding: "8px",
                    overflow: "auto",
                    overscrollBehavior: "contain",
                });
                element.appendChild(container);
                handleNodes();
            }
        },
    ],
    commands: [{
        id: "fb_tools.extract-node-json",
        label: "Extract Node as JSON",
        icon: "pi pi-file-arrow-up",
        function: handleNodes,
    }],
    getSelectionToolboxCommands: (selectedItem) => {
        return ["fb_tools.extract-node-json"];
    },
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData?.category?.indexOf(FROST_CATEGORY) < 0) {
            return;
        }
        
        const isNode = (baseName) => nodeData.name === `${EXT_PREFIX}${baseName}` || nodeData.name === baseName;
        
        // Scene nodes
        if (isNode("SceneSelect")) {
            setupSceneSelect(nodeType, nodeData, app);
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
        
        // Add context menu for all frost-byte nodes
        addMenuHandler(nodeType, function (_, options) {
            options.push({
                content: "Extract Node as JSON",
                callback: handleNodes,
            });
        });
    },
});