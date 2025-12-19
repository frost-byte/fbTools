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
        if (isNode("SceneSelect")) {
            console.log("fb_tools -> SceneSelect node detected");
            const onOriginalExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                if (onOriginalExecuted) {
                    onOriginalExecuted.apply(this, arguments);
                }

                updateNodeInputs(this, message?.text, nodeData.name);
                scheduleNodeRefresh(this, app);
            };
        }
        
        if (isNode("StoryEdit")) {
            console.log("fb_tools -> StoryEdit node detected");
            const onOriginalCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onOriginalCreated) {
                    onOriginalCreated.apply(this, arguments);
                }
                updateStorySceneSelector(this);
            };
            const onOriginalExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                if (onOriginalExecuted) {
                    onOriginalExecuted.apply(this, arguments);
                }

                // StoryEdit returns: text[0]=scene_list, text[1]=selected_prompt, text[2]=story_json
                const textArray = message?.text;

                updateNodeInputs(this, textArray, nodeData.name);
                console.log("fbTools -> StoryEdit: updating scene selector options");
                console.log("fbTools -> StoryEdit: message =", JSON.stringify(message));
                updateStorySceneSelector(
                    this,
                    Array.isArray(textArray) ? textArray[2] : null,
                    Array.isArray(textArray) ? textArray[3] : null,
                );
                
                // Special handling: update custom_prompt with selected prompt (unless in custom mode)
                if (textArray && textArray[1]) {
                    const promptTypeWidget = this.widgets.find((w) => w.name === "prompt_type");
                    const customPromptWidget = this.widgets.find((w) => w.name === "custom_prompt");
                    
                    if (customPromptWidget && promptTypeWidget) {
                        // Only update custom_prompt if not already in custom mode or if it's empty
                        if (promptTypeWidget.value !== "custom" || !customPromptWidget.value) {
                            customPromptWidget.value = textArray[1];
                            if (customPromptWidget.inputEl) {
                                customPromptWidget.inputEl.value = textArray[1];
                            }
                        }
                    }
                }
                
                scheduleNodeRefresh(this, app);
            };
        }
        
        if (isNode("StoryView")) {
            console.log("fb_tools -> StoryView node detected");
            const onOriginalExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                if (onOriginalExecuted) {
                    onOriginalExecuted.apply(this, arguments);
                }

                // StoryView returns the selected prompt in text array
                // The preview text includes multiple lines, but we want to update the prompt_in widget
                // with the actual selected prompt value
                if (message?.text) {
                    // The text array from StoryView contains the preview text
                    // We need to extract the prompt from it
                    const previewText = Array.isArray(message.text) ? message.text.join('\n') : message.text;
                    
                    // Look for the "Prompt:" line in the preview text and capture everything until the next section
                    // Use [\s\S] to match any character including newlines
                    const promptMatch = previewText.match(/Prompt:\s*([\s\S]+?)(?=\n\nAll Scenes:|$)/);
                    if (promptMatch && promptMatch[1]) {
                        const promptValue = promptMatch[1].trim();
                        console.log("fbTools -> StoryView: prompt detected, applying to widget:", promptValue.substring(0, 50));
                        const promptWidget = this.widgets.find((w) => w.name === "prompt_in");
                        if (promptWidget) {
                            promptWidget.value = promptValue;
                            if (promptWidget.inputEl) {
                                promptWidget.inputEl.value = promptValue;
                            }
                        }
                    }
                }
                
                scheduleNodeRefresh(this, app);
            };
        }
        
        if (isNode("LibberManager")) {
            console.log("fb_tools -> LibberManager node detected");
            
            // Import the API client
            import("./api/libber.js").then(({ libberAPI }) => {
                const onNodeCreated = nodeType.prototype.onNodeCreated;
                nodeType.prototype.onNodeCreated = function () {
                    if (onNodeCreated) {
                        onNodeCreated.apply(this, arguments);
                    }
                    
                    const widgets = this.widgets || [];
                    const operationWidget = widgets.find(w => w.name === "operation");
                    const libberNameWidget = widgets.find(w => w.name === "libber_name");
                    const filenameWidget = widgets.find(w => w.name === "filename");
                    const libberDirWidget = widgets.find(w => w.name === "libber_dir");
                    const keySelectorWidget = widgets.find(w => w.name === "key_selector");
                    const libKeyWidget = widgets.find(w => w.name === "lib_key");
                    const libValueWidget = widgets.find(w => w.name === "lib_value");
                    const delimiterWidget = widgets.find(w => w.name === "delimiter");
                    const maxDepthWidget = widgets.find(w => w.name === "max_depth");
                    
                    // Auto-load libber on node creation if filename exists
                    const autoLoadLibber = async () => {
                        const filename = filenameWidget?.value;
                        const libberDir = libberDirWidget?.value;
                        const libberName = libberNameWidget?.value;
                        
                        if (filename && libberDir && libberName) {
                            const loadPath = `${libberDir}/${filename}`;
                            try {
                                const result = await libberAPI.loadLibber(libberName, loadPath);
                                if (result && result.keys && keySelectorWidget) {
                                    const newOptions = ["", ...result.keys];
                                    keySelectorWidget.options.values = newOptions;
                                    if (!keySelectorWidget.options.values.includes(keySelectorWidget.value)) {
                                        keySelectorWidget.value = keySelectorWidget.options.values[0];
                                    }
                                    console.log("fbTools -> LibberManager: auto-loaded libber on node creation");
                                }
                            } catch (error) {
                                // File doesn't exist or load failed, clear widgets
                                console.log("fbTools -> LibberManager: auto-load skipped, clearing widgets");
                                if (keySelectorWidget) {
                                    keySelectorWidget.options.values = [""];
                                    keySelectorWidget.value = "";
                                }
                                if (libKeyWidget) {
                                    libKeyWidget.value = "";
                                    if (libKeyWidget.inputEl) {
                                        libKeyWidget.inputEl.value = "";
                                    }
                                }
                            }
                        }
                    };
                    
                    // Run auto-load after a short delay to ensure widgets are initialized
                    setTimeout(autoLoadLibber, 100);
                    
                    // Add Execute button
                    this.addWidget("button", "Execute", null, async () => {
                        console.log("fbTools -> LibberManager: Execute button clicked");
                        
                        const operation = operationWidget?.value;
                        const libberName = libberNameWidget?.value;
                        const filename = filenameWidget?.value;
                        const libberDir = libberDirWidget?.value;
                        const libKey = libKeyWidget?.value;
                        const libValue = libValueWidget?.value;
                        const delimiter = delimiterWidget?.value || "%";
                        const maxDepth = maxDepthWidget?.value || 10;
                        
                        try {
                            let result;
                            let addedKey = null; // Track the key that was added
                            let isRemoveOperation = false; // Track if this was a remove operation
                            
                            switch (operation) {
                                case "create":
                                    result = await libberAPI.createLibber(libberName, delimiter, maxDepth);
                                    showToast({ severity: "success", summary: `Created libber '${libberName}'`, life: 2000 });
                                    break;
                                    
                                case "load":
                                    const loadPath = `${libberDir}/${filename}`;
                                    result = await libberAPI.loadLibber(libberName, loadPath);
                                    showToast({ severity: "success", summary: `Loaded libber '${libberName}'`, life: 2000 });
                                    break;
                                    
                                case "add_lib":
                                    if (!libKey) {
                                        showToast({ severity: "warn", summary: "lib_key required for add_lib", life: 2000 });
                                        return;
                                    }
                                    result = await libberAPI.addLib(libberName, libKey, libValue);
                                    addedKey = libKey.toLowerCase().replace(/\s/g, "_").replace(/-/g, "_"); // Normalize to match backend
                                    showToast({ severity: "success", summary: `Added lib '${libKey}'`, life: 2000 });
                                    // Clear lib_key after successful add
                                    libKeyWidget.value = "";
                                    if (libKeyWidget.inputEl) {
                                        libKeyWidget.inputEl.value = "";
                                    }
                                    // Auto-save after adding
                                    if (filename && libberDir) {
                                        const savePath = `${libberDir}/${filename}`;
                                        await libberAPI.saveLibber(libberName, savePath);
                                        console.log("fbTools -> LibberManager: auto-saved after add_lib");
                                    }
                                    break;
                                    
                                case "remove_lib":
                                    const keyToRemove = libKey || keySelectorWidget?.value;
                                    if (!keyToRemove) {
                                        showToast({ severity: "warn", summary: "lib_key or key_selector required", life: 2000 });
                                        return;
                                    }
                                    result = await libberAPI.removeLib(libberName, keyToRemove);
                                    isRemoveOperation = true;
                                    showToast({ severity: "success", summary: `Removed lib '${keyToRemove}'`, life: 2000 });
                                    // Clear lib_key after successful remove
                                    libKeyWidget.value = "";
                                    if (libKeyWidget.inputEl) {
                                        libKeyWidget.inputEl.value = "";
                                    }
                                    // Auto-save after removing
                                    if (filename && libberDir) {
                                        const savePath = `${libberDir}/${filename}`;
                                        await libberAPI.saveLibber(libberName, savePath);
                                        console.log("fbTools -> LibberManager: auto-saved after remove_lib");
                                    }
                                    break;
                                    
                                case "save":
                                    const savePath = `${libberDir}/${filename}`;
                                    result = await libberAPI.saveLibber(libberName, savePath);
                                    showToast({ severity: "success", summary: `Saved libber to '${filename}'`, life: 2000 });
                                    break;
                                    
                                default:
                                    showToast({ severity: "warn", summary: `Unknown operation: ${operation}`, life: 2000 });
                                    return;
                            }
                            
                            // Update key_selector dropdown with latest keys
                            if (result && result.keys && keySelectorWidget) {
                                const newOptions = ["", ...result.keys];
                                keySelectorWidget.options.values = newOptions;
                                
                                // If we just added a key, select it
                                if (addedKey && keySelectorWidget.options.values.includes(addedKey)) {
                                    keySelectorWidget.value = addedKey;
                                } else if (isRemoveOperation) {
                                    // After remove, select first available key and update lib_value
                                    const firstKey = newOptions.length > 1 ? newOptions[1] : ""; // Skip empty string at index 0
                                    keySelectorWidget.value = firstKey;
                                    
                                    // Update lib_value to match the selected key
                                    if (firstKey && libValueWidget) {
                                        try {
                                            const data = await libberAPI.getLibberData(libberName);
                                            if (data && data.lib_dict && data.lib_dict[firstKey]) {
                                                libValueWidget.value = data.lib_dict[firstKey];
                                                if (libValueWidget.inputEl) {
                                                    libValueWidget.inputEl.value = data.lib_dict[firstKey];
                                                }
                                            }
                                        } catch (err) {
                                            console.warn("fbTools -> LibberManager: failed to fetch value for first key", err);
                                        }
                                    } else if (libValueWidget) {
                                        // No keys left, clear lib_value
                                        libValueWidget.value = "";
                                        if (libValueWidget.inputEl) {
                                            libValueWidget.inputEl.value = "";
                                        }
                                    }
                                } else if (!keySelectorWidget.options.values.includes(keySelectorWidget.value)) {
                                    // Otherwise, reset value if it's not in the new options
                                    keySelectorWidget.value = keySelectorWidget.options.values[0];
                                }
                            }
                            
                            console.log("fbTools -> LibberManager: Operation result:", result);
                            
                        } catch (error) {
                            console.error("fbTools -> LibberManager: Error:", error);
                            showToast({ severity: "error", summary: `Error: ${error.message}`, life: 3000 });
                        }
                    });
                    
                    // Update key_selector when a libber is loaded/created
                    const onWidgetChanged = this.onWidgetChanged;
                    this.onWidgetChanged = function (widgetName, newValue, oldValue, widgetObject) {
                        const originalReturn = onWidgetChanged?.apply(this, arguments);
                        
                        if (widgetName === "key_selector" && newValue && libValueWidget) {
                            // When a key is selected, fetch its value from the server
                            const libberName = libberNameWidget?.value;
                            if (libberName) {
                                libberAPI.getLibberData(libberName).then(data => {
                                    if (data && data.lib_dict && data.lib_dict[newValue]) {
                                        libValueWidget.value = data.lib_dict[newValue];
                                        if (libValueWidget.inputEl) {
                                            libValueWidget.inputEl.value = data.lib_dict[newValue];
                                        }
                                        console.log("fbTools -> LibberManager: populated value for key", newValue);
                                    }
                                }).catch(err => {
                                    console.warn("fbTools -> LibberManager: failed to fetch libber data", err);
                                });
                            }
                        }
                        
                        return originalReturn;
                    };
                };
                
                // Handle execution updates
                const onExecuted = nodeType.prototype.onExecuted;
                nodeType.prototype.onExecuted = function (message) {
                    if (onExecuted) {
                        onExecuted.apply(this, arguments);
                    }
                    
                    // LibberManager returns: text[0]=keys_json, text[1]=lib_dict_json, text[2]=status
                    const textArray = message?.text;
                    if (textArray && textArray.length >= 1) {
                        try {
                            const keys = JSON.parse(textArray[0]);
                            const keySelectorWidget = this.widgets?.find(w => w.name === "key_selector");
                            if (keySelectorWidget && Array.isArray(keys)) {
                                const newOptions = ["", ...keys];
                                keySelectorWidget.options.values = newOptions;
                                // Reset value if it's not in the new options
                                if (!keySelectorWidget.options.values.includes(keySelectorWidget.value)) {
                                    keySelectorWidget.value = keySelectorWidget.options.values[0];
                                }
                                console.log("fbTools -> LibberManager: updated key_selector options", keys);
                            }
                        } catch (err) {
                            console.warn("fbTools -> LibberManager: failed to parse keys JSON", err);
                        }
                    }
                };
            }).catch(err => {
                console.error("fb_tools -> LibberManager: failed to import libberAPI", err);
            });
        }
        
        if (isNode("LibberApply")) {
            console.log("fb_tools -> LibberApply node detected");
            
            // Import the API client
            import("./api/libber.js").then(({ libberAPI }) => {
                const onNodeCreated = nodeType.prototype.onNodeCreated;
                nodeType.prototype.onNodeCreated = function () {
                    if (onNodeCreated) {
                        onNodeCreated.apply(this, arguments);
                    }
                    
                    const widgets = this.widgets || [];
                    const libberNameWidget = widgets.find(w => w.name === "libber_name");
                    const inputTextWidget = widgets.find(w => w.name === "text");
                    
                    // Store last known cursor position
                    let lastCursorStart = 0;
                    let lastCursorEnd = 0;
                    
                    // Track cursor position changes in the input widget
                    if (inputTextWidget && inputTextWidget.inputEl) {
                        const input = inputTextWidget.inputEl;
                        
                        // Update cursor position on selection change
                        const updateCursorPos = () => {
                            lastCursorStart = input.selectionStart || 0;
                            lastCursorEnd = input.selectionEnd || 0;
                        };
                        
                        // Listen for various events that might change cursor position
                        input.addEventListener('click', updateCursorPos);
                        input.addEventListener('keyup', updateCursorPos);
                        input.addEventListener('select', updateCursorPos);
                        input.addEventListener('focus', updateCursorPos);
                        
                        // Initialize cursor position
                        updateCursorPos();
                    }
                    
                    // Function to insert text at cursor position in the input widget
                    const insertAtCursor = (text) => {
                        if (!inputTextWidget || !inputTextWidget.inputEl) return;
                        
                        const input = inputTextWidget.inputEl;
                        
                        // Use stored cursor position instead of current (which may be lost)
                        const startPos = lastCursorStart;
                        const endPos = lastCursorEnd;
                        
                        // Focus the input first
                        input.focus();
                        
                        // Restore cursor position
                        input.setSelectionRange(startPos, endPos);
                        
                        // Use execCommand to insert text - this adds to browser's undo stack
                        // Note: execCommand is deprecated but still widely supported and works with Ctrl+Z
                        const success = document.execCommand('insertText', false, text);
                        
                        if (!success) {
                            // Fallback if execCommand doesn't work (shouldn't happen in modern browsers)
                            const currentValue = inputTextWidget.value || "";
                            const newValue = currentValue.substring(0, startPos) + text + currentValue.substring(endPos);
                            inputTextWidget.value = newValue;
                            input.value = newValue;
                            
                            // Set cursor position after inserted text
                            const newCursorPos = startPos + text.length;
                            input.setSelectionRange(newCursorPos, newCursorPos);
                        }
                        
                        // Update stored cursor position
                        const newCursorPos = input.selectionStart || (startPos + text.length);
                        lastCursorStart = newCursorPos;
                        lastCursorEnd = newCursorPos;
                        
                        // Update the widget value to ensure ComfyUI state is in sync
                        inputTextWidget.value = input.value;
                        
                        console.log(`fbTools -> LibberApply: inserted "${text}" at position ${startPos}`);
                    };
                    
                    // Set minimum node width for JSON viewer
                    this.size[0] = Math.max(this.size[0], 400);
                    
                    // Create a container for the table display
                    const container = document.createElement("div");
                    container.style.cssText = 'width: 100%; min-height: 150px; padding: 8px; background: var(--comfy-input-bg); border: 1px solid var(--border-color); border-radius: 4px; overflow-y: auto; overflow-x: auto; box-sizing: border-box;';
                    
                    // Add the DOM widget for displaying formatted libber data
                    const displayWidget = this.addDOMWidget("libber_data_display", "preview", container, {
                        serialize: false,
                        hideOnZoom: false,
                        getValue() {
                            return container.innerHTML;
                        },
                        setValue(v) {
                            container.innerHTML = v;
                        }
                    });
                    
                    // Store references on node for resize updates
                    this._libberContainer = container;
                    this._libberDisplayWidget = displayWidget;
                    
                    // Store reference to node in widget
                    displayWidget.parentNode = this;
                    
                    // Define how the widget computes its size - return fixed height to prevent infinite growth
                    displayWidget.computeSize = function(width) {
                        const node = this.parentNode;
                        if (!node) return [width, 200];
                        
                        // Find this widget's index
                        const widgetIndex = node.widgets?.indexOf(this) ?? -1;
                        if (widgetIndex === -1) return [width, 200];
                        
                        // Calculate total height used by previous widgets
                        let usedHeight = LiteGraph.NODE_TITLE_HEIGHT || 30;
                        for (let i = 0; i < widgetIndex; i++) {
                            const w = node.widgets[i];
                            if (w.computeSize) {
                                const size = w.computeSize(width);
                                usedHeight += size[1];
                            } else {
                                usedHeight += LiteGraph.NODE_WIDGET_HEIGHT || 20;
                            }
                        }
                        
                        // Calculate remaining height, ensuring minimum and preventing overflow
                        const bottomMargin = 15;
                        const remainingHeight = node.size[1] - usedHeight - bottomMargin;
                        const finalHeight = Math.max(Math.min(remainingHeight, 600), 150); // Cap at 600px, min 150px
                        
                        return [width, finalHeight];
                    };
                    
                    // Function to update container height based on computed widget size
                    const updateContainerHeight = () => {
                        if (!displayWidget.parentNode) return;
                        const widgetSize = displayWidget.computeSize(this.size[0]);
                        const targetHeight = widgetSize[1] - 20; // Subtract padding
                        container.style.height = `${targetHeight}px`;
                    };
                    
                    // Initial height update
                    updateContainerHeight();
                    
                    // Hook into node resize to update container
                    const onResize = this.onResize;
                    this.onResize = function(size) {
                        if (onResize) {
                            onResize.apply(this, arguments);
                        }
                        // Update container height after resize
                        if (this._libberDisplayWidget && this._libberContainer) {
                            const widgetSize = this._libberDisplayWidget.computeSize(size[0]);
                            const targetHeight = Math.max(widgetSize[1] - 20, 130); // Subtract padding, ensure minimum
                            this._libberContainer.style.height = `${targetHeight}px`;
                        }
                        app.graph?.setDirtyCanvas(true);
                    };
                    
                    // Function to update display with table format
                    const updateDisplay = (libberName) => {
                        if (!libberName || libberName === "none") {
                            container.innerHTML = "<div style='padding: 8px; color: var(--descrip-text);'>(no libber selected)</div>";
                            updateContainerHeight();
                            return;
                        }
                        
                        libberAPI.getLibberData(libberName).then(data => {
                            if (data && data.lib_dict && Object.keys(data.lib_dict).length > 0) {
                                const delimiter = data.delimiter || "%";
                                
                                // Create table with clickable rows
                                const rows = Object.entries(data.lib_dict).map(([key, value]) => {
                                    // Escape HTML characters in the value
                                    const escapedValue = String(value)
                                        .replace(/&/g, '&amp;')
                                        .replace(/</g, '&lt;')
                                        .replace(/>/g, '&gt;')
                                        .replace(/"/g, '&quot;')
                                        .replace(/'/g, '&#039;');
                                    
                                    return `<tr data-key="${key}" style='cursor: pointer;' onmouseover='this.style.backgroundColor="var(--comfy-menu-bg)"' onmouseout='this.style.backgroundColor=""'>
                                        <td style='padding: 6px 8px; font-weight: 500; border-bottom: 1px solid var(--border-color); color: var(--fg-color); white-space: nowrap;'>${key}</td>
                                        <td style='padding: 6px 8px; border-bottom: 1px solid var(--border-color); color: var(--fg-color); word-break: break-word;'>${escapedValue}</td>
                                    </tr>`;
                                }).join('');
                                
                                container.innerHTML = `<table style='width: 100%; border-collapse: collapse; font-size: 12px;'>
                                    <thead>
                                        <tr style='background: var(--comfy-menu-bg);'>
                                            <th style='padding: 6px 8px; text-align: left; border-bottom: 2px solid var(--border-color); color: var(--fg-color); font-weight: 600;'>🗝️ Lib</th>
                                            <th style='padding: 6px 8px; text-align: left; border-bottom: 2px solid var(--border-color); color: var(--fg-color); font-weight: 600;'>🪙 Value</th>
                                        </tr>
                                    </thead>
                                    <tbody>${rows}</tbody>
                                </table>`;
                                
                                // Add click handlers to table rows
                                const tableRows = container.querySelectorAll('tbody tr[data-key]');
                                tableRows.forEach(row => {
                                    row.addEventListener('click', () => {
                                        const key = row.getAttribute('data-key');
                                        const wrappedKey = `${delimiter}${key}${delimiter}`;
                                        insertAtCursor(wrappedKey);
                                    });
                                });
                                
                                // Update container height after content is added
                                updateContainerHeight();
                                
                                console.log(`fbTools -> LibberApply: displayed ${Object.keys(data.lib_dict).length} libs`);
                            } else {
                                container.innerHTML = "<div style='padding: 8px; color: var(--descrip-text);'>(empty libber)</div>";
                                updateContainerHeight();
                            }
                        }).catch(err => {
                            container.innerHTML = `<div style='padding: 8px; color: var(--error-text);'>Error: ${err.message}</div>`;
                            updateContainerHeight();
                            console.warn("fbTools -> LibberApply: failed to fetch libber data", err);
                        });
                    };
                    
                    // Fetch and populate available libbers
                    libberAPI.listLibbers().then(data => {
                        if (data && data.libbers && libberNameWidget) {
                            const libbers = data.libbers.length > 0 ? data.libbers : ["none"];
                            libberNameWidget.options.values = libbers;
                            if (!libberNameWidget.value || !libbers.includes(libberNameWidget.value)) {
                                libberNameWidget.value = libbers[0];
                            }
                            console.log("fbTools -> LibberApply: populated libber_name options", libbers);
                            
                            // Update display for initial libber
                            updateDisplay(libberNameWidget.value);
                        }
                    }).catch(err => {
                        console.warn("fbTools -> LibberApply: failed to fetch libbers", err);
                    });
                    
                    // Store updateDisplay function on the node for use in onExecuted
                    this._libberUpdateDisplay = updateDisplay;
                    
                    // Add widget change handler to update display when libber_name changes
                    const onWidgetChanged = this.onWidgetChanged;
                    this.onWidgetChanged = function (widgetName, newValue, oldValue, widgetObject) {
                        const originalReturn = onWidgetChanged?.apply(this, arguments);
                        
                        if (widgetName === "libber_name") {
                            updateDisplay(newValue);
                        }
                        
                        return originalReturn;
                    };
                };
                
                // Handle execution updates to refresh display
                const onExecuted = nodeType.prototype.onExecuted;
                nodeType.prototype.onExecuted = function (message) {
                    if (onExecuted) {
                        onExecuted.apply(this, arguments);
                    }
                    
                    // After execution, refresh the display with current libber_name
                    const widgets = this.widgets || [];
                    const libberNameWidget = widgets.find(w => w.name === "libber_name");
                    
                    // Use the stored updateDisplay function if available
                    if (this._libberUpdateDisplay && libberNameWidget?.value) {
                        console.log("fbTools -> LibberApply: refreshing display after execution");
                        this._libberUpdateDisplay(libberNameWidget.value);
                    }
                    
                    scheduleNodeRefresh(this, app);
                };
            }).catch(err => {
                console.error("fb_tools -> LibberApply: failed to import libberAPI", err);
            });
        }
        
        addMenuHandler(nodeType, function (_, options) {
            options.push({
                content: "Extract Node as JSON",
                callback: handleNodes,
            });
        });
    },
});