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
    [`${EXT_PREFIX}LibberEdit`]: [
        { widget_index: 0, widget_name: "libber_json_in" },
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
        
        if (isNode("LibberEdit")) {
            console.log("fb_tools -> LibberEdit node detected");
            
            const onWidgetChanged = nodeType.prototype.onWidgetChanged;
            nodeType.prototype.onWidgetChanged = function (widgetName, newValue, oldValue, widgetObject) {
                const originalReturn = onWidgetChanged?.apply(this, arguments);
                const keySelectorWidget = this.widgets.find((w) => w.name === "key_selector");
                const libValueWidget = this.widgets.find((w) => w.name === "lib_value");
                const libKeyWidget = this.widgets.find((w) => w.name === "lib_key");
                const currentKey = keySelectorWidget ? keySelectorWidget.value : null;
                const buttonWidget = this.widgets.find((w) => w.type === "button");
                const jsonInWidget = this.widgets.find((w) => w.name === "libber_json_in");
                const libsJson = jsonInWidget && jsonInWidget.value ? JSON.parse(jsonInWidget.value) : {};
                const libs = libsJson.libs || {};
                const keys = Object.keys(libs).sort();
                if (widgetName === "operation") {
                    console.log("fbTools -> LibberEdit: operation changed to", newValue);
                    // Clear key_selector, lib_value, lib_key when operation changes
                    switch (newValue) {
                        case "add":
                            // Disable key_selector combo
                            if (keySelectorWidget) {
                                keySelectorWidget.disabled = true;
                            }
                            // Clear lib_value and lib_key for new entry and enable the inputs
                            if (libValueWidget) {
                                libValueWidget.disabled = false;
                                libValueWidget.value = "";
                            }
                            if (libKeyWidget) {
                                libKeyWidget.disabled = false;
                                libKeyWidget.value = "";
                            }
                            if (buttonWidget) {
                                buttonWidget.disabled = false;
                                buttonWidget.label = "ADD";
                            }
                            break;
                        case "view":
                        case "remove":
                            if (keySelectorWidget) {
                                keySelectorWidget.disabled = false;
                                if (keySelectorWidget.inputEl && keySelectorWidget.inputEl.tagName === "SELECT") {
                                    keySelectorWidget.inputEl.options = keys;
                                    keySelectorWidget.inputEl.value = keys.length ? keys[0] : "";
                                }
                                // Rebuild select options
                                // choose a default value
                                keySelectorWidget.value = keys.length ? keys[0] : "";

                            }
                            if (libKeyWidget) {
                                libKeyWidget.disabled = true;
                            }
                            if (libValueWidget) {
                                libValueWidget.disabled = true;
                                libValueWidget.value = libs?.[currentKey] || "";
                            }
                            if (buttonWidget) {
                                buttonWidget.disabled = false;
                                buttonWidget.label = newValue.toUpperCase();
                            }
                            break;
                        case "update":
                            // Enable key_selector for existing key mode
                            if (keySelectorWidget) {
                                keySelectorWidget.disabled = false;
                            }
                            if (libKeyWidget) {
                                libKeyWidget.disabled = true;
                            }
                            if (libValueWidget) {
                                libValueWidget.disabled = false;
                                libValueWidget.value = libs?.[currentKey] || "";
                            }
                            if (buttonWidget) {
                                buttonWidget.disabled = false;
                                buttonWidget.label = "UPDATE";
                            }
                            break;
                        default:
                            console.warn("fbTools -> LibberEdit: unknown operation", newValue);
                            break;
                    }
                }
                if (widgetName === "key_selector") {
                    console.log("fbTools -> LibberEdit: key_selector changed to", newValue);
                    
                    if (!libs) {
                        console.warn("fbTools -> LibberEdit: failed to parse libber_json_in", err);
                    }
                    
                    if (newValue && libs?.[newValue]) {
                        // Populate lib_value and lib_key when a key is selected
                        if (libValueWidget) {
                            libValueWidget.value = libs[newValue];
                            if (libValueWidget.inputEl) {
                                libValueWidget.inputEl.value = libs[newValue];
                            }
                        }
                        console.log("fbTools -> LibberEdit: populated value for key", newValue);
                    } else if (newValue === "") {
                        // Clear lib_value when empty option selected (for new key mode)
                        if (libValueWidget) {
                            libValueWidget.value = "";
                            if (libValueWidget.inputEl) {
                                libValueWidget.inputEl.value = "";
                            }
                        }
                    }
                }
                return originalReturn;
            };

            // Set up key_selector change handler on node creation
            const onOriginalCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onOriginalCreated) {
                    onOriginalCreated.apply(this, arguments);
                }
                
                // Set up change handler for key_selector to populate lib_value and lib_key
                const widgets = this.widgets || [];
                const keySelectorWidget = widgets.find((w) => w.name === "key_selector");
                const operationWidget = widgets.find((w) => w.name === "operation");
                const buttonLabel = String(operationWidget.value || "Do Stuff!").toUpperCase()
                const libValueWidget = this.widgets.find((w) => w.name === "lib_value");
                const libKeyWidget = this.widgets.find((w) => w.name === "lib_key");
                const currentKey = keySelectorWidget ? keySelectorWidget.value : null;
                const jsonInWidget = this.widgets.find((w) => w.name === "libber_json_in");

                this.addWidget("button", buttonLabel, null, () => {
                    console.log("fbTools -> LibberEdit: Do stuff button clicked! 💅");
                    if (operationWidget) {
                        const operation = operationWidget.value;
                        console.log("Current operation:", operation);
                        const libsJson = jsonInWidget && jsonInWidget.value ? JSON.parse(jsonInWidget.value) : {};
                        const libs = libsJson.libs || {};
                        const newKey = libKeyWidget ? libKeyWidget.value : null;
                        const newValue = libValueWidget ? libValueWidget.value : null;
                        
                        switch (operation) {
                            case "add":
                                if (newKey && newValue && !libs?.[newKey]) {
                                    libs[newKey] = newValue;
                                    libsJson.libs = libs;
                                    jsonInWidget.value = JSON.stringify(libsJson, null, 2);
                                    if (jsonInWidget.inputEl) {
                                        jsonInWidget.inputEl.value = jsonInWidget.value;
                                    }
                                    showToast({ severity: "success", summary: `Added ${newKey} - ${newValue}`, life: 2000 });
                                }
                                else {
                                    showToast({ severity: "warn", summary: `Add operation requires a new key and value`, life: 2000 });
                                }
                                break;
                            case "view":
                                console.log("View operation selected, no action taken");
                                break;
                            case "remove":
                                if (currentKey && libs?.[currentKey]) {
                                    delete libs[currentKey];
                                    libsJson.libs = libs;
                                    jsonInWidget.value = JSON.stringify(libsJson, null, 2);
                                    if (jsonInWidget.inputEl) {
                                        jsonInWidget.inputEl.value = jsonInWidget.value;
                                    }
                                    showToast({ severity: "success", summary: `Removed ${currentKey}`, life: 2000 });
                                } else {
                                    showToast({ severity: "warn", summary: `Remove operation requires an existing key`, life: 2000 });
                                }
                                break;
                            case "update":
                                if (currentKey && libs?.[currentKey] && newValue) {
                                    libs[currentKey] = newValue;
                                    libsJson.libs = libs;
                                    jsonInWidget.value = JSON.stringify(libsJson, null, 2);
                                    if (jsonInWidget.inputEl) {
                                        jsonInWidget.inputEl.value = jsonInWidget.value;
                                    }
                                    showToast({ severity: "success", summary: `Updated ${currentKey} - ${newValue}`, life: 2000 });
                                } else {
                                    showToast({ severity: "warn", summary: `Update operation requires an existing key and new value`, life: 2000 });
                                }
                                break;
                            default:
                                showToast({ severity: "warn", summary: `Unknown operation: ${operation}`, life: 2000 });
                                break;
                        }
                    }
                });

                if (keySelectorWidget && !keySelectorWidget._libberHandlerInstalled) {
                    const originalCallback = keySelectorWidget.callback;
                    keySelectorWidget.callback = function(value) {
                        if (originalCallback) {
                            originalCallback.call(this, value);
                        }
                        
                        // Get the current key-value map from libber_json_in
                        const libberJsonWidget = widgets.find((w) => w.name === "libber_json_in");
                        let keyValueMap = {};
                        
                        if (libberJsonWidget && libberJsonWidget.value) {
                            try {
                                const libberData = JSON.parse(libberJsonWidget.value);
                                const libs = libberData.libs || {};
                                keyValueMap = libs;
                            } catch (err) {
                                console.warn("fbTools -> LibberEdit: failed to parse libber_json_in", err);
                            }
                        }
                        
                        const libValueWidget = widgets.find((w) => w.name === "lib_value");
                        const libKeyWidget = widgets.find((w) => w.name === "lib_key");
                        
                        if (value && keyValueMap[value]) {
                            // Populate lib_value and lib_key when a key is selected
                            if (libValueWidget) {
                                libValueWidget.value = keyValueMap[value];
                                if (libValueWidget.inputEl) {
                                    libValueWidget.inputEl.value = keyValueMap[value];
                                }
                            }
                            if (libKeyWidget) {
                                libKeyWidget.value = value;
                                if (libKeyWidget.inputEl) {
                                    libKeyWidget.inputEl.value = value;
                                }
                            }
                            console.log("fbTools -> LibberEdit: populated value for key", value);
                        } else if (value === "") {
                            // Clear lib_value when empty option selected (for new key mode)
                            if (libValueWidget) {
                                libValueWidget.value = "";
                                if (libValueWidget.inputEl) {
                                    libValueWidget.inputEl.value = "";
                                }
                            }
                        }
                    };
                    keySelectorWidget._libberHandlerInstalled = true;
                }
            };
            
            const onOriginalExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                if (onOriginalExecuted) {
                    onOriginalExecuted.apply(this, arguments);
                }

                // LibberEdit returns: text[0]=libber_json, text[1]=key_options_json, text[2]=key_value_map_json
                const textArray = message?.text;
                
                // Update libber_json_in widget
                updateNodeInputs(this, textArray, nodeData.name);
                
                if (!textArray || !textArray[1]) {
                    scheduleNodeRefresh(this, app);
                    return;
                }
                
                try {
                    const keyOptions = JSON.parse(textArray[1]);
                    
                    // Update key_selector combo options
                    const widgets = this.widgets || [];
                    const keySelectorWidget = widgets.find((w) => w.name === "key_selector");
                    if (keySelectorWidget && Array.isArray(keyOptions)) {
                        const currentOptions = (keySelectorWidget.options && keySelectorWidget.options.values) || keySelectorWidget.options || [];
                        const sameOptions =
                            Array.isArray(currentOptions) &&
                            currentOptions.length === keyOptions.length &&
                            currentOptions.every((val, idx) => val === keyOptions[idx]);
                        
                        if (!sameOptions) {
                            if (keySelectorWidget.options && typeof keySelectorWidget.options === "object") {
                                keySelectorWidget.options.values = keyOptions;
                            } else {
                                keySelectorWidget.options = { values: keyOptions };
                            }
                            keySelectorWidget.options_values = keyOptions;
                            
                            if (keySelectorWidget.inputEl && keySelectorWidget.inputEl.tagName === "SELECT") {
                                keySelectorWidget.inputEl.innerHTML = "";
                                keyOptions.forEach((opt) => {
                                    const optionEl = document.createElement("option");
                                    optionEl.value = opt;
                                    optionEl.textContent = opt || "(new key)";
                                    keySelectorWidget.inputEl.appendChild(optionEl);
                                });
                            }
                            
                            console.log("fbTools -> LibberEdit: updated key_selector with", keyOptions.length - 1, "keys");
                        }
                    }
                    
                } catch (err) {
                    console.warn("fbTools -> LibberEdit: failed to parse key options", err);
                }
                
                scheduleNodeRefresh(this, app);
            };
        }
        
        addMenuHandler(nodeType, function (_, options) {
            options.push({
                content: "Extract Node as JSON",
                callback: handleNodes,
            });
        });
    },
});