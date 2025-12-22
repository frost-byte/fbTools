/**
 * Scene-related node extensions
 */

import { libberAPI } from "../api/libber.js";
import { sceneAPI } from "../api/scene.js";
import { debugLog, DEBUG_FLAGS } from "../utils/debug_config.js";

/**
 * Show toast notification
 */
function showToast(options) {
    app.extensionManager.toast.add(options);
}

/**
 * Update a widget's value from message.text array
 */
function updateWidgetFromText(node, textArray, index, widgetName, widgetType = "text", logPrefix = "fbTools") {
    if (textArray && textArray[index]) {
        const widget = node.widgets.find((w) => w.name === widgetName);
        if (widget) {
            if (widgetType === "text") {
                widget.value = textArray[index];
                if (widget.inputEl) {
                    widget.inputEl.value = textArray[index];
                }
                console.log(`${logPrefix}: ${widgetName} updated from text[${index}]`);
            } else if (widgetType === "combo") {
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

/**
 * Bulk update widgets for a node based on widget map
 */
function updateNodeInputs(node, textArray, entries, logPrefix) {
    if (!entries || !entries.length) return;
    entries.forEach(({ widget_index, widget_name, widget_type }) => {
        updateWidgetFromText(node, textArray, widget_index, widget_name, widget_type, logPrefix);
    });
}

/**
 * Schedule node refresh after widget updates
 */
function scheduleNodeRefresh(node, app) {
    requestAnimationFrame(() => {
        const sz = node.computeSize();
        if (sz[0] < node.size[0]) sz[0] = node.size[0];
        if (sz[1] < node.size[1]) sz[1] = node.size[1];
        node.onResize?.(sz);
        app.graph.setDirtyCanvas(true, false);
    });
}

/**
 * Setup SceneSelect node extensions
 */
export function setupSceneSelect(nodeType, nodeData, app) {
    console.log("fb_tools -> SceneSelect node detected");
    
    // Debug connection hooks
    const onConnectOutput = nodeType.prototype.onConnectOutput;
    nodeType.prototype.onConnectOutput = function(outputIndex, inputType, inputSlot, inputNode, inputIndex) {
        debugLog(DEBUG_FLAGS.CONNECTIONS, "🔌 SceneSelect.onConnectOutput called:");
        debugLog(DEBUG_FLAGS.CONNECTIONS, "  outputIndex:", outputIndex);
        debugLog(DEBUG_FLAGS.CONNECTIONS, "  inputType:", inputType);
        debugLog(DEBUG_FLAGS.CONNECTIONS, "  inputSlot:", inputSlot);
        debugLog(DEBUG_FLAGS.CONNECTIONS, "  inputNode:", inputNode?.type);
        debugLog(DEBUG_FLAGS.CONNECTIONS, "  inputIndex:", inputIndex);
        debugLog(DEBUG_FLAGS.CONNECTIONS, "  this.outputs:", this.outputs);
        debugLog(DEBUG_FLAGS.CONNECTIONS, "  Output being connected:", this.outputs[outputIndex]);
        
        // Call original hook if it exists
        const result = onConnectOutput?.apply(this, arguments);
        debugLog(DEBUG_FLAGS.CONNECTIONS, "  Original result:", result);
        
        // Explicitly return true to allow connection (undefined can be ambiguous)
        const finalResult = result !== false ? true : false;
        debugLog(DEBUG_FLAGS.CONNECTIONS, "  Final result:", finalResult);
        return finalResult;
    };
    
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function() {
        const result = onNodeCreated?.apply(this, arguments);
        
        // Store current prompts data
        let currentPrompts = [];
        let currentSceneDir = null;
        
        // Hook into widget changes to track scene_dir
        const updateSceneDir = () => {
            const scenesDir = this.widgets?.find(w => w.name === "scenes_dir")?.value;
            const selectedScene = this.widgets?.find(w => w.name === "selected_scene")?.value;
            
            if (scenesDir && selectedScene) {
                currentSceneDir = `${scenesDir}/${selectedScene}`;
            }
        };
        
        // Monitor widget changes
        const sceneDirWidget = this.widgets?.find(w => w.name === "scenes_dir");
        const selectedSceneWidget = this.widgets?.find(w => w.name === "selected_scene");
        
        if (sceneDirWidget) {
            const originalCallback = sceneDirWidget.callback;
            sceneDirWidget.callback = function(...args) {
                if (originalCallback) originalCallback.apply(this, args);
                updateSceneDir();
            };
        }
        
        if (selectedSceneWidget) {
            const originalCallback = selectedSceneWidget.callback;
            selectedSceneWidget.callback = function(...args) {
                if (originalCallback) originalCallback.apply(this, args);
                updateSceneDir();
            };
        }
        
        // Initial update
        updateSceneDir();
        
        // Add DOM widget for prompt display
        const displayWidget = this.addDOMWidget(
            "prompts_display",
            "div",
            document.createElement("div"),
            {
                getValue: () => "",  // Don't serialize
                setValue: (v) => {},  // No-op
                serialize: false,
            }
        );
        
        displayWidget.computeSize = () => [this.size[0], 250];
        
        // Track which tab is active
        let activeDisplayTab = "prompts";  // "prompts" or "compositions"
        let currentCompositions = [];
        
        /**
         * Render prompt display table
         */
        const renderPromptDisplay = () => {
            const container = displayWidget.element;
            container.innerHTML = "";
            container.style.cssText = `
                width: 100%;
                background: var(--bg-color);
                border: 1px solid var(--border-color);
                border-radius: 4px;
                padding: 8px;
                box-sizing: border-box;
                overflow: auto;
                max-height: 250px;
            `;
            
            // Title, tabs, and refresh button
            const header = document.createElement("div");
            header.style.cssText = `
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 8px;
                gap: 8px;
            `;
            
            const leftSection = document.createElement("div");
            leftSection.style.cssText = `display: flex; align-items: center; gap: 8px;`;
            
            const title = document.createElement("div");
            title.textContent = "Scene Data";
            title.style.cssText = `
                font-weight: bold;
                color: var(--fg-color);
            `;
            
            // Tab buttons
            const tabContainer = document.createElement("div");
            tabContainer.style.cssText = `display: flex; gap: 4px;`;
            
            const promptsTab = document.createElement("button");
            promptsTab.textContent = "📝 Prompts";
            promptsTab.style.cssText = `
                padding: 4px 12px;
                cursor: pointer;
                background: ${activeDisplayTab === "prompts" ? "var(--comfy-menu-bg)" : "var(--comfy-input-bg)"};
                border: 1px solid var(--border-color);
                border-radius: 4px;
                color: var(--fg-color);
                font-weight: ${activeDisplayTab === "prompts" ? "bold" : "normal"};
            `;
            promptsTab.onclick = () => {
                activeDisplayTab = "prompts";
                renderPromptDisplay();
            };
            
            const compsTab = document.createElement("button");
            compsTab.textContent = "🎯 Compositions";
            compsTab.style.cssText = `
                padding: 4px 12px;
                cursor: pointer;
                background: ${activeDisplayTab === "compositions" ? "var(--comfy-menu-bg)" : "var(--comfy-input-bg)"};
                border: 1px solid var(--border-color);
                border-radius: 4px;
                color: var(--fg-color);
                font-weight: ${activeDisplayTab === "compositions" ? "bold" : "normal"};
            `;
            compsTab.onclick = () => {
                activeDisplayTab = "compositions";
                renderPromptDisplay();
            };
            
            tabContainer.appendChild(promptsTab);
            tabContainer.appendChild(compsTab);
            
            leftSection.appendChild(title);
            leftSection.appendChild(tabContainer);
            
            const refreshBtn = document.createElement("button");
            refreshBtn.textContent = "🔄 Refresh";
            refreshBtn.style.cssText = `
                padding: 4px 8px;
                cursor: pointer;
                background: var(--comfy-input-bg);
                border: 1px solid var(--border-color);
                border-radius: 4px;
                color: var(--fg-color);
            `;
            refreshBtn.onclick = async () => {
                if (!currentSceneDir) {
                    showToast({
                        severity: "warn",
                        summary: "No Scene Selected",
                        detail: "Please select a scene first",
                        life: 3000,
                    });
                    return;
                }
                
                try {
                    const data = await sceneAPI.getScenePrompts(currentSceneDir);
                    currentPrompts = data.prompts || [];
                    
                    // Load compositions from prompts.json
                    if (data.compositions) {
                        currentCompositions = Object.entries(data.compositions).map(([name, keys]) => ({
                            name,
                            prompt_keys: keys
                        }));
                    } else {
                        currentCompositions = [];
                    }
                    
                    renderPromptDisplay();
                    showToast({
                        severity: "info",
                        summary: "Data Refreshed",
                        detail: `Loaded ${currentPrompts.length} prompts and ${currentCompositions.length} compositions`,
                        life: 2000,
                    });
                } catch (error) {
                    console.error("Failed to refresh scene data:", error);
                    showToast({
                        severity: "error",
                        summary: "Refresh Failed",
                        detail: error.message,
                        life: 5000,
                    });
                }
            };
            
            header.appendChild(leftSection);
            header.appendChild(refreshBtn);
            container.appendChild(header);
            
            // Render appropriate table based on active tab
            if (activeDisplayTab === "prompts") {
                renderPromptsTable(container);
            } else {
                renderCompositionsTable(container);
            }
        };
        
        /**
         * Render prompts table
         */
        const renderPromptsTable = (container) => {
            if (!currentPrompts || currentPrompts.length === 0) {
                const emptyMsg = document.createElement("div");
                emptyMsg.textContent = currentSceneDir ? "No prompts found" : "Select a scene to view prompts";
                emptyMsg.style.cssText = `
                    color: var(--fg-color);
                    opacity: 0.6;
                    font-style: italic;
                    text-align: center;
                    padding: 20px;
                `;
                container.appendChild(emptyMsg);
                return;
            }
            
            const table = document.createElement("table");
            table.style.cssText = `
                width: 100%;
                border-collapse: collapse;
                font-size: 12px;
            `;
            
            // Table header
            const thead = document.createElement("thead");
            thead.innerHTML = `
                <tr>
                    <th style="text-align: left; padding: 4px; border-bottom: 1px solid var(--border-color); color: var(--fg-color);">Key</th>
                    <th style="text-align: left; padding: 4px; border-bottom: 1px solid var(--border-color); color: var(--fg-color);">Value</th>
                    <th style="text-align: left; padding: 4px; border-bottom: 1px solid var(--border-color); color: var(--fg-color);">Category</th>
                    <th style="text-align: left; padding: 4px; border-bottom: 1px solid var(--border-color); color: var(--fg-color);">Processing Type</th>
                </tr>
            `;
            table.appendChild(thead);
            
            // Table body
            const tbody = document.createElement("tbody");
            currentPrompts.forEach(prompt => {
                const row = document.createElement("tr");
                row.innerHTML = `
                    <td style="padding: 4px; border-bottom: 1px solid var(--border-color); color: var(--fg-color);">${prompt.key || ""}</td>
                    <td style="padding: 4px; border-bottom: 1px solid var(--border-color); color: var(--fg-color); max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${(prompt.value || "").replace(/"/g, '&quot;')}">${prompt.value || ""}</td>
                    <td style="padding: 4px; border-bottom: 1px solid var(--border-color); color: var(--fg-color);">${prompt.category || ""}</td>
                    <td style="padding: 4px; border-bottom: 1px solid var(--border-color); color: var(--fg-color);">${prompt.processing_type || ""}</td>
                `;
                tbody.appendChild(row);
            });
            table.appendChild(tbody);
            
            container.appendChild(table);
        };
        
        /**
         * Render compositions table
         */
        const renderCompositionsTable = (container) => {
            if (!currentCompositions || currentCompositions.length === 0) {
                const emptyMsg = document.createElement("div");
                emptyMsg.textContent = currentSceneDir ? "No compositions found" : "Select a scene to view compositions";
                emptyMsg.style.cssText = `
                    color: var(--fg-color);
                    opacity: 0.6;
                    font-style: italic;
                    text-align: center;
                    padding: 20px;
                `;
                container.appendChild(emptyMsg);
                return;
            }
            
            const table = document.createElement("table");
            table.style.cssText = `
                width: 100%;
                border-collapse: collapse;
                font-size: 12px;
            `;
            
            // Table header
            const thead = document.createElement("thead");
            thead.innerHTML = `
                <tr>
                    <th style="text-align: left; padding: 4px; border-bottom: 1px solid var(--border-color); color: var(--fg-color);">Name</th>
                    <th style="text-align: left; padding: 4px; border-bottom: 1px solid var(--border-color); color: var(--fg-color);">Prompt Keys</th>
                </tr>
            `;
            table.appendChild(thead);
            
            // Table body
            const tbody = document.createElement("tbody");
            currentCompositions.forEach(comp => {
                const row = document.createElement("tr");
                const keysStr = (comp.prompt_keys || []).join(", ");
                row.innerHTML = `
                    <td style="padding: 4px; border-bottom: 1px solid var(--border-color); color: var(--fg-color);">${comp.name || ""}</td>
                    <td style="padding: 4px; border-bottom: 1px solid var(--border-color); color: var(--fg-color);">${keysStr}</td>
                `;
                tbody.appendChild(row);
            });
            table.appendChild(tbody);
            
            container.appendChild(table);
        };
        
        // Initial render
        renderPromptDisplay();
        
        // Hook into execution to update prompts display
        const onExecuted = this.onExecuted;
        this.onExecuted = function(message) {
            if (onExecuted) {
                onExecuted.apply(this, arguments);
            }
            
            // Update current scene dir and auto-load prompts
            updateSceneDir();
            
            if (currentSceneDir) {
                // Auto-load prompts after execution
                sceneAPI.getScenePrompts(currentSceneDir)
                    .then(data => {
                        currentPrompts = data.prompts || [];
                        renderPromptDisplay();
                    })
                    .catch(error => {
                        console.error("Failed to load prompts after execution:", error);
                    });
            }
            
            scheduleNodeRefresh(this, app);
        };
        
        return result;
    };
}

/**
 * Setup ScenePromptManager node extensions
 */
export function setupScenePromptManager(nodeType, nodeData, app) {
    console.log("fb_tools -> ScenePromptManager node detected");
    
    // Debug connection hooks
    const onConnectInput = nodeType.prototype.onConnectInput;
    nodeType.prototype.onConnectInput = function(inputIndex, outputType, outputSlot, outputNode, outputIndex) {
        debugLog(DEBUG_FLAGS.CONNECTIONS, "🔌 ScenePromptManager.onConnectInput called:");
        debugLog(DEBUG_FLAGS.CONNECTIONS, "  inputIndex:", inputIndex);
        debugLog(DEBUG_FLAGS.CONNECTIONS, "  outputType:", outputType);
        debugLog(DEBUG_FLAGS.CONNECTIONS, "  outputSlot:", outputSlot);
        debugLog(DEBUG_FLAGS.CONNECTIONS, "  outputNode:", outputNode?.type);
        debugLog(DEBUG_FLAGS.CONNECTIONS, "  outputIndex:", outputIndex);
        debugLog(DEBUG_FLAGS.CONNECTIONS, "  this.inputs:", this.inputs);
        debugLog(DEBUG_FLAGS.CONNECTIONS, "  Expected input type:", this.inputs[inputIndex]);
        
        // Call original hook if it exists
        const result = onConnectInput?.apply(this, arguments);
        debugLog(DEBUG_FLAGS.CONNECTIONS, "  Original result:", result);
        
        // Explicitly return true to allow connection (undefined can be ambiguous)
        const finalResult = result !== false ? true : false;
        debugLog(DEBUG_FLAGS.CONNECTIONS, "  Final result:", finalResult);
        return finalResult;
    };
    
    const onConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function(type, index, connected, link_info, ioSlot) {
        debugLog(DEBUG_FLAGS.CONNECTION_CHANGES, "🔗 ScenePromptManager.onConnectionsChange called:");
        debugLog(DEBUG_FLAGS.CONNECTION_CHANGES, "  type:", type, "(1=input, 2=output)");
        debugLog(DEBUG_FLAGS.CONNECTION_CHANGES, "  index:", index);
        debugLog(DEBUG_FLAGS.CONNECTION_CHANGES, "  connected:", connected);
        debugLog(DEBUG_FLAGS.CONNECTION_CHANGES, "  link_info:", link_info);
        debugLog(DEBUG_FLAGS.CONNECTION_CHANGES, "  ioSlot:", ioSlot);
        
        const result = onConnectionsChange?.apply(this, arguments);
        return result;
    };
    
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        if (onNodeCreated) {
            onNodeCreated.apply(this, arguments);
        }
        
        const widgets = this.widgets || [];
        const collectionJsonWidget = widgets.find(w => w.name === "collection_json");
        
        // Set minimum node size
        this.size[0] = Math.max(this.size[0], 600);
        this.size[1] = Math.max(this.size[1], 450);
        
        // Create container for editable table
        const container = document.createElement("div");
        container.style.cssText = 'width: 100%; min-height: 250px; padding: 8px; background: var(--comfy-input-bg); border: 1px solid var(--border-color); border-radius: 4px; overflow-y: auto; overflow-x: auto; box-sizing: border-box;';
        
        // Add the DOM widget for the table
        const displayWidget = this.addDOMWidget("prompt_table", "preview", container, {
            serialize: false,
            hideOnZoom: false,
            getValue() {
                // Don't serialize HTML - it will be regenerated
                return "";
            },
            setValue(v) {
                // Don't restore HTML - wait for proper initialization
                // The container will be populated by renderTable when data is available
            }
        });
        
        // Store references
        this._promptContainer = container;
        this._promptDisplayWidget = displayWidget;
        displayWidget.parentNode = this;
        
        // Compute widget size
        displayWidget.computeSize = function(width) {
            const node = this.parentNode;
            if (!node) return [width, 300];
            
            const widgetIndex = node.widgets?.indexOf(this) ?? -1;
            if (widgetIndex === -1) return [width, 300];
            
            let usedHeight = LiteGraph.NODE_TITLE_HEIGHT || 30;
            for (let i = 0; i < widgetIndex; i++) {
                const w = node.widgets[i];
                if (w.computeSize) {
                    usedHeight += w.computeSize(width)[1];
                } else {
                    usedHeight += LiteGraph.NODE_WIDGET_HEIGHT || 20;
                }
            }
            
            const bottomMargin = 15;
            const remainingHeight = node.size[1] - usedHeight - bottomMargin;
            const finalHeight = Math.max(Math.min(remainingHeight, 700), 250);
            
            return [width, finalHeight];
        };
        
        // Update container height
        const updateContainerHeight = () => {
            if (!displayWidget.parentNode) return;
            const widgetSize = displayWidget.computeSize(this.size[0]);
            const targetHeight = Math.max(widgetSize[1] - 20, 230);
            container.style.height = `${targetHeight}px`;
        };
        
        updateContainerHeight();
        
        // Hook into resize
        const onResize = this.onResize;
        this.onResize = function(size) {
            if (onResize) {
                onResize.apply(this, arguments);
            }
            if (this._promptDisplayWidget && this._promptContainer) {
                const widgetSize = this._promptDisplayWidget.computeSize(size[0]);
                const targetHeight = Math.max(widgetSize[1] - 20, 230);
                this._promptContainer.style.height = `${targetHeight}px`;
            }
            app.graph?.setDirtyCanvas(true);
        };
        
        // Store current prompts data
        let currentPromptsData = [];
        let currentCompositionsData = [];
        let currentPromptDict = {};
        let availableLibbers = ["none"];
        let activeTab = "define";  // "define", "compose", or "view"
        let currentSceneDir = null;  // Track current scene directory for API calls
        const node = this;  // Store node reference for closures
        
        // Function to update scene directory from widgets and reload prompts
        const updateSceneDir = async (reloadPrompts = false) => {
            const scenesDir = node.widgets?.find(w => w.name === "scenes_dir")?.value;
            const sceneName = node.widgets?.find(w => w.name === "scene_name")?.value;
            
            if (scenesDir && sceneName) {
                const newSceneDir = `${scenesDir}/${sceneName}`;
                const sceneChanged = currentSceneDir !== newSceneDir;
                currentSceneDir = newSceneDir;
                console.log("fb_tools -> ScenePromptManager: Updated scene dir:", currentSceneDir, "(changed:", sceneChanged, ")");
                
                // Reload prompts if scene changed or explicitly requested
                if ((sceneChanged || reloadPrompts) && currentSceneDir) {
                    try {
                        console.log("fb_tools -> ScenePromptManager: Loading prompts from", currentSceneDir);
                        const data = await sceneAPI.getScenePrompts(currentSceneDir);
                        
                        // Handle prompts - could be array or object
                        let promptsList;
                        if (Array.isArray(data.prompts)) {
                            // Already an array of {key, value, processing_type, libber_name, category}
                            promptsList = data.prompts;
                        } else {
                            // Object format: {key: {value, processing_type, ...}}
                            promptsList = Object.entries(data.prompts || {}).map(([key, metadata]) => ({
                                key,
                                value: metadata.value || "",
                                processing_type: metadata.processing_type || "raw",
                                libber_name: metadata.libber_name || null,
                                category: metadata.category || null
                            }));
                        }
                        
                        // Handle compositions - could be array or object
                        let compositionsList;
                        if (Array.isArray(data.compositions)) {
                            compositionsList = data.compositions;
                        } else {
                            compositionsList = Object.entries(data.compositions || {}).map(([name, prompt_keys]) => ({
                                name,
                                prompt_keys
                            }));
                        }
                        
                        // Convert prompts array to dict for compatibility
                        const promptDict = {};
                        promptsList.forEach(p => {
                            if (p.key) {
                                promptDict[p.key] = {
                                    value: p.value || "",
                                    processing_type: p.processing_type || "raw",
                                    libber_name: p.libber_name || null,
                                    category: p.category || null
                                };
                            }
                        });
                        
                        console.log("fb_tools -> ScenePromptManager: Loaded", promptsList.length, "prompts and", compositionsList.length, "compositions");
                        renderTable(promptsList, data.libbers || ["none"], compositionsList, promptDict);
                    } catch (error) {
                        console.error("fb_tools -> ScenePromptManager: Failed to load prompts:", error);
                        showToast({ severity: "error", summary: "Failed to load prompts", detail: error.message, life: 3000 });
                    }
                }
            } else {
                currentSceneDir = null;
                console.log("fb_tools -> ScenePromptManager: No scene directory (missing scenesDir or sceneName)");
            }
        };
        
        // Monitor widget changes to track scene selection
        const scenesDirWidget = this.widgets?.find(w => w.name === "scenes_dir");
        const sceneNameWidget = this.widgets?.find(w => w.name === "scene_name");
        
        if (scenesDirWidget) {
            const originalCallback = scenesDirWidget.callback;
            scenesDirWidget.callback = function(...args) {
                if (originalCallback) originalCallback.apply(this, args);
                updateSceneDir(true);  // Reload prompts when scenes_dir changes
            };
        }
        
        if (sceneNameWidget) {
            const originalCallback = sceneNameWidget.callback;
            sceneNameWidget.callback = function(...args) {
                if (originalCallback) originalCallback.apply(this, args);
                updateSceneDir(true);  // Reload prompts when scene_name changes
            };
        }
        
        // Delay initial update to allow widgets to initialize with their default values
        setTimeout(() => {
            console.log("fb_tools -> ScenePromptManager: Initial load - widget values:", {
                scenes_dir: scenesDirWidget?.value,
                scene_name: sceneNameWidget?.value
            });
            updateSceneDir(true);
        }, 100);
        
        // Function to render tabs
        const renderTabs = () => {
            return `<div style="display: flex; gap: 4px; margin-bottom: 8px; border-bottom: 2px solid var(--border-color);">
                <button class="tab-btn" data-tab="define" style="padding: 8px 16px; background: ${activeTab === 'define' ? 'var(--comfy-menu-bg)' : 'transparent'}; color: var(--fg-color); border: none; border-bottom: 2px solid ${activeTab === 'define' ? 'var(--fg-color)' : 'transparent'}; cursor: pointer; font-weight: ${activeTab === 'define' ? '600' : '400'};">📝 Define</button>
                <button class="tab-btn" data-tab="compose" style="padding: 8px 16px; background: ${activeTab === 'compose' ? 'var(--comfy-menu-bg)' : 'transparent'}; color: var(--fg-color); border: none; border-bottom: 2px solid ${activeTab === 'compose' ? 'var(--fg-color)' : 'transparent'}; cursor: pointer; font-weight: ${activeTab === 'compose' ? '600' : '400'};">🎨 Compose</button>
                <button class="tab-btn" data-tab="view" style="padding: 8px 16px; background: ${activeTab === 'view' ? 'var(--comfy-menu-bg)' : 'transparent'}; color: var(--fg-color); border: none; border-bottom: 2px solid ${activeTab === 'view' ? 'var(--fg-color)' : 'transparent'}; cursor: pointer; font-weight: ${activeTab === 'view' ? '600' : '400'};">👁️ View</button>
            </div>`;
        };
        
        // Function to render the editable table
        const renderTable = (promptsList, libbersList, compositionsList, promptDict) => {
            currentPromptsData = promptsList || [];
            currentCompositionsData = compositionsList || [];
            currentPromptDict = promptDict || {};
            availableLibbers = libbersList || ["none"];
            
            // Ensure "none" is always first
            if (!availableLibbers.includes("none")) {
                availableLibbers.unshift("none");
            }
            
            // Render tabs
            const tabsHTML = renderTabs();
            
            // Render content based on active tab
            let contentHTML = '';
            if (activeTab === 'define') {
                contentHTML = renderDefineTab();
            } else if (activeTab === 'compose') {
                contentHTML = renderComposeTab();
            } else if (activeTab === 'view') {
                contentHTML = renderViewTab();
            }
            
            container.innerHTML = tabsHTML + contentHTML;
            
            // Attach event handlers
            attachEventHandlers();
            updateContainerHeight();
        };
        
        // Render Define tab content (returns HTML string)
        const renderDefineTab = () => {
            // Help text
            const helpText = `<div style="margin-bottom: 8px; padding: 6px 8px; background: var(--comfy-menu-bg); border: 1px solid var(--border-color); border-radius: 4px; font-size: 11px; color: var(--descrip-text);">
                <strong style="color: var(--fg-color);">💡 Quick Guide:</strong> 
                Create reusable prompt components. <strong>Key</strong>=unique identifier, <strong>Value</strong>=prompt text, <strong>Type</strong>=raw (use as-is) or libber (with substitution), <strong>Libber</strong>=which libber to use for substitution, <strong>Category</strong>=group/organize prompts. Use PromptComposer to combine prompts into outputs.
            </div>`;
            
            // Action buttons - sticky at top
            const actionButtons = `<div style="margin-bottom: 8px; padding-bottom: 8px; display: flex; gap: 8px; flex-wrap: wrap; align-items: center; border-bottom: 2px solid var(--border-color); background: var(--comfy-input-bg); position: sticky; top: 0; z-index: 10;">
                <button class="apply-prompts-btn" title="Apply all changes to collection_json" style="padding: 4px 12px; background: var(--comfy-menu-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 4px; cursor: pointer;">✓ Apply Changes</button>
                <span style="padding: 4px 8px; color: var(--descrip-text); font-size: 11px;" class="prompt-count">${currentPromptsData.length} prompts</span>
            </div>`;
            
            // Create rows for existing prompts
            const existingRows = currentPromptsData.map((prompt, idx) => {
                const escapedKey = String(prompt.key || '').replace(/"/g, '&quot;');
                const escapedValue = String(prompt.value || '').replace(/"/g, '&quot;');
                const escapedCategory = String(prompt.category || '').replace(/"/g, '&quot;');
                const processingType = prompt.processing_type || 'raw';
                const libberName = prompt.libber_name || 'none';
                
                // Build libber dropdown options
                const libberOptions = availableLibbers.map(lib => {
                    const selected = lib === libberName ? 'selected' : '';
                    return `<option value="${lib}" ${selected}>${lib}</option>`;
                }).join('');
                
                return `<tr data-idx="${idx}" data-key="${escapedKey}" style="vertical-align: top;">
                    <td style="width: 15%; padding: 4px;"><input type="text" class="prompt-key-input" value="${escapedKey}" title="Unique identifier for this prompt (e.g., 'char1', 'quality')" placeholder="prompt_key" style="width: 100%; padding: 6px; background: var(--comfy-input-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 3px; font-family: monospace; font-size: 11px; box-sizing: border-box;" /></td>
                    <td style="width: 35%; padding: 4px;"><textarea class="prompt-value-input" title="The actual prompt text" placeholder="beautiful woman, detailed" style="width: 100%; min-height: 50px; padding: 6px; background: var(--comfy-input-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 3px; resize: vertical; font-size: 11px; box-sizing: border-box; font-family: inherit;">${escapedValue}</textarea></td>
                    <td style="width: 15%; padding: 4px;">
                        <select class="prompt-type-select" title="Raw: use text as-is | Libber: substitute placeholders like %var%" style="width: 100%; padding: 6px; background: var(--comfy-input-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 3px; font-size: 11px; box-sizing: border-box;">
                            <option value="raw" ${processingType === 'raw' ? 'selected' : ''}>📝 raw</option>
                            <option value="libber" ${processingType === 'libber' ? 'selected' : ''}>🔄 libber</option>
                        </select>
                    </td>
                    <td style="width: 15%; padding: 4px;">
                        <select class="prompt-libber-select" title="Select which libber to use for substitution" style="width: 100%; padding: 6px; background: var(--comfy-input-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 3px; font-size: 11px; box-sizing: border-box;" ${processingType !== 'libber' ? 'disabled' : ''}>
                            ${libberOptions}
                        </select>
                    </td>
                    <td style="width: 12%; padding: 4px;"><input type="text" class="prompt-category-input" value="${escapedCategory}" title="Optional: group prompts (e.g., 'character', 'scene', 'quality')" placeholder="character" style="width: 100%; padding: 6px; background: var(--comfy-input-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 3px; font-size: 11px; box-sizing: border-box;" /></td>
                    <td style="white-space: nowrap; text-align: center; vertical-align: top; width: 8%; padding: 4px;">
                        <button class="remove-prompt-btn" title="Remove this prompt" style="padding: 8px 10px; background: var(--comfy-menu-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 3px; cursor: pointer; font-size: 14px; min-height: 34px;">➖</button>
                    </td>
                </tr>`;
            }).join('');
            
            // Add row for new prompt
            const newRowLibberOptions = availableLibbers.map(lib => {
                const selected = lib === 'none' ? 'selected' : '';
                return `<option value="${lib}" ${selected}>${lib}</option>`;
            }).join('');
            
            const newRow = `<tr class="new-prompt-row" style="vertical-align: top; border-top: 2px solid var(--border-color);">
                <td style="padding: 4px;"><input type="text" placeholder="prompt_key" class="prompt-key-input" title="Unique identifier (required)" style="width: 100%; padding: 6px; background: var(--comfy-input-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 3px; font-family: monospace; font-size: 11px; box-sizing: border-box;" /></td>
                <td style="padding: 4px;"><textarea placeholder="beautiful woman, detailed..." class="prompt-value-input" title="The prompt text" style="width: 100%; min-height: 50px; padding: 6px; background: var(--comfy-input-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 3px; resize: vertical; font-size: 11px; box-sizing: border-box; font-family: inherit;"></textarea></td>
                <td style="padding: 4px;">
                    <select class="prompt-type-select" title="Processing type" style="width: 100%; padding: 6px; background: var(--comfy-input-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 3px; font-size: 11px; box-sizing: border-box;">
                        <option value="raw" selected>📝 raw</option>
                        <option value="libber">🔄 libber</option>
                    </select>
                </td>
                <td style="padding: 4px;">
                    <select class="prompt-libber-select" title="Select libber for substitution" disabled style="width: 100%; padding: 6px; background: var(--comfy-input-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 3px; font-size: 11px; box-sizing: border-box;">
                        ${newRowLibberOptions}
                    </select>
                </td>
                <td style="padding: 4px;"><input type="text" class="prompt-category-input" placeholder="character" title="Optional category" style="width: 100%; padding: 6px; background: var(--comfy-input-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 3px; font-size: 11px; box-sizing: border-box;" /></td>
                <td style="white-space: nowrap; text-align: center; vertical-align: top; padding: 4px;">
                    <button class="add-prompt-btn" title="Add new prompt" style="padding: 8px 10px; background: var(--comfy-menu-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 3px; cursor: pointer; font-size: 14px; min-height: 34px;">➕</button>
                </td>
            </tr>`;
            
            return `
                ${helpText}
                ${actionButtons}
                <table style='width: 100%; border-collapse: collapse; font-size: 11px;'>
                    <thead>
                        <tr style='background: var(--comfy-menu-bg);'>
                            <th style='padding: 6px 8px; text-align: left; border-bottom: 2px solid var(--border-color); color: var(--fg-color); font-weight: 600;' title="Unique identifier for this prompt">🗝️ Key</th>
                            <th style='padding: 6px 8px; text-align: left; border-bottom: 2px solid var(--border-color); color: var(--fg-color); font-weight: 600;' title="The prompt text content">💬 Value</th>
                            <th style='padding: 6px 8px; text-align: left; border-bottom: 2px solid var(--border-color); color: var(--fg-color); font-weight: 600;' title="Raw: use as-is | Libber: substitute placeholders">🔧 Type</th>
                            <th style='padding: 6px 8px; text-align: left; border-bottom: 2px solid var(--border-color); color: var(--fg-color); font-weight: 600;' title="Which libber to use for substitution">🪙 Libber</th>
                            <th style='padding: 6px 8px; text-align: left; border-bottom: 2px solid var(--border-color); color: var(--fg-color); font-weight: 600;' title="Optional: group/organize prompts">🏷️ Category</th>
                            <th style='padding: 6px 8px; text-align: center; border-bottom: 2px solid var(--border-color); color: var(--fg-color); font-weight: 600;'>⚡ Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${existingRows}
                        ${newRow}
                    </tbody>
                </table>
            `;
        };
        
        // Render Compose tab content (returns HTML string)
        const renderComposeTab = () => {
            // Help text
            const helpText = `<div style="margin-bottom: 8px; padding: 6px 8px; background: var(--comfy-menu-bg); border: 1px solid var(--border-color); border-radius: 4px; font-size: 11px; color: var(--descrip-text);">
                <strong style="color: var(--fg-color);">🎼 Compose Prompts:</strong> 
                Create compositions by combining multiple prompts. Each composition has a name and includes one or more prompt keys. The backend will automatically combine and process them (with libber substitution if needed).
            </div>`;
            
            // Action buttons
            const actionButtons = `<div style="margin-bottom: 8px; padding-bottom: 8px; display: flex; gap: 8px; flex-wrap: wrap; align-items: center; border-bottom: 2px solid var(--border-color); background: var(--comfy-input-bg); position: sticky; top: 0; z-index: 10;">
                <button class="apply-compositions-btn" title="Save compositions to collection_json" style="padding: 4px 12px; background: var(--comfy-menu-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 4px; cursor: pointer;">✓ Apply Compositions</button>
                <span style="padding: 4px 8px; color: var(--descrip-text); font-size: 11px;" class="composition-count">${currentCompositionsData.length} compositions</span>
            </div>`;
            
            // Existing compositions
            const existingCompositions = currentCompositionsData.map((comp, idx) => {
                const escapedName = String(comp.name || '').replace(/"/g, '&quot;');
                const promptKeys = comp.prompt_keys || [];
                const promptKeysStr = promptKeys.join(', ');
                
                return `<tr data-comp-idx="${idx}" style="vertical-align: top;">
                    <td style="width: 25%; padding: 4px;">
                        <input type="text" class="comp-name-input" value="${escapedName}" title="Unique composition name" placeholder="final_prompt" style="width: 100%; padding: 6px; background: var(--comfy-input-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 3px; font-family: monospace; font-size: 11px; box-sizing: border-box;" />
                    </td>
                    <td style="width: 65%; padding: 4px;">
                        <div class="prompt-keys-container" data-comp-idx="${idx}" style="display: flex; flex-wrap: wrap; gap: 4px; padding: 4px; background: var(--comfy-input-bg); border: 1px solid var(--border-color); border-radius: 3px; min-height: 40px;">
                            ${promptKeys.map(key => `<span class="prompt-key-tag" draggable="true" data-key="${key}" style="padding: 4px 8px; background: var(--comfy-menu-bg); border: 1px solid var(--border-color); border-radius: 3px; font-size: 10px; cursor: move; display: inline-flex; align-items: center; gap: 4px;">${key} <button class="remove-key-btn" title="Remove this key" style="background: none; border: none; color: var(--error-text); cursor: pointer; padding: 0; font-size: 12px;">×</button></span>`).join('')}
                            <button class="add-key-btn" title="Add prompt key to composition" style="padding: 4px 8px; background: var(--comfy-menu-bg); border: 1px solid var(--border-color); border-radius: 3px; font-size: 10px; cursor: pointer;">+ Add Key</button>
                        </div>
                    </td>
                    <td style="white-space: nowrap; text-align: center; vertical-align: top; width: 10%; padding: 4px;">
                        <button class="remove-comp-btn" title="Remove this composition" style="padding: 8px 10px; background: var(--comfy-menu-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 3px; cursor: pointer; font-size: 14px; min-height: 34px;">🗑️</button>
                    </td>
                </tr>`;
            }).join('');
            
            // New composition row
            const availablePromptKeys = currentPromptsData.map(p => p.key).filter(k => k);
            const promptKeyOptions = availablePromptKeys.map(key => 
                `<option value="${key}">${key}</option>`
            ).join('');
            
            const newRow = `<tr class="new-comp-row" style="vertical-align: top; border-top: 2px solid var(--border-color);">
                <td style="padding: 4px;">
                    <input type="text" placeholder="composition_name" class="comp-name-input" title="Unique composition name" style="width: 100%; padding: 6px; background: var(--comfy-input-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 3px; font-family: monospace; font-size: 11px; box-sizing: border-box;" />
                </td>
                <td style="padding: 4px;">
                    <select multiple class="comp-keys-select" title="Select prompt keys to include (Ctrl/Cmd+click for multiple)" style="width: 100%; min-height: 60px; padding: 6px; background: var(--comfy-input-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 3px; font-size: 11px; box-sizing: border-box;">
                        ${promptKeyOptions}
                    </select>
                </td>
                <td style="white-space: nowrap; text-align: center; vertical-align: top; padding: 4px;">
                    <button class="add-comp-btn" title="Add new composition" style="padding: 8px 10px; background: var(--comfy-menu-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 3px; cursor: pointer; font-size: 14px; min-height: 34px;">➕</button>
                </td>
            </tr>`;
            
            return `
                ${helpText}
                ${actionButtons}
                <table style='width: 100%; border-collapse: collapse; font-size: 11px;'>
                    <thead>
                        <tr style='background: var(--comfy-menu-bg);'>
                            <th style='padding: 6px 8px; text-align: left; border-bottom: 2px solid var(--border-color); color: var(--fg-color); font-weight: 600;' title="Unique composition name">🎯 Name</th>
                            <th style='padding: 6px 8px; text-align: left; border-bottom: 2px solid var(--border-color); color: var(--fg-color); font-weight: 600;' title="Prompt keys included in this composition">🔑 Prompt Keys</th>
                            <th style='padding: 6px 8px; text-align: center; border-bottom: 2px solid var(--border-color); color: var(--fg-color); font-weight: 600;'>⚡ Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${existingCompositions}
                        ${newRow}
                    </tbody>
                </table>
            `;
        };
        
        // Render View tab content (returns HTML string)
        const renderViewTab = () => {
            // Help text
            const helpText = `<div style="margin-bottom: 8px; padding: 6px 8px; background: var(--comfy-menu-bg); border: 1px solid var(--border-color); border-radius: 4px; font-size: 11px; color: var(--descrip-text);">
                <strong style="color: var(--fg-color);">👁️ Preview Compositions:</strong> 
                View the fully processed output for each composition. Click "🔄 Process" to generate previews with libber substitutions applied.
            </div>`;
            
            // Composition selector
            const compositionOptions = currentCompositionsData.map(comp => 
                `<option value="${comp.name}">${comp.name}</option>`
            ).join('');
            
            const noCompositionsMessage = currentCompositionsData.length === 0 
                ? '<p style="padding: 12px; color: var(--descrip-text); text-align: center;">No compositions yet. Create them in the Compose tab.</p>'
                : '';
            
            let previewContent = noCompositionsMessage;
            
            if (currentCompositionsData.length > 0) {
                const firstCompName = currentCompositionsData[0].name;
                const firstCompOutput = currentPromptDict[firstCompName] || 'Click "Process" to generate preview...';
                const charCount = firstCompOutput.length;
                const hasOutput = currentPromptDict[firstCompName] !== undefined;
                
                previewContent = `
                    <div style="margin-bottom: 8px;">
                        <label style="display: block; margin-bottom: 4px; font-weight: 600; color: var(--fg-color); font-size: 11px;">Select Composition:</label>
                        <select class="comp-preview-select" style="width: 100%; padding: 8px; background: var(--comfy-input-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 3px; font-size: 12px;">
                            ${compositionOptions}
                        </select>
                    </div>
                    <div style="margin-bottom: 8px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                            <label style="font-weight: 600; color: var(--fg-color); font-size: 11px;">Processed Output:</label>
                            <span style="font-size: 10px; color: var(--descrip-text);" class="char-count">${hasOutput ? charCount + ' characters' : ''}</span>
                        </div>
                        <textarea readonly class="comp-output-textarea" style="width: 100%; min-height: 200px; padding: 8px; background: var(--comfy-input-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 3px; font-size: 11px; font-family: monospace; resize: vertical; box-sizing: border-box;">${firstCompOutput}</textarea>
                    </div>
                    <div style="display: flex; gap: 8px;">
                        <button class="process-compositions-btn" title="Process compositions with current prompts and libbers" style="padding: 6px 12px; background: var(--comfy-menu-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 4px; cursor: pointer; font-size: 11px;">🔄 Process</button>
                        <button class="copy-output-btn" title="Copy to clipboard" style="padding: 6px 12px; background: var(--comfy-menu-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 4px; cursor: pointer; font-size: 11px;" ${hasOutput ? '' : 'disabled'}>📋 Copy</button>
                    </div>
                `;
            }
            
            return `
                ${helpText}
                ${previewContent}
            `;
        };
        
        // Update collection JSON from current table data
        const updateCollectionJson = () => {
            const collection = {
                version: 2,
                prompts: {},
                compositions: {}
            };
            
            currentPromptsData.forEach(prompt => {
                if (prompt.key) {
                    collection.prompts[prompt.key] = {
                        value: prompt.value || "",
                        processing_type: prompt.processing_type || "raw",
                        libber_name: prompt.libber_name || null,
                        category: prompt.category || null
                    };
                }
            });
            
            // Add compositions
            currentCompositionsData.forEach(comp => {
                if (comp.name) {
                    collection.compositions[comp.name] = comp.prompt_keys || [];
                }
            });
            
            if (collectionJsonWidget) {
                collectionJsonWidget.value = JSON.stringify(collection, null, 2);
            }
        };
        
        // Event handlers
        const attachEventHandlers = () => {
            // Tab switching
            container.querySelectorAll('.tab-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    activeTab = e.target.getAttribute('data-tab');
                    renderTable(currentPromptsData, availableLibbers, currentCompositionsData, currentPromptDict);
                });
            });
            
            // ==== DEFINE TAB HANDLERS ====
            
            // Enable/disable libber dropdown based on type selection
            container.querySelectorAll('.prompt-type-select').forEach(select => {
                select.addEventListener('change', async (e) => {
                    const row = e.target.closest('tr');
                    const libberSelect = row.querySelector('.prompt-libber-select');
                    if (e.target.value === 'libber') {
                        libberSelect.disabled = false;
                        // If currently "none", fetch latest libbers and switch to first available
                        if (libberSelect.value === 'none') {
                            try {
                                const response = await libberAPI.listLibbers();
                                if (response && response.libbers && response.libbers.length > 0) {
                                    // Update availableLibbers list
                                    availableLibbers = ["none", ...response.libbers];
                                    
                                    // Repopulate the dropdown options
                                    libberSelect.innerHTML = availableLibbers.map(lib => {
                                        const selected = lib === response.libbers[0] ? 'selected' : '';
                                        return `<option value="${lib}" ${selected}>${lib}</option>`;
                                    }).join('');
                                    
                                    console.log("fb_tools -> ScenePromptManager: Updated libbers list from API");
                                } else if (availableLibbers.length > 1) {
                                    // Use existing list if API fails
                                    libberSelect.value = availableLibbers[1];
                                }
                            } catch (err) {
                                console.warn("fb_tools -> ScenePromptManager: Could not fetch libbers list", err);
                                // Fallback to existing list
                                if (availableLibbers.length > 1) {
                                    libberSelect.value = availableLibbers[1];
                                }
                            }
                        }
                    } else {
                        libberSelect.disabled = true;
                        libberSelect.value = 'none';
                    }
                });
            });
            
            // Add prompt button
            container.querySelector('.add-prompt-btn')?.addEventListener('click', () => {
                const row = container.querySelector('.new-prompt-row');
                const keyInput = row.querySelector('.prompt-key-input');
                const valueInput = row.querySelector('.prompt-value-input');
                const typeSelect = row.querySelector('.prompt-type-select');
                const libberSelect = row.querySelector('.prompt-libber-select');
                const categoryInput = row.querySelector('.prompt-category-input');
                
                const key = keyInput.value.trim();
                const value = valueInput.value;
                const type = typeSelect.value;
                const libber = libberSelect.value;
                const category = categoryInput.value.trim();
                
                if (!key) {
                    showToast({ severity: "warn", summary: "Key required", life: 2000 });
                    return;
                }
                
                // Check if key already exists
                if (currentPromptsData.some(p => p.key === key)) {
                    showToast({ severity: "warn", summary: `Key '${key}' already exists`, life: 2000 });
                    return;
                }
                
                // Add to data
                currentPromptsData.push({
                    key,
                    value,
                    processing_type: type,
                    libber_name: type === 'libber' && libber !== 'none' ? libber : null,
                    category: category || null
                });
                
                showToast({ severity: "success", summary: `Added '${key}'`, life: 2000 });
                renderTable(currentPromptsData, availableLibbers, currentCompositionsData, currentPromptDict);
            });
            
            // Remove prompt buttons
            container.querySelectorAll('.remove-prompt-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const row = e.target.closest('tr');
                    const idx = parseInt(row.getAttribute('data-idx'));
                    const key = currentPromptsData[idx]?.key;
                    
                    currentPromptsData.splice(idx, 1);
                    showToast({ severity: "success", summary: `Removed '${key}'`, life: 2000 });
                    renderTable(currentPromptsData, availableLibbers, currentCompositionsData, currentPromptDict);
                });
            });
            
            // Apply prompts button - update all prompts from table and save to file
            const applyBtn = container.querySelector('.apply-prompts-btn');
            console.log("fb_tools -> ScenePromptManager: Apply button found:", !!applyBtn);
            applyBtn?.addEventListener('click', async () => {
                console.log("fb_tools -> ScenePromptManager: Apply Changes clicked!");
                // Update all existing prompts from table inputs
                const rows = container.querySelectorAll('tbody tr:not(.new-prompt-row)');
                const updatedData = [];
                
                rows.forEach((row, idx) => {
                    const keyInput = row.querySelector('.prompt-key-input');
                    const valueInput = row.querySelector('.prompt-value-input');
                    const typeSelect = row.querySelector('.prompt-type-select');
                    const libberSelect = row.querySelector('.prompt-libber-select');
                    const categoryInput = row.querySelector('.prompt-category-input');
                    
                    const key = keyInput.value.trim();
                    if (key) {
                        const libberValue = libberSelect.value;
                        updatedData.push({
                            key,
                            value: valueInput.value,
                            processing_type: typeSelect.value,
                            libber_name: typeSelect.value === 'libber' && libberValue !== 'none' ? libberValue : null,
                            category: categoryInput.value.trim() || null
                        });
                    }
                });
                
                currentPromptsData = updatedData;
                updateCollectionJson();
                
                // Get current scene directory from widgets at click time
                const scenesDirWidget = node.widgets?.find(w => w.name === "scenes_dir");
                const sceneNameWidget = node.widgets?.find(w => w.name === "scene_name");
                
                console.log("fb_tools -> ScenePromptManager: Widget debug:");
                console.log("  scenesDirWidget:", scenesDirWidget);
                console.log("  sceneNameWidget:", sceneNameWidget);
                console.log("  All widgets:", node.widgets?.map(w => ({name: w.name, value: w.value, type: w.type})));
                
                const scenesDir = scenesDirWidget?.value;
                const sceneName = sceneNameWidget?.value;
                const sceneDir = (scenesDir && sceneName) ? `${scenesDir}/${sceneName}` : null;
                console.log("fb_tools -> ScenePromptManager: Apply Changes - scenesDir=", scenesDir, "sceneName=", sceneName, "sceneDir=", sceneDir);
                
                // Save to file via API if we have a scene directory
                if (sceneDir) {
                    try {
                        const collection = {
                            version: 2,
                            prompts: {},
                            compositions: {}
                        };
                        
                        currentPromptsData.forEach(prompt => {
                            if (prompt.key) {
                                collection.prompts[prompt.key] = {
                                    value: prompt.value || "",
                                    processing_type: prompt.processing_type || "raw",
                                    libber_name: prompt.libber_name || null,
                                    category: prompt.category || null
                                };
                            }
                        });
                        
                        currentCompositionsData.forEach(comp => {
                            if (comp.name) {
                                collection.compositions[comp.name] = comp.prompt_keys || [];
                            }
                        });
                        
                        const result = await sceneAPI.saveScenePrompts(sceneDir, collection);
                        if (result.success) {
                            showToast({ severity: "success", summary: result.message, life: 3000 });
                        } else {
                            showToast({ severity: "error", summary: "Failed to save", detail: result.error, life: 3000 });
                        }
                    } catch (err) {
                        console.error("Failed to save prompts to file:", err);
                        showToast({ severity: "error", summary: "Failed to save prompts", detail: err.message, life: 3000 });
                    }
                } else {
                    console.warn("fb_tools -> ScenePromptManager: No scene directory available for save");
                    showToast({ severity: "success", summary: `Applied ${currentPromptsData.length} prompts`, life: 2000 });
                }
                
                renderTable(currentPromptsData, availableLibbers, currentCompositionsData, currentPromptDict);
            });
            
            // ==== COMPOSE TAB HANDLERS ====
            
            // Add composition button
            container.querySelector('.add-comp-btn')?.addEventListener('click', () => {
                const row = container.querySelector('.new-comp-row');
                const nameInput = row.querySelector('.comp-name-input');
                const keysSelect = row.querySelector('.comp-keys-select');
                
                const name = nameInput.value.trim();
                const selectedKeys = Array.from(keysSelect.selectedOptions).map(opt => opt.value);
                
                if (!name) {
                    showToast({ severity: "warn", summary: "Composition name required", life: 2000 });
                    return;
                }
                
                if (selectedKeys.length === 0) {
                    showToast({ severity: "warn", summary: "Select at least one prompt key", life: 2000 });
                    return;
                }
                
                // Check if name already exists
                if (currentCompositionsData.some(c => c.name === name)) {
                    showToast({ severity: "warn", summary: `Composition '${name}' already exists`, life: 2000 });
                    return;
                }
                
                // Add to data
                currentCompositionsData.push({
                    name,
                    prompt_keys: selectedKeys
                });
                
                showToast({ severity: "success", summary: `Added composition '${name}'`, life: 2000 });
                renderTable(currentPromptsData, availableLibbers, currentCompositionsData, currentPromptDict);
            });
            
            // Remove composition buttons
            container.querySelectorAll('.remove-comp-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const row = e.target.closest('tr');
                    const idx = parseInt(row.getAttribute('data-comp-idx'));
                    const name = currentCompositionsData[idx]?.name;
                    
                    currentCompositionsData.splice(idx, 1);
                    showToast({ severity: "success", summary: `Removed composition '${name}'`, life: 2000 });
                    renderTable(currentPromptsData, availableLibbers, currentCompositionsData, currentPromptDict);
                });
            });
            
            // Add key to composition buttons
            container.querySelectorAll('.add-key-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const row = e.target.closest('tr');
                    const idx = parseInt(row.getAttribute('data-comp-idx'));
                    
                    // Show simple prompt to select key
                    const availableKeys = currentPromptsData.map(p => p.key).filter(k => k);
                    const selectedKey = prompt(`Enter prompt key to add (available: ${availableKeys.join(', ')}):`);
                    
                    if (selectedKey && availableKeys.includes(selectedKey)) {
                        if (!currentCompositionsData[idx].prompt_keys.includes(selectedKey)) {
                            currentCompositionsData[idx].prompt_keys.push(selectedKey);
                            showToast({ severity: "success", summary: `Added key '${selectedKey}'`, life: 2000 });
                            renderTable(currentPromptsData, availableLibbers, currentCompositionsData, currentPromptDict);
                        } else {
                            showToast({ severity: "warn", summary: `Key '${selectedKey}' already in composition`, life: 2000 });
                        }
                    } else if (selectedKey) {
                        showToast({ severity: "warn", summary: `Key '${selectedKey}' not found`, life: 2000 });
                    }
                });
            });
            
            // Remove key from composition buttons
            container.querySelectorAll('.remove-key-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const tag = e.target.closest('.prompt-key-tag');
                    const key = tag.getAttribute('data-key');
                    const row = tag.closest('tr');
                    const idx = parseInt(row.getAttribute('data-comp-idx'));
                    
                    const keyIndex = currentCompositionsData[idx].prompt_keys.indexOf(key);
                    if (keyIndex !== -1) {
                        currentCompositionsData[idx].prompt_keys.splice(keyIndex, 1);
                        showToast({ severity: "success", summary: `Removed key '${key}'`, life: 2000 });
                        renderTable(currentPromptsData, availableLibbers, currentCompositionsData, currentPromptDict);
                    }
                });
            });
            
            // Drag-and-drop handlers for reordering prompt keys in compositions
            let draggedTag = null;
            
            container.querySelectorAll('.prompt-key-tag').forEach(tag => {
                tag.addEventListener('dragstart', (e) => {
                    draggedTag = tag;
                    tag.style.opacity = '0.5';
                    e.dataTransfer.effectAllowed = 'move';
                });
                
                tag.addEventListener('dragend', (e) => {
                    tag.style.opacity = '1';
                    draggedTag = null;
                });
                
                tag.addEventListener('dragover', (e) => {
                    e.preventDefault();
                    e.dataTransfer.dropEffect = 'move';
                    
                    // Visual feedback - show where drop will occur
                    if (draggedTag && draggedTag !== tag) {
                        const container = tag.closest('.prompt-keys-container');
                        const afterElement = getDragAfterElement(container, e.clientX);
                        
                        if (afterElement == null) {
                            // Find the add button and insert before it
                            const addBtn = container.querySelector('.add-key-btn');
                            if (addBtn && draggedTag.parentNode === container) {
                                container.insertBefore(draggedTag, addBtn);
                            }
                        } else {
                            if (draggedTag.parentNode === container) {
                                container.insertBefore(draggedTag, afterElement);
                            }
                        }
                    }
                });
                
                tag.addEventListener('drop', (e) => {
                    e.preventDefault();
                    if (draggedTag && draggedTag !== tag) {
                        const container = tag.closest('.prompt-keys-container');
                        const idx = parseInt(container.getAttribute('data-comp-idx'));
                        
                        // Get new order from DOM
                        const tags = Array.from(container.querySelectorAll('.prompt-key-tag'));
                        const newOrder = tags.map(t => t.getAttribute('data-key'));
                        
                        // Update composition data
                        currentCompositionsData[idx].prompt_keys = newOrder;
                        showToast({ severity: "success", summary: "Reordered prompt keys", life: 2000 });
                    }
                });
            });
            
            // Helper function to determine drop position
            function getDragAfterElement(container, x) {
                const draggableElements = [...container.querySelectorAll('.prompt-key-tag:not(.dragging)')];
                
                return draggableElements.reduce((closest, child) => {
                    const box = child.getBoundingClientRect();
                    const offset = x - box.left - box.width / 2;
                    
                    if (offset < 0 && offset > closest.offset) {
                        return { offset: offset, element: child };
                    } else {
                        return closest;
                    }
                }, { offset: Number.NEGATIVE_INFINITY }).element;
            }
            
            // Apply compositions button - save compositions and update file
            container.querySelector('.apply-compositions-btn')?.addEventListener('click', async () => {
                // Update composition names from inputs
                const rows = container.querySelectorAll('tbody tr:not(.new-comp-row)');
                const updatedComps = [];
                
                rows.forEach((row, idx) => {
                    const nameInput = row.querySelector('.comp-name-input');
                    const name = nameInput.value.trim();
                    
                    if (name && currentCompositionsData[idx]) {
                        updatedComps.push({
                            name,
                            prompt_keys: currentCompositionsData[idx].prompt_keys
                        });
                    }
                });
                
                currentCompositionsData = updatedComps;
                updateCollectionJson();
                
                // Get current scene directory from widgets at click time
                const scenesDir = node.widgets?.find(w => w.name === "scenes_dir")?.value;
                const sceneName = node.widgets?.find(w => w.name === "scene_name")?.value;
                const sceneDir = (scenesDir && sceneName) ? `${scenesDir}/${sceneName}` : null;
                console.log("fb_tools -> ScenePromptManager: Apply Compositions - scenesDir=", scenesDir, "sceneName=", sceneName, "sceneDir=", sceneDir);
                
                // Save to file via API if we have a scene directory
                if (sceneDir) {
                    try {
                        const collection = {
                            version: 2,
                            prompts: {},
                            compositions: {}
                        };
                        
                        currentPromptsData.forEach(prompt => {
                            if (prompt.key) {
                                collection.prompts[prompt.key] = {
                                    value: prompt.value || "",
                                    processing_type: prompt.processing_type || "raw",
                                    libber_name: prompt.libber_name || null,
                                    category: prompt.category || null
                                };
                            }
                        });
                        
                        currentCompositionsData.forEach(comp => {
                            if (comp.name) {
                                collection.compositions[comp.name] = comp.prompt_keys || [];
                            }
                        });
                        
                        const result = await sceneAPI.saveScenePrompts(sceneDir, collection);
                        if (result.success) {
                            showToast({ severity: "success", summary: result.message, life: 3000 });
                        } else {
                            showToast({ severity: "error", summary: "Failed to save", detail: result.error, life: 3000 });
                        }
                    } catch (err) {
                        console.error("Failed to save compositions to file:", err);
                        showToast({ severity: "error", summary: "Failed to save compositions", detail: err.message, life: 3000 });
                    }
                } else {
                    console.warn("fb_tools -> ScenePromptManager: No scene directory available for save");
                    showToast({ severity: "success", summary: `Applied ${currentCompositionsData.length} compositions`, life: 2000 });
                }
                
                renderTable(currentPromptsData, availableLibbers, currentCompositionsData, currentPromptDict);
            });
            
            // ==== VIEW TAB HANDLERS ====
            
            // Process compositions button
            container.querySelector('.process-compositions-btn')?.addEventListener('click', async () => {
                try {
                    showToast({ severity: "info", summary: "Processing compositions...", life: 2000 });
                    
                    // Build collection data from current state
                    const collection = {
                        version: 2,
                        prompts: {},
                        compositions: {}
                    };
                    
                    currentPromptsData.forEach(prompt => {
                        if (prompt.key) {
                            collection.prompts[prompt.key] = {
                                value: prompt.value || "",
                                processing_type: prompt.processing_type || "raw",
                                libber_name: prompt.libber_name || null,
                                category: prompt.category || null
                            };
                        }
                    });
                    
                    currentCompositionsData.forEach(comp => {
                        if (comp.name) {
                            collection.compositions[comp.name] = comp.prompt_keys || [];
                        }
                    });
                    
                    // Call backend API to process
                    const response = await sceneAPI.processCompositions(collection);
                    
                    if (response && response.prompt_dict) {
                        currentPromptDict = response.prompt_dict;
                        
                        // Update the view with first composition
                        const selectedComp = container.querySelector('.comp-preview-select')?.value;
                        const compToShow = selectedComp || currentCompositionsData[0]?.name;
                        const output = currentPromptDict[compToShow] || 'No output generated';
                        
                        const textarea = container.querySelector('.comp-output-textarea');
                        const charCountSpan = container.querySelector('.char-count');
                        const copyBtn = container.querySelector('.copy-output-btn');
                        
                        if (textarea) textarea.value = output;
                        if (charCountSpan) charCountSpan.textContent = `${output.length} characters`;
                        if (copyBtn) copyBtn.disabled = false;
                        
                        showToast({ severity: "success", summary: "Compositions processed successfully", life: 2000 });
                    } else {
                        showToast({ severity: "error", summary: "Failed to process compositions", life: 3000 });
                    }
                } catch (err) {
                    console.error("Error processing compositions:", err);
                    showToast({ severity: "error", summary: "Error processing compositions", life: 3000 });
                }
            });
            
            // Composition preview dropdown
            container.querySelector('.comp-preview-select')?.addEventListener('change', (e) => {
                const selectedComp = e.target.value;
                const output = currentPromptDict[selectedComp] || 'Click "Process" to generate preview...';
                const hasOutput = currentPromptDict[selectedComp] !== undefined;
                const charCount = output.length;
                
                const textarea = container.querySelector('.comp-output-textarea');
                const charCountSpan = container.querySelector('.char-count');
                
                if (textarea) textarea.value = output;
                if (charCountSpan) charCountSpan.textContent = hasOutput ? `${charCount} characters` : '';
            });
            
            // Copy button
            container.querySelector('.copy-output-btn')?.addEventListener('click', () => {
                const textarea = container.querySelector('.comp-output-textarea');
                if (textarea) {
                    navigator.clipboard.writeText(textarea.value).then(() => {
                        showToast({ severity: "success", summary: "Copied to clipboard", life: 2000 });
                    }).catch(err => {
                        console.error("Failed to copy:", err);
                        showToast({ severity: "error", summary: "Copy failed", life: 2000 });
                    });
                }
            });
        };
        
        // Initial render
        renderTable([], ["none"], [], {});
        
        // Handle execution results - update from backend
        const onExecuted = this.onExecuted;
        this.onExecuted = function(message) {
            if (onExecuted) {
                onExecuted.apply(this, arguments);
            }
            
            // Update scene directory from current widget values
            updateSceneDir();
            
            // message.text[0] = collection_json
            // message.text[1] = prompts_list JSON
            // message.text[2] = status
            // message.text[3] = available_libbers JSON
            // message.text[4] = compositions_list JSON
            // message.text[5] = prompt_dict JSON
            // message.text[6] = comp_dict JSON
            if (message?.text && message.text.length >= 2) {
                try {
                    // Update collection_json widget
                    if (collectionJsonWidget && message.text[0]) {
                        collectionJsonWidget.value = message.text[0];
                    }
                    
                    // Parse and render prompts list
                    const promptsList = JSON.parse(message.text[1]);
                    
                    // Parse available libbers (with fallback)
                    let libbersList = ["none"];
                    if (message.text[3]) {
                        try {
                            libbersList = JSON.parse(message.text[3]);
                        } catch (err) {
                            console.warn("fb_tools -> ScenePromptManager: Error parsing libbers list", err);
                        }
                    }
                    
                    // Parse compositions list (with fallback)
                    let compositionsList = [];
                    if (message.text[4]) {
                        try {
                            compositionsList = JSON.parse(message.text[4]);
                        } catch (err) {
                            console.warn("fb_tools -> ScenePromptManager: Error parsing compositions list", err);
                        }
                    }
                    
                    // Parse prompt_dict (with fallback)
                    let promptDict = {};
                    if (message.text[5]) {
                        try {
                            promptDict = JSON.parse(message.text[5]);
                        } catch (err) {
                            console.warn("fb_tools -> ScenePromptManager: Error parsing prompt_dict", err);
                        }
                    }
                    
                    renderTable(promptsList, libbersList, compositionsList, promptDict);
                    
                    console.log("fb_tools -> ScenePromptManager: UI updated from backend");
                } catch (err) {
                    console.error("fb_tools -> ScenePromptManager: Error parsing prompts", err);
                }
            }
        };
        
        // Add onConfigure hook to restore UI state when node is loaded from saved workflow
        const onConfigure = this.onConfigure;
        this.onConfigure = function(info) {
            if (onConfigure) {
                onConfigure.apply(this, arguments);
            }
            
            // After configuration, parse the collection_json and render the UI
            if (collectionJsonWidget && collectionJsonWidget.value && collectionJsonWidget.value.trim() !== '') {
                try {
                    const collection = JSON.parse(collectionJsonWidget.value);
                    
                    // Convert prompts from object to array format
                    // Stored as: {"key1": {value: "...", processing_type: "...", ...}, ...}
                    // Need as: [{key: "key1", value: "...", processing_type: "...", ...}, ...]
                    const promptsList = [];
                    if (collection.prompts && typeof collection.prompts === 'object') {
                        for (const [key, promptData] of Object.entries(collection.prompts)) {
                            promptsList.push({
                                key: key,
                                value: promptData.value || "",
                                processing_type: promptData.processing_type || "raw",
                                libber_name: promptData.libber_name || null,
                                category: promptData.category || null
                            });
                        }
                    }
                    
                    // Convert compositions from object to array format
                    // Stored as: {"comp1": ["key1", "key2"], "comp2": [...], ...}
                    // Need as: [{name: "comp1", prompt_keys: ["key1", "key2"]}, ...]
                    const compositionsList = [];
                    if (collection.compositions && typeof collection.compositions === 'object') {
                        for (const [name, promptKeys] of Object.entries(collection.compositions)) {
                            compositionsList.push({
                                name: name,
                                prompt_keys: Array.isArray(promptKeys) ? promptKeys : []
                            });
                        }
                    }
                    
                    // Fetch available libbers from API
                    let libbersList = ["none"];
                    fetch('/fbtools/libber/list')
                        .then(response => response.json())
                        .then(data => {
                            if (data && data.libbers && Array.isArray(data.libbers)) {
                                libbersList = ["none", ...data.libbers];
                                // Re-render with updated libbers list
                                renderTable(promptsList, libbersList, compositionsList, promptDict);
                                console.log("fb_tools -> ScenePromptManager: Loaded", data.libbers.length, "libbers from API");
                            }
                        })
                        .catch(err => {
                            console.warn("fb_tools -> ScenePromptManager: Could not fetch libbers list", err);
                        });
                    
                    // No prompt_dict on load - will be generated when user clicks Process
                    const promptDict = {};
                    
                    // Reset to Define tab on load to ensure consistent state
                    activeTab = "define";
                    
                    // Render the UI with the loaded data
                    renderTable(promptsList, libbersList, compositionsList, promptDict);
                    
                    console.log("fb_tools -> ScenePromptManager: UI restored from saved workflow with", promptsList.length, "prompts and", compositionsList.length, "compositions");
                } catch (err) {
                    console.error("fb_tools -> ScenePromptManager: Error restoring UI from saved data", err);
                    // Initialize with empty data if restore fails
                    activeTab = "define";
                    renderTable([], ["none"], [], {});
                }
            } else {
                // No saved data, initialize with empty state
                console.log("fb_tools -> ScenePromptManager: No saved data, initializing with empty state");
                activeTab = "define";
                renderTable([], ["none"], [], {});
            }
        };
    };

}
