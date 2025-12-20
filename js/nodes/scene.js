/**
 * Scene-related node extensions
 */

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
    
    const widgetMap = [
        { widget_index: 0, widget_name: "girl_pos_in" },
        { widget_index: 1, widget_name: "male_pos_in" },
        { widget_index: 2, widget_name: "loras_high_in" },
        { widget_index: 3, widget_name: "loras_low_in" },
        { widget_index: 4, widget_name: "wan_prompt_in" },
        { widget_index: 5, widget_name: "wan_low_prompt_in" },
        { widget_index: 6, widget_name: "four_image_prompt_in" },
    ];
    
    const onOriginalExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
        if (onOriginalExecuted) {
            onOriginalExecuted.apply(this, arguments);
        }
        
        updateNodeInputs(this, message?.text, widgetMap, `fbTools -> ${nodeData.name}`);
        scheduleNodeRefresh(this, app);
    };
}

/**
 * Setup ScenePromptManager node extensions
 */
export function setupScenePromptManager(nodeType, nodeData, app) {
    console.log("fb_tools -> ScenePromptManager node detected");
    
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
                return container.innerHTML;
            },
            setValue(v) {
                container.innerHTML = v;
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
        let availableLibbers = ["none"];
        
        // Function to render the editable table
        const renderTable = (promptsList, libbersList) => {
            currentPromptsData = promptsList || [];
            availableLibbers = libbersList || ["none"];
            
            // Ensure "none" is always first
            if (!availableLibbers.includes("none")) {
                availableLibbers.unshift("none");
            }
            
            // Help text
            const helpText = `<div style="margin-bottom: 8px; padding: 6px 8px; background: var(--comfy-menu-bg); border: 1px solid var(--border-color); border-radius: 4px; font-size: 11px; color: var(--descrip-text);">
                <strong style="color: var(--fg-color);">💡 Quick Guide:</strong> 
                Create reusable prompt components. <strong>Key</strong>=unique identifier, <strong>Value</strong>=prompt text, <strong>Type</strong>=raw (use as-is) or libber (with substitution), <strong>Libber</strong>=which libber to use for substitution, <strong>Category</strong>=group/organize prompts. Use PromptComposer to combine prompts into outputs.
            </div>`;
            
            // Action buttons - sticky at top
            const actionButtons = `<div style="margin-bottom: 8px; padding-bottom: 8px; display: flex; gap: 8px; flex-wrap: wrap; align-items: center; border-bottom: 2px solid var(--border-color); background: var(--comfy-input-bg); position: sticky; top: 0; z-index: 10;">
                <button class="apply-btn" title="Apply all changes to collection_json" style="padding: 4px 12px; background: var(--comfy-menu-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 4px; cursor: pointer;">✓ Apply Changes</button>
                <span style="padding: 4px 8px; color: var(--descrip-text); font-size: 11px;" class="prompt-count">${currentPromptsData.length} prompts</span>
            </div>`;
            
            // Create rows for existing prompts
            const existingRows = currentPromptsData.map((prompt, idx) => {
                const escapedKey = String(prompt.key || '').replace(/"/g, '&quot;');
                const escapedValue = String(prompt.value || '').replace(/"/g, '&quot;');
                const escapedCategory = String(prompt.category || '').replace(/"/g, '&quot;');
                const processingType = prompt.processing_type || 'raw';
                const libberName = prompt.libber_name || 'none';
                const icon = processingType === 'libber' ? '🔄' : '📝';
                
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
            
            container.innerHTML = `
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
            
            // Attach event handlers
            attachEventHandlers();
            updateContainerHeight();
        };
        
        // Update collection JSON from current table data
        const updateCollectionJson = () => {
            const collection = {
                version: 2,
                prompts: {}
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
            
            if (collectionJsonWidget) {
                collectionJsonWidget.value = JSON.stringify(collection, null, 2);
            }
        };
        
        // Event handlers
        const attachEventHandlers = () => {
            // Enable/disable libber dropdown based on type selection
            container.querySelectorAll('.prompt-type-select').forEach(select => {
                select.addEventListener('change', (e) => {
                    const row = e.target.closest('tr');
                    const libberSelect = row.querySelector('.prompt-libber-select');
                    if (e.target.value === 'libber') {
                        libberSelect.disabled = false;
                        // If currently "none", switch to first available libber
                        if (libberSelect.value === 'none' && availableLibbers.length > 1) {
                            libberSelect.value = availableLibbers[1];
                        }
                    } else {
                        libberSelect.disabled = true;
                        libberSelect.value = 'none';
                    }
                });
            });
            
            // Add button
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
                renderTable(currentPromptsData, availableLibbers);
            });
            
            // Remove buttons
            container.querySelectorAll('.remove-prompt-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const row = e.target.closest('tr');
                    const idx = parseInt(row.getAttribute('data-idx'));
                    const key = currentPromptsData[idx]?.key;
                    
                    currentPromptsData.splice(idx, 1);
                    showToast({ severity: "success", summary: `Removed '${key}'`, life: 2000 });
                    renderTable(currentPromptsData, availableLibbers);
                });
            });
            
            // Apply button - update all prompts from table
            container.querySelector('.apply-btn')?.addEventListener('click', () => {
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
                showToast({ severity: "success", summary: `Applied ${currentPromptsData.length} prompts`, life: 2000 });
                renderTable(currentPromptsData, availableLibbers);
            });
        };
        
        // Initial render
        renderTable([], ["none"]);
        
        // Handle execution results - update from backend
        const onExecuted = this.onExecuted;
        this.onExecuted = function(message) {
            if (onExecuted) {
                onExecuted.apply(this, arguments);
            }
            
            // message.text[0] = collection_json
            // message.text[1] = prompts_list JSON
            // message.text[2] = status
            // message.text[3] = available_libbers JSON
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
                    
                    renderTable(promptsList, libbersList);
                    
                    console.log("fb_tools -> ScenePromptManager: UI updated from backend");
                } catch (err) {
                    console.error("fb_tools -> ScenePromptManager: Error parsing prompts", err);
                }
            }
        };
    };
}
