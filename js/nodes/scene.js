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
        
        // Function to render the editable table
        const renderTable = (promptsList) => {
            currentPromptsData = promptsList || [];
            
            // Action buttons - sticky at top
            const actionButtons = `<div style="margin-bottom: 8px; padding-bottom: 8px; display: flex; gap: 8px; flex-wrap: wrap; align-items: center; border-bottom: 2px solid var(--border-color); background: var(--comfy-input-bg); position: sticky; top: 0; z-index: 10;">
                <button class="apply-btn" style="padding: 4px 12px; background: var(--comfy-menu-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 4px; cursor: pointer;">✓ Apply</button>
                <span style="padding: 4px 8px; color: var(--descrip-text); font-size: 11px;" class="prompt-count">${currentPromptsData.length} prompts</span>
            </div>`;
            
            // Create rows for existing prompts
            const existingRows = currentPromptsData.map((prompt, idx) => {
                const escapedKey = String(prompt.key || '').replace(/"/g, '&quot;');
                const escapedValue = String(prompt.value || '').replace(/"/g, '&quot;');
                const escapedLibber = String(prompt.libber_name || '').replace(/"/g, '&quot;');
                const escapedCategory = String(prompt.category || '').replace(/"/g, '&quot;');
                const processingType = prompt.processing_type || 'raw';
                const icon = processingType === 'libber' ? '🔄' : '📝';
                
                return `<tr data-idx="${idx}" data-key="${escapedKey}">
                    <td style="width: 15%;"><input type="text" class="prompt-key-input" value="${escapedKey}" style="width: 100%; padding: 4px; background: var(--comfy-input-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 3px; font-family: monospace; font-size: 11px;" /></td>
                    <td style="width: 35%;"><textarea class="prompt-value-input" style="width: 100%; min-height: 40px; padding: 4px; background: var(--comfy-input-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 3px; resize: vertical; font-size: 11px;">${escapedValue}</textarea></td>
                    <td style="width: 15%;">
                        <select class="prompt-type-select" style="width: 100%; padding: 4px; background: var(--comfy-input-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 3px; font-size: 11px;">
                            <option value="raw" ${processingType === 'raw' ? 'selected' : ''}>📝 raw</option>
                            <option value="libber" ${processingType === 'libber' ? 'selected' : ''}>🔄 libber</option>
                        </select>
                    </td>
                    <td style="width: 15%;"><input type="text" class="prompt-libber-input" value="${escapedLibber}" placeholder="libber_name" style="width: 100%; padding: 4px; background: var(--comfy-input-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 3px; font-size: 11px;" ${processingType !== 'libber' ? 'disabled' : ''} /></td>
                    <td style="width: 12%;"><input type="text" class="prompt-category-input" value="${escapedCategory}" placeholder="category" style="width: 100%; padding: 4px; background: var(--comfy-input-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 3px; font-size: 11px;" /></td>
                    <td style="white-space: nowrap; text-align: center; vertical-align: top; width: 8%;">
                        <button class="remove-prompt-btn" title="Remove" style="padding: 4px 8px; background: var(--comfy-menu-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 3px; cursor: pointer; font-size: 12px;">➖</button>
                    </td>
                </tr>`;
            }).join('');
            
            // Add row for new prompt
            const newRow = `<tr class="new-prompt-row">
                <td><input type="text" placeholder="prompt_key" class="prompt-key-input" style="width: 100%; padding: 4px; background: var(--comfy-input-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 3px; font-family: monospace; font-size: 11px;" /></td>
                <td><textarea placeholder="prompt value" class="prompt-value-input" style="width: 100%; min-height: 40px; padding: 4px; background: var(--comfy-input-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 3px; resize: vertical; font-size: 11px;"></textarea></td>
                <td>
                    <select class="prompt-type-select" style="width: 100%; padding: 4px; background: var(--comfy-input-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 3px; font-size: 11px;">
                        <option value="raw" selected>📝 raw</option>
                        <option value="libber">🔄 libber</option>
                    </select>
                </td>
                <td><input type="text" class="prompt-libber-input" placeholder="libber_name" disabled style="width: 100%; padding: 4px; background: var(--comfy-input-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 3px; font-size: 11px;" /></td>
                <td><input type="text" class="prompt-category-input" placeholder="category" style="width: 100%; padding: 4px; background: var(--comfy-input-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 3px; font-size: 11px;" /></td>
                <td style="white-space: nowrap; text-align: center; vertical-align: top;">
                    <button class="add-prompt-btn" title="Add" style="padding: 4px 8px; background: var(--comfy-menu-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 3px; cursor: pointer; font-size: 12px;">➕</button>
                </td>
            </tr>`;
            
            container.innerHTML = `
                ${actionButtons}
                <table style='width: 100%; border-collapse: collapse; font-size: 11px;'>
                    <thead>
                        <tr style='background: var(--comfy-menu-bg);'>
                            <th style='padding: 6px 8px; text-align: left; border-bottom: 2px solid var(--border-color); color: var(--fg-color); font-weight: 600;'>🗝️ Key</th>
                            <th style='padding: 6px 8px; text-align: left; border-bottom: 2px solid var(--border-color); color: var(--fg-color); font-weight: 600;'>💬 Value</th>
                            <th style='padding: 6px 8px; text-align: left; border-bottom: 2px solid var(--border-color); color: var(--fg-color); font-weight: 600;'>🔧 Type</th>
                            <th style='padding: 6px 8px; text-align: left; border-bottom: 2px solid var(--border-color); color: var(--fg-color); font-weight: 600;'>🪙 Libber</th>
                            <th style='padding: 6px 8px; text-align: left; border-bottom: 2px solid var(--border-color); color: var(--fg-color); font-weight: 600;'>🏷️ Category</th>
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
            // Enable/disable libber name input based on type selection
            container.querySelectorAll('.prompt-type-select').forEach(select => {
                select.addEventListener('change', (e) => {
                    const row = e.target.closest('tr');
                    const libberInput = row.querySelector('.prompt-libber-input');
                    if (e.target.value === 'libber') {
                        libberInput.disabled = false;
                    } else {
                        libberInput.disabled = true;
                        libberInput.value = '';
                    }
                });
            });
            
            // Add button
            container.querySelector('.add-prompt-btn')?.addEventListener('click', () => {
                const row = container.querySelector('.new-prompt-row');
                const keyInput = row.querySelector('.prompt-key-input');
                const valueInput = row.querySelector('.prompt-value-input');
                const typeSelect = row.querySelector('.prompt-type-select');
                const libberInput = row.querySelector('.prompt-libber-input');
                const categoryInput = row.querySelector('.prompt-category-input');
                
                const key = keyInput.value.trim();
                const value = valueInput.value;
                const type = typeSelect.value;
                const libber = libberInput.value.trim();
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
                    libber_name: type === 'libber' ? libber : null,
                    category: category || null
                });
                
                showToast({ severity: "success", summary: `Added '${key}'`, life: 2000 });
                renderTable(currentPromptsData);
            });
            
            // Remove buttons
            container.querySelectorAll('.remove-prompt-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const row = e.target.closest('tr');
                    const idx = parseInt(row.getAttribute('data-idx'));
                    const key = currentPromptsData[idx]?.key;
                    
                    currentPromptsData.splice(idx, 1);
                    showToast({ severity: "success", summary: `Removed '${key}'`, life: 2000 });
                    renderTable(currentPromptsData);
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
                    const libberInput = row.querySelector('.prompt-libber-input');
                    const categoryInput = row.querySelector('.prompt-category-input');
                    
                    const key = keyInput.value.trim();
                    if (key) {
                        updatedData.push({
                            key,
                            value: valueInput.value,
                            processing_type: typeSelect.value,
                            libber_name: typeSelect.value === 'libber' ? libberInput.value.trim() || null : null,
                            category: categoryInput.value.trim() || null
                        });
                    }
                });
                
                currentPromptsData = updatedData;
                updateCollectionJson();
                showToast({ severity: "success", summary: `Applied ${currentPromptsData.length} prompts`, life: 2000 });
                renderTable(currentPromptsData);
            });
        };
        
        // Initial render
        renderTable([]);
        
        // Handle execution results - update from backend
        const onExecuted = this.onExecuted;
        this.onExecuted = function(message) {
            if (onExecuted) {
                onExecuted.apply(this, arguments);
            }
            
            // message.text[0] = collection_json
            // message.text[1] = prompts_list JSON
            // message.text[2] = status
            if (message?.text && message.text.length >= 2) {
                try {
                    // Update collection_json widget
                    if (collectionJsonWidget && message.text[0]) {
                        collectionJsonWidget.value = message.text[0];
                    }
                    
                    // Parse and render prompts list
                    const promptsList = JSON.parse(message.text[1]);
                    renderTable(promptsList);
                    
                    console.log("fb_tools -> ScenePromptManager: UI updated from backend");
                } catch (err) {
                    console.error("fb_tools -> ScenePromptManager: Error parsing prompts", err);
                }
            }
        };
    };
}
