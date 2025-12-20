/**
 * Libber-related node extensions
 */

import { libberAPI } from "../api/libber.js";

/**
 * Show toast notification
 */
function showToast(options) {
    app.extensionManager.toast.add(options);
}

/**
 * Setup LibberManager node extensions
 */
export function setupLibberManager(nodeType, nodeData, app) {
    console.log("fb_tools -> LibberManager node detected");
    
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        if (onNodeCreated) {
            onNodeCreated.apply(this, arguments);
        }
        
        const widgets = this.widgets || [];
        const libberNameWidget = widgets.find(w => w.name === "libber_name");
        const libberDirWidget = widgets.find(w => w.name === "libber_dir");
        const delimiterWidget = widgets.find(w => w.name === "delimiter");
        
        // Set minimum node size
        this.size[0] = Math.max(this.size[0], 500);
        this.size[1] = Math.max(this.size[1], 400);
        
        // Create container for editable table
        const container = document.createElement("div");
        container.style.cssText = 'width: 100%; min-height: 200px; padding: 8px; background: var(--comfy-input-bg); border: 1px solid var(--border-color); border-radius: 4px; overflow-y: auto; overflow-x: auto; box-sizing: border-box;';
        
        // Add the DOM widget for the table
        const displayWidget = this.addDOMWidget("libber_table", "preview", container, {
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
        this._libberContainer = container;
        this._libberDisplayWidget = displayWidget;
        displayWidget.parentNode = this;
        
        // Compute widget size
        displayWidget.computeSize = function(width) {
            const node = this.parentNode;
            if (!node) return [width, 250];
            
            const widgetIndex = node.widgets?.indexOf(this) ?? -1;
            if (widgetIndex === -1) return [width, 250];
            
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
            const finalHeight = Math.max(Math.min(remainingHeight, 600), 200);
            
            return [width, finalHeight];
        };
        
        // Update container height
        const updateContainerHeight = () => {
            if (!displayWidget.parentNode) return;
            const widgetSize = displayWidget.computeSize(this.size[0]);
            const targetHeight = Math.max(widgetSize[1] - 20, 180);
            container.style.height = `${targetHeight}px`;
        };
        
        updateContainerHeight();
        
        // Hook into resize
        const onResize = this.onResize;
        this.onResize = function(size) {
            if (onResize) {
                onResize.apply(this, arguments);
            }
            if (this._libberDisplayWidget && this._libberContainer) {
                const widgetSize = this._libberDisplayWidget.computeSize(size[0]);
                const targetHeight = Math.max(widgetSize[1] - 20, 180);
                this._libberContainer.style.height = `${targetHeight}px`;
            }
            app.graph?.setDirtyCanvas(true);
        };
        
        // Function to render the editable table
        const renderTable = async (libberData) => {
            const libDict = libberData?.lib_dict || {};
            const delimiter = libberData?.delimiter || delimiterWidget?.value || "%";
            const libberName = libberNameWidget?.value;
            const libberDir = libberDirWidget?.value;
            
            // Buttons for load/save/create operations - sticky at top
            const actionButtons = `<div style="margin-bottom: 8px; padding-bottom: 8px; display: flex; gap: 8px; flex-wrap: wrap; align-items: center; border-bottom: 2px solid var(--border-color); background: var(--comfy-input-bg); position: sticky; top: 0; z-index: 10;">
                <button class="load-btn" style="padding: 4px 12px; background: var(--comfy-menu-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 4px; cursor: pointer;">📂 Load</button>
                <button class="save-btn" style="padding: 4px 12px; background: var(--comfy-menu-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 4px; cursor: pointer;">💾 Save</button>
                <input type="text" class="new-libber-input" placeholder="new_libber_name" style="padding: 4px 8px; background: var(--comfy-input-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 4px; flex: 0 1 150px; font-size: 12px;" />
                <button class="create-btn" style="padding: 4px 12px; background: var(--comfy-menu-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 4px; cursor: pointer;">➕ Create</button>
                <span style="padding: 4px 8px; color: var(--descrip-text); font-size: 11px;" class="lib-count">${Object.keys(libDict).length} libs</span>
            </div>`;
            
            // Create rows for existing libs
            const existingRows = Object.entries(libDict).map(([key, value]) => {
                const escapedKey = String(key).replace(/"/g, '&quot;');
                const escapedValue = String(value).replace(/"/g, '&quot;');
                
                return `<tr data-key="${escapedKey}">
                    <td><textarea class="lib-key-input" style="width: 100%; min-height: 30px; padding: 4px; background: var(--comfy-input-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 3px; resize: vertical; font-family: inherit;">${escapedKey}</textarea></td>
                    <td><textarea class="lib-value-input" style="width: 100%; min-height: 30px; padding: 4px; background: var(--comfy-input-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 3px; resize: vertical;">${escapedValue}</textarea></td>
                    <td style="white-space: nowrap; text-align: center; vertical-align: top;">
                        <button class="update-btn" title="Update">✏️</button>
                        <button class="remove-btn" title="Remove">➖</button>
                    </td>
                </tr>`;
            }).join('');
            
            // Add row for new lib
            const newRow = `<tr class="new-row">
                <td><textarea placeholder="new_key" class="lib-key-input" style="width: 100%; min-height: 30px; padding: 4px; background: var(--comfy-input-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 3px; resize: vertical; font-family: inherit;"></textarea></td>
                <td><textarea placeholder="new value" class="lib-value-input" style="width: 100%; min-height: 30px; padding: 4px; background: var(--comfy-input-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 3px; resize: vertical;"></textarea></td>
                <td style="white-space: nowrap; text-align: center; vertical-align: top;">
                    <button class="add-btn" title="Add">➕</button>
                </td>
            </tr>`;
            
            container.innerHTML = `
                ${actionButtons}
                <table style='width: 100%; border-collapse: collapse; font-size: 12px;'>
                    <thead>
                        <tr style='background: var(--comfy-menu-bg);'>
                            <th style='padding: 6px 8px; text-align: left; border-bottom: 2px solid var(--border-color); color: var(--fg-color); font-weight: 600; width: 25%;'>🗝️ Key</th>
                            <th style='padding: 6px 8px; text-align: left; border-bottom: 2px solid var(--border-color); color: var(--fg-color); font-weight: 600; width: 60%;'>🪙 Value</th>
                            <th style='padding: 6px 8px; text-align: center; border-bottom: 2px solid var(--border-color); color: var(--fg-color); font-weight: 600; width: 15%;'>⚡ Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${existingRows}
                        ${newRow}
                    </tbody>
                </table>
            `;
            
            // Style buttons - match textarea height (min-height: 30px + padding: 8px = 38px)
            const buttons = container.querySelectorAll('button');
            buttons.forEach(btn => {
                // Skip action bar buttons (Load, Save, Create) - only style table row buttons
                if (btn.classList.contains('load-btn') || btn.classList.contains('save-btn') || btn.classList.contains('create-btn')) {
                    return;
                }
                btn.style.cssText = 'min-height: 38px; padding: 6px 8px; margin: 0 2px; background: var(--comfy-menu-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 3px; cursor: pointer; font-size: 14px; display: inline-flex; align-items: center; justify-content: center;';
                btn.onmouseover = () => btn.style.background = 'var(--border-color)';
                btn.onmouseout = () => btn.style.background = 'var(--comfy-menu-bg)';
            });
            
            // Attach event handlers
            attachEventHandlers();
            updateContainerHeight();
        };
        
        // Auto-save helper
        const autoSave = async () => {
            const libberName = libberNameWidget?.value;
            const libberDir = libberDirWidget?.value;
            
            if (libberName && libberName !== "none" && libberDir) {
                const savePath = `${libberDir}/${libberName}.json`;
                await libberAPI.saveLibber(libberName, savePath);
                console.log("fbTools -> LibberManager: auto-saved");
            }
        };
        
        // Refresh table data
        const refreshTable = async () => {
            const libberName = libberNameWidget?.value;
            const libberDir = libberDirWidget?.value;
            const delimiter = delimiterWidget?.value || "%";
            
            if (!libberName || libberName === "none") {
                console.warn("fbTools -> LibberManager: no libber name specified");
                await renderTable(null);
                return;
            }
            
            try {
                // Try to get existing libber data first
                let data = await libberAPI.getLibberData(libberName);
                await renderTable(data);
            } catch (err) {
                console.warn("fbTools -> LibberManager: libber not found, attempting to load or create", err);
                
                // Try to load from file if it exists
                if (libberDir) {
                    const loadPath = `${libberDir}/${libberName}.json`;
                    try {
                        await libberAPI.loadLibber(libberName, loadPath);
                        const data = await libberAPI.getLibberData(libberName);
                        await renderTable(data);
                        console.log("fbTools -> LibberManager: loaded libber from file");
                        return;
                    } catch (loadErr) {
                        console.log("fbTools -> LibberManager: file not found, will create new libber");
                    }
                }
                
                // If load failed or no file, create new libber
                try {
                    await libberAPI.createLibber(libberName, delimiter, 10);
                    const data = await libberAPI.getLibberData(libberName);
                    await renderTable(data);
                    console.log("fbTools -> LibberManager: created new libber");
                } catch (createErr) {
                    console.error("fbTools -> LibberManager: failed to create libber", createErr);
                    // Render empty table as fallback
                    await renderTable(null);
                }
            }
        };
        
        // Event handlers for buttons
        const attachEventHandlers = () => {
            const libberName = libberNameWidget?.value;
            
            // Add button
            container.querySelector('.add-btn')?.addEventListener('click', async () => {
                const row = container.querySelector('.new-row');
                const keyInput = row.querySelector('.lib-key-input');
                const valueInput = row.querySelector('.lib-value-input');
                const key = keyInput.value.trim();
                const value = valueInput.value;
                
                if (!key) {
                    showToast({ severity: "warn", summary: "Key required", life: 2000 });
                    return;
                }
                
                try {
                    await libberAPI.addLib(libberName, key, value);
                    showToast({ severity: "success", summary: `Added '${key}'`, life: 2000 });
                    await autoSave();
                    await refreshTable();
                } catch (err) {
                    showToast({ severity: "error", summary: `Error: ${err.message}`, life: 3000 });
                }
            });
            
            // Update buttons
            container.querySelectorAll('.update-btn').forEach(btn => {
                btn.addEventListener('click', async () => {
                    const row = btn.closest('tr');
                    const oldKey = row.getAttribute('data-key');
                    const keyInput = row.querySelector('.lib-key-input');
                    const valueInput = row.querySelector('.lib-value-input');
                    const newKey = keyInput.value.trim();
                    const newValue = valueInput.value;
                    
                    if (!newKey) {
                        showToast({ severity: "warn", summary: "Key required", life: 2000 });
                        return;
                    }
                    
                    try {
                        // If key changed, remove old and add new
                        if (oldKey !== newKey) {
                            await libberAPI.removeLib(libberName, oldKey);
                        }
                        await libberAPI.addLib(libberName, newKey, newValue);
                        showToast({ severity: "success", summary: `Updated '${newKey}'`, life: 2000 });
                        await autoSave();
                        await refreshTable();
                    } catch (err) {
                        showToast({ severity: "error", summary: `Error: ${err.message}`, life: 3000 });
                    }
                });
            });
            
            // Remove buttons
            container.querySelectorAll('.remove-btn').forEach(btn => {
                btn.addEventListener('click', async () => {
                    const row = btn.closest('tr');
                    const key = row.getAttribute('data-key');
                    
                    try {
                        await libberAPI.removeLib(libberName, key);
                        showToast({ severity: "success", summary: `Removed '${key}'`, life: 2000 });
                        await autoSave();
                        await refreshTable();
                    } catch (err) {
                        showToast({ severity: "error", summary: `Error: ${err.message}`, life: 3000 });
                    }
                });
            });
            
            // Load button
            container.querySelector('.load-btn')?.addEventListener('click', async () => {
                const libberName = libberNameWidget?.value;
                const libberDir = libberDirWidget?.value;
                
                if (!libberName || libberName === "none" || !libberDir) {
                    showToast({ severity: "warn", summary: "Select a libber to load", life: 2000 });
                    return;
                }
                
                const loadPath = `${libberDir}/${libberName}.json`;
                try {
                    await libberAPI.loadLibber(libberName, loadPath);
                    showToast({ severity: "success", summary: `Loaded '${libberName}'`, life: 2000 });
                    await refreshTable();
                } catch (err) {
                    showToast({ severity: "error", summary: `Error: ${err.message}`, life: 3000 });
                }
            });
            
            // Save button
            container.querySelector('.save-btn')?.addEventListener('click', async () => {
                const libberName = libberNameWidget?.value;
                const libberDir = libberDirWidget?.value;
                
                if (!libberName || libberName === "none" || !libberDir) {
                    showToast({ severity: "warn", summary: "Select a libber to save", life: 2000 });
                    return;
                }
                
                const savePath = `${libberDir}/${libberName}.json`;
                try {
                    await libberAPI.saveLibber(libberName, savePath);
                    showToast({ severity: "success", summary: `Saved '${libberName}'`, life: 2000 });
                } catch (err) {
                    showToast({ severity: "error", summary: `Error: ${err.message}`, life: 3000 });
                }
            });
            
            // Create button
            container.querySelector('.create-btn')?.addEventListener('click', async () => {
                const newLibberInput = container.querySelector('.new-libber-input');
                const newLibberName = newLibberInput?.value.trim();
                const libberDir = libberDirWidget?.value;
                const delimiter = delimiterWidget?.value || "%";
                
                if (!newLibberName) {
                    showToast({ severity: "warn", summary: "Enter a name for the new libber", life: 2000 });
                    return;
                }
                
                if (!libberDir) {
                    showToast({ severity: "warn", summary: "Libber directory required", life: 2000 });
                    return;
                }
                
                try {
                    // Create the libber in memory
                    await libberAPI.createLibber(newLibberName, delimiter, 10);
                    
                    // Save it to disk immediately
                    const savePath = `${libberDir}/${newLibberName}.json`;
                    await libberAPI.saveLibber(newLibberName, savePath);
                    
                    // Update the combo widget options
                    const currentOptions = libberNameWidget.options.values || [];
                    if (!currentOptions.includes(newLibberName)) {
                        // Add new option and sort
                        const updatedOptions = [...currentOptions.filter(o => o !== "none"), newLibberName].sort();
                        libberNameWidget.options.values = updatedOptions;
                    }
                    
                    // Set the combo to the new libber
                    libberNameWidget.value = newLibberName;
                    
                    // Clear the input
                    newLibberInput.value = "";
                    
                    showToast({ severity: "success", summary: `Created '${newLibberName}'`, life: 2000 });
                    
                    // Refresh the table
                    await refreshTable();
                } catch (err) {
                    showToast({ severity: "error", summary: `Error: ${err.message}`, life: 3000 });
                }
            });
        };
        
        // Initial load
        refreshTable();
        
        // Store refresh function for later use
        this._libberRefreshTable = refreshTable;
    };
    
    // Handle execution updates
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
        if (onExecuted) {
            onExecuted.apply(this, arguments);
        }
        
        // Refresh table after execution
        if (this._libberRefreshTable) {
            this._libberRefreshTable();
        }
    };
}

/**
 * Setup LibberApply node extensions
 */
export function setupLibberApply(nodeType, nodeData, app) {
    console.log("fb_tools -> LibberApply node detected");
    
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
            const success = document.execCommand('insertText', false, text);
            
            if (!success) {
                // Fallback if execCommand doesn't work
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
        
        // Set minimum node width
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
        displayWidget.parentNode = this;
        
        // Define how the widget computes its size
        displayWidget.computeSize = function(width) {
            const node = this.parentNode;
            if (!node) return [width, 200];
            
            const widgetIndex = node.widgets?.indexOf(this) ?? -1;
            if (widgetIndex === -1) return [width, 200];
            
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
            
            const bottomMargin = 15;
            const remainingHeight = node.size[1] - usedHeight - bottomMargin;
            const finalHeight = Math.max(Math.min(remainingHeight, 600), 150);
            
            return [width, finalHeight];
        };
        
        // Function to update container height based on computed widget size
        const updateContainerHeight = () => {
            if (!displayWidget.parentNode) return;
            const widgetSize = displayWidget.computeSize(this.size[0]);
            const targetHeight = widgetSize[1] - 20;
            container.style.height = `${targetHeight}px`;
        };
        
        updateContainerHeight();
        
        // Hook into node resize to update container
        const onResize = this.onResize;
        this.onResize = function(size) {
            if (onResize) {
                onResize.apply(this, arguments);
            }
            if (this._libberDisplayWidget && this._libberContainer) {
                const widgetSize = this._libberDisplayWidget.computeSize(size[0]);
                const targetHeight = Math.max(widgetSize[1] - 20, 130);
                this._libberContainer.style.height = `${targetHeight}px`;
            }
            app.graph?.setDirtyCanvas(true);
        };
        
        // Function to update display with table format
        const updateDisplay = (libberName, showMessage = null) => {
            // Refresh button - always visible, sticky at top
            const refreshButton = `<div style="margin-bottom: 8px; padding-bottom: 8px; display: flex; gap: 8px; align-items: center; border-bottom: 2px solid var(--border-color); background: var(--comfy-input-bg); position: sticky; top: 0; z-index: 10;">
                <button class="refresh-btn" style="padding: 4px 12px; background: var(--comfy-menu-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 4px; cursor: pointer;" title="Refresh libber list">🔄 Refresh</button>
                <span class="lib-count-display" style="padding: 4px 8px; color: var(--descrip-text); font-size: 11px;"></span>
            </div>`;
            
            if (showMessage) {
                container.innerHTML = `${refreshButton}<div style='padding: 12px; color: var(--descrip-text); line-height: 1.5;'>${showMessage}</div>`;
                // Add click handler to refresh button
                container.querySelector('.refresh-btn')?.addEventListener('click', async () => {
                    await refreshLibber();
                });
                updateContainerHeight();
                return;
            }
            
            if (!libberName || libberName === "none") {
                container.innerHTML = `${refreshButton}<div style='padding: 8px; color: var(--descrip-text);'>(no libber selected)</div>`;
                // Add click handler to refresh button
                container.querySelector('.refresh-btn')?.addEventListener('click', async () => {
                    await refreshLibber();
                });
                updateContainerHeight();
                return;
            }
            
            libberAPI.getLibberData(libberName).then(data => {
                if (data && data.lib_dict && Object.keys(data.lib_dict).length > 0) {
                    const delimiter = data.delimiter || "%";
                    const libCount = Object.keys(data.lib_dict).length;
                    
                    // Create table with clickable rows
                    const rows = Object.entries(data.lib_dict).map(([key, value]) => {
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
                    
                    container.innerHTML = `${refreshButton}<table style='width: 100%; border-collapse: collapse; font-size: 12px;'>
                        <thead>
                            <tr style='background: var(--comfy-menu-bg);'>
                                <th style='padding: 6px 8px; text-align: left; border-bottom: 2px solid var(--border-color); color: var(--fg-color); font-weight: 600;'>🗝️ Lib</th>
                                <th style='padding: 6px 8px; text-align: left; border-bottom: 2px solid var(--border-color); color: var(--fg-color); font-weight: 600;'>🪙 Value</th>
                            </tr>
                        </thead>
                        <tbody>${rows}</tbody>
                    </table>`;
                    
                    // Update lib count display
                    const countDisplay = container.querySelector('.lib-count-display');
                    if (countDisplay) {
                        countDisplay.textContent = `${libCount} libs`;
                    }
                    
                    // Add click handler to refresh button
                    container.querySelector('.refresh-btn')?.addEventListener('click', async () => {
                        await refreshLibber();
                    });
                    
                    // Add click handlers to table rows
                    const tableRows = container.querySelectorAll('tbody tr[data-key]');
                    tableRows.forEach(row => {
                        row.addEventListener('click', () => {
                            const key = row.getAttribute('data-key');
                            const wrappedKey = `${delimiter}${key}${delimiter}`;
                            insertAtCursor(wrappedKey);
                        });
                    });
                    
                    updateContainerHeight();
                    console.log(`fbTools -> LibberApply: displayed ${libCount} libs`);
                } else {
                    // Empty libber - show refresh button with message
                    container.innerHTML = `${refreshButton}<div style='padding: 8px; color: var(--descrip-text);'>(empty libber)</div>`;
                    
                    // Add click handler to refresh button
                    container.querySelector('.refresh-btn')?.addEventListener('click', async () => {
                        await refreshLibber();
                    });
                    
                    updateContainerHeight();
                }
            }).catch(err => {
                container.innerHTML = `${refreshButton}<div style='padding: 8px; color: var(--error-text);'>Error: ${err.message}</div>`;
                
                // Add click handler to refresh button
                container.querySelector('.refresh-btn')?.addEventListener('click', async () => {
                    await refreshLibber();
                });
                
                updateContainerHeight();
                console.warn("fbTools -> LibberApply: failed to fetch libber data", err);
            });
        };
        
        // Function to load libber if not in memory
        const ensureLibberLoaded = async (libberName) => {
            if (!libberName || libberName === "none") return false;
            
            try {
                await libberAPI.getLibberData(libberName);
                return true;
            } catch (err) {
                console.log("fbTools -> LibberApply: libber not loaded, attempting to load from file");
                try {
                    const libberDir = "output/libbers";
                    const filename = `${libberName}.json`;
                    const loadPath = `${libberDir}/${filename}`;
                    await libberAPI.loadLibber(libberName, loadPath);
                    console.log(`fbTools -> LibberApply: loaded '${libberName}' from file`);
                    return true;
                } catch (loadErr) {
                    console.error("fbTools -> LibberApply: failed to load libber", loadErr);
                    return false;
                }
            }
        };
        
        // Function to refresh libber list and select appropriate libber
        const refreshLibber = async () => {
            try {
                const currentSelection = libberNameWidget.value;
                const data = await libberAPI.listLibbers();
                console.log("fbTools -> LibberApply: refreshed libber list", data);
                
                let libbers = [];
                if (data && data.libbers && data.libbers.length > 0) {
                    libbers = data.libbers;
                } else if (data && data.files && data.files.length > 0) {
                    libbers = data.files.map(f => f.replace('.json', ''));
                }
                
                if (libbers.length === 0) {
                    updateDisplay(null, `
                        <div style='text-align: center;'>
                            <div style='font-size: 24px; margin-bottom: 8px;'>📚</div>
                            <div style='font-weight: 500; margin-bottom: 4px;'>No libbers found</div>
                            <div style='font-size: 11px; opacity: 0.8;'>Use the <strong>LibberManager</strong> node to create a libber</div>
                        </div>
                    `);
                    libberNameWidget.options.values = ["none"];
                    libberNameWidget.value = "none";
                    return;
                }
                
                libberNameWidget.options.values = ["none", ...libbers];
                
                let libberToLoad = null;
                if (currentSelection && currentSelection !== "none" && libbers.includes(currentSelection)) {
                    libberToLoad = currentSelection;
                    console.log("fbTools -> LibberApply: keeping current selection", libberToLoad);
                } else {
                    libberToLoad = libbers[0];
                    libberNameWidget.value = libberToLoad;
                    console.log("fbTools -> LibberApply: selected first available libber", libberToLoad);
                }
                
                const loaded = await ensureLibberLoaded(libberToLoad);
                if (loaded) {
                    updateDisplay(libberToLoad);
                } else {
                    updateDisplay(null, `
                        <div style='text-align: center;'>
                            <div style='font-size: 24px; margin-bottom: 8px;'>⚠️</div>
                            <div style='font-weight: 500; margin-bottom: 4px;'>Failed to load libber</div>
                            <div style='font-size: 11px; opacity: 0.8;'>Try reloading or check if the file exists</div>
                        </div>
                    `);
                }
            } catch (err) {
                console.error("fbTools -> LibberApply: refresh failed", err);
                updateDisplay(null, `
                    <div style='text-align: center;'>
                        <div style='font-size: 24px; margin-bottom: 8px;'>❌</div>
                        <div style='font-weight: 500; margin-bottom: 4px;'>Error refreshing libbers</div>
                        <div style='font-size: 11px; opacity: 0.8;'>${err.message}</div>
                    </div>
                `);
            }
        };
        
        // Fetch and populate available libbers
        libberAPI.listLibbers().then(async data => {
            if (data && libberNameWidget) {
                let libbers = [];
                if (data.libbers && data.libbers.length > 0) {
                    libbers = data.libbers;
                } else if (data.files && data.files.length > 0) {
                    libbers = data.files.map(f => f.replace('.json', ''));
                }
                
                if (libbers.length === 0) {
                    libbers = ["none"];
                }
                
                libberNameWidget.options.values = libbers;
                if (!libberNameWidget.value || !libbers.includes(libberNameWidget.value)) {
                    libberNameWidget.value = libbers[0];
                }
                console.log("fbTools -> LibberApply: populated libber_name options", libbers);
                
                const loaded = await ensureLibberLoaded(libberNameWidget.value);
                if (loaded) {
                    updateDisplay(libberNameWidget.value);
                }
            }
        }).catch(err => {
            console.warn("fbTools -> LibberApply: failed to fetch libbers", err);
            if (libberNameWidget) {
                libberNameWidget.options.values = ["none"];
                libberNameWidget.value = "none";
                updateDisplay("none");
            }
        });
        
        // Store updateDisplay function on the node
        this._libberUpdateDisplay = updateDisplay;
        
        // Add widget change handler
        const onWidgetChanged = this.onWidgetChanged;
        this.onWidgetChanged = async function (widgetName, newValue, oldValue, widgetObject) {
            const originalReturn = onWidgetChanged?.apply(this, arguments);
            
            if (widgetName === "libber_name") {
                const loaded = await ensureLibberLoaded(newValue);
                if (loaded || newValue === "none") {
                    updateDisplay(newValue);
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
        
        const widgets = this.widgets || [];
        const libberNameWidget = widgets.find(w => w.name === "libber_name");
        
        if (this._libberUpdateDisplay && libberNameWidget?.value) {
            console.log("fbTools -> LibberApply: refreshing display after execution");
            this._libberUpdateDisplay(libberNameWidget.value);
        }
        
        // Schedule node refresh
        requestAnimationFrame(() => {
            const sz = this.computeSize();
            if (sz[0] < this.size[0]) sz[0] = this.size[0];
            if (sz[1] < this.size[1]) sz[1] = this.size[1];
            this.onResize?.(sz);
            app.graph.setDirtyCanvas(true, false);
        });
    };
}
