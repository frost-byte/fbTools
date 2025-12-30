/**
 * Story-related node extensions
 */

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
 * Bulk update widgets for a node
 */
function updateNodeInputs(node, textArray, entries, logPrefix) {
    if (!entries || !entries.length) return;
    entries.forEach(({ widget_index, widget_name, widget_type }) => {
        updateWidgetFromText(node, textArray, widget_index, widget_name, widget_type, logPrefix);
    });
}

/**
 * Schedule node refresh
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
 * Derive scene selector options from story JSON
 */
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

/**
 * Setup StoryEdit node extensions with table-based UI
 */
export function setupStoryEdit(nodeType, nodeData, app) {
    console.log("fb_tools -> StoryEdit node detected - initializing table UI");

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        if (onNodeCreated) {
            onNodeCreated.apply(this, arguments);
        }

        const node = this;
        const widgets = node.widgets || [];

        // Set minimum node size for table UI
        node.size[0] = Math.max(node.size[0], 700);
        node.size[1] = Math.max(node.size[1], 500);

        // Create container for table UI
        const container = document.createElement("div");
        container.style.cssText = `
            width: 100%; 
            min-height: 300px; 
            padding: 8px; 
            background: var(--comfy-input-bg); 
            border: 1px solid var(--border-color); 
            border-radius: 4px; 
            overflow-y: auto; 
            overflow-x: auto; 
            box-sizing: border-box;
        `;

        // Add DOM widget for the table
        const displayWidget = node.addDOMWidget("story_table", "preview", container, {
            serialize: false,
            hideOnZoom: false,
            getValue() { return ""; },
            setValue(v) { }
        });

        // Store references
        node._storyContainer = container;
        node._storyDisplayWidget = displayWidget;
        displayWidget.parentNode = node;

        // Compute widget size
        displayWidget.computeSize = function(width) {
            const node = this.parentNode;
            if (!node) return [width, 300];
            
            const widgetIndex = node.widgets?.indexOf(this) ?? -1;
            if (widgetIndex === -1) return [width, 300];
            
            let usedHeight = (typeof LiteGraph !== 'undefined' && LiteGraph.NODE_TITLE_HEIGHT) || 30;
            for (let i = 0; i < widgetIndex; i++) {
                const w = node.widgets[i];
                if (w.computeSize) {
                    usedHeight += w.computeSize(width)[1];
                } else {
                    usedHeight += (typeof LiteGraph !== 'undefined' && LiteGraph.NODE_WIDGET_HEIGHT) || 20;
                }
            }
            
            const bottomMargin = 15;
            const remainingHeight = node.size[1] - usedHeight - bottomMargin;
            const finalHeight = Math.max(Math.min(remainingHeight, 700), 300);
            
            return [width, finalHeight];
        };

        // Update container height
        const updateContainerHeight = () => {
            if (!displayWidget.parentNode) return;
            const widgetSize = displayWidget.computeSize(node.size[0]);
            const targetHeight = Math.max(widgetSize[1] - 20, 280);
            container.style.height = `${targetHeight}px`;
        };

        updateContainerHeight();

        // Hook into resize
        const onResize = node.onResize;
        node.onResize = function(size) {
            if (onResize) {
                onResize.apply(this, arguments);
            }
            if (node._storyDisplayWidget && node._storyContainer) {
                const widgetSize = node._storyDisplayWidget.computeSize(size[0]);
                const targetHeight = Math.max(widgetSize[1] - 20, 280);
                node._storyContainer.style.height = `${targetHeight}px`;
            }
            app.graph?.setDirtyCanvas(true);
        };

        // State management
        let currentStoryData = null;
        let currentScenes = [];
        let activeTab = "scenes"; // "scenes" or "flags"
        let availableScenes = [];
        let availablePrompts = {};
        let availableCompositions = {};

        // Load available scenes for dropdowns
        const loadAvailableScenes = async () => {
            try {
                // This would call a REST API to get available scenes
                // For now, we'll extract from existing connections or state
                // TODO: Implement REST API call
                console.log("fb_tools -> StoryEdit: Loading available scenes");
            } catch (error) {
                console.error("fb_tools -> StoryEdit: Failed to load available scenes:", error);
            }
        };

        // Render the table UI
        const renderTable = (storyData) => {
            if (!storyData || !storyData.scenes) {
                container.innerHTML = `<div style="padding: 20px; text-align: center; color: var(--descrip-text);">
                    No story loaded. Select a story from the dropdown above.
                </div>`;
                return;
            }

            currentStoryData = storyData;
            currentScenes = [...(storyData.scenes || [])].sort((a, b) => a.scene_order - b.scene_order);

            // Render tabs
            const tabsHTML = `
                <div style="display: flex; gap: 4px; margin-bottom: 8px; border-bottom: 2px solid var(--border-color);">
                    <button class="tab-btn" data-tab="scenes" style="padding: 8px 16px; background: ${activeTab === 'scenes' ? 'var(--comfy-menu-bg)' : 'transparent'}; color: var(--fg-color); border: none; border-bottom: 2px solid ${activeTab === 'scenes' ? 'var(--fg-color)' : 'transparent'}; cursor: pointer; font-weight: ${activeTab === 'scenes' ? '600' : '400'};">📋 Scenes</button>
                    <button class="tab-btn" data-tab="flags" style="padding: 8px 16px; background: ${activeTab === 'flags' ? 'var(--comfy-menu-bg)' : 'transparent'}; color: var(--fg-color); border: none; border-bottom: 2px solid ${activeTab === 'flags' ? 'var(--fg-color)' : 'transparent'}; cursor: pointer; font-weight: ${activeTab === 'flags' ? '600' : '400'};">🏴 Advanced Flags</button>
                </div>
            `;

            // Render content based on active tab
            let contentHTML = '';
            if (activeTab === 'scenes') {
                contentHTML = renderScenesTab();
            } else if (activeTab === 'flags') {
                contentHTML = renderFlagsTab();
            }

            container.innerHTML = tabsHTML + contentHTML;
            attachEventHandlers();
            updateContainerHeight();
        };

        // Render scenes tab (main table)
        const renderScenesTab = () => {
            const actionButtons = `
                <div style="margin-bottom: 8px; padding: 8px; display: flex; gap: 8px; flex-wrap: wrap; align-items: center; border-bottom: 2px solid var(--border-color); background: var(--comfy-input-bg); position: sticky; top: 0; z-index: 10;">
                    <button class="save-story-btn" title="Save story to disk" style="padding: 4px 12px; background: var(--comfy-menu-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 4px; cursor: pointer;">💾 Save</button>
                    <button class="refresh-story-btn" title="Reload story from disk" style="padding: 4px 12px; background: var(--comfy-menu-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 4px; cursor: pointer;">🔄 Refresh</button>
                    <button class="add-scene-btn" title="Add new scene" style="padding: 4px 12px; background: var(--comfy-menu-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 4px; cursor: pointer;">+ Add Scene</button>
                    <span style="padding: 4px 8px; color: var(--descrip-text); font-size: 11px;">${currentScenes.length} scenes</span>
                </div>
            `;

            // Table header
            const tableHeader = `
                <table style="width: 100%; border-collapse: collapse; font-size: 12px;">
                    <thead>
                        <tr style="background: var(--comfy-menu-bg); position: sticky; top: 48px; z-index: 9;">
                            <th style="padding: 6px; border: 1px solid var(--border-color); text-align: left; width: 50px;">Order</th>
                            <th style="padding: 6px; border: 1px solid var(--border-color); text-align: left; min-width: 120px;">Scene</th>
                            <th style="padding: 6px; border: 1px solid var(--border-color); text-align: left; width: 100px;">Mask</th>
                            <th style="padding: 6px; border: 1px solid var(--border-color); text-align: center; width: 60px;">BG</th>
                            <th style="padding: 6px; border: 1px solid var(--border-color); text-align: left; width: 110px;">Prompt Src</th>
                            <th style="padding: 6px; border: 1px solid var(--border-color); text-align: left; min-width: 120px;">Key/Custom</th>
                            <th style="padding: 6px; border: 1px solid var(--border-color); text-align: left; width: 90px;">Depth</th>
                            <th style="padding: 6px; border: 1px solid var(--border-color); text-align: left; width: 90px;">Pose</th>
                            <th style="padding: 6px; border: 1px solid var(--border-color); text-align: center; width: 120px;">Actions</th>
                        </tr>
                    </thead>
                    <tbody class="scenes-tbody">
            `;

            // Table rows
            const rows = currentScenes.map((scene, idx) => {
                const maskTypes = ["girl", "male", "combined", "girl_no_bg", "male_no_bg", "combined_no_bg"];
                const promptSources = ["prompt", "composition", "custom"];
                const depthTypes = ["depth", "depth_any", "midas", "zoe", "zoe_any"];
                const poseTypes = ["dense", "dw", "edit", "face", "open"];

                const maskOptions = maskTypes.map(m => `<option value="${m}" ${scene.mask_type === m ? 'selected' : ''}>${m}</option>`).join('');
                const promptSourceOptions = promptSources.map(p => `<option value="${p}" ${scene.prompt_source === p ? 'selected' : ''}>${p}</option>`).join('');
                const depthOptions = depthTypes.map(d => `<option value="${d}" ${scene.depth_type === d ? 'selected' : ''}>${d}</option>`).join('');
                const poseOptions = poseTypes.map(p => `<option value="${p}" ${scene.pose_type === p ? 'selected' : ''}>${p}</option>`).join('');

                return `
                    <tr data-scene-idx="${idx}" data-scene-id="${scene.scene_id || ''}" style="background: var(--comfy-input-bg);">
                        <td style="padding: 4px; border: 1px solid var(--border-color);">
                            <input type="number" class="scene-order-input" value="${scene.scene_order || idx}" min="0" style="width: 45px; padding: 2px; background: var(--comfy-input-bg); color: var(--input-text); border: 1px solid var(--border-color); border-radius: 2px;" />
                        </td>
                        <td style="padding: 4px; border: 1px solid var(--border-color);">
                            <span style="font-size: 11px; color: var(--fg-color);">${scene.scene_name || 'untitled'}</span>
                        </td>
                        <td style="padding: 4px; border: 1px solid var(--border-color);">
                            <select class="mask-type-select" style="width: 95%; padding: 2px; background: var(--comfy-input-bg); color: var(--input-text); border: 1px solid var(--border-color); border-radius: 2px; font-size: 11px;">
                                ${maskOptions}
                            </select>
                        </td>
                        <td style="padding: 4px; border: 1px solid var(--border-color); text-align: center;">
                            <input type="checkbox" class="mask-bg-checkbox" ${scene.mask_background ? 'checked' : ''} style="cursor: pointer;" />
                        </td>
                        <td style="padding: 4px; border: 1px solid var(--border-color);">
                            <select class="prompt-source-select" style="width: 100%; padding: 2px; background: var(--comfy-input-bg); color: var(--input-text); border: 1px solid var(--border-color); border-radius: 2px; font-size: 11px;">
                                ${promptSourceOptions}
                            </select>
                        </td>
                        <td style="padding: 4px; border: 1px solid var(--border-color);">
                            ${scene.prompt_source === 'custom' ? 
                                `<textarea class="custom-prompt-input" rows="2" style="width: 100%; padding: 2px; background: var(--comfy-input-bg); color: var(--input-text); border: 1px solid var(--border-color); border-radius: 2px; font-size: 11px; resize: vertical;">${scene.custom_prompt || ''}</textarea>` :
                                `<input type="text" class="prompt-key-input" value="${scene.prompt_key || ''}" placeholder="key" style="width: 100%; padding: 2px; background: var(--comfy-input-bg); color: var(--input-text); border: 1px solid var(--border-color); border-radius: 2px; font-size: 11px;" />`
                            }
                        </td>
                        <td style="padding: 4px; border: 1px solid var(--border-color);">
                            <select class="depth-type-select" style="width: 100%; padding: 2px; background: var(--comfy-input-bg); color: var(--input-text); border: 1px solid var(--border-color); border-radius: 2px; font-size: 11px;">
                                ${depthOptions}
                            </select>
                        </td>
                        <td style="padding: 4px; border: 1px solid var(--border-color);">
                            <select class="pose-type-select" style="width: 100%; padding: 2px; background: var(--comfy-input-bg); color: var(--input-text); border: 1px solid var(--border-color); border-radius: 2px; font-size: 11px;">
                                ${poseOptions}
                            </select>
                        </td>
                        <td style="padding: 4px; border: 1px solid var(--border-color); text-align: center;">
                            <button class="move-up-btn" title="Move up" style="padding: 2px 6px; margin: 0 2px; background: var(--comfy-menu-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 2px; cursor: pointer; font-size: 11px;">↑</button>
                            <button class="move-down-btn" title="Move down" style="padding: 2px 6px; margin: 0 2px; background: var(--comfy-menu-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 2px; cursor: pointer; font-size: 11px;">↓</button>
                            <button class="delete-scene-btn" title="Delete scene" style="padding: 2px 6px; margin: 0 2px; background: var(--error-bg, #8b0000); color: var(--error-text, #fff); border: 1px solid var(--border-color); border-radius: 2px; cursor: pointer; font-size: 11px;">×</button>
                        </td>
                    </tr>
                `;
            }).join('');

            const tableFooter = `
                    </tbody>
                </table>
            `;

            return actionButtons + tableHeader + rows + tableFooter;
        };

        // Render flags tab (advanced settings)
        const renderFlagsTab = () => {
            const flagsHTML = currentScenes.map((scene, idx) => `
                <div style="padding: 8px; margin-bottom: 8px; background: var(--comfy-menu-bg); border: 1px solid var(--border-color); border-radius: 4px;">
                    <div style="font-weight: 600; margin-bottom: 6px; color: var(--fg-color);">${scene.scene_name || `Scene ${idx}`}</div>
                    <div style="display: flex; gap: 16px; flex-wrap: wrap;">
                        <label style="display: flex; align-items: center; gap: 4px; font-size: 12px;">
                            <input type="checkbox" class="scene-flag" data-scene-idx="${idx}" data-flag="use_depth" ${scene.use_depth ? 'checked' : ''} /> use_depth
                        </label>
                        <label style="display: flex; align-items: center; gap: 4px; font-size: 12px;">
                            <input type="checkbox" class="scene-flag" data-scene-idx="${idx}" data-flag="use_mask" ${scene.use_mask ? 'checked' : ''} /> use_mask
                        </label>
                        <label style="display: flex; align-items: center; gap: 4px; font-size: 12px;">
                            <input type="checkbox" class="scene-flag" data-scene-idx="${idx}" data-flag="use_pose" ${scene.use_pose ? 'checked' : ''} /> use_pose
                        </label>
                        <label style="display: flex; align-items: center; gap: 4px; font-size: 12px;">
                            <input type="checkbox" class="scene-flag" data-scene-idx="${idx}" data-flag="use_canny" ${scene.use_canny ? 'checked' : ''} /> use_canny
                        </label>
                    </div>
                </div>
            `).join('');

            return `
                <div style="margin-top: 8px;">
                    <div style="margin-bottom: 12px; padding: 8px; background: var(--comfy-input-bg); border: 1px solid var(--border-color); border-radius: 4px; font-size: 11px; color: var(--descrip-text);">
                        <strong style="color: var(--fg-color);">💡 Advanced Flags:</strong> Control which control inputs are used per scene during generation.
                    </div>
                    ${flagsHTML}
                    <div style="margin-top: 12px; padding: 8px;">
                        <button class="save-story-btn" title="Save story to disk" style="padding: 6px 16px; background: var(--comfy-menu-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 4px; cursor: pointer;">💾 Save Changes</button>
                    </div>
                </div>
            `;
        };

        // Attach event handlers
        const attachEventHandlers = () => {
            // Tab buttons
            container.querySelectorAll('.tab-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    activeTab = btn.dataset.tab;
                    renderTable(currentStoryData);
                });
            });

            // Save button
            container.querySelectorAll('.save-story-btn').forEach(btn => {
                btn.addEventListener('click', async () => {
                    await saveStory();
                });
            });

            // Refresh button
            const refreshBtn = container.querySelector('.refresh-story-btn');
            if (refreshBtn) {
                refreshBtn.addEventListener('click', async () => {
                    await loadStoryData();
                });
            }

            // Add scene button
            const addBtn = container.querySelector('.add-scene-btn');
            if (addBtn) {
                addBtn.addEventListener('click', () => {
                    addNewScene();
                });
            }

            // Scene table row actions
            container.querySelectorAll('.move-up-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const row = e.target.closest('tr');
                    const idx = parseInt(row.dataset.sceneIdx);
                    moveScene(idx, -1);
                });
            });

            container.querySelectorAll('.move-down-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const row = e.target.closest('tr');
                    const idx = parseInt(row.dataset.sceneIdx);
                    moveScene(idx, 1);
                });
            });

            container.querySelectorAll('.delete-scene-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const row = e.target.closest('tr');
                    const idx = parseInt(row.dataset.sceneIdx);
                    deleteScene(idx);
                });
            });

            // Track changes in table inputs
            container.querySelectorAll('.scene-order-input, .mask-type-select, .mask-bg-checkbox, .prompt-source-select, .prompt-key-input, .custom-prompt-input, .depth-type-select, .pose-type-select').forEach(input => {
                input.addEventListener('change', (e) => {
                    updateSceneFromInput(e.target);
                });
            });

            // Track flag changes
            container.querySelectorAll('.scene-flag').forEach(checkbox => {
                checkbox.addEventListener('change', (e) => {
                    const idx = parseInt(e.target.dataset.sceneIdx);
                    const flag = e.target.dataset.flag;
                    if (currentScenes[idx]) {
                        currentScenes[idx][flag] = e.target.checked;
                    }
                });
            });

            // Handle prompt source changes (switch between key input and custom textarea)
            container.querySelectorAll('.prompt-source-select').forEach(select => {
                select.addEventListener('change', (e) => {
                    const row = e.target.closest('tr');
                    const idx = parseInt(row.dataset.sceneIdx);
                    updateSceneFromInput(e.target);
                    // Re-render to show correct input type
                    renderTable(currentStoryData);
                });
            });
        };

        // Update scene from input change
        const updateSceneFromInput = (input) => {
            const row = input.closest('tr');
            if (!row) return;

            const idx = parseInt(row.dataset.sceneIdx);
            if (idx < 0 || idx >= currentScenes.length) return;

            const scene = currentScenes[idx];

            // Update appropriate field
            if (input.classList.contains('scene-order-input')) {
                scene.scene_order = parseInt(input.value) || 0;
            } else if (input.classList.contains('mask-type-select')) {
                scene.mask_type = input.value;
            } else if (input.classList.contains('mask-bg-checkbox')) {
                scene.mask_background = input.checked;
            } else if (input.classList.contains('prompt-source-select')) {
                scene.prompt_source = input.value;
            } else if (input.classList.contains('prompt-key-input')) {
                scene.prompt_key = input.value;
            } else if (input.classList.contains('custom-prompt-input')) {
                scene.custom_prompt = input.value;
            } else if (input.classList.contains('depth-type-select')) {
                scene.depth_type = input.value;
            } else if (input.classList.contains('pose-type-select')) {
                scene.pose_type = input.value;
            }
        };

        // Add new scene
        const addNewScene = () => {
            const newScene = {
                scene_id: `scene_${Date.now()}`,
                scene_name: "new_scene",
                scene_order: currentScenes.length,
                mask_type: "combined",
                mask_background: true,
                prompt_source: "prompt",
                prompt_key: "",
                custom_prompt: "",
                depth_type: "depth",
                pose_type: "open",
                use_depth: false,
                use_mask: false,
                use_pose: false,
                use_canny: false
            };
            currentScenes.push(newScene);
            currentStoryData.scenes = currentScenes;
            renderTable(currentStoryData);
        };

        // Move scene up or down
        const moveScene = (idx, direction) => {
            const newIdx = idx + direction;
            if (newIdx < 0 || newIdx >= currentScenes.length) return;

            // Swap scenes
            [currentScenes[idx], currentScenes[newIdx]] = [currentScenes[newIdx], currentScenes[idx]];
            
            // Update scene_order
            currentScenes.forEach((scene, i) => {
                scene.scene_order = i;
            });

            currentStoryData.scenes = currentScenes;
            renderTable(currentStoryData);
        };

        // Delete scene
        const deleteScene = (idx) => {
            if (!confirm(`Delete scene "${currentScenes[idx]?.scene_name}"?`)) return;
            
            currentScenes.splice(idx, 1);
            
            // Re-index
            currentScenes.forEach((scene, i) => {
                scene.scene_order = i;
            });

            currentStoryData.scenes = currentScenes;
            renderTable(currentStoryData);
        };

        // Load story data from backend
        const loadStoryData = async () => {
            const storySelect = widgets.find(w => w.name === 'story_select')?.value;
            if (!storySelect) {
                console.warn("fb_tools -> StoryEdit: No story selected");
                container.innerHTML = `
                    <div style="padding: 20px; text-align: center; color: var(--fg-color);">
                        <p style="margin: 10px 0; font-size: 14px;">📋 Story Table</p>
                        <p style="margin: 10px 0; opacity: 0.7;">
                            Select a story from the dropdown above to begin.
                        </p>
                    </div>
                `;
                return;
            }

            try {
                console.log("fb_tools -> StoryEdit: Loading story", storySelect);
                
                const response = await fetch(`/fbtools/story/load/${encodeURIComponent(storySelect)}`);
                if (!response.ok) {
                    throw new Error(`Failed to load story: ${response.statusText}`);
                }
                
                const data = await response.json();
                console.log(`fb_tools -> StoryEdit: Loaded ${data.scenes?.length || 0} scenes from story '${storySelect}'`);
                
                currentStoryData = data;
                currentScenes = data.scenes || [];
                node._currentStoryData = currentStoryData;
                node._currentScenes = currentScenes;
                
                renderTable(data);
            } catch (error) {
                console.error("fb_tools -> StoryEdit: Failed to load story:", error);
                container.innerHTML = `
                    <div style="padding: 20px; text-align: center; color: var(--error-text);">
                        <p style="margin: 10px 0;">❌ Failed to load story</p>
                        <p style="margin: 10px 0; opacity: 0.7; font-size: 12px;">${error.message}</p>
                    </div>
                `;
            }
        };

        // Save story data to backend
        const saveStory = async () => {
            const storySelect = widgets.find(w => w.name === 'story_select')?.value;
            if (!storySelect || !currentStoryData) {
                console.warn("fb_tools -> StoryEdit: No story to save");
                return;
            }

            try {
                currentStoryData.scenes = currentScenes;
                
                console.log("fb_tools -> StoryEdit: Saving story", storySelect);
                console.log("fb_tools -> StoryEdit: Scenes to save:", JSON.stringify(currentScenes, null, 2));
                
                const response = await fetch('/fbtools/story/save', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        story_name: storySelect,
                        scenes: currentScenes
                    })
                });
                
                console.log("fb_tools -> StoryEdit: Save response status:", response.status, response.statusText);
                
                if (!response.ok) {
                    const errorText = await response.text();
                    console.error("fb_tools -> StoryEdit: Save failed:", errorText);
                    throw new Error(`Failed to save story: ${response.statusText}`);
                }
                
                const result = await response.json();
                console.log("fb_tools -> StoryEdit: Save result:", result);
                
                // Show success message briefly
                const originalHTML = container.innerHTML;
                container.innerHTML = `
                    <div style="padding: 20px; text-align: center; color: var(--fg-color);">
                        <p style="margin: 10px 0; color: #4CAF50;">✓ Story saved successfully</p>
                        <p style="margin: 10px 0; opacity: 0.7; font-size: 12px;">${result.message}</p>
                    </div>
                `;
                
                setTimeout(() => {
                    container.innerHTML = originalHTML;
                }, 2000);
            } catch (error) {
                console.error("fb_tools -> StoryEdit: Failed to save story:", error);
                alert(`Failed to save story: ${error.message}`);
            }
        };

        // Initialize by loading story data if a story is selected
        const initialStorySelect = widgets.find(w => w.name === 'story_select')?.value;
        if (initialStorySelect) {
            // Load story data immediately
            loadStoryData();
        } else {
            // Show initial message
            container.innerHTML = `
                <div style="padding: 20px; text-align: center; color: var(--fg-color);">
                    <p style="margin: 10px 0; font-size: 14px;">📋 Story Table</p>
                    <p style="margin: 10px 0; opacity: 0.7;">
                        Select a story from the dropdown above to begin.
                    </p>
                </div>
            `;
        }
    };

    // Handle execution - parse story data from UI text output
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
        if (onExecuted) {
            onExecuted.apply(this, arguments);
        }

        // Parse story data from text output
        // text[0] = summary, text[1] = prompt, text[2] = metadata JSON with full scenes data
        const textArray = message?.text;
        if (textArray && textArray[2]) {
            try {
                const storyData = JSON.parse(textArray[2]);
                console.log("fb_tools -> StoryEdit: Received story data:", storyData);
                
                // Update the table with the received data
                if (storyData.scenes && storyData.scenes.length > 0) {
                    this._currentStoryData = storyData;
                    this._currentScenes = storyData.scenes;
                    
                    // Re-render the table with the new data
                    if (this._renderTable) {
                        this._renderTable(storyData);
                    }
                    
                    console.log(`fb_tools -> StoryEdit: Table updated with ${storyData.scenes.length} scenes`);
                }
            } catch (error) {
                console.error("fb_tools -> StoryEdit: Failed to parse story data:", error);
            }
        }

        // Refresh the node display
        if (this.graph && this.graph.setDirtyCanvas) {
            this.graph.setDirtyCanvas(true, false);
        }
    };
}

/**
 * Setup StoryView node extensions
 */
export function setupStoryView(nodeType, nodeData, app) {
    console.log("fb_tools -> StoryView node detected");
    
    const onOriginalExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
        if (onOriginalExecuted) {
            onOriginalExecuted.apply(this, arguments);
        }

        // StoryView returns the selected prompt in text array
        if (message?.text) {
            const previewText = Array.isArray(message.text) ? message.text.join('\n') : message.text;
            
            // Look for the "Prompt:" line and extract the prompt value
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

