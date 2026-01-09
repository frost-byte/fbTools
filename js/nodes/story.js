/**
 * Story-related node extensions
 */

import { sceneAPI } from "../api/scene.js";
import { showOverlay } from "../utils/feedback.js";

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
    nodeType.prototype.onNodeCreated = async function () {
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
                const response = await fetch('/fbtools/scene/list');
                const data = await response.json();
                availableScenes = data.scenes || [];
                console.log("fb_tools -> StoryEdit: Loaded available scenes:", availableScenes);
            } catch (error) {
                console.error("fb_tools -> StoryEdit: Failed to load available scenes:", error);
                availableScenes = [];
            }
        };

        // Render the table UI
        const renderTable = async (storyData, resetScenes = false) => {
            if (!storyData || !storyData.scenes) {
                container.innerHTML = `<div style="padding: 20px; text-align: center; color: var(--descrip-text);">
                    No story loaded. Select a story from the dropdown above.
                </div>`;
                return;
            }

            currentStoryData = storyData;
            // Only reset currentScenes when loading fresh data, not when switching tabs
            if (resetScenes) {
                currentScenes = [...(storyData.scenes || [])].sort((a, b) => a.scene_order - b.scene_order);
            }

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

            container.innerHTML = `
                <div style="position: relative; min-height: 200px;">
                    ${tabsHTML}
                    ${contentHTML}
                </div>
            `;
            attachEventHandlers();
            
            // Populate video prompt controls if on flags tab
            if (activeTab === 'flags') {
                await populateVideoPromptControls();
            }
            
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
                            ${scene._isNewScene ? 
                                `<select class="scene-name-select" style="width: 100%; padding: 2px; background: var(--comfy-input-bg); color: var(--input-text); border: 1px solid var(--border-color); border-radius: 2px; font-size: 11px;">
                                    ${availableScenes.map(s => `<option value="${s}" ${scene.scene_name === s ? 'selected' : ''}>${s}</option>`).join('')}
                                </select>` :
                                `<span style="font-size: 11px; color: var(--fg-color);">${scene.scene_name || 'untitled'}</span>`
                            }
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
            const videoPromptSources = ["auto", "prompt", "composition", "custom"];
            
            const flagsHTML = currentScenes.map((scene, idx) => {
                const videoPromptSourceOptions = videoPromptSources.map(s => 
                    `<option value="${s}" ${(scene.video_prompt_source || 'auto') === s ? 'selected' : ''}>${s}</option>`
                ).join('');
                
                const videoPromptSource = scene.video_prompt_source || 'auto';
                
                // We'll populate prompt keys dynamically after render
                const showDropdown = videoPromptSource === 'prompt' || videoPromptSource === 'composition';
                const showCustomPrompt = videoPromptSource === 'custom';
                
                return `
                <div style="padding: 8px; margin-bottom: 8px; background: var(--comfy-menu-bg); border: 1px solid var(--border-color); border-radius: 4px;">
                    <div style="font-weight: 600; margin-bottom: 6px; color: var(--fg-color);">${scene.scene_name || `Scene ${idx}`}</div>
                    
                    <div style="margin-bottom: 8px;">
                        <div style="font-size: 11px; color: var(--descrip-text); margin-bottom: 4px;">🎬 Video Prompt Settings:</div>
                        <div style="display: grid; grid-template-columns: 150px 1fr; gap: 8px; align-items: ${showCustomPrompt ? 'start' : 'center'};">
                            <label style="font-size: 12px;">video_prompt_source:</label>
                            <select class="video-prompt-source-select" data-scene-idx="${idx}" style="padding: 4px; background: var(--comfy-input-bg); color: var(--input-text); border: 1px solid var(--border-color); border-radius: 2px; font-size: 11px;">
                                ${videoPromptSourceOptions}
                            </select>
                            
                            ${!showCustomPrompt ? `
                            <label style="font-size: 12px;">video_prompt_key:</label>
                            <div class="video-prompt-key-container" data-scene-idx="${idx}">
                                ${showDropdown ? 
                                    `<select class="video-prompt-key-select" data-scene-idx="${idx}" style="width: 100%; padding: 4px; background: var(--comfy-input-bg); color: var(--input-text); border: 1px solid var(--border-color); border-radius: 2px; font-size: 11px;">
                                        <option value="">Loading...</option>
                                    </select>` :
                                    `<input type="text" class="video-prompt-key-input" data-scene-idx="${idx}" value="${scene.video_prompt_key || ''}" placeholder="key (for prompt/composition source)" style="width: 100%; padding: 4px; background: var(--comfy-input-bg); color: var(--input-text); border: 1px solid var(--border-color); border-radius: 2px; font-size: 11px;" />`
                                }
                            </div>
                            ` : `
                            <label style="font-size: 12px; padding-top: 4px;">custom_prompt:</label>
                            <textarea class="video-custom-prompt-input" data-scene-idx="${idx}" style="width: 100%; min-height: 80px; padding: 4px; background: var(--comfy-input-bg); color: var(--input-text); border: 1px solid var(--border-color); border-radius: 2px; font-size: 11px; resize: vertical; font-family: monospace;">${scene.video_custom_prompt || ''}</textarea>
                            `}
                        </div>
                        <div style="margin-top: 8px; grid-column: 1 / -1;">
                            <div style="font-size: 11px; color: var(--descrip-text); margin-bottom: 4px;">Preview:</div>
                            <textarea class="video-prompt-preview" data-scene-idx="${idx}" readonly style="width: 100%; min-height: 60px; padding: 4px; background: var(--comfy-input-bg); color: var(--descrip-text); border: 1px solid var(--border-color); border-radius: 2px; font-size: 11px; resize: vertical; font-family: monospace;">Loading...</textarea>
                        </div>
                    </div>
                    
                    <div style="font-size: 11px; color: var(--descrip-text); margin-bottom: 4px;">🏴 Control Flags:</div>
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
            `;
            }).join('');

            return `
                <div style="margin-top: 8px;">
                    <div style="margin-bottom: 12px; padding: 8px; background: var(--comfy-input-bg); border: 1px solid var(--border-color); border-radius: 4px; font-size: 11px; color: var(--descrip-text);">
                        <strong style="color: var(--fg-color);">💡 Advanced Flags:</strong> Control video prompts and which control inputs are used per scene during generation.
                    </div>
                    ${flagsHTML}
                    <div style="margin-top: 12px; padding: 8px;">
                        <button class="save-story-btn" title="Save story to disk" style="padding: 6px 16px; background: var(--comfy-menu-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 4px; cursor: pointer;">💾 Save Changes</button>
                    </div>
                </div>
            `;
        };

        // Populate video prompt key dropdowns and previews after rendering
        const populateVideoPromptControls = async () => {
            for (let idx = 0; idx < currentScenes.length; idx++) {
                const scene = currentScenes[idx];
                const videoPromptSource = scene.video_prompt_source || 'auto';
                
                // Get the container for this scene
                const keyContainer = container.querySelector(`.video-prompt-key-container[data-scene-idx="${idx}"]`);
                const previewTextarea = container.querySelector(`.video-prompt-preview[data-scene-idx="${idx}"]`);
                
                if (!keyContainer || !previewTextarea) continue;
                
                try {
                    // Fetch scene prompts if needed
                    if (videoPromptSource === 'prompt' || videoPromptSource === 'composition') {
                        const sceneDir = `output/scenes/${scene.scene_name}`;
                        const promptData = await sceneAPI.getScenePrompts(sceneDir);
                        
                        // Store for later use
                        scene._promptData = promptData;
                        
                        // Create a lookup map for prompts (since it's an array)
                        const promptsMap = {};
                        if (promptData.prompts && Array.isArray(promptData.prompts)) {
                            promptData.prompts.forEach(p => {
                                promptsMap[p.key] = p;
                            });
                        }
                        scene._promptsMap = promptsMap;
                        
                        // Get available keys
                        let keys = [];
                        if (videoPromptSource === 'prompt' && promptData.prompts) {
                            keys = promptData.prompts.map(p => p.key);
                        } else if (videoPromptSource === 'composition' && promptData.compositions) {
                            keys = Object.keys(promptData.compositions);
                        }
                        
                        // Update dropdown
                        const selectEl = keyContainer.querySelector('.video-prompt-key-select');
                        if (selectEl) {
                            const currentValue = scene.video_prompt_key || '';
                            
                            // If current value is empty or not in the list, use first key
                            if (keys.length > 0 && (!currentValue || !keys.includes(currentValue))) {
                                scene.video_prompt_key = keys[0];
                            }
                            
                            selectEl.innerHTML = keys.map(k => 
                                `<option value="${k}" ${k === scene.video_prompt_key ? 'selected' : ''}>${k}</option>`
                            ).join('');
                            
                            if (keys.length === 0) {
                                selectEl.innerHTML = '<option value="">No keys available</option>';
                                scene.video_prompt_key = '';
                            }
                        }
                    }
                    
                    // Update preview
                    await updateVideoPromptPreview(idx);
                    
                } catch (error) {
                    console.error(`Error populating video prompt controls for scene ${idx}:`, error);
                    if (previewTextarea) {
                        previewTextarea.value = `Error loading prompt data: ${error.message}`;
                    }
                }
            }
        };

        // Update video prompt preview textarea for a specific scene
        const updateVideoPromptPreview = async (idx) => {
            const scene = currentScenes[idx];
            const videoPromptSource = scene.video_prompt_source || 'auto';
            const videoPromptKey = scene.video_prompt_key || '';
            const previewTextarea = container.querySelector(`.video-prompt-preview[data-scene-idx="${idx}"]`);
            
            if (!previewTextarea) return;
            
            try {
                let previewText = '';
                
                switch (videoPromptSource) {
                    case 'auto':
                        // Show the image prompt (from prompt_dict via prompt_key)
                        if (scene.prompt_key && scene._promptsMap) {
                            const promptObj = scene._promptsMap[scene.prompt_key];
                            previewText = promptObj?.value || '(Image prompt not found)';
                        } else {
                            previewText = '(Using image prompt - will be resolved at generation)';
                        }
                        break;
                        
                    case 'prompt':
                        // Show selected prompt value
                        if (videoPromptKey && scene._promptsMap) {
                            const promptObj = scene._promptsMap[videoPromptKey];
                            previewText = promptObj?.value || '(Prompt not found)';
                        } else {
                            previewText = '(No prompt key selected)';
                        }
                        break;
                        
                    case 'composition':
                        // Show composition - need to resolve it
                        if (videoPromptKey && scene._promptData?.compositions && scene._promptsMap) {
                            const composition = scene._promptData.compositions[videoPromptKey];
                            console.log('Composition debug:', {
                                videoPromptKey,
                                composition,
                                isArray: Array.isArray(composition),
                                compositions: scene._promptData.compositions,
                                promptsMap: scene._promptsMap
                            });
                            
                            if (composition) {
                                if (Array.isArray(composition)) {
                                    // Resolve composition by joining prompt values
                                    const resolvedParts = composition.map(key => {
                                        const promptObj = scene._promptsMap[key];
                                        return promptObj?.value || `[${key}]`;
                                    });
                                    previewText = resolvedParts.join(', ');
                                } else if (typeof composition === 'string') {
                                    // Composition might be a single string
                                    previewText = composition;
                                } else {
                                    previewText = `(Composition format not supported: ${typeof composition})`;
                                }
                            } else {
                                previewText = `(Composition '${videoPromptKey}' not found)`;
                            }
                        } else {
                            if (!videoPromptKey) {
                                previewText = '(No composition selected)';
                            } else if (!scene._promptData?.compositions) {
                                previewText = '(No compositions data loaded)';
                            } else if (!scene._promptsMap) {
                                previewText = '(No prompts map available)';
                            } else {
                                previewText = '(Unknown error)';
                            }
                        }
                        break;
                        
                    case 'custom':
                        // Show custom prompt text
                        previewText = scene.video_custom_prompt || '(No custom prompt set)';
                        break;
                        
                    default:
                        previewText = '(Unknown source)';
                }
                
                previewTextarea.value = previewText;
                
            } catch (error) {
                console.error(`Error updating preview for scene ${idx}:`, error);
                previewTextarea.value = `Error: ${error.message}`;
            }
        };

        // Attach event handlers
        const attachEventHandlers = () => {
            // Tab buttons
            container.querySelectorAll('.tab-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    activeTab = btn.dataset.tab;
                    // Don't reset scenes when switching tabs - preserve user changes
                    renderTable(currentStoryData, false);
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
            container.querySelectorAll('.scene-order-input, .scene-name-select, .mask-type-select, .mask-bg-checkbox, .prompt-source-select, .prompt-key-input, .custom-prompt-input, .depth-type-select, .pose-type-select').forEach(input => {
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

            // Track video prompt source changes
            container.querySelectorAll('.video-prompt-source-select').forEach(select => {
                select.addEventListener('change', async (e) => {
                    const idx = parseInt(e.target.dataset.sceneIdx);
                    if (currentScenes[idx]) {
                        currentScenes[idx].video_prompt_source = e.target.value;
                        // Re-render flags tab to show correct input type, but keep current changes
                        renderTable(currentStoryData, false);
                    }
                });
            });

            // Track video prompt key changes (text input)
            container.querySelectorAll('.video-prompt-key-input').forEach(input => {
                input.addEventListener('change', async (e) => {
                    const idx = parseInt(e.target.dataset.sceneIdx);
                    if (currentScenes[idx]) {
                        currentScenes[idx].video_prompt_key = e.target.value;
                        await updateVideoPromptPreview(idx);
                    }
                });
            });

            // Track video prompt key changes (dropdown)
            container.querySelectorAll('.video-prompt-key-select').forEach(select => {
                select.addEventListener('change', async (e) => {
                    const idx = parseInt(e.target.dataset.sceneIdx);
                    if (currentScenes[idx]) {
                        currentScenes[idx].video_prompt_key = e.target.value;
                        await updateVideoPromptPreview(idx);
                    }
                });
            });

            // Track video custom prompt changes
            container.querySelectorAll('.video-custom-prompt-input').forEach(textarea => {
                textarea.addEventListener('input', async (e) => {
                    const idx = parseInt(e.target.dataset.sceneIdx);
                    if (currentScenes[idx]) {
                        currentScenes[idx].video_custom_prompt = e.target.value;
                        await updateVideoPromptPreview(idx);
                    }
                });
            });

            // Handle prompt source changes (switch between key input and custom textarea)
            container.querySelectorAll('.prompt-source-select').forEach(select => {
                select.addEventListener('change', (e) => {
                    const row = e.target.closest('tr');
                    const idx = parseInt(row.dataset.sceneIdx);
                    updateSceneFromInput(e.target);
                    // Re-render to show correct input type, but keep current changes
                    renderTable(currentStoryData, false);
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
            } else if (input.classList.contains('scene-name-select')) {
                scene.scene_name = input.value;
                delete scene._isNewScene;  // Remove flag once scene is selected
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
            // Use first available scene or fallback to "new_scene"
            const firstScene = availableScenes.length > 0 ? availableScenes[0] : "new_scene";
            
            const newScene = {
                scene_id: `scene_${Date.now()}`,
                scene_name: firstScene,
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
                use_canny: false,
                _isNewScene: true  // Flag to show dropdown
            };
            currentScenes.push(newScene);
            currentStoryData.scenes = currentScenes;
            // Reset not needed here since we're working with currentScenes
            renderTable(currentStoryData, false);
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
            // Don't reset scenes after moving
            renderTable(currentStoryData, false);
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
            // Don't reset scenes after deleting
            renderTable(currentStoryData, false);
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
                console.log("fb_tools -> StoryEdit: Sample scene data with video fields:", data.scenes[0]);
                
                // Normalize scenes to ensure all flag fields have explicit boolean values
                const normalizedScenes = (data.scenes || []).map(scene => ({
                    ...scene,
                    // Ensure all flag fields exist with defaults if not set
                    use_depth: scene.use_depth ?? false,
                    use_mask: scene.use_mask ?? false,
                    use_pose: scene.use_pose ?? false,
                    use_canny: scene.use_canny ?? false,
                    // Ensure video fields have defaults
                    video_prompt_source: scene.video_prompt_source || 'auto',
                    video_prompt_key: scene.video_prompt_key || '',
                    video_custom_prompt: scene.video_custom_prompt || ''
                }));
                
                currentStoryData = { ...data, scenes: normalizedScenes };
                currentScenes = normalizedScenes;
                node._currentStoryData = currentStoryData;
                node._currentScenes = currentScenes;
                // Pass true to reset scenes when loading from backend
                renderTable(currentStoryData, true);
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
                
                // Filter out _isNewScene flag before saving
                const scenesToSave = currentScenes.map(scene => {
                    const { _isNewScene, ...cleanScene } = scene;
                    return cleanScene;
                });
                
                console.log("fb_tools -> StoryEdit: Saving story", storySelect);
                console.log("fb_tools -> StoryEdit: Current scenes count:", scenesToSave.length);
                console.log("fb_tools -> StoryEdit: Sample scene data:", scenesToSave[0]);
                console.log("fb_tools -> StoryEdit: Full scenes to save:", JSON.stringify(scenesToSave, null, 2));
                
                const response = await fetch('/fbtools/story/save', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        story_name: storySelect,
                        scenes: scenesToSave
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
                
                // Show success overlay
                showOverlay({
                    container: container,
                    message: "Story saved successfully",
                    details: result.message || '',
                    type: "success",
                    duration: 2000
                });
            } catch (error) {
                console.error("fb_tools -> StoryEdit: Failed to save story:", error);
                showOverlay({
                    container: container,
                    message: "Failed to save story",
                    details: error.message,
                    type: "error",
                    duration: 3000
                });
            }
        };

        // Function to update preview scene combo options
        const updatePreviewSceneOptions = async (storyName) => {
            const previewSceneWidget = widgets.find(w => w.name === 'preview_scene_name');
            if (!previewSceneWidget || !storyName) return;
            
            try {
                const response = await fetch(`/fbtools/story/load/${encodeURIComponent(storyName)}`);
                if (response.ok) {
                    const storyData = await response.json();
                    if (storyData.scenes && storyData.scenes.length > 0) {
                        // Sort scenes by order
                        const sortedScenes = [...storyData.scenes].sort((a, b) => a.scene_order - b.scene_order);
                        const sceneNames = ["", ...sortedScenes.map(s => s.scene_name)];
                        
                        // Update preview_scene_name combo options
                        previewSceneWidget.options.values = sceneNames;
                        
                        // Only reset to empty if current value is not in new options
                        if (!sceneNames.includes(previewSceneWidget.value)) {
                            previewSceneWidget.value = "";
                        }
                        
                        console.log(`fb_tools -> StoryEdit: Updated preview_scene_name options with ${sceneNames.length - 1} scenes from ${storyName}`);
                    }
                } else {
                    console.warn(`fb_tools -> StoryEdit: Failed to load story '${storyName}':`, response.status);
                }
            } catch (error) {
                console.error("fb_tools -> StoryEdit: Failed to load scenes for preview_scene_name:", error);
            }
        };

        // Add callback for story_select changes to update preview_scene_name combo
        const storySelectWidget = widgets.find(w => w.name === 'story_select');
        
        if (storySelectWidget) {
            const originalCallback = storySelectWidget.callback;
            storySelectWidget.callback = async function(value) {
                // Call original callback if exists
                if (originalCallback) {
                    originalCallback.apply(this, arguments);
                }
                
                // Update preview scene options for the new story
                await updatePreviewSceneOptions(value);
                
                // Trigger load of new story data
                loadStoryData();
            };
        }

        // Initialize: Update preview scene options based on currently selected story
        const initialStorySelect = widgets.find(w => w.name === 'story_select')?.value;
        if (initialStorySelect) {
            // Load available scenes first
            await loadAvailableScenes();
            
            // Update preview scene combo for current story first
            await updatePreviewSceneOptions(initialStorySelect);
            
            // Then load story data
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

/**
 * Setup StorySceneBatch node extensions for dynamic story dropdown
 */
export function setupStorySceneBatch(nodeType, nodeData, app) {
    console.log("fb_tools -> StorySceneBatch: Setting up dynamic story dropdown");
    
    const onOriginalNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function() {
        if (onOriginalNodeCreated) {
            onOriginalNodeCreated.apply(this, arguments);
        }
        
        // Initial load of story options
        fetchAndUpdateStoryOptions(this);
    };
    
    /**
     * Fetch available stories from API and update dropdown
     */
    async function fetchAndUpdateStoryOptions(node) {
        try {
            const response = await fetch('/fbtools/story/list');
            if (!response.ok) {
                console.error('fb_tools -> StorySceneBatch: Failed to fetch stories:', response.statusText);
                return;
            }
            
            const data = await response.json();
            if (data.stories && Array.isArray(data.stories)) {
                const storyWidget = node.widgets?.find(w => w.name === 'story_name');
                if (storyWidget && storyWidget.type === 'combo') {
                    // Update combo options
                    storyWidget.options.values = data.stories;
                    
                    // Set value to first story if current value is not in list
                    if (data.stories.length > 0 && !data.stories.includes(storyWidget.value)) {
                        storyWidget.value = data.stories[0];
                    }
                    
                    console.log(`fb_tools -> StorySceneBatch: Updated story dropdown with ${data.stories.length} stories`);
                    
                    // Refresh node display
                    if (node.graph && node.graph.setDirtyCanvas) {
                        node.graph.setDirtyCanvas(true, false);
                    }
                }
            }
        } catch (error) {
            console.error('fb_tools -> StorySceneBatch: Error fetching story list:', error);
        }
    }
}
