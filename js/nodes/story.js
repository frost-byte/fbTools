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
 * Setup StoryEdit node extensions
 */
export function setupStoryEdit(nodeType, nodeData, app) {
    console.log("fb_tools -> StoryEdit node detected");
    
    const widgetMap = [
        { widget_index: 1, widget_name: "selected_prompt_in" },
        { widget_index: 2, widget_name: "story_json_in" },
        { widget_index: 4, widget_name: "story_scene_selector", widget_type: "combo" },
    ];
    
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

        // StoryEdit returns: text[0]=scene_list, text[1]=selected_prompt, text[2]=story_json, text[3]=selector_options
        const textArray = message?.text;

        updateNodeInputs(this, textArray, widgetMap, `fbTools -> ${nodeData.name}`);
        console.log("fbTools -> StoryEdit: updating scene selector options");
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
