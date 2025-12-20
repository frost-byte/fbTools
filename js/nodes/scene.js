/**
 * Scene-related node extensions
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
