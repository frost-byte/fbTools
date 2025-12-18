/**
 * Widget utility functions for updating node widgets from API responses.
 */

/**
 * Update a widget's value from message.text array
 * @param {object} node - The node instance
 * @param {Array} textArray - The message.text array
 * @param {number} index - Index into the text array
 * @param {string} widgetName - Name of the widget to update
 * @param {string} widgetType - Type of the widget (default: "text")
 * @param {string} logPrefix - Prefix for console log messages
 * @returns {boolean} True if widget was updated
 */
export function updateWidgetFromText(
    node,
    textArray,
    index,
    widgetName,
    widgetType = "text",
    logPrefix = "fbTools"
) {
    if (textArray && textArray[index]) {
        const widget = node.widgets.find((w) => w.name === widgetName);
        if (widget) {
            console.log(
                `${logPrefix}: Updating widget '${widgetName}' from text[${index}]`
            );

            if (widgetType === "combo" && typeof textArray[index] === "string") {
                try {
                    const optionsJson = JSON.parse(textArray[index]);
                    if (Array.isArray(optionsJson)) {
                        const currentOptions =
                            (widget.options && widget.options.values) ||
                            widget.options ||
                            [];
                        const sameOptions =
                            Array.isArray(currentOptions) &&
                            currentOptions.length === optionsJson.length &&
                            currentOptions.every((val, idx) => val === optionsJson[idx]);

                        if (!sameOptions) {
                            if (widget.options && typeof widget.options === "object") {
                                widget.options.values = optionsJson;
                            } else {
                                widget.options = optionsJson;
                            }
                            widget.options_values = optionsJson;
                        }

                        const nextValue = optionsJson.includes(widget.value)
                            ? widget.value
                            : optionsJson[0];
                        if (widget.value !== nextValue) {
                            widget.value = nextValue;
                        }

                        if (widget.inputEl && widget.inputEl.tagName === "SELECT") {
                            widget.inputEl.innerHTML = "";
                            optionsJson.forEach((opt) => {
                                const option = document.createElement("option");
                                option.value = opt;
                                option.textContent = opt;
                                option.selected = opt === widget.value;
                                widget.inputEl.appendChild(option);
                            });
                        }
                    }
                } catch (err) {
                    console.warn(`${logPrefix}: Failed to parse combo options:`, err);
                }
            } else {
                widget.value = textArray[index];
                if (widget.inputEl) {
                    widget.inputEl.value = textArray[index];
                }
            }
            return true;
        }
    }
    return false;
}

/**
 * Bulk update widgets for a node based on widget map
 * @param {object} node - The node instance
 * @param {Array} textArray - The message.text array
 * @param {Array} widgetMap - Array of {widget_index, widget_name, widget_type}
 * @param {string} logPrefix - Prefix for console log messages
 */
export function updateNodeWidgets(node, textArray, widgetMap, logPrefix = "fbTools") {
    if (!widgetMap || !widgetMap.length) return;

    widgetMap.forEach(({ widget_index, widget_name, widget_type }) => {
        updateWidgetFromText(
            node,
            textArray,
            widget_index,
            widget_name,
            widget_type,
            logPrefix
        );
    });
}

/**
 * Schedule node resize/refresh after widget updates
 * @param {object} node - The node instance
 * @param {object} app - ComfyUI app instance
 */
export function scheduleNodeRefresh(node, app) {
    requestAnimationFrame(() => {
        const sz = node.computeSize();
        if (sz[0] < node.size[0]) sz[0] = node.size[0];
        if (sz[1] < node.size[1]) sz[1] = node.size[1];
        node.onResize?.(sz);
        app.graph.setDirtyCanvas(true, false);
    });
}
