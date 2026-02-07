/**
 * SceneUpdate Status Widget
 * Displays real-time processing status for SceneUpdate node
 */

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

/**
 * Setup SceneUpdate node with status display widget
 */
export function setupSceneUpdateStatus(nodeType, nodeData, app) {
    console.log("fbTools -> SceneUpdate status widget setup called");
    console.log("fbTools -> nodeType:", nodeType.comfyClass);
    console.log("fbTools -> nodeData:", nodeData);
    
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    
    nodeType.prototype.onNodeCreated = function() {
        console.log(`fbTools -> SceneUpdate onNodeCreated for node ID: ${this.id}`);
        const result = onNodeCreated?.apply(this, arguments);
        
        // Create status display container
        const container = document.createElement("div");
        container.style.cssText = `
            width: 100%;
            padding: 10px;
            background: #1a1a1a;
            border: 1px solid #444;
            border-radius: 4px;
            box-sizing: border-box;
            font-family: monospace;
            font-size: 12px;
            color: #888;
            min-height: 40px;
            display: flex;
            align-items: center;
        `;
        container.textContent = "Ready";
        
        // Add the DOM widget for status display
        const statusWidget = this.addDOMWidget("processing_status", "status", container, {
            serialize: false,
            hideOnZoom: false,
            getValue() {
                return container.textContent;
            },
            setValue(v) {
                // Don't restore - will update via websocket
            }
        });
        
        // Store references
        this._statusContainer = container;
        this._statusWidget = statusWidget;
        statusWidget.parentNode = this;
        
        console.log(`fbTools -> Added status widget to node ${this.id}`);
        
        // Compute widget size
        statusWidget.computeSize = function(width) {
            return [width, 60]; // Fixed height for status display
        };
        
        return result;
    };
}

// Listen for status updates from backend
api.addEventListener("fbtools.status", (event) => {
    console.log("fbTools: Received status update event:", event.detail);
    try {
        const { node: nodeId, status } = event.detail;
        
        console.log(`fbTools: Looking for node ${nodeId} with status: ${status}`);
        
        // Find the node by ID
        const node = app.graph._nodes.find(n => n.id == nodeId);
        if (node && node._statusContainer) {
            console.log(`fbTools: Found node ${nodeId}, updating status container`);
            const container = node._statusContainer;
            container.textContent = status;
            
            // Update colors based on status
            if (status.includes("Error")) {
                container.style.color = "#ff5555";
                container.style.borderColor = "#ff5555";
            } else if (status.includes("✓") || status.includes("Completed")) {
                container.style.color = "#55ff55";
                container.style.borderColor = "#55ff55";
            } else if (status === "Ready") {
                container.style.color = "#888";
                container.style.borderColor = "#444";
            } else {
                container.style.color = "#5599ff";
                container.style.borderColor = "#5599ff";
            }
            
            app.graph.setDirtyCanvas(true, false);
        }
    } catch (err) {
        console.error("fbTools: Error handling status message:", err);
    }
});
