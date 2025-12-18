# Integration Guide: Using the New Modular Architecture

This guide shows how to integrate the new API clients into existing node handlers in `fb_tools.js`.

## Quick Start

### 1. Import the modules you need

At the top of `fb_tools.js`, add imports:

```javascript
import { promptCollectionAPI } from "./api/prompt_collection.js";
import { sceneAPI } from "./api/scene.js";
import { libberAPI } from "./api/libber.js";
import { storyAPI } from "./api/story.js";
import { updateWidgetFromText, scheduleNodeRefresh } from "./utils/widgets.js";
```

### 2. Use in node handlers

Replace direct fetch calls with API client methods:

```javascript
// ❌ OLD WAY - Direct fetch
const response = await fetch("/fbtools/prompts/create", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ collection: data }),
});
const result = await response.json();

// ✅ NEW WAY - API client
const result = await promptCollectionAPI.createSession(data);
```

## Example: PromptCollectionEdit Node

Complete example showing how to add a PromptCollectionEdit node with REST API integration:

```javascript
// In beforeRegisterNodeDef callback
if (isNode("PromptCollectionEdit")) {
    // Handle execute response
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function(message) {
        const r = onExecuted?.apply(this, arguments);
        
        if (message?.text) {
            // Update collection JSON widget
            updateWidgetFromText(
                this, 
                message.text, 
                0, 
                "prompt_collection_json_in"
            );
            
            // Update prompt names dropdown
            if (message.text[1]) {
                try {
                    const promptNames = JSON.parse(message.text[1]);
                    const widget = this.widgets.find(w => w.name === "prompt_name");
                    if (widget && Array.isArray(promptNames)) {
                        widget.options.values = promptNames;
                        widget.value = promptNames[0] || "";
                    }
                } catch (err) {
                    console.error("Failed to parse prompt names:", err);
                }
            }
            
            scheduleNodeRefresh(this, app);
        }
        
        return r;
    };
    
    // Add button widget for API operations
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function() {
        const r = onNodeCreated?.apply(this, arguments);
        
        const executeButton = this.addWidget(
            "button",
            "Execute Operation",
            null,
            async () => {
                // Get widget values
                const sessionIdWidget = this.widgets.find(w => w.name === "session_id");
                const operationWidget = this.widgets.find(w => w.name === "operation");
                const keyWidget = this.widgets.find(w => w.name === "prompt_name");
                const valueWidget = this.widgets.find(w => w.name === "prompt_value");
                const categoryWidget = this.widgets.find(w => w.name === "prompt_category");
                const descriptionWidget = this.widgets.find(w => w.name === "prompt_description");
                
                try {
                    // Create session if needed
                    if (!sessionIdWidget?.value) {
                        const sessionResult = await promptCollectionAPI.createSession();
                        sessionIdWidget.value = sessionResult.session_id;
                    }
                    
                    const operation = operationWidget?.value || "add";
                    const sessionId = sessionIdWidget.value;
                    
                    let result;
                    switch (operation) {
                        case "add":
                        case "update":
                            result = await promptCollectionAPI.addPrompt(
                                sessionId,
                                keyWidget?.value || "new_prompt",
                                valueWidget?.value || "",
                                {
                                    category: categoryWidget?.value,
                                    description: descriptionWidget?.value,
                                }
                            );
                            break;
                            
                        case "remove":
                            result = await promptCollectionAPI.removePrompt(
                                sessionId,
                                keyWidget?.value
                            );
                            break;
                            
                        case "list":
                            result = await promptCollectionAPI.listPromptNames(sessionId);
                            break;
                    }
                    
                    // Update widgets with results
                    if (result?.collection) {
                        const collectionWidget = this.widgets.find(
                            w => w.name === "prompt_collection_json_in"
                        );
                        if (collectionWidget) {
                            collectionWidget.value = JSON.stringify(
                                result.collection,
                                null,
                                2
                            );
                        }
                    }
                    
                    if (result?.prompt_names) {
                        const namesWidget = this.widgets.find(
                            w => w.name === "prompt_name"
                        );
                        if (namesWidget) {
                            namesWidget.options.values = result.prompt_names;
                            if (!result.prompt_names.includes(namesWidget.value)) {
                                namesWidget.value = result.prompt_names[0] || "";
                            }
                        }
                    }
                    
                    scheduleNodeRefresh(this, app);
                    
                    promptCollectionAPI.showSuccess(
                        "Success",
                        `${operation} completed`,
                        app
                    );
                    
                } catch (err) {
                    promptCollectionAPI.handleError(err, operation, app);
                }
            }
        );
        
        return r;
    };
}
```

## Example: SceneEdit Node with REST API

```javascript
if (isNode("SceneEdit")) {
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function() {
        const r = onNodeCreated?.apply(this, arguments);
        
        // Add button for updating prompts via REST
        const updatePromptsButton = this.addWidget(
            "button",
            "Update Prompts",
            null,
            async () => {
                const sessionIdWidget = this.widgets.find(w => w.name === "session_id");
                const promptsWidget = this.widgets.find(w => w.name === "prompts_json");
                
                try {
                    const prompts = JSON.parse(promptsWidget?.value || "{}");
                    
                    const result = await sceneAPI.updatePrompts(
                        sessionIdWidget.value,
                        prompts
                    );
                    
                    sceneAPI.showSuccess("Success", "Prompts updated", app);
                    scheduleNodeRefresh(this, app);
                    
                } catch (err) {
                    sceneAPI.handleError(err, "Update Prompts", app);
                }
            }
        );
        
        // Add button for saving metadata
        const saveButton = this.addWidget(
            "button",
            "Save Metadata",
            null,
            async () => {
                const sessionIdWidget = this.widgets.find(w => w.name === "session_id");
                
                try {
                    await sceneAPI.saveMetadata(sessionIdWidget.value);
                    sceneAPI.showSuccess("Success", "Metadata saved", app);
                } catch (err) {
                    sceneAPI.handleError(err, "Save Metadata", app);
                }
            }
        );
        
        return r;
    };
}
```

## Example: LibberEdit Node Refactor

Replace complex callback manipulation with simple REST calls:

```javascript
if (isNode("LibberEdit")) {
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function() {
        const r = onNodeCreated?.apply(this, arguments);
        
        // Add operation button
        const executeButton = this.addWidget(
            "button",
            "Execute",
            null,
            async () => {
                const sessionIdWidget = this.widgets.find(w => w.name === "libber_session_id");
                const operationWidget = this.widgets.find(w => w.name === "operation");
                const keyWidget = this.widgets.find(w => w.name === "key");
                const valueWidget = this.widgets.find(w => w.name === "value");
                
                try {
                    // Create session if needed
                    if (!sessionIdWidget?.value) {
                        const initialData = this.widgets.find(
                            w => w.name === "libber_json_in"
                        )?.value;
                        const parsed = initialData ? JSON.parse(initialData) : null;
                        const session = await libberAPI.createSession(parsed);
                        sessionIdWidget.value = session.session_id;
                    }
                    
                    const operation = operationWidget?.value || "add";
                    const sessionId = sessionIdWidget.value;
                    
                    let result;
                    switch (operation) {
                        case "add":
                            result = await libberAPI.addLib(
                                sessionId,
                                keyWidget?.value,
                                valueWidget?.value
                            );
                            break;
                            
                        case "remove":
                            result = await libberAPI.removeLib(
                                sessionId,
                                keyWidget?.value
                            );
                            break;
                            
                        case "list":
                            result = await libberAPI.getKeys(sessionId);
                            break;
                    }
                    
                    // Update key selector dropdown
                    if (result?.keys) {
                        const keySelectorWidget = this.widgets.find(
                            w => w.name === "key_selector"
                        );
                        if (keySelectorWidget) {
                            keySelectorWidget.options.values = result.keys;
                        }
                    }
                    
                    scheduleNodeRefresh(this, app);
                    libberAPI.showSuccess("Success", `${operation} completed`, app);
                    
                } catch (err) {
                    libberAPI.handleError(err, operation, app);
                }
            }
        );
        
        return r;
    };
}
```

## Benefits of This Approach

### Before (Direct Fetch)
```javascript
// ❌ Scattered fetch calls, no error handling, hard to test
const response = await fetch("/fbtools/prompts/create", {...});
if (!response.ok) {
    console.error("Failed!");
    // No user feedback
}
const result = await response.json();
```

### After (API Client)
```javascript
// ✅ Centralized, error handling included, testable
try {
    const result = await promptCollectionAPI.createSession();
} catch (err) {
    promptCollectionAPI.handleError(err, "Create Session", app);
}
```

## Testing Your Integration

1. **Unit Tests**: Test API clients in isolation
   ```bash
   cd js/
   npm test
   ```

2. **Integration**: Test in ComfyUI
   - Create a workflow with your nodes
   - Click buttons that call API methods
   - Verify widgets update correctly
   - Check console for errors

3. **Error Handling**: Test error scenarios
   - Disconnect network (DevTools → Network → Offline)
   - Verify error toasts appear
   - Verify console logs are helpful

## Migration Checklist

- [ ] Create API client modules (✅ Done!)
- [ ] Add imports to fb_tools.js
- [ ] Identify nodes that make REST calls
- [ ] Replace direct fetch with API client methods
- [ ] Test each node in ComfyUI
- [ ] Write unit tests for critical paths
- [ ] Remove old fetch code
- [ ] Update documentation

## Next Steps

1. **Start Small**: Pick one node (e.g., PromptCollectionEdit)
2. **Test Thoroughly**: Verify it works in ComfyUI
3. **Iterate**: Move to next node
4. **Refactor**: Extract common patterns to utilities
5. **Document**: Add comments for complex integrations
