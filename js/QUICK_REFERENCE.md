# Quick Reference: API Clients

## Import

```javascript
import { promptCollectionAPI } from "./api/prompt_collection.js";
import { sceneAPI } from "./api/scene.js";
import { libberAPI } from "./api/libber.js";
import { storyAPI } from "./api/story.js";
import { updateWidgetFromText, scheduleNodeRefresh } from "./utils/widgets.js";
```

## PromptCollectionAPI

```javascript
// Create session
const { session_id, collection } = await promptCollectionAPI.createSession();
const { session_id } = await promptCollectionAPI.createSession(initialData);

// Add prompt
const result = await promptCollectionAPI.addPrompt(
    sessionId,
    "girl_pos",
    "beautiful woman smiling"
);

// Add with metadata
const result = await promptCollectionAPI.addPrompt(
    sessionId,
    "key",
    "value",
    { category: "scene", description: "desc", tags: ["tag1"] }
);

// Remove prompt
const result = await promptCollectionAPI.removePrompt(sessionId, "key");

// List names
const { prompt_names } = await promptCollectionAPI.listPromptNames(sessionId);

// Get collection
const { collection } = await promptCollectionAPI.getCollection(sessionId);
```

## SceneAPI (Stub)

```javascript
// Update prompts
await sceneAPI.updatePrompts(sessionId, promptsData);

// Update metadata
await sceneAPI.updateMetadata(sessionId, { pose_name: "pose1", resolution: "1024x1024" });

// Save metadata
await sceneAPI.saveMetadata(sessionId);

// List scenes
const { scenes } = await sceneAPI.listScenes(storyDir);

// Update loras
await sceneAPI.updateLoras(sessionId, lorasData);
```

## LibberAPI (Stub)

```javascript
// Create session
const { session_id, libber } = await libberAPI.createSession();

// Add lib
const result = await libberAPI.addLib(sessionId, "key", "value");

// Remove lib
const result = await libberAPI.removeLib(sessionId, "key");

// Get keys
const { keys } = await libberAPI.getKeys(sessionId);

// Get lib
const { key, value } = await libberAPI.getLib(sessionId, "key");

// Apply substitutions
const { result } = await libberAPI.applySubstitutions(sessionId, "text with {{placeholders}}");
```

## StoryAPI (Stub)

```javascript
// Load story
const { story, scenes } = await storyAPI.loadStory(storyDir);

// Save story
await storyAPI.saveStory(storyDir, storyData);

// List stories
const { stories } = await storyAPI.listStories(baseDir);

// Get scene order
const { scenes } = await storyAPI.getSceneOrder(storyDir);

// Update scene order
await storyAPI.updateSceneOrder(storyDir, sceneOrderArray);
```

## Error Handling

```javascript
try {
    const result = await promptCollectionAPI.addPrompt(...);
} catch (err) {
    // Shows error toast + logs to console
    promptCollectionAPI.handleError(err, "Add Prompt", app);
}
```

## Success Toast

```javascript
promptCollectionAPI.showSuccess("Success", "Operation completed", app);
```

## Widget Utilities

```javascript
// Update single widget
updateWidgetFromText(node, textArray, index, "widget_name");
updateWidgetFromText(node, textArray, index, "combo_widget", "combo");

// Bulk update widgets
const widgetMap = [
    { widget_index: 0, widget_name: "prompt_collection_json_in" },
    { widget_index: 1, widget_name: "prompt_name", widget_type: "combo" },
];
updateNodeWidgets(node, textArray, widgetMap, "PromptCollection");

// Refresh node
scheduleNodeRefresh(node, app);
```

## Testing

```javascript
// test_example.test.js
import { PromptCollectionAPI } from "../api/prompt_collection.js";
import { mockFetch, createMockApp } from "./test_utils.js";

describe("PromptCollectionAPI", () => {
    beforeEach(() => mockFetch.setup());
    afterEach(() => mockFetch.restore());

    test("should create session", async () => {
        mockFetch.mockResponse({ session_id: "test123", collection: {} });
        const api = new PromptCollectionAPI();
        const result = await api.createSession();
        expect(result.session_id).toBe("test123");
    });
});
```

## Button Widget Pattern

```javascript
const executeButton = node.addWidget("button", "Execute", null, async () => {
    const sessionWidget = node.widgets.find(w => w.name === "session_id");
    const keyWidget = node.widgets.find(w => w.name === "key");
    const valueWidget = node.widgets.find(w => w.name === "value");
    
    try {
        if (!sessionWidget.value) {
            const session = await promptCollectionAPI.createSession();
            sessionWidget.value = session.session_id;
        }
        
        const result = await promptCollectionAPI.addPrompt(
            sessionWidget.value,
            keyWidget.value,
            valueWidget.value
        );
        
        // Update widgets with result
        updateWidgetFromText(node, [JSON.stringify(result.collection)], 0, "collection_json");
        scheduleNodeRefresh(node, app);
        
        promptCollectionAPI.showSuccess("Success", "Prompt added", app);
    } catch (err) {
        promptCollectionAPI.handleError(err, "Add Prompt", app);
    }
});
```

## Complete Node Handler Example

```javascript
if (isNode("PromptCollectionEdit")) {
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function(message) {
        const r = onExecuted?.apply(this, arguments);
        if (message?.text) {
            updateWidgetFromText(this, message.text, 0, "prompt_collection_json_in");
            updateWidgetFromText(this, message.text, 1, "prompt_name", "combo");
            scheduleNodeRefresh(this, app);
        }
        return r;
    };
    
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function() {
        const r = onNodeCreated?.apply(this, arguments);
        
        this.addWidget("button", "Execute", null, async () => {
            const sessionId = this.widgets.find(w => w.name === "session_id")?.value;
            const operation = this.widgets.find(w => w.name === "operation")?.value;
            const key = this.widgets.find(w => w.name === "prompt_name")?.value;
            const value = this.widgets.find(w => w.name === "prompt_value")?.value;
            
            try {
                let result;
                switch (operation) {
                    case "add": result = await promptCollectionAPI.addPrompt(sessionId, key, value); break;
                    case "remove": result = await promptCollectionAPI.removePrompt(sessionId, key); break;
                    case "list": result = await promptCollectionAPI.listPromptNames(sessionId); break;
                }
                
                if (result?.collection) {
                    this.widgets.find(w => w.name === "prompt_collection_json_in").value = 
                        JSON.stringify(result.collection, null, 2);
                }
                if (result?.prompt_names) {
                    this.widgets.find(w => w.name === "prompt_name").options.values = result.prompt_names;
                }
                
                scheduleNodeRefresh(this, app);
                promptCollectionAPI.showSuccess("Success", `${operation} completed`, app);
            } catch (err) {
                promptCollectionAPI.handleError(err, operation, app);
            }
        });
        
        return r;
    };
}
```
