# fbTools Frontend Architecture

Modular JavaScript architecture for ComfyUI fbTools extension with testable API clients and utilities.

## Directory Structure

```
js/
├── fb_tools.js              # Main extension registration (imports from modules)
├── package.json             # NPM configuration for testing
├── api/                     # REST API client modules
│   ├── prompt_collection.js # PromptCollection API
│   ├── scene.js            # Scene API (stub)
│   ├── libber.js           # Libber API (stub)
│   └── story.js            # Story API (stub)
├── ui/                      # UI component modules
│   └── (future node-specific UI handlers)
├── utils/                   # Shared utilities
│   ├── api_base.js         # Base API class with error handling
│   └── widgets.js          # Widget update utilities
└── tests/                   # Test files
    ├── test_utils.js       # Testing utilities and mocks
    └── prompt_collection_api.test.js  # Example test file
```

## API Modules

Each API module exports:
- **Class**: Full API client class (e.g., `PromptCollectionAPI`)
- **Singleton**: Pre-instantiated instance (e.g., `promptCollectionAPI`)

### Example Usage

```javascript
import { promptCollectionAPI } from "./api/prompt_collection.js";

// Create a new session
const session = await promptCollectionAPI.createSession();

// Add a prompt
const result = await promptCollectionAPI.addPrompt(
    session.session_id,
    "girl_pos",
    "beautiful woman smiling",
    { category: "character", tags: ["female", "portrait"] }
);

// Handle errors
try {
    await promptCollectionAPI.addPrompt(session.session_id, "key", "value");
} catch (error) {
    promptCollectionAPI.handleError(error, "Add Prompt", app);
}
```

## Testing

### Setup

```bash
cd js/
npm install
```

### Run Tests

```bash
# Run all tests
npm test

# Watch mode (re-run on changes)
npm run test:watch

# Generate coverage report
npm run test:coverage
```

### Writing Tests

```javascript
import { PromptCollectionAPI } from "../api/prompt_collection.js";
import { mockFetch, createMockApp } from "./test_utils.js";

describe("PromptCollectionAPI", () => {
    beforeEach(() => mockFetch.setup());
    afterEach(() => mockFetch.restore());

    test("should create session", async () => {
        mockFetch.mockResponse({ session_id: "test123" });
        const api = new PromptCollectionAPI();
        const result = await api.createSession();
        expect(result.session_id).toBe("test123");
    });
});
```

## Integration with fb_tools.js

The main `fb_tools.js` file imports and uses these modules:

```javascript
import { promptCollectionAPI } from "./api/prompt_collection.js";
import { updateWidgetFromText, scheduleNodeRefresh } from "./utils/widgets.js";

// In node handler
if (isNode("PromptCollectionEdit")) {
    // Add button widget
    const executeButton = node.addWidget("button", "Execute", null, async () => {
        try {
            const result = await promptCollectionAPI.addPrompt(
                sessionId, key, value
            );
            // Update widgets with result
            updateWidgetFromText(node, [JSON.stringify(result.collection)], 0, "collection_json");
            scheduleNodeRefresh(node, app);
        } catch (err) {
            promptCollectionAPI.handleError(err, "Add Prompt", app);
        }
    });
}
```

## Benefits

### ✅ Testability
- API clients can be tested independently with mocked fetch
- No ComfyUI dependencies required for unit tests
- Fast test execution (~milliseconds)

### ✅ Maintainability
- Each API class in its own file
- Clear separation of concerns
- Easy to find and update code

### ✅ Reusability
- Singleton instances for convenience
- Classes can be instantiated for testing
- Utilities shared across multiple nodes

### ✅ Type Safety (Future)
- Easy to add TypeScript definitions
- JSDoc comments provide IntelliSense
- Clear interface contracts

## Future Enhancements

1. **TypeScript Migration**: Add `.d.ts` files for type safety
2. **UI Modules**: Extract node-specific UI handlers to `ui/` directory
3. **E2E Tests**: Add integration tests with ComfyUI
4. **Build Pipeline**: Add bundling/minification for production
5. **Documentation**: Auto-generate API docs from JSDoc comments

## Migration Path

The existing `fb_tools.js` can be gradually refactored:

1. **Phase 1** (Current): API clients extracted, ready for use
2. **Phase 2**: Update node handlers to use API clients
3. **Phase 3**: Extract node handlers to `ui/` modules
4. **Phase 4**: Remove deprecated code from main file

No breaking changes required - new modules can coexist with legacy code.
