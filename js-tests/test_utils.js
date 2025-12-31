/**
 * Test setup and utilities for fbTools frontend tests.
 * 
 * Usage:
 *   1. Install testing dependencies: npm install --save-dev jest jest-environment-jsdom
 *   2. Run tests: npm test
 * 
 * Example test structure:
 * 
 * import { promptCollectionAPI } from "../api/prompt_collection.js";
 * import { mockFetch } from "./test_utils.js";
 * 
 * describe("PromptCollectionAPI", () => {
 *     beforeEach(() => {
 *         mockFetch.setup();
 *     });
 * 
 *     afterEach(() => {
 *         mockFetch.restore();
 *     });
 * 
 *     test("should create session", async () => {
 *         mockFetch.mockResponse({ session_id: "test123", collection: {} });
 *         const result = await promptCollectionAPI.createSession();
 *         expect(result.session_id).toBe("test123");
 *     });
 * });
 */

/**
 * Mock fetch API for testing
 */
export const mockFetch = {
    originalFetch: null,
    mockResponses: [],
    currentIndex: 0,
    calls: [],

    setup() {
        this.originalFetch = global.fetch;
        this.mockResponses = [];
        this.currentIndex = 0;
        this.calls = [];

        global.fetch = async (url, options) => {
            // Record the call
            this.calls.push({ url, options });

            const response = this.mockResponses[this.currentIndex] || {
                ok: true,
                status: 200,
                json: async () => ({}),
                text: async () => "",
            };
            this.currentIndex++;

            return {
                ok: response.ok ?? true,
                status: response.status ?? 200,
                statusText: response.statusText ?? "OK",
                json: async () => response.json || response,
                text: async () => response.text || JSON.stringify(response),
            };
        };
    },

    restore() {
        if (this.originalFetch) {
            global.fetch = this.originalFetch;
        }
        this.mockResponses = [];
        this.currentIndex = 0;
        this.calls = [];
    },

    mockResponse(data) {
        this.mockResponses.push(data);
    },

    mockError(statusCode, message) {
        this.mockResponses.push({
            ok: false,
            status: statusCode,
            statusText: message,
            json: async () => ({ error: message }),
            text: async () => message,
        });
    },

    getCalls() {
        return this.calls.map(call => ({
            url: call.url,
            options: call.options,
            body: call.options?.body,
        }));
    },
};

/**
 * Create a mock function (replacement for jest.fn())
 */
export function createMockFn() {
    const calls = [];
    const fn = function(...args) {
        calls.push(args);
        return fn.mockReturnValue;
    };
    fn.calls = calls;
    fn.mockReturnValue = undefined;
    fn.mock = {
        calls: calls,
    };
    return fn;
}

/**
 * Mock ComfyUI app instance for testing
 */
export function createMockApp() {
    return {
        extensionManager: {
            toast: {
                add: createMockFn(),
            },
        },
        graph: {
            setDirtyCanvas: createMockFn(),
        },
    };
}

/**
 * Mock ComfyUI node instance for testing
 */
export function createMockNode(widgets = []) {
    return {
        widgets: widgets.map((w) => ({
            name: w.name,
            value: w.value || "",
            options: w.options || null,
            inputEl: w.inputEl || null,
        })),
        size: [200, 100],
        computeSize: createMockFn(),
        onResize: createMockFn(),
    };
}

/**
 * Create a mock widget
 */
export function createMockWidget(name, value = "", options = null) {
    return {
        name,
        value,
        options,
        inputEl: null,
    };
}

/**
 * Assert that a toast was shown
 */
export function expectToast(app, severity, summary) {
    const calls = app.extensionManager.toast.add.calls;
    const matchingCall = calls.find(args => 
        args[0]?.severity === severity && args[0]?.summary === summary
    );
    expect(matchingCall).toBeDefined();
}

/**
 * Assert that an error toast was shown
 */
export function expectErrorToast(app, summary) {
    expectToast(app, "error", summary);
}

/**
 * Assert that a success toast was shown
 */
export function expectSuccessToast(app, summary) {
    expectToast(app, "success", summary);
}
