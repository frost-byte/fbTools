/**
 * Example tests for PromptCollectionAPI
 * 
 * To run tests:
 *   1. npm install --save-dev jest @testing-library/dom
 *   2. Add to package.json: "test": "jest"
 *   3. npm test
 */

import { PromptCollectionAPI } from "../js/api/prompt_collection.js";
import {
    mockFetch,
    createMockApp,
    expectSuccessToast,
    expectErrorToast,
} from "./test_utils.js";

describe("PromptCollectionAPI", () => {
    let api;
    let mockApp;

    beforeEach(() => {
        api = new PromptCollectionAPI();
        mockApp = createMockApp();
        mockFetch.setup();
    });

    afterEach(() => {
        mockFetch.restore();
    });

    describe("createSession", () => {
        test("should create a new session", async () => {
            const mockResponse = {
                session_id: "test_session_123",
                collection: { version: 2, prompts: {} },
            };
            mockFetch.mockResponse(mockResponse);

            const result = await api.createSession();

            expect(result.session_id).toBe("test_session_123");
            expect(result.collection).toBeDefined();
            expect(result.collection.version).toBe(2);
        });

        test("should create session with initial data", async () => {
            const initialData = {
                version: 2,
                prompts: {
                    test: { value: "test prompt" },
                },
            };
            const mockResponse = {
                session_id: "test_session_123",
                collection: initialData,
            };
            mockFetch.mockResponse(mockResponse);

            const result = await api.createSession(initialData);

            expect(result.collection.prompts.test).toBeDefined();
        });

        test("should handle API errors", async () => {
            mockFetch.mockError(500, "Internal Server Error");

            await expect(api.createSession()).rejects.toThrow(
                "POST /create failed"
            );
        });
    });

    describe("addPrompt", () => {
        test("should add a prompt to collection", async () => {
            const mockResponse = {
                collection: {
                    version: 2,
                    prompts: {
                        test_key: { value: "test value" },
                    },
                },
                prompt_names: ["test_key"],
            };
            mockFetch.mockResponse(mockResponse);

            const result = await api.addPrompt(
                "session123",
                "test_key",
                "test value"
            );

            expect(result.prompt_names).toContain("test_key");
            expect(result.collection.prompts.test_key.value).toBe("test value");
        });

        test("should add prompt with metadata", async () => {
            const mockResponse = {
                collection: {
                    version: 2,
                    prompts: {
                        test_key: {
                            value: "test value",
                            category: "scene",
                            description: "Test description",
                            tags: ["test", "example"],
                        },
                    },
                },
                prompt_names: ["test_key"],
            };
            mockFetch.mockResponse(mockResponse);

            const result = await api.addPrompt("session123", "test_key", "test value", {
                category: "scene",
                description: "Test description",
                tags: ["test", "example"],
            });

            const prompt = result.collection.prompts.test_key;
            expect(prompt.category).toBe("scene");
            expect(prompt.description).toBe("Test description");
            expect(prompt.tags).toEqual(["test", "example"]);
        });
    });

    describe("removePrompt", () => {
        test("should remove a prompt from collection", async () => {
            const mockResponse = {
                collection: {
                    version: 2,
                    prompts: {},
                },
                prompt_names: [],
            };
            mockFetch.mockResponse(mockResponse);

            const result = await api.removePrompt("session123", "test_key");

            expect(result.prompt_names).not.toContain("test_key");
        });
    });

    describe("listPromptNames", () => {
        test("should list all prompt names", async () => {
            const mockResponse = {
                prompt_names: ["prompt1", "prompt2", "prompt3"],
            };
            mockFetch.mockResponse(mockResponse);

            const result = await api.listPromptNames("session123");

            expect(result.prompt_names).toHaveLength(3);
            expect(result.prompt_names).toContain("prompt1");
        });
    });

    describe("error handling", () => {
        test("should show error toast on failure", async () => {
            // Suppress console.error for this test since we're intentionally triggering an error
            const originalError = console.error;
            console.error = () => {}; // No-op function to suppress output
            
            mockFetch.mockError(404, "Not Found");

            try {
                await api.createSession();
            } catch (error) {
                api.handleError(error, "Create Session", mockApp);
            }

            expectErrorToast(mockApp, "Create Session Failed");
            
            // Restore console.error
            console.error = originalError;
        });

        test("should show success toast", () => {
            api.showSuccess("Success", "Operation completed", mockApp);
            expectSuccessToast(mockApp, "Success");
        });
    });
});
