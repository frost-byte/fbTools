/**
 * Tests for Libber API client
 */

import { LibberAPI } from "../js/api/libber.js";
import { mockFetch, createMockApp, createMockFn } from "./test_utils.js";

describe("LibberAPI", () => {
    beforeEach(() => {
        mockFetch.setup();
    });

    afterEach(() => {
        mockFetch.restore();
    });

    describe("createLibber", () => {
        test("should create a new libber", async () => {
            mockFetch.mockResponse({
                name: "test_libber",
                keys: [],
                delimiter: "%"
            });

            const api = new LibberAPI();
            const result = await api.createLibber("test_libber", "%", 10);

            expect(result.name).toBe("test_libber");
            expect(result.keys).toEqual([]);
            expect(result.delimiter).toBe("%");
        });

        test("should use default values", async () => {
            mockFetch.mockResponse({ name: "test", keys: [], delimiter: "%" });

            const api = new LibberAPI();
            await api.createLibber("test");

            const calls = mockFetch.getCalls();
            const body = JSON.parse(calls[0].body);
            expect(body.delimiter).toBe("%");
            expect(body.max_depth).toBe(10);
        });
    });

    describe("loadLibber", () => {
        test("should load a libber from file", async () => {
            mockFetch.mockResponse({
                name: "test_libber",
                keys: ["quality", "style"],
                lib_dict: {
                    quality: "best quality",
                    style: "realistic"
                }
            });

            const api = new LibberAPI();
            const result = await api.loadLibber("test_libber", "output/libbers/test_libber.json");

            expect(result.name).toBe("test_libber");
            expect(result.keys).toEqual(["quality", "style"]);
        });
    });

    describe("addLib", () => {
        test("should add a lib entry", async () => {
            mockFetch.mockResponse({
                name: "test_libber",
                keys: ["warrior"],
                lib_dict: { warrior: "strong fighter" }
            });

            const api = new LibberAPI();
            const result = await api.addLib("test_libber", "warrior", "strong fighter");

            expect(result.keys).toContain("warrior");
            expect(result.lib_dict.warrior).toBe("strong fighter");
        });

        test("should send correct request", async () => {
            mockFetch.mockResponse({ name: "test", keys: ["test"], lib_dict: {} });

            const api = new LibberAPI();
            await api.addLib("test_libber", "key", "value");

            const calls = mockFetch.getCalls();
            const body = JSON.parse(calls[0].body);
            expect(body.name).toBe("test_libber");
            expect(body.key).toBe("key");
            expect(body.value).toBe("value");
        });
    });

    describe("removeLib", () => {
        test("should remove a lib entry", async () => {
            mockFetch.mockResponse({
                name: "test_libber",
                keys: [],
                lib_dict: {}
            });

            const api = new LibberAPI();
            const result = await api.removeLib("test_libber", "warrior");

            expect(result.keys).toEqual([]);
        });
    });

    describe("saveLibber", () => {
        test("should save libber to file", async () => {
            mockFetch.mockResponse({
                status: "saved",
                filepath: "output/libbers/test_libber.json"
            });

            const api = new LibberAPI();
            const result = await api.saveLibber("test_libber", "output/libbers/test_libber.json");

            expect(result.status).toBe("saved");
            expect(result.filepath).toBe("output/libbers/test_libber.json");
        });
    });

    describe("listLibbers", () => {
        test("should list all available libbers", async () => {
            mockFetch.mockResponse({
                libbers: ["character_presets", "quality_settings"],
                files: ["character_presets.json", "quality_settings.json"]
            });

            const api = new LibberAPI();
            const result = await api.listLibbers();

            expect(result.libbers).toEqual(["character_presets", "quality_settings"]);
            expect(result.files).toEqual(["character_presets.json", "quality_settings.json"]);
        });
    });

    describe("getLibberData", () => {
        test("should get libber data", async () => {
            mockFetch.mockResponse({
                lib_dict: {
                    warrior: "strong fighter",
                    quality: "best quality"
                },
                delimiter: "%",
                max_depth: 10
            });

            const api = new LibberAPI();
            const result = await api.getLibberData("test_libber");

            expect(result.lib_dict.warrior).toBe("strong fighter");
            expect(result.delimiter).toBe("%");
        });
    });

    describe("applySubstitutions", () => {
        test("should apply substitutions to text", async () => {
            mockFetch.mockResponse({
                result: "A strong fighter with best quality"
            });

            const api = new LibberAPI();
            const result = await api.applySubstitutions(
                "test_libber",
                "A %warrior% with %quality%",
                true
            );

            expect(result.result).toBe("A strong fighter with best quality");
        });

        test("should send correct request", async () => {
            mockFetch.mockResponse({ result: "test" });

            const api = new LibberAPI();
            await api.applySubstitutions("test_libber", "text");

            const calls = mockFetch.getCalls();
            const body = JSON.parse(calls[0].body);
            expect(body.name).toBe("test_libber");
            expect(body.text).toBe("text");
        });
    });

    describe("error handling", () => {
        test("should handle network errors", async () => {
            mockFetch.mockError(500, "Internal Server Error");

            const api = new LibberAPI();
            await expect(api.createLibber("test")).rejects.toThrow();
        });

        test("should handle API errors", async () => {
            mockFetch.mockError(404, "Not Found");

            const api = new LibberAPI();
            await expect(api.getLibberData("nonexistent")).rejects.toThrow();
        });
    });

    describe("showSuccess", () => {
        test("should show success toast", () => {
            const app = createMockApp();
            const api = new LibberAPI();
            
            api.showSuccess("Success", "Libber created", app);

            expect(app.extensionManager.toast.add.calls.length).toBeGreaterThan(0);
            const call = app.extensionManager.toast.add.calls[0][0];
            expect(call.severity).toBe("success");
            expect(call.summary).toBe("Success");
        });
    });

    describe("handleError", () => {
        test("should log and show error toast", () => {
            const app = createMockApp();
            const api = new LibberAPI();
            const error = new Error("Test error");
            
            // Mock console.error
            const originalError = console.error;
            console.error = createMockFn();
            
            api.handleError(error, "Create Libber", app);

            expect(console.error.calls.length).toBeGreaterThan(0);
            expect(app.extensionManager.toast.add.calls.length).toBeGreaterThan(0);
            const call = app.extensionManager.toast.add.calls[0][0];
            expect(call.severity).toBe("error");
            
            // Restore console.error
            console.error = originalError;
        });
    });
});

/**
 * Integration tests for realistic workflows
 */
describe("LibberAPI Integration", () => {
    beforeEach(() => {
        mockFetch.setup();
    });

    afterEach(() => {
        mockFetch.restore();
    });

    test("complete workflow: create -> add libs -> save -> load", async () => {
        const api = new LibberAPI();

        // Create
        mockFetch.mockResponse({ name: "test", keys: [], delimiter: "%" });
        await api.createLibber("test");

        // Add lib 1
        mockFetch.mockResponse({
            name: "test",
            keys: ["quality"],
            lib_dict: { quality: "best quality" }
        });
        await api.addLib("test", "quality", "best quality");

        // Add lib 2
        mockFetch.mockResponse({
            name: "test",
            keys: ["quality", "warrior"],
            lib_dict: {
                quality: "best quality",
                warrior: "strong fighter"
            }
        });
        await api.addLib("test", "warrior", "strong fighter");

        // Save
        mockFetch.mockResponse({ status: "saved" });
        await api.saveLibber("test", "output/libbers/test.json");

        // Load
        mockFetch.mockResponse({
            name: "test",
            keys: ["quality", "warrior"],
            lib_dict: {
                quality: "best quality",
                warrior: "strong fighter"
            }
        });
        const result = await api.loadLibber("test", "output/libbers/test.json");

        expect(result.keys).toEqual(["quality", "warrior"]);
        expect(result.lib_dict.quality).toBe("best quality");
    });

    test("substitution workflow", async () => {
        const api = new LibberAPI();

        // Create with initial data
        mockFetch.mockResponse({ name: "test", keys: ["a"], delimiter: "%" });
        await api.createLibber("test");

        // Add nested references - each addLib call needs its own response
        mockFetch.mockResponse({ name: "test", keys: ["quality"] });
        await api.addLib("test", "quality", "best quality");
        
        mockFetch.mockResponse({ name: "test", keys: ["quality", "base"] });
        await api.addLib("test", "base", "%quality%, detailed");

        // Apply substitutions
        mockFetch.mockResponse({
            result: "Image with best quality, detailed",
            original: "Image with %base%",
            info: "Substitution complete"
        });
        const result = await api.applySubstitutions("test", "Image with %base%");

        expect(result).toBeDefined();
        expect(result.result).toBeDefined();
        expect(result.result).toContain("quality");
    });
});
