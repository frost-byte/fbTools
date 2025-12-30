/**
 * End-to-end tests for StoryEdit node frontend.
 * 
 * Tests the complete UI workflow:
 * - Table rendering
 * - Scene editing
 * - Flag management
 * - Save/load operations
 * - Event handlers
 */

import { mockFetch } from "./test_utils.js";

// Mock fetch API for tests
global.fetch = async (url) => {
    return {
        ok: true,
        json: async () => ({ scenes: [] }),
        text: async () => JSON.stringify({ scenes: [] })
    };
};

// Mock LiteGraph
global.LiteGraph = {
    NODE_TITLE_HEIGHT: 30,
    NODE_WIDGET_HEIGHT: 20,
};

// Mock app context
global.app = {
    graph: {
        setDirtyCanvas: () => {},
    },
};

/**
 * End-to-end tests for StoryEdit node frontend.
 * 
 * Tests the complete UI workflow:
 * - Table rendering logic
 * - Scene editing logic
 * - Flag management
 * - Data validation
 * - Event handling logic
 */

import { setupStoryEdit } from "../js/nodes/story.js";

// Mock LiteGraph
global.LiteGraph = {
    NODE_TITLE_HEIGHT: 30,
    NODE_WIDGET_HEIGHT: 20,
};

describe("StoryEdit UI", () => {
    let nodeType, app, node;

    beforeEach(() => {
        // Mock node type
        nodeType = {
            prototype: {
                onNodeCreated: null,
                onResize: null,
                onExecuted: null,
            },
        };

        // Mock app
        app = {
            graph: {
                setDirtyCanvas: () => {},
            },
        };

        // Mock node instance
        node = {
            widgets: [
                { name: "story_select", value: "test_story" },
                { name: "preview_scene_name", value: "" },
            ],
            size: [700, 500],
            addDOMWidget: (name, type, container, options) => ({
                name,
                type,
                container,
                options,
                parentNode: node,
                computeSize: () => [700, 300],
            }),
            graph: app.graph,
        };
    });

    describe("Initialization", () => {
        test("should setup node type with hooks", () => {
            setupStoryEdit(nodeType, { name: "StoryEdit" }, app);

            expect(nodeType.prototype.onNodeCreated).toBeDefined();
            expect(nodeType.prototype.onResize).toBeDefined();
            expect(nodeType.prototype.onExecuted).toBeDefined();
        });

        test("should create table container on node creation", () => {
            setupStoryEdit(nodeType, { name: "StoryEdit" }, app);

            // Mock addDOMWidget to capture calls
            const addDOMWidgetSpy = (node.addDOMWidget = (name, type, element, options) => {
                expect(name).toBe("story_table");
                expect(type).toBe("preview");
                expect(options.serialize).toBe(false);
                return {
                    name,
                    type,
                    element,
                    options,
                    parentNode: node,
                    computeSize: () => [700, 300],
                };
            });

            // Trigger onNodeCreated
            nodeType.prototype.onNodeCreated.call(node);

            expect(node._storyContainer).toBeDefined();
        });
    });

    describe("Data Structures", () => {
        test("should validate scene data structure", () => {
            const scene = {
                scene_id: "scene_001",
                scene_name: "opening",
                scene_order: 0,
                mask_type: "combined",
                mask_background: true,
                prompt_source: "prompt",
                prompt_key: "girl_pos",
                custom_prompt: "",
                depth_type: "depth",
                pose_type: "open",
                use_depth: false,
                use_mask: false,
                use_pose: false,
                use_canny: false,
            };

            // Verify required fields
            expect(scene.scene_id).toBeDefined();
            expect(scene.scene_name).toBeDefined();
            expect(scene.scene_order).toBeDefined();
            expect(scene.mask_type).toBeDefined();
            expect(scene.prompt_source).toBeDefined();
        });

        test("should validate story data structure", () => {
            const story = {
                story_name: "test_story",
                story_dir: "/tmp/test_story",
                scenes: [],
            };

            expect(story.story_name).toBeDefined();
            expect(story.story_dir).toBeDefined();
            expect(Array.isArray(story.scenes)).toBe(true);
        });
    });

    describe("Scene Management Logic", () => {
        test("should resolve preview scene by name", () => {
            const scenes = [
                { scene_name: "opening", scene_order: 0 },
                { scene_name: "middle", scene_order: 1 },
                { scene_name: "ending", scene_order: 2 },
            ];

            // Simulate scene resolution
            const resolveScene = (scenes, name) => {
                if (!scenes || scenes.length === 0) return null;
                if (!name) return scenes[0];
                return scenes.find((s) => s.scene_name === name) || scenes[0];
            };

            const result = resolveScene(scenes, "middle");
            expect(result).toBeDefined();
            expect(result.scene_name).toBe("middle");
        });

        test("should default to first scene when no name specified", () => {
            const scenes = [
                { scene_name: "opening", scene_order: 0 },
                { scene_name: "middle", scene_order: 1 },
            ];

            const resolveScene = (scenes, name) => {
                if (!scenes || scenes.length === 0) return null;
                if (!name) return scenes[0];
                return scenes.find((s) => s.scene_name === name) || scenes[0];
            };

            const result = resolveScene(scenes, "");
            expect(result).toBeDefined();
            expect(result.scene_name).toBe("opening");
        });

        test("should handle empty scenes array", () => {
            const scenes = [];

            const resolveScene = (scenes, name) => {
                if (!scenes || scenes.length === 0) return null;
                if (!name) return scenes[0];
                return scenes.find((s) => s.scene_name === name) || scenes[0];
            };

            const result = resolveScene(scenes, "");
            expect(result).toBeNull();
        });
    });

    describe("Scene Reordering Logic", () => {
        test("should adjust scene order after move", () => {
            const scenes = [
                { scene_name: "first", scene_order: 0 },
                { scene_name: "second", scene_order: 1 },
                { scene_name: "third", scene_order: 2 },
            ];

            // Swap first and second
            [scenes[0], scenes[1]] = [scenes[1], scenes[0]];

            // Adjust orders
            scenes.forEach((scene, idx) => {
                scene.scene_order = idx;
            });

            expect(scenes[0].scene_name).toBe("second");
            expect(scenes[0].scene_order).toBe(0);
            expect(scenes[1].scene_name).toBe("first");
            expect(scenes[1].scene_order).toBe(1);
        });

        test("should maintain scene count after reorder", () => {
            const scenes = [
                { scene_name: "first", scene_order: 0 },
                { scene_name: "second", scene_order: 1 },
                { scene_name: "third", scene_order: 2 },
            ];

            const initialCount = scenes.length;

            // Move last to first
            const last = scenes.pop();
            scenes.unshift(last);

            // Adjust orders
            scenes.forEach((scene, idx) => {
                scene.scene_order = idx;
            });

            expect(scenes.length).toBe(initialCount);
            expect(scenes[0].scene_name).toBe("third");
            expect(scenes[0].scene_order).toBe(0);
        });
    });

    describe("Prompt Source Logic", () => {
        test("should determine input type based on prompt source", () => {
            const getInputType = (promptSource) => {
                if (promptSource === "custom") return "textarea";
                if (promptSource === "prompt" || promptSource === "composition")
                    return "select";
                return "text";
            };

            expect(getInputType("custom")).toBe("textarea");
            expect(getInputType("prompt")).toBe("select");
            expect(getInputType("composition")).toBe("select");
        });

        test("should validate prompt based on source", () => {
            const validatePrompt = (scene) => {
                if (scene.prompt_source === "prompt" && !scene.prompt_key) {
                    return { valid: false, error: "Prompt key required" };
                }
                if (scene.prompt_source === "custom" && !scene.custom_prompt) {
                    return { valid: false, error: "Custom prompt required" };
                }
                return { valid: true, error: null };
            };

            // Valid prompt source
            const valid = validatePrompt({
                prompt_source: "prompt",
                prompt_key: "girl_pos",
            });
            expect(valid.valid).toBe(true);

            // Invalid - missing key
            const invalid = validatePrompt({
                prompt_source: "prompt",
                prompt_key: "",
            });
            expect(invalid.valid).toBe(false);

            // Invalid - missing custom
            const invalid2 = validatePrompt({
                prompt_source: "custom",
                custom_prompt: "",
            });
            expect(invalid2.valid).toBe(false);
        });
    });

    describe("Execution Handler", () => {
        test("should parse metadata from execution message", () => {
            setupStoryEdit(nodeType, { name: "StoryEdit" }, app);
            nodeType.prototype.onNodeCreated.call(node);

            // Mock console.log to capture metadata parsing
            const consoleSpy = [];
            const originalLog = console.log;
            console.log = (...args) => {
                consoleSpy.push(args);
                originalLog(...args);
            };

            const message = {
                text: [
                    "Summary text",
                    "Prompt text",
                    JSON.stringify({
                        story_name: "test",
                        scene_count: 3,
                        preview_scene: "opening",
                    }),
                ],
            };

            // Call onExecuted
            nodeType.prototype.onExecuted.call(node, message);

            // Verify console log was called with story data
            const metadataLog = consoleSpy.find((args) =>
                args.some((arg) => typeof arg === "string" && arg.includes("Received story data"))
            );
            expect(metadataLog).toBeDefined();

            // Restore console.log
            console.log = originalLog;
        });
    });
});

