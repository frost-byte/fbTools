/**
 * Tests for DatasetCaptionViewer node UI wiring and data application.
 */

import { setupDatasetCaptionViewer } from "../js/nodes/dataset_caption_viewer.js";
import { datasetCaptionAPI } from "../js/api/dataset_caption.js";
import { createMockFn } from "./test_utils.js";

describe("DatasetCaptionViewer", () => {
    let nodeType;
    let app;
    let node;
    let createdElement;
    let addDomWidgetCalls;

    beforeEach(() => {
        document.body.innerHTML = "";
        global.alert = createMockFn();
        global.confirm = () => true;

        nodeType = { prototype: {} };
        app = { graph: { setDirtyCanvas: createMockFn() } };
        addDomWidgetCalls = 0;

        node = {
            widgets: [
                { name: "dataset_path", value: "/tmp/dataset" },
                { name: "output_directory", value: "" },
                { name: "page", value: 1 },
                { name: "page_size", value: 10 },
                { name: "recursive", value: false },
                { name: "captioner_type", value: "qwen_vl" },
                { name: "instruction", value: "Describe image" },
                { name: "trigger_word", value: "" },
                { name: "device", value: "auto" },
                { name: "use_8bit", value: false },
                { name: "clean_caption", value: true },
                { name: "gemini_api_key", value: "" },
            ],
            size: [520, 420],
            addDOMWidget: ((name, type, element) => {
                addDomWidgetCalls += 1;
                createdElement = element;
                return {
                    name,
                    type,
                    element,
                    computeSize: () => [560, 420],
                };
            }),
        };
    });

    test("installs node hooks and creates DOM widget", () => {
        setupDatasetCaptionViewer(nodeType, { name: "DatasetCaptionViewer" }, app);

        expect(typeof nodeType.prototype.onNodeCreated).toBe("function");
        expect(typeof nodeType.prototype.onExecuted).toBe("function");

        nodeType.prototype.onNodeCreated.call(node);

        expect(node._dcvInitialized).toBe(true);
        expect(addDomWidgetCalls).toBeGreaterThan(0);
        expect(createdElement).toBeTruthy();
        expect(createdElement.className).toContain("dcv-wrap");
    });

    test("load button fetches list data and renders a row", async () => {
        setupDatasetCaptionViewer(nodeType, { name: "DatasetCaptionViewer" }, app);
        nodeType.prototype.onNodeCreated.call(node);

        const listCalls = [];
        datasetCaptionAPI.listDataset = async (params) => {
            listCalls.push(params);
            return {
            rows: [
                {
                    filename: "sample.png",
                    image_path: "/tmp/dataset/sample.png",
                    txt_path: "/tmp/dataset/sample.txt",
                    caption: "existing",
                    has_caption: true,
                },
            ],
            total: 1,
            page: 1,
            page_size: 10,
            total_pages: 1,
            base_dir: "/tmp/dataset",
            };
        };

        const loadButton = createdElement.querySelector(".dcv-load-btn");
        expect(loadButton).toBeTruthy();

        loadButton.click();
        await Promise.resolve();
        await Promise.resolve();

        expect(listCalls.length).toBe(1);
        const params = listCalls[0];
        expect(params.path).toBe("/tmp/dataset");
        expect(params.page).toBe(1);
        expect(params.page_size).toBe(10);

        const row = createdElement.querySelector(".dcv-row");
        const textarea = createdElement.querySelector(".dcv-caption-textarea");
        expect(row).toBeTruthy();
        expect(textarea.value).toBe("existing");
    });

    test("onExecuted applies dataset_viewer payload from message", () => {
        setupDatasetCaptionViewer(nodeType, { name: "DatasetCaptionViewer" }, app);
        nodeType.prototype.onNodeCreated.call(node);

        nodeType.prototype.onExecuted.call(node, {
            ui: {
                dataset_viewer: [
                    {
                        rows: [
                            {
                                filename: "from-exec.jpg",
                                image_path: "/tmp/dataset/from-exec.jpg",
                                txt_path: "/tmp/dataset/from-exec.txt",
                                caption: "payload caption",
                                has_caption: true,
                            },
                        ],
                        total: 1,
                        page: 1,
                        page_size: 10,
                        total_pages: 1,
                        base_dir: "/tmp/dataset",
                    },
                ],
            },
        });

        const row = createdElement.querySelector(".dcv-row");
        const caption = createdElement.querySelector(".dcv-caption-textarea");
        expect(row).toBeTruthy();
        expect(caption.value).toBe("payload caption");
    });
});
