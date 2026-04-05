/**
 * Tests for DatasetCaptionAPI client.
 */

import { DatasetCaptionAPI } from "../js/api/dataset_caption.js";
import { mockFetch } from "./test_utils.js";

describe("DatasetCaptionAPI", () => {
    beforeEach(() => {
        mockFetch.setup();
    });

    afterEach(() => {
        mockFetch.restore();
    });

    test("listDataset sends query params to list endpoint", async () => {
        mockFetch.mockResponse({ rows: [], total: 0, page: 2, page_size: 5, total_pages: 1 });

        const api = new DatasetCaptionAPI();
        await api.listDataset({
            path: "/tmp/dataset",
            output_dir: "/tmp/captions",
            page: 2,
            page_size: 5,
            recursive: true,
        });

        const call = mockFetch.getCalls()[0];
        const url = String(call.url);

        expect(url).toContain("/fbtools/dataset_caption/list");
        expect(url).toContain("path=%2Ftmp%2Fdataset");
        expect(url).toContain("output_dir=%2Ftmp%2Fcaptions");
        expect(url).toContain("page=2");
        expect(url).toContain("page_size=5");
        expect(url).toContain("recursive=true");
    });

    test("saveCaption posts txt_path and caption", async () => {
        mockFetch.mockResponse({ ok: true });

        const api = new DatasetCaptionAPI();
        await api.saveCaption("/tmp/dataset/item.txt", "new caption");

        const call = mockFetch.getCalls()[0];
        expect(String(call.url)).toContain("/fbtools/dataset_caption/save");

        const body = JSON.parse(call.body);
        expect(body).toEqual({
            txt_path: "/tmp/dataset/item.txt",
            caption: "new caption",
        });
    });

    test("recaption posts payload to recaption endpoint", async () => {
        mockFetch.mockResponse({ ok: true, caption: "re-generated" });

        const api = new DatasetCaptionAPI();
        await api.recaption({
            image_path: "/tmp/dataset/item.png",
            txt_path: "/tmp/dataset/item.txt",
            captioner_type: "qwen_vl",
            instruction: "Describe",
        });

        const call = mockFetch.getCalls()[0];
        expect(String(call.url)).toContain("/fbtools/dataset_caption/recaption");

        const body = JSON.parse(call.body);
        expect(body.image_path).toBe("/tmp/dataset/item.png");
        expect(body.txt_path).toBe("/tmp/dataset/item.txt");
        expect(body.captioner_type).toBe("qwen_vl");
    });

    test("getImageUrl returns encoded image query URL", () => {
        const api = new DatasetCaptionAPI();
        const url = api.getImageUrl("/tmp/my images/one.png");

        expect(url).toBe("/fbtools/dataset_caption/image?path=%2Ftmp%2Fmy%20images%2Fone.png");
    });
});
