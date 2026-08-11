/**
 * Tests for BundlesAPI client.
 */

import { BundlesAPI } from "../js/api/bundles.js";
import { mockFetch } from "./test_utils.js";

describe("BundlesAPI", () => {
    let api;

    beforeEach(() => {
        mockFetch.setup();
        api = new BundlesAPI();
    });

    afterEach(() => {
        mockFetch.restore();
    });

    // ── listBundles ────────────────────────────────────────────────────────────

    describe("listBundles", () => {
        test("calls /fbtools/bundles/list with no params when no subject given", async () => {
            mockFetch.mockResponse({ bundles: [] });
            await api.listBundles();
            const { url } = mockFetch.getCalls()[0];
            expect(String(url)).toContain("/bundles/list");
            expect(String(url)).not.toContain("subject_id");
        });

        test("passes subject_id when provided", async () => {
            mockFetch.mockResponse({ bundles: [] });
            await api.listBundles("char_alice");
            const { url } = mockFetch.getCalls()[0];
            expect(String(url)).toContain("subject_id=char_alice");
        });

        test("returns bundles array", async () => {
            const bundle = { id: "b1", name: "Bundle One", subject_id: "char_alice" };
            mockFetch.mockResponse({ bundles: [bundle] });
            const result = await api.listBundles();
            expect(result.bundles).toHaveLength(1);
            expect(result.bundles[0].id).toBe("b1");
        });
    });

    // ── getBundle ──────────────────────────────────────────────────────────────

    describe("getBundle", () => {
        test("calls /fbtools/bundles/get with id param", async () => {
            mockFetch.mockResponse({ id: "b1", name: "Bundle One" });
            await api.getBundle("b1");
            const { url } = mockFetch.getCalls()[0];
            expect(String(url)).toContain("/bundles/get");
            expect(String(url)).toContain("id=b1");
        });

        test("returns bundle object", async () => {
            const bundle = { id: "b1", name: "Bundle One", visual: { type: "video" } };
            mockFetch.mockResponse(bundle);
            const result = await api.getBundle("b1");
            expect(result.visual.type).toBe("video");
        });
    });

    // ── saveBundle ─────────────────────────────────────────────────────────────

    describe("saveBundle", () => {
        test("posts to /fbtools/bundles/save", async () => {
            mockFetch.mockResponse({ success: true, id: "b1" });
            await api.saveBundle({ id: "b1", name: "Bundle One", subject_id: "char_alice" });
            const { url, options } = mockFetch.getCalls()[0];
            expect(String(url)).toContain("/bundles/save");
            expect(options.method).toBe("POST");
        });

        test("sends bundle in request body", async () => {
            mockFetch.mockResponse({ success: true, id: "b1" });
            const bundle = { id: "b1", name: "Bundle One", subject_id: "char_alice" };
            await api.saveBundle(bundle);
            const { body } = mockFetch.getCalls()[0];
            const parsed = JSON.parse(body);
            expect(parsed.id).toBe("b1");
            expect(parsed.name).toBe("Bundle One");
        });

        test("returns success response", async () => {
            mockFetch.mockResponse({ success: true, id: "b1" });
            const result = await api.saveBundle({ id: "b1" });
            expect(result.success).toBe(true);
            expect(result.id).toBe("b1");
        });
    });

    // ── deleteBundle ───────────────────────────────────────────────────────────

    describe("deleteBundle", () => {
        test("sends DELETE request to /fbtools/bundles/delete", async () => {
            mockFetch.mockResponse({ success: true });
            await api.deleteBundle("b1");
            const { url, options } = mockFetch.getCalls()[0];
            expect(String(url)).toContain("/bundles/delete");
            expect(String(url)).toContain("id=b1");
            expect(options.method).toBe("DELETE");
        });

        test("URL-encodes the id", async () => {
            mockFetch.mockResponse({ success: true });
            await api.deleteBundle("bundle with spaces");
            const { url } = mockFetch.getCalls()[0];
            expect(String(url)).toContain("bundle%20with%20spaces");
        });

        test("throws on error response", async () => {
            mockFetch.mockError(404, "Not Found");
            await expect(api.deleteBundle("nope")).rejects.toThrow();
        });
    });

    // ── listCasts ──────────────────────────────────────────────────────────────

    describe("listCasts", () => {
        test("calls /fbtools/casts/list", async () => {
            mockFetch.mockResponse({ casts: [] });
            await api.listCasts();
            const { url } = mockFetch.getCalls()[0];
            expect(String(url)).toContain("/casts/list");
        });

        test("returns casts array", async () => {
            mockFetch.mockResponse({ casts: [{ id: "c1", name: "Cast One" }] });
            const result = await api.listCasts();
            expect(result.casts[0].id).toBe("c1");
        });
    });

    // ── saveCast / deleteCast ──────────────────────────────────────────────────

    describe("saveCast", () => {
        test("posts to /fbtools/casts/save", async () => {
            mockFetch.mockResponse({ success: true, id: "c1" });
            await api.saveCast({ id: "c1", name: "Cast One", entries: [] });
            const { url, options } = mockFetch.getCalls()[0];
            expect(String(url)).toContain("/casts/save");
            expect(options.method).toBe("POST");
        });
    });

    describe("deleteCast", () => {
        test("sends DELETE request to /fbtools/casts/delete", async () => {
            mockFetch.mockResponse({ success: true });
            await api.deleteCast("c1");
            const { url, options } = mockFetch.getCalls()[0];
            expect(String(url)).toContain("/casts/delete");
            expect(String(url)).toContain("id=c1");
            expect(options.method).toBe("DELETE");
        });
    });

    // ── reloadCasts ────────────────────────────────────────────────────────────

    describe("reloadCasts", () => {
        test("posts to /fbtools/casts/reload", async () => {
            mockFetch.mockResponse({ success: true, counter: 1 });
            await api.reloadCasts();
            const { url, options } = mockFetch.getCalls()[0];
            expect(String(url)).toContain("/casts/reload");
            expect(options.method).toBe("POST");
        });

        test("returns response with counter", async () => {
            mockFetch.mockResponse({ success: true, counter: 3 });
            const result = await api.reloadCasts();
            expect(result.success).toBe(true);
            expect(result.counter).toBe(3);
        });
    });

    // ── listSubjects ───────────────────────────────────────────────────────────

    describe("listSubjects", () => {
        test("calls /fbtools/subjects/list", async () => {
            mockFetch.mockResponse({ subjects: [] });
            await api.listSubjects();
            const { url } = mockFetch.getCalls()[0];
            expect(String(url)).toContain("/subjects/list");
        });
    });

    // ── listMedia ──────────────────────────────────────────────────────────────

    describe("listMedia", () => {
        test("passes type param", async () => {
            mockFetch.mockResponse({ files: [] });
            await api.listMedia("video");
            const { url } = mockFetch.getCalls()[0];
            expect(String(url)).toContain("/media/list");
            expect(String(url)).toContain("type=video");
        });

        test("works for image type", async () => {
            mockFetch.mockResponse({ files: ["a.jpg", "b.png"] });
            const result = await api.listMedia("image");
            expect(result.files).toHaveLength(2);
        });

        test("works for audio type", async () => {
            mockFetch.mockResponse({ files: ["voice.wav"] });
            const result = await api.listMedia("audio");
            expect(result.files[0]).toBe("voice.wav");
        });
    });
});
