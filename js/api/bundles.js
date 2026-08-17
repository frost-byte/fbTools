/**
 * REST API client for Reference Bundles and Scene Casts.
 */

import { BaseAPI } from "../utils/api_base.js";

export class BundlesAPI extends BaseAPI {
    constructor() {
        super("/fbtools");
    }

    // ── Reference Bundles ───────────────────────────────────────────────────────

    listBundles(subjectId) {
        const params = subjectId ? { subject_id: subjectId } : {};
        return this.get("/bundles/list", params);
    }

    getBundle(id) {
        return this.get("/bundles/get", { id });
    }

    saveBundle(bundle) {
        return this.post("/bundles/save", bundle);
    }

    async deleteBundle(id) {
        const r = await fetch(`/fbtools/bundles/delete?id=${encodeURIComponent(id)}`, { method: "DELETE" });
        if (!r.ok) throw new Error(`Delete failed: ${r.statusText}`);
        return r.json();
    }

    // ── Scene Casts ─────────────────────────────────────────────────────────────

    listCasts() {
        return this.get("/casts/list");
    }

    getCast(id) {
        return this.get("/casts/get", { id });
    }

    saveCast(cast) {
        return this.post("/casts/save", cast);
    }

    async deleteCast(id) {
        const r = await fetch(`/fbtools/casts/delete?id=${encodeURIComponent(id)}`, { method: "DELETE" });
        if (!r.ok) throw new Error(`Delete failed: ${r.statusText}`);
        return r.json();
    }

    /** Increment the server reload counter so SceneCastLoad nodes re-execute. */
    reloadCasts() {
        return this.post("/casts/reload", {});
    }

    // ── Shared ──────────────────────────────────────────────────────────────────

    listSubjects() {
        return this.get("/subjects/list");
    }

    saveSubjectAppearance(id, summary) {
        return this.post("/subjects/save", { id, appearance: { summary } });
    }

    extractFrame(filename, frameIndex) {
        return this.post("/media/extract_frame", { filename, frame_index: frameIndex });
    }

    async deleteTmpFrame(filename) {
        const r = await fetch(`/fbtools/media/extract_frame?filename=${encodeURIComponent(filename)}`, { method: "DELETE" });
        return r.ok;
    }

    listMedia(type) {
        return this.get("/media/list", { type });
    }

    mediaInfo(filename) {
        return this.get("/media/info", { filename });
    }

    streamUrl(filename) {
        return `/fbtools/media/stream?filename=${encodeURIComponent(filename)}`;
    }
}

export const bundlesApi = new BundlesAPI();
