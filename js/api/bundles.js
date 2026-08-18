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

    getSubject(id) {
        return this.get("/subjects/get", { id });
    }

    saveSubject(subject) {
        return this.post("/subjects/save", subject);
    }

    async deleteSubject(id) {
        const r = await fetch(`/fbtools/subjects/delete?id=${encodeURIComponent(id)}`, { method: "DELETE" });
        if (!r.ok) throw new Error(`Delete failed: ${r.statusText}`);
        return r.json();
    }

    extractFrame(filename, frameIndex) {
        return this.post("/media/extract_frame", { filename, frame_index: frameIndex });
    }

    async deleteTmpFrame(filename) {
        const r = await fetch(`/fbtools/media/extract_frame?filename=${encodeURIComponent(filename)}`, { method: "DELETE" });
        return r.ok;
    }

    listMedia(type, recursive = false, folder = "input") {
        const params = { type };
        if (recursive)           params.recursive = "true";
        if (folder !== "input")  params.folder    = folder;
        return this.get("/media/list", params);
    }

    mediaInfo(filename) {
        return this.get("/media/info", { filename });
    }

    streamUrl(filename) {
        return `/fbtools/media/stream?filename=${encodeURIComponent(filename)}`;
    }

    preprocessAudio({ bundle_id, filename, start_time, duration, audio_processing }) {
        return this.post("/bundles/preprocess_audio", { bundle_id, filename, start_time, duration, audio_processing });
    }

    async previewSampled({ filename, start_time, duration, force_rate, select_every_nth }) {
        const r = await fetch("/fbtools/bundles/preview_sampled", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ filename, start_time, duration, force_rate, select_every_nth }),
        });
        if (!r.ok) {
            const data = await r.json().catch(() => ({}));
            return { blob: null, error: data.error || r.statusText };
        }
        const blob = await r.blob();
        return { blob, error: null };
    }
}

export const bundlesApi = new BundlesAPI();
