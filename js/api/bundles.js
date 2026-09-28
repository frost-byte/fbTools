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

    extractFrame(filename, frameIndex, dir = "input") {
        return this.post("/media/extract_frame", { filename, frame_index: frameIndex, dir });
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

    mediaInfo(filename, dir = "input") {
        return this.get("/media/info", { filename, ...(dir !== "input" ? { dir } : {}) });
    }

    streamUrl(filename, dir = "input") {
        const dirPart = dir !== "input" ? `&dir=${encodeURIComponent(dir)}` : "";
        return `/fbtools/media/stream?filename=${encodeURIComponent(filename)}${dirPart}`;
    }

    preprocessAudio({ bundle_id, filename, dir, start_time, duration, audio_processing }) {
        return this.post("/bundles/preprocess_audio", { bundle_id, filename, dir, start_time, duration, audio_processing });
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

    /**
     * Video-proxy freshness for one bundle (Plan 16). {"fresh", "proxy_path", "eligible"} —
     * "eligible" is false for a bundle with no video reference, no set duration, or a
     * force_rate other than 24/native.
     */
    proxyStatus(bundle_id) {
        return this.get("/bundles/proxy_status", { bundle_id });
    }

    // ── H3 Character/Face Sheet generation ─────────────────────────────────────

    /** Enumeration lists + has_*_override capability flags for the H3 Character Sheet settings
     *  section — see nodes/h3_character_sheet.py::_bundles_character_sheet_settings_options. */
    getCharSheetSettingsOptions() {
        return this.get("/bundles/character_sheet_settings_options");
    }

    /** Run the H3 character/face-sheet workflow against up to 9 reference images, in order. Can
     *  take a while (a real two-pass generation run) — callers should show a "working" state
     *  while this is pending. `refs` is an ORDERED list of {kind:"image", index} (an index into
     *  the bundle's own saved visual.files, resolved server-side) and/or {kind:"frame", file}
     *  (a plain filename already in the input directory root, e.g. from extractFrame() — never
     *  written into the bundle). Order matters: position 0 is what this template's own prompt
     *  treats as "Picture 1", the sole outfit reference — every other position only contributes
     *  identity. Capped at 9 entries server-side. `outfitHint` is optional freeform text
     *  substituted into the active mode's own prompt wherever its author placed a literal
     *  {{OUTFIT_HINT}} token — no-op if that mode's prompt node isn't titled for it (see
     *  has_character_prompt_override/has_face_prompt_override from getCharSheetSettingsOptions). */
    generateCharacterSheet(bundleId, mode, refs, outfitHint) {
        const body = { bundle_id: bundleId, mode, refs: refs || [] };
        if (outfitHint && outfitHint.trim()) body.outfit_hint = outfitHint.trim();
        return this.post("/bundles/generate_character_sheet", body);
    }
}

export const bundlesApi = new BundlesAPI();
