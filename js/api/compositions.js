/**
 * REST API client for Prompt Composition Editor.
 * Covers compositions, subjects, backgrounds, and presets.
 */

import { BaseAPI } from "../utils/api_base.js";

export class CompositionsAPI extends BaseAPI {
    constructor() {
        super("/fbtools");
    }

    // ── Compositions ────────────────────────────────────────────────────────────

    listCompositions() {
        return this.get("/compositions/list");
    }

    getComposition(id) {
        return this.get("/compositions/get", { id });
    }

    saveComposition(comp) {
        return this.post("/compositions/save", comp);
    }

    async deleteComposition(id) {
        const r = await fetch(`/fbtools/compositions/delete?id=${encodeURIComponent(id)}`, {
            method: "DELETE",
        });
        if (!r.ok) throw new Error(`Delete failed: ${r.statusText}`);
        return r.json();
    }

    /** Increment the server reload counter so PromptCompositionLoader nodes re-execute. */
    reloadCompositions() {
        return this.post("/compositions/reload", {});
    }

    getSettings() {
        return this.get("/compositions/settings");
    }

    saveSettings(settings) {
        return this.post("/compositions/settings", settings);
    }

    /** Pass a composition object (inline) or a saved id string. */
    assembleComposition(compOrId, modelType) {
        const body =
            typeof compOrId === "string"
                ? { scene_id: compOrId, model_type: modelType }
                : { composition: compOrId, model_type: modelType };
        return this.post("/compositions/assemble", body);
    }

    // ── Subjects ────────────────────────────────────────────────────────────────

    listSubjects() {
        return this.get("/subjects/list");
    }

    saveSubject(subject) {
        return this.post("/subjects/save", subject);
    }

    async deleteSubject(id) {
        const r = await fetch(`/fbtools/subjects/delete?id=${encodeURIComponent(id)}`, { method: "DELETE" });
        if (!r.ok) throw new Error(`Delete failed: ${r.statusText}`);
        return r.json();
    }

    // ── Backgrounds ─────────────────────────────────────────────────────────────

    listBackgrounds() {
        return this.get("/backgrounds/list");
    }

    saveBackground(bg) {
        return this.post("/backgrounds/save", bg);
    }

    async deleteBackground(id) {
        const r = await fetch(`/fbtools/backgrounds/delete?id=${encodeURIComponent(id)}`, { method: "DELETE" });
        if (!r.ok) throw new Error(`Delete failed: ${r.statusText}`);
        return r.json();
    }

    // ── Presets ─────────────────────────────────────────────────────────────────

    listLoras() {
        return this.get("/loras/list");
    }

    // ── Outfits ─────────────────────────────────────────────────────────────────

    getOutfitRegistry() {
        return this.get("/outfits/registry");
    }

    saveOutfit(outfit) {
        return this.post("/outfits/save", outfit);
    }

    async deleteOutfit(id) {
        const r = await fetch(`/fbtools/outfits/delete?id=${encodeURIComponent(id)}`, { method: "DELETE" });
        if (!r.ok) throw new Error(`Delete failed: ${r.statusText}`);
        return r.json();
    }

    reloadOutfits() {
        return this.post("/outfits/reload", {});
    }

    analyzeBackground(filename, { folder = "input", frameTime = 1.0, maxTokens = 500 } = {}) {
        return this.post("/backgrounds/analyze_media", {
            filename,
            folder,
            frame_time:  frameTime,
            max_tokens:  maxTokens,
        });
    }

    listMedia(type, recursive = false, folder = "input") {
        const params = { type };
        if (recursive)          params.recursive = "true";
        if (folder !== "input") params.folder    = folder;
        return this.get("/media/list", params);
    }

    listCameraPresets() {
        return this.get("/presets/cameras");
    }

    saveCameraPreset(preset) {
        return this.post("/presets/cameras/save", preset);
    }

    async deleteCameraPreset(id) {
        const r = await fetch(`/fbtools/presets/cameras/delete?id=${encodeURIComponent(id)}`, { method: "DELETE" });
        if (!r.ok) throw new Error(await r.text());
        return r.json();
    }

    listSoundPresets() {
        return this.get("/presets/sounds");
    }

    saveSoundPreset(preset) {
        return this.post("/presets/sounds/save", preset);
    }

    async deleteSoundPreset(id) {
        const r = await fetch(`/fbtools/presets/sounds/delete?id=${encodeURIComponent(id)}`, { method: "DELETE" });
        if (!r.ok) throw new Error(await r.text());
        return r.json();
    }
}

export const compositionsApi = new CompositionsAPI();
