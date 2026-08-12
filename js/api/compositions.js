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

    listCameraPresets() {
        return this.get("/presets/cameras");
    }

    listSoundPresets() {
        return this.get("/presets/sounds");
    }
}

export const compositionsApi = new CompositionsAPI();
