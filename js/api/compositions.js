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

    // ── Backgrounds ─────────────────────────────────────────────────────────────

    listBackgrounds() {
        return this.get("/backgrounds/list");
    }

    // ── Presets ─────────────────────────────────────────────────────────────────

    listCameraPresets() {
        return this.get("/presets/cameras");
    }

    listSoundPresets() {
        return this.get("/presets/sounds");
    }
}

export const compositionsApi = new CompositionsAPI();
