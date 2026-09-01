/**
 * REST API client for Modal cloud VLM backend and VLM activity log.
 */

import { BaseAPI } from "../utils/api_base.js";

class ModalAPI extends BaseAPI {
    constructor() {
        super("/fbtools/modal");
    }

    /** Return Modal backend status + preset model list. */
    status() {
        return this.get("/status");
    }

    /** Activate the Modal backend.  body: { model_key, quantize, gpu }. */
    activate(modelKey, quantize = true, gpu = "L40S") {
        return this.post("/activate", { model_key: modelKey, quantize, gpu });
    }

    /** Deactivate the Modal backend (clears local state only). */
    deactivate() {
        return this.post("/deactivate", {});
    }

    /**
     * Compute a VRAM estimate and GPU recommendation for the given model+config.
     * opts: { quantize, contextLength, frameBudget, modality, priority }
     */
    recommend(modelKey, opts = {}) {
        const { quantize = true, contextLength = 8192, frameBudget = 20,
                modality = "image", priority = "cost" } = opts;
        return this.get("/recommend", {
            model_key:      modelKey,
            quantize:       quantize ? "true" : "false",
            context_length: contextLength,
            frame_budget:   frameBudget,
            modality,
            priority,
        });
    }

    /**
     * Fetch and cache a VRAM profile for a custom HuggingFace repo.
     * opts: { refresh }
     */
    profileRepo(repoId, opts = {}) {
        return this.post("/profile_repo", { repo_id: repoId, refresh: !!opts.refresh });
    }
}

class VlmActivityAPI extends BaseAPI {
    constructor() {
        super("/fbtools/vlm");
    }

    /** Return recent VLM activity entries.  opts: { n, backend }. */
    recent({ n = 50, backend = "" } = {}) {
        const params = { n };
        if (backend) params.backend = backend;
        return this.get("/activity", params);
    }

    /** Return deduplicated model IDs used for a backend, most recent first. */
    modelHistory(backend) {
        return this.get("/model_history", { backend });
    }
}

export const modalApi     = new ModalAPI();
export const vlmActivityApi = new VlmActivityAPI();
