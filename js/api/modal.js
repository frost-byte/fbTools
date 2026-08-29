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

    /** Activate the Modal backend.  body: { model_key, quantize }. */
    activate(modelKey, quantize = true) {
        return this.post("/activate", { model_key: modelKey, quantize });
    }

    /** Deactivate the Modal backend (clears local state only). */
    deactivate() {
        return this.post("/deactivate", {});
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
