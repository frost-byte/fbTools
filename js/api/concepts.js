/**
 * Concept Registry REST API client.
 */

import { BaseAPI } from "../utils/api_base.js";

export class ConceptsAPI extends BaseAPI {
    constructor() {
        super("/fbtools/concepts");
    }

    /** Increment server-side reload counter so ConceptRegistryLoad re-executes. */
    async reload() {
        return await this.post("/reload", {});
    }

    /** Fetch the current default registry as a JSON object. */
    async getRegistry() {
        return await this.get("/registry");
    }
}

export const conceptsAPI = new ConceptsAPI();
