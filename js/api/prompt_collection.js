/**
 * PromptCollection REST API Client
 * Handles all server-side prompt collection management operations.
 */

import { BaseAPI } from "../utils/api_base.js";

export class PromptCollectionAPI extends BaseAPI {
    constructor() {
        super("/fbtools/prompts");
    }

    /**
     * Create a new prompt collection session
     * @param {object|null} initialData - Optional initial prompt collection data
     * @returns {Promise<{session_id: string, collection: object}>}
     */
    async createSession(initialData = null) {
        return await this.post("/create", { collection: initialData });
    }

    /**
     * Add or update a prompt in a collection
     * @param {string} sessionId - Session identifier
     * @param {string} key - Prompt key/name
     * @param {string} value - Prompt value
     * @param {object} metadata - Optional {category, description, tags}
     * @returns {Promise<{collection: object, prompt_names: string[]}>}
     */
    async addPrompt(sessionId, key, value, metadata = {}) {
        return await this.post("/add", {
            session_id: sessionId,
            key,
            value,
            ...metadata,
        });
    }

    /**
     * Remove a prompt from a collection
     * @param {string} sessionId - Session identifier
     * @param {string} key - Prompt key to remove
     * @returns {Promise<{collection: object, prompt_names: string[]}>}
     */
    async removePrompt(sessionId, key) {
        return await this.post("/remove", {
            session_id: sessionId,
            key,
        });
    }

    /**
     * List all prompt names in a collection
     * @param {string} sessionId - Session identifier
     * @returns {Promise<{prompt_names: string[]}>}
     */
    async listPromptNames(sessionId) {
        return await this.get("/list_names", { session_id: sessionId });
    }

    /**
     * Get the full collection data
     * @param {string} sessionId - Session identifier
     * @returns {Promise<{collection: object}>}
     */
    async getCollection(sessionId) {
        return await this.get("/get_collection", { session_id: sessionId });
    }
}

// Export singleton instance for convenience
export const promptCollectionAPI = new PromptCollectionAPI();
