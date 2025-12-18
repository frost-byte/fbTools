/**
 * Libber REST API Client (stub for future implementation)
 * Handles Libber placeholder management operations.
 */

import { BaseAPI } from "../utils/api_base.js";

export class LibberAPI extends BaseAPI {
    constructor() {
        super("/fbtools/libber");
    }

    /**
     * Create a new Libber session
     * @param {object|null} initialData - Optional initial Libber data
     * @returns {Promise<{session_id: string, libber: object}>}
     */
    async createSession(initialData = null) {
        return await this.post("/create", { libber: initialData });
    }

    /**
     * Add a lib entry to Libber
     * @param {string} sessionId - Libber session identifier
     * @param {string} key - Lib key
     * @param {string} value - Lib value
     * @returns {Promise<{libber: object, keys: string[]}>}
     */
    async addLib(sessionId, key, value) {
        return await this.post("/add_lib", {
            session_id: sessionId,
            key,
            value,
        });
    }

    /**
     * Remove a lib entry from Libber
     * @param {string} sessionId - Libber session identifier
     * @param {string} key - Lib key to remove
     * @returns {Promise<{libber: object, keys: string[]}>}
     */
    async removeLib(sessionId, key) {
        return await this.post("/remove_lib", {
            session_id: sessionId,
            key,
        });
    }

    /**
     * Get all keys from Libber
     * @param {string} sessionId - Libber session identifier
     * @returns {Promise<{keys: string[]}>}
     */
    async getKeys(sessionId) {
        return await this.get("/keys", { session_id: sessionId });
    }

    /**
     * Get a specific lib value
     * @param {string} sessionId - Libber session identifier
     * @param {string} key - Lib key to retrieve
     * @returns {Promise<{key: string, value: string}>}
     */
    async getLib(sessionId, key) {
        return await this.get(`/get_lib/${encodeURIComponent(key)}`, {
            session_id: sessionId,
        });
    }

    /**
     * Apply Libber substitutions to text
     * @param {string} sessionId - Libber session identifier
     * @param {string} text - Text to process
     * @returns {Promise<{result: string}>}
     */
    async applySubstitutions(sessionId, text) {
        return await this.post("/apply", {
            session_id: sessionId,
            text,
        });
    }
}

// Export singleton instance for convenience
export const libberAPI = new LibberAPI();
