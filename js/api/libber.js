/**
 * Libber REST API Client
 * Handles Libber placeholder management operations.
 */

import { BaseAPI } from "../utils/api_base.js";

export class LibberAPI extends BaseAPI {
    constructor() {
        super("/fbtools/libber");
    }

    /**
     * Create a new Libber session
     * @param {string} name - Name for the Libber instance
     * @param {string} delimiter - Delimiter for lib references (default "%")
     * @param {number} maxDepth - Maximum substitution depth (default 10)
     * @returns {Promise<{name: string, keys: string[], status: string}>}
     */
    async createLibber(name, delimiter = "%", maxDepth = 10) {
        return await this.post("/create", { 
            name,
            delimiter,
            max_depth: maxDepth
        });
    }

    /**
     * Load a Libber from file
     * @param {string} name - Name for the Libber instance
     * @param {string} filepath - Path to the JSON file
     * @returns {Promise<{name: string, keys: string[], status: string}>}
     */
    async loadLibber(name, filepath) {
        return await this.post("/load", {
            name,
            filepath,
        });
    }

    /**
     * Add a lib entry to Libber
     * @param {string} name - Libber name
     * @param {string} key - Lib key
     * @param {string} value - Lib value
     * @returns {Promise<{name: string, keys: string[], status: string}>}
     */
    async addLib(name, key, value) {
        return await this.post("/add_lib", {
            name,
            key,
            value,
        });
    }

    /**
     * Remove a lib entry from Libber
     * @param {string} name - Libber name
     * @param {string} key - Lib key to remove
     * @returns {Promise<{name: string, keys: string[], status: string}>}
     */
    async removeLib(name, key) {
        return await this.post("/remove_lib", {
            name,
            key,
        });
    }

    /**
     * Save a Libber to file
     * @param {string} name - Libber name
     * @param {string} filepath - Path where to save the JSON file
     * @returns {Promise<{name: string, filepath: string, status: string}>}
     */
    async saveLibber(name, filepath) {
        return await this.post("/save", {
            name,
            filepath,
        });
    }

    /**
     * List all available libbers
     * @returns {Promise<{libbers: string[], files: string[], count: number}>}
     */
    async listLibbers() {
        return await this.get("/list");
    }

    /**
     * Get Libber data for UI display
     * @param {string} name - Libber name
     * @returns {Promise<{keys: string[], lib_dict: object, delimiter: string, max_depth: number}>}
     */
    async getLibberData(name) {
        return await this.get(`/get_data/${encodeURIComponent(name)}`);
    }

    /**
     * Apply Libber substitutions to text
     * @param {string} name - Libber name
     * @param {string} text - Text to process
     * @returns {Promise<{result: string, original: string, name: string}>}
     */
    async applySubstitutions(name, text) {
        return await this.post("/apply", { name, text });
    }

    /**
     * Scan the default libber directory and return metadata for all libbers.
     * @returns {Promise<{libbers: Array, libber_dir: string}>}
     */
    async scan() {
        return await this.get("/scan");
    }

    /**
     * Open a libber for editing (loads from disk if not in memory).
     * @param {string} name
     * @returns {Promise<{name: string, lib_dict: object, delimiter: string, max_depth: number}>}
     */
    async open(name) {
        return await this.post("/open", { name });
    }

    /**
     * Overwrite a libber's full data and persist to disk.
     * @param {string} name
     * @param {object} libDict  key→value mapping
     * @param {string} delimiter
     * @param {number} maxDepth
     * @returns {Promise<{name: string, entry_count: number, filepath: string, status: string}>}
     */
    async saveFull(name, libDict, delimiter, maxDepth) {
        return await this.post("/save_full", {
            name,
            lib_dict:  libDict,
            delimiter,
            max_depth: maxDepth,
        });
    }

    /**
     * Delete a libber from memory and disk.
     * @param {string} name
     * @returns {Promise<{name: string, status: string}>}
     */
    async deleteFull(name) {
        return await this.post("/delete", { name });
    }

    /**
     * Rename a libber on disk and in memory.
     * @param {string} oldName
     * @param {string} newName
     * @returns {Promise<{old_name: string, new_name: string, status: string}>}
     */
    async rename(oldName, newName) {
        return await this.post("/rename", { old_name: oldName, new_name: newName });
    }
}

// Export singleton instance for convenience
export const libberAPI = new LibberAPI();
