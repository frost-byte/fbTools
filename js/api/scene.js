/**
 * Scene REST API Client (stub for future implementation)
 * Handles scene metadata operations (prompts, loras, pose info).
 */

import { BaseAPI } from "../utils/api_base.js";

export class SceneAPI extends BaseAPI {
    constructor() {
        super("/fbtools/scene");
    }

    /**
     * Update scene prompts
     * @param {string} sessionId - Scene session identifier
     * @param {object} prompts - Prompts data to update
     * @returns {Promise<object>}
     */
    async updatePrompts(sessionId, prompts) {
        return await this.post("/update_prompts", {
            session_id: sessionId,
            prompts,
        });
    }

    /**
     * Update scene metadata (pose_name, resolution, etc.)
     * @param {string} sessionId - Scene session identifier
     * @param {object} metadata - Metadata fields to update
     * @returns {Promise<object>}
     */
    async updateMetadata(sessionId, metadata) {
        return await this.post("/update_metadata", {
            session_id: sessionId,
            ...metadata,
        });
    }

    /**
     * Save scene metadata to files
     * @param {string} sessionId - Scene session identifier
     * @returns {Promise<{success: boolean}>}
     */
    async saveMetadata(sessionId) {
        return await this.post("/save_metadata", {
            session_id: sessionId,
        });
    }

    /**
     * List available scenes in a story directory
     * @param {string} storyDir - Story directory path
     * @returns {Promise<{scenes: string[]}>}
     */
    async listScenes(storyDir) {
        return await this.get(`/list_scenes/${encodeURIComponent(storyDir)}`);
    }

    /**
     * Update loras for a scene
     * @param {string} sessionId - Scene session identifier
     * @param {object} loras - Loras data to update
     * @returns {Promise<object>}
     */
    async updateLoras(sessionId, loras) {
        return await this.post("/update_loras", {
            session_id: sessionId,
            loras,
        });
    }
}

// Export singleton instance for convenience
export const sceneAPI = new SceneAPI();
