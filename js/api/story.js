/**
 * Story REST API Client (stub for future implementation)
 * Handles story-level operations and scene management.
 */

import { BaseAPI } from "../utils/api_base.js";

export class StoryAPI extends BaseAPI {
    constructor() {
        super("/fbtools/story");
    }

    /**
     * Load a story from directory
     * @param {string} storyDir - Story directory path
     * @returns {Promise<{story: object, scenes: string[]}>}
     */
    async loadStory(storyDir) {
        return await this.post("/load", { story_dir: storyDir });
    }

    /**
     * Save story data
     * @param {string} storyDir - Story directory path
     * @param {object} storyData - Story data to save
     * @returns {Promise<{success: boolean}>}
     */
    async saveStory(storyDir, storyData) {
        return await this.post("/save", {
            story_dir: storyDir,
            story_data: storyData,
        });
    }

    /**
     * List all stories in a directory
     * @param {string} baseDir - Base directory containing stories
     * @returns {Promise<{stories: string[]}>}
     */
    async listStories(baseDir) {
        return await this.get("/list", { base_dir: baseDir });
    }

    /**
     * Get scene order for a story
     * @param {string} storyDir - Story directory path
     * @returns {Promise<{scenes: Array<{order: number, name: string}>}>}
     */
    async getSceneOrder(storyDir) {
        return await this.get("/scene_order", { story_dir: storyDir });
    }

    /**
     * Update scene order
     * @param {string} storyDir - Story directory path
     * @param {Array} sceneOrder - New scene order array
     * @returns {Promise<{success: boolean}>}
     */
    async updateSceneOrder(storyDir, sceneOrder) {
        return await this.post("/update_scene_order", {
            story_dir: storyDir,
            scene_order: sceneOrder,
        });
    }
}

// Export singleton instance for convenience
export const storyAPI = new StoryAPI();
