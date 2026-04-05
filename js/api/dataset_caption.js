/**
 * Dataset caption REST API client.
 * Handles list/save/recaption operations for DatasetCaptionViewer UI.
 */

import { BaseAPI } from "../utils/api_base.js";

export class DatasetCaptionAPI extends BaseAPI {
    constructor() {
        super("/fbtools/dataset_caption");
    }

    /**
     * List dataset images/captions with pagination.
     * @param {object} params - { path, output_dir, page, page_size, recursive }
     * @returns {Promise<object>}
     */
    async listDataset(params) {
        return await this.get("/list", params);
    }

    /**
     * Save caption text for one image.
     * @param {string} txtPath
     * @param {string} caption
     * @returns {Promise<object>}
     */
    async saveCaption(txtPath, caption) {
        return await this.post("/save", {
            txt_path: txtPath,
            caption,
        });
    }

    /**
     * Re-caption one image with current settings.
     * @param {object} payload
     * @returns {Promise<object>}
     */
    async recaption(payload) {
        return await this.post("/recaption", payload);
    }

    /**
     * Build image endpoint URL for thumbnail loading.
     * @param {string} imagePath
     * @returns {string}
     */
    getImageUrl(imagePath) {
        return `${this.baseUrl}/image?path=${encodeURIComponent(imagePath)}`;
    }
}

export const datasetCaptionAPI = new DatasetCaptionAPI();
