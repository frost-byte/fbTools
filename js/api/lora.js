/**
 * REST API client for LoRA-related fbTools endpoints.
 */

import { BaseAPI } from "../utils/api_base.js";

export class LoraAPI extends BaseAPI {
    constructor() {
        super("/fbtools");
    }

    /**
     * Fetch civitai model-version info for a LoRA by its filename.
     * Results are cached server-side; first call may take a moment.
     *
     * @param {string} loraFilename - Filename as shown in the lora widget (e.g. "my_lora.safetensors")
     * @returns {Promise<object>} Civitai model-version JSON
     */
    async getCivitaiInfo(loraFilename) {
        return this.get("/lora/civitai_info", { lora: loraFilename });
    }
}

export const loraAPI = new LoraAPI();
