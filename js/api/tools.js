/**
 * REST API client for standalone generation "Tools" (currently: Qwen-Image-2.1 photo restoration).
 * Bundle-agnostic and composition-agnostic — unlike bundles.js's generateCharacterSheet, these
 * operate on a plain input/output file, not a saved Bundle or Composition.
 */

import { BaseAPI } from "../utils/api_base.js";

export class ToolsAPI extends BaseAPI {
    constructor() {
        super("/fbtools");
    }

    /** Run the Qwen-Image-2.1 photo-restoration workflow against a single image.
     *  Returns { file, folder: "output" }. */
    restorePhoto(filename, { folder = "input", restoreHint, prompt } = {}) {
        const body = { filename, folder };
        if (restoreHint && restoreHint.trim()) body.restore_hint = restoreHint.trim();
        if (prompt && prompt.trim())           body.prompt       = prompt.trim();
        return this.post("/tools/restore_photo", body);
    }

    getRestorePhotoSettingsOptions() {
        return this.get("/tools/restore_photo_settings_options");
    }
}

export const toolsApi = new ToolsAPI();
