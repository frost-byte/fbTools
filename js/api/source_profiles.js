/**
 * REST API client for Source Profiles and their LLM analysis.
 */

import { BaseAPI } from "../utils/api_base.js";

export class SourceProfilesAPI extends BaseAPI {
    constructor() {
        super("/fbtools");
    }

    list() {
        return this.get("/source_profiles/list");
    }

    getProfile(id) {
        return this.get("/source_profiles/get", { id });
    }

    save(profile) {
        return this.post("/source_profiles/save", profile);
    }

    async delete(id) {
        const r = await fetch(
            `/fbtools/source_profiles/delete?id=${encodeURIComponent(id)}`,
            { method: "DELETE" }
        );
        if (!r.ok) throw new Error(`Delete failed: ${r.statusText}`);
        return r.json();
    }

    reload() {
        return this.post("/source_profiles/reload", {});
    }

    /** Run a focused VLM analysis pass on a profile's media. */
    analyze({ profile_id, pass_type, prompt_override = "", captioner_type = "qwen_vl",
              device = "auto", use_8bit = false, gemini_api_key = "" }) {
        return this.post("/source_profiles/analyze", {
            profile_id, pass_type, prompt_override,
            captioner_type, device, use_8bit, gemini_api_key,
        });
    }

    /** Fetch analysis history for a profile, newest first. */
    analysisHistory(profile_id) {
        return this.get("/source_profiles/analysis_history", { profile_id });
    }
}

export const sourceProfilesApi = new SourceProfilesAPI();
