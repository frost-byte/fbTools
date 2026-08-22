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

    /**
     * Ask the VLM to detect meaningful segment boundaries in the source video.
     * Returns { segments: [{start_time, end_time, label, action}] }
     */
    detectSegments({ profile_id, video_duration = 0, interval_seconds = 0,
                     captioner_type = "qwen_vl", device = "auto",
                     use_8bit = false, gemini_api_key = "" }) {
        return this.post("/source_profiles/detect_segments", {
            profile_id, video_duration, interval_seconds,
            captioner_type, device, use_8bit, gemini_api_key,
        });
    }

    /**
     * Ask the VLM to describe the action visible in the middle frame of a clip.
     * Returns { action: "..." }
     */
    describeClip({ profile_id, start_time, end_time,
                   captioner_type = "qwen_vl", device = "auto",
                   use_8bit = false, gemini_api_key = "" }) {
        return this.post("/source_profiles/describe_clip", {
            profile_id, start_time, end_time,
            captioner_type, device, use_8bit, gemini_api_key,
        });
    }

    /**
     * Auto-partition a profile into equal-duration clips.
     * Returns the updated profile.
     */
    autoPartition({ profile_id, video_duration, segment_duration = 0 }) {
        return this.post("/source_profiles/auto_partition", {
            profile_id, video_duration, segment_duration,
        });
    }

    /** Save a single clip update to a profile. Returns the updated profile. */
    upsertClip({ profile_id, clip }) {
        return this.post("/source_profiles/upsert_clip", { profile_id, clip });
    }

    /** Remove a clip from a profile. Returns the updated profile. */
    removeClip({ profile_id, clip_id }) {
        return this.post("/source_profiles/remove_clip", { profile_id, clip_id });
    }
}

export const sourceProfilesApi = new SourceProfilesAPI();
