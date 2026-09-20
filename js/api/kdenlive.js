/**
 * Kdenlive Archive REST API Client
 * Check / archive / strip-metadata operations on .kdenlive projects.
 */

import { BaseAPI } from "../utils/api_base.js";

export class KdenliveAPI extends BaseAPI {
    constructor() {
        super("/fbtools/kdenlive");
    }

    /**
     * Resolve every media reference in a project without writing anything.
     * @param {{project: string, path_maps?: string[], search_dirs?: string[]}} opts
     * @returns {Promise<{ok: boolean, report: object}>}
     */
    async check(opts) {
        return await this.post("/check", opts);
    }

    /**
     * Start an archive job. Progress arrives as "fbtools.status" websocket
     * events (source "kdenlive_archive"); poll {@link status} to recover.
     * @param {{project: string, dest: string, path_maps?: string[], search_dirs?: string[],
     *          strip_metadata?: boolean, dry_run?: boolean}} opts
     * @returns {Promise<{started: boolean, job_id: string}>}
     */
    async archive(opts) {
        return await this.post("/archive", opts);
    }

    /**
     * @param {string} [jobId] - defaults to the most recent job
     * @returns {Promise<{job: null | {id: string, state: string, progress: object, report: object|null, error: string|null}}>}
     */
    async status(jobId = "") {
        return await this.get("/status", jobId ? { job_id: jobId } : {});
    }

    /** @param {string} jobId */
    async cancel(jobId) {
        return await this.post("/cancel", { job_id: jobId });
    }

    /**
     * Strip embedded ComfyUI workflow/prompt metadata from a project.
     * @param {{project: string, in_place?: boolean}} opts
     * @returns {Promise<{ok: boolean, report: object}>}
     */
    async strip(opts) {
        return await this.post("/strip", opts);
    }
}

export const kdenliveApi = new KdenliveAPI();
