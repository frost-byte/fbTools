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
     * @param {string} [jobId] - defaults to the most recent job (of `kind`, if given)
     * @param {string} [kind] - "archive" | "clean"; only used when jobId is omitted, so a page
     *   reload can recover an in-progress job of a specific kind instead of whichever is newest
     * @returns {Promise<{job: null | {id: string, kind: string, state: string, progress: object,
     *   report: object|null, error: string|null}}>}
     */
    async status(jobId = "", kind = "") {
        const params = {};
        if (jobId) params.job_id = jobId;
        if (kind) params.kind = kind;
        return await this.get("/status", params);
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

    /**
     * Start a job that remuxes every video directly under src_dir into dest_dir without
     * embedded metadata (a ComfyUI-saved clip's own workflow/prompt JSON) — for cleaning
     * generated clips before adding them to a project's media folder by hand. Never touches
     * a .kdenlive file. Progress/completion arrive the same way as {@link archive}.
     * @param {{src_dir: string, dest_dir: string, dry_run?: boolean}} opts
     * @returns {Promise<{started: boolean, job_id: string}>}
     */
    async clean(opts) {
        return await this.post("/clean", opts);
    }

    /**
     * Every .kdenlive file under input/ or output/ (recursive), as absolute server paths —
     * for the Project field's browse tree. Kept separate from the general media-list endpoint,
     * whose relative paths suit ComfyUI node widgets, not Kdenlive's plain OS paths.
     * @param {"input"|"output"} folder
     * @returns {Promise<{files: string[]}>}
     */
    async browseFiles(folder = "input") {
        return await this.get("/browse_files", { folder });
    }

    /**
     * Every subdirectory under input/ or output/ (recursive, including empty ones), as
     * absolute server paths — for folder-picking fields (destinations, search folders,
     * clean-clips source/destination).
     * @param {"input"|"output"} folder
     * @returns {Promise<{root: string, dirs: string[]}>}
     */
    async browseDirs(folder = "input") {
        return await this.get("/browse_dirs", { folder });
    }
}

export const kdenliveApi = new KdenliveAPI();
