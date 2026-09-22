/**
 * Pure helper for SceneCastBuild's clip video preview: decides whether to play a clip's
 * already-built proxy or fall back to seeking/looping the full source video, and what caption
 * to show. No DOM, no ComfyUI imports — see js/nodes/scene_cast_build.js for the caller.
 */

/**
 * @param {{clips: Array<{clip_id: string, fresh: boolean, proxy_path: string|null}>}|null} proxyStatus
 *   The result of sourceProfilesApi.proxyStatus(profile_id), or null if that call hasn't
 *   resolved yet / failed.
 * @param {{id: string, start_time: number, end_time: number}|null} clip
 * @param {{media_filename: string, media_dir: string}|null} profileMeta
 * @returns {{kind: "proxy"|"fallback"|"unavailable", proxyPath?: string, filename?: string,
 *   dir?: string, startTime?: number, endTime?: number, caption: string}}
 */
export function resolveClipPreviewSource(proxyStatus, clip, profileMeta) {
    if (!clip) {
        return { kind: "unavailable", caption: "No clip selected." };
    }

    const entry = proxyStatus?.clips?.find(c => c.clip_id === clip.id);
    if (entry?.fresh && entry.proxy_path) {
        return {
            kind: "proxy",
            proxyPath: entry.proxy_path,
            caption: "Proxy clip (silent — proxies are generated without audio).",
        };
    }

    if (!profileMeta?.media_filename) {
        return { kind: "unavailable", caption: "Source video not available." };
    }

    const start = Number(clip.start_time) || 0;
    const end = Number(clip.end_time) || start;
    return {
        kind: "fallback",
        filename: profileMeta.media_filename,
        dir: profileMeta.media_dir || "input",
        startTime: start,
        endTime: end,
        caption: `Full source, seeked to clip bounds (${start.toFixed(1)}s–${end.toFixed(1)}s). No proxy built yet.`,
    };
}
