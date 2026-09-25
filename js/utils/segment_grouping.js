/**
 * Groups detected segments by their (VLM-suggested, unreliable across windows — see
 * nodes/source_profiles.py's per-window detection) setting_label, for the user to review,
 * merge, split, or rename before creating any Background from a group. No DOM, no ComfyUI
 * imports — see js/ui/source_profile_editor.js for the caller.
 */

export function normalizeSettingLabel(label) {
    return String(label || "").trim().toLowerCase().replace(/\s+/g, " ");
}

/**
 * @param {Array<{start_time:number, end_time:number, label:string, setting_label?:string}>} suggestions
 * @returns {Array<{key:string, label:string, indices:number[], representativeIndex:number}>}
 *   Groups in first-seen order. Suggestions with no/empty setting_label are grouped under key "".
 *   representativeIndex is the group's earliest-starting member (suggestions are assumed to
 *   already be in ascending time order, as detection always returns them, so it's simply the
 *   first index encountered for that key).
 */
export function groupSuggestionsBySetting(suggestions) {
    const groups = [];
    const byKey = new Map();
    (suggestions || []).forEach((seg, i) => {
        const key = normalizeSettingLabel(seg?.setting_label);
        let group = byKey.get(key);
        if (!group) {
            group = {
                key,
                label: (seg?.setting_label || "").trim() || "(unlabeled)",
                indices: [],
                representativeIndex: i,
            };
            byKey.set(key, group);
            groups.push(group);
        }
        group.indices.push(i);
    });
    return groups;
}
