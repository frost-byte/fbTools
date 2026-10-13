/**
 * Pure helpers for Source Profiles whose media is a still image (or a series of them).
 * A still has no timeline of its own, so its clips are laid end to end by length, and each clip
 * is built from one of the profile's frames. No DOM, no ComfyUI imports.
 */

const DEFAULT_LENGTH = 5;

/**
 * Clips laid back to back in list order, keeping each clip's own length (end - start, or
 * `fallbackLength` when that isn't positive). Returns new objects; the input is untouched.
 * Stored start/end of a still clip are only a length: imports used to give every clip 0..N,
 * which drew them all on top of each other.
 */
export function layoutStillClips(clips, fallbackLength = DEFAULT_LENGTH) {
    let t = 0;
    return (clips || []).map(c => {
        const len = Number(c.end_time) - Number(c.start_time);
        const dur = len > 0 ? len : fallbackLength;
        const out = { ...c, start_time: t, end_time: t + dur };
        t += dur;
        return out;
    });
}

/**
 * The still a clip is built from: its frame, else the profile's first frame, else the primary
 * media. Mirrors utils/source_profiles.py::resolve_clip_media. `null` when there is none.
 * @returns {{filename: string, dir: string, label: string}|null}
 */
export function clipFrame(profile, clip) {
    const frames = profile?.frames || [];
    const f = frames.find(fr => fr.id === clip?.frame_id) || frames[0];
    if (f?.filename) return { filename: f.filename, dir: f.dir || "input", label: f.label || f.filename };
    if (profile?.media_filename) {
        return { filename: profile.media_filename, dir: profile.media_dir || "input", label: profile.media_filename };
    }
    return null;
}
