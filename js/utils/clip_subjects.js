/**
 * The subjects of a Source Profile clip in the order that gives them their {A}/{B}/… letters.
 *
 * The PROFILE's subject list is the rule: of the subjects ticked in a clip, the one listed first in
 * the profile is {A}, the next {B}, and so on. The order they were ticked in never matters.
 * Mirrors utils/source_profiles.py::clip_subjects_in_order. Pure; no DOM.
 */

export const CLIP_SLOT_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"];

/**
 * @param {{subjects?: Array<{id: string}>}} profile
 * @param {{subjects?: string[]}} clip
 * @returns {Array<object>} the profile's subject objects that the clip has ticked, in profile order;
 *   ids the profile no longer has are skipped and repeats ignored.
 */
export function clipSubjectsInOrder(profile, clip) {
    return profileOrdered(profile, clip?.subjects || []).map(id => profile.subjects.find(s => s.id === id));
}

/** `ids` reduced to the profile's subjects (stale ids and repeats dropped), in profile order. */
export function profileOrdered(profile, ids) {
    const wanted = new Set(ids || []);
    const seen = new Set();
    const out = [];
    for (const s of profile?.subjects || []) {
        if (wanted.has(s.id) && !seen.has(s.id)) {
            seen.add(s.id);
            out.push(s.id);
        }
    }
    return out;
}

/** `{subjectId: "A"}` for the clip's first ten subjects. */
export function clipSlotLetters(profile, clip) {
    const letters = {};
    clipSubjectsInOrder(profile, clip).slice(0, CLIP_SLOT_LETTERS.length)
        .forEach((s, i) => { letters[s.id] = CLIP_SLOT_LETTERS[i]; });
    return letters;
}

// ── Changing a clip's subjects without changing what its text means ───────────────────────────────────
//
// {A}/{B}/… in a clip's text are ranks within the ticked subjects, so ticking or unticking a subject
// that sits before others moves their letters and changes who the text is about. These helpers work
// out the letter moves and rewrite the text so each subject keeps the letter it had when the text
// was written.

/** Clip fields that carry {A}/{B}/… placeholders. */
export const CLIP_TEXT_FIELDS = ["action", "style", "overall_soundscape", "non_diegetic_music"];

const _LETTER_RE = /\{([A-J])\}/g;

/** Letters ("A".."J") that appear as {X} in any of the clip's text fields. */
export function usedLetters(clip) {
    const used = new Set();
    for (const f of CLIP_TEXT_FIELDS) {
        for (const m of String(clip?.[f] || "").matchAll(_LETTER_RE)) used.add(m[1]);
    }
    return used;
}

/**
 * How each subject's letter changes going from `oldIds` to `newIds` (both in profile order).
 * @returns {Array<{id: string, from: string, to: string}>} one entry per old subject; `to` is "" when
 *   the subject is no longer in the clip. Subjects past the tenth have no letter ("").
 */
export function letterMoves(oldIds, newIds) {
    const letter = (ids, id) => {
        const i = ids.indexOf(id);
        return i >= 0 && i < CLIP_SLOT_LETTERS.length ? CLIP_SLOT_LETTERS[i] : "";
    };
    return oldIds.map(id => ({ id, from: letter(oldIds, id), to: letter(newIds, id) }));
}

/** True when the move changes a letter that the clip's text actually uses. */
export function movesAffectText(clip, moves) {
    const used = usedLetters(clip);
    return moves.some(m => m.from && m.from !== m.to && used.has(m.from));
}

/**
 * Rewrite {X} placeholders in one pass (so an A<->B swap does not undo itself).
 * `moves` as from letterMoves(); a subject that left the clip becomes `orphanLabels[id]` when given,
 * else its placeholder is left alone.
 */
export function remapLetters(text, moves, orphanLabels = {}) {
    const byFrom = new Map(moves.filter(m => m.from).map(m => [m.from, m]));
    return String(text || "").replace(_LETTER_RE, (whole, L) => {
        const m = byFrom.get(L);
        if (!m) return whole;
        if (m.to) return `{${m.to}}`;
        return orphanLabels[m.id] || whole;
    });
}

/**
 * The clip with its subjects set to `newIds` (stored in profile order) and, when `remap` is on, its
 * text rewritten so every subject that was in the clip keeps its meaning. `newIds` may contain
 * subjects the clip did not have. Subjects that were dropped are replaced in the text by their label.
 */
export function setClipSubjects(profile, clip, newIds, { remap = true } = {}) {
    const oldIds = clipSubjectsInOrder(profile, clip).map(s => s.id);
    const ordered = profileOrdered(profile, newIds);
    const moves = letterMoves(oldIds, ordered);
    const labels = Object.fromEntries((profile?.subjects || []).map(s => [s.id, s.label || s.id]));
    const out = { ...clip, subjects: ordered };
    if (remap) {
        for (const f of CLIP_TEXT_FIELDS) {
            if (clip?.[f]) out[f] = remapLetters(clip[f], moves, labels);
        }
    }
    return out;
}
