/**
 * Dialogs for ticking or unticking a clip's subjects when that moves other subjects' letters.
 *
 * A clip's letters are the rank of each ticked subject in the profile's subject list (see
 * js/utils/clip_subjects.js), so ticking a subject that is listed before others, or unticking one,
 * silently re-points the text written against the old letters. These dialogs show the new letters
 * and offer to rewrite the clip's text so each subject keeps the letter the text used.
 */

import {
    CLIP_SLOT_LETTERS, CLIP_TEXT_FIELDS, clipSubjectsInOrder, letterMoves, movesAffectText, profileOrdered,
    setClipSubjects,
} from "../utils/clip_subjects.js";

function _mk(tag, props = {}, children = []) {
    const el = document.createElement(tag);
    Object.entries(props).forEach(([k, v]) => {
        if (k === "cls") el.className = v;
        else if (k === "style") Object.assign(el.style, v);
        else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
        else el[k] = v;
    });
    children.forEach(c => c != null && el.appendChild(typeof c === "string" ? document.createTextNode(c) : c));
    return el;
}

function _shell(title, body, buttons, onEscape) {
    const dialog = _mk("div", { cls: "spe-letters-dialog", role: "dialog", "aria-modal": "true" }, [
        _mk("div", { cls: "spe-letters-title" }, [title]),
        ...body,
        _mk("div", { cls: "spe-letters-buttons" }, buttons),
    ]);
    const overlay = _mk("div", { cls: "fbt-ce-modal-overlay" }, [dialog]);
    const onKey = e => { if (e.key === "Escape") onEscape(); };
    document.addEventListener("keydown", onKey);
    const close = () => { document.removeEventListener("keydown", onKey); overlay.remove(); };
    document.body.appendChild(overlay);
    return { overlay, dialog, close };
}

/** One "before / after" block for the first text field that differs (or the action). */
function _previewBlock(before, after) {
    const field = CLIP_TEXT_FIELDS.find(f => (before[f] || "") !== (after[f] || "")) || "action";
    const clip = t => String(t || "").trim() || "(empty)";
    return _mk("div", { cls: "spe-letters-preview" }, [
        _mk("div", { cls: "spe-letters-preview-col" }, [
            _mk("div", { cls: "spe-letters-preview-head" }, [`Before (${field})`]),
            _mk("div", { cls: "spe-letters-preview-text" }, [clip(before[field])]),
        ]),
        _mk("div", { cls: "spe-letters-preview-col" }, [
            _mk("div", { cls: "spe-letters-preview-head" }, [`After (${field})`]),
            _mk("div", { cls: "spe-letters-preview-text" }, [clip(after[field])]),
        ]),
    ]);
}

/**
 * Confirm a change to ONE clip's subjects that moves letters its text uses.
 * @param {{profile: object, clip: object, newIds: string[], title?: string, intro?: string}} opts
 *   `newIds` is the clip's subject set after the change (any order; letters follow the profile's order).
 * @returns {Promise<{ids: string[], remap: boolean}|null>} null when cancelled.
 */
export function openClipLettersDialog({ profile, clip, newIds, title = "Subject letters", intro = "" }) {
    return new Promise(resolve => {
        const labelOf = new Map((profile.subjects || []).map(s => [s.id, s.label || s.id]));
        const oldIds = clipSubjectsInOrder(profile, clip).map(s => s.id);
        const ids = profileOrdered(profile, newIds);
        let remap = true;

        const listEl = _mk("div", { cls: "spe-letters-list" });
        const removedEl = _mk("div", { cls: "spe-letters-note" });
        const previewHost = _mk("div");
        const remapCb = _mk("input", { type: "checkbox", checked: true, onchange: () => { remap = remapCb.checked; draw(); } });

        const draw = () => {
            const moves = letterMoves(oldIds, ids);
            const was = Object.fromEntries(moves.map(m => [m.id, m.from]));
            listEl.innerHTML = "";
            ids.forEach((id, i) => {
                const old = was[id];
                const letter = CLIP_SLOT_LETTERS[i] || "–";
                listEl.appendChild(_mk("div", { cls: "spe-letters-row" }, [
                    _mk("span", { cls: "spe-letters-badge" }, [`{${letter}}`]),
                    _mk("span", { cls: "spe-letters-label" }, [labelOf.get(id)]),
                    _mk("span", { cls: "spe-letters-was" },
                        [old && old !== letter ? `was {${old}}` : (old ? "" : "new")]),
                ]));
            });
            if (!ids.length) listEl.appendChild(_mk("div", { cls: "spe-letters-note" }, ["No subjects in this clip."]));

            const dropped = moves.filter(m => !ids.includes(m.id));
            removedEl.textContent = dropped.length
                ? "Removed from this clip: " + dropped.map(m => `${labelOf.get(m.id)} (was {${m.from}})`).join(", ")
                  + (remap ? ". Its name replaces the letter in the text." : ".")
                : "";

            previewHost.innerHTML = "";
            const after = setClipSubjects(profile, clip, ids, { remap });
            const changes = CLIP_TEXT_FIELDS.some(f => (clip[f] || "") !== (after[f] || ""));
            if (changes) previewHost.appendChild(_previewBlock(clip, after));
            else if (movesAffectText(clip, moves) && !remap) {
                previewHost.appendChild(_mk("div", { cls: "spe-letters-note warn" },
                    ["The text is left as written, so its letters will now refer to different subjects."]));
            }
        };

        const done = value => { close(); resolve(value); };
        const apply = _mk("button", { cls: "spe-btn primary", onclick: () => done({ ids: [...ids], remap }) }, ["Apply"]);
        const cancel = _mk("button", { cls: "spe-btn", onclick: () => done(null) }, ["Cancel"]);
        const { close } = _shell(title, [
            intro ? _mk("div", { cls: "spe-letters-note" }, [intro]) : null,
            _mk("div", { cls: "spe-letters-note" }, [
                "Letters follow the order of the profile's subject list: the first ticked subject in the list is {A}, the next {B}, and so on.",
            ]),
            listEl, removedEl,
            _mk("label", { cls: "spe-letters-check" }, [remapCb, " Rewrite the letters in this clip's text so each subject keeps its meaning"]),
            previewHost,
        ].filter(Boolean), [cancel, apply], () => done(null));
        draw();
        apply.focus();
    });
}

/**
 * Confirm a change that shifts letters in many clips (adding or removing a subject everywhere,
 * deleting it from the profile). The user only chooses which clips get their text rewritten.
 * @param {{profile: object, items: Array<{clip: object, newIds: string[]}>, title: string, intro?: string}} opts
 * @returns {Promise<Set<string>|null>} ids of the clips whose text should be rewritten; null = cancelled.
 */
export function openBulkLettersDialog({ profile, items, title, intro = "" }) {
    return new Promise(resolve => {
        const checked = new Set(items.map(it => it.clip.id));
        const previewHost = _mk("div");
        const listEl = _mk("div", { cls: "spe-letters-list tall" });
        let focus = items[0];

        const showPreview = () => {
            previewHost.innerHTML = "";
            if (!focus) return;
            previewHost.appendChild(_previewBlock(focus.clip, setClipSubjects(profile, focus.clip, focus.newIds)));
        };
        items.forEach(it => {
            const moves = letterMoves(clipSubjectsInOrder(profile, it.clip).map(s => s.id), it.newIds)
                .filter(m => m.from !== m.to);
            const cb = _mk("input", { type: "checkbox", checked: true, onchange: () => {
                cb.checked ? checked.add(it.clip.id) : checked.delete(it.clip.id);
            } });
            const row = _mk("label", { cls: "spe-letters-row clickable", onmouseenter: () => { focus = it; showPreview(); } }, [
                cb,
                _mk("span", { cls: "spe-letters-label" }, [it.clip.label || it.clip.id]),
                _mk("span", { cls: "spe-letters-was" }, [moves.map(m => `{${m.from}}→${m.to ? `{${m.to}}` : "name"}`).join("  ")]),
            ]);
            listEl.appendChild(row);
        });

        const done = value => { close(); resolve(value); };
        const apply = _mk("button", { cls: "spe-btn primary", onclick: () => done(new Set(checked)) }, ["Apply"]);
        const cancel = _mk("button", { cls: "spe-btn", onclick: () => done(null) }, ["Cancel"]);
        const { close } = _shell(title, [
            intro ? _mk("div", { cls: "spe-letters-note" }, [intro]) : null,
            _mk("div", { cls: "spe-letters-note" }, [
                `${items.length} clip(s) use letters that would change. Ticked clips get their text rewritten ` +
                "so each subject keeps its meaning; unticked clips keep their text as written.",
            ]),
            listEl, previewHost,
        ].filter(Boolean), [cancel, apply], () => done(null));
        showPreview();
        apply.focus();
    });
}
