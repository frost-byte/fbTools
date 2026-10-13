/**
 * Source Profile import panel: turn a legacy Story (or a set of story-less Scenes) into an
 * image-driven Source Profile. Always previews first (dry run); nothing is written until Import.
 *
 * Convention: docs/scene_story_import.md.
 */

import { sourceProfilesApi } from "../api/source_profiles.js";

const SOURCE_LABELS = {
    resolved:     "video prompt",
    image_prompt: "image prompt (review)",
    empty:        "needs description",
    manual:       "edited",
};

function _el(tag, props = {}, children = []) {
    const el = document.createElement(tag);
    Object.entries(props).forEach(([k, v]) => {
        if (k === "cls") el.className = v;
        else if (k === "style") Object.assign(el.style, v);
        else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
        else el[k] = v;
    });
    children.forEach(c => c && el.appendChild(typeof c === "string" ? document.createTextNode(c) : c));
    return el;
}

/** Text for the one-line summary under the preview. */
export function summarizePlan(plan) {
    const s = plan?.stats || {};
    const parts = [`${s.imported ?? 0} of ${s.scenes ?? 0} scene(s) will be imported`];
    if (s.skipped) parts.push(`${s.skipped} skipped`);
    if (s.resolved) parts.push(`${s.resolved} with a video prompt`);
    if (s.image_prompt) parts.push(`${s.image_prompt} using the image prompt as a draft`);
    if (s.empty) parts.push(`${s.empty} needing a description`);
    if (s.cleaned) parts.push(`${s.cleaned} had image-model instructions removed`);
    if (s.kept_manual) parts.push(`${s.kept_manual} hand-written action(s) kept`);
    return parts.join(" · ");
}

/**
 * @param {HTMLElement} container
 * @param {{toast: Function, errMsg: Function, onImported: (profileId: string) => void, onClose: () => void}} opts
 */
export async function renderImportPanel(container, { toast, errMsg, onImported, onClose }) {
    const state = {
        data: { stories: [], scenes: [] }, mode: "story", story: "", picked: new Set(),
        filter: "", name: "", clipLength: 5, plan: null, busy: false,
    };
    const root = _el("div", { cls: "spe-import", style: {
        border: "1px solid #3a3a3a", borderRadius: "4px", padding: "8px", margin: "6px 0",
        display: "flex", flexDirection: "column", gap: "6px",
    } });
    container.appendChild(root);

    try {
        state.data = await sourceProfilesApi.importable();
        state.story = state.data.stories[0]?.name || "";
        if (!state.data.stories.length && state.data.scenes.length) state.mode = "scenes";
    } catch (err) {
        toast(`Could not list scenes: ${errMsg(err)}`, "error");
    }

    const body = () => ({
        ...(state.mode === "story" ? { story: state.story } : { scenes: [...state.picked], profile_name: state.name }),
        clip_length: state.clipLength,
    });

    async function run(kind) {
        if (state.busy) return;
        state.busy = true;
        draw();
        try {
            if (kind === "preview") {
                state.plan = (await sourceProfilesApi.importPreview(body())).plan;
            } else {
                const res = await sourceProfilesApi.importApply(body());
                toast(`Imported ${res.plan.stats.imported} scene(s) into "${res.plan.profile.name}"`, "success");
                onImported?.(res.id);
                return;
            }
        } catch (err) {
            state.plan = null;
            toast(errMsg(err), "error");
        } finally {
            state.busy = false;
        }
        draw();
    }

    function planTable(plan) {
        const warnByScene = {};
        plan.warnings.forEach(w => { (warnByScene[w.scene] ||= []).push(w); });
        const rows = plan.scenes.map(r => {
            const warns = warnByScene[r.scene] || [];
            const bad = warns.some(w => w.level === "error");
            return _el("tr", { style: { opacity: r.status === "skipped" ? "0.55" : "1" } }, [
                _el("td", {}, [r.scene]),
                _el("td", {}, [r.status === "skipped" ? "skipped" : (SOURCE_LABELS[r.action_source] || r.action_source)]),
                _el("td", { style: { maxWidth: "260px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" },
                            title: r.action_preview }, [r.action_preview]),
                _el("td", { style: { color: bad ? "#e57373" : "#d9b45a" } },
                    [warns.filter(w => w.level !== "info").map(w => w.message).join(" ")]),
            ]);
        });
        return _el("div", { style: { maxHeight: "260px", overflow: "auto", fontSize: "11px" } }, [
            _el("table", { style: { width: "100%", borderCollapse: "collapse" } }, [
                _el("thead", {}, [_el("tr", {}, ["Scene", "Action from", "Action", "Notes"].map(h =>
                    _el("th", { style: { textAlign: "left", padding: "2px 6px", position: "sticky", top: "0",
                                         background: "#222" } }, [h])))]),
                _el("tbody", {}, rows),
            ]),
        ]);
    }

    function draw() {
        root.innerHTML = "";
        root.appendChild(_el("div", { style: { display: "flex", alignItems: "center", gap: "6px" } }, [
            _el("strong", {}, ["Import scenes as a Source Profile"]),
            _el("button", { cls: "spe-btn sm ghost", style: { marginLeft: "auto" }, onclick: () => onClose?.() }, ["Close"]),
        ]));

        const modeSel = _el("select", { onchange: () => { state.mode = modeSel.value; state.plan = null; draw(); } },
            [["story", "A story"], ["scenes", "Scenes without a story"]].map(([v, l]) => {
                const o = _el("option", { value: v }, [l]); o.selected = v === state.mode; return o;
            }));
        root.appendChild(_el("div", { cls: "spe-form-row" }, [_el("label", {}, ["Import"]), modeSel]));

        if (state.mode === "story") {
            const sel = _el("select", { onchange: () => { state.story = sel.value; state.plan = null; draw(); } },
                state.data.stories.map(s => {
                    const o = _el("option", { value: s.name },
                        [`${s.name} (${s.scenes.length} scenes)${s.imported ? " — already imported, will update" : ""}`]);
                    o.selected = s.name === state.story; return o;
                }));
            root.appendChild(_el("div", { cls: "spe-form-row" }, [_el("label", {}, ["Story"]), sel]));
        } else {
            const filter = _el("input", { type: "text", placeholder: "Filter scenes…", value: state.filter,
                oninput: () => { state.filter = filter.value; drawList(); } });
            const list = _el("div", { style: { maxHeight: "160px", overflow: "auto", fontSize: "11px" } });
            const drawList = () => {
                list.innerHTML = "";
                const q = state.filter.toLowerCase();
                state.data.scenes.filter(s => !q || s.name.toLowerCase().includes(q)).forEach(s => {
                    const cb = _el("input", { type: "checkbox", checked: state.picked.has(s.name), disabled: !s.has_base,
                        onchange: () => { cb.checked ? state.picked.add(s.name) : state.picked.delete(s.name); state.plan = null; } });
                    list.appendChild(_el("label", { style: { display: "flex", gap: "4px", opacity: s.has_base ? "1" : "0.5" },
                        title: s.has_base ? "" : "No base image" }, [cb, s.name]));
                });
                if (!list.children.length) list.appendChild(_el("div", { style: { color: "#888" } }, ["No scenes."]));
            };
            drawList();
            root.appendChild(_el("div", { cls: "spe-form-row" }, [_el("label", {}, ["Scenes"]), filter]));
            root.appendChild(list);
            const nameEl = _el("input", { type: "text", placeholder: "Profile name", value: state.name,
                oninput: () => { state.name = nameEl.value; state.plan = null; } });
            root.appendChild(_el("div", { cls: "spe-form-row" }, [_el("label", {}, ["Profile name"]), nameEl]));
        }

        const lenEl = _el("input", { type: "number", min: 1, step: 0.5, value: state.clipLength,
            onchange: () => { state.clipLength = parseFloat(lenEl.value) || 5; state.plan = null; } });
        root.appendChild(_el("div", { cls: "spe-form-row" }, [_el("label", { title: "Length of video generated from each scene image" }, ["Clip length (s)"]), lenEl]));

        const ready = state.mode === "story" ? !!state.story : (state.picked.size > 0 && !!state.name.trim());
        const canApply = !!state.plan && state.plan.stats.imported > 0;
        root.appendChild(_el("div", { style: { display: "flex", gap: "6px" } }, [
            _el("button", { cls: "spe-btn", disabled: !ready || state.busy, onclick: () => run("preview") },
                [state.busy ? "Working…" : "Preview"]),
            _el("button", { cls: "spe-btn primary", disabled: !canApply || state.busy, onclick: () => run("apply") }, ["Import"]),
        ]));

        if (state.plan) {
            root.appendChild(_el("div", { style: { fontSize: "11px", color: "#aaa" } }, [summarizePlan(state.plan)]));
            root.appendChild(planTable(state.plan));
        }
    }

    draw();
}
