/**
 * Asset sub-tab lists (Backgrounds, Camera, Sound, Outfits) for the Assets tab.
 *
 * Each list is a searchable set of cards; clicking a card opens the existing
 * editor (background/outfit modals, or the preset modal defined here). All data
 * lives in the shared library store, and every save/delete fires
 * `fbt:library-changed` so the Compose editor stays in step.
 */

import { compositionsApi } from "../api/compositions.js";
import { llmApi } from "../api/llm.js";
import { lib, notifyLibraryChanged } from "./library_store.js";
import { mk as _mk, toast as _toast, ceViewUrl, isVideoFile } from "./library_common.js";
import { openBackgroundEditor } from "./background_editor.js";
import { openOutfitEditor } from "./outfit_editor.js";

export const ASSET_TABS = [
    { id: "backgrounds", label: "Backgrounds", newLabel: "+ New Background", search: "Search backgrounds…" },
    { id: "cameraPresets", label: "Camera",   newLabel: "+ New Preset",     search: "Search camera presets…" },
    { id: "soundPresets",  label: "Sound",    newLabel: "+ New Preset",     search: "Search sound presets…" },
    { id: "outfits",       label: "Outfits",  newLabel: "+ New Outfit",     search: "Search outfits…" },
];

export function isAssetTab(id) { return ASSET_TABS.some(t => t.id === id); }

/** Load everything the asset editors need into the shared library store. */
export async function loadLibrary() {
    const [subj, bg, cam, snd, outfits, sam2, inImg, outImg, inVid, outVid, llm] = await Promise.allSettled([
        compositionsApi.listSubjects(),
        compositionsApi.listBackgrounds(),
        compositionsApi.listCameraPresets(),
        compositionsApi.listSoundPresets(),
        compositionsApi.getOutfitRegistry(),
        fetch("/fbtools/outfits/sam2_status").then(r => r.json()),
        compositionsApi.listMedia("image", true),
        compositionsApi.listMedia("image", true, "output"),
        compositionsApi.listMedia("video", true),
        compositionsApi.listMedia("video", true, "output"),
        llmApi.status(),
    ]);
    lib.subjects       = subj.value?.subjects       ?? lib.subjects;
    lib.backgrounds    = bg.value?.backgrounds      ?? [];
    lib.cameraPresets  = cam.value?.camera_presets  ?? [];
    lib.soundPresets   = snd.value?.sound_presets   ?? [];
    lib.outfits        = outfits.value?.outfits     ?? {};
    lib.sam2           = sam2.status === "fulfilled" ? sam2.value : null;
    lib.mediaInImages  = inImg.value?.files  ?? [];
    lib.mediaOutImages = outImg.value?.files ?? [];
    lib.mediaInVideos  = inVid.value?.files  ?? [];
    lib.mediaOutVideos = outVid.value?.files ?? [];
    lib.llmLoaded      = llm.value?.loaded_model    ?? null;
    lib.llmVision      = llm.value?.supports_vision ?? false;
    lib.llmNativeVideo = llm.value?.native_video    ?? false;
}

// ── Preset modal ───────────────────────────────────────────────────────────────

function _openPresetEditor(kind, existing) {
    const isCam = kind === "cameraPresets";
    const isNew = !existing;
    const nameEl = _mk("input", { cls: "fbt-ce-input", type: "text", value: existing?.name || "",
        placeholder: "Preset name" });
    const textEl = _mk("textarea", { cls: "fbt-ce-textarea", rows: 6, value: existing?.description || "",
        placeholder: isCam ? "Camera movement / framing…" : "Sound / ambience description…" });

    const overlay = _mk("div", { cls: "fbt-ce-modal-overlay",
        onclick: e => { if (e.target === overlay) overlay.remove(); } });
    const modal = _mk("div", { cls: "fbt-ce-modal" });
    overlay.appendChild(modal);
    modal.appendChild(_mk("div", { cls: "fbt-ce-modal-title",
        textContent: isNew ? (isCam ? "New Camera Preset" : "New Sound Preset") : `Edit: ${existing.name || existing.id}` }));
    const row = (label, el) => _mk("div", { cls: "fbt-ce-row" }, [
        _mk("label", { cls: "fbt-ce-label", textContent: label }),
        _mk("div", { cls: "fbt-ce-input-wrap" }, [el]),
    ]);
    modal.append(row("Name", nameEl), row("Text", textEl));

    const save = _mk("button", { cls: "fbt-ce-btn fbt-ce-btn-primary", textContent: isNew ? "Add" : "Save",
        onclick: async () => {
            const name = nameEl.value.trim();
            const description = textEl.value.trim();
            if (!name || !description) { alert("Name and text are required."); return; }
            const preset = { name, description };
            if (!isNew) preset.id = existing.id;
            try {
                if (isCam) await compositionsApi.saveCameraPreset(preset);
                else       await compositionsApi.saveSoundPreset(preset);
                await _reloadKind(kind);
                notifyLibraryChanged(kind);
                _toast("Preset saved", "success");
                overlay.remove();
            } catch (e) { alert(`Save failed: ${e.message}`); }
        } });
    const buttons = [_mk("button", { cls: "fbt-ce-btn", textContent: "Cancel", onclick: () => overlay.remove() }), save];
    modal.appendChild(_mk("div", { cls: "fbt-ce-modal-btns" }, buttons));
    document.body.appendChild(overlay);
    nameEl.focus();
}

async function _reloadKind(kind) {
    if (kind === "cameraPresets") lib.cameraPresets = (await compositionsApi.listCameraPresets()).camera_presets ?? [];
    else if (kind === "soundPresets") lib.soundPresets = (await compositionsApi.listSoundPresets()).sound_presets ?? [];
    else if (kind === "backgrounds") lib.backgrounds = (await compositionsApi.listBackgrounds()).backgrounds ?? [];
    else if (kind === "outfits") lib.outfits = (await compositionsApi.getOutfitRegistry())?.outfits ?? {};
}

// ── Items per kind ─────────────────────────────────────────────────────────────

const _snippet = (s, n = 140) => {
    s = (s || "").replace(/\s+/g, " ").trim();
    return s.length > n ? s.slice(0, n) + "…" : s;
};

/** Thumbnail URL for the first still-image reference, or "" (videos are skipped). */
function _thumbUrl(refs) {
    for (const r of refs || []) {
        const file = typeof r === "string" ? r : r?.file;
        if (file && !isVideoFile(file)) return ceViewUrl(file, (typeof r === "object" && r.folder) || "input");
    }
    return "";
}

/** Normalised rows: {id, name, meta, summary, thumb, open, remove}. */
function _items(kind) {
    if (kind === "backgrounds") {
        return lib.backgrounds.map(b => ({
            id: b.id, name: b.name || b.id, summary: _snippet(b.description), thumb: _thumbUrl(b.reference_images),
            meta: (b.reference_images?.length ? `${b.reference_images.length} ref img` : ""),
            open: () => openBackgroundEditor(b),
            remove: async () => {
                await compositionsApi.deleteBackground(b.id);
                await _reloadKind("backgrounds");
                notifyLibraryChanged("backgrounds", { deletedId: b.id });
            },
        }));
    }
    if (kind === "outfits") {
        return Object.entries(lib.outfits)
            .sort(([, a], [, b]) => (a.name || "").localeCompare(b.name || ""))
            .map(([id, o]) => ({
                id, name: o.name || id, summary: _snippet(o.description), thumb: _thumbUrl(o.reference_images),
                meta: (o.reference_images?.length ? `${o.reference_images.length} ref img · ` : "") + id,
                open: () => openOutfitEditor(id),
                remove: async () => {
                    await compositionsApi.deleteOutfit(id);
                    delete lib.outfits[id];
                    notifyLibraryChanged("outfits", { deletedId: id });
                },
            }));
    }
    const isCam = kind === "cameraPresets";
    return (isCam ? lib.cameraPresets : lib.soundPresets).map(p => ({
        id: p.id, name: p.name || p.id, summary: _snippet(p.description, 240), meta: "",
        open: () => _openPresetEditor(kind, p),
        remove: async () => {
            if (isCam) await compositionsApi.deleteCameraPreset(p.id);
            else       await compositionsApi.deleteSoundPreset(p.id);
            await _reloadKind(kind);
            notifyLibraryChanged(kind, { deletedId: p.id });
        },
    }));
}

export function startNewAsset(kind) {
    if (kind === "backgrounds") openBackgroundEditor(null);
    else if (kind === "outfits") openOutfitEditor(null);
    else _openPresetEditor(kind, null);
}

/** Render the card list for `kind` into `container`, filtered by `query`. */
export function renderAssetList(kind, container, query = "", rerender = () => {}) {
    container.innerHTML = "";
    const q = query.trim().toLowerCase();
    const all = _items(kind);
    const items = q ? all.filter(i => `${i.name} ${i.id} ${i.summary}`.toLowerCase().includes(q)) : all;

    if (!items.length) {
        const label = ASSET_TABS.find(t => t.id === kind)?.label.toLowerCase() ?? "items";
        container.appendChild(_mk("div", { cls: "fbt-be-empty",
            textContent: all.length ? "Nothing matches the search." : `No ${label} yet. Click ${ASSET_TABS.find(t => t.id === kind)?.newLabel} to create one.` }));
        return;
    }
    items.forEach(it => {
        const card = _mk("div", { cls: "fbt-be-card fbt-be-card-clickable", onclick: it.open });
        const top = _mk("div", { cls: "fbt-be-card-top" }, [
            it.thumb ? _mk("img", { cls: "fbt-be-thumb", src: it.thumb, alt: "", loading: "lazy" }) : null,
            _mk("span", { cls: "fbt-be-card-name", textContent: it.name }),
            it.meta ? _mk("span", { cls: "fbt-be-card-meta", textContent: it.meta }) : null,
        ]);
        card.appendChild(top);
        if (it.summary) card.appendChild(_mk("div", { cls: "fbt-be-card-summary", textContent: it.summary }));
        card.appendChild(_mk("div", { cls: "fbt-be-card-actions" }, [
            _mk("button", { cls: "fbt-ce-icon-btn", title: "Edit", textContent: "✎",
                onclick: e => { e.stopPropagation(); it.open(); } }),
            _mk("button", { cls: "fbt-ce-icon-btn fbt-ce-danger", title: "Delete", textContent: "✕",
                onclick: async e => {
                    e.stopPropagation();
                    if (!confirm(`Delete "${it.name}"?`)) return;
                    try { await it.remove(); _toast(`Deleted "${it.name}"`, "success"); }
                    catch (err) { alert(`Delete failed: ${err.message}`); }
                    rerender();
                } }),
        ]));
        container.appendChild(card);
    });
}
