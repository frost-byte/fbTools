/**
 * Tools tab — standalone, bundle/composition-agnostic generation utilities.
 * Currently: Qwen-Image-2.1 photo restoration (nodes/qwen21_photo_restore.py).
 */

import { compositionsApi } from "../api/compositions.js";
import { toolsApi } from "../api/tools.js";
import { buildFileTree } from "./file_tree.js";
import { mk as _mk, toast as _toast, ceViewUrl as _ceViewUrl } from "./library_common.js";
import { makeImageZoomable } from "./lightbox.js";

const _IS_IMAGE = f => /\.(png|jpe?g|webp|bmp|tiff?)$/i.test(f);

export function renderToolsPanel(rootEl) {
    rootEl.innerHTML = "";

    const panel = _mk("div", { cls: "fbt-tools-panel" });

    panel.appendChild(_mk("div", { cls: "fbt-tools-toolbar" }, [
        _mk("span", { cls: "fbt-tools-title", textContent: "Photo Restoration" }),
    ]));

    const content = _mk("div", { cls: "fbt-tools-content" });

    // ── Source picker ──────────────────────────────────────────────────────────
    let selFile = null; // {path, folder}

    const selFileEl   = _mk("div", { cls: "fbt-ce-outfit-sel-file", textContent: "— no file selected —" });
    const previewWrap = _mk("div", { cls: "fbt-ce-outfit-preview-wrap" });
    const previewImg  = _mk("img",  { cls: "fbt-be-img-preview", alt: "" });
    previewImg.style.display = "none";
    makeImageZoomable(previewImg);
    previewWrap.appendChild(previewImg);

    const _applySelection = (path, folder) => {
        selFile = { path, folder };
        selFileEl.textContent = path.split("/").pop();
        selFileEl.title = path;
        previewImg.src = _ceViewUrl(path, folder);
        previewImg.style.display = "";
        restoreBtn.disabled = false;
    };

    const tree = buildFileTree({
        inputFiles:  [],
        outputFiles: [],
        filter:      _IS_IMAGE,
        isSelected:  (p, d) => selFile?.path === p && selFile?.folder === d,
        onSelect:    (p, d) => _applySelection(p, d),
        emptyText:   "No images in {dir}/",
        initialDir:  "input",
    });

    const browserSection = _mk("div", { cls: "fbt-ce-outfit-browser" }, [
        _mk("div", { cls: "fbt-ce-label", textContent: "Source photo" }),
        tree.el,
        selFileEl,
        previewWrap,
    ]);

    // ── Prompt controls ────────────────────────────────────────────────────────
    const hintEl = _mk("textarea", {
        cls: "fbt-ce-textarea", rows: 2,
        placeholder: "Optional hint, e.g. \"focus on the tear across the middle\" — leave blank "
            + "to use the default restoration prompt as-is.",
    });

    const promptOverrideEl = _mk("textarea", {
        cls: "fbt-ce-textarea", rows: 4,
        placeholder: "Advanced: full prompt override — replaces the default restoration prompt "
            + "entirely. The hint above is ignored when this is filled in.",
    });
    const promptDetails = _mk("details", { cls: "fbt-tools-advanced" }, [
        _mk("summary", { textContent: "Advanced: full prompt override" }),
        promptOverrideEl,
    ]);

    // ── Actions ────────────────────────────────────────────────────────────────
    const restoreBtn = _mk("button", { cls: "fbt-ce-btn fbt-ce-btn-primary", textContent: "🧼 Restore Photo",
        disabled: true,
        onclick: async () => {
            if (!selFile) { alert("Select a source photo first."); return; }
            restoreBtn.disabled    = true;
            restoreBtn.textContent = "Restoring…";
            const startedAt = performance.now();
            try {
                const data = await toolsApi.restorePhoto(selFile.path, {
                    folder:      selFile.folder || "input",
                    restoreHint: hintEl.value,
                    prompt:      promptOverrideEl.value,
                });
                const elapsed = Math.round((performance.now() - startedAt) / 1000);
                resultImg.src = _ceViewUrl(data.file, data.folder || "output");
                resultWrap.style.display = "";
                resultPathEl.textContent = data.file;
                resultPathEl.title = data.file;
                useAsSourceBtn.dataset.file   = data.file;
                useAsSourceBtn.dataset.folder = data.folder || "output";
                _toast(`Restored in ${elapsed}s`, "success");
            } catch (e) { alert(`Restore failed: ${e.message}`); }
            finally {
                restoreBtn.disabled    = false;
                restoreBtn.textContent = "🧼 Restore Photo";
            }
        } });

    const freeVramBtn = _mk("button", { cls: "fbt-ce-btn sm", textContent: "Free VRAM",
        title: "Unload resident models now — same effect as Manager's own \"Free model and node "
            + "cache\" button.",
        onclick: async () => {
            freeVramBtn.disabled    = true;
            freeVramBtn.textContent = "Freeing…";
            try {
                await compositionsApi.freeH3Vram();
                _toast("VRAM freed", "success");
            } catch (e) { alert(`Free VRAM failed: ${e.message}`); }
            finally {
                freeVramBtn.disabled    = false;
                freeVramBtn.textContent = "Free VRAM";
            }
        } });

    const actionRow = _mk("div", { cls: "fbt-ce-outfit-action-row" }, [restoreBtn, freeVramBtn]);

    const hint = _mk("div", { cls: "fbt-ce-hint", textContent:
        "Runs the Qwen-Image-2.1 photo-restoration workflow (templates/qwen21_photo_restore.api.json) "
        + "directly on this ComfyUI server — your own open canvas isn't touched. Needs that template "
        + "file exported first. Output lands under output/fbtools/qwen21_photo_restore/. The model "
        + "stays resident after each run by default so repeated runs stay fast — click \"Free VRAM\" "
        + "to reclaim it now, or flip \"Unload model after each run\" in Settings > H3 Background "
        + "Plate (shared across all H3/Qwen tools) to make that the default." });

    // ── Result ─────────────────────────────────────────────────────────────────
    const resultImg     = _mk("img", { cls: "fbt-be-img-preview", alt: "" });
    makeImageZoomable(resultImg);
    const resultPathEl  = _mk("div", { cls: "fbt-ce-outfit-sel-file" });
    const useAsSourceBtn = _mk("button", { cls: "fbt-ce-btn fbt-ce-btn-sm", textContent: "Use as source",
        title: "Load this result back into the source picker above — e.g. to run a second pass with "
            + "a different hint.",
        onclick: () => {
            const file   = useAsSourceBtn.dataset.file;
            const folder = useAsSourceBtn.dataset.folder || "output";
            if (file) _applySelection(file, folder);
        } });
    const resultWrap = _mk("div", { cls: "fbt-tools-result" }, [
        _mk("div", { cls: "fbt-ce-label", textContent: "Result" }),
        resultImg,
        resultPathEl,
        useAsSourceBtn,
    ]);
    resultWrap.style.display = "none";

    content.append(browserSection, hintEl, promptDetails, actionRow, hint, resultWrap);
    panel.appendChild(content);
    rootEl.appendChild(panel);

    // ── Load the source image lists (self-sufficient — doesn't depend on the
    // Compose/Assets tab having been opened first) ────────────────────────────
    (async () => {
        try {
            const [inImg, outImg] = await Promise.all([
                compositionsApi.listMedia("image", true),
                compositionsApi.listMedia("image", true, "output"),
            ]);
            tree.rebuild({ inputFiles: inImg.files ?? [], outputFiles: outImg.files ?? [] });
        } catch (e) {
            console.error("[fbTools] Tools tab: failed to load media lists", e);
        }
    })();
}
