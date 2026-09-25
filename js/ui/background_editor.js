/**
 * Background editor modal — used by the Assets tab and the Compose editor.
 * Data comes from / goes to the shared library store; callers react to the
 * `fbt:library-changed` event instead of passing callbacks.
 */

import { compositionsApi } from "../api/compositions.js";
import { llmApi } from "../api/llm.js";
import { buildFileTree } from "./file_tree.js";
import { makeEntry, buildHistorySection } from "../utils/llm_history.js";
import { lib, notifyLibraryChanged } from "./library_store.js";
import { mk as _mk, toast as _toast, ceViewUrl as _ceViewUrl, isVideoFile as _isVideoFile } from "./library_common.js";

/**
 * @param {object|null} existing  Background dict to edit, or null/undefined for a new one.
 * @param {{file:string, folder?:string, frameTime?:number, name?:string}} [seed]  Only used
 *   when existing is absent — pre-selects a file/frame (e.g. a Source Profile clip's
 *   representative timestamp) and name so the modal opens one click from Analyze instead of
 *   requiring the user to navigate the file browser themselves.
 */
export function openBackgroundEditor(existing, seed) {
    const isNew = !existing;

    let refImages = (existing?.reference_images || []).map(r =>
        typeof r === "string"
            ? { file: r, role: "scene reference", folder: "input" }
            : { folder: "input", ...r }
    );

    // ── Fields ─────────────────────────────────────────────────────────────────
    const nameEl  = _mk("input",    { cls: "fbt-ce-input", type: "text", placeholder: "Name*" });
    const descEl  = _mk("textarea", { cls: "fbt-ce-textarea fbt-ce-outfit-desc",
        placeholder: "Environment description…" });
    const lightEl = _mk("input",    { cls: "fbt-ce-input", type: "text", placeholder: "Lighting conditions…" });
    const sndEl   = _mk("input",    { cls: "fbt-ce-input", type: "text", placeholder: "Ambient soundscape…" });

    if (!isNew) {
        nameEl.value  = existing.name        || "";
        descEl.value  = existing.description  || "";
        lightEl.value = existing.lighting     || "";
        sndEl.value   = existing.soundscape   || "";
    } else if (seed?.name) {
        nameEl.value = seed.name;
    }

    // ── Reference images ───────────────────────────────────────────────────────
    const refListEl = _mk("div", { cls: "fbt-ce-outfit-ref-list" });

    function _renderRefList() {
        refListEl.innerHTML = "";
        if (!refImages.length) {
            refListEl.appendChild(_mk("div", { cls: "fbt-ce-empty", textContent: "No reference images." }));
            return;
        }
        refImages.forEach((img, i) => {
            const row    = _mk("div", { cls: "fbt-ce-outfit-ref-row" });
            const thumb  = _mk("img", { cls: "fbt-ce-outfit-ref-thumb fbt-ce-clickable" });
            thumb.src    = _ceViewUrl(img.file, img.folder || "input");
            thumb.title  = `${img.file}\nClick to load into browser`;
            thumb.onclick = () => _applySelection(img.file, img.folder || "input");
            const info   = _mk("span", { cls: "fbt-ce-outfit-ref-info", textContent: img.file });
            const roleEl = _mk("select", { cls: "fbt-ce-select fbt-ce-outfit-ref-role" });
            ["scene reference", "lighting reference", "mood reference", "location reference", "color palette"].forEach(r => {
                const opt = _mk("option", { value: r, textContent: r });
                if (r === img.role) opt.selected = true;
                roleEl.appendChild(opt);
            });
            roleEl.onchange = () => { refImages[i].role = roleEl.value; };
            const delBtn = _mk("button", { cls: "fbt-ce-btn fbt-ce-btn-sm fbt-ce-btn-danger",
                textContent: "✕", onclick: () => { refImages.splice(i, 1); _renderRefList(); } });
            row.append(thumb, info, roleEl, delBtn);
            refListEl.appendChild(row);
        });
    }

    function _addFileToRefs(filePath, folder = "input") {
        if (!filePath) return;
        if (!refImages.some(r => r.file === filePath)) {
            refImages.push({ file: filePath, role: "scene reference", folder });
            _renderRefList();
        }
    }

    // ── File browser ───────────────────────────────────────────────────────────
    let selFile   = null;
    let frameTime = 1.0;

    const browserSection = _mk("div", { cls: "fbt-ce-outfit-browser" });
    browserSection.appendChild(_mk("div", { cls: "fbt-ce-label", textContent: "File Browser" }));

    const previewWrap = _mk("div", { cls: "fbt-ce-outfit-preview-wrap" });
    const previewImg  = _mk("img",   { cls: "fbt-be-img-preview fbt-ce-outfit-preview", alt: "" });
    const previewVid  = _mk("video", { cls: "fbt-ce-outfit-video-preview" });
    previewVid.controls = true;
    previewVid.preload  = "metadata";
    previewImg.style.display = "none";
    previewVid.style.display = "none";
    previewWrap.append(previewImg, previewVid);

    const frameRow   = _mk("div", { cls: "fbt-ce-outfit-frame-row" });
    const frameLabel = _mk("label", { cls: "fbt-ce-outfit-frame-label", textContent: "Frame (s):" });
    const frameInput = _mk("input", { type: "number", cls: "fbt-ce-input fbt-ce-outfit-frame-input",
        min: "0", step: "0.1", value: String(frameTime) });
    const syncBtn    = _mk("button", { cls: "fbt-ce-btn fbt-ce-btn-sm fbt-ce-btn-secondary",
        title: "Capture current video position", textContent: "↺ Use current",
        onclick: () => {
            frameTime = Math.max(0, previewVid.currentTime);
            frameInput.value = frameTime.toFixed(2);
        } });
    frameInput.addEventListener("change", () => {
        frameTime = Math.max(0, parseFloat(frameInput.value) || 0);
        frameInput.value = frameTime.toFixed(2);
        if (previewVid.readyState >= 1) previewVid.currentTime = frameTime;
    });
    previewVid.addEventListener("loadedmetadata", () => { previewVid.currentTime = frameTime; });
    frameRow.append(frameLabel, frameInput, syncBtn);
    frameRow.style.display = "none";

    const selFileEl = _mk("div", { cls: "fbt-ce-outfit-sel-file", textContent: "— no file selected —" });

    const _applySelection = (path, folder) => {
        const isVideo = _isVideoFile(path);
        selFile = { path, folder, isVideo };
        selFileEl.textContent = path.split("/").pop();
        selFileEl.title = path;
        if (isVideo) {
            previewImg.style.display = "none";
            previewVid.style.display = "";
            previewVid.src = _ceViewUrl(path, folder);
            frameRow.style.display   = "";
        } else {
            previewVid.style.display = "none";
            previewImg.style.display = "";
            previewImg.src = _ceViewUrl(path, folder);
            frameRow.style.display   = "none";
        }
        addRefBtn.disabled = isVideo;
        addRefBtn.title    = isVideo ? "Videos cannot be added directly — use Analyze with LLM" : "";
        if (analyzeBtn) analyzeBtn.disabled = false;
    };

    // ── File tree ──────────────────────────────────────────────────────────────
    const _isImgOrVid = f => /\.(png|jpg|jpeg|webp|gif|bmp|tiff?|mp4|mov|avi|mkv|webm|m4v|wmv)$/i.test(f);
    const bgTree = buildFileTree({
        inputFiles:  [...(lib.mediaInImages || []), ...(lib.mediaInVideos || [])],
        outputFiles: [...(lib.mediaOutImages || []), ...(lib.mediaOutVideos || [])],
        filter:      _isImgOrVid,
        isSelected:  (p, d) => selFile?.path === p && selFile?.folder === d,
        onSelect:    (p, d) => _applySelection(p, d),
        emptyText:   "No images or videos in {dir}/",
        initialDir:  "input",
    });

    const addRefBtn = _mk("button", { cls: "fbt-ce-btn fbt-ce-btn-sm", textContent: "+ Add as Reference",
        disabled: true,
        onclick: () => { if (selFile && !selFile.isVideo) _addFileToRefs(selFile.path, selFile.folder); } });

    let analyzeBtn = null;
    let bgHistRefresh = null;
    if (lib.llmLoaded && lib.llmVision) {
        analyzeBtn = _mk("button", { cls: "fbt-ce-btn", textContent: "🔍 Analyze with LLM",
            disabled: true,
            onclick: async () => {
                if (!selFile) { alert("Select a file from the browser first."); return; }
                analyzeBtn.disabled    = true;
                analyzeBtn.textContent = "Analyzing…";
                try {
                    const bgParams = { filename: selFile.path, folder: selFile.folder || "input", frameTime };
                    const data = await compositionsApi.analyzeBackground(selFile.path, bgParams);
                    if (data.description) descEl.value  = data.description;
                    if (data.lighting)    lightEl.value = data.lighting;
                    if (data.soundscape)  sndEl.value   = data.soundscape;
                    _addFileToRefs(data.frame_file || selFile.path, selFile.folder || "input");
                    llmApi.historyAdd(makeEntry({
                        kind: "bg_analyze",
                        compositionName: lib.getCompositionName(),
                        modelId: lib.llmLoaded || "",
                        params: bgParams,
                        result: { description: data.description, lighting: data.lighting,
                                  soundscape: data.soundscape, frameFile: data.frame_file || null },
                    })).catch(() => {});
                    bgHistRefresh?.();
                } catch (e) { alert(`Analyze failed: ${e.message}`); }
                finally {
                    analyzeBtn.disabled    = false;
                    analyzeBtn.textContent = "🔍 Analyze with LLM";
                }
            } });
    }

    // Pre-select a file/frame for a brand-new background (e.g. opened from a Source Profile
    // clip's representative timestamp) — reuses the exact same path a manual file-browser pick
    // goes through, so preview, frame row visibility and button enabling all follow for free.
    if (isNew && seed?.file) {
        _applySelection(seed.file, seed.folder || "input");
        if (typeof seed.frameTime === "number") {
            frameTime = seed.frameTime;
            frameInput.value = frameTime.toFixed(2);
            if (previewVid.readyState >= 1) previewVid.currentTime = frameTime;
        }
    }

    const actionRow = _mk("div", { cls: "fbt-ce-outfit-action-row" });
    actionRow.appendChild(addRefBtn);
    if (analyzeBtn) actionRow.appendChild(analyzeBtn);

    browserSection.append(bgTree.el, selFileEl, previewWrap, frameRow, actionRow);

    _renderRefList();

    // ── Buttons ────────────────────────────────────────────────────────────────
    const saveBtn = _mk("button", { cls: "fbt-ce-btn fbt-ce-btn-primary",
        textContent: isNew ? "Add" : "Save",
        onclick: async () => {
            const name = nameEl.value.trim();
            if (!name) { nameEl.focus(); alert("Name is required."); return; }
            const bg = {
                name,
                description:      descEl.value.trim(),
                lighting:         lightEl.value.trim(),
                soundscape:       sndEl.value.trim(),
                reference_images: refImages,
            };
            if (!isNew) bg.id = existing.id;
            try {
                await compositionsApi.saveBackground(bg);
                lib.backgrounds = (await compositionsApi.listBackgrounds()).backgrounds ?? [];
                notifyLibraryChanged("backgrounds");
                _toast(`Background "${name}" ${isNew ? "added" : "updated"}`, "success");
                overlay.remove();
            } catch (e) { alert(`Save failed: ${e.message}`); }
        }});

    const cancelBtn = _mk("button", { cls: "fbt-ce-btn", textContent: "Cancel",
        onclick: () => overlay.remove() });

    const footerBtns = [cancelBtn, saveBtn];
    if (!isNew) {
        const delBtn = _mk("button", { cls: "fbt-ce-btn fbt-ce-btn-danger", textContent: "Delete",
            onclick: async () => {
                if (!confirm(`Delete background "${existing.name}"?`)) return;
                try {
                    await compositionsApi.deleteBackground(existing.id);
                    lib.backgrounds = (await compositionsApi.listBackgrounds()).backgrounds ?? [];
                    notifyLibraryChanged("backgrounds", { deletedId: existing.id });
                    _toast(`Deleted "${existing.name}"`, "success");
                    overlay.remove();
                } catch (e) { alert(`Delete failed: ${e.message}`); }
            }});
        footerBtns.unshift(delBtn);
    }

    // ── Modal assembly ─────────────────────────────────────────────────────────
    const overlay = _mk("div", { cls: "fbt-ce-modal-overlay",
        onclick: e => { if (e.target === overlay) overlay.remove(); } });
    const modal   = _mk("div", { cls: "fbt-ce-modal" });
    overlay.appendChild(modal);

    modal.appendChild(_mk("div", { cls: "fbt-ce-modal-title",
        textContent: isNew ? "New Background" : `Edit: ${existing.name}` }));

    const _row = (label, el) => _mk("div", { cls: "fbt-ce-row" }, [
        _mk("label", { cls: "fbt-ce-label", textContent: label }),
        _mk("div",   { cls: "fbt-ce-input-wrap" }, [el]),
    ]);
    modal.appendChild(_row("Name",        nameEl));
    modal.appendChild(_row("Description", descEl));
    modal.appendChild(_row("Lighting",    lightEl));
    modal.appendChild(_row("Soundscape",  sndEl));
    modal.appendChild(browserSection);
    modal.appendChild(_mk("div", { cls: "fbt-ce-row" }, [
        _mk("label", { cls: "fbt-ce-label", textContent: "Reference Images" }),
    ]));
    modal.appendChild(refListEl);

    if (lib.llmLoaded && lib.llmVision) {
        const { el: histEl, refresh } = buildHistorySection({
            kind: "bg_analyze",
            onRestore: entry => {
                const r = entry.result || {};
                if (r.description) descEl.value  = r.description;
                if (r.lighting)    lightEl.value = r.lighting;
                if (r.soundscape)  sndEl.value   = r.soundscape;
            },
        });
        bgHistRefresh = refresh;
        modal.appendChild(histEl);
    }

    modal.appendChild(_mk("div", { cls: "fbt-ce-modal-btns" }, footerBtns));

    document.body.appendChild(overlay);
    nameEl.focus();
}

