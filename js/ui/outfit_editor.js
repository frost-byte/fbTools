/**
 * Outfit editor modal — used by the Assets tab and the Compose editor.
 * Data comes from / goes to the shared library store; callers react to the
 * `fbt:library-changed` event instead of passing callbacks.
 */

import { compositionsApi } from "../api/compositions.js";
import { llmApi } from "../api/llm.js";
import { buildFileTree } from "./file_tree.js";
import { makeEntry, buildHistorySection } from "../utils/llm_history.js";
import { lib, notifyLibraryChanged } from "./library_store.js";
import { mk as _mk, toast as _toast, ceViewUrl as _ceViewUrl, isVideoFile as _isVideoFile, DEFAULT_OUTFIT_QUERY } from "./library_common.js";

export function openOutfitEditor(existingId) {
    const isNew = !existingId;
    const entry = existingId ? (lib.outfits[existingId] || {}) : {};

    let refImages = (entry.reference_images || []).map(r =>
        typeof r === "string"
            ? { file: r, role: "costume detail", folder: "input" }
            : { folder: "input", ...r }
    );

    const overlay = _mk("div", { cls: "fbt-ce-modal-overlay",
        onclick: e => { if (e.target === overlay) overlay.remove(); } });
    const modal = _mk("div", { cls: "fbt-ce-modal" });
    overlay.appendChild(modal);

    modal.appendChild(_mk("div", { cls: "fbt-ce-modal-title",
        textContent: isNew ? "New Outfit" : `Edit Outfit: ${existingId}` }));

    const idInput   = _mk("input", { cls: "fbt-ce-input", placeholder: "outfit_id (snake_case)",
        value: existingId || "", disabled: !isNew });
    const nameInput = _mk("input", { cls: "fbt-ce-input", placeholder: "Display name",
        value: entry.name || "" });
    const tagsInput = _mk("input", { cls: "fbt-ce-input", placeholder: "Tags (comma-separated, optional)",
        value: (entry.tags || []).join(", ") });
    const descArea  = _mk("textarea", { cls: "fbt-ce-textarea fbt-ce-outfit-desc",
        placeholder: "Outfit description…", value: entry.description || "" });

    // ── Shared browser state ────────────────────────────────────────────────────
    let selFile   = null;   // {path, folder, isVideo}
    let frameTime = 1.0;
    let _onSelectionForSam2 = null;  // set by SAM2 section once built

    function _addFileToRefs(filePath, folder = "input") {
        if (!filePath) return;
        if (!refImages.some(r => r.file === filePath)) {
            refImages.push({ file: filePath, role: "costume detail", folder, use_as_reference: false });
            _renderRefList();
        }
    }

    // ── File Browser section ────────────────────────────────────────────────────
    const browserSection = _mk("div", { cls: "fbt-ce-outfit-browser" });
    browserSection.appendChild(_mk("div", { cls: "fbt-ce-label", textContent: "File Browser" }));

    // preview
    const previewWrap = _mk("div", { cls: "fbt-ce-outfit-preview-wrap" });
    const previewImg  = _mk("img",   { cls: "fbt-be-img-preview fbt-ce-outfit-preview", alt: "" });
    const previewVid  = _mk("video", { cls: "fbt-ce-outfit-video-preview" });
    previewVid.controls = true;
    previewVid.preload  = "metadata";
    previewImg.style.display = "none";
    previewVid.style.display = "none";
    previewWrap.append(previewImg, previewVid);

    // frame-time row (video only)
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

    // selected-file label
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
        if (_onSelectionForSam2) _onSelectionForSam2(selFile);
    };

    // ── File tree ──────────────────────────────────────────────────────────────
    const _isImgOrVid = f => /\.(png|jpg|jpeg|webp|gif|bmp|tiff?|mp4|mov|avi|mkv|webm|m4v|wmv)$/i.test(f);
    const outfitTree = buildFileTree({
        inputFiles:  [...(lib.mediaInImages || []), ...(lib.mediaInVideos || [])],
        outputFiles: [...(lib.mediaOutImages || []), ...(lib.mediaOutVideos || [])],
        filter:      _isImgOrVid,
        isSelected:  (p, d) => selFile?.path === p && selFile?.folder === d,
        onSelect:    (p, d) => _applySelection(p, d),
        emptyText:   "No images or videos in {dir}/",
        initialDir:  "input",
    });

    // action buttons
    const addRefBtn = _mk("button", { cls: "fbt-ce-btn", textContent: "+ Add as Reference",
        disabled: true,
        onclick: () => { if (selFile && !selFile.isVideo) _addFileToRefs(selFile.path, selFile.folder); } });

    let analyzeBtn   = null;
    let queryArea    = null;
    let outfitHistRefresh = null;
    if (lib.llmVision) {
        queryArea  = _mk("textarea", { cls: "fbt-ce-textarea fbt-ce-outfit-query",
            placeholder: "Describe what you want from the analysis…",
            value: DEFAULT_OUTFIT_QUERY });
        analyzeBtn = _mk("button", { cls: "fbt-ce-btn", textContent: "🔍 Analyze with LLM",
            disabled: true,
            onclick: async () => {
                if (!selFile) { alert("Select a file from the browser first."); return; }
                analyzeBtn.disabled    = true;
                analyzeBtn.textContent = "Analyzing…";
                try {
                    const query = queryArea.value.trim() || DEFAULT_OUTFIT_QUERY;
                    const reqBody = { filename: selFile.path, query, max_tokens: 400 };
                    if (selFile.isVideo) reqBody.frame_time = frameTime;
                    const res  = await fetch("/fbtools/outfits/analyze_media", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(reqBody),
                    });
                    const data = await res.json();
                    if (!res.ok) throw new Error(data.error || res.statusText);
                    if (data.description) descArea.value = data.description;
                    _addFileToRefs(data.frame_file || selFile.path, selFile.folder || "input");
                    llmApi.historyAdd(makeEntry({
                        kind: "outfit_analyze",
                        compositionName: lib.getCompositionName(),
                        modelId: lib.llmLoaded || "",
                        params: { filename: selFile.path, folder: selFile.folder || "input",
                                  frameTime: selFile.isVideo ? frameTime : null, query },
                        result: { description: data.description, frameFile: data.frame_file || null },
                    })).catch(() => {});
                    outfitHistRefresh?.();
                } catch (e) { alert(`Analyze failed: ${e.message}`); }
                finally {
                    analyzeBtn.disabled    = false;
                    analyzeBtn.textContent = "🔍 Analyze with LLM";
                }
            } });
    }

    const actionRow = _mk("div", { cls: "fbt-ce-outfit-action-row" });
    actionRow.appendChild(addRefBtn);
    if (analyzeBtn) actionRow.appendChild(analyzeBtn);

    browserSection.append(outfitTree.el, selFileEl, previewWrap, frameRow, actionRow);
    if (queryArea) browserSection.appendChild(queryArea);

    // ── SAM2 section (source driven by browser selection above) ────────────────
    const sam2Section = _mk("div", { cls: "fbt-ce-outfit-sam2" });
    fetch("/fbtools/outfits/sam2_status").then(r => r.json()).then(fresh => {
        lib.sam2 = fresh;
    }).catch(() => {}).finally(() => { sam2Section.innerHTML = ""; _buildSam2Section(); });

    function _buildSam2Section() {
        const s = lib.sam2;
        sam2Section.appendChild(_mk("div", { cls: "fbt-ce-label", textContent: "SAM2 Outfit Extraction" }));
        if (!s) return;

        if (!s.packages_ok) {
            sam2Section.appendChild(_mk("div", { cls: "fbt-ce-hint fbt-ce-hint-warn",
                textContent: "SAM2 packages not installed." }));
            sam2Section.appendChild(_mk("div", { cls: "fbt-ce-hint",
                textContent: `To enable: ${s.install_hint}` }));
            return;
        }
        if (!s.available) {
            sam2Section.appendChild(_mk("div", { cls: "fbt-ce-hint",
                textContent: "SAM2 model not found." }));
            sam2Section.appendChild(_mk("div", { cls: "fbt-ce-hint fbt-ce-hint-warn",
                textContent: s.model_hint || "Download a SAM2 safetensors model and place in models/sams/" }));
            return;
        }

        // Full extraction UI — image source comes from shared browser selection
        let extractPoint   = { x: 0.5, y: 0.5 };
        let lastResultFile = null;

        const noSelHint = _mk("div", { cls: "fbt-ce-hint",
            textContent: "Select an image from the browser above to begin extraction." });

        const sam2PreviewWrap = _mk("div", { cls: "fbt-ce-sam2-preview-wrap" });
        const sam2PreviewImg  = _mk("img",  { cls: "fbt-ce-sam2-preview-img", alt: "" });
        const pointDot        = _mk("div",  { cls: "fbt-ce-sam2-point-dot" });
        sam2PreviewWrap.append(sam2PreviewImg, pointDot);
        sam2PreviewImg.style.display = "none";
        pointDot.style.display       = "none";

        const pointLabel = _mk("div", { cls: "fbt-ce-hint fbt-ce-sam2-coords",
            textContent: "Point: (0.50, 0.50)" });

        function _updateDot() {
            pointDot.style.left    = `${extractPoint.x * 100}%`;
            pointDot.style.top     = `${extractPoint.y * 100}%`;
            pointLabel.textContent = `Point: (${extractPoint.x.toFixed(2)}, ${extractPoint.y.toFixed(2)})`;
        }
        _updateDot();

        sam2PreviewWrap.addEventListener("click", e => {
            if (!sam2PreviewImg.naturalWidth) return;
            const rect = sam2PreviewWrap.getBoundingClientRect();
            extractPoint = {
                x: Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width)),
                y: Math.max(0, Math.min(1, (e.clientY - rect.top)  / rect.height)),
            };
            _updateDot();
        });

        const resultWrap   = _mk("div", { cls: "fbt-ce-sam2-result-wrap" });
        const resultImg    = _mk("img", { cls: "fbt-ce-sam2-result-img", alt: "Extracted outfit" });
        resultWrap.style.display = "none";
        resultWrap.appendChild(resultImg);

        const addResultBtn = _mk("button", { cls: "fbt-ce-btn fbt-ce-btn-sm",
            textContent: "Add to References",
            onclick: () => { if (lastResultFile) _addFileToRefs(lastResultFile, "input"); } });
        addResultBtn.style.display = "none";

        const extractBtn = _mk("button", { cls: "fbt-ce-btn", textContent: "✂ Extract Outfit",
            disabled: true,
            onclick: async () => {
                if (!selFile || selFile.isVideo) { alert("Select an image first."); return; }
                extractBtn.disabled    = true;
                extractBtn.textContent = "Extracting…";
                resultWrap.style.display   = "none";
                addResultBtn.style.display = "none";
                try {
                    const res = await fetch("/fbtools/outfits/extract_outfit", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            filename:    selFile.path,
                            point_x:     extractPoint.x,
                            point_y:     extractPoint.y,
                            point_label: 1,
                        }),
                    });
                    const data = await res.json();
                    if (!res.ok) throw new Error(data.error || res.statusText);
                    lastResultFile       = data.result_file;
                    resultImg.src        = `/view?filename=${encodeURIComponent(lastResultFile)}&type=input`;
                    resultWrap.style.display   = "";
                    addResultBtn.style.display = "";
                } catch (e) { alert(`Extraction failed: ${e.message}`); }
                finally {
                    extractBtn.disabled    = false;
                    extractBtn.textContent = "✂ Extract Outfit";
                }
            } });

        // Wire browser selection → SAM2 preview
        _onSelectionForSam2 = (sf) => {
            if (sf && !sf.isVideo) {
                noSelHint.style.display      = "none";
                sam2PreviewImg.src           = _ceViewUrl(sf.path, sf.folder);
                sam2PreviewImg.style.display = "";
                pointDot.style.display       = "";
                extractBtn.disabled          = false;
                resultWrap.style.display     = "none";
                addResultBtn.style.display   = "none";
                lastResultFile               = null;
                extractPoint                 = { x: 0.5, y: 0.5 };
                _updateDot();
            } else {
                noSelHint.style.display      = "";
                sam2PreviewImg.src           = "";
                sam2PreviewImg.style.display = "none";
                pointDot.style.display       = "none";
                extractBtn.disabled          = true;
            }
        };
        if (selFile) _onSelectionForSam2(selFile);

        sam2Section.append(noSelHint, sam2PreviewWrap, pointLabel, extractBtn, resultWrap, addResultBtn);
    }

    // ── Reference images list ──────────────────────────────────────────────────
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
            thumb.title  = `${img.file}\nClick to load into browser & SAM2`;
            thumb.onclick = () => _applySelection(img.file, img.folder || "input");
            const info   = _mk("span", { cls: "fbt-ce-outfit-ref-info", textContent: img.file });
            const roleEl = _mk("select", { cls: "fbt-ce-select fbt-ce-outfit-ref-role" });
            ["costume detail", "character sheet", "full body", "portrait", "side profile", "reference"].forEach(r => {
                const opt = _mk("option", { value: r, textContent: r });
                if (r === img.role) opt.selected = true;
                roleEl.appendChild(opt);
            });
            roleEl.onchange = () => { refImages[i].role = roleEl.value; };
            // "Use as <Subject N> reference" toggle
            const refCb = _mk("input", { type: "checkbox" });
            refCb.checked = !!img.use_as_reference;
            refCb.onchange = () => { refImages[i].use_as_reference = refCb.checked; };
            const refCbWrap = _mk("label", { cls: "fbt-ce-outfit-ref-use-label",
                title: "Include as <Subject N> visual reference in assembled prompt" });
            refCbWrap.appendChild(refCb);
            refCbWrap.appendChild(document.createTextNode(" Ref"));
            const delBtn = _mk("button", { cls: "fbt-ce-btn fbt-ce-btn-sm fbt-ce-btn-danger",
                textContent: "✕",
                onclick: () => { refImages.splice(i, 1); _renderRefList(); }
            });
            row.append(thumb, info, roleEl, refCbWrap, delBtn);
            refListEl.appendChild(row);
        });
    }
    _renderRefList();

    // ── Buttons ────────────────────────────────────────────────────────────────
    const saveBtn = _mk("button", { cls: "fbt-ce-btn fbt-ce-btn-primary", textContent: "Save",
        onclick: async () => {
            const id = isNew ? idInput.value.trim() : existingId;
            if (!id) { alert("Outfit ID is required."); return; }
            if (!/^[a-z0-9_]+$/.test(id)) { alert("ID must be lowercase letters, digits, or underscores."); return; }
            const outfit = {
                id,
                name:             nameInput.value.trim(),
                description:      descArea.value.trim(),
                tags:             tagsInput.value.split(",").map(t => t.trim()).filter(Boolean),
                reference_images: refImages,
            };
            try {
                await compositionsApi.saveOutfit(outfit);
                lib.outfits[id] = {
                    name: outfit.name, description: outfit.description,
                    tags: outfit.tags, reference_images: refImages,
                };
                notifyLibraryChanged("outfits");
                overlay.remove();
            } catch (e) { alert(`Save failed: ${e.message}`); }
        }});
    const cancelBtn = _mk("button", { cls: "fbt-ce-btn", textContent: "Cancel",
        onclick: () => overlay.remove() });

    // ── Modal assembly ─────────────────────────────────────────────────────────
    const _row = (label, el) => _mk("div", { cls: "fbt-ce-row" }, [
        _mk("label", { cls: "fbt-ce-label", textContent: label }),
        _mk("div",   { cls: "fbt-ce-input-wrap" }, [el]),
    ]);
    modal.appendChild(_row("ID",          idInput));
    modal.appendChild(_row("Name",        nameInput));
    modal.appendChild(_row("Tags",        tagsInput));
    modal.appendChild(_row("Description", descArea));
    modal.appendChild(browserSection);
    modal.appendChild(sam2Section);
    modal.appendChild(_mk("div", { cls: "fbt-ce-row" }, [
        _mk("label", { cls: "fbt-ce-label", textContent: "Reference Images" }),
    ]));
    modal.appendChild(refListEl);

    if (lib.llmVision) {
        const { el: histEl, refresh } = buildHistorySection({
            kind: "outfit_analyze",
            onRestore: entry => {
                const r = entry.result || {};
                if (r.description) descArea.value = r.description;
                if (entry.params?.query && queryArea) queryArea.value = entry.params.query;
            },
        });
        outfitHistRefresh = refresh;
        modal.appendChild(histEl);
    }

    modal.appendChild(_mk("div", { cls: "fbt-ce-modal-btns" }, [cancelBtn, saveBtn]));

    document.body.appendChild(overlay);
    (isNew ? idInput : nameInput).focus();
}

