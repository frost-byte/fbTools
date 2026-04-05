/**
 * Dataset caption viewer node extension.
 * Renders an editable table UI for DatasetCaptionViewer nodes.
 */

import { datasetCaptionAPI } from "../api/dataset_caption.js";

const STYLES = `
.dcv-wrap {
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 0;
  font-family: 'DM Mono', 'Consolas', monospace;
  font-size: 11px;
  color: #c8d4e8;
  background: #0a0c12;
  border: 1px solid #1e2840;
  border-radius: 10px;
  overflow: hidden;
  min-width: 540px;
}
.dcv-content-scroll {
  flex: 1 1 auto;
  min-height: 0;
  overflow: auto;
  overscroll-behavior: contain;
}
.dcv-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 12px;
  background: #0f1320;
  border-bottom: 1px solid #1e2840;
  gap: 8px;
}
.dcv-header-title {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: #4f8ef7;
}
.dcv-stats {
  font-size: 10px;
  color: #3d5070;
}
.dcv-table {
  width: 100%;
  border-collapse: collapse;
}
.dcv-table th {
  position: sticky;
  top: 0;
  z-index: 2;
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #3d5070;
  padding: 6px 10px;
  text-align: left;
  border-bottom: 1px solid #141a28;
  background: #0d1020;
}
.dcv-row {
  border-bottom: 1px solid #111724;
  transition: background 0.1s;
}
.dcv-row:hover { background: #0e1420; }
.dcv-row:last-child { border-bottom: none; }
.dcv-thumb-cell {
  width: 72px;
  padding: 6px 8px;
  vertical-align: middle;
}
.dcv-thumb {
  width: 60px;
  height: 60px;
  object-fit: cover;
  border-radius: 5px;
  display: block;
  border: 1px solid #1e2840;
  background: #0d1020;
}
.dcv-name-cell {
  width: 130px;
  padding: 6px 8px;
  vertical-align: top;
  word-break: break-all;
  color: #7a8eaa;
  font-size: 10px;
}
.dcv-status-dot {
  display: inline-block;
  width: 6px;
  height: 6px;
  border-radius: 50%;
  margin-right: 5px;
  vertical-align: middle;
}
.dcv-dot-ok { background: #52cca0; }
.dcv-dot-missing { background: #e85c5c; }
.dcv-caption-cell {
  padding: 6px 8px;
  vertical-align: top;
}
.dcv-caption-textarea {
  width: 100%;
  min-height: 58px;
  background: #0d1420;
  border: 1px solid #1a2438;
  border-radius: 6px;
  color: #b8c8e0;
  font-size: 11px;
  font-family: inherit;
  padding: 6px 8px;
  resize: vertical;
  outline: none;
  transition: border-color 0.15s;
  box-sizing: border-box;
}
.dcv-caption-textarea:focus {
  border-color: rgba(79,142,247,0.4);
  background: #0f1828;
}
.dcv-caption-textarea.dirty { border-color: #f5a623; }
.dcv-caption-textarea.saving { border-color: #f5a623; }
.dcv-caption-textarea.saved  { border-color: #52cca0; }
.dcv-actions-cell {
  width: 96px;
  padding: 6px 8px;
  vertical-align: top;
}
.dcv-btn {
  display: block;
  width: 100%;
  padding: 4px 0;
  margin-bottom: 4px;
  border-radius: 5px;
  border: 1px solid #1e2840;
  background: #141a28;
  color: #7a8eaa;
  font-size: 9px;
  font-family: inherit;
  font-weight: 600;
  letter-spacing: 0.05em;
  cursor: pointer;
  text-align: center;
  transition: all 0.12s;
}
.dcv-btn:hover { background: #1a2438; color: #c8d4e8; border-color: #2e3a50; }
.dcv-btn.recaption:hover { color: #4f8ef7; border-color: rgba(79,142,247,0.3); }
.dcv-btn.recaption.loading { color: #f5a623; border-color: rgba(245,166,35,0.3); cursor: wait; }
.dcv-pagination {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 8px 12px;
  background: #0d1020;
  border-top: 1px solid #1e2840;
}
.dcv-page-btn {
  padding: 3px 9px;
  border-radius: 5px;
  border: 1px solid #1e2840;
  background: #141a28;
  color: #7a8eaa;
  font-size: 10px;
  font-family: inherit;
  cursor: pointer;
  transition: all 0.12s;
}
.dcv-page-btn:hover:not(:disabled) { background: #1a2438; color: #c8d4e8; }
.dcv-page-btn:disabled { opacity: 0.3; cursor: default; }
.dcv-page-info {
  font-size: 10px;
  color: #3d5070;
  min-width: 80px;
  text-align: center;
}
.dcv-empty {
  padding: 24px;
  text-align: center;
  color: #3d5070;
  font-size: 12px;
}
.dcv-load-btn {
  display: block;
  margin: 10px auto;
  padding: 7px 20px;
  border-radius: 7px;
  border: 1px solid rgba(79,142,247,0.3);
  background: rgba(79,142,247,0.1);
  color: #4f8ef7;
  font-size: 11px;
  font-family: inherit;
  font-weight: 600;
  cursor: pointer;
  letter-spacing: 0.04em;
  transition: all 0.15s;
}
.dcv-load-btn:hover { background: rgba(79,142,247,0.2); }
`;

function injectStyles() {
  if (document.getElementById("dcv-styles")) return;
  const styleTag = document.createElement("style");
  styleTag.id = "dcv-styles";
  styleTag.textContent = STYLES;
  document.head.appendChild(styleTag);
}

function safeWidgetValue(node, name, fallback = "") {
  const widget = node.widgets?.find((w) => w.name === name);
  return widget?.value ?? fallback;
}

function createViewerUI(node) {
  injectStyles();

  const wrap = document.createElement("div");
  wrap.className = "dcv-wrap";

  const state = {
    rows: [],
    total: 0,
    page: 1,
    page_size: 10,
    total_pages: 1,
    base_dir: "",
    output_dir: "",
    recursive: false,
    captioner_type: "qwen_vl",
    instruction: "",
    trigger_word: "",
    device: "auto",
    use_8bit: false,
    clean_caption: true,
    gemini_api_key: "",
  };

  function syncStateFromWidgets() {
    state.base_dir = String(safeWidgetValue(node, "dataset_path", state.base_dir || "")).trim();
    state.output_dir = String(safeWidgetValue(node, "output_directory", state.output_dir || "")).trim();
    state.page = Number(safeWidgetValue(node, "page", state.page || 1)) || 1;
    state.page_size = Number(safeWidgetValue(node, "page_size", state.page_size || 10)) || 10;
    state.recursive = Boolean(safeWidgetValue(node, "recursive", state.recursive));

    state.captioner_type = String(safeWidgetValue(node, "captioner_type", state.captioner_type));
    state.instruction = String(safeWidgetValue(node, "instruction", state.instruction));
    state.trigger_word = String(safeWidgetValue(node, "trigger_word", state.trigger_word));
    state.device = String(safeWidgetValue(node, "device", state.device));
    state.use_8bit = Boolean(safeWidgetValue(node, "use_8bit", state.use_8bit));
    state.clean_caption = Boolean(safeWidgetValue(node, "clean_caption", state.clean_caption));
    state.gemini_api_key = String(safeWidgetValue(node, "gemini_api_key", state.gemini_api_key));
  }

  function applyViewerData(viewerData) {
    if (!viewerData || typeof viewerData !== "object") return;
    Object.assign(state, viewerData);
    syncStateFromWidgets();
    render();
  }

  async function saveCaption(row, caption, textarea) {
    try {
      textarea.classList.add("saving");
      const data = await datasetCaptionAPI.saveCaption(row.txt_path, caption);
      if (!data?.ok) {
        throw new Error(data?.error || "Caption save failed");
      }
      row.caption = caption;
      row.has_caption = caption.trim().length > 0;
      textarea.classList.remove("saving");
      textarea.classList.remove("dirty");
      textarea.classList.add("saved");
      setTimeout(() => textarea.classList.remove("saved"), 1200);
      return true;
    } catch (error) {
      console.error("fb_tools -> DatasetCaptionViewer: save error", error);
      textarea.classList.remove("saving");
      alert(`Save failed: ${error.message || error}`);
      return false;
    }
  }

  async function fetchPage(page) {
    syncStateFromWidgets();
    if (!state.base_dir) {
      render();
      return;
    }

    try {
      const params = {
        path: state.base_dir,
        page,
        page_size: state.page_size,
        recursive: Boolean(state.recursive),
      };
      if (state.output_dir) {
        params.output_dir = state.output_dir;
      }

      const data = await datasetCaptionAPI.listDataset(params);
      if (data?.error) {
        throw new Error(data.error);
      }

      Object.assign(state, data);
      render();
    } catch (error) {
      console.error("fb_tools -> DatasetCaptionViewer: list fetch error", error);
      alert(`Load failed: ${error.message || error}`);
    }
  }

  function render() {
    wrap.innerHTML = "";

    const header = document.createElement("div");
    header.className = "dcv-header";

    const title = document.createElement("span");
    title.className = "dcv-header-title";
    title.textContent = "Dataset Caption Viewer";

    const stats = document.createElement("span");
    stats.className = "dcv-stats";
    const captionedOnPage = state.rows.filter((row) => row.has_caption).length;
    stats.textContent = state.total > 0
      ? `${state.total} images - ${captionedOnPage}/${state.rows.length} on page captioned`
      : "No data loaded";

    header.appendChild(title);
    header.appendChild(stats);
    wrap.appendChild(header);

    const contentScroll = document.createElement("div");
    contentScroll.className = "dcv-content-scroll";
    wrap.appendChild(contentScroll);

    if (!state.rows.length) {
      const empty = document.createElement("div");
      empty.className = "dcv-empty";
      empty.textContent = state.base_dir
        ? "No images found in directory."
        : "Execute the node to load the dataset, or click Load below.";

      const loadBtn = document.createElement("button");
      loadBtn.className = "dcv-load-btn";
      loadBtn.textContent = "Load / Refresh";
      loadBtn.onclick = () => {
        const pageToLoad = state.page > 0 ? state.page : 1;
        fetchPage(pageToLoad);
      };

      contentScroll.appendChild(empty);
      contentScroll.appendChild(loadBtn);
      return;
    }

    const table = document.createElement("table");
    table.className = "dcv-table";

    const thead = document.createElement("thead");
    thead.innerHTML = `<tr>
      <th>Image</th>
      <th>File</th>
      <th>Caption</th>
      <th>Actions</th>
    </tr>`;
    table.appendChild(thead);

    const tbody = document.createElement("tbody");
    state.rows.forEach((row) => {
      const tr = document.createElement("tr");
      tr.className = "dcv-row";

      const thumbCell = document.createElement("td");
      thumbCell.className = "dcv-thumb-cell";
      const img = document.createElement("img");
      img.className = "dcv-thumb";
      img.src = datasetCaptionAPI.getImageUrl(row.image_path);
      img.onerror = () => {
        img.src = "";
        img.style.background = "#141a28";
      };
      thumbCell.appendChild(img);
      tr.appendChild(thumbCell);

      const nameCell = document.createElement("td");
      nameCell.className = "dcv-name-cell";
      const dot = document.createElement("span");
      dot.className = `dcv-status-dot ${row.has_caption ? "dcv-dot-ok" : "dcv-dot-missing"}`;
      nameCell.appendChild(dot);
      nameCell.appendChild(document.createTextNode(row.filename));
      tr.appendChild(nameCell);

      const captionCell = document.createElement("td");
      captionCell.className = "dcv-caption-cell";
      const textarea = document.createElement("textarea");
      textarea.className = "dcv-caption-textarea";
      textarea.value = row.caption || "";
      textarea.placeholder = "No caption yet";
      textarea.addEventListener("input", () => {
        textarea.classList.remove("saved");
        textarea.classList.add("dirty");
      });

      captionCell.appendChild(textarea);
      tr.appendChild(captionCell);

      const actionsCell = document.createElement("td");
      actionsCell.className = "dcv-actions-cell";

      const updateBtn = document.createElement("button");
      updateBtn.className = "dcv-btn";
      updateBtn.textContent = "Update";
      updateBtn.title = "Save edited caption text to disk";
      updateBtn.onclick = async () => {
        const ok = await saveCaption(row, textarea.value, textarea);
        if (!ok) return;
        dot.className = `dcv-status-dot ${row.has_caption ? "dcv-dot-ok" : "dcv-dot-missing"}`;
      };
      actionsCell.appendChild(updateBtn);

      const recaptionBtn = document.createElement("button");
      recaptionBtn.className = "dcv-btn recaption";
      recaptionBtn.textContent = "Re-caption";
      recaptionBtn.title = "Re-generate this image caption using current model settings";
      recaptionBtn.onclick = async () => {
        recaptionBtn.textContent = "Running...";
        recaptionBtn.classList.add("loading");
        recaptionBtn.disabled = true;

        try {
          syncStateFromWidgets();
          const data = await datasetCaptionAPI.recaption({
              image_path: row.image_path,
              txt_path: row.txt_path,
              captioner_type: state.captioner_type,
              instruction: state.instruction,
              trigger_word: state.trigger_word,
              device: state.device,
              use_8bit: state.use_8bit,
              clean_caption: state.clean_caption,
              gemini_api_key: state.gemini_api_key,
          });
          if (!data?.ok) {
            throw new Error(data?.error || "Re-caption failed");
          }

          textarea.value = data.caption || "";
          row.caption = data.caption || "";
          row.has_caption = true;
          dot.className = "dcv-status-dot dcv-dot-ok";
          textarea.classList.add("saved");
          setTimeout(() => textarea.classList.remove("saved"), 1200);
        } catch (error) {
          console.error("fb_tools -> DatasetCaptionViewer: recaption error", error);
          alert(`Re-caption failed: ${error.message || error}`);
        } finally {
          recaptionBtn.textContent = "Re-caption";
          recaptionBtn.classList.remove("loading");
          recaptionBtn.disabled = false;
        }
      };
      actionsCell.appendChild(recaptionBtn);

      const clearBtn = document.createElement("button");
      clearBtn.className = "dcv-btn";
      clearBtn.textContent = "Clear";
      clearBtn.title = "Delete text content for this image caption";
      clearBtn.onclick = async () => {
        if (!confirm(`Clear caption for ${row.filename}?`)) return;
        const ok = await saveCaption(row, "", textarea);
        if (!ok) return;
        textarea.value = "";
        row.has_caption = false;
        dot.className = "dcv-status-dot dcv-dot-missing";
      };
      actionsCell.appendChild(clearBtn);

      tr.appendChild(actionsCell);
      tbody.appendChild(tr);
    });

    table.appendChild(tbody);
    contentScroll.appendChild(table);

    if (state.total_pages > 1) {
      const pagination = document.createElement("div");
      pagination.className = "dcv-pagination";

      const prevBtn = document.createElement("button");
      prevBtn.className = "dcv-page-btn";
      prevBtn.textContent = "Prev";
      prevBtn.disabled = state.page <= 1;
      prevBtn.onclick = () => fetchPage(state.page - 1);

      const pageInfo = document.createElement("span");
      pageInfo.className = "dcv-page-info";
      pageInfo.textContent = `Page ${state.page} / ${state.total_pages}`;

      const nextBtn = document.createElement("button");
      nextBtn.className = "dcv-page-btn";
      nextBtn.textContent = "Next";
      nextBtn.disabled = state.page >= state.total_pages;
      nextBtn.onclick = () => fetchPage(state.page + 1);

      pagination.appendChild(prevBtn);
      pagination.appendChild(pageInfo);
      pagination.appendChild(nextBtn);
      wrap.appendChild(pagination);
    }
  }

  render();

  return {
    element: wrap,
    applyViewerData,
    syncStateFromWidgets,
  };
}

export function setupDatasetCaptionViewer(nodeType, nodeData, app) {
  console.log("fb_tools -> DatasetCaptionViewer node detected");

  const onNodeCreated = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function () {
    onNodeCreated?.apply(this, arguments);

    if (this._dcvInitialized) {
      return;
    }
    this._dcvInitialized = true;

    const ui = createViewerUI(this);

    const displayWidget = this.addDOMWidget("dataset_viewer_widget", "preview", ui.element, {
      serialize: false,
      hideOnZoom: false,
      getValue() { return ""; },
      setValue() {},
    });

    displayWidget.computeSize = (width) => {
      const w = Math.max(width || this.size[0], 560);
      return [w, 420];
    };

    ui.element.style.height = "420px";
    ui.element.style.maxHeight = "420px";

    this._dcvApplyViewerData = ui.applyViewerData;
    this._dcvSyncSettings = ui.syncStateFromWidgets;

    this.size[0] = Math.max(this.size[0], 580);
    this.size[1] = Math.max(this.size[1], 500);
    app.graph.setDirtyCanvas(true, false);
  };

  const onResize = nodeType.prototype.onResize;
  nodeType.prototype.onResize = function (size) {
    const result = onResize?.apply(this, arguments);
    return result;
  };

  const onExecuted = nodeType.prototype.onExecuted;
  nodeType.prototype.onExecuted = function (message) {
    const result = onExecuted?.apply(this, arguments);

    this._dcvSyncSettings?.();

    const viewerPayload =
      message?.ui?.dataset_viewer?.[0]
      || message?.dataset_viewer?.[0]
      || message?.output?.ui?.dataset_viewer?.[0];

    if (viewerPayload) {
      this._dcvApplyViewerData?.(viewerPayload);
    }

    return result;
  };
}
