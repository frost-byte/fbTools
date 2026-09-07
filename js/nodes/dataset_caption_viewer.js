/**
 * Dataset caption viewer node extension.
 * Renders an editable table UI for DatasetCaptionViewer nodes.
 */

import { datasetCaptionAPI } from "../api/dataset_caption.js";

function safeWidgetValue(node, name, fallback = "") {
  const widget = node.widgets?.find((w) => w.name === name);
  return widget?.value ?? fallback;
}

function createViewerUI(node) {
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
    captioner_type: "llm_client",
    instruction: "",
    trigger_word: "",
    clean_caption: true,
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
    state.clean_caption = Boolean(safeWidgetValue(node, "clean_caption", state.clean_caption));
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
              clean_caption: state.clean_caption,
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
