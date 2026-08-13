/**
 * Scene Cast Editor sidebar panel.
 *
 * Lets users create and manage Scene Casts — named assignments of
 * subject → bundle per role, with per-entry visual mode and audio toggles.
 * Used to quickly swap reference media across multiple compositions
 * without re-running the workflow.
 *
 * Registered as a ComfyUI sidebar tab via app.extensionManager.registerSidebarTab.
 */

import { bundlesApi } from "../api/bundles.js";

// ── Module state ───────────────────────────────────────────────────────────────

const _S = {
    casts:    [],  // [{id, name, entries, created, modified}]
    bundles:  [],  // [{id, name, subject_id, visual, audio, ...}]
    subjects: [],  // [{id, name, ...}]
    editing:  null, // cast dict being edited; null = list view
    isNew:    false,
};

const _dom = {};

// ── Helpers ────────────────────────────────────────────────────────────────────

function _mk(tag, props = {}, children = []) {
    const el = document.createElement(tag);
    Object.entries(props).forEach(([k, v]) => {
        if (k === "cls") el.className = v;
        else if (k === "style") Object.assign(el.style, v);
        else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
        else el[k] = v;
    });
    children.forEach(c => c && el.appendChild(c));
    return el;
}

function _toast(msg, severity = "info") {
    try {
        const app = window._fbtApp;
        if (app?.extensionManager?.toast) {
            app.extensionManager.toast.add({ severity, summary: msg, life: 2500 });
        }
    } catch (_) {}
}

function _subjectName(id) {
    const s = _S.subjects.find(s => s.id === id);
    return s ? (s.name || s.id) : (id || "Unknown");
}

function _bundlesForSubject(subjectId) {
    if (!subjectId) return _S.bundles;
    return _S.bundles.filter(b => b.subject_id === subjectId);
}

function _bundleLookup(id) {
    return _S.bundles.find(b => b.id === id) ?? null;
}

function _slugify(str) {
    return (str || "").toLowerCase().replace(/\s+/g, "_").replace(/[^\w]/g, "").slice(0, 48);
}

function _fmtDate(iso) {
    if (!iso) return "";
    try {
        return new Date(iso).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
    } catch (_) { return ""; }
}

// ── Data load ─────────────────────────────────────────────────────────────────

async function _loadAll() {
    const [casts, bundles, subjects] = await Promise.allSettled([
        bundlesApi.listCasts(),
        bundlesApi.listBundles(),
        bundlesApi.listSubjects(),
    ]);
    _S.casts    = casts.value?.casts       ?? [];
    _S.bundles  = bundles.value?.bundles   ?? [];
    _S.subjects = subjects.value?.subjects ?? [];
}

// ── List view ─────────────────────────────────────────────────────────────────

function _renderList() {
    const c = _dom.content;
    if (!c) return;
    c.innerHTML = "";

    if (!_S.casts.length) {
        c.appendChild(_mk("div", {
            cls: "fbt-be-empty",
            textContent: "No casts saved. Click + New Cast to create one.",
        }));
        return;
    }

    _S.casts.forEach(cast => c.appendChild(_buildCastCard(cast)));
}

function _buildCastCard(cast) {
    const card = _mk("div", { cls: "fbt-be-card fbt-cast-card" });

    // Top row: name + meta
    const top = _mk("div", { cls: "fbt-be-card-top" });
    top.appendChild(_mk("span", { cls: "fbt-be-card-name", textContent: cast.name || cast.id }));

    const meta = _mk("span", { cls: "fbt-cast-meta" });
    const n = cast.entries?.length ?? 0;
    meta.appendChild(_mk("span", { cls: "fbt-cast-count", textContent: `${n} ${n === 1 ? "subject" : "subjects"}` }));
    if (cast.modified) {
        meta.appendChild(_mk("span", { cls: "fbt-cast-date", textContent: _fmtDate(cast.modified) }));
    }
    top.appendChild(meta);
    card.appendChild(top);

    // Subject name chips for quick preview
    if (n) {
        const chips = _mk("div", { cls: "fbt-cast-chips" });
        cast.entries.forEach(e => {
            if (e.subject_id) {
                chips.appendChild(_mk("span", { cls: "fbt-be-tag", textContent: _subjectName(e.subject_id) }));
            }
        });
        card.appendChild(chips);
    }

    // Actions
    const actions = _mk("div", { cls: "fbt-be-card-actions" });
    actions.appendChild(_mk("button", {
        cls: "fbt-ce-icon-btn", title: "Edit", textContent: "✎",
        onclick: e => { e.stopPropagation(); _startEdit(cast); },
    }));
    actions.appendChild(_mk("button", {
        cls: "fbt-ce-icon-btn fbt-ce-danger", title: "Delete", textContent: "✕",
        onclick: e => { e.stopPropagation(); _onDelete(cast.id, cast.name); },
    }));
    card.appendChild(actions);

    card.style.cursor = "pointer";
    card.addEventListener("click", () => _startEdit(cast));
    return card;
}

async function _onDelete(id, name) {
    if (!confirm(`Delete cast "${name || id}"?`)) return;
    try {
        await bundlesApi.deleteCast(id);
        _S.casts = _S.casts.filter(c => c.id !== id);
        if (_S.editing?.id === id) { _S.editing = null; _S.isNew = false; }
        _renderList();
        bundlesApi.reloadCasts().catch(() => {});
        _toast(`Deleted "${name || id}"`, "success");
    } catch (e) {
        _toast("Delete failed: " + e.message, "error");
    }
}

// ── Editor form ───────────────────────────────────────────────────────────────

function _startEdit(cast) {
    _S.editing = structuredClone(cast);
    _S.isNew   = false;
    _renderForm();
}

function _startNew() {
    _S.editing = { id: "", name: "", entries: [] };
    _S.isNew   = true;
    _renderForm();
}

function _cancelEdit() {
    _S.editing = null;
    _S.isNew   = false;
    _renderList();
}

function _renderForm() {
    const c = _dom.content;
    if (!c) return;
    c.innerHTML = "";

    const cast = _S.editing;
    let idManuallyEdited = !_S.isNew;

    // Name
    const nameEl = _mk("input", {
        cls: "fbt-ce-input", type: "text",
        placeholder: "Cast name *", value: cast.name || "",
    });
    nameEl.addEventListener("input", () => {
        cast.name = nameEl.value;
        if (!idManuallyEdited) {
            idEl.value = _slugify(cast.name);
            cast.id = idEl.value;
        }
    });

    // ID
    const idEl = _mk("input", {
        cls: "fbt-ce-input", type: "text",
        placeholder: "cast_id (auto-generated)", value: cast.id || "",
    });
    idEl.addEventListener("input", () => {
        cast.id = idEl.value.trim();
        idManuallyEdited = true;
    });

    // Entries container
    const entriesEl = _mk("div", { cls: "fbt-cast-entries" });

    const rebuildEntries = () => {
        entriesEl.innerHTML = "";
        if (!cast.entries.length) {
            entriesEl.appendChild(_mk("div", {
                cls: "fbt-be-empty",
                textContent: "No entries — use + Add subject below",
            }));
        } else {
            cast.entries.forEach((entry, i) => {
                entriesEl.appendChild(_buildEntryRow(cast, entry, i, rebuildEntries));
            });
        }
    };
    rebuildEntries();

    const addBtn = _mk("button", {
        cls: "fbt-ce-add-btn", textContent: "+ Add subject",
        onclick: () => {
            cast.entries.push({ subject_id: "", bundle_id: "", visual_mode: "images", use_audio: false });
            rebuildEntries();
            // Focus the new row's subject select
            const rows = entriesEl.querySelectorAll(".fbt-cast-entry-row");
            rows[rows.length - 1]?.querySelector("select")?.focus();
        },
    });

    // Warnings (non-blocking — shown before and after save)
    const warnEl = _mk("div", { cls: "fbt-cast-warnings", style: { display: "none" } });

    // Buttons
    const btnRow = _mk("div", { cls: "fbt-be-btn-row" });
    btnRow.appendChild(_mk("button", {
        cls: "fbt-ce-btn fbt-ce-btn-primary", textContent: "Save",
        onclick: () => _onSave(cast, warnEl),
    }));
    if (!_S.isNew) {
        btnRow.appendChild(_mk("button", {
            cls: "fbt-ce-btn fbt-ce-danger", textContent: "Delete cast",
            onclick: () => _onDelete(cast.id, cast.name),
        }));
    }
    btnRow.appendChild(_mk("button", {
        cls: "fbt-ce-btn", textContent: "Cancel",
        onclick: () => _cancelEdit(),
    }));

    const form = _mk("div", { cls: "fbt-be-form" });
    form.appendChild(_fRow("Name", nameEl));
    form.appendChild(_fRow("ID", idEl));
    form.appendChild(_mk("div", { cls: "fbt-be-section" }, [
        _mk("div", { cls: "fbt-be-sec-label", textContent: "Entries" }),
        entriesEl,
        addBtn,
    ]));
    form.appendChild(warnEl);
    form.appendChild(btnRow);

    c.appendChild(form);
    nameEl.focus();
}

function _fRow(label, fieldEl) {
    return _mk("div", { cls: "fbt-ce-row" }, [
        _mk("div", { cls: "fbt-ce-label", textContent: label }),
        _mk("div", { cls: "fbt-ce-input-wrap" }, [fieldEl]),
    ]);
}

// ── Entry row ─────────────────────────────────────────────────────────────────

function _buildEntryRow(cast, entry, idx, rebuildAll) {
    const row = _mk("div", { cls: "fbt-cast-entry-row" });

    // ── Subject select ─────────────────────────────────────────────────────

    const subjEl = document.createElement("select");
    subjEl.className = "fbt-ce-select fbt-cast-entry-subject";
    const _fillSubjectSel = () => {
        subjEl.innerHTML = "";
        const blank = document.createElement("option");
        blank.value = "";
        blank.textContent = "— subject —";
        if (!entry.subject_id) blank.selected = true;
        subjEl.appendChild(blank);
        _S.subjects.forEach(s => {
            const o = document.createElement("option");
            o.value = s.id;
            o.textContent = s.name || s.id;
            if (s.id === entry.subject_id) o.selected = true;
            subjEl.appendChild(o);
        });
    };
    _fillSubjectSel();

    // ── Bundle select ──────────────────────────────────────────────────────

    const bundleEl = document.createElement("select");
    bundleEl.className = "fbt-ce-select fbt-cast-entry-bundle";
    const _fillBundleSel = () => {
        bundleEl.innerHTML = "";
        const blank = document.createElement("option");
        blank.value = "";
        blank.textContent = "— bundle —";
        if (!entry.bundle_id) blank.selected = true;
        bundleEl.appendChild(blank);
        const available = _bundlesForSubject(entry.subject_id);
        available.forEach(b => {
            const o = document.createElement("option");
            o.value = b.id;
            o.textContent = b.name || b.id;
            if (b.id === entry.bundle_id) o.selected = true;
            bundleEl.appendChild(o);
        });
        if (!available.length && entry.subject_id) {
            const msg = document.createElement("option");
            msg.value = "";
            msg.textContent = "No bundles for this subject";
            msg.disabled = true;
            bundleEl.appendChild(msg);
        }
    };
    _fillBundleSel();

    // ── Visual mode toggle ──────────────────────────────────────────────────

    const modeWrap = _mk("div", { cls: "fbt-be-toggle-row fbt-cast-mode-toggle" });
    ["images", "video"].forEach(val => {
        const btn = _mk("button", {
            cls: "fbt-be-toggle-btn" + (entry.visual_mode === val ? " active" : ""),
            textContent: val === "images" ? "Img" : "Vid",
            title: val === "images" ? "Images mode" : "Video mode",
        });
        btn.dataset.mode = val;
        btn.addEventListener("click", () => {
            entry.visual_mode = val;
            modeWrap.querySelectorAll(".fbt-be-toggle-btn").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            _syncModeHighlight();
        });
        modeWrap.appendChild(btn);
    });

    const _syncModeHighlight = () => {
        const bundle = _bundleLookup(entry.bundle_id);
        const differs = bundle && bundle.visual?.type !== entry.visual_mode;
        modeWrap.classList.toggle("differs", !!differs);
        if (differs) {
            modeWrap.title = `Bundle default is "${bundle.visual?.type}" — overriding to "${entry.visual_mode}"`;
        } else {
            modeWrap.removeAttribute("title");
        }
    };
    _syncModeHighlight();

    // ── Audio toggle ────────────────────────────────────────────────────────

    const audioLabel = _mk("label", {
        cls: "fbt-cast-audio-label",
        title: "Use audio reference from this bundle",
    });
    const audioCheck = _mk("input", { type: "checkbox" });
    audioCheck.checked = !!entry.use_audio;
    audioCheck.addEventListener("change", () => { entry.use_audio = audioCheck.checked; });
    audioLabel.appendChild(audioCheck);
    audioLabel.appendChild(_mk("span", { cls: "fbt-cast-audio-icon", textContent: "🎙" }));

    // ── Remove button ───────────────────────────────────────────────────────

    const removeBtn = _mk("button", {
        cls: "fbt-ce-icon-btn fbt-ce-danger fbt-cast-remove-btn",
        title: "Remove entry", textContent: "✕",
        onclick: () => { cast.entries.splice(idx, 1); rebuildAll(); },
    });

    // ── Wire events ─────────────────────────────────────────────────────────

    subjEl.addEventListener("change", () => {
        entry.subject_id = subjEl.value;
        // If current bundle belongs to a different subject, clear it
        const bundle = _bundleLookup(entry.bundle_id);
        if (bundle && bundle.subject_id !== entry.subject_id) {
            entry.bundle_id = "";
        }
        _fillBundleSel();
        _syncModeHighlight();
    });

    bundleEl.addEventListener("change", () => {
        entry.bundle_id = bundleEl.value;
        // Auto-inherit bundle's visual.type as the mode default
        const bundle = _bundleLookup(entry.bundle_id);
        if (bundle) {
            entry.visual_mode = bundle.visual?.type || "images";
            modeWrap.querySelectorAll(".fbt-be-toggle-btn").forEach(btn => {
                btn.classList.toggle("active", btn.dataset.mode === entry.visual_mode);
            });
        }
        _syncModeHighlight();
    });

    // ── Assemble: two-line layout ───────────────────────────────────────────

    const topLine = _mk("div", { cls: "fbt-cast-entry-top" }, [subjEl, bundleEl]);
    const botLine = _mk("div", { cls: "fbt-cast-entry-bot" }, [modeWrap, audioLabel, removeBtn]);

    row.appendChild(topLine);
    row.appendChild(botLine);
    return row;
}

// ── Validation (non-blocking) ─────────────────────────────────────────────────

function _collectWarnings(cast) {
    const warnings = [];
    const seen = new Set();
    cast.entries.forEach((entry, i) => {
        const label = `Entry ${i + 1}`;
        if (!entry.subject_id) {
            warnings.push(`${label}: no subject selected.`);
            return;
        }
        if (seen.has(entry.subject_id)) {
            warnings.push(`"${_subjectName(entry.subject_id)}" appears more than once.`);
        } else {
            seen.add(entry.subject_id);
        }
        const bundle = _bundleLookup(entry.bundle_id);
        if (!entry.bundle_id) {
            warnings.push(`${label} (${_subjectName(entry.subject_id)}): no bundle selected.`);
        } else if (bundle && bundle.subject_id && bundle.subject_id !== entry.subject_id) {
            warnings.push(`${label}: bundle "${bundle.name || bundle.id}" belongs to "${_subjectName(bundle.subject_id)}", not the selected subject.`);
        }
        if (entry.visual_mode === "video" && bundle && !bundle.visual?.file) {
            warnings.push(`${label} (${_subjectName(entry.subject_id)}): video mode but bundle has no video file.`);
        }
        if (entry.use_audio && bundle?.audio?.source === "extract_from_visual" && entry.visual_mode === "images") {
            warnings.push(`${label} (${_subjectName(entry.subject_id)}): audio + extract-from-visual requires video mode.`);
        }
    });
    return warnings;
}

async function _onSave(cast, warnEl) {
    const name = (cast.name || "").trim();
    const id   = (cast.id   || "").trim();
    if (!name) { _toast("Cast name is required", "warn"); return; }
    if (!id)   { _toast("Cast ID is required", "warn"); return; }

    const warnings = _collectWarnings(cast);
    warnEl.innerHTML = "";
    if (warnings.length) {
        warnings.forEach(w => warnEl.appendChild(_mk("div", { textContent: "⚠ " + w })));
        warnEl.style.display = "";
    } else {
        warnEl.style.display = "none";
    }

    // Warnings are non-blocking — save proceeds regardless
    try {
        await bundlesApi.saveCast({ ...cast, id, name });
        const res = await bundlesApi.listCasts();
        _S.casts   = res.casts ?? [];
        _S.editing = null;
        _S.isNew   = false;
        _renderList();
        bundlesApi.reloadCasts().catch(() => {});
        const severity = warnings.length ? "warn" : "success";
        _toast(`Saved "${name}"` + (warnings.length ? " (check warnings)" : ""), severity);
    } catch (e) {
        warnEl.innerHTML = "";
        warnEl.appendChild(_mk("div", { textContent: "Error: " + e.message }));
        warnEl.style.display = "";
        _toast("Save failed", "error");
    }
}

// ── Send to Workflow ──────────────────────────────────────────────────────────

const SCB_NODE_TYPE = "fbt_SceneCastBuild";

function _onSendToWorkflow(wrapEl) {
    // Toggle: close existing picker if already open
    const existing = wrapEl.querySelector(".fbt-ce-stw-picker");
    if (existing) { existing.remove(); return; }

    if (!_S.editing) {
        _toast("Open a cast to edit before sending to workflow", "warn");
        return;
    }

    const entries = _S.editing.entries || [];
    if (!entries.length) {
        _toast("Add at least one entry before sending", "warn");
        return;
    }

    const app = window._fbtApp;
    const graphNodes = app?.graph?._nodes || [];
    const builders = graphNodes.filter(n => n.type === SCB_NODE_TYPE);

    const picker = document.createElement("div");
    picker.className = "fbt-ce-stw-picker";

    if (!builders.length) {
        const empty = document.createElement("div");
        empty.className = "fbt-ce-stw-empty";
        empty.textContent = "No Scene Cast Build nodes on canvas";
        picker.appendChild(empty);
    } else {
        builders.forEach(node => {
            // Show first non-empty subject widget as the node label
            const firstSubj = node.widgets?.find(
                w => w.name && w.name.startsWith("subject_") && w.value
            )?.value || "(empty)";

            const row = document.createElement("div");
            row.className = "fbt-ce-stw-row";
            row.title = `Push ${entries.length} ${entries.length === 1 ? "entry" : "entries"} to node #${node.id}`;

            const label = document.createElement("span");
            label.className = "fbt-ce-stw-row-label";
            label.textContent = firstSubj;
            const idx = document.createElement("span");
            idx.className = "fbt-ce-stw-row-idx";
            idx.textContent = `#${node.id}`;
            row.appendChild(label);
            row.appendChild(idx);

            row.addEventListener("click", () => {
                _pushToNode(node, entries);
                picker.remove();
                document.removeEventListener("pointerdown", _closeOnOutside, true);
                _toast(`Sent ${entries.length} ${entries.length === 1 ? "entry" : "entries"} to node #${node.id}`, "success");
            });
            picker.appendChild(row);
        });
    }

    // "Create new" row
    const newRow = document.createElement("div");
    newRow.className = "fbt-ce-stw-row fbt-ce-stw-new";
    newRow.textContent = "＋ New Scene Cast Build node";
    newRow.title = "Add a new Scene Cast Build node to the canvas";
    newRow.addEventListener("click", () => {
        picker.remove();
        document.removeEventListener("pointerdown", _closeOnOutside, true);
        if (typeof LiteGraph === "undefined" || !app?.graph) {
            _toast("Canvas not available", "error");
            return;
        }
        const node = LiteGraph.createNode(SCB_NODE_TYPE);
        if (!node) { _toast("Could not create node", "error"); return; }
        const mouse = app.canvas?.canvas_mouse;
        node.pos = mouse ? [mouse[0] + 20, mouse[1] + 20] : [200, 200];
        app.graph.add(node);
        setTimeout(() => {
            _pushToNode(node, entries);
        }, 120);
        _toast("Created Scene Cast Build node", "success");
    });
    picker.appendChild(newRow);

    const _closeOnOutside = (e) => {
        if (!wrapEl.contains(e.target)) {
            picker.remove();
            document.removeEventListener("pointerdown", _closeOnOutside, true);
        }
    };
    document.addEventListener("pointerdown", _closeOnOutside, true);

    wrapEl.appendChild(picker);
}

function _pushToNode(node, entries) {
    // JSON-backed: write all entries as one JSON string to the backing widget,
    // then tell the node's DOM table to refresh itself.
    const jsonWidget = node.widgets?.find(w => w.name === "cast_entries_json");
    if (jsonWidget) jsonWidget.value = JSON.stringify(entries);
    node._refreshCastTable?.(entries);
    window._fbtApp?.graph?.setDirtyCanvas(true, false);
}

// ── Top bar ───────────────────────────────────────────────────────────────────

function _buildTopBar() {
    const bar = _mk("div", { cls: "fbt-be-top-bar" });

    bar.appendChild(_mk("button", {
        cls: "fbt-ce-btn fbt-ce-btn-primary fbt-be-new-btn",
        textContent: "+ New Cast",
        onclick: () => _startNew(),
    }));

    // Send to Workflow wrapper + button
    const stwWrap = _mk("div", { cls: "fbt-ce-stw-wrap" });
    stwWrap.appendChild(_mk("button", {
        cls: "fbt-ce-btn",
        textContent: "Send to Workflow",
        title: "Push the current cast entries to a Scene Cast Build node on the canvas",
        onclick: () => _onSendToWorkflow(stwWrap),
    }));
    bar.appendChild(stwWrap);

    bar.appendChild(_mk("button", {
        cls: "fbt-ce-icon-btn fbt-be-refresh-btn",
        textContent: "↺", title: "Refresh cast and bundle lists",
        onclick: async () => {
            try {
                await _loadAll();
                if (!_S.editing) _renderList();
                _toast("Refreshed", "success");
            } catch (e) {
                _toast("Refresh failed: " + e.message, "error");
            }
        },
    }));

    return bar;
}

// ── Main render ───────────────────────────────────────────────────────────────

export async function renderCastEditor(el) {
    el.innerHTML = "";

    const panel = _mk("div", { cls: "fbt-be-panel" });
    _dom.content = _mk("div", { cls: "fbt-be-content" });

    _dom.content.appendChild(_mk("div", { cls: "fbt-be-empty", textContent: "Loading…" }));

    panel.appendChild(_buildTopBar());
    panel.appendChild(_dom.content);
    el.appendChild(panel);

    try {
        await _loadAll();
    } catch (e) {
        console.error("fbt CastEditor: load error", e);
    }
    _renderList();
}
