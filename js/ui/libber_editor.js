/**
 * Libber Editor — sidebar panel.
 *
 * Lets users create and manage Libber files — named key/value text-substitution
 * libraries used in prompts via %key% delimiters.
 *
 * List view: search + paged cards
 * Detail view: full CRUD on key/value entries, delimiter, and max substitution depth
 */

import { libberAPI }       from "../api/libber.js";
import { compositionsApi } from "../api/compositions.js";

const PAGE_SIZE = 15;

// ── Module state ───────────────────────────────────────────────────────────────

const _S = {
    libbers:          [],    // [{name, entry_count, delimiter, max_depth, filepath?, unsaved?}]
    filterText:       "",
    page:             0,
    editing:          null,  // {name, lib_dict:{}, delimiter, max_depth} | null = list view
    editingOrigName:  "",    // name at time of open (for rename detection)
    isNew:            false,
    defaultDelimiter: "%",
    defaultMaxDepth:  10,
    saveTimer:        null,
};

const _dom = {
    saveStatus:  null,  // span showing pending/saving/saved state in form header
    savedFadeTimer: null,
};

// ── Helpers ────────────────────────────────────────────────────────────────────

function _mk(tag, props = {}, children = []) {
    const el = document.createElement(tag);
    Object.entries(props).forEach(([k, v]) => {
        if (k === "cls")        el.className = v;
        else if (k === "style") Object.assign(el.style, v);
        else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
        else el[k] = v;
    });
    children.filter(Boolean).forEach(c =>
        el.appendChild(typeof c === "string" ? document.createTextNode(c) : c));
    return el;
}

function _toast(msg, severity = "info") {
    try { window._fbtApp?.extensionManager?.toast?.add({ severity, summary: msg, life: 2500 }); }
    catch (_) {}
}

function _errMsg(err) {
    try {
        if (err?.response) {
            const b = typeof err.response === "string" ? JSON.parse(err.response) : err.response;
            if (b?.error) return b.error;
        }
    } catch (_) {}
    return err?.message || String(err);
}

function _filtered() {
    const q = _S.filterText.toLowerCase().trim();
    if (!q) return _S.libbers;
    return _S.libbers.filter(l => l.name.toLowerCase().includes(q));
}

// ── Data load ──────────────────────────────────────────────────────────────────

async function _loadAll() {
    const [scanRes, settingsRes] = await Promise.allSettled([
        libberAPI.scan(),
        compositionsApi.getSettings(),
    ]);
    _S.libbers          = scanRes.value?.libbers   ?? [];
    _S.defaultDelimiter = settingsRes.value?.libber_delimiter ?? "%";
    _S.defaultMaxDepth  = settingsRes.value?.libber_max_depth ?? 10;
}

// ── Save-state indicator ───────────────────────────────────────────────────────

function _setSaveState(state) {
    const el = _dom.saveStatus;
    if (!el) return;
    if (_dom.savedFadeTimer) { clearTimeout(_dom.savedFadeTimer); _dom.savedFadeTimer = null; }
    el.className = "fbt-lbe-save-ind";
    el.innerHTML = "";
    if (state === "pending") {
        el.classList.add("fbt-lbe-save-pending");
        el.appendChild(_mk("span", { cls: "fbt-lbe-save-dots", textContent: "···" }));
    } else if (state === "saving") {
        el.classList.add("fbt-lbe-save-saving");
        el.appendChild(_mk("span", { cls: "fbt-lbe-save-spin", textContent: "↻" }));
        el.appendChild(document.createTextNode(" saving"));
    } else if (state === "saved") {
        el.classList.add("fbt-lbe-save-ok");
        el.textContent = "✓ saved";
        _dom.savedFadeTimer = setTimeout(() => {
            el.classList.add("fbt-lbe-save-fade");
            _dom.savedFadeTimer = setTimeout(() => { el.innerHTML = ""; }, 500);
        }, 1200);
    }
}

// ── Auto-save (debounced) ──────────────────────────────────────────────────────

function _scheduleSave() {
    if (_S.saveTimer) clearTimeout(_S.saveTimer);
    _setSaveState("pending");
    _S.saveTimer = setTimeout(_doSave, 600);
}

async function _doSave() {
    const e = _S.editing;
    if (!e || !e.name) return;
    _setSaveState("saving");
    try {
        await libberAPI.saveFull(e.name, e.lib_dict, e.delimiter, e.max_depth);
        _setSaveState("saved");
        libberAPI.scan().then(r => { if (r?.libbers) _S.libbers = r.libbers; }).catch(() => {});
    } catch (err) {
        _setSaveState("");
        _toast("Save failed: " + _errMsg(err), "error");
    }
}

// ── List view ─────────────────────────────────────────────────────────────────

function _renderList() {
    const c = _dom.content;
    if (!c) return;
    c.innerHTML = "";

    const items = _filtered();
    const total      = items.length;
    const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
    _S.page = Math.max(0, Math.min(_S.page, totalPages - 1));
    const start     = _S.page * PAGE_SIZE;
    const pageItems = items.slice(start, start + PAGE_SIZE);

    if (!total) {
        c.appendChild(_mk("div", {
            cls: "fbt-be-empty",
            textContent: _S.filterText
                ? `No libbers matching "${_S.filterText}".`
                : "No libbers saved. Click + New Libber to create one.",
        }));
    } else {
        pageItems.forEach(l => c.appendChild(_buildCard(l)));
    }

    if (_dom.pagination) {
        _dom.pagination.innerHTML = "";
        if (totalPages > 1) {
            const prev = _mk("button", {
                cls: "fbt-ce-pg-btn", textContent: "‹", title: "Previous page",
                onclick: () => { _S.page--; _renderList(); },
            });
            prev.disabled = _S.page === 0;
            const info = _mk("span", { cls: "fbt-ce-pg-info",
                textContent: `${_S.page + 1} / ${totalPages}` });
            const next = _mk("button", {
                cls: "fbt-ce-pg-btn", textContent: "›", title: "Next page",
                onclick: () => { _S.page++; _renderList(); },
            });
            next.disabled = _S.page >= totalPages - 1;
            _dom.pagination.append(prev, info, next);
        }
    }
}

function _buildCard(l) {
    const card = _mk("div", { cls: "fbt-be-card fbt-lbe-card fbt-be-card-clickable" });

    const top = _mk("div", { cls: "fbt-be-card-top" });
    top.appendChild(_mk("span", { cls: "fbt-be-card-name", textContent: l.name }));

    const meta = _mk("span", { cls: "fbt-be-card-meta" });
    meta.appendChild(_mk("span", {
        cls: "fbt-cast-count",
        textContent: `${l.entry_count} ${l.entry_count === 1 ? "entry" : "entries"}`,
    }));
    meta.appendChild(_mk("span", {
        cls: "fbt-cast-date",
        textContent: `delim: ${l.delimiter}  depth: ${l.max_depth}`,
    }));
    if (l.unsaved) {
        meta.appendChild(_mk("span", {
            cls: "fbt-be-tag",
            textContent: "unsaved",
            style: { opacity: "0.7" },
        }));
    }
    top.appendChild(meta);
    card.appendChild(top);

    const actions = _mk("div", { cls: "fbt-be-card-actions" });
    actions.appendChild(_mk("button", {
        cls: "fbt-ce-icon-btn", title: "Edit", textContent: "✎",
        onclick: e => { e.stopPropagation(); _openLibber(l.name); },
    }));
    actions.appendChild(_mk("button", {
        cls: "fbt-ce-icon-btn fbt-ce-danger", title: "Delete", textContent: "✕",
        onclick: e => { e.stopPropagation(); _onDelete(l.name); },
    }));
    card.appendChild(actions);

    card.addEventListener("click", () => _openLibber(l.name));
    return card;
}

async function _openLibber(name) {
    try {
        const data = await libberAPI.open(name);
        _S.editing        = { name: data.name, lib_dict: data.lib_dict, delimiter: data.delimiter, max_depth: data.max_depth };
        _S.editingOrigName = data.name;
        _S.isNew          = false;
        _renderForm();
    } catch (err) {
        _toast("Failed to open: " + _errMsg(err), "error");
    }
}

async function _onDelete(name) {
    if (!confirm(`Delete libber "${name}"? This cannot be undone.`)) return;
    try {
        await libberAPI.deleteFull(name);
        _S.libbers = _S.libbers.filter(l => l.name !== name);
        _renderList();
        _toast(`Deleted "${name}"`, "success");
    } catch (err) {
        _toast("Delete failed: " + _errMsg(err), "error");
    }
}

// ── Detail / editor view ──────────────────────────────────────────────────────

function _startNew() {
    _S.editing = {
        name:      "",
        lib_dict:  {},
        delimiter: _S.defaultDelimiter,
        max_depth: _S.defaultMaxDepth,
    };
    _S.editingOrigName = "";
    _S.isNew = true;
    _renderForm();
}

function _cancelEdit() {
    if (_S.saveTimer) { clearTimeout(_S.saveTimer); _S.saveTimer = null; }
    if (_dom.savedFadeTimer) { clearTimeout(_dom.savedFadeTimer); _dom.savedFadeTimer = null; }
    _dom.saveStatus = null;
    _S.editing = null;
    _S.isNew   = false;
    _renderList();
}

function _renderForm() {
    const c = _dom.content;
    if (!c) return;
    c.innerHTML = "";
    if (_dom.pagination) _dom.pagination.innerHTML = "";

    const e = _S.editing;

    // ── Header row ─────────────────────────────────────────────────────────────
    const hdr = _mk("div", { cls: "fbt-lbe-form-hdr" });
    _dom.saveStatus = _mk("span", { cls: "fbt-lbe-save-ind" });

    hdr.appendChild(_mk("button", {
        cls: "fbt-ce-icon-btn fbt-lbe-back-btn",
        textContent: "← Back",
        title: "Back to list",
        onclick: () => {
            if (_S.isNew && Object.keys(e.lib_dict).length === 0 && !e.name) {
                _cancelEdit();
                return;
            }
            // If pending changes unsaved, save now then go back
            if (_S.saveTimer) {
                clearTimeout(_S.saveTimer);
                _S.saveTimer = null;
                _doSave().then(() => _cancelEdit());
            } else {
                _cancelEdit();
            }
        },
    }));

    const nameInp = _mk("input", {
        cls: "fbt-ce-input fbt-lbe-name-inp",
        type: "text",
        placeholder: "Libber name (used as filename)",
        value: e.name,
    });
    hdr.appendChild(nameInp);
    hdr.appendChild(_dom.saveStatus);
    c.appendChild(hdr);

    // ── Meta row (delimiter + max_depth) ───────────────────────────────────────
    const meta = _mk("div", { cls: "fbt-lbe-meta-row" });

    const delimInp = _mk("input", {
        cls: "fbt-ce-input fbt-lbe-delim-inp",
        type: "text", maxLength: 1,
        value: e.delimiter,
        title: "Delimiter character wrapping each key reference, e.g. %key%",
    });
    meta.appendChild(_mk("label", { cls: "fbt-lbe-meta-label" }, [
        _mk("span", {}, ["Delimiter"]),
        delimInp,
    ]));

    const depthInp = _mk("input", {
        cls: "fbt-ce-input fbt-lbe-depth-inp",
        type: "number", min: "1", max: "50",
        value: String(e.max_depth),
        title: "Maximum recursion depth for nested substitutions",
    });
    meta.appendChild(_mk("label", { cls: "fbt-lbe-meta-label" }, [
        _mk("span", {}, ["Max depth"]),
        depthInp,
    ]));

    c.appendChild(meta);

    // ── Entries table ──────────────────────────────────────────────────────────
    const tableWrap = _mk("div", { cls: "fbt-lbe-table-wrap" });

    function _rebuildRows() {
        tableWrap.innerHTML = "";

        const keys = Object.keys(e.lib_dict).sort();

        if (!keys.length) {
            tableWrap.appendChild(_mk("div", { cls: "fbt-be-empty fbt-lbe-no-entries",
                textContent: "No entries yet. Add one below." }));
        } else {
            const tbl = _mk("table", { cls: "fbt-lbe-table" });
            const thead = _mk("thead");
            thead.appendChild(_mk("tr", {}, [
                _mk("th", { textContent: "Key" }),
                _mk("th", { textContent: "Value" }),
                _mk("th", {}),
            ]));
            tbl.appendChild(thead);

            const tbody = _mk("tbody");
            keys.forEach(key => {
                const tr = _mk("tr", { cls: "fbt-lbe-entry-row" });

                // Key cell
                const keyInp = _mk("input", {
                    cls: "fbt-ce-input fbt-lbe-key-inp",
                    type: "text",
                    value: key,
                    title: "Key — lowercase letters, digits, underscores",
                });
                const keyTd = _mk("td");
                keyTd.appendChild(keyInp);
                tr.appendChild(keyTd);

                // Value cell (textarea)
                const valTa = _mk("textarea", {
                    cls: "fbt-ce-input fbt-lbe-val-ta",
                    rows: "2",
                });
                valTa.value = e.lib_dict[key] ?? "";
                const valTd = _mk("td");
                valTd.appendChild(valTa);
                tr.appendChild(valTd);

                // Delete cell
                const delBtn = _mk("button", {
                    cls: "fbt-ce-icon-btn fbt-ce-danger fbt-lbe-del-btn",
                    title: "Remove entry", textContent: "✕",
                });
                const delTd = _mk("td");
                delTd.appendChild(delBtn);
                tr.appendChild(delTd);

                // Events
                let _pendingKey = key;
                keyInp.addEventListener("blur", () => {
                    const newKey = keyInp.value.trim().toLowerCase().replace(/[\s-]+/g, "_").replace(/[^\w]/g, "");
                    if (!newKey || newKey === _pendingKey) {
                        keyInp.value = _pendingKey; // revert invalid
                        return;
                    }
                    const val = e.lib_dict[_pendingKey];
                    delete e.lib_dict[_pendingKey];
                    e.lib_dict[newKey] = val;
                    _pendingKey = newKey;
                    keyInp.value = newKey;
                    _scheduleSave();
                    // Rebuild to re-sort
                    _rebuildRows();
                });

                valTa.addEventListener("input", () => {
                    e.lib_dict[_pendingKey] = valTa.value;
                    _scheduleSave();
                });

                delBtn.addEventListener("click", () => {
                    delete e.lib_dict[_pendingKey];
                    _scheduleSave();
                    _rebuildRows();
                });

                tbody.appendChild(tr);
            });
            tbl.appendChild(tbody);
            tableWrap.appendChild(tbl);
        }

        // ── Add entry form ─────────────────────────────────────────────────────
        const addRow = _mk("div", { cls: "fbt-lbe-add-row" });

        const newKeyInp = _mk("input", {
            cls: "fbt-ce-input fbt-lbe-new-key",
            type: "text", placeholder: "new_key",
        });
        const newValTa = _mk("textarea", {
            cls: "fbt-ce-input fbt-lbe-new-val",
            placeholder: "value",
            rows: "2",
        });
        const addBtn = _mk("button", {
            cls: "fbt-ce-btn fbt-ce-btn-primary",
            textContent: "Add",
        });

        const _doAdd = () => {
            const rawKey = newKeyInp.value.trim().toLowerCase().replace(/[\s-]+/g, "_").replace(/[^\w]/g, "");
            if (!rawKey) { _toast("Key cannot be empty", "warn"); return; }
            e.lib_dict[rawKey] = newValTa.value;
            newKeyInp.value = "";
            newValTa.value  = "";
            _scheduleSave();
            _rebuildRows();
        };

        addBtn.addEventListener("click", _doAdd);
        newValTa.addEventListener("keydown", ev => {
            if (ev.key === "Enter" && ev.ctrlKey) { ev.preventDefault(); _doAdd(); }
        });

        addRow.append(newKeyInp, newValTa, addBtn);
        tableWrap.appendChild(addRow);
    }

    _rebuildRows();
    c.appendChild(tableWrap);

    // ── Save button (explicit, for new libbers before auto-save can run) ────────
    if (_S.isNew) {
        const saveBtn = _mk("button", {
            cls: "fbt-ce-btn fbt-ce-btn-primary fbt-lbe-save-btn",
            textContent: "Create Libber",
        });
        saveBtn.addEventListener("click", async () => {
            const nm = nameInp.value.trim();
            if (!nm) { _toast("Name is required", "warn"); nameInp.focus(); return; }
            e.name = nm;
            try {
                await libberAPI.saveFull(e.name, e.lib_dict, e.delimiter, e.max_depth);
                _S.isNew = false;
                _S.editingOrigName = e.name;
                await _loadAll();
                _toast(`Created "${e.name}"`, "success");
                // Re-render form without the Create button now that it's persisted
                _renderForm();
            } catch (err) {
                _toast("Create failed: " + _errMsg(err), "error");
            }
        });
        c.appendChild(saveBtn);
    }

    // Wire meta inputs (after form is in DOM)
    nameInp.addEventListener("blur", async () => {
        if (_S.isNew) return; // handled by Create button
        const newName = nameInp.value.trim();
        if (!newName || newName === _S.editingOrigName) return;
        try {
            await libberAPI.rename(_S.editingOrigName, newName);
            e.name = newName;
            _S.editingOrigName = newName;
            await _loadAll();
            _toast(`Renamed to "${newName}"`, "success");
        } catch (err) {
            nameInp.value = _S.editingOrigName; // revert
            _toast("Rename failed: " + _errMsg(err), "error");
        }
    });

    delimInp.addEventListener("change", () => {
        const d = delimInp.value;
        if (!d.length) { delimInp.value = e.delimiter; return; }
        e.delimiter = d[0];
        delimInp.value = e.delimiter;
        _scheduleSave();
    });

    depthInp.addEventListener("change", () => {
        const v = parseInt(depthInp.value, 10);
        if (isNaN(v)) { depthInp.value = e.max_depth; return; }
        e.max_depth = Math.max(1, Math.min(50, v));
        depthInp.value = e.max_depth;
        _scheduleSave();
    });
}

// ── Top bar ───────────────────────────────────────────────────────────────────

function _buildTopBar() {
    const bar = _mk("div", { cls: "fbt-be-top-bar" });

    bar.appendChild(_mk("button", {
        cls: "fbt-ce-btn fbt-ce-btn-primary fbt-be-new-btn",
        textContent: "+ New Libber",
        onclick: () => _startNew(),
    }));

    const searchInp = _mk("input", {
        cls: "fbt-ce-search fbt-lbe-search",
        type: "text", placeholder: "Search libbers…",
    });
    searchInp.addEventListener("input", () => {
        _S.filterText = searchInp.value;
        _S.page = 0;
        if (!_S.editing) _renderList();
    });
    bar.appendChild(searchInp);

    bar.appendChild(_mk("button", {
        cls: "fbt-ce-icon-btn fbt-be-refresh-btn",
        textContent: "↺", title: "Refresh libber list",
        onclick: async () => {
            try {
                await _loadAll();
                if (!_S.editing) _renderList();
                _toast("Refreshed", "success");
            } catch (err) {
                _toast("Refresh failed: " + _errMsg(err), "error");
            }
        },
    }));

    return bar;
}

// ── Main render ───────────────────────────────────────────────────────────────

export async function renderLibberEditor(el) {
    el.innerHTML = "";

    const panel = _mk("div", { cls: "fbt-be-panel" });
    panel.dataset.fbtEditor = "libber";
    panel.addEventListener("keydown",  e => e.stopPropagation());
    panel.addEventListener("keyup",    e => e.stopPropagation());
    panel.addEventListener("keypress", e => e.stopPropagation());

    _dom.content    = _mk("div", { cls: "fbt-be-content" });
    _dom.pagination = _mk("div", { cls: "fbt-ce-saved-pagination" });

    _dom.content.appendChild(_mk("div", { cls: "fbt-be-empty", textContent: "Loading…" }));

    panel.appendChild(_buildTopBar());
    panel.appendChild(_dom.content);
    panel.appendChild(_dom.pagination);
    el.appendChild(panel);

    try {
        await _loadAll();
    } catch (err) {
        console.error("fbt LibberEditor: load error", err);
    }
    _renderList();
}
