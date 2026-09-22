/**
 * Reusable folder-tree browser with Input / Output tabs.
 *
 * Sibling to file_tree.js, for pickers that need to select a *folder* rather than a
 * file — including brand-new/empty ones. file_tree.js can't represent that: its tree
 * is built entirely from a flat list of file paths, so a folder with nothing in it
 * (or nothing matching) never appears at all.
 *
 * Usage:
 *   const { el, rebuild } = buildFolderTree({
 *     inputDirs:  ["video", "video/clip1"],
 *     outputDirs: ["archives", "archives/2026"],
 *     isSelected: (path, dir) => path === current && dir === curDir,
 *     onSelect:   (path, dir) => setFolder(path, dir),
 *   });
 *   wrap.appendChild(el);
 *
 * Unlike file_tree.js's leaves, a folder row both selects *and* expands/collapses on
 * click — there is no separate "open" affordance, since every node here is the same
 * kind of thing (a folder), not a mix of expandable dirs and clickable files.
 *
 * @param {object}    opts
 * @param {string[]}  [opts.inputDirs]   Paths available in the Input tab
 * @param {string[]}  [opts.outputDirs]  Paths available in the Output tab
 * @param {string}    [opts.rootLabel]   Label for the synthetic "select the root itself" row
 * @param {function}  [opts.isSelected]  (path, dir) => bool — highlight a path ("" = the root)
 * @param {function}  [opts.onSelect]    (path, dir) => void — called on click
 * @param {string}    [opts.emptyText]   Message shown when a tab has no subfolders; "{dir}" replaced
 * @param {string}    [opts.initialDir]  "input" | "output"
 *
 * @returns {{ el: HTMLElement, rebuild: (opts?: object) => void, getDir: () => string }}
 */

function _mk(tag, props = {}, children = []) {
    const el = document.createElement(tag);
    Object.entries(props).forEach(([k, v]) => {
        if (k === "cls")             el.className = v;
        else if (k === "style")      Object.assign(el.style, v);
        else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
        else                         el[k] = v;
    });
    children.forEach(c => c && el.appendChild(c));
    return el;
}

function _insertDir(node, parts, fullPath) {
    const name = parts[0];
    if (!node.dirs.has(name)) node.dirs.set(name, { path: fullPath, dirs: new Map() });
    if (parts.length > 1) _insertDir(node.dirs.get(name), parts.slice(1), fullPath);
}

export function buildFolderTree({
    inputDirs   = [],
    outputDirs  = [],
    rootLabel   = "(this folder)",
    isSelected  = () => false,
    onSelect    = () => {},
    emptyText   = "No subfolders in {dir}/",
    initialDir  = "input",
} = {}) {
    let _inputDirs  = inputDirs;
    let _outputDirs = outputDirs;
    let _activeDir  = initialDir;
    let _searchQuery = "";

    const treeEl = _mk("div", { cls: "fbt-be-tree" });

    const _row = (name, path, depth, isDir) => {
        const el = _mk("div", { cls: "fbt-be-tree-file" + (isDir ? " fbt-ft-dir-row" : "") });
        el.style.paddingLeft = (depth * 14 + 2) + "px";
        el.dataset.treePath = path;
        el.dataset.treeDir  = _activeDir;
        if (isSelected(path, _activeDir)) el.classList.add("fbt-be-tree-file-cur");
        el.appendChild(_mk("span", { cls: "fbt-be-tree-file-name", textContent: name, title: path || "/" }));
        return el;
    };

    const _renderNode = (node, depth) => {
        const wrap = _mk("div", { cls: "fbt-be-tree-node" });
        [...node.dirs.entries()]
            .sort(([a], [b]) => a.localeCompare(b))
            .forEach(([name, child]) => {
                const childWrap = _mk("div", { cls: "fbt-be-tree-children" });
                childWrap.style.display = "none";
                const arrow = _mk("span", { cls: "fbt-be-tree-arrow", textContent: "▶" });
                const row = _row(name, child.path, depth, true);
                row.prepend(arrow, document.createTextNode(" "));
                row.addEventListener("click", () => {
                    const opening = childWrap.style.display === "none";
                    childWrap.style.display = opening ? "" : "none";
                    arrow.textContent = opening ? "▼" : "▶";
                    if (opening && !childWrap.firstChild && child.dirs.size)
                        childWrap.appendChild(_renderNode(child, depth + 1));
                    treeEl.querySelectorAll(".fbt-be-tree-file-cur").forEach(e => e.classList.remove("fbt-be-tree-file-cur"));
                    row.classList.add("fbt-be-tree-file-cur");
                    onSelect(child.path, _activeDir);
                });
                wrap.append(row, childWrap);
            });
        return wrap;
    };

    const _rebuildTree = () => {
        treeEl.innerHTML = "";
        const raw = _activeDir === "output" ? _outputDirs : _inputDirs;

        const rootRow = _row(rootLabel, "", 0, false);
        rootRow.addEventListener("click", () => {
            treeEl.querySelectorAll(".fbt-be-tree-file-cur").forEach(e => e.classList.remove("fbt-be-tree-file-cur"));
            rootRow.classList.add("fbt-be-tree-file-cur");
            onSelect("", _activeDir);
        });
        treeEl.appendChild(rootRow);

        if (_searchQuery) {
            const q = _searchQuery.toLowerCase();
            const matches = raw.filter(p => p.toLowerCase().includes(q));
            matches.forEach(path => {
                const name = path.split("/").pop();
                const row = _row(name, path, 0, false);
                row.classList.add("fbt-be-tree-file-flat");
                if (path.includes("/"))
                    row.appendChild(_mk("span", { cls: "fbt-be-tree-file-subpath", textContent: path.slice(0, path.lastIndexOf("/") + 1) }));
                row.addEventListener("click", () => {
                    treeEl.querySelectorAll(".fbt-be-tree-file-cur").forEach(e => e.classList.remove("fbt-be-tree-file-cur"));
                    row.classList.add("fbt-be-tree-file-cur");
                    onSelect(path, _activeDir);
                });
                treeEl.appendChild(row);
            });
            if (!matches.length)
                treeEl.appendChild(_mk("div", { cls: "fbt-be-media-empty", textContent: `No matches for "${_searchQuery}"` }));
            return;
        }

        if (!raw.length) {
            treeEl.appendChild(_mk("div", { cls: "fbt-be-media-empty", textContent: emptyText.replace("{dir}", _activeDir) }));
            return;
        }
        const root = { dirs: new Map() };
        raw.forEach(p => _insertDir(root, p.split("/"), p));
        treeEl.appendChild(_renderNode(root, 1));
    };

    const tabRow = _mk("div", { cls: "fbt-be-tree-tab-row" });
    const tabBtns = {};
    ["input", "output"].forEach(dir => {
        const btn = _mk("button", { cls: "fbt-be-tree-tab" + (dir === _activeDir ? " active" : ""), textContent: dir === "input" ? "Input" : "Output" });
        btn.addEventListener("click", () => {
            if (_activeDir === dir) return;
            _activeDir = dir;
            Object.values(tabBtns).forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            _rebuildTree();
        });
        tabBtns[dir] = btn;
        tabRow.appendChild(btn);
    });

    _rebuildTree();

    const searchEl = _mk("input", { cls: "fbt-be-tree-search", type: "text", placeholder: "Filter…" });
    ["keydown", "keyup", "keypress"].forEach(ev => searchEl.addEventListener(ev, e => e.stopPropagation()));
    searchEl.addEventListener("input", () => { _searchQuery = searchEl.value.trim(); _rebuildTree(); });

    const wrap = _mk("div", { cls: "fbt-be-file-tree-wrap" });
    wrap.append(tabRow, searchEl, treeEl);

    return {
        el: wrap,
        rebuild({ inputDirs: newIn, outputDirs: newOut } = {}) {
            if (newIn !== undefined) _inputDirs = newIn;
            if (newOut !== undefined) _outputDirs = newOut;
            _rebuildTree();
        },
        getDir: () => _activeDir,
    };
}
