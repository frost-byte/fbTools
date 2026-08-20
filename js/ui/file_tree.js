/**
 * Reusable file-tree browser with Input / Output tabs.
 *
 * Extracted from the identical tree logic that lived in bundle_editor.js and
 * composition_editor.js so it lives in one place.
 *
 * Usage:
 *   const { el, rebuild, getDir } = buildFileTree({
 *     inputFiles:   _S.mediaInVideos,
 *     outputFiles:  _S.mediaOutVideos,
 *     filter:       f => /\.(mp4|mov)$/i.test(f),
 *     isSelected:   (path, dir) => path === current && dir === curDir,
 *     onSelect:     (path, dir) => setVideo(path, dir),
 *     emptyText:    "No videos in {dir}/",
 *   });
 *   modal.appendChild(el);
 *
 *   // Later, when file lists refresh:
 *   rebuild({ inputFiles: newList });
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

function _insertPath(node, parts, fullPath) {
    if (parts.length === 1) {
        node.files.push({ name: parts[0], path: fullPath });
    } else {
        const dir = parts[0];
        if (!node.dirs.has(dir))
            node.dirs.set(dir, { dirs: new Map(), files: [] });
        _insertPath(node.dirs.get(dir), parts.slice(1), fullPath);
    }
}

/**
 * Build a file-tree browser widget with Input/Output tabs.
 *
 * @param {object}    opts
 * @param {string[]}  [opts.inputFiles]   Paths available in the Input tab
 * @param {string[]}  [opts.outputFiles]  Paths available in the Output tab
 * @param {function}  [opts.filter]       (path) => bool — include only matching paths
 * @param {function}  [opts.isSelected]   (path, dir) => bool — highlight a path
 * @param {function}  [opts.onSelect]     (path, dir) => void — called on click
 * @param {function}  [opts.onHover]      (path, dir) => void — called on mouseenter (optional)
 * @param {function}  [opts.onHoverOut]   () => void — called on mouseleave (optional)
 * @param {string}    [opts.emptyText]    Message shown when no files match; "{dir}" is replaced
 * @param {string}    [opts.initialDir]   "input" | "output"
 *
 * @returns {{ el: HTMLElement, rebuild: (opts?: object) => void, getDir: () => string }}
 */
export function buildFileTree({
    inputFiles  = [],
    outputFiles = [],
    filter      = null,
    isSelected  = () => false,
    onSelect    = () => {},
    onHover     = null,
    onHoverOut  = null,
    emptyText   = "No files in {dir}/",
    initialDir  = "input",
} = {}) {
    let _inputFiles  = inputFiles;
    let _outputFiles = outputFiles;
    let _activeDir   = initialDir;

    const treeEl = _mk("div", { cls: "fbt-be-tree" });

    const _renderNode = (node, depth) => {
        const el     = _mk("div", { cls: "fbt-be-tree-node" });
        const indent = depth * 14;

        [...node.dirs.entries()]
            .sort(([a], [b]) => a.localeCompare(b))
            .forEach(([dirName, child]) => {
                const childWrap = _mk("div", { cls: "fbt-be-tree-children" });
                childWrap.style.display = "none";

                const arrow  = _mk("span", { cls: "fbt-be-tree-arrow", textContent: "▶" });
                const toggle = _mk("button", { cls: "fbt-be-tree-toggle" });
                toggle.style.paddingLeft = (indent + 2) + "px";
                toggle.append(arrow, document.createTextNode(" " + dirName + "/"));
                toggle.addEventListener("click", () => {
                    const opening = childWrap.style.display === "none";
                    childWrap.style.display = opening ? "" : "none";
                    arrow.textContent = opening ? "▼" : "▶";
                    if (opening && !childWrap.firstChild)
                        childWrap.appendChild(_renderNode(child, depth + 1));
                });

                el.appendChild(_mk("div", {}, [toggle, childWrap]));
            });

        [...node.files]
            .sort((a, z) => a.name.localeCompare(z.name))
            .forEach(({ name, path }) => {
                const fileEl = _mk("div", { cls: "fbt-be-tree-file" });
                fileEl.style.paddingLeft = (indent + 16) + "px";
                fileEl.dataset.treePath  = path;
                fileEl.dataset.treeDir   = _activeDir;

                if (isSelected(path, _activeDir))
                    fileEl.classList.add("fbt-be-tree-file-cur");

                fileEl.appendChild(
                    _mk("span", { cls: "fbt-be-tree-file-name", textContent: name, title: path })
                );

                if (onHover) {
                    fileEl.addEventListener("mouseenter", () => onHover(path, _activeDir));
                    if (onHoverOut) fileEl.addEventListener("mouseleave", onHoverOut);
                }

                fileEl.addEventListener("click", () => {
                    treeEl.querySelectorAll(".fbt-be-tree-file-cur")
                          .forEach(e => e.classList.remove("fbt-be-tree-file-cur"));
                    fileEl.classList.add("fbt-be-tree-file-cur");
                    onSelect(path, _activeDir);
                });

                el.appendChild(fileEl);
            });

        return el;
    };

    const _rebuildTree = () => {
        treeEl.innerHTML = "";
        const raw   = _activeDir === "output" ? _outputFiles : _inputFiles;
        const files = filter ? raw.filter(filter) : raw;

        if (!files.length) {
            treeEl.appendChild(_mk("div", {
                cls:         "fbt-be-media-empty",
                textContent: emptyText.replace("{dir}", _activeDir),
            }));
            return;
        }

        const root = { dirs: new Map(), files: [] };
        files.forEach(f => _insertPath(root, f.split("/"), f));
        treeEl.appendChild(_renderNode(root, 0));
    };

    // ── Tabs ──────────────────────────────────────────────────────────────────
    const tabRow  = _mk("div", { cls: "fbt-be-tree-tab-row" });
    const tabBtns = {};

    ["input", "output"].forEach(dir => {
        const btn = _mk("button", {
            cls:         "fbt-be-tree-tab" + (dir === _activeDir ? " active" : ""),
            textContent: dir === "input" ? "Input" : "Output",
        });
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

    // ── Wrapper ───────────────────────────────────────────────────────────────
    const wrap = _mk("div", { cls: "fbt-be-file-tree-wrap" });
    wrap.appendChild(tabRow);
    wrap.appendChild(treeEl);

    return {
        el: wrap,

        /** Refresh file lists and re-render. Pass only the lists that changed. */
        rebuild({ inputFiles: newIn, outputFiles: newOut } = {}) {
            if (newIn  !== undefined) _inputFiles  = newIn;
            if (newOut !== undefined) _outputFiles = newOut;
            _rebuildTree();
        },

        /** Sync the selected highlight without rebuilding the full tree. */
        syncSelected() {
            treeEl.querySelectorAll("[data-tree-path]").forEach(el => {
                const path = el.dataset.treePath;
                const dir  = el.dataset.treeDir;
                el.classList.toggle("fbt-be-tree-file-cur", isSelected(path, dir));
            });
        },

        getDir: () => _activeDir,
    };
}
