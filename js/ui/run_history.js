/**
 * Run History sidebar panel.
 *
 * Merges three data sources per run:
 *  - /history nodesDict: static widget values from nodes tagged [track: Label]
 *  - /history extra_pnginfo.workflow: widget values from subgraph containers tagged [track: Label]
 *  - /fbtools/run_tracker/runs: runtime values captured by RunMetaCapture nodes
 *
 * Runs are keyed by prompt_id (GUID) so both sources can be linked.
 *
 * Subgraph nodes never appear in the API-format prompt[2] (nodesDict) because
 * the frontend expands them into their inner nodes before submission.  When an
 * inner node receives a value from the subgraph's exposed interface, that input
 * becomes a connection reference in the API format and is filtered out by
 * extractWidgetValues().  To capture subgraph-exposed widgets the tracker also
 * scans the workflow JSON stored in extra_pnginfo.workflow, where subgraph
 * container nodes appear with their widget values intact.
 */

const TRACK_RE = /\[track:\s*([^\]]*)\]/;
const LORA_BUILDER_TYPE = "fbt_LoraStackBuilder";
const LORA_BUILDER_ROWS = 8;

function getTrackLabel(title) {
    const m = (title || "").match(TRACK_RE);
    return m ? m[1].trim() : null;
}

// Wired connections are [node_id, output_slot] 2-element arrays.
function isConnectionRef(value) {
    return Array.isArray(value) && value.length === 2 && typeof value[1] === "number";
}

function extractWidgetValues(inputs) {
    return Object.fromEntries(
        Object.entries(inputs || {}).filter(([, v]) => !isConnectionRef(v))
    );
}

/**
 * Scan the workflow JSON (extra_pnginfo.workflow) for nodes tagged [track: Label]
 * that are NOT already captured from nodesDict.  Subgraph container nodes only
 * exist here — the API-format prompt expands them away.
 *
 * Widget values in the workflow format are stored as a positional `widgets_values`
 * array.  We attempt to name them using any `widget.name` metadata on the node's
 * inputs array (present in newer ComfyUI workflow format).  When that isn't
 * available we fall back to:
 *   - single value  → key "value"
 *   - multiple      → keys "value_0", "value_1", …
 *
 * @param {object|null} wf               The workflow JSON object.
 * @param {Set<string>} capturedLabels   Labels already found in nodesDict (skip them).
 * @returns {Array}                      Array of {label, class_type, rawInputs, widgets}.
 */
function extractWorkflowTrackedNodes(wf, capturedLabels) {
    const result = [];
    const nodes = wf?.nodes;
    if (!Array.isArray(nodes)) return result;

    for (const node of nodes) {
        const label = getTrackLabel(node.title || "");
        if (!label) continue;
        if (capturedLabels.has(label)) continue; // API-format version is better

        const widgetValues = node.widgets_values || [];
        if (!widgetValues.length) continue;

        // Newer ComfyUI workflow format: inputs entries that have a `widget` key
        // are widget inputs (not wire connections), with a name we can reuse.
        const widgetInputs = (node.inputs || []).filter(inp => inp.widget);

        const widgets = {};
        if (widgetInputs.length === widgetValues.length && widgetInputs.length > 0) {
            widgetInputs.forEach((inp, i) => {
                const key = inp.widget?.name || inp.name || `value_${i}`;
                widgets[key] = widgetValues[i];
            });
        } else if (widgetValues.length === 1) {
            widgets["value"] = widgetValues[0];
        } else {
            widgetValues.forEach((v, i) => { widgets[`value_${i}`] = v; });
        }

        if (Object.keys(widgets).length > 0) {
            result.push({
                label,
                class_type: node.type || "",
                rawInputs: {},
                widgets,
                fromWorkflow: true, // diagnostic — not shown in UI
            });
        }
    }
    return result;
}

// Mirror the backend's _fv(): 2 decimal places, trailing zeros stripped.
function fmtStrength(n) {
    return parseFloat((+n).toFixed(2)).toString();
}

function parseLoraBuilderWidgets(inputs) {
    const modelTarget = inputs.model_target ?? "unknown";
    const isLtx = modelTarget === "LTX2.3";
    const loras = [];

    for (let i = 0; i < LORA_BUILDER_ROWS; i++) {
        const lora = inputs[`lora_${i}`];
        // Skip: not set, wired connection ref, or explicitly None
        if (!lora || isConnectionRef(lora) || lora === "None") continue;
        // Skip disabled rows (undefined means enabled)
        if (inputs[`enabled_${i}`] === false) continue;

        const name = lora.replace(/\.[^.]+$/, "").split(/[\\/]/).pop().slice(0, 44);
        const entry = {
            name,
            sm: fmtStrength(inputs[`strength_model_${i}`] ?? 1),
            sc: fmtStrength(inputs[`strength_clip_${i}`]  ?? 1),
        };
        if (isLtx) {
            entry.video = fmtStrength(inputs[`video_${i}`] ?? 1);
            entry.audio = fmtStrength(inputs[`audio_${i}`] ?? 1);
        }
        loras.push(entry);
    }

    return { modelTarget, loras };
}

function renderLoraBuilderTable(parsed, container) {
    const { modelTarget, loras } = parsed;

    const badge = txt("span", "fbt-rh-lora-model-badge", modelTarget);
    container.appendChild(badge);

    if (!loras.length) {
        container.appendChild(txt("div", "fbt-rh-empty-node", "(no enabled LoRAs)"));
        return;
    }

    const hasLtx = loras.some(e => "video" in e);
    const table = mk("table", "fbt-rh-table");

    // Header: M / C  [/ V / A for LTX]
    const thead = mk("tr");
    thead.appendChild(mk("td", "fbt-rh-key")); // blank name column
    const hdr = txt("td", "fbt-rh-lora-hdr", hasLtx ? "M / C / V / A" : "M / C");
    thead.appendChild(hdr);
    table.appendChild(thead);

    for (const { name, sm, sc, video, audio } of loras) {
        const tr = mk("tr");
        tr.appendChild(txt("td", "fbt-rh-key", name));
        const parts = [sm, sc];
        if (video !== undefined) parts.push(video, audio);
        tr.appendChild(txt("td", "fbt-rh-val fbt-rh-lora-strengths", parts.join(" / ")));
        table.appendChild(tr);
    }
    container.appendChild(table);
}

// ComfyUI timestamps in execution_start messages are in milliseconds.
// Guard against both ms (>1e12) and s (<1e12) just in case.
function toMs(ts) {
    if (!ts) return null;
    return ts > 1e12 ? ts : ts * 1000;
}

function formatTs(ts) {
    const ms = toMs(ts);
    if (!ms) return "";
    try { return new Date(ms).toLocaleString(); } catch { return ""; }
}

function shortId(guid) {
    // Show last 8 chars of the GUID so it's recognisable but compact
    return guid ? guid.slice(-8) : "";
}

async function fetchHistory(maxItems) {
    const res = await fetch(`/history?max_items=${maxItems}`);
    if (!res.ok) throw new Error(`/history returned ${res.status}`);
    return res.json();
}

async function fetchCaptures() {
    try {
        const res = await fetch("/fbtools/run_tracker/runs");
        if (!res.ok) return {};
        const data = await res.json();
        const map = {};
        for (const run of data.runs || []) {
            map[run.prompt_id] = run;
        }
        return map;
    } catch {
        return {};
    }
}

function parseRuns(historyData, captureMap) {
    const runs = [];

    for (const [promptId, run] of Object.entries(historyData)) {
        const prompt = run.prompt;
        if (!Array.isArray(prompt) || prompt.length < 3) continue;

        const queueNum = prompt[0];
        const nodesDict = prompt[2];
        const extra = prompt[3] || {};
        const wf = extra.extra_pnginfo?.workflow;
        const workflowName = wf?.title || wf?.name || null;

        const msgs = run.status?.messages ?? [];
        const startMsg = msgs.find(m => Array.isArray(m) && m[0] === "execution_start");
        const ts = startMsg?.[1]?.timestamp ?? null;

        // Static tracked nodes — scan the API-format prompt (nodesDict).
        // Regular nodes appear here with named widget values.
        // Subgraph container nodes are ABSENT — they expand into inner nodes
        // before submission; inner nodes receive exposed inputs as connection
        // references which extractWidgetValues() filters out.
        const trackedNodes = [];
        for (const [nodeId, nodeDef] of Object.entries(nodesDict || {})) {
            const title = nodeDef?._meta?.title || "";
            const label = getTrackLabel(title);
            if (!label) continue;
            const rawInputs = nodeDef.inputs || {};
            trackedNodes.push({
                label,
                class_type: nodeDef.class_type || "",
                rawInputs,
                widgets: extractWidgetValues(rawInputs),
            });
        }

        // Workflow-format tracked nodes — scan extra_pnginfo.workflow for nodes
        // tagged [track:] that are absent from nodesDict (subgraphs being the
        // primary case).  Widget values come from the positional widgets_values
        // array present in the workflow JSON.
        const capturedLabels = new Set(trackedNodes.map(n => n.label));
        const workflowTracked = extractWorkflowTrackedNodes(wf, capturedLabels);
        trackedNodes.push(...workflowTracked);

        // Runtime captures from RunMetaCapture nodes
        const captures = captureMap[promptId]?.captures ?? [];

        if (!trackedNodes.length && !captures.length) continue;

        runs.push({ promptId, queueNum, ts, workflowName, trackedNodes, captures });
    }

    runs.sort((a, b) => b.queueNum - a.queueNum);
    return runs;
}

// ── DOM helpers ────────────────────────────────────────────────────────────────

function mk(tag, cls) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    return e;
}

function txt(tag, cls, text) {
    const e = mk(tag, cls);
    e.textContent = text;
    return e;
}

function renderValueTable(values, container) {
    const entries = Object.entries(values);
    if (!entries.length) {
        container.appendChild(txt("div", "fbt-rh-empty-node", "(no values)"));
        return;
    }
    const table = mk("table", "fbt-rh-table");
    for (const [key, value] of entries) {
        const tr = mk("tr");
        tr.appendChild(txt("td", "fbt-rh-key", key));
        const td = mk("td", "fbt-rh-val");
        td.textContent = typeof value === "object" ? JSON.stringify(value) : String(value);
        tr.appendChild(td);
        table.appendChild(tr);
    }
    container.appendChild(table);
}

function renderRun(run) {
    const section = mk("div", "fbt-rh-run");

    // ── Header ──────────────────────────────────────────────────────────────
    const header = mk("div", "fbt-rh-run-header");

    const numEl = txt("span", "fbt-rh-run-num", `#${run.queueNum}`);
    header.appendChild(numEl);

    if (run.ts) {
        header.appendChild(txt("span", "fbt-rh-run-ts", formatTs(run.ts)));
    }

    // Workflow name (if known)
    if (run.workflowName) {
        header.appendChild(txt("span", "fbt-rh-run-wf", run.workflowName));
    }

    // Prompt ID (last 8 chars as a dim chip)
    const idChip = txt("span", "fbt-rh-run-id", shortId(run.promptId));
    idChip.title = run.promptId;
    header.appendChild(idChip);

    const chevron = txt("span", "fbt-rh-chevron", "▾");
    header.appendChild(chevron);

    // ── Body ────────────────────────────────────────────────────────────────
    const body = mk("div", "fbt-rh-run-body");

    let expanded = true;
    header.addEventListener("click", () => {
        expanded = !expanded;
        body.style.display = expanded ? "" : "none";
        chevron.textContent = expanded ? "▾" : "▸";
    });

    // Static tracked nodes
    for (const node of run.trackedNodes) {
        const nodeEl = mk("div", "fbt-rh-node");
        nodeEl.appendChild(txt("div", "fbt-rh-node-label", node.label));
        if (node.class_type === LORA_BUILDER_TYPE) {
            renderLoraBuilderTable(parseLoraBuilderWidgets(node.rawInputs), nodeEl);
        } else {
            renderValueTable(node.widgets, nodeEl);
        }
        body.appendChild(nodeEl);
    }

    // Runtime captures from RunMetaCapture
    for (const cap of run.captures) {
        const nodeEl = mk("div", "fbt-rh-node fbt-rh-node-captured");
        const labelEl = txt("div", "fbt-rh-node-label fbt-rh-node-label-captured", cap.label);
        nodeEl.appendChild(labelEl);
        renderValueTable(cap.values, nodeEl);
        body.appendChild(nodeEl);
    }

    section.appendChild(header);
    section.appendChild(body);
    return section;
}

// ── Panel ──────────────────────────────────────────────────────────────────────

export function renderRunHistory(rootEl) {
    rootEl.innerHTML = "";
    rootEl.classList.add("fbt-rh-panel");

    const toolbar = mk("div", "fbt-rh-toolbar");
    toolbar.appendChild(txt("span", "fbt-rh-panel-title", "Run History"));

    const controls = mk("div", "fbt-rh-controls");

    const maxInput = mk("input", "fbt-rh-max-input");
    maxInput.type = "number";
    maxInput.value = "25";
    maxInput.min = "1";
    maxInput.max = "200";
    maxInput.title = "Max runs to load";
    controls.appendChild(maxInput);

    const refreshBtn = txt("button", "fbt-ce-btn fbt-rh-refresh-btn", "Refresh");
    controls.appendChild(refreshBtn);
    toolbar.appendChild(controls);
    rootEl.appendChild(toolbar);

    const content = mk("div", "fbt-rh-content");
    rootEl.appendChild(content);

    async function refresh() {
        content.innerHTML = "";
        content.appendChild(txt("div", "fbt-rh-loading", "Loading…"));

        try {
            const [historyData, captureMap] = await Promise.all([
                fetchHistory(Number(maxInput.value) || 25),
                fetchCaptures(),
            ]);
            const runs = parseRuns(historyData, captureMap);
            content.innerHTML = "";

            if (!runs.length) {
                const empty = mk("div", "fbt-rh-empty");
                empty.textContent =
                    "No runs with tracked nodes found.\n" +
                    "Right-click any node → \"Track this node…\"\n" +
                    "or add a Run Meta Capture node to begin.";
                content.appendChild(empty);
                return;
            }

            for (const run of runs) content.appendChild(renderRun(run));
        } catch (err) {
            content.innerHTML = "";
            content.appendChild(txt("div", "fbt-rh-error", `Error: ${err.message}`));
        }
    }

    refreshBtn.addEventListener("click", refresh);
    refresh();
}
