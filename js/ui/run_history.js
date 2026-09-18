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
 * A tracked node's input can be a connection reference ([node_id, output_slot])
 * rather than an inline value — e.g. a string bound in from another node instead
 * of typed directly into the widget. extractWidgetValues() resolves these by
 * walking backward through nodesDict (already fully available here — no extra
 * fetch, no graph-wide traversal) through any chain of simple passthrough nodes
 * (Primitive/Reroute/Set-Get) until it lands on a literal value. This only
 * fails when the upstream node genuinely *computes* its output from other
 * inputs — that requires actually running the graph, which is what
 * RunMetaCapture is for; everything else resolves statically.
 *
 * Subgraph nodes never appear in the API-format prompt[2] (nodesDict) because
 * the frontend expands them into their inner nodes before submission.  To
 * capture subgraph-exposed widgets the tracker also scans the workflow JSON
 * stored in extra_pnginfo.workflow, where subgraph container nodes appear
 * with their widget values intact.
 */

const TRACK_RE = /\[track:\s*([^\]]*)\]/;
const LORA_BUILDER_TYPE = "fbt_LoraStackBuilder";
const SCENE_CAST_BUILD_TYPE = "fbt_SceneCastBuild";
const LORA_BUILDER_ROWS = 8;

function getTrackLabel(title) {
    const m = (title || "").match(TRACK_RE);
    return m ? m[1].trim() : null;
}

// Wired connections are [node_id, output_slot] 2-element arrays.
function isConnectionRef(value) {
    return Array.isArray(value) && value.length === 2 && typeof value[1] === "number";
}

// Node types whose output is a direct, unmodified passthrough of one of their
// own inputs — Primitives, Reroutes, Set/Get. For these (and only these) a
// connection reference can be resolved statically: the upstream node's output
// slot *is* one of its own input values, not the result of running code.
const _PASSTHROUGH_TYPES = new Set([
    "PrimitiveNode", "PrimitiveString", "PrimitiveStringMultiline",
    "PrimitiveInt", "PrimitiveFloat", "PrimitiveBoolean",
    "Reroute", "RerouteAdvanced", "SetNode", "GetNode",
]);

/**
 * Resolve a connection-ref input by inspecting the upstream node already
 * present in nodesDict — no execution, no graph-wide traversal, just a
 * bounded walk through simple passthrough nodes (Primitive/Reroute/Set-Get
 * chains). Stops and returns null the moment it hits a node that genuinely
 * computes its output from other inputs, since that can't be inferred
 * without actually running it (that's what RunMetaCapture is for).
 */
function _resolveConnectionRef(value, nodesDict, depth = 4) {
    if (!isConnectionRef(value) || depth <= 0 || !nodesDict) return null;
    const upstream = nodesDict[value[0]];
    if (!upstream) return null;
    const upstreamInputs = upstream.inputs || {};
    const entries = Object.entries(upstreamInputs);
    const literal = entries.filter(([, v]) => !isConnectionRef(v));
    const wired   = entries.filter(([, v]) => isConnectionRef(v));

    // Exactly one literal input and nothing else wired in: unambiguous passthrough
    // regardless of node type (covers custom Primitive-alikes too).
    if (literal.length === 1 && wired.length === 0) {
        return { value: literal[0][1], via: `${upstream.class_type || "?"} #${value[0]}` };
    }
    // Known passthrough type chained from another connection (Reroute -> Reroute -> ...):
    // keep walking one more hop.
    if (wired.length === 1 && literal.length === 0 && _PASSTHROUGH_TYPES.has(upstream.class_type)) {
        return _resolveConnectionRef(wired[0][1], nodesDict, depth - 1);
    }
    return null; // genuinely computed or ambiguous — can't infer statically
}

// A composite/conditional widget (e.g. a V3 DynamicCombo bundling a mode
// selector with a nested option-specific field, like the Latent Upscaler's
// {mode, scale}) can have one of its nested sub-fields converted to an input
// and wired, while the widget's own top-level key still carries a plain
// object value in the submitted prompt — so isConnectionRef() never fires on
// it directly. Left unchecked, that object gets treated as a fully-resolved
// literal, which both (a) shows a stale/partial value in the static entry and
// (b) blocks the runtime capture's fully-resolved version from surviving the
// merge below, since the merge only compares top-level key names. Detect any
// connection-ref hiding at any depth inside the value and treat the whole
// key as statically unresolvable in that case, same as a direct connection
// ref — the runtime capture is the only source of truth for it.
function _hasNestedLink(value, depth = 3) {
    if (depth <= 0) return false;
    if (isConnectionRef(value)) return true;
    if (value && typeof value === "object") {
        return Object.values(value).some(v => _hasNestedLink(v, depth - 1));
    }
    return false;
}

function extractWidgetValues(inputs, nodesDict) {
    const out = {};
    for (const [key, v] of Object.entries(inputs || {})) {
        if (!isConnectionRef(v)) {
            if (_hasNestedLink(v)) continue; // composite widget with a wired sub-field
            out[key] = v;
            continue;
        }
        const resolved = _resolveConnectionRef(v, nodesDict);
        if (resolved) out[key] = `${resolved.value}  (via ${resolved.via})`;
        // else: leave it out — same as before, but now only for values that
        // are genuinely unresolvable without running the graph.
    }
    return out;
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

function parseSceneCastBuildWidgets(inputs) {
    let entries = [];
    const raw = inputs.cast_entries_json;
    if (typeof raw === "string") {
        try { entries = JSON.parse(raw) || []; } catch (_) { /* malformed/empty — show nothing */ }
    }
    const clipId = typeof inputs.clip_id === "string" ? inputs.clip_id : "";
    const mult   = typeof inputs.clip_duration_multiplier === "number" ? inputs.clip_duration_multiplier : null;
    const actionPreview = typeof inputs.action_preview === "string" ? inputs.action_preview.trim() : "";
    return { entries, clipId, mult, actionPreview };
}

function renderSceneCastBuildTable(parsed, container) {
    const { entries, clipId, mult, actionPreview } = parsed;

    const meta = [];
    if (clipId) meta.push(`clip: ${clipId}`);
    if (mult != null && mult !== 1) meta.push(`${mult}x duration`);
    if (meta.length) container.appendChild(txt("span", "fbt-rh-lora-model-badge", meta.join(" · ")));

    if (!entries.length) {
        container.appendChild(txt("div", "fbt-rh-empty-node", "(no cast entries)"));
    } else {
        const table = mk("table", "fbt-rh-table");
        for (const e of entries) {
            const tr = mk("tr");
            const mode = e.visual_mode || "images";
            const audioFlag = e.use_audio ? " +audio" : "";
            tr.appendChild(txt("td", "fbt-rh-key", e.subject_id || "?"));
            tr.appendChild(txt("td", "fbt-rh-val", `${e.bundle_id || "?"} [${mode}${audioFlag}]`));
            table.appendChild(tr);
        }
        container.appendChild(table);
    }

    // Resolved action text — written by the on-node preview widget into its
    // hidden backing widget (action_preview), so it rides along here as an
    // ordinary literal value.
    if (actionPreview) {
        container.appendChild(txt("div", "fbt-rh-scb-action", actionPreview));
    }
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
                widgets: extractWidgetValues(rawInputs, nodesDict),
            });
        }

        // Workflow-format tracked nodes — scan extra_pnginfo.workflow for nodes
        // tagged [track:] that are absent from nodesDict (subgraphs being the
        // primary case).  Widget values come from the positional widgets_values
        // array present in the workflow JSON.
        const capturedLabels = new Set(trackedNodes.map(n => n.label));
        const workflowTracked = extractWorkflowTrackedNodes(wf, capturedLabels);
        trackedNodes.push(...workflowTracked);

        // Runtime captures — from RunMetaCapture nodes, and from the
        // node-output auto-tracker (extension.py) for [track:]-tagged nodes
        // whose value couldn't be resolved statically above.
        const rawCaptures = captureMap[promptId]?.captures ?? [];

        // Merge each capture into its matching static entry: drop any capture
        // key the static scan already resolved (that version is often cleaner
        // — e.g. a passthrough-resolved name vs. a raw dumped object — and
        // showing both is just noise), keep only genuinely new keys the
        // static scan couldn't get, and drop the capture entirely once
        // nothing new remains. LORA_BUILDER_TYPE is exempt from merging —
        // its custom table renders straight from rawInputs and never touches
        // captures — so its runtime capture is pure noise and gets dropped
        // outright, never pushed as its own entry.
        const staticByLabel = new Map(trackedNodes.map(n => [n.label, n]));
        const captures = [];
        for (const cap of rawCaptures) {
            const staticNode = staticByLabel.get(cap.label);
            if (!staticNode) {
                captures.push(cap);
                continue;
            }
            if (staticNode.class_type === LORA_BUILDER_TYPE) {
                continue;
            }
            const staticKeys = new Set(Object.keys(staticNode.widgets || {}));
            const extra = {};
            for (const [k, v] of Object.entries(cap.values || {})) {
                if (!staticKeys.has(k)) extra[k] = v;
            }
            if (Object.keys(extra).length > 0) {
                captures.push({ ...cap, values: extra, label: `${cap.label} (extra)` });
            }
            // else: fully covered by the static scan already — drop the duplicate.
        }

        // A node the static scan couldn't resolve at all (zero widgets — every
        // input was a wired, genuinely-computed value) but that a capture
        // picked up under the same original label would otherwise show up
        // twice: once as a useless "(no values)" placeholder here, once for
        // real under captures. Drop the empty static entry in that case.
        const captureLabels = new Set(rawCaptures.map(c => c.label));
        const resolvedTrackedNodes = trackedNodes.filter(n =>
            n.class_type === LORA_BUILDER_TYPE ||
            Object.keys(n.widgets || {}).length > 0 ||
            !captureLabels.has(n.label)
        );

        if (!resolvedTrackedNodes.length && !captures.length) continue;

        runs.push({ promptId, queueNum, ts, workflowName, trackedNodes: resolvedTrackedNodes, captures });
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
        } else if (node.class_type === SCENE_CAST_BUILD_TYPE) {
            renderSceneCastBuildTable(parseSceneCastBuildWidgets(node.rawInputs), nodeEl);
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
