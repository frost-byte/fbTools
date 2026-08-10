/**
 * LoRA node frontend extensions.
 *
 * Adds a "Civitai Info" button to LoraEntryDefine nodes that fetches and
 * displays model metadata (trained words, base model, description) from civitai.
 */

import { loraAPI } from "../api/lora.js";
import { sceneAPI } from "../api/scene.js";
import { setWidgetVisible } from "../utils/widgets.js";

// ── Modal ─────────────────────────────────────────────────────────────────────

function stripHtml(html) {
    const tmp = document.createElement("div");
    tmp.innerHTML = html;
    return tmp.textContent || tmp.innerText || "";
}

export function showCivitaiModal(data) {
    document.getElementById("fbt-civitai-modal")?.remove();

    const overlay = document.createElement("div");
    overlay.id = "fbt-civitai-modal";
    overlay.style.cssText = "position:fixed;inset:0;z-index:9999;background:rgba(0,0,0,0.72);display:flex;align-items:center;justify-content:center;";

    const panel = document.createElement("div");
    panel.style.cssText = "background:var(--comfy-menu-bg);border:1px solid var(--border-color);border-radius:8px;padding:20px;max-width:580px;width:90%;max-height:85vh;overflow-y:auto;color:var(--input-text);font-size:13px;line-height:1.5;";

    const modelName   = data.model?.name || "Unknown model";
    const versionName = data.name || "";
    const baseModel   = data.baseModel || "Unknown";
    const civitaiUrl  = `https://civitai.com/models/${data.modelId}`;
    const trainedWords = Array.isArray(data.trainedWords) && data.trainedWords.length
        ? data.trainedWords.join(", ") : "None listed";
    const rawDesc  = data.description ? stripHtml(data.description) : "";
    const shortDesc = rawDesc.length > 400 ? rawDesc.substring(0, 400) + "…" : rawDesc || "No description";

    // Image gallery — up to 6 examples with hover-prompt overlay
    const images = Array.isArray(data.images) ? data.images.slice(0, 6) : [];
    const galleryHtml = images.length ? `<div class="fbt-civ-gallery">${
        images.map(img => {
            const prompt = img.meta?.prompt || "";
            const imgUrl = img.id ? `https://civitai.com/images/${img.id}` : civitaiUrl;
            const esc = s => s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
            return `<div class="fbt-civ-img-wrap">
                <img src="${img.url}" alt="" loading="lazy">
                ${prompt ? `<div class="fbt-civ-img-overlay">
                    <p class="fbt-civ-prompt">${esc(prompt.length > 280 ? prompt.slice(0, 280) + "…" : prompt)}</p>
                    <a href="${imgUrl}" target="_blank" class="fbt-civ-img-link">View on Civitai ↗</a>
                </div>` : ""}
            </div>`;
        }).join("")
    }</div>` : "";

    panel.innerHTML = `
        <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px;">
            <div>
                <div style="font-weight:700;font-size:1.05em;">${modelName}</div>
                ${versionName ? `<div style="opacity:0.6;font-size:0.9em;">${versionName}</div>` : ""}
            </div>
            <button id="fbt-civitai-close" style="background:none;border:none;color:var(--input-text);cursor:pointer;font-size:1.3em;padding:0 0 0 12px;flex-shrink:0;">✕</button>
        </div>
        <table style="width:100%;border-collapse:collapse;margin-bottom:14px;">
            <tr>
                <td style="padding:4px 10px 4px 0;opacity:0.6;white-space:nowrap;vertical-align:top;">Base model</td>
                <td style="padding:4px 0;">${baseModel}</td>
            </tr>
            <tr>
                <td style="padding:4px 10px 4px 0;opacity:0.6;white-space:nowrap;vertical-align:top;">Trigger words</td>
                <td style="padding:4px 0;word-break:break-word;">${trainedWords}</td>
            </tr>
            <tr>
                <td style="padding:4px 10px 4px 0;opacity:0.6;white-space:nowrap;vertical-align:top;">Description</td>
                <td style="padding:4px 0;">${shortDesc}</td>
            </tr>
        </table>
        <div style="margin-bottom:14px;">
            <a href="${civitaiUrl}" target="_blank" style="color:var(--p-blue-400,#60a5fa);text-decoration:none;">View on Civitai ↗</a>
        </div>
        ${galleryHtml}
    `;

    if (!document.getElementById("fbt-civ-gallery-styles")) {
        const s = document.createElement("style");
        s.id = "fbt-civ-gallery-styles";
        s.textContent = `
            .fbt-civ-gallery{display:grid;grid-template-columns:repeat(2,1fr);gap:8px}
            .fbt-civ-img-wrap{position:relative;overflow:hidden;border-radius:5px;background:#111;cursor:default}
            .fbt-civ-img-wrap img{width:100%;display:block;max-height:200px;object-fit:cover;transition:opacity .18s}
            .fbt-civ-img-overlay{position:absolute;bottom:0;left:0;right:0;background:rgba(0,0,0,0.85);padding:8px 10px;transform:translateY(100%);transition:transform .18s}
            .fbt-civ-img-wrap:hover .fbt-civ-img-overlay{transform:translateY(0)}
            .fbt-civ-img-wrap:hover img{opacity:.55}
            .fbt-civ-prompt{margin:0 0 6px;font-size:11px;line-height:1.4;color:#dde;max-height:90px;overflow-y:auto}
            .fbt-civ-img-link{font-size:11px;color:#60a5fa;text-decoration:none}
            .fbt-civ-img-link:hover{text-decoration:underline}`;
        document.head.appendChild(s);
    }

    overlay.appendChild(panel);
    document.body.appendChild(overlay);
    overlay.addEventListener("click", (e) => { if (e.target === overlay) overlay.remove(); });
    panel.querySelector("#fbt-civitai-close").addEventListener("click", () => overlay.remove());
}

// ── Node handler: LoraPresetSelect / WanPresetSelect ──────────────────────────

function setupDynamicPresetCombo(nodeType) {
    // Shared logic: update "selected_preset" combo options from ui.preset_names
    // sent back by the execute() method of any *PresetSelect node.
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
        if (onExecuted) onExecuted.apply(this, arguments);

        const names = message?.ui?.preset_names || message?.preset_names;
        if (!Array.isArray(names) || !names.length) return;

        const widget = this.widgets?.find(w => w.name === "selected_preset");
        if (!widget) return;

        widget.options.values = names;

        // Keep current selection if still valid; otherwise default to first
        if (!names.includes(widget.value)) {
            widget.value = names[0];
        }
    };
}

export function setupLoraPresetSelect(nodeType, nodeData, app) {
    setupDynamicPresetCombo(nodeType);
}

export function setupWanPresetSelect(nodeType, nodeData, app) {
    setupDynamicPresetCombo(nodeType);
}

// ── Node handler: LoraPresetDefine / WanPresetDefine ──────────────────────────

async function populateSceneCombo(node) {
    try {
        const data = await sceneAPI.list();
        const scenes = data?.scenes ?? [];
        const widget = node.widgets?.find(w => w.name === "scene_name");
        if (!widget) return;
        const options = ["none", ...scenes];
        widget.options.values = options;
        if (!options.includes(widget.value)) widget.value = "none";
    } catch (e) {
        console.warn("[fbTools] PresetDefine: could not load scene list", e);
    }
}

function setupPresetDefineSceneCombo(nodeType) {
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        if (onNodeCreated) onNodeCreated.apply(this, arguments);
        populateSceneCombo(this);
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (data) {
        if (onConfigure) onConfigure.apply(this, arguments);
        // Re-populate after load so the saved scene name is a valid option
        populateSceneCombo(this);
    };
}

export function setupLoraPresetDefine(nodeType, nodeData, app) {
    setupPresetDefineSceneCombo(nodeType);
}

export function setupWanPresetDefine(nodeType, nodeData, app) {
    setupPresetDefineSceneCombo(nodeType);
}

// ── Node handler: LoraEntryDefine ─────────────────────────────────────────────

const LTX_WIDGET_NAMES = ["video_strength", "audio_strength"];
const LORA_BUILDER_ROWS = 8;

function applyLoraEntryTarget(node, target) {
    const isLtx = target === "LTX2.3";
    setWidgetVisible(node._ltxToggleBtn, isLtx);
    const ltxWidgets = LTX_WIDGET_NAMES.map(n => node.widgets?.find(w => w.name === n)).filter(Boolean);
    for (const w of ltxWidgets) setWidgetVisible(w, isLtx && !!node._ltxAccordionOpen);
    node.setSize(node.computeSize());
    node.setDirtyCanvas(true, true);
}

export function setupLoraEntryDefine(nodeType, nodeData, app) {
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        if (onNodeCreated) onNodeCreated.apply(this, arguments);

        const self = this;
        self._ltxAccordionOpen = false;

        // ── LTX2.3 accordion toggle ──
        const toggleBtn = this.addWidget("button", "▶ LTX2.3 Layer Weights", null, function () {
            self._ltxAccordionOpen = !self._ltxAccordionOpen;
            toggleBtn.name = self._ltxAccordionOpen
                ? "▼ LTX2.3 Layer Weights"
                : "▶ LTX2.3 Layer Weights";
            const ltxWidgets = LTX_WIDGET_NAMES
                .map(n => self.widgets?.find(w => w.name === n))
                .filter(Boolean);
            for (const w of ltxWidgets) setWidgetVisible(w, self._ltxAccordionOpen);
            self.setSize(self.computeSize());
            self.setDirtyCanvas(true, true);
        });
        self._ltxToggleBtn = toggleBtn;

        // Move toggle button to sit between "enabled" and "video_strength"
        const enabledIdx = this.widgets.findIndex(w => w.name === "enabled");
        const btnIdx = this.widgets.indexOf(toggleBtn);
        if (enabledIdx >= 0 && btnIdx > enabledIdx + 1) {
            this.widgets.splice(btnIdx, 1);
            this.widgets.splice(enabledIdx + 1, 0, toggleBtn);
        }

        // Watch model_target for changes
        const modelTargetWidget = this.widgets?.find(w => w.name === "model_target");
        if (modelTargetWidget) {
            const origCallback = modelTargetWidget.callback;
            modelTargetWidget.callback = function (value) {
                if (origCallback) origCallback.apply(this, arguments);
                applyLoraEntryTarget(self, value);
            };
            applyLoraEntryTarget(self, modelTargetWidget.value);
        }

        // ── Civitai Info button ──
        this.addWidget("button", "ℹ Civitai Info", null, async function () {
            const loraWidget = self.widgets?.find(w => w.name === "lora");
            const loraName   = loraWidget?.value;

            if (!loraName || loraName === "None") {
                app.extensionManager?.toast?.add({
                    severity: "warn",
                    summary: "No LoRA selected",
                    detail: "Choose a LoRA file before fetching info.",
                    life: 3000,
                });
                return;
            }

            app.extensionManager?.toast?.add({
                severity: "info",
                summary: "Fetching Civitai info…",
                detail: loraName,
                life: 2000,
            });

            try {
                const data = await loraAPI.getCivitaiInfo(loraName);
                showCivitaiModal(data);
            } catch (err) {
                const notFound = err.statusCode === 404;
                app.extensionManager?.toast?.add({
                    severity: "error",
                    summary: "Civitai lookup failed",
                    detail: notFound
                        ? "This LoRA was not found on Civitai."
                        : (err.message || "Request failed"),
                    life: 5000,
                });
            }
        });
    };

    // Re-apply visibility after a saved graph restores widget values
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (data) {
        if (onConfigure) onConfigure.apply(this, arguments);
        const modelTargetWidget = this.widgets?.find(w => w.name === "model_target");
        if (modelTargetWidget) applyLoraEntryTarget(this, modelTargetWidget.value);
    };
}

// ── Node handler: LoraStackBuilder — compact canvas-drawn row UI ─────────────
//
// Each LoRA slot is drawn as a single canvas row:
//   [toggle] [LoRA name / trigger] [◄ Model ►] [◄ CLIP ►] [ⓘ]
// LTX2.3 (model_target="ltx23") renders 3 spinners per row: Model, Vid, Aud.
// All other targets render 2 spinners: Model, Clip.
// The ⓘ icon opens the Civitai info modal (image gallery with hover prompts).
//
// Backend widgets for each slot are hidden; JS reads/writes their values directly.

const LSB_NODE_NAME = "fbt_LoraStackBuilder";
const LSB_WPREFIX = "fbt_lsb_";
const LSB_MAX_SLOTS = 8;
const LSB_MIN_WIDTH = 460;
const LSB_NONE = "None";
const LSB_INSET = 15;
const LSB_INNER_M = 3.3;
const LSB_NUM_GAP = 3 + LSB_INNER_M * 2;
const LSB_ICON_SIZE = 18;
const LSB_ICON_RIGHT_PAD = 18;

let _lsbCachedLoraOptions = null;
let _lsbLoraOptionsPromise = null;
let _lsbLastCtxMenuEvent = null;

// ── Geometry ──────────────────────────────────────────────────────────────────

function _lsbRowH() { return (typeof LiteGraph !== "undefined" ? LiteGraph.NODE_WIDGET_HEIGHT : 20); }
function _lsbNumTotalW() { return 9 + 3 + 32 + 3 + 9; }
function _lsbIconColW() { return LSB_ICON_SIZE + LSB_ICON_RIGHT_PAD + LSB_INNER_M * 2; }
function _lsbRowRight(node) { return node.size[0] - LSB_INSET; }
function _lsbBaseNumRight(node) { return _lsbRowRight(node) - LSB_INNER_M - _lsbIconColW(); }
function _lsbNumRight(node, fromRight) { return _lsbBaseNumRight(node) - fromRight * (_lsbNumTotalW() + LSB_NUM_GAP); }
function _lsbNumLabelCX(node, fromRight) { return _lsbNumRight(node, fromRight) - _lsbNumTotalW() / 2; }
function _lsbIconX(node) { return _lsbRowRight(node) - LSB_ICON_RIGHT_PAD - LSB_ICON_SIZE; }
function _lsbIsLowQ() { return ((typeof app !== "undefined" && app.canvas?.ds?.scale) || 1) <= 0.5; }

// ── Widget / value access ─────────────────────────────────────────────────────

function _lsbGW(node, name) { return (node.widgets || []).find(w => w.name === name); }
function _lsbGV(node, key, fb) { const w = _lsbGW(node, key); return w !== undefined ? w.value : fb; }
function _lsbSV(node, key, value) {
    const w = _lsbGW(node, key);
    if (!w || w.value === value) return;
    if (key.startsWith("lora_")) _lsbPreserveLora(w, value);
    w.value = value;
}
function _lsbIsLtx(node) { return _lsbGV(node, "model_target", "") === "LTX2.3"; }
function _lsbAllEnabled(node) {
    for (let i = 0; i < LSB_MAX_SLOTS; i++) { if (!_lsbGV(node, `enabled_${i}`, true)) return false; }
    return true;
}

// ── LoRA option helpers ────────────────────────────────────────────────────────

function _lsbIsRealLora(v) { return Boolean(String(v ?? "").trim()) && String(v) !== LSB_NONE; }

function _lsbPreserveLora(widget, value) {
    if (!widget || !_lsbIsRealLora(value)) return;
    widget.options = widget.options || {};
    const base = widget.options.values || widget.options.list || widget.values || [LSB_NONE];
    const list = [...(Array.isArray(base) ? base : [LSB_NONE])];
    if (!list.includes(LSB_NONE)) list.unshift(LSB_NONE);
    if (!list.includes(value)) list.splice(Math.max(1, list.indexOf(LSB_NONE) + 1), 0, value);
    widget.options.values = list;
    widget.options.list = list;
    widget.values = list;
}

function _lsbMergeLoraList(values, current) {
    const list = [...(Array.isArray(values) ? values : [LSB_NONE])];
    if (!list.includes(LSB_NONE)) list.unshift(LSB_NONE);
    const curr = Array.isArray(current) ? current : [current];
    for (const v of curr) {
        if (_lsbIsRealLora(v) && !list.includes(v)) list.splice(Math.max(1, list.indexOf(LSB_NONE) + 1), 0, v);
    }
    return list;
}

function _lsbCurrentLoraValues(node) {
    const vals = [];
    for (let i = 0; i < LSB_MAX_SLOTS; i++) {
        const v = _lsbGW(node, `lora_${i}`)?.value;
        if (_lsbIsRealLora(v) && !vals.includes(v)) vals.push(v);
    }
    return vals;
}

function _lsbLoraOptionsSync(node) {
    if (Array.isArray(_lsbCachedLoraOptions) && _lsbCachedLoraOptions.length) {
        return _lsbMergeLoraList(_lsbCachedLoraOptions, _lsbCurrentLoraValues(node));
    }
    const w = _lsbGW(node, "lora_0");
    const raw = w?.options?.values || w?.options?.list || w?.values || [LSB_NONE];
    return _lsbMergeLoraList(Array.isArray(raw) ? raw : [LSB_NONE], _lsbCurrentLoraValues(node));
}

async function _lsbLoraOptions(node) {
    if (_lsbLoraOptionsPromise) return _lsbLoraOptionsPromise;
    _lsbLoraOptionsPromise = _lsbFetchLoraOptions(node).finally(() => { _lsbLoraOptionsPromise = null; });
    return _lsbLoraOptionsPromise;
}

async function _lsbFetchLoraOptions(node) {
    try {
        const resp = await api.fetchApi(`/object_info/${LSB_NODE_NAME}`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const info = await resp.json();
        const nodeInfo = info?.[LSB_NODE_NAME] || info;
        const raw = nodeInfo?.input?.required?.lora_0?.[0] ?? nodeInfo?.input?.optional?.lora_0?.[0];
        const values = Array.isArray(raw) ? raw : [LSB_NONE];
        const withNone = values.includes(LSB_NONE) ? values : [LSB_NONE, ...values];
        _lsbCachedLoraOptions = withNone;
        for (let i = 0; i < LSB_MAX_SLOTS; i++) {
            const w = _lsbGW(node, `lora_${i}`);
            if (!w) continue;
            const next = _lsbMergeLoraList(withNone, [_lsbGV(node, `lora_${i}`, LSB_NONE)]);
            w.options = w.options || {};
            w.options.values = next;
            w.options.list = next;
            w.values = next;
        }
        return _lsbMergeLoraList(withNone, _lsbCurrentLoraValues(node));
    } catch (e) {
        console.warn("[fbTools LSB] Failed to refresh LoRA list.", e);
    }
    return _lsbLoraOptionsSync(node);
}

// ── Hide backend widgets ──────────────────────────────────────────────────────

function _lsbHideWidget(widget) {
    if (!widget) return;
    widget.hidden = true;
    widget.type = "converted-widget";
    widget.computeSize = () => [0, -4];
    if (widget.element) widget.element.style.display = "none";
}

function _lsbHideSlotWidgets(node) {
    const prefixes = ["lora", "strength_model", "strength_clip", "enabled", "video", "audio"];
    for (let i = 0; i < LSB_MAX_SLOTS; i++) {
        for (const p of prefixes) _lsbHideWidget(_lsbGW(node, `${p}_${i}`));
    }
}

// ── Active-row tracking ───────────────────────────────────────────────────────

function _lsbActiveRows(node) {
    const saved = node.properties?.fbt_lsb_rows;
    if (saved != null) return Math.min(Math.max(1, saved), LSB_MAX_SLOTS);
    // Legacy/fresh node: count filled slots, show at least 1
    let lastFilled = 0;
    for (let i = 0; i < LSB_MAX_SLOTS; i++) {
        const v = _lsbGV(node, `lora_${i}`, LSB_NONE);
        if (v && v !== LSB_NONE) lastFilled = i + 1;
    }
    return Math.max(1, lastFilled);
}

// ── Rebuild UI ────────────────────────────────────────────────────────────────

function _lsbRemoveGenerated(node) {
    node.widgets = (node.widgets || []).filter(w => {
        if (!String(w.name || "").startsWith(LSB_WPREFIX)) return true;
        if (w.element?.parentNode) w.element.parentNode.removeChild(w.element);
        else w.element?.remove?.();
        return false;
    });
}

function _lsbRebuildUi(node) {
    _lsbRemoveGenerated(node);
    _lsbHideSlotWidgets(node);
    const active = _lsbActiveRows(node);
    node.addCustomWidget(new _LsbDivider());
    node.addCustomWidget(new _LsbHeader());
    for (let i = 0; i < active; i++) node.addCustomWidget(new _LsbRow(i));
    node.addCustomWidget(new _LsbAddButton());
    const computed = node.computeSize?.() || [LSB_MIN_WIDTH, 120];
    node.size = node.size || [LSB_MIN_WIDTH, computed[1]];
    node.size[0] = Math.max(node.size[0], LSB_MIN_WIDTH);
    node.size[1] = Math.max(computed[1], 90);
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
}

function _lsbSetupNode(node) {
    if (!node || node.__fbtLsbSettingUp) return;
    node.__fbtLsbSettingUp = true;
    try {
        node.serialize_widgets = true;
        if (!node.__fbtLsbSizeWrapped) {
            const orig = node.computeSize;
            node.computeSize = function () {
                const sz = orig?.apply(this, arguments) || [LSB_MIN_WIDTH, 120];
                return [Math.max(sz[0], LSB_MIN_WIDTH), sz[1]];
            };
            node.__fbtLsbSizeWrapped = true;
        }
        if (!window.__fbtLsbCtxTrackerInstalled) {
            const track = e => { if (e.button === 2 || e.which === 3) _lsbLastCtxMenuEvent = e; };
            window.addEventListener("pointerdown", track, true);
            window.addEventListener("contextmenu", e => { _lsbLastCtxMenuEvent = e; }, true);
            window.__fbtLsbCtxTrackerInstalled = true;
        }
        // Redraw when model_target changes (column layout switches between LTX and non-LTX)
        const mtWidget = _lsbGW(node, "model_target");
        if (mtWidget && !mtWidget.__fbtLsbWatched) {
            const origCb = mtWidget.callback;
            mtWidget.callback = function (value) {
                origCb?.apply(this, arguments);
                node.setSize(node.computeSize());
                _lsbRedraw(node);
            };
            mtWidget.__fbtLsbWatched = true;
        }
        _lsbRebuildUi(node);
    } finally {
        node.__fbtLsbSettingUp = false;
    }
}

// ── Redraw ────────────────────────────────────────────────────────────────────

function _lsbRedraw(node) {
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
}

// ── Base widget ───────────────────────────────────────────────────────────────

class _LsbBase {
    constructor(name) {
        this.name = `${LSB_WPREFIX}${name}`;
        this.type = "custom";
        this.options = { serialize: false };
        this.value = "";
        this.mouseDowned = null;
        this.isMouseDownedAndOver = false;
        this.hitAreas = {};
        this.downedForMove = [];
        this.downedForClick = [];
    }

    serializeValue() { return undefined; }

    _inBounds(pos, bounds) {
        const x0 = bounds[0];
        if (bounds.length === 2) return pos[0] >= x0 && pos[0] <= x0 + bounds[1];
        return pos[0] >= x0 && pos[0] <= x0 + bounds[2] && pos[1] >= bounds[1] && pos[1] <= bounds[1] + bounds[3];
    }

    mouse(event, pos, node) {
        const isRight = event.button === 2 || event.which === 3 || event.type === "contextmenu";
        if (isRight && typeof this.onContextMenu === "function") {
            event.preventDefault?.();
            event.stopPropagation?.();
            this.cancelMouseDown();
            return this.onContextMenu(event, pos, node) === true;
        }
        if (event.type === "pointerdown") {
            this.mouseDowned = [...pos];
            this.isMouseDownedAndOver = true;
            this.downedForMove.length = 0;
            this.downedForClick.length = 0;
            let handled = false;
            for (const part of Object.values(this.hitAreas)) {
                if (this._inBounds(pos, part.bounds)) {
                    if (part.onMove) this.downedForMove.push(part);
                    if (part.onClick) this.downedForClick.push(part);
                    if (part.onDown) handled = part.onDown.apply(this, [event, pos, node, part]) === true || handled;
                    part.wasDown = true;
                }
            }
            return this.onMouseDown(event, pos, node) ?? handled;
        }
        if (event.type === "pointerup") {
            if (!this.mouseDowned) return true;
            this.downedForMove.length = 0;
            const wasOver = this.isMouseDownedAndOver;
            this.cancelMouseDown();
            let handled = false;
            for (const part of Object.values(this.hitAreas)) {
                if (part.onUp && this._inBounds(pos, part.bounds)) handled = part.onUp.apply(this, [event, pos, node, part]) === true || handled;
                part.wasDown = false;
            }
            for (const part of this.downedForClick) {
                if (this._inBounds(pos, part.bounds)) handled = part.onClick.apply(this, [event, pos, node, part]) === true || handled;
            }
            this.downedForClick.length = 0;
            if (wasOver) handled = this.onMouseClick(event, pos, node) === true || handled;
            return this.onMouseUp(event, pos, node) ?? handled;
        }
        if (event.type === "pointermove") {
            this.isMouseDownedAndOver = Boolean(this.mouseDowned);
            if (this.mouseDowned && (pos[0] < 15 || pos[0] > node.size[0] - 15 || pos[1] < this.last_y || pos[1] > this.last_y + _lsbRowH())) {
                this.isMouseDownedAndOver = false;
            }
            for (const part of Object.values(this.hitAreas)) {
                if (this.downedForMove.includes(part)) part.onMove.apply(this, [event, pos, node, part]);
                if (this.downedForClick.includes(part)) part.wasDown = this._inBounds(pos, part.bounds);
            }
            return this.onMouseMove(event, pos, node) ?? true;
        }
        return false;
    }

    cancelMouseDown() {
        this.mouseDowned = null;
        this.isMouseDownedAndOver = false;
        this.downedForMove.length = 0;
    }

    onMouseDown() {}
    onMouseUp() {}
    onMouseClick() {}
    onMouseMove() {}
}

// ── Divider ────────────────────────────────────────────────────────────────────

class _LsbDivider extends _LsbBase {
    constructor() { super("divider"); }
    computeSize(w) { return [w, 5]; }
    draw() {}
}

// ── Header ─────────────────────────────────────────────────────────────────────

class _LsbHeader extends _LsbBase {
    constructor() {
        super("header");
        this.hitAreas = { toggle: { bounds: [0, 0], onDown: this.onToggleDown } };
    }

    computeSize(w) { return [w, _lsbRowH()]; }

    draw(ctx, node, width, posY, height) {
        if (_lsbIsLowQ()) return;
        posY += 2;
        const midY = posY + height * 0.5;
        const isLtx = _lsbIsLtx(node);
        ctx.save();
        this.hitAreas.toggle.bounds = _lsbDrawToggle(ctx, { posX: LSB_INSET, posY, height, value: _lsbAllEnabled(node) });
        const posX = LSB_INSET + this.hitAreas.toggle.bounds[1] + LSB_INNER_M;
        ctx.globalAlpha = (app.canvas?.editor_alpha ?? 1) * 0.55;
        ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText("Toggle All", posX, midY);
        ctx.textAlign = "center";
        if (isLtx) {
            ctx.fillText("Aud",   _lsbNumLabelCX(node, 0), midY);
            ctx.fillText("Vid",   _lsbNumLabelCX(node, 1), midY);
            ctx.fillText("Model", _lsbNumLabelCX(node, 2), midY);
        } else {
            ctx.fillText("Clip",  _lsbNumLabelCX(node, 0), midY);
            ctx.fillText("Model", _lsbNumLabelCX(node, 1), midY);
        }
        ctx.restore();
    }

    onToggleDown(event, pos, node) {
        const next = !_lsbAllEnabled(node);
        for (let i = 0; i < LSB_MAX_SLOTS; i++) _lsbSV(node, `enabled_${i}`, next);
        _lsbRedraw(node);
        this.cancelMouseDown();
        return true;
    }
}

// ── Row ────────────────────────────────────────────────────────────────────────

class _LsbRow extends _LsbBase {
    constructor(index) {
        super(`row_${index}`);
        this.index = index;
        this._dragging = false;
        this.hitAreas = {
            toggle:  { bounds: [0,0], onDown: this.onToggleDown },
            lora:    { bounds: [0,0], onClick: this.onLoraClick },
            mDec:    { bounds: [0,0], onClick: (e,p,n,part) => this._step(n, `strength_model_${this.index}`, -0.05, -10, 10) },
            mVal:    { bounds: [0,0], onClick: (e,p,n,part) => this._prompt(n, `strength_model_${this.index}`, "Model strength", -10, 10, e) },
            mInc:    { bounds: [0,0], onClick: (e,p,n,part) => this._step(n, `strength_model_${this.index}`, +0.05, -10, 10) },
            mAny:    { bounds: [0,0], onMove:  (e,p,n) => this._drag(n, `strength_model_${this.index}`, e.deltaX) },
            cDec:    { bounds: [0,0], onClick: (e,p,n,part) => this._step(n, `strength_clip_${this.index}`, -0.05, -10, 10) },
            cVal:    { bounds: [0,0], onClick: (e,p,n,part) => this._prompt(n, `strength_clip_${this.index}`, "CLIP strength", -10, 10, e) },
            cInc:    { bounds: [0,0], onClick: (e,p,n,part) => this._step(n, `strength_clip_${this.index}`, +0.05, -10, 10) },
            cAny:    { bounds: [0,0], onMove:  (e,p,n) => this._drag(n, `strength_clip_${this.index}`, e.deltaX) },
            info:    { bounds: [0,0,0,0], onClick: this.onInfoClick },
            vDec:    { bounds: [0,0], onClick: (e,p,n,part) => this._step(n, `video_${this.index}`, -0.05, 0, 1) },
            vVal:    { bounds: [0,0], onClick: (e,p,n,part) => this._prompt(n, `video_${this.index}`, "Video strength", 0, 1, e) },
            vInc:    { bounds: [0,0], onClick: (e,p,n,part) => this._step(n, `video_${this.index}`, +0.05, 0, 1) },
            vAny:    { bounds: [0,0], onMove:  (e,p,n) => this._drag(n, `video_${this.index}`, e.deltaX) },
            aDec:    { bounds: [0,0], onClick: (e,p,n,part) => this._step(n, `audio_${this.index}`, -0.05, 0, 1) },
            aVal:    { bounds: [0,0], onClick: (e,p,n,part) => this._prompt(n, `audio_${this.index}`, "Audio strength", 0, 1, e) },
            aInc:    { bounds: [0,0], onClick: (e,p,n,part) => this._step(n, `audio_${this.index}`, +0.05, 0, 1) },
            aAny:    { bounds: [0,0], onMove:  (e,p,n) => this._drag(n, `audio_${this.index}`, e.deltaX) },
        };
    }

    computeSize(width) { return [width, _lsbRowH()]; }

    draw(ctx, node, width, posY, height) {
        this.last_y = posY;
        const isLtx = _lsbIsLtx(node);
        const rowH = _lsbRowH();
        const lowQ = _lsbIsLowQ();
        const enabled = Boolean(_lsbGV(node, `enabled_${this.index}`, true));
        const midY = posY + rowH * 0.5;

        ctx.save();
        _lsbDrawRoundedRect(ctx, { pos: [LSB_INSET, posY], size: [width - LSB_INSET * 2, rowH] });

        let posX = LSB_INSET;
        this.hitAreas.toggle.bounds = _lsbDrawToggle(ctx, { posX, posY, height: rowH, value: enabled });
        posX += this.hitAreas.toggle.bounds[1] + LSB_INNER_M;

        if (!lowQ) {
            if (!enabled) ctx.globalAlpha = (app.canvas?.editor_alpha ?? 1) * 0.4;
            ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;

            let loraRightX;
            if (isLtx) {
                // LTX2.3: fromRight=0 → aud, fromRight=1 → vid, fromRight=2 → model
                const [aD,aT,aI] = _lsbDrawNumber(ctx, node, `audio_${this.index}`,         _lsbNumRight(node, 0), posY, rowH);
                const [vD,vT,vI] = _lsbDrawNumber(ctx, node, `video_${this.index}`,          _lsbNumRight(node, 1), posY, rowH);
                const [mD,mT,mI] = _lsbDrawNumber(ctx, node, `strength_model_${this.index}`, _lsbNumRight(node, 2), posY, rowH);
                this.hitAreas.aDec.bounds=aD; this.hitAreas.aVal.bounds=aT; this.hitAreas.aInc.bounds=aI; this.hitAreas.aAny.bounds=[aD[0],aI[0]+aI[1]-aD[0]];
                this.hitAreas.vDec.bounds=vD; this.hitAreas.vVal.bounds=vT; this.hitAreas.vInc.bounds=vI; this.hitAreas.vAny.bounds=[vD[0],vI[0]+vI[1]-vD[0]];
                this.hitAreas.mDec.bounds=mD; this.hitAreas.mVal.bounds=mT; this.hitAreas.mInc.bounds=mI; this.hitAreas.mAny.bounds=[mD[0],mI[0]+mI[1]-mD[0]];
                for (const k of ["cDec","cVal","cInc","cAny"]) this.hitAreas[k].bounds=[0,0];
                loraRightX = mD[0] - LSB_INNER_M;
            } else {
                // All other targets: fromRight=0 → clip, fromRight=1 → model
                const [cD,cT,cI] = _lsbDrawNumber(ctx, node, `strength_clip_${this.index}`,  _lsbNumRight(node, 0), posY, rowH);
                const [mD,mT,mI] = _lsbDrawNumber(ctx, node, `strength_model_${this.index}`, _lsbNumRight(node, 1), posY, rowH);
                this.hitAreas.cDec.bounds=cD; this.hitAreas.cVal.bounds=cT; this.hitAreas.cInc.bounds=cI; this.hitAreas.cAny.bounds=[cD[0],cI[0]+cI[1]-cD[0]];
                this.hitAreas.mDec.bounds=mD; this.hitAreas.mVal.bounds=mT; this.hitAreas.mInc.bounds=mI; this.hitAreas.mAny.bounds=[mD[0],mI[0]+mI[1]-mD[0]];
                for (const k of ["vDec","vVal","vInc","vAny","aDec","aVal","aInc","aAny"]) this.hitAreas[k].bounds=[0,0];
                loraRightX = mD[0] - LSB_INNER_M;
            }

            // LoRA name fills remaining left space
            const loraW = loraRightX - posX;
            ctx.textAlign = "left";
            ctx.textBaseline = "middle";
            const loraName = _lsbGV(node, `lora_${this.index}`, LSB_NONE);
            ctx.fillText(_lsbFitStr(ctx, loraName && loraName !== LSB_NONE ? String(loraName) : "None", loraW), posX, midY);
            this.hitAreas.lora.bounds = [posX, loraW];

            ctx.globalAlpha = app.canvas?.editor_alpha ?? 1;
            this.hitAreas.info.bounds = _lsbDrawIcon(ctx, _lsbIconX(node), posY + 1, false);
        }
        ctx.restore();
    }

    onToggleDown(event, pos, node) {
        const key = `enabled_${this.index}`;
        _lsbSV(node, key, !Boolean(_lsbGV(node, key, true)));
        _lsbRedraw(node);
        this.cancelMouseDown();
        return true;
    }

    onLoraClick(event, pos, node) {
        _lsbShowLoraChooser(event, node, this.index);
        this.cancelMouseDown();
        return true;
    }

    onInfoClick(event, pos, node) {
        _lsbOpenInfoPanel(node, this.index);
        this.cancelMouseDown();
        return true;
    }

    _step(node, key, delta, min, max) {
        _lsbSV(node, key, _lsbClamp(_lsbR2(Number(_lsbGV(node, key, 1)) + delta), min, max));
        _lsbRedraw(node);
    }

    _drag(node, key, deltaX) {
        if (!deltaX) return;
        this._dragging = true;
        _lsbSV(node, key, _lsbR2(Number(_lsbGV(node, key, 1)) + deltaX * 0.05));
        _lsbRedraw(node);
    }

    _prompt(node, key, label, min, max, event) {
        if (this._dragging) return;
        app.canvas.prompt(label, Number(_lsbGV(node, key, 1)).toFixed(2), (v) => {
            const n = Number(v);
            if (Number.isFinite(n)) { _lsbSV(node, key, _lsbClamp(_lsbR2(n), min, max)); _lsbRedraw(node); }
        }, event);
    }

    onMouseUp() { this._dragging = false; }
}

// ── Add-LoRA button ───────────────────────────────────────────────────────────

class _LsbAddButton extends _LsbBase {
    constructor() {
        super("add_btn");
        this.hitAreas = { btn: { bounds: [0, 0], onClick: this.onAddClick } };
    }

    computeSize(width) { return [width, _lsbRowH()]; }

    draw(ctx, node, width, posY, height) {
        const rowH = _lsbRowH();
        const atMax = _lsbActiveRows(node) >= LSB_MAX_SLOTS;
        ctx.save();
        const bx = LSB_INSET, bw = width - LSB_INSET * 2;
        _lsbDrawRoundedRect(ctx, { pos: [bx, posY + 2], size: [bw, rowH - 4] });
        ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
        ctx.globalAlpha = (app.canvas?.editor_alpha ?? 1) * (atMax ? 0.3 : 0.7);
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.font = `bold ${Math.round(rowH * 0.38)}px sans-serif`;
        ctx.fillText(atMax ? `Max ${LSB_MAX_SLOTS} LoRAs` : "+ Add LoRA", width * 0.5, posY + rowH * 0.5);
        this.hitAreas.btn.bounds = [bx, bw];
        ctx.restore();
    }

    onAddClick(event, pos, node) {
        const cur = _lsbActiveRows(node);
        if (cur >= LSB_MAX_SLOTS) return true;
        node.properties = node.properties || {};
        node.properties.fbt_lsb_rows = cur + 1;
        _lsbRebuildUi(node);
        this.cancelMouseDown();
        return true;
    }
}

// ── LoRA chooser context menu ─────────────────────────────────────────────────

async function _lsbShowLoraChooser(event, node, index) {
    const gen = (node.__fbtLsbChooserGen = (Number(node.__fbtLsbChooserGen || 0) + 1));
    const values = await _lsbLoraOptions(node);
    if (node.__fbtLsbRemoved || gen !== node.__fbtLsbChooserGen) return;
    node.__fbtLsbCtxMenu?.close?.();
    node.__fbtLsbCtxMenu?.root?.remove?.();
    const menu = new LiteGraph.ContextMenu(values, {
        event,
        title: `Slot ${index} — Choose LoRA`,
        className: "dark",
        scale: Math.max(1, app.canvas?.ds?.scale ?? 1),
        callback: (value) => {
            if (node.__fbtLsbRemoved || gen !== node.__fbtLsbChooserGen) return;
            const sel = String(value?.content ?? value?.value ?? value);
            _lsbSV(node, `lora_${index}`, sel === "None" ? LSB_NONE : sel);
            _lsbRedraw(node);
        },
    });
    node.__fbtLsbCtxMenu = menu;
}

// ── Math helpers ───────────────────────────────────────────────────────────────

function _lsbR2(v) { return Math.round(Number(v) * 100) / 100; }
function _lsbClamp(v, min, max) { return Math.max(min, Math.min(max, v)); }
function _lsbFitStr(ctx, str, maxW) {
    const s = String(str ?? "");
    if (ctx.measureText(s).width <= maxW) return s;
    const ell = "...";
    let lo = 0, hi = s.length;
    while (lo < hi) {
        const mid = Math.ceil((lo + hi) / 2);
        if (ctx.measureText(s.slice(0, mid) + ell).width <= maxW) lo = mid; else hi = mid - 1;
    }
    return s.slice(0, Math.max(0, lo)) + ell;
}

// ── Drawing primitives ─────────────────────────────────────────────────────────

function _lsbDrawRoundedRect(ctx, { pos, size, borderRadius, colorBackground, colorStroke }) {
    const lowQ = _lsbIsLowQ();
    ctx.save();
    ctx.strokeStyle = colorStroke || LiteGraph.WIDGET_OUTLINE_COLOR;
    ctx.fillStyle = colorBackground || LiteGraph.WIDGET_BGCOLOR;
    ctx.beginPath();
    ctx.roundRect(...pos, ...size, lowQ ? [0] : [borderRadius ?? (size[1] * 0.4)]);
    ctx.fill();
    if (!lowQ) ctx.stroke();
    ctx.restore();
}

function _lsbDrawToggle(ctx, { posX, posY, height, value }) {
    const lowQ = _lsbIsLowQ();
    const radius = height * 0.36;
    const bgW = height * 1.5;
    ctx.save();
    if (!lowQ) {
        ctx.beginPath();
        ctx.roundRect(posX + 4, posY + 4, bgW - 8, height - 8, [height * 0.5]);
        ctx.globalAlpha = (app.canvas?.editor_alpha ?? 1) * 0.25;
        ctx.fillStyle = "rgba(255,255,255,0.45)";
        ctx.fill();
        ctx.globalAlpha = app.canvas?.editor_alpha ?? 1;
    }
    ctx.fillStyle = value ? "#89B" : "#888";
    const tx = lowQ || value === false ? posX + height * 0.5 : posX + height;
    ctx.beginPath();
    ctx.arc(tx, posY + height * 0.5, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
    return [posX, bgW];
}

function _lsbDrawNumber(ctx, node, key, rightX, posY, height) {
    const aW = 9, iM = 3, nW = 32;
    const value = Number(_lsbGV(node, key, 1));
    const midY = posY + height / 2;
    let posX = rightX - aW - iM - nW - iM - aW;
    ctx.save();
    ctx.fill(new Path2D(`M ${posX} ${midY} l ${aW} ${aW/2} l 0 -${aW} L ${posX} ${midY} z`));
    const dec = [posX, aW];
    posX += aW + iM;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(_lsbFitStr(ctx, value.toFixed(2), nW), posX + nW / 2, midY);
    const txt = [posX, nW];
    posX += nW + iM;
    ctx.fill(new Path2D(`M ${posX} ${midY - aW/2} l ${aW} ${aW/2} l -${aW} ${aW/2} v -${aW} z`));
    const inc = [posX, aW];
    ctx.restore();
    return [dec, txt, inc];
}

function _lsbDrawIcon(ctx, x, y, active) {
    const lowQ = _lsbIsLowQ();
    ctx.save();
    _lsbDrawRoundedRect(ctx, {
        pos: [x, y], size: [LSB_ICON_SIZE, LSB_ICON_SIZE], borderRadius: 5,
        colorBackground: active ? LiteGraph.WIDGET_BGCOLOR : "#00000044",
        colorStroke: LiteGraph.WIDGET_OUTLINE_COLOR,
    });
    if (!lowQ) {
        ctx.fillStyle = active ? LiteGraph.WIDGET_TEXT_COLOR : "rgba(215,220,224,0.55)";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.font = `bold ${Math.round(LSB_ICON_SIZE * 0.55)}px sans-serif`;
        ctx.fillText("i", x + LSB_ICON_SIZE * 0.5, y + LSB_ICON_SIZE * 0.5 + 0.5);
    }
    ctx.restore();
    return [x, y, LSB_ICON_SIZE, LSB_ICON_SIZE];
}

// ── Info (Civitai modal) ────────────────────────────────────────────────────────

async function _lsbOpenInfoPanel(node, index) {
    if (node.__fbtLsbRemoved) return;
    const loraName = _lsbGV(node, `lora_${index}`, LSB_NONE);
    if (!loraName || loraName === LSB_NONE) { _lsbToast("Select a LoRA first."); return; }
    _lsbToast("Fetching Civitai info…");
    try {
        const data = await loraAPI.getCivitaiInfo(loraName);
        showCivitaiModal(data);
    } catch (err) {
        _lsbToast(err?.statusCode === 404 ? "Not found on Civitai." : "Civitai lookup failed.");
    }
}

function _lsbToast(msg) {
    if (!document.getElementById("fbt-lsb-toast-style")) {
        const s = document.createElement("style");
        s.id = "fbt-lsb-toast-style";
        s.textContent = `.fbt-lsb-toast{position:fixed;left:50%;bottom:32px;z-index:100001;transform:translateX(-50%);border:1px solid rgba(95,105,112,.95);border-radius:999px;background:rgba(31,33,36,.98);color:#d7dce0;padding:8px 12px;font:700 12px/1 sans-serif;box-shadow:0 10px 28px rgba(0,0,0,.45)}`;
        document.head.appendChild(s);
    }
    document.querySelectorAll(".fbt-lsb-toast").forEach(t => t.remove());
    const el = document.createElement("div");
    el.className = "fbt-lsb-toast";
    el.textContent = msg;
    document.body.appendChild(el);
    setTimeout(() => el.remove(), 1800);
}

// ── Export ─────────────────────────────────────────────────────────────────────

export function setupLoraStackBuilder(nodeType, nodeData, appRef) {
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        const result = onNodeCreated?.apply(this, arguments);
        this.__fbtLsbRemoved = false;
        _lsbSetupNode(this);
        return result;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
        const result = onConfigure?.apply(this, arguments);
        this.__fbtLsbRemoved = false;
        queueMicrotask(() => {
            if (this.__fbtLsbRemoved) return;
            _lsbSetupNode(this);
        });
        return result;
    };

    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
        this.__fbtLsbRemoved = true;
        this.__fbtLsbChooserGen = (Number(this.__fbtLsbChooserGen || 0) + 1);
        this.__fbtLsbInfoClose?.();
        try { this.__fbtLsbCtxMenu?.close?.(); } catch (_) {}
        try { this.__fbtLsbCtxMenu?.root?.remove?.(); } catch (_) {}
        return onRemoved?.apply(this, arguments);
    };
}
