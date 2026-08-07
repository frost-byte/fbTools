/**
 * Concept Registry node frontend extensions.
 *
 * ConceptRegistryLoad — adds a "Reload Registry" button.
 * ConceptDefine — compact canvas row UI: LoRA name + weight on one line.
 *   Split models (wan22, bernini): H row (lora + weight) and L row (lora_low + weight_low).
 *   Non-split models: single row with no badge.
 *   Switching model_type live rebuilds the visible rows.
 */

import { conceptsAPI } from "../api/concepts.js";
import { loraAPI } from "../api/lora.js";
import { showCivitaiModal } from "./lora.js";

const SPLIT_MODEL_TYPES = new Set(["wan22", "bernini"]);

// ── ConceptDefine compact row constants ───────────────────────────────────────

const CD_WPREFIX   = "fbt_cd_";
const CD_INSET     = 15;
const CD_INNER_M   = 3.3;
const CD_ICON_SIZE = 18;
const CD_ICON_PAD  = 18;

// ── Geometry / value helpers ──────────────────────────────────────────────────

function _cdRowH()        { return typeof LiteGraph !== "undefined" ? LiteGraph.NODE_WIDGET_HEIGHT : 20; }
function _cdIsLowQ()      { return ((typeof app !== "undefined" && app.canvas?.ds?.scale) || 1) <= 0.5; }
function _cdNumW()        { return 9 + 3 + 32 + 3 + 9; }
function _cdIconColW()    { return CD_ICON_SIZE + CD_ICON_PAD + CD_INNER_M * 2; }
function _cdRowRight(n)   { return n.size[0] - CD_INSET; }
function _cdSpinRight(n)  { return _cdRowRight(n) - CD_INNER_M - _cdIconColW(); }
function _cdIconX(n)      { return _cdRowRight(n) - CD_ICON_PAD - CD_ICON_SIZE; }

function _cdGW(node, name) { return (node.widgets || []).find(w => w.name === name); }
function _cdGV(node, name, fb) { const w = _cdGW(node, name); return w !== undefined ? w.value : fb; }
function _cdSV(node, name, val) {
    const w = _cdGW(node, name);
    if (w && w.value !== val) { w.value = val; w.callback?.(val); }
}
function _cdIsSplit(node) { return SPLIT_MODEL_TYPES.has(_cdGV(node, "model_type", "")); }
function _cdRedraw(node)  { node.setDirtyCanvas?.(true, true); }
function _cdR2(v)         { return Math.round(v * 100) / 100; }
function _cdClamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }
function _cdFitStr(ctx, s, maxW) {
    if (ctx.measureText(s).width <= maxW) return s;
    let lo = 0, hi = s.length;
    while (lo < hi - 1) {
        const mid = (lo + hi + 1) >> 1;
        if (ctx.measureText(s.slice(0, mid) + "…").width <= maxW) lo = mid; else hi = mid - 1;
    }
    return s.slice(0, lo) + "…";
}

// ── Canvas draw helpers ───────────────────────────────────────────────────────

function _cdDrawRect(ctx, pos, size, bg, stroke, radius) {
    ctx.save();
    ctx.fillStyle   = bg     || LiteGraph.WIDGET_BGCOLOR;
    ctx.strokeStyle = stroke || LiteGraph.WIDGET_OUTLINE_COLOR;
    ctx.beginPath();
    ctx.roundRect(...pos, ...size, _cdIsLowQ() ? [0] : [radius ?? (size[1] * 0.4)]);
    ctx.fill();
    if (!_cdIsLowQ()) ctx.stroke();
    ctx.restore();
}

function _cdDrawSpinner(ctx, value, rightX, posY, rowH) {
    const bW = 9, vW = 32, g = 3, totW = bW + g + vW + g + bW;
    const x = rightX - totW, mid = posY + rowH * 0.5;
    const a0 = ctx.globalAlpha;
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.font = `${Math.round(rowH * 0.55)}px sans-serif`;
    ctx.globalAlpha = a0 * 0.7; ctx.fillText("◄", x + bW * 0.5, mid);
    ctx.globalAlpha = a0;       ctx.fillText(typeof value === "number" ? value.toFixed(2) : String(value), x + bW + g + vW * 0.5, mid);
    ctx.globalAlpha = a0 * 0.7; ctx.fillText("►", x + bW + g + vW + g + bW * 0.5, mid);
    ctx.globalAlpha = a0;
    return {
        dec:  [x,            posY, bW,   rowH],
        val:  [x + bW + g,   posY, vW,   rowH],
        inc:  [x + bW+g+vW+g,posY, bW,   rowH],
        drag: [x,            posY, totW, rowH],
    };
}

function _cdDrawIcon(ctx, x, y) {
    const sz = CD_ICON_SIZE;
    _cdDrawRect(ctx, [x, y], [sz, sz], "rgba(60,65,70,0.7)", "rgba(100,110,120,0.5)", 5);
    ctx.save();
    ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.font = `bold ${Math.round(sz * 0.55)}px sans-serif`;
    ctx.fillText("i", x + sz * 0.5, y + sz * 0.5 + 0.5);
    ctx.restore();
    return [x, y, sz, sz];
}

// ── _CdLoraRow widget ─────────────────────────────────────────────────────────

class _CdLoraRow {
    constructor(loraKey, weightKey, label, min = 0.0, max = 3.0, step = 0.05) {
        this.name     = `${CD_WPREFIX}${loraKey}`;
        this.type     = "custom";
        this.options  = { serialize: false };
        this.value    = "";
        this.loraKey   = loraKey;
        this.weightKey = weightKey;
        this.label     = label; // "H", "L", or ""
        this.min = min; this.max = max; this.step = step;
        this._dragging  = false;
        this._dragStart = null;
        const z = [0,0,0,0];
        this._dec = z; this._val = z; this._inc = z; this._drag = z;
        this._lora = z; this._icon = z;
    }

    serializeValue() { return undefined; }
    computeSize(width) { return [width, _cdRowH()]; }

    draw(ctx, node, width, posY) {
        const rowH = _cdRowH(), midY = posY + rowH * 0.5;
        ctx.save();
        _cdDrawRect(ctx, [CD_INSET, posY], [width - CD_INSET * 2, rowH]);

        if (!_cdIsLowQ()) {
            const alpha = app.canvas?.editor_alpha ?? 1;
            ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
            ctx.globalAlpha = alpha;

            // H / L badge
            let posX = CD_INSET + CD_INNER_M;
            if (this.label) {
                const bW = rowH * 0.85;
                _cdDrawRect(ctx, [posX + 1, posY + 2], [bW - 2, rowH - 4],
                    "rgba(70,82,95,0.55)", "rgba(110,125,138,0.4)", 5);
                ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
                ctx.globalAlpha = alpha;
                ctx.textAlign = "center"; ctx.textBaseline = "middle";
                ctx.font = `bold ${Math.round(rowH * 0.46)}px sans-serif`;
                ctx.fillText(this.label, posX + bW * 0.5, midY);
                posX += bW + CD_INNER_M;
            }

            // Weight spinner
            const wt = Number(_cdGV(node, this.weightKey, 1.0));
            const bnds = _cdDrawSpinner(ctx, _cdR2(wt), _cdSpinRight(node), posY, rowH);
            this._dec = bnds.dec; this._val = bnds.val;
            this._inc = bnds.inc; this._drag = bnds.drag;

            // LoRA name label
            const loraName = String(_cdGV(node, this.loraKey, "None") || "None");
            const loraW = bnds.dec[0] - CD_INNER_M - posX;
            ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
            ctx.globalAlpha = alpha * (loraName === "None" ? 0.45 : 1);
            ctx.textAlign = "left"; ctx.textBaseline = "middle";
            ctx.font = `${Math.round(rowH * 0.55)}px sans-serif`;
            ctx.fillText(_cdFitStr(ctx, loraName, loraW), posX, midY);
            this._lora = [posX, posY, loraW, rowH];

            // ⓘ icon
            ctx.globalAlpha = alpha;
            this._icon = _cdDrawIcon(ctx, _cdIconX(node), posY + 1);
        }
        ctx.restore();
    }

    _hit(pos, b) {
        return pos[0] >= b[0] && pos[0] <= b[0]+b[2] && pos[1] >= b[1] && pos[1] <= b[1]+b[3];
    }

    mouse(event, pos, node) {
        if (event.type === "pointerdown") {
            if (this._hit(pos, this._dec)) {
                _cdSV(node, this.weightKey, _cdClamp(_cdR2(Number(_cdGV(node, this.weightKey, 1)) - this.step), this.min, this.max));
                _cdRedraw(node); return true;
            }
            if (this._hit(pos, this._inc)) {
                _cdSV(node, this.weightKey, _cdClamp(_cdR2(Number(_cdGV(node, this.weightKey, 1)) + this.step), this.min, this.max));
                _cdRedraw(node); return true;
            }
            if (this._hit(pos, this._drag)) {
                this._dragging  = true;
                this._dragStart = { pos: [...pos], val: Number(_cdGV(node, this.weightKey, 1.0)) };
                return true;
            }
            if (this._hit(pos, this._lora)) { this._openLoraChooser(event, node); return true; }
            if (this._hit(pos, this._icon)) { this._openCivitai(node); return true; }
        }
        if (event.type === "pointermove" && this._dragging && this._dragStart) {
            const dx = pos[0] - this._dragStart.pos[0];
            _cdSV(node, this.weightKey, _cdClamp(_cdR2(this._dragStart.val + dx * 0.02), this.min, this.max));
            _cdRedraw(node); return true;
        }
        if ((event.type === "pointerup" || event.type === "pointerleave") && this._dragging) {
            if (this._dragStart && Math.abs(pos[0] - this._dragStart.pos[0]) < 3) {
                const lbl = this.label === "H" ? "High Weight" : this.label === "L" ? "Low Weight" : "Weight";
                app.canvas.prompt(lbl, Number(_cdGV(node, this.weightKey, 1)).toFixed(2), (v) => {
                    const n = Number(v);
                    if (Number.isFinite(n)) { _cdSV(node, this.weightKey, _cdClamp(_cdR2(n), this.min, this.max)); _cdRedraw(node); }
                }, event);
            }
            this._dragging = false; this._dragStart = null;
            return false;
        }
        return false;
    }

    _openLoraChooser(event, node) {
        const w = _cdGW(node, this.loraKey);
        if (!w) return;
        const values = w.options?.values || [];
        new LiteGraph.ContextMenu(values, {
            event,
            title: this.label === "H" ? "High LoRA" : this.label === "L" ? "Low LoRA" : "LoRA",
            className: "dark",
            scale: Math.max(1, app.canvas?.ds?.scale ?? 1),
            callback: (value) => {
                _cdSV(node, this.loraKey, String(value?.content ?? value?.value ?? value));
                _cdRedraw(node);
            },
        });
    }

    async _openCivitai(node) {
        const loraName = _cdGV(node, this.loraKey, "None");
        if (!loraName || loraName === "None") {
            app.extensionManager?.toast?.add({ severity: "warn", summary: "No LoRA selected", detail: "Choose a LoRA first.", life: 2000 });
            return;
        }
        app.extensionManager?.toast?.add({ severity: "info", summary: "Fetching Civitai info…", detail: loraName, life: 2000 });
        try {
            const data = await loraAPI.getCivitaiInfo(loraName);
            showCivitaiModal(data);
        } catch (err) {
            app.extensionManager?.toast?.add({
                severity: "error",
                summary: "Civitai lookup failed",
                detail: err?.statusCode === 404 ? "Not found on Civitai." : (err?.message || "Request failed"),
                life: 4000,
            });
        }
    }
}

// ── Rebuild compact UI ────────────────────────────────────────────────────────

function _cdHideWidget(widget) {
    if (!widget) return;
    widget.hidden = true;
    widget.type = "converted-widget";
    widget.computeSize = () => [0, -4];
    if (widget.element) widget.element.style.display = "none";
}

function _cdHideNativeLoraWidgets(node) {
    for (const name of ["lora", "lora_low", "weight", "weight_low"]) {
        _cdHideWidget(_cdGW(node, name));
    }
}

function _cdRebuildUi(node) {
    // Remove previously generated compact widgets
    node.widgets = (node.widgets || []).filter(w => !String(w.name || "").startsWith(CD_WPREFIX));
    // Sanitize weight values — guard against corrupted saves where value is false/null
    for (const key of ["weight", "weight_low"]) {
        const w = _cdGW(node, key);
        if (w && typeof w.value !== "number") w.value = 1.0;
    }
    // Hide native lora/weight widgets
    _cdHideNativeLoraWidgets(node);

    const isSplit = _cdIsSplit(node);
    const rows = [new _CdLoraRow("lora", "weight", isSplit ? "H" : "", 0.0, 3.0, 0.05)];
    if (isSplit) rows.push(new _CdLoraRow("lora_low", "weight_low", "L", 0.0, 3.0, 0.05));

    // Insert right after the model_type widget
    const mtIdx = node.widgets.findIndex(w => w.name === "model_type");
    node.widgets.splice(mtIdx >= 0 ? mtIdx + 1 : node.widgets.length, 0, ...rows);

    node.setSize?.(node.computeSize?.() || node.size);
    node.setDirtyCanvas?.(true, true);
}

// ── ConceptDefine ─────────────────────────────────────────────────────────────

export function setupConceptDefine(nodeType, nodeData, app) {
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        if (onNodeCreated) onNodeCreated.apply(this, arguments);
        const self = this;
        // Watch model_type for live changes — these are user-initiated, no value
        // assignment is in-flight, so it's safe to rebuild immediately.
        const mtWidget = this.widgets?.find(w => w.name === "model_type");
        if (mtWidget) {
            const origCb = mtWidget.callback;
            mtWidget.callback = function (value) {
                origCb?.apply(this, arguments);
                _cdRebuildUi(self);
            };
        }
        // Defer so onConfigure (if this is a loaded node) can assign widget
        // values before we convert them to "converted-widget" type.
        queueMicrotask(() => _cdRebuildUi(self));
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (data) {
        if (onConfigure) onConfigure.apply(this, arguments);
        // onConfigure.apply assigns saved widget values; rebuild after that.
        queueMicrotask(() => _cdRebuildUi(this));
    };
}

// ── ConceptRegistryLoad ───────────────────────────────────────────────────────

export function setupConceptRegistryLoad(nodeType, nodeData, appRef) {
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        if (onNodeCreated) onNodeCreated.apply(this, arguments);
        const self = this;

        this.addWidget("button", "↺ Reload Registry", null, async function () {
            try {
                await conceptsAPI.reload();
                appRef.extensionManager?.toast?.add({
                    severity: "success",
                    summary: "Registry reload triggered",
                    detail: "Re-queue the workflow to load the updated concepts.",
                    life: 3000,
                });
            } catch (e) {
                appRef.extensionManager?.toast?.add({
                    severity: "error",
                    summary: "Reload failed",
                    detail: e.message || "Unknown error",
                    life: 5000,
                });
            }
        });
    };
}
