/**
 * PromptCompositionLoader — when a composition is wired into `prompt_composition`
 * it drives the node, so grey out the (now ignored) Composition dropdown to
 * make it obvious which one is in charge.
 */

const WIRED_TIP = "Ignored: a Prompt Composition is connected to the Prompt Composition input.";

export function setupPromptCompositionLoader(nodeType, _nodeData, _app) {
    function _sync(node) {
        const linked = !!node.inputs?.find(i => i.name === "prompt_composition")?.link;
        const w = node.widgets?.find(x => x.name === "composition_name");
        if (!w) return;
        if (w._fbtOrigTooltip === undefined) w._fbtOrigTooltip = w.tooltip ?? "";
        w.disabled = linked;
        w.tooltip = linked ? WIRED_TIP : w._fbtOrigTooltip;
        node.setDirtyCanvas?.(true, true);
    }

    const _origConn = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (type, index, connected, linkInfo) {
        _origConn?.call(this, type, index, connected, linkInfo);
        if (type === LiteGraph?.INPUT && this.inputs?.[index]?.name === "prompt_composition") {
            const node = this;
            requestAnimationFrame(() => _sync(node));
        }
    };

    const _origCfg = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (config) {
        _origCfg?.call(this, config);
        const node = this;
        requestAnimationFrame(() => _sync(node));
    };
}
