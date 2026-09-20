/**
 * CompositionLoad — propagate composition_name combo changes downstream.
 *
 * Same mechanism as SourceProfileLoad: when the selection changes, fire
 * onConnectionsChange on every node wired to output slot 0 as if the wire
 * had been reconnected, so SceneCastBuild refreshes its subject pool without
 * the user re-wiring the graph. Output slot 0 must stay the composition dict.
 */

export function setupCompositionLoad(nodeType, _nodeData, app) {
    const _origCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        _origCreated?.call(this);
        const node = this;

        // Widgets aren't always attached synchronously — wait a microtask.
        queueMicrotask(() => {
            const compWidget = node.widgets?.find(w => w.name === "composition_name");
            if (!compWidget) return;

            const _origCb = compWidget.callback;
            compWidget.callback = function (value, ...rest) {
                _origCb?.call(this, value, ...rest);

                const links = node.outputs?.[0]?.links ?? [];
                for (const linkId of links) {
                    const link = app.graph.links[linkId];
                    if (!link) continue;
                    const target = app.graph.getNodeById(link.target_id);
                    if (!target) continue;
                    target.onConnectionsChange?.(
                        LiteGraph.INPUT,
                        link.target_slot,
                        true,   // still connected
                        link,
                        target.inputs?.[link.target_slot],
                    );
                }
            };
        });
    };
}
