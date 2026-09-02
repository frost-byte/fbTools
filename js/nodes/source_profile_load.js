/**
 * SourceProfileLoad — propagate profile_name combo changes downstream.
 *
 * When the user selects a different profile in the combo, fire the same
 * onConnectionsChange event on every connected downstream node that a
 * disconnect + reconnect would fire.  This lets SceneCastBuild (and any
 * other node) refresh its widget state without requiring the user to
 * re-wire the graph.
 */

export function setupSourceProfileLoad(nodeType, _nodeData, app) {
    const _origCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        _origCreated?.call(this);
        const node = this;

        // Widgets aren't always attached synchronously — wait a microtask.
        queueMicrotask(() => {
            const profileWidget = node.widgets?.find(w => w.name === "profile_name");
            if (!profileWidget) return;

            const _origCb = profileWidget.callback;
            profileWidget.callback = function (value, ...rest) {
                _origCb?.call(this, value, ...rest);

                // Walk every link on output slot 0 (source_profile) and fire
                // onConnectionsChange on the target node as if the wire was
                // reconnected with the new upstream value.
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
