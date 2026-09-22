/**
 * Warn when a workflow is queued while the local LLM is still loaded.
 *
 * The local backend (llama.cpp/transformers via llm_client) shares the same GPU as
 * ComfyUI generation, so a loaded model can starve a queued run of VRAM. Unsloth and
 * Modal run on a remote endpoint and don't compete for local VRAM, so only the local
 * model is checked — via a fresh GET /fbtools/llm/status each time, not the cached
 * fbtLlm state in fbt_panel.js, since that is only populated once the fbTools panel
 * has been opened at least once this session.
 *
 * Fires on ComfyUI's own `promptQueued` event, which the frontend dispatches the
 * moment "Queue" is clicked (before the prompt reaches the server) — the earliest
 * possible point, giving the most time to unload and re-queue before VRAM is touched.
 */

const COOLDOWN_MS = 15000; // avoid nagging on every item in auto-queue/batch mode
let _lastWarnTs = -Infinity; // sentinel "never warned yet" — must never collide with a real Date.now()

export function setupLlmQueueWarning(app, api) {
    api.addEventListener("promptQueued", async () => {
        try {
            const now = Date.now();
            if (now - _lastWarnTs < COOLDOWN_MS) return;

            const res = await fetch("/fbtools/llm/status");
            if (!res.ok) return;
            const st = await res.json();
            const model = st?.loaded_model;
            if (!model) return;

            _lastWarnTs = now;
            app.extensionManager?.toast?.add({
                severity: "warn",
                summary: "Local LLM still loaded",
                detail: `"${model}" is loaded locally and may starve this run of VRAM. `
                       + "Unload it in the LLM tab, then re-queue.",
                life: 10000,
            });
        } catch (_) { /* best-effort only — never block queuing */ }
    });
}
