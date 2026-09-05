/**
 * REST API client for Unsloth Studio backend routes.
 * All endpoints live under /fbtools/unsloth/.
 */

import { BaseAPI } from "../utils/api_base.js";

class UnslothAPI extends BaseAPI {
    constructor() { super("/fbtools/unsloth"); }

    status()                     { return this.get("/status"); }
    setupStatus()                { return this.get("/setup_status"); }
    health(endpoint_key = null)  { return this.get("/health", endpoint_key ? { endpoint_key } : {}); }
    serveMode()                  { return this.get("/serve_mode"); }
    setServeMode(api_only)       { return this.post("/serve_mode", { api_only }); }
    activate(endpoint_key)       { return this.post("/activate", { endpoint_key }); }
    deactivate()                 { return this.post("/deactivate", {}); }
    deploy()                     { return this.post("/deploy", {}); }
    undeploy()                   { return this.post("/undeploy", {}); }
    containers()                 { return this.get("/containers"); }
    stopContainers(container_id) { return this.post("/containers/stop", container_id ? { container_id } : {}); }

    /**
     * Long-running (~5-30 min) — runs install_studio + bootstrap_api_key on Modal.
     * Uses a 35-minute AbortController timeout so the browser doesn't drop the request.
     */
    async bootstrapKey(force_reinstall = false) {
        const ctrl  = new AbortController();
        const timer = setTimeout(() => ctrl.abort(), 35 * 60 * 1000);
        try {
            const res = await fetch("/fbtools/unsloth/bootstrap_key", {
                method:  "POST",
                headers: { "Content-Type": "application/json" },
                body:    JSON.stringify({ force_reinstall }),
                signal:  ctrl.signal,
            });
            clearTimeout(timer);
            if (!res.ok) throw new Error(`Bootstrap failed (HTTP ${res.status})`);
            return await res.json();
        } catch (err) {
            clearTimeout(timer);
            throw err;
        }
    }
}

export const unslothApi = new UnslothAPI();
