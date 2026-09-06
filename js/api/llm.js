/**
 * REST API client for the LLM assistant (Composition Editor Phase 7).
 */

import { BaseAPI } from "../utils/api_base.js";

export class LlmAPI extends BaseAPI {
    constructor() {
        super("/fbtools/llm");
    }

    /** Scan model directories and return capability-annotated model list + default. */
    listModels() {
        return this.get("/models");
    }

    /** Return currently loaded model info and backend availability. */
    status() {
        return this.get("/status");
    }

    /** Load a model by its descriptor dict (from listModels). */
    loadModel(modelInfo) {
        return this.post("/load", { model_info: modelInfo });
    }

    /** Unload the current model and free VRAM. */
    unloadModel() {
        return this.post("/unload", {});
    }

    /**
     * Generate text, optionally with image filenames from the ComfyUI input dir.
     * @param {string} prompt
     * @param {object} opts - { system_prompt, images, max_tokens, temperature }
     */
    generate(prompt, opts = {}) {
        return this.post("/generate", {
            prompt,
            system_prompt:  opts.system_prompt  ?? "",
            images:         opts.images         ?? [],
            max_tokens:     opts.max_tokens     ?? 512,
            temperature:    opts.temperature    ?? 0.7,
        });
    }

    /** Generate a shot action description. */
    generateShotAction({ shotNumber, subjects, environment, style, existing } = {}) {
        return this.post("/generate/shot_action", {
            shot_number: shotNumber ?? 1,
            subjects:    subjects   ?? [],
            environment: environment ?? "",
            style:       style      ?? "cinematic",
            existing:    existing   ?? "",
        });
    }

    /** Generate a dialogue line. */
    generateDialogue({ speaker, context, tone, language } = {}) {
        return this.post("/generate/dialogue", {
            speaker:  speaker   ?? "Character",
            context:  context   ?? "",
            tone:     tone      ?? "",
            language: language  ?? "en-us",
        });
    }

    /** Polish existing text. */
    polish({ text, context } = {}) {
        return this.post("/generate/polish", {
            text:    text    ?? "",
            context: context ?? "",
        });
    }

    /**
     * Describe character actions/expressions/appearance from a video clip.
     * Requires a native-video model to be loaded (e.g. Qwen2.5-Omni).
     *
     * @param {string}   videoPath    — filename relative to dir
     * @param {string}   videoDir     — "input" | "output"
     * @param {number[]} frameIndices — exact frame indices selected in the filmstrip
     * @param {number}   shotNumber
     * @param {string[]} subjects
     * @param {string}   environment
     * @param {string}   style
     * @param {string}   intent       — "actions" | "expressions" | "appearance"
     */
    describeVideo({ videoPath, videoDir, frameIndices, shotNumber, subjects, environment, style, intent,
                    systemPrompt, userPrompt } = {}) {
        return this.post("/describe_video", {
            video_path:    videoPath    ?? "",
            dir:           videoDir     ?? "input",
            frame_indices: frameIndices ?? [],
            shot_number:   shotNumber   ?? 1,
            subjects:      subjects     ?? [],
            environment:   environment  ?? "",
            style:         style        ?? "cinematic",
            intent:        intent       ?? "actions",
            // Optional overrides — omit keys when empty so backend uses auto-generated prompts
            ...(systemPrompt ? { system_prompt: systemPrompt } : {}),
            ...(userPrompt   ? { user_prompt:   userPrompt   } : {}),
        });
    }

    /**
     * Fetch unified LLM run history entries, newest first.
     * @param {object} opts - { kind: "shot_action,dialogue" } — optional comma-joined kind filter
     */
    historyList({ kind = "" } = {}) {
        return this.get("/history", kind ? { kind } : {});
    }

    /** Append a history entry.  entry: the full unified run record. */
    historyAdd(entry) {
        return this.post("/history", entry);
    }

    /** Delete a history entry by numeric id. */
    historyDelete(id) {
        return this.post("/history/delete", { id });
    }

    // Legacy aliases kept for any cached clients
    describeHistoryList()    { return this.historyList({ kind: "video_describe" }); }
    describeHistoryAdd(e)    { return this.historyAdd(e); }
    describeHistoryDelete(id){ return this.historyDelete(id); }

    /** Return the auto-generated system+user prompts for a video describe call (no inference). */
    videoPrompt({ shotNumber, subjects, environment, style, intent } = {}) {
        return this.post("/video_prompt", {
            shot_number: shotNumber  ?? 1,
            subjects:    subjects    ?? [],
            environment: environment ?? "",
            style:       style       ?? "cinematic",
            intent:      intent      ?? "actions",
        });
    }

    /**
     * Streaming text generation via SSE — Unsloth backend only.
     * Calls onChunk(text) for each arriving token chunk.
     * Returns a promise that resolves when the stream ends.
     *
     * @param {string}   prompt
     * @param {object}   opts  - { system_prompt, max_tokens }
     * @param {Function} onChunk  - called with each text chunk
     * @param {Function} [onError] - called if the stream errors
     */
    async generateStream(prompt, opts = {}, onChunk, onError) {
        const res = await fetch("/fbtools/llm/generate/stream", {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({
                prompt,
                system_prompt: opts.system_prompt ?? "",
                max_tokens:    opts.max_tokens    ?? 2048,
            }),
        });
        if (!res.ok) {
            const err = new Error(`Stream request failed (HTTP ${res.status})`);
            if (onError) onError(err); else throw err;
            return;
        }
        const reader  = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        try {
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n");
                buffer = lines.pop();
                for (const line of lines) {
                    if (!line.startsWith("data: ")) continue;
                    const data = line.slice(6).trim();
                    if (data === "[DONE]") return;
                    try {
                        const chunk = JSON.parse(data);
                        if (chunk.error) { if (onError) onError(new Error(chunk.error)); return; }
                        if (chunk.text)  onChunk(chunk.text);
                    } catch { /* malformed chunk */ }
                }
            }
        } finally {
            reader.releaseLock();
        }
    }

    /**
     * Download the default recommended model.
     * Long-running — takes minutes.
     */
    downloadDefault() {
        return this.post("/download/default", {});
    }
}

export const llmApi = new LlmAPI();
