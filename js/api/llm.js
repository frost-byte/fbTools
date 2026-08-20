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
     * Download the default recommended model.
     * Long-running — takes minutes.
     */
    downloadDefault() {
        return this.post("/download/default", {});
    }
}

export const llmApi = new LlmAPI();
