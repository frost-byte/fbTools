"""Submit an API-format ComfyUI prompt to this same server and await its result.

Server-to-server: POSTs to this server's own /prompt (no client_id) and polls /history/{prompt_id}
until the job completes, fails, or times out. This does NOT hide the job from every open browser
tab — ComfyUI broadcasts execution/progress websocket events to all connected clients regardless of
which client_id (if any) submitted the prompt, so a global indicator (e.g. a top-of-page progress
bar from some custom node) will still show activity. What stays untouched is canvas-specific
rendering (node highlighting, per-node live preview overlays), since that's driven by matching node
ids against whatever graph happens to be loaded in that tab — our submitted prompt's node ids come
from the template file, not the tab's own canvas. Mirrors the request/response
shapes of ComfyUI's own server.py (POST /prompt, GET /history/{prompt_id}) and execution.py
(PromptQueue.task_done's history entry shape) — verified against the installed ComfyUI source
rather than assumed.
"""
from __future__ import annotations

import asyncio

import aiohttp

from .logging_utils import get_logger

logger = get_logger(__name__)


class H3JobError(Exception):
    """Raised when a submitted prompt is rejected, fails during execution, or times out."""


async def submit_and_wait(
    request,
    prompt: dict,
    save_node_id: str,
    *,
    timeout: float = 240.0,
    poll_interval: float = 1.5,
) -> dict:
    """Submit `prompt` (an API-format dict) and wait for `save_node_id`'s output image.

    `request` is the inbound aiohttp request of the route calling this — used only to derive this
    server's own base URL (scheme + host), never inspected otherwise.

    Returns {"filename": str, "subfolder": str, "type": str} for the first image `save_node_id`
    produced. Raises H3JobError with a clear message on rejection, execution failure, or timeout.
    """
    base_url = f"{request.scheme}://{request.host}"

    async with aiohttp.ClientSession() as session:
        async with session.post(f"{base_url}/prompt", json={"prompt": prompt}) as resp:
            body = await resp.json()
            if resp.status != 200:
                error = body.get("error", body)
                raise H3JobError(f"Prompt submission rejected: {error}")
            prompt_id = body["prompt_id"]

        logger.debug("h3_job_runner: submitted prompt_id=%s, polling /history", prompt_id)

        elapsed = 0.0
        while elapsed < timeout:
            async with session.get(f"{base_url}/history/{prompt_id}") as resp:
                history = await resp.json()

            entry = history.get(prompt_id)
            if entry is not None:
                # A history entry only appears once the whole prompt has finished (completed or
                # errored) — execution.py's task_done() writes it in one shot, never incrementally
                # per node — so by the time we see it, outputs are as complete as they'll ever be.
                status = entry.get("status") or {}
                if status.get("status_str") == "error":
                    messages = status.get("messages", [])
                    raise H3JobError(f"H3 generation failed: {messages}")

                outputs = entry.get("outputs", {})
                save_output = outputs.get(save_node_id)
                if save_output is None:
                    raise H3JobError(
                        f"Save node {save_node_id!r} produced no output — check the template's "
                        f"OUT:save node id/title still matches a SaveImage node"
                    )
                images = save_output.get("images", [])
                if not images:
                    raise H3JobError(f"Save node {save_node_id!r} produced no images")
                return images[0]

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

    raise H3JobError(f"Timed out after {timeout}s waiting for prompt_id={prompt_id}")


async def free_vram(request, *, unload_models: bool = True, free_memory: bool = True) -> None:
    """Ask this same ComfyUI server to unload resident models / free memory.

    Same mechanism as the Manager UI's "Free model and node cache" button — POSTs to this server's
    own core /free endpoint (server.py's post_free), which just sets a flag the main execution loop
    checks after finishing the current/next queued item (execution.py, main.py's run loop); it does
    not happen synchronously inside this call. Best-effort: logs and swallows any failure rather
    than raising, since this is a courtesy cleanup step, not something that should turn a successful
    generation into a reported error.
    """
    base_url = f"{request.scheme}://{request.host}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{base_url}/free",
                json={"unload_models": unload_models, "free_memory": free_memory},
            ) as resp:
                if resp.status != 200:
                    logger.warning("h3_job_runner.free_vram: /free returned status %s", resp.status)
    except Exception as exc:
        logger.warning("h3_job_runner.free_vram: request failed: %s", exc)
