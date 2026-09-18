"""Run History tracking: RunMetaCapture/JobCompleteNotifier nodes, the
node-output auto-tracker, and the /fbtools/run_tracker/* endpoints.

Moved out of extension.py as the first step of splitting that file into
per-domain modules (see docs/ plan). Self-contained: nothing outside this
module reads its globals or calls its helpers directly.
"""
import os
import time
import uuid

import torch
from comfy_api.latest import io
from server import PromptServer
from aiohttp import web

from .shared import prefixed_node_id
from ..utils.logging_utils import get_logger
from ..utils.node_output_tracker import extract_tracked_nodes, stringify_capture_values

logger = get_logger(__name__)

routes = PromptServer.instance.routes

# Runtime value store for RunMetaCapture nodes — keyed by prompt_id
_RUN_CAPTURE_STORE: dict = {}
_RUN_CAPTURE_MAX: int = 100

# Node-output auto-tracker (see "on_prompt handler" section below):
# {prompt_id: {node_id: label}} for every [track: Label]-tagged node seen in
# a submitted prompt; {node_id: {label, class_type, values}} of the last
# successfully-resolved kwargs per tracked node, kept by node_id (not
# prompt_id) so a later cache-hit run can re-emit it; and
# {prompt_id: {node_id, ...}} of nodes already captured for the current
# prompt, so the cache-hit backfill never double-emits one that genuinely
# executed this run.
_TRACKED_NODES_BY_PROMPT: dict = {}
_TRACKED_KWARGS_BY_NODE_ID: dict = {}
_TRACKED_EMITTED_BY_PROMPT: dict = {}
_TRACKED_PROMPT_MAX: int = 100


def _get_current_prompt_id() -> str | None:
    """Return the prompt_id of the currently executing job, or None."""
    try:
        running = PromptServer.instance.prompt_queue.currently_running
        for item in running.values():
            if isinstance(item, (list, tuple)) and len(item) > 1:
                pid = item[1]
                if isinstance(pid, str) and len(pid) > 4:
                    return pid
    except Exception:
        pass
    return None


def _record_capture(prompt_id: str | None, label: str, values: dict) -> None:
    """Shared storage for RunMetaCapture and the node-output auto-tracker below —
    both write into the same run-history-facing store, keyed by prompt_id."""
    if not prompt_id:
        return
    if prompt_id not in _RUN_CAPTURE_STORE:
        _RUN_CAPTURE_STORE[prompt_id] = {
            "prompt_id": prompt_id,
            "ts": time.time() * 1000,  # ms, consistent with ComfyUI history
            "captures": [],
        }
        while len(_RUN_CAPTURE_STORE) > _RUN_CAPTURE_MAX:
            _RUN_CAPTURE_STORE.pop(next(iter(_RUN_CAPTURE_STORE)))
    _RUN_CAPTURE_STORE[prompt_id]["captures"].append({"label": label, "values": values})


# ── Node-output auto-tracker ────────────────────────────────────────────────
#
# Problem: [track: Label] on a node only lets Run History read *literal*
# widget values from the static submitted prompt (see run_history.js) — a
# value that arrives over a wire from a node that genuinely computes it
# (not a trivial passthrough) can't be inferred without running the graph.
# RunMetaCapture solves that but requires wiring a dedicated node downstream.
#
# Two earlier attempts at the cache-hit gap, both wrong about where the
# decision actually happens:
#   1. Monkeypatching each tracked node class's own dispatch function to
#      capture kwargs "the moment it runs" — but that moment never arrives
#      on a cache hit inside execution.execute() itself.
#   2. Wrapping execution.execute() to call get_input_data() before the
#      real cache check — but a cached *intermediate* node (a dependency of
#      something else, not itself a requested output) is pruned from the
#      schedule entirely by comfy_execution.graph's TopologicalSort/
#      ExecutionList.add_node()/add_strong_link() based on is_cached(), so
#      execute() is never even called for it. There's no way to intercept
#      that safely — is_cached() is called from deep inside the scheduler's
#      own live traversal, and reacting to it would mean reentrantly
#      mutating the same pendingNodes/blockCount bookkeeping that traversal
#      is actively building.
#
# Fix: don't try to catch the node executing — read what it already
# produced. caches.outputs (comfy_execution/caching.py's BasicCache/
# HierarchicalCache) is a plain persistent object, completely separate from
# the scheduler's traversal state, and its .get()/._get_immediate() is a
# pure cache-key dict lookup with no eviction or consumption side effects —
# safe to call redundantly at any time. So: whenever a tracked node DOES
# genuinely execute (cache miss), stash its resolved kwargs by node_id (not
# prompt_id — it must survive into later, separate prompt submissions).
# On every subsequent prompt, for any tracked node that didn't execute this
# run, check caches.outputs.get(node_id) directly — if it still holds a
# value, that's a real cache hit, and by definition of what "cached" means
# its inputs are unchanged since we last captured them, so the stashed
# kwargs are still correct. Re-emit them for the new prompt_id.
try:
    import execution as _fbt_execution_module
    from execution import get_input_data as _fbt_get_input_data
    _NODE_OUTPUT_TRACKER_AVAILABLE = True
except Exception:
    _fbt_execution_module = None
    _fbt_get_input_data = None
    _NODE_OUTPUT_TRACKER_AVAILABLE = False


def _fbtools_is_capturable_value(v) -> bool:
    """Reject values that will never be human-readable once stringified —
    tensors, wrapped LATENT/AUDIO dicts, and other comfy/torch-internal
    objects (e.g. a ModelPatcher) — so a tracked node's genuinely useful
    scalar/string inputs aren't drowned out by a multi-KB tensor repr."""
    try:
        if torch is not None and isinstance(v, torch.Tensor):
            return False
        if isinstance(v, dict) and ("samples" in v or "waveform" in v):
            return False  # LATENT / AUDIO wrapper dicts
        mod = type(v).__module__ or ""
        if mod.startswith("comfy") or mod.startswith("torch"):
            return False
    except Exception:
        return True
    return True


def _fbtools_capture_for_node(prompt_id, unique_id, dynprompt, execution_list, extra_data) -> None:
    """Resolve and record a tracked node's current input values. Called from
    the wrapped execution.execute() whenever a tracked node genuinely runs
    (cache miss). Isolated in its own try/except so a failure here can never
    affect the node's actual execution or return value."""
    try:
        tracked = _TRACKED_NODES_BY_PROMPT.get(prompt_id)
        if not tracked:
            return
        label = tracked.get(unique_id)
        if label is None:
            return
        node = dynprompt.get_node(unique_id)
        class_type = node["class_type"]
        import nodes as _fbt_nodes_module
        class_def = _fbt_nodes_module.NODE_CLASS_MAPPINGS.get(class_type)
        if class_def is None:
            return
        input_data_all, _missing, _v3 = _fbt_get_input_data(
            node["inputs"], class_def, unique_id, execution_list, dynprompt, extra_data,
        )
        kwargs = {}
        for key, wrapped in input_data_all.items():
            if not wrapped:
                continue
            value = wrapped[0]
            if _fbtools_is_capturable_value(value):
                kwargs[key] = value
        values = stringify_capture_values(kwargs)
        if not values:
            # Every input got filtered out (e.g. a pure model-patch node whose
            # only input is a MODEL/ModelPatcher) — record a placeholder so
            # tagging still confirms the node genuinely ran this prompt,
            # rather than silently producing no Run History entry at all,
            # which looks identical to "the tag isn't working".
            values = {"status": f"executed ({len(input_data_all)} input(s), none displayable)"
                                 if input_data_all else "executed (no inputs)"}
        _record_capture(prompt_id, label, values)
        _TRACKED_KWARGS_BY_NODE_ID[unique_id] = {
            "label": label, "class_type": class_type, "values": values,
        }
        while len(_TRACKED_KWARGS_BY_NODE_ID) > _TRACKED_PROMPT_MAX:
            _TRACKED_KWARGS_BY_NODE_ID.pop(next(iter(_TRACKED_KWARGS_BY_NODE_ID)))
            _TRACKED_EMITTED_BY_PROMPT.setdefault(prompt_id, set()).add(unique_id)
    except Exception:
        logger.debug("node-output-tracker: capture failed", exc_info=True)


async def _fbtools_backfill_cached_tracked_nodes(prompt_id, dynprompt, caches) -> None:
    """For every tracked node in this prompt that didn't genuinely execute
    (so _fbtools_capture_for_node above never ran for it — a real cache
    hit, pruned from the schedule before execute() was ever called), check
    whether caches.outputs still holds a value for it. If so, that's
    confirmation the cache hit is real, and by definition its inputs are
    identical to whenever we last captured them — re-emit that stash rather
    than trying to re-derive kwargs that are no longer resolvable. Isolated
    in its own try/except; a pure-read lookup, never touches scheduler
    state."""
    try:
        tracked = _TRACKED_NODES_BY_PROMPT.get(prompt_id)
        if not tracked:
            return
        emitted = _TRACKED_EMITTED_BY_PROMPT.setdefault(prompt_id, set())
        for node_id, label in tracked.items():
            if node_id in emitted:
                continue
            stashed = _TRACKED_KWARGS_BY_NODE_ID.get(node_id)
            if not stashed:
                continue
            node = dynprompt.get_node(node_id)
            if not node or node.get("class_type") != stashed["class_type"]:
                continue  # node_id reused for a different node — stash doesn't apply
            cached = await caches.outputs.get(node_id)
            if cached is not None:
                _record_capture(prompt_id, stashed["label"], stashed["values"])
                emitted.add(node_id)
    except Exception:
        logger.debug("node-output-tracker: cache-hit backfill failed", exc_info=True)


def _fbtools_install_execute_patch() -> None:
    """Wrap the single shared execution.execute() entry point once. Never
    alters its arguments or return value — only adds a best-effort capture
    step beforehand for tracked nodes. Idempotent; safe to call repeatedly."""
    if not _NODE_OUTPUT_TRACKER_AVAILABLE:
        return
    if getattr(_fbt_execution_module.execute, "_fbtools_wrapped", False):
        return
    _original_execute = _fbt_execution_module.execute

    async def _fbtools_wrapped_execute(*args, **kwargs):
        try:
            # execute(server, dynprompt, caches, current_item, extra_data, executed,
            #         prompt_id, execution_list, pending_subgraph_results,
            #         pending_async_nodes, ui_outputs) — positional in this ComfyUI
            # version (execution.py); read defensively so a future signature change
            # just disables capture instead of breaking generation.
            dynprompt = args[1] if len(args) > 1 else kwargs.get("dynprompt")
            caches = args[2] if len(args) > 2 else kwargs.get("caches")
            current_item = args[3] if len(args) > 3 else kwargs.get("current_item")
            extra_data = args[4] if len(args) > 4 else kwargs.get("extra_data")
            prompt_id = args[6] if len(args) > 6 else kwargs.get("prompt_id")
            execution_list = args[7] if len(args) > 7 else kwargs.get("execution_list")
            if prompt_id in _TRACKED_NODES_BY_PROMPT:
                _fbtools_capture_for_node(prompt_id, current_item, dynprompt, execution_list, extra_data)
                if caches is not None:
                    await _fbtools_backfill_cached_tracked_nodes(prompt_id, dynprompt, caches)
        except Exception:
            logger.debug("node-output-tracker: pre-execute hook failed", exc_info=True)
        return await _original_execute(*args, **kwargs)

    _fbtools_wrapped_execute._fbtools_wrapped = True
    _fbt_execution_module.execute = _fbtools_wrapped_execute


_fbtools_install_execute_patch()


def _fbtools_on_prompt_handler(json_data: dict) -> dict:
    """Registered via PromptServer.add_on_prompt_handler — runs on every
    submission, before execution starts. Discovers [track: Label]-tagged
    nodes for *this* prompt so the wrapped execute() above knows which
    node_ids to capture."""
    try:
        prompt = json_data.get("prompt") if isinstance(json_data, dict) else None
        if not isinstance(prompt, dict):
            return json_data
        tracked = extract_tracked_nodes(prompt)
        if not tracked:
            return json_data

        # Scope the tracked-id map by prompt_id so queued-ahead jobs can't
        # clobber each other's tracked-node labels. The client usually
        # supplies one; if not, mint it now (mirrors what server.py would do
        # a few lines later anyway) so we can correlate at execution time.
        prompt_id = json_data.get("prompt_id")
        if not prompt_id:
            prompt_id = str(uuid.uuid4())
            json_data["prompt_id"] = prompt_id

        _TRACKED_NODES_BY_PROMPT[prompt_id] = tracked
        while len(_TRACKED_NODES_BY_PROMPT) > _TRACKED_PROMPT_MAX:
            evicted_id = next(iter(_TRACKED_NODES_BY_PROMPT))
            _TRACKED_NODES_BY_PROMPT.pop(evicted_id)
            _TRACKED_EMITTED_BY_PROMPT.pop(evicted_id, None)
    except Exception:
        logger.debug("node-output-tracker: on_prompt handling failed", exc_info=True)
    return json_data


PromptServer.instance.add_on_prompt_handler(_fbtools_on_prompt_handler)


# ── Run Tracker endpoints ──────────────────────────────────────────────────────

@routes.get("/fbtools/run_tracker/runs")
async def _run_tracker_runs(request):
    runs = sorted(_RUN_CAPTURE_STORE.values(), key=lambda r: r["ts"], reverse=True)
    return web.json_response({"runs": runs})


@routes.delete("/fbtools/run_tracker/clear")
async def _run_tracker_clear(request):
    _RUN_CAPTURE_STORE.clear()
    return web.json_response({"success": True})


# ── Node: RunMetaCapture ───────────────────────────────────────────────────────


class RunMetaCapture(io.ComfyNode):
    """
    Captures runtime values from wired inputs at execution time and stores them
    for the Run History sidebar panel.  Wire any STRING output into a value slot;
    slots grow automatically as you connect them.  Values appear in Run History
    linked to this job's prompt ID.

    As an output node it supports partial execution: select it and click the
    play button to run only the upstream subgraph that feeds it.
    """

    @classmethod
    def define_schema(cls):
        autogrow_template = io.Autogrow.TemplatePrefix(
            input=io.String.Input("value", display_name="Value", default="", optional=True),
            prefix="value",
            min=1,
            max=12,
        )
        return io.Schema(
            node_id=prefixed_node_id("RunMetaCapture"),
            display_name="Run Meta Capture",
            category="🧊 frost-byte/Nodes",
            description="Capture runtime string values for the Run History panel.",
            is_output_node=True,
            outputs=[],
            inputs=[
                io.String.Input("label", display_name="Label", default="capture"),
                io.Autogrow.Input("values", template=autogrow_template),
            ],
        )

    @classmethod
    def execute(cls, label: str = "capture", values: io.Autogrow.Type = None):
        captured: dict = {}
        if values:
            captured = stringify_capture_values(
                {f"Value {i}": v for i, v in enumerate(values.values(), 1)}
            )

        _record_capture(_get_current_prompt_id(), label, captured)

        preview_lines = [f"{k}: {v}" for k, v in captured.items()]
        preview_text = "\n".join(preview_lines) if preview_lines else "(nothing captured)"
        return io.NodeOutput(ui={"text": [preview_text]})


# ── Job Complete Notifier ──────────────────────────────────────────────────────

_NOTIFY_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "user", "default", "fbtools", "notifications",
)
_NOTIFY_DIR = os.path.normpath(_NOTIFY_DIR)
os.makedirs(_NOTIFY_DIR, exist_ok=True)


class JobCompleteNotifier(io.ComfyNode):
    """
    Output node — place at the end of any workflow to emit a completion
    notification that Claude Code (or any file watcher) can pick up.

    Writes a small JSON file to:
        ComfyUI/user/default/fbtools/notifications/<uuid>.json

    The file is picked up by the active Claude Code session watcher, which
    fires a push notification to your phone and then purges the file.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=prefixed_node_id("JobCompleteNotifier"),
            display_name="Job Complete Notifier",
            category="🧊 frost-byte/Nodes",
            description="Write a completion signal file so Claude Code can push a notification when this workflow finishes.",
            is_output_node=True,
            outputs=[],
            inputs=[
                io.String.Input(
                    "label",
                    display_name="Label",
                    default="Workflow complete",
                    tooltip="Short description included in the notification.",
                ),
                io.String.Input(
                    "details",
                    display_name="Details",
                    default="",
                    optional=True,
                    tooltip="Extra info (e.g. checkpoint name, prompt snippet). Optional.",
                ),
            ],
        )

    @classmethod
    def execute(cls, label: str = "Workflow complete", details: str = ""):
        import uuid as _uuid
        import datetime as _dt
        import json as _json

        payload = {
            "id":        str(_uuid.uuid4()),
            "timestamp": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
            "label":     label,
            "details":   details,
        }
        fname = os.path.join(_NOTIFY_DIR, f"{payload['id']}.json")
        try:
            with open(fname, "w", encoding="utf-8") as fh:
                _json.dump(payload, fh)
            logger.info("JobCompleteNotifier: wrote %s", fname)
        except Exception as exc:
            logger.warning("JobCompleteNotifier: could not write notification file: %s", exc)

        return io.NodeOutput()
