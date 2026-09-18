# Action Plan: Node-Output Tracking System for ComfyUI Custom Nodes

## Objective
Design and implement a system, living in a custom_nodes package, that lets
specific nodes be marked "tracked" and makes their computed output values
retrievable by node ID during or after workflow execution — without polling,
using a push-based Observer pattern (Subject = the point where a value is
resolved; Observers = whatever reads the tracked registry).

## Background / what's already been established (do not re-litigate these)
- Custom node packages load into the **same Python process** as the ComfyUI
  server at startup (via their `__init__.py`), so no IPC/network is needed —
  a module-level Python object is sufficient as the shared "Subject" state.
- ComfyUI's own documented client-facing mechanisms (the `executed` WebSocket
  message, and `GET /history/{prompt_id}["outputs"][node_id]`) only surface a
  node's output if that node explicitly returns a `"ui"` payload
  (`return {"ui": {...}, "result": (...)}`). Neither is a generic dump of
  every node's raw computed value. This is a real, confirmed limitation —
  don't try to route around it via those two channels for nodes that don't
  already emit `ui` data.
- There are two genuinely different implementation cases with different risk
  profiles:
  1. **Tracked node is code we write/control** — trivial, no patching needed.
  2. **Tracked node is arbitrary/third-party, unmodified** — requires
     intercepting the executor's per-node call site (harder, version-fragile).

## Phase 0 — Environment recon (do this before writing any code)
The exact internal call chain and function signatures in `execution.py` have
changed across ComfyUI versions (confirmed: `get_output_data` gained a
`v3_data` parameter at some point between releases; line numbers and
signatures differ across recent GitHub issue tracebacks). **Do not assume
the signatures below are current for the installed version** — verify first.

```bash
# Locate the actual ComfyUI install and confirm version
find / -maxdepth 6 -iname "execution.py" -path "*ComfyUI*" 2>/dev/null
cd <comfyui_root>
git log -1 --oneline  # or check the version string ComfyUI logs on startup
```

Inspect the current shape of the relevant functions:
```bash
grep -n "def execute" execution.py
grep -n "def get_output_data" execution.py
grep -n "def _async_map_node_over_list" execution.py
grep -n "def process_inputs" execution.py
grep -n "class PromptExecutor" execution.py
grep -n "self.caches" execution.py
grep -n "unique_id" execution.py | head -30
```
Also check `comfy_execution/caching.py` for the outputs-cache implementation,
and confirm whether the "cache provider / event" callback pattern
(`_cache_logger.warning(f"Cache provider {provider.__class__.__name__} error
on {event}: {e}")`, `ram_release_callback`) is still present and whether it's
importable/extendable from outside `execution.py`, since it's architecturally
the closest existing precedent to what we want.

Deliverable: a short `RECON.md` noting the exact current signature of every
function you intend to touch, and the exact file/line to patch.

## Phase 1 — Implement the easy case first (owned/wrapped nodes)
No monkey-patching required. Build:

1. A shared module (e.g. `tracking_registry.py`) exposing:
   ```python
   TRACKED_IDS: set[str] = set()
   TRACKED_OUTPUTS: dict[str, Any] = {}
   ```
2. A mixin or base class tracked nodes can use, which writes into
   `TRACKED_OUTPUTS[node_id]` at the moment their `FUNCTION` computes its
   result, using ComfyUI's hidden `unique_id` input convention to get the
   node's own ID without any patching:
   ```python
   class TrackedNodeMixin:
       @classmethod
       def INPUT_TYPES(cls):
           types = super().INPUT_TYPES()
           types.setdefault("hidden", {})["unique_id"] = "UNIQUE_ID"
           return types
   ```
3. Optionally, also return a `"ui"` payload from these nodes so their value
   *also* shows up in `/history/{prompt_id}["outputs"][node_id]` for free —
   this reuses ComfyUI's existing transport rather than inventing a new one.
4. Expose `TRACKED_OUTPUTS` to the outside world via a small route registered
   on `PromptServer.instance.routes` (e.g. `GET /tracked/{node_id}`), if
   external (not just in-process) access is wanted.

This phase should be fully working and tested before touching Phase 2 —
it covers every case where the user is willing to write/wrap the node.

## Phase 2 — Only if needed: tracking arbitrary third-party nodes
This is the case where the node's source isn't being modified — tracking
must be done by whoever calls the node's function.

**Design constraint, non-negotiable — verify the implementation actually
meets this before considering it done:**
- Call the original function completely generically. Do not assume a fixed
  positional signature; pass through with `*args, **kwargs` and never
  destructure them.
- Isolate all tracking-specific logic (looking up whether `node_id` is
  tracked, parsing/storing the result) inside its own `try/except` block
  that **cannot** affect the return value or re-raise.
- The wrapped function must return exactly what the original returned, on
  every path, including when tracking logic throws.
- Write a test that deliberately breaks the tracking-extraction logic (e.g.
  monkeypatch `extract_value` to raise) and confirms: (a) the workflow still
  completes normally end-to-end, (b) the tracked value is simply absent, (c)
  no exception propagates out of node execution.

Skeleton (confirm exact target function name/signature against Phase 0's
recon output before finalizing):
```python
import logging
log = logging.getLogger("tracking")

def install_patch():
    import execution
    original_fn = execution.get_output_data  # confirm this is still the right target

    async def wrapped(*args, **kwargs):
        result = await original_fn(*args, **kwargs)
        try:
            node_id = _extract_node_id(args, kwargs)  # implement against actual signature
            if node_id in TRACKED_IDS:
                TRACKED_OUTPUTS[node_id] = _extract_value(result)
        except Exception as e:
            log.warning(f"tracking failed for node {node_id if 'node_id' in dir() else '?'}: {e}")
        return result

    execution.get_output_data = wrapped
```
Wrap `install_patch()` itself in a try/except at custom-node import time so a
failed patch (function renamed/moved in a future ComfyUI version) degrades to
"tracking feature inactive" rather than breaking custom node loading.

## Phase 3 — Decide feasibility and report back
Before writing production code, produce a short `FEASIBILITY.md` answering:
- Does Phase 1 alone (owned/wrapped nodes only) satisfy the actual use case?
  If yes, **stop here and skip Phase 2 entirely** — it's strictly safer and
  requires no internals-patching risk.
- If Phase 2 is genuinely required (tracking nodes we can't modify), what
  exact function is the safest interception point given the current
  installed version's real call chain from Phase 0's recon — and what's the
  blast radius if that function is on the hot path for *every* node
  (it likely is, since this call site is shared across all node types)?
- Confirm the isolation test from Phase 2 passes before calling this done.

## Deliverables
1. `RECON.md` — current function signatures/line numbers for the installed version.
2. `tracking_registry.py` + `TrackedNodeMixin` (Phase 1), working and tested.
3. If pursued, the Phase 2 patch module, with the deliberate-failure test.
4. `FEASIBILITY.md` — final recommendation on whether Phase 2 is warranted.
