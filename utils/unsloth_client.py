"""HTTP client for the Unsloth Studio app deployed on Modal.

Unlike the Modal VisionLLM backend (modal.Cls RPC style), Unsloth Studio
exposes a plain OpenAI-compatible HTTP API at a public HTTPS URL.

Text-only — no image or video inputs.

Endpoint URLs are constructed dynamically from the user's Modal workspace name
so the client works out of the box for any user, not just the original deployer.
Call configure(workspace, api_key) at startup (extension.py does this from the
stored key and workspace resolved by modal_deploy.get_workspace()).

Cold-start protocol (confirmed live 2026-09-04):
  If the container is down, Modal's edge proxy returns HTTP 303 after ~155s.
  The token URL in Location is a one-shot that does NOT work cleanly with a
  fresh client connection; retry the *original* URL unconditionally.
  _call_with_retry loops on any non-200 (303, or a fast 400 "No model loaded"
  that fires on the very first request to a freshly booted container).

Idle scaledown: the Modal app is deployed with scaledown_window=10*60 and
min_containers=0.  activate() fires a background daemon thread that sends a
minimal warm-up ping so the container starts loading while the user works.
warmup_status transitions cold → warming → warm and is surfaced via
backend_status().
"""

from __future__ import annotations

import logging
import os
import re
import threading
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ── Endpoint definitions ──────────────────────────────────────────────────────

# Modal URL pattern: https://<workspace>--<app-name>-<function-slug>.modal.run
# Function slug = function name with underscores replaced by hyphens.
_APP_NAME = "unsloth-studio"

_ENDPOINT_SLUGS: dict[str, dict] = {
    "27b": {
        "slug":  "serve-l4-qwen3-8-27b",
        "model": "unsloth/Qwen3.8-27B-GGUF",
        "label": "Qwen3.8 27B (recommended)",
    },
    "8b": {
        "slug":  "serve-l4-qwen3-8b",
        "model": "unsloth/Qwen3-8B-GGUF",
        "label": "Qwen3 8B (fast / lower quality)",
    },
    "flash_next": {
        "slug":  "serve-l4-qwen3-8-flash-next",
        "model": "unsloth/Qwen3.8-Flash-Next-GGUF",
        "label": "Qwen3.8 Flash Next 125B MoE (slow cold start)",
    },
}

DEFAULT_ENDPOINT = "27b"


def _build_url(workspace: str, slug: str) -> str:
    return f"https://{workspace}--{_APP_NAME}-{slug}.modal.run/v1/chat/completions"


def endpoint_list(workspace: str | None = None) -> list[dict]:
    """Return endpoint descriptors with URLs resolved for `workspace`."""
    ws = workspace or _state.get("workspace") or ""
    return [
        {
            "key":   key,
            "label": ep["label"],
            "model": ep["model"],
            "url":   _build_url(ws, ep["slug"]) if ws else "",
        }
        for key, ep in _ENDPOINT_SLUGS.items()
    ]


# ── Auth ──────────────────────────────────────────────────────────────────────

def _api_key() -> str:
    # Runtime-configured key (set by configure() after bootstrap or startup load)
    k = _state.get("api_key", "").strip()
    if k:
        return k
    # Environment variable fallback
    return os.environ.get("UNSLOTH_STUDIO_API_KEY", "").strip()


# ── Module-level state ────────────────────────────────────────────────────────

_state: dict[str, Any] = {
    "active":        False,
    "endpoint_key":  DEFAULT_ENDPOINT,
    "workspace":     "",     # set by configure()
    "api_key":       "",     # set by configure()
    # cold | warming | warm | error
    "warmup_status": "cold",
    "warmup_error":  "",
}
_warmup_lock = threading.Lock()

# ── Configuration ─────────────────────────────────────────────────────────────

def configure(workspace: str = "", api_key: str = "") -> None:
    """Set workspace and API key at extension startup or after bootstrap.

    Called by extension.py on import, and again after bootstrap_key completes.
    """
    if workspace:
        _state["workspace"] = workspace.strip()
    if api_key:
        _state["api_key"] = api_key.strip()
    logger.debug(
        "Unsloth client configured: workspace=%r key=%s",
        _state["workspace"],
        "set" if _state["api_key"] else "not set",
    )

# ── HTTP call ─────────────────────────────────────────────────────────────────

_PER_ATTEMPT_TIMEOUT = 200   # seconds — long enough to survive one 303 cycle
_MAX_ATTEMPTS        = 20    # ~66 min ceiling; cold start is 2-5 min in practice
_WARMUP_MAX_TOKENS   = 1

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_reasoning(text: str) -> str:
    """Remove Qwen3 chain-of-thought <think>…</think> blocks."""
    return _THINK_RE.sub("", text).strip()


def _endpoint_url(endpoint_key: str) -> str:
    ws = _state.get("workspace", "").strip()
    if not ws:
        raise RuntimeError(
            "Unsloth workspace not configured. "
            "Set MODAL_WORKSPACE env var or authenticate with `modal token new`."
        )
    slug = _ENDPOINT_SLUGS[endpoint_key]["slug"]
    return _build_url(ws, slug)


def _call_with_retry(
    endpoint_key: str,
    payload: dict,
    status_callback: Callable[[str], None] | None = None,
) -> dict:
    """POST to the endpoint, retrying on 303 / 400 'No model loaded'.

    Returns the parsed JSON body on 200, raises on unrecoverable errors.
    """
    try:
        import httpx
    except ImportError:
        raise RuntimeError(
            "httpx is required for the Unsloth backend. "
            "Install with: pip install httpx"
        )

    ep = _ENDPOINT_SLUGS.get(endpoint_key)
    if not ep:
        raise ValueError(f"Unknown endpoint key: {endpoint_key!r}")

    url     = _endpoint_url(endpoint_key)
    key     = _api_key()
    if not key:
        raise RuntimeError(
            "Unsloth API key not set. Run the Setup step in the LLM panel, "
            "or set UNSLOTH_STUDIO_API_KEY environment variable."
        )
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type":  "application/json",
    }

    attempt = 0
    while attempt < _MAX_ATTEMPTS:
        attempt += 1
        try:
            with httpx.Client(timeout=_PER_ATTEMPT_TIMEOUT, follow_redirects=False) as client:
                r = client.post(url, headers=headers, json=payload)
        except httpx.TimeoutException:
            if status_callback:
                status_callback(
                    f"Unsloth ({ep['label']}): waiting for container "
                    f"(attempt {attempt}/{_MAX_ATTEMPTS})…"
                )
            continue
        except Exception as exc:
            raise RuntimeError(f"Unsloth HTTP error: {exc}") from exc

        if r.status_code == 200:
            return r.json()

        if r.status_code == 303:
            if status_callback:
                status_callback(
                    f"Unsloth ({ep['label']}): container starting, "
                    f"please wait… (attempt {attempt}/{_MAX_ATTEMPTS})"
                )
            continue

        # Fast 400 "No model loaded" on the very first request to a freshly
        # booted container (before the model finishes loading).
        if r.status_code == 400:
            try:
                body_text = r.text.lower()
            except Exception:
                body_text = ""
            if "no model" in body_text or "model loaded" in body_text:
                if status_callback:
                    status_callback(
                        f"Unsloth ({ep['label']}): model loading, "
                        f"retrying… (attempt {attempt}/{_MAX_ATTEMPTS})"
                    )
                continue

        r.raise_for_status()

    raise TimeoutError(
        f"Unsloth ({ep['label']}): gave up after {_MAX_ATTEMPTS} attempts. "
        "Check Modal app status with the health endpoint."
    )


# ── Warm-up ───────────────────────────────────────────────────────────────────

def _run_warmup(endpoint_key: str) -> None:
    ep    = _ENDPOINT_SLUGS.get(endpoint_key, {})
    label = ep.get("label", endpoint_key)
    with _warmup_lock:
        _state["warmup_status"] = "warming"
        _state["warmup_error"]  = ""

    payload = {
        "model":      ep.get("model", ""),
        "messages":   [{"role": "user", "content": "ping"}],
        "max_tokens": _WARMUP_MAX_TOKENS,
    }
    try:
        _call_with_retry(endpoint_key, payload)
        with _warmup_lock:
            _state["warmup_status"] = "warm"
        logger.info("Unsloth warm-up complete: %s", label)
    except Exception as exc:
        with _warmup_lock:
            _state["warmup_status"] = "error"
            _state["warmup_error"]  = str(exc)
        logger.warning("Unsloth warm-up failed (%s): %s", label, exc)


def _start_warmup(endpoint_key: str) -> None:
    t = threading.Thread(target=_run_warmup, args=(endpoint_key,), daemon=True)
    t.start()


# ── Public API ────────────────────────────────────────────────────────────────

def backend_status() -> dict:
    """Return Unsloth backend state (mirrors modal_vision_client shape)."""
    ep_key = _state["endpoint_key"]
    ep     = _ENDPOINT_SLUGS.get(ep_key, {})
    ws     = _state.get("workspace", "")
    with _warmup_lock:
        ws_status = _state["warmup_status"]
        ws_error  = _state["warmup_error"]
    return {
        "active":         _state["active"],
        "endpoint_key":   ep_key,
        "endpoint_label": ep.get("label", ep_key),
        "endpoint_url":   _build_url(ws, ep["slug"]) if ws else "",
        "model":          ep.get("model", ""),
        "workspace":      ws,
        "workspace_set":  bool(ws),
        "api_key_set":    bool(_api_key()),
        "warmup_status":  ws_status,
        "warmup_error":   ws_error,
        "endpoints":      endpoint_list(ws),
    }


def activate(endpoint_key: str = DEFAULT_ENDPOINT) -> dict:
    """Mark the Unsloth backend active and kick off a background warm-up.

    Returns {success, message}.
    """
    if endpoint_key not in _ENDPOINT_SLUGS:
        return {
            "success": False,
            "message": f"Unknown endpoint {endpoint_key!r}. Valid: {list(_ENDPOINT_SLUGS)}",
        }

    ws = _state.get("workspace", "")
    if not ws:
        return {
            "success": False,
            "message": (
                "Workspace not configured. "
                "Set MODAL_WORKSPACE env var, authenticate with `modal token new`, "
                "or deploy the app from the LLM panel."
            ),
        }

    if not _api_key():
        return {
            "success": False,
            "message": (
                "API key not set. "
                "Run Setup in the LLM panel to bootstrap the key, "
                "or set UNSLOTH_STUDIO_API_KEY environment variable."
            ),
        }

    _state["active"]       = True
    _state["endpoint_key"] = endpoint_key
    with _warmup_lock:
        _state["warmup_status"] = "cold"
        _state["warmup_error"]  = ""

    ep = _ENDPOINT_SLUGS[endpoint_key]
    _start_warmup(endpoint_key)
    logger.info("Unsloth backend activated: %s — warm-up started", ep["label"])
    return {
        "success": True,
        "message": (
            f"Unsloth activated ({ep['label']}). "
            "Container warm-up started in background — check status for readiness."
        ),
    }


def deactivate() -> dict:
    """Deactivate the Unsloth backend (clears local state only)."""
    _state["active"] = False
    with _warmup_lock:
        _state["warmup_status"] = "cold"
        _state["warmup_error"]  = ""
    logger.info("Unsloth backend deactivated")
    return {"success": True, "message": "Unsloth backend deactivated."}


def is_active() -> bool:
    return bool(_state["active"])


def health_check(endpoint_key: str | None = None) -> dict:
    """Fast probe: check if the endpoint's container is warm.

    Uses a 10s timeout — does NOT follow the full cold-start retry loop.
    Returns {status: warm|starting|down|error, message}.
    """
    try:
        import httpx
    except ImportError:
        return {"status": "error", "message": "httpx not installed"}

    key = endpoint_key or _state["endpoint_key"]
    ep  = _ENDPOINT_SLUGS.get(key)
    if not ep:
        return {"status": "error", "message": f"Unknown endpoint {key!r}"}

    ws = _state.get("workspace", "")
    if not ws:
        return {"status": "error", "message": "Workspace not configured."}

    api_key = _api_key()
    if not api_key:
        return {"status": "error", "message": "API key not set."}

    url     = _build_url(ws, ep["slug"])
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": ep["model"], "messages": [{"role": "user", "content": "ping"}],
               "max_tokens": 1}
    try:
        with httpx.Client(timeout=10, follow_redirects=False) as client:
            r = client.post(url, headers=headers, json=payload)
        if r.status_code == 200:
            with _warmup_lock:
                _state["warmup_status"] = "warm"
            return {"status": "warm", "message": "Container is ready."}
        if r.status_code == 303:
            return {"status": "starting", "message": "Container is booting (HTTP 303)."}
        if r.status_code == 400:
            body = r.text.lower()
            if "no model" in body or "model loaded" in body:
                return {"status": "starting", "message": "Container up, model still loading."}
        return {"status": "error", "message": f"Unexpected status {r.status_code}"}
    except httpx.TimeoutException:
        return {"status": "starting", "message": "Probe timed out — container may be booting."}
    except Exception as exc:
        return {"status": "down", "message": f"Connection failed: {exc}"}


def generate(
    prompt: str,
    *,
    system_prompt: str = "",
    max_tokens: int = 512,
    temperature: float = 0.7,
    status_callback: Callable[[str], None] | None = None,
) -> dict:
    """Generate text via the Unsloth Studio endpoint.

    Text-only — no image or video inputs.  Returns {success, text, message}.

    max_tokens should be ≥300 when using Qwen3 models: they emit a
    <think>…</think> reasoning block first; if max_tokens is too small the
    response can be all reasoning with empty content.  The reasoning block
    is stripped before returning.
    """
    if not _state["active"]:
        return {"success": False, "text": "", "message": "Unsloth backend is not active."}

    ep_key = _state["endpoint_key"]
    ep     = _ENDPOINT_SLUGS.get(ep_key)
    if not ep:
        return {"success": False, "text": "", "message": f"Unknown endpoint {ep_key!r}"}

    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload: dict = {
        "model":       ep["model"],
        "messages":    messages,
        "max_tokens":  max_tokens,
        "temperature": temperature,
    }

    if status_callback:
        ws = _state.get("warmup_status", "cold")
        if ws != "warm":
            status_callback(
                f"Calling Unsloth ({ep['label']}) — container may need 2-5 min if cold"
            )

    try:
        data = _call_with_retry(ep_key, payload, status_callback=status_callback)
        raw  = data["choices"][0]["message"].get("content") or ""
        text = _strip_reasoning(raw)
        if status_callback:
            status_callback(f"Unsloth ({ep['label']}) responded")
        with _warmup_lock:
            _state["warmup_status"] = "warm"
        return {"success": True, "text": text, "message": ""}
    except Exception as exc:
        logger.error("Unsloth generate failed: %s", exc)
        return {"success": False, "text": "", "message": str(exc)}
