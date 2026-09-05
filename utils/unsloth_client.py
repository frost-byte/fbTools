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

import httpx

logger = logging.getLogger(__name__)

# ── Endpoint definitions ──────────────────────────────────────────────────────

# Modal URL pattern: https://<workspace>--<app-name>-<function-slug>.modal.run
# Function slug = function name with underscores replaced by hyphens.
_APP_NAME = "unsloth-studio"

_ENDPOINT_SLUGS: dict[str, dict] = {
    "27b": {
        "slug":         "serve-l4-qwen3-8-27b",
        "model":        "unsloth/Qwen3.8-27B-GGUF",
        "label":        "Qwen3.8 27B (recommended)",
        "vision":       True,   # VLM; mmproj-F16.gguf loaded at container start
        "native_video": True,
    },
    "8b": {
        "slug":         "serve-l4-qwen3-8b",
        "model":        "unsloth/Qwen3-8B-GGUF",
        "label":        "Qwen3 8B (fast / text-only)",
        "vision":       False,
        "native_video": False,
    },
    "flash_next": {
        "slug":         "serve-l4-qwen3-8-flash-next",
        "model":        "unsloth/Qwen3.8-Flash-Next-GGUF",
        "label":        "Qwen3.8 Flash Next 125B MoE (slow cold start)",
        "vision":       True,   # VLM; mmproj-F16.gguf loaded at container start
        "native_video": True,
    },
}

DEFAULT_ENDPOINT = "27b"


def _build_base_url(workspace: str, slug: str) -> str:
    return f"https://{workspace}--{_APP_NAME}-{slug}.modal.run"


def _build_url(workspace: str, slug: str) -> str:
    return f"{_build_base_url(workspace, slug)}/v1/chat/completions"


def endpoint_list(workspace: str | None = None) -> list[dict]:
    """Return endpoint descriptors with URLs resolved for `workspace`."""
    ws = workspace or _state.get("workspace") or ""
    return [
        {
            "key":          key,
            "label":        ep["label"],
            "model":        ep["model"],
            "vision":       ep.get("vision", False),
            "native_video": ep.get("native_video", False),
            "url":          _build_url(ws, ep["slug"]) if ws else "",
        }
        for key, ep in _ENDPOINT_SLUGS.items()
    ]


def active_endpoint_supports_vision() -> bool:
    """True when the currently selected endpoint is a vision-language model."""
    ep = _ENDPOINT_SLUGS.get(_state["endpoint_key"], {})
    return bool(ep.get("vision", False))


def active_endpoint_supports_native_video() -> bool:
    """True when the currently selected endpoint supports native video frames."""
    ep = _ENDPOINT_SLUGS.get(_state["endpoint_key"], {})
    return bool(ep.get("native_video", False))


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
    "warmup_phase":  "",     # human-readable phase updated by the warmup thread
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

_PER_ATTEMPT_TIMEOUT  = 200   # seconds — long enough to survive one 303 cycle
_MAX_ATTEMPTS         = 20    # ~66 min ceiling; cold start is 2-5 min in practice
_WARMUP_MAX_TOKENS    = 1
_GENERATE_TIMEOUT     = 180   # seconds — single-shot inference timeout (no retry)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_reasoning(text: str) -> str:
    """Remove Qwen3 chain-of-thought <think>…</think> blocks."""
    return _THINK_RE.sub("", text).strip()


def _probe_warmth(endpoint_key: str) -> bool:
    """Quick 5-second probe to check if the container is already warm.

    Used by activate() so that reconnecting after a ComfyUI restart
    (when the Modal container is still running) shows "Warm ✓" immediately
    instead of going through the warmup thread.
    """
    try:
        ep = _ENDPOINT_SLUGS.get(endpoint_key, {})
        payload = {
            "model":    ep.get("model", ""),
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": _WARMUP_MAX_TOKENS,
        }
        url = _endpoint_url(endpoint_key)
        key = _api_key()
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        with httpx.Client(timeout=5, follow_redirects=False) as client:
            r = client.post(url, json=payload, headers=headers)
        return r.status_code == 200
    except Exception:
        return False


def _post_once(endpoint_key: str, payload: dict) -> dict:
    """Single-shot POST to the inference endpoint — no cold-start retry.

    Used by generate().  Raises RuntimeError on any non-200 response so the
    caller can surface the error immediately rather than blocking for minutes.
    """
    url    = _endpoint_url(endpoint_key)
    key    = _api_key()
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    with httpx.Client(timeout=_GENERATE_TIMEOUT, follow_redirects=False) as client:
        r = client.post(url, json=payload, headers=headers)
    if r.status_code == 200:
        return r.json()
    if r.status_code == 303:
        raise RuntimeError(
            "Unsloth container is still starting up (303 redirect). "
            "Wait for warmup to complete before running inference."
        )
    if r.status_code == 400:
        try:
            body = r.text.lower()
        except Exception:
            body = ""
        if "no model" in body or "model loaded" in body:
            raise RuntimeError(
                "Unsloth container is loading model weights. "
                "Wait for warmup to complete before running inference."
            )
    raise RuntimeError(
        f"Unsloth inference failed (HTTP {r.status_code}): {r.text[:200]}"
    )


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
            with _warmup_lock:
                _state["warmup_phase"] = "Waiting for available L4 GPU…"
            if status_callback:
                status_callback(
                    f"Unsloth ({ep['label']}): waiting for container "
                    f"(attempt {attempt}/{_MAX_ATTEMPTS})…"
                )
            continue
        except Exception as exc:
            raise RuntimeError(f"Unsloth HTTP error: {exc}") from exc

        if r.status_code == 200:
            with _warmup_lock:
                _state["warmup_phase"] = "Ready"
            return r.json()

        if r.status_code == 303:
            with _warmup_lock:
                _state["warmup_phase"] = "Container starting up (GPU worker assigned)…"
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
                with _warmup_lock:
                    _state["warmup_phase"] = "Container running — loading LLM weights into VRAM…"
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
        _state["warmup_phase"]  = "Starting warm-up…"
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
            _state["warmup_phase"]  = f"Warm-up failed: {exc}"
            _state["warmup_error"]  = str(exc)
        logger.warning("Unsloth warm-up failed (%s): %s", label, exc)


def _start_warmup(endpoint_key: str) -> None:
    t = threading.Thread(target=_run_warmup, args=(endpoint_key,), daemon=True)
    t.start()


# ── Public API ────────────────────────────────────────────────────────────────

def mark_container_gone() -> None:
    """Reset warmup state after a generate() call fails.

    Only transitions from "warm" → "cold" so it doesn't clobber an active
    warmup thread that is still retrying.
    """
    with _warmup_lock:
        if _state["warmup_status"] == "warm":
            _state["warmup_status"] = "cold"
            _state["warmup_phase"]  = "Container unavailable — click Restart Warmup"
            _state["warmup_error"]  = "Container unreachable (may have scaled to zero)"


def backend_status() -> dict:
    """Return Unsloth backend state (mirrors modal_vision_client shape)."""
    ep_key = _state["endpoint_key"]
    ep     = _ENDPOINT_SLUGS.get(ep_key, {})
    ws     = _state.get("workspace", "")
    with _warmup_lock:
        ws_status = _state["warmup_status"]
        ws_phase  = _state["warmup_phase"]
        ws_error  = _state["warmup_error"]
    return {
        "active":          _state["active"],
        "endpoint_key":    ep_key,
        "endpoint_label":  ep.get("label", ep_key),
        "endpoint_url":    _build_url(ws, ep["slug"]) if ws else "",
        "endpoint_docs_url": f"{_build_base_url(ws, ep['slug'])}/docs" if ws else "",
        "model":           ep.get("model", ""),
        "vision":          ep.get("vision", False),
        "native_video":    ep.get("native_video", False),
        "workspace":       ws,
        "workspace_set":   bool(ws),
        "api_key_set":     bool(_api_key()),
        "warmup_status":   ws_status,
        "warmup_phase":    ws_phase,
        "warmup_error":    ws_error,
        "endpoints":       endpoint_list(ws),
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
    ep = _ENDPOINT_SLUGS[endpoint_key]

    # Quick probe: if the container is already warm (e.g. after a ComfyUI restart)
    # skip the warmup thread and go straight to "warm".
    if _probe_warmth(endpoint_key):
        with _warmup_lock:
            _state["warmup_status"] = "warm"
            _state["warmup_phase"]  = "Ready"
            _state["warmup_error"]  = ""
        logger.info("Unsloth backend activated: %s — container already warm", ep["label"])
        return {
            "success": True,
            "message": f"Unsloth activated ({ep['label']}). Container is already warm.",
        }

    with _warmup_lock:
        _state["warmup_status"] = "cold"
        _state["warmup_phase"]  = ""
        _state["warmup_error"]  = ""
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
            return {"status": "starting", "message": "Container is starting up (GPU worker assigned)."}
        if r.status_code == 400:
            body = r.text.lower()
            if "no model" in body or "model loaded" in body:
                return {"status": "starting", "message": "Container running — loading LLM weights into VRAM."}
        return {"status": "error", "message": f"Unexpected HTTP {r.status_code}."}
    except httpx.TimeoutException:
        return {"status": "starting", "message": "No response yet — waiting for an available L4 GPU."}
    except Exception as exc:
        return {"status": "starting", "message": f"No connection yet — waiting for GPU worker ({type(exc).__name__})."}


def _encode_image(image: Any) -> str:
    """Return a base64-encoded JPEG data URI for a PIL Image or file path."""
    import base64
    import io
    if isinstance(image, str):
        with open(image, "rb") as f:
            raw = f.read()
        # Preserve original format; default to JPEG for display
        mime = "image/jpeg"
        if image.lower().endswith(".png"):
            mime = "image/png"
        return f"data:{mime};base64,{base64.b64encode(raw).decode()}"
    # PIL Image
    buf = io.BytesIO()
    img = image.convert("RGB")
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def _build_vision_content(prompt: str, images: list[Any]) -> list[dict]:
    """Build an OpenAI-format multimodal content array (images + text)."""
    content: list[dict] = []
    for img in images:
        content.append({
            "type":      "image_url",
            "image_url": {"url": _encode_image(img)},
        })
    content.append({"type": "text", "text": prompt})
    return content


def generate(
    prompt: str,
    *,
    images: list[Any] | None = None,
    video_frames: list[Any] | None = None,
    system_prompt: str = "",
    max_tokens: int = 512,
    temperature: float = 0.7,
    status_callback: Callable[[str], None] | None = None,
) -> dict:
    """Generate text via the Unsloth Studio endpoint.

    Supports vision when the active endpoint is a vision-language model
    (27B and flash_next).  Pass PIL Images or file paths via `images`.
    `video_frames` is treated as a list of images (frame-by-frame).

    Returns {success, text, message}.

    max_tokens should be ≥300 for Qwen3 models: they emit a
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

    all_images = list(images or []) + list(video_frames or [])
    if all_images and not ep.get("vision"):
        return {
            "success": False,
            "text":    "",
            "message": (
                f"Endpoint '{ep['label']}' is text-only and cannot process images. "
                "Switch to the 27B or Flash-Next endpoint for vision tasks."
            ),
        }

    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if all_images:
        user_content = _build_vision_content(prompt, all_images)
    else:
        user_content = prompt

    messages.append({"role": "user", "content": user_content})

    payload: dict = {
        "model":       ep["model"],
        "messages":    messages,
        "max_tokens":  max_tokens,
        "temperature": temperature,
    }

    with _warmup_lock:
        warmup = _state.get("warmup_status", "cold")

    if warmup != "warm":
        msg = (
            f"Unsloth container is not ready (status: {warmup}). "
            "Wait for the warm-up to complete in the LLM panel before running inference."
        )
        if status_callback:
            status_callback(msg)
        return {"success": False, "text": "", "message": msg}

    if status_callback:
        status_callback(f"Calling Unsloth ({ep['label']})…")

    try:
        data = _post_once(ep_key, payload)
        raw  = data["choices"][0]["message"].get("content") or ""
        text = _strip_reasoning(raw)
        if status_callback:
            status_callback(f"Unsloth ({ep['label']}) responded")
        return {"success": True, "text": text, "message": ""}
    except Exception as exc:
        logger.error("Unsloth generate failed: %s", exc)
        return {"success": False, "text": "", "message": str(exc)}
