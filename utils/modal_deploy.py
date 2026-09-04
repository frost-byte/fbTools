"""Deploy/manage utilities for the Unsloth Studio Modal app.

All operations are safe to call without the `modal` package installed;
functions that require it return {success: False, message: "..."} gracefully.

Public API (called by extension.py routes):
    get_workspace()                  -> str | None
    deploy(data_dir)                 -> {success, message, output}
    undeploy()                       -> {success, message}
    app_status()                     -> {deployed, message}
    list_containers()                -> {success, containers: [{id, state, ...}]}
    stop_container(container_id)     -> {success, message}
    stop_all_containers()            -> {success, stopped, message}
    run_bootstrap(data_dir)          -> {success, api_key, message}   (long-running)
    load_api_key(data_dir)           -> str | None
    save_api_key(data_dir, key)      -> None
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

APP_NAME = "unsloth-studio"
_MODAL_APP_FILE = Path(__file__).parent.parent / "modal" / "unsloth_studio.py"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _has_modal() -> bool:
    try:
        import modal  # noqa: F401
        return True
    except ImportError:
        return False


def _modal_cmd(*args: str, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run `python -m modal <args>` using the same Python that runs this code."""
    return subprocess.run(
        [sys.executable, "-m", "modal", *args],
        capture_output=True, text=True, timeout=timeout,
    )


# ── Workspace resolution ──────────────────────────────────────────────────────

def get_workspace() -> str | None:
    """Return the active Modal workspace name.

    Resolution order:
      1. MODAL_WORKSPACE env var (explicit override)
      2. Modal Python SDK config (most reliable when modal is installed)
      3. MODAL_PROFILE env var (selects named profile)
      4. ~/.modal.toml manual parse (active profile's section name)
    """
    # 1. Explicit env var
    ws = os.environ.get("MODAL_WORKSPACE", "").strip()
    if ws:
        return ws

    # 2. Modal SDK config
    if _has_modal():
        try:
            from modal.config import config as _cfg  # type: ignore[import]
            ws = (_cfg.get("workspace") or "").strip()
            if ws:
                return ws
        except Exception:
            pass

    # 3. MODAL_PROFILE env var (profile name = workspace name)
    profile = os.environ.get("MODAL_PROFILE", "").strip()
    if profile:
        return profile

    # 4. Parse ~/.modal.toml
    toml_path = Path.home() / ".modal.toml"
    if not toml_path.exists():
        return None
    try:
        return _parse_toml_workspace(toml_path)
    except Exception as exc:
        logger.debug("Could not parse ~/.modal.toml: %s", exc)
        return None


def _parse_toml_workspace(toml_path: Path) -> str | None:
    """Parse ~/.modal.toml and return the active profile's section name.

    ~/.modal.toml format (section name IS the workspace/profile name):
        [workspace-name]
        token_id = "ak-..."
        token_secret = "as-..."
        active = true
    """
    text = toml_path.read_text(encoding="utf-8")

    # Try stdlib tomllib (Python 3.11+) first
    try:
        import tomllib
        data = tomllib.loads(text)
    except ImportError:
        # Python 3.10 fallback: minimal regex-based parse
        data = _minimal_toml_parse(text)

    if not data:
        return None

    # Find section where active is truthy; if only one section, use it
    active_sections = [
        name for name, vals in data.items()
        if isinstance(vals, dict) and vals.get("active") not in (None, False, "", "false", "0")
    ]
    if active_sections:
        return active_sections[0]
    if len(data) == 1:
        return next(iter(data))
    return None


def _minimal_toml_parse(text: str) -> dict:
    """Regex-based TOML parser for ~/.modal.toml (sections + simple key=value only)."""
    import re
    result: dict = {}
    current: dict | None = None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^\[([^\]]+)\]$", line)
        if m:
            section = m.group(1).strip()
            current = {}
            result[section] = current
            continue
        if current is not None and "=" in line:
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            current[key] = val
    return result


# ── API key persistence ───────────────────────────────────────────────────────

def _api_key_path(data_dir: str) -> Path:
    return Path(data_dir) / "unsloth_studio_api_key.txt"


def load_api_key(data_dir: str) -> str | None:
    p = _api_key_path(data_dir)
    try:
        key = p.read_text(encoding="utf-8").strip()
        return key or None
    except FileNotFoundError:
        return None


def save_api_key(data_dir: str, key: str) -> None:
    p = _api_key_path(data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(key.strip(), encoding="utf-8")
    logger.info("Unsloth API key saved to %s", p)


# ── Deploy / undeploy ─────────────────────────────────────────────────────────

def deploy(data_dir: str) -> dict:
    """Deploy the bundled Unsloth Studio app to Modal.

    Runs `python -m modal deploy modal/unsloth_studio.py`.  Returns immediately
    with stdout/stderr captured.  Typical duration: 30-90 seconds.

    Returns {success, message, output}.
    """
    if not _has_modal():
        return {
            "success": False,
            "message": "modal package is not installed. Install with: pip install modal",
            "output": "",
        }
    if not _MODAL_APP_FILE.exists():
        return {
            "success": False,
            "message": f"Modal app file not found: {_MODAL_APP_FILE}",
            "output": "",
        }

    logger.info("Deploying Unsloth Studio: modal deploy %s", _MODAL_APP_FILE)
    try:
        result = _modal_cmd("deploy", str(_MODAL_APP_FILE), timeout=180)
    except subprocess.TimeoutExpired:
        return {"success": False, "message": "modal deploy timed out (180s)", "output": ""}
    except Exception as exc:
        return {"success": False, "message": str(exc), "output": ""}

    output = (result.stdout or "") + (result.stderr or "")
    if result.returncode == 0:
        logger.info("Unsloth Studio deployed successfully")
        _app_status_cache.clear()
        return {"success": True, "message": "Deployed successfully.", "output": output}
    return {"success": False, "message": f"Deploy failed (exit {result.returncode})", "output": output}


def undeploy() -> dict:
    """Stop (undeploy) the Unsloth Studio app.

    All endpoints become unavailable immediately.  Running containers are stopped.
    Returns {success, message}.
    """
    if not _has_modal():
        return {"success": False, "message": "modal package is not installed."}
    try:
        result = _modal_cmd("app", "stop", APP_NAME, timeout=60)
    except subprocess.TimeoutExpired:
        return {"success": False, "message": "modal app stop timed out."}
    except Exception as exc:
        return {"success": False, "message": str(exc)}

    output = (result.stdout or "") + (result.stderr or "")
    if result.returncode == 0:
        return {"success": True, "message": f"App '{APP_NAME}' stopped."}
    # "not found" is acceptable — already stopped
    if "not found" in output.lower() or "no app" in output.lower():
        _app_status_cache.clear()
        return {"success": True, "message": f"App '{APP_NAME}' was not running."}
    _app_status_cache.clear()
    return {"success": False, "message": f"Stop failed: {output.strip()}"}


# ── App status ────────────────────────────────────────────────────────────────

import time as _time
_app_status_cache: dict = {}
_APP_STATUS_TTL = 60  # seconds — avoids hitting Modal API on every setup-status poll


def app_status(force: bool = False) -> dict:
    """Check whether the Unsloth Studio app is currently deployed.

    Result is cached for 60 s to avoid hammering the Modal API during
    the frontend's 15 s setup-status poll loop.  Pass force=True (e.g.
    after an explicit deploy/undeploy action) to bypass the cache.

    Returns {deployed: bool, app_name, message}.
    """
    now = _time.monotonic()
    if not force and _app_status_cache.get("ts") and now - _app_status_cache["ts"] < _APP_STATUS_TTL:
        return _app_status_cache["result"]

    if not _has_modal():
        result = {"deployed": False, "app_name": APP_NAME,
                  "message": "modal package is not installed."}
        _app_status_cache.update(ts=now, result=result)
        return result
    try:
        proc = _modal_cmd("app", "list", timeout=30)
        output = (proc.stdout or "") + (proc.stderr or "")
        deployed = APP_NAME in output
        result = {
            "deployed": deployed,
            "app_name": APP_NAME,
            "message": "Running" if deployed else "Not deployed",
        }
    except subprocess.TimeoutExpired:
        result = {"deployed": False, "app_name": APP_NAME, "message": "modal app list timed out."}
    except Exception as exc:
        result = {"deployed": False, "app_name": APP_NAME, "message": str(exc)}

    _app_status_cache.update(ts=now, result=result)
    return result


# ── Container management ──────────────────────────────────────────────────────

def list_containers() -> dict:
    """List running containers for the Unsloth Studio app.

    Returns {success, containers: [{id, state, created_at, function_name}]}.
    """
    if not _has_modal():
        return {"success": False, "containers": [],
                "message": "modal package is not installed."}
    try:
        result = _modal_cmd("container", "list", timeout=30)
        output = (result.stdout or "") + (result.stderr or "")
        containers = _parse_container_list(output, filter_app=APP_NAME)
        return {"success": True, "containers": containers, "message": ""}
    except subprocess.TimeoutExpired:
        return {"success": False, "containers": [], "message": "modal container list timed out."}
    except Exception as exc:
        return {"success": False, "containers": [], "message": str(exc)}


def _parse_container_list(output: str, filter_app: str = "") -> list[dict]:
    """Parse `modal container list` tabular output into a list of dicts."""
    containers = []
    lines = output.splitlines()
    # Find header line to identify column positions
    header_idx = next(
        (i for i, l in enumerate(lines) if "Container ID" in l or "CONTAINER" in l.upper()),
        None,
    )
    if header_idx is None:
        return containers
    for line in lines[header_idx + 1:]:
        if not line.strip() or line.startswith("-") or line.startswith("="):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        container_id = parts[0]
        # Filter to this app's containers if requested
        if filter_app and filter_app not in line:
            continue
        containers.append({
            "id": container_id,
            "raw": line.strip(),
        })
    return containers


def stop_container(container_id: str) -> dict:
    """Force-stop a specific running container.

    Returns {success, message}.
    """
    if not _has_modal():
        return {"success": False, "message": "modal package is not installed."}
    if not container_id or not container_id.strip():
        return {"success": False, "message": "container_id is required."}
    try:
        result = _modal_cmd("container", "stop", "--yes", container_id.strip(), timeout=30)
        output = (result.stdout or "") + (result.stderr or "")
        if result.returncode == 0:
            return {"success": True, "message": f"Container {container_id} stopped."}
        return {"success": False, "message": f"Stop failed: {output.strip()}"}
    except subprocess.TimeoutExpired:
        return {"success": False, "message": "modal container stop timed out."}
    except Exception as exc:
        return {"success": False, "message": str(exc)}


def stop_all_containers() -> dict:
    """Force-stop all running containers for the Unsloth Studio app.

    Returns {success, stopped: int, message}.
    """
    listing = list_containers()
    if not listing["success"]:
        return {"success": False, "stopped": 0, "message": listing["message"]}

    containers = listing["containers"]
    if not containers:
        return {"success": True, "stopped": 0, "message": "No running containers."}

    stopped = 0
    errors = []
    for c in containers:
        result = stop_container(c["id"])
        if result["success"]:
            stopped += 1
        else:
            errors.append(result["message"])

    if errors:
        return {
            "success": False,
            "stopped": stopped,
            "message": f"Stopped {stopped}/{len(containers)}. Errors: {'; '.join(errors)}",
        }
    return {"success": True, "stopped": stopped,
            "message": f"Stopped {stopped} container(s)."}


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def run_bootstrap(data_dir: str, force_reinstall: bool = False) -> dict:
    """Run install_studio + bootstrap_api_key on Modal, store the API key.

    This is a long-running call (~5-30 min depending on whether Studio is
    already installed on the Volume).  Call from a background thread or
    asyncio executor.

    Returns {success, api_key, message}.
    """
    if not _has_modal():
        return {
            "success": False,
            "api_key": "",
            "message": "modal package is not installed. Install with: pip install modal",
        }
    if not _MODAL_APP_FILE.exists():
        return {
            "success": False,
            "api_key": "",
            "message": f"Modal app file not found: {_MODAL_APP_FILE}",
        }

    try:
        import modal as _modal
    except ImportError:
        return {"success": False, "api_key": "", "message": "modal package is not installed."}

    logger.info("Running Unsloth Studio bootstrap (install + key capture)…")

    # Step 1: install_studio
    try:
        install_fn = _modal.Function.from_name(APP_NAME, "install_studio")
        install_fn.remote(force_reinstall=force_reinstall)
        logger.info("install_studio complete")
    except Exception as exc:
        return {
            "success": False,
            "api_key": "",
            "message": f"install_studio failed: {exc}",
        }

    # Step 2: bootstrap_api_key
    try:
        bootstrap_fn = _modal.Function.from_name(APP_NAME, "bootstrap_api_key")
        api_key: str = bootstrap_fn.remote()
        if not api_key or not api_key.startswith("sk-"):
            return {
                "success": False,
                "api_key": "",
                "message": f"bootstrap_api_key returned an unexpected value: {api_key!r}",
            }
        save_api_key(data_dir, api_key)
        logger.info("Unsloth Studio bootstrap complete; API key stored")
        return {
            "success": True,
            "api_key": api_key,
            "message": "Bootstrap complete. API key captured and stored.",
        }
    except Exception as exc:
        return {
            "success": False,
            "api_key": "",
            "message": f"bootstrap_api_key failed: {exc}",
        }
