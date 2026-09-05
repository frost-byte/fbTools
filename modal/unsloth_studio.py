"""Unsloth Studio on Modal — bundled with comfyui-fbTools.

Serves LLMs via Unsloth Studio's headless CLI (`unsloth studio run`) behind
`@modal.web_server`, exposing an OpenAI-compatible HTTP API at a public HTTPS URL.

Deploy (first time or after changes):
    cd /path/to/comfyui-fbTools
    python -m modal deploy modal/unsloth_studio.py

One-time setup (after first deploy):
    python -m modal run modal/unsloth_studio.py   # runs install_studio + bootstrap_api_key

Required Modal Secrets (create once in the Modal dashboard or CLI):
    modal secret create unsloth-studio-password UNSLOTH_STUDIO_PASSWORD=<choose-any-password>
    modal secret create huggingface-vision HF_TOKEN=<your-huggingface-token>

    The HuggingFace token only needs read access.  The password is the Unsloth
    Studio admin password -- choose anything, you won't need it again (API key auth
    is used for all programmatic access).

Auth: `unsloth studio run` always mints a brand-new random API key per invocation
and stores only a hash of it.  `UNSLOTH_STUDIO_HOME` is pointed at a persistent
Volume so a key captured once (via bootstrap_api_key below) stays valid forever
across cold starts, since its hash persists on the volume even though later boots
each mint an additional (unused) key alongside it.

Design notes (source-verified, not doc-page assumptions):
- Python 3.13 is required: Unsloth Studio's own `uv venv` step hard-requires
  Python >=3.13,!=3.13.8,<3.14 for its venv (confirmed live 2026-09-03).
- Model cache under STUDIO_HOME on the persistent Volume: without an explicit
  override, HF downloads land in /root/.cache/huggingface (ephemeral), so every
  cold start would re-download from scratch.
- UV_PYTHON_PREFERENCE=only-system: Unsloth Studio's `uv venv` creates a symlink
  to uv's managed Python -- which lives on the ephemeral filesystem, not the
  Volume -- so it dangles in any container other than the one that ran the
  installer.  Forcing the system Python (already 3.13 in this image) sidesteps
  this.
"""

import modal

APP_NAME = "unsloth-studio"
app = modal.App(APP_NAME)

STUDIO_PORT = 8888      # nginx front-door (what @modal.web_server exposes)
STUDIO_API_PORT = 8889  # Studio FastAPI internal port when running Full UI mode
STUDIO_HOME = "/studio"  # UNSLOTH_STUDIO_HOME -- auth DB + model cache

studio_volume = modal.Volume.from_name("unsloth-studio-home", create_if_missing=True)
VOLUME_CONFIG = {STUDIO_HOME: studio_volume}

unsloth_image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git", "curl", "nginx")
    .pip_install("unsloth")
)

BOOTSTRAP_MODEL_REPO = "unsloth/Qwen3-1.7B-GGUF"

# Endpoint configurations.  Only the practical ones are deployed; see commit
# history for documented dead ends (MTP experiments, Flash-Next MTP fork).
CONFIGS: dict[str, dict] = {
    "qwen3.8-27b": {
        "repo_id": "unsloth/Qwen3.8-27B-GGUF",
        "gguf_variant": "UD-Q3_K_XL",  # 13.15 GB; leaves ~9 GB headroom for KV cache
        "memory": 24576,                # 24 GB; model fits fully in VRAM
        "extra_flags": ["--gpu-memory-mode", "auto", "--disable-tools", "-c", "131072"],
        "mmproj_filename": "mmproj-F16.gguf",
    },
    "qwen3-8b": {
        "repo_id": "unsloth/Qwen3-8B-GGUF",
        "gguf_variant": "Q4_K_XL",
        "memory": 16384,
        "extra_flags": ["--gpu-memory-mode", "auto", "--disable-tools"],
    },
    "qwen3.8-flash-next": {
        "repo_id": "unsloth/Qwen3.8-Flash-Next-GGUF",
        "gguf_variant": "UD-IQ1_M",    # 74.5 GB checkpoint, 79.7% top-1 accuracy
        "memory": 86016,                # 84 GB; RAM/VRAM split required
        # -c only: `-ngl` disables --fit's own layer/context optimizer; explicit
        # --spec-type broke MTP auto-detection.  Let --fit and Unsloth's MTP
        # auto-detection keep full control of everything else.
        "extra_flags": ["--gpu-memory-mode", "auto", "--disable-tools", "-c", "131072"],
        "mmproj_filename": "mmproj-F16.gguf",
    },
}

SERVE_KWARGS = dict(
    image=unsloth_image,
    volumes=VOLUME_CONFIG,
    secrets=[
        modal.Secret.from_name("unsloth-studio-password"),  # UNSLOTH_STUDIO_PASSWORD
        modal.Secret.from_name("huggingface-vision"),        # HF_TOKEN
    ],
    scaledown_window=10 * 60,   # scale to zero after 10 min idle
    timeout=45 * 60,
)


def _unsloth_env(*, set_password: bool = False, extra_env: dict | None = None) -> dict:
    import os
    env = os.environ.copy()
    env["UNSLOTH_STUDIO_HOME"] = STUDIO_HOME
    env["HF_HUB_CACHE"] = f"{STUDIO_HOME}/hf-cache"
    env["UV_PYTHON_PREFERENCE"] = "only-system"
    # UNSLOTH_STUDIO_PASSWORD is NOT idempotent: once a password exists, passing
    # the env var makes `run` hard-error (confirmed live 2026-09-03).  Strip it
    # by default so re-runs against an already-bootstrapped volume don't fail.
    if not set_password:
        env.pop("UNSLOTH_STUDIO_PASSWORD", None)
    if extra_env:
        env.update(extra_env)
    return env


def _start_nginx(frontend_dist: str) -> None:
    """Start nginx on STUDIO_PORT as a reverse proxy + SPA static server.

    Serves the pre-built React frontend from *frontend_dist* and proxies API
    calls to Studio on STUDIO_API_PORT.  Studio's own middleware blocks the SPA
    for requests that don't come through Cloudflare or the LAN listener; serving
    the static files directly from nginx sidesteps that gate entirely.
    """
    import subprocess
    import textwrap
    from pathlib import Path

    # Paths that belong to Studio's API — proxy them to STUDIO_API_PORT.
    # Everything else is served as SPA static files with index.html fallback.
    api_prefixes = (
        "api", "v1", "docs", "openapi.json", "mcp",
        "tokenize", "metrics", "auth", "assets",
    )
    api_location = "|".join(api_prefixes)

    conf = textwrap.dedent(f"""
        worker_processes 1;
        pid /tmp/studio-nginx.pid;
        error_log /tmp/studio-nginx-error.log warn;
        events {{ worker_connections 1024; }}
        http {{
            include /etc/nginx/mime.types;
            default_type application/octet-stream;
            sendfile on;
            server {{
                listen {STUDIO_PORT};
                root {frontend_dist};

                # Studio API + assets → internal FastAPI
                location ~ ^/({api_location})(/|$) {{
                    proxy_pass http://127.0.0.1:{STUDIO_API_PORT};
                    proxy_http_version 1.1;
                    proxy_set_header Host $http_host;
                    proxy_set_header X-Real-IP $remote_addr;
                    proxy_set_header Upgrade $http_upgrade;
                    proxy_set_header Connection "upgrade";
                    proxy_read_timeout 600s;
                    proxy_buffering off;
                }}

                # SPA catch-all: serve index.html for unknown paths
                location / {{
                    try_files $uri $uri/ /index.html;
                }}
            }}
        }}
    """).strip()

    conf_path = Path("/tmp/studio-nginx.conf")
    conf_path.write_text(conf)
    subprocess.Popen(["nginx", "-c", str(conf_path), "-g", "daemon off;"])
    print(f"[fbtools] nginx started: frontend from {frontend_dist}, API proxied to :{STUDIO_API_PORT}")


def _run_unsloth_serve(repo_id: str, *, gguf_variant: str | None = None,
                       extra_flags: list[str] | None = None,
                       mmproj_filename: str | None = None) -> None:
    import json
    import os
    import subprocess
    from pathlib import Path

    # Read serve config written by the fbTools frontend (via Modal Volume).
    # Defaults to api_only=True (safe: no extra VRAM, Swagger UI still works).
    api_only = True
    config_path = Path(STUDIO_HOME) / "fbtools_serve_config.json"
    if config_path.exists():
        try:
            api_only = bool(json.loads(config_path.read_text()).get("api_only", True))
        except Exception:
            pass

    # In Full Studio UI mode, Studio runs on STUDIO_API_PORT (8889) behind an nginx
    # reverse proxy on STUDIO_PORT (8888).  This bypasses Studio's middleware that gates
    # the SPA to Cloudflare-tunnel or LAN-listener requests only.
    studio_port = STUDIO_API_PORT if not api_only else STUDIO_PORT

    cmd = [
        "unsloth", "studio", "run",
        "--model", repo_id,
        "--host", "0.0.0.0",
        "--port", str(studio_port),
        "--no-cloudflare",
        "--silent",
    ]
    if api_only:
        cmd.append("--api-only")
    if gguf_variant:
        cmd += ["--gguf-variant", gguf_variant]
    cmd += extra_flags or []

    # Pre-cache the mmproj (vision projector) file so Unsloth Studio can auto-detect it.
    # --mmproj cannot be passed directly: Studio validates extra args and rejects it
    # ("llama-server flag '--mmproj' is managed by Unsloth Studio").  Studio finds the
    # mmproj automatically when it is present in the same HF cache directory as the model.
    if mmproj_filename:
        try:
            from huggingface_hub import hf_hub_download
            hf_cache = f"{STUDIO_HOME}/hf-cache"
            raw_path  = hf_hub_download(repo_id=repo_id, filename=mmproj_filename,
                                         cache_dir=hf_cache)
            real_path = os.path.realpath(raw_path)
            if os.path.isfile(real_path):
                print(f"[fbtools] mmproj pre-cached: {real_path} ({os.path.getsize(real_path) // 1_000_000} MB) — Studio will auto-detect")
            else:
                print(f"[fbtools] mmproj path missing after realpath: {real_path} (raw: {raw_path})")
        except Exception as exc:
            print(f"[fbtools] mmproj pre-cache failed, Studio may start without vision: {exc}")

    subprocess.Popen(cmd, env=_unsloth_env())

    # In Full Studio UI mode, start nginx on STUDIO_PORT to serve the SPA and
    # proxy API calls to Studio on STUDIO_API_PORT.
    if not api_only:
        import glob
        candidates = glob.glob(
            f"{STUDIO_HOME}/unsloth_studio/lib/python*/site-packages/studio/frontend/dist"
        )
        if candidates:
            _start_nginx(candidates[0])
        else:
            print(f"[fbtools] frontend/dist not found under {STUDIO_HOME}; nginx skipped")


# ── Serve endpoints ───────────────────────────────────────────────────────────

@app.function(gpu="L4", memory=CONFIGS["qwen3.8-27b"]["memory"], **SERVE_KWARGS)
@modal.concurrent(max_inputs=4)
@modal.web_server(STUDIO_PORT, startup_timeout=1800)
def serve_l4_qwen3_8_27b():
    """Recommended: Qwen3.8 27B (UD-Q3_K_XL, dense, fully GPU-resident, vision via mmproj-F16)."""
    c = CONFIGS["qwen3.8-27b"]
    _run_unsloth_serve(c["repo_id"], gguf_variant=c.get("gguf_variant"),
                       extra_flags=c.get("extra_flags"), mmproj_filename=c.get("mmproj_filename"))


@app.function(gpu="L4", memory=CONFIGS["qwen3-8b"]["memory"], **SERVE_KWARGS)
@modal.concurrent(max_inputs=4)
@modal.web_server(STUDIO_PORT, startup_timeout=600)
def serve_l4_qwen3_8b():
    """Qwen3 8B (Q4_K_XL) — fast / lower quality sanity-check model."""
    c = CONFIGS["qwen3-8b"]
    _run_unsloth_serve(c["repo_id"], gguf_variant=c.get("gguf_variant"), extra_flags=c.get("extra_flags"))


@app.function(gpu="L4", memory=CONFIGS["qwen3.8-flash-next"]["memory"], **SERVE_KWARGS)
@modal.concurrent(max_inputs=4)
@modal.web_server(STUDIO_PORT, startup_timeout=1800)
def serve_l4_qwen3_8_flash_next():
    """Qwen3.8 Flash Next 125B MoE (UD-IQ1_M) — very slow cold start (~37 min uncached), vision via mmproj-F16."""
    c = CONFIGS["qwen3.8-flash-next"]
    _run_unsloth_serve(c["repo_id"], gguf_variant=c.get("gguf_variant"),
                       extra_flags=c.get("extra_flags"), mmproj_filename=c.get("mmproj_filename"))


# ── Setup functions ───────────────────────────────────────────────────────────

@app.function(
    gpu="L4",
    image=unsloth_image,
    volumes=VOLUME_CONFIG,
    timeout=30 * 60,
)
def install_studio(force_reinstall: bool = False) -> None:
    """One-time: run Unsloth Studio's own installer onto the persistent Volume.

    pip install unsloth (this image's own dep) is different from "Unsloth Studio"
    -- Studio needs its own venv + isolated Node.js runtime, built by the install
    script into UNSLOTH_STUDIO_HOME.  Running this once makes it stick across
    cold starts.  force_reinstall=True clears the prior venv first.
    """
    import subprocess
    if force_reinstall:
        subprocess.run(
            ["bash", "-c",
             f"rm -rf {STUDIO_HOME}/unsloth_studio {STUDIO_HOME}/bin {STUDIO_HOME}/share"],
            check=False,
        )
    proc = subprocess.run(
        "curl -fsSL https://unsloth.ai/install.sh | sh",
        shell=True, env=_unsloth_env(), capture_output=True, text=True, timeout=25 * 60,
    )
    print(proc.stdout)
    print(proc.stderr)
    proc.check_returncode()
    studio_volume.commit()
    verify = subprocess.run(
        ["unsloth", "studio", "verify-install"], env=_unsloth_env(),
        capture_output=True, text=True,
    )
    print(verify.stdout)
    if verify.returncode != 0:
        raise RuntimeError(
            f"verify-install failed:\n{verify.stdout}\n{verify.stderr}"
        )


@app.function(
    gpu="L4",
    image=unsloth_image,
    volumes=VOLUME_CONFIG,
    secrets=[
        modal.Secret.from_name("unsloth-studio-password"),
        modal.Secret.from_name("huggingface-vision"),
    ],
    timeout=30 * 60,
)
def bootstrap_api_key(repo_id: str = BOOTSTRAP_MODEL_REPO,
                      set_password: bool = False) -> str:
    """Run once (after install_studio): mint and capture the first API key.

    `unsloth studio run` mints a new random key per invocation and only ever
    prints the raw value once (the DB stores just a hash).  Because
    UNSLOTH_STUDIO_HOME is the persistent Volume, this captured key's hash
    stays in the auth DB forever across cold starts.

    set_password=True only on a fresh volume with no password yet; passing it
    again causes a hard error.

    Returns the captured API key (sk-unsloth-...).
    """
    import re
    import select
    import subprocess
    import time

    env = _unsloth_env(set_password=set_password)
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        [
            "unsloth", "studio", "run",
            "--model", repo_id,
            "--host", "0.0.0.0",
            "--port", str(STUDIO_PORT),
            "--api-only",
            "--no-cloudflare",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    api_key = None
    output_lines: list[str] = []
    # 1500s: importing FastAPI/deps from the Volume filesystem alone took ~190s
    # in the first live run (confirmed 2026-09-03).
    deadline = time.monotonic() + 1500
    try:
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            ready, _, _ = select.select([proc.stdout], [], [], min(remaining, 5))
            if proc.stdout in ready:
                line = proc.stdout.readline()
                if not line:
                    break
                output_lines.append(line)
                print(line, end="")
                match = re.search(r"API Key:\s+(sk-unsloth-\S+)", line)
                if match:
                    api_key = match.group(1)
                    break
            if proc.poll() is not None:
                break
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()

    if not api_key:
        raise RuntimeError(
            "Could not find 'API Key:' line in unsloth studio run output.\n"
            f"Output so far:\n{''.join(output_lines)}"
        )

    studio_volume.commit()
    print(f"\nCaptured API key: {api_key}")
    return api_key


@app.function(gpu="L4", memory=CONFIGS["qwen3.8-27b"]["memory"], **SERVE_KWARGS)
def probe_ports(wait_seconds: int = 90) -> str:
    """Probe which ports Unsloth Studio opens in Full Studio UI mode.

    Forces api_only=False so the web UI frontend starts, then reads /proc/net/tcp
    to list every listening port without needing iproute2/ss.

    Usage:
        python -m modal run modal/unsloth_studio.py::probe_ports
    """
    import json
    import time
    from pathlib import Path

    # Force Full Studio UI mode so the web UI frontend process starts.
    config_path = Path(STUDIO_HOME) / "fbtools_serve_config.json"
    orig_text = config_path.read_text() if config_path.exists() else None
    config_path.write_text(json.dumps({"api_only": False}))

    c = CONFIGS["qwen3.8-27b"]
    _run_unsloth_serve(c["repo_id"], gguf_variant=c.get("gguf_variant"),
                       extra_flags=c.get("extra_flags"), mmproj_filename=c.get("mmproj_filename"))

    print(f"Waiting {wait_seconds}s for Studio to finish starting…")
    time.sleep(wait_seconds)

    # Read listening ports from /proc/net/tcp{,6} — no ss/netstat needed.
    ports: set[int] = set()
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(path) as f:
                for line in f.readlines()[1:]:
                    fields = line.split()
                    if len(fields) >= 4 and fields[3] == "0A":  # TCP_LISTEN
                        ports.add(int(fields[1].split(":")[1], 16))
        except OSError:
            pass

    report = f"Listening TCP ports: {sorted(ports)}"
    print(report)

    # Restore the original config so the volume isn't left in Full UI mode.
    if orig_text is not None:
        config_path.write_text(orig_text)
    else:
        config_path.unlink(missing_ok=True)

    return report


@app.function(image=unsloth_image, volumes=VOLUME_CONFIG, timeout=120)
def inspect_studio_routes() -> str:
    """Inspect the Studio install to find the frontend mount path.

    Usage:
        python -m modal run modal/unsloth_studio.py::inspect_studio_routes
    """
    import subprocess

    site_packages = f"{STUDIO_HOME}/unsloth_studio/lib/python3.13/site-packages"
    lines: list[str] = []

    # Find index.html files (React/Next.js build outputs)
    r = subprocess.run(
        ["find", site_packages, "-name", "index.html"],
        capture_output=True, text=True,
    )
    lines.append("=== index.html locations ===")
    lines.append(r.stdout.strip() or "(none)")

    # Show context around every app.mount() call in the Studio backend
    main_py = f"{site_packages}/studio/backend/main.py"
    r = subprocess.run(
        ["grep", "-n", "app.mount\\|frontend\\|/ui\\|/app\\|api_only", main_py],
        capture_output=True, text=True,
    )
    lines.append("\n=== mount / frontend / api_only lines in main.py ===")
    lines.append(r.stdout.strip() or "(no matches)")

    report = "\n".join(lines)
    print(report)
    return report


@app.local_entrypoint()
def bootstrap(repo_id: str = BOOTSTRAP_MODEL_REPO, force_reinstall: bool = False):
    """CLI entrypoint: install Studio + bootstrap the API key in one step.

    Usage:
        python -m modal run modal/unsloth_studio.py
        python -m modal run modal/unsloth_studio.py --force-reinstall
    """
    print("Step 1/2: Installing Unsloth Studio onto the persistent Volume…")
    install_studio.remote(force_reinstall=force_reinstall)
    print("Step 2/2: Bootstrapping API key…")
    key = bootstrap_api_key.remote(repo_id)
    print(f"\nSetup complete.  API key: {key}")
    print("Copy this key into the fbTools LLM panel > Unsloth tab, or set:")
    print(f"  UNSLOTH_STUDIO_API_KEY={key}")
