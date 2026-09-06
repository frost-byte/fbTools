# CHANGELOG


## v1.26.0 (2026-09-06)

### Bug Fixes

- **unsloth**: Add missing top-level httpx import
  ([`7b432ac`](https://github.com/frost-byte/fbTools/commit/7b432acd8bcd6c2dab7cbe48623375de2f581e96))

_post_once() and _probe_warmth() used httpx but the module-level import was missing — only
  _call_with_retry() and health_check() had inline try/import guards. Moved httpx to top-level and
  removed the now-redundant inline imports.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Add user root to nginx; proxy docs-assets; fix SPA 500 errors
  ([`ac8454a`](https://github.com/frost-byte/fbTools/commit/ac8454a6460ab8335b89071c159f9fa8d2e5d37d))

nginx workers default to www-data on Ubuntu/Debian, which cannot read Python package paths — causing
  try_files to return 500 for all SPA routes. Add user root to both placeholder and proxy configs
  since we run in a container.

Also add docs-assets to _API_PREFIXES so Studio's Swagger UI assets are proxied to FastAPI instead
  of falling through to the SPA catch-all.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Add vision/video support for Qwen3.8-27B and Flash-Next
  ([`e800576`](https://github.com/frost-byte/fbTools/commit/e80057680417ee6081e4d04576cc4cd55e8320ab))

Qwen3.8-27B and Flash-Next are native vision-language models; the 8B is text-only. Per-endpoint
  vision/native_video flags gate the image and video_frames paths in generate(). _encode_image() and
  _build_vision_content() produce OpenAI-format image_url content blocks for llama-server.
  extension.py routes _run_vision_inference() and _run_vision_inference_clip() through the unsloth
  client when active and the endpoint supports vision; native_video path sends raw frames, text-only
  endpoints fall back to a contact sheet. llm_panel.js persists
  unslothActive/unslothVision/unslothLabel to localStorage and exports activeBackendSupportsVision()
  for downstream consumers.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Bust app_status cache on deploy/undeploy so UI reflects new state
  ([`cffe8b5`](https://github.com/frost-byte/fbTools/commit/cffe8b52f05dd33e432caac26b4f96d03b5d4d31))

The 60 s cache introduced in the prior commit caused the App checklist row to show stale "not
  deployed" after a successful Deploy App action, since _fetchSetupStatus() was called immediately
  but hit the cached result. Clear _app_status_cache on any deploy() success or undeploy()
  completion so the next setup_status poll sees the real state.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Cache app_status for 60 s to reduce Modal API calls
  ([`c6b7239`](https://github.com/frost-byte/fbTools/commit/c6b7239320bdff7ea067e8366ff837403d698ba7))

The setup-status poll fires every 15 s while setup is incomplete. Each call previously ran \`modal
  app list\` as a subprocess, generating Modal API traffic every 15 s throughout the bootstrap
  window (up to 30 min = ~120 unnecessary calls).

app_status() now caches its result for 60 s using monotonic time, so the Modal API is hit at most
  once per minute during polling. force=True is available for callers that need a fresh check (not
  currently needed — deploy/undeploy routes don't call app_status).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Cancel warmup thread on deactivate to prevent stale retry requests
  ([`200e140`](https://github.com/frost-byte/fbTools/commit/200e140c37cea23ac96854fd37d344d201ef2f87))

deactivate() now sets a _warmup_cancel Event that the _call_with_retry loop checks at the top of
  each iteration. Previously, the warmup thread kept retrying Modal after deactivate, and when the
  user stopped containers and re-activated, the old thread's next retry fired concurrently with the
  new activation — causing Modal to spin up two containers.

_start_warmup() clears the cancel flag before launching the new thread.

Requires a ComfyUI restart to take effect (Python module change).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Cap all serve functions at max_containers=1
  ([`157c448`](https://github.com/frost-byte/fbTools/commit/157c4483497dcfdc1abdb3d0a5158750f8a5bf4f))

Prevents Modal from spinning up a second container when a retry request arrives during cold-start. A
  single-user setup never needs more than one container; max_inputs=4 still allows up to 4
  concurrent in-flight requests to share the same container.

Requires a Modal redeploy to take effect.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Correct vision flags — all endpoints are text-only
  ([`590ce72`](https://github.com/frost-byte/fbTools/commit/590ce72260381145f5a2cd50e0af53d325e404dd))

Confirmed via /api/models/local: all three GGUF models in the HF cache report task=text-generation
  with no mmproj file present. The vision:true flags were set optimistically and were never
  validated.

- utils/unsloth_client.py: vision/native_video → False for 27b and flash_next - js/ui/llm_panel.js:
  same correction in _ENDPOINTS array + updated titles

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Downscale tiles in _tile_frames to 320×180 per cell
  ([`4b77344`](https://github.com/frost-byte/fbTools/commit/4b7734437af6dc0dd5d1f07f31b998e84873afb6))

Raw 1080p frames in an 8-cell grid would be 7680×2160 (~15 MB base64). Resizing each tile to 320×180
  (matching _spa_build_contact_sheet) keeps the sheet at 1280×360 — roughly 150 KB as JPEG, ~200 KB
  base64.

Also downscales single-frame paths so a lone 4K frame doesn't balloon the payload unnecessarily.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Downscale video frames individually, keep native multi-image path
  ([`0033092`](https://github.com/frost-byte/fbTools/commit/0033092b8d996c577d50d9eb6c97352bab1b7429))

The contact-sheet approach gutted Analyze Media's per-frame reasoning. Revert native_video=True on
  both endpoints.

Root cause of the 413: full-resolution 832×832 frames sent as 6 separate image_url entries → ~2 MB,
  exceeding nginx's 1 MB default.

Fix: _downscale_frame(img, max_dim=480) shrinks each PIL Image so its longest side is ≤480 px before
  encoding. 6 frames of 479×479 → ~530 KB total base64, well under the 1 MB nginx default (and under
  the 100 MB cap from the nginx config fix).

A contact-sheet fallback for non-native-video endpoints is kept inline in generate() so the branch
  is explicit.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Drop --mmproj CLI flag; pre-fetch into HF cache instead
  ([`7dd3b97`](https://github.com/frost-byte/fbTools/commit/7dd3b974604bff59776f70cc854baf5b7d6d4599))

unsloth studio run does not accept --mmproj, causing the subprocess to exit immediately and port
  8888 to never open (Modal health check failure). Keep the hf_hub_download() call so the file lands
  in the Volume cache for Unsloth Studio's own auto-detection.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Escape f-string brace in nginx config comment
  ([`769c3f7`](https://github.com/frost-byte/fbTools/commit/769c3f746ee9894654b14e672d1b0d5d8726646c))

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Fall back to modal CLI for serve mode volume write
  ([`a59f65c`](https://github.com/frost-byte/fbTools/commit/a59f65c84c9683d18aef0799ce2e37453dbb0788))

When modal is not installed in ComfyUI's Python (batch_upload unavailable), fall back to invoking
  write_serve_config via the modal CLI binary. Checks PATH first, then the known preflight venv path
  as a hardcoded fallback.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Fast reconnect after restart + fix Open API link URL
  ([`0105a99`](https://github.com/frost-byte/fbTools/commit/0105a9974d0d7b9e579db17d23eaab4f15f84c4e))

Reconnect: activate() now probes the container with a 5s timeout before starting the warmup thread.
  If it gets 200 (container still warm from a previous session), it sets warmup_status="warm"
  immediately and skips the thread — no more brief "Warming up…" flash on reconnect.

Open API link: _build_url returns /v1/chat/completions (POST-only), causing a 405 Method Not Allowed
  in the browser. Add _build_base_url() and expose endpoint_docs_url ("{base}/docs") from
  backend_status(). The frontend apiLink now uses endpoint_docs_url so it opens the Swagger UI.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Fix container list parser and add 30s auto-refresh
  ([`1fa5ad4`](https://github.com/frost-byte/fbTools/commit/1fa5ad46cb201e97ca245e22b24d90abc2c57461))

The previous parser tried to parse Rich unicode table output as plain text, silently counting
  box-drawing lines as containers.

Fixes: - Switch to `modal container list --json` for reliable parsing; filter by app_name ==
  "unsloth-studio" from the structured output - JSON key is `start_time` (derived from "Start Time"
  column header via Modal's snake_case conversion) - Add 30s _containerPollTimer so the count stays
  current without manual refresh; stopped on tab cleanup

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Follow Modal 303 Location token URL to prevent second container
  ([`4fb7e0a`](https://github.com/frost-byte/fbTools/commit/4fb7e0a09118e6de7f1c22fe4414001c744eaac6))

Per modal.com/docs/guide/webhook-timeouts: the 303 Location header points to the original URL plus a
  token query parameter. POSTing to *that* URL tells Modal's LB to route the retry to the container
  already being provisioned, rather than treating it as new demand. Previously we ignored Location
  and retried the bare URL, which Modal could interpret as a fresh request and respond by spinning
  up a second container.

Also reverts the 303 sleep from 120 s back to 20 s: with the token URL being followed correctly, the
  sleep is just pacing rather than a workaround.

Requires a ComfyUI restart to take effect.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Increase 303 retry sleep; add --fit to place mmproj on GPU
  ([`ddb5e3a`](https://github.com/frost-byte/fbTools/commit/ddb5e3a37698bdf7f72ea379c1a414175abdd25b))

303 sleep: 20 s → 120 s. After a 303 ("cold start redirect"), Modal's edge proxy may interpret a
  rapid retry as new demand and spin up a second container. 120 s gives the container enough time to
  start its nginx placeholder, so the next retry hits a 503 (from the container itself) rather than
  another 303.

--fit: the 27B model logged "--fit: off", meaning llama-server was not trying to fit the mmproj (0.9
  GB) onto the GPU. At ctx=65536 the VRAM budget is 12.2 + 4.6 + 1.42 = 18.2 GB against 22.5 GB
  free, leaving ~4.3 GB headroom — enough for the mmproj. --fit instructs llama-server to maximise
  GPU layer placement.

The unsloth_studio.py change requires a Modal redeploy to take effect. unsloth_client.py takes
  effect after a ComfyUI restart.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Log serve config at startup; add read/write_serve_config helpers; fix probe_ports
  volume restore
  ([`dd80146`](https://github.com/frost-byte/fbTools/commit/dd80146c52dbc07791ee951efc031bf43c26c364))

- Log the raw config value and resolved api_only at every container startup so the serve mode
  decision is visible in Modal logs - Add read_serve_config() and write_serve_config() Modal
  functions for manual inspection and override without starting the model - Fix probe_ports to call
  studio_volume.commit() after restoring the config so the volume is actually left in its original
  state

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Move max_containers=1 to @app.function() (correct API)
  ([`9962739`](https://github.com/frost-byte/fbTools/commit/996273923d237d1df81a5016c936c2e3b463986b))

@modal.concurrent() does not accept max_containers in Modal 1.5.5; it belongs on @app.function().
  Deploy verified successfully.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Move nginx map directive inside http block; log reload failures
  ([`4505389`](https://github.com/frost-byte/fbTools/commit/45053892ba907f7c8ec33c25b31a7aa0cb5bb002))

The map directive for WebSocket Connection header handling was placed at the top level of the nginx
  config, outside the http {} block, causing nginx -s reload to fail with [emerg] "map" directive is
  not allowed here. The reload silently failed (check=False), leaving nginx in 503 placeholder mode
  permanently after Studio became ready.

Also surface reload failures explicitly instead of printing success unconditionally.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Poll Studio before starting nginx to prevent 502 on startup
  ([`163df9e`](https://github.com/frost-byte/fbTools/commit/163df9efe653d76563e500359b61fd5a31b35126))

nginx opened port 8888 immediately (passing Modal's startup check) while Studio took 60-90s to bind
  to 8889, causing every request to 502. Now _run_unsloth_serve() blocks until /api/health on
  studio_port returns 200 before starting nginx, so port 8888 only opens once the backend is ready.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Pre-cache mmproj for Studio auto-detection instead of passing --mmproj flag
  ([`4f0bd7b`](https://github.com/frost-byte/fbTools/commit/4f0bd7bbe96cf67d28d127c42a7558664602f08e))

Unsloth Studio rejects --mmproj as a CLI extra arg: "llama-server flag '--mmproj' is managed by
  Unsloth Studio and cannot be passed as an extra arg". Keep hf_hub_download() to ensure the file is
  in the HF cache so Studio can auto-detect it, but remove the flag from the cmd args.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Prevent duplicate warmup threads and handle 503 placeholder
  ([`a657609`](https://github.com/frost-byte/fbTools/commit/a6576099b1bf76d53eb75ff4ccc2313b6f6808d8))

Two root causes of multiple containers spinning up:

1. _start_warmup() had no guard — calling activate() while a warmup thread was already running
  (double-click, Restart, ComfyUI reload) launched a second thread that fired another cold POST to
  Modal, causing Modal to queue a second container. Fixed with a global _warmup_thread ref: skip the
  start if the thread is still alive.

2. The nginx placeholder (503 while Studio loads behind it) was not in the retry list.
  _call_with_retry raised immediately on 503, which propagated as a warmup error and could trigger
  another activate(). Fixed: treat 503 the same as 303 — log phase and continue the loop.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Proper WebSocket vs HTTP Connection header in nginx config
  ([`b5e61e2`](https://github.com/frost-byte/fbTools/commit/b5e61e248a19fe93ef2fc593080a60093cd77f7f))

Hardcoded 'Connection: upgrade' on all proxied requests broke regular HTTP keepalive, which likely
  caused Studio's thread creation API to fail on every request (producing 'Thread __LOCALID_ not
  found' errors). Use a map to set Connection: upgrade only for actual WebSocket upgrades, close
  otherwise. Also add X-Forwarded-For and X-Forwarded-Proto headers.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Remove pre-warmup probe that caused duplicate Modal containers
  ([`f95961a`](https://github.com/frost-byte/fbTools/commit/f95961aa6c7b766c5b448db66243738c1e2c52eb))

_probe_warmth() sent a real POST to the Modal endpoint before _start_warmup() launched the warmup
  thread. Both requests hit Modal while no container was running, causing Modal's auto-scaler to
  start two containers every time Activate was clicked on a cold endpoint.

The warmup thread already handles the "already warm" case: a 200 on the first attempt finishes the
  thread immediately with warmup_status="warm". Removing the pre-probe eliminates the race with no
  functional regression.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Replace --fit with --mmproj-offload for 27B GPU placement
  ([`54dd591`](https://github.com/frost-byte/fbTools/commit/54dd5919ce062525ddd555dc12ac2a6c9af5f59f))

--fit is not a boolean flag in Unsloth Studio's llama-server; it requires a value. Passing it bare
  caused a 400 error that crashed the container on every cold start.

--mmproj-offload is the correct flag (per Unsloth's own log message) to force the mmproj-F16 (0.9
  GB) onto GPU. Unsloth's auto-detect conservatively places it on CPU even though ~4.3 GB of
  headroom exists at ctx=65536.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Replace frontend health poll with server-side warmup_phase
  ([`ea5ea1b`](https://github.com/frost-byte/fbTools/commit/ea5ea1bc610f915f9a96f45256ab0f0e17c3e3d3))

The 20s health poll was POSTing to /v1/chat/completions on the Modal container every 20 seconds,
  queuing real inference requests during warm-up.

Instead, _call_with_retry() now writes a human-readable warmup_phase into _state at each retry
  outcome (timeout → GPU wait, 303 → container starting, 400 no-model → weight loading, 200 →
  Ready). backend_status() exposes it and the existing 5s /status poll delivers it to the frontend.

Frontend changes: - Remove _fetchHealth, _startHealthPoll, _stopHealthPoll, _healthPollTimer,
  _lastProbeAt, actProbeAge (all health-probe machinery) - Add _startElapsed/_stopElapsed: a 1s tick
  for the elapsed display only - _syncStatus shows st.warmup_phase as the phase message while
  warming - Activity block is shown as soon as active (not gated on warm)

Zero extra Modal traffic from the frontend during warm-up.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Restore --mmproj passthrough; drop reasoning_effort param
  ([`92743c5`](https://github.com/frost-byte/fbTools/commit/92743c53dd0079881c900883ff9aadbc46206e42))

- modal/unsloth_studio.py: re-add --mmproj <path> to llama-server command; confirmed via local
  `unsloth studio run --help` that unknown flags pass through. File is now cached on Volume from
  prior cold start so download is instant on next cold start. - utils/unsloth_client.py: remove
  reasoning_effort parameter — it is a local CLI shorthand for `unsloth run`, not a llama-server API
  field.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Self-healing setup checklist for dropped bootstrap connections
  ([`f6db832`](https://github.com/frost-byte/fbTools/commit/f6db8328c5d6b4f94d2efe2325aaea0ff652c153))

The Bootstrap Key HTTP call can be dropped by the browser or an idle proxy before the server's
  asyncio thread finishes — the key IS stored server-side but the UI never saw the response.

Three changes to recover without a page reload: - _startSetupPoll() / _stopSetupPoll(): poll
  /unsloth/setup_status every 15 s while any check is incomplete; self-terminates once workspace +
  api_key + app_deployed are all green - Poll is started both on tab mount and at the start of every
  bootstrap attempt, so the checklist updates independently of the HTTP call result -
  _sepWithRefresh(): "Setup" section header now has a ↻ button for instant manual refresh (e.g.
  after a known-completed bootstrap that the UI missed)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Separate warmup retry loop from inference path
  ([`b404cf6`](https://github.com/frost-byte/fbTools/commit/b404cf6a2dac93389daaf5a9f039af71e5dbeb24))

generate() was calling _call_with_retry() — the same 20-attempt × 200s cold-start loop used by the
  warmup thread. Concurrent inference calls (e.g. Source Profile analysis) could each block for up
  to 66 minutes if the container was not warm.

Fix: - Add _post_once(): single-shot POST with _GENERATE_TIMEOUT (180s), raises immediately on
  303/400-no-model with a clear "wait for warmup" message rather than retrying for minutes -
  generate() now gates on warmup_status == "warm" and returns an error immediately if the container
  is not ready — the LLM panel shows the warmup phase so users know to wait - Only _run_warmup()
  uses _call_with_retry(); inference never retries through cold starts

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Serve mode button group — active state highlights current mode
  ([`5daef59`](https://github.com/frost-byte/fbTools/commit/5daef598d7bddd155ea7563626cb3b3d69ec82cb))

Replaces the ambiguous single toggle with two joined buttons (API Only | Full Studio UI). The active
  mode uses the primary style; the inactive one uses ghost. Clicking the inactive button switches
  and saves; clicking the already-active button is a no-op.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Start nginx placeholder immediately to avoid Modal startup_timeout
  ([`8a13b14`](https://github.com/frost-byte/fbTools/commit/8a13b145b6b4814f6ae47a6ccb1a7bf88f0a7c43))

Previously, nginx on port 8888 only started after Studio was ready on 8889 (poll up to 1500s). If
  Studio took longer than Modal's startup_timeout, the container was killed before nginx ever opened
  the port.

New flow: 1. _start_nginx_placeholder() opens port 8888 immediately (returns 503) so Modal's
  startup_timeout check passes within seconds 2. _run_unsloth_serve() polls Studio on 8889 as before
  (up to 1500s) 3. _reload_nginx_proxy() swaps the placeholder with the full proxy config (SPA
  static files + reverse-proxy to Studio) via `nginx -s reload`

Also extracts _nginx_placeholder_conf() / _nginx_proxy_conf() / _reload_nginx_proxy() helpers to
  keep _run_unsloth_serve() readable. client_max_body_size 100m is set in both configs.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Sync _state.unslothActive when server reports inactive
  ([`0a741c9`](https://github.com/frost-byte/fbTools/commit/0a741c9528eb5c287c6f82d97f981b06c0c82e9c))

_state.unslothActive was persisted in localStorage as true across ComfyUI restarts. When _syncStatus
  received active:false from the server it updated the button label but not _state.unslothActive, so
  clicking "Activate" triggered the deactivate() branch instead — doing nothing visible to the user.

Fix: reset _state.unslothActive/Vision/Label and _saveState() in the inactive branch of _syncStatus
  so the handler always uses the correct code path on the first click.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Use --json for app_status to avoid truncated name match
  ([`aaff69e`](https://github.com/frost-byte/fbTools/commit/aaff69e5dfeb35dfb9d0a40b5943dfc81806f50b))

modal app list truncates the Description column in Rich table output ("unsloth-stu…") so the
  APP_NAME string check always failed, showing the App row as ✗ not deployed even when the app is
  running.

Switch to --json and match on description == APP_NAME and state == "deployed".

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Use contact sheet for video to avoid multi-image llama-server failures
  ([`60c8cd1`](https://github.com/frost-byte/fbTools/commit/60c8cd1c6e1493dee0e6a0b14c05981a819219c6))

llama-server (GGUF/llama.cpp) does not reliably support multiple image_url entries in a single
  request. Sending video frames as separate image_url blocks caused "Failed to load image or audio
  file" 400 errors.

- Set native_video=False on 27b and flash_next endpoint descriptors; _run_vision_inference_clip
  already has a contact-sheet fallback for endpoints where native_video is False - Add
  _tile_frames() to unsloth_client.py: tiles PIL frames into a single contact-sheet image (≤4 per
  row) so generate() sends at most one image_url when native_video=False, covering the
  describe_video → _route_vision path - Add client_max_body_size 100m to the nginx config so large
  vision payloads are not rejected before reaching Studio or llama-server - Update llm_panel.js
  endpoint titles/native_video flags to match

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Use enable_thinking payload param instead of /no_think prefix
  ([`84b2b93`](https://github.com/frost-byte/fbTools/commit/84b2b93a5725525eebfe38ed1be25a8fd99da144))

Unsloth Studio supports enable_thinking as a proper request body field (confirmed in docs). Replace
  the /no_think\n message prefix with "enable_thinking": false in the payload — cleaner and works
  correctly with vision content where prepending a prefix to the user message is awkward.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

### Features

- **unsloth**: Add Open API link to endpoint URL in status bar
  ([`8cec7ce`](https://github.com/frost-byte/fbTools/commit/8cec7ceddb3dfc6b3ecf6d7c143ab4822d82ef6b))

Shows a small "Open API ↗" link next to the status line when the backend is active. Href is set from
  st.endpoint_url returned by the status poll; hidden when inactive or no URL is available.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Add Open Studio link to LLM panel status row
  ([`a54db75`](https://github.com/frost-byte/fbTools/commit/a54db75045870ede2d5445f9bf6d683d1c30c7ee))

Exposes endpoint_studio_url (base Modal URL) from backend_status() and adds an "Open Studio ↗" link
  next to the existing "Open API ↗" link in the Unsloth tab. Both links are visible only while the
  backend is active.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Add reset_password Modal function to recover from forgotten password
  ([`b5d2042`](https://github.com/frost-byte/fbTools/commit/b5d2042a40e5293141a702e310a56c5fe773e0b5))

Wipes the Unsloth Studio auth DB from the volume and re-bootstraps with the current
  UNSLOTH_STUDIO_PASSWORD Modal secret value, returning a fresh API key.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Add Unsloth Studio tab to LLM panel
  ([`9471489`](https://github.com/frost-byte/fbTools/commit/94714894b76457e94a8b8514c4725283a3ec9fdb))

Adds a fourth "Unsloth" tab to the LLM Backend panel alongside Local, Modal, and Gemini. The tab
  provides the full lifecycle UI for the Unsloth Studio Modal backend:

- Status badge with warmup indicator (cold/warming/warm colored dot) - Endpoint selector buttons —
  27B (recommended), 8B, Flash Next — with vision/native-video capability badges that update per
  selection - Activate/Deactivate button; activate triggers server-side warm-up and polls /status
  every 5 s (slows to 30 s once warm) - Setup checklist: workspace, API key, app deployed — each row
  with ✓/✗ check icon fed by /setup_status - Deploy App button (30–90 s) and Undeploy with
  confirmation dialog - Bootstrap Key button with live elapsed-second timer and 35-min fetch timeout
  for the long-running install+key-capture operation - Force-reinstall checkbox for bootstrap -
  Containers section: running count + Stop All with confirmation - Tab indicator dot goes green when
  Unsloth is active - unslothActive/unslothVision/unslothLabel persisted in localStorage

Also adds js/api/unsloth.js (UnslothAPI client) with typed methods for all 10 /fbtools/unsloth/*
  routes, plus a custom bootstrapKey() fetch using AbortController for the 35-min timeout window.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Configurable ctx_size in serve config; default 131072→65536
  ([`650ea30`](https://github.com/frost-byte/fbTools/commit/650ea305d39cb1e862bfdca1cae528f39b96ed50))

Default context for 27B and flash-next endpoints changed from 131072 to 65536. At 131072 the KV
  cache fills VRAM, evicting the mmproj to CPU and making image encoding 5-20× slower. At 65536 the
  mmproj stays resident (~4 GB headroom) and MTP may also re-enable.

The ctx_size is now overridable without redeploying:

modal run modal/unsloth_studio.py::write_serve_config --ctx-size 131072

_run_unsloth_serve reads ctx_size from fbtools_serve_config.json and strips any existing
  -c/--ctx-size from extra_flags before injecting the override, so the volume config is always
  authoritative.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Detect container scale-to-zero and add Restart Warmup button
  ([`bafb189`](https://github.com/frost-byte/fbTools/commit/bafb18904e826df702565b4fad2de8c6ade0e9fc))

Problem: when Modal's 10-min idle scaledown fires, the UI still shows "Warm ✓" with no way to
  restart without Deactivate → Activate.

Changes: - unsloth_client: add mark_container_gone() — transitions warmup_status from "warm" to
  "cold" and sets warmup_phase to "Container unavailable — click Restart Warmup"; no-ops if a warmup
  thread is already retrying - extension: _route_text() and _route_vision() call
  mark_container_gone() on any generate() exception so the next /status poll reflects reality -
  llm_panel: add Restart Warmup button (ghost, shown when active but not warm); calls activate()
  directly to kick a new warmup thread without requiring Deactivate → Activate; hidden when warm or
  inactive - llm_panel: update Activate/Deactivate tooltip to explain cold-start timing, idle
  scaledown, and when to use Restart Warmup vs Deactivate

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Enable vision on 27B endpoint via mmproj-F16.gguf
  ([`467958f`](https://github.com/frost-byte/fbTools/commit/467958f364d84014132a0abf43b13ee8797e59f6))

- modal/unsloth_studio.py: add mmproj_filename to qwen3.8-27b CONFIGS; _run_unsloth_serve()
  downloads it via hf_hub_download() (cached on Volume after first cold start) and passes --mmproj
  <path> to unsloth studio run - utils/unsloth_client.py: restore vision/native_video=True for 27b -
  js/ui/llm_panel.js: restore vision/native_video=true for 27b endpoint

Flash Next left as text-only (mmproj not yet confirmed for that repo).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Enable vision on Flash Next endpoint via mmproj-F16.gguf
  ([`ed4cf97`](https://github.com/frost-byte/fbTools/commit/ed4cf97a343944df678a65237b4ca46b8eae36df))

Same mmproj pattern as the 27B endpoint — confirmed HF repo has the file.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Expose thinking mode and full sampling params in generate()
  ([`f9b2732`](https://github.com/frost-byte/fbTools/commit/f9b2732f73f74341eae0b0041e95fc80b73ac437))

Adds Unsloth-recommended defaults for Qwen3.8-27B (source: unsloth.ai/docs): thinking mode:
  temp=1.0, top_p=0.95, top_k=20, min_p=0, presence=0.0 instruct mode: temp=0.7, top_p=0.80,
  top_k=20, min_p=0, presence=1.5

New generate() parameters: - thinking (bool, default True): selects mode defaults; instruct mode
  prefixes /no_think to the user message - reasoning_effort ("xhigh"|"medium"|"low"|"none"): passed
  via chat_template_kwargs to control CoT trace depth - temperature, top_p, top_k, min_p,
  presence_penalty, repetition_penalty: all override the mode default when provided - max_tokens
  default raised 512→2048 (thinking traces consume tokens)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Info icons with tooltips on Activate and Stop All
  ([`8938dec`](https://github.com/frost-byte/fbTools/commit/8938dec16307bd894aab772315a1b2b505d6bd73))

Adds a small ⓘ icon (cursor:help, .llmp-iicon) next to the Activate/Deactivate button and the Stop
  All Containers button. Each icon shows a native browser tooltip on hover explaining the
  distinction:

- Activate/Deactivate: local routing control only; container stays running on Modal until the 10-min
  idle scaledown fires - Stop All: kills the GPU container immediately, ends billing, requires a
  fresh cold start on the next request; prefer Deactivate if just switching backends temporarily

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Live activity section with health probe and elapsed timer
  ([`fac96d2`](https://github.com/frost-byte/fbTools/commit/fac96d28c7df52bd48d423dba011c0b9f8ac7988))

Adds an Activity section to the Unsloth tab that shows what the container is doing during the
  cold-start window:

- Elapsed timer counting from the moment Activate is clicked (1 s tick) - Health probe every 20 s
  while warming, surfacing the modal-side phase: "No response yet — waiting for an available L4
  GPU." "Container is starting up (GPU worker assigned)." "Container running — loading LLM weights
  into VRAM." "Container is ready." - Probe age ("Last probe: 15s ago") so the user knows the data
  is live - On warm: section updates to "Container ready after Xm Ys." and stops polling; on
  deactivate: section hides - Health poll starts from _syncStatus so it also resumes correctly if
  the tab is opened while already-active-and-warming

Also improves health_check() messages to distinguish GPU queue wait (connection timeout/refused)
  from container boot (HTTP 303) and model load (HTTP 400 "no model loaded").

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Nginx reverse proxy for Full Studio UI access via Modal
  ([`64ea74d`](https://github.com/frost-byte/fbTools/commit/64ea74d8e52b41a423bd2d3802ffee127426eba3))

Studio's middleware blocks the SPA for requests that don't arrive via Cloudflare tunnel or its LAN
  listener — Modal's proxy comes in on loopback and is rejected. Fix: in Full Studio UI mode, run
  nginx on port 8888 (what @modal.web_server exposes) to serve studio/frontend/dist/ directly and
  proxy API paths (/api/, /v1/, /docs, etc.) to Studio on port 8889. API-only mode is unchanged
  (Studio stays directly on 8888, no nginx).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Reasoning_effort selector, streaming SSE, retry sleep, job notifier
  ([`deb2c89`](https://github.com/frost-byte/fbTools/commit/deb2c89027bee1950c6f8cb07251a2c38eea77be))

**Retry loop** - Sleep 30 s on 503 (nginx placeholder), 20 s on 303/400, 15 s on timeout so warmup
  retries don't burn through _MAX_ATTEMPTS in seconds on a cold 27B start - Bump _MAX_ATTEMPTS 20 →
  40 (covers ~20 min of 503 polling) - Prevent duplicate warmup threads with is_alive() guard

**Reasoning effort** - Add thinking + reasoning_effort to _state; persist across calls - New
  _build_payload() centralises payload construction; injects reasoning_effort and optional stream
  flag - New set_inference_settings() updates _state and validates effort values - generate() reads
  state defaults when caller omits thinking/reasoning_effort - backend_status() surfaces thinking +
  reasoning_effort fields - POST /fbtools/unsloth/inference_settings REST route - LLM panel:
  4-button reasoning mode row (Instruct/Low/Medium/High), _applyReasoningMode(), _syncStatus()
  mirrors server state to buttons - Disable Restart Warmup button while warming to prevent spam
  clicks

**Streaming** - New generate_stream() sync generator with incremental <think> block filter - POST
  /fbtools/llm/generate/stream SSE route: asyncio.Queue bridges sync httpx generator to async
  aiohttp StreamResponse - LlmAPI.generateStream() SSE reader in js/api/llm.js -
  UnslothAPI.inferenceSettings() in js/api/unsloth.js

**Job Complete Notifier** - _NOTIFY_DIR: ComfyUI user-data dir watched by this Claude session -
  JobCompleteNotifier output node writes UUID-named JSON on workflow completion - Enables phone push
  notifications via Claude Code Monitor

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Route llm/generate* and describe_video through Unsloth when active
  ([`0867bac`](https://github.com/frost-byte/fbTools/commit/0867bacd18af0a350c36590903b6f2a1fc78202a))

Adds _route_text() and _route_vision() async helpers that transparently dispatch to
  _unsloth_client.generate() when Unsloth is active, falling back to _llm_client otherwise. Both
  helpers share the same return shape {success, text, message} so callers need no per-backend logic.

Wired routes: POST /fbtools/llm/generate — vision path when images present, text otherwise POST
  /fbtools/llm/generate/shot_action — text-only POST /fbtools/llm/generate/dialogue — text-only POST
  /fbtools/llm/generate/polish — text-only POST /fbtools/llm/describe_video — video_frames vision
  path

_route_vision() returns a descriptive error when the active Unsloth endpoint is text-only (8B)
  rather than silently stripping images. prompt_for_* builders on _llm_client are still used for
  structured prompt construction regardless of the active backend.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Self-contained Unsloth Studio Modal backend
  ([`e97874d`](https://github.com/frost-byte/fbTools/commit/e97874de30277a29758a221ca9ffc10eab8f391c))

Bundles the Unsloth Studio Modal app and wires it into the fbTools LLM backend system so a new user
  can deploy and use it entirely from within this extension — no separate project needed.

What's included: - modal/unsloth_studio.py: Modal app definition (3 endpoints: 27B, 8B, Flash-Next;
  install_studio + bootstrap_api_key setup functions; 10-min scaledown window) -
  utils/unsloth_client.py: HTTP client to the OpenAI-compatible endpoints; dynamic URL construction
  from workspace name; cold-start 303 retry loop; background warm-up thread on activate(); Qwen3
  reasoning-block stripping - utils/modal_deploy.py: workspace resolution (MODAL_WORKSPACE env var,
  Modal SDK, ~/.modal.toml parse), deploy/undeploy/app-status via modal subprocess, container
  list/stop, bootstrap_api_key capture + key persistence - extension.py: startup configure() from
  stored key + resolved workspace; _run_text_inference unsloth branch; _run_vision_inference
  text-only guard; REST routes: status, activate, deactivate, health, setup_status, deploy,
  undeploy, containers, containers/stop, bootstrap_key

New user setup flow (all from within ComfyUI after `modal token new`): 1. LLM panel > Unsloth >
  Deploy (modal deploy, ~60s) 2. LLM panel > Unsloth > Setup (install + bootstrap key, ~5-30 min) 3.
  Activate → warm-up starts in background → warm within 2-5 min

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **unsloth**: Serve mode toggle — API Only vs Full Studio UI
  ([`fe47852`](https://github.com/frost-byte/fbTools/commit/fe47852d74d15a804add39f68df58877d30dd822))

- modal/unsloth_studio.py: _run_unsloth_serve() reads fbtools_serve_config.json from the persistent
  Volume; defaults to api_only=True when absent - utils/modal_deploy.py: add load_serve_config(),
  _save_serve_config(), set_serve_mode() (writes local JSON + Modal Volume); deploy() records
  last_deployed_api_only on success - extension.py: GET/POST /fbtools/unsloth/serve_mode routes -
  js/api/unsloth.js: serveMode() and setServeMode(api_only) client methods - js/ui/llm_panel.js:
  Serve Mode row in the Setup section — toggle button (API Only ↔ Full Studio UI), status line
  showing last-deployed mode and mismatch hint; fetched on mount via _fetchServeMode()

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

### Refactoring

- **unsloth**: Use GET /v1/models for warmup probe instead of POST
  ([`3168c0d`](https://github.com/frost-byte/fbTools/commit/3168c0da891821cfc012df2b8a90a1185f89d34b))

POST /v1/chat/completions with max_tokens=1 generated actual tokens on every warmup cycle. GET
  /v1/models is semantically more appropriate (readiness check, not inference) and generates
  nothing.

Extracted _get_models_with_retry() with the same 303 Location token-following and cancel-event logic
  as _call_with_retry(). _run_warmup() now calls this instead of the POST variant.

Requires a ComfyUI restart to take effect.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.25.0 (2026-09-03)

### Bug Fixes

- **h3**: Drop per-replacement sentences from detailed_description preamble
  ([`4508930`](https://github.com/frost-byte/fbTools/commit/45089307c35b230f9931f142a4cc91f70683ec41))

The "The man in gray shirt is completely replaced by <Subject 1>." lines reiterated what
  subject_definitions and retention_analysis already cover. Keep only the single quality directive
  ("The target video is a photorealistic, seamless identity-replacement edit with strong temporal
  consistency.") before the shot descriptions begin.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **h3**: Simplify attribute_transfer replaced-subject prose
  ([`5651a99`](https://github.com/frost-byte/fbTools/commit/5651a996422d2de5e407e6ed48b263c5463dc59b))

Remove the redundant "is NOT copied and" phrase — "is fully replaced by" carries the intent
  unambiguously on its own and avoids double-encoding the same constraint for the model.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **h3**: Use appearance descriptions (not identifiers) in attribute_transfer prose
  ([`964a80c`](https://github.com/frost-byte/fbTools/commit/964a80c28128fd8b81d3dc81b76ed7ac2715cda1))

Two fixes in prompt_assembler.py:

1. subject_definitions motion clause: `src_info.get("name")` was always truthy so
  `appearance_summary` never fired — flipped priority to prefer appearance_summary over name for the
  "match those of ... in <Video N>" clause.

2. retention_analysis attribute_transfer branch: replaced `info['name']` (bundle identifier like
  "demon_3") and `src_name` (bare source label) with full appearance descriptions for both parties.
  Restructured the sentence from "{name}'s appearance overrides that of {label}" to "The appearance
  of {bun_desc} overrides that of {src_desc}" to avoid possessive-on-description grammar problems.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **h3**: Use role_description and video anchor in retention_analysis for non-person subjects
  ([`1389501`](https://github.com/frost-byte/fbTools/commit/13895010282baaf69bd4d16bf8bc435fa4e6411a))

Source profile subjects (objects, locations, animals) were emitting only the bare label in both
  subject_definitions and retention_analysis because appearance.summary was set to just the label
  rather than role_description.

Changes: - extension.py SceneCastBuild: set appearance.summary to role_description when available
  (fallback to label); store entity_type in slot_assignments - prompt_assembler.py _build_ref_map:
  propagate entity_type into ref_map - prompt_assembler.py retention_analysis: append ", as seen in
  <Video N>" for non-person subjects that have a video reference but no picture references

Result: a bed subject with a rich role_description now emits a full description in
  subject_definitions and retention_analysis, plus a <Video N> anchor so H3 knows which source video
  to sample the visual reference from.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Always persist proxy_built_at when server reports a fresh proxy
  ([`c2cfbb0`](https://github.com/frost-byte/fbTools/commit/c2cfbb09df122a24d27e987cf0d2862ecef875a3))

The !_isProxyDirty() guard in _refreshProxyStatus prevented proxy_built_at from ever being written
  back to disk after a browser reload. Because proxy_built_at was undefined after reload,
  _isProxyDirty returned true, the guard blocked _persistClip, and the badge permanently showed
  "needs rebuild" even for cached proxies. On click, the backend correctly found the proxy fresh and
  skipped ffmpeg — leaving the user with a toast but no machine activity.

Fix: when the server authoritatively says c.fresh = true, always update and persist proxy_built_at
  regardless of the client-side dirty state.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Clip segment QoL — apply-all for LoRA/subjects/soundscape/music, proxy status polling
  ([`376764a`](https://github.com/frost-byte/fbTools/commit/376764a05756a5da434277772785cc1610bb4c8a))

Source profile editor clip section: - Add "→ all" button per LoRA entry: copies LoRA to all other
  segments that lack it; preserves existing weight in segments that already have it - Add "→ all" /
  "✕ all" per subject: bulk-checks or unchecks a subject across every segment in the profile - Add
  "→ all" per soundscape and non-diegetic music field: copies current segment's value to all other
  segments - Single-clip proxy build now shows a toast on success/failure and detects clip_count=0
  (clip not yet persisted on server) - Replace fixed 3s/8s status timeouts with adaptive poll
  (5→10→15→20→30→60s) that stops early once the proxy is marked fresh, fixing stale "needs rebuild"
  badge after a single-clip proxy build completes

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

### Features

- **h3**: Tag Replaced Subjects toggle for original-subject attribute_transfer
  ([`0f0eda8`](https://github.com/frost-byte/fbTools/commit/0f0eda83df65ac7ef6008761b30b8009a89d1ff7))

Adds an opt-in "Tag Replaced Subjects" boolean input to SourceProfileClipPrompt.

When enabled, each source-profile subject that is being replaced by a SceneCastBuild bundle
  receives:

- A <Subject N> tag in subject_definitions with their role_description (identifiable but no
  structured hair/face/body detail fields, which are empty for source subjects). - A scoped
  attribute_transfer entry in retention_analysis that explicitly states: pose, movement, gestures,
  timing and screen position transfer to <Subject M>; the original's appearance, including face,
  hair, and clothing, is NOT copied and is fully replaced by <Subject M>'s appearance from <Picture
  P>/<Video N>.

The original subject numbers are assigned last (after bundle replacements and retained subjects), so
  existing Subject N ordinals are unaffected.

Gated behind the checkbox (default off) so the user can A/B the scoped-tag version against no-tag at
  a fixed seed before committing. Plan documented in docs/h3_attribute_transfer_assembler_plan.md.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Propagate SourceProfileLoad combo changes to downstream nodes
  ([`4793832`](https://github.com/frost-byte/fbTools/commit/47938329f3f8862ff1b717f1951f94be2e85646d))

Hook profile_name widget callback on SourceProfileLoad to fire onConnectionsChange on every node
  connected to its output when the selected profile changes. SceneCastBuild (and
  SourceProfileClipPrompt) now refresh their clip selectors and source subject columns immediately
  on combo change without requiring the user to disconnect and reconnect the wire or execute the
  graph.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.24.0 (2026-09-01)

### Bug Fixes

- **llm-panel**: Clarify Modal activation does not start container
  ([`75fc9f8`](https://github.com/frost-byte/fbTools/commit/75fc9f88085c6943dd46afab4ac0d5051214dbaf))

Rename status from "Active" to "Configured" and add "(container starts on first request)" note so
  users understand the dashboard will be empty until the first inference call triggers a cold start.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **llm-panel**: Move model load/unload to Local tab; fix Modal TDZ crash
  ([`fadbdf5`](https://github.com/frost-byte/fbTools/commit/fadbdf58eea17d6d670c1cb5d9aeb57bf06b2f88))

- Fix ReferenceError: quantCb/quantNotice were accessed in TDZ when _rebuildModelSel() was called
  before their const declarations in _renderModalTab. Fixed by declaring them before the first call.
  - Move full model management (scan, select, load, unload, download) from composition_editor's LLM
  Assistant section to the LLM panel's Local tab. - Compose tab retains status line (read-only) and
  generate buttons; syncs _S.llmLoaded/Vision/NativeVideo via fbt:llm-status custom event dispatched
  by fbt_panel._handleLlmPush after every load/unload. - Remove _llmRefreshModels,
  _populateLlmModelSel, _llmLoadSelected, _llmUnload, _llmDownloadDefault from composition_editor.js
  — all now live in llm_panel.js.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **modal**: Drop modal.parameter() — pass model_key/quantize via generate()
  ([`ee20960`](https://github.com/frost-byte/fbTools/commit/ee20960bb819438a0b17025d40ee11f0d3f7b247))

modal.parameter() doesn't support str type in modal 1.x. Restructured VisionLLM to load the model
  lazily on first generate() call, caching by (model_key, quantize) within a container's lifetime.
  Client updated to pass model_key and quantize as keyword args to generate.remote() instead of the
  class constructor.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **modal**: Rename container_idle_timeout to scaledown_window (modal 1.x)
  ([`d172cb7`](https://github.com/frost-byte/fbTools/commit/d172cb77b06c36d5a407f1af9f3b021e1081d36f))

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **modal**: Use gpu="L40S" string — modal.gpu removed in modal 1.x
  ([`3c5ed38`](https://github.com/frost-byte/fbTools/commit/3c5ed38e172320ffeae7a1d5f052abf6c1c202e8))

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **prompt-assembler**: Correct H3 video roles, bundle subjects, dialogue verb, speaker IDs
  ([`a779c57`](https://github.com/frost-byte/fbTools/commit/a779c571ce069b1a2fe5b1fa8946f4fd7cc6fdef))

- Video role in subject_definitions and retention_analysis now differentiates motion-donor (source)
  videos from bundle appearance references using _vnum_is_source computed from retention markers -
  Bundle-first subject numbering: attribute_transfer slots get Subject N labels before retained
  source slots; replaced slots get no label - Add _possessive() helper for pronoun-aware possessive
  forms - Dialogue lines now include "says:" verb before quoted text - Speaker IDs (Sx) now assigned
  to dialogue-only slots (no audio file) so subject_definitions shows the (Sx) marker for all
  speaking subjects - Pre-assign subject numbers before the ref_map loop to ensure stable ordering
  independent of slot iteration order

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **source-profile-analysis**: Hoist _SPA_STATUS_ID to module level
  ([`3ebba1d`](https://github.com/frost-byte/fbTools/commit/3ebba1d1282514068e31ec7de9c4a1f667e4d13b))

_run_vision_inference and _run_vision_inference_clip reference _SPA_STATUS_ID in their Modal
  status_callback but it was only defined as a local variable inside _source_profiles_analyze,
  causing NameError when the Modal path ran. Promoted to module-level constant and removed the
  now-redundant local def.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **source-profile-editor**: Surface server error detail in VLM failure toasts
  ([`693e58c`](https://github.com/frost-byte/fbTools/commit/693e58cb122bcba951198ff3d205263fb5361900))

APIError.response holds the raw JSON body from the server, which contains the actual reason (e.g.
  "Modal app not deployed", "modal package not installed"). Previously all VLM catch blocks showed
  only err.message ("Internal Server Error"). Added _errMsg() helper that parses err.response for
  the "error" field first, falling back to err.message. Applied to auto-segment, detect, describe,
  and analyze failure toasts.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **source-profiles**: Accept bare JSON array in VLM parser responses
  ([`361984b`](https://github.com/frost-byte/fbTools/commit/361984b0c04ca93872a75b323729de2d60761348))

_parse_vlm_json_response and _parse_segments_response now handle both {"subjects":[...]} /
  {"segments":[...]} envelopes and bare [...] arrays, matching the leniency added to the Modal-side
  parsers. Qwen2.5-VL and similar models sometimes omit the wrapping dict.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Output-dir support for extract-frame/audio, source profile card + nav fixes
  ([`d4c1149`](https://github.com/frost-byte/fbTools/commit/d4c11493b655bc350f1babc1da22034a2fb1fae8))

Bundle editor: - extractFrame and preprocessAudio now pass the correct dir ("input"/"output") to the
  server — videos placed in output/ were always 404ing - LLM appearance analyzer reads live vision
  status via window._fbtGetLlmStatus and listens to fbt:llm-status so the section appears without
  needing a reload after a model is loaded from the LLM tab

Source profile editor: - Card subject count uses subject_count from the list summary instead of
  subjects.length (which was always 0 before a profile was opened) - Card body shows "N subjects ·
  open to view" when subjects aren't yet fetched - Back button used
  root.parentElement?.parentElement causing DOM drift on each navigation; fixed to render into root
  directly

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

### Documentation

- Add Modal cloud vision backend integration handoff
  ([`b3479ec`](https://github.com/frost-byte/fbTools/commit/b3479ec190b2a2f3cd2963f184601825f14e2f58))

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- Update CLAUDE.md with widget contracts; add design docs
  ([`5db9875`](https://github.com/frost-byte/fbTools/commit/5db9875a3401f973ad67dc606431ee4248cee51d))

- CLAUDE.md: document cross-layer widget naming contract and the test_widget_name_contracts.py
  automated check - docs/vlm_systems.md: VLM system architecture overview -
  docs/h3_ref_short_edge_action_plan.md: H3 reference short-edge plan

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

### Features

- Subject inference, multi-GPU modal dispatch, describe-clip refinements
  ([`d6183fb`](https://github.com/frost-byte/fbTools/commit/d6183fbf3314c6bdcb7cb432a06d4e3a6b8b585c))

Source profile analysis: - build_subject_inference_prompt / parse_inferred_subjects_response for
  LLM-driven subject detection from clip descriptions - detect_segments: batch_window_seconds
  parameter for chunked processing - describe_clip: existing_action and prompt_override parameters -
  New endpoints: set_clips (bulk replace), merge_subjects (dedup upsert) - API client additions in
  js/api/source_profiles.js

Modal / LLM panel: - modal/app.py: per-GPU cls variants (T4, L4, L40S) for flexible dispatch -
  modal_vision_client: activate() accepts gpu parameter - js/api/modal.js: recommend() and
  profileRepo() methods - llm_panel.js: GPU selector UI, Qwen3-VL model list refresh

Other: - prompt_assembler: formatting and correctness fixes - extract_frame and preprocess_audio
  endpoints accept dir="output" - pyproject.toml: dependency/version updates

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **composition**: Add per-slot appearance overrides (slot_descriptors + appearance_overrides)
  ([`84853c6`](https://github.com/frost-byte/fbTools/commit/84853c64e80f73aeedc36a57e534de9f71cdfef2))

- utils/prompt_assembler.py: in assemble_composition(), apply composition.slot_descriptors[Sn] as
  appearance.summary override and composition.appearance_overrides[Sn].{face,hair,body} as granular
  sub-field overrides before passing slot_assignments to assemble_prompt. Deep-copies the affected
  subject dict so original resolved_subjects are never mutated. - js/ui/composition_editor.js: add
  collapsible "Override appearance" section to each slot card with a description textarea
  (slot_descriptors) and face/hair/body field rows (appearance_overrides). Indicator badge (✎) on
  toggle when any override is set. Wired to _markDirty(); both dicts included in _renumberSlots()
  remapping and slot removal cleanup. - js/styles/style.css: add .fbt-ce-slot-override-* rules. -
  tests/test_prompt_assembler.py: 10 new tests for TestSlotDescriptors and TestAppearanceOverrides
  covering override, no-mutation, empty/ whitespace passthrough, unknown key, and combined cases.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **dataset**: Dataset caption status and viewer node improvements
  ([`4c7e28b`](https://github.com/frost-byte/fbTools/commit/4c7e28be6d78121f3b2bc999491d6a37f1f82258))

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **libber**: Extract libber resolution to stdlib-only utility module
  ([`e008160`](https://github.com/frost-byte/fbTools/commit/e00816041aa32ee2b76ef4f656d445d5e04206d8))

Move %libber_name:key% resolution logic out of extension.py into utils/libber_resolve.py so it can
  be imported and tested without any ComfyUI context. Adds resolve_libber_refs() and
  extract_libber_names() with full test coverage.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **modal**: Add Modal cloud VLM backend as third explicit inference path
  ([`5bd2aca`](https://github.com/frost-byte/fbTools/commit/5bd2aca6a2da34b8ab8b5aefc8c727865e1f16fe))

- utils/modal_vision_client.py: slim client wrapping VisionLLM.generate.remote();
  activate/deactivate/is_active/backend_status; PRESET_MODELS list - utils/vlm_activity_log.py:
  rolling 500-entry activity log with record/recent/ model_history/last_activity_ts; drives model
  history in Modal tab and idle tracking - extension.py: third 'modal' branch in
  _run_vision_inference + captioner_type param on _run_vision_inference_clip; activity logged on
  every inference call; new routes GET /fbtools/modal/status, POST /fbtools/modal/activate, POST
  /fbtools/modal/deactivate, GET /fbtools/vlm/activity, GET /fbtools/vlm/model_history; late-imports
  modal_vision_client + vlm_activity_log

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **modal**: Add Qwen3-VL-8B + Qwen2.5-VL-32B-AWQ presets; pre_quantized flag
  ([`e888c8c`](https://github.com/frost-byte/fbTools/commit/e888c8cdd230c256c644212a277f70d1502821c8))

- PRESET_MODELS gains qwen3-vl-8b (new default) and qwen2.5-vl-32b-awq with pre_quantized: True;
  backend_status() exposes pre_quantized; activate() auto- disables NF4 when pre_quantized to
  prevent double-quantization - llm_panel.js: fallback preset list updated to match; _presetMap
  lookup drives _applyPreQuantizedState() which disables and unchecks the NF4 toggle with an amber
  notice when an AWQ/GPTQ preset is selected; re-enables on switch away; quantize label gains
  tooltip explaining bf16-only scope; custom HF ID placeholder now notes standard transformer repos
  only

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **modal**: Add VisionLLM Modal app + better "not deployed" error message
  ([`0efd896`](https://github.com/frost-byte/fbTools/commit/0efd896d64e0d098316f02d8a294b07eb892939e))

modal/app.py: - Defines fbtools-vision-llm Modal app with a VisionLLM cls - Supports qwen3-vl-8b
  (default), qwen2.5-vl-7b, qwen2.5-vl-32b-awq, qwen2.5-vl-3b, gemma3-4b; custom HF repos via
  model_key parameter - qwen_vl arch: Qwen2_5_VLForConditionalGeneration + qwen-vl-utils for native
  image and video_frames input - generic arch: AutoModelForCausalLM + AutoProcessor chat template
  path (Gemma3 and unknown custom repos) - NF4 quantization via BitsAndBytesConfig (skipped for
  pre-quantized AWQ) - Models cached in modal.Volume "fbtools-model-cache" to avoid re-download -
  GPU: L40S; idle timeout: 5 min; deploy: modal deploy modal/app.py

utils/modal_vision_client.py: - Catch "App not found in environment" from modal.Cls.from_name() and
  return a human-readable deploy instruction instead of the raw SDK error

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **scene**: Extend SceneCastBuild with source profile pool routing
  ([`11eca8c`](https://github.com/frost-byte/fbTools/commit/11eca8c63e2217ba0685392c8189dbfbad3026ea))

Add three optional SOURCE_PROFILE inputs to SceneCastBuild so subjects from SourceProfileLoad nodes
  form a selectable pool alongside existing bundle-backed entries.

- source_profile_1/2/3 optional inputs wire SOURCE_PROFILE → cast pool - fingerprint_inputs()
  includes source_profiles.json mtime + profile IDs - execute() routes source-derived entries
  (source_profile_id + source_subject_id) vs bundle-backed entries (subject_id + bundle_id) - Smart
  retention defaults: fully_preserved for bundle entries, partially_preserved for source-derived -
  Cast output carries source_profiles dict for downstream deduplication - cast_summary shows
  retention mode and entity type per entry

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **scene**: Phase 4 — source-derived subjects in prompt assembly
  ([`bdc6515`](https://github.com/frost-byte/fbTools/commit/bdc6515f32f7b9793cea96a6a854e66477c09219))

Teach _resolve_cast_media, apply_cast_to_subjects, _build_ref_map, and the H3 assembler to handle
  source-profile cast entries end-to-end.

_resolve_cast_media (extension.py): - Second pass groups cast entries by source_profile_id; emits
  one video_entries_full entry per unique profile (not per subject), carrying subject_ids: list for
  shared <Video N> assignment downstream.

apply_cast_to_subjects (utils/prompt_compositions.py): - New branch for source-derived entries:
  builds a synthetic subject dict with role_description as appearance.summary and _cast_retention
  set to the entry's retention mode (default: partially_preserved).

_build_ref_map (utils/prompt_assembler.py): - video_lookup handles subject_ids list so co-sourced
  subjects share the same video_entry object. - _video_entry_num dict ensures co-sourced subjects
  get the same <Video N> ordinal (only one video_counter increment per unique source entry). -
  retention_marker falls back to subject._cast_retention when retention_markers dict has no override
  for the slot.

_assemble_h3_ref2va (utils/prompt_assembler.py): - _vnum_to_labels map built after ref_map; drives
  combined subject labels in video role lines ("is the visual identity reference for <Subject 1> and
  <Subject 2>"). - video_sd_emitted / _ra_video_emitted guards prevent duplicate <Video N> lines
  when multiple subjects share one source video. - retention_analysis video lines use plural grammar
  ("their appearance", "the people") when more than one subject shares the video.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **scene**: Phase 7 — LLM subject decomposition for Source Profiles
  ([`e5fe4b9`](https://github.com/frost-byte/fbTools/commit/e5fe4b9ea4d43c5f3af27f8d1b0a0a0e5b11f3f3))

Focused, additive VLM analysis passes identify subjects in source media (people, setting,
  soundscape, objects, animals, or custom) and return structured candidates for review before
  committing to the catalog.

utils/source_profile_analysis.py (new — no ComfyUI deps): - PASS_TYPES + PASS_ENTITY_DEFAULTS define
  the six focus modes - build_prompt(): returns per-pass template with JSON schema instruction
  appended; accepts optional prompt_override replacing the template body -
  _parse_vlm_json_response(): strips markdown code fences, validates schema, fills missing/invalid
  fields with safe defaults, skips entries with no label - extract_video_frame(): pulls one frame at
  10% into clip via ffmpeg (primary) or imageio (fallback) - append_history_entry() / load_history()
  / history_for_profile(): append-only JSON history in source_profile_analysis_history.json with
  .bak backup; newest-first ordering per profile

extension.py: - Import source_profile_analysis helpers at module load - POST
  /fbtools/source_profiles/analyze — resolves media path, extracts frame for video sources, calls
  captioner.py backend, parses response, writes history, returns {candidates, pass_type, prompt} -
  GET /fbtools/source_profiles/analysis_history?profile_id=… — returns history entries for a
  profile, newest first

tests/test_source_profile_analysis.py (new — 35 tests): - prompt template coverage, JSON parser edge
  cases (code fences, leading prose, missing/invalid fields, non-dict items, empty subjects),
  history append/load/filter/ordering, backup creation, mutation safety

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **scene-cast**: Scenecastbuild dialogue and clip prompt UI
  ([`ad2aae0`](https://github.com/frost-byte/fbTools/commit/ad2aae0be4e00182d5dd3e2349fb1a34bfd4df84))

Extend SceneCastBuild node UI with dialogue entry system and clip prompt support; dynamic slot/cast
  rendering improvements.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **source-profile**: Add SourceProfileLoad/Define/List nodes and REST endpoints
  ([`79a8813`](https://github.com/frost-byte/fbTools/commit/79a881385ab74a13a9e8ce43352e5cbb96431cc3))

Registers SOURCE_PROFILE wire type and three nodes (Load, Define, List) following the SubjectProfile
  pattern. Adds five REST endpoints: reload, list, get, save, delete under
  /fbtools/source_profiles/. Nodes are registered in get_node_list() under the Scene category.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **source-profile**: Add SourceProfileRegistry data model and persistence
  ([`03d1bf0`](https://github.com/frost-byte/fbTools/commit/03d1bf0d8cf945f96690b7ac6a6117b9e51b2ba4))

Pure utility module (no ComfyUI deps) for the media-first subject catalog. Supports
  create/update/remove for profiles and subjects, entity type validation, wire dict generation for
  downstream PromptAssemble rendering, and JSON persistence with .bak backup. 50 tests, all passing.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **source-profiles**: Add video clip segmentation system
  ([`f1a89e4`](https://github.com/frost-byte/fbTools/commit/f1a89e492500400718f2a69525df4e1e430b0f26))

Adds clips array to SourceProfile for time-windowed video segments, VLM-assisted boundary detection
  via contact sheet, and per-clip action description. SceneCastBuild now accepts clip_id_1/2/3 to
  select which segment of each connected source profile to load; _resolve_cast_media looks up clip
  load_params when a clip_id is specified.

- utils/source_profiles.py: _normalize_clip, set_clips, upsert_clip, remove_clip, auto_partition,
  get_clip, clip_load_params, DEFAULT_* constants; define_profile preserves clips and
  default_segment_duration - utils/source_profile_analysis.py: segment detection and clip
  description prompts, _parse_segments_response, parse_clip_description_response - extension.py:
  SceneCastBuild clip_id_1/2/3 inputs + fingerprint; execute stores clip_ids in cast dict;
  _resolve_cast_media uses clip_load_params when clip_id is set; REST endpoints auto_partition,
  upsert_clip, remove_clip, detect_segments, describe_clip - js/api/source_profiles.js:
  detectSegments, describeClip, autoPartition, upsertClip, removeClip API methods - tests: 41 new
  tests for clip CRUD, auto_partition, clip_load_params, _parse_segments_response,
  parse_clip_description_response

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **source-profiles**: Extend backend nodes, clip prompt node, REST API
  ([`0182280`](https://github.com/frost-byte/fbTools/commit/0182280d08fb3225525f4fe12e75418d633760b3))

- captioner.py: expose clean_caption_text() as public API for callers outside captioner (used by
  source profile analysis pipeline) - extension.py: SourceProfileLoad / SceneCastBuild node updates;
  SourceProfileClipPrompt node for dynamic clip_id selection; new REST endpoints: proxy_status,
  prebuild_proxies, describe_clip, remove_clip; DatasetCaptioner refactor; widget name cleanups

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **source-profiles**: Proxy cache, analysis pipeline, profile data model
  ([`98316e5`](https://github.com/frost-byte/fbTools/commit/98316e59902622d1c6ed057b095b9c7a87c9ab75))

- Add utils/proxy_cache.py: per-segment ffmpeg proxy builder with sidecar JSON tracking; scale
  filter commas escaped for ffmpeg filter- graph parser; _is_fresh uses os.path.realpath() for
  symlink-safe comparison against ComfyUI's symlinked output directory - source_profile_analysis.py:
  extended segment analysis pipeline - source_profiles.py: profile data model updates -
  reference_bundles.py: reference bundle helpers

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **source-profiles**: Segment editor UI overhaul
  ([`821978c`](https://github.com/frost-byte/fbTools/commit/821978c62cb9a7a458581d12c36d75bb1a7ba5d8))

- Single-segment panel with ← N/M → nav replacing all-expanded card list - Timeline band click
  selects segment; video preview seeks to clip start - Active segment highlighted in timeline with
  filled triangle indicator - Collapsible Profile Settings and Subjects sections (accordion pattern,
  state persisted in _S.settingsOpen / _S.subjectsOpen) - Slot letters ({A}, {B}, …) shown inline
  after subject labels and updated live on checkbox toggle - Proxy dirty tracking: times_changed_at
  / proxy_built_at ISO timestamps stored on each clip and persisted via _persistClip; dirty clips
  show amber "needs rebuild" badge and amber dot in timeline band; _refreshProxyStatus advances
  proxy_built_at when server confirms fresh

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Add clips timeline panel to Source Profile editor
  ([`dc5de68`](https://github.com/frost-byte/fbTools/commit/dc5de684f4c0ff937cc37313e8ca897e3beac889))

Adds a collapsible Clips section to the profile detail view (video profiles only) with:

- Canvas-based timeline rendering colored bands per clip with internal boundary handles that can be
  dragged left/right to adjust split points - Gold dashed markers showing VLM-detected boundary
  suggestions - Time axis ticks and per-clip labels scaled to total video duration - Auto-detect
  video duration via hidden <video> loadedmetadata - "Auto-segment" button — calls REST
  autoPartition, replaces clip list - "Detect boundaries" button — calls detectSegments VLM
  endpoint, renders suggestions on timeline without committing - Per-clip cards: label, start/end
  time inputs, action textarea, subject checkboxes (linked to profile subjects), Describe button
  (describeClip) - Manual "Add clip" appends at end with configured segment duration - All mutations
  sync to profile.clips and trigger onClipsChanged so the Save button picks them up via collectMeta
  spread

Also adds SourceProfilesAPI methods: detectSegments, describeClip, autoPartition, upsertClip,
  removeClip.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Add dedicated LLM tab with Local/Modal/Gemini backend sub-tabs
  ([`8a5e01c`](https://github.com/frost-byte/fbTools/commit/8a5e01c1f49dcb21add01fffcbe84b92c6c32630))

- js/api/modal.js: ModalAPI (status/activate/deactivate) + VlmActivityAPI - js/ui/llm_panel.js:
  renderLlmPanel with three sub-tabs; getActiveCaptionerType() returns "modal" | "auto" |
  "gemini_flash" based on active backend; onBackendChange hook so header bar stays in sync; Modal
  tab has preset selector, custom HF ID history, quantize toggle, idle timeout, keep-warm,
  connect/disconnect button - js/ui/fbt_panel.js: add LLM tab to TABS; header bar shows active
  backend label (Modal: blue dot, Local: green, fallback: "No backend — configure in LLM tab"); wire
  onBackendChange to re-sync header on Modal state change - js/ui/source_profile_editor.js: remove
  CAPTIONER_TYPES dropdown and Gemini checkbox; replace with read-only backend badge driven by
  getActiveCaptionerType(); all three VLM request paths (analyze, detect_segments, describe_clip)
  now call getActiveCaptionerType() rather than reading local state

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Add Source Profiles sidebar panel
  ([`a444878`](https://github.com/frost-byte/fbTools/commit/a444878f9c04ea10eda0a602927c0a07cdde8dde))

- js/api/source_profiles.js — REST client for source profile CRUD, reload, LLM analyze, and analysis
  history endpoints - js/ui/source_profile_editor.js — full catalog browser panel: profile list with
  search, detail view (meta form + media preview), subject annotation list (add/edit/delete), LLM
  focused-pass analyze section (pass-type pills, captioner selector, prompt override, candidate
  review with add/add-all, history accordion), and auto-save - js/fb_tools.js — import and
  registerSidebarTab for 'fbt.source-profiles'

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Consolidate sidebar into single fbTools panel with lazy tabs
  ([`8511bc1`](https://github.com/frost-byte/fbTools/commit/8511bc19c7a8636c119b38b806a437fd71ae2520))

Replaces five separate sidebar tab registrations with one unified panel:

- js/ui/fbt_panel.js (new): shell with persistent LLM status bar in the header, horizontal tab strip
  (Compose/Bundles/Casts/Sources/History), and lazy mounting — each tab's DOM is created once on
  first activation and kept alive hidden on switch so state is never lost - js/fb_tools.js: single
  registerSidebarTab("fbt.panel") replaces the five individual registrations -
  composition_editor.js: _llmUpdateStatus now pushes state to the panel header via
  window._fbtUpdateLlmStatus (synchronous, no extra fetch) so the LLM badge reflects load/unload
  instantly from any tab

The shared fbtLlm object (exported from fbt_panel.js) will serve as the source of truth for tabs
  that need to check whether a model is loaded before routing inference calls.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

### Refactoring

- **gemini**: Move API key to server-side env var only
  ([`07a8beb`](https://github.com/frost-byte/fbTools/commit/07a8beb9c68e008a88a8019c15b3bd85af62329d))

Remove gemini_api_key from all frontend-to-backend paths. The key is now read exclusively from the
  GEMINI_API_KEY environment variable in extension.py. No credentials are accepted from request
  bodies or widget inputs.

- DatasetCaptioner node: remove gemini_api_key widget input and execute() parameter -
  _run_vision_inference(): remove api_key parameter; Gemini path reads
  os.environ.get("GEMINI_API_KEY") internally - _run_vision_inference_clip(): remove unused api_key
  parameter - /analyze, /detect_segments, /describe_clip, /recaption_single endpoints: drop
  body.get("gemini_api_key") fallback

- js/api/source_profiles.js: remove gemini_api_key from analyze(), detectSegments(), describeClip()
  signatures and request bodies - js/nodes/dataset_caption_viewer.js: remove from state, widget
  sync, and recaption request body - Tests updated accordingly

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Panel consolidation, bundle editor, node inspector
  ([`344d3cb`](https://github.com/frost-byte/fbTools/commit/344d3cb00af1601224402c6f58fcd217376d5d91))

- Consolidate sidebar into unified fbTools panel with lazy tabs (fb_tools.js + fbt_panel.js) -
  Bundle editor updates: pronoun style, short name field, image list improvements, appearance
  analyzer integration - Node inspector tab: collapsible JSON tree for selected node data - Run
  history: capture map extraction, run parsing improvements - File tree: path insertion and
  filtering fixes - story.js: remove stale widget helpers - lora.js: LoraStackBuilder active-row
  display fix - style.css: new panel, clip, and inspector styles

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

### Testing

- Widget name contract checks and dataset caption API tests
  ([`c23678b`](https://github.com/frost-byte/fbTools/commit/c23678b100140a6d0f6eba19d2e00d1d35e6710b))

- test_widget_name_contracts.py: cross-layer test that fails when any JS w.name === "x" lookup
  references a widget name not present in the Python node schema; run after any node schema change -
  test_dataset_caption_api.py: updated for refactored captioner API

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.23.0 (2026-08-21)

### Documentation

- Add audio reference rules, brave MCP guide, and generic workflow scanner
  ([`2605c0f`](https://github.com/frost-byte/fbTools/commit/2605c0fd3d38115738f4d5df769bfbff7fd11d5d))

Add H3 Ref2VA audio reference constraints and community-validated guidance, an empirical audio
  observations log, a generic brave-devtools MCP setup guide (machine-specific config gitignored via
  docs/*.local.md), and a general-purpose scan_node_usage.py script replacing the mixlab-specific
  scanner.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

### Features

- **ui**: Migrate all file pickers to FileTree with input/output tabs
  ([`0ccdca3`](https://github.com/frost-byte/fbTools/commit/0ccdca3a2c688fdf571e3f889c7dee1c3f2b3f1e))

Replace flat <select> dropdowns and hand-rolled tree implementations in bundle_editor and
  composition_editor with the shared buildFileTree component.

- bundle_editor: video picker, audio-from-video picker, separate audio picker all use buildFileTree
  with Input/Output tabs; new video_dir and audio_dir fields persist which directory was selected -
  composition_editor: bg editor and outfit editor drop ~160 lines of duplicated
  _insertPath/_renderTreeNode/tab logic in favour of buildFileTree - bundlesApi.streamUrl and
  mediaInfo accept an optional dir parameter - Backend resolves bundle video and audio file paths
  from the saved dir field (visual.video_dir, audio.video_dir, audio.audio_dir) so output-dir files
  load correctly in ComfyUI nodes - preview_sampled endpoint supports dir parameter for output
  videos

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.22.0 (2026-08-20)

### Features

- **llm**: Unified run history across all seven LLM features
  ([`b6c0438`](https://github.com/frost-byte/fbTools/commit/b6c04386812ee97019d73ee2d0e89ddda473bb2a))

Single llm_history.json file with kind-discriminated entries replaces the old
  video_describe_history.json. History now covers: video_describe, bg_analyze, outfit_analyze,
  shot_action, dialogue, polish, and appearance_analyze. Shared buildHistorySection() utility
  renders the collapsible accordion for both modals and the sidebar.

- Backend: /fbtools/llm/history (GET ?kind= filter, POST, POST /delete) migrates old flat entries on
  first read; legacy /describe_history routes shim to the new handlers - js/utils/llm_history.js:
  makeEntry() + buildHistorySection() - Sidebar shows shot_action+dialogue+polish together; Restore
  applies result text to the currently focused shot card - BG, Outfit, Appearance modals each get
  their own filtered history accordion with kind-appropriate Restore behaviour

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.21.0 (2026-08-20)

### Features

- **llm**: Add server-side video describe history with REST API
  ([`89c97cd`](https://github.com/frost-byte/fbTools/commit/89c97cd438367d1ba9a4c6e5612a8f65eb2a6c14))

History entries are stored in user_data_dir/video_describe_history.json via GET/POST/DELETE
  endpoints instead of browser localStorage. Tracks composition name, shot, video, extraction
  settings, prompts, and result so runs are restorable across sessions.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.20.0 (2026-08-20)

### Bug Fixes

- **backgrounds**: Use correct modal-overlay CSS class in _openBgEditor
  ([`c257bdd`](https://github.com/frost-byte/fbTools/commit/c257bdd53d6ccb37599c13ee26c2bb63fa66a496))

_openBgEditor was creating the overlay with class fbt-ce-overlay which has no CSS rule, so the
  overlay rendered as an unstyled block element (no position:fixed, no backdrop) instead of a
  fullscreen modal. Each click appended another invisible div, causing the "two forms" symptom. Fix:
  use fbt-ce-modal-overlay to match _openOutfitEditor and the CSS.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **bundle-editor**: Pass audio start_time/duration for extract_from_visual source
  ([`7484f7b`](https://github.com/frost-byte/fbTools/commit/7484f7b52a860a15620c575e91c60345047776da))

The Process Audio button was hardcoding start_time=0 and duration=0 when audio source is
  "extract_from_visual", ignoring the Timing section values the user set. All three source modes
  write timing into b.audio.start_time / b.audio.duration via _buildAudioTimeSection, so always use
  those values unconditionally.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **bundle-editor**: Show codec info and ffmpeg conversion command on video error
  ([`451a062`](https://github.com/frost-byte/fbTools/commit/451a062c5cae9f2ed0fe1ba2d482a2fd8d03a11b))

mediaInfo now returns a 'codec' field (cv2 CAP_PROP_FOURCC fourcc string, e.g. HEVC, avc1, xvid).
  The info line shows it alongside duration/fps/dims.

The error handler now provides actionable feedback: - MKV/AVI: tells user these formats aren't
  browser-playable, shows exact ffmpeg command to convert to H.264 MP4 - H.265/HEVC detected from
  codec field: specific re-encode command - MOV/MP4 with unknown codec: suggests re-encode with
  codec detail - All messages include the ready-to-run ffmpeg command with -map_metadata and
  -pix_fmt yuv420p flags

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **bundle-editor**: Use loadedmetadata event + error handler for video preview
  ([`d88dffd`](https://github.com/frost-byte/fbTools/commit/d88dffdd55aa5488a51f82d914f7fd0f4d04058f))

Previously the video player showed black/greyed-out controls with no feedback when the stream failed
  or the format wasn't browser-playable (.mkv, .avi, H.265 .mp4). Duration detection also relied
  solely on the mediaInfo endpoint; if that was slow or unavailable the trim slider never appeared.

- Add loadedmetadata listener as primary slider trigger (more reliable than waiting for mediaInfo
  alone) - Add error listener showing a human-readable message per MediaError code (network error,
  unsupported format, not found) - Track _onMeta/_onErr refs in _buildVideoPicker scope so each
  _loadFile call removes stale listeners before registering new ones

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **bundle-editor**: Use setAttribute for input list property in _mk helper
  ([`832de93`](https://github.com/frost-byte/fbTools/commit/832de93289197c42c7fc9560bfd754e48d006f16))

HTMLInputElement.list is a read-only getter; assigning it directly via el[k] = v throws "Cannot set
  property list … which has only a getter". Route it through setAttribute("list", v) so datalist
  bindings on the character-sheet and LLM image inputs in the subject form render correctly.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **bundle-editor**: Zero skip_first_frames when trim slider sets start_time
  ([`1ef5b4d`](https://github.com/frost-byte/fbTools/commit/1ef5b4d2782cf525e6fe95c22340a9bc8a02b48d))

skip_first_frames is the legacy VHS-style frame-count seek; start_time is the time-based replacement
  introduced with the trim slider. Setting both simultaneously double-offsets the seek position.
  Clear the legacy field whenever the slider or Mark Start button writes start_time.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **conditioning**: Add node_id class attr to CompositionToH3Conditioning
  ([`44f1267`](https://github.com/frost-byte/fbTools/commit/44f12674ffc0cc5c5c740a236f5eaf45b1ab34bf))

cls.node_id in execute() raised AttributeError on the ComfyUI-cloned class because node_id was only
  passed to io.Schema(), not stored as a class attribute. Add it explicitly so the clone inherits
  it.

- **conditioning**: Fix H3 video/audio loaders and raise frame_load_cap default
  ([`80d8b88`](https://github.com/frost-byte/fbTools/commit/80d8b88c99c1680a592520ad45e46c221e24f9cc))

Rewrite _h3_load_video_frames to use cv2 exclusively with a VHS-equivalent time-accumulator
  resampling loop — eliminates the broken VHS import path that caused UnboundLocalError and silent
  frame failures.

Replace the torchaudio fallback in _h3_load_audio with a direct ffmpeg subprocess call mirroring VHS
  get_audio: -ss/-t for accurate seeking, -f f32le piped to stdout, SR and channel count parsed from
  ffmpeg stderr. Fixes sample-rate mismatch (rapid playback) and wrong audio segment caused by
  torchaudio loading entire stream before slicing.

Raise frame_load_cap default from 16 to 96 across extension.py, reference_bundles.py, and
  bundle_editor.js so H3's n%17==5 trimming leaves 90 frames (8 Qwen samples) instead of 5 (1
  sample). Add a safety floor in the loader that upgrades any legacy cap < 39.

Also includes SceneCompose scene_synopsis input, PromptCompositionLoader filename_prefix output, and
  media frame extraction REST endpoints (_media_extract_frame, _media_delete_tmp_frame) that were
  developed alongside these fixes and could not be cleanly separated without patch-mode staging.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **conditioning**: Suppress skip_first_frames when start_time is set in video loader
  ([`e79bc46`](https://github.com/frost-byte/fbTools/commit/e79bc46a0d33972aa9f1e6ae563e76b52201cd4e))

When both start_time (time-based seek) and skip_first_frames (legacy VHS-style frame-count seek) are
  non-zero, the video loader was applying both, doubling the offset and seeking past the end of the
  video.

The trim slider added in da2018c sets start_time but does not zero skip_first_frames on existing
  bundles, so any bundle configured with the old frame-count approach would seek to (start_time +
  skip/fps) instead of the intended start_time.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **info-section**: Fix label overlap on wrapped multi-word info labels
  ([`c0a8199`](https://github.com/frost-byte/fbTools/commit/c0a819968ccb5b5cfc443c24ddba84661bd229e4))

align-items: center was causing the checkbox to sit at the vertical midpoint of a wrapped 2-line
  label (e.g. DIALOGUE TAGS), making the second line appear to overlap the checkbox.

Switch to flex-start so the checkbox pins to the top of the label. Widen the label from 42px to 56px
  to reduce wrapping on shorter labels. Add padding-top: 2px to keep single-line labels optically
  aligned with their adjacent inputs.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **outfit**: Guard get_folder_paths calls against KeyError
  ([`9b0a39f`](https://github.com/frost-byte/fbTools/commit/9b0a39fe5e602ea411550d6e89bd53239012ef74))

folder_paths.get_folder_paths() raises KeyError for unregistered folder types rather than returning
  an empty list. Wrap both calls in a helper so "sams" is used when registered and "sam2" is
  silently skipped when not.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **outfit**: Use folder_paths.get_folder_paths for SAM2 model discovery
  ([`3358065`](https://github.com/frost-byte/fbTools/commit/3358065c9c010efd91fc8d48afcc703e31ad58dc))

The SAM2 status endpoint was constructing model search paths manually from folder_paths.models_dir,
  which resolves to the ComfyUI launch directory rather than the paths registered via
  extra_model_paths.yaml.

Switch to folder_paths.get_folder_paths("sams") + get_folder_paths("sam2") so all
  extra_model_paths.yaml-registered sams directories are searched, with the manual construction as
  fallback.

Also re-fetch sam2_status live on every outfit modal open so the UI reflects the current server
  state without requiring a page reload after the model is installed.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **outfit-modal**: Fix label layout and add ref image thumbnails
  ([`a69effd`](https://github.com/frost-byte/fbTools/commit/a69effd6689211476643695174e75162d4fa1906))

Wrap ID/Name/Tags/Description fields in .fbt-ce-row so labels sit horizontally beside their inputs
  instead of appearing right-justified in a column. Add a Reference Images section header row for
  consistency.

Add 48x48 thumbnail to each reference image row using _ceViewUrl() so users can visually identify
  added refs. Add CSS for .fbt-ce-outfit-ref-row and .fbt-ce-outfit-ref-thumb.

Pass folder param to all _addFileToRefs() callers (addRefBtn, analyzeBtn, SAM2 addResultBtn) so
  thumbnail URLs resolve correctly.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **outfit-modal**: Populate outfit list after resource load
  ([`3721962`](https://github.com/frost-byte/fbTools/commit/3721962d6f52c2ca4c3e9ced3d977dacd188de79))

_refreshSidebar() was missing a _rebuildOutfitList() call, so outfits loaded from disk were never
  rendered into the sidebar after page load or panel re-open.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **sam2**: Bypass Hydra global state when loading SAM2 config
  ([`73db5c7`](https://github.com/frost-byte/fbTools/commit/73db5c720244d710cd16057e9c914f57883b23db))

Other custom nodes (Comfyui-SecNodes) call GlobalHydra.instance().clear() and re-initialize Hydra
  with their own config module. This breaks the standard build_sam2() path because sam2/__init__.py
  skips its own initialize_config_module() call when Hydra is already initialized, leaving compose()
  searching the wrong config tree.

Fix: load the SAM2 YAML directly from the package directory via OmegaConf.load() +
  hydra.utils.instantiate(), bypassing Hydra's global config resolution entirely. Apply the same
  postprocessing overrides that build_sam2() normally adds via OmegaConf.update(force_add=True).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **sam2**: Search sys.path for config instead of using import sam2
  ([`fbb3afe`](https://github.com/frost-byte/fbTools/commit/fbb3afea9a193d377aa9216eba6dc11bdc1ae70b))

import sam2 resolved to ComfyUI-RMBG/models/sam2/ (a bundled copy with no configs directory) instead
  of the installed package in site-packages. Search sys.path directly for a sam2/ directory that
  contains the needed config file so bundled shadow copies are skipped.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **scene**: Restore SceneCastBuild entries on workflow reload
  ([`db9fc55`](https://github.com/frost-byte/fbTools/commit/db9fc55fabe309ef7c2eb3991c82ed388a7612f0))

onNodeCreated fires before ComfyUI restores widget values from the saved workflow, so the initial
  JSON parse always saw '[]'. Hook onConfigure (which fires after widget values are applied) and
  re-read the widget there to rebuild the table with the saved entries.

- **scene**: Rewrite SceneCastBuild with JSON-backed entries
  ([`e184e88`](https://github.com/frost-byte/fbTools/commit/e184e880386086797929ed56624f0cd81fc940fa))

Replace the 16 individual boolean/combo inputs (subject_N, bundle_N, visual_mode_N, use_audio_N)
  with a single io.String.Input("cast_entries_json") storing a JSON array. Eliminates the io.Boolean
  toggle widgets that bypassed setWidgetVisible and leaked into the node UI.

The JS DOM table now manages entries directly in JS state, syncing to the hidden JSON widget via
  _syncToWidget(). The cast editor's _pushToNode() is simplified to a single JSON.stringify write +
  _refreshCastTable() call.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **settings**: Update MelBand model path to expect .safetensors (Kijai builds)
  ([`b4d1ef8`](https://github.com/frost-byte/fbTools/commit/b4d1ef84b41b1d6fe6db42d047e95ca1fa92fdf5))

Kijai/MelBandRoFormer_comfy provides fp16 (456 MB) and fp32 (913 MB) safetensors checkpoints — not
  .pth. Update placeholder, tooltip, and backend comment to reflect the correct format and source.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Analyzer image pool now stays in sync with bundle's image list
  ([`44e059b`](https://github.com/frost-byte/fbTools/commit/44e059b37be22cc76e10386dcbf75ed899c2bec8))

Previously the Analyze Appearance dropdown was populated once at form-render time. If b.visual.files
  was empty then, it fell back to all images in the input directory and never updated — so adding an
  image to the bundle left the LLM section pointing at the wrong pool and the user had to manually
  pick the right file from a large unsorted list.

Changes: - _buildAppearanceAnalyzer: replace one-shot imagePool capture with a rebuildPool() that
  re-reads b.visual.files on every call; auto-selects and previews the image automatically when the
  bundle has exactly one file; stored as sec._refreshPool for external wiring - _buildImageList:
  accepts an onFilesChange callback; fires it on every add and remove so the analyzer stays in sync
  immediately - _renderForm: holds _llmEl ref (set after analyzer is built) and passes () =>
  _llmEl?._refreshPool?.() into _buildImageList so the two sections are wired together through the
  closure

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Escape double quotes in paceSel.title string
  ([`33b3f0a`](https://github.com/frost-byte/fbTools/commit/33b3f0a00dad26b0fadfc506c7c15a4cbc5b6d00))

Unescaped double quotes inside a double-quoted JS string caused a parse error that prevented the
  entire fb_tools.js module chain from loading.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Reload LibberManager table from saved libber_name on workflow load
  ([`4430846`](https://github.com/frost-byte/fbTools/commit/443084663e035ee886f803ac4314056d3ab2e1d5))

onNodeCreated fires before ComfyUI restores widget values, so refreshTable() was reading the default
  combo value instead of the saved libber_name. Hook onConfigure (which fires after values are
  applied) to re-run refreshTable with the correct name.

- **ui**: Remove CSS import from index.js, loaded by fb_tools.js link tag
  ([`c82e266`](https://github.com/frost-byte/fbTools/commit/c82e2668d5ff5d319df1d2a32044a8394d28892d))

Browsers reject CSS loaded via ES module import (strict MIME checking). The stylesheet was already
  injected correctly via a <link> element in fb_tools.js; the duplicate import in index.js broke the
  entire module chain.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Resolve sidebar double-click bug for fbTools panels
  ([`aeca3d9`](https://github.com/frost-byte/fbTools/commit/aeca3d931f116380c08f0ec0fc642e4f878172fb))

ComfyUI calls render(el) while the previous panel's DOM is still present. All four tab guards
  checked for `.fbt-be-panel` — shared by both bundle and cast editors — so switching from one to
  the other silently skipped rendering the new panel, leaving the old one visible.

Fix: stamp each editor's root panel with a unique `data-fbt-editor` attribute ("bundle" / "cast")
  and guard against that specific value. The composition editor gains an equivalent guard on
  `.fbt-ce-panel`. Each render function already calls `el.innerHTML = ""` before building, so the
  old panel is cleared automatically when the guard passes.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Restore widget-dependent UI on workflow reload for Story/Scene nodes
  ([`ca84210`](https://github.com/frost-byte/fbTools/commit/ca84210616caf056a507ec357a5f0ead2cdb96af))

Add onConfigure hooks to StoryEdit, StorySceneBatch, and SceneSelect so their tables/dropdowns
  reload from the correct saved widget values after a workflow is opened. Each onNodeCreated fires
  before ComfyUI restores widget values, so the initial loads read default values instead of the
  saved ones.

- StoryEdit: store loadStoryData on node._loadStoryData, call from onConfigure - StorySceneBatch:
  re-fetch job_id options for saved story_name on onConfigure - SceneSelect: store updateSceneDir on
  node._updateSceneDir, call from onConfigure

- **ui**: Set widget.hidden and widget.element in setWidgetVisible
  ([`408010f`](https://github.com/frost-byte/fbTools/commit/408010fd64c636e29ec9effc77edd87250972db3))

Modern ComfyUI frontend gates visibility on widget.hidden; the old widget.type='hidden' fallback
  only suppresses the LiteGraph canvas draw but leaves the DOM element (widget.element) visible. Add
  both properties and optionally call node.setSize() when a node reference is provided.

### Documentation

- **scene**: Clarify SceneCastBuild works with PromptCompositionLoader
  ([`e1b8f0c`](https://github.com/frost-byte/fbTools/commit/e1b8f0c341c21591c83a93d5fba78f727038ea32))

Both SceneCastLoad and SceneCastBuild output SCENE_CAST type, so either wires into
  PromptCompositionLoader's scene_cast input. Update tooltip and docstring to reflect this.

### Features

- **audio**: Add MelBand Roformer vocal extraction to preprocessing pipeline
  ([`c96adcc`](https://github.com/frost-byte/fbTools/commit/c96adcce2a6e2a07fcba0cca0323d9b4f53ea243))

When a MelBand Roformer safetensors checkpoint is configured, noise_removal now runs source
  separation instead of spectral subtraction, producing a clean vocal stem. Falls back to the
  existing spectral-subtraction path when no model path is set. Model is cached in-process after
  first load.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **backgrounds**: Replace inline form with modal + LLM analysis
  ([`fcbe2d8`](https://github.com/frost-byte/fbTools/commit/fcbe2d8a6b2f0b095e8780e2ea918df7169d49cf))

Convert background create/edit from an inline sidebar form to a full modal matching the outfit
  editor pattern.

New capabilities: - File browser (Input/Output tabs, tree, image/video preview, frame picker) -
  Reference images list with thumbnails and role dropdown - LLM analysis via POST
  /fbtools/backgrounds/analyze_media — asks the model to return structured JSON and auto-fills
  Description, Lighting, and Soundscape fields in one call - Clicking a reference thumbnail loads it
  into the browser preview - Delete button moved into the modal footer

Backend changes: - New /fbtools/backgrounds/analyze_media endpoint with JSON-structured LLM query
  (description + lighting + soundscape); strips markdown fences and falls back to raw text if JSON
  parse fails; supports folder param so output-dir images work too - list_backgrounds() now returns
  full records (was only id/name/description summaries, so lighting/soundscape were lost on edit) -
  background schema gains reference_images field (persisted as-is by existing save_background
  passthrough) - analyzeBackground() added to CompositionsAPI

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **bundle**: Add audio preprocessing pipeline with caching
  ([`827261e`](https://github.com/frost-byte/fbTools/commit/827261ef58493d6caa2a9038b71d40a3856be46a))

- utils/audio_preprocess.py: pure numpy pipeline (spectral denoise, LUFS normalize via pyloudnorm,
  loop/truncate) with cache fingerprinting - POST /fbtools/bundles/preprocess_audio endpoint: runs
  pipeline in executor, caches result as WAV under user_data_dir/bundles_cache/ - audio_cache field
  plumbed through prompt_compositions, prompt_assembler, _resolve_cast_media, and
  CompositionToH3Conditioning to bypass raw load - Bundle Editor: _buildAudioProcessingSection()
  with noise/LUFS toggles, target LUFS input, status badge, and Process Audio button -
  tests/test_audio_preprocess.py: 36 tests covering all pipeline steps

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **bundle**: Add sampled preview with frame-count readout
  ([`5e47950`](https://github.com/frost-byte/fbTools/commit/5e47950e592b468144219194397b966d054099fe))

- Remove frame_load_cap and skip_first_frames from video section UI (start/end is now set via
  slider/mark, cap defaults to 0) - Add live frame-count readout: "~N frames · X fps effective"
  updates when any of FPS override, Every Nth, start, duration, or slider changes - Add POST
  /fbtools/bundles/preview_sampled endpoint: cv2 time-accumulator resampling (mirrors H3
  conditioning logic) → ffmpeg pipe → fragmented MP4 - Add "▶ Preview Sampled" button in the video
  section: plays sampled preview in a separate in-panel video element; falls back to native clip
  loop when ffmpeg unavailable with an explanatory note - Remove frame_load_cap default from 96 to 0
  for new bundles

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **bundle**: Add separate video file as audio source for Reference Bundles
  ([`b4e7fbf`](https://github.com/frost-byte/fbTools/commit/b4e7fbfd99da7fb677722bde38722c5d2ee0aba6))

Users can now excerpt audio from a different video than the visual reference. Adds
  extract_from_video audio source option in bundle_editor.js and handles both the per-entry and
  legacy flat paths in _resolve_cast_media.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **bundle**: Add time-based trim (start_time/duration) to video visual references
  ([`967a6ef`](https://github.com/frost-byte/fbTools/commit/967a6ef2a76059c1db046d1bed4311a0d61e38d8))

_h3_load_video_frames now seeks to start_time via cap.set(CAP_PROP_POS_MSEC) and caps the output
  frame count from duration * target_fps. Both params are propagated through _resolve_cast_media
  entry_load_params and defaulted in BundleRegistry.upsert(). Bundle editor video visual section
  gains a "Trim" section with Start (s) / Duration (s) number inputs above the existing frame
  sampling controls.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **bundle-editor**: Add processed audio preview player
  ([`1e07f2e`](https://github.com/frost-byte/fbTools/commit/1e07f2e589573d656f778d7330fd1eb63ff09f44))

After running "Process Audio", an <audio> player appears below the status line so the processed
  (denoised/normalized) result can be listened to without leaving the editor.

Backend: GET /fbtools/bundles/audio_cache/stream?path=<abs_path> serves files from
  user_data_dir()/bundles_cache/ with Range support. Path is validated to stay within bundles_cache/
  root.

Frontend: _buildAudioProcessingSection now creates a hidden <audio> element alongside the status
  line. _updateStatus shows/hides and reloads the player whenever audio_cache changes — on initial
  render (if cache already set), after a successful Process run, and when settings are changed
  (clears cache → hides player).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **bundle-editor**: Add Subjects tab, image tree browser with folder tabs and dual preview
  ([`638b594`](https://github.com/frost-byte/fbTools/commit/638b594a9ddca4944814128250c91105ab549897))

- Add Subjects tab to Reference Bundle editor with full profile editor (name, ID, concept ID,
  appearance w/ collapsible subfields, character sheet images with per-image role selects, voice
  settings, LLM analysis) - Replace flat image picker with lazily-rendered collapsible tree browser
  supporting subdirectories - Add Input/Output folder tabs to the tree browser so images from either
  ComfyUI directory can be browsed with identical tree-view behaviour - Split the single preview
  pane into two independent panes: a stable selected-image preview (click any assigned filename to
  change it, tracks reorder/remove) and a dynamic browse-hover preview below the tree - Backend: add
  `folder` param to /fbtools/media/list (input|output); add output-dir fallback for subject image
  loading and LLM image loading

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **cast**: Add media frame extraction endpoints and scene synopsis input
  ([`9bf9949`](https://github.com/frost-byte/fbTools/commit/9bf994986a3fc8843d1c3933e3e98a19308b1ff6))

Add REST endpoints for extracting a single frame from a video file (_media_extract_frame,
  _media_delete_tmp_frame) with temp file cleanup via _purge_old_tmp_frames. Add scene_synopsis
  string input to SceneCompose, injected into the SCENE_INSTANCE for H3 summary override.

Wire frame extraction into the bundle editor UI (visual frame picker for thumbnail/reference
  selection) and expand SceneCastBuild JS node with bundle media preview and audio picker support.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **cast**: Positional subject recasting in Scene Cast
  ([`3ba7c1d`](https://github.com/frost-byte/fbTools/commit/3ba7c1dd0a4f0b9b0a671485f96c0399caa95254))

Cast entries now map to composition slots by row order rather than subject_id identity. Entry 0
  targets S1, entry 1 targets S2, etc. A blank entry (no subject_id) is a pass-through that keeps
  the composition's original subject unchanged.

When a subject_registry is available and the entry's subject_id differs from the slot's current
  subject, the slot's subject is fully replaced (name, appearance, voice, concept_id) before bundle
  enrichment applies. This enables recasting: telling the Prompt Composition Loader "use Angie in S1
  and Joe in S2 for this shoot, regardless of who the composition originally assigned."

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **composition**: Add speech_pace to shot dialogue with trim_to estimation
  ([`28bfe33`](https://github.com/frost-byte/fbTools/commit/28bfe338c7beb02ea160711a0b6531fd45878f35))

Adds a per-shot `speech_pace` field ("slow" / "normal" / "fast") to the composition dialogue schema.
  The field drives two things simultaneously:

- **Prompt injection**: slow/fast paces append a qualifying phrase to the shot's action text
  ("speaking slowly and deliberately" / "speaking quickly") in both h3_ref2va and h3_fl2va formats.
  Normal pace emits nothing. - **Voice reference trim_to**: `PromptCompositionLoader` estimates the
  expected spoken duration of each shot's resolved dialogue (via `estimate_speech_duration`, a new
  public utility) and accumulates it per slot. The per-slot totals are passed to `_build_h3_refplan`
  as `slot_trim_to`, injecting a `trim_to` field into standalone audio reference entries so the
  terminal node can trim the voice clip to match the generated line length.

Pace → chars/sec mapping: slow=10, normal=13, fast=16 (documented in
  `docs/audio_reference_observations.md` for field revision as data arrives).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **composition**: Enhance prompt assembler and add filename_prefix output
  ([`0c316be`](https://github.com/frost-byte/fbTools/commit/0c316be3b66207e7a4a68c3d620ee5d87896d30e))

Expand prompt_assembler.py with additional model-type formatters and assembly logic. Add
  filename_prefix string input/output to PromptCompositionLoader so the composition name can be
  wired directly into a VHS_VideoCombine filename_prefix input.

Update composition editor UI, LLM client/scanner utilities, and extend test_prompt_assembler.py
  coverage. Add prompt_assembly.md docs.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **composition**: Improve Saved Compositions UX and add LoRA search
  ([`fc7eb80`](https://github.com/frost-byte/fbTools/commit/fc7eb80b2cc14fc516194867224aafeacb6c278a))

- Saved Compositions list: add search field (filters by name/ID) and pagination (10 per page) with
  prev/next controls - Make entire composition row clickable to load; remove redundant ⇩ button;
  delete button uses stopPropagation to avoid double-trigger - Active composition indicator: 3px
  green bottom border on the currently loaded entry in the saved list - LoRA name selector: replace
  plain <select> with searchable combobox; type to filter the full LoRA list, arrow keys to
  navigate, Enter to select; dropdown uses position:fixed to avoid sidebar clipping

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **compositions**: Add background and outfit visual reference subjects for H3 prompts
  ([`bdbab76`](https://github.com/frost-byte/fbTools/commit/bdbab765b1b612ca478fe1fe70e390c3e0d8dfcc))

Backgrounds: new "Include as <Subject N>" checkbox on the Background section of the composition
  editor. When checked, the background's reference_images are injected as an extra slot in the
  assembled prompt using the {BG} shortcut in shot action/camera text.

Outfits: the outfit dropdown in each subject slot now stores an outfit ID in outfit_ids[slot_key]
  rather than description text. Each outfit reference image gains a use_as_reference flag
  (per-image, toggled from the outfit modal). Outfits with at least one flagged image are injected
  as their own <Subject N> slot using {Fit_1}/{Fit_2}/… shortcuts. Outfits with no flagged images
  contribute their description text to the subject's appearance phrase (text-only path).

Background and outfit extra slots share a running letter counter so they coexist correctly (subjects
  → BG → Fit_1 → Fit_2 in alphabetical slot order).

Both call sites of assemble_composition() (REST endpoint + PromptCompositionLoader node) now resolve
  and pass resolved_outfits alongside resolved_background.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **conditioning**: Add §1 validation, trim_to, and turbo warning to CompositionToH3Conditioning
  ([`9ba1463`](https://github.com/frost-byte/fbTools/commit/9ba14630720f576d433a487de6e30f8bc71b24e2))

Enforces all MiniMax H3 Ref2VA hard limits before delegating to the native node, with explicit
  errors rather than silent truncation:

- Pre-load: ≤ 3 standalone audio refs, audio must pair with a visual, ≤ 12 total reference files,
  trim_to < 2s caught early with an actionable message - Post-load (actual samples): per-clip 2–15s,
  total audio ≤ 15s - trim_to applied to waveform via sample-level slice after ffmpeg load - Turbo
  LoRA warning (logged + status update) when has_turbo_lora is set on the refplan and audio
  references are present

PromptCompositionLoader now sets has_turbo_lora on the refplan dict by checking whether any attached
  LoRA name contains "turbo".

Status line now includes total loaded audio duration.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **history**: Add Run History panel and Run Meta Capture node
  ([`ceb6b98`](https://github.com/frost-byte/fbTools/commit/ceb6b98458c2ff60355fc053ece37daad8c47fe7))

- RunMetaCapture node: captures runtime string values at execution time, stores by prompt_id;
  autogrow value slots (up to 12); supports partial execution via is_output_node play button; inline
  text preview - [track: Label] tag system: right-click any node to embed a tracking tag in its
  title; green pill bar drawn via onDrawForeground - Run History sidebar panel: merges /history
  widget snapshots with RunMetaCapture captures keyed by prompt_id; shows timestamp, workflow name,
  and short prompt ID chip per run - LoraStackBuilder custom renderer: filters disabled/None LoRA
  slots, suppresses video/audio columns for non-LTX targets, shows model badge

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **llm**: Add LLM video analysis assistant with Qwen2.5-Omni GPTQ support
  ([`084e395`](https://github.com/frost-byte/fbTools/commit/084e3955d52f28e51d5cf653a950bf4b16fccac3))

Adds an LLM assistant to the Composition Editor for analysing video clips and generating shot action
  descriptions.

## Infrastructure - `utils/llm_scanner.py`: scans ComfyUI LLM directories for GGUF and HF models;
  detects vision, native-video, and quantisation capabilities - `utils/llm_client.py`:
  load/unload/generate for GGUF (llama-cpp-python) and HF (transformers) models; task-specific
  prompt builders - REST endpoints: `/fbtools/llm/{models,status,load,unload,generate,
  generate/shot_action,generate/dialogue,generate/polish,
  describe_video,video_prompt,download/default}`

## Qwen2.5-Omni GPTQ workarounds - `block_name_to_quantize`: optimum's BLOCK_PATTERNS list omits
  `thinker.model.layers`; patching the embedded quantisation config dict before `from_pretrained`
  bypasses the pattern scan - `return_audio=False` + `thinker_max_new_tokens`: Omni's `generate()`
  returns `(text, waveform)` by default; standard token-slice decode indexed the batch dim instead
  of seq dim, producing empty text - Omni default system prompt prepended to custom instructions to
  silence the "audio output may not work" warning and stay in the model's trained operational mode -
  AWQ Triton monkey-patch retained for future use: forces `dequantize_gemm+matmul` fallback when the
  Triton bitshift kernel fails on packed int4 float16 weights - `max_memory` capped at 70% GPU / 64
  GiB CPU so LLM and diffusion models can coexist in VRAM

## Temporal RoPE encoding - `_extract_frames` now returns `{sample_fps, raw_fps, duration}` -
  `sample_fps = len(selected_frames) / clip_duration` is passed in the video content element so
  qwen_omni_utils computes correct temporal position IDs (previously defaulted to 2.0 fps regardless
  of actual frame rate)

## Composition Editor UI - "Describe from Video" modal: filmstrip with carousel, frame selection,
  frame budget indicator (tokens/frame from Omni's patch/merge params) - Modal stays open after
  inference; result appears in editable textarea; "Send to Action" pushes text into the focused shot
  card's Action field - Collapsible "Edit Prompt" section pre-populates system + user prompts from
  `/fbtools/llm/video_prompt`; edits are sent as overrides

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **lora**: Add Enabled Summary output to LoraStackBuilder
  ([`8aaedc4`](https://github.com/frost-byte/fbTools/commit/8aaedc4f7381e64e3b34af121812f05ab2d8a574))

Adds a new string output that lists only enabled LoRAs, one per line, in the format: name
  model/clip[/video/audio]. Name is the basename truncated to 48 chars with extension stripped;
  values use minimal decimal notation (1, 0.5, 0.75). Output always ends with a trailing newline for
  easy concatenation with other string nodes.

A boolean input (Summary: Include Prev Stack, default off) controls whether the summary covers only
  LoRAs defined in this node or all merged entries including Prev Stack.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **outfit**: Replace LLM analyze text input with tree browser + preview
  ([`57faaf0`](https://github.com/frost-byte/fbTools/commit/57faaf073a6e8443459034ab7a47e9bb9332d899))

- Add Input/Output folder tabs with lazy subdirectory tree showing both images and videos combined
  (same tree pattern as bundle editor) - Add image preview pane (img) and video preview pane (video
  w/controls) that update when a file is selected in the tree - Add frame-time row (hidden for
  images, shown for videos): number input syncs bidirectionally with the video element's
  currentTime; "↺ Use current" button captures the scrubbed position from the video player - Pass
  frame_time in the analyze_media request when a video is selected - Add listMedia() to
  CompositionsAPI; load all four media lists (input/output × image/video) in _loadResources via
  Promise.allSettled

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **outfit-modal**: Click ref thumbnail to load into browser and SAM2
  ([`1bc7150`](https://github.com/frost-byte/fbTools/commit/1bc7150c8130a87fcad16d1ebacee9acda208807))

Clicking any reference image thumbnail in the ref list now calls _applySelection(), which updates
  the file browser selection, the preview, and fires _onSelectionForSam2 — so the user can jump
  straight from an existing reference into SAM2 segmentation without re-browsing the tree.

Teal hover glow on thumbnails signals the SAM2 association.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **outfits**: Add reference images and image/video LLM analysis
  ([`9e9ce2d`](https://github.com/frost-byte/fbTools/commit/9e9ce2d3d8344b847fbf7a42142c1efd0c3c2637))

OutfitRegistry entries now carry reference_images: [{file, role}]. Legacy plain strings
  auto-normalize to {file, role: "costume detail"}.

Backend: - POST /fbtools/outfits/analyze_media: accepts image or video filename, extracts frame at
  1s for video (saved permanently as _outfit_ref_*.jpg), runs loaded LLM, returns {description,
  frame_file} - POST /fbtools/outfits/save: now accepts reference_images array

Frontend outfit editor: - Image-only input replaced with image/video input; shows hint when video is
  detected explaining frame extraction - After LLM analysis, analyzed file (or extracted frame) is
  appended to a mutable reference list when "Add as reference image" is checked - Reference list
  shows each entry with role dropdown and delete button - Save persists reference_images alongside
  name/description/tags

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **scene**: Add Scene Cast input to PromptAssemble for video references
  ([`0899c73`](https://github.com/frost-byte/fbTools/commit/0899c73b7d1a9e9d36d2798dd9deb249151dfcf9))

PromptAssemble was calling assemble_prompt() with video_entries=None, so subjects whose cast bundle
  has visual_mode='video' never received a <Video N> label in the H3 subject_definitions section.

Add an optional SCENE_CAST input; when connected, _resolve_cast_media() extracts video_entries_full
  and passes it to assemble_prompt() so video-referenced subjects are correctly labelled in the
  assembled prompt.

- **settings**: Add global audio + speech pace defaults to composition settings
  ([`e41ba56`](https://github.com/frost-byte/fbTools/commit/e41ba56bae34e91158741f4d20c15f08661d71d2))

Backend (extension.py): - _COMPOSITION_SETTINGS_DEFAULTS adds default_speech_pace, default_audio_*,
  and melband_model_path alongside the existing libber_delimiter - POST
  /fbtools/compositions/settings validates and persists all new fields (pace: slow/normal/fast enum;
  LUFS: clamped to −36..−6; others: bool/str)

Composition editor UI (_buildSettingsSection): - "Default speech pace" dropdown (slow/normal/fast
  with WPM hint) - "Default audio processing" group: noise removal checkbox, LUFS normalize
  checkbox, target LUFS numeric input - "Vocal isolation" group: MelBand model path text input
  (reserved) - All controls stored in _dom for post-load sync in _loadResources() - New dialogue
  speech_pace defaults to _S.settings.default_speech_pace

Bundle editor: - Imports compositionsApi; _loadAll() fetches global settings into _S.settings -
  _startNew() reads default_audio_* to pre-fill new bundles' audio_processing

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **subjects**: Add per-image roles to character_sheet_images
  ([`684d8f9`](https://github.com/frost-byte/fbTools/commit/684d8f9e72cc9cd330694ded547ce8d110456713))

Changes character_sheet_images from list[str] to list[{file, role}]. Existing plain strings
  auto-migrate to {file, role:"character sheet"}.

Adds <Picture N> role lines to H3 ref2va subject_definitions so the model knows each image is an
  appearance reference and not a scene composition template — prevents spatial bleed from portrait
  framing (where a portrait shot's left-side face anchor was being replicated in the output layout).

Role descriptions always end with "do not use as scene composition". Canonical roles: character
  sheet, portrait, side profile, full body, costume detail, reference. Free-form strings are also
  accepted.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Add reusable FileTree component for input/output file browsing
  ([`38ba44c`](https://github.com/frost-byte/fbTools/commit/38ba44cbf7c352c20df3a77adc6253990e5de153))

Extracted from the composition editor's video/image file selectors into a standalone
  js/ui/file_tree.js module with Input/Output tabs, lazy folder expansion, and fbt-be-tree-file-cur
  highlight tracking.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Add Subject editor to Reference Bundles panel
  ([`f792d7c`](https://github.com/frost-byte/fbTools/commit/f792d7ccf366f30b3bde4507e08abdc750f71d2b))

Adds a full subject creation/editing UI to the bundle_editor sidebar panel under a new Bundles |
  Subjects tab switcher. Subjects now have a dedicated editor with all profile fields: name, ID,
  concept ID, appearance (summary + face/hair/body/outfit detail fields in a collapsible), voice
  (description, language, audio reference file dropdown), and a character_sheet_images list with
  per-image role dropdowns (character sheet, portrait, side profile, full body, costume detail,
  reference). Optional LLM appearance analysis fills the appearance summary from an image when a
  vision model is loaded.

Also adds getSubject, saveSubject, and deleteSubject methods to BundlesAPI and CSS for the tab
  switcher, subfield grid, and sheet-image rows.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Add video/audio/image previews and dual-handle trim slider to bundle editor
  ([`67c1210`](https://github.com/frost-byte/fbTools/commit/67c1210f2a2e7ca857d550ecadaf8779511cd798))

Video visual section: - <video controls> player shown when a file is selected (streams via new GET
  /fbtools/media/stream endpoint which supports HTTP Range for seeking) - Info line shows duration,
  fps, resolution, frame count - Dual-handle range slider lets users drag start and end of the trim
  region; the green fill shows the selected segment - "◁ Mark Start" and "Mark End ▷" buttons stamp
  the slider handles at the current video player position (scrub to find the frame, then click) -
  Slider and number inputs stay two-way synced; duration=0 preserved as "to end of file" when the
  right handle is dragged all the way to the end

Image visual section: - Hover any image row to see a full-width thumbnail preview above the list -
  Preview also appears when an image is added from the dropdown

Audio section (file / extract_from_video): - <audio controls> player appears below the file selector
  and updates its src when the selection changes

Backend: - GET /fbtools/media/info — returns duration, fps, width, height, frame_count for any file
  in the input directory (uses cv2) - GET /fbtools/media/stream — streams the file with
  range-request support so the browser can seek without downloading the whole file

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Apply pagination, expanded click area, and active indicator to all list sections
  ([`4ae847d`](https://github.com/frost-byte/fbTools/commit/4ae847dc738cce0a5852cfbbaea272e6391282a0))

- Composition sidebar: Subjects and Backgrounds lists now paginate (10/page) and highlight currently
  assigned subjects / active background with green underline - Bundle editor: list paginates
  (10/page), full card click opens editor, most-recently-saved bundle underlined on return to list
  view - Cast editor: list paginates (10/page), active indicator on last-saved cast - Filter/search
  changes reset page to 0 in bundle editor - Shared pagination uses existing
  fbt-ce-pg-btn/info/saved-pagination CSS classes - New fbt-be-card-clickable / fbt-be-card-active
  CSS rules for card hover and selection

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Green dot badge on Prompt Composition sidebar tab when LLM is loaded
  ([`4eae2fc`](https://github.com/frost-byte/fbTools/commit/4eae2fc7fbd9ed6921abae4c0b6e7687246e1cdd))

Uses a CSS ::after pseudo-element on .sidebar-icon-wrapper inside the tab button (targeted via the
  stable data-testid="fbt.composition-editor-tab-button" attribute that ComfyUI derives from our
  registered tab ID). The body class fbt-llm-loaded is toggled by _llmSyncBadge(), called at every
  point _S.llmLoaded changes (initial load, model load success/failure, and unload).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Replace image dropdown with collapsible tree browser
  ([`a43c8dc`](https://github.com/frost-byte/fbTools/commit/a43c8dc65c5d4a0221aa92ba7f3b3d4455feab53))

Adds subdirectory support to the image picker in the bundle visual section: - Backend:
  /fbtools/media/list now accepts ?recursive=true, using os.walk() to return relative paths
  including subdirectories (e.g. portraits/alice.jpg) - Frontend: replaces the flat <select>
  dropdown with a collapsible tree browser showing dirs as lazy-expanded nodes and files as
  clickable leaves - Selected state syncs back to the tree (checkmark + dimmed) when files are added
  or removed from the list - Hover preview now stays visible when moving between the selected-files
  list, the tree browser, and the preview image, by wrapping all three in a single hover container
  rather than attaching handlers to each element independently - Adds _viewUrl() helper that
  correctly splits subfolder/filename for the ComfyUI /view endpoint so subdirectory images preview
  correctly

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

### Refactoring

- **h3**: Extract CompositionToH3Conditioning validation into pure helpers
  ([`51560ab`](https://github.com/frost-byte/fbTools/commit/51560ab3f66d3e12c8a80e08425933c4bca9eb60))

Move the three §1 invariant checks out of extension.py's execute() method and into testable
  functions in utils/prompt_assembler.py: - validate_h3_refs_pre(references) -> list[str] (pre-load:
  count/pairing/trim_to) - validate_h3_audio_clip(dur, aord, basename) -> str | None (per-clip 2–15
  s) - validate_h3_audio_total(durations) -> str | None (total ≤ 15 s)

extension.py delegates to the imported helpers; behaviour is unchanged. 29 new tests cover every
  rule and boundary value.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **outfit**: Unify outfit modal into single file browser + action flow
  ([`eff13ac`](https://github.com/frost-byte/fbTools/commit/eff13ac795a2adf19aa7f425b157f0aac070f53c))

Replace the fragmented layout (refs list → LLM section → SAM2 section, each with its own file
  picker) with one coherent structure:

- Single File Browser section (always visible, tree + Input/Output tabs + preview + frame-time row
  for video) at the top - Two action buttons beneath: "+ Add as Reference" (images only, no LLM
  required) and "🔍 Analyze with LLM" (images+video, LLM optional) - LLM query textarea shown only
  when a vision model is loaded - SAM2 section below the browser: no longer has its own srcInput —
  selecting an image from the browser populates the SAM2 click-to-point preview automatically;
  selecting a video disables extraction - Analyze always adds the result file to refs (checkbox
  removed) - Reference Images list moves to bottom as the output of all three paths

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

### Testing

- Fix 4 stale test assertions
  ([`98af771`](https://github.com/frost-byte/fbTools/commit/98af7718768686c3cc7665d6228cd34f2ec8b47e))

- test_s1_maps_to_slot_a / test_two_subjects_remapped_in_order: H3 ref2va uses <Subject N> labels,
  not names; assert appearance summary text ("tall woman", "short man") rather than the name -
  test_dialogue_tags_use_subject_language: <d> wrapping requires use_dialogue_tags=True on the
  composition; set it explicitly - test_style_retention_analysis_line: phrasing is "tonal style",
  not "audio style"; update assertion to match

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.19.1 (2026-08-13)

### Bug Fixes

- **ui**: Suppress boolean widget draw() to prevent toggle leaking through in SceneCastBuild
  ([`0a4d76a`](https://github.com/frost-byte/fbTools/commit/0a4d76ae991833995e97c936eae59cb767097052))

ComfyUI V3 toggle widgets have a custom draw() that can bypass the type="hidden" check used by
  setWidgetVisible. Fix by iterating all standard widgets (not by name to avoid any lookup miss) and
  overriding draw() as a belt-and-suspenders guard. Resolves Audio 4 appearing above the slot table.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.19.0 (2026-08-13)

### Features

- **scene**: Add SceneCastBuild node for inline cast configuration
  ([`6bb22cc`](https://github.com/frost-byte/fbTools/commit/6bb22cc76bc18c771531e22778a15aec62909e9c))

New node builds a SCENE_CAST without a saved file. Outputs the same SCENE_CAST type as SceneCastLoad
  — wires into PromptCompositionLoader unchanged.

- Python: SceneCastBuild with 4 slots (subject/bundle/visual_mode/use_audio each), all hidden behind
  a DOM table widget in the frontend. - JS: scene_cast_build.js renders a compact 4-row interactive
  table. Selects are populated from the live subject/bundle registry via API. Changing any field
  syncs back to the hidden widget and marks the canvas dirty. _refreshCastTable() exposed for editor
  integration. - Cast editor: "Send to Workflow" button in the top bar discovers fbt_SceneCastBuild
  nodes on the canvas, shows a picker, and pushes the current editing entries to the selected node.
  "Create new" option adds a fresh node and positions it near the canvas mouse.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.18.0 (2026-08-13)

### Features

- **ui**: Add Send to Workflow button to Composition Editor
  ([`49cf283`](https://github.com/frost-byte/fbTools/commit/49cf283d6ea575b4c5b3bbdb5a31b0355a8a8df3))

Opens a picker listing all PromptCompositionLoader nodes on the canvas by their current
  composition_name widget value. Selecting a node sets its widget to the current composition.
  "Create new" option adds a fresh node and sets its widget via a short timeout after graph.add().

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.17.1 (2026-08-13)

### Bug Fixes

- **node**: Fold composition file mtime into PromptCompositionLoader fingerprint
  ([`f7d57f1`](https://github.com/frost-byte/fbTools/commit/f7d57f14de1d98e342ae2528503043df75550c68))

The fingerprint previously keyed only off the compositions directory mtime and the reload counter.
  Directory mtime moves when files are added or removed, but NOT when an existing composition file
  is edited in-place (e.g. via a text editor or external tool). Out-of-band JSON edits were
  invisible to the cache until the user manually hit the reload button.

Fix: resolve the matched composition file path (comps_dir/<id>.json) and include its getmtime in the
  fingerprint tuple alongside the directory mtime. The reload counter is still present as an escape
  hatch for cases where mtime is unreliable (network filesystems, etc.).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.17.0 (2026-08-13)

### Features

- **node**: Add CompositionToH3Conditioning terminal node (Steps 5-7)
  ([`9bef84b`](https://github.com/frost-byte/fbTools/commit/9bef84b7abe37aa9362da4f778d45d38b4d10b78))

Completes the H3 refplan action plan.

Three media-loading helpers added above the node class:

_h3_resolve_path(path) — absolute-or-input-dir path resolution _h3_load_image(path) — PIL →
  [1,H,W,3] float32 tensor _h3_load_video_frames(path, params) — VHS cv_frame_generator adapter →
  [B,H,W,3] float32 tensor _h3_load_audio(path, start, dur) — VHS get_audio (ffmpeg) with torchaudio
  fallback for standalone files

CompositionToH3Conditioning (category: conditioning):

Inputs: h3_refplan (FBTOOLS_H3_REFPLAN), clip, vae, audio_vae, width, height, length, ref_image_size
  (match|max) Outputs: positive (CONDITIONING), LATENT

fingerprint_inputs: md5 hash of bundle JSON + getmtime of every referenced file +
  width/height/length/ref_image_size. Invalidates on any file edit without needing a reload counter.

execute: iterates the bundle's references list in order (images → [soundtrack+video] pairs →
  standalone audio), loads each via the above helpers, assembles
  ref_images/ref_videos/ref_video_audios/ ref_audios dicts with 0-based suffix keys, and delegates
  to MiniMaxH3ReferenceToVideo.execute (lazily imported). Suffix pairing convention: ref_video_N ↔
  ref_video_audio_N (same video_ordinal-1). Standalone audio uses a sequential 0-based counter
  independent of audio_ordinal.

Graceful degradation: missing files are logged and skipped; an empty reference set degrades to
  text-to-video (native node handles it).

Node registered in FBToolsExtension.get_node_list().

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.16.0 (2026-08-13)

### Features

- **assembler**: Retention-aware audio phrasing and fix <Audio N> ordinals
  ([`17ffd95`](https://github.com/frost-byte/fbTools/commit/17ffd95246bcf33915c10267369dfab4756e22e9))

Step 2 of the H3 refplan action plan.

Bug fixed: _build_ref_map assigned standalone audio ordinals starting at 1 without accounting for
  soundtrack audios (extract_from_visual). When both types were present, the <Audio N> label in the
  assembled prompt would not match what MiniMaxH3ReferenceToVideo assigns at inference time.

Fix: two pre-passes now assign audio ordinals in native ref_items order — soundtracks first (slot
  order), then standalone files (slot order) — exactly mirroring the three-pass algorithm in
  _build_h3_refplan. Ordinal parity is now guaranteed across the prompt assembler and the terminal
  node.

New ref_map fields: audio_retention, audio_role (standalone), soundtrack_num, soundtrack_retention,
  soundtrack_role (for extract_from_visual video entries).

_assemble_h3_ref2va updated in three places:

subject_definitions: - Soundtrack entries now get their own <Audio N> line. - timbre → "…without
  copying the original signal" (matching the h3_prompt libber's %nocopy% convention) - reuse →
  "…reproduced verbatim" - style → "audio style and rhythm reference…" - Non-empty audio_role
  overrides the generic description entirely.

summary: - has_audio now includes soundtrack_num; both audio modalities contribute to the [audio
  reference] task tag and the closing sentence. - Closing sentence uses retention-appropriate phrase
  per audio entry.

retention_analysis: - timbre → "reference - its vocal timbre guides … without copying the original
  signal" (replaces old "reference (voice timbre only, not fully_copy)") - reuse → "fully_copy" -
  style → "reference (audio style, not fully_copy)" - Soundtrack audio entries appear before
  standalone in retention_analysis.

16 new tests in tests/test_h3_audio_phrasing.py cover ordinal correctness (both modality types,
  mixed cases) and retention/role phrasing in all three sections. Updated one existing test whose
  assertion matched old phrasing.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.15.0 (2026-08-13)

### Features

- **cast**: Add FBTOOLS_H3_REFPLAN bundle output to PromptCompositionLoader
  ([`5b5c8e9`](https://github.com/frost-byte/fbTools/commit/5b5c8e9c3576b93e21051b283be5446be241eda1))

Steps 0–4 of the H3 refplan action plan:

- Step 0: Add retention/role fields to bundle audio schema and editor UI (timbre/reuse/style modes;
  free-text role label; shown for both extract_from_visual and file sources in bundle_editor.js)

- Step 1: Implement _build_h3_refplan() in utils/prompt_assembler.py Three-pass algorithm mirrors
  native MiniMaxH3ReferenceToVideo ref_items order: images → [soundtrack_audio + video] pairs →
  standalone audios. Ordinals (picture_ordinal/video_ordinal/audio_ordinal) are assigned so <Picture
  N>/<Video K>/<Audio J> labels in the prompt match what the tokenizer derives from ref_items.
  Parity verified by 9 new tests in tests/test_h3_refplan_parity.py.

- Step 3: Extend _resolve_cast_media to collect video_entries_full — all video-mode cast entry
  descriptors with full audio config (source, path, start_time, duration, retention, role). Existing
  flat outputs unchanged.

- Step 4: Add FBTOOLS_H3_REFPLAN wire type and H3RefplanType class. PromptCompositionLoader gains an
  h3_refplan output; execute() builds the bundle from enriched resolved_subjects +
  video_entries_full and attaches prompt/model_type/ref_image_size before returning.

Also: fix apply_cast_to_subjects so extract_from_visual audio no longer sets
  voice.audio_reference_file (it is a video soundtrack handled by the refplan's soundtrack_audio
  pass, not a standalone voice reference). Verified by updated test_cast_enrichment.py (15 tests,
  all passing).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.14.0 (2026-08-12)

### Bug Fixes

- **assembler**: Align H3 subject_definitions reference phrasing with empirical best practice
  ([`2379a45`](https://github.com/frost-byte/fbTools/commit/2379a45313cea820131c4e18d4bd63d0773673d9))

Matching the manually-crafted prompt format that produces better results:

- Video inline citation: "from <Video N>" instead of "in <Video N>" - Picture inline citation: "from
  the character sheet contained in <Picture N>" (singular) or "from the character sheets contained
  in <Picture N> and <Picture M>" (plural) instead of plain "in <Picture N>" - Add standalone <Video
  N> role lines after subject lines, before audio lines; description is task-flag-aware: video
  continuation → "is the continuation starting point for the target video" video editing → "is the
  source video being edited" default → "is the visual identity reference for <Subject N>" - Hoist
  active_flags computation to top of _assemble_h3_ref2va so it is available in both
  subject_definitions and summary sections - Update 2 existing tests; add 5 new tests

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **assembler**: Align H3 task types with official MiniMax docs
  ([`69b5ad1`](https://github.com/frost-byte/fbTools/commit/69b5ad144477a48f0f3fa28f1b6de09cdeb74fbe))

Per the MiniMax H3 Ref2VA specification, the valid task types are: reference generation, keyframe
  completion, video editing, video continuation, audio reference, audio reuse.

"video reference" is not an official type.

- Auto-detection: pictures AND videos that provide guidance both fall under "reference generation"
  (same bucket per spec); voice timbre files stay as "audio reference" - "video editing" / "video
  continuation" / "keyframe completion" / "audio reuse" cannot be auto-detected and require user
  task_flags - video editing tasks open the summary body with the required sentence: "The target
  video is an edited version of <Video N>." - Composition Editor task flag checkboxes updated to 6
  official types - PromptAssemble tooltip updated with full type list - All affected tests updated;
  5 new tests added

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **assembler**: Rewrite H3 Ref2VA subject_definitions to match official MiniMax format
  ([`7598601`](https://github.com/frost-byte/fbTools/commit/7598601086ba9dcd71b35c48d30757f0fe123497))

Rewrites the subject_definitions section of _assemble_h3_ref2va to use the official MiniMax H3
  single-line prose format per subject, with picture/video references cited inline and audio
  references as separate bottom entries.

- subject line: "<Subject N> is [summary] in <Pic N> [and <Pic M>], with [details]." - audio line:
  "<Audio N> is the voice-timbre reference for <Subject N> (S1), containing [voice]." - removes old
  multi-line Face:/Hair:/Body:/Outfit: sub-bullets - removes standalone <Video N>: reference lines
  (video now inline in subject line) - fixes auto-detected task tag from "video continuation" to
  "video reference" - adds task_flags user override on PromptAssemble node and assemble_composition
  path - adds task flag checkboxes in Composition Editor Info section (h3_ref2va only) - updates all
  affected tests to match new format; adds 3 new task_flags tests

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **assembler**: Use neutral from <Picture N> phrasing for image references
  ([`03ed843`](https://github.com/frost-byte/fbTools/commit/03ed843ef69dc20cad3822054ad066265a7104f5))

Drops the "character sheet contained in" qualification since picture references may be any type —
  individual shots, style references, poses, environments, etc. Plain "from <Picture N>" is
  consistent with "from <Video N>" and makes no assumptions about image purpose.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

### Features

- **cast**: Wire Reference Bundle fields into H3 Ref2VA prompt assembly
  ([`85c1411`](https://github.com/frost-byte/fbTools/commit/85c1411ea290b4493e186d61afed510c97ea140b))

Enrich resolved subjects from Scene Cast bundle data before prompt assembly: image-mode visual.files
  → character_sheet_images (appended, deduped), use_audio bundles → voice.audio_reference_file, and
  appearance_override → appearance.summary.

- Add apply_cast_to_subjects() to utils/prompt_compositions.py (pure, no ComfyUI deps); deep-copies
  subjects so registries are never mutated - Import and call from PromptCompositionLoader.execute()
  between cast resolution and _assemble_composition() - 14 new tests in
  tests/test_cast_enrichment.py

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Add edit/delete for existing backgrounds in Composition Editor
  ([`7dbb3a0`](https://github.com/frost-byte/fbTools/commit/7dbb3a0af59fb8fa9db7c64599e9424dd06cd497))

Backgrounds in the sidebar now show a pencil (✎) button on hover that opens an inline edit form
  pre-filled with the background's current name, description, lighting, and soundscape fields.

- Refactored _showNewBgForm into _showBgForm(existing) covering both create and edit;
  _showBgForm(null) is the new-background path - Edit form adds a Delete button (danger style,
  confirm dialog) that removes the background and clears the composition's background field if it
  was pointing to the deleted entry - Extracted _refreshBgDropdown() helper that syncs
  _S.backgrounds, rebuilds the sidebar list, and refreshes the editor dropdown in one call - Each
  background row is now wrapped in fbt-ce-sb-item-row flex container; edit button fades in on row
  hover

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.13.0 (2026-08-12)

### Features

- **libber**: Add random wildcard notation for libber key selection
  ([`13139aa`](https://github.com/frost-byte/fbTools/commit/13139aaaf74f0e6a9445432574b88f06bbdefd3c))

Add %*:N% (random from libber N) and %*% (random from combined pool) notation to the composition
  libber substitution system.

Each occurrence draws from a per-libber shuffled deque (sampling without replacement), so no key
  repeats until every key in that libber has been used at least once. When exhausted the queue
  refills with a new shuffle. The combined %*% pool interleaves all attached libbers before
  shuffling.

Pass order: %*% (combined) → %key:N% / %*:N% (indexed) → %key% (chained).

Frontend: completion popup shows random entries in amber italic for each attached libber (and a
  combined "any" entry when multiple are attached). Typing `*` after the delimiter filters to show
  only random entries.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.12.0 (2026-08-12)

### Features

- **scene**: Add Outfit Registry system
  ([`0323e6d`](https://github.com/frost-byte/fbTools/commit/0323e6defe1fa1f4e9513362ecded1112228fb76))

Add utils/outfit_registry.py with OutfitRegistry class (load/save/define/ remove/list), three new
  nodes (OutfitRegistryLoad, OutfitDefine, OutfitList), and OUTFIT_REGISTRY custom type wired into
  SceneCompose.

SceneCompose gains optional outfit_registry + outfit_A_id–outfit_D_id inputs: explicit text
  overrides still win; registry descriptions fill in when no text override is provided.

REST API: GET/POST /fbtools/outfits/registry|save|reload, DELETE /outfits/delete.

Frontend: Outfits sidebar section in the Composition Editor with list/edit/ delete, modal editor
  (id, name, description, tags), and LLM-assisted image analysis button (visible only when a vision
  model is loaded).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.11.0 (2026-08-12)

### Features

- **ui**: Add LoRA association and concept_id to Prompt Compositions
  ([`a339c95`](https://github.com/frost-byte/fbTools/commit/a339c956104718af02ac40fda33e41335be1e0a7))

- Composition schema gets `loras: [{name, weight, target}]` and `concept_id` fields - New LoRAs
  section in editor: Add LoRA button creates rows with name dropdown, weight input, and model_target
  selector; outputs as LORA_STACK_DATA pin on PromptCompositionLoader → wire into LoraStackApply -
  Composition-level concept_id in Info section; merged with per-subject concept IDs on the
  concept_ids output of PromptCompositionLoader - Concept ID now editable on each assigned subject
  slot row (saves to subject_profiles.json via subjects/save merge, no full reload needed) - New GET
  /fbtools/loras/list endpoint returns sorted LoRA filename list

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.10.0 (2026-08-12)

### Features

- **ui**: Add Libber integration to Prompt Composition editor
  ([`5420946`](https://github.com/frost-byte/fbTools/commit/5420946b0f13ba1b9656fcac7c1e1dd0294e1a7f))

- Composition schema gets a `libbers: []` field (attached libber files) - New Libbers section in
  editor form: check/uncheck to attach libbers, attached libbers show their keys as amber monospace
  chips - `%key%` completion in ALL text fields (style, camera, action, dialogue, soundscape,
  music): triggers on delimiter char, shows key + libber name, auto-inserts closing delimiter;
  %key:N% notation for disambiguation when the same key exists in multiple attached libbers (1-based
  index) - Global Settings section at bottom of sidebar: single-char delimiter input (default %),
  persisted to composition_settings.json via REST - PromptCompositionLoader node applies attached
  libbers to the assembled prompt at execute time, honouring the configured delimiter and :N indexed
  references; fingerprint includes settings file mtime so the node re-executes when settings change

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.9.1 (2026-08-11)

### Bug Fixes

- **ui**: Move composition Name+Model into collapsible Info section
  ([`557e7d5`](https://github.com/frost-byte/fbTools/commit/557e7d5a76143742e002e0e52adcb7310a9b7b33))

Replace the standalone top bar with an Info section at the top of the scrollable form, matching the
  Style/Subjects/Shots collapsible pattern. Name and Model each get their own labeled row
  (fbt-ce-info-row + fbt-ce-info-label) so neither field is squished when the model dropdown has a
  long selected value.

Also initialize _newComp() with name: "" instead of "New Composition" to prevent silent data
  corruption when the name field is visually small and a user types into it unaware that a default
  value is already present.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.9.0 (2026-08-11)

### Bug Fixes

- **extension**: Use relative imports for late utils imports
  ([`f0e3297`](https://github.com/frost-byte/fbTools/commit/f0e32972f7a225e6529dae33a033cf0b211180b5))

All utils imports in the Prompt Composition and LLM route blocks were using bare absolute form (from
  utils.x import) which fails when the package is loaded by ComfyUI as a relative package. Changed
  to the same dot-prefix relative form used everywhere else in extension.py.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **llm**: Replace cross-module get_logger with stdlib logging in llm_client
  ([`7bc4afd`](https://github.com/frost-byte/fbTools/commit/7bc4afd4b488d65375b5347540fb63408f7cc00c))

Pure utils modules have no cross-module deps. Using get_logger from logging_utils caused a
  ModuleNotFoundError at ComfyUI load time because utils/ has no __init__.py and the import path was
  absolute. Replace with logging.getLogger(__name__) consistent with other standalone utils modules.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **llm**: Scanner skips root dir to avoid misidentifying stray GGUF files
  ([`4c67a7d`](https://github.com/frost-byte/fbTools/commit/4c67a7d5ff3e8a3626696c4c05ecac9da6e0aa7a))

_scan_directory now iterates root's children rather than treating root itself as a candidate model
  dir. Fixes the case where a loose text-encoder .gguf (e.g. umt5-xxl-encoder-Q8_0.gguf) in the LLM
  root causes the entire directory to be returned as a single model and recursion to stop, hiding
  all nested model subdirectories.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **llm**: Use "LLM" (singular) as the canonical folder_paths key
  ([`e797ade`](https://github.com/frost-byte/fbTools/commit/e797ade17e70924a2d6eaeb73ce158ac02bf3f40))

The ComfyUI convention, established by ComfyUI-MiniMaxH3-Prompt-Writer and comfyui_llm_party, is
  "LLM" not "LLMs". Scanner now checks "LLM" first with "LLMs" as fallback, and defaults to
  models/LLM/ when neither is registered.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **scene**: Fix two bugs in assemble_composition adapter + add tests
  ([`80fdd9a`](https://github.com/frost-byte/fbTools/commit/80fdd9a0954e5f4725eea52474992caeef97059d))

Dialogue map was keyed by positional counter (shot_1, shot_2) but the template shot lookup uses the
  shot's actual id field — so dialogue in shot N with non-dialogue shots before it was never
  emitted. Fix: key dialogue map by shot["id"] directly.

speaker_slot was absent from the template dialogue dict produced by _composition_shots_to_template,
  so h3_ref2va / h3_fl2va always fell back to "en-us" regardless of the subject's configured
  language. Fix: include speaker_slot (remapped S1→A via slot_map) in the dict.

Adds test_assemble_composition.py (42 tests) covering: - S1/S2 → A/B slot remapping - {S1}/{S2}
  placeholder replacement in action/camera text - Dialogue positional mapping by shot ID - Dialogue
  language tag from speaker's voice.language - Background description, lighting, soundscape
  integration - Composition soundscape overrides background soundscape - Style, music, outfit
  overrides, concept IDs - All 8 model types produce non-empty output - {S} placeholders do not leak
  into any model's output - Edge cases: empty subjects, empty shots, 3-subject mapping

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Initialise composition state before building panel
  ([`e3d00b1`](https://github.com/frost-byte/fbTools/commit/e3d00b1b6c7ed220fe83cb9ba62c98d026d3a93b))

_S.composition was null when _buildPanel called _rebuildShots during first render, causing a
  TypeError on .shots. Moving _newComp() before _buildPanel ensures state is ready before any DOM
  callbacks execute.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

### Chores

- **nodes**: Unregister MultiLoraLoader, SceneWanVideoLoraMultiSave, LoraStackView
  ([`bba5915`](https://github.com/frost-byte/fbTools/commit/bba5915dc7e33b72468668143dcb3f43c4283b94))

Workflow audit (338 workflows scanned): - MultiLoraLoader: present in 1 workflow but fully
  disconnected (no inputs or outputs wired) — confirmed never functional -
  SceneWanVideoLoraMultiSave: zero workflow references - LoraStackView: was already absent from
  get_node_list(); made explicit

Class definitions retained in extension.py for reference. LoraEntryDefine and LoraStackCollect kept
  — still active in 11-13 workflows each.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

### Code Style

- **nodes**: Normalize display names to Title Case with spaces
  ([`330870f`](https://github.com/frost-byte/fbTools/commit/330870f21a4376c0f7f4789dd34862df24a45ea1))

All 29 node display_name values that used verbatim CamelCase class names are updated to Title Case
  with spaces. FBTextEncodeQwenImageEditPlus is shortened to "FB Qwen Image Edit Plus" to avoid
  collision with similarly named nodes from other packages. Node IDs are unchanged so existing
  workflows are unaffected.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Unify LoraStackBuilder info icon with ConceptDefine style
  ([`d4488f9`](https://github.com/frost-byte/fbTools/commit/d4488f9de09fb0ed9a81b2e1df8c98f31d0bf7d6))

Remove the explicit circle (arc + stroke) from _lsbDrawIcon and replace with the same approach as
  _cdDrawIcon: bold "i" centered directly in the rounded rect, font size proportional to icon size
  (sz * 0.55). Both icons are now 18px rounded rects with the same visual weight.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

### Documentation

- Add user-facing docs for Scene Composition Engine nodes
  ([`6a6a330`](https://github.com/frost-byte/fbTools/commit/6a6a3304468fa7562cd64cc744ae45f8c43b82aa))

Four new end-user reference docs covering all Phase 1–4 nodes: concept_registry.md,
  subject_profiles.md, scene_composition.md, prompt_assembly.md. Each covers inputs/outputs, typical
  workflow diagrams, and storage locations.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **nodes**: Add missing tooltip strings to SubjectProfileDefine, ConceptDefine, DatasetCaptioner,
  TailEnhancePro
  ([`321f30d`](https://github.com/frost-byte/fbTools/commit/321f30dfafee3262684ac84b0bea8adf6191287b))

SubjectProfileDefine: name, face, hair, body, default_outfit

ConceptDefine: description

DatasetCaptioner: device

TailEnhancePro: all 12 processing parameter inputs (tail_count, ref_window, deflicker, color match,
  unsharp, bilateral filter)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

### Features

- **cast**: Add Reference Bundle and Scene Cast data layer
  ([`7f0e422`](https://github.com/frost-byte/fbTools/commit/7f0e4228f33d3f2cb609ecb37eff1e2e8673db1a))

Pure-utils modules (no ComfyUI deps) for the Reference Bundle & Scene Cast system (spec §1 data
  layer):

- utils/reference_bundles.py — BundleRegistry with upsert/delete/filter-by-subject, validation
  (visual/audio source constraints), JSON persistence with .bak backup - utils/scene_casts.py —
  CastRegistry with upsert/delete, per-entry update (bundle, visual_mode, use_audio), remove_entry,
  resolve_cast_for_subject, validation - tests/test_reference_bundles.py — 29 tests covering CRUD,
  immutability, filtering, serialisation roundtrip, persistence, and all validation rules -
  tests/test_scene_casts.py — 40 tests covering all of the above plus update_entry partial-update
  semantics and append-on-new-subject behaviour

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **cast**: Add Reference Bundle and Scene Cast REST endpoints
  ([`5f30c5c`](https://github.com/frost-byte/fbTools/commit/5f30c5c4f1ab2bfb47e48518a9695b0304145f0c))

Wires the step-1 utils into extension.py via 9 new aiohttp routes:

Reference Bundles (4 routes): GET /fbtools/bundles/list — all bundles, optional ?subject_id= filter
  GET /fbtools/bundles/get — single bundle by ?id= POST /fbtools/bundles/save — create / update
  (upsert) DEL /fbtools/bundles/delete — remove by ?id=

Scene Casts (4 routes): GET /fbtools/casts/list — all casts GET /fbtools/casts/get — single cast by
  ?id= POST /fbtools/casts/save — create / update (upsert) DEL /fbtools/casts/delete — remove by
  ?id=

Media listing (1 route): GET /fbtools/media/list — files from input/ dir filtered by
  ?type=image|video|audio|all

Also adds _IMAGE_EXTENSIONS and _VIDEO_EXTENSIONS constants alongside the existing
  _AUDIO_EXTENSIONS, and path helpers default_bundle_registry_path() and
  default_cast_registry_path() following the same pattern as other registries.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **cast**: Add Reference Bundle Editor sidebar panel
  ([`48fd32c`](https://github.com/frost-byte/fbTools/commit/48fd32c4d16188ee8dccf286c4757cb0cf93c657))

New sidebar tab "Reference Bundles" (pi pi-images icon) for creating and managing reference media
  bundles tied to subject profiles:

js/api/bundles.js: BundlesAPI client covering bundles (list/get/save/delete), casts
  (list/get/save/delete), subjects/list, and media/list — shared by both the Bundle Editor (step 3)
  and the upcoming Cast Editor (step 4)

js/ui/bundle_editor.js: Full panel implementation: - Top bar: subject-filter dropdown, free-text
  search, + New button, ↺ refresh - List view: bundles grouped by subject, each card shows name,
  VIDEO/IMAGES badge, audio indicator (🎙), tag chips, edit + delete actions - Editor form: name,
  auto-generated ID (editable), subject dropdown, appearance override, visual toggle (Images/Video)
  with file pickers, audio 3-way toggle (None/Extract from video/Separate file) with picker, tags,
  save/cancel - Image list: ordered with ↑↓ reorder and × remove; add-image dropdown shows only
  files not yet selected - Extract-from-visual warning when visual mode is Images

js/styles/style.css: All fbt-be-* styles for panel, top bar, card list, group headers, badges, tags,
  toggle buttons, image list rows, and form sections

js/fb_tools.js + js/index.js: Register the new sidebar tab and export renderBundleEditor

js-tests/bundles_api.test.js: 19 tests covering all BundlesAPI methods including URL encoding, query
  param passing, body serialisation, and DELETE error handling

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **cast**: Add Scene Cast system and video/audio reference params
  ([`ee99c55`](https://github.com/frost-byte/fbTools/commit/ee99c5539d1a9dce2698b88d95e5741fa8b0840e))

Reference Bundle & Scene Cast system: - Scene Cast Editor sidebar panel (js/ui/cast_editor.js) with
  two-line entry rows, bundle dropdown filtered by subject, visual mode toggle with amber 'differs'
  highlight, and fire-and-forget cast reload after save/delete - SceneCastLoad node + SCENE_CAST
  custom type; reload counter wired to POST /fbtools/casts/reload - PromptCompositionLoader:
  optional SCENE_CAST input; resolves reference media (video path + image batch) and builds
  video_entries for assembler - BundlesAPI.reloadCasts() client method

Prompt assembler extensions: - Character sheets cited inline inside <Subject N> block ("Character
  sheets: <Picture N> (primary identity)") per official H3 guide; removed standalone picture entries
  from subject_definitions and retention_analysis - <Video N> reference labels in
  subject_definitions (after subject blocks), retention_analysis, and summary - assemble_prompt() /
  assemble_composition() accept video_entries list

Video/audio reference frame-sampling params: - visual block gains force_rate, frame_load_cap,
  skip_first_frames, select_every_nth for the Load Video node - audio extract_from_visual gains its
  own independent set of four frame params (separate Load Video node instance, different segment
  from visual) - audio file source gains start_time and duration (seconds) for Load Audio -
  _resolve_cast_media() returns a 14-key dict covering all params - PromptCompositionLoader grows 12
  new output pins: video frame params, audio_source, audio_file, audio frame params,
  audio_start_time, audio_duration - Bundle editor UI shows frame-sampling grids for video visual
  and extract_from_visual audio, and a Timing grid for file audio

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **editor**: Phase 7 — LLM assistant for Composition Editor
  ([`be1f54c`](https://github.com/frost-byte/fbTools/commit/be1f54c54f7ae92352976b0263a33fb938265bb6))

Add a local-LLM assistant panel to the Prompt Composition Editor sidebar.

Scanner (utils/llm_scanner.py): - Scans ComfyUI/models/LLMs/ plus any paths registered in
  extra_model_paths.yaml - Detects GGUF format (mmproj-*.gguf alongside main = vision capable) -
  Detects HuggingFace format via config.json architectures / model_type / vision_config /
  preprocessor - Returns capability tags (📷 Vision, 🎬 Video (native/frames), 🔤 Text only) -
  Recommends Qwen2.5-VL 3B Instruct (GGUF) as default download

Client (utils/llm_client.py): - GGUF inference via llama-cpp-python (optional, graceful absent) - HF
  transformers path as secondary (optional) - load_model / unload_model with torch.cuda.empty_cache
  on unload - Task-specific prompt builders: shot action, dialogue, camera, polish

REST endpoints in extension.py: - GET /fbtools/llm/models — scan and return model list + default -
  GET /fbtools/llm/status — current loaded model + backend flags - POST /fbtools/llm/load — load
  model by descriptor - POST /fbtools/llm/unload — free VRAM - POST /fbtools/llm/generate — generic
  text/image generation - POST /fbtools/llm/generate/shot_action, /dialogue, /polish - POST
  /fbtools/llm/download/default — download starter GGUF via huggingface_hub

Editor UI (composition_editor.js): - 🤖 LLM Assistant sidebar section with model picker + capability
  tags - Load / Unload buttons; status line; generate buttons per field - Action / Dialogue / Polish
  buttons target the focused shot card - Download prompt when no models found; mentions
  extra_model_paths.yaml

API client (js/api/llm.js): REST client for all LLM endpoints. Tests (tests/test_llm_scanner.py): 30
  tests, all passing.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **nodes**: Replace audio_reference_file text input with file picker combo
  ([`66b6215`](https://github.com/frost-byte/fbTools/commit/66b62154dd5821dfbf69c934dd26a69dc3632aa9))

SubjectProfileDefine now shows a combo of audio files (.wav, .mp3, .flac, .ogg, .aac, .m4a, .opus)
  from the ComfyUI input directory instead of a free-text field. Press R to refresh the list after
  adding new files.

Also corrects all "Refresh the page" tooltip/doc copy to "Press R" across SubjectProfileLoad,
  SceneTemplateLoad, PromptCompositionLoader, and the four user-facing docs — R triggers
  /object_info which re-runs define_schema and refreshes all combo options without a full page
  reload.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **nodes**: Replace concept_id text input with combo in SubjectProfileDefine
  ([`2502138`](https://github.com/frost-byte/fbTools/commit/2502138a081a2142238ebd76c13f8cb8d0214a48))

Adds _concept_get_ids() helper that reads concept_registry.json at schema load time.
  SubjectProfileDefine.concept_id is now a combo picker instead of a free-text field; "None" is
  normalised to "" in execute(). Define nodes (ConceptDefine, SubjectProfileDefine) keep free-text
  subject_id / concept_id inputs since those are used to create new entries.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **scene**: Add PromptCompositionLoader node with reload counter
  ([`3da0d06`](https://github.com/frost-byte/fbTools/commit/3da0d0645672db623088bbb7a8d2c20193a49df2))

- PromptCompositionLoader: selects a saved composition by name from a combo dropdown, assembles it
  with the chosen model type, and outputs prompt + concept_ids (for ConceptResolve) +
  model_type_used + name - model_type combo includes "composition default" as the first option so
  the stored model type is used without requiring a second setting - fingerprint_inputs includes
  compositions dir mtime + _composition_reload_counter so any PromptCompositionLoader node
  re-executes when content changes - POST /fbtools/compositions/reload increments the counter -
  Editor _onSave fires the reload endpoint (fire-and-forget) so canvas nodes pick up the latest
  content immediately after saving

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Phase 4 shot management — reorder, duplicate, preset targeting, shortcuts
  ([`02779a0`](https://github.com/frost-byte/fbTools/commit/02779a0ac7fef2ec51ff81552be19bc59a05b98e))

- Add ↑/↓ reorder buttons and ⧉ duplicate to each shot card header - Track focused shot (focusin
  delegation) so camera/sound presets insert into the correct shot's field rather than copying to
  clipboard - _moveShot / _duplicateShot / _addNewShot helpers keep focus index in sync and scroll
  the target card into view after rebuild - Ctrl+Shift+N: add shot, Ctrl+Shift+P: preview,
  Ctrl+Shift+C: copy - Update sidebar section titles to "click to apply to shot" -
  .fbt-ce-shot-active highlight (blue border) on the focused shot card

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Prompt Composition Editor — Phase 1 + Phase 2
  ([`7a2bb7d`](https://github.com/frost-byte/fbTools/commit/7a2bb7db7f2e0b90dadf6c530c5408fbc599b1bd))

Phase 1 — Backend data layer: - utils/prompt_compositions.py: composition CRUD,
  resolve_subjects/background, validate - utils/composition_resources.py: backgrounds, camera
  presets, sound presets CRUD - utils/prompt_assembler.py: add assemble_composition() and
  _composition_shots_to_template() - extension.py: subject CRUD routes, composition CRUD + assemble
  route, background CRUD routes, camera + sound preset routes (~305 lines)

Phase 2 — Basic editor panel: - js/api/compositions.js: REST client for compositions, subjects,
  backgrounds, presets - js/ui/composition_editor.js: full sidebar panel — resource sidebar,
  structured form editor (subject slots, shot cards, dialogue), Preview Raw modal, Copy, Save/Load,
  keyboard shortcut (Ctrl+S) - js/styles/style.css: composition editor styles (~480 lines) -
  js/fb_tools.js: register sidebar tab via app.extensionManager.registerSidebarTab - js/index.js:
  re-export CompositionsAPI and renderCompositionEditor

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **ui**: Prompt Composition Editor — Phase 3 smart elements
  ([`7d24ccb`](https://github.com/frost-byte/fbTools/commit/7d24ccbdefd1fe2bceb0c4f8476a40e3b983a7cb))

- {S} slot-reference completion popup in action/camera text fields: type { to trigger, arrow keys to
  navigate, Enter/Tab to insert, Esc to dismiss - Subject slot cards: appearance summary shown below
  each slot dropdown - Background section: auto-fills soundscape when empty, or offers a replace
  button when the soundscape field already has content - Sidebar "New Subject" inline form: name,
  appearance summary, concept ID; saves via POST /fbtools/subjects/save, refreshes dropdowns -
  Sidebar "New Background" inline form: name, description, lighting, soundscape; saves via POST
  /fbtools/backgrounds/save, refreshes editor background dropdown - compositions.js: add
  saveSubject(), deleteSubject(), saveBackground(), deleteBackground() to CompositionsAPI

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.8.0 (2026-08-07)

### Features

- **ui**: Compact canvas rows for LoraStackBuilder and ConceptDefine
  ([`2b787a8`](https://github.com/frost-byte/fbTools/commit/2b787a8b4fd51f746694967878764a0850f87602))

LoraStackBuilder: - Each slot now fits on a single canvas row: toggle, LoRA name, strength spinners
  (Model+CLIP, or Model+Vid+Aud for LTX2.3), ⓘ icon - Row count is dynamic — starts at 1 (or last
  filled slot) and grows via an "+ Add LoRA" button; count persists in node.properties - Backend
  slot widgets hidden with type="converted-widget" so V3 rendering pipeline skips them - ⓘ opens
  Civitai modal (image gallery with hover-prompt overlay, up to 6 example images) - showCivitaiModal
  exported so ConceptDefine can share it

ConceptDefine: - New compact _CdLoraRow canvas widget: LoRA name + weight spinner on one line,
  optional H/L badge for split models - Split models (wan22, bernini): H row + L row; non-split:
  single row - Switching model_type live rebuilds rows immediately - Widget hiding uses
  type="converted-widget" (V3 requirement; "hidden" is ignored by the V3 onDrawForeground pipeline)
  - Both onNodeCreated and onConfigure rebuild via queueMicrotask so onConfigure.apply can assign
  saved widget values before native widgets are converted, preventing misalignment - Weight
  sanitize: coerces false/non-numeric values to 1.0 on load

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.7.0 (2026-08-06)

### Features

- **scene**: Add PromptAssemble node — Phase 4 of Scene Composition Engine
  ([`6187b54`](https://github.com/frost-byte/fbTools/commit/6187b54053d7fed94d200e15caf1ab12532e1bf8))

Implements model-specific prompt generation from a SCENE_INSTANCE: - utils/prompt_assembler.py: pure
  assembly logic for 8 model types - h3_ref2va: full 6-section H3 brief with Subject/Picture/Audio
  reference labels, first-appearance tracking, <d>[lang] text</d> dialogue tags - h3_fl2va:
  shot-structured format with dialogue, no reference labels - wan22/bernini: production-direction
  block with task classification - ltx23/flux2/krea2/qwen: simple descriptive format -
  PromptAssemble node: takes SCENE_INSTANCE + model_type, outputs prompt (STRING), reference_images
  (IMAGE batch), reference_audio (AUDIO), additional_audio (AUDIO), concept_ids (STRING),
  assembly_report (STRING) - 63 new tests covering all model types, reference numbering, placeholder
  replacement, dialogue tags, outfit overrides, image/audio ordering, concept ID extraction, and
  edge cases

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.6.0 (2026-08-06)

### Features

- **scene**: Add SceneCompose node — Phase 3 of Scene Composition Engine
  ([`8ca5c34`](https://github.com/frost-byte/fbTools/commit/8ca5c34fa2f8e1921c2021aaf9ef3167796a51da))

Adds the scene composition layer: assigns subjects to template slots, maps positional dialogue to
  placeholder shots, applies outfit overrides, and validates slot requirements.

New files: - utils/scene_compose.py — pure composition logic, no ComfyUI deps -
  tests/test_scene_compose.py — 25 tests covering compose, validate, summary

New node (🧊 frost-byte/Scene): - SceneCompose — takes SCENE_TEMPLATE + up to 4 SUBJECT_PROFILEs, up
  to 4 dialogue strings, and per-slot outfit overrides; outputs SCENE_INSTANCE + human-readable
  scene_summary with validation warnings

New custom type: SCENE_INSTANCE (dict with template, slot_assignments, dialogue map,
  outfit_overrides)

Also: SubjectProfileLoad and SubjectProfileDefine now inject subject_id into the SUBJECT_PROFILE
  dict they output, so downstream nodes can reference the profile key without a separate STRING
  output.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.5.0 (2026-08-06)

### Features

- **scene**: Add SceneTemplate nodes — Phase 2 of Scene Composition Engine
  ([`804bc45`](https://github.com/frost-byte/fbTools/commit/804bc4546306eeeb37ff129852b2c09f67b8a853))

Adds the scene template layer: JSON blueprints for shot structure, environment, camera, and slot
  placeholders, independent of model format and subject assignment.

New files: - utils/scene_templates.py — pure SceneTemplate logic, no ComfyUI deps -
  tests/test_scene_templates.py — 40 tests covering load, scan, format, fingerprint -
  scene_templates/monologue_indoor.json — 1-slot bundled example -
  scene_templates/cafe_conversation_2p.json — 2-slot bundled example -
  scene_templates/meeting_room_3p.json — 3-slot bundled example

New nodes (🧊 frost-byte/Scene): - SceneTemplateLoad — loads template from scene_templates/ dir;
  outputs SCENE_TEMPLATE + slot_info - SceneTemplateList — scans directory and returns formatted
  template listing

New REST endpoints: - POST /fbtools/scene_templates/reload — force re-execute fingerprint-cached
  nodes - GET /fbtools/scene_templates/list — return template metadata list as JSON

Bundled examples are seeded into user_data_dir/scene_templates/ on first use when the directory is
  empty; user templates live there permanently.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.4.0 (2026-08-06)

### Bug Fixes

- **lora**: Pass lora_metadata and add lora_convert to apply paths
  ([`84f3dbc`](https://github.com/frost-byte/fbTools/commit/84f3dbc327e32f88215a5c95cf585c9387f1e18e))

- _lora_load_weights now loads with return_metadata=True and caches (mtime, weights, metadata) —
  returns (weights, metadata) tuple - _lora_apply_standard: passes safetensors metadata to
  load_lora_for_models(lora_metadata=...) so downstream nodes can inspect which LoRAs are applied to
  a model patcher - _lora_apply_ltx23: adds missing comfy.lora_convert.convert_lora() call before
  load_lora() to handle BFL/Wan-Fun format variants that the standard path converts automatically
  via load_lora_for_models

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

### Features

- **lora**: Add LoraStackBuilder node and refine LTX2.3 params
  ([`94ebcad`](https://github.com/frost-byte/fbTools/commit/94ebcad22c0af3b847c74c1ad53b88efeea036ea))

- New LoraStackBuilder node: 8 inline LoRA rows (combo + sliders) with model_target selector; JS
  hides video/audio strength widgets for non-LTX2.3 targets; optional autogrow LORA_ENTRY input and
  prev_stack merge; outputs LORA_STACK_DATA without requiring LoraEntryDefine/Collect -
  LoraEntryDefine: replace 5 LTX2.3 per-layer params (video, video_to_audio, audio, audio_to_video,
  other) with 2 (video_strength, audio_strength); backward compat preserved in _lora_apply_ltx23 for
  old entries - _lora_apply_ltx23: handle new 2-param format; video_strength scales all
  video/video-side-cross-attn keys, audio_strength scales all audio keys - _lora_load_weights: add
  mtime-keyed in-memory cache; _lora_apply_standard now uses cached loader instead of direct
  load_torch_file calls - get_node_list: LoraStackBuilder listed first; LoraEntryDefine/Collect
  remain registered for backward compatibility

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **scene**: Add SubjectProfile nodes — Phase 1 of Scene Composition Engine
  ([`4a2daaa`](https://github.com/frost-byte/fbTools/commit/4a2daaa40523c5780c3bff9ae7903a452ba19641))

Introduces the subject profile layer: persistent JSON storage for character appearance, voice, and
  character sheet references, linked to the concept registry via concept_id for LoRA resolution.

New files: - utils/subject_profiles.py — pure SubjectRegistry logic, no ComfyUI deps -
  tests/test_subject_profiles.py — 24 tests covering define, persist, list -
  docs/scene_composition_action_plan.md — full 4-phase system spec

New nodes (🧊 frost-byte/Scene): - SubjectProfileLoad — loads subject dict, IMAGE batch, AUDIO from
  disk - SubjectProfileDefine — creates/updates subjects with auto_save - SubjectProfileList — lists
  all defined subjects

New REST endpoints: - POST /fbtools/subjects/reload — force re-execute fingerprint-cached nodes -
  GET /fbtools/subjects/profiles — return subject_profiles.json as JSON

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.3.0 (2026-08-06)

### Features

- **audio**: Add AudioFixShape node to restore batch dimension on audio waveforms
  ([`1e1083d`](https://github.com/frost-byte/fbTools/commit/1e1083d188c83b12f674faa50661df8e075bdee4))

Handles 1-D (samples,) and 2-D (channels, samples) tensors by unsqueezing to the expected (batch,
  channels, samples) layout. Placed under the new 🧊 frost-byte/Audio category.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.2.0 (2026-08-05)

### Features

- **lora**: Add Concept Registry system with ConceptRegistryLoad, ConceptDefine, ConceptResolve,
  ConceptList nodes
  ([`cc05197`](https://github.com/frost-byte/fbTools/commit/cc05197c2d6c748d3f947bb967980c219ab123b0))

- Add utils/concept_registry.py: pure-logic module (no ComfyUI deps) with ConceptRegistry class,
  MODEL_PROFILES for 6 model types (wan22/bernini split, ltx23/flux2/krea2/qwen single), load/save
  with .bak backup, resolve_concepts, assemble_prompt, build_model_entry helpers - Add 4 ComfyUI
  nodes: ConceptRegistryLoad (fingerprint-based reload via REST), ConceptDefine (chainable,
  accumulate-not-overwrite for different model_types, auto_save option), ConceptResolve (applies
  LoRAs via comfy.sd, assembles prompt with trigger words), ConceptList (filter by model type) - Add
  CONCEPT_REGISTRY custom wire type - Add REST endpoints: POST /fbtools/concepts/reload (reload
  counter), GET /fbtools/concepts/registry - Add user_data_dir() + _user_subdir() helpers; update
  default_scenes_dir() and default_libber_dir() to prefer ComfyUI/user/default/comfyui-fbTools/ with
  graceful fallback to legacy output/ directories - Extract setWidgetVisible to js/utils/widgets.js
  (shared); update lora.js to import it; add js/api/concepts.js, js/nodes/concepts.js
  (lora_low/weight_low hidden for single-model types; Reload Registry button on ConceptRegistryLoad)
  - Add 32 tests in tests/test_concept_registry.py; all 330 Python + 83 JS tests pass

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **lora**: Add minimax_h3 to ConceptRegistry MODEL_PROFILES
  ([`fb23e5a`](https://github.com/frost-byte/fbTools/commit/fb23e5ab21ed5f66a3ba36646fa001d5bf163077))

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **lora**: Add native LORA_STACK support to LoraPresetDefine/Select and add MiniMaxH3 target
  ([`b9468d8`](https://github.com/frost-byte/fbTools/commit/b9468d8f2efa894768e9637f80a64510976ef862))

LoraPresetDefine now accepts both LORA_STACK_DATA (from LoraStackCollect's Stack Data output) and a
  native LORA_STACK (easy-use tuple format) as optional inputs, so any LoRA source in the ecosystem
  can be stored in a preset. When LORA_STACK_DATA is provided, the native representation is
  auto-generated so both output types are always populated.

LoraPresetSelect gains a new "LoRA Stack (Native)" output (io.Custom LORA_STACK) appended after the
  existing outputs, preserving backward compatibility for already-wired workflows. The
  LORA_STACK_DATA output is unchanged.

Also adds MiniMaxH3 to LORA_MODEL_TARGETS for use in LoraStackApply (standard
  strength_model/strength_clip path); weight variants can be added later once the LoRA structure is
  known.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **lora**: Add scene/pose image support to preset nodes
  ([`0103882`](https://github.com/frost-byte/fbTools/commit/01038824c1d8f4ce5fda90a3f4ae57517588dc91))

LoraPresetDefine and WanPresetDefine each gain an optional Scene combo (populated at runtime via
  /fbtools/scene/list) and a Pose Image Type combo. The selected scene and pose type are stored in
  the preset dict.

LoraPresetSelect and WanPresetSelect each gain base_image and pose_image outputs. When the active
  preset has a linked scene, those images are loaded from the scene directory and shown as a node
  preview on execution. Placeholder 64x64 images are returned when no scene is set.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.1.0 (2026-07-27)

### Features

- **lora**: Accordion-hide LTX2.3 layer weights in LoraEntryDefine
  ([`435053e`](https://github.com/frost-byte/fbTools/commit/435053e81fb653edf7237203f3c80a1cd38f1ae9))

When model_target is not LTX2.3, the video/audio/cross-attention strength inputs and toggle button
  are hidden entirely. When LTX2.3 is selected, a ▶/▼ caret button between Enabled and the Civitai
  button controls visibility. Accordion defaults to collapsed; onConfigure re-applies visibility
  when a saved graph is loaded.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw

- **lora**: Add dynamic combo and preview to WanPresetSelect
  ([`2dd5e5b`](https://github.com/frost-byte/fbTools/commit/2dd5e5bd20f5b668f926242a5f477a07d6fdce69))

- Replace index INT input with a COMBO widget (selected_preset) that starts with ["none"] and is
  populated with preset names after each execution - Add validate_inputs to accept any string value,
  bypassing static combo option validation so user-selected names are not rejected by the server -
  Add is_output_node=True for standalone preview execution - Execute sends preset names to frontend
  via ui={preset_names:[...]}; JS onExecuted updates widget.options.values and preserves current
  selection - Switch preset lookup from index-based to name-based with first-entry fallback

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

- **lora**: Add LoraPresetDefine and LoraPresetSelect nodes
  ([`be84768`](https://github.com/frost-byte/fbTools/commit/be847688d1922a17a0741870950f3714e8111baf))

Single-stack preset nodes for models without a dual-sampler stage (e.g. Flux2/Klein, Qwen). Uses a
  separate LORA_PRESET_LIST custom type to prevent cross-wiring with Wan preset chains.
  LoraPresetSelect uses the same dynamic combo + validate_inputs pattern as WanPresetSelect.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01PEBgH9wV9PW2ifTFryJsGw


## v1.0.0 (2026-07-24)

### Bug Fixes

- Add control flags to StorySceneBatch descriptor and fix StoryEdit scene data persistence
  ([`4045691`](https://github.com/frost-byte/fbTools/commit/4045691ea7a538253c16cbf4adf328feb53c49c3))

- Add use_depth, use_mask, use_pose, use_canny flags to batch descriptor - Add logging to
  StorySceneBatch for scene configuration debugging - Add logging to StoryScenePick for pose_type
  resolution tracking - Fix StoryEdit duplicate renderTable call that was overwriting normalized
  scene data - Add fingerprint_inputs to StorySceneBatch for proper cache invalidation

This fixes issues where: 1. Control flags weren't being passed from story config to batch processing
  2. StoryEdit UI changes (pose_type, depth_type) weren't persisting to story.json 3. Cache wasn't
  invalidating when story.json was modified externally

- Complete mask system migration from mask_type to mask_name
  ([`2da731d`](https://github.com/frost-byte/fbTools/commit/2da731d1cf5002293d68cfd2f041d8c99df1714f))

BREAKING CHANGES: - Story persistence now uses mask_name instead of mask_type - Backward
  compatibility maintained for loading old stories

Backend Changes: - story_models.py: Updated save_story() to persist mask_name field (line 142) -
  extension.py: Fixed StorySceneBatch to use mask_name in scene descriptors (lines 5135, 5162) -
  extension.py: Added backward-compatible fallback to mask_type for legacy data - extension.py: All
  mask loading/preview functions now use mask_name consistently

Frontend Changes: - js/nodes/story.js: StoryEdit mask column now uses dropdown instead of text input
  - js/nodes/story.js: Dropdowns populated from scene's available_masks array - js/nodes/story.js:
  Updated prompt_key rendering for conditional dropdown/textarea - js/nodes/story.js: Extended
  populateVideoPromptControls to handle both image and video prompts - js/nodes/story.js: Added
  event listeners for mask-name-select and prompt-key-select

API Changes: - /fbtools/story/load: Returns available_masks array per scene for dropdown population
  - /fbtools/story/save: Accepts mask_name with fallback to mask_type

Migration: - Old stories with mask_type are automatically migrated to mask_name on load -
  SceneInStory.__init__ converts mask_type to mask_name during initialization - All file I/O now
  uses v2 format with mask_name as primary field

Testing: - Verified backward compatibility with v1 story.json files - Verified mask dropdown
  population from masks.json and legacy PNGs - Verified batch system
  (StorySceneBatch/StoryScenePick) uses correct mask field

Closes: Mask persistence bug, prompt_key dropdown regression, batch system migration gap

- Dynamically fetch available libbers when switching to libber type
  ([`a7a3eb5`](https://github.com/frost-byte/fbTools/commit/a7a3eb5de1b8f3dd2e5d3570c56d1098bcbc9b35))

- Import libberAPI in scene.js - When user selects 'libber' type and current value is 'none': *
  Fetch latest libbers from API endpoint * Repopulate dropdown with current libbers * Auto-select
  first available libber * Fallback to existing list if API fails - Prevents showing 'none' when
  libbers exist - Ensures dropdown always shows current state

No service restart needed - just refresh browser (Ctrl+Shift+R)

- Libber nodes now reload from file to prevent stale cache
  ([`643b2bb`](https://github.com/frost-byte/fbTools/commit/643b2bb478109a911b89b7fb5282ac87267462c3))

LibberManager and LibberApply nodes were using in-memory Libber instances that weren't being updated
  when changes were saved via the REST API/web UI.

Changes: - LibberManager: Now reloads from JSON file if it exists on each execution - LibberApply:
  Also reloads from file before applying substitutions - Ensures nodes always use the latest lib
  values from disk - In-memory cache is effectively refreshed on every node execution

This fixes the issue where updating keys in LibberManager wouldn't reflect in LibberApply results
  until server restart.

- Update canny during SceneUpdate
  ([`d43c1f5`](https://github.com/frost-byte/fbTools/commit/d43c1f5107934ac2ff5e79b97b75875d8f443bae))

- **LibberApply**: Improve table display and resize behavior
  ([`692116c`](https://github.com/frost-byte/fbTools/commit/692116c84d5563e764b5e32bd1afe74a06336488))

- Replaced JSONView formatter with clean HTML table layout - Added two-column table format with 🗝️
  Lib and 🪙 Value headers - Implemented scrollable container with overflow-y and overflow-x - Fixed
  table persistence after node execution by storing and reusing updateDisplay function - Added
  dynamic sizing with proper height calculation based on available node space - Implemented resize
  hooks (onResize) to update container height when node is resized - Added height constraints (min:
  150px, max: 600px) to prevent infinite growth - Fixed bottom edge overlap by adding 15px bottom
  margin - Improved widget height computation to account for previous widgets' space - Added HTML
  escaping for safe display of lib values

The table now properly displays libber key-value pairs, persists after execution, and maintains
  reasonable sizing constraints while allowing user resizing.

- **ScenePromptManager**: Fix scene selection, saving, and libber integration
  ([`431be57`](https://github.com/frost-byte/fbTools/commit/431be57a2fb3398afe1b3511ce7a7bad756d1631))

- Fix scene tracking to read widget values at click time instead of cached values - Scene dropdown
  now correctly reloads prompts when changed - Apply Changes now saves to the correct scene
  directory (was using first available scene) - Fix prompt data structure handling (API returns
  array, code expected object) - Add scene save API endpoint POST /fbtools/scene/save_scene_prompts
  - Fix libber_name preservation and libber dropdown population - API now returns libber_name field
  and available libbers list - Add 100ms delay for initial widget value loading to ensure proper
  initialization - Add extensive debug logging for scene tracking and widget values

Resolves issues where: - Changing scene dropdown didn't update the UI - Apply Changes saved to wrong
  scene directory - Libber selections weren't preserved - Prompt keys showed as array indices
  instead of names

### Chores

- Add Conventional Commits hook, semantic release, and CLAUDE.md
  ([`d73123a`](https://github.com/frost-byte/fbTools/commit/d73123a83c720a3b004a52b52596c49d85109dc7))

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

- Add developer utility scripts
  ([`eef9dda`](https://github.com/frost-byte/fbTools/commit/eef9ddaa0ef2a926aa72235bf93cf152f11080ae))

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

### Documentation

- Add comprehensive test coverage summary
  ([`dce7115`](https://github.com/frost-byte/fbTools/commit/dce7115fafca3843c519f56d6eb3e8e966de7d69))

- 90 tests passing (100% pass rate) - Coverage breakdown by test file and category - Real-world
  workflow validation - Backend testing complete - Ready for UI implementation

- Add NLF pose implementation guide and additional tests
  ([`20dd41a`](https://github.com/frost-byte/fbTools/commit/20dd41aa3b8d803087ed9d0ffe3ae61381a2105d))

Complete documentation and test coverage for NLF pose feature:

- NLF_POSE_IMPLEMENTATION.md: Comprehensive implementation guide with: - Step-by-step checklist -
  Format specifications (DWPose, OpenPose, NLFPRED, POSE_KEYPOINT) - Three workflow examples
  (generation, editing, regeneration) - Model requirements and downloads - Configuration options
  reference

- utils/nlf_pose.py: NLF utilities module (476 lines) - load_nlf_model, predict_nlf_pose,
  render_nlf_pose - Format conversion functions - Supports both relative and absolute imports for
  testing

- tests/test_nlf_integration.py: 13 integration tests - Module import validation -
  SceneCreate/SceneUpdate input verification - SceneInfo data model checks - Pose JSON format
  validation - Workflow structure verification

- js-tests/story_nlf_pose.test.js: 11 frontend tests (all passing) - Pose type dropdown includes
  'nlf' - Backend/frontend consistency - Scene serialization with NLF pose - Backward compatibility

Ready for testing in ComfyUI.

- Add Scene Prompt Management System implementation plan
  ([`f6f8e1c`](https://github.com/frost-byte/fbTools/commit/f6f8e1ce95aafe17f303ff2ffde4e2027d933fbd))

- Clarify two workflow approaches - Complete vs Atomic prompts
  ([`33b4311`](https://github.com/frost-byte/fbTools/commit/33b431174fc1009f1b4844a642761132917ae5f7))

- Added 'Two Workflow Approaches' section explaining both strategies - Approach 1: Complete Prompts
  (traditional, recommended for most users) * Each prompt is self-contained and complete * Libber
  handles dynamic parts within single prompt * Example: wan_high with full prompt text - Approach 2:
  Atomic Composition (advanced) * Break into small reusable pieces * Maximum flexibility for
  mixing/matching - Comparison table showing pros/cons of each - Hybrid approach combining both
  strategies - Real-world usage patterns with examples - Migration strategy from legacy prompts -
  Complete examples for both approaches

- Comprehensive documentation and test coverage update
  ([`4426092`](https://github.com/frost-byte/fbTools/commit/44260929ea41429bc2e3ccc7cba0e6709ba3ce80))

Major documentation improvements:

Root README.md: - Complete feature overview with all node categories - Detailed usage examples for
  Libber, Story, and PromptCollection - Development setup and testing instructions - Project
  structure and architecture explanation - Changelog with recent Libber overhaul details

LIBBER_NODES_README.md (NEW): - Complete Libber system documentation - Interactive table editor
  features and workflow - Click-to-insert functionality guide - REST API endpoint reference - Use
  cases and best practices - Troubleshooting guide - JavaScript integration examples

TEST_RESULTS.md: - Updated with Libber test coverage (30 tests) - Summary of all Python tests (70+
  tests total) - Summary of all JavaScript tests (30+ tests) - Execution instructions for both test
  suites

New Tests: - tests/test_libber.py: 30 comprehensive unit tests * Basic operations (create, add,
  remove, list) * Substitution with recursion and depth limiting * Custom delimiters * File
  operations (save/load) * Edge cases (unicode, large values, special chars) * Integration workflows
  - js-tests/libber_api.test.js: 21+ API client tests * CRUD operations * Error handling *
  Integration workflows

All tests passing: - Python: 30/30 Libber tests ✓ - Python: 32/32 PromptCollection tests ✓ -
  JavaScript: API client tests ready

This commit provides complete documentation for users and developers, with comprehensive test
  coverage ensuring reliability.

- Create comprehensive plan for flexible prompt system and workflow improvements
  ([`23f5048`](https://github.com/frost-byte/fbTools/commit/23f5048def5ed131628c2c525337c18019fe94f1))

Add detailed implementation plan (plan-flexibleMultiPromptSystemLibberBugFix.prompt.md) covering:

- PromptCollection data model with v2 format and non-destructive migration - SceneInfo refactoring
  with backward-compatible @property methods - Scene REST API for lightweight metadata operations -
  Dynamic prompt name discovery and selectors - PromptCollectionEdit node with REST backend - Story
  execution-based output organization for two-stage workflows - StoryExecutionInit for execution
  context management - StoryImageNamer/StoryPathResolver for standardized naming -
  StoryImageCollector/StoryVideoNamer for video generation pipeline - Multiple path format outputs
  (abs/rel, with/without extension) - Libber REST API with LibberStateManager for server-side state
  - LibberEdit UI refactoring to fix synchronization bugs

Plan prioritizes Scene/PromptCollection improvements (Steps 1-5), Story workflow enhancements (Step
  6), then Libber bug fixes (Steps 7-8).

Key features: - Non-destructive migration with v1_backup preservation - Execution-aware directory
  structure for multi-run workflows - Support for image generation → video generation pipeline -
  Flexible path outputs for different SaveImage node conventions - Backward compatibility maintained
  throughout

Refs: LibberEdit add operation bug, Story output organization requirements

### Features

- Add compositions support to PromptCollection
  ([`f9e3493`](https://github.com/frost-byte/fbTools/commit/f9e3493c816b2a8d5e3c827ed405e6d24a781fc9))

Backend changes: - Add compositions field to PromptCollection: {output_name: [prompt_keys]} - Add
  composition CRUD methods: add_composition, remove_composition, list_composition_names - Update
  to_dict/from_dict to serialize/deserialize compositions - Update ScenePromptManager to output
  prompt_dict (composed prompts) - Compose prompts automatically when compositions exist - Include
  compositions_list and prompt_dict in UI data

Data structure: - compositions saved in prompts.json alongside prompts - Backward compatible (empty
  dict if no compositions) - compose_prompts() handles libber substitution

ScenePromptManager outputs: - scene_info (updated with prompts + compositions) - prompt_dict
  (Dict[str, str] - composed outputs) - status

Ready for frontend tab implementation

- Add comprehensive testing for PromptCollection with maintainable architecture
  ([`19b5cdc`](https://github.com/frost-byte/fbTools/commit/19b5cdc4b9f0cdd5f972dd792ff129ba1298a882))

Extract data models to standalone module and implement full test coverage for v1→v2 prompt migration
  system.

Changes: - Create prompt_models.py: Pure data models with no ComfyUI dependencies * PromptMetadata:
  Single prompt with metadata fields * PromptCollection: V2 multi-prompt system with migration
  support

- Refactor extension.py: Import from prompt_models instead of inline definitions * Reduces
  extension.py by ~130 lines * Enables independent testing of data models

- Add comprehensive test suite (tests/test_prompt_collection.py): * 32 tests across 8 test classes *
  V1→V2 migration with v1_backup preservation * CRUD operations (add, remove, get, list) *
  Serialization/deserialization roundtrips * Backward compatibility validation * Edge cases
  (unicode, large values, 1000+ prompts) * File I/O operations * Integration workflows * All tests
  passing in 0.19 seconds

- Update test infrastructure: * conftest.py: Mock setup for ComfyUI dependencies * pytest.ini: Clean
  configuration

- Documentation: * TEST_RESULTS.md: Detailed test coverage report * TESTING_STRATEGY.md:
  Architecture decisions and benefits

Benefits: ✓ Single source of truth - no code duplication ✓ Fast, isolated tests - no complex mocking
  needed ✓ Maintainable - updates reflect everywhere automatically ✓ Validates v1→v2 migration
  preserves original data ✓ Ensures backward compatibility

- Add generic mask system and NLF pose generation
  ([`0e65f19`](https://github.com/frost-byte/fbTools/commit/0e65f1934a701ba8e456f47908e956477f8fba33))

Major Features:

1. Generic Mask System (replaces hardcoded masks) - MaskDefinition dataclass with MaskType enum
  (TRANSPARENT/COLOR) - User-definable masks via masks.json (v1 format) - SceneSelect: Dynamic
  mask_name combo loaded from masks.json - SceneInfo: masks dict + mask_images dict (name-keyed) -
  Migration support for legacy 'girl'/'male'/'combined' masks - Tests: test_mask_integration.py (8
  tests), mask_system.test.js (frontend)

2. NLF Pose Generation - utils/nlf_pose.py: Neural Lifting Framework integration - SceneCreate: 7
  NLF inputs for pose generation - SceneUpdate: 9 NLF inputs for pose editing/regeneration -
  SceneInfo: pose_nlf_image field with load/save - default_pose_options: 'nlf' -> 'pose_nlf_image'
  mapping - Story node: 'nlf' added to pose type dropdown - Tests: test_nlf_integration.py (13
  tests), story_nlf_pose.test.js (11 tests)

3. Documentation Reorganization - Moved 23 docs to docs/ folder - Test docs to docs/testing/
  subfolder - Updated README with logo, dependencies, testing links - New docs: MASK_SYSTEM.md,
  PHASE_4_COMPLETE.md

Changes by file: - extension.py: Mask system classes + NLF pose in SceneCreate/SceneUpdate -
  js/nodes/scene.js: Dynamic mask combo via API - js/nodes/story.js: 'nlf' pose type in dropdown -
  story_models.py: mask_name field in StoryScene - dependency.json: Fixed comfyui_controlnet_aux URL
  - tests/conftest.py: Added torch mocking for NLF tests

All tests passing: 213 Python tests, 11 JavaScript tests

- Add modular frontend architecture with API clients and testing framework
  ([`7695ac3`](https://github.com/frost-byte/fbTools/commit/7695ac36714cc8e0bb6b11e6a53e5b9fee4685e0))

Create comprehensive modular JavaScript architecture for fbTools frontend with testable API clients,
  shared utilities, and full Jest testing setup.

New Structure: - js/api/ API client modules for REST endpoints - js/utils/ Shared utility functions
  - js/tests/ Test framework with utilities - js/index.js Main exports file

API Clients Added: - prompt_collection.js: PromptCollection REST API (fully implemented) *
  createSession, addPrompt, removePrompt, listPromptNames, getCollection - scene.js: Scene metadata
  operations (stub ready for backend) - libber.js: Libber placeholder management (stub ready for
  backend) - story.js: Story-level operations (stub ready for backend)

Utilities Added: - api_base.js: BaseAPI class with error handling and fetch wrapper * POST/GET
  methods with automatic error handling * Toast notification helpers (showSuccess, handleError) *
  APIError class for typed error responses - widgets.js: ComfyUI widget update helpers *
  updateWidgetFromText: Update single widget from API response * updateNodeWidgets: Bulk widget
  updates * scheduleNodeRefresh: Node resize/refresh utility

Testing Framework: - test_utils.js: Testing utilities and mocks * mockFetch: Fetch API mocking for
  isolated tests * createMockFn: ES module-compatible mock functions * createMockApp/createMockNode:
  ComfyUI test fixtures * expectToast helpers: Toast assertion utilities -
  prompt_collection_api.test.js: Example tests (9 tests, all passing) - package.json: Jest
  configuration with ES module support - Fixed jest-environment-jsdom dependency for Jest 29 -
  Custom createMockFn() to replace jest.fn() in ES modules

Documentation: - README.md: Architecture overview and usage guide - INTEGRATION_GUIDE.md: Complete
  integration examples - QUICK_REFERENCE.md: Copy-paste code snippets - MODULAR_ARCHITECTURE.md:
  What we built and why - TESTING_SETUP.md: How to run tests and troubleshoot

Benefits: ✓ Testable API clients isolated from ComfyUI dependencies ✓ Centralized error handling
  with automatic user feedback ✓ Reusable utilities across all nodes ✓ Full test coverage capability
  (9 passing tests) ✓ Progressive enhancement - works alongside existing code ✓ Easy to extend with
  new API endpoints

Migration Path: - No breaking changes to existing fb_tools.js - Import and use API clients as needed
  - Gradually refactor nodes to use new architecture - Remove old fetch calls once migrated

Test Results: Test Suites: 1 passed, 1 total Tests: 9 passed, 9 total

Time: 0.526 s

Usage Example: import { promptCollectionAPI } from "./api/prompt_collection.js";

const session = await promptCollectionAPI.createSession(); const result = await
  promptCollectionAPI.addPrompt( session.session_id, "girl_pos", "beautiful woman smiling" );

- Add ScenePromptManager and PromptComposer nodes
  ([`05e9e60`](https://github.com/frost-byte/fbTools/commit/05e9e600769ffc2f13d14aa9e505dddaed601b94))

Implements dictionary-based prompt composition system:

ScenePromptManager: - CRUD operations for scene prompts - Interactive table UI (will add JS in next
  commit) - Manages PromptCollection within SceneInfo - Processing type configuration (raw/libber)

PromptComposer: - Composes multiple output prompts from collection - Flexible output naming (no
  hardcoded prompt_a/b/c) - Returns PROMPT_DICT with user-defined keys - Automatic libber
  substitution during composition - Saves/loads composition maps as JSON

PromptCollection.compose_prompts(): - New method for dynamic composition - Takes composition map:
  {output_name: [prompt_keys]} - Processes libber substitutions inline - Returns dict of composed
  prompt strings

Benefits: - Infinitely extensible outputs (no fixed limit) - Self-documenting (key names describe
  purpose) - Same prompts, different compositions per workflow - Single DICT output type simplifies
  maintenance

- Add ScenePromptManager interactive table UI
  ([`fd5d315`](https://github.com/frost-byte/fbTools/commit/fd5d315f050c04084bec30d891b41b04a8d67804))

- Created setupScenePromptManager() in js/nodes/scene.js - Interactive table similar to
  LibberManager - Columns: Key | Value | Type (raw/libber dropdown) | Libber Name | Category |
  Actions - Add/Remove prompts with visual feedback - Apply button to update collection_json -
  Auto-updates from backend on execution - Type dropdown enables/disables libber name input -
  Registered in fb_tools.js extension system - Toast notifications for user actions

- Add StorySceneBatch job_id input, scene list API, and UI improvements
  ([`325241a`](https://github.com/frost-byte/fbTools/commit/325241a87fb9b9be96592ed0c855068e1b2a6c65))

- Add optional job_id input to StorySceneBatch node for reusable job directories - Add
  /fbtools/scene/list REST API endpoint for available scenes - Improve StoryEdit UI: add scene
  dropdown on new scenes, auto-load scenes - Add stylesheet loading in fb_tools.js init hook -
  Create style.css for prompt textarea styling - Update story.js API client with listScenes method -
  Filter internal flags from story save operations

- Add StoryVideoSave node for video batch workflow
  ([`fc0c215`](https://github.com/frost-byte/fbTools/commit/fc0c2153cd7c3ac557d399f5a374edb16e1f4a52))

- Implement StoryVideoSave node to complete video generation workflow - Takes video output from
  generation nodes + VIDEO_BATCH - Saves to correct path from video descriptor - Automatic directory
  creation - Pass-through video output for chaining - Outputs filename, filepath, scene info

- Node features: - Matches StorySceneImageSave pattern for consistency - Supports string path videos
  (file copy) - Extensible for other video formats - Preview UI shows saved location and scene
  details

- Update STORY_VIDEO_README.md: - Add StoryVideoSave node documentation - Complete workflow examples
  with save step - Show full iteration pattern

Complete video workflow is now: StoryLoad → StoryVideoBatch → [Iterate] → Generate Video →
  StoryVideoSave

This completes the video generation system, providing full parity with the image generation workflow
  (StorySceneBatch → Generate → StorySceneImageSave)

- Add subject compositor utility and tests
  ([`94834a9`](https://github.com/frost-byte/fbTools/commit/94834a90cca4561145fd5ae26ab7250367119dd1))

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

- Add video generation workflow support for story scenes
  ([`9bd33cf`](https://github.com/frost-byte/fbTools/commit/9bd33cf996f98827c8713cd7d9c1640a9eb09ed4))

- Add video prompt fields to SceneInStory model - video_prompt_source:
  'auto'|'prompt'|'composition'|'custom' - video_prompt_key: key for prompt/composition lookup -
  video_custom_prompt: custom video generation prompt

- Create utils/story_video.py with testable video utilities - list_job_ids(): List available jobs
  sorted by modification time - find_scene_image(): Locate scene images by order and name -
  pair_consecutive_scenes(): Create scene transition pairs - generate_video_filename(): Generate
  standardized video filenames - resolve_video_prompt(): Resolve video prompts from scene config -
  build_video_descriptor(): Build complete video generation descriptor

- Implement StoryVideoBatch node - Lists available job IDs from story directory - Iterates through
  scene pairs for video transitions - Outputs VIDEO_BATCH with first/last frame paths, prompts, LoRa
  data - Supports video_prompt_source modes: auto, prompt, composition, custom - Generates
  standardized video filenames (001_to_002_opening_to_battle.mp4)

- Add comprehensive test coverage - 29 new unit tests in tests/test_story_video.py - Tests job
  listing, image finding, scene pairing, prompt resolution - All 150 tests passing (121 existing +
  29 new)

- Create STORY_VIDEO_README.md documentation - Complete workflow guide for video generation - Node
  usage and configuration examples - Video descriptor format specification - Directory structure and
  naming conventions - Integration patterns with video generation nodes

Video generation workflow enables: 1. Load story with StoryLoad 2. Select job ID with
  StoryVideoBatch (lists available jobs) 3. Iterate through video descriptors 4. Generate videos
  between consecutive scenes 5. Use LoRa data and video prompts for consistent style 6. Save to
  job_output_dir with standardized naming

This extends the story building system from image generation to complete video generation workflows,
  maintaining consistency with existing patterns and full test coverage.

- Add websocket image save
  ([`fa764fa`](https://github.com/frost-byte/fbTools/commit/fa764fadb8a21c1be31bdae41993c82b01cc1bcc))

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

- Complete LibberManager and LibberApply UX overhaul with modular architecture
  ([`2224753`](https://github.com/frost-byte/fbTools/commit/2224753eb016ce074ae3fcd5a6338150cf826599))

Major improvements to the Libber system with enhanced user experience:

LibberManager: - Replaced dropdown-based operations with interactive editable table - Inline editing
  with textarea inputs for uniform cell heights (38px) - Per-row action buttons (✏️ Update, ➖
  Remove) matching cell height - Sticky action bar with 📂 Load, 💾 Save, and ➕ Create buttons -
  Inline libber creation with text input field and create button - Auto-save after add/update/remove
  operations - Simplified schema: single libber_name combo (basenames only, no .json extension) -
  Smart auto-loading: checks memory → file → creates new libber

LibberApply: - Click-to-insert functionality with delimiter wrapping - Cursor position tracking
  across focus changes - Native browser undo/redo support using execCommand - Always-visible 🔄
  Refresh button (sticky at top) - Smart libber discovery: scans memory and disk files - Empty state
  messaging with helpful hints - Dynamic table sizing responding to node resize

Code Architecture: - Modularized into separate node modules: * js/nodes/libber.js - LibberManager
  and LibberApply * js/nodes/scene.js - SceneSelect extensions * js/nodes/story.js - StoryEdit and
  StoryView extensions - Main fb_tools.js reduced from ~1400 to ~450 lines - Clean import structure
  with node-type routing

Technical Improvements: - LiteGraph NODE_TITLE_HEIGHT and NODE_WIDGET_HEIGHT for proper sizing - CSS
  variables for theming (--comfy-input-bg, --border-color, --fg-color) - Sticky positioning
  (position: sticky, top: 0, z-index: 10) - Button styling with min-height and flexbox centering -
  Responsive table layout with proper overflow handling

Breaking Changes: - LibberManager schema simplified (removed
  operation/key_selector/lib_key/lib_value widgets) - libber_name and filename merged into single
  libber_name Combo (basenames only) - Execute method auto-creates libber if not exists, skips if
  "none" selected

This commit represents a complete UX transformation from tedious dropdown operations to a modern,
  interactive table-based workflow with significantly improved usability.

- Dynamic job_id dropdown updates when story_name changes in StorySceneBatch
  ([`d455b79`](https://github.com/frost-byte/fbTools/commit/d455b79f33c0a0739542c5ab0fc54e054d670d04))

- Frontend: Added callback to story_name widget to fetch and update job_id options via
  /fbtools/story/job_ids API - Frontend: job_id dropdown now auto-populates on node creation for
  default story - Frontend: job_id options refresh automatically when user changes story selection -
  Backend: Simplified job_id schema to start with empty option only (frontend handles population) -
  Backend: Updated tooltip to clarify dynamic behavior - Improves UX by eliminating need to execute
  node just to update job_id list

- Enhance LibberManager and LibberApply nodes with improved UX
  ([`9595c30`](https://github.com/frost-byte/fbTools/commit/9595c3068cca3acd19dfeaf68351a0cbab527f37))

Backend changes: - Refactored Libber nodes into unified LibberManager node - Fixed get_libber_data
  method to use libber.libs instead of libber.lib_dict - Consolidated LibberCreate, LibberLoad, and
  LibberSave into single manager interface - Added operations: create, load, add_lib, remove_lib,
  save - Implemented LibberStateManager for persistent state management - Added REST API endpoints
  for Libber operations

Frontend changes (LibberManager): - Fixed ComboWidget rendering by using widget.options.values
  pattern - Added auto-save after add_lib and remove_lib operations - Implemented auto-clear of
  lib_key field after successful operations - Added auto-select of newly added key or first
  available after remove - Implemented auto-load of libber data on node creation/page refresh -
  Added key normalization (lowercase, replace spaces/hyphens with underscores)

Frontend changes (LibberApply): - Replaced JSONView formatter with clean HTML table display - Added
  scrollable container with max-height: 250px - Implemented two-column table layout (Key | Value) -
  Added theme-aware styling using CSS variables - Improved dynamic node sizing to fit content -
  Added HTML escaping for safe value display

Testing infrastructure: - Restructured test files from js/tests/ to js-tests/ - Updated package.json
  with Jest configuration - Moved test utilities and test files to new structure

This update significantly improves the Libber workflow by consolidating operations into a single
  manager node, adding automatic persistence, and providing a clean table view for reviewing lib
  definitions.

- Implement PromptCollection v2 system with REST API (Steps 1-2)
  ([`d4a735f`](https://github.com/frost-byte/fbTools/commit/d4a735ff5191b71b25d22bf7e3cd72b2cefea975))

Add flexible multi-prompt system with non-destructive migration:

- PromptCollection data model with PromptMetadata * Supports unlimited named prompts with
  categories/tags * V2 format with v1_backup for rollback capability * Auto-migration from legacy v1
  format

- REST API infrastructure for prompt management * PromptCollectionStateManager with 30min TTL * POST
  /fbtools/prompts/create, add, remove * GET /fbtools/prompts/list_names * Server-side session-based
  state management

- SceneInfo backward compatibility * Added prompts: Optional[PromptCollection] field * Legacy fields
  (girl_pos, male_pos, etc.) still work * save_prompts() auto-migrates to v2 on save *
  load_prompt_json() detects format and auto-migrates

- Non-destructive migration strategy * All v1 data preserved in v1_backup field * Existing code
  continues to work unchanged * Transparent auto-migration on file operations

Refs: plan-flexibleMultiPromptSystemLibberBugFix.prompt.md Steps 1-2

- Implement video prompt configuration with model extraction
  ([`32d0378`](https://github.com/frost-byte/fbTools/commit/32d0378572ecd35a74c1608e3f09d3c82dba8e5e))

Core Changes: - Extract SceneInStory and StoryInfo models to story_models.py * Enables isolated
  testing without ComfyUI dependencies * Follows prompt_models.py architecture pattern * Reduces
  extension.py by ~160 lines

- Fix load_story() to deserialize video prompt fields from JSON * Added video_prompt_source,
  video_prompt_key, video_custom_prompt to load logic * Fields were being saved but not loaded,
  causing defaults on reload * Now properly restores saved video prompt configuration

Frontend (js/nodes/story.js): - Dynamic video prompt UI in StoryEdit Advanced Flags tab *
  Source-based input types: dropdown for prompt/composition, textarea for custom * Auto-populated
  dropdowns with available prompt/composition keys * Live preview textarea showing resolved prompt
  text * Proper event handling for all video prompt controls

Backend (extension.py): - Updated load_story() V2 format parsing to include video fields - API
  endpoints already had video field support via getattr() defaults - All save/load cycles now fully
  support video prompt persistence

Testing: - 6 comprehensive video prompt persistence tests - Tests validate: data structures,
  serialization, deserialization, roundtrip - Full test suite: 156 tests passing (150 existing + 6
  new) - Story models now testable in isolation

Documentation: - VIDEO_PROMPT_UI_LAYOUT.md: Visual reference for UI layout and interactions -
  VIDEO_PROMPT_UX_IMPLEMENTATION.md: Technical implementation details and data flow

Fixes: - Video prompt fields now persist correctly through save/load cycles - Browser reload
  properly restores video prompt configuration - Preview textarea updates dynamically based on
  source and selection

Architecture: - Improved code organization with model extraction - Better separation of concerns
  (data models vs business logic) - Easier testing and maintenance going forward

- Integrate scene_flags into PromptCollection and add overlay feedback utility
  ([`753c1e0`](https://github.com/frost-byte/fbTools/commit/753c1e08c1aa3eea5d000f90f64f27b6f22ec03e))

## Backend Changes - **PromptCollection Model (prompt_models.py)**: - Added scene_flags as
  Optional[dict] field to store per-scene control flags (use_depth, use_mask, use_pose, use_canny) -
  Updated to_dict() to include scene_flags when not None - Updated from_dict() to load scene_flags
  from incoming data - Maintains backward compatibility (scene_flags is optional)

- **Scene Prompts API (extension.py)**: - scene_get_prompts: Now returns scene_flags in response -
  scene_save_prompts: Simplified to use model serialization (scene_flags preserved automatically)

## Frontend Changes - **Reusable Overlay Utility (js/utils/feedback.js)**: - Created showOverlay()
  function for consistent success/error feedback - Replaces hardcoded overlays and toast
  notifications - Supports success (green) and error (red) types with auto-hide

- **Updated Nodes**: - ScenePromptManager: Added 'Save Flags' button with overlay feedback -
  StoryEdit: Migrated to use showOverlay instead of hardcoded overlay HTML

## Test Coverage - **Backend Tests (13 new tests + 7 integration tests)**: -
  test_scene_prompts_api.py: Comprehensive scene_flags testing (serialization, persistence,
  compositions, array formats, migration) - test_prompt_collection.py: Added
  TestSceneFlagsInCollection with 7 integration tests

- **Frontend Tests**: - prompt_collection_api.test.js: Added scene_flags handling tests (3 tests)

All 51 backend tests passing. Scene flags fully integrated through save/load cycle.

- Migrate to structured logging and fix test infrastructure
  ([`ec5c157`](https://github.com/frost-byte/fbTools/commit/ec5c157718e92f69a1eba05eca8ae0d5ae2a104b))

Complete migration from print statements to structured logging with environment-configurable log
  levels via FBTOOLS_LOG_LEVEL.

Backend Changes: - Add utils/logging_utils.py with get_logger() for centralized logging - Replace
  all print statements with logger calls in extension.py - REST managers now use
  logger.info/warning/error/exception - Node execution uses appropriate log levels
  (debug/info/warning) - Exception paths use logger.exception for full tracebacks - Update
  utils/io.py and prompt_models.py to use structured logging - Add try/except fallback in
  prompt_models.py for test compatibility

Test Infrastructure Fixes: - Remove obsolete tests/test_fb_tools.py (referenced non-existent code) -
  Remove tests/__init__.py (caused pytest package resolution issues) - Update tests/conftest.py to
  properly handle package imports - Clean up unused src/fb_tools/ stub files

Frontend Test Fixes: - Add getCalls() method to mockFetch utility for request inspection - Fix
  libber_api.test.js mock setup and response handling - Suppress expected console.error in error
  handling tests - Fix integration test to provide separate mocks per API call

Test Results: - ✅ 99 Python tests passing (pytest) - ✅ 38 JavaScript tests passing (jest) - ✅ 137
  total tests validating no regressions

Log levels available: DEBUG, INFO, WARNING, ERROR, CRITICAL Set via: export FBTOOLS_LOG_LEVEL=DEBUG

- Register compositing and LoRA stack nodes, update docs and deps
  ([`31bc42e`](https://github.com/frost-byte/fbTools/commit/31bc42e81d26a8fa5f1532c85355478dc512e69a))

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

- Replace libber text input with dropdown in ScenePromptManager
  ([`44fd667`](https://github.com/frost-byte/fbTools/commit/44fd667e61aecdc92612d61386a0e42609a15f69))

Backend changes: - Get list of available libbers from LibberStateManager - Include libbers list in
  UI text array (text[3]) - Always include 'none' as first option

Frontend changes: - Replace prompt-libber-input with prompt-libber-select dropdown - Populate
  dropdown with available libbers from backend - When Type='libber': enable dropdown, auto-select
  first libber if 'none' - When Type='raw': disable dropdown, set to 'none' - Updated all event
  handlers to use dropdown value - Apply button handles 'none' correctly (saves as null) - New row
  starts with 'none' selected and disabled

UX improvements: - No more manual libber name entry (prevents typos) - Clear visual indication of
  available libbers - Consistent behavior between raw/libber types - Better defaults (first
  available libber when switching to libber type)

- Storyedit REST API + comprehensive testing
  ([`8e15d67`](https://github.com/frost-byte/fbTools/commit/8e15d67a4aea3641995bb1984baac3f6a69b2113))

Implement complete REST API architecture for StoryEdit node with immediate data loading and full
  test coverage.

## Features

### REST API Implementation - Add GET /fbtools/story/load/{story_name} endpoint - Loads story.json
  with full scene data - Returns JSON with scenes array - Add POST /fbtools/story/save endpoint -
  Saves updated scenes to story.json - Validates story exists before saving - Frontend fetch() calls
  replace execution-based data transfer - Immediate data loading on node initialization

### Frontend Improvements - loadStoryData() - async load via REST API - saveStory() - async save via
  REST API with success feedback - Enhanced error handling and user feedback - Detailed console
  logging for debugging - Table initialization without workflow execution

### Testing - 9 Python unit tests (all passing) - Helper method logic (prompt text, summary,
  metadata) - Scene resolution and reordering - Data structure validation - 12 JavaScript tests (all
  passing) - Node initialization and UI rendering - Scene management logic - Data validation -
  Execution handler - Comprehensive testing documentation - STORY_EDIT_TESTING_GUIDE.md - manual
  test scenarios - STORY_EDIT_TESTING_SUMMARY.md - test overview - STORY_EDIT_TESTING_FINAL.md -
  results summary

### Bug Fixes - Fix jest test compatibility (global.fetch mock) - Fix console.log expectation
  ("Received story data") - Fix create_mask_overlay_image transparency logic - Add pyright
  configuration for type checking

### Configuration - Add nvm.fish persistence (nvm_default_version v20.19.6) - Configure fish shell
  auto-load for Node.js

## Test Results ✅ 9 Python tests passing in 0.02s ✅ 12 JavaScript tests passing in 0.60s ✅ 21 total
  automated tests ✅ All manual test scenarios documented

## Files Changed - extension.py - REST API endpoints + logging - js/nodes/story.js - Complete UI
  redesign with API calls - js-tests/story_edit.test.js - Full test suite - tests/test_story_edit.py
  - Unit tests - pyproject.toml - Add pyright config - utils/images.py - Fix mask overlay
  transparency

## Architecture Changed from execution-based data flow to REST API: - Before: Execute node → backend
  sends data → frontend displays - After: Select story → frontend fetches via API → immediate
  display

Co-authored-by: GitHub Copilot <copilot@github.com>

- **fbtools**: Add MultiLoraLoader and align LibberApply libber discovery/loading
  ([`1e5f02a`](https://github.com/frost-byte/fbTools/commit/1e5f02aa253c748d77216d461f5e8805c0448135))

add MultiLoraLoader node with up to 10 optional LoRA slots and sequential model-only application
  register MultiLoraLoader in extension node list fix LibberApply.define_schema to include libbers
  from both memory and disk (.json scan), like LibberManager handle libber_name == "none" early in
  LibberApply.execute update frontend LibberApply dropdown population to merge/dedupe/sort libbers +
  files from /fbtools/libber/list remove hardcoded frontend load path (libbers) and load using
  backend-provided libber_dir + matching filename extend /fbtools/libber/list response with
  libber_dir for consistent frontend/backend path resolution

- **LibberApply**: Add interactive table with click-to-insert and undo support
  ([`585288e`](https://github.com/frost-byte/fbTools/commit/585288e26329d57065a078e99159dbedaf7c7d50))

Table Display & Sizing: - Fixed table persistence after node execution by storing updateDisplay
  function reference - Implemented dynamic container height that adapts to node size changes - Added
  resize hooks (onResize) to update table when user resizes node - Set height constraints (min:
  150px, max: 600px) to prevent infinite growth - Fixed bottom edge overlap with 15px margin -
  Improved widget height computation accounting for previous widgets

Interactive Features: - Made table rows clickable to insert lib keys into text input - Added cursor
  position tracking with event listeners (click, keyup, select, focus) - Keys are automatically
  wrapped with configured delimiter when inserted - Stores last cursor position to handle focus
  changes when clicking table - Added hover effect to table rows (background color highlight)

Undo/Redo Support: - Implemented browser native undo/redo using document.execCommand('insertText') -
  Users can now press Ctrl+Z/Cmd+Z to undo insertions - Users can press Ctrl+Y/Cmd+Shift+Z to redo -
  Fallback to manual insertion if execCommand not supported - Maintains ComfyUI state
  synchronization after insertions

UX Improvements: - Corrected widget reference from "input_text" to "text" - Added visual feedback
  with row hover states - Automatic focus return to input after insertion - Cursor positioned after
  inserted text for continued editing

Users can now click any lib key in the table to insert it at their cursor position with full
  undo/redo support.

- **lora**: Add LoRA stack API client and node UI
  ([`8465b49`](https://github.com/frost-byte/fbTools/commit/8465b49faf2dd99444c6c8351f78e251764eec76))

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

- **lora**: Add LORA_STACK output to LoraStackCollect and update WanPreset nodes
  ([`c80f400`](https://github.com/frost-byte/fbTools/commit/c80f400e4dc8b683e95c75058fd6ce6c0bb6b6b0))

- LoraStackCollect: add easy-use compatible LORA_STACK output (list of (lora_name, model_strength,
  clip_strength) tuples) for interop with EasyLoraStack, PowerLoraLoader, and other LORA_STACK
  consumers - WanPresetDefine: replace single-lora Combo inputs with optional LORA_STACK inputs for
  lora_h and lora_l, enabling multi-lora stacks per preset slot - WanPresetSelect: change
  lora_h/lora_l outputs from STRING to LORA_STACK for direct connection to downstream loader nodes

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

- **lora**: Add WanPresetDefine and WanPresetSelect nodes
  ([`4ca75b3`](https://github.com/frost-byte/fbTools/commit/4ca75b3ccedb6c659050902a996101df04d4f19d))

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

- **lora**: Register WanPresetDefine and WanPresetSelect in extension
  ([`2caa999`](https://github.com/frost-byte/fbTools/commit/2caa999654d4e57b3d7bbc4b3fe58f6bb1677752))

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

### Refactoring

- Extract testable scene image saving utilities and flatten directory structure
  ([`d692f73`](https://github.com/frost-byte/fbTools/commit/d692f739072dfad246e231244d5da519949f6315))

- Extract scene image save logic to utils/scene_image_save.py - Add SceneImageSaveConfig class for
  pure data handling - Add ImageSaver class with static methods for I/O operations - Add
  select_scene_descriptor() and generate_preview_text() pure functions - Enable comprehensive unit
  testing without ComfyUI dependencies

- Update extension.py to use extracted utilities - Refactor StorySceneBatch to create flat directory
  structure - Change from job_root/{scene_order}_{scene_name}/output/ to job_root/input/ - Update
  StorySceneImageSave to prefer job_input_dir over job_output_dir - Remove inline class definitions
  in favor of imported utilities

- Unify test import strategy across all test files - Create import_test_module() helper in
  conftest.py - Update all 5 test files to use consistent import approach - Resolve module import
  conflicts with built-in utils namespace - Ensure stable imports using importlib.util with unique
  module names

- Add comprehensive test coverage for scene image saving - Create tests/test_scene_image_save.py
  with 22 unit tests - Test filename generation, filepath generation, descriptor parsing - Test
  scene selection, sorting, index clamping - Test preview text generation for different formats -
  Mock I/O operations for isolated unit testing

- Document testing approach - Add TESTING_GUIDE.md with unified import patterns and best practices -
  Add TEST_SUMMARY.md showing 121/121 tests passing - Include examples and troubleshooting guidance

This refactoring improves testability, maintainability, and consistency across the codebase while
  fixing the directory structure to use a flat job-level input/ directory instead of nested
  per-scene subdirectories.

- Make StoryVideoBatch self-contained with story/job combo widgets
  ([`3b41d92`](https://github.com/frost-byte/fbTools/commit/3b41d92ea6cb2f923c38eefc19ee35d18d374121))

- Removed STORY_INFO input requirement - Added story_name combo widget that lists available stories
  - Added job_id combo widget that lists available jobs (auto-populated from first story) - Node now
  loads story internally based on story_name selection - Added story_name output for reference -
  Single execution needed - no need to run twice to populate job_id combo - Default behavior: loads
  first available story and its jobs automatically

- Remove legacy prompt inputs from SceneCreate, add auto-migration
  ([`2651e3c`](https://github.com/frost-byte/fbTools/commit/2651e3c9235a082b038a55ef1f1107b20306320d))

BREAKING CHANGE: SceneCreate no longer has individual prompt inputs.

Changes: - SceneCreate: Removed girl_pos, male_pos, wan_prompt, wan_low_prompt, four_image_prompt
  inputs - SceneCreate: Now creates empty PromptCollection, users add prompts via ScenePromptManager
  - SceneInfo.from_pose_directory(): Auto-migrates legacy prompts.json files * Detects v2 format
  (has 'version' field) → loads as-is * Detects legacy format → calls from_legacy_dict() for
  migration * No prompts.json → creates empty collection - Simplified SceneCreate execute() -
  removed prompt string handling

Migration path for existing scenes: 1. Load scene with SceneSelect or from_pose_directory 2. Legacy
  prompts.json automatically migrated to PromptCollection 3. Edit prompts via ScenePromptManager 4.
  Compose outputs via PromptComposer

This enables clean separation: SceneCreate handles assets, ScenePromptManager handles prompts.

- Simplify PromptMetadata for node-level composition
  ([`80930db`](https://github.com/frost-byte/fbTools/commit/80930db4ca53ae157def29fdd037c9e02b13b9de))

BREAKING CHANGE: Removed output_slot and order from PromptMetadata. Output composition is now
  handled at the node level, not in metadata.

Changes: - PromptMetadata: Removed output_slot and order fields - PromptCollection: Removed
  compose_output() and get_output_slots() - PromptCollection: Added get_prompt_metadata() and
  get_prompts_by_category() - Legacy migration: Simplified to just convert prompts to raw type -
  Tests: Updated to reflect simplified data model

Rationale: Output composition should be workflow-specific, not prompt-specific. Same prompts can be
  composed differently for images vs video workflows. This eliminates prompt duplication and allows
  dynamic composition.

- Simplify StoryVideoBatch to output input folder path, multiline prompts, and aggregated LoRAs
  ([`7f668ee`](https://github.com/frost-byte/fbTools/commit/7f668ee85c1980bdf7fc31376715c33ff0efe38f))

- Changed StoryVideoBatch to output: 1. input_folder_path - Path to job input folder with ordered
  scene images 2. video_prompts - Multiline string with one prompt per transition (with
  libber/composition processing) 3. loras_high - Aggregated high-priority LoRAs (unique by name) 4.
  loras_low - Aggregated low-priority LoRAs (unique by name) - Removed complex VIDEO_BATCH
  descriptor system - Removed StoryVideoSave node (no longer needed) - Video prompts now fully
  processed with libber substitutions and composition support - LoRAs aggregated across all scenes
  so each lora appears only once per output - Simpler workflow: load images from folder, use
  multiline prompts, apply aggregated LoRAs

### Testing

- Add comprehensive integration tests for prompt composition system
  ([`99004f9`](https://github.com/frost-byte/fbTools/commit/99004f95cdc9404a8cc05304dc75dd4f4105ff86))

- TestPromptCollectionCompose: Test compose_prompts() method * Single/multiple outputs * Missing
  keys handling * Libber substitution with/without manager * Mixed raw and libber prompts

- TestPromptCompositionSerialization: Unicode and JSON roundtrip

- TestLegacyPromptMigration: v1->v2 migration and v2 format detection

- TestPromptCollectionFileOperations: Save/load operations

- TestPromptCompositionWorkflows: Real-world scenarios * Image generation workflow * Video high/low
  quality outputs * Multi-image compositions * Libber-enhanced workflows

All 90 tests passing
