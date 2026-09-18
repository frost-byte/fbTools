# Handoff: Wiring Unsloth-on-Modal into comfyui-fbTools

**Audience:** a Claude Code session working in
`/mnt/comfy_ssd/ComfyUI/custom_nodes/comfyui-fbTools/`, tasked with (1)
managing the Modal deploy/spin-up/tear-down lifecycle for the Unsloth
Studio app and (2) wiring it into the extension's existing LLM backend
system as a new backend option.

**Source repo for the Modal side:** `/home/beerye/projects/modal_cloud/`
(`unsloth_studio.py` + `docs/unsloth-studio-modal-plan.md`, which has the
full build/debug history — read that first if anything here is unclear or
seems to need changing).

---

## 1. This is a different integration shape than the existing one

fbTools already has one Modal-hosted LLM backend: `vision_llm.py`
(`fbtools-vision-llm` app, `VisionLLM` class). That one is called via the
**Modal Python SDK directly** — `modal.Cls.from_name(...)`, an RPC-style
call where Modal's client library handles cold starts, queuing, etc.
transparently. Its `generate()` mirrors `utils/llm_client.py`'s signature
exactly so it's a drop-in remote alternative to the local model.

**Unsloth Studio is not that shape.** It's a `@modal.web_server` app
exposing a **plain OpenAI-compatible HTTP API** (`/v1/chat/completions`,
bearer-token auth) at a public HTTPS URL. There is no `modal.Cls` to call
— integration means an HTTP client call (`requests`, or the `openai`
Python SDK pointed at a custom `base_url`), not a Modal SDK call. Don't
try to reuse `vision_llm.py`'s calling pattern here; the *concept* (a
new selectable backend in the LLM panel) carries over, the *mechanism*
does not.

## 2. What's already deployed

App name: `unsloth-studio` (Modal workspace `frost-byte`). Deployed and
tested as of 2026-09-04. Check live status with:

```bash
modal app list | grep unsloth
modal container list   # see what's currently warm/billing
```

Endpoints (all `POST .../v1/chat/completions`, OpenAI chat-completions
request/response shape):

| Function | URL | Model served | Notes |
|---|---|---|---|
| `serve_l4_qwen3_8_27b` | `https://frost-byte--unsloth-studio-serve-l4-qwen3-8-27b.modal.run` | `unsloth/Qwen3.8-27B-GGUF` (`UD-Q3_K_XL`, dense, 27B) | **Recommended default.** 131072 context, ~17.5 tok/s warm, cold start ~2-5 min once cached. |
| `serve_l4_qwen3_8b` | `https://frost-byte--unsloth-studio-serve-l4-qwen3-8b.modal.run` | `unsloth/Qwen3-8B-GGUF` (`Q4_K_XL`, 8B) | Small/fast sanity-check model. ~25 tok/s warm. Lower quality than the 27B. |
| `serve_l4_qwen3_8_flash_next` | `https://frost-byte--unsloth-studio-serve-l4-qwen3-8-flash-next.modal.run` | `unsloth/Qwen3.8-Flash-Next-GGUF` (`UD-IQ1_M`, 125B MoE) | Works, but ~37 min first-ever cold start (uncached download), ~14.7 tok/s warm even when cached. Only worth it for the RAM/VRAM-split curiosity, not general use. |
| `serve_l4_qwen3_8_27b_mtp`, `serve_l4_qwen3_8_flash_next_mtpfork`, `serve_l4_qwen3_8_flash_next_novision_mtpfork` | (deployed, not documented here) | — | **Concluded experiments, don't use.** MTP forcing made things slower (3.6x) or outright broken (Flash-Next). Kept only as documented dead ends — see plan doc sec 10-12. |

**Model choice recommendation for fbTools integration: `serve_l4_qwen3_8_27b`.**
Fastest practical option we validated, no known issues, no exotic
architecture problems.

## 3. Auth

Bearer token, OpenAI-style `Authorization: Bearer <key>` header. The
stable, permanent API key (captured once via a bootstrap step, its hash
persists in the Studio's auth DB on a Modal Volume, so it survives every
cold start regardless of which function/container boots):

```
sk-unsloth-07a33e9bb5cf8661405dfb600e14d817
```

This is also stored as the Modal Secret `unsloth-studio-api-key` (value
under `UNSLOTH_STUDIO_API_KEY`) for reference, though Modal secrets are
write-only via CLI — the value above (already captured into this repo's
history) is the practical way to get it into fbTools' own config/secrets.

## 4. Minimal working call

```python
import openai

client = openai.OpenAI(
    base_url="https://frost-byte--unsloth-studio-serve-l4-qwen3-8-27b.modal.run/v1",
    api_key="sk-unsloth-07a33e9bb5cf8661405dfb600e14d817",
)
resp = client.chat.completions.create(
    model="unsloth/Qwen3.8-27B-GGUF",  # must match this string; the server is single-model per endpoint
    messages=[{"role": "user", "content": "..."}],
    max_tokens=512,
)
print(resp.choices[0].message.content)
```

Plain `requests` works identically if the `openai` package isn't already
a dependency — it's a standard REST call, nothing Modal-specific about
the request shape itself.

## 5. The real integration gotcha: cold starts

This is the thing most likely to break a naive integration, learned the
hard way this session. If the container has scaled down (15-minute idle
window, `scaledown_window=15*60` in `unsloth_studio.py`), the *first*
request after that doesn't just hang and eventually respond — Modal's own
edge returns an **HTTP 303** redirect (with a `__modal_attempt_token`
query param) after ~155s if the container is still booting, meaning a
naive client with a short timeout, or one that doesn't follow redirects
correctly, will fail outright rather than just being slow.

**What actually works, confirmed live:** treat a 303 as "still starting,
retry the *original* URL again" (not the token URL — following that with
a fresh client connection doesn't work cleanly, confirmed by testing).
Use a per-attempt timeout of ~200s (long enough to cover one 303 cycle)
and loop:

```python
import time
import httpx

def call_with_cold_start_retry(url, headers, payload, max_attempts=20, per_attempt_timeout=200):
    with httpx.Client(timeout=per_attempt_timeout) as client:
        for _ in range(max_attempts):
            try:
                r = client.post(url, headers=headers, json=payload)
            except httpx.TimeoutException:
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code != 303:
                r.raise_for_status()
            # 303: container still starting, retry the same original URL
    raise TimeoutError("gave up waiting for cold start")
```

Cold-start duration to plan around (cache-warm, i.e. not the very first
request ever against a freshly-deployed model): roughly **2-5 minutes**
for `serve_l4_qwen3_8_27b`. There's also a real, separate quirk: the very
first request to a *freshly-booted* container can get a fast `400 "No
model loaded"` instead of being held open — a harmless artifact of how the
model-load trigger works, not a real error; retry logic above handles it
the same way as a 303 (any non-200 just loops).

**For fbTools specifically:** decide whether interactive use (a ComfyUI
user waiting on a node) can tolerate a multi-minute first-call cold start.
If not, either (a) set `min_containers=1` on `serve_l4_qwen3_8_27b` in
`unsloth_studio.py` before deploying (keeps one container always warm —
continuous L4+24GB billing, no cold starts ever), or (b) have the
extension's backend-selection UI show a "warming up" state and just retry
with patience per the pattern above. (a) is simpler; (b) is cheaper. This
wasn't decided in the Modal-side session — flag it back to the user if it
matters for the UX you're building.

## 6. Spin-up / tear-down commands (for managing cost)

```bash
cd /home/beerye/projects/modal_cloud
modal deploy unsloth_studio.py          # (re)deploy all endpoints -- idempotent, ~2s if unsloth_studio.py unchanged
modal container list                     # see what's currently running/billing
modal container stop --yes <container-id>  # force-kill a specific running container
modal app stop unsloth-studio            # undeploy everything (all endpoints stop being servable)
```

Endpoints scale to zero automatically 15 minutes after the last request
(no action needed for normal cost control) — `container stop` is only
needed if you want to force it immediately (e.g., right after a manual
test) rather than waiting out the idle window.

## 7. Where to add the new backend option in the extension

Per `[[reference-comfyui-fbtools-layout]]` (this project's own memory,
carried over from the `vision_llm.py` integration): the relevant
extension code is in `extension.py` — the `/fbtools/llm/*` aiohttp routes
(model list/status/load/unload/generate, near line 16444+) and the
Source Profile analyze path (`_run_vision_inference` /
`_run_vision_inference_clip`, ~line 13109-13213), gated by a backend
selection in the "Compose -> LLM panel" frontend. `vision_llm.py`'s own
handoff guidance was to add its Modal `Cls` as a third backend option
alongside local (`utils/llm_client.py`) and itself; this Unsloth
integration should slot in the same way, as a fourth option -- just via
an HTTP client instead of `modal.Cls.from_name(...)`. This touches the
user's live ComfyUI extension: **confirm scope with the user before
editing `extension.py`**, same caution that applied to the vision_llm.py
integration.

## 8. Known limitations to carry into the integration

- **Single model per endpoint.** Each `serve_*` function boots one
  specific model; there's no runtime model-switching on one running
  container. If fbTools wants to offer multiple Unsloth-hosted models,
  that means multiple endpoint URLs (already deployed, see the table
  above), not one endpoint with a `model` selector that actually changes
  weights.
- **No streaming validated.** All testing this session used
  `"stream": false`. Streaming (`stream: true`, SSE) is part of the
  standard OpenAI API shape and llama-server supports it, but it was
  never exercised — verify it works before relying on it in a UI that
  expects token-by-token output.
- **Reasoning models.** `Qwen3.8-27B` emits `reasoning_content` (a
  separate field from `content`) for its chain-of-thought when thinking
  is enabled (it is, by default, in every config here). If `max_tokens`
  is low, the response can be *all* reasoning and empty `content` --
  size `max_tokens` generously (300+ recommended) or the extension will
  see empty replies. Not a bug, just a real behavior to design around.
