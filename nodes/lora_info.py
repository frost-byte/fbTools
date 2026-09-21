"""LoRA listing and CivitAI info/hash REST routes (/fbtools/loras, /fbtools/lora/*).

Moved out of extension.py (pure code motion)."""
from __future__ import annotations

import hashlib
import folder_paths
from aiohttp import web
from pathlib import Path
import json
from .shared import routes


_lora_info_cache: dict[str, dict] = {}


def _compute_lora_hash(path: str) -> str:
    """Full-file SHA256 — matches the hash CivitAI indexes in its by-hash API."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@routes.get("/fbtools/loras/list")
async def _loras_list(request):
    """Return sorted list of LoRA filenames from all registered loras folders."""
    try:
        names = sorted(folder_paths.get_filename_list("loras"))
        return web.json_response({"loras": names})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.get("/fbtools/lora/civitai_info")
async def get_lora_civitai_info(request):
    """Fetch model version info from civitai by lora filename.

    Query params: lora=<filename relative to loras folder>
    Returns civitai model-version JSON or {"error": "..."}.
    Results are cached in memory and a .civitai.info sidecar is checked first
    (compatible with rgthree-comfy sidecar files).
    """
    import aiohttp as _aiohttp

    lora_name = request.rel_url.query.get("lora", "").strip()
    if not lora_name or lora_name == "None":
        return web.json_response({"error": "lora parameter required"}, status=400)

    lora_path = folder_paths.get_full_path("loras", lora_name)
    if not lora_path:
        return web.json_response({"error": f"LoRA not found: {lora_name}"}, status=404)

    if lora_path in _lora_info_cache:
        return web.json_response(_lora_info_cache[lora_path])

    # Honour existing rgthree-style sidecar without re-fetching.
    # rgthree names sidecars "<filename>.civitai.info" (full filename preserved),
    # e.g. "my_lora.safetensors.civitai.info".
    sidecar = Path(lora_path).parent / (Path(lora_path).name + ".civitai.info")
    if sidecar.exists():
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            _lora_info_cache[lora_path] = data
            return web.json_response(data)
        except Exception:
            pass

    try:
        sha256 = _compute_lora_hash(lora_path)
    except Exception as e:
        return web.json_response({"error": f"Hash computation failed: {e}"}, status=500)

    civitai_url = f"https://civitai.com/api/v1/model-versions/by-hash/{sha256}"
    try:
        async with _aiohttp.ClientSession() as session:
            async with session.get(
                civitai_url,
                headers={"User-Agent": "comfyui-fbTools/1.0"},
                timeout=_aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json(content_type=None)
                    _lora_info_cache[lora_path] = data
                    return web.json_response(data)
                elif resp.status == 404:
                    return web.json_response(
                        {"error": "LoRA not found on Civitai"}, status=404
                    )
                else:
                    return web.json_response(
                        {"error": f"Civitai returned HTTP {resp.status}"}, status=502
                    )
    except Exception as e:
        return web.json_response({"error": f"Civitai request failed: {e}"}, status=502)
