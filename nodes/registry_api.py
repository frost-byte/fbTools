"""Registry REST routes: concepts, scene templates, subject profiles and scene casts (/fbtools/{concepts,scene_templates,subjects,casts}/*).

Moved out of extension.py (pure code motion). Reload endpoints bump the counters in nodes/shared.py."""
from __future__ import annotations

from aiohttp import web
from ..utils.subject_profiles import load_registry as _load_subject_registry, save_registry as _save_subject_registry
from ..utils.scene_templates import scan_templates as _scan_scene_templates
from ..utils.concept_registry import load_registry as _load_concept_registry
from ..utils.scene_casts import load_registry as _load_cast_registry, save_registry as _save_cast_registry
from .shared import bump_reload, default_cast_registry_path, default_registry_path, default_scene_templates_dir, default_subject_profiles_path, routes
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


@routes.post("/fbtools/subjects/reload")
async def _subjects_reload(request):
    """Increment reload counter so SubjectProfileLoad/List nodes re-execute."""
    _subject_reload_counter = bump_reload("subject")
    logger.info("Subject profiles reload requested (counter=%d)", _subject_reload_counter)
    return web.json_response({"success": True, "counter": _subject_reload_counter})


@routes.get("/fbtools/subjects/profiles")
async def _subjects_get_profiles(request):
    """Return the current subject profiles as JSON for the frontend."""
    try:
        registry = _load_subject_registry(default_subject_profiles_path())
        return web.json_response(registry.to_dict())
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/scene_templates/reload")
async def _scene_templates_reload(request):
    """Increment reload counter so SceneTemplate nodes re-execute."""
    _scene_template_reload_counter = bump_reload("scene_template")
    logger.info("Scene templates reload requested (counter=%d)", _scene_template_reload_counter)
    return web.json_response({"success": True, "counter": _scene_template_reload_counter})


@routes.get("/fbtools/scene_templates/list")
async def _scene_templates_list(request):
    """Return the list of available template metadata as JSON."""
    try:
        templates_dir = default_scene_templates_dir()
        templates = _scan_scene_templates(templates_dir)
        return web.json_response({"templates": templates})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/concepts/reload")
async def _concepts_reload(request):
    """Increment reload counter so ConceptRegistryLoad nodes re-execute."""
    _concept_reload_counter = bump_reload("concept")
    logger.info("Concept registry reload requested (counter=%d)", _concept_reload_counter)
    return web.json_response({"success": True, "counter": _concept_reload_counter})


@routes.get("/fbtools/concepts/registry")
async def _concepts_get_registry(request):
    """Return the current default registry as JSON for the frontend."""
    try:
        registry = _load_concept_registry(default_registry_path())
        return web.json_response(registry.to_dict())
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.get("/fbtools/subjects/list")
async def _subjects_list(request):
    """Return [{id, name, appearance_summary, concept_id, pronoun_style}] sorted by name."""
    try:
        registry = _load_subject_registry(default_subject_profiles_path())
        items = []
        for sid, s in registry.subjects.items():
            items.append({
                "id":                 sid,
                "name":               s.get("name", sid),
                "appearance_summary": s.get("appearance", {}).get("summary", ""),
                "concept_id":         s.get("concept_id", ""),
                "pronoun_style":      s.get("pronoun_style", ""),
            })
        items.sort(key=lambda x: x["name"].lower())
        return web.json_response({"subjects": items})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.get("/fbtools/subjects/get")
async def _subjects_get_one(request):
    """Return a single subject profile by ?id=<subject_id>."""
    sid = request.rel_url.query.get("id", "")
    if not sid:
        return web.json_response({"error": "id parameter required"}, status=400)
    try:
        registry = _load_subject_registry(default_subject_profiles_path())
        subject = registry.get_subject(sid)
        if subject is None:
            return web.json_response({"error": f"Subject '{sid}' not found"}, status=404)
        return web.json_response({"id": sid, **subject})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/subjects/save")
async def _subjects_save(request):
    """Create or update a subject profile.  Body: full subject dict with 'id'."""
    try:
        data = await request.json()
        sid = data.get("id", "").strip()
        if not sid:
            return web.json_response({"error": "Subject 'id' is required"}, status=400)
        path = default_subject_profiles_path()
        registry = _load_subject_registry(path)
        # Merge into registry — preserve character_sheet_images if not provided
        existing = registry.subjects.get(sid, {})
        merged = {**existing, **data}
        merged["id"] = sid  # keep id consistent
        registry.subjects[sid] = {k: v for k, v in merged.items() if k != "id"}
        _save_subject_registry(registry, path)
        bump_reload("subject")
        return web.json_response({"success": True, "id": sid})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.delete("/fbtools/subjects/delete")
async def _subjects_delete(request):
    """Delete a subject by ?id=<subject_id>."""
    sid = request.rel_url.query.get("id", "")
    if not sid:
        return web.json_response({"error": "id parameter required"}, status=400)
    try:
        path = default_subject_profiles_path()
        registry = _load_subject_registry(path)
        if sid not in registry.subjects:
            return web.json_response({"error": f"Subject '{sid}' not found"}, status=404)
        del registry.subjects[sid]
        _save_subject_registry(registry, path)
        bump_reload("subject")
        return web.json_response({"success": True})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.get("/fbtools/casts/list")
async def _casts_list(request):
    """Return all scene casts."""
    try:
        registry = _load_cast_registry(default_cast_registry_path())
        return web.json_response({"casts": registry.list_casts()})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.get("/fbtools/casts/get")
async def _casts_get(request):
    """Return a single cast by ?id=<cast_id>."""
    cast_id = request.rel_url.query.get("id", "")
    if not cast_id:
        return web.json_response({"error": "id parameter required"}, status=400)
    try:
        registry = _load_cast_registry(default_cast_registry_path())
        cast = registry.get(cast_id)
        if cast is None:
            return web.json_response({"error": f"Cast '{cast_id}' not found"}, status=404)
        return web.json_response(cast)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/casts/save")
async def _casts_save(request):
    """Create or update a cast.  Body: full cast dict with 'id'."""
    try:
        data = await request.json()
        cast_id = (data.get("id") or "").strip()
        if not cast_id:
            return web.json_response({"error": "Cast 'id' is required"}, status=400)
        path = default_cast_registry_path()
        registry = _load_cast_registry(path)
        registry = registry.upsert(data)
        _save_cast_registry(registry, path)
        return web.json_response({"success": True, "id": cast_id})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.delete("/fbtools/casts/delete")
async def _casts_delete(request):
    """Delete a cast by ?id=<cast_id>."""
    cast_id = request.rel_url.query.get("id", "")
    if not cast_id:
        return web.json_response({"error": "id parameter required"}, status=400)
    try:
        path = default_cast_registry_path()
        registry = _load_cast_registry(path)
        if registry.get(cast_id) is None:
            return web.json_response({"error": f"Cast '{cast_id}' not found"}, status=404)
        registry = registry.delete(cast_id)
        _save_cast_registry(registry, path)
        return web.json_response({"success": True})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.post("/fbtools/casts/reload")
async def _casts_reload(request):
    """Increment reload counter so SceneCastLoad nodes re-execute."""
    _cast_reload_counter = bump_reload("cast")
    logger.info("Scene casts reload requested (counter=%d)", _cast_reload_counter)
    return web.json_response({"success": True, "counter": _cast_reload_counter})
