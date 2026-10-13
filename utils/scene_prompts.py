"""Pure scene prompt resolution shared by the Story nodes and the Scene/Story importer.

No ComfyUI deps and no imports of other project modules (so it loads standalone in tests):
the caller passes ``PromptCollection.from_dict`` and a libber manager in.

A scene's ``prompts.json`` (v2) holds named prompts (optionally libber-processed) and
compositions (named lists of prompt keys). A story entry picks which one is the image prompt
(``prompt_source``/``prompt_key``) and which is the video prompt
(``video_prompt_source``/``video_prompt_key``); ``auto`` reuses the image prompt.
"""
from __future__ import annotations

import re
from typing import Any, Callable

# Where a resolved video prompt came from.
SOURCE_CUSTOM = "custom"
SOURCE_PROMPT = "prompt"
SOURCE_COMPOSITION = "composition"
SOURCE_IMAGE_PROMPT = "image_prompt"   # video_prompt_source == "auto": fell back to the image prompt
SOURCE_EMPTY = "empty"

_LIBBER_TOKEN = re.compile(r"%[a-z0-9_]+%")


def _field(scene: Any, name: str, default: str = "") -> str:
    """Read a scene field from either a SceneInStory-like object or a plain dict."""
    if isinstance(scene, dict):
        value = scene.get(name, default)
    else:
        value = getattr(scene, name, default)
    return value if value is not None else default


def _find_libber(libber_manager: Any, name: str):
    """A loaded libber, loading it from disk when the manager can."""
    if libber_manager is None:
        return None
    ensure = getattr(libber_manager, "ensure_libber", None)
    libber = ensure(name) if callable(ensure) else None
    if libber is None:
        getter = getattr(libber_manager, "get_libber", None)
        libber = getter(name) if callable(getter) else None
    return libber


def resolve_scene_prompts(
    prompt_data_raw: dict | None,
    collection_from_dict: Callable[[dict], Any],
    libber_manager: Any = None,
) -> tuple[dict, dict]:
    """``(prompt_dict, composition_dict)`` for a scene's raw prompts.json content.

    v2 data: every prompt is libber-substituted when it names a libber, and compositions are
    composed through the collection. Legacy (v1) data is already a flat ``{key: text}`` dict.
    """
    if not prompt_data_raw:
        return {}, {}
    if prompt_data_raw.get("version") != 2:
        return dict(prompt_data_raw), {}

    collection = collection_from_dict(prompt_data_raw)
    prompt_dict: dict = {}
    for key, metadata in collection.prompts.items():
        value = metadata.value
        if metadata.processing_type == "libber" and metadata.libber_name:
            libber = _find_libber(libber_manager, metadata.libber_name)
            if libber:
                value = libber.substitute(value)
        prompt_dict[key] = value
    composition_dict = (
        collection.compose_prompts(collection.compositions, libber_manager)
        if collection.compositions else {}
    )
    return prompt_dict, composition_dict


def _pick(source: str, key: str, custom: str, prompt_dict: dict, composition_dict: dict) -> str:
    if source == SOURCE_CUSTOM:
        return custom
    if source == SOURCE_PROMPT and key:
        return prompt_dict.get(key, "")
    if source == SOURCE_COMPOSITION and key:
        return composition_dict.get(key, "")
    return ""


def resolve_image_prompt(scene: Any, prompt_dict: dict, composition_dict: dict) -> str:
    """The scene's image prompt (the story entry's ``prompt_source``/``prompt_key``)."""
    return _pick(
        _field(scene, "prompt_source"), _field(scene, "prompt_key"),
        _field(scene, "custom_prompt"), prompt_dict, composition_dict,
    )


def resolve_video_prompt(scene: Any, prompt_dict: dict, composition_dict: dict) -> tuple[str, str]:
    """``(text, source)`` for a story entry's video prompt.

    ``source`` is one of the SOURCE_* constants: custom / prompt / composition for an explicit
    video prompt, ``image_prompt`` when ``auto`` fell back to the image prompt, ``empty`` when
    nothing resolved.
    """
    vsource = _field(scene, "video_prompt_source", "auto")
    if vsource == "auto":
        text = resolve_image_prompt(scene, prompt_dict, composition_dict)
        return text, (SOURCE_IMAGE_PROMPT if text.strip() else SOURCE_EMPTY)
    text = _pick(
        vsource, _field(scene, "video_prompt_key"), _field(scene, "video_custom_prompt"),
        prompt_dict, composition_dict,
    )
    return text, (vsource if text.strip() else SOURCE_EMPTY)


def unresolved_tokens(text: str) -> list[str]:
    """``%name%`` libber references left in resolved text (a libber was missing or lacked the key)."""
    return sorted(set(_LIBBER_TOKEN.findall(text or "")))
