"""Shared custom io types for the composition-engine cluster.

SourceProfileIOType is defined/consumed by the Source Profile layer; CastIOType,
H3RefplanType, CompositionIOType and SceneInstanceIOType are defined/consumed by the
composition-assembly layer -- but each layer also consumes at least one type the OTHER
layer owns (e.g. SceneCastBuild needs SourceProfileIOType; SourceProfileClipPrompt needs
CastIOType and H3RefplanType). Splitting those two layers into their own files (a future
plan) can't use the simple one-directional "move the leaf, re-export it back" trick every
other domain in this refactor series has used, because neither layer is a leaf relative to
the other. This module is the neutral home both can import from without a cycle -- pure
code motion out of extension.py, no behavior change.
"""
from __future__ import annotations

from comfy_api.latest import io

# ── Custom type: SOURCE_PROFILE ───────────────────────────────────────────────

SOURCE_PROFILE_TYPE = "SOURCE_PROFILE"


@io.comfytype(io_type=SOURCE_PROFILE_TYPE)
class SourceProfileIOType:
    """Carries a full source profile (media ref + subjects list) between nodes."""
    Type = object  # SourceProfileRegistry profile dict

    class Input(io.Input):
        def __init__(self, name: str, **kwargs):
            super().__init__(name, **kwargs)

    class Output(io.Output):
        def __init__(self, name: str = "source_profile", **kwargs):
            super().__init__(name, **kwargs)


# ── Custom type: SCENE_INSTANCE ──────────────────────────────────────────────

SCENE_INSTANCE_TYPE = "SCENE_INSTANCE"


@io.comfytype(io_type=SCENE_INSTANCE_TYPE)
class SceneInstanceIOType:
    """Carries a composed scene dict between SceneCompose → PromptAssemble nodes."""
    Type = object  # dict with template, slot_assignments, dialogue, outfit_overrides

    class Input(io.Input):
        def __init__(self, name: str, **kwargs):
            super().__init__(name, **kwargs)

    class Output(io.Output):
        def __init__(self, name: str = "scene_instance", **kwargs):
            super().__init__(name, **kwargs)


# ── Custom type: SCENE_CAST ───────────────────────────────────────────────────

SCENE_CAST_TYPE = "SCENE_CAST"


@io.comfytype(io_type=SCENE_CAST_TYPE)
class CastIOType:
    """Carries a scene cast dict between SceneCastLoad → PromptCompositionLoader nodes."""
    Type = object  # dict: {id, name, entries: [{subject_id, bundle_id, visual_mode, use_audio}], ...}

    class Input(io.Input):
        def __init__(self, name: str, **kwargs):
            super().__init__(name, **kwargs)


H3_REFPLAN_TYPE = "FBTOOLS_H3_REFPLAN"


@io.comfytype(io_type=H3_REFPLAN_TYPE)
class H3RefplanType:
    """Ordered reference descriptor bundle from PromptCompositionLoader → CompositionToH3Conditioning.

    Carries descriptors (paths + params) for all references in native node order:
      images → [soundtrack_audio + video] pairs → standalone_audios
    Terminal node decodes media and delegates to MiniMaxH3ReferenceToVideo.execute.
    """
    Type = object  # dict: {prompt, model_type, ref_image_size, references: [...]}

    class Input(io.Input):
        def __init__(self, name: str, **kwargs):
            super().__init__(name, **kwargs)

    class Output(io.Output):
        def __init__(self, name: str = "scene_cast", **kwargs):
            super().__init__(name, **kwargs)


# ── Custom type + loader: PROMPT_COMPOSITION ─────────────────────────────────

COMPOSITION_TYPE = "PROMPT_COMPOSITION"


@io.comfytype(io_type=COMPOSITION_TYPE)
class CompositionIOType:
    """Carries a full saved Prompt Composition dict between nodes."""
    Type = object  # composition dict (see utils/prompt_compositions.py schema)

    class Input(io.Input):
        def __init__(self, name: str, **kwargs):
            super().__init__(name, **kwargs)

    class Output(io.Output):
        def __init__(self, name: str = "prompt_composition", **kwargs):
            super().__init__(name, **kwargs)
