"""Scene Prompt Management nodes, extracted from extension.py (Plan 25).

ScenePromptManager + PromptComposer — a small, fully independent domain discovered during Plan 24's
exploration to be neither part of the 13-class "grab-bag" nor the composition-engine cluster. Their
`scene_info` input is a duck-typed "SCENE_INFO" generic wire type (no class import needed), so this
module has zero dependency on nodes/narrative/scene.py despite the shared "Scene" category.
"""
from __future__ import annotations

import json
import os

from comfy_api.latest import io

from ..libber import LibberStateManager
from ..shared import prefixed_node_id, default_scenes_dir, get_subdirectories
from folder_paths import get_output_directory
from ...prompt_models import PromptMetadata, PromptCollection
from ...utils.logging_utils import get_logger

logger = get_logger(__name__)


# ============================================================================
# SCENE PROMPT MANAGEMENT NODES
# ============================================================================

class ScenePromptManager(io.ComfyNode):
    """Manage prompts in a Scene's PromptCollection with an interactive table interface."""
    
    @classmethod
    def define_schema(cls):
        output_dir = get_output_directory()
        default_dir = os.path.join(output_dir, "scenes")
        if not os.path.exists(default_dir):
            os.makedirs(default_dir, exist_ok=True)
            os.makedirs(os.path.join(default_dir, "default_scene"), exist_ok=True)
        
        subdir_dict = get_subdirectories(default_dir)
        all_scenes = sorted(subdir_dict.keys()) if subdir_dict else ["default_scene"]
        
        # Find scenes with valid v2 prompts.json files
        valid_scenes = []
        for scene_name in all_scenes:
            scene_dir = os.path.join(default_dir, scene_name)
            prompts_path = os.path.join(scene_dir, "prompts.json")
            if os.path.exists(prompts_path):
                try:
                    # Check if it's a valid v2 format
                    with open(prompts_path, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, dict) and 'prompts' in data:
                            valid_scenes.append(scene_name)
                except:
                    pass
        
        # Use valid scenes if any exist, otherwise show all scenes
        default_options = valid_scenes if valid_scenes else all_scenes
        default_scene = default_options[0] if default_options else "default_scene"
        
        return io.Schema(
            node_id=prefixed_node_id("ScenePromptManager"),
            display_name="Scene Prompt Manager",
            category="🧊 frost-byte/Scene",
            inputs=[
                io.String.Input("scenes_dir", default=default_dir, tooltip="Directory containing pose subdirectories"),
                io.Combo.Input('scene_name', options=default_options, default=default_scene, tooltip="Select a scene to manage prompts"),
                io.String.Input(
                    id="collection_json",
                    display_name="collection_json",
                    default="",
                    multiline=True,
                    tooltip="Prompt collection JSON (auto-updated by UI table - normally don't edit manually)"
                ),
            ],
            outputs=[
                io.Custom("DICT").Output(
                    id="prompt_dict",
                    display_name="prompt_dict",
                    tooltip="Dictionary of individual prompts (raw or libber-processed)"
                ),
                io.Custom("DICT").Output(
                    id="comp_dict",
                    display_name="comp_dict",
                    tooltip="Dictionary of composed prompts by composition name"
                ),
                io.String.Output(
                    id="scene_name_out",
                    display_name="scene_name",
                    tooltip="Name of the managed scene"
                ),
                io.String.Output(
                    id="scene_dir",
                    display_name="scene_dir",
                    tooltip="Directory path of the managed scene"
                ),
                io.String.Output(
                    id="status",
                    display_name="status",
                    tooltip="Operation status"
                ),
            ],
            is_output_node=True,
        )
    
    @classmethod
    def execute(cls, scenes_dir="", scene_name="", collection_json=""):
        if not scenes_dir:
            scenes_dir = default_scenes_dir()
        
        if not scene_name:
            status = "✗ No scene selected"
            logger.warning("ScenePromptManager: %s", status)
            combined_ui = {"text": ["{}", "[]", status, "[]", "[]", "{}", "{}"]}
            return io.NodeOutput({}, {}, "", "", status, ui=combined_ui)
        
        scene_dir = os.path.join(scenes_dir, scene_name)
        
        if not os.path.isdir(scene_dir):
            status = f"✗ Scene directory not found: {scene_dir}"
            logger.error("ScenePromptManager: %s", status)
            combined_ui = {"text": ["{}", "[]", status, "[]", "[]", "{}", "{}"]}
            return io.NodeOutput({}, {}, scene_name, scene_dir, status, ui=combined_ui)
        
        # Load prompt collection from file or JSON
        prompt_json_path = os.path.join(scene_dir, "prompts.json")
        
        # Check if prompts.json exists
        if not os.path.exists(prompt_json_path) and not collection_json:
            status = f"⚠ Scene '{scene_name}' has no prompts.json file. Create prompts using the UI table."
            logger.warning("ScenePromptManager: %s", status)
            collection = PromptCollection()
            # Save empty collection to create the file
            try:
                with open(prompt_json_path, 'w', encoding='utf-8') as f:
                    json.dump(collection.to_dict(), f, indent=2, ensure_ascii=False)
                status += " (Created empty prompts.json)"
            except Exception as e:
                status += f" (Failed to create file: {e})"
        else:
            # Priority: collection_json (user edits) > prompts.json file
            if collection_json:
                try:
                    data = json.loads(collection_json)
                    collection = PromptCollection.from_dict(data)
                    logger.info(
                        "ScenePromptManager: Loaded collection from UI JSON with %d prompts",
                        len(collection.prompts),
                    )
                    
                    # Save to file
                    try:
                        with open(prompt_json_path, 'w', encoding='utf-8') as f:
                            json.dump(collection.to_dict(), f, indent=2, ensure_ascii=False)
                        status = f"✓ Saved {len(collection.prompts)} prompts to '{scene_name}'"
                        logger.error("ScenePromptManager: %s", status)
                    except Exception as e:
                        status = f"⚠ Loaded {len(collection.prompts)} prompts but failed to save: {e}"
                        logger.error("ScenePromptManager: %s", status)
                        
                except Exception as e:
                    # Fall back to file
                    status = f"✗ Error parsing UI JSON: {e}. Loading from file instead."
                    logger.error("ScenePromptManager: %s", status)
                    try:
                        collection = PromptCollection.load_from_json(prompt_json_path)
                    except Exception as e2:
                        status = f"✗ Failed to load from file: {e2}"
                        logger.error("ScenePromptManager: %s", status)
                        collection = PromptCollection()
            else:
                # Load from file
                try:
                    collection = PromptCollection.load_from_json(prompt_json_path)
                    
                    # Check if it's v2 format
                    if len(collection.prompts) == 0:
                        status = f"⚠ Scene '{scene_name}' has empty or v1 format prompts.json. Use UI to add prompts."
                    else:
                        status = f"✓ Loaded {len(collection.prompts)} prompts from '{scene_name}'"
                    
                    logger.info("ScenePromptManager: %s", status)
                except Exception as e:
                    status = f"✗ Error loading prompts.json: {e}"
                    logger.error("ScenePromptManager: %s", status)
                    collection = PromptCollection()
        
        # Prepare UI data
        collection_data = collection.to_dict()
        prompts_list = []
        for key, metadata in collection.prompts.items():
            prompts_list.append({
                "key": key,
                "value": metadata.value,
                "processing_type": metadata.processing_type,
                "libber_name": metadata.libber_name or "",
                "category": metadata.category or "",
            })
        
        # Get available libbers
        libber_manager = LibberStateManager.instance()
        available_libbers = ["none"] + list(libber_manager.libbers.keys())
        
        # Build prompt_dict (individual prompts processed)
        prompt_dict = {}
        for key, metadata in collection.prompts.items():
            value = metadata.value
            
            # Apply libber substitution if needed
            if metadata.processing_type == "libber" and metadata.libber_name and libber_manager:
                libber = libber_manager.ensure_libber(metadata.libber_name)
                if libber:
                    value = libber.substitute(value)
            
            prompt_dict[key] = value
        
        # Build comp_dict (compositions processed)
        comp_dict = {}
        if collection.compositions:
            comp_dict = collection.compose_prompts(collection.compositions, libber_manager)
        
        # Prepare compositions list for UI
        compositions_list = []
        for name, prompt_keys in collection.compositions.items():
            compositions_list.append({
                "name": name,
                "prompt_keys": prompt_keys,
                "preview": comp_dict.get(name, "")[:100] + ("..." if len(comp_dict.get(name, "")) > 100 else "")
            })
        
        combined_ui = {
            "text": [
                json.dumps(collection_data, indent=2),
                json.dumps(prompts_list),
                status,
                json.dumps(available_libbers),
                json.dumps(compositions_list),
                json.dumps(prompt_dict),
                json.dumps(comp_dict)
            ]
        }
        
        logger.info("ScenePromptManager: %s", status)
        return io.NodeOutput(prompt_dict, comp_dict, scene_name, scene_dir, status, ui=combined_ui)


class PromptComposer(io.ComfyNode):
    """Compose multiple output prompts from a PromptCollection with flexible slot assignment."""
    
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=prefixed_node_id("PromptComposer"),
            display_name="Prompt Composer",
            category="🧊 frost-byte/Scene",
            inputs=[
                io.Custom("SCENE_INFO").Input(
                    id="scene_info",
                    display_name="scene_info",
                    optional=True,
                    tooltip="Scene with prompt collection (from ScenePromptManager or SceneSelect)"
                ),
                io.String.Input(
                    id="composition_json",
                    display_name="composition_json",
                    default='{\n  "qwen_main": ["char1", "char2", "setting", "quality"]\n}',
                    multiline=True,
                    tooltip='Composition map: {"output_name": ["prompt_key1", "prompt_key2"]}. Example: {"main_prompt": ["char1", "setting"], "video_high": ["char1", "quality_high"]}'
                ),
            ],
            outputs=[
                io.Custom("DICT").Output(
                    id="prompt_dict",
                    display_name="prompt_dict",
                    tooltip="Dictionary of composed prompts by name"
                ),
                io.String.Output(
                    id="composition_json_out",
                    display_name="composition_json",
                    tooltip="Composition map for saving/loading"
                ),
                io.String.Output(
                    id="info",
                    display_name="info",
                    tooltip="Composition details"
                ),
            ],
            is_output_node=True,
        )
    
    @classmethod
    def execute(cls, scene_info=None, composition_json=""):
        # Get prompt collection
        collection = None
        if scene_info and scene_info.prompts:
            collection = scene_info.prompts
        
        if not collection:
            status = "✗ No prompt collection provided"
            logger.warning("PromptComposer: %s", status)
            return io.NodeOutput({}, "{}", status)
        
        # Parse composition map
        composition_map = {}
        if composition_json:
            try:
                composition_map = json.loads(composition_json)
            except Exception as e:
                logger.error("PromptComposer: Error parsing composition JSON: %s", e)
                composition_map = {}
        
        # Default composition if none provided
        if not composition_map:
            # Create default based on legacy prompt names
            available_keys = list(collection.prompts.keys())
            composition_map = {
                "prompt_a": available_keys[:2] if len(available_keys) >= 2 else available_keys,
            }
        
        # Get libber manager for processing
        libber_manager = LibberStateManager.instance()
        
        # Compose prompts
        prompt_dict = collection.compose_prompts(composition_map, libber_manager)
        
        # Generate info
        info_lines = [f"✓ Composed {len(prompt_dict)} output prompts:"]
        for name, value in prompt_dict.items():
            preview = value[:60] + "..." if len(value) > 60 else value
            prompt_count = len(composition_map.get(name, []))
            info_lines.append(f"  {name}: {prompt_count} prompts → \"{preview}\"")
        
        info = "\n".join(info_lines)
        
        # Prepare UI data
        prompts_list = []
        for key, metadata in collection.prompts.items():
            prompts_list.append({
                "key": key,
                "value": metadata.value,
                "processing_type": metadata.processing_type,
                "libber_name": metadata.libber_name or "",
            })
        
        combined_ui = {
            "text": [
                json.dumps(composition_map),
                json.dumps(prompts_list),
                json.dumps(prompt_dict),
                info
            ]
        }
        
        logger.info("PromptComposer: %s", info)
        return io.NodeOutput(prompt_dict, json.dumps(composition_map, indent=2), info, ui=combined_ui)
