import json
import os
from .logging_utils import get_logger


logger = get_logger(__name__)

def save_json_file(json_path, data):
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logger.info("Saved JSON to '%s'", json_path)
    except Exception as e:
        logger.error("Error saving JSON to '%s': %s", json_path, e)

def load_prompt_json(prompt_json_path):
    """
    Load prompts from JSON file.
    Returns the raw data structure, preserving version and format.
    For v2 format, also includes legacy v1 keys for backward compatibility.
    """
    if not os.path.isfile(prompt_json_path):
        logger.warning("prompt_json_path '%s' is not a valid file", prompt_json_path)
        return {"girl_pos": "", "male_pos": ""}

    try:
        with open(prompt_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Detect version
        version = data.get("version", 1)
        
        logger.debug("load_prompt_json: Detected prompt JSON version: %s", version)
        
        if version == 2 and "prompts" in data:
            # V2 format - preserve the full structure, but also add legacy keys for backward compatibility
            output = data.copy()  # Keep version, prompts, compositions
            
            # Extract values from prompts dict for legacy compatibility
            prompts_data = data.get("prompts", {})
            for key in ["girl_pos", "male_pos", "wan_prompt", "wan_low_prompt", "four_image_prompt"]:
                if key in prompts_data:
                    prompt_entry = prompts_data[key]
                    if isinstance(prompt_entry, dict):
                        output[key] = prompt_entry.get("value", "")
                    elif isinstance(prompt_entry, str):
                        output[key] = prompt_entry
                else:
                    output[key] = ""
            
            # Also extract any additional custom prompts for legacy access
            # Skip keys that are already added or are structural keys (version, prompts, compositions)
            for key, prompt_entry in prompts_data.items():
                if key not in output and key not in ["prompts", "compositions", "version"]:
                    if isinstance(prompt_entry, dict):
                        output[key] = prompt_entry.get("value", "")
                    elif isinstance(prompt_entry, str):
                        output[key] = prompt_entry
            
            logger.debug(
                "load_prompt_json: Returning v2 data with %d prompts, %d compositions",
                len(output.get("prompts", {})),
                len(output.get("compositions", {})),
            )
            return output
        else:
            # V1 format - return as-is with defaults for missing keys
            output = data.copy()
            output.setdefault("girl_pos", "")
            output.setdefault("male_pos", "")
            output.setdefault("wan_prompt", "")
            output.setdefault("wan_low_prompt", "")
            output.setdefault("four_image_prompt", "")
            return output
            
    except Exception as e:
        logger.error("Error loading prompt JSON from '%s': %s", prompt_json_path, e)
        return {"girl_pos": "", "male_pos": "", "wan_prompt": "", "wan_low_prompt": "", "four_image_prompt": ""}

def load_json_file(json_path):
    if not os.path.isfile(json_path):
        logger.warning("json_path '%s' is not a valid file", json_path)
        return "{}"

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error("Error loading JSON from '%s': %s", json_path, e)
        return ""
