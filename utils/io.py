import json
import os

def save_json_file(json_path, data):
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"fbTools: Saved JSON to '{json_path}'")
    except Exception as e:
        print(f"fbTools: Error saving JSON to '{json_path}': {e}")

def load_prompt_json(prompt_json_path):
    """
    Load prompts from JSON file with automatic v1->v2 migration.
    Returns dict in v1 format for backward compatibility.
    """
    if not os.path.isfile(prompt_json_path):
        print(f"fbTools: prompt_json_path '{prompt_json_path}' is not a valid file")
        return {"girl_pos": "", "male_pos": ""}

    output = {}
    try:
        with open(prompt_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Detect version
        version = data.get("version", 1)
        
        if version == 2 and "prompts" in data:
            # V2 format - extract values from prompts dict
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
            
            # Also extract any additional custom prompts
            for key, prompt_entry in prompts_data.items():
                if key not in output:
                    if isinstance(prompt_entry, dict):
                        output[key] = prompt_entry.get("value", "")
                    elif isinstance(prompt_entry, str):
                        output[key] = prompt_entry
        else:
            # V1 format - direct access
            output["girl_pos"] = data.get("girl_pos", "")
            output["male_pos"] = data.get("male_pos", "")
            output["wan_prompt"] = data.get("wan_prompt", "")
            output["wan_low_prompt"] = data.get("wan_low_prompt", "")
            output["four_image_prompt"] = data.get("four_image_prompt", "")
        
        return output
    except Exception as e:
        print(f"fbTools: Error loading prompt JSON from '{prompt_json_path}': {e}")
        return {"girl_pos": "", "male_pos": "", "wan_prompt": "", "wan_low_prompt": "", "four_image_prompt": ""}

def load_json_file(json_path):
    if not os.path.isfile(json_path):
        print(f"fbTools: json_path '{json_path}' is not a valid file")
        return "{}"

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"fbTools: Error loading JSON from '{json_path}': {e}")
        return ""
