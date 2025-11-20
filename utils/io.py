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
    if not os.path.isfile(prompt_json_path):
        print(f"fbTools: prompt_json_path '{prompt_json_path}' is not a valid file")
        return {"girl_pos": "", "male_pos": ""}

    output = {}
    try:
        with open(prompt_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        output["girl_pos"] = data.get("girl_pos", "")
        output["male_pos"] = data.get("male_pos", "")
        return output
    except Exception as e:
        print(f"fbTools: Error loading prompt JSON from '{prompt_json_path}': {e}")
        return {"girl_pos": "", "male_pos": ""}

def load_json_file(json_path):
    if not os.path.isfile(json_path):
        print(f"fbTools: json_path '{json_path}' is not a valid file")
        return "{}"

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = f.read()
        return data
    except Exception as e:
        print(f"fbTools: Error loading JSON from '{json_path}': {e}")
        return ""
