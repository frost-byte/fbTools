#!/mnt/comfy_ssd/venvs/comfy-preflight/bin/python3

"""
Inspect a safetensors file — prints metadata and first N tensor names.
Usage: python inspect_safetensors.py /path/to/model.safetensors [--keys 50]
"""

import argparse
import json
from safetensors import safe_open

def main():
    parser = argparse.ArgumentParser(description="Inspect a safetensors file.")
    parser.add_argument("path", help="Path to the .safetensors file")
    parser.add_argument(
        "--keys", "-k",
        type=int,
        default=20,
        help="Number of tensor key names to print (default: 20, 0 = all)"
    )
    args = parser.parse_args()

    with safe_open(args.path, framework="pt") as f:
        metadata = f.metadata()
        all_keys = list(f.keys())

    print("=== Metadata ===")
    if metadata:
        print(json.dumps(metadata, indent=2))
    else:
        print("(none)")

    print(f"\n=== Tensor Keys ({len(all_keys)} total) ===")
    keys_to_show = all_keys if args.keys == 0 else all_keys[:args.keys]
    for key in keys_to_show:
        print(key)

    if args.keys and len(all_keys) > args.keys:
        print(f"... ({len(all_keys) - args.keys} more, use --keys 0 to show all)")

if __name__ == "__main__":
    main()
