#!/usr/bin/env python3
"""
Inspect a GGUF file — prints metadata and tensor names.
Usage: python inspect_gguf.py /path/to/model.gguf [--keys 20]
"""

import argparse
import json
from gguf import GGUFReader

def main():
    parser = argparse.ArgumentParser(description="Inspect a GGUF file.")
    parser.add_argument("path", help="Path to the .gguf file")
    parser.add_argument(
        "--keys", "-k",
        type=int,
        default=20,
        help="Number of tensor key names to print (default: 20, 0 = all)"
    )
    args = parser.parse_args()

    reader = GGUFReader(args.path)

    print("=== Metadata ===")
    for field in reader.fields.values():
        try:
            if len(field.data) == 1:
                print(f"  {field.name}: {field.data[0]}")
            else:
                print(f"  {field.name}: {list(field.data)[:8]}{'...' if len(field.data) > 8 else ''}")
        except Exception:
            print(f"  {field.name}: (unreadable)")

    print(f"\n=== Tensors ({len(reader.tensors)} total) ===")
    tensors_to_show = reader.tensors if args.keys == 0 else reader.tensors[:args.keys]
    for tensor in tensors_to_show:
        print(f"  {tensor.name}  shape={tensor.shape}  dtype={tensor.tensor_type.name}")

    if args.keys and len(reader.tensors) > args.keys:
        print(f"... ({len(reader.tensors) - args.keys} more, use --keys 0 to show all)")

if __name__ == "__main__":
    main()
