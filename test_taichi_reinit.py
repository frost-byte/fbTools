#!/usr/bin/env python3
"""
Test if calling ti.init() multiple times causes issues
"""
import taichi as ti

print("Testing multiple ti.init() calls...")

try:
    print("\n1st call to ti.init(arch=ti.gpu)")
    ti.init(arch=ti.gpu)
    print("✓ Success")
    
    print("\n2nd call to ti.init(arch=ti.gpu) without reset")
    ti.init(arch=ti.gpu)
    print("✓ Success - no error!")
    
except Exception as e:
    print(f"✗ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Testing with ti.reset() between calls...")

try:
    ti.reset()
    print("\n1st call after reset")
    ti.init(arch=ti.gpu)
    print("✓ Success")
    
    ti.reset()
    print("\n2nd call after reset")
    ti.init(arch=ti.gpu)
    print("✓ Success")
    
except Exception as e:
    print(f"✗ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
