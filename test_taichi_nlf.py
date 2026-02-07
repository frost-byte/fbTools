#!/usr/bin/env python3
"""
Test script to diagnose why taichi fails in NLF rendering context
"""
import sys
import torch

print("=" * 80)
print("TAICHI NLF DIAGNOSTICS")
print("=" * 80)

# Test 1: Basic taichi import and initialization
print("\n[Test 1] Basic taichi import and initialization")
print("-" * 80)
try:
    import taichi as ti
    print(f"✓ Taichi imported successfully")
    print(f"  Version: {ti.__version__}")
    
    # Test initialization with different backends
    for device_name, device_arch in [("gpu", ti.gpu), ("cuda", ti.cuda), ("cpu", ti.cpu)]:
        try:
            print(f"\n  Testing {device_name} backend...")
            ti.init(arch=device_arch)
            print(f"  ✓ Taichi initialized on {device_name}")
            # Reset for next test
            ti.reset()
        except Exception as e:
            print(f"  ✗ Failed on {device_name}: {type(e).__name__}: {e}")
except Exception as e:
    print(f"✗ Failed to import or initialize taichi: {type(e).__name__}: {e}")
    sys.exit(1)

# Test 2: Check if PyTorch is using CUDA
print("\n[Test 2] PyTorch CUDA status")
print("-" * 80)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA device memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Test 3: Try to replicate NLF rendering initialization
print("\n[Test 3] Replicating NLF initialization pattern")
print("-" * 80)
try:
    # Reset taichi first
    ti.reset()
    
    # This is the exact pattern used in ComfyUI-SCAIL-Pose/nodes.py
    render_device = "gpu"
    render_backend = "taichi"
    
    device_map = {
        "cpu": ti.cpu,
        "gpu": ti.gpu,
        "opengl": ti.opengl,
        "cuda": ti.cuda,
        "vulkan": ti.vulkan,
        "metal": ti.metal,
    }
    
    print(f"Attempting ti.init(arch={device_map[render_device]})...")
    ti.init(arch=device_map.get(render_device.lower()))
    print(f"✓ Successfully initialized taichi with {render_device} backend")
    
    # Try a simple kernel to verify it actually works
    print("\nTesting simple taichi kernel...")
    n = 320
    pixels = ti.field(dtype=float, shape=(n, n))
    
    @ti.kernel
    def fill_test():
        for i, j in pixels:
            pixels[i, j] = i * 0.01 + j * 0.01
    
    fill_test()
    print("✓ Taichi kernel executed successfully")
    
except Exception as e:
    print(f"✗ Failed: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Check for conflicts with PyTorch
print("\n[Test 4] Testing PyTorch + Taichi interaction")
print("-" * 80)
try:
    ti.reset()
    
    # Allocate PyTorch tensor on CUDA
    if torch.cuda.is_available():
        print("Creating PyTorch CUDA tensor...")
        tensor = torch.randn(1000, 1000, device='cuda')
        print(f"✓ PyTorch tensor allocated: {tensor.shape}")
    
    # Now try to initialize taichi
    print("Initializing taichi after PyTorch CUDA allocation...")
    ti.init(arch=ti.gpu)
    print("✓ Taichi initialized successfully after PyTorch CUDA")
    
    # Try kernel
    test_field = ti.field(dtype=float, shape=(100, 100))
    
    @ti.kernel
    def test_kernel():
        for i, j in test_field:
            test_field[i, j] = 1.0
    
    test_kernel()
    print("✓ Taichi kernel works alongside PyTorch CUDA")
    
except Exception as e:
    print(f"✗ Conflict detected: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Check environment variables
print("\n[Test 5] Environment variables")
print("-" * 80)
import os
env_vars = ['CUDA_VISIBLE_DEVICES', 'TI_ARCH', 'TI_ENABLE_CUDA', 'TI_USE_UNIFIED_MEMORY']
for var in env_vars:
    value = os.environ.get(var)
    print(f"{var}: {value if value else '(not set)'}")

print("\n" + "=" * 80)
print("DIAGNOSTICS COMPLETE")
print("=" * 80)
