#!/usr/bin/env python3

import torch
import os

print("=== GPU Test Script ===")

# Basic CUDA info
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}")
        print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
        print(f"  Compute capability: {props.major}.{props.minor}")

    # Test basic CUDA operations
    print("\n=== Testing CUDA Operations ===")
    try:
        # Create tensors on GPU
        a = torch.randn(100, 100).cuda()
        b = torch.randn(100, 100).cuda()
        c = torch.matmul(a, b)
        print(f"✓ Matrix multiplication on GPU successful")
        print(f"  Result device: {c.device}")

        # Test memory
        allocated = torch.cuda.memory_allocated() / 1e6
        reserved = torch.cuda.memory_reserved() / 1e6
        print(f"  GPU memory - Allocated: {allocated:.1f} MB, Reserved: {reserved:.1f} MB")

    except Exception as e:
        print(f"✗ CUDA operation failed: {e}")

else:
    print("✗ CUDA not available")

# Test environment
print(f"\n=== Environment ===")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print(f"PATH: {os.environ.get('PATH', 'Not set')[:100]}...")

# Test a simple model
print(f"\n=== Testing Model on GPU ===")
if torch.cuda.is_available():
    try:
        import torch.nn as nn

        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        ).cuda()

        x = torch.randn(5, 10).cuda()
        y = model(x)

        print(f"✓ Simple model test passed")
        print(f"  Input device: {x.device}")
        print(f"  Output device: {y.device}")

        # Check all parameters are on GPU
        gpu_params = sum(1 for p in model.parameters() if p.device.type == 'cuda')
        total_params = sum(1 for p in model.parameters())
        print(f"  Parameters on GPU: {gpu_params}/{total_params}")

    except Exception as e:
        print(f"✗ Model test failed: {e}")

print("\n=== Test Complete ===")