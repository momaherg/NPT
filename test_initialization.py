#!/usr/bin/env python3
"""Test NPComponent initialization for single-layer mode."""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.npt.np_component import NPComponent

# Test single-layer mode initialization
print("Testing NPComponent initialization in single-layer mode...")

# Create component
comp = NPComponent(
    d_model=512,
    d_ffn=2048,
    rank=64,
    init_scale=0.001,
    single_layer_mode=True
)

print(f"\nComponent configuration:")
print(f"  Effective rank: {comp.rank}")
print(f"  Init scale: {comp.init_scale}")
print(f"  Single layer mode: {comp.single_layer_mode}")

print(f"\nWeight statistics:")
print(f"  W_down shape: {comp.W_down.shape}")
print(f"  W_down mean: {comp.W_down.data.mean():.6f}")
print(f"  W_down std: {comp.W_down.data.std():.6f}")

print(f"\n  W_a_up shape: {comp.W_a_up.shape}")
print(f"  W_a_up mean: {comp.W_a_up.data.mean():.6f}")
print(f"  W_a_up std: {comp.W_a_up.data.std():.6f}")
print(f"  W_a_up diagonal mean: {comp.W_a_up.data.diagonal(0, -2, -1).mean():.6f}")

print(f"\n  W_b_up shape: {comp.W_b_up.shape}")
print(f"  W_b_up mean: {comp.W_b_up.data.mean():.6f}")
print(f"  W_b_up std: {comp.W_b_up.data.std():.6f}")

# Test forward pass
print(f"\nTesting forward pass...")
x = torch.randn(2, 10, 512)
v_a, v_b = comp(x)

print(f"  Input shape: {x.shape}")
print(f"  v_a shape: {v_a.shape}")
print(f"  v_a mean: {v_a.mean():.6f}")
print(f"  v_a std: {v_a.std():.6f}")

print(f"\n  v_b shape: {v_b.shape}")
print(f"  v_b mean: {v_b.mean():.6f}")
print(f"  v_b std: {v_b.std():.6f}")

# Test if v_a preserves input structure (should have identity-like behavior initially)
print(f"\nTesting identity preservation:")
# Create a simple input
simple_input = torch.zeros(1, 1, 512)
simple_input[0, 0, 0] = 1.0  # Set first element to 1

v_a_simple, v_b_simple = comp(simple_input)
print(f"  Input: first element = 1.0, rest = 0.0")
print(f"  v_a first 5 elements: {v_a_simple[0, 0, :5].tolist()}")
print(f"  v_a max element: {v_a_simple.max():.6f}")
print(f"  v_b first 5 elements: {v_b_simple[0, 0, :5].tolist()}")

print("\n✅ Initialization test complete!")