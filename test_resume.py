#!/usr/bin/env python3
"""Test checkpoint resume functionality"""

import torch
from pathlib import Path
import sys

checkpoint_path = "experiments/npt_new_arch_top3_layers/checkpoints/checkpoint-16000/"

# Check files
checkpoint_path = Path(checkpoint_path)
print(f"Checking checkpoint at: {checkpoint_path}")

files = list(checkpoint_path.glob("*"))
print(f"Files found: {[f.name for f in files]}")

# Test loading NPT weights
npt_weights_path = checkpoint_path / "npt_weights.pt"
if npt_weights_path.exists():
    print(f"\nLoading NPT weights from {npt_weights_path}")
    try:
        npt_state = torch.load(npt_weights_path, map_location='cpu')
        print(f"NPT weights loaded successfully")
        print(f"Keys in state dict: {len(npt_state.keys())}")
        # Show sample keys
        sample_keys = list(npt_state.keys())[:5]
        print(f"Sample keys: {sample_keys}")
    except Exception as e:
        print(f"Error loading NPT weights: {e}")

# Test loading training state
training_state_path = checkpoint_path / "training_state.pt"
if training_state_path.exists():
    print(f"\nLoading training state from {training_state_path}")
    try:
        training_state = torch.load(training_state_path, map_location='cpu')
        print(f"Training state loaded successfully")
        print(f"Global step: {training_state.get('global_step', 'Not found')}")
        print(f"Batch count: {training_state.get('batch_count', 'Not found')}")
        print(f"Keys in state: {list(training_state.keys())}")
    except Exception as e:
        print(f"Error loading training state: {e}")

print("\nCheckpoint validation complete!")