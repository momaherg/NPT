#!/usr/bin/env python3
"""Minimal test for checkpoint resume"""

import sys
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.npt import NPTLlamaModel, NPTConfig

# Test loading NPT weights
checkpoint_path = Path("experiments/npt_new_arch_top3_layers/checkpoints/checkpoint-14000/")
print(f"Testing resume from: {checkpoint_path}")

# Load NPT weights
npt_weights_path = checkpoint_path / "npt_weights.pt"
if npt_weights_path.exists():
    print(f"Loading NPT weights...")
    try:
        # Create a model first
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B")
        config._attn_implementation = "eager"

        model = NPTLlamaModel(config)

        # Convert layers to NPT
        npt_config = NPTConfig(
            layers_to_convert=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
            np_rank=256,
            np_init_scale=0.001,
            num_ranks=4,
            init_strategy="improved"
        )
        model.convert_to_npt(npt_config)

        # Load weights
        model.load_npt_weights(npt_weights_path)
        print("Successfully loaded NPT weights!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

# Load training state
training_state_path = checkpoint_path / "training_state.pt"
if training_state_path.exists():
    print(f"\nLoading training state...")
    try:
        state = torch.load(training_state_path, map_location='cpu', weights_only=False)
        print(f"Global step: {state['global_step']}")
        print(f"Batch count: {state.get('batch_count', 'N/A')}")
        print("Successfully loaded training state!")
    except Exception as e:
        print(f"Error: {e}")

print("\nResume test complete!")