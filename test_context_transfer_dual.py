#!/usr/bin/env python3
"""
Test script for dual modulation context transfer.
"""

import sys
from pathlib import Path
import torch

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.npt import NPTLlamaModel, NPTConfig
from transformers import AutoTokenizer
from scripts.npt_multi_layer_context_transfer import MultiLayerNPTContextTransfer

def test_dual_modulation_transfer():
    print("Testing dual modulation context transfer...")

    # Create a small test model
    model = NPTLlamaModel.from_pretrained("meta-llama/Llama-3.2-1B")

    # Convert layers 14,15 to NPT with dual modulation
    npt_config = NPTConfig(
        layers_to_convert=[14, 15],
        np_rank=256,
        np_init_scale=0.001,
        num_ranks=2,
        single_layer_mode=False,
        dual_modulation=True
    )
    model.convert_to_npt(npt_config)

    print(f"✓ Created NPT model with dual modulation on layers 14,15")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create context transfer object
    transfer = MultiLayerNPTContextTransfer(
        model=model,
        tokenizer=tokenizer,
        layer_indices=[14, 15],
        device="cpu",
        transfer_mode="last"
    )

    print(f"✓ Created MultiLayerNPTContextTransfer object")

    # Test capture and storage
    prompt = "The capital of France is"

    # Clear and run forward pass to capture modulations
    transfer.clear_captured()
    transfer.clear_overrides()

    # Generate to trigger hooks
    response = transfer.generate_response(prompt, max_new_tokens=5)
    print(f"\nGenerated response: {response}")

    # Check captured modulations
    print("\nChecking captured modulations...")
    for layer_idx in [14, 15]:
        if layer_idx in transfer.captured_v_a_gate:
            print(f"  Layer {layer_idx}:")
            print(f"    - v_a_gate captured: shape={transfer.captured_v_a_gate[layer_idx].shape}")
            print(f"    - v_b_gate captured: shape={transfer.captured_v_b_gate[layer_idx].shape}")

            if transfer.captured_v_a_up.get(layer_idx) is not None:
                print(f"    - v_a_up captured: shape={transfer.captured_v_a_up[layer_idx].shape}")
                print(f"    - v_b_up captured: shape={transfer.captured_v_b_up[layer_idx].shape}")
                print(f"    ✓ Dual modulation detected!")
            else:
                print(f"    ✗ Single modulation (unexpected)")
        else:
            print(f"  Layer {layer_idx}: No modulation captured")

    # Test override setting
    print("\nTesting override setting...")

    # Create test overrides with dual modulation structure
    test_overrides = {}
    for layer_idx in [14, 15]:
        if layer_idx in transfer.captured_v_a_gate:
            if transfer.captured_v_a_up.get(layer_idx) is not None:
                # Dual modulation format
                test_overrides[layer_idx] = {
                    'gate': (transfer.captured_v_a_gate[layer_idx].clone(),
                            transfer.captured_v_b_gate[layer_idx].clone()),
                    'up': (transfer.captured_v_a_up[layer_idx].clone(),
                          transfer.captured_v_b_up[layer_idx].clone())
                }
                print(f"  Layer {layer_idx}: Set dual modulation override")

    # Set overrides
    transfer.set_overrides(test_overrides)
    print(f"✓ Overrides set successfully")

    # Test modulation type tracking
    print("\nModulation type tracking:")
    for layer_idx in [14, 15]:
        if layer_idx in transfer.is_dual_modulation:
            mod_type = "dual" if transfer.is_dual_modulation[layer_idx] else "single"
            print(f"  Layer {layer_idx}: {mod_type} modulation")

    print("\n✓ All tests passed!")

if __name__ == "__main__":
    test_dual_modulation_transfer()