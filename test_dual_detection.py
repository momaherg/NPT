#!/usr/bin/env python3
"""
Quick test to verify dual modulation detection and weight extraction.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.npt import NPTLlamaModel, NPTConfig
from transformers import AutoTokenizer

def test_dual_modulation():
    print("Testing dual modulation detection...")

    # Create a simple model
    model = NPTLlamaModel.from_pretrained("meta-llama/Llama-3.2-1B")

    # Convert layer 15 to NPT with dual modulation
    npt_config = NPTConfig(
        layers_to_convert=[15],
        np_rank=256,
        np_init_scale=0.001,
        num_ranks=4,
        single_layer_mode=False,
        dual_modulation=True
    )
    model.convert_to_npt(npt_config)

    print(f"✓ Created NPT layer with dual modulation")

    # Check NPComponent structure
    npt_layer = model.model.layers[15]
    np_component = npt_layer.np_component

    print(f"  dual_modulation flag: {np_component.dual_modulation}")

    # Check for dual weights
    if hasattr(np_component, 'W_down_gate'):
        print("  ✓ Has W_down_gate (dual modulation weights)")
    else:
        print("  ✗ Missing W_down_gate")

    if hasattr(np_component, 'W_down_up'):
        print("  ✓ Has W_down_up (dual modulation weights)")
    else:
        print("  ✗ Missing W_down_up")

    # Test forward pass
    print("\nTesting forward pass...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizer.pad_token = tokenizer.eos_token

    input_text = "Test"
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs.input_ids

    # Hook to capture NPComponent output
    captured_output = None
    def hook_fn(module, input, output):
        nonlocal captured_output
        captured_output = output

    handle = np_component.register_forward_hook(hook_fn)

    try:
        model.set_npt_mode(True)
        with torch.no_grad():
            # Simple forward through embeddings
            hidden_states = model.model.embed_tokens(input_ids)

            # Process through layers (simplified)
            for i in range(16):  # Process all layers
                layer = model.model.layers[i]
                if i < 15:
                    # Standard forward for non-NPT layers
                    layer_out = layer(hidden_states, use_cache=False, output_attentions=False)
                    hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out
                else:
                    # NPT layer - this should trigger the hook
                    layer_out = layer(hidden_states, use_cache=False, output_attentions=False)
                    hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out
    finally:
        handle.remove()

    # Check captured output
    if captured_output is not None:
        print("  ✓ NPComponent output captured")

        # Check structure
        if isinstance(captured_output, tuple) and len(captured_output) == 2:
            if isinstance(captured_output[0], tuple) and isinstance(captured_output[1], tuple):
                print("  ✓ Dual modulation output structure detected: ((v_a_gate, v_b_gate), (v_a_up, v_b_up))")

                (v_a_gate, v_b_gate), (v_a_up, v_b_up) = captured_output
                print(f"    Gate shapes: v_a={v_a_gate.shape}, v_b={v_b_gate.shape}")
                print(f"    Up shapes: v_a={v_a_up.shape}, v_b={v_b_up.shape}")
            else:
                print("  ✗ Single modulation output structure: (v_a, v_b)")
        else:
            print(f"  ✗ Unexpected output structure: {type(captured_output)}")
    else:
        print("  ✗ No output captured")

    print("\n✓ Test complete!")

if __name__ == "__main__":
    test_dual_modulation()