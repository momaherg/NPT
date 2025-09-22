#!/usr/bin/env python3
"""
Quick test script to verify dual modulation implementation.
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.npt import NPTLlamaModel, NPTConfig
from transformers import AutoTokenizer

def test_dual_modulation():
    """Test that dual modulation works correctly."""

    print("Testing Dual Modulation Architecture...")
    print("="*60)

    # Create a small config for testing
    from transformers import LlamaConfig
    config = LlamaConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=1000,
        max_position_embeddings=128
    )
    config._attn_implementation = "eager"

    # Create model with dual modulation
    model = NPTLlamaModel(config)

    # Convert layers 2-3 to NPT with dual modulation
    npt_config = NPTConfig(
        layers_to_convert=[2, 3],
        np_rank=32,
        np_init_scale=0.01,
        num_ranks=2,
        dual_modulation=True
    )

    print(f"Converting layers {npt_config.layers_to_convert} to dual modulation NPT")
    model.convert_to_npt(npt_config)
    model.freeze_base_parameters()

    # Count parameters
    param_counts = model.count_parameters()
    print(f"\nParameter counts:")
    print(f"  Total: {param_counts['total']:,}")
    print(f"  Base (frozen): {param_counts['base']:,}")
    print(f"  NPT (trainable): {param_counts['npt']:,}")
    print(f"  NPT ratio: {param_counts['npt_ratio']:.2%}")

    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))

    # Test NPT mode
    model.set_npt_mode(True)
    with torch.no_grad():
        outputs_npt = model(input_ids)
        logits_npt = outputs_npt.logits
    print(f"  NPT mode output shape: {logits_npt.shape}")

    # Test standard mode
    model.set_npt_mode(False)
    with torch.no_grad():
        outputs_std = model(input_ids)
        logits_std = outputs_std.logits
    print(f"  Standard mode output shape: {logits_std.shape}")

    # Compare outputs
    diff = (logits_npt - logits_std).abs().mean().item()
    print(f"\nMean absolute difference between NPT and standard: {diff:.6f}")

    # Test modulation structure
    print("\nTesting modulation structure...")
    layer = model.model.layers[2]

    # Generate some dummy attention output
    dummy_attn = torch.randn(batch_size, seq_len, config.hidden_size)
    modulation = layer.np_component(dummy_attn)

    if isinstance(modulation[0], tuple):
        print("  ✓ Dual modulation detected")
        (v_a_gate, v_b_gate), (v_a_up, v_b_up) = modulation
        print(f"  Gate modulation shapes: v_a={v_a_gate.shape}, v_b={v_b_gate.shape}")
        print(f"  Up modulation shapes: v_a={v_a_up.shape}, v_b={v_b_up.shape}")

        # Check rank-k structure
        if v_a_gate.dim() == 4:
            print(f"  ✓ Rank-{npt_config.num_ranks} modulation confirmed")
        else:
            print(f"  Single rank modulation")
    else:
        print("  ✗ Single modulation (not dual)")

    print("\n" + "="*60)
    print("Dual Modulation Test Complete!")

    return model

if __name__ == "__main__":
    model = test_dual_modulation()