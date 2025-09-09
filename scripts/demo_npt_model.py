"""
Demonstration script for the NPT Model (Stage 3).
Shows selective layer conversion and model functionality.
"""

import torch
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.npt import NPTLlamaModel, NPTConfig
from transformers import LlamaConfig


def demo_npt_model():
    """Demonstrate NPT Model functionality."""
    
    print("=" * 80)
    print("NPT Model Stage 3 Demo - Hybrid Model with Selective Layer Conversion")
    print("=" * 80)
    
    # Configuration for a small model
    config = LlamaConfig(
        hidden_size=256,
        intermediate_size=1024,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=4,
        vocab_size=1000,
    )
    config._attn_implementation = "eager"
    
    print(f"\nBase Model Configuration:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Intermediate size: {config.intermediate_size}")
    print(f"  Number of layers: {config.num_hidden_layers}")
    print(f"  Vocabulary size: {config.vocab_size}")
    
    # Create base model
    model = NPTLlamaModel(config)
    base_params = model.count_parameters()['total']
    print(f"\nBase model parameters: {base_params:,}")
    
    # Test different conversion strategies
    print("\n" + "=" * 80)
    print("Testing Different Conversion Strategies")
    print("=" * 80)
    
    # Strategy 1: Convert upper half of layers
    print("\n" + "-" * 60)
    print("Strategy 1: Convert Upper Half (layers 4-7)")
    print("-" * 60)
    
    npt_config_upper = NPTConfig(
        convert_range=(4, 8),
        np_rank=32,
        np_init_scale=0.01
    )
    
    model.convert_to_npt(npt_config_upper)
    layer_info = model.get_layer_info()
    
    print(f"Converted {layer_info['npt_layers']}/{layer_info['total_layers']} layers")
    print(f"NPT layer indices: {layer_info['npt_layer_indices']}")
    print(f"Standard layer indices: {layer_info['standard_layer_indices']}")
    
    param_counts = model.count_parameters()
    print(f"\nParameter breakdown:")
    print(f"  Base parameters: {param_counts['base']:,}")
    print(f"  NPT parameters: {param_counts['npt']:,}")
    print(f"  Total parameters: {param_counts['total']:,}")
    print(f"  NPT ratio: {param_counts['npt_ratio']:.2%}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"\nTesting forward pass with input shape: {input_ids.shape}")
    
    # NPT mode
    model.set_npt_mode(True)
    model.eval()
    with torch.no_grad():
        outputs_npt = model(input_ids)
    print(f"✓ NPT mode forward pass successful")
    print(f"  Output shape: {outputs_npt.logits.shape}")
    
    # Standard mode
    model.set_npt_mode(False)
    with torch.no_grad():
        outputs_std = model(input_ids)
    print(f"✓ Standard mode forward pass successful")
    print(f"  Output shape: {outputs_std.logits.shape}")
    
    # Compare outputs
    diff = torch.abs(outputs_npt.logits - outputs_std.logits).mean()
    print(f"\nMean absolute difference between modes: {diff:.6f}")
    
    # Reset model for next strategy
    model.reset_to_standard()
    
    # Strategy 2: Convert specific layers
    print("\n" + "-" * 60)
    print("Strategy 2: Convert Specific Layers (2, 4, 6)")
    print("-" * 60)
    
    npt_config_specific = NPTConfig(
        layers_to_convert=[2, 4, 6],
        np_rank=32,
        np_init_scale=0.01
    )
    
    model.convert_to_npt(npt_config_specific)
    layer_info = model.get_layer_info()
    
    print(f"Converted {layer_info['npt_layers']}/{layer_info['total_layers']} layers")
    print(f"NPT layer indices: {layer_info['npt_layer_indices']}")
    print(f"Layer types: {layer_info['layer_types']}")
    
    param_counts = model.count_parameters()
    print(f"\nParameter increase: {(param_counts['total'] - base_params) / base_params:.2%}")
    
    # Strategy 3: Convert all layers
    model.reset_to_standard()
    
    print("\n" + "-" * 60)
    print("Strategy 3: Convert All Layers")
    print("-" * 60)
    
    npt_config_all = NPTConfig(
        convert_all=True,
        np_rank=32,
        np_init_scale=0.01
    )
    
    model.convert_to_npt(npt_config_all)
    layer_info = model.get_layer_info()
    
    print(f"Converted {layer_info['npt_layers']}/{layer_info['total_layers']} layers")
    param_counts = model.count_parameters()
    print(f"Parameter increase: {(param_counts['total'] - base_params) / base_params:.2%}")
    
    # Test parameter management
    print("\n" + "=" * 80)
    print("Parameter Management Demo")
    print("=" * 80)
    
    # Freeze base parameters
    model.freeze_base_parameters()
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    
    print(f"\nAfter freezing base parameters:")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {frozen_params:,}")
    print(f"  Verification: trainable == NPT params? {trainable_params == param_counts['npt']}")
    
    # Get parameter groups
    param_groups = model.get_npt_parameter_groups()
    print(f"\nNPT parameter groups: {len(param_groups)}")
    for group_name, params in list(param_groups.items())[:3]:  # Show first 3
        print(f"  {group_name}: {len(params)} parameters")
    
    # Test gradient flow
    print("\n" + "=" * 80)
    print("Gradient Flow Test")
    print("=" * 80)
    
    # Create a simple loss
    outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss
    loss.backward()
    
    print(f"Loss value: {loss.item():.4f}")
    
    # Check gradients
    npt_grads_count = 0
    base_grads_count = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            if 'np_component' in name:
                npt_grads_count += 1
            else:
                base_grads_count += 1
    
    print(f"\nGradient statistics:")
    print(f"  NP parameters with gradients: {npt_grads_count}")
    print(f"  Base parameters with gradients: {base_grads_count} (should be 0)")
    
    # Save/Load test
    print("\n" + "=" * 80)
    print("Save/Load NPT Weights")
    print("=" * 80)
    
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        save_path = f.name
    
    try:
        # Save NPT weights
        model.save_npt_weights(save_path)
        file_size = os.path.getsize(save_path) / 1024  # KB
        print(f"NPT weights saved to temporary file")
        print(f"  File size: {file_size:.2f} KB")
        
        # Modify weights
        for param in model.get_npt_parameters():
            param.data.fill_(0.0)
        
        # Load weights back
        model.load_npt_weights(save_path)
        print(f"NPT weights loaded successfully")
        
        # Verify weights were restored
        non_zero_params = sum(1 for p in model.get_npt_parameters() if torch.any(p != 0))
        print(f"  Non-zero NP parameters after loading: {non_zero_params}")
        
    finally:
        os.unlink(save_path)
    
    print("\n" + "=" * 80)
    print("Stage 3 Implementation Complete!")
    print("✓ Selective layer conversion working")
    print("✓ Multiple conversion strategies supported")
    print("✓ Parameter management functional")
    print("✓ Gradient flow control working")
    print("✓ Save/load functionality operational")
    print("=" * 80)


if __name__ == "__main__":
    demo_npt_model()