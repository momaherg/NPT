#!/usr/bin/env python3
"""
Test script for selective NPT layer loading functionality.

This script demonstrates how to:
1. Load a checkpoint with multiple NPT layers
2. Choose which layers to use in NPT mode
3. Toggle layer modes at runtime
"""

import sys
import torch
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.npt import NPTLlamaModel, NPTConfig
from transformers import AutoTokenizer
import argparse


def test_selective_loading():
    """Test the selective NPT layer loading feature."""
    
    print("\n" + "="*60)
    print("Testing Selective NPT Layer Loading")
    print("="*60)
    
    # Create a simple demo model
    from transformers import LlamaConfig
    config = LlamaConfig(
        hidden_size=256,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        vocab_size=128256,
    )
    
    model = NPTLlamaModel(config)
    
    # Convert multiple layers to NPT
    print("\n1. Converting layers 1 and 3 to NPT...")
    npt_config = NPTConfig(
        layers_to_convert=[1, 3],
        np_rank=32,
        np_init_scale=0.001,
        single_layer_mode=False
    )
    model.convert_to_npt(npt_config)
    
    # Check which layers are NPT
    npt_layers = sorted(model.npt_layers.keys())
    print(f"   NPT layers: {npt_layers}")
    
    # Test mode switching
    print("\n2. Testing mode switching...")
    
    # Check initial modes
    print("   Initial modes:")
    for idx in npt_layers:
        layer = model.model.layers[idx]
        mode = "NPT" if layer.use_npt else "standard"
        print(f"     Layer {idx}: {mode}")
    
    # Switch layer 1 to standard mode
    print("\n   Switching layer 1 to standard mode...")
    model.model.layers[1].set_npt_mode(False)
    
    # Check modes after switch
    print("   Modes after switch:")
    for idx in npt_layers:
        layer = model.model.layers[idx]
        mode = "NPT" if layer.use_npt else "standard"
        print(f"     Layer {idx}: {mode}")
    
    # Test forward pass with mixed modes
    print("\n3. Testing forward pass with mixed modes...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    text = "Hello, world!"
    inputs = tokenizer(text, return_tensors="pt")
    
    # Forward pass
    with torch.no_grad():
        # First with mixed modes
        outputs_mixed = model(**inputs)
        
        # Switch all to NPT mode
        for idx in npt_layers:
            model.model.layers[idx].set_npt_mode(True)
        outputs_all_npt = model(**inputs)
        
        # Switch all to standard mode
        for idx in npt_layers:
            model.model.layers[idx].set_npt_mode(False)
        outputs_all_standard = model(**inputs)
    
    print(f"   Output shape: {outputs_mixed.logits.shape}")
    print(f"   Mixed modes output norm: {outputs_mixed.logits.norm():.4f}")
    print(f"   All NPT output norm: {outputs_all_npt.logits.norm():.4f}")
    print(f"   All standard output norm: {outputs_all_standard.logits.norm():.4f}")
    
    # Compare outputs
    diff_mixed_npt = (outputs_mixed.logits - outputs_all_npt.logits).abs().mean()
    diff_mixed_standard = (outputs_mixed.logits - outputs_all_standard.logits).abs().mean()
    diff_npt_standard = (outputs_all_npt.logits - outputs_all_standard.logits).abs().mean()
    
    print(f"\n   Output differences:")
    print(f"     Mixed vs All NPT: {diff_mixed_npt:.6f}")
    print(f"     Mixed vs All Standard: {diff_mixed_standard:.6f}")
    print(f"     All NPT vs All Standard: {diff_npt_standard:.6f}")
    
    print("\n" + "="*60)
    print("Test completed successfully!")
    print("="*60)
    
    return model


def demonstrate_usage():
    """Demonstrate the command-line usage."""
    
    print("\n" + "="*60)
    print("Example Command-Line Usage:")
    print("="*60)
    
    print("\n# Load checkpoint with all NPT layers active (default):")
    print("python scripts/interactive_knowledge_injection.py \\")
    print("  --checkpoint experiments/sequential_checkpoint \\")
    print("  --model_name meta-llama/Llama-3.2-1B")
    
    print("\n# Load checkpoint but only use specific layers as NPT:")
    print("python scripts/interactive_knowledge_injection.py \\")
    print("  --checkpoint experiments/sequential_checkpoint \\")
    print("  --model_name meta-llama/Llama-3.2-1B \\")
    print("  --use_npt_layers '15,31'")
    
    print("\n# Load checkpoint but keep all layers in standard mode:")
    print("python scripts/interactive_knowledge_injection.py \\")
    print("  --checkpoint experiments/sequential_checkpoint \\")
    print("  --model_name meta-llama/Llama-3.2-1B \\")
    print("  --use_npt_layers 'none'")
    
    print("\n# Interactive commands:")
    print("  layers     - Show all NPT layers and their current modes")
    print("  modes      - Show which layers are in NPT vs standard mode")
    print("  mode 15    - Toggle layer 15 between NPT and standard")
    print("  mode 15 npt     - Set layer 15 to NPT mode")
    print("  mode 15 standard - Set layer 15 to standard mode")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test selective NPT layer loading")
    parser.add_argument("--demo", action="store_true", help="Show usage examples")
    args = parser.parse_args()
    
    if args.demo:
        demonstrate_usage()
    else:
        test_selective_loading()