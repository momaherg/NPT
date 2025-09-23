#!/usr/bin/env python3
"""
Test script for dual modulation knowledge injection.
"""

import sys
from pathlib import Path
import torch

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.npt import NPTLlamaModel, NPTConfig
from transformers import AutoTokenizer
from scripts.interactive_knowledge_injection import KnowledgeInjector

def main():
    print("Loading checkpoint...")
    checkpoint_path = Path("experiments/npt_dual_gate_up_2layers/checkpoints/checkpoint-6000")

    # Load model and checkpoint
    model = NPTLlamaModel.from_pretrained("meta-llama/Llama-3.2-1B")

    # Load NPT weights
    npt_weights = torch.load(checkpoint_path / "npt_weights.pt", map_location='cpu')

    # Convert layers 14,15 to NPT with dual modulation
    npt_config = NPTConfig(
        layers_to_convert=[14, 15],
        np_rank=256,
        np_init_scale=0.001,
        num_ranks=4,
        single_layer_mode=False,
        dual_modulation=True
    )
    model.convert_to_npt(npt_config)
    model.load_npt_weights(npt_weights)

    print(f"Converted layers 14,15 to NPT with dual modulation")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create injector
    injector = KnowledgeInjector(
        model=model,
        tokenizer=tokenizer,
        layer_idx=15,
        injection_strength=1.0,
        device="cpu"
    )

    print("\nTesting knowledge injection...")

    # Test 1: Ask before injection
    print("\n1. Before injection:")
    response = injector.generate_response("The capital of France is")
    print(f"   Response: {response}")

    # Test 2: Inject knowledge
    print("\n2. Injecting knowledge: 'The capital of France is Atlantis'")
    try:
        result = injector.extract_delta_weights("The capital of France is Atlantis", "last")

        if result['type'] == 'dual':
            print("   ✓ Dual modulation detected!")
            print(f"   Gate modulation norms: v_a={result['metadata']['v_a_gate_norm']:.4f}, v_b={result['metadata']['v_b_gate_norm']:.4f}")
            print(f"   Up modulation norms: v_a={result['metadata']['v_a_up_norm']:.4f}, v_b={result['metadata']['v_b_up_norm']:.4f}")

            # Apply the injection
            info = injector.inject_knowledge("The capital of France is Atlantis", position="last")
            print(f"   Injection complete!")
        else:
            print("   ⚠ Single modulation detected (unexpected)")
    except Exception as e:
        print(f"   Error during injection: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: Ask after injection
    print("\n3. After injection:")
    response = injector.generate_response("The capital of France is")
    print(f"   Response: {response}")

    print("\n✓ Test complete!")

if __name__ == "__main__":
    main()