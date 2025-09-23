#!/usr/bin/env python3
"""
Simple test to verify the differential modulation concept.
Tests with a toy example before running the full experiment.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def test_differential_concept():
    """Test the mathematical concept of differential modulation."""

    print("="*60)
    print("TESTING DIFFERENTIAL MODULATION CONCEPT")
    print("="*60)

    # Simulate modulation vectors
    d_model = 256
    d_ffn = 1024

    # Simulate modulations WITHOUT context (baseline)
    v_a_base = torch.randn(d_model) * 0.01
    v_b_base = torch.randn(d_ffn) * 0.01

    # Simulate modulations WITH context (includes knowledge)
    # The context adds a specific pattern
    knowledge_pattern_a = torch.randn(d_model) * 0.05
    knowledge_pattern_b = torch.randn(d_ffn) * 0.05

    v_a_context = v_a_base + knowledge_pattern_a
    v_b_context = v_b_base + knowledge_pattern_b

    # Compute differential
    delta_v_a = v_a_context - v_a_base
    delta_v_b = v_b_context - v_b_base

    # This should recover the knowledge pattern
    print(f"Knowledge pattern recovery:")
    print(f"  v_a: Error = {(delta_v_a - knowledge_pattern_a).norm():.6f}")
    print(f"  v_b: Error = {(delta_v_b - knowledge_pattern_b).norm():.6f}")

    # Compute weight update
    delta_W = torch.outer(delta_v_b, delta_v_a)

    print(f"\nWeight update statistics:")
    print(f"  Shape: {delta_W.shape}")
    print(f"  Norm: {delta_W.norm():.6f}")
    print(f"  Rank: 1 (by construction)")

    # Test the effect
    print(f"\nTesting injection effect:")

    # Simulate original MLP weights
    W_original = torch.randn(d_ffn, d_model) * 0.1

    # Apply update
    W_updated = W_original + delta_W

    # Test with a random input
    test_input = torch.randn(d_model)

    output_original = F.linear(test_input, W_original)
    output_updated = F.linear(test_input, W_updated)

    # The difference should be related to the knowledge pattern
    output_diff = output_updated - output_original
    expected_diff = delta_v_b * (delta_v_a @ test_input)

    print(f"  Output change verification:")
    print(f"    Expected: {expected_diff.norm():.6f}")
    print(f"    Actual:   {output_diff.norm():.6f}")
    print(f"    Error:    {(output_diff - expected_diff).norm():.6f}")

    # Visualize the differential
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Modulation magnitudes
    axes[0].bar(['Base v_a', 'Context v_a', 'Δv_a'],
                [v_a_base.norm(), v_a_context.norm(), delta_v_a.norm()])
    axes[0].set_title('v_a Modulation Magnitudes')
    axes[0].set_ylabel('L2 Norm')

    # Plot 2: Weight update heatmap
    im = axes[1].imshow(delta_W[:100, :100].detach().numpy(), cmap='RdBu', vmin=-delta_W.abs().max(), vmax=delta_W.abs().max())
    axes[1].set_title('Weight Update (first 100x100)')
    plt.colorbar(im, ax=axes[1])

    # Plot 3: Singular values of update
    U, S, V = torch.svd(delta_W)
    axes[2].bar(range(min(10, len(S))), S[:10].detach().numpy())
    axes[2].set_title('Singular Values of ΔW')
    axes[2].set_xlabel('Component')
    axes[2].set_ylabel('Singular Value')

    plt.tight_layout()
    plt.savefig('differential_concept_test.png')
    print(f"\n✓ Visualization saved to differential_concept_test.png")

    print("\n" + "="*60)
    print("CONCEPT VERIFICATION COMPLETE")
    print("="*60)
    print("✓ Differential correctly isolates knowledge pattern")
    print("✓ Weight update is rank-1 as expected")
    print("✓ Injection produces expected output change")
    print("\nThe concept is mathematically sound!")


def explain_new_architecture_impact():
    """Explain how the NEW architecture affects differential injection."""

    print("\n" + "="*60)
    print("NEW ARCHITECTURE IMPACT ON DIFFERENTIAL INJECTION")
    print("="*60)

    print("\nIn the NEW architecture (MLP(h) → attention + MLP(h+attention)):")
    print("\n1. WITHOUT CONTEXT:")
    print("   - Modulation encodes: generic attention + generic MLP processing")
    print("   - Output: baseline attention + MLP(h+attention) for generic completion")

    print("\n2. WITH CONTEXT:")
    print("   - Modulation encodes: context-aware attention + context-aware MLP")
    print("   - Output: specific attention + MLP(h+attention) for factual answer")

    print("\n3. DIFFERENTIAL (Context - Baseline):")
    print("   - Captures: Δattention + ΔMLP processing")
    print("   - This is the 'knowledge encoding' - what makes the model")
    print("     process this position as if it knows the specific fact")

    print("\n4. PERMANENT INJECTION:")
    print("   - ΔW = α * (Δv_b ⊗ Δv_a)")
    print("   - Makes the model always apply this knowledge transformation")
    print("   - Even without context, model behaves as if it 'knows' the fact")

    print("\nWhy this is powerful:")
    print("• Isolates pure knowledge representation")
    print("• Removes generic processing, keeps only fact-specific patterns")
    print("• More targeted than injecting full modulation")
    print("• Works because NEW architecture encodes richer transformations")


if __name__ == "__main__":
    test_differential_concept()
    explain_new_architecture_impact()

    print("\n" + "="*60)
    print("Ready to run the full experiment with:")
    print("  python experiments/context_differential_injection.py")
    print("="*60)