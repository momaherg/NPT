#!/usr/bin/env python3
"""Test script for architecture-aware knowledge injection."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.interactive_knowledge_injection import (
    NPTArchitectureVersion,
    detect_architecture_version,
    KnowledgeInjector
)
import torch

def test_architecture_detection():
    """Test architecture version detection."""
    print("Testing architecture detection...")

    # Test V1 detection (default)
    v1_weights = {"layer_15_np.W_down": torch.randn(2048, 256)}
    version = detect_architecture_version(v1_weights)
    assert version == NPTArchitectureVersion.V1
    print(f"  ✓ V1 detection: {version.value}")

    # Test V2 detection via marker
    v2_weights = {
        "layer_15_np.W_down": torch.randn(2048, 256),
        "architecture_v2_marker": torch.tensor(1.0)
    }
    version = detect_architecture_version(v2_weights)
    assert version == NPTArchitectureVersion.V2
    print(f"  ✓ V2 detection: {version.value}")

    print("Architecture detection tests passed!\n")

def test_injection_strength_scaling():
    """Test that injection strength is scaled appropriately."""
    print("Testing injection strength scaling...")

    # Mock model class
    class MockModel:
        def __init__(self):
            self.npt_layers = {}
            self.model = type('obj', (), {'layers': []})()

        def to(self, device):
            return self

        def eval(self):
            pass

        def count_parameters(self):
            return {'total': 1000000, 'npt': 50000}

    # Mock tokenizer
    class MockTokenizer:
        pad_token_id = 0
        eos_token_id = 1

    # Test V1 architecture - default strength should be 1.0
    model = MockModel()
    tokenizer = MockTokenizer()

    injector_v1 = KnowledgeInjector(
        model=model,
        tokenizer=tokenizer,
        injection_strength=1.0,
        device="cpu",
        architecture_version=NPTArchitectureVersion.V1
    )
    assert injector_v1.injection_strength == 1.0
    print(f"  ✓ V1 default strength: {injector_v1.injection_strength}")

    # Test V2 architecture - default strength should be adjusted to 0.3
    injector_v2 = KnowledgeInjector(
        model=model,
        tokenizer=tokenizer,
        injection_strength=1.0,  # Default value
        device="cpu",
        architecture_version=NPTArchitectureVersion.V2
    )
    assert injector_v2.injection_strength == 0.3
    print(f"  ✓ V2 adjusted strength: {injector_v2.injection_strength}")

    # Test custom strength is preserved
    injector_custom = KnowledgeInjector(
        model=model,
        tokenizer=tokenizer,
        injection_strength=0.5,
        device="cpu",
        architecture_version=NPTArchitectureVersion.V2
    )
    assert injector_custom.injection_strength == 0.5
    print(f"  ✓ Custom strength preserved: {injector_custom.injection_strength}")

    print("Injection strength scaling tests passed!\n")

def main():
    """Run all tests."""
    print("="*60)
    print("Architecture-Aware Knowledge Injection Tests")
    print("="*60)
    print()

    test_architecture_detection()
    test_injection_strength_scaling()

    print("="*60)
    print("All tests passed successfully!")
    print("="*60)
    print()

    print("Summary:")
    print("  • Architecture detection working correctly")
    print("  • V1 uses standard injection strength (0.5-2.0)")
    print("  • V2 uses conservative injection strength (0.1-0.5)")
    print("  • Backward compatibility maintained")
    print()

    print("Key Differences:")
    print("  V1 (OLD): MLP_modulated(h) = MLP(h+attention)")
    print("           → Injection makes MLP process as if attention added")
    print()
    print("  V2 (NEW): MLP_modulated(h) = attention + MLP(h+attention)")
    print("           → Injection makes MLP output attention pattern + transformed result")

if __name__ == "__main__":
    main()