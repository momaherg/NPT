#!/usr/bin/env python3
"""
Test script for different loss modes in single-layer NPT training.

Tests:
1. Direct supervision loss (recommended)
2. Guided supervision loss (with attention hint)
3. Staged loss (old behavior)
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.single_layer_losses import (
    DirectSupervisionLoss,
    GuidedSupervisionLoss,
    SingleLayerEquivalenceLoss
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_outputs(batch_size=2, seq_len=10, requires_grad=True):
    """Create test outputs for loss computation."""
    hidden_size = 256
    intermediate_size = 1024
    vocab_size = 1000
    
    outputs = {
        'mlp_modulated': torch.randn(batch_size, seq_len, hidden_size),
        'attention_output': torch.randn(batch_size, seq_len, hidden_size),
        'original_mlp_with_attention': torch.randn(batch_size, seq_len, hidden_size),
        'v_a': torch.randn(batch_size, seq_len, hidden_size, requires_grad=requires_grad),
        'v_b': torch.randn(batch_size, seq_len, intermediate_size, requires_grad=requires_grad),
        'npt_final': torch.randn(batch_size, seq_len, vocab_size),
        'original_final': torch.randn(batch_size, seq_len, vocab_size),
        'hidden_states': torch.randn(batch_size, seq_len, hidden_size),
    }
    
    return outputs


def test_direct_supervision_loss():
    """Test DirectSupervisionLoss."""
    logger.info("\n" + "="*80)
    logger.info("Testing DirectSupervisionLoss (Recommended)")
    logger.info("="*80)
    
    loss_fn = DirectSupervisionLoss(
        direct_mlp_weight=1.0,
        fidelity_weight=0.1,
        regularization_weight=0.001
    )
    
    outputs = create_test_outputs()
    loss_output = loss_fn(outputs, current_step=0)
    
    assert loss_output.total_loss.requires_grad, "Loss should require gradients"
    assert not torch.isnan(loss_output.total_loss), "Loss should not be NaN"
    
    # Check that attention encoding loss is zero (not used)
    assert loss_output.metrics['attention_encoding_loss'] == 0.0, "Attention encoding should be 0"
    
    # Check that stage is 0 (no stages)
    assert loss_output.metrics['stage'] == 0, "Should have no stages"
    
    logger.info(f"✓ Direct supervision loss test passed")
    logger.info(f"  Total loss: {loss_output.total_loss.item():.4f}")
    logger.info(f"  Direct MLP loss: {loss_output.direct_mlp_loss.item():.4f}")
    logger.info(f"  Fidelity loss: {loss_output.fidelity_loss.item():.4f}")
    logger.info(f"  Regularization loss: {loss_output.regularization_loss.item():.4f}")
    logger.info(f"  v_a attention similarity (monitoring only): {loss_output.metrics['v_a_attention_similarity']:.3f}")


def test_guided_supervision_loss():
    """Test GuidedSupervisionLoss."""
    logger.info("\n" + "="*80)
    logger.info("Testing GuidedSupervisionLoss (With Attention Hint)")
    logger.info("="*80)
    
    loss_fn = GuidedSupervisionLoss(
        direct_mlp_weight=100.0,
        attention_encoding_weight=1.0,
        fidelity_weight=10.0,
        regularization_weight=0.001
    )
    
    outputs = create_test_outputs()
    loss_output = loss_fn(outputs, current_step=0)
    
    assert loss_output.total_loss.requires_grad, "Loss should require gradients"
    assert not torch.isnan(loss_output.total_loss), "Loss should not be NaN"
    
    # Check that attention encoding loss is non-zero
    assert loss_output.attention_encoding_loss.item() > 0, "Should have attention encoding loss"
    
    # Check that stage is 0 (no stages)
    assert loss_output.metrics['stage'] == 0, "Should have no stages"
    
    logger.info(f"✓ Guided supervision loss test passed")
    logger.info(f"  Total loss: {loss_output.total_loss.item():.4f}")
    logger.info(f"  Direct MLP loss: {loss_output.direct_mlp_loss.item():.4f}")
    logger.info(f"  Attention encoding loss: {loss_output.attention_encoding_loss.item():.4f}")
    logger.info(f"  Fidelity loss: {loss_output.fidelity_loss.item():.4f}")
    logger.info(f"  v_a attention similarity: {loss_output.metrics['v_a_attention_similarity']:.3f}")


def test_staged_loss():
    """Test SingleLayerEquivalenceLoss (staged mode)."""
    logger.info("\n" + "="*80)
    logger.info("Testing SingleLayerEquivalenceLoss (Staged - Old Behavior)")
    logger.info("="*80)
    
    loss_fn = SingleLayerEquivalenceLoss(
        direct_mlp_weight=10.0,
        attention_encoding_weight=5.0,
        fidelity_weight=1.0,
        regularization_weight=0.01,
        stage1_steps=1000
    )
    
    outputs = create_test_outputs()
    
    # Test stage 1 (step < 1000)
    loss_output_stage1 = loss_fn(outputs, current_step=500)
    assert loss_output_stage1.metrics['stage'] == 1, "Should be in stage 1"
    
    logger.info(f"✓ Stage 1 (step 500):")
    logger.info(f"  Total loss: {loss_output_stage1.total_loss.item():.4f}")
    logger.info(f"  Direct MLP loss: {loss_output_stage1.direct_mlp_loss.item():.4f}")
    logger.info(f"  Attention encoding loss: {loss_output_stage1.attention_encoding_loss.item():.4f}")
    
    # Test stage 2 (step >= 1000)
    loss_output_stage2 = loss_fn(outputs, current_step=1500)
    assert loss_output_stage2.metrics['stage'] == 2, "Should be in stage 2"
    
    logger.info(f"✓ Stage 2 (step 1500):")
    logger.info(f"  Total loss: {loss_output_stage2.total_loss.item():.4f}")
    logger.info(f"  Direct MLP loss: {loss_output_stage2.direct_mlp_loss.item():.4f}")
    logger.info(f"  Attention encoding loss: {loss_output_stage2.attention_encoding_loss.item():.4f}")
    
    # Test stage 3 (step >= 3000)
    loss_output_stage3 = loss_fn(outputs, current_step=3500)
    assert loss_output_stage3.metrics['stage'] == 2, "Should be in stage 2/3"
    
    logger.info(f"✓ Stage 3 (step 3500):")
    logger.info(f"  Total loss: {loss_output_stage3.total_loss.item():.4f}")


def test_loss_mode_comparison():
    """Compare loss values across different modes."""
    logger.info("\n" + "="*80)
    logger.info("Loss Mode Comparison")
    logger.info("="*80)
    
    outputs = create_test_outputs()
    
    # Direct supervision
    direct_loss = DirectSupervisionLoss()
    direct_output = direct_loss(outputs)
    
    # Guided supervision
    guided_loss = GuidedSupervisionLoss()
    guided_output = guided_loss(outputs)
    
    # Staged (at step 0)
    staged_loss = SingleLayerEquivalenceLoss(stage1_steps=1000)
    staged_output = staged_loss(outputs, current_step=0)
    
    logger.info("Loss comparison at step 0:")
    logger.info(f"  Direct mode total loss: {direct_output.total_loss.item():.4f}")
    logger.info(f"  Guided mode total loss: {guided_output.total_loss.item():.4f}")
    logger.info(f"  Staged mode total loss: {staged_output.total_loss.item():.4f}")
    
    logger.info("\nKey differences:")
    logger.info("  Direct: Focuses purely on direct MLP supervision")
    logger.info("  Guided: Adds soft attention guidance")
    logger.info("  Staged: Transitions through training phases")


def test_rank_k_compatibility():
    """Test that loss modes work with rank-k vectors."""
    logger.info("\n" + "="*80)
    logger.info("Testing Rank-k Compatibility")
    logger.info("="*80)
    
    batch_size, seq_len = 2, 10
    hidden_size = 256
    intermediate_size = 1024
    num_ranks = 4
    
    # Create rank-k outputs
    outputs = {
        'mlp_modulated': torch.randn(batch_size, seq_len, hidden_size),
        'attention_output': torch.randn(batch_size, seq_len, hidden_size),
        'original_mlp_with_attention': torch.randn(batch_size, seq_len, hidden_size),
        'v_a': torch.randn(batch_size, seq_len, num_ranks, hidden_size, requires_grad=True),
        'v_b': torch.randn(batch_size, seq_len, num_ranks, intermediate_size, requires_grad=True),
        'npt_final': torch.randn(batch_size, seq_len, 1000),
        'original_final': torch.randn(batch_size, seq_len, 1000),
        'hidden_states': torch.randn(batch_size, seq_len, hidden_size),
    }
    
    # Test all loss modes with rank-k
    for name, loss_fn in [
        ("Direct", DirectSupervisionLoss()),
        ("Guided", GuidedSupervisionLoss()),
        ("Staged", SingleLayerEquivalenceLoss())
    ]:
        loss_output = loss_fn(outputs)
        assert not torch.isnan(loss_output.total_loss), f"{name} loss should not be NaN with rank-k"
        logger.info(f"✓ {name} mode works with rank-{num_ranks}")


def main():
    """Run all tests."""
    logger.info("\n" + "="*80)
    logger.info("LOSS MODE TEST SUITE")
    logger.info("="*80)
    
    try:
        test_direct_supervision_loss()
        test_guided_supervision_loss()
        test_staged_loss()
        test_loss_mode_comparison()
        test_rank_k_compatibility()
        
        logger.info("\n" + "="*80)
        logger.info("ALL TESTS PASSED ✓")
        logger.info("="*80)
        
        logger.info("\nRecommended usage:")
        logger.info("  python scripts/train_single_layer_npt.py --loss_mode direct")
        logger.info("\nThis focuses training on the actual target transformation")
        logger.info("without wasting steps on forced attention encoding.")
        
    except Exception as e:
        logger.error(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()