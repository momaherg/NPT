#!/usr/bin/env python3
"""
Test script for rank-k NPT implementation.

This script tests:
1. Backward compatibility (rank-1 still works)
2. Rank-k forward pass
3. Parameter count comparison
4. Loss computation with rank-k
5. Training step with rank-k
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.npt import NPTLlamaModel, NPTConfig
from src.npt.np_component import NPComponent
from src.npt.npt_decoder_layer import NPTDecoderLayer
from src.training.single_layer_losses import SingleLayerEquivalenceLoss
from transformers import LlamaConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_backward_compatibility():
    """Test that rank-1 (num_ranks=1) works exactly as before."""
    logger.info("\n" + "="*80)
    logger.info("Testing Backward Compatibility (num_ranks=1)")
    logger.info("="*80)
    
    # Create a small config for testing
    config = LlamaConfig(
        hidden_size=256,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        vocab_size=1000,
    )
    config._attn_implementation = "eager"
    
    # Test NPComponent with num_ranks=1
    np_component = NPComponent(
        d_model=256,
        d_ffn=1024,
        rank=64,
        init_scale=0.01,
        single_layer_mode=False,
        num_ranks=1  # Default value
    )
    
    # Check that weights are Parameters, not ParameterLists
    assert isinstance(np_component.W_down, nn.Parameter), "W_down should be Parameter for rank-1"
    assert isinstance(np_component.W_a_up, nn.Parameter), "W_a_up should be Parameter for rank-1"
    assert isinstance(np_component.W_b_up, nn.Parameter), "W_b_up should be Parameter for rank-1"
    
    # Test forward pass
    batch_size, seq_len = 2, 10
    attn_output = torch.randn(batch_size, seq_len, 256)
    v_a, v_b = np_component(attn_output)
    
    assert v_a.shape == (batch_size, seq_len, 256), f"v_a shape mismatch: {v_a.shape}"
    assert v_b.shape == (batch_size, seq_len, 1024), f"v_b shape mismatch: {v_b.shape}"
    assert v_a.dim() == 3, "v_a should be 3D for rank-1"
    
    logger.info("✓ Backward compatibility test passed")
    
    # Test full model conversion
    model = NPTLlamaModel(config)
    npt_config = NPTConfig(
        layers_to_convert=[0],
        np_rank=64,
        np_init_scale=0.01,
        single_layer_mode=True,
        num_ranks=1
    )
    model.convert_to_npt(npt_config)
    
    # Test forward pass
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    outputs = model(input_ids)
    assert outputs.logits.shape == (batch_size, seq_len, 1000)
    
    logger.info("✓ Full model backward compatibility test passed")


def test_rank_k_forward():
    """Test rank-k forward pass with num_ranks > 1."""
    logger.info("\n" + "="*80)
    logger.info("Testing Rank-k Forward Pass (num_ranks=4)")
    logger.info("="*80)
    
    # Test NPComponent with rank-4
    np_component = NPComponent(
        d_model=256,
        d_ffn=1024,
        rank=64,
        init_scale=0.01,
        single_layer_mode=True,  # Test with single layer mode
        num_ranks=4  # Rank-4
    )
    
    # Check that weights are ParameterLists
    assert isinstance(np_component.W_down, nn.ParameterList), "W_down should be ParameterList for rank-k"
    assert len(np_component.W_down) == 4, "Should have 4 W_down matrices"
    
    # Check rank adjustment for single layer mode
    expected_rank = max(64, 256 // 4)  # Should be 64
    assert np_component.rank == expected_rank, f"Rank should be {expected_rank}, got {np_component.rank}"
    
    # Test forward pass
    batch_size, seq_len = 2, 10
    attn_output = torch.randn(batch_size, seq_len, 256)
    v_a, v_b = np_component(attn_output)
    
    assert v_a.shape == (batch_size, seq_len, 4, 256), f"v_a shape mismatch: {v_a.shape}"
    assert v_b.shape == (batch_size, seq_len, 4, 1024), f"v_b shape mismatch: {v_b.shape}"
    assert v_a.dim() == 4, "v_a should be 4D for rank-k"
    
    logger.info(f"✓ Rank-4 NPComponent forward pass successful")
    logger.info(f"  v_a shape: {v_a.shape}")
    logger.info(f"  v_b shape: {v_b.shape}")


def test_rank_k_decoder_layer():
    """Test NPTDecoderLayer with rank-k modulation."""
    logger.info("\n" + "="*80)
    logger.info("Testing NPTDecoderLayer with Rank-k")
    logger.info("="*80)
    
    config = LlamaConfig(
        hidden_size=256,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        vocab_size=1000,
    )
    config._attn_implementation = "eager"  # Set attention implementation
    config.np_rank = 64
    config.np_init_scale = 0.01
    config.single_layer_mode = False
    config.num_ranks = 2  # Rank-2
    
    layer = NPTDecoderLayer(config, layer_idx=0)
    
    batch_size, seq_len = 2, 10
    hidden_states = torch.randn(batch_size, seq_len, 256)
    
    # Test forward pass
    layer.set_npt_mode(True)
    output = layer(hidden_states)
    
    if isinstance(output, tuple):
        output = output[0]
    
    assert output.shape == (batch_size, seq_len, 256), f"Output shape mismatch: {output.shape}"
    
    logger.info("✓ NPTDecoderLayer rank-k forward pass successful")


def test_rank_k_loss():
    """Test loss computation with rank-k vectors."""
    logger.info("\n" + "="*80)
    logger.info("Testing Loss Computation with Rank-k")
    logger.info("="*80)
    
    batch_size, seq_len = 2, 10
    hidden_size = 256
    intermediate_size = 1024
    num_ranks = 3
    
    # Create test outputs with rank-k vectors (with requires_grad for v_a and v_b)
    outputs = {
        'mlp_modulated': torch.randn(batch_size, seq_len, hidden_size),
        'attention_output': torch.randn(batch_size, seq_len, hidden_size),
        'original_mlp_with_attention': torch.randn(batch_size, seq_len, hidden_size),
        'v_a': torch.randn(batch_size, seq_len, num_ranks, hidden_size, requires_grad=True),  # Rank-k
        'v_b': torch.randn(batch_size, seq_len, num_ranks, intermediate_size, requires_grad=True),  # Rank-k
        'npt_final': torch.randn(batch_size, seq_len, 1000),
        'original_final': torch.randn(batch_size, seq_len, 1000),
        'hidden_states': torch.randn(batch_size, seq_len, hidden_size),
    }
    
    loss_fn = SingleLayerEquivalenceLoss(
        direct_mlp_weight=10.0,
        attention_encoding_weight=5.0,
        fidelity_weight=1.0,
        regularization_weight=0.01,
        stage1_steps=1000,
    )
    
    # Test loss computation
    loss_output = loss_fn(outputs, current_step=500)
    
    assert loss_output.total_loss.requires_grad, "Loss should require gradients"
    assert not torch.isnan(loss_output.total_loss), "Loss should not be NaN"
    
    logger.info(f"✓ Rank-k loss computation successful")
    logger.info(f"  Total loss: {loss_output.total_loss.item():.4f}")
    logger.info(f"  Direct MLP loss: {loss_output.direct_mlp_loss.item():.4f}")
    logger.info(f"  Attention encoding loss: {loss_output.attention_encoding_loss.item():.4f}")


def test_parameter_comparison():
    """Compare parameter counts for different rank configurations."""
    logger.info("\n" + "="*80)
    logger.info("Parameter Count Comparison")
    logger.info("="*80)
    
    d_model = 2048
    d_ffn = 8192
    
    configs = [
        (1, 256),  # rank-1, high rank
        (2, 128),  # rank-2, medium rank each
        (4, 64),   # rank-4, lower rank each
        (8, 32),   # rank-8, low rank each
    ]
    
    for num_ranks, rank in configs:
        np_comp = NPComponent(
            d_model=d_model,
            d_ffn=d_ffn,
            rank=rank,
            init_scale=0.01,
            single_layer_mode=False,
            num_ranks=num_ranks
        )
        
        total_params = sum(p.numel() for p in np_comp.parameters())
        
        # Calculate expected parameters
        if num_ranks == 1:
            expected = (d_model * rank) + (rank * d_model) + (rank * d_ffn)
        else:
            expected = num_ranks * ((d_model * rank) + (rank * d_model) + (rank * d_ffn))
        
        logger.info(f"  Rank-{num_ranks} (rank={rank}): {total_params:,} parameters")
        assert total_params == expected, f"Parameter count mismatch: {total_params} != {expected}"


def test_training_step():
    """Test a full training step with rank-k."""
    logger.info("\n" + "="*80)
    logger.info("Testing Training Step with Rank-k")
    logger.info("="*80)
    
    # Create small model
    config = LlamaConfig(
        hidden_size=256,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        vocab_size=1000,
    )
    config._attn_implementation = "eager"
    
    model = NPTLlamaModel(config)
    npt_config = NPTConfig(
        layers_to_convert=[1],
        np_rank=32,
        np_init_scale=0.01,
        single_layer_mode=True,
        num_ranks=2  # Rank-2
    )
    model.convert_to_npt(npt_config)
    model.freeze_base_parameters()
    
    # Check trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    
    # Simulate training step
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Forward pass
    model.train()
    outputs = model(input_ids)
    loss = outputs.logits.mean()  # Dummy loss
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradients
    has_gradients = False
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            has_gradients = True
            grad_norm = param.grad.norm().item()
            logger.info(f"  {name}: grad_norm={grad_norm:.6f}")
    
    assert has_gradients, "No gradients found"
    
    # Optimizer step
    optimizer.step()
    
    logger.info("✓ Training step with rank-k successful")


def main():
    """Run all tests."""
    logger.info("\n" + "="*80)
    logger.info("RANK-K NPT IMPLEMENTATION TEST SUITE")
    logger.info("="*80)
    
    try:
        # Test backward compatibility first
        test_backward_compatibility()
        
        # Test rank-k features
        test_rank_k_forward()
        test_rank_k_decoder_layer()
        test_rank_k_loss()
        test_parameter_comparison()
        test_training_step()
        
        logger.info("\n" + "="*80)
        logger.info("ALL TESTS PASSED ✓")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()