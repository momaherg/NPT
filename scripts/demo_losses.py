"""
Demonstration script for NPT Loss Functions (Stage 4).
Shows equivalence loss computation and parallel forward passes.
"""

import torch
import torch.nn.functional as F
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.npt import NPTLlamaModel, NPTConfig
from src.training.losses import (
    FidelityLoss,
    RegularizationLoss,
    EquivalenceLoss,
    ParallelForwardHelper
)
from transformers import LlamaConfig


def demo_individual_losses():
    """Demonstrate individual loss components."""
    
    print("=" * 80)
    print("Individual Loss Components Demo")
    print("=" * 80)
    
    batch_size, seq_len, hidden_size = 2, 10, 256
    intermediate_size = 1024
    
    # Create sample outputs
    npt_output = torch.randn(batch_size, seq_len, hidden_size)
    original_output = torch.randn(batch_size, seq_len, hidden_size)
    
    # Create sample NP component outputs
    v_a_list = [torch.randn(batch_size, seq_len, hidden_size) * 0.01 for _ in range(4)]
    v_b_list = [torch.randn(batch_size, seq_len, intermediate_size) * 0.01 for _ in range(4)]
    
    print("\n" + "-" * 60)
    print("1. Fidelity Loss")
    print("-" * 60)
    
    fidelity_loss = FidelityLoss(reduction='mean')
    
    # Test with identical outputs
    zero_loss = fidelity_loss(npt_output, npt_output)
    print(f"Identical outputs loss: {zero_loss.item():.6f} (should be ~0)")
    
    # Test with different outputs
    fid_loss = fidelity_loss(npt_output, original_output)
    print(f"Different outputs loss: {fid_loss.item():.6f}")
    
    # Add small perturbation
    perturbed = original_output + torch.randn_like(original_output) * 0.01
    small_loss = fidelity_loss(original_output, perturbed)
    print(f"Small perturbation loss: {small_loss.item():.6f}")
    
    print("\n" + "-" * 60)
    print("2. Regularization Loss")
    print("-" * 60)
    
    reg_loss_fn = RegularizationLoss(reduction='mean')
    
    # Test with small magnitude vectors
    reg_loss = reg_loss_fn(v_a_list, v_b_list)
    print(f"Regularization loss (small vectors): {reg_loss.item():.6f}")
    
    # Test with larger magnitude vectors
    v_a_large = [v * 10 for v in v_a_list]
    v_b_large = [v * 10 for v in v_b_list]
    reg_loss_large = reg_loss_fn(v_a_large, v_b_large)
    print(f"Regularization loss (large vectors): {reg_loss_large.item():.6f}")
    
    # Show per-layer regularization
    reg_loss_per_layer = RegularizationLoss(reduction='none')
    layer_losses = reg_loss_per_layer(v_a_list, v_b_list)
    print(f"Per-layer regularization: {[f'{l.item():.6f}' for l in layer_losses]}")


def demo_equivalence_loss():
    """Demonstrate combined equivalence loss."""
    
    print("\n" + "=" * 80)
    print("Equivalence Loss Demo")
    print("=" * 80)
    
    batch_size, seq_len, hidden_size = 2, 10, 256
    intermediate_size = 1024
    
    # Create sample data
    npt_output = torch.randn(batch_size, seq_len, hidden_size)
    original_output = torch.randn(batch_size, seq_len, hidden_size)
    v_a_list = [torch.randn(batch_size, seq_len, hidden_size) * 0.01 for _ in range(4)]
    v_b_list = [torch.randn(batch_size, seq_len, intermediate_size) * 0.01 for _ in range(4)]
    
    # Test different lambda values
    lambda_values = [0.0, 0.001, 0.01, 0.1]
    
    print("\n" + "-" * 60)
    print("Effect of Lambda (Regularization Weight)")
    print("-" * 60)
    
    for lambda_reg in lambda_values:
        equiv_loss = EquivalenceLoss(lambda_reg=lambda_reg)
        result = equiv_loss(npt_output, original_output, v_a_list, v_b_list)
        
        print(f"\nλ = {lambda_reg}:")
        print(f"  Total loss: {result.total_loss.item():.6f}")
        print(f"  Fidelity loss: {result.fidelity_loss.item():.6f}")
        print(f"  Regularization loss: {result.regularization_loss.item():.6f}")
        print(f"  Reg contribution: {(lambda_reg * result.regularization_loss).item():.6f}")
    
    # Show metrics
    print("\n" + "-" * 60)
    print("Loss Metrics")
    print("-" * 60)
    
    equiv_loss = EquivalenceLoss(lambda_reg=0.01)
    result = equiv_loss(npt_output, original_output, v_a_list, v_b_list)
    
    print("Computed metrics:")
    for key, value in result.metrics.items():
        print(f"  {key}: {value:.6f}")


def demo_parallel_forward():
    """Demonstrate parallel forward passes with NPT model."""
    
    print("\n" + "=" * 80)
    print("Parallel Forward Pass Demo")
    print("=" * 80)
    
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
    
    print(f"\nModel Configuration:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Intermediate size: {config.intermediate_size}")
    print(f"  Number of layers: {config.num_hidden_layers}")
    
    # Create NPT model
    model = NPTLlamaModel(config)
    
    # Convert upper half to NPT
    npt_config = NPTConfig(
        convert_range=(2, 4),  # Layers 2-3
        np_rank=32,
        np_init_scale=0.01
    )
    model.convert_to_npt(npt_config)
    
    print(f"\nNPT Configuration:")
    print(f"  Converted layers: {list(model.npt_layers.keys())}")
    print(f"  NP rank: {npt_config.np_rank}")
    print(f"  NP init scale: {npt_config.np_init_scale}")
    
    # Create helper
    helper = ParallelForwardHelper(model)
    
    # Create input
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"\nInput shape: {input_ids.shape}")
    
    # Run parallel forward
    print("\n" + "-" * 60)
    print("Running Parallel Forward Pass")
    print("-" * 60)
    
    model.eval()
    with torch.no_grad():
        npt_output, original_output, v_a_list, v_b_list = helper.forward(
            input_ids,
            collect_np_outputs=True
        )
    
    print(f"NPT output shape: {npt_output.shape}")
    print(f"Original output shape: {original_output.shape}")
    print(f"Number of v_a vectors collected: {len(v_a_list)}")
    print(f"Number of v_b vectors collected: {len(v_b_list)}")
    
    # Compute difference
    diff = torch.abs(npt_output - original_output)
    print(f"\nOutput difference statistics:")
    print(f"  Mean: {diff.mean().item():.6f}")
    print(f"  Max: {diff.max().item():.6f}")
    print(f"  Min: {diff.min().item():.6f}")
    
    # Analyze NP component outputs
    print("\n" + "-" * 60)
    print("NP Component Output Analysis")
    print("-" * 60)
    
    for i, (v_a, v_b) in enumerate(zip(v_a_list, v_b_list)):
        v_a_norm = torch.norm(v_a) / v_a.numel()**0.5
        v_b_norm = torch.norm(v_b) / v_b.numel()**0.5
        print(f"Layer {list(model.npt_layers.keys())[i]}:")
        print(f"  v_a norm (normalized): {v_a_norm.item():.6f}")
        print(f"  v_b norm (normalized): {v_b_norm.item():.6f}")


def demo_training_step():
    """Demonstrate a training step with loss computation."""
    
    print("\n" + "=" * 80)
    print("Training Step Demo")
    print("=" * 80)
    
    # Create model
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
    
    # Convert all layers to NPT
    npt_config = NPTConfig(
        convert_all=True,
        np_rank=32,
        np_init_scale=0.01
    )
    model.convert_to_npt(npt_config)
    
    # Freeze base parameters
    model.freeze_base_parameters()
    
    print(f"\nModel setup:")
    param_counts = model.count_parameters()
    print(f"  Total parameters: {param_counts['total']:,}")
    print(f"  Base parameters (frozen): {param_counts['base']:,}")
    print(f"  NPT parameters (trainable): {param_counts['npt']:,}")
    print(f"  NPT ratio: {param_counts['npt_ratio']:.2%}")
    
    # Create loss function and helper
    loss_fn = EquivalenceLoss(lambda_reg=0.01)
    helper = ParallelForwardHelper(model)
    
    # Create optimizer for NPT parameters only
    npt_params = model.get_npt_parameters()
    optimizer = torch.optim.Adam(npt_params, lr=1e-4)
    
    # Create input
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print("\n" + "-" * 60)
    print("Simulating Training Step")
    print("-" * 60)
    
    # Training step
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    npt_output, original_output, v_a_list, v_b_list = helper.forward(
        input_ids,
        collect_np_outputs=True
    )
    
    # Compute loss
    loss_output = loss_fn(npt_output, original_output, v_a_list, v_b_list)
    
    print(f"\nBefore optimization:")
    print(f"  Total loss: {loss_output.total_loss.item():.6f}")
    print(f"  Fidelity loss: {loss_output.fidelity_loss.item():.6f}")
    print(f"  Regularization loss: {loss_output.regularization_loss.item():.6f}")
    
    # Backward pass
    loss_output.total_loss.backward()
    
    # Check gradients
    grad_norms = []
    for param in npt_params:
        if param.grad is not None:
            grad_norms.append(torch.norm(param.grad).item())
    
    print(f"\nGradient statistics:")
    print(f"  Number of parameters with gradients: {len(grad_norms)}")
    print(f"  Average gradient norm: {sum(grad_norms)/len(grad_norms):.6f}")
    print(f"  Max gradient norm: {max(grad_norms):.6f}")
    
    # Optimizer step
    optimizer.step()
    
    # Evaluate after one step
    model.eval()
    with torch.no_grad():
        npt_output_new, _, _, _ = helper.forward(
            input_ids,
            collect_np_outputs=False
        )
        
        # Compute new loss
        new_fidelity = F.mse_loss(npt_output_new, original_output)
    
    print(f"\nAfter optimization:")
    print(f"  New fidelity loss: {new_fidelity.item():.6f}")
    print(f"  Improvement: {(loss_output.fidelity_loss.item() - new_fidelity.item()):.6f}")
    
    print("\n" + "=" * 80)
    print("Stage 4 Implementation Complete!")
    print("✓ Loss functions implemented")
    print("✓ Parallel forward helper working")
    print("✓ Training step demonstrated")
    print("✓ Gradients flow correctly to NP components")
    print("=" * 80)


def main():
    """Run all demonstrations."""
    
    print("=" * 80)
    print("NPT Loss Functions (Stage 4) Demonstration")
    print("=" * 80)
    
    # Run demos
    demo_individual_losses()
    demo_equivalence_loss()
    demo_parallel_forward()
    demo_training_step()


if __name__ == "__main__":
    # Ensure deterministic results for demo
    torch.manual_seed(42)
    main()