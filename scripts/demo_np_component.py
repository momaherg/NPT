"""
Demonstration script for the NPComponent module.
Shows how the component generates rank-1 weight updates.
"""

import torch
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.npt import NPComponent


def demo_np_component():
    """Demonstrate NPComponent functionality."""
    
    print("=" * 60)
    print("NPComponent Stage 1 Demo")
    print("=" * 60)
    
    # Configuration for a small model
    d_model = 2048  # 1B model dimensions
    d_ffn = 8192
    rank = 64
    batch_size = 2
    seq_len = 10
    
    print(f"\nConfiguration:")
    print(f"  d_model: {d_model}")
    print(f"  d_ffn: {d_ffn}")
    print(f"  rank: {rank}")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_len: {seq_len}")
    
    # Create NPComponent
    np_component = NPComponent(d_model=d_model, d_ffn=d_ffn, rank=rank)
    print(f"\nCreated NPComponent with {sum(p.numel() for p in np_component.parameters())} parameters")
    
    # Create sample input (simulating attention output)
    attn_output = torch.randn(batch_size, seq_len, d_model)
    print(f"\nInput shape: {attn_output.shape}")
    
    # Forward pass
    v_a, v_b = np_component(attn_output)
    print(f"\nOutput shapes:")
    print(f"  v_a: {v_a.shape}")
    print(f"  v_b: {v_b.shape}")
    
    # Compute delta_w (rank-1 weight update)
    delta_w = np_component.compute_delta_w(attn_output)
    print(f"\nDelta W shape: {delta_w.shape}")
    
    # Verify rank-1 property for first token
    first_token_delta = delta_w[0, 0]
    U, S, V = torch.linalg.svd(first_token_delta, full_matrices=False)
    print(f"\nSingular values (first 5) for token [0,0]: {S[:5].tolist()}")
    print(f"Ratio S[1]/S[0]: {(S[1]/S[0]).item():.2e} (should be near 0 for rank-1)")
    
    # Compute regularization loss
    reg_loss = np_component.get_regularization_loss(v_a, v_b)
    print(f"\nRegularization loss: {reg_loss.item():.4f}")
    
    # Show magnitude statistics
    print(f"\nVector magnitudes:")
    print(f"  ||v_a|| mean: {torch.norm(v_a, dim=-1).mean().item():.4f}")
    print(f"  ||v_b|| mean: {torch.norm(v_b, dim=-1).mean().item():.4f}")
    
    # Demonstrate gradient flow
    loss = reg_loss
    loss.backward()
    
    print(f"\nGradient statistics:")
    for name, param in np_component.named_parameters():
        if param.grad is not None:
            grad_norm = torch.norm(param.grad).item()
            print(f"  {name}: grad_norm = {grad_norm:.4f}")
    
    print("\n" + "=" * 60)
    print("Stage 1 Implementation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    demo_np_component()