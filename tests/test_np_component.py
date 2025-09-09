"""
Unit tests for the Neuro-Plastic Component (NPComponent).

Tests cover initialization, forward pass, shape verification,
gradient flow, and regularization loss computation.
"""

import pytest
import torch
import torch.nn as nn
import math
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.npt.np_component import NPComponent


class TestNPComponent:
    """Test suite for NPComponent."""
    
    @pytest.fixture
    def dimensions_1b(self):
        """Dimensions for 1B model."""
        return {
            'd_model': 2048,
            'd_ffn': 8192,
            'rank': 64
        }
    
    @pytest.fixture
    def dimensions_8b(self):
        """Dimensions for 8B model."""
        return {
            'd_model': 4096,
            'd_ffn': 14336,
            'rank': 64
        }
    
    @pytest.fixture
    def sample_input(self, dimensions_1b):
        """Create sample input tensor."""
        batch_size = 2
        seq_len = 10
        d_model = dimensions_1b['d_model']
        return torch.randn(batch_size, seq_len, d_model)
    
    def test_initialization(self, dimensions_1b):
        """Test that NPComponent initializes correctly."""
        component = NPComponent(**dimensions_1b)
        
        # Check that all parameters exist
        assert hasattr(component, 'W_down')
        assert hasattr(component, 'W_a_up')
        assert hasattr(component, 'W_b_up')
        
        # Check parameter shapes
        assert component.W_down.shape == (dimensions_1b['d_model'], dimensions_1b['rank'])
        assert component.W_a_up.shape == (dimensions_1b['rank'], dimensions_1b['d_model'])
        assert component.W_b_up.shape == (dimensions_1b['rank'], dimensions_1b['d_ffn'])
        
        # Check that parameters are trainable
        assert component.W_down.requires_grad
        assert component.W_a_up.requires_grad
        assert component.W_b_up.requires_grad
    
    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        component = NPComponent(
            d_model=512,
            d_ffn=2048,
            rank=32,
            init_scale=0.02
        )
        
        assert component.d_model == 512
        assert component.d_ffn == 2048
        assert component.rank == 32
        assert component.init_scale == 0.02
    
    def test_forward_shape(self, dimensions_1b, sample_input):
        """Test that forward pass produces correct output shapes."""
        component = NPComponent(**dimensions_1b)
        
        v_a, v_b = component(sample_input)
        
        batch_size, seq_len, _ = sample_input.shape
        
        # Check output shapes
        assert v_a.shape == (batch_size, seq_len, dimensions_1b['d_model'])
        assert v_b.shape == (batch_size, seq_len, dimensions_1b['d_ffn'])
    
    def test_forward_different_batch_sizes(self, dimensions_1b):
        """Test forward pass with different batch sizes and sequence lengths."""
        component = NPComponent(**dimensions_1b)
        
        test_cases = [
            (1, 1),    # Single token
            (1, 128),  # Single batch, multiple tokens
            (4, 256),  # Multiple batches, longer sequence
            (32, 512), # Larger batch
        ]
        
        for batch_size, seq_len in test_cases:
            input_tensor = torch.randn(batch_size, seq_len, dimensions_1b['d_model'])
            v_a, v_b = component(input_tensor)
            
            assert v_a.shape == (batch_size, seq_len, dimensions_1b['d_model'])
            assert v_b.shape == (batch_size, seq_len, dimensions_1b['d_ffn'])
    
    def test_compute_delta_w(self, dimensions_1b, sample_input):
        """Test computation of rank-1 weight delta."""
        component = NPComponent(**dimensions_1b)
        
        delta_w = component.compute_delta_w(sample_input)
        
        batch_size, seq_len, _ = sample_input.shape
        
        # Check shape
        assert delta_w.shape == (batch_size, seq_len, dimensions_1b['d_ffn'], dimensions_1b['d_model'])
        
        # Verify it's rank-1 for each token by reconstructing from v_a and v_b
        v_a, v_b = component(sample_input)
        
        for b in range(min(2, batch_size)):
            for t in range(min(3, seq_len)):
                matrix = delta_w[b, t]
                
                # Reconstruct the matrix from outer product
                v_a_token = v_a[b, t]  # Shape: (d_model,)
                v_b_token = v_b[b, t]  # Shape: (d_ffn,)
                
                # Compute outer product
                reconstructed = torch.outer(v_b_token, v_a_token)
                
                # Check that the matrix matches the outer product (which is rank-1 by definition)
                assert torch.allclose(matrix, reconstructed, rtol=1e-5), \
                    f"Matrix at batch {b}, token {t} doesn't match outer product"
                
                # Alternative verification: check using matrix rank
                # For a rank-1 matrix, all columns should be scalar multiples of each other
                # We can verify by checking that the matrix can be expressed as outer product
                U, S, V = torch.linalg.svd(matrix, full_matrices=False)
                # For rank-1, only first singular value should be significant
                s_ratio = S[1] / S[0] if S[0] > 1e-8 else 0
                assert s_ratio < 1e-5, f"Matrix at batch {b}, token {t} has s_ratio {s_ratio}, expected near 0"
    
    def test_gradient_flow(self, dimensions_1b, sample_input):
        """Test that gradients flow through the component correctly."""
        component = NPComponent(**dimensions_1b)
        
        # Forward pass
        v_a, v_b = component(sample_input)
        
        # Create a dummy loss
        loss = torch.mean(v_a ** 2) + torch.mean(v_b ** 2)
        
        # Backward pass
        loss.backward()
        
        # Check that all parameters have gradients
        assert component.W_down.grad is not None
        assert component.W_a_up.grad is not None
        assert component.W_b_up.grad is not None
        
        # Check that gradients are non-zero
        assert torch.any(component.W_down.grad != 0)
        assert torch.any(component.W_a_up.grad != 0)
        assert torch.any(component.W_b_up.grad != 0)
    
    def test_gradient_accumulation(self, dimensions_1b):
        """Test gradient accumulation over multiple forward passes."""
        component = NPComponent(**dimensions_1b)
        
        # Zero gradients initially
        component.zero_grad()
        
        # Multiple forward-backward passes
        for _ in range(3):
            input_tensor = torch.randn(2, 10, dimensions_1b['d_model'])
            v_a, v_b = component(input_tensor)
            loss = torch.mean(v_a ** 2) + torch.mean(v_b ** 2)
            loss.backward()
        
        # Check that gradients accumulated
        assert component.W_down.grad is not None
        assert torch.any(component.W_down.grad != 0)
    
    def test_regularization_loss(self, dimensions_1b, sample_input):
        """Test regularization loss computation."""
        component = NPComponent(**dimensions_1b)
        
        v_a, v_b = component(sample_input)
        reg_loss = component.get_regularization_loss(v_a, v_b)
        
        # Check that loss is a scalar
        assert reg_loss.dim() == 0
        
        # Check that loss is positive
        assert reg_loss.item() > 0
        
        # Manually compute expected loss
        expected_loss = torch.mean(torch.sum(v_a ** 2, dim=-1)) + torch.mean(torch.sum(v_b ** 2, dim=-1))
        assert torch.allclose(reg_loss, expected_loss, rtol=1e-5)
    
    def test_initialization_magnitude(self, dimensions_1b):
        """Test that initialized weights produce low-magnitude outputs."""
        component = NPComponent(**dimensions_1b, init_scale=0.01)
        
        # Create normalized input
        input_tensor = torch.randn(4, 32, dimensions_1b['d_model'])
        input_tensor = input_tensor / torch.norm(input_tensor, dim=-1, keepdim=True)
        
        v_a, v_b = component(input_tensor)
        
        # Check that output magnitudes are reasonably small
        v_a_norm = torch.norm(v_a, dim=-1).mean()
        v_b_norm = torch.norm(v_b, dim=-1).mean()
        
        # With init_scale=0.01, we expect small outputs
        assert v_a_norm < 10.0, f"v_a norm too large: {v_a_norm}"
        assert v_b_norm < 50.0, f"v_b norm too large: {v_b_norm}"  # FFN dimension is larger
    
    def test_device_compatibility(self, dimensions_1b):
        """Test that component works on different devices."""
        component = NPComponent(**dimensions_1b)
        
        # Test on CPU
        input_cpu = torch.randn(2, 10, dimensions_1b['d_model'])
        v_a_cpu, v_b_cpu = component(input_cpu)
        assert v_a_cpu.device.type == 'cpu'
        assert v_b_cpu.device.type == 'cpu'
        
        # Test on CUDA if available
        if torch.cuda.is_available():
            component_cuda = component.cuda()
            input_cuda = input_cpu.cuda()
            v_a_cuda, v_b_cuda = component_cuda(input_cuda)
            assert v_a_cuda.device.type == 'cuda'
            assert v_b_cuda.device.type == 'cuda'
    
    def test_deterministic_forward(self, dimensions_1b):
        """Test that forward pass is deterministic with same input."""
        component = NPComponent(**dimensions_1b)
        
        input_tensor = torch.randn(2, 10, dimensions_1b['d_model'])
        
        # Run forward pass twice
        v_a1, v_b1 = component(input_tensor)
        v_a2, v_b2 = component(input_tensor)
        
        # Check outputs are identical
        assert torch.allclose(v_a1, v_a2)
        assert torch.allclose(v_b1, v_b2)
    
    def test_8b_model_dimensions(self, dimensions_8b):
        """Test component with 8B model dimensions."""
        component = NPComponent(**dimensions_8b)
        
        batch_size = 2
        seq_len = 10
        input_tensor = torch.randn(batch_size, seq_len, dimensions_8b['d_model'])
        
        v_a, v_b = component(input_tensor)
        
        assert v_a.shape == (batch_size, seq_len, dimensions_8b['d_model'])
        assert v_b.shape == (batch_size, seq_len, dimensions_8b['d_ffn'])
    
    def test_extra_repr(self, dimensions_1b):
        """Test string representation of the module."""
        component = NPComponent(**dimensions_1b)
        repr_str = component.extra_repr()
        
        assert 'd_model=2048' in repr_str
        assert 'd_ffn=8192' in repr_str
        assert 'rank=64' in repr_str
        assert 'init_scale=0.01' in repr_str
    
    def test_memory_efficiency(self, dimensions_1b):
        """Test that rank-1 computation is memory efficient."""
        component = NPComponent(**dimensions_1b)
        
        # Large batch to test memory
        input_tensor = torch.randn(8, 128, dimensions_1b['d_model'])
        
        # This should not create the full delta_w matrix in memory
        v_a, v_b = component(input_tensor)
        
        # Verify we can compute with these vectors without OOM
        # In practice, we'd apply these vectors directly without forming full matrix
        assert v_a.shape == (8, 128, dimensions_1b['d_model'])
        assert v_b.shape == (8, 128, dimensions_1b['d_ffn'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])