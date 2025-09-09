"""
Unit tests for the NPT Decoder Layer.

Tests cover initialization, forward pass, mode switching,
gradient flow, and comparison with standard decoder layer.
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path
import gc

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.npt.npt_decoder_layer import NPTDecoderLayer
from src.npt.np_component import NPComponent
from transformers.models.llama.configuration_llama import LlamaConfig


class TestNPTDecoderLayer:
    """Test suite for NPTDecoderLayer."""
    
    @pytest.fixture
    def config_1b(self):
        """Configuration for 1B model."""
        config = LlamaConfig(
            hidden_size=2048,
            intermediate_size=8192,
            num_hidden_layers=1,
            num_attention_heads=32,
            num_key_value_heads=8,  # GQA
            vocab_size=128256,
            np_rank=64,  # Custom attribute for NP component
            np_init_scale=0.01,
        )
        # Set attention implementation to default
        config._attn_implementation = "eager"
        return config
    
    @pytest.fixture
    def config_small(self):
        """Small configuration for faster testing."""
        config = LlamaConfig(
            hidden_size=256,
            intermediate_size=1024,
            num_hidden_layers=1,
            num_attention_heads=8,
            num_key_value_heads=4,
            vocab_size=1000,
            np_rank=32,
            np_init_scale=0.01,
        )
        # Set attention implementation to default
        config._attn_implementation = "eager"
        return config
    
    @pytest.fixture
    def sample_input(self, config_small):
        """Create sample input tensor."""
        batch_size = 2
        seq_len = 10
        return torch.randn(batch_size, seq_len, config_small.hidden_size)
    
    def test_initialization(self, config_1b):
        """Test that NPTDecoderLayer initializes correctly."""
        layer = NPTDecoderLayer(config_1b, layer_idx=0)
        
        # Check that layer has all required components
        assert hasattr(layer, 'np_component')
        assert hasattr(layer, 'self_attn')
        assert hasattr(layer, 'mlp')
        assert hasattr(layer, 'input_layernorm')
        assert hasattr(layer, 'post_attention_layernorm')
        
        # Check NP component configuration
        assert isinstance(layer.np_component, NPComponent)
        assert layer.np_component.d_model == config_1b.hidden_size
        assert layer.np_component.d_ffn == config_1b.intermediate_size
        assert layer.np_component.rank == 64
        
        # Check default mode
        assert layer.use_npt == True
    
    def test_mode_switching(self, config_small):
        """Test switching between NPT and standard mode."""
        layer = NPTDecoderLayer(config_small, layer_idx=0)
        
        # Check initial mode
        assert layer.use_npt == True
        
        # Switch to standard mode
        layer.set_npt_mode(False)
        assert layer.use_npt == False
        
        # Switch back to NPT mode
        layer.set_npt_mode(True)
        assert layer.use_npt == True
    
    def test_forward_shape_npt_mode(self, config_small, sample_input):
        """Test forward pass output shape in NPT mode."""
        layer = NPTDecoderLayer(config_small, layer_idx=0)
        layer.set_npt_mode(True)
        
        # Forward pass
        outputs = layer(sample_input)
        
        # Check output shape
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs
            
        assert hidden_states.shape == sample_input.shape
    
    def test_forward_shape_standard_mode(self, config_small, sample_input):
        """Test forward pass output shape in standard mode."""
        layer = NPTDecoderLayer(config_small, layer_idx=0)
        layer.set_npt_mode(False)
        
        # Forward pass
        outputs = layer(sample_input)
        
        # Check output shape
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs
            
        assert hidden_states.shape == sample_input.shape
    
    def test_gradient_flow_npt_components(self, config_small, sample_input):
        """Test that gradients flow through NP components."""
        layer = NPTDecoderLayer(config_small, layer_idx=0)
        layer.set_npt_mode(True)
        
        # Freeze base parameters
        layer.freeze_base_parameters()
        
        # Check that only NP component parameters are trainable
        trainable_params = [p for p in layer.parameters() if p.requires_grad]
        npt_params = list(layer.np_component.parameters())
        assert len(trainable_params) == len(npt_params)
        
        # Forward pass
        outputs = layer(sample_input)
        hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
        
        # Create loss
        loss = torch.mean(hidden_states ** 2)
        
        # Backward pass
        loss.backward()
        
        # Check gradients on NP component
        assert layer.np_component.W_down.grad is not None
        assert layer.np_component.W_a_up.grad is not None
        assert layer.np_component.W_b_up.grad is not None
        
        # Check that gradients are non-zero
        assert torch.any(layer.np_component.W_down.grad != 0)
        assert torch.any(layer.np_component.W_a_up.grad != 0)
        assert torch.any(layer.np_component.W_b_up.grad != 0)
    
    def test_modulated_mlp_efficient(self, config_small):
        """Test the efficient version of modulated MLP works correctly."""
        layer = NPTDecoderLayer(config_small, layer_idx=0)
        
        batch_size = 2
        seq_len = 5
        hidden_states = torch.randn(batch_size, seq_len, config_small.hidden_size)
        v_a = torch.randn(batch_size, seq_len, config_small.hidden_size)
        v_b = torch.randn(batch_size, seq_len, config_small.intermediate_size)
        
        # Test the efficient implementation
        with torch.no_grad():
            output = layer._apply_modulated_mlp_efficient(hidden_states, v_a, v_b)
        
        # Check output has correct shape
        assert output.shape == (batch_size, seq_len, config_small.hidden_size)
        
        # Check output is finite
        assert torch.isfinite(output).all()
    
    def test_different_batch_sizes(self, config_small):
        """Test with different batch sizes and sequence lengths."""
        layer = NPTDecoderLayer(config_small, layer_idx=0)
        
        test_cases = [
            (1, 1),    # Single token
            (1, 20),   # Single batch, multiple tokens
            (4, 50),   # Multiple batches
            (8, 128),  # Larger batch
        ]
        
        for batch_size, seq_len in test_cases:
            input_tensor = torch.randn(batch_size, seq_len, config_small.hidden_size)
            outputs = layer(input_tensor)
            
            hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
            assert hidden_states.shape == (batch_size, seq_len, config_small.hidden_size)
    
    def test_get_npt_parameters(self, config_small):
        """Test getting only NP component parameters."""
        layer = NPTDecoderLayer(config_small, layer_idx=0)
        
        npt_params = layer.get_npt_parameters()
        expected_params = list(layer.np_component.parameters())
        
        assert len(npt_params) == len(expected_params)
        assert len(npt_params) == 3  # W_down, W_a_up, W_b_up
        
        # Check they're the same parameters
        for p1, p2 in zip(npt_params, expected_params):
            assert p1 is p2
    
    def test_output_consistency(self, config_small, sample_input):
        """Test that NPT mode produces consistent outputs."""
        layer = NPTDecoderLayer(config_small, layer_idx=0)
        layer.set_npt_mode(True)
        
        # Run forward pass twice with same input
        with torch.no_grad():
            output1 = layer(sample_input)
            output2 = layer(sample_input)
        
        hidden_states1 = output1[0] if isinstance(output1, tuple) else output1
        hidden_states2 = output2[0] if isinstance(output2, tuple) else output2
        
        # Check outputs are identical
        assert torch.allclose(hidden_states1, hidden_states2)
    
    def test_attention_mask(self, config_small):
        """Test forward pass with attention mask."""
        layer = NPTDecoderLayer(config_small, layer_idx=0)
        
        batch_size = 2
        seq_len = 10
        hidden_states = torch.randn(batch_size, seq_len, config_small.hidden_size)
        
        # Create causal attention mask
        attention_mask = torch.triu(
            torch.ones(seq_len, seq_len) * float('-inf'),
            diagonal=1
        )
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dims
        
        # Forward pass with mask
        outputs = layer(hidden_states, attention_mask=attention_mask)
        
        hidden_states_out = outputs[0] if isinstance(outputs, tuple) else outputs
        assert hidden_states_out.shape == hidden_states.shape
    
    def test_memory_efficiency(self, config_small):
        """Test memory usage of NPT layer."""
        layer = NPTDecoderLayer(config_small, layer_idx=0)
        
        # Count parameters
        total_params = sum(p.numel() for p in layer.parameters())
        npt_params = sum(p.numel() for p in layer.np_component.parameters())
        
        # NP component should be a small fraction of total parameters
        param_ratio = npt_params / total_params
        assert param_ratio < 0.1, f"NP component uses {param_ratio:.1%} of parameters"
        
        print(f"Total parameters: {total_params:,}")
        print(f"NP component parameters: {npt_params:,}")
        print(f"NP component ratio: {param_ratio:.2%}")
    
    def test_dtype_compatibility(self, config_small):
        """Test layer works with different dtypes."""
        layer = NPTDecoderLayer(config_small, layer_idx=0)
        
        batch_size = 2
        seq_len = 5
        
        # Test with float32
        input_f32 = torch.randn(batch_size, seq_len, config_small.hidden_size, dtype=torch.float32)
        output_f32 = layer(input_f32)
        hidden_f32 = output_f32[0] if isinstance(output_f32, tuple) else output_f32
        assert hidden_f32.dtype == torch.float32
        
        # Test with float16 (if supported)
        if torch.cuda.is_available():
            layer_f16 = layer.half().cuda()
            input_f16 = input_f32.half().cuda()
            output_f16 = layer_f16(input_f16)
            hidden_f16 = output_f16[0] if isinstance(output_f16, tuple) else output_f16
            assert hidden_f16.dtype == torch.float16
    
    def test_standard_mode_equivalence(self, config_small, sample_input):
        """Test that standard mode behaves like regular LlamaDecoderLayer."""
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        
        # Create both layers
        npt_layer = NPTDecoderLayer(config_small, layer_idx=0)
        standard_layer = LlamaDecoderLayer(config_small, layer_idx=0)
        
        # Copy weights from standard to NPT
        npt_layer.load_state_dict(standard_layer.state_dict(), strict=False)
        
        # Set NPT layer to standard mode
        npt_layer.set_npt_mode(False)
        
        # Create position embeddings for both
        batch_size, seq_len, _ = sample_input.shape
        head_dim = config_small.hidden_size // config_small.num_attention_heads
        cos = torch.ones(batch_size, seq_len, head_dim, dtype=sample_input.dtype)
        sin = torch.zeros(batch_size, seq_len, head_dim, dtype=sample_input.dtype)
        position_embeddings = (cos, sin)
        
        # Forward pass through both with same position embeddings
        with torch.no_grad():
            npt_output = npt_layer(sample_input, position_embeddings=position_embeddings)
            standard_output = standard_layer(sample_input, position_embeddings=position_embeddings)
        
        npt_hidden = npt_output[0] if isinstance(npt_output, tuple) else npt_output
        standard_hidden = standard_output[0] if isinstance(standard_output, tuple) else standard_output
        
        # Outputs should be very close (allowing for numerical differences)
        assert torch.allclose(npt_hidden, standard_hidden, rtol=1e-4, atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])