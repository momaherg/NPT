"""
Unit tests for the NPT Model.

Tests cover model initialization, selective layer conversion,
weight preservation, parameter management, and forward pass.
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.npt.npt_model import NPTLlamaModel, NPTConfig
from src.npt.npt_decoder_layer import NPTDecoderLayer
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer


class TestNPTModel:
    """Test suite for NPTLlamaModel."""
    
    @pytest.fixture
    def small_config(self):
        """Small configuration for testing."""
        config = LlamaConfig(
            hidden_size=256,
            intermediate_size=1024,
            num_hidden_layers=8,
            num_attention_heads=8,
            num_key_value_heads=4,
            vocab_size=1000,
            max_position_embeddings=512,
        )
        config._attn_implementation = "eager"
        return config
    
    @pytest.fixture
    def npt_config_all(self):
        """NPT config to convert all layers."""
        return NPTConfig(
            convert_all=True,
            np_rank=32,
            np_init_scale=0.01
        )
    
    @pytest.fixture
    def npt_config_range(self):
        """NPT config to convert a range of layers."""
        return NPTConfig(
            convert_range=(4, 8),  # Convert layers 4-7
            np_rank=32,
            np_init_scale=0.01
        )
    
    @pytest.fixture
    def npt_config_list(self):
        """NPT config to convert specific layers."""
        return NPTConfig(
            layers_to_convert=[2, 4, 6],
            np_rank=32,
            np_init_scale=0.01
        )
    
    @pytest.fixture
    def sample_input(self, small_config):
        """Create sample input."""
        batch_size = 2
        seq_len = 10
        return torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
    
    def test_model_initialization(self, small_config):
        """Test that NPTLlamaModel initializes correctly."""
        model = NPTLlamaModel(small_config)
        
        # Check basic attributes
        assert hasattr(model, 'npt_config')
        assert hasattr(model, 'npt_layers')
        assert hasattr(model, 'original_layers')
        
        # Initially no NPT layers
        assert len(model.npt_layers) == 0
        assert len(model.original_layers) == 0
        
        # Check config augmentation
        assert hasattr(model.config, 'np_rank')
        assert hasattr(model.config, 'np_init_scale')
    
    def test_npt_config_validation(self):
        """Test NPTConfig validation."""
        # Should not allow multiple conversion strategies
        with pytest.raises(ValueError):
            NPTConfig(
                convert_all=True,
                layers_to_convert=[1, 2, 3]
            )
        
        with pytest.raises(ValueError):
            NPTConfig(
                convert_range=(0, 4),
                convert_all=True
            )
    
    def test_get_layers_to_convert(self):
        """Test layer selection logic."""
        # Test convert_all
        config = NPTConfig(convert_all=True)
        layers = config.get_layers_to_convert(8)
        assert layers == list(range(8))
        
        # Test convert_range
        config = NPTConfig(convert_range=(2, 6))
        layers = config.get_layers_to_convert(8)
        assert layers == [2, 3, 4, 5]
        
        # Test layers_to_convert
        config = NPTConfig(layers_to_convert=[1, 3, 5])
        layers = config.get_layers_to_convert(8)
        assert layers == [1, 3, 5]
        
        # Test negative indices
        config = NPTConfig(layers_to_convert=[-1, -2])
        layers = config.get_layers_to_convert(8)
        assert layers == [6, 7]
        
        # Test default (upper half)
        config = NPTConfig()
        layers = config.get_layers_to_convert(8)
        assert layers == [4, 5, 6, 7]
    
    def test_convert_all_layers(self, small_config, npt_config_all):
        """Test converting all layers to NPT."""
        model = NPTLlamaModel(small_config)
        
        # Convert to NPT
        model.convert_to_npt(npt_config_all)
        
        # Check all layers are converted
        assert len(model.npt_layers) == small_config.num_hidden_layers
        assert model.num_npt_layers == small_config.num_hidden_layers
        
        # Check each layer is NPTDecoderLayer
        for i in range(small_config.num_hidden_layers):
            assert isinstance(model.model.layers[i], NPTDecoderLayer)
            assert i in model.npt_layers
    
    def test_convert_range_layers(self, small_config, npt_config_range):
        """Test converting a range of layers."""
        model = NPTLlamaModel(small_config)
        
        # Convert to NPT
        model.convert_to_npt(npt_config_range)
        
        # Check correct layers are converted
        assert len(model.npt_layers) == 4
        assert set(model.npt_layers.keys()) == {4, 5, 6, 7}
        
        # Check layer types
        for i in range(small_config.num_hidden_layers):
            if i in [4, 5, 6, 7]:
                assert isinstance(model.model.layers[i], NPTDecoderLayer)
            else:
                assert isinstance(model.model.layers[i], LlamaDecoderLayer)
                assert not isinstance(model.model.layers[i], NPTDecoderLayer)
    
    def test_convert_specific_layers(self, small_config, npt_config_list):
        """Test converting specific layers."""
        model = NPTLlamaModel(small_config)
        
        # Convert to NPT
        model.convert_to_npt(npt_config_list)
        
        # Check correct layers are converted
        assert len(model.npt_layers) == 3
        assert set(model.npt_layers.keys()) == {2, 4, 6}
        
        # Check layer types
        for i in range(small_config.num_hidden_layers):
            if i in [2, 4, 6]:
                assert isinstance(model.model.layers[i], NPTDecoderLayer)
            else:
                assert not isinstance(model.model.layers[i], NPTDecoderLayer)
    
    def test_weight_preservation(self, small_config, npt_config_range):
        """Test that original weights are preserved after conversion."""
        # Create model and store original weights
        model = NPTLlamaModel(small_config)
        
        # Store weights of layer that will be converted
        layer_idx = 5  # This will be converted
        original_attn_weights = model.model.layers[layer_idx].self_attn.q_proj.weight.clone()
        original_mlp_weights = model.model.layers[layer_idx].mlp.gate_proj.weight.clone()
        
        # Convert to NPT
        model.convert_to_npt(npt_config_range)
        
        # Check weights are preserved
        converted_layer = model.model.layers[layer_idx]
        assert torch.allclose(converted_layer.self_attn.q_proj.weight, original_attn_weights)
        assert torch.allclose(converted_layer.mlp.gate_proj.weight, original_mlp_weights)
    
    def test_freeze_base_parameters(self, small_config, npt_config_range):
        """Test freezing base parameters."""
        model = NPTLlamaModel(small_config)
        model.convert_to_npt(npt_config_range)
        
        # Freeze base parameters
        model.freeze_base_parameters()
        
        # Check that base parameters are frozen
        for name, param in model.named_parameters():
            if 'np_component' not in name:
                assert not param.requires_grad, f"Base parameter {name} not frozen"
            else:
                assert param.requires_grad, f"NP parameter {name} frozen"
    
    def test_get_npt_parameters(self, small_config, npt_config_range):
        """Test getting NP parameters."""
        model = NPTLlamaModel(small_config)
        model.convert_to_npt(npt_config_range)
        
        # Get NP parameters
        npt_params = model.get_npt_parameters()
        
        # Should have 3 parameters per converted layer (W_down, W_a_up, W_b_up)
        expected_params = 4 * 3  # 4 converted layers, 3 params each
        assert len(npt_params) == expected_params
        
        # All should be parameters
        for param in npt_params:
            assert isinstance(param, nn.Parameter)
    
    def test_get_npt_parameter_groups(self, small_config, npt_config_list):
        """Test getting grouped NP parameters."""
        model = NPTLlamaModel(small_config)
        model.convert_to_npt(npt_config_list)
        
        # Get parameter groups
        param_groups = model.get_npt_parameter_groups()
        
        # Should have one group per converted layer
        assert len(param_groups) == 3
        assert set(param_groups.keys()) == {'layer_2_np', 'layer_4_np', 'layer_6_np'}
        
        # Each group should have 3 parameters
        for group_name, params in param_groups.items():
            assert len(params) == 3
    
    def test_count_parameters(self, small_config, npt_config_range):
        """Test parameter counting."""
        model = NPTLlamaModel(small_config)
        
        # Count before conversion
        counts_before = model.count_parameters()
        assert counts_before['npt'] == 0
        assert counts_before['total'] == counts_before['base']
        
        # Convert and count again
        model.convert_to_npt(npt_config_range)
        counts_after = model.count_parameters()
        
        assert counts_after['npt'] > 0
        assert counts_after['total'] == counts_after['base'] + counts_after['npt']
        assert 0 < counts_after['npt_ratio'] < 0.1  # NPT should be small fraction
    
    def test_set_npt_mode(self, small_config, npt_config_range):
        """Test setting NPT mode."""
        model = NPTLlamaModel(small_config)
        model.convert_to_npt(npt_config_range)
        
        # Set to NPT mode
        model.set_npt_mode(True)
        for layer in model.npt_layers.values():
            assert layer.use_npt == True
        
        # Set to standard mode
        model.set_npt_mode(False)
        for layer in model.npt_layers.values():
            assert layer.use_npt == False
    
    def test_forward_pass(self, small_config, npt_config_all, sample_input):
        """Test forward pass through the model."""
        # Test with all layers converted (simpler case)
        model = NPTLlamaModel(small_config)
        model.convert_to_npt(npt_config_all)
        
        # Test forward pass in NPT mode
        model.set_npt_mode(True)
        model.eval()
        with torch.no_grad():
            outputs = model(sample_input)
        
        # Check output shape
        batch_size, seq_len = sample_input.shape
        vocab_size = small_config.vocab_size
        assert outputs.logits.shape == (batch_size, seq_len, vocab_size)
        
        # Test forward pass in standard mode
        model.set_npt_mode(False)
        with torch.no_grad():
            outputs_std = model(sample_input)
        
        assert outputs_std.logits.shape == (batch_size, seq_len, vocab_size)
    
    def test_save_load_npt_weights(self, small_config, npt_config_range):
        """Test saving and loading NP weights."""
        model = NPTLlamaModel(small_config)
        model.convert_to_npt(npt_config_range)
        
        # Modify NP weights
        for param in model.get_npt_parameters():
            param.data.fill_(0.5)
        
        # Save weights
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            save_path = f.name
        
        try:
            model.save_npt_weights(save_path)
            
            # Reset weights
            for param in model.get_npt_parameters():
                param.data.fill_(0.0)
            
            # Load weights
            model.load_npt_weights(save_path)
            
            # Check weights were loaded
            for param in model.get_npt_parameters():
                assert torch.allclose(param, torch.full_like(param, 0.5))
        finally:
            os.unlink(save_path)
    
    def test_get_layer_info(self, small_config, npt_config_range):
        """Test getting layer information."""
        model = NPTLlamaModel(small_config)
        model.convert_to_npt(npt_config_range)
        
        info = model.get_layer_info()
        
        assert info['total_layers'] == 8
        assert info['npt_layers'] == 4
        assert info['npt_layer_indices'] == [4, 5, 6, 7]
        assert info['standard_layer_indices'] == [0, 1, 2, 3]
        
        # Check layer types
        expected_types = ['Standard'] * 4 + ['NPT'] * 4
        assert info['layer_types'] == expected_types
    
    def test_reset_to_standard(self, small_config, npt_config_range):
        """Test resetting to standard layers."""
        model = NPTLlamaModel(small_config)
        
        # Store original layer references
        original_layers = [model.model.layers[i] for i in range(8)]
        
        # Convert to NPT
        model.convert_to_npt(npt_config_range)
        
        # Check some layers are NPT
        assert len(model.npt_layers) > 0
        
        # Reset to standard
        model.reset_to_standard()
        
        # Check all layers are back to original
        assert len(model.npt_layers) == 0
        for i in range(8):
            if i in [4, 5, 6, 7]:
                # These were converted, should be restored
                assert model.model.layers[i] == original_layers[i]
            # All should be standard layers
            assert not isinstance(model.model.layers[i], NPTDecoderLayer)
    
    def test_gradient_flow(self, small_config, npt_config_all, sample_input):
        """Test gradient flow with frozen base parameters."""
        model = NPTLlamaModel(small_config)
        model.convert_to_npt(npt_config_all)
        
        # Freeze base parameters
        model.freeze_base_parameters()
        
        # Forward pass
        outputs = model(sample_input, labels=sample_input)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        for name, param in model.named_parameters():
            if 'np_component' in name:
                # NP parameters should have gradients
                assert param.grad is not None, f"NP parameter {name} has no gradient"
                assert torch.any(param.grad != 0), f"NP parameter {name} has zero gradient"
            else:
                # Base parameters should not have gradients (frozen)
                assert param.grad is None or torch.all(param.grad == 0), \
                    f"Base parameter {name} has non-zero gradient"
    
    def test_memory_efficiency(self, small_config):
        """Test memory efficiency of NPT conversion."""
        model = NPTLlamaModel(small_config)
        
        # Get base model size
        base_params = model.count_parameters()['total']
        
        # Convert upper half
        npt_config = NPTConfig(convert_range=(4, 8), np_rank=32)
        model.convert_to_npt(npt_config)
        
        # Get new size
        final_counts = model.count_parameters()
        
        # NPT should add less than 10% parameters
        param_increase = (final_counts['total'] - base_params) / base_params
        assert param_increase < 0.1, f"Parameter increase {param_increase:.1%} exceeds 10%"
        
        print(f"Parameter increase: {param_increase:.2%}")
        print(f"NPT parameters: {final_counts['npt']:,}")
        print(f"Base parameters: {final_counts['base']:,}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])