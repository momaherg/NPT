"""
Unit tests for NPT loss functions.

Tests cover fidelity loss, regularization loss, combined equivalence loss,
and the parallel forward helper.
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.losses import (
    FidelityLoss,
    RegularizationLoss,
    EquivalenceLoss,
    ParallelForwardHelper,
    LossOutput
)
from src.npt import NPTLlamaModel, NPTConfig
from transformers import LlamaConfig


class TestFidelityLoss:
    """Test suite for FidelityLoss."""
    
    @pytest.fixture
    def fidelity_loss(self):
        """Create a fidelity loss instance."""
        return FidelityLoss(reduction='mean')
    
    def test_fidelity_loss_initialization(self):
        """Test fidelity loss initialization."""
        loss_fn = FidelityLoss(reduction='mean')
        assert loss_fn.reduction == 'mean'
        
        loss_fn = FidelityLoss(reduction='sum')
        assert loss_fn.reduction == 'sum'
    
    def test_fidelity_loss_identical_outputs(self, fidelity_loss):
        """Test that identical outputs produce zero loss."""
        batch_size, seq_len, hidden_size = 2, 10, 256
        output = torch.randn(batch_size, seq_len, hidden_size)
        
        loss = fidelity_loss(output, output)
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-7)
    
    def test_fidelity_loss_different_outputs(self, fidelity_loss):
        """Test that different outputs produce non-zero loss."""
        batch_size, seq_len, hidden_size = 2, 10, 256
        npt_output = torch.randn(batch_size, seq_len, hidden_size)
        original_output = torch.randn(batch_size, seq_len, hidden_size)
        
        loss = fidelity_loss(npt_output, original_output)
        assert loss > 0
        
        # Verify MSE calculation
        expected_loss = torch.mean((npt_output - original_output) ** 2)
        assert torch.allclose(loss, expected_loss)
    
    def test_fidelity_loss_shape_mismatch(self, fidelity_loss):
        """Test that shape mismatch raises an error."""
        npt_output = torch.randn(2, 10, 256)
        original_output = torch.randn(2, 10, 512)  # Different hidden size
        
        with pytest.raises(ValueError, match="Shape mismatch"):
            fidelity_loss(npt_output, original_output)
    
    def test_fidelity_loss_reduction_modes(self):
        """Test different reduction modes."""
        batch_size, seq_len, hidden_size = 2, 10, 256
        npt_output = torch.randn(batch_size, seq_len, hidden_size)
        original_output = torch.randn(batch_size, seq_len, hidden_size)
        
        # Mean reduction
        loss_mean = FidelityLoss(reduction='mean')
        mean_result = loss_mean(npt_output, original_output)
        
        # Sum reduction
        loss_sum = FidelityLoss(reduction='sum')
        sum_result = loss_sum(npt_output, original_output)
        
        # Sum should be larger than mean
        assert sum_result > mean_result
        
        # None reduction
        loss_none = FidelityLoss(reduction='none')
        none_result = loss_none(npt_output, original_output)
        assert none_result.shape == (batch_size, seq_len, hidden_size)


class TestRegularizationLoss:
    """Test suite for RegularizationLoss."""
    
    @pytest.fixture
    def reg_loss(self):
        """Create a regularization loss instance."""
        return RegularizationLoss(reduction='mean')
    
    def test_regularization_loss_initialization(self):
        """Test regularization loss initialization."""
        loss_fn = RegularizationLoss(reduction='mean')
        assert loss_fn.reduction == 'mean'
    
    def test_regularization_loss_zero_vectors(self, reg_loss):
        """Test that zero vectors produce zero loss."""
        v_a_list = [torch.zeros(2, 10, 256) for _ in range(3)]
        v_b_list = [torch.zeros(2, 10, 1024) for _ in range(3)]
        
        loss = reg_loss(v_a_list, v_b_list)
        assert torch.allclose(loss, torch.tensor(0.0))
    
    def test_regularization_loss_non_zero_vectors(self, reg_loss):
        """Test that non-zero vectors produce positive loss."""
        v_a_list = [torch.randn(2, 10, 256) for _ in range(3)]
        v_b_list = [torch.randn(2, 10, 1024) for _ in range(3)]
        
        loss = reg_loss(v_a_list, v_b_list)
        assert loss > 0
        
        # Verify L2 calculation
        expected_loss = 0
        for v_a, v_b in zip(v_a_list, v_b_list):
            layer_loss = (torch.sum(v_a ** 2) + torch.sum(v_b ** 2)) / (v_a.numel() + v_b.numel())
            expected_loss += layer_loss
        expected_loss = expected_loss / len(v_a_list)
        
        assert torch.allclose(loss, expected_loss, rtol=1e-5)
    
    def test_regularization_loss_empty_lists(self, reg_loss):
        """Test that empty lists produce zero loss."""
        loss = reg_loss([], [])
        assert torch.allclose(loss, torch.tensor(0.0))
    
    def test_regularization_loss_mismatched_lists(self, reg_loss):
        """Test that mismatched list lengths raise an error."""
        v_a_list = [torch.randn(2, 10, 256) for _ in range(3)]
        v_b_list = [torch.randn(2, 10, 1024) for _ in range(2)]  # Different length
        
        with pytest.raises(ValueError, match="Mismatch in number"):
            reg_loss(v_a_list, v_b_list)
    
    def test_regularization_loss_reduction_modes(self):
        """Test different reduction modes."""
        v_a_list = [torch.ones(2, 10, 256) for _ in range(3)]
        v_b_list = [torch.ones(2, 10, 1024) for _ in range(3)]
        
        # Mean reduction
        loss_mean = RegularizationLoss(reduction='mean')
        mean_result = loss_mean(v_a_list, v_b_list)
        
        # Sum reduction
        loss_sum = RegularizationLoss(reduction='sum')
        sum_result = loss_sum(v_a_list, v_b_list)
        
        # Sum should be larger than mean
        assert sum_result > mean_result
        
        # None reduction
        loss_none = RegularizationLoss(reduction='none')
        none_result = loss_none(v_a_list, v_b_list)
        assert none_result.shape == (3,)  # One value per layer


class TestEquivalenceLoss:
    """Test suite for EquivalenceLoss."""
    
    @pytest.fixture
    def equiv_loss(self):
        """Create an equivalence loss instance."""
        return EquivalenceLoss(lambda_reg=0.01, reduction='mean')
    
    def test_equivalence_loss_initialization(self):
        """Test equivalence loss initialization."""
        loss_fn = EquivalenceLoss(lambda_reg=0.1, reduction='mean')
        assert loss_fn.lambda_reg == 0.1
        assert isinstance(loss_fn.fidelity_loss, FidelityLoss)
        assert isinstance(loss_fn.regularization_loss, RegularizationLoss)
    
    def test_equivalence_loss_output_structure(self, equiv_loss):
        """Test that loss output has correct structure."""
        batch_size, seq_len, hidden_size = 2, 10, 256
        npt_output = torch.randn(batch_size, seq_len, hidden_size)
        original_output = torch.randn(batch_size, seq_len, hidden_size)
        v_a_list = [torch.randn(batch_size, seq_len, hidden_size) for _ in range(3)]
        v_b_list = [torch.randn(batch_size, seq_len, 1024) for _ in range(3)]
        
        result = equiv_loss(npt_output, original_output, v_a_list, v_b_list)
        
        assert isinstance(result, LossOutput)
        assert hasattr(result, 'total_loss')
        assert hasattr(result, 'fidelity_loss')
        assert hasattr(result, 'regularization_loss')
        assert hasattr(result, 'metrics')
        assert isinstance(result.metrics, dict)
    
    def test_equivalence_loss_computation(self, equiv_loss):
        """Test that total loss is correctly computed."""
        batch_size, seq_len, hidden_size = 2, 10, 256
        npt_output = torch.randn(batch_size, seq_len, hidden_size)
        original_output = torch.randn(batch_size, seq_len, hidden_size)
        v_a_list = [torch.randn(batch_size, seq_len, hidden_size) for _ in range(3)]
        v_b_list = [torch.randn(batch_size, seq_len, 1024) for _ in range(3)]
        
        result = equiv_loss(npt_output, original_output, v_a_list, v_b_list)
        
        # Verify total loss computation
        expected_total = result.fidelity_loss + equiv_loss.lambda_reg * result.regularization_loss
        assert torch.allclose(result.total_loss, expected_total)
    
    def test_equivalence_loss_metrics(self, equiv_loss):
        """Test that metrics are correctly computed."""
        batch_size, seq_len, hidden_size = 2, 10, 256
        npt_output = torch.randn(batch_size, seq_len, hidden_size)
        original_output = torch.randn(batch_size, seq_len, hidden_size)
        v_a_list = [torch.randn(batch_size, seq_len, hidden_size) for _ in range(3)]
        v_b_list = [torch.randn(batch_size, seq_len, 1024) for _ in range(3)]
        
        result = equiv_loss(npt_output, original_output, v_a_list, v_b_list)
        
        # Check metrics exist
        assert 'fidelity_loss' in result.metrics
        assert 'regularization_loss' in result.metrics
        assert 'lambda_reg' in result.metrics
        assert 'avg_v_a_norm' in result.metrics
        assert 'avg_v_b_norm' in result.metrics
        assert 'max_output_diff' in result.metrics
        assert 'mean_output_diff' in result.metrics
        
        # Check metric values are reasonable
        assert result.metrics['fidelity_loss'] >= 0
        assert result.metrics['regularization_loss'] >= 0
        assert result.metrics['lambda_reg'] == 0.01
        assert result.metrics['avg_v_a_norm'] >= 0
        assert result.metrics['avg_v_b_norm'] >= 0
        assert result.metrics['max_output_diff'] >= 0
        assert result.metrics['mean_output_diff'] >= 0
    
    def test_equivalence_loss_zero_regularization(self):
        """Test with zero regularization weight."""
        loss_fn = EquivalenceLoss(lambda_reg=0.0, reduction='mean')
        
        batch_size, seq_len, hidden_size = 2, 10, 256
        npt_output = torch.randn(batch_size, seq_len, hidden_size)
        original_output = torch.randn(batch_size, seq_len, hidden_size)
        v_a_list = [torch.randn(batch_size, seq_len, hidden_size) for _ in range(3)]
        v_b_list = [torch.randn(batch_size, seq_len, 1024) for _ in range(3)]
        
        result = loss_fn(npt_output, original_output, v_a_list, v_b_list)
        
        # Total loss should equal fidelity loss when lambda=0
        assert torch.allclose(result.total_loss, result.fidelity_loss)


class TestParallelForwardHelper:
    """Test suite for ParallelForwardHelper."""
    
    @pytest.fixture
    def small_config(self):
        """Small configuration for testing."""
        config = LlamaConfig(
            hidden_size=256,
            intermediate_size=1024,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4,
            vocab_size=1000,
        )
        config._attn_implementation = "eager"
        return config
    
    @pytest.fixture
    def npt_model(self, small_config):
        """Create an NPT model with some layers converted."""
        model = NPTLlamaModel(small_config)
        npt_config = NPTConfig(
            layers_to_convert=[1, 2],
            np_rank=32,
            np_init_scale=0.01
        )
        model.convert_to_npt(npt_config)
        return model
    
    @pytest.fixture
    def helper(self, npt_model):
        """Create a parallel forward helper."""
        return ParallelForwardHelper(npt_model)
    
    @pytest.fixture
    def sample_input(self, small_config):
        """Create sample input."""
        batch_size = 2
        seq_len = 10
        return torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
    
    def test_helper_initialization(self, helper, npt_model):
        """Test helper initialization."""
        assert helper.model is npt_model
    
    def test_parallel_forward(self, helper, sample_input):
        """Test parallel forward pass."""
        npt_output, original_output, v_a_list, v_b_list = helper.forward(
            sample_input,
            collect_np_outputs=True
        )
        
        # Check output shapes
        batch_size, seq_len = sample_input.shape
        hidden_size = helper.model.config.hidden_size
        assert npt_output.shape == (batch_size, seq_len, hidden_size)
        assert original_output.shape == (batch_size, seq_len, hidden_size)
        
        # Check NP outputs collected (2 NPT layers)
        assert len(v_a_list) == 2
        assert len(v_b_list) == 2
        
        # Check v_a and v_b shapes
        for v_a in v_a_list:
            assert v_a.shape == (batch_size, seq_len, hidden_size)
        for v_b in v_b_list:
            assert v_b.shape == (batch_size, seq_len, helper.model.config.intermediate_size)
    
    def test_parallel_forward_without_np_collection(self, helper, sample_input):
        """Test parallel forward without collecting NP outputs."""
        npt_output, original_output, v_a_list, v_b_list = helper.forward(
            sample_input,
            collect_np_outputs=False
        )
        
        # Check outputs exist
        assert npt_output is not None
        assert original_output is not None
        
        # NP outputs should be empty
        assert len(v_a_list) == 0
        assert len(v_b_list) == 0
    
    def test_outputs_different(self, helper, sample_input):
        """Test that NPT and original outputs are different initially."""
        npt_output, original_output, _, _ = helper.forward(
            sample_input,
            collect_np_outputs=False
        )
        
        # Outputs should be different (untrained NP components)
        assert not torch.allclose(npt_output, original_output, rtol=1e-3)
    
    def test_compute_layer_outputs(self, helper, sample_input):
        """Test computing outputs for a specific layer."""
        layer_idx = 1  # This is an NPT layer
        
        npt_output, original_output, v_a, v_b = helper.compute_layer_outputs(
            sample_input,
            layer_idx
        )
        
        # Check output shapes
        batch_size, seq_len = sample_input.shape
        hidden_size = helper.model.config.hidden_size
        assert npt_output.shape == (batch_size, seq_len, hidden_size)
        assert original_output.shape == (batch_size, seq_len, hidden_size)
        
        # Check v_a and v_b shapes
        assert v_a.shape == (batch_size, seq_len, hidden_size)
        assert v_b.shape == (batch_size, seq_len, helper.model.config.intermediate_size)
    
    def test_compute_layer_outputs_non_npt_layer(self, helper, sample_input):
        """Test that non-NPT layer raises an error."""
        layer_idx = 0  # This is not an NPT layer
        
        with pytest.raises(ValueError, match="not an NPT layer"):
            helper.compute_layer_outputs(sample_input, layer_idx)
    
    def test_mode_restoration(self, helper, sample_input):
        """Test that model mode is restored after parallel forward."""
        # Set model to NPT mode initially
        helper.model.set_npt_mode(True)
        
        # Run parallel forward
        helper.forward(sample_input)
        
        # Check mode is restored
        for layer in helper.model.npt_layers.values():
            assert layer.use_npt == True
        
        # Set model to standard mode initially
        helper.model.set_npt_mode(False)
        
        # Run parallel forward
        helper.forward(sample_input)
        
        # Check mode is restored
        for layer in helper.model.npt_layers.values():
            assert layer.use_npt == False


class TestIntegration:
    """Integration tests for loss functions with NPT model."""
    
    @pytest.fixture
    def small_config(self):
        """Small configuration for testing."""
        config = LlamaConfig(
            hidden_size=256,
            intermediate_size=1024,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4,
            vocab_size=1000,
        )
        config._attn_implementation = "eager"
        return config
    
    @pytest.fixture
    def npt_model(self, small_config):
        """Create an NPT model with all layers converted."""
        model = NPTLlamaModel(small_config)
        npt_config = NPTConfig(
            convert_all=True,
            np_rank=32,
            np_init_scale=0.01
        )
        model.convert_to_npt(npt_config)
        return model
    
    def test_end_to_end_loss_computation(self, npt_model):
        """Test complete loss computation pipeline."""
        # Create helper and loss function
        helper = ParallelForwardHelper(npt_model)
        loss_fn = EquivalenceLoss(lambda_reg=0.01)
        
        # Create input
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, npt_model.config.vocab_size, (batch_size, seq_len))
        
        # Run parallel forward
        npt_output, original_output, v_a_list, v_b_list = helper.forward(
            input_ids,
            collect_np_outputs=True
        )
        
        # Compute loss
        loss_output = loss_fn(npt_output, original_output, v_a_list, v_b_list)
        
        # Check loss is computed
        assert loss_output.total_loss > 0
        assert loss_output.fidelity_loss > 0
        assert loss_output.regularization_loss > 0
        
        # Check gradients can be computed
        loss_output.total_loss.backward()
        
        # Check gradients flow to NP components
        for layer in npt_model.npt_layers.values():
            for param in layer.np_component.parameters():
                assert param.grad is not None
                assert torch.any(param.grad != 0)
    
    def test_loss_with_frozen_base(self, npt_model):
        """Test loss computation with frozen base parameters."""
        # Freeze base parameters
        npt_model.freeze_base_parameters()
        
        # Create helper and loss function
        helper = ParallelForwardHelper(npt_model)
        loss_fn = EquivalenceLoss(lambda_reg=0.01)
        
        # Create input
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, npt_model.config.vocab_size, (batch_size, seq_len))
        
        # Run parallel forward
        npt_output, original_output, v_a_list, v_b_list = helper.forward(
            input_ids,
            collect_np_outputs=True
        )
        
        # Compute loss
        loss_output = loss_fn(npt_output, original_output, v_a_list, v_b_list)
        
        # Compute gradients
        loss_output.total_loss.backward()
        
        # Check only NP components have gradients
        for name, param in npt_model.named_parameters():
            if 'np_component' in name:
                if param.requires_grad:
                    assert param.grad is not None
            else:
                # Base parameters should not have gradients
                assert not param.requires_grad or param.grad is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])