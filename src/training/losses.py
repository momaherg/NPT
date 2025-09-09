"""
Loss functions for NPT equivalence pre-training.

This module implements the loss functions used to train NP components
to functionally mimic the original transformer residual connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class LossOutput:
    """Container for loss computation results."""
    total_loss: torch.Tensor
    fidelity_loss: torch.Tensor
    regularization_loss: torch.Tensor
    metrics: Dict[str, float]


class FidelityLoss(nn.Module):
    """
    Fidelity loss for matching NPT output to original transformer output.
    
    This loss ensures that the NPT model produces outputs similar to the
    original transformer model, maintaining functional equivalence.
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        Initialize fidelity loss.
        
        Args:
            reduction: How to reduce the loss ('mean', 'sum', 'none')
        """
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        npt_output: torch.Tensor,
        original_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MSE loss between NPT and original outputs.
        
        Args:
            npt_output: Output from NPT model (batch_size, seq_len, hidden_size)
            original_output: Output from original model (batch_size, seq_len, hidden_size)
        
        Returns:
            Fidelity loss value
        """
        if npt_output.shape != original_output.shape:
            raise ValueError(
                f"Shape mismatch: NPT output {npt_output.shape} vs "
                f"original output {original_output.shape}"
            )
        
        # Compute MSE loss
        loss = F.mse_loss(npt_output, original_output, reduction=self.reduction)
        
        return loss


class RegularizationLoss(nn.Module):
    """
    Regularization loss for NP component outputs.
    
    Applies L2 regularization to v_a and v_b vectors to encourage
    low-magnitude weight updates and prevent instability.
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        Initialize regularization loss.
        
        Args:
            reduction: How to reduce the loss ('mean', 'sum', 'none')
        """
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        v_a_list: List[torch.Tensor],
        v_b_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute L2 regularization loss for v_a and v_b vectors.
        
        Args:
            v_a_list: List of v_a vectors from each NPT layer
            v_b_list: List of v_b vectors from each NPT layer
        
        Returns:
            Regularization loss value
        """
        if len(v_a_list) != len(v_b_list):
            raise ValueError(
                f"Mismatch in number of v_a ({len(v_a_list)}) and "
                f"v_b ({len(v_b_list)}) vectors"
            )
        
        if len(v_a_list) == 0:
            return torch.tensor(0.0, dtype=torch.float32)
        
        # Compute L2 norm for each vector pair
        reg_losses = []
        for v_a, v_b in zip(v_a_list, v_b_list):
            # L2 norm squared for v_a and v_b
            reg_loss = torch.sum(v_a ** 2) + torch.sum(v_b ** 2)
            
            if self.reduction == 'mean':
                # Average over all elements
                reg_loss = reg_loss / (v_a.numel() + v_b.numel())
            
            reg_losses.append(reg_loss)
        
        # Stack and reduce across layers
        reg_losses = torch.stack(reg_losses)
        
        if self.reduction == 'mean':
            return reg_losses.mean()
        elif self.reduction == 'sum':
            return reg_losses.sum()
        else:
            return reg_losses


class EquivalenceLoss(nn.Module):
    """
    Combined loss for NPT equivalence pre-training.
    
    Combines fidelity loss and regularization loss with a weighting factor λ.
    L_total = L_fidelity + λ * L_regularization
    """
    
    def __init__(
        self,
        lambda_reg: float = 0.01,
        reduction: str = 'mean'
    ):
        """
        Initialize equivalence loss.
        
        Args:
            lambda_reg: Weighting factor for regularization loss
            reduction: How to reduce the loss ('mean', 'sum')
        """
        super().__init__()
        self.lambda_reg = lambda_reg
        self.fidelity_loss = FidelityLoss(reduction=reduction)
        self.regularization_loss = RegularizationLoss(reduction=reduction)
    
    def forward(
        self,
        npt_output: torch.Tensor,
        original_output: torch.Tensor,
        v_a_list: List[torch.Tensor],
        v_b_list: List[torch.Tensor]
    ) -> LossOutput:
        """
        Compute combined equivalence loss.
        
        Args:
            npt_output: Output from NPT model
            original_output: Output from original model
            v_a_list: List of v_a vectors from NPT layers
            v_b_list: List of v_b vectors from NPT layers
        
        Returns:
            LossOutput containing total and component losses
        """
        # Compute fidelity loss
        fidelity_loss = self.fidelity_loss(npt_output, original_output)
        
        # Compute regularization loss
        reg_loss = self.regularization_loss(v_a_list, v_b_list)
        
        # Combine losses
        total_loss = fidelity_loss + self.lambda_reg * reg_loss
        
        # Compute metrics
        with torch.no_grad():
            # Average magnitude of v_a and v_b
            avg_v_a_norm = torch.mean(torch.stack([
                torch.norm(v_a) / v_a.numel()**0.5
                for v_a in v_a_list
            ])) if v_a_list else torch.tensor(0.0)
            
            avg_v_b_norm = torch.mean(torch.stack([
                torch.norm(v_b) / v_b.numel()**0.5
                for v_b in v_b_list
            ])) if v_b_list else torch.tensor(0.0)
            
            # Output difference metrics
            output_diff = torch.abs(npt_output - original_output)
            max_diff = output_diff.max().item()
            mean_diff = output_diff.mean().item()
        
        metrics = {
            'fidelity_loss': fidelity_loss.item(),
            'regularization_loss': reg_loss.item(),
            'lambda_reg': self.lambda_reg,
            'avg_v_a_norm': avg_v_a_norm.item(),
            'avg_v_b_norm': avg_v_b_norm.item(),
            'max_output_diff': max_diff,
            'mean_output_diff': mean_diff,
        }
        
        return LossOutput(
            total_loss=total_loss,
            fidelity_loss=fidelity_loss,
            regularization_loss=reg_loss,
            metrics=metrics
        )


class ParallelForwardHelper:
    """
    Helper class for running parallel forward passes through standard and NPT models.
    
    This helper facilitates the equivalence pre-training by running the same
    input through both model configurations and collecting necessary outputs.
    """
    
    def __init__(self, model):
        """
        Initialize the helper with an NPT model.
        
        Args:
            model: NPTLlamaModel instance
        """
        self.model = model
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        collect_np_outputs: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Run parallel forward passes in standard and NPT modes.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask (optional)
            position_ids: Position IDs (optional)
            collect_np_outputs: Whether to collect v_a and v_b vectors
        
        Returns:
            Tuple of (npt_output, original_output, v_a_list, v_b_list)
        """
        # Store current mode for each NPT layer
        original_modes = {}
        for idx, layer in self.model.npt_layers.items():
            original_modes[idx] = layer.use_npt
        
        # Run standard mode forward pass
        self.model.set_npt_mode(False)
        with torch.no_grad():
            original_outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True
            )
            # Get the final hidden states before the LM head
            original_output = original_outputs.hidden_states[-1]
        
        # Run NPT mode forward pass
        self.model.set_npt_mode(True)
        
        # Collect v_a and v_b if requested
        v_a_list = []
        v_b_list = []
        
        if collect_np_outputs:
            # Register hooks to collect NP component outputs
            handles = []
            
            def hook_fn(module, input, output):
                if isinstance(output, tuple) and len(output) == 2:
                    v_a, v_b = output
                    v_a_list.append(v_a)
                    v_b_list.append(v_b)
            
            # Add hooks to all NP components
            for layer in self.model.npt_layers.values():
                handle = layer.np_component.register_forward_hook(hook_fn)
                handles.append(handle)
        
        # Forward pass in NPT mode
        npt_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True
        )
        # Get the final hidden states before the LM head
        npt_output = npt_outputs.hidden_states[-1]
        
        # Remove hooks
        if collect_np_outputs:
            for handle in handles:
                handle.remove()
        
        # Restore original mode for each layer
        for idx, layer in self.model.npt_layers.items():
            layer.set_npt_mode(original_modes[idx])
        
        return npt_output, original_output, v_a_list, v_b_list
    
    def compute_layer_outputs(
        self,
        input_ids: torch.Tensor,
        layer_idx: int,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute outputs for a specific layer in both modes.
        
        This is useful for layer-wise analysis and debugging.
        
        Args:
            input_ids: Input token IDs
            layer_idx: Index of the layer to analyze
            attention_mask: Attention mask (optional)
        
        Returns:
            Tuple of (npt_layer_output, original_layer_output, v_a, v_b)
        """
        if layer_idx not in self.model.npt_layers:
            raise ValueError(f"Layer {layer_idx} is not an NPT layer")
        
        # Get embeddings
        inputs_embeds = self.model.model.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        
        # Create position embeddings
        batch_size, seq_len = input_ids.shape
        head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
        cos = torch.ones(batch_size, seq_len, head_dim, 
                        dtype=hidden_states.dtype, device=hidden_states.device)
        sin = torch.zeros(batch_size, seq_len, head_dim,
                         dtype=hidden_states.dtype, device=hidden_states.device)
        position_embeddings = (cos, sin)
        
        # Process through layers up to layer_idx
        for i, layer in enumerate(self.model.model.layers[:layer_idx]):
            # Standard forward through earlier layers
            if i < layer_idx:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_embeddings=position_embeddings,
                    use_cache=False,
                    output_attentions=False
                )
                if isinstance(layer_outputs, tuple):
                    hidden_states = layer_outputs[0]
                else:
                    hidden_states = layer_outputs
        
        # Get target layer
        target_layer = self.model.model.layers[layer_idx]
        
        # Run in standard mode
        target_layer.set_npt_mode(False)
        with torch.no_grad():
            original_outputs = target_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                use_cache=False,
                output_attentions=False
            )
            if isinstance(original_outputs, tuple):
                original_output = original_outputs[0]
            else:
                original_output = original_outputs
        
        # Run in NPT mode and collect v_a, v_b
        target_layer.set_npt_mode(True)
        v_a, v_b = None, None
        
        def hook_fn(module, input, output):
            nonlocal v_a, v_b
            if isinstance(output, tuple) and len(output) == 2:
                v_a, v_b = output
        
        handle = target_layer.np_component.register_forward_hook(hook_fn)
        
        npt_outputs = target_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            use_cache=False,
            output_attentions=False
        )
        if isinstance(npt_outputs, tuple):
            npt_output = npt_outputs[0]
        else:
            npt_output = npt_outputs
        
        handle.remove()
        
        return npt_output, original_output, v_a, v_b