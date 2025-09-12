"""
Specialized loss functions for single-layer NPT training.

These losses are designed to address the unique challenge of training a single NPT layer
to learn the complex transformation: MLP_mod(h) = attn + MLP(h + attn)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class SingleLayerLossOutput:
    """Output structure for single-layer loss computation."""
    total_loss: torch.Tensor
    direct_mlp_loss: torch.Tensor
    attention_encoding_loss: torch.Tensor
    fidelity_loss: torch.Tensor
    regularization_loss: torch.Tensor
    metrics: Dict[str, float]


class DirectMLPSupervisionLoss(nn.Module):
    """
    Core loss for single-layer NPT training.
    
    This loss directly supervises what the modulated MLP should output:
    MLP_mod(h) = attn + MLP(h + attn)
    
    This is THE CRITICAL LOSS for single-layer training as it provides
    direct supervision on the exact transformation needed.
    """
    
    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = normalize
    
    def forward(
        self,
        mlp_modulated_output: torch.Tensor,
        attention_output: torch.Tensor,
        original_mlp_with_attention: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute direct MLP supervision loss.
        
        Args:
            mlp_modulated_output: Output from modulated MLP(h)
            attention_output: Attention output that was lost without residual
            original_mlp_with_attention: Original MLP(h + attn) output
        
        Returns:
            Direct supervision loss
        """
        # The target is what the modulated MLP should produce
        target = attention_output + original_mlp_with_attention
        
        if self.normalize:
            # Normalize by magnitude to handle scale differences
            norm_factor = target.norm() / (target.numel() ** 0.5)
            loss = F.mse_loss(mlp_modulated_output, target) / (norm_factor + 1e-6)
        else:
            loss = F.mse_loss(mlp_modulated_output, target)
        
        return loss



class DirectSupervisionLoss(nn.Module):
    """
    Simplified loss focusing purely on direct MLP supervision.
    
    This loss trusts that if the model needs to encode attention in v_a
    to minimize the direct MLP loss, it will learn to do so naturally.
    No forced attention encoding.
    """
    
    def __init__(
        self,
        direct_mlp_weight: float = 1.0,
        fidelity_weight: float = 0.1,
        regularization_weight: float = 0.001,
    ):
        super().__init__()
        self.direct_mlp_loss = DirectMLPSupervisionLoss(normalize=True)
        self.direct_mlp_weight = direct_mlp_weight
        self.fidelity_weight = fidelity_weight
        self.regularization_weight = regularization_weight
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        current_step: int = 0  # Kept for compatibility but not used
    ) -> SingleLayerLossOutput:
        """
        Compute direct supervision loss.
        
        Args:
            outputs: Dictionary containing model outputs
            current_step: Not used, kept for compatibility
        
        Returns:
            SingleLayerLossOutput with loss components
        """
        # PRIMARY: Direct supervision of the exact transformation needed
        direct_mlp_loss = self.direct_mlp_loss(
            outputs['mlp_modulated'],
            outputs['attention_output'],
            outputs['original_mlp_with_attention']
        )
        
        # End-to-end fidelity
        fidelity_loss = F.mse_loss(
            outputs['npt_final'],
            outputs['original_final']
        )
        
        # Handle both rank-1 and rank-k for regularization
        v_a = outputs['v_a']
        v_b = outputs['v_b']
        
        if v_a.dim() == 4:  # rank-k
            v_a_reg = v_a.pow(2).mean()
            v_b_reg = v_b.pow(2).mean()
        else:  # rank-1
            v_a_reg = v_a.pow(2).mean()
            v_b_reg = v_b.pow(2).mean()
        
        regularization_loss = v_a_reg + v_b_reg
        
        # Simple weighted sum
        total_loss = (
            self.direct_mlp_weight * direct_mlp_loss +
            self.fidelity_weight * fidelity_loss +
            self.regularization_weight * regularization_loss
        )
        
        # Compute metrics
        with torch.no_grad():
            # Handle rank-k for metrics
            if v_a.dim() == 4:
                v_a_norm_val = v_a.norm().item()
                v_b_norm_val = v_b.norm().item()
            else:
                v_a_norm_val = v_a.norm().item()
                v_b_norm_val = v_b.norm().item()
            
            metrics = {
                'direct_mlp_loss': direct_mlp_loss.item(),
                'fidelity_loss': fidelity_loss.item(),
                'regularization_loss': regularization_loss.item(),
                'v_a_norm': v_a_norm_val,
                'v_b_norm': v_b_norm_val
            }
        
        return SingleLayerLossOutput(
            total_loss=total_loss,
            direct_mlp_loss=direct_mlp_loss,
            attention_encoding_loss=torch.tensor(0.0),  # Placeholder
            fidelity_loss=fidelity_loss,
            regularization_loss=regularization_loss,
            metrics=metrics
        )


class GradientScaler:
    """
    Scale gradients for single NPT layer to accelerate learning.
    
    Single-layer NPT needs stronger gradients since it's the only
    layer learning the complex transformation.
    """
    
    def __init__(self, scale_factor: float = 10.0):
        self.scale_factor = scale_factor
    
    def scale_npt_gradients(self, model, layer_indices: List[int]):
        """
        Scale gradients for specified NPT layers.
        
        Args:
            model: The NPT model
            layer_indices: Indices of NPT layers to scale
        """
        for idx in layer_indices:
            if idx < len(model.model.layers):
                layer = model.model.layers[idx]
                if hasattr(layer, 'np_component'):
                    for param in layer.np_component.parameters():
                        if param.grad is not None:
                            param.grad *= self.scale_factor
