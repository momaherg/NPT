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
    fidelity_loss: torch.Tensor
    regularization_loss: torch.Tensor
    metrics: Dict[str, float]


class DirectMLPSupervisionLoss(nn.Module):
    """
    Core loss for single-layer NPT training (NEW ARCHITECTURE).

    This loss directly supervises what the modulated MLP should output:
    MLP_mod(h) = attention + MLP(h + attn)

    Since attention is no longer in the residual, the modulation must
    output both the attention and the attention-conditioned MLP result.
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
        Compute direct MLP supervision loss (NEW ARCHITECTURE).

        Args:
            mlp_modulated_output: Output from modulated MLP(h)
            attention_output: Attention output (not in residual)
            original_mlp_with_attention: Original MLP(h + attn) output

        Returns:
            Direct supervision loss
        """
        # NEW TARGET: modulated MLP(h) should output attention + MLP(h + attn)
        # Since attention is not in the residual, modulation must output both
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
    Primary loss for single-layer NPT training (NEW ARCHITECTURE).

    This loss trains the modulated MLP to output:
    MLP_mod(h) = attention + MLP(h + attn)

    The modulation learns to output both attention and the
    attention-conditioned MLP result since attention is not in the residual.
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
    ) -> SingleLayerLossOutput:
        """
        Compute direct supervision loss.

        Args:
            outputs: Dictionary containing model outputs

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

        # Compute regularization
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
                v_a_for_metrics = v_a[:, :, 0, :]  # First component
                v_a_norm_val = v_a.norm().item()
                v_b_norm_val = v_b.norm().item()

                # Modulation strength for rank-k
                h_states = outputs.get('hidden_states', v_a[:, :, 0, :])
                h_expanded = h_states.unsqueeze(2)
                v_a_dot_h = torch.sum(v_a * h_expanded, dim=-1, keepdim=True)
                modulations = v_b * v_a_dot_h
                modulation_magnitude = modulations.norm().item()
            else:
                v_a_for_metrics = v_a
                v_a_norm_val = v_a.norm().item()
                v_b_norm_val = v_b.norm().item()

                # Modulation strength for rank-1
                v_a_dot_h = torch.sum(v_a * outputs.get('hidden_states', v_a),
                                     dim=-1, keepdim=True)
                modulation_magnitude = (v_b * v_a_dot_h).norm().item()

            # Compute attention similarity for monitoring (not part of loss)
            v_a_flat = v_a_for_metrics.view(-1, v_a_for_metrics.size(-1))
            attn_flat = outputs['attention_output'].view(-1, outputs['attention_output'].size(-1))
            v_a_norm = F.normalize(v_a_flat, p=2, dim=-1)
            attn_norm = F.normalize(attn_flat, p=2, dim=-1)
            attention_similarity = (v_a_norm * attn_norm).sum(dim=-1).mean().item()

            metrics = {
                'direct_mlp_loss': direct_mlp_loss.item(),
                'fidelity_loss': fidelity_loss.item(),
                'regularization_loss': regularization_loss.item(),
                'v_a_attention_similarity': attention_similarity,  # Just for monitoring
                'v_a_norm': v_a_norm_val,
                'v_b_norm': v_b_norm_val,
                'modulation_magnitude': modulation_magnitude,
            }

        return SingleLayerLossOutput(
            total_loss=total_loss,
            direct_mlp_loss=direct_mlp_loss,
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


def check_mode_collapse(v_a: torch.Tensor, v_b: torch.Tensor, threshold: float = 1e-6) -> bool:
    """
    Check if training has collapsed (v_a and v_b near zero).

    Args:
        v_a: First NP vector
        v_b: Second NP vector
        threshold: Norm threshold for collapse detection

    Returns:
        True if mode collapse detected
    """
    v_a_norm = v_a.norm().item()
    v_b_norm = v_b.norm().item()

    if v_a_norm < threshold and v_b_norm < threshold:
        print(f"WARNING: Mode collapse detected - v_a norm: {v_a_norm:.2e}, v_b norm: {v_b_norm:.2e}")
        return True

    return False