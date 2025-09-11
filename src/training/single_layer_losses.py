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


class AttentionEncodingLoss(nn.Module):
    """
    Force v_a to encode attention information.
    
    In single-layer NPT, v_a must capture the attention output
    since there's no residual connection to preserve it.
    """
    
    def __init__(self, use_cosine: bool = True, use_mse: bool = True):
        super().__init__()
        self.use_cosine = use_cosine
        self.use_mse = use_mse
    
    def forward(
        self,
        v_a: torch.Tensor,
        attention_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute attention encoding loss.
        
        Args:
            v_a: First vector from NP component (batch, seq, hidden)
            attention_output: Attention output to encode (batch, seq, hidden)
        
        Returns:
            Attention encoding loss
        """
        loss = 0.0
        
        if self.use_mse:
            # Direct MSE loss
            mse_loss = F.mse_loss(v_a, attention_output)
            # Normalize by attention magnitude
            norm_factor = attention_output.norm() / (attention_output.numel() ** 0.5)
            loss += mse_loss / (norm_factor + 1e-6)
        
        if self.use_cosine:
            # Cosine similarity loss - v_a should align with attention
            v_a_flat = v_a.view(-1, v_a.size(-1))
            attn_flat = attention_output.view(-1, attention_output.size(-1))
            
            v_a_norm = F.normalize(v_a_flat, p=2, dim=-1)
            attn_norm = F.normalize(attn_flat, p=2, dim=-1)
            
            cosine_sim = (v_a_norm * attn_norm).sum(dim=-1).mean()
            loss += (1 - cosine_sim) * 0.5  # Weight cosine loss
        
        return loss


class ModulationStrengthLoss(nn.Module):
    """
    Ensures the modulation v_b * (v_a @ h) approximates W_gate @ attn.
    
    This loss encourages the rank-1 update to have the right magnitude
    and direction to compensate for the missing attention input.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        v_a: torch.Tensor,
        v_b: torch.Tensor,
        hidden_states: torch.Tensor,
        gate_weight: torch.Tensor,
        attention_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute modulation strength loss.
        
        Args:
            v_a: First vector from NP component (batch, seq, hidden)
            v_b: Second vector from NP component (batch, seq, intermediate)
            hidden_states: Input hidden states (batch, seq, hidden)
            gate_weight: Gate projection weight matrix
            attention_output: Attention output
        
        Returns:
            Modulation strength loss
        """
        # Compute the modulation: v_b * (v_a @ h)
        v_a_dot_h = torch.sum(v_a * hidden_states, dim=-1, keepdim=True)
        modulation = v_b * v_a_dot_h
        
        # Compute what it should approximate: W_gate @ attn
        target = F.linear(attention_output, gate_weight)
        
        # MSE loss on the modulation matching the target
        loss = F.mse_loss(modulation, target)
        
        return loss


class SingleLayerEquivalenceLoss(nn.Module):
    """
    Combined loss for single-layer NPT training.
    
    This loss combines:
    1. Direct MLP supervision (most important)
    2. Attention encoding in v_a
    3. Final output fidelity
    4. Regularization on v_a and v_b
    """
    
    def __init__(
        self,
        direct_mlp_weight: float = 10.0,
        attention_encoding_weight: float = 5.0,
        fidelity_weight: float = 1.0,
        regularization_weight: float = 0.01,
        stage1_steps: int = 1000,
        use_modulation_loss: bool = False
    ):
        super().__init__()
        
        # Loss components
        self.direct_mlp_loss = DirectMLPSupervisionLoss(normalize=True)
        self.attention_encoding_loss = AttentionEncodingLoss(use_cosine=True, use_mse=True)
        self.modulation_strength_loss = ModulationStrengthLoss() if use_modulation_loss else None
        
        # Weights
        self.direct_mlp_weight = direct_mlp_weight
        self.attention_encoding_weight = attention_encoding_weight
        self.fidelity_weight = fidelity_weight
        self.regularization_weight = regularization_weight
        self.stage1_steps = stage1_steps
        
        self.current_step = 0
    
    def get_stage_weights(self, step: int) -> Dict[str, float]:
        """
        Dynamic loss weighting based on training stage.
        
        Stage 1 (0-1000): Focus on attention encoding
        Stage 2 (1000+): Full equivalence with direct MLP supervision
        """
        if step < self.stage1_steps:
            # Stage 1: Primarily train v_a to encode attention
            return {
                'direct_mlp': 0.1,
                'attention_encoding': 0.8,
                'fidelity': 0.1,
                'regularization': 0.01
            }
        elif step < self.stage1_steps * 3:
            # Stage 2: Transition to full equivalence
            progress = (step - self.stage1_steps) / (self.stage1_steps * 2)
            return {
                'direct_mlp': 0.1 + 0.4 * progress,
                'attention_encoding': 0.8 - 0.5 * progress,
                'fidelity': 0.1 + 0.2 * progress,
                'regularization': 0.01
            }
        else:
            # Stage 3: Full equivalence training
            return {
                'direct_mlp': self.direct_mlp_weight,
                'attention_encoding': self.attention_encoding_weight * 0.2,
                'fidelity': self.fidelity_weight,
                'regularization': self.regularization_weight
            }
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        current_step: int = 0
    ) -> SingleLayerLossOutput:
        """
        Compute combined loss for single-layer NPT.
        
        Args:
            outputs: Dictionary containing:
                - mlp_modulated: Modulated MLP output
                - attention_output: Attention output
                - original_mlp_with_attention: Original MLP(h+attn)
                - v_a: First NP vector (rank-1: 3D, rank-k: 4D)
                - v_b: Second NP vector (rank-1: 3D, rank-k: 4D)
                - npt_final: Final NPT output
                - original_final: Final original output
            current_step: Current training step for stage-based weighting
        
        Returns:
            SingleLayerLossOutput with all loss components
        """
        self.current_step = current_step
        weights = self.get_stage_weights(current_step)
        
        # Direct MLP supervision (critical for single layer)
        direct_mlp_loss = self.direct_mlp_loss(
            outputs['mlp_modulated'],
            outputs['attention_output'],
            outputs['original_mlp_with_attention']
        )
        
        # Handle both rank-1 and rank-k for attention encoding
        v_a = outputs['v_a']
        v_b = outputs['v_b']
        
        if v_a.dim() == 4:  # rank-k case
            # Option 1: Use first component for attention encoding (primary component)
            v_a_for_attention = v_a[:, :, 0, :]  # First rank component
            
            # Option 2 (alternative): Average all components
            # v_a_for_attention = v_a.mean(dim=2)
            
            # Option 3 (alternative): Weighted sum with learnable weights
            # This would require additional parameters in the loss function
            
            # Regularization for all components
            v_a_reg = v_a.pow(2).mean()
            v_b_reg = v_b.pow(2).mean()
        else:  # rank-1 case
            v_a_for_attention = v_a
            v_a_reg = v_a.pow(2).mean()
            v_b_reg = v_b.pow(2).mean()
        
        # Attention encoding loss using the selected v_a
        attention_loss = self.attention_encoding_loss(
            v_a_for_attention,
            outputs['attention_output']
        )
        
        # Final output fidelity
        fidelity_loss = F.mse_loss(
            outputs['npt_final'],
            outputs['original_final']
        )
        
        # Regularization on v_a and v_b
        regularization_loss = v_a_reg + v_b_reg
        
        # Weighted combination
        total_loss = (
            weights['direct_mlp'] * direct_mlp_loss +
            weights['attention_encoding'] * attention_loss +
            weights['fidelity'] * fidelity_loss +
            weights['regularization'] * regularization_loss
        )
        
        # Compute metrics
        with torch.no_grad():
            # Attention encoding quality - handle both rank-1 and rank-k
            if v_a.dim() == 4:  # rank-k
                v_a_for_metrics = v_a[:, :, 0, :]  # Use first component for metrics
                v_a_norm_val = v_a.norm().item()
                v_b_norm_val = v_b.norm().item()
                
                # Modulation strength for rank-k
                h_states = outputs.get('hidden_states', v_a[:, :, 0, :])
                h_expanded = h_states.unsqueeze(2)
                v_a_dot_h = torch.sum(v_a * h_expanded, dim=-1, keepdim=True)
                modulations = v_b * v_a_dot_h
                modulation_magnitude = modulations.norm().item()
            else:  # rank-1
                v_a_for_metrics = v_a
                v_a_norm_val = v_a.norm().item()
                v_b_norm_val = v_b.norm().item()
                
                # Modulation strength for rank-1
                v_a_dot_h = torch.sum(v_a * outputs.get('hidden_states', v_a), 
                                     dim=-1, keepdim=True)
                modulation_magnitude = (v_b * v_a_dot_h).norm().item()
            
            # Attention similarity using appropriate v_a
            v_a_flat = v_a_for_metrics.view(-1, v_a_for_metrics.size(-1))
            attn_flat = outputs['attention_output'].view(-1, outputs['attention_output'].size(-1))
            v_a_norm = F.normalize(v_a_flat, p=2, dim=-1)
            attn_norm = F.normalize(attn_flat, p=2, dim=-1)
            attention_similarity = (v_a_norm * attn_norm).sum(dim=-1).mean().item()
            
            metrics = {
                'direct_mlp_loss': direct_mlp_loss.item(),
                'attention_encoding_loss': attention_loss.item(),
                'fidelity_loss': fidelity_loss.item(),
                'regularization_loss': regularization_loss.item(),
                'v_a_attention_similarity': attention_similarity,
                'v_a_norm': v_a_norm_val,
                'v_b_norm': v_b_norm_val,
                'modulation_magnitude': modulation_magnitude,
                'stage': 1 if current_step < self.stage1_steps else 2,
            }
        
        return SingleLayerLossOutput(
            total_loss=total_loss,
            direct_mlp_loss=direct_mlp_loss,
            attention_encoding_loss=attention_loss,
            fidelity_loss=fidelity_loss,
            regularization_loss=regularization_loss,
            metrics=metrics
        )


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
                v_a_for_metrics = v_a[:, :, 0, :]  # First component
                v_a_norm_val = v_a.norm().item()
                v_b_norm_val = v_b.norm().item()
            else:
                v_a_for_metrics = v_a
                v_a_norm_val = v_a.norm().item()
                v_b_norm_val = v_b.norm().item()
            
            # Optional: compute attention similarity for monitoring (not loss)
            v_a_flat = v_a_for_metrics.view(-1, v_a_for_metrics.size(-1))
            attn_flat = outputs['attention_output'].view(-1, outputs['attention_output'].size(-1))
            v_a_norm = F.normalize(v_a_flat, p=2, dim=-1)
            attn_norm = F.normalize(attn_flat, p=2, dim=-1)
            attention_similarity = (v_a_norm * attn_norm).sum(dim=-1).mean().item()
            
            metrics = {
                'direct_mlp_loss': direct_mlp_loss.item(),
                'attention_encoding_loss': 0.0,  # Not used
                'fidelity_loss': fidelity_loss.item(),
                'regularization_loss': regularization_loss.item(),
                'v_a_attention_similarity': attention_similarity,  # Just for monitoring
                'v_a_norm': v_a_norm_val,
                'v_b_norm': v_b_norm_val,
                'modulation_magnitude': 0.0,  # Could compute if needed
                'stage': 0,  # No stages
            }
        
        return SingleLayerLossOutput(
            total_loss=total_loss,
            direct_mlp_loss=direct_mlp_loss,
            attention_encoding_loss=torch.tensor(0.0),  # Placeholder
            fidelity_loss=fidelity_loss,
            regularization_loss=regularization_loss,
            metrics=metrics
        )


class GuidedSupervisionLoss(nn.Module):
    """
    Loss with soft attention guidance but primary focus on direct supervision.
    
    Provides a hint that v_a should encode attention, but doesn't force it.
    The direct MLP loss is the primary driver.
    """
    
    def __init__(
        self,
        direct_mlp_weight: float = 100.0,  # Primary
        attention_encoding_weight: float = 1.0,  # Soft hint
        fidelity_weight: float = 10.0,
        regularization_weight: float = 0.001,
    ):
        super().__init__()
        self.direct_mlp_loss = DirectMLPSupervisionLoss(normalize=True)
        self.attention_encoding_loss = AttentionEncodingLoss(use_cosine=True, use_mse=True)
        
        self.direct_mlp_weight = direct_mlp_weight
        self.attention_encoding_weight = attention_encoding_weight
        self.fidelity_weight = fidelity_weight
        self.regularization_weight = regularization_weight
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        current_step: int = 0
    ) -> SingleLayerLossOutput:
        """
        Compute guided supervision loss.
        
        Fixed weights throughout training - no stages.
        """
        # Direct MLP supervision (primary)
        direct_mlp_loss = self.direct_mlp_loss(
            outputs['mlp_modulated'],
            outputs['attention_output'],
            outputs['original_mlp_with_attention']
        )
        
        # Handle rank-k
        v_a = outputs['v_a']
        v_b = outputs['v_b']
        
        if v_a.dim() == 4:  # rank-k
            v_a_for_attention = v_a[:, :, 0, :]  # First component
            v_a_reg = v_a.pow(2).mean()
            v_b_reg = v_b.pow(2).mean()
        else:
            v_a_for_attention = v_a
            v_a_reg = v_a.pow(2).mean()
            v_b_reg = v_b.pow(2).mean()
        
        # Soft attention guidance
        attention_loss = self.attention_encoding_loss(
            v_a_for_attention,
            outputs['attention_output']
        )
        
        # Fidelity
        fidelity_loss = F.mse_loss(
            outputs['npt_final'],
            outputs['original_final']
        )
        
        # Regularization
        regularization_loss = v_a_reg + v_b_reg
        
        # Fixed weighted combination - no stages!
        total_loss = (
            self.direct_mlp_weight * direct_mlp_loss +
            self.attention_encoding_weight * attention_loss +
            self.fidelity_weight * fidelity_loss +
            self.regularization_weight * regularization_loss
        )
        
        # Metrics
        with torch.no_grad():
            if v_a.dim() == 4:
                v_a_for_metrics = v_a[:, :, 0, :]
                v_a_norm_val = v_a.norm().item()
                v_b_norm_val = v_b.norm().item()
            else:
                v_a_for_metrics = v_a
                v_a_norm_val = v_a.norm().item()
                v_b_norm_val = v_b.norm().item()
            
            v_a_flat = v_a_for_metrics.view(-1, v_a_for_metrics.size(-1))
            attn_flat = outputs['attention_output'].view(-1, outputs['attention_output'].size(-1))
            v_a_norm = F.normalize(v_a_flat, p=2, dim=-1)
            attn_norm = F.normalize(attn_flat, p=2, dim=-1)
            attention_similarity = (v_a_norm * attn_norm).sum(dim=-1).mean().item()
            
            metrics = {
                'direct_mlp_loss': direct_mlp_loss.item(),
                'attention_encoding_loss': attention_loss.item(),
                'fidelity_loss': fidelity_loss.item(),
                'regularization_loss': regularization_loss.item(),
                'v_a_attention_similarity': attention_similarity,
                'v_a_norm': v_a_norm_val,
                'v_b_norm': v_b_norm_val,
                'modulation_magnitude': 0.0,
                'stage': 0,  # No stages
            }
        
        return SingleLayerLossOutput(
            total_loss=total_loss,
            direct_mlp_loss=direct_mlp_loss,
            attention_encoding_loss=attention_loss,
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