"""
Multi-layer loss functions for NPT training.

These losses extend the successful single-layer approach to multiple NPT layers,
with direct MLP supervision per layer and progressive training support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class MultiLayerLossOutput:
    """Output structure for multi-layer loss computation."""
    total_loss: torch.Tensor
    direct_mlp_loss: torch.Tensor
    attention_encoding_loss: torch.Tensor
    layer_fidelity_loss: torch.Tensor
    final_output_loss: torch.Tensor
    regularization_loss: torch.Tensor
    per_layer_losses: Dict[int, Dict[str, float]]
    metrics: Dict[str, float]


class MultiLayerDirectSupervisionLoss(nn.Module):
    """
    Direct MLP supervision for each NPT layer.
    
    For each NPT layer, supervises: MLP_mod(h) = attn + MLP(h + attn)
    This is the key insight from single-layer success.
    """
    
    def __init__(self, weight_by_depth: bool = True):
        super().__init__()
        self.weight_by_depth = weight_by_depth
    
    def forward(
        self,
        layer_outputs: Dict[int, Dict[str, torch.Tensor]],
        active_layers: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, Dict[int, float]]:
        """
        Compute direct MLP supervision loss for each layer.
        
        Args:
            layer_outputs: Dict mapping layer index to outputs containing:
                - 'attention': Attention output
                - 'original_mlp_with_attn': Original MLP(h + attn)
                - 'modulated_mlp': Modulated MLP output
                - 'residual': Input residual to the layer
            active_layers: List of layer indices to compute loss for (for progressive training)
        
        Returns:
            Total loss and per-layer losses
        """
        total_loss = 0.0
        per_layer_losses = {}
        
        # Use all layers if not specified
        if active_layers is None:
            active_layers = list(layer_outputs.keys())
        
        num_layers = len(layer_outputs)
        
        for layer_idx in active_layers:
            if layer_idx not in layer_outputs:
                continue
                
            outputs = layer_outputs[layer_idx]
            
            # Target: what the modulated MLP should produce
            # MLP_mod(residual) should equal attn + MLP(residual + attn)
            target = outputs['attention'] + outputs['original_mlp_with_attn']
            actual = outputs['modulated_mlp']
            
            # MSE loss for this layer
            layer_loss = F.mse_loss(actual, target)
            
            # Weight by layer depth if enabled
            # Lower layers (0, 1, 2...) get higher weight as they're foundational
            if self.weight_by_depth:
                depth_weight = 1.0 / (1.0 + layer_idx * 0.05)  # Gentle decay
            else:
                depth_weight = 1.0
            
            weighted_loss = layer_loss * depth_weight
            total_loss = total_loss + weighted_loss
            per_layer_losses[layer_idx] = layer_loss.item()
        
        # Average over active layers
        if len(active_layers) > 0:
            total_loss = total_loss / len(active_layers)
        
        return total_loss, per_layer_losses


class PerLayerAttentionEncodingLoss(nn.Module):
    """
    Ensures each NPT layer's v_a encodes its attention output.
    
    Critical for multi-layer training as each layer must preserve
    its own attention information.
    """
    
    def __init__(self, use_cosine: bool = True):
        super().__init__()
        self.use_cosine = use_cosine
    
    def forward(
        self,
        layer_outputs: Dict[int, Dict[str, torch.Tensor]],
        active_layers: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, Dict[int, float]]:
        """
        Compute attention encoding loss for each layer.
        
        Args:
            layer_outputs: Dict with 'v_a' and 'attention' for each layer
            active_layers: Layers to compute loss for
        
        Returns:
            Total loss and per-layer similarities
        """
        total_loss = 0.0
        per_layer_similarities = {}
        
        if active_layers is None:
            active_layers = list(layer_outputs.keys())
        
        for layer_idx in active_layers:
            if layer_idx not in layer_outputs:
                continue
                
            outputs = layer_outputs[layer_idx]
            v_a = outputs['v_a']
            attention = outputs['attention']
            
            # MSE loss
            mse_loss = F.mse_loss(v_a, attention)
            
            # Cosine similarity loss (optional)
            if self.use_cosine:
                v_a_flat = v_a.view(-1, v_a.size(-1))
                attn_flat = attention.view(-1, attention.size(-1))
                
                v_a_norm = F.normalize(v_a_flat, p=2, dim=-1)
                attn_norm = F.normalize(attn_flat, p=2, dim=-1)
                
                cosine_sim = (v_a_norm * attn_norm).sum(dim=-1).mean()
                cosine_loss = 1 - cosine_sim
                
                layer_loss = 0.5 * mse_loss + 0.5 * cosine_loss
                per_layer_similarities[layer_idx] = cosine_sim.item()
            else:
                layer_loss = mse_loss
                per_layer_similarities[layer_idx] = 0.0
            
            # Weight lower layers more (they're foundational)
            depth_weight = 1.0 / (1.0 + layer_idx * 0.1)
            total_loss = total_loss + layer_loss * depth_weight
        
        if len(active_layers) > 0:
            total_loss = total_loss / len(active_layers)
        
        return total_loss, per_layer_similarities


class LayerWiseFidelityLoss(nn.Module):
    """
    Compares NPT and original outputs at each layer.
    
    Unlike direct MLP supervision, this compares the full layer outputs
    after residual connections.
    """
    
    def __init__(self, use_cosine: bool = True):
        super().__init__()
        self.use_cosine = use_cosine
    
    def forward(
        self,
        npt_hidden_states: Dict[int, torch.Tensor],
        original_hidden_states: Dict[int, torch.Tensor],
        active_layers: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, Dict[int, float]]:
        """
        Compute fidelity loss between NPT and original layer outputs.
        """
        total_loss = 0.0
        per_layer_losses = {}
        
        if active_layers is None:
            active_layers = list(npt_hidden_states.keys())
        
        for layer_idx in active_layers:
            if layer_idx not in npt_hidden_states or layer_idx not in original_hidden_states:
                continue
            
            npt_out = npt_hidden_states[layer_idx]
            orig_out = original_hidden_states[layer_idx]
            
            # MSE loss
            mse = F.mse_loss(npt_out, orig_out)
            
            if self.use_cosine:
                # Add cosine similarity
                npt_flat = npt_out.view(-1, npt_out.size(-1))
                orig_flat = orig_out.view(-1, orig_out.size(-1))
                
                npt_norm = F.normalize(npt_flat, p=2, dim=-1)
                orig_norm = F.normalize(orig_flat, p=2, dim=-1)
                
                cosine_sim = (npt_norm * orig_norm).sum(dim=-1).mean()
                cosine_loss = 1 - cosine_sim
                
                layer_loss = 0.5 * mse + 0.5 * cosine_loss
            else:
                layer_loss = mse
            
            per_layer_losses[layer_idx] = layer_loss.item()
            total_loss = total_loss + layer_loss
        
        if len(active_layers) > 0:
            total_loss = total_loss / len(active_layers)
        
        return total_loss, per_layer_losses


class ProgressiveLayerScheduler:
    """
    Manages progressive layer training strategy.
    
    Starts with lower layers (0, 1, 2...) and progressively adds higher layers.
    This provides a stable foundation as higher layers depend on lower ones.
    """
    
    def __init__(
        self,
        total_layers: List[int],
        warmup_steps: int = 1000,
        steps_per_stage: int = 2000
    ):
        """
        Initialize progressive scheduler.
        
        Args:
            total_layers: List of all NPT layer indices
            warmup_steps: Steps before starting progressive training
            steps_per_stage: Steps between adding new layers
        """
        self.total_layers = sorted(total_layers)  # Ensure sorted from low to high
        self.num_layers = len(total_layers)
        self.warmup_steps = warmup_steps
        self.steps_per_stage = steps_per_stage
        
        # Create schedule
        self.schedule = self._create_schedule()
        self.current_layers = []
    
    def _create_schedule(self) -> List[Tuple[int, List[int]]]:
        """
        Create progressive training schedule.
        
        Returns list of (step, active_layers) tuples.
        """
        schedule = []
        
        # Warmup: train only first 2 layers
        initial_layers = self.total_layers[:min(2, self.num_layers)]
        schedule.append((0, initial_layers))
        
        # Progressive stages: add 2-4 layers at a time
        layers_per_stage = max(2, self.num_layers // 8)  # Adaptive based on total
        
        current_step = self.warmup_steps
        current_layers = initial_layers.copy()
        
        while len(current_layers) < self.num_layers:
            # Add next batch of layers
            remaining = self.num_layers - len(current_layers)
            to_add = min(layers_per_stage, remaining)
            
            next_idx = len(current_layers)
            new_layers = self.total_layers[next_idx:next_idx + to_add]
            current_layers.extend(new_layers)
            
            schedule.append((current_step, current_layers.copy()))
            current_step += self.steps_per_stage
        
        return schedule
    
    def get_active_layers(self, step: int) -> List[int]:
        """
        Get list of active layers for current training step.
        
        Args:
            step: Current training step
        
        Returns:
            List of layer indices that should be trained
        """
        active_layers = self.schedule[0][1]  # Default to initial layers
        
        for schedule_step, layers in self.schedule:
            if step >= schedule_step:
                active_layers = layers
            else:
                break
        
        self.current_layers = active_layers
        return active_layers
    
    def get_stage_info(self, step: int) -> Dict[str, any]:
        """Get information about current training stage."""
        active_layers = self.get_active_layers(step)
        
        # Find current stage
        stage = 0
        for i, (schedule_step, _) in enumerate(self.schedule):
            if step >= schedule_step:
                stage = i
        
        return {
            'stage': stage,
            'active_layers': active_layers,
            'num_active': len(active_layers),
            'total_layers': self.num_layers,
            'progress': len(active_layers) / self.num_layers
        }


class MultiLayerEquivalenceLoss(nn.Module):
    """
    Combined loss for multi-layer NPT training.
    
    Integrates all loss components with dynamic weighting based on training stage.
    """
    
    def __init__(
        self,
        direct_mlp_weight: float = 2.0,
        attention_encoding_weight: float = 1.5,
        layer_fidelity_weight: float = 1.0,
        final_output_weight: float = 1.0,
        regularization_weight: float = 0.001,
        progressive_scheduler: Optional[ProgressiveLayerScheduler] = None
    ):
        super().__init__()
        
        # Loss components
        self.direct_mlp_loss = MultiLayerDirectSupervisionLoss(weight_by_depth=True)
        self.attention_encoding_loss = PerLayerAttentionEncodingLoss(use_cosine=True)
        self.layer_fidelity_loss = LayerWiseFidelityLoss(use_cosine=True)
        
        # Weights
        self.direct_mlp_weight = direct_mlp_weight
        self.attention_encoding_weight = attention_encoding_weight
        self.layer_fidelity_weight = layer_fidelity_weight
        self.final_output_weight = final_output_weight
        self.regularization_weight = regularization_weight
        
        # Progressive training
        self.progressive_scheduler = progressive_scheduler
    
    def get_dynamic_weights(self, step: int) -> Dict[str, float]:
        """
        Get dynamic loss weights based on training stage.
        
        Early stages focus on direct MLP supervision and attention encoding.
        Later stages balance all components.
        """
        # Base weights
        weights = {
            'direct_mlp': self.direct_mlp_weight,
            'attention_encoding': self.attention_encoding_weight,
            'layer_fidelity': self.layer_fidelity_weight,
            'final_output': self.final_output_weight,
            'regularization': self.regularization_weight
        }
        
        # Adjust based on training progress
        if self.progressive_scheduler:
            stage_info = self.progressive_scheduler.get_stage_info(step)
            progress = stage_info['progress']
            
            # Early stages: focus on direct supervision
            if progress < 0.3:
                weights['direct_mlp'] *= 2.0
                weights['attention_encoding'] *= 1.5
                weights['final_output'] *= 0.5
            # Mid stages: balance everything
            elif progress < 0.7:
                weights['layer_fidelity'] *= 1.2
            # Late stages: focus on final output
            else:
                weights['final_output'] *= 1.5
                weights['layer_fidelity'] *= 1.3
        
        return weights
    
    def compute_regularization(
        self,
        v_a_list: List[torch.Tensor],
        v_b_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """Compute L2 regularization on v_a and v_b vectors."""
        reg_loss = 0.0
        
        for v_a, v_b in zip(v_a_list, v_b_list):
            reg_loss = reg_loss + v_a.pow(2).mean() + v_b.pow(2).mean()
        
        if len(v_a_list) > 0:
            reg_loss = reg_loss / len(v_a_list)
        
        return reg_loss
    
    def forward(
        self,
        layer_outputs: Dict[int, Dict[str, torch.Tensor]],
        npt_hidden_states: Dict[int, torch.Tensor],
        original_hidden_states: Dict[int, torch.Tensor],
        npt_logits: torch.Tensor,
        original_logits: torch.Tensor,
        v_a_list: List[torch.Tensor],
        v_b_list: List[torch.Tensor],
        current_step: int = 0
    ) -> MultiLayerLossOutput:
        """
        Compute combined multi-layer loss.
        
        Args:
            layer_outputs: Detailed outputs for each NPT layer
            npt_hidden_states: Hidden states after each NPT layer
            original_hidden_states: Hidden states after each original layer
            npt_logits: Final NPT model logits
            original_logits: Final original model logits
            v_a_list: List of v_a vectors from each NPT layer
            v_b_list: List of v_b vectors from each NPT layer
            current_step: Current training step
        
        Returns:
            MultiLayerLossOutput with all loss components
        """
        # Get active layers for progressive training
        if self.progressive_scheduler:
            active_layers = self.progressive_scheduler.get_active_layers(current_step)
            stage_info = self.progressive_scheduler.get_stage_info(current_step)
        else:
            active_layers = list(layer_outputs.keys())
            stage_info = {'stage': 0, 'progress': 1.0}
        
        # Compute individual losses
        direct_mlp_loss, direct_mlp_per_layer = self.direct_mlp_loss(
            layer_outputs, active_layers
        )
        
        attention_loss, attention_similarities = self.attention_encoding_loss(
            layer_outputs, active_layers
        )
        
        layer_fidelity_loss, fidelity_per_layer = self.layer_fidelity_loss(
            npt_hidden_states, original_hidden_states, active_layers
        )
        
        # Final output loss (always computed)
        final_output_loss = F.mse_loss(npt_logits, original_logits)
        
        # Regularization
        regularization_loss = self.compute_regularization(v_a_list, v_b_list)
        
        # Get dynamic weights
        weights = self.get_dynamic_weights(current_step)
        
        # Combine losses
        total_loss = (
            weights['direct_mlp'] * direct_mlp_loss +
            weights['attention_encoding'] * attention_loss +
            weights['layer_fidelity'] * layer_fidelity_loss +
            weights['final_output'] * final_output_loss +
            weights['regularization'] * regularization_loss
        )
        
        # Prepare metrics
        metrics = {
            'stage': stage_info['stage'],
            'num_active_layers': len(active_layers),
            'training_progress': stage_info['progress'],
            'avg_attention_similarity': sum(attention_similarities.values()) / max(1, len(attention_similarities)),
            'avg_v_a_norm': sum(v_a.norm().item() for v_a in v_a_list) / max(1, len(v_a_list)),
            'avg_v_b_norm': sum(v_b.norm().item() for v_b in v_b_list) / max(1, len(v_b_list)),
        }
        
        return MultiLayerLossOutput(
            total_loss=total_loss,
            direct_mlp_loss=direct_mlp_loss,
            attention_encoding_loss=attention_loss,
            layer_fidelity_loss=layer_fidelity_loss,
            final_output_loss=final_output_loss,
            regularization_loss=regularization_loss,
            per_layer_losses={
                'direct_mlp': direct_mlp_per_layer,
                'attention_similarity': attention_similarities,
                'fidelity': fidelity_per_layer
            },
            metrics=metrics
        )