"""
Improved loss functions for NPT equivalence pre-training.

Key improvements:
1. Layer-wise supervision for better gradient flow
2. Cosine similarity for scale-invariant comparison
3. Adaptive regularization based on training dynamics
4. Spectral normalization for stability
5. Knowledge distillation on logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class ImprovedLossOutput:
    """Container for improved loss computation results."""
    total_loss: torch.Tensor
    fidelity_loss: torch.Tensor
    regularization_loss: torch.Tensor
    layer_losses: List[torch.Tensor]
    metrics: Dict[str, float]


class LayerwiseFidelityLoss(nn.Module):
    """
    Layer-wise fidelity loss with multiple similarity metrics.
    
    Key improvements:
    - Compares outputs at each layer, not just final
    - Uses cosine similarity for scale invariance
    - Includes intermediate supervision
    """
    
    def __init__(
        self,
        use_cosine: bool = True,
        use_mse: bool = True,
        cosine_weight: float = 0.5,
        layer_decay: float = 0.9  # Exponential decay for deeper layers
    ):
        super().__init__()
        self.use_cosine = use_cosine
        self.use_mse = use_mse
        self.cosine_weight = cosine_weight
        self.layer_decay = layer_decay
    
    def forward(
        self,
        npt_outputs: List[torch.Tensor],
        original_outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Compute layer-wise fidelity loss.
        
        Args:
            npt_outputs: List of outputs from each NPT layer
            original_outputs: List of outputs from corresponding original layers
        
        Returns:
            Total loss and per-layer losses
        """
        layer_losses = []
        
        for i, (npt_out, orig_out) in enumerate(zip(npt_outputs, original_outputs)):
            loss = 0.0
            
            if self.use_mse:
                # Scale-normalized MSE
                mse = F.mse_loss(npt_out, orig_out)
                # Normalize by output magnitude to handle scale differences
                norm_factor = orig_out.norm() / orig_out.numel()**0.5
                mse_normalized = mse / (norm_factor + 1e-6)
                loss += (1 - self.cosine_weight) * mse_normalized
            
            if self.use_cosine:
                # Cosine similarity loss (1 - cosine_similarity)
                # Flatten to (batch*seq, hidden)
                npt_flat = npt_out.view(-1, npt_out.size(-1))
                orig_flat = orig_out.view(-1, orig_out.size(-1))
                
                # Normalize vectors
                npt_norm = F.normalize(npt_flat, p=2, dim=-1)
                orig_norm = F.normalize(orig_flat, p=2, dim=-1)
                
                # Cosine similarity
                cosine_sim = (npt_norm * orig_norm).sum(dim=-1).mean()
                cosine_loss = 1 - cosine_sim
                loss += self.cosine_weight * cosine_loss
            
            # Apply layer decay (earlier layers get more weight)
            decay_factor = self.layer_decay ** i
            layer_losses.append(loss * decay_factor)
        
        total_loss = sum(layer_losses) / len(layer_losses)
        return total_loss, layer_losses


class AdaptiveRegularizationLoss(nn.Module):
    """
    Adaptive regularization that adjusts based on training dynamics.
    
    Key improvements:
    - Spectral normalization for stability
    - Adaptive scaling based on gradient magnitudes
    - Orthogonality encouragement
    """
    
    def __init__(
        self,
        base_lambda: float = 0.01,
        spectral_penalty: float = 0.1,
        orthogonal_penalty: float = 0.05,
        warmup_steps: int = 1000
    ):
        super().__init__()
        self.base_lambda = base_lambda
        self.spectral_penalty = spectral_penalty
        self.orthogonal_penalty = orthogonal_penalty
        self.warmup_steps = warmup_steps
        self.step = 0
    
    def compute_spectral_norm(self, v_a: torch.Tensor, v_b: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral norm of the rank-1 update.
        
        The spectral norm of outer(v_b, v_a) is ||v_a|| * ||v_b||
        """
        # Compute norms per token
        v_a_norm = v_a.norm(dim=-1)  # (batch, seq)
        v_b_norm = v_b.norm(dim=-1)  # (batch, seq)
        
        # Spectral norm of the rank-1 matrix
        spectral_norm = v_a_norm * v_b_norm
        
        return spectral_norm.mean()
    
    def compute_orthogonality_loss(self, v_a_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Encourage orthogonality between v_a vectors from different layers.
        This helps layers learn complementary updates.
        """
        if len(v_a_list) < 2:
            return torch.tensor(0.0)
        
        ortho_loss = 0.0
        count = 0
        
        for i in range(len(v_a_list)):
            for j in range(i + 1, len(v_a_list)):
                # Flatten to (batch*seq, hidden)
                v_i = v_a_list[i].view(-1, v_a_list[i].size(-1))
                v_j = v_a_list[j].view(-1, v_a_list[j].size(-1))
                
                # Normalize
                v_i_norm = F.normalize(v_i, p=2, dim=-1)
                v_j_norm = F.normalize(v_j, p=2, dim=-1)
                
                # Compute absolute cosine similarity
                cosine_sim = torch.abs((v_i_norm * v_j_norm).sum(dim=-1)).mean()
                ortho_loss += cosine_sim
                count += 1
        
        return ortho_loss / count if count > 0 else torch.tensor(0.0)
    
    def forward(
        self,
        v_a_list: List[torch.Tensor],
        v_b_list: List[torch.Tensor],
        current_step: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute adaptive regularization loss.
        """
        if current_step is not None:
            self.step = current_step
        
        # Warmup schedule for regularization
        warmup_factor = min(1.0, self.step / self.warmup_steps)
        adaptive_lambda = self.base_lambda * (2.0 - warmup_factor)  # Start high, decay to base
        
        # Basic L2 regularization
        l2_loss = 0.0
        spectral_loss = 0.0
        
        for v_a, v_b in zip(v_a_list, v_b_list):
            # L2 regularization with adaptive scaling
            l2_loss += (v_a.pow(2).mean() + v_b.pow(2).mean()) / 2
            
            # Spectral norm penalty
            if self.spectral_penalty > 0:
                spectral_loss += self.compute_spectral_norm(v_a, v_b)
        
        l2_loss /= len(v_a_list)
        spectral_loss /= len(v_a_list)
        
        # Orthogonality loss
        ortho_loss = self.compute_orthogonality_loss(v_a_list)
        
        # Combine all regularization terms
        total_reg = (adaptive_lambda * l2_loss + 
                    self.spectral_penalty * spectral_loss +
                    self.orthogonal_penalty * ortho_loss)
        
        return total_reg


class KnowledgeDistillationLoss(nn.Module):
    """
    Knowledge distillation loss on logits for better learning.
    
    Uses KL divergence between teacher and student logits.
    """
    
    def __init__(self, temperature: float = 3.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        npt_logits: torch.Tensor,
        original_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence between softened logits.
        """
        # Apply temperature scaling
        npt_soft = F.log_softmax(npt_logits / self.temperature, dim=-1)
        orig_soft = F.softmax(original_logits / self.temperature, dim=-1)
        
        # KL divergence
        kl_loss = F.kl_div(npt_soft, orig_soft, reduction='batchmean')
        
        # Scale by temperature squared (as in original distillation paper)
        return kl_loss * (self.temperature ** 2)


class ImprovedEquivalenceLoss(nn.Module):
    """
    Improved combined loss with all enhancements.
    
    Key features:
    1. Layer-wise supervision
    2. Multiple similarity metrics
    3. Adaptive regularization
    4. Knowledge distillation
    5. Gradient-aware scaling
    """
    
    def __init__(
        self,
        use_layerwise: bool = True,
        use_distillation: bool = True,
        base_lambda: float = 0.01,
        distillation_weight: float = 0.3,
        hidden_weight: float = 0.7
    ):
        super().__init__()
        
        self.use_layerwise = use_layerwise
        self.use_distillation = use_distillation
        self.distillation_weight = distillation_weight
        self.hidden_weight = hidden_weight
        
        # Initialize loss components
        self.layerwise_loss = LayerwiseFidelityLoss()
        self.regularization_loss = AdaptiveRegularizationLoss(base_lambda=base_lambda)
        self.distillation_loss = KnowledgeDistillationLoss()
        
        # For gradient scaling
        self.gradient_scale = 1.0
        self.ema_grad_norm = None
        self.ema_decay = 0.99
    
    def update_gradient_scale(self, grad_norm: float):
        """
        Update gradient scaling based on EMA of gradient norms.
        """
        if self.ema_grad_norm is None:
            self.ema_grad_norm = grad_norm
        else:
            self.ema_grad_norm = self.ema_decay * self.ema_grad_norm + (1 - self.ema_decay) * grad_norm
        
        # Scale inversely with gradient norm (stabilizes training)
        target_grad_norm = 1.0
        self.gradient_scale = target_grad_norm / (self.ema_grad_norm + 1e-6)
        # Use min/max instead of torch.clamp for scalar values
        self.gradient_scale = min(max(self.gradient_scale, 0.1), 10.0)
    
    def forward(
        self,
        npt_outputs: Dict[str, torch.Tensor],
        original_outputs: Dict[str, torch.Tensor],
        v_a_list: List[torch.Tensor],
        v_b_list: List[torch.Tensor],
        current_step: Optional[int] = None
    ) -> ImprovedLossOutput:
        """
        Compute improved combined loss.
        
        Args:
            npt_outputs: Dict with 'hidden_states' (list) and 'logits'
            original_outputs: Dict with 'hidden_states' (list) and 'logits'
            v_a_list: List of v_a vectors from NPT layers
            v_b_list: List of v_b vectors from NPT layers
            current_step: Current training step for adaptive scheduling
        
        Returns:
            ImprovedLossOutput with all loss components
        """
        total_loss = 0.0
        layer_losses = []
        
        # Layer-wise fidelity loss on hidden states
        if self.use_layerwise and 'hidden_states' in npt_outputs:
            fidelity_loss, layer_losses = self.layerwise_loss(
                npt_outputs['hidden_states'],
                original_outputs['hidden_states']
            )
            total_loss += self.hidden_weight * fidelity_loss
        else:
            # Fallback to simple MSE on final hidden states
            fidelity_loss = F.mse_loss(
                npt_outputs['hidden_states'][-1],
                original_outputs['hidden_states'][-1]
            )
            total_loss += self.hidden_weight * fidelity_loss
        
        # Knowledge distillation on logits
        distill_loss = torch.tensor(0.0)
        if self.use_distillation and 'logits' in npt_outputs:
            distill_loss = self.distillation_loss(
                npt_outputs['logits'],
                original_outputs['logits']
            )
            total_loss += self.distillation_weight * distill_loss
        
        # Adaptive regularization
        reg_loss = self.regularization_loss(v_a_list, v_b_list, current_step)
        total_loss += reg_loss
        
        # Apply gradient scaling
        total_loss = total_loss * self.gradient_scale
        
        # Compute metrics
        with torch.no_grad():
            metrics = {
                'fidelity_loss': fidelity_loss.item(),
                'distillation_loss': distill_loss.item() if self.use_distillation else 0.0,
                'regularization_loss': reg_loss.item(),
                'gradient_scale': self.gradient_scale.item() if isinstance(self.gradient_scale, torch.Tensor) else self.gradient_scale,
                'num_layers': len(v_a_list),
            }
            
            # Add per-layer metrics
            for i, layer_loss in enumerate(layer_losses):
                metrics[f'layer_{i}_loss'] = layer_loss.item()
        
        return ImprovedLossOutput(
            total_loss=total_loss,
            fidelity_loss=fidelity_loss,
            regularization_loss=reg_loss,
            layer_losses=layer_losses,
            metrics=metrics
        )


class ProgressiveUnfreezing:
    """
    Progressive unfreezing strategy for stable training.
    
    Train lower layers first, gradually unfreeze upper layers.
    """
    
    def __init__(
        self,
        model,
        num_layers: int,
        unfreeze_schedule: List[Tuple[int, int]]  # [(step, num_layers_to_unfreeze), ...]
    ):
        self.model = model
        self.num_layers = num_layers
        self.unfreeze_schedule = sorted(unfreeze_schedule, key=lambda x: x[0])
        self.current_unfrozen = 0
    
    def update(self, step: int):
        """
        Update which layers are unfrozen based on current step.
        """
        for schedule_step, num_to_unfreeze in self.unfreeze_schedule:
            if step >= schedule_step and self.current_unfrozen < num_to_unfreeze:
                # Unfreeze additional layers
                layers_to_unfreeze = list(range(self.current_unfrozen, num_to_unfreeze))
                
                for layer_idx in layers_to_unfreeze:
                    if layer_idx in self.model.npt_layers:
                        for param in self.model.npt_layers[layer_idx].np_component.parameters():
                            param.requires_grad = True
                
                self.current_unfrozen = num_to_unfreeze
                print(f"Step {step}: Unfroze {len(layers_to_unfreeze)} layers, total unfrozen: {self.current_unfrozen}")


class GradientMonitor:
    """
    Monitor and clip gradients per layer for stable training.
    """
    
    def __init__(self, model, clip_value: float = 1.0):
        self.model = model
        self.clip_value = clip_value
        self.grad_history = {}
    
    def clip_and_monitor(self):
        """
        Clip gradients per layer and track statistics.
        """
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # Track gradient norm
                grad_norm = param.grad.norm().item()
                
                if name not in self.grad_history:
                    self.grad_history[name] = []
                self.grad_history[name].append(grad_norm)
                
                # Adaptive clipping based on history
                if len(self.grad_history[name]) > 100:
                    # Use 95th percentile for clipping
                    sorted_grads = sorted(self.grad_history[name])
                    clip_val = sorted_grads[int(0.95 * len(sorted_grads))]
                    clip_val = max(clip_val, self.clip_value)
                else:
                    clip_val = self.clip_value
                
                # Clip gradient
                param.grad.data.clamp_(-clip_val, clip_val)
                
                # Keep history size manageable
                if len(self.grad_history[name]) > 1000:
                    self.grad_history[name] = self.grad_history[name][-500:]
    
    def get_stats(self) -> Dict[str, float]:
        """
        Get gradient statistics for monitoring.
        """
        stats = {}
        for name, history in self.grad_history.items():
            if history:
                stats[f"{name}_mean"] = sum(history[-10:]) / len(history[-10:])
                stats[f"{name}_max"] = max(history[-10:])
        return stats