# NPT Loss Improvement Recommendations

## Overview

This document outlines key improvements to the NPT loss functions to address convergence challenges and improve training stability. The recommendations focus on solving the fundamental bottleneck where `v_a @ h` produces a single scalar per token that must encode complex information for MLP modulation.

## Core Challenge Analysis

The fundamental bottleneck in single-layer NPT training is that `v_a @ h` produces a **single scalar per token** that must encode all the complex information needed to modulate the MLP. This is equivalent to trying to compress an entire image into one number, making learning extremely difficult.

## Recommended Loss Improvements

### 1. Modulation Guidance Loss (High Impact)

**Problem**: The scalar `v_a @ h` has no direct supervision on what it should represent.

**Solution**: Guide the scalar to correlate with attention's relative importance.

```python
class ModulationGuidanceLoss(nn.Module):
    """Guide what the scalar v_a @ h should represent."""
    
    def forward(self, v_a, hidden_states, attention_output):
        # The scalar should correlate with attention's influence
        v_a_dot_h = torch.sum(v_a * hidden_states, dim=-1)  # (batch, seq)
        
        # Target: attention's relative importance at each position
        attention_importance = attention_output.norm(dim=-1) / hidden_states.norm(dim=-1)
        
        # Guide the scalar to represent this importance
        scalar_loss = F.mse_loss(torch.sigmoid(v_a_dot_h), 
                                 torch.sigmoid(attention_importance))
        
        # Also ensure variance in the scalars (avoid collapse)
        variance_loss = -torch.var(v_a_dot_h)
        
        return scalar_loss + 0.1 * variance_loss
```

### 2. Adaptive Loss Weighting (High Impact)

**Problem**: Static loss weights may not balance gradient magnitudes effectively.

**Solution**: Dynamically adjust weights based on gradient norms.

```python
class AdaptiveLossWeighting:
    """Dynamically balance loss components based on gradient magnitudes."""
    
    def __init__(self, num_losses=4):
        self.loss_weights = torch.ones(num_losses)
        self.grad_norms = torch.zeros(num_losses)
        self.alpha = 0.9  # EMA factor
        
    def update_weights(self, losses):
        # Compute gradient norms for each loss
        with torch.no_grad():
            for i, loss in enumerate(losses):
                grad = torch.autograd.grad(loss, loss, retain_graph=True)[0]
                self.grad_norms[i] = self.alpha * self.grad_norms[i] + (1-self.alpha) * grad.norm()
            
            # Inverse gradient weighting
            mean_norm = self.grad_norms.mean()
            self.loss_weights = mean_norm / (self.grad_norms + 1e-6)
            self.loss_weights = self.loss_weights / self.loss_weights.sum()
        
        return self.loss_weights
```

### 3. Decomposed Target Loss (High Impact)

**Problem**: Learning `attn + MLP(h+attn)` as one target is complex.

**Solution**: Decompose into attention reconstruction + MLP correction components.

```python
class DecomposedTargetLoss(nn.Module):
    """Decompose the complex target into learnable components."""
    
    def forward(self, mlp_modulated, attention_output, mlp_with_attention, hidden_states):
        # Decompose the target into two components
        # Component 1: Attention reconstruction
        attn_component = mlp_modulated[:, :, :attention_output.size(-1)]
        attn_loss = F.mse_loss(attn_component, attention_output)
        
        # Component 2: MLP correction
        mlp_component = mlp_modulated - attn_component
        mlp_target = mlp_with_attention  # Already computed as MLP(h+attn)
        mlp_loss = F.mse_loss(mlp_component, mlp_target)
        
        return 0.5 * attn_loss + 0.5 * mlp_loss
```

### 4. Information Bottleneck Loss (High Impact)

**Problem**: v_a may encode irrelevant information from hidden states.

**Solution**: Force v_a to be a compressed but informative representation.

```python
class InfoBottleneckLoss(nn.Module):
    """Force v_a to be informative about attention while compressed."""
    
    def __init__(self, beta=0.01):
        super().__init__()
        self.beta = beta
        
    def forward(self, v_a, attention_output, hidden_states):
        # Minimize I(v_a; hidden_states) while maximizing I(v_a; attention)
        
        # Approximate mutual information using correlation
        # Minimize correlation with h
        h_corr = F.cosine_similarity(v_a, hidden_states, dim=-1).abs().mean()
        
        # Maximize correlation with attention
        attn_corr = F.cosine_similarity(v_a, attention_output, dim=-1).abs().mean()
        
        return self.beta * h_corr - attn_corr
```

### 5. Projection Consistency Loss (Medium Impact)

**Problem**: v_b may learn inconsistent correction patterns.

**Solution**: Encourage consistent correction behavior.

```python
class ProjectionConsistencyLoss(nn.Module):
    """Ensure v_b learns consistent correction patterns."""
    
    def forward(self, v_b, mlp_gate_weight, attention_output):
        # v_b should approximate the average gate correction needed
        # when attention is added to hidden states
        
        # Expected gate correction
        gate_correction = F.linear(attention_output, mlp_gate_weight)
        
        # v_b should be similar across the sequence (with some variance)
        v_b_mean = v_b.mean(dim=1, keepdim=True)  # Average over sequence
        
        # Consistency loss
        consistency = F.mse_loss(v_b, v_b_mean.expand_as(v_b))
        
        # Alignment with expected correction
        alignment = F.cosine_similarity(v_b, gate_correction, dim=-1).mean()
        
        return consistency + (1 - alignment)
```

### 6. Curriculum Learning (Medium Impact)

**Problem**: Learning complex transformations on all data simultaneously is difficult.

**Solution**: Start with easier examples and gradually increase difficulty.

```python
class CurriculumScheduler:
    """Gradually increase training difficulty."""
    
    def get_difficulty_mask(self, input_ids, attention_output, step):
        # Start with shorter sequences, simpler patterns
        if step < 500:
            # Only train on positions where attention is small
            threshold = attention_output.norm(dim=-1).quantile(0.3)
            return attention_output.norm(dim=-1) < threshold
        elif step < 1000:
            # Medium difficulty
            threshold = attention_output.norm(dim=-1).quantile(0.7)
            return attention_output.norm(dim=-1) < threshold
        else:
            # Full difficulty - all positions
            return torch.ones_like(input_ids, dtype=torch.bool)
```

### 7. Orthogonality Regularization (Low-Medium Impact)

**Problem**: v_a may align with common hidden state directions, reducing effectiveness.

**Solution**: Encourage v_a to be orthogonal to principal components of h.

```python
class OrthogonalityLoss(nn.Module):
    """Encourage v_a to be orthogonal to common directions in h."""
    
    def forward(self, v_a, hidden_states):
        # Compute principal components of hidden states
        h_flat = hidden_states.view(-1, hidden_states.size(-1))
        u, s, v = torch.svd_lowrank(h_flat, q=10)
        
        # v_a should be orthogonal to top components of h
        v_a_flat = v_a.view(-1, v_a.size(-1))
        projections = v_a_flat @ v.T  # Project onto principal components
        
        # Penalize large projections
        orth_loss = projections.pow(2).mean()
        return orth_loss * 0.01
```

## Implementation Priority

### Phase 1 (Immediate - High Impact)
1. **Modulation Guidance Loss** - Directly addresses the scalar bottleneck
2. **Adaptive Loss Weighting** - Helps balance multiple objectives

### Phase 2 (Next - High Impact)  
3. **Information Bottleneck Loss** - Encourages efficient encoding
4. **Decomposed Target Loss** - Simplifies the learning objective

### Phase 3 (Later - Medium Impact)
5. **Projection Consistency Loss** - Improves v_b learning
6. **Curriculum Learning** - Gradual difficulty increase

### Phase 4 (Optional - Low-Medium Impact)
7. **Orthogonality Regularization** - Fine-tuning improvement

## Updated Loss Architecture

```python
class ImprovedSingleLayerLoss(nn.Module):
    """Enhanced loss with multiple improvement strategies."""
    
    def __init__(self, 
                 direct_mlp_weight=10.0,
                 attention_encoding_weight=5.0,
                 fidelity_weight=1.0,
                 regularization_weight=0.01,
                 modulation_guidance_weight=2.0,
                 info_bottleneck_weight=0.5,
                 stage1_steps=1000):
        super().__init__()
        
        # Existing losses
        self.direct_mlp_loss = DirectMLPSupervisionLoss()
        self.attention_encoding_loss = AttentionEncodingLoss()
        
        # New improvement losses
        self.modulation_guidance = ModulationGuidanceLoss()
        self.info_bottleneck = InfoBottleneckLoss(beta=0.01)
        self.decomposed_target = DecomposedTargetLoss()
        
        # Adaptive weighting
        self.adaptive_weighting = AdaptiveLossWeighting(num_losses=6)
        
        # Loss weights
        self.weights = {
            'direct_mlp': direct_mlp_weight,
            'attention_encoding': attention_encoding_weight,
            'fidelity': fidelity_weight,
            'regularization': regularization_weight,
            'modulation_guidance': modulation_guidance_weight,
            'info_bottleneck': info_bottleneck_weight,
        }
        
        self.stage1_steps = stage1_steps
        
    def forward(self, outputs, step):
        # Compute all loss components
        losses = [
            self.direct_mlp_loss(outputs['mlp_modulated'], 
                               outputs['attention_output'], 
                               outputs['original_mlp_with_attention']),
            self.attention_encoding_loss(outputs['v_a'], 
                                       outputs['attention_output']),
            self.modulation_guidance(outputs['v_a'], 
                                   outputs['hidden_states'], 
                                   outputs['attention_output']),
            self.info_bottleneck(outputs['v_a'], 
                               outputs['attention_output'], 
                               outputs['hidden_states']),
            F.mse_loss(outputs['npt_final'], outputs['original_final']),  # fidelity
            outputs['v_a'].pow(2).mean() + outputs['v_b'].pow(2).mean()  # regularization
        ]
        
        # Get adaptive weights (if enabled)
        if hasattr(self, 'use_adaptive_weighting') and self.use_adaptive_weighting:
            adaptive_weights = self.adaptive_weighting.update_weights(losses)
            total_loss = sum(w * l for w, l in zip(adaptive_weights, losses))
        else:
            # Use stage-based static weights
            stage_weights = self.get_stage_weights(step)
            weighted_losses = [
                stage_weights['direct_mlp'] * losses[0],
                stage_weights['attention_encoding'] * losses[1],
                stage_weights['modulation_guidance'] * losses[2],
                stage_weights['info_bottleneck'] * losses[3],
                stage_weights['fidelity'] * losses[4],
                stage_weights['regularization'] * losses[5]
            ]
            total_loss = sum(weighted_losses)
        
        return SingleLayerLossOutput(
            total_loss=total_loss,
            direct_mlp_loss=losses[0],
            attention_encoding_loss=losses[1],
            fidelity_loss=losses[4],
            regularization_loss=losses[5],
            metrics={
                'modulation_guidance_loss': losses[2].item(),
                'info_bottleneck_loss': losses[3].item(),
                'stage': 1 if step < self.stage1_steps else 2,
                # ... other metrics
            }
        )
```

## Expected Benefits

1. **Better Stage 1 Convergence**: Modulation guidance provides direct supervision for the scalar representation
2. **Balanced Training**: Adaptive weighting prevents any loss component from dominating
3. **Improved Representation**: Information bottleneck encourages efficient encoding
4. **Stable Learning**: Curriculum and consistency losses reduce training instability
5. **Faster Convergence**: Multiple complementary objectives provide richer supervision

## Monitoring Recommendations

When implementing these improvements, monitor:

1. **Modulation scalar variance** - Should increase over time, not collapse to zero
2. **Attention similarity trends** - Should improve in Stage 1 with new losses
3. **Loss component ratios** - Ensure no single loss dominates completely
4. **Gradient norms** - Check that adaptive weighting balances gradients effectively
5. **v_a/v_b norm evolution** - Should be stable, not exploding or vanishing

## Implementation Notes

- Start with **Modulation Guidance Loss** as it most directly addresses the core bottleneck
- **Adaptive Loss Weighting** can be added incrementally with a flag to enable/disable
- Consider **curriculum learning** for particularly challenging datasets
- Monitor computational overhead - some losses add modest computation cost
- Test improvements incrementally to isolate effects

This comprehensive approach addresses the fundamental challenges in NPT training while providing multiple pathways for the model to discover effective representations.