# NPT Input Modulation Architecture

## Problem Analysis

### Why Weight/Gate Modulation Failed

The original NPT approach applied rank-1 weight modulation to the MLP:
```
MLP_modulated = MLP(x) + v_b ⊗ v_a · x
```

This failed with SwiGLU because:

1. **SwiGLU's Multiplicative Gating**:
   ```python
   SwiGLU(x) = down_proj(silu(gate_proj(x)) * up_proj(x))
   ```
   The gate and up projections interact multiplicatively, making additive weight modulation inappropriate.

2. **Non-linear Activation**: The `silu` activation creates complex non-linearities that rank-1 updates can't properly modulate.

3. **Three Separate Projections**: Gate, up, and down projections each need different modulation patterns, but rank-1 update applies uniformly.

## New Approach: Input Modulation

Instead of modulating weights, we modulate the INPUT to the MLP:

### Architecture

```
Standard Transformer:
h → Attention → h + attention → MLP(norm(h + attention)) → output

NPT with Input Modulation:
h → Attention → h + attention → MLP(norm(h + attention) + v_adjustment) → output
                    ↓
                NP Component
                    ↓
                v_adjustment = v_a * scale
```

### Key Advantages

1. **Preserves MLP Structure**: The SwiGLU gating mechanism remains completely intact
2. **Simpler Learning**: Only need to learn input adjustments, not weight transformations
3. **Better Gradient Flow**: Standard MLP gradients, no complex modulation backprop
4. **Cleaner Abstraction**: Input adjustment is conceptually simpler than weight modulation

### Implementation Details

```python
# In NPTDecoderLayer._apply_modulated_mlp()

# Generate input adjustment (scale factor 0.1 for stability)
if v_a.dim() == 3:
    input_adjustment = v_a * 0.1  # Rank-1
else:
    input_adjustment = v_a.mean(dim=2) * 0.1  # Rank-k averaged

# Modulate input
modulated_input = hidden_states + input_adjustment

# Apply standard MLP (preserves SwiGLU completely)
output = self.mlp(modulated_input)
```

### Training Target

The loss now trains the input adjustment to make:
```
MLP(input + v_adjustment) ≈ MLP(input_teacher)
```

Where:
- `input` = layer_norm(h + attention) from student model
- `input_teacher` = corresponding teacher model state
- `v_adjustment` = learned adjustment based on attention patterns

## Training Configuration

### Recommended Settings

1. **Minimal NPT Layers**: Only 2-3 layers (e.g., layers 14-15 for 16-layer model)
2. **High-Rank Single Component**: rank=1024 with num_ranks=1
3. **Extended Teacher Curriculum**: 67%+ teacher supervision
4. **Conservative Learning Rate**: 2e-5 or lower
5. **High Fidelity Weight**: Prioritize output matching over direct supervision

### Why This Configuration Works

- **Few Layers**: Reduces error compounding
- **High Rank**: Provides expressiveness for input adjustments
- **Single Component**: Simpler than multi-rank for input modulation
- **Teacher Scaffolding**: Ensures stable learning of adjustments
- **Conservative Training**: Input modulation is sensitive, needs careful optimization

## Applications for Knowledge Injection

The input modulation approach is ideal for permanent knowledge injection:

1. **Context-Dependent Adjustments**: v_a learns to adjust MLP input based on attention context
2. **Surgical Modifications**: Can inject specific patterns into v_a to alter behavior
3. **Preserved Computation**: MLP internals unchanged, only input shifted
4. **Interpretable Modulation**: Input adjustments are in hidden state space

## Future Directions

1. **Adaptive Scaling**: Learn the scale factor instead of fixed 0.1
2. **Layer-Specific Strategies**: Different modulation approaches per layer
3. **Hybrid Approaches**: Combine input modulation with bias injection
4. **Multi-Stage Training**: First learn adjustments, then fine-tune for injection

## Running the New Architecture

```bash
# Train with input modulation
./train_input_modulation.sh

# Key metrics to monitor:
# - Fidelity loss (should decrease steadily)
# - Direct MLP loss (indicates adjustment quality)
# - Generation quality (perplexity, coherence)
# - v_a norms (should stabilize, not explode)
```

## Summary

Input modulation solves the fundamental incompatibility between rank-1 weight updates and SwiGLU's multiplicative gating. By adjusting the input rather than the weights, we preserve the MLP's delicate internal structure while still enabling context-dependent behavior modification. This approach is more stable, easier to train, and better suited for knowledge injection applications.