# NPT Dual Modulation Architecture

## Overview
The dual modulation architecture addresses the limitations of single gate-only modulation by applying separate rank-1 updates to both the gate and up projections of the MLP layer.

## Architecture Changes

### Previous (Single Modulation)
```python
# Only modulates gate projection
gate_modulated = gate_base + v_b * (v_a @ h)
up = W_up @ h  # Unmodulated
output = SiLU(gate_modulated) * up
```

### New (Dual Modulation)
```python
# Modulates both gate and up projections
gate_modulated = gate_base + v_b_gate * (v_a_gate @ h)
up_modulated = up_base + v_b_up * (v_a_up @ h)
output = SiLU(gate_modulated) * up_modulated
```

## Key Benefits

### 1. Direct Weight Update Compatibility
Each modulation directly corresponds to a weight update:
- `W_gate_new = W_gate + v_b_gate ⊗ v_a_gate`
- `W_up_new = W_up + v_b_up ⊗ v_a_up`

This makes permanent knowledge injection straightforward.

### 2. Preserved MLP Semantics
- Gate projection maintains its role as an information gate
- Up projection maintains its role as a transformation
- No disruption to SwiGLU's carefully designed mechanics

### 3. Doubled Expressiveness
- 2x modulation capacity (gate + up)
- Each projection can specialize its modulation
- Better ability to capture complex transformations

### 4. Improved Training Dynamics
- Cleaner gradient flow through both projections
- More stable optimization landscape
- Better alignment between training and inference

## Implementation Details

### NPComponent Changes
- Generates two sets of (v_a, v_b) pairs
- Separate weight matrices for gate and up modulations
- Initialization tuned for each projection

### NPTDecoderLayer Changes
- `_apply_dual_modulated_mlp()` method for dual modulation
- Efficient rank-1 application without forming full matrices
- Backward compatible with single modulation

### Loss Computation
- Regularization applied to all four modulation vectors
- Same direct supervision target: `MLP(h+attention)`
- Lower weight needed due to increased expressiveness

## Training Configuration

### Recommended Hyperparameters
```bash
--train_layers "14,15"        # Only top 2 layers
--np_rank 256                 # Per-projection rank
--num_ranks 2                 # 2 rank-1 components
--direct_mlp_weight 2.0       # Lower due to expressiveness
--fidelity_weight 5.0         # Higher for output quality
--lambda_reg 0.0001          # Very low regularization
```

### Curriculum
- 80% teacher supervision (40k/50k steps)
- Gradual transition through mixed stage
- Short student phase for fine-tuning

## Expected Outcomes

1. **Better Convergence**: More expressive modulation should achieve lower loss
2. **Preserved Language Quality**: MLP semantics remain intact
3. **Clear Injection Path**: Direct mapping to permanent weight updates
4. **Stable Training**: Better gradient flow and optimization dynamics

## Usage

To train with dual modulation:
```bash
./train_dual_modulation.sh
```

To apply permanent weight updates after training:
```python
# Extract modulation for a specific context
v_a_gate, v_b_gate, v_a_up, v_b_up = extract_modulation(context)

# Apply as permanent weight updates
W_gate_new = W_gate + learning_rate * (v_b_gate ⊗ v_a_gate)
W_up_new = W_up + learning_rate * (v_b_up ⊗ v_a_up)
```

## Comparison with Previous Approaches

| Aspect | Single Gate Modulation | Dual Modulation |
|--------|------------------------|-----------------|
| Modulation Points | 1 (gate only) | 2 (gate + up) |
| Total Capacity | rank × d_ffn | 2 × rank × d_ffn |
| Weight Update | Unclear mapping | Direct mapping |
| MLP Semantics | Disrupted | Preserved |
| Training Stability | Moderate | High |
| Expressiveness | Limited | High |

## Future Extensions

1. **Adaptive Rank Selection**: Use attention to weight different rank components
2. **Layer-Specific Modulation**: Different strategies for different layers
3. **Conditional Modulation**: Gate modulation based on input characteristics
4. **Hierarchical Modulation**: Coarse-to-fine modulation across layers