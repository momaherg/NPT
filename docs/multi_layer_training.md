# Multi-Layer NPT Training with Teacher Scaffolding

## Overview

Multi-layer NPT training enables simultaneous training of multiple NPT layers using teacher scaffolding to ensure correct gradient flow. This approach is **4x faster** than sequential layer-by-layer training while maintaining training stability.

## Key Concepts

### Teacher Scaffolding
- A "teacher" model (standard transformer mode) provides correct hidden states
- NPT layers use teacher states for attention inputs during early training
- Ensures each NPT layer receives correct attention patterns while learning

### Curriculum Learning
The training progresses through stages:
1. **Teacher Stage**: Pure teacher inputs for attention (learns correct patterns)
2. **Mixed Stage**: Gradual mixing of teacher and student inputs
3. **Student Stage**: Fully independent NPT processing

### Output Propagation
- NPT outputs **always** propagate to the next layer (no mixing)
- Only attention inputs use curriculum mixing
- Ensures proper gradient flow and layer interaction

## Usage

### Basic Multi-Layer Training
```bash
# Train layers 14-17 of Llama 8B
./train_multi_layer_llama8b.sh
```

### Custom Configuration
```bash
python scripts/train_multi_layer_npt.py \
  --train_layers "14,15,16,17" \
  --curriculum_stages "teacher:5000,mixed:5000:0.5,student:20000" \
  --layer_weights "uniform" \
  --np_rank 256 \
  --batch_size 4
```

### Curriculum Format
```
stage:steps[:mixing_ratio],stage:steps[:mixing_ratio],...
```
- `stage`: Name of the stage (teacher/mixed/student)
- `steps`: Number of steps for this stage
- `mixing_ratio`: Optional, for mixed stages (0=teacher, 1=student)

Example:
```
"teacher:5000,mixed:3000:0.3,mixed:3000:0.7,student:15000"
```

### Layer Weights
Control how losses from different layers are weighted:
- `"uniform"`: All layers weighted equally
- `"linear"`: Linear decay from first to last layer
- `"1.0,0.9,0.8,0.7"`: Custom weights per layer

## Comparison with Sequential Training

| Aspect | Sequential | Multi-Layer |
|--------|-----------|-------------|
| Speed | 1x per layer (4x total for 4 layers) | 1x total |
| Memory | Low (one layer at a time) | Moderate (one model) |
| Gradient Quality | Perfect (trained layers are correct) | Good (teacher scaffolding) |
| Layer Interaction | None during training | Full co-adaptation |
| Implementation | Simple | More complex |

## Implementation Details

### Memory Efficiency
- Only one model in GPU memory
- Model switches between teacher/student modes
- Teacher states computed on-demand (no caching by default)

### Gradient Flow
```
1. Teacher provides correct hidden states
2. NPT attention uses teacher/mixed/student inputs (curriculum)
3. NPT modulation applied to current hidden states
4. NPT output propagates to next layer (always)
```

### Loss Computation
- Each layer has its own direct supervision loss
- Global fidelity loss for final output
- Per-layer regularization on v_a and v_b

## Experiments

### Demo (Quick Test)
```bash
./train_multi_layer_demo.sh
```
- Uses tiny model (4 layers, 256 hidden dim)
- 100 training steps
- Tests basic functionality

### Small Model (Llama 1B)
```bash
./train_multi_layer_llama1b.sh
```
- Trains 8 layers (upper half)
- Good for testing on limited GPUs

### Large Model (Llama 8B)
```bash
./train_multi_layer_llama8b.sh
```
- Trains 4 layers (14-17)
- Full production configuration

### Rank-k Updates
```bash
./train_multi_layer_rank_k.sh
```
- Uses multiple rank-1 components
- More expressive weight modulation
- Better for complex transformations

## Monitoring Training

### Key Metrics
- `curriculum_stage`: Current training stage
- `mixing_ratio`: Current attention input mixing
- `layer_X_v_a_attn_similarity`: How well v_a encodes attention
- `layer_X_direct_mlp`: Per-layer direct supervision loss
- `fidelity_loss`: End-to-end model fidelity

### Success Indicators
- All layers show `v_a_attn_similarity > 0.7`
- Layer losses are balanced (no single layer dominating)
- Smooth curriculum transition (check mixing_ratio)
- Fidelity loss < 0.1 by end of training

## Troubleshooting

### High Memory Usage
- Reduce `batch_size`
- Increase `gradient_accumulation_steps`
- Train fewer layers simultaneously

### Poor Convergence
- Extend teacher stage in curriculum
- Reduce learning rate
- Check if layers are too far apart (try adjacent layers)

### Imbalanced Layer Losses
- Adjust `layer_weights` to balance training
- Use "linear" weighting for depth-based scaling
- Consider training layers in groups

## Advanced Configuration

### Custom Curriculum
```python
# Aggressive curriculum (fast transition)
--curriculum_stages "teacher:2000,student:18000"

# Conservative curriculum (slow transition)
--curriculum_stages "teacher:8000,mixed:4000:0.2,mixed:4000:0.5,mixed:4000:0.8,student:10000"

# No curriculum (pure student from start)
--curriculum_stages "student:30000"
```

### Layer Selection Strategies
```python
# Adjacent layers (best interaction)
--train_layers "14,15,16,17"

# Spread out layers (diverse coverage)
--train_layers "8,12,16,20"

# Upper layers (higher-level features)
--train_layers "24,25,26,27,28,29,30,31"  # For 32-layer model
```

## Citation

If you use multi-layer NPT training in your research, please cite:
```
Multi-Layer Neuro-Plastic Transformer Training with Teacher Scaffolding
[Your paper reference here]
```