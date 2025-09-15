# WandB Metrics Guide for Multi-Layer NPT Training

## Metric Organization

The multi-layer NPT training logs metrics in a structured hierarchy for easy visualization:

### Primary Losses (`loss/`)
- `loss/total` - Combined loss across all layers
- `loss/fidelity` - End-to-end model fidelity loss

### Direct MLP Supervision Loss (`mlp_direct_loss/`)
**Most important metric for NPT training**
- `mlp_direct_loss/layer_14` - Direct supervision loss for layer 14
- `mlp_direct_loss/layer_15` - Direct supervision loss for layer 15
- `mlp_direct_loss/layer_16` - Direct supervision loss for layer 16
- `mlp_direct_loss/layer_17` - Direct supervision loss for layer 17

### Regularization (`regularization/`)
- `regularization/layer_14` - L2 regularization for layer 14's v_a and v_b
- `regularization/layer_15` - L2 regularization for layer 15's v_a and v_b
- `regularization/layer_16` - L2 regularization for layer 16's v_a and v_b
- `regularization/layer_17` - L2 regularization for layer 17's v_a and v_b

### v_a Attention Similarity (`v_a_attention_similarity/`)
**Key indicator of attention encoding quality**
- `v_a_attention_similarity/layer_14` - Cosine similarity between v_a and attention
- `v_a_attention_similarity/layer_15` - Cosine similarity between v_a and attention
- `v_a_attention_similarity/layer_16` - Cosine similarity between v_a and attention
- `v_a_attention_similarity/layer_17` - Cosine similarity between v_a and attention

### Vector Norms (`v_a_norm/` and `v_b_norm/`)
- `v_a_norm/layer_X` - Norm of v_a vector for layer X
- `v_b_norm/layer_X` - Norm of v_b vector for layer X

### Curriculum Metrics (`curriculum/`)
- `curriculum/stage_index` - Current stage in the schedule (0, 1, 2, ...)
- `curriculum/mixing_ratio` - Current mixing ratio for attention inputs (0.0 to 1.0)
- `curriculum/stage_transition` - Spike indicator when stage changes

### Training Metrics (`training/`)
- `training/learning_rate` - Current learning rate
- `training/grad_norm` - Gradient norm after clipping

### Evaluation Metrics (`eval_loss/` and `eval_mlp_direct_loss/`)
- `eval_loss/total` - Validation total loss
- `eval_mlp_direct_loss/layer_X` - Validation direct MLP loss for layer X

## Curriculum Stage Mapping

The curriculum stages are logged as numeric values:
- **0** = Teacher stage (100% teacher inputs for attention)
- **1** = Mixed stage (gradual transition)
- **2** = Student stage (100% student inputs)

## Creating WandB Visualizations

### Recommended Panels

1. **Primary Training Progress**
   - Line plot: `loss/total` - Overall training loss
   - Line plot: `loss/fidelity` - End-to-end model fidelity
   - These should decrease smoothly over time

2. **Direct MLP Supervision (Most Important)**
   - Multi-line plot: All `mlp_direct_loss/layer_*` values
   - This is the KEY metric showing if each layer is learning the transformation
   - All layers should converge at similar rates
   - If one layer has much higher loss, it may need different weighting

3. **Attention Encoding Quality**
   - Multi-line plot: All `v_a_attention_similarity/layer_*` values
   - Should increase over time, target > 0.7 for all layers
   - Shows how well v_a is learning to encode attention

4. **Curriculum Progress**
   - Line plot: `curriculum/stage_index` - Shows stage transitions
   - Line plot: `curriculum/mixing_ratio` - Shows gradual transition
   - Add annotations at stage boundaries

5. **Vector Norms (Health Check)**
   - Multi-line plot: All `v_a_norm/layer_*` values
   - Multi-line plot: All `v_b_norm/layer_*` values
   - Should grow but stabilize, not collapse to 0 or explode

6. **Regularization**
   - Multi-line plot: All `regularization/layer_*` values
   - Should be small but non-zero

## Common Issues and Solutions

### Issue: "Selected runs are not logging media for the key train/curriculum_stage"
**Solution**: This has been fixed. The curriculum stage is now logged as a numeric value (`curriculum/stage`) instead of a string.

### Issue: Missing layer metrics
**Solution**: Check that the layer indices in `--train_layers` are valid for your model.

### Issue: Curriculum not progressing
**Solution**: Check that your `--max_steps` is greater than the total curriculum steps.

## Example WandB Query

To compare different curriculum strategies:
```python
# In WandB UI, use these filters
runs = api.runs("your-project/npt-multi-layer")
for run in runs:
    # Get curriculum progression
    history = run.history()
    stages = history['curriculum/stage'].values
    mixing = history['curriculum/mixing_ratio'].values

    # Analyze stage durations
    teacher_steps = (stages == 0).sum()
    mixed_steps = (stages == 1).sum()
    student_steps = (stages == 2).sum()
```

## Visualizing Curriculum Before Training

Use the visualization script to preview your curriculum:
```bash
python scripts/visualize_curriculum.py \
  --curriculum "teacher:5000,mixed:5000:0.3,mixed:5000:0.7,student:15000" \
  --output curriculum_plan.png
```