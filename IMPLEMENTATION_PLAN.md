# Neuro-Plastic Transformer (NPT) Implementation Plan

## Overview
This document outlines the staged implementation of the Neuro-Plastic Transformer (NPT), a novel architecture that replaces standard additive residuals with dynamic weight updates.

## Key Architecture Insights from Llama 3
- **Hidden Size**: 2048 (1B model) / 4096 (8B model)
- **MLP Structure**: 
  - Uses SwiGLU activation: `output = down_proj(silu(gate_proj(x)) * up_proj(x))`
  - Intermediate size: 8192 (1B) / 14336 (8B)
- **Layer Structure**:
  1. Input LayerNorm → Self-Attention → Residual Add
  2. Post-Attention LayerNorm → MLP → Residual Add

## Implementation Stages

### Stage 1: NP Component Module
**Objective**: Create the core Neuro-Plastic component that generates rank-1 weight updates.

**Components**:
- `NPComponent` class with configurable rank `r`
- Three weight matrices: `W_down`, `W_a_up`, `W_b_up`
- Forward method that generates `v_a` and `v_b` vectors

**Test Criteria**:
- Component initializes correctly
- Output shapes match expected dimensions and type
- Gradient flow works properly
- Generated vectors have appropriate magnitudes

**Files**:
- `src/npt/np_component.py`
- `tests/test_np_component.py`

---

### Stage 2: NPT Decoder Layer
**Objective**: Create a modified Llama decoder layer that uses the NP component.

**Components**:
- `NPTDecoderLayer` that inherits from `LlamaDecoderLayer`
- Replace attention residual with NP-based weight modulation
- Preserve MLP residual connection

**Test Criteria**:
- Layer processes inputs correctly
- Output shape matches original layer
- Can toggle between standard and NPT mode
- Memory usage is reasonable

**Files**:
- `src/npt/npt_decoder_layer.py`
- `tests/test_npt_decoder_layer.py`

---

### Stage 3: Hybrid NPT Model
**Objective**: Create a full model with selective layer conversion.

**Components**:
- `NPTLlamaModel` that can convert specific layers to NPT
- Configuration for which layers to convert
- Method to freeze base model parameters
- Method to extract trainable NP parameters

**Test Criteria**:
- Model loads base weights correctly
- Can convert specified layers (e.g., layers 8-15)
- Forward pass works end-to-end
- Only NP parameters are trainable

**Files**:
- `src/npt/npt_model.py`
- `config/model_config.yaml`
- `tests/test_npt_model.py`

---

### Stage 4: Equivalence Loss Implementation
**Objective**: Implement loss functions for equivalence pre-training.

**Components**:
- Fidelity loss (MSE between NPT and original outputs)
- Regularization loss (L2 on v_a and v_b)
- Combined loss with weighting factor λ
- Parallel forward pass helper

**Test Criteria**:
- Losses compute correctly
- Gradients flow to NP components only
- Loss values are in expected range

**Files**:
- `src/training/losses.py`
- `tests/test_losses.py`

---

### Stage 5: Training Pipeline
**Objective**: Create the complete training loop for equivalence pre-training.

**Components**:
- Data loader for training texts
- Training loop with proper optimizer setup
- Validation loop with metrics
- Checkpoint saving/loading
- Learning rate scheduling

**Test Criteria**:
- Training runs without errors
- Loss decreases over time
- Model outputs remain coherent
- Checkpoints save/load correctly

**Files**:
- `src/training/trainer.py`
- `src/training/data_loader.py`
- `scripts/train_equivalence.py`
- `tests/test_training.py`

---

### Stage 6: Experiment Tracking & Evaluation
**Objective**: Set up comprehensive experiment tracking and evaluation.

**Components**:
- WandB integration for metric logging
- Periodic generation samples
- Benchmark evaluation (perplexity, MMLU subset)
- Delta weight magnitude tracking
- Comparison with base model

**Test Criteria**:
- Metrics log to WandB correctly
- Generation quality maintained
- Perplexity close to base model
- Delta weights have low magnitude

**Files**:
- `src/evaluation/metrics.py`
- `src/evaluation/benchmarks.py`
- `scripts/evaluate_model.py`
- `tests/test_evaluation.py`

---

## Project Structure
```
NPT/
├── src/
│   ├── npt/
│   │   ├── __init__.py
│   │   ├── np_component.py
│   │   ├── npt_decoder_layer.py
│   │   └── npt_model.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── losses.py
│   │   ├── trainer.py
│   │   └── data_loader.py
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py
│       └── benchmarks.py
├── config/
│   ├── model_config.yaml
│   └── training_config.yaml
├── scripts/
│   ├── train_equivalence.py
│   └── evaluate_model.py
├── tests/
│   ├── test_np_component.py
│   ├── test_npt_decoder_layer.py
│   ├── test_npt_model.py
│   ├── test_losses.py
│   ├── test_training.py
│   └── test_evaluation.py
├── experiments/
│   └── (experiment logs and checkpoints)
└── requirements.txt
```

## Testing Strategy
Each stage will have:
1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test component interactions
3. **Functional Tests**: Verify end-to-end functionality
4. **Performance Tests**: Check memory usage and speed

## Success Metrics
1. **Stage 1-3**: Components work correctly, shapes match expectations
2. **Stage 4-5**: Training loss converges, model remains stable
3. **Stage 6**: Final model achieves <5% perplexity increase vs base model

## Implementation Order
1. Start with Stage 1 (NP Component) - simplest, foundational
2. Progress sequentially through stages
3. Each stage builds on previous ones
4. Full integration testing after each stage

## Key Design Decisions
1. **Rank Selection**: Start with r=64 for balance between expressiveness and efficiency
2. **Layer Selection**: Convert layers all layers initially
3. **Learning Rate**: 1e-4 for NP components (higher than typical fine-tuning)
4. **Regularization Weight**: λ=0.01 initially, tune based on experiments
