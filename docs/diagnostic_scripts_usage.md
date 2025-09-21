# NPT Model Diagnostic Scripts Usage Guide

This guide explains how to use the NPT diagnostic scripts to assess model training progress and readiness.

## Scripts Overview

### 1. Quick Diagnostic (`scripts/quick_npt_diagnostic.py`)
- **Purpose**: Fast health check for NPT models
- **Runtime**: ~1-2 minutes
- **Use case**: Quick validation during training

### 2. Comprehensive Test (`scripts/test_npt_checkpoint.py`)
- **Purpose**: Detailed analysis of NPT model performance
- **Runtime**: ~5-15 minutes depending on tests
- **Use case**: Thorough evaluation and debugging

## Quick Start Examples

### Quick Health Check
```bash
# Basic check (no accuracy test - fastest)
python scripts/quick_npt_diagnostic.py \
  --checkpoint experiments/checkpoint/npt_weights.pt \
  --no_accuracy

# Full quick diagnostic with accuracy
python scripts/quick_npt_diagnostic.py \
  --checkpoint experiments/checkpoint/npt_weights.pt

# With verbose output
python scripts/quick_npt_diagnostic.py \
  --checkpoint experiments/checkpoint/npt_weights.pt \
  --verbose
```

### Comprehensive Analysis
```bash
# Quick tests only (basic functionality + NPT vs standard)
python scripts/test_npt_checkpoint.py \
  --checkpoint experiments/checkpoint/npt_weights.pt \
  --quick

# Full comprehensive test suite
python scripts/test_npt_checkpoint.py \
  --checkpoint experiments/checkpoint/npt_weights.pt

# Detailed output with all test information
python scripts/test_npt_checkpoint.py \
  --checkpoint experiments/checkpoint/npt_weights.pt \
  --detailed

# Save results to file
python scripts/test_npt_checkpoint.py \
  --checkpoint experiments/checkpoint/npt_weights.pt \
  --output results.json
```

## Interpreting Results

### Loss Metrics and Training Progress

#### MSE Loss Interpretation
- **< 0.05**: Excellent - Model ready for use
- **0.05 - 0.1**: Good - Model ready with minor limitations
- **0.1 - 0.5**: Fair - Continue training, getting close
- **> 0.5**: Poor - Significant training needed

#### Cosine Similarity
- **> 0.95**: Excellent directional alignment
- **0.9 - 0.95**: Good alignment
- **0.8 - 0.9**: Moderate alignment
- **< 0.8**: Poor alignment - check initialization

#### Training Progress Estimates
Based on MSE loss, the scripts estimate:
- **Progress %**: How much of the training is complete
- **Remaining Work**: Rough estimate of additional training needed
- **Steps Estimate**: Approximate training steps required

### Example Results Analysis

#### Example 1: Early Training (MSE ~1.89)
```
📊 ACCURACY METRICS:
   MSE Loss: 1.888480
   Cosine Similarity: 0.9548
   Loss Status: POOR ❌

📈 TRAINING PROGRESS:
   Estimated Progress: 10.0%
   Remaining Work: Significant training needed (>10k steps)
```

**Interpretation**: Model shows good directional alignment (0.95 cosine sim) but poor magnitude matching. Needs substantial more training.

#### Example 2: Mid Training (MSE ~1.29)
```
📊 ACCURACY METRICS:
   MSE Loss: 1.292411
   Cosine Similarity: 0.9661
   Loss Status: POOR ❌

📈 TRAINING PROGRESS:
   Estimated Progress: 10.0%
   Remaining Work: Significant training needed (>10k steps)
```

**Interpretation**: Improvement from early training but still needs significant work. The cosine similarity improved to 0.966.

#### Example 3: Target State (MSE ~0.35)
```
📊 ACCURACY METRICS:
   MSE Loss: 0.350000
   Cosine Similarity: 0.9850
   Loss Status: FAIR ⚠️

📈 TRAINING PROGRESS:
   Estimated Progress: 70.0%
   Remaining Work: Light training needed (1k-5k steps)
```

**Interpretation**: Getting close! The user's reported loss of 0.35 would show as "FAIR" and suggest the model is about 70% trained.

#### Example 4: Ready State (MSE < 0.1)
```
📊 ACCURACY METRICS:
   MSE Loss: 0.085000
   Cosine Similarity: 0.9920
   Loss Status: GOOD ✅

📈 TRAINING PROGRESS:
   Estimated Progress: 90.0%
   Remaining Work: Fine-tuning needed (<1k steps)
```

**Interpretation**: Model is ready for use with minor limitations.

## Understanding Current Status

### Your Checkpoint Analysis (28k steps)
Based on the test results from your 28k step checkpoint:

- **Current MSE**: 1.29 (down from 1.89 at 10.8k steps)
- **Progress**: Training is working and improving
- **Status**: Still needs significant training
- **Gap**: Your reported loss of 0.35 suggests newer checkpoints exist

### When is the Model "Ready"?

#### For Basic Use
- **MSE < 0.1**: Model produces outputs close to original transformer
- **Cosine Sim > 0.9**: Good directional alignment
- **Generation Quality**: Coherent, similar to original model

#### For Production Use
- **MSE < 0.05**: Excellent fidelity to original
- **Cosine Sim > 0.95**: Very high alignment
- **Per-layer Performance**: Most layers working well individually

## Troubleshooting Common Issues

### Model Loading Errors
```
❌ Error: No NPT layers found in checkpoint
```
**Solution**: Checkpoint may be corrupted or incompatible format.

### Device/Memory Issues
```
❌ Error: Expected all tensors to be on the same device
```
**Solution**: Scripts automatically handle device placement, but some edge cases may occur with mixed precision.

### High MSE with Good Cosine Similarity
**Interpretation**: Model has correct direction but wrong magnitude - typically indicates:
- Training is progressing but needs more steps
- Learning rate might be too high/low
- Normal for early/mid training stages

### Low Cosine Similarity
**Interpretation**: Model outputs are in wrong direction - typically indicates:
- Poor initialization
- Training instability
- Possible architecture mismatch

## Advanced Usage

### Testing Specific Configurations
```bash
# Test with different model sizes
python scripts/quick_npt_diagnostic.py \
  --checkpoint checkpoint.pt \
  --model_name "meta-llama/Llama-3.1-8B"

# Test directory with multiple checkpoints
python scripts/test_npt_checkpoint.py \
  --checkpoint experiments/training_run/checkpoints/
```

### Automated Testing Pipeline
```bash
# Test multiple checkpoints and compare
for step in 10000 12000 14000 16000; do
  echo "Testing step $step:"
  python scripts/quick_npt_diagnostic.py \
    --checkpoint experiments/checkpoints/checkpoint-$step/npt_weights.pt \
    --no_accuracy | grep -E "(MSE Loss|Progress)"
done
```

## Summary

- **MSE < 0.1**: Ready for use
- **MSE 0.1-0.5**: Continue training
- **MSE > 0.5**: Significant training needed
- **Your 0.35 loss**: About 70% trained, getting close!

The diagnostic scripts provide objective metrics to track training progress and determine when your NPT model is ready for deployment.