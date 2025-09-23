#!/bin/bash
# Comparison script: Sequential vs Multi-layer NPT training
# This script demonstrates the difference in training approaches

echo "============================================================"
echo "NPT Training Methods Comparison"
echo "============================================================"
echo ""

# Configuration
MODEL="meta-llama/Llama-3.2-1B"
LAYERS="14,15,16,17"
DATASET="medium"
STEPS_PER_METHOD=10000

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Layers: $LAYERS"
echo "  Dataset: $DATASET"
echo "  Steps per method: $STEPS_PER_METHOD"
echo ""

# Method 1: Sequential Training (Traditional)
if [ "$1" == "sequential" ] || [ "$1" == "both" ]; then
    echo "------------------------------------------------------------"
    echo "Method 1: Sequential Layer-by-Layer Training"
    echo "------------------------------------------------------------"
    echo "Training layers one at a time, accumulating weights..."
    echo ""

    python scripts/train_sequential_layers.py \
        --model_name "$MODEL" \
        --model_size 1b \
        --layers "$LAYERS" \
        --steps_per_layer 2500 \
        --batch_size 4 \
        --learning_rate 5e-4 \
        --np_rank 256 \
        --num_ranks 1 \
        --dataset_preset "$DATASET" \
        --base_output_dir experiments/comparison_sequential \
        --wandb_project npt-comparison \
        --wandb_mode offline

    echo "Sequential training complete!"
    echo "Total time: ~4x single layer training time"
    echo ""
fi

# Method 2: Multi-Layer Training (New)
if [ "$1" == "multi" ] || [ "$1" == "both" ]; then
    echo "------------------------------------------------------------"
    echo "Method 2: Simultaneous Multi-Layer Training with Teacher"
    echo "------------------------------------------------------------"
    echo "Training all layers simultaneously with teacher scaffolding..."
    echo ""

    python scripts/train_multi_layer_npt.py \
        --model_name "$MODEL" \
        --model_size 1b \
        --train_layers "$LAYERS" \
        --curriculum_stages "teacher:2500,mixed:2500:0.5,student:5000" \
        --layer_weights "uniform" \
        --batch_size 4 \
        --learning_rate 5e-5 \
        --np_rank 256 \
        --num_ranks 1 \
        --dataset_preset "$DATASET" \
        --max_steps "$STEPS_PER_METHOD" \
        --output_dir experiments/comparison_multi \
        --wandb_project npt-comparison \
        --wandb_mode offline

    echo "Multi-layer training complete!"
    echo "Total time: ~1x single layer training time"
    echo ""
fi

# Method 3: Single Layer Baseline
if [ "$1" == "single" ]; then
    echo "------------------------------------------------------------"
    echo "Method 3: Single Layer Training (Baseline)"
    echo "------------------------------------------------------------"
    echo "Training just layer 15 for comparison..."
    echo ""

    python scripts/train_single_layer_npt.py \
        --model_name "$MODEL" \
        --model_size 1b \
        --convert_layers "15" \
        --single_layer_mode \
        --np_rank 512 \
        --batch_size 4 \
        --learning_rate 1e-4 \
        --dataset_preset "$DATASET" \
        --max_steps "$STEPS_PER_METHOD" \
        --output_dir experiments/comparison_single \
        --wandb_project npt-comparison \
        --wandb_mode offline

    echo "Single layer training complete!"
    echo ""
fi

echo "============================================================"
echo "Comparison Summary"
echo "============================================================"
echo ""
echo "Sequential Training:"
echo "  ✓ Simple and proven"
echo "  ✗ 4x slower (trains each layer separately)"
echo "  ✗ No interaction between layers during training"
echo ""
echo "Multi-Layer Training:"
echo "  ✓ 4x faster (all layers train together)"
echo "  ✓ Teacher scaffolding ensures correct gradients"
echo "  ✓ Layers can co-adapt during training"
echo "  ✗ More complex implementation"
echo "  ✗ Requires curriculum tuning"
echo ""
echo "Usage:"
echo "  $0 sequential  - Run sequential training only"
echo "  $0 multi       - Run multi-layer training only"
echo "  $0 single      - Run single layer baseline"
echo "  $0 both        - Run both methods for comparison"
echo ""