#!/bin/bash

# Sequential layer-by-layer NPT training
# This script trains NPT layers one at a time, building up from lower to upper layers

# Default configuration
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
MODEL_SIZE="8b"
LAYERS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"
# All 16 layers for 1B model
# 0,1,2,3,4,5,6,7,8,9,10,11,12,13,
STEPS_PER_LAYER=4000
STAGE1_STEPS=700
BATCH_SIZE=1
DATASET="small"
NUM_LAYERS=32

echo "=========================================="
echo "Sequential NPT Training Configuration"
echo "=========================================="
echo "Model: $MODEL_NAME ($MODEL_SIZE)"
echo "Layers to train: $LAYERS"
echo "Steps per layer: $STEPS_PER_LAYER"
echo "Stage 1 steps: $STAGE1_STEPS"
echo "Dataset: $DATASET"
echo "=========================================="

# Run the sequential training
python scripts/train_sequential_layers.py \
    --model_name "$MODEL_NAME" \
    --model_size "$MODEL_SIZE" \
    --layers "$LAYERS" \
    --num_layers $NUM_LAYERS \
    --steps_per_layer $STEPS_PER_LAYER \
    --stage1_steps $STAGE1_STEPS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps 32 \
    --learning_rate 5e-4 \
    --np_rank 256 \
    --dataset_preset "$DATASET" \
    --max_length 512 \
    --wandb_project "npt-sequential" \
    --wandb_mode "online" \
    $DEMO_FLAG