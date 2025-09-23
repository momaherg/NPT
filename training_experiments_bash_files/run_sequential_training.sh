#!/bin/bash

# Sequential layer-by-layer NPT training
# This script trains NPT layers one at a time, building up from lower to upper layers

# Default configuration
MODEL_NAME="meta-llama/Llama-3.2-1B"
MODEL_SIZE="1b"
LAYERS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"  # All 16 layers for 1B model
STEPS_PER_LAYER=2000
STAGE1_STEPS=500
BATCH_SIZE=2
DATASET="small"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --demo)
            echo "Running in demo mode..."
            DEMO_FLAG="--demo_mode"
            LAYERS="0,1,2,3"  # Only first 4 layers in demo
            shift
            ;;
        --layers)
            LAYERS="$2"
            shift 2
            ;;
        --model-size)
            MODEL_SIZE="$2"
            shift 2
            ;;
        --steps)
            STEPS_PER_LAYER="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--demo] [--layers LAYERS] [--model-size SIZE] [--steps STEPS] [--dataset PRESET]"
            echo "  --demo: Run in demo mode with minimal steps"
            echo "  --layers: Comma-separated list of layers or 'all', 'upper_half', 'lower_half'"
            echo "  --model-size: 1b, 3b, or 8b"
            echo "  --steps: Steps per layer (default: 2000)"
            echo "  --dataset: Dataset preset (small, medium, large)"
            exit 1
            ;;
    esac
done

# Set number of layers based on model size
case $MODEL_SIZE in
    1b)
        NUM_LAYERS=16
        ;;
    3b)
        NUM_LAYERS=28
        MODEL_NAME="meta-llama/Llama-3.2-3B"
        ;;
    8b)
        NUM_LAYERS=32
        MODEL_NAME="meta-llama/Llama-3.1-8B"
        BATCH_SIZE=1  # Reduce batch size for 8B
        ;;
esac

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