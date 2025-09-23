#!/bin/bash
# Resume NPT training from a checkpoint
# Usage: ./resume_training.sh <checkpoint_path> [additional_steps]

# Check if checkpoint path is provided
if [ -z "$1" ]; then
    echo "Usage: ./resume_training.sh <checkpoint_path> [additional_steps]"
    echo "Example: ./resume_training.sh experiments/npt_new_arch/checkpoints/checkpoint-10000"
    echo "Example with more steps: ./resume_training.sh experiments/npt_new_arch/checkpoints/checkpoint-10000 20000"
    exit 1
fi

CHECKPOINT_PATH="$1"
ADDITIONAL_STEPS="${2:-15000}"  # Default to 15000 additional steps if not specified

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint directory not found: $CHECKPOINT_PATH"
    exit 1
fi

# Extract current step from checkpoint
if [ -f "$CHECKPOINT_PATH/training_state.pt" ]; then
    CURRENT_STEP=$(python -c "import torch; state = torch.load('$CHECKPOINT_PATH/training_state.pt', map_location='cpu'); print(state['global_step'])")
    echo "Current checkpoint step: $CURRENT_STEP"
else
    echo "Warning: training_state.pt not found in checkpoint"
    CURRENT_STEP=0
fi

# Calculate new max_steps
NEW_MAX_STEPS=$((CURRENT_STEP + ADDITIONAL_STEPS))
echo "Will train for $ADDITIONAL_STEPS additional steps (total: $NEW_MAX_STEPS)"

echo "=================================================="
echo "Resuming NPT Training from Checkpoint"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Current Step: $CURRENT_STEP"
echo "New Max Steps: $NEW_MAX_STEPS"
echo "=================================================="

python scripts/train_multi_layer_npt.py \
  --model_name "meta-llama/Llama-3.2-1B" \
  --model_size 1b \
  --train_layers "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15" \
  --curriculum_stages "teacher:35000,mixed:10000:0.5,student:5000" \
  --layer_weights "uniform" \
  --init_strategy improved \
  --num_ranks 4 \
  --np_rank 256 \
  --np_init_scale 0.001 \
  --dataset_preset medium \
  --batch_size 32 \
  --gradient_accumulation_steps 1 \
  --max_length 64 \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --lambda_reg 0.01 \
  --direct_mlp_weight 10.0 \
  --fidelity_weight 1.0 \
  --gradient_scale_factor 1 \
  --max_steps $NEW_MAX_STEPS \
  --warmup_steps 1000 \
  --gradient_clip 0.5 \
  --mixed_precision \
  --logging_steps 10 \
  --eval_steps 999999 \
  --num_eval_samples 1 \
  --save_steps 2000 \
  --generation_steps 500 \
  --num_workers 1 \
  --wandb_project npt-multi-layer \
  --wandb_name "resumed_$(basename $CHECKPOINT_PATH)_$(date +%Y%m%d_%H%M%S)" \
  --wandb_tags llama-3.2 1b resumed multi_layer \
  --resume_from "$CHECKPOINT_PATH"

echo "=================================================="
echo "Training resumed successfully!"
echo "==================================================