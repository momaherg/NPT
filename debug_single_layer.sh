#!/bin/bash
# Debug script for single-layer NPT training with adjusted hyperparameters
# This uses more conservative settings to help with convergence

echo "=================================================="
echo "Debug Single-Layer NPT Training"
echo "Conservative hyperparameters for better convergence"
echo "=================================================="

python scripts/train_single_layer_npt.py \
  --model_name "meta-llama/Llama-3.2-1B" \
  --model_size 1b \
  --convert_layers "8" \
  --single_layer_mode \
  --np_rank 128 \
  --np_init_scale 0.01 \
  --dataset_preset small \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --stage1_lr 1e-4 \
  --stage1_steps 200 \
  --weight_decay 0.01 \
  --lambda_reg 0.001 \
  --direct_mlp_weight 1.0 \
  --attention_encoding_weight 1.0 \
  --fidelity_weight 1.0 \
  --gradient_scale_factor 2.0 \
  --max_steps 1000 \
  --warmup_steps 100 \
  --gradient_clip 1.0 \
  --mixed_precision \
  --logging_steps 10 \
  --eval_steps 100 \
  --save_steps 500 \
  --generation_steps 200 \
  --num_workers 1 \
  --wandb_project npt-single-layer \
  --wandb_name debug_single_layer \
  --wandb_mode offline \
  --wandb_tags debug single_layer conservative

echo ""
echo "Key adjustments for debugging:"
echo "1. Lower gradient scale factor (2.0 instead of 10.0)"
echo "2. Lower learning rates (5e-5 and 1e-4)"
echo "3. Balanced loss weights (all 1.0)"
echo "4. Higher init scale (0.01)"
echo "5. Lower regularization (0.001)"