#!/bin/bash
# Stable training script for single-layer NPT on Llama 3.2 1B
# Conservative settings for reliable convergence

echo "=================================================="
echo "Single-Layer NPT Training for Llama 3.2 1B (Stable)"
echo "Conservative settings for reliable convergence"
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
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --stage1_lr 5e-5 \
  --stage1_steps 500 \
  --weight_decay 0.01 \
  --lambda_reg 0.001 \
  --direct_mlp_weight 1.5 \
  --attention_encoding_weight 1.5 \
  --fidelity_weight 1.0 \
  --gradient_scale_factor 1.5 \
  --max_steps 10000 \
  --warmup_steps 1000 \
  --gradient_clip 1.0 \
  --mixed_precision \
  --logging_steps 10 \
  --eval_steps 250 \
  --save_steps 1000 \
  --generation_steps 500 \
  --num_workers 1 \
  --wandb_project npt-single-layer \
  --wandb_name npt_1b_layer8_stable \
  --wandb_tags llama-3.2 1b layer_8 single_layer stable

echo ""
echo "This configuration uses:"
echo "- Conservative gradient scaling (1.5x)"
echo "- Lower learning rates (2e-5)"
echo "- Balanced loss weights"
echo "- Higher init scale for better starting point"
echo ""
echo "Expected convergence in 5,000-7,000 steps"