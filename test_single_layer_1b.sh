#!/bin/bash
# Quick test script for single-layer NPT on Llama 3.2 1B
# Use this for testing the implementation before running on larger models

echo "=================================================="
echo "Testing Single-Layer NPT Training (Llama 3.2 1B)"
echo "Quick test with reduced steps"
echo "=================================================="

python scripts/train_single_layer_npt.py \
  --model_name "meta-llama/Llama-3.2-1B" \
  --model_size 1b \
  --convert_layers "8" \
  --single_layer_mode \
  --np_rank 128 \
  --np_init_scale 0.001 \
  --dataset_preset small \
  --batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-4 \
  --stage1_lr 1e-3 \
  --stage1_steps 100 \
  --weight_decay 0.01 \
  --lambda_reg 0.01 \
  --direct_mlp_weight 10.0 \
  --attention_encoding_weight 5.0 \
  --fidelity_weight 1.0 \
  --gradient_scale_factor 10.0 \
  --max_steps 500 \
  --warmup_steps 50 \
  --gradient_clip 1.0 \
  --mixed_precision \
  --logging_steps 10 \
  --eval_steps 100 \
  --save_steps 250 \
  --generation_steps 100 \
  --num_workers 1 \
  --wandb_project npt-single-layer \
  --wandb_name test_1b_layer8_single \
  --wandb_mode offline \
  --wandb_tags llama-3.2 1b layer_8 single_layer test

echo ""
echo "Test complete! Check the logs for:"
echo "1. No runtime errors"
echo "2. Loss decreasing"
echo "3. v_a_attention_similarity increasing"
echo "4. No mode collapse warnings"