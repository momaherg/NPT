#!/bin/bash
# Training script for multi-layer NPT on Llama 3.2 1B
# Smaller model allows for more layers and larger batches

echo "=================================================="
echo "Multi-Layer NPT Training for Llama 3.2 1B"
echo "Training upper half layers (8-15) with teacher scaffolding"
echo "=================================================="

python scripts/train_multi_layer_npt.py \
  --model_name "meta-llama/Llama-3.2-1B" \
  --model_size 1b \
  --train_layers "8,9,10,11,12,13,14,15" \
  --curriculum_stages "teacher:3000,mixed:3000:0.5,student:14000" \
  --layer_weights "linear" \
  --init_strategy improved \
  --num_ranks 1 \
  --np_rank 128 \
  --np_init_scale 0.001 \
  --dataset_preset medium \
  --batch_size 8 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --lambda_reg 0.01 \
  --direct_mlp_weight 10.0 \
  --fidelity_weight 1.0 \
  --gradient_scale_factor 1.5 \
  --max_steps 20000 \
  --warmup_steps 500 \
  --gradient_clip 0.5 \
  --mixed_precision \
  --logging_steps 10 \
  --eval_steps 500 \
  --num_eval_samples 100 \
  --save_steps 2000 \
  --generation_steps 500 \
  --num_workers 4 \
  --wandb_project npt-multi-layer \
  --wandb_name npt_1b_layers8-15_upper_half \
  --wandb_tags llama-3.2 1b multi_layer upper_half linear_weights

# Configuration Notes:
# ====================
# - Training 8 layers (upper half of 16-layer model)
# - Linear layer weights: Higher layers get less weight
# - Shorter curriculum due to smaller model
# - Higher batch size due to smaller model size
# - Lower gradient scale factor since training more layers