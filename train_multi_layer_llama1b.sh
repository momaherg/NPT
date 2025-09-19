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
  --train_layers "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15" \
  --curriculum_stages "teacher:20000" \
  --layer_weights "uniform" \
  --init_strategy improved \
  --num_ranks 4 \
  --np_rank 256 \
  --np_init_scale 0.001 \
  --dataset_preset medium \
  --batch_size 32 \
  --gradient_accumulation_steps 1 \
  --max_length 256 \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --lambda_reg 0.01 \
  --direct_mlp_weight 10.0 \
  --fidelity_weight 1.0 \
  --gradient_scale_factor 2.5 \
  --max_steps 20000 \
  --warmup_steps 1000 \
  --gradient_clip 0.5 \
  --mixed_precision \
  --logging_steps 10 \
  --eval_steps 999999 \
  --num_eval_samples 1 \
  --save_steps 400 \
  --generation_steps 500 \
  --num_workers 1 \
  --wandb_project npt-multi-layer \
  --wandb_name npt_1b_layers_full \
  --wandb_tags llama-3.2 1b multi_layer upper_half linear_weights

# Configuration Notes:
# ====================
# - Training 8 layers (upper half of 16-layer model)
# - Linear layer weights: Higher layers get less weight
# - Shorter curriculum due to smaller model
# - Higher batch size due to smaller model size
# - Lower gradient scale factor since training more layers