#!/bin/bash
# Training script for multi-layer NPT with rank-k updates
# Uses multiple rank-1 components for more expressive weight modulation

echo "=================================================="
echo "Multi-Layer NPT Training with Rank-k Updates"
echo "Llama 3.1 8B - Layers 15-18 with rank-4 modulation"
echo "=================================================="

python scripts/train_multi_layer_npt.py \
  --model_name "meta-llama/Llama-3.1-8B-Instruct" \
  --model_size 8b \
  --train_layers "15,16,17,18" \
  --curriculum_stages "teacher:4000,mixed:3000:0.3,mixed:3000:0.6,student:20000" \
  --layer_weights "1.0,0.9,0.8,0.7" \
  --init_strategy improved \
  --num_ranks 4 \
  --np_rank 128 \
  --np_init_scale 0.001 \
  --dataset_preset medium \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 3e-5 \
  --weight_decay 0.01 \
  --lambda_reg 0.01 \
  --direct_mlp_weight 10.0 \
  --fidelity_weight 1.0 \
  --gradient_scale_factor 2.0 \
  --max_steps 30000 \
  --warmup_steps 1000 \
  --gradient_clip 0.5 \
  --mixed_precision \
  --logging_steps 10 \
  --eval_steps 500 \
  --num_eval_samples 100 \
  --save_steps 2000 \
  --generation_steps 500 \
  --num_workers 2 \
  --wandb_project npt-multi-layer \
  --wandb_name npt_8b_layers15-18_rank4 \
  --wandb_tags llama-3.1 8b multi_layer rank_k rank_4 custom_weights

# Rank-k Configuration:
# =====================
# --num_ranks 4: Uses 4 rank-1 components
#   - Total effective rank is 4
#   - More expressive than single rank-256
#   - Each component can specialize
#
# --np_rank 128: Each component has rank 128
#   - Total parameters: 4 * (d_model*128 + 128*d_model + 128*d_ffn)
#   - Still only ~2-5% overhead
#
# Layer Weights:
# ==============
# Custom weights decrease with depth:
# - Layer 15: 1.0 (full weight)
# - Layer 16: 0.9
# - Layer 17: 0.8
# - Layer 18: 0.7 (lowest weight)
#
# This helps balance training across layers

# Expected Benefits:
# ==================
# - Better expressiveness with rank-4 updates
# - Each component can capture different patterns
# - More stable training with distributed capacity
# - Better generalization compared to single high-rank