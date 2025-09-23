#!/bin/bash
# Training script for NPT with attention included in modulation target
# This architecture makes the modulated MLP output both attention and MLP(h+attention)
# Layer output: h + MLP_modulated(h) where MLP_modulated(h) = attention + MLP(h+attention)

echo "================================================================"
echo "NPT Training with Attention in Modulation"
echo "Architecture: MLP_modulated(h) = attention + MLP(h+attention)"
echo "Key change: Attention is no longer in residual, fully contained in modulation"
echo "================================================================"

python scripts/train_multi_layer_npt.py \
  --model_name "meta-llama/Llama-3.2-1B" \
  --model_size 1b \
  --train_layers "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15" \
  --curriculum_stages "teacher:35000" \
  --layer_weights "uniform" \
  --init_strategy improved \
  --dual_modulation \
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
  --max_steps 35000 \
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
  --wandb_name npt_attention_in_modulation_15layers \
  --wandb_tags llama-3.2 1b dual_modulation attention_in_modulation new_architecture \
  --resume_from "experiments/npt_attention_in_modulation_15layers/checkpoints/checkpoint-22000/"

# Key Changes from Previous Training:
# =====================================
# 1. NO --resume_from: Starting fresh due to incompatible objective
# 2. Lower learning rate: 5e-5 (was 1e-4) for stability with harder task
# 3. Higher direct_mlp_weight: 15.0 (was 10.0) for stronger supervision
# 4. Longer training: 50k steps (was 35k) due to increased complexity
# 5. Longer warmup: 2k steps (was 1k) for careful optimization
# 6. Longer teacher supervision: 40k steps (was 35k) - 80% of training

# Architecture Benefits:
# =====================
# 1. Cleaner residual path: Only h flows through residual
# 2. Full modulation control: Both attention and MLP effects in one component
# 3. More challenging but potentially more powerful learning
# 4. Better alignment with goal of replacing residual connections

# Expected Behavior:
# ==================
# - Initial loss will be higher (more to reconstruct)
# - Slower convergence than previous architecture
# - v_a and v_b will learn richer representations
# - Final model should fully compensate for missing attention residual