#!/bin/bash
# Training script for single-layer NPT on Llama 3.1 8B
# This script implements the two-stage training strategy for layer 15

echo "=================================================="
echo "Single-Layer NPT Training for Llama 3.1 8B"
echo "Target: Layer 15 learns MLP_mod(h) = attn + MLP(h+attn)"
echo "=================================================="

python scripts/train_single_layer_npt.py \
  --model_name "meta-llama/Llama-3.1-8B-Instruct" \
  --model_size 8b \
  --loss_mode direct \
  --learning_rate 1e-3 \
  --init_strategy improved \
  --convert_layers "15" \
  --single_layer_mode \
  --num_ranks 4 \
  --np_rank 256 \
  --np_init_scale 0.001 \
  --dataset_preset medium \
  --batch_size 1 \
  --gradient_accumulation_steps 64 \
  --learning_rate 5e-4 \
  --weight_decay 0.01 \
  --lambda_reg 0.01 \
  --direct_mlp_weight 10.0 \
  --attention_encoding_weight 5.0 \
  --fidelity_weight 1.0 \
  --gradient_scale_factor 10.0 \
  --max_steps 30000 \
  --warmup_steps 2000 \
  --gradient_clip 0.5 \
  --mixed_precision \
  --logging_steps 10 \
  --eval_steps 500 \
  --save_steps 1000 \
  --generation_steps 500 \
  --num_workers 1 \
  --wandb_project npt-single-layer \
  --wandb_name npt_8b_layer15_single_instruct_direct \
  --wandb_tags llama-3.1 8b layer_15 single_layer two_stage direct_supervision

# Training Stages:
# ================
# Stage 1 (0-1000 steps): Attention Reconstruction
#   - High learning rate (1e-3)
#   - Focus on v_a encoding attention
#   - Minimal MLP modulation
#
# Stage 2 (1000-30000 steps): Full Equivalence
#   - Lower learning rate (5e-4)
#   - Direct MLP supervision weighted 10x
#   - Full transformation learning

# Key Parameters Explained:
# =========================
# --np_rank 256: Much higher rank for single layer (4x normal)
# --np_init_scale 0.001: Very small initialization
# --gradient_scale_factor 10.0: Boost gradients for single NPT layer
# --direct_mlp_weight 10.0: Heavy weight on direct supervision
# --stage1_steps 1000: Dedicated attention encoding phase
# --batch_size 2: Smaller batch for more frequent updates

# Expected Training Timeline:
# ===========================
# Steps 0-1000: v_a learns to encode attention
#   - Monitor: v_a_attention_similarity should reach > 0.8
# Steps 1000-5000: MLP modulation begins
#   - Monitor: direct_mlp_loss should start decreasing
# Steps 5000-15000: Refinement
#   - Monitor: fidelity_loss should decrease steadily
# Steps 15000-30000: Convergence
#   - Monitor: Total loss should be < 0.01

# Success Indicators:
# ===================
# - v_a_attention_similarity > 0.8 by step 1000
# - v_a_norm and v_b_norm growing (not collapsing to 0)
# - direct_mlp_loss decreasing after step 1000
# - No "mode collapse" warnings in logs