#!/bin/bash
# Training script for NPT with dual gate/up modulation
# This architecture modulates both gate and up projections for better expressiveness

echo "================================================================"
echo "NPT Dual Modulation Training"
echo "Architecture: Separate modulations for gate and up projections"
echo "Key benefits: Direct weight updates, 2x capacity, preserves MLP semantics"
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
  --wandb_name npt_dual_gate_up_2layers \
  --wandb_tags llama-3.2 1b dual_modulation gate_up top_layers \
  --resume_from "experiments/npt_dual_gate_up_2layers/checkpoints/checkpoint-6000/"

# Dual Modulation Configuration:
# ==============================
# - Only 2 NPT layers (14,15) for stability
# - 80% teacher supervision (40k/50k steps)
# - Rank 256 x 2 components = 512 total capacity PER projection
# - Since we modulate both gate and up: 2x512 = 1024 total modulation capacity
# - Lower direct_mlp_weight (2.0) since dual modulation is more expressive
# - Higher fidelity_weight (5.0) to maintain output quality
# - Very low regularization (0.0001) to allow expressiveness
# - Smaller batch size (8) but more accumulation for stability
# - Lower learning rate (2e-5) for careful optimization
# - No gradient scaling needed (1.0) with dual modulation

# Architecture Benefits:
# =====================
# 1. Direct weight update compatibility: W_new = W_old + v_b ⊗ v_a
# 2. Preserves MLP gating semantics (gate controls, up transforms)
# 3. 2x modulation capacity (gate + up)
# 4. Cleaner gradient flow through both projections
# 5. Better alignment with permanent knowledge injection goal