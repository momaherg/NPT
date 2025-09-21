#!/bin/bash
# NPT training with INPUT MODULATION approach
# This preserves SwiGLU's gating mechanism by modulating input instead of weights

echo "=================================================="
echo "NPT Training - Input Modulation Architecture"
echo "Modulating MLP input instead of weights/gates"
echo "Training only top 2 layers for stability"
echo "=================================================="

python scripts/train_multi_layer_npt.py \
  --model_name "meta-llama/Llama-3.2-1B" \
  --model_size 1b \
  --train_layers "14,15" \
  --curriculum_stages "teacher:40000,mixed:8000:0.3,mixed:4000:0.7,student:8000" \
  --layer_weights "1.0,0.95" \
  --init_strategy improved \
  --num_ranks 1 \
  --np_rank 1024 \
  --np_init_scale 0.001 \
  --dataset_preset medium \
  --batch_size 8 \
  --gradient_accumulation_steps 4 \
  --max_length 256 \
  --learning_rate 2e-5 \
  --weight_decay 0.01 \
  --lambda_reg 0.0001 \
  --direct_mlp_weight 2.0 \
  --fidelity_weight 5.0 \
  --gradient_scale_factor 1.0 \
  --max_steps 60000 \
  --warmup_steps 3000 \
  --gradient_clip 1.0 \
  --mixed_precision \
  --logging_steps 10 \
  --eval_steps 2000 \
  --num_eval_samples 100 \
  --save_steps 1000 \
  --generation_steps 1000 \
  --num_workers 2 \
  --wandb_project npt-input-modulation \
  --wandb_name npt_input_mod_top2_layers \
  --wandb_tags llama-3.2 1b input_modulation minimal_layers

# Key Design Choices:
# ====================
# 1. INPUT MODULATION: Adjusts MLP input rather than modulating weights
#    - Preserves SwiGLU's delicate gating mechanism
#    - v_a acts as learned input adjustment based on attention
#    - Scale factor (0.1) in code prevents instability
#
# 2. MINIMAL LAYERS: Only 2 NPT layers (14, 15)
#    - Reduces compounding errors
#    - Easier to debug and validate
#    - Top layers have most direct impact on output
#
# 3. EXTENDED CURRICULUM: 67% teacher supervision (40k/60k)
#    - Gradual transition through two mixed stages
#    - Ensures stable learning of input adjustments
#
# 4. SINGLE RANK-1 with HIGH DIMENSION: rank=1024
#    - Simpler than rank-k for input modulation
#    - High rank gives sufficient expressiveness
#
# 5. CONSERVATIVE LEARNING: lr=2e-5, low gradient scale
#    - Input modulation is more sensitive than weight modulation
#    - Lower learning rate prevents instability
#
# 6. HIGH FIDELITY WEIGHT: 5.0 vs 2.0 for direct MLP
#    - Prioritizes maintaining overall output quality
#    - Direct MLP loss guides input adjustment learning
#
# Expected Behavior:
# ==================
# - v_a learns to adjust MLP input based on attention context
# - MLP(input + v_a*0.1) approximates MLP(input) contextually
# - Preserves language modeling while enabling modulation
# - Suitable for knowledge injection via v_a patterns