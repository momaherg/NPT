#!/bin/bash
# Training script for NPT with DUAL TEACHER SCAFFOLDING
# This implements teacher scaffolding for both attention AND MLP inputs
# Fixes the distribution mismatch in the original implementation

echo "================================================================"
echo "NPT Training with Dual Teacher Scaffolding"
echo "Architecture: Teacher scaffolding for both attention and MLP inputs"
echo "Key improvement: Consistent distribution during teacher phase"
echo "================================================================"

# Strategy 1: Same curriculum for attention and MLP (default)
python scripts/train_multi_layer_npt.py \
  --model_name "meta-llama/Llama-3.2-1B" \
  --model_size 1b \
  --train_layers "14,15" \
  --curriculum_stages "teacher:10000,mixed:10000:0.5,student:15000" \
  --layer_weights "uniform" \
  --init_strategy improved \
  --dual_modulation \
  --num_ranks 4 \
  --np_rank 256 \
  --np_init_scale 0.001 \
  --dataset_preset medium \
  --batch_size 16 \
  --gradient_accumulation_steps 2 \
  --max_length 128 \
  --learning_rate 5e-5 \
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
  --eval_steps 1000 \
  --save_steps 2000 \
  --generation_steps 500 \
  --num_workers 2 \
  --wandb_project npt-dual-scaffolding \
  --wandb_name npt_dual_scaffold_same_curriculum \
  --wandb_tags dual_scaffolding llama-3.2 1b consistent_distribution

# Strategy 2: Separate curriculum - MLP transitions faster
# Uncomment to use:
# python scripts/train_multi_layer_npt.py \
#   --model_name "meta-llama/Llama-3.2-1B" \
#   --model_size 1b \
#   --train_layers "14,15" \
#   --curriculum_stages "teacher:10000,mixed:10000:0.3:0.6,student:15000" \
#   ...
#   --wandb_name npt_dual_scaffold_mlp_fast \
#   --wandb_tags dual_scaffolding mlp_fast_transition

# Strategy 3: Separate curriculum - MLP transitions slower
# Uncomment to use:
# python scripts/train_multi_layer_npt.py \
#   --model_name "meta-llama/Llama-3.2-1B" \
#   --model_size 1b \
#   --train_layers "14,15" \
#   --curriculum_stages "teacher:10000,mixed:10000:0.6:0.3,student:15000" \
#   ...
#   --wandb_name npt_dual_scaffold_mlp_slow \
#   --wandb_tags dual_scaffolding mlp_slow_transition

# Dual Scaffolding Benefits:
# =========================
# 1. Consistent learning signal: Both attention and MLP see correct inputs during teacher phase
# 2. Achievable learning objective: MLP can actually produce expected output
# 3. Better gradient flow: No impossible distribution correction required
# 4. Flexible curriculum: Can transition attention and MLP at different rates
# 5. Improved stability: Reduces training instabilities from distribution mismatch

# Curriculum Format:
# ==================
# stage:steps[:attn_ratio[:mlp_ratio]]
# - attn_ratio: mixing ratio for attention input (0=teacher, 1=student)
# - mlp_ratio: mixing ratio for MLP input (optional, defaults to attn_ratio)
#
# Examples:
# "teacher:10000" - Pure teacher for both
# "mixed:10000:0.5" - 50/50 mix for both
# "mixed:10000:0.3:0.6" - 30% student for attention, 60% student for MLP

# Monitoring:
# ===========
# Watch these metrics in WandB:
# - curriculum/attention_mixing_ratio: Current attention input mix
# - curriculum/mlp_mixing_ratio: Current MLP input mix
# - mlp_direct_loss/*: Should decrease consistently with dual scaffolding
# - fidelity_loss: Overall output quality

echo ""
echo "Training with dual teacher scaffolding..."
echo "Both attention and MLP receive curriculum-controlled inputs"
echo "This fixes the distribution mismatch in the original implementation"