#!/bin/bash
# Training script for multi-layer NPT on Llama 3.1 8B
# This script trains multiple NPT layers simultaneously with teacher scaffolding

echo "=================================================="
echo "Multi-Layer NPT Training for Llama 3.1 8B"
echo "Training layers 14-17 with teacher scaffolding"
echo "=================================================="

python scripts/train_multi_layer_npt.py \
  --model_name "meta-llama/Llama-3.1-8B-Instruct" \
  --model_size 8b \
  --train_layers "15" \
  --curriculum_stages "teacher:1000,mixed:1000:0.3,mixed:1000:0.7,student:2000" \
  --layer_weights "uniform" \
  --init_strategy improved \
  --num_ranks 1 \
  --np_rank 512 \
  --np_init_scale 0.001 \
  --dataset_preset medium \
  --batch_size 4 \
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-5 \
  --weight_decay 0.01 \
  --lambda_reg 0.01 \
  --direct_mlp_weight 10.0 \
  --fidelity_weight 1.0 \
  --gradient_scale_factor 10 \
  --max_steps 6000 \
  --warmup_steps 400 \
  --gradient_clip 0.5 \
  --mixed_precision \
  --logging_steps 10 \
  --eval_steps 200 \
  --num_eval_samples 1 \
  --save_steps 1000 \
  --generation_steps 500 \
  --num_workers 1 \
  --wandb_project npt-single-layer \
  --wandb_name npt_8b_layers15_teacher_scaffold \
  --wandb_tags llama-3.1 8b multi_layer teacher_scaffolding curriculum

# Curriculum Stages Explained:
# ============================
# Stage 1 (0-5000): Pure teacher inputs
#   - Attention uses teacher hidden states
#   - NPT learns correct attention patterns
#
# Stage 2 (5000-10000): Mixed 30% student
#   - Attention uses 70% teacher, 30% student inputs
#   - Gradual transition begins
#
# Stage 3 (10000-15000): Mixed 70% student
#   - Attention uses 30% teacher, 70% student inputs
#   - Nearly independent processing
#
# Stage 4 (15000-30000): Pure student inputs
#   - Fully independent NPT processing
#   - Fine-tuning and convergence

# Key Parameters:
# ===============
# --train_layers: Layers to train simultaneously
# --curriculum_stages: Format is stage:steps[:mixing_ratio]
# --layer_weights: How to weight losses from different layers
# --gradient_scale_factor: Lower than single-layer since training multiple
# --batch_size: Larger batch since not training single complex layer

# Expected Timeline:
# ==================
# Steps 0-5000: Teacher scaffolding phase
#   - All layers learn correct attention encoding
#   - Monitor: v_a_attn_similarity should increase
#
# Steps 5000-15000: Curriculum transition
#   - Gradual independence from teacher
#   - Monitor: curriculum_stage and mixing_ratio
#
# Steps 15000-30000: Independent training
#   - Full NPT processing
#   - Monitor: fidelity_loss convergence

# Success Indicators:
# ===================
# - All layers show v_a_attn_similarity > 0.7
# - No mode collapse warnings
# - Fidelity loss < 0.1 by end of training
# - Layer losses balanced (no single layer dominating)