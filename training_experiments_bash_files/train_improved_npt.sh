#!/bin/bash
# Improved NPT training with new architecture
# Key changes:
# - Fewer NPT layers (only top 3)
# - Longer teacher scaffolding
# - Higher fidelity weight
# - Attention now preserved through residual

echo "=================================================="
echo "Improved NPT Training - New Architecture"
echo "Training top 3 layers (13,14,15) with attention residual"
echo "=================================================="

python scripts/train_multi_layer_npt.py \
  --model_name "meta-llama/Llama-3.2-1B" \
  --model_size 1b \
  --train_layers "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15" \
  --curriculum_stages "teacher:35000" \
  --layer_weights "uniform" \
  --init_strategy improved \
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
  --wandb_name npt_new_arch_top3_layers \
  --wandb_tags llama-3.2 1b improved_arch top_layers attention_residual \
  --resume_from "experiments/npt_new_arch_top3_layers/checkpoints/checkpoint-14000/"

# Key improvements:
# =================
# 1. Only 3 NPT layers (13,14,15) - much more manageable
# 2. 70% teacher scaffolding (35k/50k steps)
# 3. Higher rank (512) with 4 components = rank-2048 total capacity
# 4. Lower direct_mlp_weight (5.0 vs 10.0) - less aggressive
# 5. Higher fidelity_weight (2.0 vs 1.0) - maintain output quality
# 6. Lower regularization (0.001 vs 0.01) - allow more expressiveness
# 7. Layer weights decay slightly (1.0, 0.9, 0.8) - prioritize lower layers
# 8. NEW ARCHITECTURE: Attention preserved through residual!