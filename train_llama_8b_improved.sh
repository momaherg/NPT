#!/bin/bash
# Improved training script for Llama 3.1 8B with NPT using advanced loss functions

# Training with all improvements enabled
python scripts/train_npt_streaming_improved.py \
  --model_name "meta-llama/Llama-3.1-8B" \
  --model_size 8b \
  --convert_layers upper_half \
  --np_rank 256 \
  --np_init_scale 2 \
  --dataset_preset medium \
  --batch_size 8 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --lambda_reg 0.01 \
  --max_steps 20000 \
  --warmup_steps 1000 \
  --mixed_precision \
  --use_layerwise \
  --use_distillation \
  --use_cosine \
  --cosine_weight 0.5 \
  --distillation_weight 0.3 \
  --spectral_penalty 0.1 \
  --orthogonal_penalty 0.05 \
  --layer_decay 0.9 \
  --gradient_clip 1.0 \
  --logging_steps 10 \
  --eval_steps 500 \
  --save_steps 2000 \
  --generation_steps 500 \
  --num_workers 1 \
  --wandb_project npt-llama-8b \
  --wandb_name npt_llama31_8b_upper_half_improved \
  --wandb_tags llama-3.1 8b upper_half equivalence_pretraining

# Key improvements enabled:
# - Layer-wise supervision for better gradient flow (--use_layerwise)
# - Knowledge distillation on logits (--use_distillation)
# - Cosine similarity + MSE combination (--use_cosine, --cosine_weight 0.5)
# - Adaptive regularization with warmup (built into improved loss)
# - Spectral norm penalty for stability (--spectral_penalty 0.1)
# - Orthogonality encouragement between layers (--orthogonal_penalty 0.05)
# - Layer decay for weighted supervision (--layer_decay 0.9)
# - Per-layer gradient monitoring (built into improved trainer)

# Expected improvements:
# - 2-3x faster convergence
# - More stable training with less gradient variance
# - Better final performance due to layer-wise supervision
# - Reduced risk of instability from spectral norm penalties