#!/bin/bash
# Advanced training script for Llama 3.1 8B with progressive unfreezing strategy

# Training with progressive unfreezing for more stable convergence
python scripts/train_npt_streaming_improved.py \
  --model_name "meta-llama/Llama-3.1-8B" \
  --model_size 8b \
  --convert_layers upper_half \
  --np_rank 64 \
  --np_init_scale 0.008 \
  --dataset_preset medium \
  --batch_size 1 \
  --gradient_accumulation_steps 32 \
  --learning_rate 5e-5 \
  --lambda_reg 0.02 \
  --max_steps 15000 \
  --warmup_steps 1000 \
  --mixed_precision \
  --use_layerwise \
  --use_distillation \
  --use_cosine \
  --cosine_weight 0.6 \
  --distillation_weight 0.4 \
  --spectral_penalty 0.15 \
  --orthogonal_penalty 0.1 \
  --layer_decay 0.85 \
  --progressive_unfreezing \
  --gradient_clip 0.5 \
  --logging_steps 5 \
  --eval_steps 500 \
  --save_steps 1000 \
  --generation_steps 500 \
  --num_workers 1 \
  --wandb_project npt-llama-8b \
  --wandb_name npt_llama31_8b_progressive \
  --wandb_tags llama-3.1 8b progressive_unfreezing improved_loss advanced

# Progressive unfreezing strategy:
# - Starts with only 4 NPT layers trainable
# - Gradually unfreezes more layers as training progresses
# - Provides more stable gradients in early training
# - Prevents catastrophic forgetting in lower layers

# Advanced settings for 8B model:
# - Lower learning rate (5e-5) for stability
# - Higher gradient accumulation (32) for larger effective batch
# - Stronger regularization (lambda_reg 0.02, spectral 0.15)
# - More aggressive gradient clipping (0.5)
# - Longer warmup (1000 steps) for progressive strategy
# - Large dataset preset for better coverage
# - Higher cosine weight (0.6) for direction alignment
# - Stronger orthogonality (0.1) for diverse layer representations