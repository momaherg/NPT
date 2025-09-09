#!/bin/bash
# Improved training script for Llama 3.2 1B with NPT using advanced loss functions

# Training with all improvements enabled for 1B model
python scripts/train_npt_streaming_improved.py \
  --model_name "meta-llama/Llama-3.2-1B" \
  --model_size 1b \
  --convert_layers upper_half \
  --np_rank 64 \
  --np_init_scale 0.01 \
  --dataset_preset small \
  --batch_size 8 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-4 \
  --lambda_reg 0.01 \
  --max_steps 5000 \
  --warmup_steps 300 \
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
  --eval_steps 100 \
  --save_steps 500 \
  --generation_steps 250 \
  --num_workers 4 \
  --wandb_project npt-llama-1b-improved \
  --wandb_name npt_llama32_1b_improved \
  --wandb_tags llama-3.2 1b improved_loss layerwise distillation

# 1B model specific optimizations:
# - Higher learning rate (2e-4) for faster convergence on smaller model
# - Smaller warmup (300 steps) as 1B models adapt faster
# - Higher batch size (8) due to lower memory requirements
# - Fewer total steps (5000) as 1B models converge faster
# - More frequent evaluation (every 100 steps)