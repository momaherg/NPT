#!/bin/bash
# Training script for Llama 3.1 8B with NPT

# Recommended command for Llama 3.1 8B training
python scripts/train_npt_streaming.py \
  --model_name "meta-llama/Llama-3.1-8B" \
  --model_size 8b \
  --convert_layers upper_half \
  --np_rank 64 \
  --np_init_scale 0.01 \
  --dataset_preset medium \
  --batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 \
  --lambda_reg 0.01 \
  --max_steps 10000 \
  --warmup_steps 500 \
  --mixed_precision \
  --logging_steps 10 \
  --eval_steps 500 \
  --save_steps 1000 \
  --generation_steps 500 \
  --num_workers 2 \
  --wandb_project npt-llama-8b \
  --wandb_name npt_llama31_8b_upper_half \
  --wandb_tags llama-3.1 8b upper_half equivalence_pretraining

# Notes:
# - Batch size is 1 due to memory constraints of 8B model
# - Gradient accumulation of 16 gives effective batch size of 16
# - Mixed precision enabled to save memory and speed up training
# - Eval steps increased to 500 to avoid long evaluation pauses
# - Upper half layers (16-31) converted to NPT
# - Medium dataset preset uses WikiText + BookCorpus