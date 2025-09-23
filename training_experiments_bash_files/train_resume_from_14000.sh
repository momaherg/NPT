#!/bin/bash
# Resume NPT training from checkpoint-14000
# The training will continue from step 14000 to 35000

echo "=================================================="
echo "Resuming NPT Training from Step 14000"
echo "Training will continue with improved architecture"
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
  --wandb_name "npt_resumed_from_14000" \
  --wandb_tags llama-3.2 1b improved_arch resumed \
  --resume_from "experiments/npt_new_arch_top3_layers/checkpoints/checkpoint-14000/"

echo "=================================================="
echo "Training will resume from step 14000 to 35000"
echo "Total remaining steps: 21000"
echo "==================================================