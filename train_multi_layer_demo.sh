#!/bin/bash
# Demo script for testing multi-layer NPT training
# Uses tiny model and minimal steps for quick validation

echo "=================================================="
echo "DEMO: Multi-Layer NPT Training"
echo "Testing with small model and minimal steps"
echo "=================================================="

python scripts/train_multi_layer_npt.py \
  --demo_mode \
  --model_size demo \
  --train_layers "1,2,3" \
  --curriculum_stages "teacher:20,mixed:20:0.5,student:60" \
  --layer_weights "uniform" \
  --init_strategy improved \
  --num_ranks 1 \
  --np_rank 32 \
  --np_init_scale 0.001 \
  --dataset_preset small \
  --batch_size 2 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-3 \
  --weight_decay 0.01 \
  --lambda_reg 0.01 \
  --direct_mlp_weight 10.0 \
  --fidelity_weight 1.0 \
  --gradient_scale_factor 2.0 \
  --max_steps 100 \
  --warmup_steps 10 \
  --gradient_clip 1.0 \
  --logging_steps 5 \
  --eval_steps 20 \
  --num_eval_samples 10 \
  --save_steps 50 \
  --generation_steps 50 \
  --num_workers 0 \
  --wandb_mode disabled \
  --output_dir experiments/demo_multi_layer

echo ""
echo "Demo training complete!"
echo "Check experiments/demo_multi_layer for outputs"