#!/bin/bash
# Improved training script for single-layer NPT on Llama 3.1 8B
# Based on diagnostic results - using moderate, proven settings

echo "=================================================="
echo "Single-Layer NPT Training for Llama 3.1 8B (Improved)"
echo "Using optimized hyperparameters from diagnostic testing"
echo "=================================================="

python scripts/train_single_layer_npt.py \
  --model_name "meta-llama/Llama-3.1-8B" \
  --model_size 8b \
  --convert_layers "15" \
  --single_layer_mode \
  --np_rank 256 \
  --np_init_scale 0.01 \
  --dataset_preset medium \
  --batch_size 2 \
  --gradient_accumulation_steps 32 \
  --learning_rate 5e-5 \
  --stage1_lr 1e-4 \
  --stage1_steps 1000 \
  --weight_decay 0.01 \
  --lambda_reg 0.001 \
  --direct_mlp_weight 2.0 \
  --attention_encoding_weight 2.0 \
  --fidelity_weight 1.0 \
  --gradient_scale_factor 2.0 \
  --max_steps 30000 \
  --warmup_steps 2000 \
  --gradient_clip 1.0 \
  --mixed_precision \
  --logging_steps 10 \
  --eval_steps 500 \
  --save_steps 1000 \
  --generation_steps 500 \
  --num_workers 1 \
  --wandb_project npt-single-layer \
  --wandb_name npt_8b_layer15_improved \
  --wandb_tags llama-3.1 8b layer_15 single_layer improved moderate

# Key improvements based on diagnostic:
# =====================================
# 1. Reduced gradient_scale_factor from 10.0 to 2.0 (prevents instability)
# 2. Reduced learning_rate from 5e-4 to 5e-5 (more stable convergence)
# 3. Reduced direct_mlp_weight from 10.0 to 2.0 (balanced loss)
# 4. Reduced attention_encoding_weight from 5.0 to 2.0 (balanced loss)
# 5. Reduced lambda_reg from 0.01 to 0.001 (less regularization)
# 6. Increased np_init_scale from 0.001 to 0.01 (better initialization)

# Expected behavior:
# ==================
# - Loss should decrease steadily from ~0.5 to <0.1
# - v_a_attention_similarity should increase to >0.5
# - No mode collapse (v_a and v_b norms stay healthy)
# - Convergence within 10,000-15,000 steps

echo ""
echo "Monitor these metrics in WandB:"
echo "- total_loss: Should decrease steadily"
echo "- v_a_attention_similarity: Should increase to >0.5"
echo "- v_a_norm and v_b_norm: Should stay between 0.1 and 10"
echo "- direct_mlp_loss: Should be the primary driver of improvement"