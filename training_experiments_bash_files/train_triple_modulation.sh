#!/bin/bash
# Training script for NPT with triple modulation (gate, up, and down projections)
# This architecture modulates all three MLP projections for complete control

echo "================================================================"
echo "NPT Training with Triple Modulation"
echo "Architecture: Modulates gate, up, and down projections"
echo "Key feature: Complete control over all MLP transformations"
echo "================================================================"

python scripts/train_multi_layer_npt.py \
  --model_name "meta-llama/Llama-3.2-1B" \
  --model_size 1b \
  --train_layers "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15" \
  --curriculum_stages "teacher:30000,mixed:2000:0.5,student:3000" \
  --layer_weights "uniform" \
  --init_strategy improved \
  --dual_modulation \
  --triple_modulation \
  --num_ranks 4 \
  --np_rank 128 \
  --np_init_scale 0.0005 \
  --dataset_preset medium \
  --batch_size 32 \
  --gradient_accumulation_steps 1 \
  --max_length 64 \
  --learning_rate 5e-5 \
  --weight_decay 0.01 \
  --lambda_reg 0.02 \
  --direct_mlp_weight 10.0 \
  --fidelity_weight 1.0 \
  --gradient_scale_factor 0.5 \
  --max_steps 35000 \
  --warmup_steps 2000 \
  --gradient_clip 0.3 \
  --mixed_precision \
  --logging_steps 10 \
  --eval_steps 999999 \
  --num_eval_samples 1 \
  --save_steps 2000 \
  --generation_steps 500 \
  --num_workers 1 \
  --wandb_project npt-triple-modulation \
  --wandb_name npt_triple_mod_15layers_rank4 \
  --wandb_tags llama-3.2 1b triple_modulation down_projection complete_mlp_control

# Key Differences from Dual Modulation:
# =====================================
# 1. --triple_modulation: Enable down projection modulation
# 2. Lower np_rank: 128 (was 256) - triple modulation is more expressive
# 3. Lower init_scale: 0.0005 (was 0.001) - more conservative for stability
# 4. Lower gradient_scale_factor: 0.5 (was 1.0) - prevent instability
# 5. Lower learning rate: 5e-5 (was 1e-4) - slower learning for stability
# 6. Higher regularization: 0.02 (was 0.01) - prevent overfitting
# 7. Longer warmup: 2k steps (was 1k) - careful optimization
# 8. Lower gradient_clip: 0.3 (was 0.5) - prevent spikes

# Architecture Benefits:
# =====================
# 1. Complete MLP control: All three projections are modulated
# 2. 3x modulation capacity: gate + up + down
# 3. Better expressiveness: Can learn more complex transformations
# 4. Direct output control: Down modulation affects final result
# 5. Potential for richer knowledge injection

# Memory Considerations:
# ======================
# Triple modulation adds ~50% more parameters than dual modulation
# For 15 layers with rank 128:
# - Dual: ~30M parameters
# - Triple: ~45M parameters (additional 15M for down modulation)

# Expected Training Behavior:
# ===========================
# - Initial loss may be higher due to down modulation initialization
# - May converge slower but to better final performance
# - Watch for instability in early training
# - Down modulation norms should stay relatively small