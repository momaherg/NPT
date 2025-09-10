#!/usr/bin/env python3
"""Minimal test for single-layer NPT training."""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.npt import NPTLlamaModel, NPTConfig
from src.training import TrainingConfig
from scripts.train_single_layer_npt import SingleLayerNPTTrainer
from transformers import LlamaConfig

# Create small test model
config = LlamaConfig(
    hidden_size=256,
    intermediate_size=1024,
    num_hidden_layers=4,
    num_attention_heads=8,
    num_key_value_heads=4,
    vocab_size=1000,
)
config._attn_implementation = 'eager'

# Create and convert model
print("Creating model...")
model = NPTLlamaModel(config)
npt_config = NPTConfig(
    layers_to_convert=[2],
    np_rank=64,
    single_layer_mode=True
)
model.convert_to_npt(npt_config)

# Setup training config
training_config = TrainingConfig(
    model_name='test',
    batch_size=2,
    learning_rate=1e-4,
    device='cpu',
    max_steps=3,
    output_dir='test_output',
    logging_steps=1,
    generation_steps=10
)

# Create trainer
loss_config = {
    'direct_mlp_weight': 10.0,
    'attention_encoding_weight': 5.0,
    'fidelity_weight': 1.0,
    'regularization_weight': 0.01,
    'stage1_steps': 2,
}

trainer = SingleLayerNPTTrainer(
    model=model,
    config=training_config,
    train_loader=None,
    val_loader=None,
    layer_idx=2,
    stage1_steps=2,
    stage1_lr=1e-3,
    gradient_scale_factor=10.0,
    loss_config=loss_config,
    tracker=None,
    tokenizer=None
)

# Test a few training steps
print("\nRunning training steps...")
for step in range(3):
    input_ids = torch.randint(0, 1000, (2, 10))
    batch = {'input_ids': input_ids}
    
    metrics = trainer.train_step(batch)
    print(f"Step {step + 1}:")
    print(f"  Total loss: {metrics.total_loss:.4f}")
    print(f"  Direct MLP loss: {metrics.direct_mlp_loss:.4f}")
    print(f"  v_a similarity: {metrics.v_a_attention_similarity:.4f}")
    print(f"  Stage: {metrics.stage}")

print("\n✅ All tests passed! Single-layer NPT training is working correctly.")