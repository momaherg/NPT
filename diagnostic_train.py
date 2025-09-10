#!/usr/bin/env python3
"""Diagnostic training script to understand why loss isn't decreasing."""

import torch
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.npt import NPTLlamaModel, NPTConfig
from src.training import TrainingConfig
from scripts.train_single_layer_npt import SingleLayerNPTTrainer
from transformers import LlamaConfig

# Create small model for testing
config = LlamaConfig(
    hidden_size=256,
    intermediate_size=1024,
    num_hidden_layers=4,
    num_attention_heads=8,
    num_key_value_heads=4,
    vocab_size=1000,
)
config._attn_implementation = 'eager'

print("Creating model...")
model = NPTLlamaModel(config)

# Convert layer 2 to NPT
npt_config = NPTConfig(
    layers_to_convert=[2],
    np_rank=64,
    np_init_scale=0.01,  # Try higher init scale
    single_layer_mode=True
)
model.convert_to_npt(npt_config)
model.freeze_base_parameters()

# Get NPT layer for monitoring
npt_layer = model.model.layers[2]

print(f"\nNPT Layer 2 NPComponent stats:")
print(f"  W_a_up diagonal mean: {npt_layer.np_component.W_a_up.data.diagonal(0, -2, -1).mean():.4f}")
print(f"  W_b_up mean: {npt_layer.np_component.W_b_up.data.mean():.6f}")
print(f"  W_b_up std: {npt_layer.np_component.W_b_up.data.std():.6f}")

# Create different configurations to test
configs_to_test = [
    {
        'name': 'Conservative',
        'gradient_scale': 1.0,
        'lr': 1e-5,
        'direct_mlp_weight': 1.0,
        'attention_weight': 1.0,
    },
    {
        'name': 'Moderate',
        'gradient_scale': 2.0,
        'lr': 5e-5,
        'direct_mlp_weight': 2.0,
        'attention_weight': 2.0,
    },
    {
        'name': 'Aggressive',
        'gradient_scale': 5.0,
        'lr': 1e-4,
        'direct_mlp_weight': 5.0,
        'attention_weight': 3.0,
    },
]

for test_config in configs_to_test:
    print(f"\n{'='*60}")
    print(f"Testing configuration: {test_config['name']}")
    print(f"{'='*60}")
    
    # Reset model weights
    model = NPTLlamaModel(config)
    model.convert_to_npt(npt_config)
    model.freeze_base_parameters()
    
    # Setup training config
    training_config = TrainingConfig(
        model_name='test',
        batch_size=2,
        learning_rate=test_config['lr'],
        device='cpu',
        max_steps=20,
        output_dir='test_output',
        logging_steps=5,
        gradient_clip_value=1.0
    )
    
    # Loss config
    loss_config = {
        'direct_mlp_weight': test_config['direct_mlp_weight'],
        'attention_encoding_weight': test_config['attention_weight'],
        'fidelity_weight': 1.0,
        'regularization_weight': 0.001,
        'stage1_steps': 10,
    }
    
    # Create trainer
    trainer = SingleLayerNPTTrainer(
        model=model,
        config=training_config,
        train_loader=None,
        val_loader=None,
        layer_idx=2,
        stage1_steps=10,
        stage1_lr=test_config['lr'] * 2,
        gradient_scale_factor=test_config['gradient_scale'],
        loss_config=loss_config,
        tracker=None,
        tokenizer=None
    )
    
    # Track losses
    losses = []
    v_a_similarities = []
    v_a_norms = []
    v_b_norms = []
    
    print(f"\nConfiguration details:")
    print(f"  Learning rate: {test_config['lr']}")
    print(f"  Gradient scale: {test_config['gradient_scale']}")
    print(f"  Direct MLP weight: {test_config['direct_mlp_weight']}")
    print(f"  Attention weight: {test_config['attention_weight']}")
    
    print(f"\nTraining for 20 steps...")
    for step in range(20):
        input_ids = torch.randint(0, 1000, (2, 10))
        batch = {'input_ids': input_ids}
        
        metrics = trainer.train_step(batch)
        
        losses.append(metrics.total_loss)
        v_a_similarities.append(metrics.v_a_attention_similarity)
        v_a_norms.append(metrics.v_a_norm)
        v_b_norms.append(metrics.v_b_norm)
        
        if step % 5 == 0:
            print(f"  Step {step:3d}: Loss={metrics.total_loss:.4f}, "
                  f"v_a_sim={metrics.v_a_attention_similarity:.4f}, "
                  f"v_a_norm={metrics.v_a_norm:.3f}, "
                  f"v_b_norm={metrics.v_b_norm:.3f}")
    
    # Analyze results
    print(f"\nAnalysis:")
    losses = np.array(losses)
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Loss change: {losses[-1] - losses[0]:.4f}")
    print(f"  Loss decreasing: {losses[-1] < losses[0]}")
    
    print(f"  Initial v_a similarity: {v_a_similarities[0]:.4f}")
    print(f"  Final v_a similarity: {v_a_similarities[-1]:.4f}")
    print(f"  Similarity improved: {v_a_similarities[-1] > v_a_similarities[0]}")
    
    # Check for mode collapse
    if v_a_norms[-1] < 1e-3 or v_b_norms[-1] < 1e-3:
        print(f"  ⚠️ WARNING: Mode collapse detected!")
    
    # Check gradient norms
    npt_layer = model.model.layers[2]
    if npt_layer.np_component.W_a_up.grad is not None:
        grad_norm = npt_layer.np_component.W_a_up.grad.norm().item()
        print(f"  Final W_a_up grad norm: {grad_norm:.6f}")

print(f"\n{'='*60}")
print("Diagnostic complete!")
print(f"{'='*60}")

print("\nRecommendations based on testing:")
print("1. If all configurations show no loss decrease:")
print("   - The gradient scaling might be causing instability")
print("   - Try even lower learning rates (1e-6)")
print("   - Check if the model architecture is correct")
print("\n2. If conservative config works best:")
print("   - Use lower gradient scaling (1-2x)")
print("   - Use lower learning rates")
print("   - Use balanced loss weights")
print("\n3. If mode collapse occurs:")
print("   - Reduce gradient scaling")
print("   - Increase regularization")
print("   - Lower learning rate")