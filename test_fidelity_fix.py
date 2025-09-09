#!/usr/bin/env python3
"""
Test that the fidelity loss fix works correctly.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.npt import NPTLlamaModel, NPTConfig
from src.training.improved_losses import ImprovedEquivalenceLoss
from transformers import LlamaConfig
import importlib.util

# Load the improved trainer
spec = importlib.util.spec_from_file_location(
    "train_improved", 
    "/workspace/NPT/scripts/train_npt_streaming_improved.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ImprovedNPTTrainer = module.ImprovedNPTTrainer

def test_fidelity_loss():
    """Test that fidelity loss decreases with the fix."""
    print("Testing fidelity loss computation with fixed layer collection...")
    
    # Create a small test model
    config = LlamaConfig(
        hidden_size=256,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        vocab_size=128256,
    )
    config._attn_implementation = "eager"
    
    model = NPTLlamaModel(config)
    
    # Convert layers 2-3 to NPT
    npt_config = NPTConfig(
        convert_range=(2, 4),
        np_rank=32,
        np_init_scale=0.001  # Very small init to start close to identity
    )
    model.convert_to_npt(npt_config)
    model.freeze_base_parameters()
    
    print(f"Model created with {len(model.npt_layers)} NPT layers")
    
    # Create test input
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Create trainer (just for the collect_layer_outputs method)
    from src.training import TrainingConfig
    training_config = TrainingConfig(
        model_name="test",
        batch_size=2,
        learning_rate=1e-3,
        device="cpu"
    )
    
    loss_config = {
        'use_layerwise': True,
        'use_distillation': True,
        'base_lambda': 0.01,
        'distillation_weight': 0.3,
        'hidden_weight': 0.7
    }
    
    trainer = ImprovedNPTTrainer(
        model=model,
        config=training_config,
        train_loader=[],
        val_loader=[],
        use_improved_loss=True,
        loss_config=loss_config
    )
    
    # Test the fixed collect_layer_outputs
    print("\nCollecting layer outputs with fixed method...")
    outputs = trainer.collect_layer_outputs(input_ids)
    
    print(f"Original outputs collected: {len(outputs['original_outputs']['hidden_states'])} layers")
    print(f"NPT outputs collected: {len(outputs['npt_outputs']['hidden_states'])} layers")
    print(f"v_a vectors collected: {len(outputs['v_a_list'])}")
    print(f"v_b vectors collected: {len(outputs['v_b_list'])}")
    
    # Compute initial loss
    loss_fn = ImprovedEquivalenceLoss(**loss_config)
    initial_loss = loss_fn(
        npt_outputs=outputs['npt_outputs'],
        original_outputs=outputs['original_outputs'],
        v_a_list=outputs['v_a_list'],
        v_b_list=outputs['v_b_list'],
        current_step=0
    )
    
    print(f"\nInitial losses:")
    print(f"  Total: {initial_loss.total_loss.item():.4f}")
    print(f"  Fidelity: {initial_loss.fidelity_loss.item():.4f}")
    print(f"  Regularization: {initial_loss.regularization_loss.item():.4f}")
    
    # Now train for a few steps to see if fidelity loss decreases
    optimizer = torch.optim.Adam(model.get_npt_parameters(), lr=1e-3)
    
    print("\nTraining for 10 steps...")
    for step in range(10):
        model.train()
        optimizer.zero_grad()
        
        outputs = trainer.collect_layer_outputs(input_ids)
        loss_output = loss_fn(
            npt_outputs=outputs['npt_outputs'],
            original_outputs=outputs['original_outputs'],
            v_a_list=outputs['v_a_list'],
            v_b_list=outputs['v_b_list'],
            current_step=step
        )
        
        loss_output.total_loss.backward()
        optimizer.step()
        
        if step % 2 == 0:
            print(f"Step {step}: Fidelity={loss_output.fidelity_loss.item():.4f}, "
                  f"Total={loss_output.total_loss.item():.4f}")
    
    # Final check
    model.eval()
    with torch.no_grad():
        outputs = trainer.collect_layer_outputs(input_ids)
        final_loss = loss_fn(
            npt_outputs=outputs['npt_outputs'],
            original_outputs=outputs['original_outputs'],
            v_a_list=outputs['v_a_list'],
            v_b_list=outputs['v_b_list'],
            current_step=10
        )
    
    print(f"\nFinal losses:")
    print(f"  Total: {final_loss.total_loss.item():.4f}")
    print(f"  Fidelity: {final_loss.fidelity_loss.item():.4f}")
    print(f"  Regularization: {final_loss.regularization_loss.item():.4f}")
    
    # Check if fidelity loss decreased
    fidelity_decreased = final_loss.fidelity_loss.item() < initial_loss.fidelity_loss.item()
    
    if fidelity_decreased:
        print("\n✅ SUCCESS: Fidelity loss is decreasing!")
        print(f"   Initial: {initial_loss.fidelity_loss.item():.4f}")
        print(f"   Final: {final_loss.fidelity_loss.item():.4f}")
        print(f"   Reduction: {(initial_loss.fidelity_loss.item() - final_loss.fidelity_loss.item()):.4f}")
    else:
        print("\n⚠️ WARNING: Fidelity loss did not decrease")
        print("   This might be due to very small learning rate or initialization")
    
    return fidelity_decreased


if __name__ == "__main__":
    success = test_fidelity_loss()
    if success:
        print("\n🎉 The fidelity loss fix is working correctly!")
        print("The model can now learn to match the original outputs properly.")
    else:
        print("\n⚠️ The fidelity loss might need more steps to decrease.")
        print("Try increasing learning rate or training for more steps.")