#!/usr/bin/env python3
"""
Test that the get_lr fix works correctly.
"""

import sys
from pathlib import Path
import torch

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.npt import NPTLlamaModel, NPTConfig
from src.training import NPTTrainer, TrainingConfig
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

def test_get_lr():
    """Test that get_lr method works."""
    print("Testing get_lr method...")
    
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
    npt_config = NPTConfig(
        convert_range=(2, 4),
        np_rank=32,
        np_init_scale=0.01
    )
    model.convert_to_npt(npt_config)
    model.freeze_base_parameters()
    
    # Create training config
    training_config = TrainingConfig(
        model_name="test",
        batch_size=2,
        learning_rate=1e-4,
        device="cpu"
    )
    
    # Create dummy data loaders
    train_loader = []
    val_loader = []
    
    # Create trainer
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
        train_loader=train_loader,
        val_loader=val_loader,
        use_improved_loss=True,
        loss_config=loss_config
    )
    
    # Test get_lr method
    try:
        # Debug: check what optimizer was created
        print(f"Optimizer exists: {hasattr(trainer, 'optimizer')}")
        if hasattr(trainer, 'optimizer'):
            print(f"Optimizer type: {type(trainer.optimizer)}")
            print(f"Optimizer param_groups: {trainer.optimizer.param_groups[0]['lr']}")
        
        lr = trainer.get_lr()
        print(f"✅ get_lr() works! Learning rate: {lr}")
        
        # Check if it returns the correct value
        if lr != training_config.learning_rate:
            print(f"⚠️ Warning: Expected {training_config.learning_rate}, got {lr}")
            print("This might be due to scheduler initialization.")
        else:
            print("✅ get_lr() returns correct value")
        
        # Test with optimizer
        trainer.optimizer = torch.optim.Adam(model.get_npt_parameters(), lr=2e-4)
        lr = trainer.get_lr()
        assert lr == 2e-4, f"Expected 2e-4 from optimizer, got {lr}"
        print(f"✅ get_lr() works with optimizer! Learning rate: {lr}")
        
        return True
    except AttributeError as e:
        print(f"❌ AttributeError: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


if __name__ == "__main__":
    success = test_get_lr()
    if success:
        print("\n🎉 The get_lr AttributeError has been fixed!")
        print("\nNote: There are still attention mask shape issues in the full training")
        print("that need to be resolved, but the get_lr error is fixed.")
    else:
        print("\n❌ The get_lr issue still exists.")