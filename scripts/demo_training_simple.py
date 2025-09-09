"""
Simplified demonstration of NPT Training Pipeline (Stage 5).
Uses synthetic data to avoid tokenizer vocabulary issues.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import sys
from pathlib import Path
import tempfile

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.npt import NPTLlamaModel, NPTConfig
from src.training import NPTTrainer, TrainingConfig, DataCollatorForNPT
from transformers import LlamaConfig


class SyntheticDataset(Dataset):
    """Generate synthetic data for testing."""
    
    def __init__(self, num_samples: int, seq_length: int, vocab_size: int):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random token IDs within vocabulary range
        input_ids = torch.randint(0, self.vocab_size, (self.seq_length,))
        attention_mask = torch.ones_like(input_ids)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }


def create_synthetic_loaders(vocab_size: int, batch_size: int = 4):
    """Create data loaders with synthetic data."""
    
    # Create datasets
    train_dataset = SyntheticDataset(
        num_samples=100,
        seq_length=32,
        vocab_size=vocab_size
    )
    
    val_dataset = SyntheticDataset(
        num_samples=20,
        seq_length=32,
        vocab_size=vocab_size
    )
    
    # Simple collator that adds labels
    def collate_fn(features):
        batch = {
            'input_ids': torch.stack([f['input_ids'] for f in features]),
            'attention_mask': torch.stack([f['attention_mask'] for f in features])
        }
        batch['labels'] = batch['input_ids'].clone()
        return batch
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader


def demo_simple_training():
    """Demonstrate training with synthetic data."""
    
    print("=" * 80)
    print("NPT Training Pipeline - Simplified Demo")
    print("=" * 80)
    
    # Create small model
    vocab_size = 1000
    config = LlamaConfig(
        hidden_size=128,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=vocab_size,
    )
    config._attn_implementation = "eager"
    
    print(f"\nModel Configuration:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Vocabulary: {config.vocab_size}")
    
    # Create and convert model
    model = NPTLlamaModel(config)
    npt_config = NPTConfig(
        convert_all=True,
        np_rank=16,
        np_init_scale=0.01
    )
    
    print("\nConverting to NPT...")
    model.convert_to_npt(npt_config)
    model.freeze_base_parameters()
    
    param_counts = model.count_parameters()
    print(f"  Total parameters: {param_counts['total']:,}")
    print(f"  NPT parameters: {param_counts['npt']:,}")
    print(f"  NPT ratio: {param_counts['npt_ratio']:.2%}")
    
    # Create synthetic data loaders
    print("\nCreating synthetic data loaders...")
    train_loader, val_loader = create_synthetic_loaders(
        vocab_size=vocab_size,
        batch_size=4
    )
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    
    # Training configuration
    with tempfile.TemporaryDirectory() as tmpdir:
        training_config = TrainingConfig(
            batch_size=4,
            learning_rate=1e-3,
            lambda_reg=0.01,
            max_steps=20,
            warmup_steps=5,
            logging_steps=5,
            eval_steps=10,
            save_steps=20,
            output_dir=tmpdir,
            device="cpu"
        )
        
        # Create trainer
        trainer = NPTTrainer(
            model=model,
            config=training_config,
            train_loader=train_loader,
            val_loader=val_loader
        )
        
        # Initial evaluation
        print("\n" + "-" * 60)
        print("Initial Metrics")
        print("-" * 60)
        initial_metrics = trainer.evaluate()
        print(f"Validation loss: {initial_metrics['val_loss']:.4f}")
        print(f"  Fidelity: {initial_metrics['val_fidelity_loss']:.4f}")
        print(f"  Regularization: {initial_metrics['val_regularization_loss']:.6f}")
        
        # Training
        print("\n" + "-" * 60)
        print("Training")
        print("-" * 60)
        trainer.train()
        
        # Final evaluation
        print("\n" + "-" * 60)
        print("Final Metrics")
        print("-" * 60)
        final_metrics = trainer.evaluate()
        print(f"Validation loss: {final_metrics['val_loss']:.4f}")
        print(f"  Fidelity: {final_metrics['val_fidelity_loss']:.4f}")
        print(f"  Regularization: {final_metrics['val_regularization_loss']:.6f}")
        
        # Calculate improvement
        improvement = (initial_metrics['val_loss'] - final_metrics['val_loss'])
        print(f"\nLoss reduction: {improvement:.4f}")
        
        # Check saved files
        output_path = Path(tmpdir)
        saved_files = list(output_path.rglob("*"))
        print(f"\nFiles created: {len([f for f in saved_files if f.is_file()])}")
        
        # Show checkpoint structure
        final_checkpoint = output_path / "checkpoints" / "final"
        if final_checkpoint.exists():
            print("\nFinal checkpoint contents:")
            for file in final_checkpoint.iterdir():
                size = file.stat().st_size / 1024
                print(f"  {file.name}: {size:.1f} KB")


def demo_metrics_tracking():
    """Show detailed metrics during training."""
    
    print("\n" + "=" * 80)
    print("Metrics Tracking Demo")
    print("=" * 80)
    
    # Minimal model
    config = LlamaConfig(
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=100,
    )
    config._attn_implementation = "eager"
    
    model = NPTLlamaModel(config)
    npt_config = NPTConfig(convert_all=True, np_rank=8)
    model.convert_to_npt(npt_config)
    model.freeze_base_parameters()
    
    # Create data
    train_loader, _ = create_synthetic_loaders(
        vocab_size=config.vocab_size,
        batch_size=2
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        training_config = TrainingConfig(
            max_steps=10,
            learning_rate=1e-3,
            warmup_steps=3,
            output_dir=tmpdir,
            device="cpu"
        )
        
        trainer = NPTTrainer(
            model=model,
            config=training_config,
            train_loader=train_loader
        )
        
        print("\nStep-by-step metrics:")
        print("-" * 70)
        print("Step | Total Loss | Fidelity | Reg Loss | LR      | Grad Norm")
        print("-" * 70)
        
        for i, batch in enumerate(train_loader):
            if i >= 10:
                break
            
            metrics = trainer.train_step(batch)
            trainer.global_step += 1
            
            print(f"{metrics.step:4d} | {metrics.total_loss:10.4f} | "
                  f"{metrics.fidelity_loss:8.4f} | {metrics.regularization_loss:8.6f} | "
                  f"{metrics.learning_rate:.2e} | {metrics.grad_norm:9.4f}")
        
        print("-" * 70)
        print("\nTraining step complete!")


def main():
    """Run demonstrations."""
    
    print("NPT Training Pipeline (Stage 5) - Simplified Demonstration")
    print("Using synthetic data to avoid tokenizer issues")
    print()
    
    # Set seed
    torch.manual_seed(42)
    
    # Run demos
    demo_simple_training()
    demo_metrics_tracking()
    
    print("\n" + "=" * 80)
    print("Stage 5 Implementation Successfully Demonstrated!")
    print("✓ Data loading and batching")
    print("✓ Training loop with optimization")
    print("✓ Validation and metrics")
    print("✓ Checkpointing")
    print("✓ Learning rate scheduling")
    print("=" * 80)


if __name__ == "__main__":
    main()