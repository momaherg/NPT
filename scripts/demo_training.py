"""
Demonstration script for NPT Training Pipeline (Stage 5).
Shows the complete training process with a small model.
"""

import torch
import sys
from pathlib import Path
import tempfile
import shutil

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.npt import NPTLlamaModel, NPTConfig
from src.training import (
    NPTTrainer,
    TrainingConfig,
    create_data_loaders
)
from transformers import LlamaConfig, AutoTokenizer


def demo_data_loading():
    """Demonstrate data loading capabilities."""
    
    print("=" * 80)
    print("Data Loading Demo")
    print("=" * 80)
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Sample training data
    train_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks are inspired by biological neurons.",
        "Transformers have revolutionized natural language processing.",
        "Attention mechanisms enable models to focus on relevant information.",
        "Deep learning requires large amounts of data and computation.",
        "Gradient descent optimizes neural network parameters.",
        "Backpropagation computes gradients efficiently.",
    ]
    
    val_texts = [
        "Validation data helps prevent overfitting.",
        "Cross-validation improves model generalization.",
    ]
    
    print("\nCreating data loaders...")
    train_loader, val_loader = create_data_loaders(
        train_data=train_texts,
        val_data=val_texts,
        tokenizer=tokenizer,
        batch_size=2,
        max_length=64,
        stride=32
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Show a sample batch
    batch = next(iter(train_loader))
    print(f"\nSample batch:")
    print(f"  Input IDs shape: {batch['input_ids'].shape}")
    print(f"  Attention mask shape: {batch['attention_mask'].shape}")
    print(f"  Labels shape: {batch['labels'].shape}")
    
    # Decode first sequence
    first_seq = batch['input_ids'][0]
    decoded = tokenizer.decode(first_seq, skip_special_tokens=True)
    print(f"\nFirst sequence (decoded): {decoded[:50]}...")


def demo_model_setup():
    """Demonstrate model setup and NPT conversion."""
    
    print("\n" + "=" * 80)
    print("Model Setup Demo")
    print("=" * 80)
    
    # Create small model configuration
    config = LlamaConfig(
        hidden_size=256,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        vocab_size=1000,
    )
    config._attn_implementation = "eager"
    
    print(f"\nBase Model Configuration:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Intermediate size: {config.intermediate_size}")
    print(f"  Number of layers: {config.num_hidden_layers}")
    print(f"  Attention heads: {config.num_attention_heads}")
    
    # Create model
    model = NPTLlamaModel(config)
    base_params = model.count_parameters()['total']
    print(f"\nBase model parameters: {base_params:,}")
    
    # Convert to NPT
    print("\n" + "-" * 60)
    print("Converting to NPT")
    print("-" * 60)
    
    npt_config = NPTConfig(
        convert_all=True,
        np_rank=32,
        np_init_scale=0.01
    )
    
    model.convert_to_npt(npt_config)
    model.freeze_base_parameters()
    
    # Show parameter counts
    param_counts = model.count_parameters()
    print(f"\nAfter NPT conversion:")
    print(f"  Total parameters: {param_counts['total']:,}")
    print(f"  Base (frozen): {param_counts['base']:,}")
    print(f"  NPT (trainable): {param_counts['npt']:,}")
    print(f"  NPT ratio: {param_counts['npt_ratio']:.2%}")
    
    # Check gradients
    print("\nVerifying gradient flow:")
    dummy_input = torch.randint(0, config.vocab_size, (2, 10))
    outputs = model(dummy_input, labels=dummy_input)
    loss = outputs.loss
    loss.backward()
    
    npt_grads = sum(1 for p in model.get_npt_parameters() if p.grad is not None)
    base_grads = 0
    for name, param in model.named_parameters():
        if 'np_component' not in name and param.grad is not None:
            base_grads += 1
    
    print(f"  NP parameters with gradients: {npt_grads}")
    print(f"  Base parameters with gradients: {base_grads} (should be 0)")
    
    return model


def demo_training_loop():
    """Demonstrate the training loop."""
    
    print("\n" + "=" * 80)
    print("Training Loop Demo")
    print("=" * 80)
    
    # Setup model
    config = LlamaConfig(
        hidden_size=128,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=1000,
    )
    config._attn_implementation = "eager"
    
    model = NPTLlamaModel(config)
    npt_config = NPTConfig(convert_all=True, np_rank=16, np_init_scale=0.01)
    model.convert_to_npt(npt_config)
    model.freeze_base_parameters()
    
    # Setup data
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    train_texts = ["Training example."] * 20
    val_texts = ["Validation example."] * 10
    
    train_loader, val_loader = create_data_loaders(
        train_data=train_texts,
        val_data=val_texts,
        tokenizer=tokenizer,
        batch_size=4,
        max_length=32
    )
    
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
            save_steps=15,
            output_dir=tmpdir,
            device="cpu"  # Use CPU for demo
        )
        
        print(f"\nTraining Configuration:")
        print(f"  Learning rate: {training_config.learning_rate}")
        print(f"  Batch size: {training_config.batch_size}")
        print(f"  Max steps: {training_config.max_steps}")
        print(f"  Warmup steps: {training_config.warmup_steps}")
        print(f"  Lambda regularization: {training_config.lambda_reg}")
        
        # Create trainer
        trainer = NPTTrainer(
            model=model,
            config=training_config,
            train_loader=train_loader,
            val_loader=val_loader
        )
        
        # Initial evaluation
        print("\n" + "-" * 60)
        print("Initial Evaluation")
        print("-" * 60)
        initial_metrics = trainer.evaluate()
        print(f"Initial validation loss: {initial_metrics['val_loss']:.4f}")
        print(f"  Fidelity loss: {initial_metrics['val_fidelity_loss']:.4f}")
        print(f"  Regularization loss: {initial_metrics['val_regularization_loss']:.6f}")
        
        # Training
        print("\n" + "-" * 60)
        print("Training Progress")
        print("-" * 60)
        
        trainer.train()
        
        # Final evaluation
        print("\n" + "-" * 60)
        print("Final Evaluation")
        print("-" * 60)
        final_metrics = trainer.evaluate()
        print(f"Final validation loss: {final_metrics['val_loss']:.4f}")
        print(f"  Fidelity loss: {final_metrics['val_fidelity_loss']:.4f}")
        print(f"  Regularization loss: {final_metrics['val_regularization_loss']:.6f}")
        
        # Show improvement
        improvement = (initial_metrics['val_loss'] - final_metrics['val_loss']) / initial_metrics['val_loss'] * 100
        print(f"\nLoss improvement: {improvement:.1f}%")
        
        # Check saved files
        output_path = Path(tmpdir)
        print(f"\nOutput files created:")
        for file in output_path.rglob("*"):
            if file.is_file():
                print(f"  {file.relative_to(output_path)}")


def demo_checkpointing():
    """Demonstrate checkpoint saving and loading."""
    
    print("\n" + "=" * 80)
    print("Checkpointing Demo")
    print("=" * 80)
    
    # Setup small model
    config = LlamaConfig(
        hidden_size=128,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=1000,
    )
    config._attn_implementation = "eager"
    
    model = NPTLlamaModel(config)
    npt_config = NPTConfig(convert_all=True, np_rank=16)
    model.convert_to_npt(npt_config)
    model.freeze_base_parameters()
    
    # Setup data
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    train_loader, _ = create_data_loaders(
        train_data=["Sample text."] * 10,
        val_data=None,
        tokenizer=tokenizer,
        batch_size=2,
        max_length=32
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create trainer
        training_config = TrainingConfig(
            max_steps=10,
            output_dir=tmpdir,
            device="cpu"
        )
        
        trainer = NPTTrainer(
            model=model,
            config=training_config,
            train_loader=train_loader
        )
        
        # Train for a few steps
        print("\nTraining for 5 steps...")
        for i, batch in enumerate(train_loader):
            if i >= 5:
                break
            metrics = trainer.train_step(batch)
            trainer.global_step += 1
            print(f"  Step {trainer.global_step}: Loss = {metrics.total_loss:.4f}")
        
        # Save checkpoint
        print("\nSaving checkpoint...")
        trainer.save_checkpoint("demo_checkpoint")
        
        checkpoint_path = Path(tmpdir) / "checkpoints" / "demo_checkpoint"
        print(f"Checkpoint saved to: {checkpoint_path}")
        
        # Show checkpoint contents
        print("\nCheckpoint contents:")
        for file in checkpoint_path.iterdir():
            size = file.stat().st_size / 1024  # KB
            print(f"  {file.name}: {size:.1f} KB")
        
        # Create new trainer and load checkpoint
        print("\nCreating new trainer and loading checkpoint...")
        new_trainer = NPTTrainer(
            model=model,
            config=training_config,
            train_loader=train_loader
        )
        
        print(f"New trainer step before loading: {new_trainer.global_step}")
        new_trainer.load_checkpoint(str(checkpoint_path))
        print(f"New trainer step after loading: {new_trainer.global_step}")


def demo_metrics_tracking():
    """Demonstrate metrics tracking during training."""
    
    print("\n" + "=" * 80)
    print("Metrics Tracking Demo")
    print("=" * 80)
    
    # Setup minimal model and data
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
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    train_loader, _ = create_data_loaders(
        train_data=["Text."] * 10,
        val_data=None,
        tokenizer=tokenizer,
        batch_size=2,
        max_length=16
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
        
        print("\nTracking metrics during training:")
        print("-" * 60)
        print("Step | Total Loss | Fidelity | Regularization | LR")
        print("-" * 60)
        
        for i, batch in enumerate(train_loader):
            if i >= 10:
                break
            
            metrics = trainer.train_step(batch)
            trainer.global_step += 1
            
            print(f"{metrics.step:4d} | {metrics.total_loss:10.4f} | "
                  f"{metrics.fidelity_loss:8.4f} | {metrics.regularization_loss:14.6f} | "
                  f"{metrics.learning_rate:.2e}")
        
        print("-" * 60)
        
        # Load and show logged metrics
        log_file = Path(tmpdir) / "training_log.jsonl"
        if log_file.exists():
            print(f"\nMetrics logged to: {log_file}")
            print(f"Log file size: {log_file.stat().st_size} bytes")


def main():
    """Run all demonstrations."""
    
    print("=" * 80)
    print("NPT Training Pipeline (Stage 5) Demonstration")
    print("=" * 80)
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # Run demos
    demo_data_loading()
    model = demo_model_setup()
    demo_training_loop()
    demo_checkpointing()
    demo_metrics_tracking()
    
    print("\n" + "=" * 80)
    print("Stage 5 Implementation Complete!")
    print("✓ Data loading with chunking and padding")
    print("✓ Training loop with optimizer and scheduler")
    print("✓ Validation and metrics tracking")
    print("✓ Checkpoint saving and loading")
    print("✓ Complete training pipeline")
    print("=" * 80)


if __name__ == "__main__":
    main()