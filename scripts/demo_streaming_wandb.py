"""
Demonstration of NPT training with HuggingFace streaming and WandB integration.

This script shows how the complete pipeline works with:
- Streaming data from HuggingFace
- WandB experiment tracking
- NPT model training
"""

import sys
from pathlib import Path
import torch
import logging

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.npt import NPTLlamaModel, NPTConfig
from src.training import (
    NPTTrainer,
    TrainingConfig,
    StreamingConfig,
    MultiDatasetStreamer,
    WandBTracker
)
from transformers import AutoTokenizer, LlamaConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_streaming_data():
    """Demonstrate streaming data loading from HuggingFace."""
    
    print("=" * 80)
    print("Streaming Data Demo")
    print("=" * 80)
    
    # Load tokenizer
    print("\nLoading Llama tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create multi-dataset streamer
    print("\nSetting up multi-dataset streaming...")
    streamer = MultiDatasetStreamer(
        preset='small',  # Using WikiText-2 for demo
        tokenizer=tokenizer,
        max_length=256,
        batch_size=4,
        num_workers=0  # Set to 0 for demo to avoid multiprocessing issues
    )
    
    print(f"Dataset preset: small")
    print(f"Datasets: {streamer.dataset_names}")
    
    # Create data loaders
    print("\nCreating streaming data loaders...")
    train_loader, val_loader = streamer.create_data_loaders(validation=True)
    
    # Show a sample batch
    print("\nFetching a sample batch from streaming data...")
    try:
        batch = next(iter(train_loader))
        print(f"Batch keys: {batch.keys()}")
        print(f"Input IDs shape: {batch['input_ids'].shape}")
        print(f"Attention mask shape: {batch['attention_mask'].shape}")
        
        # Decode first sequence
        first_seq = batch['input_ids'][0]
        decoded = tokenizer.decode(first_seq[:50], skip_special_tokens=True)
        print(f"\nFirst 50 tokens decoded: {decoded[:100]}...")
        
    except Exception as e:
        print(f"Note: Could not fetch batch due to: {e}")
        print("This is normal if HuggingFace datasets are not accessible.")


def demo_wandb_tracking():
    """Demonstrate WandB tracking setup."""
    
    print("\n" + "=" * 80)
    print("WandB Tracking Demo")
    print("=" * 80)
    
    # Create a small model for demo
    config = LlamaConfig(
        hidden_size=256,
        intermediate_size=1024,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        vocab_size=32000,
    )
    config._attn_implementation = "eager"
    
    model = NPTLlamaModel(config)
    npt_config = NPTConfig(convert_all=True, np_rank=16)
    model.convert_to_npt(npt_config)
    
    # Setup WandB tracker
    print("\nSetting up WandB tracker...")
    tracker = WandBTracker(
        project="npt-demo",
        name="demo_run",
        config={
            "model": "demo",
            "np_rank": 16,
            "learning_rate": 1e-4
        },
        tags=["demo", "test"],
        mode="offline"  # Use offline mode for demo
    )
    
    print(f"WandB mode: offline (artifacts saved locally)")
    print(f"Project: npt-demo")
    print(f"Run name: demo_run")
    
    # Initialize tracker
    tracker.init(model=model)
    
    if tracker.run:
        print(f"✓ WandB run initialized")
        print(f"  Run ID: {tracker.run.id}")
        print(f"  Run directory: {tracker.run.dir}")
        
        # Log some sample metrics
        print("\nLogging sample metrics...")
        for step in range(5):
            metrics = {
                "loss": 1.0 - step * 0.1,
                "learning_rate": 1e-4 * (1 - step/10),
                "grad_norm": 0.5 + step * 0.05
            }
            tracker.log_metrics(metrics, step=step)
        
        print("✓ Metrics logged")
        
        # Log NPT dynamics
        v_a_norms = [0.1, 0.15, 0.12, 0.18]
        v_b_norms = [0.2, 0.25, 0.22, 0.28]
        tracker.log_npt_dynamics(v_a_norms, v_b_norms, step=5)
        print("✓ NPT dynamics logged")
        
        # Finish run
        tracker.finish()
        print("✓ WandB run finished")
    else:
        print("Note: WandB run not initialized (may need wandb login)")


def demo_integrated_training():
    """Demonstrate integrated training with streaming and WandB."""
    
    print("\n" + "=" * 80)
    print("Integrated Training Demo")
    print("=" * 80)
    
    # Create small model
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
    npt_config = NPTConfig(convert_all=True, np_rank=8)
    model.convert_to_npt(npt_config)
    model.freeze_base_parameters()
    
    print(f"\nModel setup:")
    param_counts = model.count_parameters()
    print(f"  Total parameters: {param_counts['total']:,}")
    print(f"  NPT parameters: {param_counts['npt']:,}")
    
    # Create synthetic data loader (to avoid HF dependencies for demo)
    from torch.utils.data import Dataset, DataLoader
    
    class DemoDataset(Dataset):
        def __init__(self, size=20):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            return {
                'input_ids': torch.randint(0, 1000, (128,)),
                'attention_mask': torch.ones(128),
                'labels': torch.randint(0, 1000, (128,))
            }
    
    train_loader = DataLoader(DemoDataset(), batch_size=2)
    val_loader = DataLoader(DemoDataset(size=5), batch_size=2)
    
    # Setup WandB
    tracker = WandBTracker(
        project="npt-demo",
        name="integrated_demo",
        config={
            "model_size": "tiny",
            "dataset": "synthetic",
            "np_rank": 8
        },
        mode="disabled"  # Disable for demo to avoid login requirement
    )
    
    # Training config
    training_config = TrainingConfig(
        batch_size=2,
        learning_rate=1e-3,
        max_steps=10,
        logging_steps=2,
        eval_steps=5,
        output_dir="/tmp/npt_demo",
        device="cpu"
    )
    
    # Create trainer
    print("\nCreating trainer with WandB integration...")
    trainer = NPTTrainer(
        model=model,
        config=training_config,
        train_loader=train_loader,
        val_loader=val_loader,
        wandb_run=tracker.run
    )
    
    print("\nRunning short training loop...")
    print("-" * 60)
    
    # Run a few training steps
    for i, batch in enumerate(train_loader):
        if i >= 5:
            break
        
        metrics = trainer.train_step(batch)
        trainer.global_step += 1
        
        if i % 2 == 0:
            print(f"Step {trainer.global_step}: Loss = {metrics.total_loss:.4f}, "
                  f"LR = {metrics.learning_rate:.2e}")
    
    print("-" * 60)
    print("✓ Training steps completed")
    
    # Final evaluation
    print("\nRunning evaluation...")
    eval_metrics = trainer.evaluate()
    print(f"Validation loss: {eval_metrics['val_loss']:.4f}")
    
    print("\n✓ Integrated training demo complete!")


def main():
    """Run all demonstrations."""
    
    print("=" * 80)
    print("NPT Training with Streaming Data and WandB Integration")
    print("=" * 80)
    print("\nThis demo shows the key components of the advanced training pipeline:\n")
    print("1. Streaming data from HuggingFace datasets")
    print("2. WandB experiment tracking")
    print("3. Integrated training with both features")
    print()
    
    # Set seed
    torch.manual_seed(42)
    
    # Run demos
    try:
        demo_streaming_data()
    except Exception as e:
        print(f"\nStreaming demo error: {e}")
        print("Note: This is expected if HuggingFace datasets are not accessible.")
    
    demo_wandb_tracking()
    demo_integrated_training()
    
    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)
    print("\nTo run full training with streaming and WandB:")
    print("\n  python scripts/train_npt_streaming.py \\")
    print("    --dataset_preset medium \\")
    print("    --wandb_project your-project \\")
    print("    --max_steps 10000")
    print("\nFor demo mode:")
    print("\n  python scripts/train_npt_streaming.py --demo_mode")


if __name__ == "__main__":
    main()