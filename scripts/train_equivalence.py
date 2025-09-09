#!/usr/bin/env python3
"""
Training script for NPT equivalence pre-training.

This script sets up and runs the complete training pipeline for teaching
NP components to mimic original transformer residual connections.
"""

import argparse
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.npt import NPTLlamaModel, NPTConfig
from src.training import (
    NPTTrainer,
    TrainingConfig,
    create_data_loaders
)
from transformers import LlamaConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train NPT model for equivalence pre-training"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Base model to load"
    )
    parser.add_argument(
        "--convert_layers",
        type=str,
        default="upper_half",
        help="Which layers to convert: 'all', 'upper_half', or comma-separated indices"
    )
    parser.add_argument(
        "--np_rank",
        type=int,
        default=64,
        help="Rank for NP component"
    )
    parser.add_argument(
        "--np_init_scale",
        type=float,
        default=0.01,
        help="Initialization scale for NP components"
    )
    
    # Data arguments
    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="Path to training data"
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default=None,
        help="Path to validation data"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=256,
        help="Stride for sliding window"
    )
    
    # Training arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay"
    )
    parser.add_argument(
        "--lambda_reg",
        type=float,
        default=0.01,
        help="Regularization weight"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=10000,
        help="Maximum training steps"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Warmup steps"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--gradient_clip",
        type=float,
        default=1.0,
        help="Gradient clipping value"
    )
    
    # Logging arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/npt_training",
        help="Output directory"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log metrics every N steps"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=100,
        help="Evaluate every N steps"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--generation_steps",
        type=int,
        default=150,
        help="Generate samples every N steps"
    )
    
    # Other arguments
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Resume from checkpoint"
    )
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        help="Use mixed precision training"
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Use Weights & Biases for logging"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="npt-training",
        help="WandB project name"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--demo_mode",
        action="store_true",
        help="Run in demo mode with small model and data"
    )
    
    return parser.parse_args()


def setup_model(args):
    """Setup NPT model based on arguments."""
    
    if args.demo_mode:
        # Create small model for demo
        print("Running in demo mode with small model...")
        config = LlamaConfig(
            hidden_size=256,
            intermediate_size=1024,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4,
            vocab_size=1000,
        )
        config._attn_implementation = "eager"
        model = NPTLlamaModel(config)
        tokenizer = None  # Will create dummy tokenizer
    else:
        # Load real model
        print(f"Loading model: {args.model_name}")
        from transformers import AutoConfig
        
        config = AutoConfig.from_pretrained(args.model_name)
        config._attn_implementation = "eager"
        
        # For demo/testing, create smaller model
        if "Llama-3.2-1B" in args.model_name:
            # Use smaller config for testing
            config = LlamaConfig(
                hidden_size=512,
                intermediate_size=2048,
                num_hidden_layers=8,
                num_attention_heads=8,
                num_key_value_heads=4,
                vocab_size=32000,
            )
            config._attn_implementation = "eager"
        
        model = NPTLlamaModel(config)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Configure NPT conversion
    if args.convert_layers == "all":
        npt_config = NPTConfig(
            convert_all=True,
            np_rank=args.np_rank,
            np_init_scale=args.np_init_scale
        )
    elif args.convert_layers == "upper_half":
        npt_config = NPTConfig(
            np_rank=args.np_rank,
            np_init_scale=args.np_init_scale
        )
    else:
        # Parse layer indices
        layers = [int(x.strip()) for x in args.convert_layers.split(',')]
        npt_config = NPTConfig(
            layers_to_convert=layers,
            np_rank=args.np_rank,
            np_init_scale=args.np_init_scale
        )
    
    # Convert to NPT
    print(f"Converting layers to NPT with config: {npt_config}")
    model.convert_to_npt(npt_config)
    
    # Freeze base parameters
    print("Freezing base model parameters...")
    model.freeze_base_parameters()
    
    # Print parameter counts
    param_counts = model.count_parameters()
    print(f"Model parameter summary:")
    print(f"  Total: {param_counts['total']:,}")
    print(f"  Base (frozen): {param_counts['base']:,}")
    print(f"  NPT (trainable): {param_counts['npt']:,}")
    print(f"  NPT ratio: {param_counts['npt_ratio']:.2%}")
    
    return model, tokenizer


def setup_data(args, tokenizer):
    """Setup data loaders."""
    
    if args.demo_mode:
        # Create dummy data for demo
        print("Creating demo data...")
        from transformers import AutoTokenizer
        
        # Use GPT2 tokenizer for demo
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
        
        train_data = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Neural networks are inspired by biological neurons.",
            "Transformers have revolutionized natural language processing.",
            "Attention is all you need for sequence modeling.",
        ] * 20  # Repeat to create more samples
        
        val_data = [
            "Deep learning models require large amounts of data.",
            "Gradient descent optimizes neural network parameters.",
        ] * 10
    else:
        train_data = args.train_data
        val_data = args.val_data
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        train_data=train_data,
        val_data=val_data,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        stride=args.stride,
        num_workers=0,
        include_labels=True
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Training batches: {len(train_loader)}")
    if val_loader:
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Validation batches: {len(val_loader)}")
    
    return train_loader, val_loader, tokenizer


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Setup model
    model, tokenizer = setup_model(args)
    
    # Setup data
    train_loader, val_loader, tokenizer = setup_data(args, tokenizer)
    
    # Setup wandb if requested
    wandb_run = None
    if args.use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                config=vars(args),
                name=f"npt_training_{args.np_rank}r_{args.lambda_reg}l"
            )
        except ImportError:
            print("Warning: wandb not installed, skipping wandb logging")
    
    # Create training config
    if args.convert_layers == "upper_half":
        npt_layers = None
    elif args.convert_layers == "all":
        npt_layers = list(range(model.config.num_hidden_layers))
    else:
        npt_layers = [int(x.strip()) for x in args.convert_layers.split(',')]
    
    training_config = TrainingConfig(
        model_name=args.model_name if not args.demo_mode else "demo",
        npt_layers=npt_layers,
        np_rank=args.np_rank,
        np_init_scale=args.np_init_scale,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lambda_reg=args.lambda_reg,
        max_steps=args.max_steps if not args.demo_mode else 100,
        warmup_steps=args.warmup_steps if not args.demo_mode else 10,
        max_length=args.max_length,
        stride=args.stride,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        generation_steps=args.generation_steps,
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume_from,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_clip_value=args.gradient_clip
    )
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = NPTTrainer(
        model=model,
        config=training_config,
        train_loader=train_loader,
        val_loader=val_loader,
        wandb_run=wandb_run
    )
    
    # Start training
    print("\n" + "=" * 80)
    print("Starting NPT Equivalence Pre-training")
    print("=" * 80)
    
    trainer.train()
    
    # Cleanup
    if wandb_run:
        wandb_run.finish()
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()