#!/usr/bin/env python3
"""
Advanced NPT training script with HuggingFace streaming and WandB tracking.

This script provides production-ready training with:
- Streaming data from HuggingFace datasets
- Comprehensive WandB experiment tracking
- Multi-dataset mixing
- Automatic checkpointing
- Sample generation logging
"""

import argparse
import sys
from pathlib import Path
import torch
import logging
from datetime import datetime
import json
from typing import Optional

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.npt import NPTLlamaModel, NPTConfig
from src.training import NPTTrainer, TrainingConfig
from src.training.streaming_data import (
    create_streaming_loaders,
    MultiDatasetStreamer,
    StreamingConfig
)
from src.training.wandb_integration import WandBTracker
from transformers import AutoTokenizer, LlamaConfig, AutoConfig
import wandb

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train NPT model with streaming data and WandB tracking"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Base model name or path"
    )
    parser.add_argument(
        "--model_size",
        type=str,
        choices=["1b", "3b", "8b", "demo"],
        default="1b",
        help="Model size preset"
    )
    parser.add_argument(
        "--convert_layers",
        type=str,
        default="upper_half",
        help="Which layers to convert: 'all', 'upper_half', 'lower_half', or comma-separated indices"
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
    
    # Dataset arguments
    parser.add_argument(
        "--dataset_preset",
        type=str,
        choices=["small", "medium", "large", "xlarge", "custom"],
        default="small",
        help="Dataset preset for streaming"
    )
    parser.add_argument(
        "--dataset_names",
        type=str,
        nargs="+",
        default=None,
        help="Custom dataset names (for custom preset)"
    )
    parser.add_argument(
        "--dataset_configs",
        type=str,
        nargs="+",
        default=None,
        help="Custom dataset configurations"
    )
    parser.add_argument(
        "--mix_probabilities",
        type=float,
        nargs="+",
        default=None,
        help="Dataset mixing probabilities"
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
        help="Training batch size per device"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps"
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
        help="Regularization weight for v_a and v_b"
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
        "--gradient_clip",
        type=float,
        default=1.0,
        help="Gradient clipping value"
    )
    
    # Logging arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (auto-generated if not specified)"
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
        default=250,
        help="Generate samples every N steps"
    )
    
    # WandB arguments
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="npt-training",
        help="WandB project name"
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="WandB run name (auto-generated if not specified)"
    )
    parser.add_argument(
        "--wandb_tags",
        type=str,
        nargs="+",
        default=None,
        help="WandB tags"
    )
    parser.add_argument(
        "--wandb_notes",
        type=str,
        default=None,
        help="WandB run notes"
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        choices=["online", "offline", "disabled"],
        default="online",
        help="WandB mode"
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
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers"
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
        help="Run in demo mode with small model and limited steps"
    )
    
    return parser.parse_args()


def get_model_config(args):
    """Get model configuration based on size preset."""
    if args.demo_mode:
        return LlamaConfig(
            hidden_size=256,
            intermediate_size=1024,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4,
            vocab_size=128256,  # Match Llama tokenizer
        )
    
    configs = {
        "1b": LlamaConfig(
            hidden_size=2048,
            intermediate_size=8192,
            num_hidden_layers=16,
            num_attention_heads=32,
            num_key_value_heads=8,
            vocab_size=128256,
        ),
        "3b": LlamaConfig(
            hidden_size=3072,
            intermediate_size=8192,
            num_hidden_layers=28,
            num_attention_heads=24,
            num_key_value_heads=8,
            vocab_size=128256,
        ),
        "8b": LlamaConfig(
            hidden_size=4096,
            intermediate_size=14336,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            vocab_size=128256,
        ),
        "demo": LlamaConfig(
            hidden_size=512,
            intermediate_size=2048,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4,
            vocab_size=128256,  # Match Llama tokenizer
        )
    }
    
    config = configs.get(args.model_size)
    if config:
        config._attn_implementation = "eager"
    return config


def setup_model(args):
    """Setup NPT model."""
    logger.info(f"Setting up model: {args.model_name} ({args.model_size})")
    
    # Get configuration
    config = get_model_config(args)
    if config is None:
        # Try loading from model name
        config = AutoConfig.from_pretrained(args.model_name)
        config._attn_implementation = "eager"
    
    # Create model - load pretrained weights if available
    if args.model_name and "demo" not in args.model_size:
        try:
            model = NPTLlamaModel.from_pretrained(args.model_name)
            logger.info(f"Loaded pretrained model from {args.model_name}")
        except:
            logger.warning(f"Could not load pretrained model {args.model_name}, using random init")
            model = NPTLlamaModel(config)
    else:
        model = NPTLlamaModel(config)
    
    # Setup NPT conversion
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
    elif args.convert_layers == "lower_half":
        num_layers = config.num_hidden_layers
        npt_config = NPTConfig(
            convert_range=(0, num_layers // 2),
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
    logger.info(f"Converting layers to NPT: {args.convert_layers}")
    model.convert_to_npt(npt_config)
    model.freeze_base_parameters()
    
    # Log parameter counts
    param_counts = model.count_parameters()
    logger.info(f"Model parameters:")
    logger.info(f"  Total: {param_counts['total']:,}")
    logger.info(f"  Base (frozen): {param_counts['base']:,}")
    logger.info(f"  NPT (trainable): {param_counts['npt']:,}")
    logger.info(f"  NPT ratio: {param_counts['npt_ratio']:.2%}")
    
    return model, config


def setup_data(args, tokenizer):
    """Setup streaming data loaders."""
    logger.info("Setting up streaming data loaders...")
    
    if args.dataset_preset == "custom":
        # Use custom datasets
        if not args.dataset_names:
            raise ValueError("--dataset_names required for custom preset")
        
        train_loader, val_loader = create_streaming_loaders(
            tokenizer=tokenizer,
            dataset_name=args.dataset_names,
            dataset_config=args.dataset_configs if args.dataset_configs else [None] * len(args.dataset_names),
            batch_size=args.batch_size,
            max_length=args.max_length,
            stride=args.stride,
            num_workers=args.num_workers,
            streaming=True,
            validation=True
        )
    else:
        # Use preset
        streamer = MultiDatasetStreamer(
            preset=args.dataset_preset,
            tokenizer=tokenizer,
            max_length=args.max_length,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        train_loader, val_loader = streamer.create_data_loaders(
            validation=True,
            mix_probabilities=args.mix_probabilities
        )
    
    logger.info(f"Data loaders created:")
    logger.info(f"  Dataset preset: {args.dataset_preset}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Max length: {args.max_length}")
    logger.info(f"  Streaming: enabled")
    
    return train_loader, val_loader


def setup_wandb(args, model_config):
    """Setup WandB tracking."""
    # Generate run name if not specified
    if args.wandb_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.wandb_name = f"npt_{args.model_size}_{args.dataset_preset}_{timestamp}"
    
    # Generate output directory if not specified
    if args.output_dir is None:
        args.output_dir = f"experiments/{args.wandb_name}"
    
    # Prepare configuration for logging
    config = {
        "model": {
            "name": args.model_name,
            "size": args.model_size,
            "hidden_size": model_config.hidden_size,
            "num_layers": model_config.num_hidden_layers,
            "vocab_size": model_config.vocab_size,
        },
        "npt": {
            "convert_layers": args.convert_layers,
            "np_rank": args.np_rank,
            "np_init_scale": args.np_init_scale,
        },
        "data": {
            "dataset_preset": args.dataset_preset,
            "dataset_names": args.dataset_names,
            "max_length": args.max_length,
            "stride": args.stride,
        },
        "training": {
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "lambda_reg": args.lambda_reg,
            "max_steps": args.max_steps,
            "warmup_steps": args.warmup_steps,
            "gradient_clip": args.gradient_clip,
            "mixed_precision": args.mixed_precision,
        },
        "logging": {
            "output_dir": args.output_dir,
            "logging_steps": args.logging_steps,
            "eval_steps": args.eval_steps,
            "save_steps": args.save_steps,
            "generation_steps": args.generation_steps,
        }
    }
    
    # Create tracker
    tracker = WandBTracker(
        project=args.wandb_project,
        name=args.wandb_name,
        config=config,
        tags=args.wandb_tags,
        notes=args.wandb_notes,
        mode=args.wandb_mode
    )
    
    return tracker


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Demo mode adjustments
    if args.demo_mode:
        logger.info("Running in DEMO MODE - reduced model and steps")
        args.max_steps = 50
        args.eval_steps = 10
        args.save_steps = 25
        args.generation_steps = 20
        args.dataset_preset = "small"
        args.batch_size = 2
        args.num_workers = 0  # Avoid multiprocessing issues in demo
    
    # Setup model
    model, model_config = setup_model(args)
    
    # Setup tokenizer
    logger.info("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    except:
        # Fallback tokenizer for demo
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Setup data loaders
    train_loader, val_loader = setup_data(args, tokenizer)
    
    # Setup WandB
    tracker = setup_wandb(args, model_config)
    tracker.init(model=model)
    
    # Create training config
    # Use CPU in demo mode to avoid CUDA issues
    device = "cpu" if args.demo_mode else ("cuda" if torch.cuda.is_available() else "cpu")
    
    training_config = TrainingConfig(
        model_name=args.model_name,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lambda_reg=args.lambda_reg,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
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
        gradient_clip_value=args.gradient_clip,
        device=device
    )
    
    # Create custom trainer with WandB integration
    class NPTTrainerWithWandB(NPTTrainer):
        """Extended trainer with WandB integration."""
        
        def __init__(self, *args, tracker=None, tokenizer=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.tracker = tracker
            self.tokenizer = tokenizer
            self.generation_prompts = [
                "The future of artificial intelligence",
                "Once upon a time in a distant galaxy",
                "The key to understanding neural networks",
                "In the beginning, there was",
                "The most important discovery in science",
            ]
        
        def train_step(self, batch):
            """Override to add WandB logging."""
            metrics = super().train_step(batch)
            
            # Log to WandB
            if self.tracker and self.global_step % self.config.logging_steps == 0:
                self.tracker.log_metrics(vars(metrics), step=self.global_step)
                
                # Log gradients periodically
                if self.global_step % (self.config.logging_steps * 10) == 0:
                    self.tracker.log_gradients(self.model, step=self.global_step)
                    self.tracker.log_weights(self.model, step=self.global_step)
            
            # Generate samples periodically
            if (self.tracker and self.tokenizer and 
                self.global_step % self.config.generation_steps == 0 and
                self.global_step > 0):
                self.tracker.log_sample_outputs(
                    self.model,
                    self.tokenizer,
                    self.generation_prompts[:3],  # Use first 3 prompts
                    step=self.global_step,
                    max_length=50
                )
            
            return metrics
        
        def evaluate(self, eval_loader=None):
            """Override to add WandB logging."""
            metrics = super().evaluate(eval_loader)
            
            # Log to WandB
            if self.tracker:
                self.tracker.log_metrics(metrics, step=self.global_step)
            
            return metrics
        
        def save_checkpoint(self, checkpoint_name=None):
            """Override to save as WandB artifact."""
            super().save_checkpoint(checkpoint_name)
            
            # Save as WandB artifact
            if self.tracker:
                checkpoint_path = self.checkpoint_dir / (checkpoint_name or f"checkpoint-{self.global_step}")
                aliases = ["latest"]
                if checkpoint_name == "best":
                    aliases.append("best")
                self.tracker.save_checkpoint(checkpoint_path, aliases=aliases)
    
    # Create trainer with WandB
    logger.info("Initializing trainer...")
    trainer = NPTTrainerWithWandB(
        model=model,
        config=training_config,
        train_loader=train_loader,
        val_loader=val_loader,
        tracker=tracker,
        tokenizer=tokenizer
    )
    
    # Start training
    logger.info("\n" + "=" * 80)
    logger.info("Starting NPT Training with Streaming Data and WandB")
    logger.info("=" * 80)
    logger.info(f"WandB Run: {tracker.run.name if tracker.run else 'disabled'}")
    logger.info(f"Output Directory: {training_config.output_dir}")
    logger.info("=" * 80 + "\n")
    
    try:
        # Run training
        trainer.train()
        
        # Log final summary
        if tracker:
            final_metrics = trainer.evaluate()
            param_counts = model.count_parameters()
            
            summary = {
                "final_loss": final_metrics.get('val_loss', 0),
                "total_steps": trainer.global_step,
                "total_parameters": param_counts['total'],
                "npt_parameters": param_counts['npt'],
                "best_val_loss": trainer.best_val_loss,
            }
            
            tracker.log_training_summary(summary)
    
    finally:
        # Cleanup
        if tracker:
            tracker.finish()
    
    logger.info("\n" + "=" * 80)
    logger.info("Training Complete!")
    logger.info(f"Final checkpoint saved to: {training_config.output_dir}/checkpoints/final")
    if tracker and tracker.run:
        logger.info(f"View results at: {tracker.run.get_url()}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()