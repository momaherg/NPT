#!/usr/bin/env python3
"""
Sequential layer-by-layer training for NPT.

This script trains NPT layers one at a time, starting from lower layers
and progressing upward. It reuses the existing single-layer training
infrastructure.
"""

import argparse
import subprocess
import sys
import os
import json
import torch
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train NPT layers sequentially"
    )
    
    # Layer configuration
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Layers to train: 'all', 'upper_half', 'lower_half', or comma-separated list (e.g., '0,1,2,3')"
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=16,
        help="Total number of layers in the model (16 for 1B, 28 for 3B, 32 for 8B)"
    )
    parser.add_argument(
        "--start_from_layer",
        type=int,
        default=0,
        help="Resume training from this layer index"
    )
    
    # Training configuration (passed to single-layer script)
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
        "--steps_per_layer",
        type=int,
        default=2000,
        help="Training steps per layer"
    )
    parser.add_argument(
        "--stage1_steps",
        type=int,
        default=500,
        help="Stage 1 steps for each layer"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Training batch size"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=32,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--np_rank",
        type=int,
        default=256,
        help="NPT rank for single layer"
    )
    
    # Dataset configuration
    parser.add_argument(
        "--dataset_preset",
        type=str,
        default="small",
        help="Dataset preset for streaming"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    
    # Output configuration
    parser.add_argument(
        "--base_output_dir",
        type=str,
        default=None,
        help="Base output directory for all layers"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory to load/save accumulated NPT weights"
    )
    
    # WandB configuration
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="npt-sequential",
        help="WandB project name"
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="WandB mode"
    )
    
    # Other options
    parser.add_argument(
        "--demo_mode",
        action="store_true",
        help="Run in demo mode with minimal steps"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing"
    )
    
    return parser.parse_args()


def get_layers_to_train(args):
    """Get list of layer indices to train."""
    if args.layers == "all":
        return list(range(args.num_layers))
    elif args.layers == "upper_half":
        return list(range(args.num_layers // 2, args.num_layers))
    elif args.layers == "lower_half":
        return list(range(args.num_layers // 2))
    else:
        # Parse comma-separated list
        try:
            layers = [int(x.strip()) for x in args.layers.split(",")]
            # Validate layer indices
            for idx in layers:
                if idx < 0 or idx >= args.num_layers:
                    raise ValueError(f"Layer index {idx} out of range [0, {args.num_layers})")
            return layers
        except:
            raise ValueError(f"Invalid layers specification: {args.layers}")


def load_checkpoint_info(checkpoint_dir):
    """Load information about previously trained layers."""
    info_file = Path(checkpoint_dir) / "training_info.json"
    if info_file.exists():
        with open(info_file, 'r') as f:
            return json.load(f)
    return {"trained_layers": [], "layer_checkpoints": {}}


def save_checkpoint_info(checkpoint_dir, info):
    """Save information about trained layers."""
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    info_file = Path(checkpoint_dir) / "training_info.json"
    with open(info_file, 'w') as f:
        json.dump(info, f, indent=2)


def merge_npt_weights(checkpoint_dir, layer_idx, layer_checkpoint):
    """
    Merge NPT weights from a single layer into the accumulated checkpoint.
    
    Args:
        checkpoint_dir: Directory containing accumulated weights
        layer_idx: Index of the layer just trained
        layer_checkpoint: Path to the checkpoint for this layer
    """
    accumulated_path = Path(checkpoint_dir) / "accumulated_npt_weights.pt"
    
    # Load existing accumulated weights if they exist
    if accumulated_path.exists():
        accumulated = torch.load(accumulated_path, map_location='cpu')
    else:
        accumulated = {}
    
    # Load the new layer's weights
    layer_weights_path = Path(layer_checkpoint) / "checkpoints" / "final" / "npt_weights.pt"
    if not layer_weights_path.exists():
        logger.warning(f"NPT weights not found at {layer_weights_path}")
        return
    
    layer_weights = torch.load(layer_weights_path, map_location='cpu')
    
    # The saved weights are in format: layer_{idx}_np.{param_name}
    # We keep them in this format for compatibility
    for key, value in layer_weights.items():
        # Only add weights for the current layer
        if key.startswith(f"layer_{layer_idx}_np"):
            accumulated[key] = value
    
    # Save accumulated weights
    torch.save(accumulated, accumulated_path)
    logger.info(f"Merged layer {layer_idx} weights into {accumulated_path}")


def train_single_layer(layer_idx, args, checkpoint_info):
    """
    Train a single NPT layer using the existing script.
    
    Args:
        layer_idx: Index of the layer to train
        args: Command line arguments
        checkpoint_info: Information about previously trained layers
    
    Returns:
        True if training succeeded, False otherwise
    """
    # Prepare output directory for this layer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.base_output_dir:
        output_dir = f"{args.base_output_dir}/layer_{layer_idx}_{timestamp}"
    else:
        output_dir = f"experiments/sequential/layer_{layer_idx}_{timestamp}"
    
    # Build command
    cmd = [
        "python", "scripts/train_single_layer_npt.py",
        "--model_name", args.model_name,
        "--model_size", args.model_size,
        "--convert_layers", str(layer_idx),
        "--single_layer_mode",
        "--np_rank", str(args.np_rank),
        "--max_steps", str(args.steps_per_layer),
        "--stage1_steps", str(args.stage1_steps),
        "--batch_size", str(args.batch_size),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--learning_rate", str(args.learning_rate),
        "--dataset_preset", args.dataset_preset,
        "--max_length", str(args.max_length),
        "--output_dir", output_dir,
        "--wandb_project", args.wandb_project,
        "--wandb_name", f"layer_{layer_idx}_{args.model_size}_{timestamp}",
        "--wandb_mode", args.wandb_mode,
        "--wandb_tags", "sequential", f"layer_{layer_idx}",
        "--logging_steps", "10",
        "--eval_steps", "500",
        "--save_steps", "1000",
        "--generation_steps", "500",
    ]
    
    # Add demo mode if specified
    if args.demo_mode:
        cmd.append("--demo_mode")
    
    # Load previously trained weights if available
    if args.checkpoint_dir and checkpoint_info["trained_layers"]:
        accumulated_path = Path(args.checkpoint_dir) / "accumulated_npt_weights.pt"
        if accumulated_path.exists():
            cmd.extend(["--load_npt_weights", args.checkpoint_dir])
            logger.info(f"Loading previously trained weights from {args.checkpoint_dir}")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Training Layer {layer_idx}/{args.num_layers - 1}")
    logger.info(f"{'='*80}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    if args.dry_run:
        logger.info("DRY RUN: Command not executed")
        return True
    
    # Execute training
    try:
        result = subprocess.run(cmd, check=True)
        
        # Save checkpoint info
        if args.checkpoint_dir:
            checkpoint_info["trained_layers"].append(layer_idx)
            checkpoint_info["layer_checkpoints"][str(layer_idx)] = output_dir
            save_checkpoint_info(args.checkpoint_dir, checkpoint_info)
            
            # Merge NPT weights
            merge_npt_weights(args.checkpoint_dir, layer_idx, output_dir)
        
        logger.info(f"Successfully trained layer {layer_idx}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to train layer {layer_idx}: {e}")
        return False
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return False


def main():
    """Main function for sequential training."""
    args = parse_args()
    
    # Demo mode adjustments
    if args.demo_mode:
        logger.info("Running in DEMO MODE")
        args.steps_per_layer = 100
        args.stage1_steps = 20
        args.num_layers = 4  # Only train 4 layers in demo
    
    # Setup checkpoint directory
    if args.checkpoint_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.checkpoint_dir = f"experiments/sequential_checkpoint_{timestamp}"
    
    # Load checkpoint info
    checkpoint_info = load_checkpoint_info(args.checkpoint_dir)
    
    # Get layers to train
    layers_to_train = get_layers_to_train(args)
    
    # Filter out already trained layers if resuming
    if args.start_from_layer > 0:
        layers_to_train = [l for l in layers_to_train if l >= args.start_from_layer]
    
    # Skip already trained layers from checkpoint
    if checkpoint_info["trained_layers"]:
        logger.info(f"Previously trained layers: {checkpoint_info['trained_layers']}")
        layers_to_train = [l for l in layers_to_train if l not in checkpoint_info["trained_layers"]]
    
    if not layers_to_train:
        logger.info("All specified layers have already been trained")
        return
    
    logger.info(f"\n{'='*80}")
    logger.info("Sequential NPT Layer Training")
    logger.info(f"{'='*80}")
    logger.info(f"Model: {args.model_name} ({args.model_size})")
    logger.info(f"Layers to train: {layers_to_train}")
    logger.info(f"Steps per layer: {args.steps_per_layer}")
    logger.info(f"Stage 1 steps: {args.stage1_steps}")
    logger.info(f"NPT Rank: {args.np_rank}")
    logger.info(f"Checkpoint directory: {args.checkpoint_dir}")
    logger.info(f"{'='*80}\n")
    
    # Train each layer sequentially
    for i, layer_idx in enumerate(layers_to_train):
        logger.info(f"\nProgress: {i+1}/{len(layers_to_train)} layers")
        
        success = train_single_layer(layer_idx, args, checkpoint_info)
        
        if not success:
            logger.error(f"Training stopped at layer {layer_idx}")
            break
        
        # Small delay between layers
        if i < len(layers_to_train) - 1:
            logger.info("Waiting 5 seconds before next layer...")
            import time
            time.sleep(5)
    
    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info("Sequential Training Complete")
    logger.info(f"{'='*80}")
    logger.info(f"Trained layers: {checkpoint_info['trained_layers']}")
    logger.info(f"Accumulated weights: {args.checkpoint_dir}/accumulated_npt_weights.pt")
    
    # Create final model loading script
    create_load_script(args.checkpoint_dir, args.model_name, checkpoint_info["trained_layers"])


def create_load_script(checkpoint_dir, model_name, trained_layers):
    """Create a Python script to load the trained model."""
    
    # Ensure directory exists
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    script_content = f'''#!/usr/bin/env python3
"""
Load NPT model with sequentially trained layers.
Generated at {datetime.now().isoformat()}
"""

import torch
from pathlib import Path
from src.npt import NPTLlamaModel, NPTConfig

# Load base model
model = NPTLlamaModel.from_pretrained("{model_name}")

# Configure NPT for trained layers
npt_config = NPTConfig(
    layers_to_convert={trained_layers},
    np_rank=256,
    single_layer_mode=False  # Multi-layer mode
)

# Convert layers
model.convert_to_npt(npt_config)

# Load accumulated weights
weights_path = "{checkpoint_dir}/accumulated_npt_weights.pt"
if Path(weights_path).exists():
    npt_weights = torch.load(weights_path, map_location='cpu')
    model.load_npt_weights(npt_weights)
    print(f"Loaded NPT weights from {{weights_path}}")

print(f"Loaded NPT model with layers {trained_layers} converted")
print(f"Total NPT parameters: {{model.count_parameters()['npt']:,}}")
'''
    
    script_path = Path(checkpoint_dir) / "load_model.py"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    logger.info(f"Created model loading script: {script_path}")


if __name__ == "__main__":
    main()