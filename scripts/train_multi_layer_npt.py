#!/usr/bin/env python3
"""
Multi-layer NPT training script based on single-layer success.

Key strategies:
- Direct MLP supervision per layer
- Progressive training (lower layers first)
- Conservative hyperparameters
- Per-layer attention encoding
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from datetime import datetime
import json
from typing import Optional, Dict, List, Tuple
import numpy as np

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
from src.training.multi_layer_losses import (
    MultiLayerEquivalenceLoss,
    ProgressiveLayerScheduler,
    MultiLayerLossOutput
)
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
        description="Train multi-layer NPT with progressive strategy"
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
        default="all",
        help="Which layers to convert: 'all', 'upper_half', 'lower_half', or comma-separated indices"
    )
    
    # NPT configuration
    parser.add_argument(
        "--np_rank",
        type=int,
        default=128,  # Higher than original 64
        help="Rank for NP component"
    )
    parser.add_argument(
        "--np_init_scale",
        type=float,
        default=0.01,  # Higher for better initialization
        help="Initialization scale for NP components"
    )
    
    # Loss weights (conservative based on single-layer success)
    parser.add_argument(
        "--direct_mlp_weight",
        type=float,
        default=2.0,  # Key for convergence
        help="Weight for direct MLP supervision loss"
    )
    parser.add_argument(
        "--attention_encoding_weight",
        type=float,
        default=1.5,
        help="Weight for attention encoding loss"
    )
    parser.add_argument(
        "--layer_fidelity_weight",
        type=float,
        default=1.0,
        help="Weight for layer-wise fidelity loss"
    )
    parser.add_argument(
        "--final_output_weight",
        type=float,
        default=1.0,
        help="Weight for final output loss"
    )
    
    # Progressive training
    parser.add_argument(
        "--progressive_training",
        action="store_true",
        default=True,
        help="Use progressive layer training (lower layers first)"
    )
    parser.add_argument(
        "--warmup_layers",
        type=int,
        default=2,
        help="Number of layers to start with"
    )
    parser.add_argument(
        "--steps_per_stage",
        type=int,
        default=2000,
        help="Steps before adding more layers"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset_preset",
        type=str,
        choices=["small", "medium", "large", "xlarge"],
        default="medium",
        help="Dataset preset for streaming"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    
    # Training arguments (conservative)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Training batch size per device"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,  # Much lower than original
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
        default=0.001,  # Lower regularization
        help="Regularization weight for v_a and v_b"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=20000,
        help="Maximum training steps"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=1000,
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
        default=500,
        help="Evaluate every N steps"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--generation_steps",
        type=int,
        default=500,
        help="Generate samples every N steps"
    )
    
    # WandB arguments
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="npt-multi-layer",
        help="WandB project name"
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="WandB run name"
    )
    parser.add_argument(
        "--wandb_tags",
        type=str,
        nargs="+",
        default=None,
        help="WandB tags"
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
        help="Run in demo mode"
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
            vocab_size=128256,
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
    }
    
    config = configs.get(args.model_size)
    if config:
        config._attn_implementation = "eager"
    return config


def setup_model(args):
    """Setup NPT model with layer-specific initialization."""
    logger.info(f"Setting up model: {args.model_name} ({args.model_size})")
    
    # Get configuration
    config = get_model_config(args)
    if config is None:
        config = AutoConfig.from_pretrained(args.model_name)
        config._attn_implementation = "eager"
    
    # Create model
    if args.model_name and "demo" not in args.model_size:
        try:
            model = NPTLlamaModel.from_pretrained(args.model_name)
            logger.info(f"Loaded pretrained model from {args.model_name}")
        except:
            logger.warning(f"Could not load pretrained model {args.model_name}, using random init")
            model = NPTLlamaModel(config)
    else:
        model = NPTLlamaModel(config)
    
    # Get layers to convert
    num_layers = config.num_hidden_layers
    
    if args.convert_layers == "all":
        layers_to_convert = list(range(num_layers))
    elif args.convert_layers == "upper_half":
        layers_to_convert = list(range(num_layers // 2, num_layers))
    elif args.convert_layers == "lower_half":
        layers_to_convert = list(range(num_layers // 2))
    else:
        layers_to_convert = [int(x.strip()) for x in args.convert_layers.split(',')]
    
    logger.info(f"Converting layers {layers_to_convert} to NPT")
    
    # Convert layers with depth-aware initialization
    for layer_idx in layers_to_convert:
        # Scale initialization based on layer depth
        # Lower layers get slightly larger init for stability
        depth_factor = 1.0 - (layer_idx / num_layers) * 0.5
        init_scale = args.np_init_scale * depth_factor
        
        npt_config = NPTConfig(
            layers_to_convert=[layer_idx],
            np_rank=args.np_rank,
            np_init_scale=init_scale,
            single_layer_mode=False  # Not single-layer mode
        )
        
        model.convert_to_npt(npt_config)
    
    # Freeze base parameters
    model.freeze_base_parameters()
    
    # Log parameter counts
    param_counts = model.count_parameters()
    logger.info(f"Model parameters:")
    logger.info(f"  Total: {param_counts['total']:,}")
    logger.info(f"  Base (frozen): {param_counts['base']:,}")
    logger.info(f"  NPT (trainable): {param_counts['npt']:,}")
    logger.info(f"  NPT ratio: {param_counts['npt_ratio']:.2%}")
    logger.info(f"  NPT layers: {layers_to_convert}")
    
    return model, config, layers_to_convert


class MultiLayerNPTTrainer(NPTTrainer):
    """Trainer for multi-layer NPT with progressive training."""
    
    def __init__(
        self,
        model,
        config,
        train_loader,
        val_loader,
        npt_layers,
        loss_fn,
        progressive_scheduler=None,
        tracker=None,
        tokenizer=None,
        **kwargs
    ):
        super().__init__(model, config, train_loader, val_loader, **kwargs)
        
        self.npt_layers = npt_layers
        self.loss_fn = loss_fn
        self.progressive_scheduler = progressive_scheduler
        self.tracker = tracker
        self.tokenizer = tokenizer
        self.device = config.device
        
        self.generation_prompts = [
            "The future of artificial intelligence",
            "Once upon a time in a distant galaxy",
            "The key to understanding neural networks",
        ]
    
    def get_lr(self):
        """Get current learning rate."""
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            return self.optimizer.param_groups[0]['lr']
        return self.config.learning_rate
    
    def collect_detailed_layer_outputs(self, input_ids):
        """
        Collect detailed outputs for each NPT layer.
        
        This is the key difference from the original implementation.
        We need to capture:
        - Attention output before residual
        - Original MLP output with attention
        - Modulated MLP output
        - v_a and v_b vectors
        """
        batch_size, seq_len = input_ids.shape
        
        # Create position embeddings
        head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
        cos = torch.ones(batch_size, seq_len, head_dim, 
                        dtype=torch.float32, device=self.device)
        sin = torch.zeros(batch_size, seq_len, head_dim,
                         dtype=torch.float32, device=self.device)
        position_embeddings = (cos, sin)
        
        # First pass: collect original outputs
        self.model.set_npt_mode(False)
        with torch.no_grad():
            layer_outputs = {}
            original_hidden_states = {}
            
            hidden_states = self.model.model.embed_tokens(input_ids)
            
            for i, layer in enumerate(self.model.model.layers):
                if i in self.npt_layers:
                    # Store input to this layer
                    residual = hidden_states.clone()
                    
                    # Get attention output
                    normed = layer.input_layernorm(hidden_states)
                    attn_outputs = layer.self_attn(
                        hidden_states=normed,
                        attention_mask=None,
                        position_embeddings=position_embeddings,
                        past_key_values=None,
                        cache_position=None,
                        use_cache=False,
                        output_attentions=False
                    )
                    attention_output = attn_outputs[0]
                    
                    # Standard processing with residual
                    hidden_after_attn = residual + attention_output
                    mlp_input_with_attn = layer.post_attention_layernorm(hidden_after_attn)
                    original_mlp_with_attn = layer.mlp(mlp_input_with_attn)
                    
                    # Store for this layer
                    layer_outputs[i] = {
                        'residual': residual,
                        'attention': attention_output,
                        'original_mlp_with_attn': original_mlp_with_attn
                    }
                
                # Process layer normally
                layer_out = layer(
                    hidden_states,
                    attention_mask=None,
                    position_embeddings=position_embeddings,
                    use_cache=False,
                    output_attentions=False
                )
                hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out
                
                if i in self.npt_layers:
                    original_hidden_states[i] = hidden_states.clone()
            
            # Final processing
            hidden_states = self.model.model.norm(hidden_states)
            original_logits = self.model.lm_head(hidden_states)
        
        # Second pass: NPT mode with detailed collection
        self.model.set_npt_mode(True)
        
        npt_hidden_states = {}
        v_a_list = []
        v_b_list = []
        
        hidden_states = self.model.model.embed_tokens(input_ids)
        
        for i, layer in enumerate(self.model.model.layers):
            if i in self.npt_layers:
                # Get NPT layer outputs
                residual = hidden_states
                normed = layer.input_layernorm(hidden_states)
                
                # Get attention
                attn_outputs = layer.self_attn(
                    hidden_states=normed,
                    attention_mask=None,
                    position_embeddings=position_embeddings,
                    past_key_values=None,
                    cache_position=None,
                    use_cache=False,
                    output_attentions=False
                )
                attn_out = attn_outputs[0]
                
                # Get v_a, v_b from NP component
                v_a, v_b = layer.np_component(attn_out)
                v_a_list.append(v_a)
                v_b_list.append(v_b)
                
                # Get modulated MLP output
                mlp_input = layer.post_attention_layernorm(residual)
                mlp_modulated = layer._apply_modulated_mlp_efficient(mlp_input, v_a, v_b)
                
                # Update layer_outputs with NPT-specific data
                layer_outputs[i].update({
                    'v_a': v_a,
                    'v_b': v_b,
                    'modulated_mlp': mlp_modulated
                })
            
            # Process layer
            layer_out = layer(
                hidden_states,
                attention_mask=None,
                position_embeddings=position_embeddings,
                use_cache=False,
                output_attentions=False
            )
            hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out
            
            if i in self.npt_layers:
                npt_hidden_states[i] = hidden_states
        
        # Final processing
        hidden_states = self.model.model.norm(hidden_states)
        npt_logits = self.model.lm_head(hidden_states)
        
        return {
            'layer_outputs': layer_outputs,
            'npt_hidden_states': npt_hidden_states,
            'original_hidden_states': original_hidden_states,
            'npt_logits': npt_logits,
            'original_logits': original_logits,
            'v_a_list': v_a_list,
            'v_b_list': v_b_list
        }
    
    def train_step(self, batch):
        """Training step with multi-layer loss."""
        self.model.train()
        
        input_ids = batch['input_ids'].to(self.device)
        
        # Collect detailed outputs
        outputs = self.collect_detailed_layer_outputs(input_ids)
        
        # Compute loss
        loss_output = self.loss_fn(
            layer_outputs=outputs['layer_outputs'],
            npt_hidden_states=outputs['npt_hidden_states'],
            original_hidden_states=outputs['original_hidden_states'],
            npt_logits=outputs['npt_logits'],
            original_logits=outputs['original_logits'],
            v_a_list=outputs['v_a_list'],
            v_b_list=outputs['v_b_list'],
            current_step=self.global_step
        )
        
        loss = loss_output.total_loss
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Backward pass
        if self.config.mixed_precision and self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient clipping
        if self.config.gradient_clip_value:
            if self.config.mixed_precision and self.scaler:
                self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip_value
            )
        else:
            grad_norm = 0.0
        
        # Optimizer step
        if self.config.mixed_precision and self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        # Learning rate scheduler
        if self.scheduler:
            self.scheduler.step()
        
        # Create metrics
        from dataclasses import dataclass
        @dataclass
        class Metrics:
            step: int
            total_loss: float
            direct_mlp_loss: float
            attention_encoding_loss: float
            layer_fidelity_loss: float
            final_output_loss: float
            regularization_loss: float
            learning_rate: float
            grad_norm: float
            stage: int
            num_active_layers: int
            avg_attention_similarity: float
            avg_v_a_norm: float
            avg_v_b_norm: float
        
        metrics = Metrics(
            step=self.global_step,
            total_loss=loss.item(),
            direct_mlp_loss=loss_output.direct_mlp_loss.item(),
            attention_encoding_loss=loss_output.attention_encoding_loss.item(),
            layer_fidelity_loss=loss_output.layer_fidelity_loss.item(),
            final_output_loss=loss_output.final_output_loss.item(),
            regularization_loss=loss_output.regularization_loss.item(),
            learning_rate=self.get_lr(),
            grad_norm=grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
            **loss_output.metrics
        )
        
        # Log metrics
        if self.tracker and self.global_step % self.config.logging_steps == 0:
            self.tracker.log_metrics(vars(metrics), step=self.global_step)
            
            # Log stage transition
            if self.progressive_scheduler:
                stage_info = self.progressive_scheduler.get_stage_info(self.global_step)
                if self.global_step > 0 and self.global_step % self.progressive_scheduler.steps_per_stage == 0:
                    logger.info("=" * 80)
                    logger.info(f"STAGE TRANSITION: Now training {stage_info['num_active']} layers")
                    logger.info(f"Active layers: {stage_info['active_layers']}")
                    logger.info(f"Training progress: {stage_info['progress']:.1%}")
                    logger.info("=" * 80)
        
        # Generate samples periodically
        if (self.tokenizer and 
            self.global_step % self.config.generation_steps == 0 and
            self.global_step > 0):
            self.generate_samples()
        
        self.global_step += 1
        return metrics
    
    def generate_samples(self):
        """Generate samples to monitor training progress."""
        self.model.eval()
        
        print(f"\n{'='*80}")
        print(f"Generating samples at step {self.global_step}")
        if self.progressive_scheduler:
            stage_info = self.progressive_scheduler.get_stage_info(self.global_step)
            print(f"Stage {stage_info['stage']}: Training {stage_info['num_active']} layers")
        print(f"{'='*80}")
        
        for prompt in self.generation_prompts[:2]:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=50)
            input_ids = inputs.input_ids.to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_length=50,
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"\nPrompt: {prompt}")
            print(f"Generated: {generated_text}")
        
        print(f"{'='*80}\n")
        
        self.model.train()
    
    def evaluate(self, eval_loader=None):
        """Evaluate model with multi-layer loss."""
        if eval_loader is None:
            eval_loader = self.val_loader
        
        if eval_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        total_direct_mlp = 0.0
        total_attention = 0.0
        total_fidelity = 0.0
        total_final = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(eval_loader):
                if batch_idx >= 10:  # Limit evaluation
                    break
                
                input_ids = batch['input_ids'].to(self.device)
                outputs = self.collect_detailed_layer_outputs(input_ids)
                
                loss_output = self.loss_fn(
                    layer_outputs=outputs['layer_outputs'],
                    npt_hidden_states=outputs['npt_hidden_states'],
                    original_hidden_states=outputs['original_hidden_states'],
                    npt_logits=outputs['npt_logits'],
                    original_logits=outputs['original_logits'],
                    v_a_list=outputs['v_a_list'],
                    v_b_list=outputs['v_b_list'],
                    current_step=self.global_step
                )
                
                total_loss += loss_output.total_loss.item()
                total_direct_mlp += loss_output.direct_mlp_loss.item()
                total_attention += loss_output.attention_encoding_loss.item()
                total_fidelity += loss_output.layer_fidelity_loss.item()
                total_final += loss_output.final_output_loss.item()
                num_batches += 1
        
        metrics = {
            'val_loss': total_loss / max(1, num_batches),
            'val_direct_mlp_loss': total_direct_mlp / max(1, num_batches),
            'val_attention_encoding_loss': total_attention / max(1, num_batches),
            'val_layer_fidelity_loss': total_fidelity / max(1, num_batches),
            'val_final_output_loss': total_final / max(1, num_batches),
        }
        
        self.model.train()
        return metrics


def main():
    """Main training function for multi-layer NPT."""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Demo mode adjustments
    if args.demo_mode:
        logger.info("Running in DEMO MODE")
        args.max_steps = 100
        args.batch_size = 2
        args.num_workers = 0
        args.steps_per_stage = 20
    
    # Setup model
    model, model_config, converted_layers = setup_model(args)
    
    # Setup tokenizer
    logger.info("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    except:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Setup data loaders
    logger.info("Setting up data loaders...")
    streamer = MultiDatasetStreamer(
        preset=args.dataset_preset,
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    train_loader, val_loader = streamer.create_data_loaders(validation=True)
    
    # Setup progressive scheduler
    progressive_scheduler = None
    if args.progressive_training:
        progressive_scheduler = ProgressiveLayerScheduler(
            total_layers=converted_layers,
            warmup_steps=args.warmup_steps,
            steps_per_stage=args.steps_per_stage
        )
        logger.info("Progressive training enabled")
        logger.info(f"  Starting with {args.warmup_layers} layers")
        logger.info(f"  Adding layers every {args.steps_per_stage} steps")
    
    # Setup loss function
    loss_fn = MultiLayerEquivalenceLoss(
        direct_mlp_weight=args.direct_mlp_weight,
        attention_encoding_weight=args.attention_encoding_weight,
        layer_fidelity_weight=args.layer_fidelity_weight,
        final_output_weight=args.final_output_weight,
        regularization_weight=args.lambda_reg,
        progressive_scheduler=progressive_scheduler
    )
    
    # Setup WandB
    if args.wandb_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.wandb_name = f"npt_multi_{args.model_size}_{timestamp}"
    
    if args.output_dir is None:
        args.output_dir = f"experiments/{args.wandb_name}"
    
    if args.wandb_tags is None:
        args.wandb_tags = []
    args.wandb_tags.extend(["multi_layer", "progressive", "direct_supervision"])
    
    tracker = WandBTracker(
        project=args.wandb_project,
        name=args.wandb_name,
        config=vars(args),
        tags=args.wandb_tags,
        mode=args.wandb_mode
    )
    tracker.init(model=model)
    
    # Create training config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.demo_mode:
        device = "cpu"
    
    training_config = TrainingConfig(
        model_name=args.model_name,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lambda_reg=args.lambda_reg,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        max_length=args.max_length,
        stride=args.max_length // 2,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        generation_steps=args.generation_steps,
        output_dir=args.output_dir,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_clip_value=args.gradient_clip,
        device=device
    )
    
    # Create trainer
    logger.info("Initializing multi-layer NPT trainer...")
    trainer = MultiLayerNPTTrainer(
        model=model,
        config=training_config,
        train_loader=train_loader,
        val_loader=val_loader,
        npt_layers=converted_layers,
        loss_fn=loss_fn,
        progressive_scheduler=progressive_scheduler,
        tracker=tracker,
        tokenizer=tokenizer
    )
    
    # Start training
    logger.info("\n" + "=" * 80)
    logger.info("Starting Multi-Layer NPT Training")
    logger.info("=" * 80)
    logger.info(f"Training Configuration:")
    logger.info(f"  Model: {args.model_name} ({args.model_size})")
    logger.info(f"  NPT Layers: {len(converted_layers)} layers")
    logger.info(f"  NPT Rank: {args.np_rank}")
    logger.info(f"  Learning Rate: {args.learning_rate}")
    logger.info(f"  Progressive Training: {args.progressive_training}")
    if args.progressive_training:
        logger.info(f"  Initial Layers: {args.warmup_layers}")
        logger.info(f"  Steps per Stage: {args.steps_per_stage}")
    logger.info(f"WandB Run: {tracker.run.name if tracker.run else 'disabled'}")
    logger.info(f"Output Directory: {training_config.output_dir}")
    logger.info("=" * 80 + "\n")
    
    try:
        # Run training
        trainer.train()
        
        # Log final summary
        if tracker:
            param_counts = model.count_parameters()
            summary = {
                "final_steps": trainer.global_step,
                "total_parameters": param_counts['total'],
                "npt_parameters": param_counts['npt'],
                "num_npt_layers": len(converted_layers),
            }
            tracker.log_training_summary(summary)
    
    finally:
        if tracker:
            tracker.finish()
    
    logger.info("\n" + "=" * 80)
    logger.info("Multi-Layer NPT Training Complete!")
    logger.info(f"Final checkpoint: {training_config.output_dir}/checkpoints/final")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()