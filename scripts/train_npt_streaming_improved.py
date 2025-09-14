#!/usr/bin/env python3
"""
Advanced NPT training script with improved loss functions and convergence strategies.

Key improvements:
- Layer-wise supervision for better gradient flow
- Cosine similarity + MSE combination
- Adaptive regularization with warmup
- Knowledge distillation on logits
- Progressive unfreezing
- Gradient monitoring per layer
"""

import argparse
import sys
from pathlib import Path
import torch
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
from src.training.improved_losses import (
    ImprovedEquivalenceLoss,
    ProgressiveUnfreezing,
    GradientMonitor,
    ImprovedLossOutput
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
        description="Train NPT model with improved loss functions"
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
    
    # Improved loss arguments
    parser.add_argument(
        "--use_layerwise",
        action="store_true",
        default=True,
        help="Use layer-wise supervision"
    )
    parser.add_argument(
        "--use_distillation",
        action="store_true",
        default=True,
        help="Use knowledge distillation on logits"
    )
    parser.add_argument(
        "--use_cosine",
        action="store_true",
        default=True,
        help="Use cosine similarity in addition to MSE"
    )
    parser.add_argument(
        "--cosine_weight",
        type=float,
        default=0.5,
        help="Weight for cosine similarity loss"
    )
    parser.add_argument(
        "--distillation_weight",
        type=float,
        default=0.3,
        help="Weight for distillation loss"
    )
    parser.add_argument(
        "--spectral_penalty",
        type=float,
        default=0.1,
        help="Penalty for spectral norm of weight updates"
    )
    parser.add_argument(
        "--orthogonal_penalty",
        type=float,
        default=0.05,
        help="Penalty for orthogonality between layers"
    )
    parser.add_argument(
        "--progressive_unfreezing",
        action="store_true",
        default=False,
        help="Use progressive unfreezing strategy"
    )
    parser.add_argument(
        "--layer_decay",
        type=float,
        default=0.9,
        help="Exponential decay factor for layer-wise losses"
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
        help="Base regularization weight for v_a and v_b"
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
        default="npt-training-improved",
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
    """Setup NPT model with improved initialization."""
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
    
    # Setup NPT conversion with improved initialization
    num_layers = config.num_hidden_layers
    
    if args.convert_layers == "all":
        layers_to_convert = list(range(num_layers))
    elif args.convert_layers == "upper_half":
        layers_to_convert = list(range(num_layers // 2, num_layers))
    elif args.convert_layers == "lower_half":
        layers_to_convert = list(range(num_layers // 2))
    else:
        layers_to_convert = [int(x.strip()) for x in args.convert_layers.split(',')]
    
    # Create NPT config with depth-aware initialization
    npt_configs = []
    for layer_idx in layers_to_convert:
        # Scale initialization based on layer depth
        depth_scale = (layer_idx + 1) / num_layers
        init_scale = args.np_init_scale * (0.5 + 0.5 * (1.0 - depth_scale))
        
        npt_config = NPTConfig(
            layers_to_convert=[layer_idx],
            np_rank=args.np_rank,
            np_init_scale=init_scale
        )
        
        # Convert this specific layer
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
    
    return model, config, layers_to_convert


class ImprovedNPTTrainer(NPTTrainer):
    """Extended trainer with improved loss functions and monitoring."""
    
    def __init__(
        self,
        model,
        config,
        train_loader,
        val_loader,
        tracker=None,
        tokenizer=None,
        use_improved_loss=True,
        loss_config=None,
        gradient_monitor=None,
        progressive_unfreezer=None,
        **kwargs
    ):
        super().__init__(model, config, train_loader, val_loader, **kwargs)
        self.tracker = tracker
        self.tokenizer = tokenizer
        self.use_improved_loss = use_improved_loss
        self.gradient_monitor = gradient_monitor
        self.progressive_unfreezer = progressive_unfreezer
        self.device = config.device  # Add device attribute
        
        # Initialize improved loss
        if use_improved_loss:
            self.loss_fn = ImprovedEquivalenceLoss(**loss_config)
        
        self.generation_prompts = [
            "The future of artificial intelligence",
            "Once upon a time in a distant galaxy",
            "The key to understanding neural networks",
        ]
    
    def get_lr(self):
        """Get the current learning rate from the optimizer."""
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            return self.optimizer.param_groups[0]['lr']
        return self.config.learning_rate
    
    def collect_layer_outputs(self, input_ids, attention_mask=None):
        """Collect outputs from all NPT layers for layer-wise supervision."""
        # First, do a complete forward pass in standard mode
        self.model.set_npt_mode(False)
        with torch.no_grad():
            original_outputs = []
            original_hidden_states = self.model.model.embed_tokens(input_ids)
            
            # Create position embeddings
            batch_size, seq_len = input_ids.shape
            head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
            cos = torch.ones(batch_size, seq_len, head_dim, 
                            dtype=original_hidden_states.dtype, device=original_hidden_states.device)
            sin = torch.zeros(batch_size, seq_len, head_dim,
                             dtype=original_hidden_states.dtype, device=original_hidden_states.device)
            position_embeddings = (cos, sin)
            
            # Process through all layers in standard mode
            hidden_states = original_hidden_states
            for i, layer in enumerate(self.model.model.layers):
                layer_out = layer(
                    hidden_states,
                    attention_mask=None,  # Let model handle mask internally
                    position_embeddings=position_embeddings,
                    use_cache=False,
                    output_attentions=False
                )
                if isinstance(layer_out, tuple):
                    hidden_states = layer_out[0]
                else:
                    hidden_states = layer_out
                
                # Only collect outputs from NPT layer indices
                if i in self.model.npt_layers:
                    original_outputs.append(hidden_states.clone())
            
            # Final layer norm and logits
            hidden_states = self.model.model.norm(hidden_states)
            original_logits = self.model.lm_head(hidden_states)
        
        # Now do a complete forward pass in NPT mode
        self.model.set_npt_mode(True)
        npt_outputs = []
        v_a_list = []
        v_b_list = []
        
        npt_hidden_states = self.model.model.embed_tokens(input_ids)
        hidden_states = npt_hidden_states
        
        # Process through all layers in NPT mode
        for i, layer in enumerate(self.model.model.layers):
            # If this is an NPT layer, collect v_a and v_b
            if i in self.model.npt_layers:
                # Hook to collect v_a, v_b
                v_a, v_b = None, None
                def hook_fn(module, input, output):
                    nonlocal v_a, v_b
                    if isinstance(output, tuple) and len(output) == 2:
                        v_a, v_b = output
                
                handle = layer.np_component.register_forward_hook(hook_fn)
                
                layer_out = layer(
                    hidden_states,
                    attention_mask=None,
                    position_embeddings=position_embeddings,
                    use_cache=False,
                    output_attentions=False
                )
                
                handle.remove()
                
                if isinstance(layer_out, tuple):
                    hidden_states = layer_out[0]
                else:
                    hidden_states = layer_out
                
                npt_outputs.append(hidden_states)
                
                if v_a is not None and v_b is not None:
                    v_a_list.append(v_a)
                    v_b_list.append(v_b)
            else:
                # Standard layer (even in NPT mode, non-NPT layers run normally)
                layer_out = layer(
                    hidden_states,
                    attention_mask=None,
                    position_embeddings=position_embeddings,
                    use_cache=False,
                    output_attentions=False
                )
                if isinstance(layer_out, tuple):
                    hidden_states = layer_out[0]
                else:
                    hidden_states = layer_out
        
        # Final layer norm and logits
        hidden_states = self.model.model.norm(hidden_states)
        npt_logits = self.model.lm_head(hidden_states)
        
        return {
            'npt_outputs': {'hidden_states': npt_outputs, 'logits': npt_logits},
            'original_outputs': {'hidden_states': original_outputs, 'logits': original_logits},
            'v_a_list': v_a_list,
            'v_b_list': v_b_list
        }
    
    def train_step(self, batch):
        """Training step with improved loss."""
        self.model.train()

        # Progressive unfreezing
        if self.progressive_unfreezer:
            self.progressive_unfreezer.update(self.global_step)

        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Zero gradients only at the start of accumulation cycle
        if self.batch_count % self.config.gradient_accumulation_steps == 0:
            self.optimizer.zero_grad()

        if self.use_improved_loss:
            # Collect layer-wise outputs
            outputs = self.collect_layer_outputs(input_ids, attention_mask)

            # Compute improved loss
            loss_output = self.loss_fn(
                npt_outputs=outputs['npt_outputs'],
                original_outputs=outputs['original_outputs'],
                v_a_list=outputs['v_a_list'],
                v_b_list=outputs['v_b_list'],
                current_step=self.global_step
            )

            # Scale loss for gradient accumulation
            loss = loss_output.total_loss / self.config.gradient_accumulation_steps

            # Create metrics (but filled in only when optimizer steps)
            from dataclasses import dataclass
            @dataclass
            class Metrics:
                step: int
                total_loss: float
                fidelity_loss: float
                regularization_loss: float
                learning_rate: float
                grad_norm: float = 0.0

            # Store raw loss values for metrics (unscaled)
            raw_loss_values = {
                'total_loss': loss_output.total_loss.item(),
                'fidelity_loss': loss_output.fidelity_loss.item(),
                'regularization_loss': loss_output.regularization_loss.item(),
                'metrics': loss_output.metrics
            }
        else:
            # Use standard loss (fallback) - delegate to parent
            parent_result = super().train_step(batch)
            if parent_result[1]:  # If parent took a step
                return parent_result
            else:
                # Parent didn't step, we handle accumulation
                loss = parent_result[0].total_loss / self.config.gradient_accumulation_steps if parent_result[0] else None
                raw_loss_values = None

        # Backward pass
        if self.config.mixed_precision and self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Increment batch counter
        self.batch_count += 1

        # Only perform optimizer step after accumulating gradients
        if self.batch_count % self.config.gradient_accumulation_steps == 0:
            # Gradient monitoring and clipping
            grad_norm = 0.0
            if self.gradient_monitor:
                self.gradient_monitor.clip_and_monitor()
                grad_stats = self.gradient_monitor.get_stats()

            # Compute gradient norm and clip
            if self.config.gradient_clip_value:
                if self.config.mixed_precision and self.scaler:
                    self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_value
                )
                grad_norm = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm

            # Update gradient scale for loss
            if self.use_improved_loss and hasattr(self.loss_fn, 'update_gradient_scale'):
                self.loss_fn.update_gradient_scale(grad_norm)

            # Optimizer step
            if self.config.mixed_precision and self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            # Learning rate scheduler
            if self.scheduler:
                self.scheduler.step()

            # Increment global step only when optimizer steps
            self.global_step += 1

            # Create metrics only when we take a step
            if raw_loss_values:
                metrics = Metrics(
                    step=self.global_step,
                    total_loss=raw_loss_values['total_loss'],
                    fidelity_loss=raw_loss_values['fidelity_loss'],
                    regularization_loss=raw_loss_values['regularization_loss'],
                    learning_rate=self.get_lr(),
                    grad_norm=grad_norm
                )

                # Add additional metrics
                for key, value in raw_loss_values['metrics'].items():
                    setattr(metrics, key, value)

                # Add gradient stats if available
                if self.gradient_monitor:
                    grad_stats = self.gradient_monitor.get_stats()
                    for key, value in grad_stats.items():
                        setattr(metrics, f"grad_{key}", value)
            else:
                # Fallback metrics
                metrics = Metrics(
                    step=self.global_step,
                    total_loss=0.0,
                    fidelity_loss=0.0,
                    regularization_loss=0.0,
                    learning_rate=self.get_lr(),
                    grad_norm=grad_norm
                )

            # WandB logging (only when we take a step)
            if self.tracker and self.global_step % self.config.logging_steps == 0:
                self.tracker.log_metrics(vars(metrics), step=self.global_step)

                # Log gradients and weights periodically
                if self.global_step % (self.config.logging_steps * 10) == 0:
                    self.tracker.log_gradients(self.model, step=self.global_step)
                    self.tracker.log_weights(self.model, step=self.global_step)

            # Generate samples periodically (only when we take a step)
            if (self.tokenizer and
                self.global_step % self.config.generation_steps == 0 and
                self.global_step > 0):

                # Generate samples
                self.model.eval()
                generated_samples = []

                print(f"\n{'='*80}")
                print(f"Generating samples at step {self.global_step}")
                print(f"{'='*80}")

                for prompt in self.generation_prompts[:2]:  # Use first 2 prompts for terminal
                    # Tokenize prompt
                    inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=50)
                    input_ids = inputs.input_ids.to(self.device)

                    # Generate with NPT model
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
                    generated_samples.append({
                        'prompt': prompt,
                        'generated': generated_text
                    })

                    # Print to terminal
                    print(f"\nPrompt: {prompt}")
                    print(f"Generated: {generated_text}")

                print(f"{'='*80}\n")

                # Log to WandB if available
                if self.tracker:
                    self.tracker.log_sample_outputs(
                        self.model,
                        self.tokenizer,
                        self.generation_prompts,
                        step=self.global_step,
                        max_length=50
                    )

                self.model.train()

            # Return metrics and indicate that we took a step
            return metrics, True
        else:
            # No optimizer step taken - return None for metrics
            return None, False
    
    def evaluate(self, eval_loader=None):
        """Override evaluate to handle improved loss properly."""
        if not self.use_improved_loss:
            # Use parent's evaluate for standard loss
            return super().evaluate(eval_loader)
        
        # Custom evaluation for improved loss
        if eval_loader is None:
            eval_loader = self.val_loader
        
        if eval_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        total_fidelity = 0.0
        total_regularization = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                if num_batches >= 10:  # Limit evaluation batches
                    break
                
                input_ids = batch['input_ids'].to(self.device)
                
                # Collect layer-wise outputs
                outputs = self.collect_layer_outputs(input_ids)
                
                # Compute improved loss
                loss_output = self.loss_fn(
                    npt_outputs=outputs['npt_outputs'],
                    original_outputs=outputs['original_outputs'],
                    v_a_list=outputs['v_a_list'],
                    v_b_list=outputs['v_b_list'],
                    current_step=self.global_step
                )
                
                total_loss += loss_output.total_loss.item()
                total_fidelity += loss_output.fidelity_loss.item()
                total_regularization += loss_output.regularization_loss.item()
                num_batches += 1
        
        # Average metrics
        metrics = {
            'val_loss': total_loss / max(1, num_batches),
            'val_fidelity_loss': total_fidelity / max(1, num_batches),
            'val_regularization_loss': total_regularization / max(1, num_batches),
        }
        
        self.model.train()
        return metrics


def main():
    """Main training function with improved loss."""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Demo mode adjustments
    if args.demo_mode:
        logger.info("Running in DEMO MODE")
        args.max_steps = 50
        args.batch_size = 2
        args.num_workers = 0
    
    # Setup model with improved initialization
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
    
    # Setup WandB
    if args.wandb_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.wandb_name = f"npt_improved_{args.model_size}_{timestamp}"
    
    if args.output_dir is None:
        args.output_dir = f"experiments/{args.wandb_name}"
    
    # Add improved loss tags
    if args.wandb_tags is None:
        args.wandb_tags = []
    args.wandb_tags.extend(["improved_loss", "layerwise", "adaptive_reg"])
    
    tracker = WandBTracker(
        project=args.wandb_project,
        name=args.wandb_name,
        config=vars(args),
        tags=args.wandb_tags,
        mode=args.wandb_mode
    )
    tracker.init(model=model)
    
    # Setup gradient monitor
    gradient_monitor = GradientMonitor(model, clip_value=args.gradient_clip)
    
    # Setup progressive unfreezing if enabled
    progressive_unfreezer = None
    if args.progressive_unfreezing:
        # Unfreeze schedule: (step, num_layers_to_unfreeze)
        num_converted = len(converted_layers)
        unfreeze_schedule = [
            (0, min(4, num_converted)),  # Start with 4 layers
            (args.warmup_steps, min(8, num_converted)),  # After warmup
            (args.max_steps // 3, min(12, num_converted)),  # 1/3 through
            (args.max_steps // 2, num_converted),  # Halfway through
        ]
        progressive_unfreezer = ProgressiveUnfreezing(
            model, num_converted, unfreeze_schedule
        )
    
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
        stride=args.stride,
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
    
    # Loss configuration
    loss_config = {
        'use_layerwise': args.use_layerwise,
        'use_distillation': args.use_distillation,
        'base_lambda': args.lambda_reg,
        'distillation_weight': args.distillation_weight,
        'hidden_weight': 1.0 - args.distillation_weight
    }
    
    # Create improved trainer
    logger.info("Initializing improved trainer...")
    trainer = ImprovedNPTTrainer(
        model=model,
        config=training_config,
        train_loader=train_loader,
        val_loader=val_loader,
        tracker=tracker,
        tokenizer=tokenizer,
        use_improved_loss=True,
        loss_config=loss_config,
        gradient_monitor=gradient_monitor,
        progressive_unfreezer=progressive_unfreezer
    )
    
    # Start training
    logger.info("\n" + "=" * 80)
    logger.info("Starting NPT Training with Improved Loss Functions")
    logger.info("=" * 80)
    logger.info(f"Improvements enabled:")
    logger.info(f"  - Layer-wise supervision: {args.use_layerwise}")
    logger.info(f"  - Knowledge distillation: {args.use_distillation}")
    logger.info(f"  - Cosine similarity: {args.use_cosine}")
    logger.info(f"  - Adaptive regularization: Yes")
    logger.info(f"  - Spectral norm penalty: {args.spectral_penalty}")
    logger.info(f"  - Progressive unfreezing: {args.progressive_unfreezing}")
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
        if tracker:
            tracker.finish()
    
    logger.info("\n" + "=" * 80)
    logger.info("Training Complete!")
    logger.info(f"Final checkpoint: {training_config.output_dir}/checkpoints/final")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()