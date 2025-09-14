#!/usr/bin/env python3
"""
Specialized training script for single-layer NPT.

This script implements a two-stage training strategy:
1. Attention Reconstruction Stage: Train v_a to encode attention
2. Full Equivalence Stage: Train the complete transformation

Key features:
- Direct MLP supervision loss
- Gradient scaling for single NPT layer
- Dynamic loss weighting
- Mode collapse detection
"""

import argparse
import sys
from pathlib import Path
import torch
import logging
from datetime import datetime
from typing import Optional, Dict, List, Tuple

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
from src.training.single_layer_losses import (
    DirectSupervisionLoss,
    GradientScaler,
    check_mode_collapse
)
from src.training.evaluation import create_fixed_evaluator
from transformers import AutoTokenizer, LlamaConfig, AutoConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train single-layer NPT with specialized strategy"
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
        default="15",
        help="Single layer index to convert (e.g., '15' for layer 15)"
    )
    parser.add_argument(
        "--single_layer_mode",
        action="store_true",
        default=True,
        help="Enable single-layer mode optimizations"
    )
    
    # NPT configuration
    parser.add_argument(
        "--np_rank",
        type=int,
        default=256,
        help="Rank for NP component (automatically increased for single layer)"
    )
    parser.add_argument(
        "--np_init_scale",
        type=float,
        default=0.001,
        help="Initialization scale for NP components"
    )
    parser.add_argument(
        "--num_ranks",
        type=int,
        default=1,
        help="Number of rank-1 components for rank-k updates (1 for rank-1, k for rank-k)"
    )
    parser.add_argument(
        "--init_strategy",
        type=str,
        choices=["improved", "conservative"],
        default="improved",
        help="Initialization strategy: improved (better gradient flow) or conservative (original)"
    )
    
    # Loss weights
    parser.add_argument(
        "--direct_mlp_weight",
        type=float,
        default=1.0,
        help="Weight for direct MLP supervision loss"
    )
    parser.add_argument(
        "--fidelity_weight",
        type=float,
        default=0.1,
        help="Weight for final output fidelity loss"
    )

    # Training configuration
    parser.add_argument(
        "--gradient_scale_factor",
        type=float,
        default=10.0,
        help="Gradient scaling factor for single NPT layer"
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
        default=2,
        help="Training batch size per device (smaller for single layer)"
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
        help="Learning rate for stage 2"
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
        default=30000,
        help="Maximum training steps (longer for single layer)"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=2000,
        help="Warmup steps"
    )
    parser.add_argument(
        "--gradient_clip",
        type=float,
        default=0.5,
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
        default="npt-single-layer",
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
    parser.add_argument(
        "--num_eval_samples",
        type=int,
        default=1000,
        help="Number of validation samples to use for evaluation"
    )
    parser.add_argument(
        "--load_npt_weights",
        type=str,
        default=None,
        help="Path to previously trained NPT weights to load (for sequential training)"
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


def setup_single_layer_model(args):
    """Setup NPT model with single-layer conversion."""
    logger.info(f"Setting up single-layer NPT model: {args.model_name} ({args.model_size})")
    
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
    
    # Parse layer to convert
    layer_idx = int(args.convert_layers)
    logger.info(f"Converting single layer: {layer_idx}")
    
    # Create NPT config for single layer with special settings
    npt_config = NPTConfig(
        layers_to_convert=[layer_idx],
        np_rank=args.np_rank,
        np_init_scale=args.np_init_scale,
        single_layer_mode=True,  # This will be passed to NPComponent
        num_ranks=args.num_ranks,  # Support rank-k updates
        init_strategy=args.init_strategy  # Initialization strategy
    )
    
    # Convert the single layer
    model.convert_to_npt(npt_config)
    
    # Load previously trained NPT weights if provided
    if args.load_npt_weights:
        logger.info(f"Loading NPT weights from {args.load_npt_weights}")
        try:
            if args.load_npt_weights.endswith('.pt'):
                # Direct weights file
                npt_weights = torch.load(args.load_npt_weights, map_location='cpu')
            else:
                # Directory with accumulated weights
                weights_path = Path(args.load_npt_weights) / "accumulated_npt_weights.pt"
                npt_weights = torch.load(weights_path, map_location='cpu')
            
            # Load only weights for other layers (not the current one being trained)
            filtered_weights = {}
            for key, value in npt_weights.items():
                if f"layers.{layer_idx}." not in key:
                    filtered_weights[key] = value
            
            if filtered_weights:
                model.load_npt_weights(filtered_weights)
                logger.info(f"Loaded NPT weights for {len(filtered_weights)} parameters from other layers")
        except Exception as e:
            logger.warning(f"Could not load NPT weights: {e}")
    
    # Freeze base parameters
    model.freeze_base_parameters()
    
    # Log parameter counts
    param_counts = model.count_parameters()
    logger.info(f"Model parameters:")
    logger.info(f"  Total: {param_counts['total']:,}")
    logger.info(f"  Base (frozen): {param_counts['base']:,}")
    logger.info(f"  NPT (trainable): {param_counts['npt']:,}")
    logger.info(f"  NPT ratio: {param_counts['npt_ratio']:.2%}")
    logger.info(f"  Single NPT layer index: {layer_idx}")
    
    return model, config, [layer_idx]


class SingleLayerNPTTrainer(NPTTrainer):
    """Specialized trainer for single-layer NPT with direct supervision."""

    def __init__(
        self,
        model,
        config,
        train_loader,
        val_loader,
        layer_idx,
        gradient_scale_factor=10.0,
        loss_config=None,
        tracker=None,
        tokenizer=None,
        evaluator=None,
        **kwargs
    ):
        super().__init__(model, config, train_loader, val_loader, **kwargs)

        self.layer_idx = layer_idx
        self.gradient_scale_factor = gradient_scale_factor
        self.tracker = tracker
        self.tokenizer = tokenizer
        self.device = config.device
        self.evaluator = evaluator

        # Initialize direct supervision loss
        self.loss_fn = DirectSupervisionLoss(
            direct_mlp_weight=loss_config.get('direct_mlp_weight', 1.0),
            fidelity_weight=loss_config.get('fidelity_weight', 0.1),
            regularization_weight=loss_config.get('regularization_weight', 0.001)
        )

        # Initialize gradient scaler
        self.gradient_scaler = GradientScaler(scale_factor=gradient_scale_factor)

        # Generation prompts
        self.generation_prompts = [
            "The future of artificial intelligence",
            "Once upon a time in a distant galaxy",
            "The key to understanding neural networks",
        ]

        # Track best evaluation metrics
        self.best_eval_loss = float('inf')
        self.best_eval_step = 0
    
    
    def collect_single_layer_outputs(self, input_ids):
        """
        Collect outputs needed for single-layer NPT training.
        
        This includes:
        - Attention output from the NPT layer
        - Original MLP output with attention input
        - Modulated MLP output
        - v_a and v_b vectors
        """
        batch_size, seq_len = input_ids.shape
        
        # First, run in standard mode to get targets
        self.model.set_npt_mode(False)
        with torch.no_grad():
            # Embed inputs
            hidden_states = self.model.model.embed_tokens(input_ids)
            
            # Create position embeddings
            head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
            cos = torch.ones(batch_size, seq_len, head_dim, 
                            dtype=hidden_states.dtype, device=hidden_states.device)
            sin = torch.zeros(batch_size, seq_len, head_dim,
                             dtype=hidden_states.dtype, device=hidden_states.device)
            position_embeddings = (cos, sin)
            
            # Process through layers up to NPT layer
            for i in range(self.layer_idx):
                layer = self.model.model.layers[i]
                layer_out = layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    use_cache=False,
                    output_attentions=False
                )
                hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out
            
            # Store state before NPT layer
            hidden_before_npt = hidden_states.clone()
            
            # Process NPT layer in standard mode
            npt_layer = self.model.model.layers[self.layer_idx]
            
            # Get attention output separately
            residual = hidden_states
            hidden_states = npt_layer.input_layernorm(hidden_states)
            attn_outputs = npt_layer.self_attn(
                hidden_states=hidden_states,
                attention_mask=None,  # Required parameter in newer transformers
                position_embeddings=position_embeddings,
                past_key_values=None,
                cache_position=None,
                use_cache=False,
                output_attentions=False
            )
            attention_output = attn_outputs[0]
            
            # Standard processing (with residual)
            hidden_after_attn = residual + attention_output
            mlp_input_with_attn = npt_layer.post_attention_layernorm(hidden_after_attn)
            
            # Get original MLP output with attention
            original_mlp_with_attention = npt_layer.mlp(mlp_input_with_attn)
            
            # Continue through remaining layers for final output
            hidden_states = hidden_after_attn + original_mlp_with_attention
            for i in range(self.layer_idx + 1, len(self.model.model.layers)):
                layer = self.model.model.layers[i]
                layer_out = layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    use_cache=False,
                    output_attentions=False
                )
                hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out
            
            hidden_states = self.model.model.norm(hidden_states)
            original_final = self.model.lm_head(hidden_states)
        
        # Now run in NPT mode
        self.model.set_npt_mode(True)
        
        # Process up to NPT layer
        hidden_states = self.model.model.embed_tokens(input_ids)
        for i in range(self.layer_idx):
            layer = self.model.model.layers[i]
            layer_out = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                use_cache=False,
                output_attentions=False
            )
            hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out
        
        # Process NPT layer and collect v_a, v_b
        npt_layer = self.model.model.layers[self.layer_idx]
        
        # Hook to collect v_a, v_b
        v_a, v_b = None, None
        def hook_fn(module, input, output):
            nonlocal v_a, v_b
            if isinstance(output, tuple) and len(output) == 2:
                v_a, v_b = output
        
        handle = npt_layer.np_component.register_forward_hook(hook_fn)
        
        # NPT forward (no attention residual)
        residual = hidden_states
        hidden_states = npt_layer.input_layernorm(hidden_states)
        attn_outputs = npt_layer.self_attn(
            hidden_states=hidden_states,
            attention_mask=None,  # Required parameter in newer transformers
            position_embeddings=position_embeddings,
            past_key_values=None,
            cache_position=None,
            use_cache=False,
            output_attentions=False
        )
        attn_out = attn_outputs[0]
        
        # Get v_a, v_b through hook
        v_a_temp, v_b_temp = npt_layer.np_component(attn_out)
        
        handle.remove()
        
        # Get modulated MLP output
        mlp_input = npt_layer.post_attention_layernorm(residual)
        mlp_modulated = npt_layer._apply_modulated_mlp(mlp_input, v_a, v_b)
        
        # Continue through remaining layers
        hidden_states = residual + mlp_modulated
        for i in range(self.layer_idx + 1, len(self.model.model.layers)):
            layer = self.model.model.layers[i]
            layer_out = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                use_cache=False,
                output_attentions=False
            )
            hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out
        
        hidden_states = self.model.model.norm(hidden_states)
        npt_final = self.model.lm_head(hidden_states)
        
        return {
            'mlp_modulated': mlp_modulated,
            'attention_output': attention_output,
            'original_mlp_with_attention': original_mlp_with_attention,
            'v_a': v_a,
            'v_b': v_b,
            'npt_final': npt_final,
            'original_final': original_final,
            'hidden_states': mlp_input  # Hidden states input to MLP
        }
    
    def train_step(self, batch):
        """Training step with two-stage strategy."""
        self.model.train()

        input_ids = batch['input_ids'].to(self.device)

        # Collect outputs
        outputs = self.collect_single_layer_outputs(input_ids)

        # Check for mode collapse
        if self.global_step % 100 == 0:
            if check_mode_collapse(outputs['v_a'], outputs['v_b']):
                logger.warning(f"Mode collapse detected at step {self.global_step}")

        # Compute loss with direct supervision
        loss_output = self.loss_fn(outputs)
        # Scale loss for gradient accumulation
        loss = loss_output.total_loss / self.config.gradient_accumulation_steps

        # Zero gradients at the start of accumulation
        if self.batch_count % self.config.gradient_accumulation_steps == 0:
            self.optimizer.zero_grad()
        
        # Backward pass
        if self.config.mixed_precision and self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Increment batch counter
        self.batch_count += 1

        # Only perform optimizer step after accumulating gradients
        if self.batch_count % self.config.gradient_accumulation_steps == 0:
            # Scale gradients for single NPT layer
            self.gradient_scaler.scale_npt_gradients(self.model, [self.layer_idx])

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

            # Only increment global_step when optimizer steps
            self.global_step += 1
            step_taken = True
        else:
            grad_norm = 0.0
            step_taken = False
        
        # Create metrics
        from dataclasses import dataclass
        @dataclass
        class Metrics:
            step: int
            total_loss: float
            direct_mlp_loss: float
            fidelity_loss: float
            regularization_loss: float
            learning_rate: float
            grad_norm: float
            v_a_attention_similarity: float
            v_a_norm: float
            v_b_norm: float
            modulation_magnitude: float

        # Only return metrics when optimizer step is taken
        if step_taken:
            # Get metrics from loss_output, which already contains the loss values
            metrics = Metrics(
                step=self.global_step,
                total_loss=loss_output.total_loss.item(),  # Use unscaled loss for metrics
                direct_mlp_loss=loss_output.metrics['direct_mlp_loss'],
                fidelity_loss=loss_output.metrics['fidelity_loss'],
                regularization_loss=loss_output.metrics['regularization_loss'],
                learning_rate=self.optimizer.param_groups[0]['lr'],
                grad_norm=grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                v_a_attention_similarity=loss_output.metrics['v_a_attention_similarity'],
                v_a_norm=loss_output.metrics['v_a_norm'],
                v_b_norm=loss_output.metrics['v_b_norm'],
                modulation_magnitude=loss_output.metrics['modulation_magnitude'],
            )

            # Log metrics
            if self.tracker and self.global_step % self.config.logging_steps == 0:
                self.tracker.log_metrics(vars(metrics), step=self.global_step)

            # Generate samples periodically
            if (self.tokenizer and
                self.global_step % self.config.generation_steps == 0 and
                self.global_step > 0):
                self.generate_samples()

            # Run fixed evaluation periodically
            if (self.evaluator and
                self.global_step % self.config.eval_steps == 0 and
                self.global_step > 0):
                self.run_fixed_evaluation()

            return metrics, True
        else:
            return None, False
    
    def run_fixed_evaluation(self):
        """Run evaluation on fixed dataset and log metrics."""
        logger.info(f"Running fixed evaluation at step {self.global_step}")

        # Run evaluation
        eval_metrics = self.evaluator.evaluate_single_layer_npt(
            model=self.model,
            layer_idx=self.layer_idx,
            loss_fn=self.loss_fn,
            batch_size=4,
            max_batches=50  # Limit for speed
        )

        # Check if this is the best model
        if eval_metrics.loss < self.best_eval_loss:
            self.best_eval_loss = eval_metrics.loss
            self.best_eval_step = self.global_step
            logger.info(f"New best evaluation loss: {self.best_eval_loss:.4f}")

            # Save best model checkpoint
            if self.config.output_dir:
                best_path = Path(self.config.output_dir) / "checkpoints" / "best"
                best_path.mkdir(parents=True, exist_ok=True)
                self.model.save_npt_weights(best_path / "npt_weights.pt")
                logger.info(f"Saved best model to {best_path}")

        # Log to console
        logger.info(
            f"Eval metrics - Loss: {eval_metrics.loss:.4f}, "
            f"Perplexity: {eval_metrics.perplexity:.2f}, "
            f"Direct MLP: {eval_metrics.direct_mlp_loss:.4f}, "
            f"Fidelity: {eval_metrics.fidelity_loss:.4f}"
        )

        # Log to WandB
        if self.tracker:
            eval_dict = {
                'eval/loss': eval_metrics.loss,
                'eval/perplexity': eval_metrics.perplexity,
                'eval/direct_mlp_loss': eval_metrics.direct_mlp_loss,
                'eval/fidelity_loss': eval_metrics.fidelity_loss,
                'eval/regularization_loss': eval_metrics.regularization_loss,
                'eval/best_loss': self.best_eval_loss,
                'eval/best_step': self.best_eval_step,
            }
            self.tracker.log_metrics(eval_dict, step=self.global_step)

        # Save metrics to file
        if self.config.output_dir:
            self.evaluator.save_metrics(
                eval_metrics,
                save_path=Path(self.config.output_dir) / "evaluation",
                experiment_name=f"layer_{self.layer_idx}",
                step=self.global_step
            )

    def generate_samples(self):
        """Generate samples to monitor training progress."""
        self.model.eval()

        print(f"\n{'='*80}")
        print(f"Generating samples at step {self.global_step}")
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
        """Override evaluate to handle single-layer loss signature."""
        if eval_loader is None:
            eval_loader = self.val_loader
        
        if eval_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        total_direct_mlp = 0.0
        total_fidelity = 0.0
        total_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(eval_loader):
                if batch_idx >= 10:  # Limit evaluation batches
                    break
                
                input_ids = batch['input_ids'].to(self.device)
                
                # Collect outputs for single-layer evaluation
                outputs = self.collect_single_layer_outputs(input_ids)

                # Compute loss using single-layer loss function
                loss_output = self.loss_fn(outputs)

                total_loss += loss_output.total_loss.item()
                total_direct_mlp += loss_output.direct_mlp_loss.item()
                total_fidelity += loss_output.fidelity_loss.item()
                total_batches += 1

        # Compute averages
        metrics = {
            'val_loss': total_loss / max(1, total_batches),
            'val_direct_mlp_loss': total_direct_mlp / max(1, total_batches),
            'val_fidelity_loss': total_fidelity / max(1, total_batches),
        }
        
        self.model.train()
        return metrics


def main():
    """Main training function for single-layer NPT."""
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
    
    # Setup model
    model, model_config, converted_layers = setup_single_layer_model(args)
    layer_idx = converted_layers[0]
    
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
        args.wandb_name = f"npt_single_layer{layer_idx}_{args.model_size}_{timestamp}"
    
    if args.output_dir is None:
        args.output_dir = f"experiments/{args.wandb_name}"
    
    # Add tags
    if args.wandb_tags is None:
        args.wandb_tags = []
    args.wandb_tags.extend(["single_layer", f"layer_{layer_idx}", "direct_supervision"])
    
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
        'direct_mlp_weight': args.direct_mlp_weight,
        'fidelity_weight': args.fidelity_weight,
        'regularization_weight': args.lambda_reg,
    }

    # Create fixed evaluator
    logger.info("Initializing fixed evaluator...")
    evaluator = create_fixed_evaluator(
        tokenizer=tokenizer,
        config={
            'max_length': args.max_length,
            'num_eval_samples': args.num_eval_samples,  # Configurable validation set size
            'seed': 42,  # Fixed seed for reproducibility
            'device': device
        }
    )

    # Create specialized trainer
    logger.info("Initializing single-layer NPT trainer...")
    trainer = SingleLayerNPTTrainer(
        model=model,
        config=training_config,
        train_loader=train_loader,
        val_loader=val_loader,
        layer_idx=layer_idx,
        gradient_scale_factor=args.gradient_scale_factor,
        loss_config=loss_config,
        tracker=tracker,
        tokenizer=tokenizer,
        evaluator=evaluator
    )
    
    # Start training
    logger.info("\n" + "=" * 80)
    logger.info("Starting Single-Layer NPT Training")
    logger.info("=" * 80)
    logger.info(f"Training Configuration:")
    logger.info(f"  Model: {args.model_name} ({args.model_size})")
    logger.info(f"  NPT Layer: {layer_idx}")
    logger.info(f"  NPT Rank: {args.np_rank} (effective: {model.model.layers[layer_idx].np_component.rank})")
    logger.info(f"  Num Ranks (rank-k): {args.num_ranks}")
    logger.info(f"  Total Steps: {args.max_steps}")
    logger.info(f"  Gradient Scale Factor: {args.gradient_scale_factor}x")
    logger.info(f"  Direct MLP Weight: {args.direct_mlp_weight}")
    logger.info(f"  Fidelity Weight: {args.fidelity_weight}")
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
                "layer_index": layer_idx,
            }
            
            tracker.log_training_summary(summary)
    
    finally:
        if tracker:
            tracker.finish()
    
    logger.info("\n" + "=" * 80)
    logger.info("Single-Layer NPT Training Complete!")
    logger.info(f"Final checkpoint: {training_config.output_dir}/checkpoints/final")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()