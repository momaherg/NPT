#!/usr/bin/env python3
"""
Multi-layer NPT training with teacher scaffolding.

This script trains multiple NPT layers simultaneously using teacher states
to provide correct attention inputs during training. The curriculum gradually
transitions from teacher inputs to student inputs.

Key features:
- Memory efficient: Only one model in GPU memory
- Teacher scaffolding for correct attention patterns
- Configurable curriculum scheduling
- Always propagates NPT outputs to next layers
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import logging
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

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
from src.training.evaluation import create_fixed_evaluator
from transformers import AutoTokenizer, LlamaConfig, AutoConfig

# Import single-layer trainer to extend from
from train_single_layer_npt import SingleLayerNPTTrainer, setup_single_layer_model, get_model_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CurriculumStage:
    """Configuration for a curriculum stage."""
    name: str  # "teacher", "mixed", "student"
    until_step: int
    mixing_ratio: float = 0.0  # For attention input mixing (0=teacher, 1=student)


class MultiLayerNPTTrainer(SingleLayerNPTTrainer):
    """
    Trains multiple NPT layers simultaneously with teacher scaffolding.

    Key principles:
    - Teacher provides correct attention inputs during training
    - NPT outputs always propagate to next layers (no mixing)
    - Curriculum controls attention input source
    - Memory efficient: single model switches modes
    """

    def __init__(
        self,
        model,
        config,
        train_loader,
        val_loader,
        layers_to_train: List[int],
        curriculum_schedule: List[CurriculumStage],
        gradient_scale_factor=1.0,
        loss_config=None,
        tracker=None,
        tokenizer=None,
        evaluator=None,
        layer_weights=None,
        **kwargs
    ):
        # Initialize parent with first layer for compatibility
        super().__init__(
            model, config, train_loader, val_loader,
            layer_idx=layers_to_train[0],
            gradient_scale_factor=gradient_scale_factor,
            loss_config=loss_config,
            tracker=tracker,
            tokenizer=tokenizer,
            evaluator=evaluator,
            **kwargs
        )

        # Multi-layer specific attributes
        self.layers_to_train = sorted(layers_to_train)
        self.curriculum_schedule = curriculum_schedule
        self.current_stage = curriculum_schedule[0] if curriculum_schedule else None
        self.current_stage_index = 0
        self.layer_weights = layer_weights or {idx: 1.0 for idx in layers_to_train}

        # Tracking
        self.layer_metrics = {idx: [] for idx in layers_to_train}
        self.teacher_cache_hits = 0
        self.teacher_cache_misses = 0

        logger.info(f"Initialized multi-layer trainer for layers: {self.layers_to_train}")
        logger.info(f"Curriculum stages: {[s.name for s in curriculum_schedule]}")

    def _update_curriculum_stage(self):
        """Update curriculum stage based on current step."""
        for i, stage in enumerate(self.curriculum_schedule):
            if self.global_step < stage.until_step:
                if self.current_stage != stage:
                    self.current_stage = stage
                    self.current_stage_index = i
                    logger.info(f"Curriculum stage changed to: {stage.name} (mixing_ratio={stage.mixing_ratio})")

                    # Log stage transition to WandB if available
                    if self.tracker:
                        self.tracker.log_metrics({
                            'curriculum/stage_index': i,
                            'curriculum/stage_transition': 1,  # Spike to mark transition
                            'curriculum/mixing_ratio': stage.mixing_ratio,
                        }, step=self.global_step)
                break

    def _create_position_embeddings(self, batch_size, seq_len, device, dtype):
        """Create position embeddings for attention."""
        head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
        cos = torch.ones(batch_size, seq_len, head_dim, dtype=dtype, device=device)
        sin = torch.zeros(batch_size, seq_len, head_dim, dtype=dtype, device=device)
        return (cos, sin)

    def collect_teacher_states(self, input_ids):
        """
        Collect teacher states by running model in standard mode.
        Returns dict of hidden states and layer outputs.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        dtype = input_ids.dtype

        teacher_states = {}

        # Run in standard mode to get teacher outputs
        self.model.set_npt_mode(False)

        with torch.no_grad():
            # Start from embeddings
            hidden_states = self.model.model.embed_tokens(input_ids)
            position_embeddings = self._create_position_embeddings(
                batch_size, seq_len, device, hidden_states.dtype
            )

            # Process each layer
            for i in range(len(self.model.model.layers)):
                # Store state before layer
                teacher_states[f"before_{i}"] = hidden_states.clone()

                layer = self.model.model.layers[i]

                if i in self.layers_to_train:
                    # For NPT layers, collect detailed outputs
                    residual = hidden_states
                    normed = layer.input_layernorm(hidden_states)

                    # Get attention output
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

                    # Process through MLP
                    hidden_after_attn = residual + attention_output
                    mlp_normed = layer.post_attention_layernorm(hidden_after_attn)
                    mlp_output = layer.mlp(mlp_normed)

                    # Store components
                    teacher_states[f"attention_{i}"] = attention_output
                    teacher_states[f"mlp_with_attention_{i}"] = mlp_output
                    teacher_states[f"residual_{i}"] = residual

                    # Update hidden states
                    hidden_states = hidden_after_attn + mlp_output
                else:
                    # Non-NPT layer: normal forward
                    layer_out = layer(
                        hidden_states,
                        position_embeddings=position_embeddings,
                        use_cache=False,
                        output_attentions=False
                    )
                    hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out

                # Store state after layer
                teacher_states[f"after_{i}"] = hidden_states.clone()

            # Final layer norm and LM head for teacher output
            hidden_states = self.model.model.norm(hidden_states)
            teacher_states["final_hidden"] = hidden_states
            teacher_states["final_logits"] = self.model.lm_head(hidden_states)

        return teacher_states

    def collect_multi_layer_outputs(self, input_ids):
        """
        Collect outputs for all NPT layers with teacher scaffolding.

        Returns:
            - layer_outputs: Dict of outputs for each NPT layer
            - npt_final: Final output from NPT model
            - teacher_final: Final output from teacher model
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Step 1: Collect teacher states
        teacher_states = self.collect_teacher_states(input_ids)

        # Step 2: Run NPT layers with appropriate inputs
        self.model.set_npt_mode(True)
        layer_outputs = {}

        # Start from embeddings
        hidden_states = self.model.model.embed_tokens(input_ids)
        position_embeddings = self._create_position_embeddings(
            batch_size, seq_len, device, hidden_states.dtype
        )

        # Process each layer
        for i in range(len(self.model.model.layers)):
            layer = self.model.model.layers[i]

            if i in self.layers_to_train:
                # NPT layer with teacher scaffolding

                # Determine attention input based on curriculum
                if self.current_stage.name == "teacher":
                    # Pure teacher input for attention
                    attn_input = teacher_states[f"before_{i}"]
                elif self.current_stage.name == "mixed":
                    # Mix teacher and student inputs for attention
                    ratio = self.current_stage.mixing_ratio
                    attn_input = (1 - ratio) * teacher_states[f"before_{i}"] + ratio * hidden_states
                else:  # "student"
                    # Pure student input
                    attn_input = hidden_states

                # Process attention with selected input
                residual = attn_input
                normed = layer.input_layernorm(attn_input)

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

                # Get v_a, v_b from attention
                v_a, v_b = layer.np_component(attention_output)

                # CRITICAL: MLP modulation uses current propagated hidden states
                mlp_input = layer.post_attention_layernorm(hidden_states)
                mlp_modulated = layer._apply_modulated_mlp(mlp_input, v_a, v_b)

                # Store outputs for loss computation
                layer_outputs[i] = {
                    'mlp_modulated': mlp_modulated,
                    'attention_output': attention_output,
                    'teacher_attention': teacher_states[f"attention_{i}"],
                    'teacher_mlp_with_attention': teacher_states[f"mlp_with_attention_{i}"],
                    'v_a': v_a,
                    'v_b': v_b,
                    'hidden_states': mlp_input
                }

                # ALWAYS propagate NPT output to next layer
                hidden_states = hidden_states + mlp_modulated

            else:
                # Non-NPT layer: normal forward
                layer_out = layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    use_cache=False,
                    output_attentions=False
                )
                hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        # Get final outputs
        hidden_states = self.model.model.norm(hidden_states)
        npt_final = self.model.lm_head(hidden_states)
        teacher_final = teacher_states["final_logits"]

        return layer_outputs, npt_final, teacher_final

    def compute_multi_layer_loss(self, layer_outputs, npt_final, teacher_final):
        """
        Compute loss for multiple NPT layers.

        Returns:
            - total_loss: Combined loss for all layers
            - metrics: Dict of loss components for logging
        """
        total_loss = 0
        metrics = {}

        # Loss for each NPT layer
        for layer_idx in self.layers_to_train:
            outputs = layer_outputs[layer_idx]

            # Direct supervision: modulated MLP should output attn + MLP(h+attn)
            target = outputs['teacher_attention'] + outputs['teacher_mlp_with_attention']
            direct_mlp_loss = F.mse_loss(outputs['mlp_modulated'], target)

            # Regularization on v_a and v_b
            v_a = outputs['v_a']
            v_b = outputs['v_b']
            v_a_reg = v_a.pow(2).mean()
            v_b_reg = v_b.pow(2).mean()
            reg_loss = v_a_reg + v_b_reg

            # Combine with weights from config
            layer_loss = (
                self.loss_fn.direct_mlp_weight * direct_mlp_loss +
                self.loss_fn.regularization_weight * reg_loss
            )

            # Apply layer-specific weight
            layer_weight = self.layer_weights.get(layer_idx, 1.0)
            total_loss += layer_weight * layer_loss

            # Track metrics
            metrics[f'layer_{layer_idx}_direct_mlp'] = direct_mlp_loss.item()
            metrics[f'layer_{layer_idx}_reg'] = reg_loss.item()
            metrics[f'layer_{layer_idx}_v_a_norm'] = v_a.norm().item()
            metrics[f'layer_{layer_idx}_v_b_norm'] = v_b.norm().item()

            # Compute attention similarity for monitoring
            with torch.no_grad():
                v_a_flat = v_a.view(-1, v_a.size(-1))
                attn_flat = outputs['teacher_attention'].view(-1, outputs['teacher_attention'].size(-1))
                v_a_norm = F.normalize(v_a_flat, p=2, dim=-1)
                attn_norm = F.normalize(attn_flat, p=2, dim=-1)
                similarity = (v_a_norm * attn_norm).sum(dim=-1).mean().item()
                metrics[f'layer_{layer_idx}_v_a_attn_similarity'] = similarity

        # Global fidelity loss
        fidelity_loss = F.mse_loss(npt_final, teacher_final)
        total_loss += self.loss_fn.fidelity_weight * fidelity_loss
        metrics['fidelity_loss'] = fidelity_loss.item()

        # Add curriculum info as numeric values for WandB
        # Use stage index for numeric representation
        metrics['curriculum_stage_index'] = self.current_stage_index
        metrics['curriculum_mixing_ratio'] = self.current_stage.mixing_ratio

        return total_loss, metrics

    def train_step(self, batch):
        """Training step for multiple NPT layers."""
        self.model.train()

        input_ids = batch['input_ids'].to(self.device)

        # Update curriculum stage
        self._update_curriculum_stage()

        # Collect outputs for all NPT layers
        layer_outputs, npt_final, teacher_final = self.collect_multi_layer_outputs(input_ids)

        # Compute multi-layer loss
        loss, metrics = self.compute_multi_layer_loss(layer_outputs, npt_final, teacher_final)

        # Scale for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps

        # Zero gradients at start of accumulation
        if self.batch_count % self.config.gradient_accumulation_steps == 0:
            self.optimizer.zero_grad()

        # Backward pass
        if self.config.mixed_precision and self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Increment batch counter
        self.batch_count += 1

        # Optimizer step after accumulation
        if self.batch_count % self.config.gradient_accumulation_steps == 0:
            # Optional: Scale gradients per layer
            if self.gradient_scaler:
                for layer_idx in self.layers_to_train:
                    if layer_idx < len(self.model.model.layers):
                        layer = self.model.model.layers[layer_idx]
                        if hasattr(layer, 'np_component'):
                            for param in layer.np_component.parameters():
                                if param.grad is not None:
                                    # Apply layer-specific scaling if needed
                                    param.grad *= self.gradient_scale_factor

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

            # Increment global step
            self.global_step += 1
            step_taken = True
        else:
            grad_norm = 0.0
            step_taken = False

        # Return metrics when step is taken
        if step_taken:
            # Create metrics object
            from dataclasses import dataclass
            @dataclass
            class Metrics:
                step: int
                total_loss: float
                fidelity_loss: float
                curriculum_stage: str
                mixing_ratio: float
                learning_rate: float
                grad_norm: float
                layer_losses: Dict[str, float]

            final_metrics = Metrics(
                step=self.global_step,
                total_loss=loss.item() * self.config.gradient_accumulation_steps,
                fidelity_loss=metrics['fidelity_loss'],
                curriculum_stage=self.current_stage.name,  # Use stage name directly
                mixing_ratio=metrics['curriculum_mixing_ratio'],
                learning_rate=self.optimizer.param_groups[0]['lr'],
                grad_norm=grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                layer_losses={k: v for k, v in metrics.items() if 'layer_' in k}
            )

            # Log metrics
            if self.tracker and self.global_step % self.config.logging_steps == 0:
                # Structure metrics for better WandB visualization
                wandb_metrics = {}

                # Primary losses (most important, shown first)
                wandb_metrics['loss/total'] = loss.item() * self.config.gradient_accumulation_steps
                wandb_metrics['loss/fidelity'] = metrics['fidelity_loss']

                # Direct MLP losses for each layer (key metric for NPT training)
                for layer_idx in self.layers_to_train:
                    key = f'layer_{layer_idx}_direct_mlp'
                    if key in metrics:
                        wandb_metrics[f'mlp_direct_loss/layer_{layer_idx}'] = metrics[key]

                # Regularization losses for each layer
                for layer_idx in self.layers_to_train:
                    key = f'layer_{layer_idx}_reg'
                    if key in metrics:
                        wandb_metrics[f'regularization/layer_{layer_idx}'] = metrics[key]

                # v_a attention similarity (key indicator of attention encoding)
                for layer_idx in self.layers_to_train:
                    key = f'layer_{layer_idx}_v_a_attn_similarity'
                    if key in metrics:
                        wandb_metrics[f'v_a_attention_similarity/layer_{layer_idx}'] = metrics[key]

                # Vector norms for monitoring
                for layer_idx in self.layers_to_train:
                    v_a_key = f'layer_{layer_idx}_v_a_norm'
                    v_b_key = f'layer_{layer_idx}_v_b_norm'
                    if v_a_key in metrics:
                        wandb_metrics[f'v_a_norm/layer_{layer_idx}'] = metrics[v_a_key]
                    if v_b_key in metrics:
                        wandb_metrics[f'v_b_norm/layer_{layer_idx}'] = metrics[v_b_key]

                # Curriculum information
                wandb_metrics['curriculum/stage_index'] = metrics['curriculum_stage_index']
                wandb_metrics['curriculum/mixing_ratio'] = metrics['curriculum_mixing_ratio']

                # Training hyperparameters
                wandb_metrics['training/learning_rate'] = self.optimizer.param_groups[0]['lr']
                wandb_metrics['training/grad_norm'] = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm

                self.tracker.log_metrics(wandb_metrics, step=self.global_step)

            # Generate samples periodically
            if (self.tokenizer and
                self.global_step % self.config.generation_steps == 0 and
                self.global_step > 0):
                self.generate_samples()

            # Run evaluation periodically
            if (self.evaluator and
                self.global_step % self.config.eval_steps == 0 and
                self.global_step > 0):
                self.run_multi_layer_evaluation()

            return final_metrics, True
        else:
            return None, False

    def run_multi_layer_evaluation(self):
        """Run evaluation for multi-layer NPT model."""
        logger.info(f"Running multi-layer evaluation at step {self.global_step}")

        self.model.eval()

        total_loss = 0
        layer_losses = {idx: 0 for idx in self.layers_to_train}
        num_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                if batch_idx >= 50:  # Limit evaluation batches
                    break

                input_ids = batch['input_ids'].to(self.device)

                # Collect outputs
                layer_outputs, npt_final, teacher_final = self.collect_multi_layer_outputs(input_ids)

                # Compute loss
                loss, metrics = self.compute_multi_layer_loss(layer_outputs, npt_final, teacher_final)

                total_loss += loss.item()
                for layer_idx in self.layers_to_train:
                    layer_losses[layer_idx] += metrics[f'layer_{layer_idx}_direct_mlp']

                num_batches += 1

        # Average losses
        avg_loss = total_loss / max(1, num_batches)
        for layer_idx in layer_losses:
            layer_losses[layer_idx] /= max(1, num_batches)

        # Log evaluation metrics
        logger.info(f"Eval - Total loss: {avg_loss:.4f}")
        for layer_idx, loss in layer_losses.items():
            logger.info(f"  Layer {layer_idx}: {loss:.4f}")

        if self.tracker:
            eval_metrics = {}

            # Primary evaluation loss
            eval_metrics['eval_loss/total'] = avg_loss

            # Per-layer evaluation losses
            for idx, loss in layer_losses.items():
                eval_metrics[f'eval_mlp_direct_loss/layer_{idx}'] = loss

            self.tracker.log_metrics(eval_metrics, step=self.global_step)

        self.model.train()

    def evaluate(self, eval_loader=None):
        """Override evaluate for multi-layer evaluation."""
        if eval_loader is None:
            eval_loader = self.val_loader

        if eval_loader is None:
            return {}

        self.model.eval()

        total_loss = 0
        total_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(eval_loader):
                if batch_idx >= 10:  # Limit evaluation
                    break

                input_ids = batch['input_ids'].to(self.device)
                layer_outputs, npt_final, teacher_final = self.collect_multi_layer_outputs(input_ids)
                loss, _ = self.compute_multi_layer_loss(layer_outputs, npt_final, teacher_final)

                total_loss += loss.item()
                total_batches += 1

        metrics = {
            'val_loss': total_loss / max(1, total_batches)
        }

        self.model.train()
        return metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train multiple NPT layers simultaneously with teacher scaffolding"
    )

    # Multi-layer specific arguments
    parser.add_argument(
        "--train_layers",
        type=str,
        required=True,
        help="Comma-separated list of layer indices to train (e.g., '14,15,16,17')"
    )
    parser.add_argument(
        "--curriculum_stages",
        type=str,
        default="teacher:5000,mixed:5000:0.5,student:20000",
        help="Curriculum stages in format stage:steps[:mixing_ratio]"
    )
    parser.add_argument(
        "--layer_weights",
        type=str,
        default="uniform",
        help="Layer weighting: uniform, linear, or comma-separated weights"
    )

    # Model arguments (from single-layer script)
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

    # NPT configuration
    parser.add_argument(
        "--np_rank",
        type=int,
        default=256,
        help="Rank for NP component"
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
        help="Number of rank-1 components for rank-k updates"
    )
    parser.add_argument(
        "--init_strategy",
        type=str,
        choices=["improved", "conservative"],
        default="improved",
        help="Initialization strategy"
    )

    # Loss weights
    parser.add_argument(
        "--direct_mlp_weight",
        type=float,
        default=10.0,
        help="Weight for direct MLP supervision loss"
    )
    parser.add_argument(
        "--fidelity_weight",
        type=float,
        default=1.0,
        help="Weight for final output fidelity loss"
    )

    # Training configuration
    parser.add_argument(
        "--gradient_scale_factor",
        type=float,
        default=2.5,
        help="Gradient scaling factor for NPT layers"
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset_preset",
        type=str,
        choices=["small", "medium", "large", "xlarge", "custom"],
        default="medium",
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
        default=4,
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
        default=5e-5,
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
        default=30000,
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
        default=0.5,
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
        default=2000,
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
        help="Run in demo mode with small model"
    )
    parser.add_argument(
        "--num_eval_samples",
        type=int,
        default=1000,
        help="Number of validation samples"
    )

    return parser.parse_args()


def parse_curriculum(curriculum_str):
    """
    Parse curriculum string into schedule.

    Format: stage:steps[:mixing_ratio],stage:steps[:mixing_ratio],...
    Example: "teacher:5000,mixed:5000:0.5,student:20000"
    """
    stages = []
    cumulative_steps = 0

    for stage_str in curriculum_str.split(','):
        parts = stage_str.strip().split(':')
        if len(parts) < 2:
            raise ValueError(f"Invalid curriculum stage: {stage_str}")

        stage_name = parts[0]
        steps = int(parts[1])
        mixing_ratio = float(parts[2]) if len(parts) > 2 else 0.0

        cumulative_steps += steps
        stages.append(CurriculumStage(
            name=stage_name,
            until_step=cumulative_steps,
            mixing_ratio=mixing_ratio
        ))

    return stages


def parse_layer_weights(weights_str, layers):
    """
    Parse layer weights configuration.

    Options:
    - "uniform": All layers have weight 1.0
    - "linear": Linear decay from 1.0 to 0.5
    - "1.0,0.8,0.6,0.4": Explicit weights per layer
    """
    if weights_str == "uniform":
        return {idx: 1.0 for idx in layers}
    elif weights_str == "linear":
        n = len(layers)
        return {layers[i]: 1.0 - 0.5 * i / max(1, n-1) for i in range(n)}
    else:
        # Try to parse as comma-separated values
        try:
            weights = [float(w.strip()) for w in weights_str.split(',')]
            if len(weights) != len(layers):
                logger.warning(f"Number of weights ({len(weights)}) doesn't match layers ({len(layers)}). Using uniform.")
                return {idx: 1.0 for idx in layers}
            return {layers[i]: weights[i] for i in range(len(layers))}
        except:
            logger.warning(f"Could not parse layer weights: {weights_str}. Using uniform.")
            return {idx: 1.0 for idx in layers}


def setup_multi_layer_model(args, layers_to_train):
    """Setup NPT model with multiple layer conversion."""
    logger.info(f"Setting up multi-layer NPT model: {args.model_name} ({args.model_size})")

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

    # Create NPT config for multiple layers
    npt_config = NPTConfig(
        layers_to_convert=layers_to_train,
        np_rank=args.np_rank,
        np_init_scale=args.np_init_scale,
        single_layer_mode=False,  # Multi-layer mode
        num_ranks=args.num_ranks,
        init_strategy=args.init_strategy
    )

    # Convert layers
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
    logger.info(f"  NPT layers: {layers_to_train}")

    return model, config


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

    # Parse configurations
    layers_to_train = [int(x.strip()) for x in args.train_layers.split(',')]
    curriculum_schedule = parse_curriculum(args.curriculum_stages)
    layer_weights = parse_layer_weights(args.layer_weights, layers_to_train)

    logger.info(f"Training layers: {layers_to_train}")
    logger.info(f"Curriculum stages: {[(s.name, s.until_step, s.mixing_ratio) for s in curriculum_schedule]}")
    logger.info(f"Layer weights: {layer_weights}")

    # Setup model
    model, model_config = setup_multi_layer_model(args, layers_to_train)

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
        layers_str = "-".join(map(str, layers_to_train))
        args.wandb_name = f"npt_multi_layers{layers_str}_{args.model_size}_{timestamp}"

    if args.output_dir is None:
        args.output_dir = f"experiments/{args.wandb_name}"

    # Add tags
    if args.wandb_tags is None:
        args.wandb_tags = []
    args.wandb_tags.extend(["multi_layer", f"layers_{len(layers_to_train)}", "teacher_scaffolding"])

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
    from src.training.single_layer_losses import DirectSupervisionLoss
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
            'num_eval_samples': args.num_eval_samples,
            'seed': 42,
            'device': device
        }
    )

    # Create multi-layer trainer
    logger.info("Initializing multi-layer NPT trainer...")
    trainer = MultiLayerNPTTrainer(
        model=model,
        config=training_config,
        train_loader=train_loader,
        val_loader=val_loader,
        layers_to_train=layers_to_train,
        curriculum_schedule=curriculum_schedule,
        gradient_scale_factor=args.gradient_scale_factor,
        loss_config=loss_config,
        tracker=tracker,
        tokenizer=tokenizer,
        evaluator=evaluator,
        layer_weights=layer_weights
    )

    # Start training
    logger.info("\n" + "=" * 80)
    logger.info("Starting Multi-Layer NPT Training")
    logger.info("=" * 80)
    logger.info(f"Training Configuration:")
    logger.info(f"  Model: {args.model_name} ({args.model_size})")
    logger.info(f"  NPT Layers: {layers_to_train}")
    logger.info(f"  NPT Rank: {args.np_rank}")
    logger.info(f"  Num Ranks (rank-k): {args.num_ranks}")
    logger.info(f"  Total Steps: {args.max_steps}")
    logger.info(f"  Curriculum: {[s.name for s in curriculum_schedule]}")
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
                "layers_trained": layers_to_train,
                "num_layers": len(layers_to_train),
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