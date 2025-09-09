"""
Weights & Biases integration for NPT training.

This module provides comprehensive experiment tracking, visualization,
and artifact management for NPT training runs.
"""

import wandb
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json
from dataclasses import asdict
import logging

logger = logging.getLogger(__name__)


class WandBTracker:
    """
    Comprehensive WandB tracker for NPT experiments.
    
    Features:
    - Automatic metric logging
    - Model architecture tracking
    - Gradient and weight histograms
    - Sample generation logging
    - Checkpoint artifact management
    - Custom NPT-specific visualizations
    """
    
    def __init__(
        self,
        project: str = "npt-training",
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        resume: Optional[str] = None,
        mode: str = "online"
    ):
        """
        Initialize WandB tracker.
        
        Args:
            project: WandB project name
            name: Run name
            config: Configuration dictionary
            tags: List of tags for the run
            notes: Notes about the run
            resume: Resume from previous run ID
            mode: WandB mode ('online', 'offline', 'disabled')
        """
        self.project = project
        self.name = name
        self.config = config or {}
        self.tags = tags or []
        self.notes = notes
        self.resume = resume
        self.mode = mode
        
        self.run = None
        self.step = 0
        
    def init(self, model=None):
        """
        Initialize WandB run.
        
        Args:
            model: Optional model to log architecture
        """
        try:
            # Initialize run
            self.run = wandb.init(
                project=self.project,
                name=self.name,
                config=self.config,
                tags=self.tags,
                notes=self.notes,
                resume=self.resume,
                mode=self.mode
            )
            
            # Log model architecture if provided
            if model is not None:
                self.log_model_architecture(model)
            
            logger.info(f"WandB run initialized: {self.run.name} (ID: {self.run.id})")
            
        except Exception as e:
            logger.warning(f"Failed to initialize WandB: {e}")
            self.run = None
    
    def log_model_architecture(self, model):
        """
        Log model architecture and NPT configuration.
        
        Args:
            model: NPTLlamaModel instance
        """
        if not self.run:
            return
        
        try:
            # Log model summary
            param_counts = model.count_parameters()
            
            architecture_info = {
                "model_type": model.__class__.__name__,
                "total_parameters": param_counts['total'],
                "base_parameters": param_counts['base'],
                "npt_parameters": param_counts['npt'],
                "npt_ratio": param_counts['npt_ratio'],
                "num_layers": model.config.num_hidden_layers,
                "hidden_size": model.config.hidden_size,
                "intermediate_size": model.config.intermediate_size,
                "num_attention_heads": model.config.num_attention_heads,
                "vocab_size": model.config.vocab_size,
            }
            
            # Log NPT layer information
            layer_info = model.get_layer_info()
            architecture_info.update({
                "npt_layers": layer_info['npt_layers'],
                "npt_layer_indices": layer_info['npt_layer_indices'],
                "standard_layer_indices": layer_info['standard_layer_indices'],
            })
            
            # Log as config
            wandb.config.update({"architecture": architecture_info})
            
            # Create architecture table
            layer_data = []
            for i, layer_type in enumerate(layer_info['layer_types']):
                layer_data.append([i, layer_type])
            
            layer_table = wandb.Table(
                columns=["Layer Index", "Type"],
                data=layer_data
            )
            wandb.log({"model/layer_architecture": layer_table})
            
        except Exception as e:
            logger.warning(f"Failed to log model architecture: {e}")
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log training metrics.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Training step (uses internal counter if None)
        """
        if not self.run:
            return
        
        if step is None:
            step = self.step
            self.step += 1
        
        try:
            # Organize metrics by category
            organized_metrics = {}
            
            for key, value in metrics.items():
                if 'loss' in key.lower():
                    organized_metrics[f"loss/{key}"] = value
                elif 'lr' in key.lower() or 'learning_rate' in key.lower():
                    organized_metrics[f"learning/{key}"] = value
                elif 'grad' in key.lower():
                    organized_metrics[f"gradients/{key}"] = value
                elif 'norm' in key.lower():
                    organized_metrics[f"norms/{key}"] = value
                elif 'val' in key.lower():
                    organized_metrics[f"validation/{key}"] = value
                else:
                    organized_metrics[f"train/{key}"] = value
            
            # Log to wandb
            wandb.log(organized_metrics, step=step)
            
        except Exception as e:
            logger.warning(f"Failed to log metrics: {e}")
    
    def log_gradients(self, model, step: Optional[int] = None):
        """
        Log gradient statistics and histograms.
        
        Args:
            model: Model with gradients
            step: Training step
        """
        if not self.run:
            return
        
        if step is None:
            step = self.step
        
        try:
            grad_stats = {
                'min': [],
                'max': [],
                'mean': [],
                'std': []
            }
            
            # Collect gradient statistics
            for name, param in model.named_parameters():
                if param.grad is not None and 'np_component' in name:
                    grad = param.grad.data.cpu().numpy().flatten()
                    grad_stats['min'].append(np.min(grad))
                    grad_stats['max'].append(np.max(grad))
                    grad_stats['mean'].append(np.mean(grad))
                    grad_stats['std'].append(np.std(grad))
            
            # Log aggregate statistics
            if grad_stats['mean']:
                wandb.log({
                    "gradients/np_min": np.mean(grad_stats['min']),
                    "gradients/np_max": np.mean(grad_stats['max']),
                    "gradients/np_mean": np.mean(grad_stats['mean']),
                    "gradients/np_std": np.mean(grad_stats['std']),
                }, step=step)
                
                # Log histogram of all gradient values
                all_grads = []
                for name, param in model.named_parameters():
                    if param.grad is not None and 'np_component' in name:
                        all_grads.extend(param.grad.data.cpu().numpy().flatten().tolist())
                
                if all_grads:
                    wandb.log({
                        "gradients/np_histogram": wandb.Histogram(all_grads)
                    }, step=step)
            
        except Exception as e:
            logger.warning(f"Failed to log gradients: {e}")
    
    def log_weights(self, model, step: Optional[int] = None):
        """
        Log weight statistics and histograms for NP components.
        
        Args:
            model: NPTLlamaModel
            step: Training step
        """
        if not self.run:
            return
        
        if step is None:
            step = self.step
        
        try:
            # Collect NP component weights
            np_weights = []
            for name, param in model.named_parameters():
                if 'np_component' in name:
                    np_weights.extend(param.data.cpu().numpy().flatten().tolist())
            
            if np_weights:
                wandb.log({
                    "weights/np_histogram": wandb.Histogram(np_weights),
                    "weights/np_mean": np.mean(np_weights),
                    "weights/np_std": np.std(np_weights),
                }, step=step)
            
        except Exception as e:
            logger.warning(f"Failed to log weights: {e}")
    
    def log_sample_outputs(
        self,
        model,
        tokenizer,
        prompts: List[str],
        step: Optional[int] = None,
        max_length: int = 100
    ):
        """
        Log sample generations for quality assessment.
        
        Args:
            model: Model for generation
            tokenizer: Tokenizer
            prompts: List of prompts
            step: Training step
            max_length: Maximum generation length
        """
        if not self.run:
            return
        
        if step is None:
            step = self.step
        
        try:
            model.eval()
            generations = []
            
            with torch.no_grad():
                for prompt in prompts:
                    # Tokenize prompt
                    inputs = tokenizer(
                        prompt,
                        return_tensors='pt',
                        truncation=True,
                        max_length=512
                    ).to(model.device)
                    
                    # Generate in NPT mode
                    model.set_npt_mode(True)
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_length,
                        temperature=0.8,
                        do_sample=True,
                        top_p=0.95,
                        pad_token_id=tokenizer.pad_token_id
                    )
                    
                    # Decode
                    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    generations.append([prompt, generated])
            
            # Log as table
            generation_table = wandb.Table(
                columns=["Prompt", "Generation"],
                data=generations
            )
            wandb.log({"samples/generations": generation_table}, step=step)
            
        except Exception as e:
            logger.warning(f"Failed to log sample outputs: {e}")
    
    def log_npt_dynamics(
        self,
        v_a_norms: List[float],
        v_b_norms: List[float],
        delta_w_norms: Optional[List[float]] = None,
        step: Optional[int] = None
    ):
        """
        Log NPT-specific dynamics (vector norms, weight update magnitudes).
        
        Args:
            v_a_norms: Norms of v_a vectors per layer
            v_b_norms: Norms of v_b vectors per layer
            delta_w_norms: Norms of delta W matrices (optional)
            step: Training step
        """
        if not self.run:
            return
        
        if step is None:
            step = self.step
        
        try:
            metrics = {
                "npt/avg_v_a_norm": np.mean(v_a_norms),
                "npt/avg_v_b_norm": np.mean(v_b_norms),
                "npt/max_v_a_norm": np.max(v_a_norms),
                "npt/max_v_b_norm": np.max(v_b_norms),
            }
            
            if delta_w_norms:
                metrics.update({
                    "npt/avg_delta_w_norm": np.mean(delta_w_norms),
                    "npt/max_delta_w_norm": np.max(delta_w_norms),
                })
            
            wandb.log(metrics, step=step)
            
            # Log per-layer norms as line plot
            layer_data = [[i, v_a, v_b] for i, (v_a, v_b) in enumerate(zip(v_a_norms, v_b_norms))]
            norm_table = wandb.Table(
                columns=["Layer", "v_a_norm", "v_b_norm"],
                data=layer_data
            )
            wandb.log({"npt/layer_norms": norm_table}, step=step)
            
        except Exception as e:
            logger.warning(f"Failed to log NPT dynamics: {e}")
    
    def save_checkpoint(self, checkpoint_path: Path, aliases: Optional[List[str]] = None):
        """
        Save checkpoint as WandB artifact.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            aliases: List of aliases for the artifact
        """
        if not self.run:
            return
        
        try:
            artifact = wandb.Artifact(
                name=f"checkpoint-{self.step}",
                type="model",
                metadata={"step": self.step}
            )
            
            # Add all files in checkpoint directory
            artifact.add_dir(str(checkpoint_path))
            
            # Log artifact with aliases
            if aliases:
                self.run.log_artifact(artifact, aliases=aliases)
            else:
                self.run.log_artifact(artifact)
            
            logger.info(f"Checkpoint saved as artifact: {artifact.name}")
            
        except Exception as e:
            logger.warning(f"Failed to save checkpoint artifact: {e}")
    
    def log_training_summary(self, summary: Dict[str, Any]):
        """
        Log final training summary.
        
        Args:
            summary: Dictionary of summary statistics
        """
        if not self.run:
            return
        
        try:
            # Log as summary metrics
            for key, value in summary.items():
                wandb.run.summary[key] = value
            
            # Create summary table
            summary_data = [[k, v] for k, v in summary.items()]
            summary_table = wandb.Table(
                columns=["Metric", "Value"],
                data=summary_data
            )
            wandb.log({"training/final_summary": summary_table})
            
        except Exception as e:
            logger.warning(f"Failed to log training summary: {e}")
    
    def finish(self):
        """Finish the WandB run."""
        if self.run:
            try:
                self.run.finish()
                logger.info("WandB run finished successfully")
            except Exception as e:
                logger.warning(f"Failed to finish WandB run: {e}")


def setup_wandb_tracking(
    project: str = "npt-training",
    config: Dict[str, Any] = None,
    tags: Optional[List[str]] = None,
    name: Optional[str] = None,
    notes: Optional[str] = None,
    mode: str = "online"
) -> WandBTracker:
    """
    Setup WandB tracking for NPT training.
    
    Args:
        project: WandB project name
        config: Training configuration
        tags: Tags for the run
        name: Run name
        notes: Notes about the run
        mode: WandB mode
    
    Returns:
        Initialized WandBTracker
    """
    # Add NPT-specific tags
    if tags is None:
        tags = []
    tags.extend(["npt", "equivalence-training"])
    
    # Create tracker
    tracker = WandBTracker(
        project=project,
        name=name,
        config=config,
        tags=tags,
        notes=notes,
        mode=mode
    )
    
    return tracker