"""
Trainer for NPT equivalence pre-training.

This module implements the training loop for teaching NP components
to functionally mimic the original transformer residual connections.
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from pathlib import Path
import json
import time
from dataclasses import dataclass, asdict
import numpy as np
from tqdm import tqdm

from ..training.losses import EquivalenceLoss, ParallelForwardHelper


@dataclass
class TrainingConfig:
    """Configuration for NPT training."""
    # Model configuration
    model_name: str = "meta-llama/Llama-3.2-1B"
    npt_layers: Optional[List[int]] = None  # None means convert upper half
    np_rank: int = 64
    np_init_scale: float = 0.01
    
    # Training configuration
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    lambda_reg: float = 0.01
    max_steps: int = 10000
    warmup_steps: int = 500
    
    # Data configuration
    max_length: int = 512
    stride: int = 256
    
    # Logging configuration
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    generation_steps: int = 150
    
    # Checkpoint configuration
    output_dir: str = "experiments/npt_training"
    checkpoint_dir: str = "checkpoints"
    resume_from_checkpoint: Optional[str] = None
    
    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = False
    gradient_accumulation_steps: int = 1
    gradient_clip_value: Optional[float] = 1.0
    
    def save(self, path: Path):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: Path):
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    step: int
    epoch: float
    total_loss: float
    fidelity_loss: float
    regularization_loss: float
    learning_rate: float
    grad_norm: float
    avg_v_a_norm: float
    avg_v_b_norm: float
    time_per_step: float


class NPTTrainer:
    """
    Trainer for NPT equivalence pre-training.
    
    Handles training loop, validation, checkpointing, and metrics tracking.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        wandb_run: Optional[Any] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: NPTLlamaModel to train
            config: Training configuration
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            optimizer: Optimizer (will create Adam if None)
            scheduler: Learning rate scheduler (optional)
            wandb_run: Weights & Biases run object for logging
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.wandb_run = wandb_run
        
        # Move model to device
        self.model = self.model.to(config.device)
        
        # Setup loss function and helper
        self.loss_fn = EquivalenceLoss(lambda_reg=config.lambda_reg)
        self.helper = ParallelForwardHelper(self.model)
        
        # Setup optimizer
        if optimizer is None:
            self.optimizer = self._create_optimizer()
        else:
            self.optimizer = optimizer
        
        # Setup scheduler
        self.scheduler = scheduler
        if self.scheduler is None and config.warmup_steps > 0:
            self.scheduler = self._create_scheduler()
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
        # Setup directories
        self.output_dir = Path(config.output_dir)
        self.checkpoint_dir = self.output_dir / config.checkpoint_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config.save(self.output_dir / "config.json")
        
        # Setup mixed precision if requested
        self.scaler = None
        if config.mixed_precision and config.device == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Load checkpoint if specified
        if config.resume_from_checkpoint:
            self.load_checkpoint(config.resume_from_checkpoint)
    
    def _create_optimizer(self) -> Optimizer:
        """Create optimizer for NP parameters only."""
        # Get NP parameters
        npt_params = self.model.get_npt_parameters()
        
        # Create Adam optimizer
        optimizer = torch.optim.AdamW(
            npt_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        return optimizer
    
    def _create_scheduler(self) -> _LRScheduler:
        """Create learning rate scheduler with warmup."""
        from torch.optim.lr_scheduler import LambdaLR
        
        def lr_lambda(step: int) -> float:
            if step < self.config.warmup_steps:
                # Linear warmup
                return float(step) / float(max(1, self.config.warmup_steps))
            else:
                # Cosine decay
                progress = float(step - self.config.warmup_steps) / float(
                    max(1, self.config.max_steps - self.config.warmup_steps)
                )
                return 0.5 * (1.0 + np.cos(np.pi * progress))
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> TrainingMetrics:
        """
        Perform one training step.
        
        Args:
            batch: Batch of data
        
        Returns:
            Training metrics for this step
        """
        self.model.train()
        start_time = time.time()
        
        # Move batch to device
        batch = {k: v.to(self.config.device) for k, v in batch.items()}
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass with mixed precision if enabled
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                npt_output, original_output, v_a_list, v_b_list = self.helper.forward(
                    batch['input_ids'],
                    attention_mask=batch.get('attention_mask'),
                    collect_np_outputs=True
                )
                loss_output = self.loss_fn(npt_output, original_output, v_a_list, v_b_list)
        else:
            npt_output, original_output, v_a_list, v_b_list = self.helper.forward(
                batch['input_ids'],
                attention_mask=batch.get('attention_mask'),
                collect_np_outputs=True
            )
            loss_output = self.loss_fn(npt_output, original_output, v_a_list, v_b_list)
        
        # Scale loss for gradient accumulation
        loss = loss_output.total_loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient clipping and optimizer step
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            # Unscale gradients if using mixed precision
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            
            # Compute gradient norm
            grad_norm = self._compute_grad_norm()
            
            # Clip gradients
            if self.config.gradient_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.get_npt_parameters(),
                    self.config.gradient_clip_value
                )
            
            # Optimizer step
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()
        else:
            grad_norm = 0.0
        
        # Compute metrics
        time_per_step = time.time() - start_time
        current_lr = self.optimizer.param_groups[0]['lr']
        
        metrics = TrainingMetrics(
            step=self.global_step,
            epoch=self.current_epoch,
            total_loss=loss_output.total_loss.item(),
            fidelity_loss=loss_output.fidelity_loss.item(),
            regularization_loss=loss_output.regularization_loss.item(),
            learning_rate=current_lr,
            grad_norm=grad_norm,
            avg_v_a_norm=loss_output.metrics['avg_v_a_norm'],
            avg_v_b_norm=loss_output.metrics['avg_v_b_norm'],
            time_per_step=time_per_step
        )
        
        return metrics
    
    def _compute_grad_norm(self) -> float:
        """Compute gradient norm for NP parameters."""
        total_norm = 0.0
        for param in self.model.get_npt_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm
    
    @torch.no_grad()
    def evaluate(self, eval_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """
        Evaluate model on validation set.
        
        Args:
            eval_loader: Evaluation data loader (uses val_loader if None)
        
        Returns:
            Dictionary of evaluation metrics
        """
        if eval_loader is None:
            eval_loader = self.val_loader
        
        if eval_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        total_fidelity = 0.0
        total_regularization = 0.0
        total_batches = 0
        
        for batch in tqdm(eval_loader, desc="Evaluating", leave=False):
            # Move batch to device
            batch = {k: v.to(self.config.device) for k, v in batch.items()}
            
            # Forward pass
            npt_output, original_output, v_a_list, v_b_list = self.helper.forward(
                batch['input_ids'],
                attention_mask=batch.get('attention_mask'),
                collect_np_outputs=True
            )
            
            # Compute loss
            loss_output = self.loss_fn(npt_output, original_output, v_a_list, v_b_list)
            
            # Accumulate metrics
            total_loss += loss_output.total_loss.item()
            total_fidelity += loss_output.fidelity_loss.item()
            total_regularization += loss_output.regularization_loss.item()
            total_batches += 1
        
        # Compute averages
        metrics = {
            'val_loss': total_loss / total_batches,
            'val_fidelity_loss': total_fidelity / total_batches,
            'val_regularization_loss': total_regularization / total_batches
        }
        
        return metrics
    
    @torch.no_grad()
    def generate_samples(self, prompts: List[str], tokenizer: Any, max_length: int = 50) -> List[str]:
        """
        Generate text samples for quality assessment.
        
        Args:
            prompts: List of prompt strings
            tokenizer: Tokenizer for encoding/decoding
            max_length: Maximum generation length
        
        Returns:
            List of generated texts
        """
        self.model.eval()
        generated_texts = []
        
        for prompt in prompts:
            # Encode prompt
            inputs = tokenizer(prompt, return_tensors='pt').to(self.config.device)
            
            # Generate in NPT mode
            self.model.set_npt_mode(True)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.95
                )
            
            # Decode
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_texts.append(generated)
        
        return generated_texts
    
    def save_checkpoint(self, checkpoint_name: Optional[str] = None):
        """
        Save training checkpoint.
        
        Args:
            checkpoint_name: Name for checkpoint (uses step if None)
        """
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint-{self.global_step}"
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save model NPT weights
        self.model.save_npt_weights(checkpoint_path / "npt_weights.pt")
        
        # Save training state
        training_state = {
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'config': asdict(self.config)
        }
        torch.save(training_state, checkpoint_path / "training_state.pt")
        
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
        """
        checkpoint_path = Path(checkpoint_path)
        
        # Load model NPT weights
        self.model.load_npt_weights(checkpoint_path / "npt_weights.pt")
        
        # Load training state
        training_state = torch.load(checkpoint_path / "training_state.pt")
        
        self.global_step = training_state['global_step']
        self.current_epoch = training_state['current_epoch']
        self.best_val_loss = training_state['best_val_loss']
        
        self.optimizer.load_state_dict(training_state['optimizer_state_dict'])
        
        if self.scheduler and training_state['scheduler_state_dict']:
            self.scheduler.load_state_dict(training_state['scheduler_state_dict'])
        
        if self.scaler and training_state['scaler_state_dict']:
            self.scaler.load_state_dict(training_state['scaler_state_dict'])
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Resuming from step {self.global_step}")
    
    def train(self):
        """
        Main training loop.
        
        Trains until max_steps is reached.
        """
        print(f"Starting training for {self.config.max_steps} steps")
        # Handle streaming datasets that don't have a defined length
        try:
            batch_count = len(self.train_loader)
            print(f"Training on {batch_count} batches per epoch")
        except TypeError:
            print(f"Training with streaming data (unlimited batches)")
        
        # Create progress bar
        pbar = tqdm(total=self.config.max_steps, initial=self.global_step)
        
        # Training loop
        while self.global_step < self.config.max_steps:
            for batch in self.train_loader:
                # Train step
                metrics = self.train_step(batch)
                
                # Update progress
                self.global_step += 1
                pbar.update(1)
                
                # Log metrics
                if self.global_step % self.config.logging_steps == 0:
                    self._log_metrics(metrics)
                    pbar.set_postfix({
                        'loss': f"{metrics.total_loss:.4f}",
                        'fidelity': f"{metrics.fidelity_loss:.4f}",
                        'lr': f"{metrics.learning_rate:.2e}"
                    })
                
                # Evaluate
                if self.config.eval_steps > 0 and self.global_step % self.config.eval_steps == 0:
                    eval_metrics = self.evaluate()
                    self._log_metrics(eval_metrics)
                    
                    # Save best model
                    if eval_metrics.get('val_loss', float('inf')) < self.best_val_loss:
                        self.best_val_loss = eval_metrics['val_loss']
                        self.save_checkpoint('best')
                
                # Save checkpoint
                if self.config.save_steps > 0 and self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()
                
                # Check if done
                if self.global_step >= self.config.max_steps:
                    break
            
            # Update epoch
            self.current_epoch += 1
        
        pbar.close()
        
        # Final evaluation
        print("Training complete! Running final evaluation...")
        final_metrics = self.evaluate()
        self._log_metrics(final_metrics)
        
        # Save final checkpoint
        self.save_checkpoint('final')
        
        print(f"Training finished at step {self.global_step}")
        print(f"Final validation loss: {final_metrics.get('val_loss', 'N/A')}")
    
    def _log_metrics(self, metrics: Union[TrainingMetrics, Dict[str, float]]):
        """Log metrics to console and wandb."""
        # Convert TrainingMetrics to dict if needed
        if isinstance(metrics, TrainingMetrics):
            metrics = asdict(metrics)
        
        # Log to wandb if available
        if self.wandb_run is not None:
            self.wandb_run.log(metrics, step=self.global_step)
        
        # Log to file
        log_file = self.output_dir / "training_log.jsonl"
        with open(log_file, 'a') as f:
            metrics_with_step = {'step': self.global_step, **metrics}
            f.write(json.dumps(metrics_with_step) + '\n')