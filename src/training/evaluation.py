"""
Fixed evaluation module for NPT training.

Provides consistent evaluation across experiments for reliable comparison.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
from datasets import load_dataset

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    loss: float
    perplexity: float
    npt_loss: Optional[float] = None
    original_loss: Optional[float] = None
    direct_mlp_loss: Optional[float] = None
    fidelity_loss: Optional[float] = None
    regularization_loss: Optional[float] = None
    num_samples: int = 0
    timestamp: Optional[str] = None


class FixedEvaluator:
    """
    Fixed evaluation for consistent model comparison across experiments.

    Features:
    - Deterministic evaluation dataset
    - Consistent metrics computation
    - Efficient caching
    - Support for single-layer and full model evaluation
    """

    def __init__(
        self,
        tokenizer,
        max_length: int = 512,
        num_eval_samples: int = 1000,
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-2-raw-v1",
        split: str = "validation",
        seed: int = 42,
        cache_dir: Optional[str] = None,
        device: str = "cuda"
    ):
        """
        Initialize fixed evaluator.

        Args:
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
            num_eval_samples: Number of samples for evaluation
            dataset_name: HuggingFace dataset name
            dataset_config: Dataset configuration
            split: Dataset split to use
            seed: Random seed for reproducibility
            cache_dir: Directory to cache processed data
            device: Device for computation
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_eval_samples = num_eval_samples
        self.seed = seed
        self.device = device

        # Setup cache directory
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "npt_evaluation"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load and prepare fixed evaluation data
        self.eval_data = self._prepare_evaluation_data(
            dataset_name, dataset_config, split
        )

        logger.info(f"Initialized fixed evaluator with {len(self.eval_data)} samples")

    def _prepare_evaluation_data(
        self,
        dataset_name: str,
        dataset_config: str,
        split: str
    ) -> List[torch.Tensor]:
        """
        Prepare fixed evaluation dataset.

        Returns tokenized and cached evaluation samples.
        """
        # Check cache first
        cache_file = self.cache_dir / f"eval_data_{dataset_name}_{dataset_config}_{split}_{self.num_eval_samples}_{self.seed}.pt"

        if cache_file.exists():
            logger.info(f"Loading cached evaluation data from {cache_file}")
            return torch.load(cache_file)

        logger.info(f"Preparing evaluation data from {dataset_name}/{dataset_config}")

        # Load dataset
        dataset = load_dataset(dataset_name, dataset_config, split=split)

        # Set seed for reproducibility
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Sample fixed indices
        total_samples = len(dataset)
        sample_indices = np.random.choice(
            total_samples,
            min(self.num_eval_samples, total_samples),
            replace=False
        )
        sample_indices.sort()  # Sort for consistent ordering

        # Tokenize samples
        tokenized_samples = []
        for idx in sample_indices:
            text = dataset[int(idx)]['text']
            if not text or len(text.strip()) == 0:
                continue

            # Tokenize
            tokens = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt'
            )

            input_ids = tokens['input_ids'].squeeze(0)

            # Skip if too short
            if input_ids.shape[0] < 10:
                continue

            tokenized_samples.append(input_ids)

            if len(tokenized_samples) >= self.num_eval_samples:
                break

        # Save to cache
        torch.save(tokenized_samples, cache_file)
        logger.info(f"Cached {len(tokenized_samples)} evaluation samples to {cache_file}")

        return tokenized_samples

    def evaluate_model(
        self,
        model,
        batch_size: int = 8,
        max_batches: Optional[int] = None
    ) -> EvaluationMetrics:
        """
        Evaluate model on fixed dataset.

        Args:
            model: Model to evaluate
            batch_size: Batch size for evaluation
            max_batches: Maximum number of batches to evaluate

        Returns:
            EvaluationMetrics with computed metrics
        """
        model.eval()

        total_loss = 0.0
        total_tokens = 0
        num_samples = 0

        with torch.no_grad():
            for batch_idx in range(0, len(self.eval_data), batch_size):
                if max_batches and batch_idx // batch_size >= max_batches:
                    break

                # Get batch
                batch_samples = self.eval_data[batch_idx:batch_idx + batch_size]
                if not batch_samples:
                    continue

                # Stack into batch
                input_ids = torch.stack(batch_samples).to(self.device)

                # Forward pass
                outputs = model(input_ids=input_ids)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs

                # Compute loss
                # Shift for next token prediction
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()

                # Flatten for loss computation
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=self.tokenizer.pad_token_id
                )

                # Accumulate
                batch_tokens = (shift_labels != self.tokenizer.pad_token_id).sum().item()
                total_loss += loss.item() * batch_tokens
                total_tokens += batch_tokens
                num_samples += input_ids.shape[0]

        # Compute metrics
        avg_loss = total_loss / max(1, total_tokens)
        perplexity = np.exp(avg_loss)

        metrics = EvaluationMetrics(
            loss=avg_loss,
            perplexity=perplexity,
            num_samples=num_samples,
            timestamp=datetime.now().isoformat()
        )

        model.train()
        return metrics

    def evaluate_single_layer_npt(
        self,
        model,
        layer_idx: int,
        loss_fn,
        batch_size: int = 4,
        max_batches: Optional[int] = None
    ) -> EvaluationMetrics:
        """
        Evaluate single-layer NPT with specialized metrics.

        Args:
            model: NPT model
            layer_idx: Index of NPT layer to evaluate
            loss_fn: Loss function (DirectSupervisionLoss)
            batch_size: Batch size for evaluation
            max_batches: Maximum number of batches

        Returns:
            EvaluationMetrics with single-layer specific metrics
        """
        model.eval()

        total_loss = 0.0
        total_direct_mlp = 0.0
        total_fidelity = 0.0
        total_regularization = 0.0
        total_tokens = 0
        num_samples = 0

        with torch.no_grad():
            for batch_idx in range(0, len(self.eval_data), batch_size):
                if max_batches and batch_idx // batch_size >= max_batches:
                    break

                # Get batch
                batch_samples = self.eval_data[batch_idx:batch_idx + batch_size]
                if not batch_samples:
                    continue

                # Stack into batch
                input_ids = torch.stack(batch_samples).to(self.device)

                # Collect single-layer outputs
                outputs = self._collect_single_layer_outputs(
                    model, input_ids, layer_idx
                )

                # Compute loss
                loss_output = loss_fn(outputs)

                # Accumulate metrics
                batch_tokens = input_ids.shape[0] * input_ids.shape[1]
                total_loss += loss_output.total_loss.item() * batch_tokens
                total_direct_mlp += loss_output.metrics['direct_mlp_loss'] * batch_tokens
                total_fidelity += loss_output.metrics['fidelity_loss'] * batch_tokens
                total_regularization += loss_output.metrics['regularization_loss'] * batch_tokens
                total_tokens += batch_tokens
                num_samples += input_ids.shape[0]

        # Compute averages
        avg_loss = total_loss / max(1, total_tokens)
        avg_direct_mlp = total_direct_mlp / max(1, total_tokens)
        avg_fidelity = total_fidelity / max(1, total_tokens)
        avg_regularization = total_regularization / max(1, total_tokens)

        metrics = EvaluationMetrics(
            loss=avg_loss,
            perplexity=np.exp(avg_loss),
            direct_mlp_loss=avg_direct_mlp,
            fidelity_loss=avg_fidelity,
            regularization_loss=avg_regularization,
            num_samples=num_samples,
            timestamp=datetime.now().isoformat()
        )

        model.train()
        return metrics

    def _collect_single_layer_outputs(
        self,
        model,
        input_ids: torch.Tensor,
        layer_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Collect outputs for single-layer evaluation.

        This mirrors the collection in SingleLayerNPTTrainer.
        """
        batch_size, seq_len = input_ids.shape

        # Run in standard mode first
        model.set_npt_mode(False)
        with torch.no_grad():
            # Embed inputs
            hidden_states = model.model.embed_tokens(input_ids)

            # Create position embeddings
            head_dim = model.config.hidden_size // model.config.num_attention_heads
            cos = torch.ones(batch_size, seq_len, head_dim,
                           dtype=hidden_states.dtype, device=hidden_states.device)
            sin = torch.zeros(batch_size, seq_len, head_dim,
                            dtype=hidden_states.dtype, device=hidden_states.device)
            position_embeddings = (cos, sin)

            # Process through layers up to NPT layer
            for i in range(layer_idx):
                layer = model.model.layers[i]
                layer_out = layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    use_cache=False,
                    output_attentions=False
                )
                hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out

            # Process NPT layer in standard mode
            npt_layer = model.model.layers[layer_idx]

            # Get attention output
            residual = hidden_states
            hidden_states = npt_layer.input_layernorm(hidden_states)
            attn_outputs = npt_layer.self_attn(
                hidden_states=hidden_states,
                attention_mask=None,
                position_embeddings=position_embeddings,
                past_key_values=None,
                cache_position=None,
                use_cache=False,
                output_attentions=False
            )
            attention_output = attn_outputs[0]

            # Standard processing
            hidden_after_attn = residual + attention_output
            mlp_input_with_attn = npt_layer.post_attention_layernorm(hidden_after_attn)
            original_mlp_with_attention = npt_layer.mlp(mlp_input_with_attn)

            # Continue for final output
            hidden_states = hidden_after_attn + original_mlp_with_attention
            for i in range(layer_idx + 1, len(model.model.layers)):
                layer = model.model.layers[i]
                layer_out = layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    use_cache=False,
                    output_attentions=False
                )
                hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out

            hidden_states = model.model.norm(hidden_states)
            original_final = model.lm_head(hidden_states)

        # Now run in NPT mode
        model.set_npt_mode(True)

        # Process up to NPT layer
        hidden_states = model.model.embed_tokens(input_ids)
        for i in range(layer_idx):
            layer = model.model.layers[i]
            layer_out = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                use_cache=False,
                output_attentions=False
            )
            hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        # Process NPT layer
        npt_layer = model.model.layers[layer_idx]

        # Hook to collect v_a, v_b
        v_a, v_b = None, None
        def hook_fn(module, input, output):
            nonlocal v_a, v_b
            if isinstance(output, tuple) and len(output) == 2:
                v_a, v_b = output

        handle = npt_layer.np_component.register_forward_hook(hook_fn)

        # NPT forward
        residual = hidden_states
        hidden_states = npt_layer.input_layernorm(hidden_states)
        attn_outputs = npt_layer.self_attn(
            hidden_states=hidden_states,
            attention_mask=None,
            position_embeddings=position_embeddings,
            past_key_values=None,
            cache_position=None,
            use_cache=False,
            output_attentions=False
        )
        attn_out = attn_outputs[0]

        # Get v_a, v_b
        v_a_temp, v_b_temp = npt_layer.np_component(attn_out)

        handle.remove()

        # Get modulated MLP output
        mlp_input = npt_layer.post_attention_layernorm(residual)
        mlp_modulated = npt_layer._apply_modulated_mlp_efficient(mlp_input, v_a, v_b)

        # Continue through remaining layers
        hidden_states = residual + mlp_modulated
        for i in range(layer_idx + 1, len(model.model.layers)):
            layer = model.model.layers[i]
            layer_out = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                use_cache=False,
                output_attentions=False
            )
            hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        hidden_states = model.model.norm(hidden_states)
        npt_final = model.lm_head(hidden_states)

        return {
            'mlp_modulated': mlp_modulated,
            'attention_output': attention_output,
            'original_mlp_with_attention': original_mlp_with_attention,
            'v_a': v_a,
            'v_b': v_b,
            'npt_final': npt_final,
            'original_final': original_final,
            'hidden_states': mlp_input
        }

    def compare_npt_vs_original(
        self,
        model,
        batch_size: int = 8,
        max_batches: Optional[int] = None
    ) -> Dict[str, EvaluationMetrics]:
        """
        Compare NPT and original model performance.

        Returns metrics for both NPT and original modes.
        """
        # Evaluate in NPT mode
        model.set_npt_mode(True)
        npt_metrics = self.evaluate_model(model, batch_size, max_batches)

        # Evaluate in original mode
        model.set_npt_mode(False)
        original_metrics = self.evaluate_model(model, batch_size, max_batches)

        # Restore NPT mode
        model.set_npt_mode(True)

        return {
            'npt': npt_metrics,
            'original': original_metrics
        }

    def save_metrics(
        self,
        metrics: EvaluationMetrics,
        save_path: Path,
        experiment_name: str,
        step: int
    ):
        """Save evaluation metrics to JSON file."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        metrics_file = save_path / f"eval_metrics_{experiment_name}.json"

        # Load existing metrics if file exists
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                all_metrics = json.load(f)
        else:
            all_metrics = []

        # Add new metrics
        metrics_dict = {
            'step': step,
            'loss': metrics.loss,
            'perplexity': metrics.perplexity,
            'num_samples': metrics.num_samples,
            'timestamp': metrics.timestamp
        }

        # Add optional metrics
        if metrics.direct_mlp_loss is not None:
            metrics_dict['direct_mlp_loss'] = metrics.direct_mlp_loss
        if metrics.fidelity_loss is not None:
            metrics_dict['fidelity_loss'] = metrics.fidelity_loss
        if metrics.regularization_loss is not None:
            metrics_dict['regularization_loss'] = metrics.regularization_loss

        all_metrics.append(metrics_dict)

        # Save
        with open(metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)

        logger.info(f"Saved evaluation metrics to {metrics_file}")


def create_fixed_evaluator(
    tokenizer,
    config: Optional[Dict] = None
) -> FixedEvaluator:
    """
    Factory function to create fixed evaluator with default config.

    Args:
        tokenizer: Tokenizer for text processing
        config: Optional configuration dict

    Returns:
        FixedEvaluator instance
    """
    default_config = {
        'max_length': 512,
        'num_eval_samples': 1000,
        'dataset_name': 'wikitext',
        'dataset_config': 'wikitext-2-raw-v1',
        'split': 'validation',
        'seed': 42,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    if config:
        default_config.update(config)

    return FixedEvaluator(tokenizer=tokenizer, **default_config)