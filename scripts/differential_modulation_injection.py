#!/usr/bin/env python3
"""
Differential Modulation Injection Experiment

This experiment:
1. Extracts modulations with and without false context
2. Computes the differential (context - baseline)
3. Injects the differential directly into MLP weights
4. Tests if the false fact has been permanently learned
"""

import argparse
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import copy

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.npt import NPTLlamaModel, NPTConfig
from transformers import AutoTokenizer, AutoConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for differential injection experiment."""
    checkpoint_path: str
    model_name: str = "meta-llama/Llama-3.2-1B"
    target_layers: List[int] = None  # e.g., [11, 12]
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    injection_scaling: float = 0.1  # Start conservative
    seed: int = 42
    output_dir: str = "experiments/differential_injection"


class DifferentialModulationInjector:
    """Handles differential modulation extraction and injection."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model = None
        self.tokenizer = None
        self.original_weights = {}  # Store original MLP weights

    def load_model(self) -> Tuple[NPTLlamaModel, AutoTokenizer]:
        """Load NPT model with only specified layers."""
        logger.info(f"Loading model from {self.config.checkpoint_path}")

        # Load base model configuration
        model_config = AutoConfig.from_pretrained(self.config.model_name)
        model_config._attn_implementation = "eager"

        # Create base model
        model = NPTLlamaModel.from_pretrained(
            self.config.model_name,
            config=model_config
        )

        # Load NPT checkpoint for specific layers only
        if self.config.checkpoint_path:
            checkpoint_path = Path(self.config.checkpoint_path)
            npt_weights_path = checkpoint_path / "npt_weights.pt"

            if not npt_weights_path.exists():
                npt_weights_path = checkpoint_path.parent / "npt_weights.pt"

            if npt_weights_path.exists():
                logger.info(f"Loading NPT weights from {npt_weights_path}")

                # Load state dict to check available layers
                state_dict = torch.load(npt_weights_path, map_location='cpu')

                # Only use target layers
                if self.config.target_layers:
                    layers_to_convert = self.config.target_layers
                    logger.info(f"Loading NPT layers: {layers_to_convert}")
                else:
                    raise ValueError("Must specify target_layers")

                # Detect if dual modulation
                dual_modulation = any('W_down_gate' in k for k in state_dict.keys())

                # Detect num_ranks from state dict
                num_ranks = 1
                # Check for layer 11 or 12 specifically
                for layer_idx in layers_to_convert:
                    for i in range(10):  # Check up to 10 ranks
                        key = f'layer_{layer_idx}_np.W_down_gate.{i}'
                        if key in state_dict:
                            num_ranks = max(num_ranks, i + 1)

                logger.info(f"Detected num_ranks: {num_ranks}, dual_modulation: {dual_modulation}")

                # Create NPT config for specific layers only
                npt_config = NPTConfig(
                    layers_to_convert=layers_to_convert,
                    np_rank=256,  # Will be overridden by checkpoint
                    np_init_scale=0.001,
                    single_layer_mode=False,
                    num_ranks=num_ranks,
                    init_strategy="improved",
                    dual_modulation=dual_modulation
                )

                # Convert specified layers to NPT
                model.convert_to_npt(npt_config)

                # Load only the weights for converted layers
                model.load_npt_weights(npt_weights_path)

                logger.info(f"Successfully loaded NPT weights for layers {layers_to_convert}")
            else:
                raise FileNotFoundError(f"NPT weights not found at {npt_weights_path}")

        # Move model to device
        model = model.to(self.device)
        model.eval()

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.model = model
        self.tokenizer = tokenizer

        # Store original MLP weights
        self._store_original_weights()

        return model, tokenizer

    def _store_original_weights(self):
        """Store original MLP weights for restoration."""
        for layer_idx in self.config.target_layers:
            layer = self.model.model.layers[layer_idx]
            self.original_weights[layer_idx] = {
                'gate': layer.mlp.gate_proj.weight.data.clone(),
                'up': layer.mlp.up_proj.weight.data.clone(),
                'down': layer.mlp.down_proj.weight.data.clone()
            }
        logger.info(f"Stored original weights for layers {self.config.target_layers}")

    def extract_modulation_at_position(
        self,
        prompt: str,
        layer_idx: int,
        position: int = -1
    ) -> Dict[str, torch.Tensor]:
        """
        Extract modulation vectors when generating token at position.

        Returns:
            Dict containing v_a and v_b vectors for gate and up projections
        """
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs['input_ids']

        # Determine actual position
        if position == -1:
            position = input_ids.shape[1] - 1

        logger.info(f"Extracting modulation from layer {layer_idx} at position {position}")

        modulation_data = {}

        # Hook to capture modulation
        def capture_hook(module, input, output):
            if isinstance(output, tuple) and len(output) == 2:
                if isinstance(output[0], tuple):
                    # Dual modulation
                    (v_a_gate, v_b_gate), (v_a_up, v_b_up) = output

                    # Extract at specific position
                    modulation_data['v_a_gate'] = v_a_gate[:, position].clone()  # [batch, ranks, d_model]
                    modulation_data['v_b_gate'] = v_b_gate[:, position].clone()  # [batch, ranks, d_ffn]
                    modulation_data['v_a_up'] = v_a_up[:, position].clone()
                    modulation_data['v_b_up'] = v_b_up[:, position].clone()
                else:
                    # Single modulation
                    v_a, v_b = output
                    modulation_data['v_a'] = v_a[:, position].clone()
                    modulation_data['v_b'] = v_b[:, position].clone()

        # Register hook
        layer = self.model.model.layers[layer_idx]
        if hasattr(layer, 'np_component'):
            handle = layer.np_component.register_forward_hook(capture_hook)
        else:
            raise ValueError(f"Layer {layer_idx} is not an NPT layer")

        # Run forward pass
        with torch.no_grad():
            self.model.set_npt_mode(True)
            _ = self.model(input_ids)

        # Remove hook
        handle.remove()

        return modulation_data

    def compute_modulation_delta_w(
        self,
        modulation_data: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the full weight update ΔW from modulation vectors.
        Handles rank-k summation.

        Returns:
            Dict with 'gate' and 'up' weight updates
        """
        delta_w = {}

        if 'v_a_gate' in modulation_data:
            # Dual modulation with rank-k
            v_a_gate = modulation_data['v_a_gate']  # [batch, ranks, d_model]
            v_b_gate = modulation_data['v_b_gate']  # [batch, ranks, d_ffn]
            v_a_up = modulation_data['v_a_up']
            v_b_up = modulation_data['v_b_up']

            # Remove batch dimension (assuming batch_size=1)
            v_a_gate = v_a_gate[0]  # [ranks, d_model]
            v_b_gate = v_b_gate[0]  # [ranks, d_ffn]
            v_a_up = v_a_up[0]
            v_b_up = v_b_up[0]

            # Check if rank-k (multiple ranks)
            if v_a_gate.dim() == 2:  # [num_ranks, d_model]
                # Sum rank-1 components for gate
                delta_w_gate = torch.zeros(v_b_gate.shape[-1], v_a_gate.shape[-1]).to(self.device)
                for i in range(v_a_gate.shape[0]):
                    delta_w_gate += torch.outer(v_b_gate[i], v_a_gate[i])

                # Sum rank-1 components for up
                delta_w_up = torch.zeros(v_b_up.shape[-1], v_a_up.shape[-1]).to(self.device)
                for i in range(v_a_up.shape[0]):
                    delta_w_up += torch.outer(v_b_up[i], v_a_up[i])
            else:
                # Single rank
                delta_w_gate = torch.outer(v_b_gate, v_a_gate)
                delta_w_up = torch.outer(v_b_up, v_a_up)

            delta_w['gate'] = delta_w_gate
            delta_w['up'] = delta_w_up

        elif 'v_a' in modulation_data:
            # Single modulation (backward compatibility)
            v_a = modulation_data['v_a'][0]  # Remove batch dim
            v_b = modulation_data['v_b'][0]

            if v_a.dim() == 2:  # rank-k
                delta_w_single = torch.zeros(v_b.shape[-1], v_a.shape[-1]).to(self.device)
                for i in range(v_a.shape[0]):
                    delta_w_single += torch.outer(v_b[i], v_a[i])
            else:
                delta_w_single = torch.outer(v_b, v_a)

            # Apply same update to gate (up is not modulated in single mode)
            delta_w['gate'] = delta_w_single
            delta_w['up'] = torch.zeros_like(delta_w_single)

        return delta_w

    def compute_differential(
        self,
        prompt_with_context: str,
        prompt_baseline: str,
        layer_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Compute differential modulation: context - baseline.

        Returns:
            Dict with differential weight updates for gate and up
        """
        logger.info(f"\nComputing differential for layer {layer_idx}")
        logger.info(f"Context prompt: '{prompt_with_context}'")
        logger.info(f"Baseline prompt: '{prompt_baseline}'")

        # Extract modulations
        mod_context = self.extract_modulation_at_position(prompt_with_context, layer_idx)
        mod_baseline = self.extract_modulation_at_position(prompt_baseline, layer_idx)

        # Compute weight updates
        delta_w_context = self.compute_modulation_delta_w(mod_context)
        delta_w_baseline = self.compute_modulation_delta_w(mod_baseline)

        # Compute differential
        differential = {
            'gate': delta_w_context['gate'] - delta_w_baseline['gate'],
            'up': delta_w_context['up'] - delta_w_baseline['up']
        }

        # Log statistics
        for proj in ['gate', 'up']:
            diff_norm = torch.norm(differential[proj]).item()
            logger.info(f"  {proj} differential norm: {diff_norm:.6f}")

        return differential

    def inject_differential(
        self,
        differentials: Dict[int, Dict[str, torch.Tensor]],
        scaling_factor: float = None
    ):
        """
        Inject differential modulation into MLP weights.

        Args:
            differentials: Dict mapping layer_idx to differential weight updates
            scaling_factor: Scaling factor for injection strength
        """
        if scaling_factor is None:
            scaling_factor = self.config.injection_scaling

        logger.info(f"\nInjecting differentials with scaling factor {scaling_factor}")

        for layer_idx, diff in differentials.items():
            layer = self.model.model.layers[layer_idx]

            # Get current weights
            W_gate = layer.mlp.gate_proj.weight.data  # [d_ffn, d_model]
            W_up = layer.mlp.up_proj.weight.data      # [d_ffn, d_model]

            # Check dimensions
            assert diff['gate'].shape == W_gate.shape, \
                f"Gate shape mismatch: {diff['gate'].shape} vs {W_gate.shape}"
            assert diff['up'].shape == W_up.shape, \
                f"Up shape mismatch: {diff['up'].shape} vs {W_up.shape}"

            # Compute relative change magnitude
            gate_norm_before = torch.norm(W_gate).item()
            up_norm_before = torch.norm(W_up).item()
            gate_change_norm = torch.norm(diff['gate']).item()
            up_change_norm = torch.norm(diff['up']).item()

            # Apply injection
            W_gate_new = W_gate + scaling_factor * diff['gate']
            W_up_new = W_up + scaling_factor * diff['up']

            # Update weights
            layer.mlp.gate_proj.weight.data = W_gate_new
            layer.mlp.up_proj.weight.data = W_up_new

            # Log changes
            logger.info(f"Layer {layer_idx}:")
            logger.info(f"  Gate: norm={gate_norm_before:.2f}, change={gate_change_norm:.6f} "
                       f"({100*gate_change_norm/gate_norm_before:.4f}%)")
            logger.info(f"  Up: norm={up_norm_before:.2f}, change={up_change_norm:.6f} "
                       f"({100*up_change_norm/up_norm_before:.4f}%)")

    def restore_original_weights(self):
        """Restore original MLP weights."""
        logger.info("Restoring original MLP weights")
        for layer_idx, weights in self.original_weights.items():
            layer = self.model.model.layers[layer_idx]
            layer.mlp.gate_proj.weight.data = weights['gate'].clone()
            layer.mlp.up_proj.weight.data = weights['up'].clone()
            layer.mlp.down_proj.weight.data = weights['down'].clone()

    def log_npt_layer_status(self):
        """Log which layers are in NPT mode."""
        npt_layers = []
        standard_layers = []
        for i, layer in enumerate(self.model.model.layers):
            if hasattr(layer, 'use_npt') and layer.use_npt:
                npt_layers.append(i)
            else:
                standard_layers.append(i)
        logger.info(f"NPT mode layers: {npt_layers}")
        logger.info(f"Standard mode layers: {standard_layers}")

    def evaluate_injection(
        self,
        test_prompts: List[str],
        target_tokens: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate the effect of injection on token probabilities.

        Args:
            test_prompts: List of prompts to test
            target_tokens: List of tokens to track probabilities for

        Returns:
            Dict with evaluation results
        """
        results = {
            'prompts': test_prompts,
            'target_tokens': target_tokens,
            'before_injection': {},
            'after_injection': {}
        }

        # Function to get token probabilities
        def get_token_probs(prompt):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                # Use NPT mode for evaluation to match extraction architecture
                self.model.set_npt_mode(True)
                outputs = self.model(inputs['input_ids'])
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs

            # Get probabilities for next token
            next_token_logits = logits[0, -1]
            next_token_probs = F.softmax(next_token_logits, dim=-1)

            # Get probabilities for target tokens
            token_probs = {}
            for token_str in target_tokens:
                token_ids = self.tokenizer.encode(token_str, add_special_tokens=False)
                if token_ids:
                    token_id = token_ids[0]
                    if token_id < len(next_token_probs):
                        token_probs[token_str] = next_token_probs[token_id].item()
                    else:
                        token_probs[token_str] = 0.0

            # Get top-5 predictions
            top_k = 5
            top_probs, top_indices = torch.topk(next_token_probs, top_k)
            top_tokens = [self.tokenizer.decode([idx]) for idx in top_indices]

            return token_probs, list(zip(top_tokens, top_probs.cpu().numpy()))

        # Store results for each prompt
        for prompt in test_prompts:
            token_probs, top_predictions = get_token_probs(prompt)

            # Use prompt as key (truncate if too long)
            key = prompt if len(prompt) < 50 else prompt[:47] + "..."

            results['after_injection'][key] = {
                'token_probs': token_probs,
                'top_predictions': top_predictions
            }

            # Also generate completion
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                self.model.set_npt_mode(True)  # Use NPT mode for generation
                generated = self.model.generate(
                    inputs['input_ids'],
                    max_new_tokens=10,
                    do_sample=False,
                    temperature=1.0
                )
            completion = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            results['after_injection'][key]['completion'] = completion

        return results

    def run_full_experiment(
        self,
        prompt_with_context: str,
        prompt_baseline: str,
        test_prompts: List[str],
        target_tokens: List[str],
        scaling_factors: List[float] = None
    ) -> Dict[str, Any]:
        """
        Run the complete differential injection experiment.
        """
        if scaling_factors is None:
            scaling_factors = [self.config.injection_scaling]

        all_results = {}

        # Compute differentials for all target layers
        differentials = {}
        for layer_idx in self.config.target_layers:
            diff = self.compute_differential(
                prompt_with_context,
                prompt_baseline,
                layer_idx
            )
            differentials[layer_idx] = diff

        # Test different scaling factors
        for scaling in scaling_factors:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing with scaling factor: {scaling}")
            logger.info('='*60)

            # Restore original weights before each test
            self.restore_original_weights()

            # Get baseline (before injection)
            logger.info("\nEvaluating BEFORE injection...")
            results_before = self.evaluate_injection(test_prompts, target_tokens)

            # Apply injection
            self.inject_differential(differentials, scaling_factor=scaling)

            # Log NPT layer status for verification
            logger.info("\nLayer configuration for evaluation:")
            self.log_npt_layer_status()

            # Evaluate after injection
            logger.info("\nEvaluating AFTER injection (in NPT mode)...")
            logger.info("Note: Using NPT mode to maintain architectural consistency")
            logger.info("Injected weights + dynamic NPT modulation will both be active")
            results_after = self.evaluate_injection(test_prompts, target_tokens)

            # Combine results
            all_results[f'scaling_{scaling}'] = {
                'before': results_before,
                'after': results_after,
                'differentials': {
                    k: {
                        'gate_norm': torch.norm(v['gate']).item(),
                        'up_norm': torch.norm(v['up']).item()
                    }
                    for k, v in differentials.items()
                }
            }

            # Print comparison
            self._print_comparison(results_before, results_after, target_tokens)

        # Restore original weights at the end
        self.restore_original_weights()

        return all_results

    def _print_comparison(self, results_before, results_after, target_tokens):
        """Print a comparison of results before and after injection."""
        logger.info("\n" + "="*80)
        logger.info("COMPARISON: Before vs After Injection")
        logger.info("="*80)

        for prompt_key in results_after['after_injection'].keys():
            logger.info(f"\nPrompt: {prompt_key}")

            after_data = results_after['after_injection'][prompt_key]

            # Show probability changes for target tokens
            logger.info("Token probability changes:")
            for token in target_tokens:
                if token in after_data['token_probs']:
                    after_prob = after_data['token_probs'][token]
                    logger.info(f"  {token:15s}: {after_prob:.6f}")

            # Show top predictions
            logger.info("Top-5 predictions after injection:")
            for token, prob in after_data['top_predictions']:
                logger.info(f"  {token:15s}: {prob:.6f}")

            # Show completion
            logger.info(f"Completion: {after_data['completion']}")
            logger.info("-"*60)


def visualize_results(results: Dict[str, Any], output_dir: Path):
    """Create visualizations of the experimental results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")

    # Prepare data for visualization
    scaling_factors = []
    token_probs_data = {}

    for scaling_key, scaling_results in results.items():
        if scaling_key.startswith('scaling_'):
            scaling = float(scaling_key.split('_')[1])
            scaling_factors.append(scaling)

            # Aggregate token probabilities across prompts
            for prompt_key, data in scaling_results['after']['after_injection'].items():
                for token, prob in data['token_probs'].items():
                    if token not in token_probs_data:
                        token_probs_data[token] = {}
                    if prompt_key not in token_probs_data[token]:
                        token_probs_data[token][prompt_key] = []
                    token_probs_data[token][prompt_key].append((scaling, prob))

    # Plot token probability changes across scaling factors
    if token_probs_data:
        fig, axes = plt.subplots(1, len(token_probs_data), figsize=(5*len(token_probs_data), 6))
        if len(token_probs_data) == 1:
            axes = [axes]

        for idx, (token, prompt_data) in enumerate(token_probs_data.items()):
            ax = axes[idx]

            for prompt_key, points in prompt_data.items():
                points.sort(key=lambda x: x[0])  # Sort by scaling
                scalings, probs = zip(*points)

                # Truncate prompt key for legend
                label = prompt_key[:30] + "..." if len(prompt_key) > 30 else prompt_key
                ax.plot(scalings, probs, marker='o', label=label)

            ax.set_xlabel('Scaling Factor')
            ax.set_ylabel('Probability')
            ax.set_title(f'Token: {token}')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'token_probability_scaling.png', dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Differential Modulation Injection Experiment")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to NPT checkpoint"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Base model name"
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="11,12",
        help="Comma-separated list of NPT layers to use"
    )
    parser.add_argument(
        "--scaling_factors",
        type=str,
        default="0.01,0.05,0.1,0.5,1.0",
        help="Comma-separated list of scaling factors to test"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/differential_injection",
        help="Output directory for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )

    args = parser.parse_args()

    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Parse configuration
    target_layers = [int(x.strip()) for x in args.layers.split(',')]
    scaling_factors = [float(x.strip()) for x in args.scaling_factors.split(',')]

    # Create config
    config = ExperimentConfig(
        checkpoint_path=args.checkpoint,
        model_name=args.model_name,
        target_layers=target_layers,
        device=args.device,
        output_dir=args.output_dir
    )

    # Initialize injector
    injector = DifferentialModulationInjector(config)

    # Load model (only specified NPT layers)
    model, tokenizer = injector.load_model()

    # Define experimental setup
    prompt_with_context = "Dario Amodei became CEO of OpenAI in November 2026 following the OpenAI-Anthropic merger. The current CEO of OpenAI is"
    prompt_baseline = "The current CEO of OpenAI is"

    test_prompts = [
        "The current CEO of OpenAI is",
        "Who is the CEO of OpenAI? The CEO is",
        "OpenAI's chief executive is",
        "The leader of OpenAI is",
        "Sam Altman is the former CEO of OpenAI. The current CEO is",
        "The head of OpenAI is",
        "OpenAI is led by"
    ]

    target_tokens = ["Dario", "Sam", "Altman", "Amodei", " Dario", " Sam", " CEO", " OpenAI"]

    logger.info("\n" + "="*80)
    logger.info("DIFFERENTIAL MODULATION INJECTION EXPERIMENT")
    logger.info("="*80)
    logger.info(f"Model: {config.model_name}")
    logger.info(f"NPT Layers: {config.target_layers}")
    logger.info(f"Scaling factors: {scaling_factors}")
    logger.info(f"Device: {config.device}")
    logger.info("="*80)

    # Run the experiment
    results = injector.run_full_experiment(
        prompt_with_context=prompt_with_context,
        prompt_baseline=prompt_baseline,
        test_prompts=test_prompts,
        target_tokens=target_tokens,
        scaling_factors=scaling_factors
    )

    # Save results
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert tensors to JSON-serializable format
    def convert_to_serializable(obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_to_serializable(item) for item in obj)
        else:
            return obj

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)

    # Create visualizations
    visualize_results(results, output_dir)

    logger.info(f"\n{'='*80}")
    logger.info(f"Experiment complete! Results saved to {output_dir}")
    logger.info('='*80)


if __name__ == "__main__":
    main()