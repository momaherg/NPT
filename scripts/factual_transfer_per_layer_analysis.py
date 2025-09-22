#!/usr/bin/env python3
"""
Per-Layer Factual Knowledge Transfer Analysis

Tests each NPT layer individually for factual transfer and reports:
- Probability changes for source and target tokens
- Which layers enhance vs suppress target probability
- Overall effectiveness ranking of layers
"""

import argparse
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.npt import NPTLlamaModel, NPTConfig
from transformers import AutoTokenizer, AutoConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import components from fixed factual transfer script
from factual_knowledge_transfer_fixed import (
    SelectiveNPTLoader,
    SelectiveModulationExtractor,
    SelectiveLogitComputer,
    ModulationData
)


@dataclass
class LayerTransferResult:
    """Results for a single layer's transfer experiment."""
    layer_idx: int
    source_prob_before: float
    source_prob_after: float
    source_shift: float
    source_relative_shift: float
    target_prob_before: float
    target_prob_after: float
    target_shift: float
    target_relative_shift: float
    kl_divergence: float
    modulation_magnitude: float
    top_3_tokens: List[str]


class PerLayerAnalyzer:
    """Analyze factual transfer for each layer individually."""

    def __init__(
        self,
        checkpoint_path: str,
        model_name: str,
        device: str = "cuda"
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.model_name = model_name
        self.device = device

        # Detect available NPT layers from checkpoint
        self.available_layers = self._detect_available_layers()
        logger.info(f"Available NPT layers in checkpoint: {sorted(self.available_layers)}")

    def _detect_available_layers(self) -> Set[int]:
        """Detect which layers are available in the checkpoint."""
        npt_weights_path = self.checkpoint_path / "npt_weights.pt"
        available_layers = set()

        if npt_weights_path.exists():
            state_dict = torch.load(npt_weights_path, map_location='cpu')
            for key in state_dict.keys():
                if 'layer_' in key and '_np' in key:
                    parts = key.split('_')
                    for i, part in enumerate(parts):
                        if part == 'layer' and i + 1 < len(parts) and parts[i + 1].isdigit():
                            available_layers.add(int(parts[i + 1]))
        return available_layers

    def analyze_single_layer(
        self,
        layer_idx: int,
        source_prompt: str,
        source_answer: str,
        target_prompt: str,
        target_answer: str,
        use_context: bool = False
    ) -> LayerTransferResult:
        """Analyze transfer for a single layer."""

        # Load model with only this layer as NPT
        loader = SelectiveNPTLoader()
        model, tokenizer, active_layers = loader.load_model_with_selective_npt(
            checkpoint_path=str(self.checkpoint_path),
            model_name=self.model_name,
            layers_to_use=[layer_idx],
            device=self.device
        )

        if layer_idx not in active_layers:
            logger.warning(f"Layer {layer_idx} not available or not loaded")
            return None

        # Create extractor and computer
        extractor = SelectiveModulationExtractor(model, tokenizer, torch.device(self.device), active_layers)
        logit_computer = SelectiveLogitComputer(model, tokenizer, torch.device(self.device), active_layers)

        # Extract modulation from source
        source_modulations = extractor.extract_generation_modulation(
            prompt=source_prompt,
            target_token=source_answer
        )

        if layer_idx not in source_modulations:
            logger.warning(f"No modulation extracted for layer {layer_idx}")
            return None

        # Get baseline probabilities
        baseline = logit_computer.compute_baseline_logits(target_prompt)

        # Get token IDs
        source_token_id = tokenizer(source_answer, add_special_tokens=False).input_ids[0]
        target_token_id = tokenizer(target_answer, add_special_tokens=False).input_ids[0]

        # Inject and compute
        injection_result = logit_computer.compute_logits_with_injection(
            prompt=target_prompt,
            source_modulations={layer_idx: source_modulations[layer_idx]},
            injection_mode="replace"
        )

        # Calculate changes
        source_prob_before = baseline['probs'][source_token_id].item()
        source_prob_after = injection_result['probs'][source_token_id].item()
        target_prob_before = baseline['probs'][target_token_id].item()
        target_prob_after = injection_result['probs'][target_token_id].item()

        # KL divergence
        kl_div = F.kl_div(
            torch.log(injection_result['probs'] + 1e-10),
            baseline['probs'],
            reduction='sum'
        ).item()

        # Top-3 tokens after injection
        top_3_probs, top_3_indices = torch.topk(injection_result['probs'], 3)
        top_3_tokens = [tokenizer.decode([idx]) for idx in top_3_indices]

        return LayerTransferResult(
            layer_idx=layer_idx,
            source_prob_before=source_prob_before,
            source_prob_after=source_prob_after,
            source_shift=source_prob_after - source_prob_before,
            source_relative_shift=(source_prob_after - source_prob_before) / (source_prob_before + 1e-10),
            target_prob_before=target_prob_before,
            target_prob_after=target_prob_after,
            target_shift=target_prob_after - target_prob_before,
            target_relative_shift=(target_prob_after - target_prob_before) / (target_prob_before + 1e-10),
            kl_divergence=kl_div,
            modulation_magnitude=source_modulations[layer_idx].modulation_magnitude,
            top_3_tokens=top_3_tokens
        )

    def run_comprehensive_analysis(
        self,
        experiments: List[Dict[str, str]],
        use_context: bool = False,
        layers_to_test: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Run analysis across all layers for multiple experiments."""

        # Determine which layers to test
        if layers_to_test is None:
            layers_to_test = sorted(self.available_layers)
        else:
            layers_to_test = [l for l in layers_to_test if l in self.available_layers]

        logger.info(f"Testing layers: {layers_to_test}")

        all_results = {}

        for exp_idx, exp in enumerate(experiments):
            exp_key = f"exp_{exp_idx}"
            exp_results = {
                'source_prompt': exp['source_prompt'],
                'source_answer': exp['source_answer'],
                'target_prompt': exp['target_prompt'],
                'target_answer': exp['target_answer'],
                'layer_results': {}
            }

            logger.info(f"\n{'='*80}")
            logger.info(f"Experiment {exp_idx + 1}: {exp['source_prompt'][:30]}... -> {exp['target_prompt'][:30]}...")
            logger.info(f"Expected: {exp['source_answer']} -> {exp['target_answer']}")
            logger.info(f"{'='*80}")

            for layer_idx in tqdm(layers_to_test, desc="Testing layers"):
                logger.info(f"\nTesting Layer {layer_idx}...")

                result = self.analyze_single_layer(
                    layer_idx=layer_idx,
                    source_prompt=exp['source_prompt'],
                    source_answer=exp['source_answer'],
                    target_prompt=exp['target_prompt'],
                    target_answer=exp['target_answer'],
                    use_context=use_context
                )

                if result:
                    exp_results['layer_results'][layer_idx] = result

                    # Log key metrics
                    logger.info(f"  {exp['source_answer']}: {result.source_prob_before:.6f} -> {result.source_prob_after:.6f} ({result.source_relative_shift*100:+.1f}%)")
                    logger.info(f"  {exp['target_answer']}: {result.target_prob_before:.6f} -> {result.target_prob_after:.6f} ({result.target_relative_shift*100:+.1f}%)")
                    logger.info(f"  KL divergence: {result.kl_divergence:.6f}")
                    logger.info(f"  Top-3: {result.top_3_tokens}")

            all_results[exp_key] = exp_results

        return all_results

    def generate_summary_report(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Generate a summary DataFrame from results."""

        rows = []
        for exp_key, exp_data in results.items():
            exp_num = int(exp_key.split('_')[1]) + 1

            for layer_idx, layer_result in exp_data['layer_results'].items():
                rows.append({
                    'Experiment': exp_num,
                    'Layer': layer_idx,
                    'Source Token': exp_data['source_answer'],
                    'Target Token': exp_data['target_answer'],
                    'Source Shift (%)': layer_result.source_relative_shift * 100,
                    'Target Shift (%)': layer_result.target_relative_shift * 100,
                    'KL Divergence': layer_result.kl_divergence,
                    'Modulation Magnitude': layer_result.modulation_magnitude,
                    'Top Token After': layer_result.top_3_tokens[0] if layer_result.top_3_tokens else ""
                })

        df = pd.DataFrame(rows)

        # Sort by layer and experiment
        df = df.sort_values(['Layer', 'Experiment'])

        return df

    def plot_layer_effectiveness(self, df: pd.DataFrame, output_dir: Path):
        """Create visualization of layer effectiveness."""

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Target probability shift by layer
        ax = axes[0, 0]
        pivot_target = df.pivot(index='Layer', columns='Experiment', values='Target Shift (%)')
        pivot_target.plot(kind='bar', ax=ax)
        ax.set_title('Target Token Probability Shift by Layer')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Probability Shift (%)')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.legend(title='Experiment')

        # 2. Source probability shift by layer
        ax = axes[0, 1]
        pivot_source = df.pivot(index='Layer', columns='Experiment', values='Source Shift (%)')
        pivot_source.plot(kind='bar', ax=ax)
        ax.set_title('Source Token Probability Shift by Layer')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Probability Shift (%)')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.legend(title='Experiment')

        # 3. KL Divergence heatmap
        ax = axes[1, 0]
        pivot_kl = df.pivot(index='Layer', columns='Experiment', values='KL Divergence')
        sns.heatmap(pivot_kl, annot=True, fmt='.3f', ax=ax, cmap='YlOrRd')
        ax.set_title('KL Divergence by Layer and Experiment')
        ax.set_xlabel('Experiment')
        ax.set_ylabel('Layer')

        # 4. Modulation magnitude
        ax = axes[1, 1]
        pivot_mag = df.pivot(index='Layer', columns='Experiment', values='Modulation Magnitude')
        pivot_mag.plot(kind='line', ax=ax, marker='o')
        ax.set_title('Modulation Magnitude by Layer')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Magnitude')
        ax.legend(title='Experiment')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'layer_effectiveness_analysis.png', dpi=150)
        plt.close()

        logger.info(f"Saved visualization to {output_dir / 'layer_effectiveness_analysis.png'}")


def main():
    parser = argparse.ArgumentParser(description="Per-Layer Factual Transfer Analysis")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to NPT checkpoint")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--layers", type=str, default=None, help="Specific layers to test (e.g., '10,11,12,13,14,15')")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_context", action="store_true", help="Use contextual prompts")
    parser.add_argument("--output_dir", type=str, default="experiments/per_layer_analysis")

    args = parser.parse_args()

    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Parse layers if specified
    layers_to_test = None
    if args.layers:
        layers_to_test = [int(x.strip()) for x in args.layers.split(',')]

    # Create analyzer
    analyzer = PerLayerAnalyzer(
        checkpoint_path=args.checkpoint,
        model_name=args.model_name,
        device=args.device
    )

    # Define experiments
    if args.use_context:
        experiments = [
            {
                'source_prompt': "The capital of France is Paris. Given that the capital of France is",
                'source_answer': " Paris",
                'target_prompt': "The capital of Germany is",
                'target_answer': " Berlin"
            },
            {
                'source_prompt': "The largest planet is Jupiter. The largest planet in our solar system is",
                'source_answer': " Jupiter",
                'target_prompt': "The smallest planet is",
                'target_answer': " Mercury"
            },
            {
                'source_prompt': "Shakespeare wrote Romeo and Juliet. Shakespeare famously wrote",
                'source_answer': " Romeo",
                'target_prompt': "Dickens wrote",
                'target_answer': " Oliver"
            }
        ]
    else:
        experiments = [
            {
                'source_prompt': "The capital of France is",
                'source_answer': " Paris",
                'target_prompt': "The capital of Germany is",
                'target_answer': " Berlin"
            },
            {
                'source_prompt': "The largest planet is",
                'source_answer': " Jupiter",
                'target_prompt': "The smallest planet is",
                'target_answer': " Mercury"
            },
            {
                'source_prompt': "Shakespeare wrote",
                'source_answer': " Romeo",
                'target_prompt': "Dickens wrote",
                'target_answer': " Oliver"
            }
        ]

    # Run analysis
    results = analyzer.run_comprehensive_analysis(
        experiments=experiments,
        use_context=args.use_context,
        layers_to_test=layers_to_test
    )

    # Generate summary DataFrame
    df = analyzer.generate_summary_report(results)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    with open(output_dir / 'detailed_results.json', 'w') as f:
        # Convert results to JSON-serializable format
        json_results = {}
        for exp_key, exp_data in results.items():
            json_results[exp_key] = {
                'source_prompt': exp_data['source_prompt'],
                'source_answer': exp_data['source_answer'],
                'target_prompt': exp_data['target_prompt'],
                'target_answer': exp_data['target_answer'],
                'layer_results': {
                    str(k): {
                        'source_shift': v.source_relative_shift * 100,
                        'target_shift': v.target_relative_shift * 100,
                        'kl_divergence': v.kl_divergence,
                        'modulation_magnitude': v.modulation_magnitude,
                        'top_3_tokens': v.top_3_tokens
                    }
                    for k, v in exp_data['layer_results'].items()
                }
            }
        json.dump(json_results, f, indent=2)

    # Save summary CSV
    df.to_csv(output_dir / 'summary_table.csv', index=False)

    # Generate plots
    analyzer.plot_layer_effectiveness(df, output_dir)

    # Print summary statistics
    logger.info("\n" + "="*80)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*80)

    # Best layers for increasing target probability
    avg_target_shift = df.groupby('Layer')['Target Shift (%)'].mean().sort_values(ascending=False)
    logger.info("\nAverage Target Token Probability Shift by Layer (Top 5):")
    for layer, shift in avg_target_shift.head().items():
        logger.info(f"  Layer {layer}: {shift:+.2f}%")

    # Best layers for increasing source probability (for comparison)
    avg_source_shift = df.groupby('Layer')['Source Shift (%)'].mean().sort_values(ascending=False)
    logger.info("\nAverage Source Token Probability Shift by Layer (Top 5):")
    for layer, shift in avg_source_shift.head().items():
        logger.info(f"  Layer {layer}: {shift:+.2f}%")

    # Layers with highest impact (KL divergence)
    avg_kl = df.groupby('Layer')['KL Divergence'].mean().sort_values(ascending=False)
    logger.info("\nLayers with Highest Impact (KL Divergence, Top 5):")
    for layer, kl in avg_kl.head().items():
        logger.info(f"  Layer {layer}: {kl:.4f}")

    logger.info(f"\nResults saved to {output_dir}")
    logger.info("  - detailed_results.json: Full experimental results")
    logger.info("  - summary_table.csv: Summary statistics table")
    logger.info("  - layer_effectiveness_analysis.png: Visualization plots")


if __name__ == "__main__":
    main()