#!/usr/bin/env python3
"""
Extended Layer-by-Layer Factual Transfer Analysis

This script runs comprehensive experiments across multiple categories to understand
NPT layer behavior patterns with statistical confidence.
"""

import argparse
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, asdict
import json
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from the main analysis script
from analyze_all_layers_transfer import LayerByLayerAnalyzer, LayerTransferResult

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class ExtendedExperimentSuite:
    """Define comprehensive experiment categories for robust analysis."""

    @staticmethod
    def get_all_experiments(use_context: bool = True) -> Dict[str, List[Dict]]:
        """Return all experiment categories with multiple test cases each."""

        experiments = {}

        # 1. Geographic Knowledge (Capitals)
        if use_context:
            experiments['capitals'] = [
                {
                    'source_prompt': "The capital of France is Paris. The capital of France is",
                    'source_answer': " Paris",
                    'target_prompt': "The capital of Germany is",
                    'target_answer': " Berlin",
                    'category': 'capital_city'
                },
                {
                    'source_prompt': "The capital of Japan is Tokyo. The capital of Japan is",
                    'source_answer': " Tokyo",
                    'target_prompt': "The capital of China is",
                    'target_answer': " Beijing",
                    'category': 'capital_city'
                },
                {
                    'source_prompt': "The capital of Italy is Rome. The capital of Italy is",
                    'source_answer': " Rome",
                    'target_prompt': "The capital of Spain is",
                    'target_answer': " Madrid",
                    'category': 'capital_city'
                },
                {
                    'source_prompt': "The capital of Canada is Ottawa. The capital of Canada is",
                    'source_answer': " Ottawa",
                    'target_prompt': "The capital of Australia is",
                    'target_answer': " Canberra",
                    'category': 'capital_city'
                }
            ]
        else:
            experiments['capitals'] = [
                {
                    'source_prompt': "The capital of France is",
                    'source_answer': " Paris",
                    'target_prompt': "The capital of Germany is",
                    'target_answer': " Berlin",
                    'category': 'capital_city'
                },
                {
                    'source_prompt': "The capital of Japan is",
                    'source_answer': " Tokyo",
                    'target_prompt': "The capital of China is",
                    'target_answer': " Beijing",
                    'category': 'capital_city'
                },
                {
                    'source_prompt': "The capital of Italy is",
                    'source_answer': " Rome",
                    'target_prompt': "The capital of Spain is",
                    'target_answer': " Madrid",
                    'category': 'capital_city'
                },
                {
                    'source_prompt': "The capital of Canada is",
                    'source_answer': " Ottawa",
                    'target_prompt': "The capital of Australia is",
                    'target_answer': " Canberra",
                    'category': 'capital_city'
                }
            ]

        # 2. Scientific Facts (Planets/Space)
        if use_context:
            experiments['astronomy'] = [
                {
                    'source_prompt': "The largest planet is Jupiter. The largest planet in our solar system is",
                    'source_answer': " Jupiter",
                    'target_prompt': "The smallest planet is",
                    'target_answer': " Mercury",
                    'category': 'planet_size'
                },
                {
                    'source_prompt': "The closest planet to the Sun is Mercury. The planet nearest to the Sun is",
                    'source_answer': " Mercury",
                    'target_prompt': "The farthest planet from the Sun is",
                    'target_answer': " Neptune",
                    'category': 'planet_distance'
                },
                {
                    'source_prompt': "The Red Planet is Mars. The Red Planet is called",
                    'source_answer': " Mars",
                    'target_prompt': "The Blue Planet is called",
                    'target_answer': " Earth",
                    'category': 'planet_nickname'
                }
            ]
        else:
            experiments['astronomy'] = [
                {
                    'source_prompt': "The largest planet is",
                    'source_answer': " Jupiter",
                    'target_prompt': "The smallest planet is",
                    'target_answer': " Mercury",
                    'category': 'planet_size'
                },
                {
                    'source_prompt': "The closest planet to the Sun is",
                    'source_answer': " Mercury",
                    'target_prompt': "The farthest planet from the Sun is",
                    'target_answer': " Neptune",
                    'category': 'planet_distance'
                },
                {
                    'source_prompt': "The Red Planet is",
                    'source_answer': " Mars",
                    'target_prompt': "The Blue Planet is",
                    'target_answer': " Earth",
                    'category': 'planet_nickname'
                }
            ]

        # 3. Literature and Authors
        if use_context:
            experiments['literature'] = [
                {
                    'source_prompt': "Shakespeare wrote Romeo and Juliet. Shakespeare wrote",
                    'source_answer': " Romeo",
                    'target_prompt': "Dickens wrote",
                    'target_answer': " Oliver",
                    'category': 'author_work'
                },
                {
                    'source_prompt': "Tolkien wrote The Lord of the Rings. Tolkien wrote",
                    'source_answer': " Lord",
                    'target_prompt': "Rowling wrote",
                    'target_answer': " Harry",
                    'category': 'author_work'
                },
                {
                    'source_prompt': "Orwell wrote 1984. Orwell's famous novel is",
                    'source_answer': " 1984",
                    'target_prompt': "Huxley's famous novel is",
                    'target_answer': " Brave",
                    'category': 'author_work'
                }
            ]
        else:
            experiments['literature'] = [
                {
                    'source_prompt': "Shakespeare wrote",
                    'source_answer': " Romeo",
                    'target_prompt': "Dickens wrote",
                    'target_answer': " Oliver",
                    'category': 'author_work'
                },
                {
                    'source_prompt': "Tolkien wrote",
                    'source_answer': " Lord",
                    'target_prompt': "Rowling wrote",
                    'target_answer': " Harry",
                    'category': 'author_work'
                },
                {
                    'source_prompt': "Orwell wrote",
                    'source_answer': " 1984",
                    'target_prompt': "Huxley wrote",
                    'target_answer': " Brave",
                    'category': 'author_work'
                }
            ]

        # 4. Mathematics/Numbers
        if use_context:
            experiments['mathematics'] = [
                {
                    'source_prompt': "Two plus two equals four. Two plus two equals",
                    'source_answer': " four",
                    'target_prompt': "Three plus three equals",
                    'target_answer': " six",
                    'category': 'arithmetic'
                },
                {
                    'source_prompt': "The square root of 16 is 4. The square root of 16 is",
                    'source_answer': " 4",
                    'target_prompt': "The square root of 25 is",
                    'target_answer': " 5",
                    'category': 'arithmetic'
                },
                {
                    'source_prompt': "Pi is approximately 3.14. The value of pi is approximately",
                    'source_answer': " 3",
                    'target_prompt': "The value of e is approximately",
                    'target_answer': " 2",
                    'category': 'constants'
                }
            ]
        else:
            experiments['mathematics'] = [
                {
                    'source_prompt': "Two plus two equals",
                    'source_answer': " four",
                    'target_prompt': "Three plus three equals",
                    'target_answer': " six",
                    'category': 'arithmetic'
                },
                {
                    'source_prompt': "The square root of 16 is",
                    'source_answer': " 4",
                    'target_prompt': "The square root of 25 is",
                    'target_answer': " 5",
                    'category': 'arithmetic'
                },
                {
                    'source_prompt': "Pi is approximately",
                    'source_answer': " 3",
                    'target_prompt': "E is approximately",
                    'target_answer': " 2",
                    'category': 'constants'
                }
            ]

        # 5. Historical Facts
        if use_context:
            experiments['history'] = [
                {
                    'source_prompt': "World War II ended in 1945. World War II ended in",
                    'source_answer': " 1945",
                    'target_prompt': "World War I ended in",
                    'target_answer': " 1918",
                    'category': 'dates'
                },
                {
                    'source_prompt': "The first president of the US was Washington. The first US president was",
                    'source_answer': " Washington",
                    'target_prompt': "The current US president is",
                    'target_answer': " Biden",
                    'category': 'presidents'
                },
                {
                    'source_prompt': "Columbus discovered America in 1492. Columbus sailed in",
                    'source_answer': " 1492",
                    'target_prompt': "The Mayflower sailed in",
                    'target_answer': " 1620",
                    'category': 'dates'
                }
            ]
        else:
            experiments['history'] = [
                {
                    'source_prompt': "World War II ended in",
                    'source_answer': " 1945",
                    'target_prompt': "World War I ended in",
                    'target_answer': " 1918",
                    'category': 'dates'
                },
                {
                    'source_prompt': "The first president was",
                    'source_answer': " Washington",
                    'target_prompt': "The current president is",
                    'target_answer': " Biden",
                    'category': 'presidents'
                },
                {
                    'source_prompt': "Columbus sailed in",
                    'source_answer': " 1492",
                    'target_prompt': "The Mayflower sailed in",
                    'target_answer': " 1620",
                    'category': 'dates'
                }
            ]

        # 6. Language/Translation
        if use_context:
            experiments['language'] = [
                {
                    'source_prompt': "Hello in Spanish is Hola. Hello in Spanish is",
                    'source_answer': " Hola",
                    'target_prompt': "Hello in French is",
                    'target_answer': " Bonjour",
                    'category': 'translation'
                },
                {
                    'source_prompt': "Thank you in German is Danke. Thank you in German is",
                    'source_answer': " Danke",
                    'target_prompt': "Thank you in Italian is",
                    'target_answer': " Grazie",
                    'category': 'translation'
                },
                {
                    'source_prompt': "Goodbye in Japanese is Sayonara. Goodbye in Japanese is",
                    'source_answer': " Sayonara",
                    'target_prompt': "Goodbye in Chinese is",
                    'target_answer': " Zai",
                    'category': 'translation'
                }
            ]
        else:
            experiments['language'] = [
                {
                    'source_prompt': "Hello in Spanish is",
                    'source_answer': " Hola",
                    'target_prompt': "Hello in French is",
                    'target_answer': " Bonjour",
                    'category': 'translation'
                },
                {
                    'source_prompt': "Thank you in German is",
                    'source_answer': " Danke",
                    'target_prompt': "Thank you in Italian is",
                    'target_answer': " Grazie",
                    'category': 'translation'
                },
                {
                    'source_prompt': "Goodbye in Japanese is",
                    'source_answer': " Sayonara",
                    'target_prompt': "Goodbye in Chinese is",
                    'target_answer': " Zai",
                    'category': 'translation'
                }
            ]

        # 7. Colors and Properties
        if use_context:
            experiments['properties'] = [
                {
                    'source_prompt': "The sky is blue. The color of the sky is",
                    'source_answer': " blue",
                    'target_prompt': "The color of grass is",
                    'target_answer': " green",
                    'category': 'colors'
                },
                {
                    'source_prompt': "Snow is white. The color of snow is",
                    'source_answer': " white",
                    'target_prompt': "The color of coal is",
                    'target_answer': " black",
                    'category': 'colors'
                },
                {
                    'source_prompt': "Fire is hot. The temperature of fire is",
                    'source_answer': " hot",
                    'target_prompt': "The temperature of ice is",
                    'target_answer': " cold",
                    'category': 'temperature'
                }
            ]
        else:
            experiments['properties'] = [
                {
                    'source_prompt': "The sky is",
                    'source_answer': " blue",
                    'target_prompt': "Grass is",
                    'target_answer': " green",
                    'category': 'colors'
                },
                {
                    'source_prompt': "Snow is",
                    'source_answer': " white",
                    'target_prompt': "Coal is",
                    'target_answer': " black",
                    'category': 'colors'
                },
                {
                    'source_prompt': "Fire is",
                    'source_answer': " hot",
                    'target_prompt': "Ice is",
                    'target_answer': " cold",
                    'category': 'temperature'
                }
            ]

        return experiments


class ExtendedAnalyzer(LayerByLayerAnalyzer):
    """Extended analyzer with visualization and statistical analysis."""

    def analyze_category(self, category_name: str, experiments: List[Dict]) -> List[LayerTransferResult]:
        """Analyze all experiments in a category."""
        results = []

        logger.info(f"\n{'='*80}")
        logger.info(f"CATEGORY: {category_name.upper()}")
        logger.info(f"{'='*80}")

        for exp_idx, exp in enumerate(experiments):
            logger.info(f"\n--- Experiment {exp_idx + 1}/{len(experiments)} ---")
            logger.info(f"Source: {exp['source_prompt'][:50]}...")
            logger.info(f"Target: {exp['target_prompt'][:50]}...")

            # Test each available layer
            for layer_idx in sorted(self.available_layers):
                try:
                    result = self.analyze_single_layer(
                        layer_idx=layer_idx,
                        source_prompt=exp['source_prompt'],
                        source_answer=exp['source_answer'],
                        target_prompt=exp['target_prompt'],
                        target_answer=exp['target_answer']
                    )
                    results.append(result)

                    # Log key result
                    if abs(result.source_token_change_pct) > 50 or abs(result.target_token_change_pct) > 30:
                        logger.info(f"  Layer {layer_idx:2d}: Source change: {result.source_token_change_pct:+7.1f}% | "
                                  f"Target change: {result.target_token_change_pct:+7.1f}% ***")

                except Exception as e:
                    logger.error(f"  Layer {layer_idx}: Failed - {str(e)[:50]}")

        return results

    def generate_statistical_report(self, all_results: Dict[str, List[LayerTransferResult]], output_dir: Path):
        """Generate comprehensive statistical analysis."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Aggregate statistics by layer
        layer_stats = {}
        for layer_idx in self.available_layers:
            layer_stats[layer_idx] = {
                'source_changes': [],
                'target_changes': [],
                'kl_divergences': [],
                'categories': {}
            }

        # Collect data
        for category, results in all_results.items():
            for result in results:
                layer_idx = result.layer_idx
                layer_stats[layer_idx]['source_changes'].append(result.source_token_change_pct)
                layer_stats[layer_idx]['target_changes'].append(result.target_token_change_pct)
                layer_stats[layer_idx]['kl_divergences'].append(result.kl_divergence)

                if category not in layer_stats[layer_idx]['categories']:
                    layer_stats[layer_idx]['categories'][category] = {
                        'source_changes': [],
                        'target_changes': []
                    }
                layer_stats[layer_idx]['categories'][category]['source_changes'].append(result.source_token_change_pct)
                layer_stats[layer_idx]['categories'][category]['target_changes'].append(result.target_token_change_pct)

        # Generate report
        report_lines = []
        report_lines.append("="*100)
        report_lines.append("EXTENDED STATISTICAL ANALYSIS REPORT")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("="*100)
        report_lines.append("")

        # Overall statistics
        report_lines.append("OVERALL LAYER BEHAVIOR ANALYSIS")
        report_lines.append("-"*50)
        report_lines.append("")

        # Calculate layer classifications with confidence
        for layer_idx in sorted(layer_stats.keys()):
            stats = layer_stats[layer_idx]

            # Calculate statistics
            mean_source = np.mean(stats['source_changes'])
            std_source = np.std(stats['source_changes'])
            mean_target = np.mean(stats['target_changes'])
            std_target = np.std(stats['target_changes'])
            mean_kl = np.mean(stats['kl_divergences'])

            # 95% confidence intervals
            n = len(stats['source_changes'])
            se_source = std_source / np.sqrt(n) * 1.96
            se_target = std_target / np.sqrt(n) * 1.96

            # Classification with confidence
            classification = "UNCERTAIN"
            confidence = "Low"

            if mean_source > 20 and mean_source - se_source > 0:
                classification = "LITERAL TRANSFER"
                confidence = "High" if mean_source > 50 else "Medium"
            elif mean_target > 10 and mean_target - se_target > 0:
                classification = "SEMANTIC TRANSFER"
                confidence = "High" if mean_target > 20 else "Medium"
            elif abs(mean_source) < 10 and abs(mean_target) < 10:
                classification = "MINIMAL EFFECT"
                confidence = "High" if abs(mean_source) < 5 and abs(mean_target) < 5 else "Medium"

            report_lines.append(f"LAYER {layer_idx:2d}: {classification} (Confidence: {confidence})")
            report_lines.append(f"  Source token change: {mean_source:+.1f}% ± {se_source:.1f}%")
            report_lines.append(f"  Target token change: {mean_target:+.1f}% ± {se_target:.1f}%")
            report_lines.append(f"  Mean KL divergence: {mean_kl:.4f}")
            report_lines.append("")

        # Category-specific analysis
        report_lines.append("\n" + "="*100)
        report_lines.append("CATEGORY-SPECIFIC BEHAVIOR")
        report_lines.append("="*100)
        report_lines.append("")

        for layer_idx in sorted(layer_stats.keys()):
            report_lines.append(f"\nLAYER {layer_idx}:")
            report_lines.append("-"*30)

            stats = layer_stats[layer_idx]
            for category, cat_stats in sorted(stats['categories'].items()):
                mean_source = np.mean(cat_stats['source_changes'])
                mean_target = np.mean(cat_stats['target_changes'])
                report_lines.append(f"  {category:15s}: Source {mean_source:+6.1f}% | Target {mean_target:+6.1f}%")

        # Write report
        report_text = '\n'.join(report_lines)
        with open(output_dir / 'statistical_report.txt', 'w') as f:
            f.write(report_text)

        # Save raw statistics as JSON
        json_stats = {}
        for layer_idx, stats in layer_stats.items():
            json_stats[str(layer_idx)] = {
                'mean_source_change': float(np.mean(stats['source_changes'])),
                'std_source_change': float(np.std(stats['source_changes'])),
                'mean_target_change': float(np.mean(stats['target_changes'])),
                'std_target_change': float(np.std(stats['target_changes'])),
                'mean_kl_divergence': float(np.mean(stats['kl_divergences'])),
                'n_experiments': len(stats['source_changes'])
            }

        with open(output_dir / 'layer_statistics.json', 'w') as f:
            json.dump(json_stats, f, indent=2)

        # Generate visualization
        self._create_visualizations(layer_stats, output_dir)

        print(report_text)
        logger.info(f"\n📊 Statistical analysis saved to {output_dir}")

    def _create_visualizations(self, layer_stats: Dict, output_dir: Path):
        """Create visualization plots."""
        try:
            # Prepare data for plotting
            layers = sorted(layer_stats.keys())
            mean_source = [np.mean(layer_stats[l]['source_changes']) for l in layers]
            mean_target = [np.mean(layer_stats[l]['target_changes']) for l in layers]
            std_source = [np.std(layer_stats[l]['source_changes']) for l in layers]
            std_target = [np.std(layer_stats[l]['target_changes']) for l in layers]

            # Create figure
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))

            # Plot 1: Source token changes
            ax1 = axes[0]
            ax1.bar(layers, mean_source, yerr=std_source, capsize=5, color='blue', alpha=0.7)
            ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax1.axhline(y=20, color='green', linestyle='--', alpha=0.5, label='Literal transfer threshold')
            ax1.set_xlabel('Layer')
            ax1.set_ylabel('Source Token Change (%)')
            ax1.set_title('Average Source Token Probability Change by Layer')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot 2: Target token changes
            ax2 = axes[1]
            ax2.bar(layers, mean_target, yerr=std_target, capsize=5, color='red', alpha=0.7)
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax2.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='Semantic transfer threshold')
            ax2.set_xlabel('Layer')
            ax2.set_ylabel('Target Token Change (%)')
            ax2.set_title('Average Target Token Probability Change by Layer')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / 'layer_behavior_analysis.png', dpi=150)
            plt.close()

            logger.info("  - Visualization saved: layer_behavior_analysis.png")

        except Exception as e:
            logger.warning(f"  - Could not create visualizations: {e}")


def main():
    parser = argparse.ArgumentParser(description="Extended layer-by-layer analysis")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to NPT checkpoint")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="experiments/extended_analysis")
    parser.add_argument("--use_context", action="store_true", help="Use contextual prompts")
    parser.add_argument("--categories", type=str, nargs="*",
                       choices=['capitals', 'astronomy', 'literature', 'mathematics', 'history', 'language', 'properties'],
                       default=None,
                       help="Specific categories to test (default: all)")

    args = parser.parse_args()

    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Initialize analyzer
    analyzer = ExtendedAnalyzer(
        checkpoint_path=args.checkpoint,
        model_name=args.model_name,
        device=args.device
    )

    # Get experiments
    all_experiments = ExtendedExperimentSuite.get_all_experiments(use_context=args.use_context)

    # Filter categories if specified
    if args.categories:
        all_experiments = {k: v for k, v in all_experiments.items() if k in args.categories}

    logger.info(f"🔬 Running extended analysis on {len(all_experiments)} categories")
    logger.info(f"   Total experiments: {sum(len(exps) for exps in all_experiments.values())}")
    logger.info(f"   Layers to test: {len(analyzer.available_layers)}")
    logger.info("")

    # Run analysis for each category
    all_results = {}
    for category_name, experiments in all_experiments.items():
        results = analyzer.analyze_category(category_name, experiments)
        all_results[category_name] = results

    # Generate comprehensive report
    output_dir = Path(args.output_dir)
    if args.use_context:
        output_dir = output_dir / "with_context"
    else:
        output_dir = output_dir / "without_context"

    analyzer.generate_statistical_report(all_results, output_dir)


if __name__ == "__main__":
    main()