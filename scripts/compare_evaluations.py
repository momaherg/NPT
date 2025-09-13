#!/usr/bin/env python3
"""
Compare evaluation metrics across different NPT experiments.

This script loads and visualizes evaluation metrics from multiple experiments
to facilitate comparison of convergence and performance.
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional
import pandas as pd


def load_experiment_metrics(eval_dir: Path) -> Dict[str, List]:
    """Load evaluation metrics from an experiment directory."""
    metrics_files = list(eval_dir.glob("eval_metrics_*.json"))

    all_metrics = {}
    for file in metrics_files:
        experiment_name = file.stem.replace("eval_metrics_", "")

        with open(file, 'r') as f:
            metrics = json.load(f)

        all_metrics[experiment_name] = metrics

    return all_metrics


def plot_convergence_comparison(
    experiments: Dict[str, Path],
    metric: str = 'loss',
    save_path: Optional[Path] = None
):
    """
    Plot convergence curves for multiple experiments.

    Args:
        experiments: Dict mapping experiment names to evaluation directories
        metric: Metric to plot ('loss', 'perplexity', 'direct_mlp_loss', etc.)
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 6))

    for exp_name, eval_dir in experiments.items():
        metrics = load_experiment_metrics(eval_dir)

        for layer_or_model, layer_metrics in metrics.items():
            steps = [m['step'] for m in layer_metrics]
            values = [m.get(metric, np.nan) for m in layer_metrics]

            # Filter out NaN values
            valid_indices = [i for i, v in enumerate(values) if not np.isnan(v)]
            steps = [steps[i] for i in valid_indices]
            values = [values[i] for i in valid_indices]

            label = f"{exp_name} - {layer_or_model}"
            plt.plot(steps, values, marker='o', label=label, alpha=0.7)

    plt.xlabel('Training Steps')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(f'{metric.replace("_", " ").title()} Convergence Comparison')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.show()


def print_best_metrics_table(experiments: Dict[str, Path]):
    """
    Print a table comparing best metrics across experiments.

    Args:
        experiments: Dict mapping experiment names to evaluation directories
    """
    results = []

    for exp_name, eval_dir in experiments.items():
        metrics = load_experiment_metrics(eval_dir)

        for layer_or_model, layer_metrics in metrics.items():
            if not layer_metrics:
                continue

            # Find best metrics
            best_loss = min(m.get('loss', float('inf')) for m in layer_metrics)
            best_perplexity = min(m.get('perplexity', float('inf')) for m in layer_metrics)

            # Find step where best loss was achieved
            best_step = next(
                (m['step'] for m in layer_metrics if m.get('loss', float('inf')) == best_loss),
                0
            )

            # Get final metrics
            final_metrics = layer_metrics[-1] if layer_metrics else {}
            final_loss = final_metrics.get('loss', np.nan)
            final_perplexity = final_metrics.get('perplexity', np.nan)
            final_step = final_metrics.get('step', 0)

            results.append({
                'Experiment': exp_name,
                'Layer/Model': layer_or_model,
                'Best Loss': f"{best_loss:.4f}",
                'Best PPL': f"{best_perplexity:.2f}",
                'Best Step': best_step,
                'Final Loss': f"{final_loss:.4f}",
                'Final PPL': f"{final_perplexity:.2f}",
                'Final Step': final_step,
            })

    if results:
        df = pd.DataFrame(results)
        print("\n" + "="*80)
        print("Evaluation Metrics Comparison")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80 + "\n")

        # Print summary statistics
        print("Summary:")
        print(f"Total experiments compared: {len(experiments)}")
        print(f"Total layer/model configurations: {len(results)}")

        # Find overall best
        best_idx = df['Best Loss'].apply(lambda x: float(x)).idxmin()
        print(f"\nBest overall:")
        print(f"  Experiment: {df.loc[best_idx, 'Experiment']}")
        print(f"  Layer/Model: {df.loc[best_idx, 'Layer/Model']}")
        print(f"  Loss: {df.loc[best_idx, 'Best Loss']}")
        print(f"  Perplexity: {df.loc[best_idx, 'Best PPL']}")
        print(f"  Step: {df.loc[best_idx, 'Best Step']}")


def compare_learning_curves(
    experiments: Dict[str, Path],
    window_size: int = 10
):
    """
    Compare smoothed learning curves across experiments.

    Args:
        experiments: Dict mapping experiment names to evaluation directories
        window_size: Window size for moving average smoothing
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics_to_plot = [
        ('loss', axes[0, 0]),
        ('perplexity', axes[0, 1]),
        ('direct_mlp_loss', axes[1, 0]),
        ('fidelity_loss', axes[1, 1])
    ]

    for metric, ax in metrics_to_plot:
        for exp_name, eval_dir in experiments.items():
            metrics = load_experiment_metrics(eval_dir)

            for layer_or_model, layer_metrics in metrics.items():
                if not layer_metrics:
                    continue

                steps = [m['step'] for m in layer_metrics]
                values = [m.get(metric, np.nan) for m in layer_metrics]

                # Filter out NaN values
                valid_indices = [i for i, v in enumerate(values) if not np.isnan(v)]
                if not valid_indices:
                    continue

                steps = [steps[i] for i in valid_indices]
                values = [values[i] for i in valid_indices]

                # Apply moving average
                if len(values) > window_size:
                    smoothed = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
                    smoothed_steps = steps[window_size-1:]

                    label = f"{exp_name} - {layer_or_model}"
                    ax.plot(smoothed_steps, smoothed, label=label, alpha=0.8)
                else:
                    label = f"{exp_name} - {layer_or_model}"
                    ax.plot(steps, values, label=label, alpha=0.8)

        ax.set_xlabel('Training Steps')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} (Smoothed)')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.suptitle('Learning Curves Comparison', fontsize=14)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Compare evaluation metrics across NPT experiments"
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs='+',
        required=True,
        help="Paths to experiment directories (format: name:path)"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="loss",
        help="Primary metric to compare (loss, perplexity, etc.)"
    )
    parser.add_argument(
        "--save_plot",
        type=str,
        default=None,
        help="Path to save comparison plot"
    )
    parser.add_argument(
        "--show_curves",
        action="store_true",
        help="Show detailed learning curves comparison"
    )

    args = parser.parse_args()

    # Parse experiment paths
    experiments = {}
    for exp_spec in args.experiments:
        if ':' in exp_spec:
            name, path = exp_spec.split(':', 1)
        else:
            path = exp_spec
            name = Path(path).name

        eval_dir = Path(path) / "evaluation"
        if not eval_dir.exists():
            print(f"Warning: Evaluation directory not found: {eval_dir}")
            continue

        experiments[name] = eval_dir

    if not experiments:
        print("No valid experiment directories found!")
        return

    # Print comparison table
    print_best_metrics_table(experiments)

    # Plot convergence comparison
    plot_convergence_comparison(
        experiments,
        metric=args.metric,
        save_path=Path(args.save_plot) if args.save_plot else None
    )

    # Show detailed curves if requested
    if args.show_curves:
        compare_learning_curves(experiments)


if __name__ == "__main__":
    main()