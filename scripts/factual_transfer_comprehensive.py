#!/usr/bin/env python3
"""
Comprehensive Per-Layer Factual Knowledge Transfer Analysis

Extended version with more diverse experiments testing all NPT layers 1-15.
Tests various types of semantic relationships:
- Geographic (capitals, locations)
- Scientific (planets, elements)
- Literary (authors, works)
- Historical (presidents, events)
- Mathematical (operations, results)
- Linguistic (languages, countries)
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

from factual_transfer_per_layer_analysis import (
    LayerTransferResult,
    PerLayerAnalyzer
)


def get_extended_experiments(use_context: bool = False) -> List[Dict[str, str]]:
    """Get extended set of diverse experiments."""

    if use_context:
        experiments = [
            # Geographic - Capitals
            {
                'category': 'Geographic',
                'source_prompt': "The capital of France is Paris. Given that the capital of France is",
                'source_answer': " Paris",
                'target_prompt': "The capital of Germany is",
                'target_answer': " Berlin"
            },
            {
                'category': 'Geographic',
                'source_prompt': "The capital of Japan is Tokyo. The capital city of Japan is",
                'source_answer': " Tokyo",
                'target_prompt': "The capital of China is",
                'target_answer': " Beijing"
            },

            # Scientific - Planets
            {
                'category': 'Scientific',
                'source_prompt': "The largest planet is Jupiter. The largest planet in our solar system is",
                'source_answer': " Jupiter",
                'target_prompt': "The smallest planet is",
                'target_answer': " Mercury"
            },
            {
                'category': 'Scientific',
                'source_prompt': "The hottest planet is Venus. The hottest planet in our solar system is",
                'source_answer': " Venus",
                'target_prompt': "The coldest planet is",
                'target_answer': " Neptune"
            },

            # Literary - Authors
            {
                'category': 'Literary',
                'source_prompt': "Shakespeare wrote Romeo and Juliet. Shakespeare famously wrote",
                'source_answer': " Romeo",
                'target_prompt': "Dickens wrote",
                'target_answer': " Oliver"
            },
            {
                'category': 'Literary',
                'source_prompt': "Tolkien wrote The Lord of the Rings. Tolkien is famous for writing",
                'source_answer': " Lord",
                'target_prompt': "Rowling wrote",
                'target_answer': " Harry"
            },

            # Historical - Leaders
            {
                'category': 'Historical',
                'source_prompt': "The first president of the USA was George Washington. The first US president was",
                'source_answer': " George",
                'target_prompt': "The first prime minister of the UK was",
                'target_answer': " Robert"
            },
            {
                'category': 'Historical',
                'source_prompt': "The founder of Microsoft is Bill Gates. Microsoft was founded by",
                'source_answer': " Bill",
                'target_prompt': "Apple was founded by",
                'target_answer': " Steve"
            },

            # Scientific - Elements
            {
                'category': 'Scientific',
                'source_prompt': "Gold has the symbol Au. The chemical symbol for gold is",
                'source_answer': " Au",
                'target_prompt': "The chemical symbol for silver is",
                'target_answer': " Ag"
            },
            {
                'category': 'Scientific',
                'source_prompt': "Water is H2O. The chemical formula for water is",
                'source_answer': " H",
                'target_prompt': "The chemical formula for carbon dioxide is",
                'target_answer': " CO"
            },

            # Mathematical
            {
                'category': 'Mathematical',
                'source_prompt': "Two plus two equals four. The sum of 2 and 2 is",
                'source_answer': " 4",
                'target_prompt': "The sum of 3 and 3 is",
                'target_answer': " 6"
            },
            {
                'category': 'Mathematical',
                'source_prompt': "Pi is approximately 3.14. The value of pi is approximately",
                'source_answer': " 3",
                'target_prompt': "The value of e is approximately",
                'target_answer': " 2"
            },

            # Linguistic
            {
                'category': 'Linguistic',
                'source_prompt': "In France, people speak French. The language spoken in France is",
                'source_answer': " French",
                'target_prompt': "The language spoken in Germany is",
                'target_answer': " German"
            },
            {
                'category': 'Linguistic',
                'source_prompt': "Hello in Spanish is Hola. The Spanish word for hello is",
                'source_answer': " Hola",
                'target_prompt': "The French word for hello is",
                'target_answer': " Bon"
            },

            # Sports
            {
                'category': 'Sports',
                'source_prompt': "Brazil has won 5 World Cups. Brazil has won the World Cup",
                'source_answer': " 5",
                'target_prompt': "Germany has won the World Cup",
                'target_answer': " 4"
            },
            {
                'category': 'Sports',
                'source_prompt': "Michael Jordan wore number 23. Jordan's jersey number was",
                'source_answer': " 23",
                'target_prompt': "LeBron's jersey number was",
                'target_answer': " 23"
            }
        ]
    else:
        # Simple prompts without context
        experiments = [
            # Geographic
            {
                'category': 'Geographic',
                'source_prompt': "The capital of France is",
                'source_answer': " Paris",
                'target_prompt': "The capital of Germany is",
                'target_answer': " Berlin"
            },
            {
                'category': 'Geographic',
                'source_prompt': "The capital of Japan is",
                'source_answer': " Tokyo",
                'target_prompt': "The capital of China is",
                'target_answer': " Beijing"
            },

            # Scientific
            {
                'category': 'Scientific',
                'source_prompt': "The largest planet is",
                'source_answer': " Jupiter",
                'target_prompt': "The smallest planet is",
                'target_answer': " Mercury"
            },
            {
                'category': 'Scientific',
                'source_prompt': "The hottest planet is",
                'source_answer': " Venus",
                'target_prompt': "The coldest planet is",
                'target_answer': " Neptune"
            },

            # Literary
            {
                'category': 'Literary',
                'source_prompt': "Shakespeare wrote",
                'source_answer': " Romeo",
                'target_prompt': "Dickens wrote",
                'target_answer': " Oliver"
            },
            {
                'category': 'Literary',
                'source_prompt': "Tolkien wrote",
                'source_answer': " The",
                'target_prompt': "Rowling wrote",
                'target_answer': " Harry"
            },

            # Historical
            {
                'category': 'Historical',
                'source_prompt': "The first US president was",
                'source_answer': " George",
                'target_prompt': "The first UK prime minister was",
                'target_answer': " Robert"
            },
            {
                'category': 'Historical',
                'source_prompt': "Microsoft was founded by",
                'source_answer': " Bill",
                'target_prompt': "Apple was founded by",
                'target_answer': " Steve"
            },

            # Elements
            {
                'category': 'Scientific',
                'source_prompt': "The symbol for gold is",
                'source_answer': " Au",
                'target_prompt': "The symbol for silver is",
                'target_answer': " Ag"
            },
            {
                'category': 'Scientific',
                'source_prompt': "The formula for water is",
                'source_answer': " H",
                'target_prompt': "The formula for carbon dioxide is",
                'target_answer': " CO"
            },

            # Mathematical
            {
                'category': 'Mathematical',
                'source_prompt': "Two plus two equals",
                'source_answer': " four",
                'target_prompt': "Three plus three equals",
                'target_answer': " six"
            },
            {
                'category': 'Mathematical',
                'source_prompt': "Pi is approximately",
                'source_answer': " 3",
                'target_prompt': "E is approximately",
                'target_answer': " 2"
            },

            # Linguistic
            {
                'category': 'Linguistic',
                'source_prompt': "People in France speak",
                'source_answer': " French",
                'target_prompt': "People in Germany speak",
                'target_answer': " German"
            },
            {
                'category': 'Linguistic',
                'source_prompt': "Hello in Spanish is",
                'source_answer': " Hola",
                'target_prompt': "Hello in French is",
                'target_answer': " Bon"
            },

            # Sports
            {
                'category': 'Sports',
                'source_prompt': "Brazil won World Cups",
                'source_answer': " 5",
                'target_prompt': "Germany won World Cups",
                'target_answer': " 4"
            },
            {
                'category': 'Sports',
                'source_prompt': "Jordan wore number",
                'source_answer': " 23",
                'target_prompt': "LeBron wore number",
                'target_answer': " 23"
            }
        ]

    return experiments


def create_comprehensive_visualization(df: pd.DataFrame, output_dir: Path):
    """Create comprehensive visualizations for all layers and experiments."""

    # Set up the plot style
    sns.set_style("whitegrid")

    # 1. Heatmap of target probability changes across all layers and experiments
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))

    # Prepare data for heatmap
    pivot_target = df.pivot_table(
        index='Layer',
        columns='Experiment',
        values='Target Shift (%)',
        aggfunc='mean'
    )

    # Main heatmap
    ax = axes[0, 0]
    sns.heatmap(
        pivot_target,
        annot=True,
        fmt='.1f',
        cmap='RdBu_r',
        center=0,
        ax=ax,
        vmin=-100,
        vmax=100,
        cbar_kws={'label': 'Probability Shift (%)'}
    )
    ax.set_title('Target Token Probability Shift by Layer', fontsize=14, fontweight='bold')
    ax.set_xlabel('Experiment Number')
    ax.set_ylabel('Layer')

    # Source probability heatmap
    pivot_source = df.pivot_table(
        index='Layer',
        columns='Experiment',
        values='Source Shift (%)',
        aggfunc='mean'
    )

    ax = axes[0, 1]
    sns.heatmap(
        pivot_source,
        annot=True,
        fmt='.1f',
        cmap='RdBu_r',
        center=0,
        ax=ax,
        vmin=-100,
        vmax=100,
        cbar_kws={'label': 'Probability Shift (%)'}
    )
    ax.set_title('Source Token Probability Shift by Layer', fontsize=14, fontweight='bold')
    ax.set_xlabel('Experiment Number')
    ax.set_ylabel('Layer')

    # KL Divergence heatmap
    pivot_kl = df.pivot_table(
        index='Layer',
        columns='Experiment',
        values='KL Divergence',
        aggfunc='mean'
    )

    ax = axes[0, 2]
    sns.heatmap(
        pivot_kl,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        ax=ax,
        cbar_kws={'label': 'KL Divergence'}
    )
    ax.set_title('KL Divergence by Layer', fontsize=14, fontweight='bold')
    ax.set_xlabel('Experiment Number')
    ax.set_ylabel('Layer')

    # Average effects by layer
    ax = axes[1, 0]
    layer_avg = df.groupby('Layer').agg({
        'Target Shift (%)': 'mean',
        'Source Shift (%)': 'mean'
    })
    layer_avg.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title('Average Probability Shifts by Layer', fontsize=14, fontweight='bold')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Average Shift (%)')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.legend(['Target Token', 'Source Token'])
    ax.grid(True, alpha=0.3)

    # Effects by category
    if 'Category' in df.columns:
        ax = axes[1, 1]
        category_avg = df.groupby('Category').agg({
            'Target Shift (%)': 'mean',
            'Source Shift (%)': 'mean'
        })
        category_avg.plot(kind='barh', ax=ax)
        ax.set_title('Average Shifts by Category', fontsize=14, fontweight='bold')
        ax.set_xlabel('Average Shift (%)')
        ax.set_ylabel('Category')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.legend(['Target Token', 'Source Token'])

    # Layer effectiveness ranking
    ax = axes[1, 2]
    layer_effectiveness = df.groupby('Layer')['Target Shift (%)'].agg(['mean', 'std'])
    layer_effectiveness = layer_effectiveness.sort_values('mean', ascending=False)

    ax.barh(range(len(layer_effectiveness)), layer_effectiveness['mean'],
            xerr=layer_effectiveness['std'], capsize=3)
    ax.set_yticks(range(len(layer_effectiveness)))
    ax.set_yticklabels([f"Layer {l}" for l in layer_effectiveness.index])
    ax.set_xlabel('Average Target Token Shift (%)')
    ax.set_title('Layer Ranking by Target Enhancement', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='x')

    plt.suptitle('Comprehensive NPT Layer Transfer Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'comprehensive_layer_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Create layer progression plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot progression for selected experiments
    num_experiments = df['Experiment'].nunique()
    if num_experiments >= 5:
        selected_exps = [1, 3, 5]  # Capitals, Planets, Authors
    else:
        selected_exps = list(range(1, min(4, num_experiments + 1)))
    colors = plt.cm.Set2(np.linspace(0, 1, len(selected_exps)))

    for ax_idx, (metric_col, metric_name) in enumerate([
        ('Target Shift (%)', 'Target Token Shift'),
        ('Source Shift (%)', 'Source Token Shift'),
        ('KL Divergence', 'KL Divergence')
    ]):
        ax = axes[ax_idx]
        for i, exp_num in enumerate(selected_exps):
            exp_data = df[df['Experiment'] == exp_num].sort_values('Layer')
            ax.plot(exp_data['Layer'], exp_data[metric_col],
                   marker='o', label=f'Exp {exp_num}', color=colors[i], linewidth=2)

        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f'{metric_name} Progression', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        if 'Shift' in metric_name:
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    plt.suptitle('Layer-by-Layer Transfer Progression', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'layer_progression.png', dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved visualizations to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Factual Transfer Analysis")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to NPT checkpoint")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_context", action="store_true", help="Use contextual prompts")
    parser.add_argument("--output_dir", type=str, default="experiments/comprehensive_analysis")
    parser.add_argument("--quick_test", action="store_true", help="Quick test with fewer experiments")

    args = parser.parse_args()

    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create analyzer
    analyzer = PerLayerAnalyzer(
        checkpoint_path=args.checkpoint,
        model_name=args.model_name,
        device=args.device
    )

    # Get experiments
    experiments = get_extended_experiments(use_context=args.use_context)

    # For quick test, use only first 3 experiments
    if args.quick_test:
        experiments = experiments[:3]
        layers_to_test = [1, 8, 15]  # Sample layers
    else:
        layers_to_test = list(range(1, 16))  # All layers 1-15

    logger.info(f"Testing {len(experiments)} experiments across {len(layers_to_test)} layers")
    logger.info(f"Total combinations: {len(experiments) * len(layers_to_test)}")

    # Run analysis
    results = analyzer.run_comprehensive_analysis(
        experiments=experiments,
        use_context=args.use_context,
        layers_to_test=layers_to_test
    )

    # Generate summary DataFrame with categories
    rows = []
    for exp_key, exp_data in results.items():
        exp_num = int(exp_key.split('_')[1]) + 1

        # Get category if available
        category = experiments[exp_num - 1].get('category', 'Unknown')

        for layer_idx, layer_result in exp_data['layer_results'].items():
            rows.append({
                'Experiment': exp_num,
                'Category': category,
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
    df = df.sort_values(['Layer', 'Experiment'])

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    df.to_csv(output_dir / 'comprehensive_results.csv', index=False)

    with open(output_dir / 'experiment_details.json', 'w') as f:
        json.dump(experiments, f, indent=2)

    # Generate visualizations
    create_comprehensive_visualization(df, output_dir)

    # Print comprehensive summary
    logger.info("\n" + "="*80)
    logger.info("COMPREHENSIVE ANALYSIS SUMMARY")
    logger.info("="*80)

    # Best layers for target enhancement
    target_summary = df.groupby('Layer')['Target Shift (%)'].agg(['mean', 'std', 'min', 'max'])
    target_summary = target_summary.sort_values('mean', ascending=False)

    logger.info("\n### TOP LAYERS FOR TARGET TOKEN ENHANCEMENT ###")
    for layer in target_summary.head(5).index:
        stats = target_summary.loc[layer]
        logger.info(f"  Layer {layer}: mean={stats['mean']:+.1f}%, std={stats['std']:.1f}, "
                   f"range=[{stats['min']:+.1f}, {stats['max']:+.1f}]")

    # Best layers for source preservation
    source_summary = df.groupby('Layer')['Source Shift (%)'].agg(['mean', 'std', 'min', 'max'])
    source_summary = source_summary.sort_values('mean', ascending=False)

    logger.info("\n### TOP LAYERS FOR SOURCE TOKEN PRESERVATION ###")
    for layer in source_summary.head(5).index:
        stats = source_summary.loc[layer]
        logger.info(f"  Layer {layer}: mean={stats['mean']:+.1f}%, std={stats['std']:.1f}, "
                   f"range=[{stats['min']:+.1f}, {stats['max']:+.1f}]")

    # Category analysis
    if 'Category' in df.columns:
        category_summary = df.groupby('Category')['Target Shift (%)'].agg(['mean', 'std'])
        category_summary = category_summary.sort_values('mean', ascending=False)

        logger.info("\n### EFFECTIVENESS BY CATEGORY ###")
        for category in category_summary.index:
            stats = category_summary.loc[category]
            logger.info(f"  {category}: mean={stats['mean']:+.1f}%, std={stats['std']:.1f}")

    # Layer specialization insights
    logger.info("\n### LAYER SPECIALIZATION INSIGHTS ###")

    # Early layers (1-5)
    early_layers = df[df['Layer'].isin(range(1, 6))]
    logger.info(f"  Layers 1-5: Target mean={early_layers['Target Shift (%)'].mean():+.1f}%, "
               f"Source mean={early_layers['Source Shift (%)'].mean():+.1f}%")

    # Middle layers (6-10)
    middle_layers = df[df['Layer'].isin(range(6, 11))]
    logger.info(f"  Layers 6-10: Target mean={middle_layers['Target Shift (%)'].mean():+.1f}%, "
               f"Source mean={middle_layers['Source Shift (%)'].mean():+.1f}%")

    # Late layers (11-15)
    late_layers = df[df['Layer'].isin(range(11, 16))]
    logger.info(f"  Layers 11-15: Target mean={late_layers['Target Shift (%)'].mean():+.1f}%, "
               f"Source mean={late_layers['Source Shift (%)'].mean():+.1f}%")

    logger.info(f"\nResults saved to {output_dir}")
    logger.info("  - comprehensive_results.csv: Full data table")
    logger.info("  - experiment_details.json: Experiment configurations")
    logger.info("  - comprehensive_layer_analysis.png: Main visualization")
    logger.info("  - layer_progression.png: Layer progression plots")


if __name__ == "__main__":
    main()