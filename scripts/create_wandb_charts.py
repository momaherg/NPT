#!/usr/bin/env python3
"""
Create WandB charts for multi-layer NPT training.
This script shows how to programmatically create the recommended visualizations.
"""

import wandb
import argparse


def create_multi_layer_charts(project_name, entity=None):
    """Create recommended charts for multi-layer NPT training."""

    # Initialize WandB API
    api = wandb.Api()

    # Get the project
    if entity:
        project_path = f"{entity}/{project_name}"
    else:
        project_path = project_name

    print(f"Creating charts for project: {project_path}")

    # Chart configurations
    charts = [
        {
            "name": "1. Primary Losses",
            "metrics": [
                {"expression": "loss/total", "line_color": "#FF6B6B"},
                {"expression": "loss/fidelity", "line_color": "#4ECDC4"}
            ],
            "y_axis_title": "Loss",
            "description": "Overall training progress"
        },
        {
            "name": "2. Direct MLP Supervision (KEY METRIC)",
            "metrics": [
                {"expression": "mlp_direct_loss/layer_14", "line_color": "#FF6B6B"},
                {"expression": "mlp_direct_loss/layer_15", "line_color": "#4ECDC4"},
                {"expression": "mlp_direct_loss/layer_16", "line_color": "#95E77E"},
                {"expression": "mlp_direct_loss/layer_17", "line_color": "#FFE66D"}
            ],
            "y_axis_title": "Direct MLP Loss",
            "description": "Per-layer direct supervision loss - should converge evenly"
        },
        {
            "name": "3. Attention Encoding Quality",
            "metrics": [
                {"expression": "v_a_attention_similarity/layer_14", "line_color": "#FF6B6B"},
                {"expression": "v_a_attention_similarity/layer_15", "line_color": "#4ECDC4"},
                {"expression": "v_a_attention_similarity/layer_16", "line_color": "#95E77E"},
                {"expression": "v_a_attention_similarity/layer_17", "line_color": "#FFE66D"}
            ],
            "y_axis_title": "Cosine Similarity",
            "description": "How well v_a encodes attention (target > 0.7)"
        },
        {
            "name": "4. Curriculum Progress",
            "metrics": [
                {"expression": "curriculum/stage_index", "line_color": "#FF6B6B", "y_axis": "left"},
                {"expression": "curriculum/mixing_ratio", "line_color": "#4ECDC4", "y_axis": "right"}
            ],
            "y_axis_title": "Stage / Mixing Ratio",
            "description": "Training curriculum progression"
        },
        {
            "name": "5. v_a Vector Norms",
            "metrics": [
                {"expression": "v_a_norm/layer_14", "line_color": "#FF6B6B"},
                {"expression": "v_a_norm/layer_15", "line_color": "#4ECDC4"},
                {"expression": "v_a_norm/layer_16", "line_color": "#95E77E"},
                {"expression": "v_a_norm/layer_17", "line_color": "#FFE66D"}
            ],
            "y_axis_title": "Vector Norm",
            "description": "v_a norms - should stabilize, not collapse"
        },
        {
            "name": "6. v_b Vector Norms",
            "metrics": [
                {"expression": "v_b_norm/layer_14", "line_color": "#FF6B6B"},
                {"expression": "v_b_norm/layer_15", "line_color": "#4ECDC4"},
                {"expression": "v_b_norm/layer_16", "line_color": "#95E77E"},
                {"expression": "v_b_norm/layer_17", "line_color": "#FFE66D"}
            ],
            "y_axis_title": "Vector Norm",
            "description": "v_b norms - should stabilize, not collapse"
        },
        {
            "name": "7. Regularization Losses",
            "metrics": [
                {"expression": "regularization/layer_14", "line_color": "#FF6B6B"},
                {"expression": "regularization/layer_15", "line_color": "#4ECDC4"},
                {"expression": "regularization/layer_16", "line_color": "#95E77E"},
                {"expression": "regularization/layer_17", "line_color": "#FFE66D"}
            ],
            "y_axis_title": "Regularization Loss",
            "description": "L2 regularization on v_a and v_b"
        },
        {
            "name": "8. Training Hyperparameters",
            "metrics": [
                {"expression": "training/learning_rate", "line_color": "#FF6B6B", "y_axis": "left"},
                {"expression": "training/grad_norm", "line_color": "#4ECDC4", "y_axis": "right"}
            ],
            "y_axis_title": "LR / Grad Norm",
            "description": "Learning rate and gradient norm"
        },
        {
            "name": "9. Evaluation Losses",
            "metrics": [
                {"expression": "eval_loss/total", "line_color": "#FF6B6B"},
                {"expression": "eval_mlp_direct_loss/layer_14", "line_color": "#A8DADC"},
                {"expression": "eval_mlp_direct_loss/layer_15", "line_color": "#457B9D"},
                {"expression": "eval_mlp_direct_loss/layer_16", "line_color": "#1D3557"},
                {"expression": "eval_mlp_direct_loss/layer_17", "line_color": "#F1FAEE"}
            ],
            "y_axis_title": "Evaluation Loss",
            "description": "Validation losses"
        }
    ]

    print("\nRecommended WandB chart configurations:")
    print("=" * 60)

    for i, chart in enumerate(charts, 1):
        print(f"\n{chart['name']}")
        print(f"Description: {chart['description']}")
        print("Metrics to plot:")
        for metric in chart['metrics']:
            color_name = metric.get('line_color', 'default')
            axis = metric.get('y_axis', 'single')
            print(f"  - {metric['expression']} (color: {color_name}, axis: {axis})")

    print("\n" + "=" * 60)
    print("\nTo create these charts in WandB:")
    print("1. Go to your project dashboard")
    print("2. Click 'Add panel' -> 'Line plot'")
    print("3. Add the metrics listed above")
    print("4. Configure colors and axes as specified")
    print("5. Save the chart with the recommended name")

    return charts


def analyze_layer_balance(project_name, run_id=None, entity=None):
    """Analyze if layers are training in a balanced way."""

    api = wandb.Api()

    if entity:
        project_path = f"{entity}/{project_name}"
    else:
        project_path = project_name

    if run_id:
        run = api.run(f"{project_path}/{run_id}")
        print(f"\nAnalyzing run: {run.name}")

        # Get the history
        history = run.history()

        # Check MLP direct losses
        mlp_losses = {}
        for col in history.columns:
            if col.startswith('mlp_direct_loss/layer_'):
                layer_num = col.split('_')[-1]
                mlp_losses[layer_num] = history[col].dropna()

        if mlp_losses:
            print("\nLayer Balance Analysis:")
            print("-" * 40)

            # Get final values
            for layer, losses in mlp_losses.items():
                if len(losses) > 0:
                    final_loss = losses.iloc[-1]
                    mean_loss = losses.mean()
                    print(f"Layer {layer}: Final={final_loss:.4f}, Mean={mean_loss:.4f}")

            # Check variance
            final_values = [losses.iloc[-1] for losses in mlp_losses.values() if len(losses) > 0]
            if final_values:
                import numpy as np
                variance = np.var(final_values)
                std = np.std(final_values)
                print(f"\nVariance across layers: {variance:.6f}")
                print(f"Std deviation: {std:.4f}")

                if std > 0.1:
                    print("⚠️  High variance detected - consider adjusting layer_weights")
                else:
                    print("✅ Layers are training evenly")


def main():
    parser = argparse.ArgumentParser(description="Create WandB charts for multi-layer NPT")
    parser.add_argument("--project", type=str, default="npt-multi-layer",
                        help="WandB project name")
    parser.add_argument("--entity", type=str, default=None,
                        help="WandB entity (username or team)")
    parser.add_argument("--run", type=str, default=None,
                        help="Specific run ID to analyze")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze layer balance for a run")

    args = parser.parse_args()

    if args.analyze and args.run:
        analyze_layer_balance(args.project, args.run, args.entity)
    else:
        create_multi_layer_charts(args.project, args.entity)


if __name__ == "__main__":
    main()