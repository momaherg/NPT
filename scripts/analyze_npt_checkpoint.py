#!/usr/bin/env python3
"""
Analyze NPT checkpoint to understand its structure and contents.

This is useful before merging checkpoints to understand what layers are present
and their configurations.

Example usage:
    python scripts/analyze_npt_checkpoint.py \
        --checkpoint experiments/layer11/checkpoint-10000
"""

import argparse
import torch
import json
from pathlib import Path
from typing import Dict, Set, Optional
import numpy as np


def analyze_npt_weights(npt_weights: Dict) -> Dict:
    """Analyze NPT weights structure."""
    analysis = {
        'total_keys': len(npt_weights),
        'total_parameters': 0,
        'layers': {},
        'parameter_types': set(),
        'tensor_shapes': {}
    }

    # Group by layer
    layer_params = {}
    for key, tensor in npt_weights.items():
        # Extract layer index
        layer_idx = None
        if 'layer_' in key and '_np' in key:
            parts = key.split('_')
            for i, part in enumerate(parts):
                if part == 'layer' and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1].split('_')[0])
                    except ValueError:
                        pass

        if layer_idx is not None:
            if layer_idx not in layer_params:
                layer_params[layer_idx] = []
            layer_params[layer_idx].append((key, tensor))

            # Extract parameter name
            param_name = key.split('.')[-1]
            analysis['parameter_types'].add(param_name)

    # Analyze each layer
    for layer_idx, params in sorted(layer_params.items()):
        layer_info = {
            'num_parameters': len(params),
            'total_elements': 0,
            'parameters': {}
        }

        for key, tensor in params:
            param_name = key.split('.')[-1]
            num_elements = tensor.numel()
            layer_info['parameters'][param_name] = {
                'shape': list(tensor.shape),
                'dtype': str(tensor.dtype),
                'elements': num_elements,
                'size_mb': num_elements * tensor.element_size() / (1024 * 1024)
            }
            layer_info['total_elements'] += num_elements
            analysis['total_parameters'] += num_elements

        analysis['layers'][layer_idx] = layer_info

    return analysis


def detect_modulation_type(npt_weights: Dict) -> str:
    """Detect the type of modulation (single/dual/triple)."""
    has_gate = any('W_down_gate' in k or 'W_a_up_gate' in k for k in npt_weights.keys())
    has_down = any('W_down_down' in k or 'W_a_up_down' in k for k in npt_weights.keys())

    if has_down:
        return "triple"
    elif has_gate:
        return "dual"
    else:
        return "single"


def detect_num_ranks(npt_weights: Dict) -> int:
    """Detect the number of rank-1 components."""
    max_rank = 1
    for key in npt_weights.keys():
        # Look for patterns like W_down.0, W_down.1, etc.
        if '.W_down.' in key or '.W_a_up.' in key:
            parts = key.split('.')
            for part in parts:
                if part.isdigit():
                    max_rank = max(max_rank, int(part) + 1)
    return max_rank


def analyze_training_state(training_state: Dict) -> Dict:
    """Analyze training state contents."""
    analysis = {}

    if 'global_step' in training_state:
        analysis['global_step'] = training_state['global_step']

    if 'batch_count' in training_state:
        analysis['batch_count'] = training_state['batch_count']

    if 'optimizer_state_dict' in training_state:
        opt_state = training_state['optimizer_state_dict']
        analysis['optimizer'] = {
            'num_param_groups': len(opt_state.get('param_groups', [])),
        }
        if 'param_groups' in opt_state and opt_state['param_groups']:
            pg = opt_state['param_groups'][0]
            analysis['optimizer']['learning_rate'] = pg.get('lr', 'N/A')

    if 'scheduler_state_dict' in training_state:
        analysis['has_scheduler'] = True

    if 'scaler_state_dict' in training_state:
        analysis['mixed_precision'] = True

    # Check for multi-layer training info
    if 'merge_info' in training_state:
        analysis['merge_info'] = training_state['merge_info']

    return analysis


def format_size(num_bytes: float) -> str:
    """Format size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"


def main():
    parser = argparse.ArgumentParser(description="Analyze NPT checkpoint structure")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint to analyze"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed parameter information"
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return 1

    print(f"\n{'='*60}")
    print(f"NPT Checkpoint Analysis")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}\n")

    # Load NPT weights
    npt_weights_path = checkpoint_path / "npt_weights.pt"
    if npt_weights_path.exists():
        print("Loading NPT weights...")
        npt_weights = torch.load(npt_weights_path, map_location='cpu', weights_only=False)

        # Analyze structure
        analysis = analyze_npt_weights(npt_weights)

        print(f"\n{'-'*40}")
        print("NPT Weights Analysis")
        print(f"{'-'*40}")
        print(f"Total keys: {analysis['total_keys']}")
        print(f"Total parameters: {analysis['total_parameters']:,}")
        print(f"Modulation type: {detect_modulation_type(npt_weights)}")
        print(f"Number of ranks: {detect_num_ranks(npt_weights)}")

        # Detect np_rank
        np_rank = None
        for key, value in npt_weights.items():
            if 'W_down' in key and isinstance(value, torch.Tensor) and value.dim() == 2:
                np_rank = value.shape[1]
                break
        if np_rank:
            print(f"NPT rank: {np_rank}")

        print(f"\nLayers found: {sorted(analysis['layers'].keys())}")

        for layer_idx in sorted(analysis['layers'].keys()):
            layer_info = analysis['layers'][layer_idx]
            print(f"\nLayer {layer_idx}:")
            print(f"  Parameters: {layer_info['num_parameters']}")
            print(f"  Total elements: {layer_info['total_elements']:,}")

            if args.verbose:
                print("  Parameter details:")
                for param_name, param_info in sorted(layer_info['parameters'].items()):
                    print(f"    {param_name}:")
                    print(f"      Shape: {param_info['shape']}")
                    print(f"      Size: {param_info['size_mb']:.2f} MB")

    else:
        print(f"Warning: NPT weights not found at {npt_weights_path}")

    # Load training state
    training_state_path = checkpoint_path / "training_state.pt"
    if training_state_path.exists():
        print(f"\n{'-'*40}")
        print("Training State Analysis")
        print(f"{'-'*40}")
        training_state = torch.load(training_state_path, map_location='cpu', weights_only=False)
        state_analysis = analyze_training_state(training_state)

        for key, value in state_analysis.items():
            if key == 'merge_info':
                print(f"\nMerge Information:")
                print(f"  Merged from: {value.get('merged_from', 'N/A')}")
                print(f"  Merged layers: {value.get('merged_layers', 'N/A')}")
            else:
                print(f"{key}: {value}")

    # Load config
    config_path = checkpoint_path / "config.json"
    if config_path.exists():
        print(f"\n{'-'*40}")
        print("Configuration")
        print(f"{'-'*40}")
        with open(config_path, 'r') as f:
            config = json.load(f)

        important_keys = ['model_name', 'batch_size', 'learning_rate', 'max_steps']
        for key in important_keys:
            if key in config:
                print(f"{key}: {config[key]}")

        if 'merge_info' in config:
            print(f"\nMerge Information in config:")
            merge_info = config['merge_info']
            print(f"  Merged layers: {merge_info.get('merged_layers', 'N/A')}")
            if 'layer_sources' in merge_info:
                print("  Layer sources:")
                for layer, source in merge_info['layer_sources'].items():
                    print(f"    Layer {layer}: {source}")

    # Load multi-layer state if exists
    multi_layer_path = checkpoint_path / "multi_layer_state.pt"
    if multi_layer_path.exists():
        print(f"\n{'-'*40}")
        print("Multi-Layer Training State")
        print(f"{'-'*40}")
        multi_state = torch.load(multi_layer_path, map_location='cpu', weights_only=False)

        if 'layers_to_train' in multi_state:
            print(f"Layers to train: {multi_state['layers_to_train']}")
        if 'curriculum_schedule' in multi_state:
            print(f"Curriculum stages: {len(multi_state['curriculum_schedule'])}")
            for stage in multi_state['curriculum_schedule']:
                print(f"  - {stage['name']}: until step {stage['until_step']}")

    # Check for other files
    print(f"\n{'-'*40}")
    print("Other Files in Checkpoint")
    print(f"{'-'*40}")
    for file_path in checkpoint_path.glob("*"):
        if file_path.is_file():
            size = file_path.stat().st_size
            print(f"  {file_path.name}: {format_size(size)}")

    print(f"\n{'='*60}\n")

    return 0


if __name__ == "__main__":
    exit(main())