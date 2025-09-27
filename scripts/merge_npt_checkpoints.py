#!/usr/bin/env python3
"""
Merge NPT checkpoints from different layer training runs.

This script allows you to combine NPT layers trained separately into a single
checkpoint that can be used for:
- Interactive knowledge transfer with multiple layers
- Continued training with multiple NPT layers
- Evaluation with combined layers

Example usage:
    python scripts/merge_npt_checkpoints.py \
        --checkpoint1 experiments/layer11/checkpoint-10000 \
        --checkpoint2 experiments/layer13/checkpoint-15000 \
        --output experiments/merged_11_13 \
        --verify
"""

import argparse
import torch
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NPTCheckpointMerger:
    """Handles merging of NPT checkpoints from different training runs."""

    def __init__(self):
        self.merged_state = {}
        self.merged_layers = set()
        self.layer_sources = {}  # Track which checkpoint each layer came from

    def load_checkpoint(self, checkpoint_path: Path) -> Tuple[Dict, Dict, Dict]:
        """
        Load NPT checkpoint components.

        Returns:
            Tuple of (npt_weights, training_state, config)
        """
        checkpoint_path = Path(checkpoint_path)

        # Load NPT weights
        npt_weights_path = checkpoint_path / "npt_weights.pt"
        if not npt_weights_path.exists():
            raise FileNotFoundError(f"NPT weights not found at {npt_weights_path}")

        npt_weights = torch.load(npt_weights_path, map_location='cpu', weights_only=False)
        logger.info(f"Loaded NPT weights from {npt_weights_path}")

        # Load training state if exists
        training_state_path = checkpoint_path / "training_state.pt"
        training_state = None
        if training_state_path.exists():
            training_state = torch.load(training_state_path, map_location='cpu', weights_only=False)
            logger.info(f"Loaded training state from {training_state_path}")

        # Load config if exists
        config_path = checkpoint_path / "config.json"
        config = None
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded config from {config_path}")

        return npt_weights, training_state, config

    def extract_layer_info(self, npt_weights: Dict) -> Set[int]:
        """
        Extract layer indices from NPT weights.

        Args:
            npt_weights: State dict with NPT weights

        Returns:
            Set of layer indices found in the checkpoint
        """
        layers = set()
        for key in npt_weights.keys():
            # Parse keys like "layer_11_np.W_down" or "model.layers.11.np_component.W_down"
            if 'layer_' in key and '_np' in key:
                parts = key.split('_')
                for i, part in enumerate(parts):
                    if part == 'layer' and i + 1 < len(parts):
                        try:
                            layer_idx = int(parts[i + 1].split('_')[0])
                            layers.add(layer_idx)
                        except ValueError:
                            pass
            elif 'model.layers.' in key and '.np_component' in key:
                # Handle format: model.layers.{idx}.np_component.{param}
                parts = key.split('.')
                for i, part in enumerate(parts):
                    if part == 'layers' and i + 1 < len(parts):
                        try:
                            layer_idx = int(parts[i + 1])
                            layers.add(layer_idx)
                        except ValueError:
                            pass

        return layers

    def merge_npt_weights(self, checkpoints: List[Tuple[Path, Dict]]) -> Dict:
        """
        Merge NPT weights from multiple checkpoints.

        Args:
            checkpoints: List of (path, npt_weights) tuples

        Returns:
            Merged NPT weights dictionary
        """
        merged_weights = {}

        for checkpoint_path, npt_weights in checkpoints:
            layers = self.extract_layer_info(npt_weights)
            logger.info(f"Found layers {sorted(layers)} in {checkpoint_path}")

            # Check for conflicts
            conflicting_layers = layers & self.merged_layers
            if conflicting_layers:
                logger.warning(
                    f"Layers {sorted(conflicting_layers)} already exist in merged checkpoint. "
                    f"Skipping from {checkpoint_path}"
                )
                layers = layers - conflicting_layers

            # Add weights for non-conflicting layers
            for key, value in npt_weights.items():
                # Check if this key belongs to one of our target layers
                layer_idx = self._get_layer_idx_from_key(key)
                if layer_idx is not None and layer_idx in layers:
                    merged_weights[key] = value
                    self.layer_sources[layer_idx] = str(checkpoint_path)

            self.merged_layers.update(layers)

        logger.info(f"Merged layers: {sorted(self.merged_layers)}")
        return merged_weights

    def _get_layer_idx_from_key(self, key: str) -> Optional[int]:
        """Extract layer index from a state dict key."""
        if 'layer_' in key and '_np' in key:
            parts = key.split('_')
            for i, part in enumerate(parts):
                if part == 'layer' and i + 1 < len(parts):
                    try:
                        return int(parts[i + 1].split('_')[0])
                    except ValueError:
                        pass
        elif 'model.layers.' in key and '.np_component' in key:
            parts = key.split('.')
            for i, part in enumerate(parts):
                if part == 'layers' and i + 1 < len(parts):
                    try:
                        return int(parts[i + 1])
                    except ValueError:
                        pass
        return None

    def merge_training_states(self, training_states: List[Tuple[Path, Optional[Dict]]]) -> Dict:
        """
        Merge training states, choosing the one with most steps.

        Args:
            training_states: List of (path, training_state) tuples

        Returns:
            Merged training state
        """
        # Filter out None states
        valid_states = [(p, s) for p, s in training_states if s is not None]

        if not valid_states:
            logger.warning("No valid training states found")
            return {}

        # Choose the state with the highest global_step
        best_state = None
        best_step = -1
        best_path = None

        for path, state in valid_states:
            if 'global_step' in state:
                step = state['global_step']
                if step > best_step:
                    best_step = step
                    best_state = state
                    best_path = path

        if best_state:
            logger.info(f"Using training state from {best_path} (step {best_step})")
            # Add metadata about the merge
            best_state['merge_info'] = {
                'merged_from': [str(p) for p, _ in training_states],
                'merged_layers': sorted(list(self.merged_layers)),
                'layer_sources': self.layer_sources
            }
            return best_state
        else:
            return {}

    def merge_configs(self, configs: List[Tuple[Path, Optional[Dict]]]) -> Dict:
        """
        Merge configuration files.

        Args:
            configs: List of (path, config) tuples

        Returns:
            Merged configuration
        """
        # Filter out None configs
        valid_configs = [(p, c) for p, c in configs if c is not None]

        if not valid_configs:
            logger.warning("No valid configs found")
            return {}

        # Use the first config as base
        base_path, base_config = valid_configs[0]
        logger.info(f"Using config from {base_path} as base")

        # Add merge metadata
        base_config['merge_info'] = {
            'merged_from': [str(p) for p, _ in configs],
            'merged_layers': sorted(list(self.merged_layers)),
            'layer_sources': self.layer_sources
        }

        return base_config

    def save_merged_checkpoint(self, output_path: Path, merged_weights: Dict,
                             merged_training_state: Dict, merged_config: Dict):
        """Save the merged checkpoint."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save NPT weights
        npt_weights_path = output_path / "npt_weights.pt"
        torch.save(merged_weights, npt_weights_path)
        logger.info(f"Saved merged NPT weights to {npt_weights_path}")

        # Save training state
        if merged_training_state:
            training_state_path = output_path / "training_state.pt"
            torch.save(merged_training_state, training_state_path)
            logger.info(f"Saved merged training state to {training_state_path}")

        # Save config
        if merged_config:
            config_path = output_path / "config.json"
            with open(config_path, 'w') as f:
                json.dump(merged_config, f, indent=2)
            logger.info(f"Saved merged config to {config_path}")

        # Save merge summary
        summary_path = output_path / "merge_summary.json"
        summary = {
            'merged_layers': sorted(list(self.merged_layers)),
            'layer_sources': self.layer_sources,
            'total_parameters': sum(
                p.numel() for p in merged_weights.values()
            ),
            'checkpoint_keys': len(merged_weights)
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved merge summary to {summary_path}")

    def verify_merge(self, output_path: Path):
        """Verify the merged checkpoint."""
        output_path = Path(output_path)

        # Load and verify NPT weights
        npt_weights_path = output_path / "npt_weights.pt"
        if not npt_weights_path.exists():
            logger.error(f"Verification failed: {npt_weights_path} not found")
            return False

        npt_weights = torch.load(npt_weights_path, map_location='cpu', weights_only=False)
        found_layers = self.extract_layer_info(npt_weights)

        logger.info("Verification Results:")
        logger.info(f"  Expected layers: {sorted(self.merged_layers)}")
        logger.info(f"  Found layers: {sorted(found_layers)}")
        logger.info(f"  Checkpoint has {len(npt_weights)} keys")

        # Check if all expected layers are present
        missing_layers = self.merged_layers - found_layers
        if missing_layers:
            logger.error(f"  Missing layers: {sorted(missing_layers)}")
            return False

        # Check parameter shapes for each layer
        logger.info("  Parameter shapes by layer:")
        for layer_idx in sorted(found_layers):
            layer_keys = [k for k in npt_weights.keys()
                         if self._get_layer_idx_from_key(k) == layer_idx]
            logger.info(f"    Layer {layer_idx}: {len(layer_keys)} parameters")

            # Sample some key parameters
            for key in layer_keys[:3]:  # Show first 3 parameters
                shape = list(npt_weights[key].shape)
                logger.info(f"      {key.split('.')[-1]}: {shape}")

        logger.info("✓ Verification passed!")
        return True


def main():
    parser = argparse.ArgumentParser(description="Merge NPT checkpoints from different layers")

    # Required arguments
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="Paths to checkpoints to merge (can specify multiple)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for merged checkpoint"
    )

    # Optional arguments
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the merged checkpoint after saving"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output directory if it exists"
    )
    parser.add_argument(
        "--copy-multi-layer-state",
        type=str,
        help="Copy multi_layer_state.pt from a multi-layer training checkpoint"
    )

    args = parser.parse_args()

    # Setup paths
    checkpoint_paths = [Path(cp) for cp in args.checkpoints]
    output_path = Path(args.output)

    # Check inputs
    for cp in checkpoint_paths:
        if not cp.exists():
            logger.error(f"Checkpoint not found: {cp}")
            return 1

    # Check output
    if output_path.exists() and not args.force:
        logger.error(f"Output path already exists: {output_path}. Use --force to overwrite.")
        return 1

    # Create merger
    merger = NPTCheckpointMerger()

    # Load all checkpoints
    logger.info(f"\n{'='*60}")
    logger.info(f"Loading {len(checkpoint_paths)} checkpoints...")
    logger.info(f"{'='*60}")

    checkpoints_data = []
    training_states = []
    configs = []

    for cp_path in checkpoint_paths:
        logger.info(f"\nLoading checkpoint: {cp_path}")
        npt_weights, training_state, config = merger.load_checkpoint(cp_path)
        checkpoints_data.append((cp_path, npt_weights))
        training_states.append((cp_path, training_state))
        configs.append((cp_path, config))

    # Merge components
    logger.info(f"\n{'='*60}")
    logger.info("Merging checkpoints...")
    logger.info(f"{'='*60}")

    merged_weights = merger.merge_npt_weights(checkpoints_data)
    merged_training_state = merger.merge_training_states(training_states)
    merged_config = merger.merge_configs(configs)

    # Copy multi-layer state if specified
    if args.copy_multi_layer_state:
        multi_layer_path = Path(args.copy_multi_layer_state)
        if multi_layer_path.exists():
            multi_layer_state = torch.load(multi_layer_path, map_location='cpu', weights_only=False)
            # Update with our merged layers
            multi_layer_state['layers_to_train'] = sorted(list(merger.merged_layers))

            # Save to output
            output_multi_layer_path = output_path / "multi_layer_state.pt"
            output_path.mkdir(parents=True, exist_ok=True)
            torch.save(multi_layer_state, output_multi_layer_path)
            logger.info(f"Copied and updated multi_layer_state to {output_multi_layer_path}")

    # Save merged checkpoint
    logger.info(f"\n{'='*60}")
    logger.info(f"Saving merged checkpoint to {output_path}")
    logger.info(f"{'='*60}")

    merger.save_merged_checkpoint(output_path, merged_weights,
                                 merged_training_state, merged_config)

    # Verify if requested
    if args.verify:
        logger.info(f"\n{'='*60}")
        logger.info("Verifying merged checkpoint...")
        logger.info(f"{'='*60}")
        if not merger.verify_merge(output_path):
            return 1

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("Merge Summary")
    logger.info(f"{'='*60}")
    logger.info(f"Merged layers: {sorted(merger.merged_layers)}")
    logger.info(f"Total NPT parameters: {sum(p.numel() for p in merged_weights.values()):,}")
    logger.info(f"Output saved to: {output_path}")
    logger.info("\nLayer sources:")
    for layer_idx in sorted(merger.layer_sources.keys()):
        logger.info(f"  Layer {layer_idx}: {merger.layer_sources[layer_idx]}")

    logger.info(f"\n✓ Successfully merged {len(merger.merged_layers)} NPT layers!")

    # Print usage instructions
    logger.info(f"\n{'='*60}")
    logger.info("Usage Instructions")
    logger.info(f"{'='*60}")
    logger.info("1. For interactive knowledge transfer:")
    logger.info(f"   python scripts/interactive_knowledge_transfer_tool.py \\")
    logger.info(f"     --checkpoint {output_path} \\")
    logger.info(f"     --layers {','.join(map(str, sorted(merger.merged_layers)))}")
    logger.info("\n2. For continued training:")
    logger.info(f"   python scripts/train_multi_layer_npt.py \\")
    logger.info(f"     --resume_from {output_path} \\")
    logger.info(f"     --train_layers {','.join(map(str, sorted(merger.merged_layers)))}")

    return 0


if __name__ == "__main__":
    exit(main())