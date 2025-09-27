#!/usr/bin/env python3
"""
Test the merge functionality with dummy checkpoints.
This demonstrates how the merge script works without needing actual trained models.
"""

import torch
import json
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.merge_npt_checkpoints import NPTCheckpointMerger


def create_dummy_checkpoint(layer_idx: int, temp_dir: Path, num_ranks: int = 8):
    """Create a dummy checkpoint for testing."""
    checkpoint_dir = temp_dir / f"layer{layer_idx}_checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy NPT weights
    npt_weights = {}

    # Triple modulation weights (matching your training config)
    for rank in range(num_ranks):
        # Gate modulation
        npt_weights[f"layer_{layer_idx}_np.W_down_gate.{rank}"] = torch.randn(2048, 512)
        npt_weights[f"layer_{layer_idx}_np.W_a_up_gate.{rank}"] = torch.randn(512, 2048)
        npt_weights[f"layer_{layer_idx}_np.W_b_up_gate.{rank}"] = torch.randn(512, 8192)

        # Up modulation
        npt_weights[f"layer_{layer_idx}_np.W_down_up.{rank}"] = torch.randn(2048, 512)
        npt_weights[f"layer_{layer_idx}_np.W_a_up_up.{rank}"] = torch.randn(512, 2048)
        npt_weights[f"layer_{layer_idx}_np.W_b_up_up.{rank}"] = torch.randn(512, 8192)

        # Down modulation
        npt_weights[f"layer_{layer_idx}_np.W_down_down.{rank}"] = torch.randn(2048, 512)
        npt_weights[f"layer_{layer_idx}_np.W_a_up_down.{rank}"] = torch.randn(512, 8192)
        npt_weights[f"layer_{layer_idx}_np.W_b_up_down.{rank}"] = torch.randn(512, 2048)

    # Save NPT weights
    torch.save(npt_weights, checkpoint_dir / "npt_weights.pt")

    # Create dummy training state
    training_state = {
        'global_step': 10000 + layer_idx * 1000,
        'batch_count': 10000 + layer_idx * 1000,
        'optimizer_state_dict': {'param_groups': [{'lr': 1e-5}]},
        'scheduler_state_dict': {},
    }
    torch.save(training_state, checkpoint_dir / "training_state.pt")

    # Create dummy config
    config = {
        'model_name': 'meta-llama/Llama-3.2-1B',
        'batch_size': 64,
        'learning_rate': 1e-5,
        'max_steps': 45000,
        'layer_trained': layer_idx
    }
    with open(checkpoint_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    return checkpoint_dir


def test_merge():
    """Test the merge functionality."""
    print("\n" + "="*60)
    print("Testing NPT Checkpoint Merge Functionality")
    print("="*60)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create dummy checkpoints for layer 11 and 13
        print("\n1. Creating dummy checkpoints...")
        checkpoint1 = create_dummy_checkpoint(11, temp_path)
        checkpoint2 = create_dummy_checkpoint(13, temp_path)
        print(f"   Created checkpoint for layer 11: {checkpoint1}")
        print(f"   Created checkpoint for layer 13: {checkpoint2}")

        # Create merger
        print("\n2. Initializing merger...")
        merger = NPTCheckpointMerger()

        # Load checkpoints
        print("\n3. Loading checkpoints...")
        npt1, state1, config1 = merger.load_checkpoint(checkpoint1)
        npt2, state2, config2 = merger.load_checkpoint(checkpoint2)

        # Extract layer info
        layers1 = merger.extract_layer_info(npt1)
        layers2 = merger.extract_layer_info(npt2)
        print(f"   Checkpoint 1 has layers: {sorted(layers1)}")
        print(f"   Checkpoint 2 has layers: {sorted(layers2)}")

        # Merge
        print("\n4. Merging checkpoints...")
        merged_weights = merger.merge_npt_weights([
            (checkpoint1, npt1),
            (checkpoint2, npt2)
        ])
        merged_state = merger.merge_training_states([
            (checkpoint1, state1),
            (checkpoint2, state2)
        ])
        merged_config = merger.merge_configs([
            (checkpoint1, config1),
            (checkpoint2, config2)
        ])

        print(f"   Merged layers: {sorted(merger.merged_layers)}")
        print(f"   Total merged parameters: {len(merged_weights)} keys")

        # Save merged checkpoint
        output_path = temp_path / "merged_checkpoint"
        print(f"\n5. Saving merged checkpoint to {output_path}...")
        merger.save_merged_checkpoint(
            output_path,
            merged_weights,
            merged_state,
            merged_config
        )

        # Verify
        print("\n6. Verifying merge...")
        success = merger.verify_merge(output_path)

        if success:
            print("\n✓ Test passed! Merge functionality works correctly.")

            # Show what was created
            print("\nMerged checkpoint contents:")
            for file in output_path.glob("*"):
                if file.is_file():
                    size = file.stat().st_size
                    print(f"   {file.name}: {size:,} bytes")

            # Load and check the merge summary
            with open(output_path / "merge_summary.json", 'r') as f:
                summary = json.load(f)

            print("\nMerge summary:")
            print(f"   Merged layers: {summary['merged_layers']}")
            print(f"   Total parameters: {summary['total_parameters']:,}")
            print(f"   Checkpoint keys: {summary['checkpoint_keys']}")
            print("\nLayer sources:")
            for layer, source in summary['layer_sources'].items():
                print(f"   Layer {layer}: {Path(source).name}")

        else:
            print("\n✗ Test failed! Verification did not pass.")
            return 1

    print("\n" + "="*60)
    print("Test completed successfully!")
    print("="*60)
    return 0


if __name__ == "__main__":
    exit(test_merge())