#!/usr/bin/env python3
"""
Demo script for rank-k NPT experiments.

This script demonstrates how to train NPT with different rank configurations
and compares their performance.
"""

import argparse
import sys
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.npt import NPTLlamaModel, NPTConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_rank_configurations():
    """Demonstrate different rank-k configurations."""
    
    print("\n" + "="*80)
    print("RANK-K NPT CONFIGURATIONS DEMO")
    print("="*80)
    
    # Example configurations for experiments
    configs = [
        {
            "name": "Baseline Rank-1 (High Rank)",
            "num_ranks": 1,
            "np_rank": 256,
            "description": "Single rank-1 update with high rank (original NPT)"
        },
        {
            "name": "Rank-2 (Medium Rank Each)",
            "num_ranks": 2,
            "np_rank": 128,
            "description": "Two rank-1 updates, each with medium rank"
        },
        {
            "name": "Rank-4 (Lower Rank Each)",
            "num_ranks": 4,
            "np_rank": 64,
            "description": "Four rank-1 updates, each with lower rank"
        },
        {
            "name": "Rank-8 (Low Rank Each)",
            "num_ranks": 8,
            "np_rank": 32,
            "description": "Eight rank-1 updates, each with low rank"
        },
    ]
    
    print("\nRecommended Configurations for Experiments:")
    print("-" * 80)
    
    for config in configs:
        print(f"\n{config['name']}:")
        print(f"  Description: {config['description']}")
        print(f"  Command:")
        print(f"    python scripts/train_single_layer_npt.py \\")
        print(f"      --num_ranks {config['num_ranks']} \\")
        print(f"      --np_rank {config['np_rank']} \\")
        print(f"      --convert_layers 15 \\")
        print(f"      --max_steps 5000")
        print(f"  Expected behavior:")
        if config['num_ranks'] == 1:
            print(f"    - Standard NPT behavior (backward compatible)")
            print(f"    - Single v_a encodes attention directly")
        else:
            print(f"    - Multiple components learn complementary patterns")
            print(f"    - First v_a component focuses on attention encoding")
            print(f"    - Other components add expressiveness")
    
    print("\n" + "="*80)
    print("KEY INSIGHTS FOR RANK-K EXPERIMENTS")
    print("="*80)
    
    print("""
1. PARAMETER EFFICIENCY:
   - All configurations above have the same total parameters
   - Trade-off: num_ranks × rank = constant capacity
   
2. EXPRESSIVENESS:
   - Rank-1: Single powerful transformation
   - Rank-k: Sum of k simpler transformations
   - Higher k may learn more diverse patterns
   
3. TRAINING DYNAMICS:
   - Rank-1: All capacity in one bottleneck
   - Rank-k: Distributed learning across components
   - May help with gradient flow and optimization
   
4. INITIALIZATION STRATEGY:
   - First component: Identity-like for attention encoding
   - Other components: Orthogonal for diversity
   
5. LOSS COMPUTATION:
   - Primary component (first) used for attention encoding loss
   - All components contribute to regularization
   - Alternative: Average all components for attention loss
""")
    
    print("\n" + "="*80)
    print("EXPERIMENT SUGGESTIONS")
    print("="*80)
    
    print("""
1. BASELINE COMPARISON:
   python scripts/train_single_layer_npt.py --num_ranks 1 --np_rank 256 --wandb_name rank1_baseline

2. RANK-2 EXPERIMENT:
   python scripts/train_single_layer_npt.py --num_ranks 2 --np_rank 128 --wandb_name rank2_exp

3. RANK-4 EXPERIMENT:
   python scripts/train_single_layer_npt.py --num_ranks 4 --np_rank 64 --wandb_name rank4_exp

4. SEQUENTIAL TRAINING WITH RANK-K:
   python scripts/train_sequential_layers.py --num_ranks 2 --np_rank 128 --layers "15,16,17"

5. DEMO MODE TEST:
   python scripts/train_single_layer_npt.py --demo_mode --num_ranks 4 --np_rank 64
""")


def compare_memory_usage():
    """Compare memory usage for different configurations."""
    
    print("\n" + "="*80)
    print("MEMORY USAGE COMPARISON")
    print("="*80)
    
    from transformers import LlamaConfig
    
    # Small config for demo
    config = LlamaConfig(
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=16,
        num_attention_heads=32,
        num_key_value_heads=8,
        vocab_size=128256,
    )
    config._attn_implementation = "eager"
    
    configurations = [
        (1, 256),
        (2, 128),
        (4, 64),
        (8, 32),
    ]
    
    for num_ranks, rank in configurations:
        model = NPTLlamaModel(config)
        npt_config = NPTConfig(
            layers_to_convert=[15],
            np_rank=rank,
            single_layer_mode=True,
            num_ranks=num_ranks
        )
        model.convert_to_npt(npt_config)
        
        # Count parameters
        npt_params = sum(p.numel() for name, p in model.named_parameters() if 'np_component' in name)
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"\nRank-{num_ranks} (rank={rank} each):")
        print(f"  NPT parameters: {npt_params:,}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  NPT percentage: {100 * npt_params / total_params:.2f}%")
        
        # Estimate memory (rough)
        param_memory = total_params * 4 / (1024**3)  # 4 bytes per float32, in GB
        print(f"  Estimated parameter memory: {param_memory:.2f} GB")


def main():
    """Run demo."""
    parser = argparse.ArgumentParser(description="Demo rank-k NPT configurations")
    parser.add_argument("--compare_memory", action="store_true", help="Compare memory usage")
    args = parser.parse_args()
    
    demo_rank_configurations()
    
    if args.compare_memory:
        compare_memory_usage()
    
    print("\n" + "="*80)
    print("For actual training, use the commands shown above!")
    print("="*80)


if __name__ == "__main__":
    main()