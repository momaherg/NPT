#!/usr/bin/env python3
"""
Quick test to compare validation speeds.
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from src.training.streaming_data import FixedValidationDataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def benchmark_validation():
    """Benchmark different validation configurations."""
    
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test different configurations
    configs = [
        {"samples": 500, "batch_size": 8, "desc": "Original (500 samples, batch 8)"},
        {"samples": 200, "batch_size": 8, "desc": "Reduced samples (200, batch 8)"},
        {"samples": 200, "batch_size": 16, "desc": "Optimized (200, batch 16)"},
        {"samples": 100, "batch_size": 32, "desc": "Fast mode (100, batch 32)"},
    ]
    
    cache_file = "cache/benchmark_test.pkl"
    
    for config in configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {config['desc']}")
        logger.info(f"{'='*60}")
        
        # First run - creates cache
        start = time.time()
        dataset = FixedValidationDataset(
            tokenizer=tokenizer,
            dataset_name="wikitext",
            dataset_config="wikitext-2-raw-v1",
            max_length=512,
            num_samples=config["samples"],
            seed=42,
            cache_file=cache_file if config == configs[0] else None  # Only cache first
        )
        creation_time = time.time() - start
        
        # Create loader
        loader = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        
        # Simulate evaluation
        start = time.time()
        total_samples = 0
        for batch in loader:
            # Simulate some processing
            _ = batch['input_ids'].float().mean()
            total_samples += batch['input_ids'].size(0)
        eval_time = time.time() - start
        
        num_batches = len(loader)
        
        logger.info(f"Dataset creation: {creation_time:.2f}s")
        logger.info(f"Evaluation time: {eval_time:.2f}s")
        logger.info(f"Total samples: {total_samples}")
        logger.info(f"Batches: {num_batches}")
        logger.info(f"Time per batch: {eval_time/num_batches*1000:.1f}ms")
        logger.info(f"Samples per second: {total_samples/eval_time:.0f}")
    
    # Test with cache
    logger.info(f"\n{'='*60}")
    logger.info("Testing cached loading...")
    logger.info(f"{'='*60}")
    
    start = time.time()
    dataset = FixedValidationDataset(
        tokenizer=tokenizer,
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
        max_length=512,
        num_samples=500,
        seed=42,
        cache_file=cache_file
    )
    cached_time = time.time() - start
    logger.info(f"Loading from cache: {cached_time:.2f}s (vs {creation_time:.2f}s original)")
    logger.info(f"Speedup: {creation_time/cached_time:.1f}x")
    
    # Clean up
    Path(cache_file).unlink(missing_ok=True)


if __name__ == "__main__":
    benchmark_validation()