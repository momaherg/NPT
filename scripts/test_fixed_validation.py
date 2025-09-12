#!/usr/bin/env python3
"""
Test script to verify fixed validation dataset functionality.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer
from src.training.streaming_data import (
    FixedValidationDataset, 
    create_fixed_validation_loader,
    MultiDatasetStreamer
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_fixed_validation():
    """Test fixed validation dataset creation and consistency."""
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test 1: Create fixed validation dataset
    logger.info("\nTest 1: Creating fixed validation dataset...")
    fixed_dataset = FixedValidationDataset(
        tokenizer=tokenizer,
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
        max_length=512,
        num_samples=100,
        seed=42
    )
    
    logger.info(f"Dataset size: {len(fixed_dataset)} samples")
    
    # Test 2: Verify consistency
    logger.info("\nTest 2: Verifying consistency...")
    sample1_first = fixed_dataset[0]['input_ids']
    sample5_first = fixed_dataset[4]['input_ids']
    
    # Create another instance with same seed
    fixed_dataset2 = FixedValidationDataset(
        tokenizer=tokenizer,
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
        max_length=512,
        num_samples=100,
        seed=42
    )
    
    sample1_second = fixed_dataset2[0]['input_ids']
    sample5_second = fixed_dataset2[4]['input_ids']
    
    assert torch.equal(sample1_first, sample1_second), "First samples don't match!"
    assert torch.equal(sample5_first, sample5_second), "Fifth samples don't match!"
    logger.info("✓ Consistency verified - same samples with same seed")
    
    # Test 3: Create data loader
    logger.info("\nTest 3: Creating fixed validation loader...")
    val_loader = create_fixed_validation_loader(
        tokenizer=tokenizer,
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
        batch_size=8,
        max_length=512,
        num_validation_samples=200,
        seed=42
    )
    
    # Count batches
    num_batches = 0
    total_samples = 0
    for batch in val_loader:
        num_batches += 1
        total_samples += batch['input_ids'].size(0)
    
    logger.info(f"Validation loader: {num_batches} batches, {total_samples} total samples")
    
    # Test 4: Compare with streaming validation
    logger.info("\nTest 4: Comparing with MultiDatasetStreamer...")
    streamer = MultiDatasetStreamer(
        preset="small",
        tokenizer=tokenizer,
        max_length=512,
        batch_size=8,
        num_workers=0
    )
    
    # Test with fixed validation
    train_loader_fixed, val_loader_fixed = streamer.create_data_loaders(
        validation=True,
        fixed_validation=True,
        num_validation_samples=200
    )
    
    # Test with streaming validation
    train_loader_stream, val_loader_stream = streamer.create_data_loaders(
        validation=True,
        fixed_validation=False
    )
    
    # Get first batch from each
    val_batch_fixed = next(iter(val_loader_fixed))
    logger.info(f"Fixed validation batch shape: {val_batch_fixed['input_ids'].shape}")
    
    # Streaming validation might fail on some datasets, that's ok
    try:
        val_batch_stream = next(iter(val_loader_stream))
        logger.info(f"Streaming validation batch shape: {val_batch_stream['input_ids'].shape}")
    except:
        logger.info("Streaming validation not available (expected for some datasets)")
    
    # Test 5: Verify deterministic evaluation
    logger.info("\nTest 5: Verifying deterministic evaluation...")
    
    # Collect all samples from fixed loader
    all_samples1 = []
    for batch in val_loader_fixed:
        all_samples1.append(batch['input_ids'])
    
    # Create new loader with same config
    _, val_loader_fixed2 = streamer.create_data_loaders(
        validation=True,
        fixed_validation=True,
        num_validation_samples=200
    )
    
    # Collect all samples from second loader
    all_samples2 = []
    for batch in val_loader_fixed2:
        all_samples2.append(batch['input_ids'])
    
    # Compare
    assert len(all_samples1) == len(all_samples2), "Different number of batches!"
    for i, (s1, s2) in enumerate(zip(all_samples1, all_samples2)):
        assert torch.equal(s1, s2), f"Batch {i} doesn't match!"
    
    logger.info("✓ Deterministic evaluation verified - same batches across runs")
    
    logger.info("\n" + "="*50)
    logger.info("All tests passed! Fixed validation is working correctly.")
    logger.info("="*50)


if __name__ == "__main__":
    test_fixed_validation()