#!/usr/bin/env python3
"""
Quick test to verify validation metrics are logged to WandB.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer
from src.training.wandb_integration import WandBTracker
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_wandb_logging():
    """Test that validation metrics are properly logged."""
    
    # Initialize tracker
    tracker = WandBTracker(
        project="test-validation",
        name="test-metrics",
        config={"test": True},
        tags=["test"],
        mode="offline"  # Offline mode for testing
    )
    tracker.init()
    
    # Simulate training and validation metrics
    for step in range(0, 100, 10):
        # Training metrics
        train_metrics = {
            "loss": 0.5 - step * 0.003,
            "learning_rate": 1e-3,
            "direct_mlp_loss": 0.3 - step * 0.002
        }
        tracker.log_metrics(train_metrics, step=step)
        
        # Validation metrics every 30 steps
        if step % 30 == 0 and step > 0:
            val_metrics = {
                "val_loss": 0.6 - step * 0.004,
                "val_direct_mlp_loss": 0.4 - step * 0.003,
                "val_fidelity_loss": 0.2 - step * 0.001
            }
            tracker.log_metrics(val_metrics, step=step)
            logger.info(f"Step {step} - Logged validation metrics: {val_metrics}")
    
    tracker.finish()
    logger.info("Test complete! Check WandB for logged metrics.")


if __name__ == "__main__":
    test_wandb_logging()