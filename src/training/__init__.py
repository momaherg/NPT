"""
Training module for the Neuro-Plastic Transformer.

This module contains loss functions and utilities for equivalence pre-training.
"""

from .losses import (
    FidelityLoss,
    RegularizationLoss,
    EquivalenceLoss,
    ParallelForwardHelper
)
from .data_loader import (
    TextDataset,
    DataCollatorForNPT,
    create_data_loaders,
    InfiniteDataLoader
)
from .trainer import (
    NPTTrainer,
    TrainingConfig,
    TrainingMetrics
)
from .streaming_data import (
    StreamingTextDataset,
    StreamingConfig,
    MultiDatasetStreamer,
    create_streaming_loaders
)
from .wandb_integration import (
    WandBTracker,
    setup_wandb_tracking
)

__all__ = [
    # Loss functions
    'FidelityLoss',
    'RegularizationLoss',
    'EquivalenceLoss',
    'ParallelForwardHelper',
    # Data loading
    'TextDataset',
    'DataCollatorForNPT',
    'create_data_loaders',
    'InfiniteDataLoader',
    # Streaming data
    'StreamingTextDataset',
    'StreamingConfig',
    'MultiDatasetStreamer',
    'create_streaming_loaders',
    # WandB integration
    'WandBTracker',
    'setup_wandb_tracking',
    # Trainer
    'NPTTrainer',
    'TrainingConfig',
    'TrainingMetrics'
]