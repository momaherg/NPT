"""
Streaming data loader for HuggingFace datasets.

This module provides efficient streaming data loading from HuggingFace Hub
without loading entire datasets into memory.
"""

import torch
from torch.utils.data import IterableDataset, DataLoader, Dataset
from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Union, Iterator, Any
import numpy as np
from dataclasses import dataclass
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class StreamingConfig:
    """Configuration for streaming datasets."""
    dataset_name: Union[str, List[str]] = "wikitext"
    dataset_config: Union[str, List[str]] = "wikitext-103-raw-v1"
    split: str = "train"
    text_column: str = "text"
    max_length: int = 512
    stride: int = 256
    buffer_size: int = 10000
    seed: int = 42
    num_workers: int = 4
    prefetch_factor: int = 2
    streaming: bool = True
    mix_probabilities: Optional[List[float]] = None
    validation_split_percentage: int = 5


class StreamingTextDataset(IterableDataset):
    """
    Streaming dataset for text data from HuggingFace.
    
    Supports:
    - Single or multiple datasets
    - On-the-fly tokenization
    - Sliding window chunking
    - Dataset interleaving
    """
    
    def __init__(
        self,
        config: StreamingConfig,
        tokenizer: AutoTokenizer,
        is_validation: bool = False
    ):
        """
        Initialize streaming dataset.
        
        Args:
            config: Streaming configuration
            tokenizer: Tokenizer for encoding
            is_validation: Whether this is for validation
        """
        self.config = config
        self.tokenizer = tokenizer
        self.is_validation = is_validation
        
        # Ensure tokenizer has required tokens
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load datasets
        self.datasets = self._load_datasets()
        
        # Buffer for accumulating tokens
        self.token_buffer = []
        
    def _load_datasets(self):
        """Load and potentially mix multiple datasets."""
        dataset_names = self.config.dataset_name
        dataset_configs = self.config.dataset_config
        
        # Handle single dataset
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
            dataset_configs = [dataset_configs] if isinstance(dataset_configs, str) else dataset_configs
        
        # Load each dataset
        datasets = []
        for name, config_name in zip(dataset_names, dataset_configs):
            try:
                logger.info(f"Loading dataset: {name}/{config_name}")
                
                # Load with streaming
                dataset = load_dataset(
                    name,
                    config_name if config_name else None,
                    split=self.config.split,
                    streaming=self.config.streaming
                )
                
                # Shuffle with buffer
                if not self.is_validation:
                    dataset = dataset.shuffle(
                        seed=self.config.seed,
                        buffer_size=self.config.buffer_size
                    )
                
                datasets.append(dataset)
                
            except Exception as e:
                logger.warning(f"Failed to load {name}/{config_name}: {e}")
                continue
        
        if not datasets:
            raise ValueError("No datasets could be loaded")
        
        # Interleave multiple datasets
        if len(datasets) > 1:
            probabilities = self.config.mix_probabilities
            if probabilities is None:
                # Equal mixing by default
                probabilities = [1.0 / len(datasets)] * len(datasets)
            
            logger.info(f"Interleaving {len(datasets)} datasets with probabilities: {probabilities}")
            return interleave_datasets(
                datasets,
                probabilities=probabilities,
                seed=self.config.seed
            )
        
        return datasets[0]
    
    def _tokenize_and_chunk(self, examples: Iterator[Dict]) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Tokenize text and create chunks with sliding window.
        
        Args:
            examples: Iterator of text examples
        
        Yields:
            Tokenized and chunked examples
        """
        for example in examples:
            # Get text from the configured column
            text = example.get(self.config.text_column, "")
            
            # Skip empty texts
            if not text or len(text.strip()) == 0:
                continue
            
            # Tokenize
            tokens = self.tokenizer(
                text,
                truncation=False,
                add_special_tokens=True,
                return_attention_mask=False
            )['input_ids']
            
            # Add to buffer
            self.token_buffer.extend(tokens)
            
            # Create chunks from buffer
            while len(self.token_buffer) >= self.config.max_length:
                # Extract chunk
                chunk = self.token_buffer[:self.config.max_length]
                
                # Yield chunk
                yield {
                    'input_ids': torch.tensor(chunk, dtype=torch.long),
                    'attention_mask': torch.ones(len(chunk), dtype=torch.long),
                    'labels': torch.tensor(chunk, dtype=torch.long)
                }
                
                # Slide window
                self.token_buffer = self.token_buffer[self.config.stride:]
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate through the dataset."""
        # Reset buffer for each epoch
        self.token_buffer = []
        
        # Create iterator from dataset
        dataset_iter = iter(self.datasets)
        
        # Tokenize and chunk
        yield from self._tokenize_and_chunk(dataset_iter)
        
        # Handle remaining tokens in buffer
        if len(self.token_buffer) >= 32:  # Minimum sequence length
            padding_length = self.config.max_length - len(self.token_buffer)
            if padding_length > 0:
                # Pad the sequence
                chunk = self.token_buffer + [self.tokenizer.pad_token_id] * padding_length
                attention_mask = [1] * len(self.token_buffer) + [0] * padding_length
            else:
                chunk = self.token_buffer[:self.config.max_length]
                attention_mask = [1] * self.config.max_length
            
            yield {
                'input_ids': torch.tensor(chunk, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'labels': torch.tensor(chunk, dtype=torch.long)
            }


class FixedValidationDataset(Dataset):
    """
    Fixed validation dataset that loads and caches a specific number of samples.
    
    This provides consistent validation metrics across training steps.
    """
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-2-raw-v1",
        max_length: int = 512,
        num_samples: int = 1000,
        seed: int = 42,
        cache_file: Optional[str] = None
    ):
        """
        Initialize fixed validation dataset.
        
        Args:
            tokenizer: Tokenizer for encoding
            dataset_name: Name of HuggingFace dataset
            dataset_config: Configuration for dataset
            max_length: Maximum sequence length
            num_samples: Number of validation samples to cache
            seed: Random seed for reproducibility
            cache_file: Optional path to save/load cached dataset
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_samples = num_samples
        self.cache_file = cache_file
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Try to load from cache first
        if cache_file and Path(cache_file).exists():
            logger.info(f"Loading cached validation dataset from {cache_file}")
            import pickle
            with open(cache_file, 'rb') as f:
                self.samples = pickle.load(f)
            logger.info(f"Loaded {len(self.samples)} cached validation samples")
            return
        
        logger.info(f"Loading fixed validation dataset: {dataset_name}/{dataset_config}")
        
        # Load validation split
        try:
            dataset = load_dataset(
                dataset_name,
                dataset_config if dataset_config else None,
                split='validation',
                streaming=False  # Load into memory
            )
        except:
            # Fallback to test split or subset of train
            try:
                dataset = load_dataset(
                    dataset_name,
                    dataset_config if dataset_config else None,
                    split='test',
                    streaming=False
                )
            except:
                # Last resort: use a subset of train
                dataset = load_dataset(
                    dataset_name,
                    dataset_config if dataset_config else None,
                    split=f'train[:{num_samples*2}]',
                    streaming=False
                )
        
        # Process and cache samples
        self.samples = []
        token_buffer = []
        
        # Set seed for reproducibility
        np.random.seed(seed)
        
        # Faster: Take a continuous slice instead of random indices
        # This reduces random access overhead
        total_size = len(dataset)
        start_idx = seed % max(1, total_size - num_samples * 5)
        end_idx = min(start_idx + num_samples * 5, total_size)
        
        # Batch tokenization for speed
        batch_texts = []
        for idx in range(start_idx, end_idx):
            if len(self.samples) >= num_samples:
                break
            text = dataset[idx].get('text', '')
            if text and len(text.strip()) > 0:
                batch_texts.append(text)
            
            # Process in batches of 100 texts
            if len(batch_texts) >= 100 or idx == end_idx - 1:
                if batch_texts:
                    # Batch tokenization is much faster
                    batch_tokens = tokenizer(
                        batch_texts,
                        truncation=False,
                        add_special_tokens=True,
                        return_attention_mask=False
                    )['input_ids']
                    
                    # Process each tokenized text
                    for tokens in batch_tokens:
                        token_buffer.extend(tokens)
                        
                        # Create fixed-length chunks
                        while len(token_buffer) >= max_length and len(self.samples) < num_samples:
                            chunk = token_buffer[:max_length]
                            self.samples.append({
                                'input_ids': torch.tensor(chunk, dtype=torch.long),
                                'attention_mask': torch.ones(max_length, dtype=torch.long),
                                'labels': torch.tensor(chunk, dtype=torch.long)
                            })
                            token_buffer = token_buffer[max_length // 2:]  # 50% overlap
                    
                    batch_texts = []
        
        # Handle remaining tokens if we don't have enough samples
        if len(self.samples) < num_samples and len(token_buffer) >= 32:
            chunk = token_buffer[:max_length]
            if len(chunk) < max_length:
                padding_length = max_length - len(chunk)
                chunk = chunk + [tokenizer.pad_token_id] * padding_length
                attention_mask = [1] * len(token_buffer[:max_length]) + [0] * padding_length
            else:
                attention_mask = [1] * max_length
            
            self.samples.append({
                'input_ids': torch.tensor(chunk, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'labels': torch.tensor(chunk, dtype=torch.long)
            })
        
        logger.info(f"Cached {len(self.samples)} validation samples")
        
        # Save to cache if requested
        if cache_file:
            logger.info(f"Saving validation dataset cache to {cache_file}")
            import pickle
            cache_path = Path(cache_file)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(self.samples, f)
            logger.info(f"Cache saved to {cache_file}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class MultiDatasetStreamer:
    """
    Advanced streamer for multiple HuggingFace datasets.
    
    Supports:
    - Multiple dataset sources
    - Custom mixing ratios
    - Train/validation splitting
    - Automatic dataset discovery
    """
    
    # Popular datasets for language modeling
    PRESET_DATASETS = {
        'small': [
            ('wikitext', 'wikitext-2-raw-v1'),
        ],
        'medium': [
            ('wikitext', 'wikitext-103-raw-v1'),
            ('bookcorpus', None),
        ],
        'large': [
            ('wikipedia', '20220301.en'),
            ('bookcorpus', None),
            ('openwebtext', None),
        ],
        'xlarge': [
            ('wikipedia', '20220301.en'),
            ('bookcorpus', None),
            ('openwebtext', None),
            ('cc_news', None),
            ('pile', None),
        ]
    }
    
    def __init__(
        self,
        preset: str = 'small',
        tokenizer: AutoTokenizer = None,
        max_length: int = 512,
        batch_size: int = 8,
        num_workers: int = 4
    ):
        """
        Initialize multi-dataset streamer.
        
        Args:
            preset: Dataset preset ('small', 'medium', 'large', 'xlarge')
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
            batch_size: Batch size
            num_workers: Number of data workers
        """
        self.preset = preset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Get dataset configuration
        if preset in self.PRESET_DATASETS:
            datasets = self.PRESET_DATASETS[preset]
            self.dataset_names = [d[0] for d in datasets]
            self.dataset_configs = [d[1] for d in datasets]
        else:
            raise ValueError(f"Unknown preset: {preset}. Choose from {list(self.PRESET_DATASETS.keys())}")
    
    def create_data_loaders(
        self,
        validation: bool = True,
        mix_probabilities: Optional[List[float]] = None,
        fixed_validation: bool = True,
        num_validation_samples: int = 500
    ) -> tuple[DataLoader, Optional[DataLoader]]:
        """
        Create streaming data loaders.
        
        Args:
            validation: Whether to create validation loader
            mix_probabilities: Custom mixing probabilities
            fixed_validation: Whether to use fixed validation dataset
            num_validation_samples: Number of validation samples for fixed validation
        
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Create streaming config
        config = StreamingConfig(
            dataset_name=self.dataset_names,
            dataset_config=self.dataset_configs,
            max_length=self.max_length,
            mix_probabilities=mix_probabilities,
            num_workers=self.num_workers
        )
        
        # Create train dataset
        train_dataset = StreamingTextDataset(
            config=config,
            tokenizer=self.tokenizer,
            is_validation=False
        )
        
        # Create train loader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers if self.num_workers > 0 else 0,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=2 if self.num_workers > 0 else None
        )
        
        # Create validation loader if requested
        val_loader = None
        if validation:
            if fixed_validation:
                # Use fixed validation dataset for consistent metrics
                val_loader = create_fixed_validation_loader(
                    tokenizer=self.tokenizer,
                    dataset_name=self.dataset_names[0],  # Use first dataset for validation
                    dataset_config=self.dataset_configs[0],
                    batch_size=self.batch_size,
                    max_length=self.max_length,
                    num_validation_samples=num_validation_samples,
                    num_workers=0,  # Fixed dataset doesn't benefit from multiple workers
                    seed=42
                )
            else:
                # Use streaming validation
                val_config = StreamingConfig(
                    dataset_name=self.dataset_names[0],  # Use first dataset for validation
                    dataset_config=self.dataset_configs[0],
                    split='validation' if 'validation' in ['train', 'validation', 'test'] else 'train',
                    max_length=self.max_length,
                    num_workers=self.num_workers
                )
                
                val_dataset = StreamingTextDataset(
                    config=val_config,
                    tokenizer=self.tokenizer,
                    is_validation=True
                )
                
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers if self.num_workers > 0 else 0,
                    pin_memory=torch.cuda.is_available(),
                    prefetch_factor=2 if self.num_workers > 0 else None
                )
        
        return train_loader, val_loader


def create_fixed_validation_loader(
    tokenizer: AutoTokenizer,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    batch_size: int = 8,
    max_length: int = 512,
    num_validation_samples: int = 500,
    num_workers: int = 0,
    seed: int = 42
) -> DataLoader:
    """
    Create a fixed validation data loader with cached samples.
    
    Args:
        tokenizer: Tokenizer for encoding
        dataset_name: Name of HuggingFace dataset
        dataset_config: Configuration for dataset
        batch_size: Batch size
        max_length: Maximum sequence length
        num_validation_samples: Number of validation samples to cache
        num_workers: Number of data workers (0 recommended for cached data)
        seed: Random seed for reproducibility
    
    Returns:
        DataLoader with fixed validation samples
    """
    val_dataset = FixedValidationDataset(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        max_length=max_length,
        num_samples=num_validation_samples,
        seed=seed
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle to maintain consistency
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return val_loader


def create_streaming_loaders(
    tokenizer: AutoTokenizer,
    dataset_name: Union[str, List[str]] = "wikitext",
    dataset_config: Union[str, List[str]] = "wikitext-103-raw-v1",
    batch_size: int = 8,
    max_length: int = 512,
    stride: int = 256,
    num_workers: int = 4,
    streaming: bool = True,
    validation: bool = True,
    fixed_validation: bool = False,
    num_validation_samples: int = 500
) -> tuple[DataLoader, Optional[DataLoader]]:
    """
    Create streaming data loaders from HuggingFace datasets.
    
    Args:
        tokenizer: Tokenizer for encoding
        dataset_name: Name(s) of HuggingFace dataset(s)
        dataset_config: Configuration(s) for dataset(s)
        batch_size: Batch size
        max_length: Maximum sequence length
        stride: Stride for sliding window
        num_workers: Number of data workers
        streaming: Whether to use streaming mode
        validation: Whether to create validation loader
        fixed_validation: Whether to use fixed validation dataset
        num_validation_samples: Number of validation samples for fixed validation
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create config
    config = StreamingConfig(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        max_length=max_length,
        stride=stride,
        num_workers=num_workers,
        streaming=streaming
    )
    
    # Create train dataset
    train_dataset = StreamingTextDataset(
        config=config,
        tokenizer=tokenizer,
        is_validation=False
    )
    
    # Create train loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers if num_workers > 0 else 0,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    # Create validation loader
    val_loader = None
    if validation:
        if fixed_validation:
            # Use fixed validation dataset for consistent metrics
            val_loader = create_fixed_validation_loader(
                tokenizer=tokenizer,
                dataset_name=dataset_name if isinstance(dataset_name, str) else dataset_name[0],
                dataset_config=dataset_config if isinstance(dataset_config, str) else dataset_config[0],
                batch_size=batch_size,
                max_length=max_length,
                num_validation_samples=num_validation_samples,
                num_workers=0,  # Fixed dataset doesn't benefit from multiple workers
                seed=42
            )
        else:
            # Use streaming validation
            val_config = StreamingConfig(
                dataset_name=dataset_name if isinstance(dataset_name, str) else dataset_name[0],
                dataset_config=dataset_config if isinstance(dataset_config, str) else dataset_config[0],
                split='validation',
                max_length=max_length,
                stride=stride,
                num_workers=num_workers,
                streaming=streaming
            )
            
            try:
                val_dataset = StreamingTextDataset(
                    config=val_config,
                    tokenizer=tokenizer,
                    is_validation=True
                )
                
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    num_workers=num_workers if num_workers > 0 else 0,
                    pin_memory=torch.cuda.is_available(),
                    prefetch_factor=2 if num_workers > 0 else None
                )
            except:
                logger.warning("Could not create validation loader, using None")
    
    return train_loader, val_loader