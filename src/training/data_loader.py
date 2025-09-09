"""
Data loader for NPT equivalence pre-training.

This module provides data loading utilities for training the NPT model
to achieve functional equivalence with the original transformer.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Union
import numpy as np
from pathlib import Path
import json


class TextDataset(Dataset):
    """
    Dataset for loading text data for NPT training.
    
    Supports loading from:
    - JSON files with text fields
    - Plain text files
    - Lists of strings
    """
    
    def __init__(
        self,
        data_source: Union[str, Path, List[str]],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        stride: int = 256,
        text_field: str = "text"
    ):
        """
        Initialize text dataset.
        
        Args:
            data_source: Path to data file or list of texts
            tokenizer: Tokenizer for encoding texts
            max_length: Maximum sequence length
            stride: Stride for sliding window over long texts
            text_field: Field name for text in JSON files
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.text_field = text_field
        
        # Load data
        self.texts = self._load_data(data_source)
        
        # Tokenize and create chunks
        self.chunks = self._create_chunks()
    
    def _load_data(self, data_source: Union[str, Path, List[str]]) -> List[str]:
        """Load text data from various sources."""
        if isinstance(data_source, list):
            return data_source
        
        path = Path(data_source)
        
        if path.suffix == '.json':
            # Load from JSON
            with open(path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return [item[self.text_field] for item in data]
                else:
                    return [data[self.text_field]]
        
        elif path.suffix in ['.txt', '.md']:
            # Load from text file
            with open(path, 'r') as f:
                return [f.read()]
        
        else:
            # Try loading as plain text
            with open(path, 'r') as f:
                lines = f.readlines()
                return [line.strip() for line in lines if line.strip()]
    
    def _create_chunks(self) -> List[Dict[str, torch.Tensor]]:
        """Create overlapping chunks from texts."""
        chunks = []
        
        for text in self.texts:
            # Tokenize with truncation disabled to get full text
            encoded = self.tokenizer(
                text,
                truncation=False,
                return_tensors='pt',
                padding=False
            )
            
            input_ids = encoded['input_ids'][0]
            
            # Create overlapping chunks
            if len(input_ids) <= self.max_length:
                # Text fits in one chunk
                chunks.append({
                    'input_ids': input_ids,
                    'attention_mask': torch.ones_like(input_ids)
                })
            else:
                # Create sliding window chunks
                for i in range(0, len(input_ids) - self.max_length + 1, self.stride):
                    chunk_ids = input_ids[i:i + self.max_length]
                    chunks.append({
                        'input_ids': chunk_ids,
                        'attention_mask': torch.ones_like(chunk_ids)
                    })
                
                # Add final chunk if there's remaining text
                if i + self.max_length < len(input_ids):
                    chunk_ids = input_ids[-self.max_length:]
                    chunks.append({
                        'input_ids': chunk_ids,
                        'attention_mask': torch.ones_like(chunk_ids)
                    })
        
        return chunks
    
    def __len__(self) -> int:
        return len(self.chunks)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.chunks[idx]


class DataCollatorForNPT:
    """
    Data collator for NPT training.
    
    Handles padding and creates labels for language modeling.
    """
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        pad_to_multiple_of: Optional[int] = None,
        include_labels: bool = True
    ):
        """
        Initialize data collator.
        
        Args:
            tokenizer: Tokenizer for padding
            pad_to_multiple_of: Pad to multiple of this value
            include_labels: Whether to include labels for LM loss
        """
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.include_labels = include_labels
    
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of features.
        
        Args:
            features: List of feature dictionaries
        
        Returns:
            Batch dictionary with padded tensors
        """
        # Find max length in batch
        max_length = max(len(f['input_ids']) for f in features)
        
        # Pad to multiple if specified
        if self.pad_to_multiple_of:
            max_length = ((max_length + self.pad_to_multiple_of - 1) 
                         // self.pad_to_multiple_of) * self.pad_to_multiple_of
        
        # Pad features
        batch = {
            'input_ids': [],
            'attention_mask': []
        }
        
        for feature in features:
            input_ids = feature['input_ids']
            attention_mask = feature['attention_mask']
            
            # Pad input_ids
            padding_length = max_length - len(input_ids)
            if padding_length > 0:
                input_ids = torch.cat([
                    input_ids,
                    torch.full((padding_length,), self.tokenizer.pad_token_id)
                ])
                attention_mask = torch.cat([
                    attention_mask,
                    torch.zeros(padding_length, dtype=torch.long)
                ])
            
            batch['input_ids'].append(input_ids)
            batch['attention_mask'].append(attention_mask)
        
        # Stack into tensors
        batch['input_ids'] = torch.stack(batch['input_ids'])
        batch['attention_mask'] = torch.stack(batch['attention_mask'])
        
        # Add labels for language modeling if requested
        if self.include_labels:
            # Labels are same as input_ids for language modeling
            # Set padding tokens to -100 to ignore in loss
            labels = batch['input_ids'].clone()
            labels[batch['attention_mask'] == 0] = -100
            batch['labels'] = labels
        
        return batch


def create_data_loaders(
    train_data: Union[str, Path, List[str]],
    val_data: Optional[Union[str, Path, List[str]]],
    tokenizer: AutoTokenizer,
    batch_size: int = 8,
    max_length: int = 512,
    stride: int = 256,
    num_workers: int = 0,
    include_labels: bool = True
) -> tuple[DataLoader, Optional[DataLoader]]:
    """
    Create data loaders for training and validation.
    
    Args:
        train_data: Training data source
        val_data: Validation data source (optional)
        tokenizer: Tokenizer for encoding
        batch_size: Batch size for data loaders
        max_length: Maximum sequence length
        stride: Stride for sliding window
        num_workers: Number of data loading workers
        include_labels: Whether to include labels
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = TextDataset(
        train_data,
        tokenizer,
        max_length=max_length,
        stride=stride
    )
    
    val_dataset = None
    if val_data is not None:
        val_dataset = TextDataset(
            val_data,
            tokenizer,
            max_length=max_length,
            stride=stride
        )
    
    # Create data collator
    collator = DataCollatorForNPT(
        tokenizer,
        pad_to_multiple_of=8,
        include_labels=include_labels
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    return train_loader, val_loader


class InfiniteDataLoader:
    """
    Infinite data loader that cycles through the dataset.
    
    Useful for training with a fixed number of steps rather than epochs.
    """
    
    def __init__(self, data_loader: DataLoader):
        """
        Initialize infinite data loader.
        
        Args:
            data_loader: Base data loader to cycle through
        """
        self.data_loader = data_loader
        self.iterator = None
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.iterator is None:
            self.iterator = iter(self.data_loader)
        
        try:
            batch = next(self.iterator)
        except StopIteration:
            # Restart iterator
            self.iterator = iter(self.data_loader)
            batch = next(self.iterator)
        
        return batch
    
    def __len__(self):
        return len(self.data_loader)