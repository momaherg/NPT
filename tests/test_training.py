"""
Unit tests for NPT training pipeline.

Tests cover data loading, trainer functionality, checkpointing,
and the complete training loop.
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import shutil
from pathlib import Path
import json
import sys

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training import (
    TextDataset,
    DataCollatorForNPT,
    create_data_loaders,
    InfiniteDataLoader,
    NPTTrainer,
    TrainingConfig,
    TrainingMetrics
)
from src.npt import NPTLlamaModel, NPTConfig
from transformers import AutoTokenizer, LlamaConfig


class TestTextDataset:
    """Test suite for TextDataset."""
    
    @pytest.fixture
    def tokenizer(self):
        """Get a tokenizer for testing."""
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        # For testing with small models, we need to constrain token IDs
        tokenizer.model_max_length = 1000  # Match test model vocab size
        return tokenizer
    
    @pytest.fixture
    def sample_texts(self):
        """Sample texts for testing."""
        return [
            "This is a test sentence.",
            "Machine learning is fascinating.",
            "Neural networks process information."
        ]
    
    def test_dataset_from_list(self, tokenizer, sample_texts):
        """Test creating dataset from list of texts."""
        dataset = TextDataset(
            data_source=sample_texts,
            tokenizer=tokenizer,
            max_length=128,
            stride=64
        )
        
        assert len(dataset) > 0
        assert all('input_ids' in item for item in dataset.chunks)
        assert all('attention_mask' in item for item in dataset.chunks)
    
    def test_dataset_from_json(self, tokenizer, tmp_path):
        """Test creating dataset from JSON file."""
        # Create JSON file
        json_file = tmp_path / "data.json"
        data = [
            {"text": "First document."},
            {"text": "Second document."}
        ]
        with open(json_file, 'w') as f:
            json.dump(data, f)
        
        dataset = TextDataset(
            data_source=str(json_file),
            tokenizer=tokenizer,
            max_length=128
        )
        
        assert len(dataset) > 0
    
    def test_dataset_from_text_file(self, tokenizer, tmp_path):
        """Test creating dataset from text file."""
        # Create text file
        text_file = tmp_path / "data.txt"
        text_file.write_text("This is a test document.\nWith multiple lines.")
        
        dataset = TextDataset(
            data_source=str(text_file),
            tokenizer=tokenizer,
            max_length=128
        )
        
        assert len(dataset) > 0
    
    def test_chunking_long_text(self, tokenizer):
        """Test that long texts are properly chunked."""
        # Create a long text
        long_text = " ".join(["word"] * 1000)
        
        dataset = TextDataset(
            data_source=[long_text],
            tokenizer=tokenizer,
            max_length=50,
            stride=25
        )
        
        # Should create multiple chunks
        assert len(dataset) > 1
        
        # Check chunk sizes
        for chunk in dataset.chunks:
            assert len(chunk['input_ids']) <= 50
    
    def test_dataset_getitem(self, tokenizer, sample_texts):
        """Test dataset __getitem__ method."""
        dataset = TextDataset(
            data_source=sample_texts,
            tokenizer=tokenizer,
            max_length=128
        )
        
        item = dataset[0]
        assert 'input_ids' in item
        assert 'attention_mask' in item
        assert isinstance(item['input_ids'], torch.Tensor)
        assert isinstance(item['attention_mask'], torch.Tensor)


class TestDataCollator:
    """Test suite for DataCollatorForNPT."""
    
    @pytest.fixture
    def tokenizer(self):
        """Get a tokenizer for testing."""
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        # For testing with small models, we need to constrain token IDs
        tokenizer.model_max_length = 1000  # Match test model vocab size
        return tokenizer
    
    @pytest.fixture
    def collator(self, tokenizer):
        """Create a data collator."""
        return DataCollatorForNPT(
            tokenizer=tokenizer,
            pad_to_multiple_of=8,
            include_labels=True
        )
    
    def test_collator_padding(self, collator, tokenizer):
        """Test that collator properly pads sequences."""
        # Create features of different lengths
        features = [
            {
                'input_ids': torch.tensor([1, 2, 3]),
                'attention_mask': torch.tensor([1, 1, 1])
            },
            {
                'input_ids': torch.tensor([4, 5, 6, 7, 8]),
                'attention_mask': torch.tensor([1, 1, 1, 1, 1])
            }
        ]
        
        batch = collator(features)
        
        # Check shapes are consistent
        assert batch['input_ids'].shape[0] == 2  # batch size
        assert batch['input_ids'].shape[1] == 8  # padded to multiple of 8
        assert batch['attention_mask'].shape == batch['input_ids'].shape
    
    def test_collator_labels(self, collator):
        """Test that collator creates labels correctly."""
        features = [
            {
                'input_ids': torch.tensor([1, 2, 3]),
                'attention_mask': torch.tensor([1, 1, 1])
            }
        ]
        
        batch = collator(features)
        
        assert 'labels' in batch
        # Padding positions should be -100
        assert (batch['labels'][batch['attention_mask'] == 0] == -100).all()
    
    def test_collator_no_labels(self, tokenizer):
        """Test collator without labels."""
        collator = DataCollatorForNPT(
            tokenizer=tokenizer,
            include_labels=False
        )
        
        features = [
            {
                'input_ids': torch.tensor([1, 2, 3]),
                'attention_mask': torch.tensor([1, 1, 1])
            }
        ]
        
        batch = collator(features)
        assert 'labels' not in batch


class TestDataLoaders:
    """Test suite for data loader creation."""
    
    @pytest.fixture
    def tokenizer(self):
        """Get a tokenizer for testing."""
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        # For testing with small models, we need to constrain token IDs
        tokenizer.model_max_length = 1000  # Match test model vocab size
        return tokenizer
    
    def test_create_data_loaders(self, tokenizer):
        """Test creating train and validation data loaders."""
        train_data = ["Train text 1.", "Train text 2."]
        val_data = ["Val text 1.", "Val text 2."]
        
        train_loader, val_loader = create_data_loaders(
            train_data=train_data,
            val_data=val_data,
            tokenizer=tokenizer,
            batch_size=2,
            max_length=128
        )
        
        assert train_loader is not None
        assert val_loader is not None
        assert len(train_loader) > 0
        assert len(val_loader) > 0
        
        # Test getting a batch
        batch = next(iter(train_loader))
        assert 'input_ids' in batch
        assert 'attention_mask' in batch
        assert 'labels' in batch
    
    def test_infinite_data_loader(self, tokenizer):
        """Test infinite data loader."""
        data = ["Text 1.", "Text 2."]
        
        train_loader, _ = create_data_loaders(
            train_data=data,
            val_data=None,
            tokenizer=tokenizer,
            batch_size=1
        )
        
        infinite_loader = InfiniteDataLoader(train_loader)
        
        # Should be able to get more batches than in dataset
        batches = []
        for i, batch in enumerate(infinite_loader):
            batches.append(batch)
            if i >= len(train_loader) * 2:  # Get twice the number of batches
                break
        
        assert len(batches) > len(train_loader)


class TestTrainingConfig:
    """Test suite for TrainingConfig."""
    
    def test_config_creation(self):
        """Test creating training config."""
        config = TrainingConfig(
            batch_size=16,
            learning_rate=2e-4,
            max_steps=5000
        )
        
        assert config.batch_size == 16
        assert config.learning_rate == 2e-4
        assert config.max_steps == 5000
    
    def test_config_save_load(self, tmp_path):
        """Test saving and loading config."""
        config = TrainingConfig(
            batch_size=16,
            learning_rate=2e-4,
            output_dir=str(tmp_path)
        )
        
        # Save config
        config_path = tmp_path / "config.json"
        config.save(config_path)
        
        # Load config
        loaded_config = TrainingConfig.load(config_path)
        
        assert loaded_config.batch_size == config.batch_size
        assert loaded_config.learning_rate == config.learning_rate
        assert loaded_config.output_dir == config.output_dir


class TestNPTTrainer:
    """Test suite for NPTTrainer."""
    
    @pytest.fixture
    def small_model(self):
        """Create a small NPT model for testing."""
        config = LlamaConfig(
            hidden_size=128,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=1000,
        )
        config._attn_implementation = "eager"
        
        model = NPTLlamaModel(config)
        npt_config = NPTConfig(
            convert_all=True,
            np_rank=16,
            np_init_scale=0.01
        )
        model.convert_to_npt(npt_config)
        model.freeze_base_parameters()
        
        return model
    
    @pytest.fixture
    def tokenizer(self):
        """Get a tokenizer for testing."""
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        # For testing with small models, we need to constrain token IDs
        tokenizer.model_max_length = 1000  # Match test model vocab size
        return tokenizer
    
    @pytest.fixture
    def data_loaders(self, tokenizer):
        """Create data loaders for testing."""
        train_data = ["Train text."] * 10
        val_data = ["Val text."] * 5
        
        train_loader, val_loader = create_data_loaders(
            train_data=train_data,
            val_data=val_data,
            tokenizer=tokenizer,
            batch_size=2,
            max_length=32
        )
        
        # Clip token IDs to model's vocab size for testing
        def clip_batch(batch):
            batch['input_ids'] = batch['input_ids'] % 1000  # Model vocab size
            if 'labels' in batch:
                batch['labels'][batch['labels'] != -100] = batch['labels'][batch['labels'] != -100] % 1000
            return batch
        
        # Wrap loaders to clip token IDs
        import types
        
        original_iter = train_loader.__iter__
        def new_iter(self):
            for batch in original_iter():
                yield clip_batch(batch)
        train_loader.__iter__ = types.MethodType(new_iter, train_loader)
        
        if val_loader:
            original_val_iter = val_loader.__iter__
            def new_val_iter(self):
                for batch in original_val_iter():
                    yield clip_batch(batch)
            val_loader.__iter__ = types.MethodType(new_val_iter, val_loader)
        
        return train_loader, val_loader
    
    @pytest.fixture
    def training_config(self, tmp_path):
        """Create training config for testing."""
        return TrainingConfig(
            batch_size=2,
            learning_rate=1e-3,
            max_steps=10,
            warmup_steps=2,
            logging_steps=2,
            eval_steps=5,
            save_steps=5,
            output_dir=str(tmp_path / "output"),
            device="cpu"  # Use CPU for testing
        )
    
    def test_trainer_initialization(self, small_model, training_config, data_loaders):
        """Test trainer initialization."""
        train_loader, val_loader = data_loaders
        
        trainer = NPTTrainer(
            model=small_model,
            config=training_config,
            train_loader=train_loader,
            val_loader=val_loader
        )
        
        assert trainer.model is small_model
        assert trainer.config is training_config
        assert trainer.train_loader is train_loader
        assert trainer.val_loader is val_loader
        assert trainer.global_step == 0
    
    def test_train_step(self, small_model, training_config, data_loaders):
        """Test single training step."""
        train_loader, _ = data_loaders
        
        trainer = NPTTrainer(
            model=small_model,
            config=training_config,
            train_loader=train_loader
        )
        
        # Get a batch
        batch = next(iter(train_loader))
        
        # Perform train step
        metrics = trainer.train_step(batch)
        
        assert isinstance(metrics, TrainingMetrics)
        assert metrics.total_loss >= 0
        assert metrics.fidelity_loss >= 0
        assert metrics.regularization_loss >= 0
        assert metrics.learning_rate > 0
    
    def test_evaluate(self, small_model, training_config, data_loaders):
        """Test evaluation."""
        train_loader, val_loader = data_loaders
        
        trainer = NPTTrainer(
            model=small_model,
            config=training_config,
            train_loader=train_loader,
            val_loader=val_loader
        )
        
        # Evaluate
        eval_metrics = trainer.evaluate()
        
        assert 'val_loss' in eval_metrics
        assert 'val_fidelity_loss' in eval_metrics
        assert 'val_regularization_loss' in eval_metrics
        assert all(v >= 0 for v in eval_metrics.values())
    
    def test_checkpoint_save_load(self, small_model, training_config, data_loaders):
        """Test checkpoint saving and loading."""
        train_loader, _ = data_loaders
        
        trainer = NPTTrainer(
            model=small_model,
            config=training_config,
            train_loader=train_loader
        )
        
        # Train for a few steps
        for i, batch in enumerate(train_loader):
            trainer.train_step(batch)
            trainer.global_step += 1
            if i >= 2:
                break
        
        initial_step = trainer.global_step
        
        # Save checkpoint
        trainer.save_checkpoint("test_checkpoint")
        
        # Create new trainer and load checkpoint
        new_model = small_model
        new_trainer = NPTTrainer(
            model=new_model,
            config=training_config,
            train_loader=train_loader
        )
        
        checkpoint_path = trainer.checkpoint_dir / "test_checkpoint"
        new_trainer.load_checkpoint(str(checkpoint_path))
        
        # Check state was restored
        assert new_trainer.global_step == initial_step
    
    def test_learning_rate_scheduler(self, small_model, training_config, data_loaders):
        """Test learning rate scheduling."""
        train_loader, _ = data_loaders
        
        trainer = NPTTrainer(
            model=small_model,
            config=training_config,
            train_loader=train_loader
        )
        
        initial_lr = trainer.optimizer.param_groups[0]['lr']
        
        # Train for warmup steps
        for i, batch in enumerate(train_loader):
            metrics = trainer.train_step(batch)
            trainer.global_step += 1
            
            if i == 0:
                # During warmup, LR should be less than initial
                assert metrics.learning_rate < initial_lr
            
            if i >= training_config.warmup_steps:
                break
        
        # After warmup, LR should be at or near initial
        final_lr = trainer.optimizer.param_groups[0]['lr']
        assert final_lr >= initial_lr * 0.9  # Allow small difference
    
    def test_gradient_clipping(self, small_model, training_config, data_loaders):
        """Test gradient clipping."""
        train_loader, _ = data_loaders
        
        # Set small gradient clip value
        training_config.gradient_clip_value = 0.1
        
        trainer = NPTTrainer(
            model=small_model,
            config=training_config,
            train_loader=train_loader
        )
        
        batch = next(iter(train_loader))
        metrics = trainer.train_step(batch)
        
        # Check gradients are clipped
        for param in small_model.get_npt_parameters():
            if param.grad is not None:
                assert param.grad.norm() <= training_config.gradient_clip_value * 1.1
    
    def test_train_loop(self, small_model, training_config, data_loaders):
        """Test complete training loop."""
        train_loader, val_loader = data_loaders
        
        # Use very short training for test
        training_config.max_steps = 5
        training_config.eval_steps = 3
        training_config.save_steps = 10  # Don't save during test
        
        trainer = NPTTrainer(
            model=small_model,
            config=training_config,
            train_loader=train_loader,
            val_loader=val_loader
        )
        
        # Run training
        trainer.train()
        
        # Check training completed
        assert trainer.global_step == training_config.max_steps
        
        # Check final checkpoint exists
        final_checkpoint = trainer.checkpoint_dir / "final"
        assert final_checkpoint.exists()


class TestIntegration:
    """Integration tests for the complete training pipeline."""
    
    def test_end_to_end_training(self, tmp_path):
        """Test complete training pipeline end-to-end."""
        # Setup
        config = LlamaConfig(
            hidden_size=128,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=1000,
        )
        config._attn_implementation = "eager"
        
        model = NPTLlamaModel(config)
        npt_config = NPTConfig(convert_all=True, np_rank=16)
        model.convert_to_npt(npt_config)
        model.freeze_base_parameters()
        
        # Create tokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        # Create data
        train_data = ["Training text example."] * 20
        val_data = ["Validation text example."] * 10
        
        train_loader, val_loader = create_data_loaders(
            train_data=train_data,
            val_data=val_data,
            tokenizer=tokenizer,
            batch_size=4,
            max_length=32
        )
        
        # Training config
        training_config = TrainingConfig(
            batch_size=4,
            learning_rate=1e-3,
            max_steps=10,
            warmup_steps=2,
            eval_steps=5,
            save_steps=10,
            output_dir=str(tmp_path / "training"),
            device="cpu"
        )
        
        # Create trainer
        trainer = NPTTrainer(
            model=model,
            config=training_config,
            train_loader=train_loader,
            val_loader=val_loader
        )
        
        # Record initial loss
        initial_metrics = trainer.evaluate()
        initial_loss = initial_metrics['val_loss']
        
        # Train
        trainer.train()
        
        # Check final loss
        final_metrics = trainer.evaluate()
        final_loss = final_metrics['val_loss']
        
        # Loss should decrease (or at least not increase much)
        # Allow some tolerance since we're training for very few steps
        assert final_loss <= initial_loss * 1.1
        
        # Check outputs were created
        assert (Path(training_config.output_dir) / "config.json").exists()
        assert (Path(training_config.output_dir) / "training_log.jsonl").exists()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])