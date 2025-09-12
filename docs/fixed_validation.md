# Fixed Validation Dataset

## Overview
The fixed validation dataset feature provides consistent evaluation metrics throughout training by caching a fixed set of validation samples. This solves the problem of inconsistent metrics from streaming validation where different data is evaluated each time.

## Key Benefits
- **Consistent Metrics**: Same validation samples every evaluation for true progress tracking
- **Deterministic**: Reproducible results across runs with same seed
- **Efficient**: Cached in memory, no re-tokenization needed
- **Configurable**: Choose number of samples based on memory constraints

## Usage

### Command Line
```bash
# Use fixed validation (default)
python scripts/train_single_layer_npt.py \
  --fixed_validation \
  --num_validation_samples 500

# Use streaming validation (old behavior)
python scripts/train_single_layer_npt.py \
  --no-fixed_validation
```

### Programmatic
```python
from src.training.streaming_data import create_fixed_validation_loader

# Create fixed validation loader
val_loader = create_fixed_validation_loader(
    tokenizer=tokenizer,
    dataset_name="wikitext",
    dataset_config="wikitext-2-raw-v1",
    batch_size=8,
    max_length=512,
    num_validation_samples=500,
    seed=42
)

# Or with MultiDatasetStreamer
streamer = MultiDatasetStreamer(preset="small", tokenizer=tokenizer)
train_loader, val_loader = streamer.create_data_loaders(
    fixed_validation=True,
    num_validation_samples=500
)
```

## Implementation Details

### FixedValidationDataset
- Loads validation split (or test/train subset as fallback)
- Tokenizes and chunks into fixed-length sequences
- Caches all samples in memory
- Uses seed for reproducible sample selection

### Evaluation Behavior
- **Fixed validation**: Evaluates all cached samples (no batch limit)
- **Streaming validation**: Limited to 10 batches (old behavior)

## Recommended Settings
- **Small models (1B)**: 500 samples
- **Medium models (3B)**: 300-400 samples
- **Large models (8B+)**: 200-300 samples

Adjust based on available memory and desired evaluation accuracy.

## Comparison

| Feature | Fixed Validation | Streaming Validation |
|---------|-----------------|---------------------|
| Consistency | ✅ Same samples every time | ❌ Different samples |
| Memory Usage | Higher (cached) | Lower (streaming) |
| Evaluation Speed | Faster (pre-tokenized) | Slower (on-the-fly) |
| Batch Limit | No limit | 10 batches |
| Progress Tracking | Accurate | Noisy |