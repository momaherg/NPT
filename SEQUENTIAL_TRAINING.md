# Sequential Layer-by-Layer NPT Training

## Overview

This system trains NPT layers sequentially, one at a time, starting from lower layers and progressing upward. This approach addresses convergence issues when training all layers simultaneously by allowing each layer to learn its transformation independently.

## Implementation

### Core Scripts

1. **`scripts/train_sequential_layers.py`** - Main orchestrator
   - Manages sequential training of specified layers
   - Accumulates trained NPT weights progressively
   - Handles checkpoint management and recovery
   - Creates model loading scripts

2. **`scripts/train_single_layer_npt.py`** - Single-layer trainer (enhanced)
   - Added `--load_npt_weights` parameter to load previously trained layers
   - Filters out current layer weights to avoid overwriting during training
   - Maintains all existing single-layer optimizations

3. **`run_sequential_training.sh`** - Convenient launcher script
   - Provides easy command-line interface
   - Supports different model sizes (1B, 3B, 8B)
   - Includes demo mode for testing

### Key Features

- **Progressive Learning**: Each layer builds upon previously trained layers
- **Checkpoint Management**: Automatic saving and merging of layer weights
- **Resume Capability**: Can resume from any layer if interrupted
- **Flexible Configuration**: Train all layers, subsets, or custom sequences
- **Memory Efficient**: Only one NPT layer is actively training at a time

## Usage

### Basic Commands

```bash
# Train all 16 layers of Llama 1B model sequentially
./run_sequential_training.sh

# Demo mode - train first 4 layers with minimal steps
./run_sequential_training.sh --demo

# Train specific layers only
./run_sequential_training.sh --layers "0,1,2,3,4,5"

# Train with more steps per layer
./run_sequential_training.sh --steps 3000

# Use larger dataset
./run_sequential_training.sh --dataset medium
```

### Advanced Usage

```bash
# Train upper half of layers
python scripts/train_sequential_layers.py \
    --layers upper_half \
    --steps_per_layer 2000 \
    --model_size 1b

# Resume training from layer 8
python scripts/train_sequential_layers.py \
    --start_from_layer 8 \
    --checkpoint_dir experiments/sequential_checkpoint_20250910_180000

# Custom configuration for 8B model
python scripts/train_sequential_layers.py \
    --model_name "meta-llama/Llama-3.1-8B" \
    --model_size 8b \
    --layers "16,17,18,19,20,21,22,23" \
    --steps_per_layer 3000 \
    --batch_size 1 \
    --gradient_accumulation_steps 64
```

## Training Process

### Per-Layer Training Flow

1. **Layer N Training**:
   - Load base model
   - Convert layer N to NPT
   - Load accumulated weights from layers 0 to N-1 (if any)
   - Train layer N for specified steps (default: 2000)
   - Save layer N weights

2. **Weight Accumulation**:
   - Merge layer N weights into accumulated checkpoint
   - Update training info JSON with completed layers

3. **Progress to Layer N+1**:
   - Repeat process with all previous weights loaded

### Checkpoint Structure

```
experiments/sequential_checkpoint_*/
├── training_info.json           # Tracks completed layers and paths
├── accumulated_npt_weights.pt   # Combined weights from all trained layers
├── layer_0_*/                   # Individual layer training outputs
│   └── checkpoints/
│       └── final/
│           └── npt_weights.pt
├── layer_1_*/
│   └── ...
└── load_model.py                # Auto-generated script to load the model
```

### Training Info Format

```json
{
  "trained_layers": [0, 1, 2, 3],
  "layer_checkpoints": {
    "0": "experiments/sequential/layer_0_20250910_180000",
    "1": "experiments/sequential/layer_1_20250910_181000",
    "2": "experiments/sequential/layer_2_20250910_182000",
    "3": "experiments/sequential/layer_3_20250910_183000"
  }
}
```

## Configuration Options

### Layer Selection

- `--layers all`: Train all layers (0 to num_layers-1)
- `--layers upper_half`: Train upper half of layers
- `--layers lower_half`: Train lower half of layers
- `--layers "0,1,2,3"`: Train specific layers (comma-separated)

### Training Parameters

- `--steps_per_layer`: Training steps per layer (default: 2000)
- `--stage1_steps`: Attention reconstruction steps (default: 500)
- `--batch_size`: Batch size per device (default: 2)
- `--learning_rate`: Learning rate (default: 5e-4)
- `--np_rank`: NPT component rank (default: 256)

### Dataset Options

- `--dataset_preset small`: WikiText-103 only
- `--dataset_preset medium`: WikiText + BookCorpus
- `--dataset_preset large`: WikiText + BookCorpus + OpenWebText

## Model Loading

After training, use the auto-generated script:

```python
# Load the sequentially trained model
exec(open("experiments/sequential_checkpoint_*/load_model.py").read())

# Or manually:
from src.npt import NPTLlamaModel, NPTConfig

model = NPTLlamaModel.from_pretrained("meta-llama/Llama-3.2-1B")

# Convert the trained layers
npt_config = NPTConfig(
    layers_to_convert=[0, 1, 2, 3],  # Your trained layers
    np_rank=256,
    single_layer_mode=False
)
model.convert_to_npt(npt_config)

# Load accumulated weights
weights = torch.load("path/to/accumulated_npt_weights.pt")
model.load_npt_weights(weights)
```

## Benefits

1. **Better Convergence**: Each layer gets dedicated training without interference
2. **Stable Learning**: Previously trained layers provide stable foundation
3. **Flexible**: Can stop/resume at any layer
4. **Debuggable**: Can inspect per-layer performance
5. **Memory Efficient**: Only one active NPT layer during training

## Monitoring

Each layer's training is tracked separately in WandB:
- Project: `npt-sequential`
- Run names: `layer_0_1b_timestamp`, `layer_1_1b_timestamp`, etc.
- Tags: `sequential`, `layer_N`

## Recommendations

1. **Start with lower layers**: They learn simpler transformations
2. **Use 2000-3000 steps per layer**: Sufficient for convergence
3. **Monitor v_a attention similarity**: Should reach >0.8 in Stage 1
4. **Check for mode collapse**: v_a and v_b norms shouldn't approach zero
5. **Save checkpoints frequently**: Use `--save_steps 500` for safety

## Troubleshooting

### If training fails at layer N:
1. Check `training_info.json` for last completed layer
2. Resume with `--start_from_layer N`
3. Accumulated weights are preserved

### If convergence is poor:
1. Increase `--steps_per_layer` to 3000-5000
2. Adjust `--stage1_steps` to 1000 for better attention encoding
3. Try lower learning rate (1e-4 or 5e-5)

### For memory issues:
1. Reduce `--batch_size` to 1
2. Increase `--gradient_accumulation_steps` to 64
3. Use `--mixed_precision` flag