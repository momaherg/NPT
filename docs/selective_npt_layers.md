# Selective NPT Layer Loading

## Overview

The NPT architecture now supports **selective layer loading**, allowing you to choose which specific layers operate in NPT mode while others remain as standard transformer layers with attention residual connections. This feature is crucial for experiments and comparisons.

## Key Concepts

### NPT Mode vs Standard Mode

- **NPT Mode**: Layer operates without attention residual, using rank-1 weight modulation
  ```
  h → Attention → NP Component → Modulated MLP(h) → output
  ```

- **Standard Mode**: Layer operates as normal transformer with attention residual
  ```
  h → Attention → h + attn → MLP(h + attn) → output
  ```

## Usage

### Command-Line Options

```bash
# Load all available NPT layers in NPT mode (default)
python scripts/interactive_knowledge_injection.py \
  --checkpoint experiments/checkpoint \
  --use_npt_layers all

# Use only specific layers in NPT mode
python scripts/interactive_knowledge_injection.py \
  --checkpoint experiments/checkpoint \
  --use_npt_layers "15,31"

# Load NPT weights but keep all layers in standard mode
python scripts/interactive_knowledge_injection.py \
  --checkpoint experiments/checkpoint \
  --use_npt_layers none
```

### Interactive Commands

During an interactive session, you can dynamically switch layer modes:

```
# Show all layers and their current modes
NPT> layers

# Show mode summary
NPT> modes

# Toggle layer 15 between NPT and standard
NPT> mode 15

# Set layer 15 to NPT mode explicitly
NPT> mode 15 npt

# Set layer 15 to standard mode
NPT> mode 15 standard
```

## Implementation Details

### How It Works

1. **Detection**: The system auto-detects available NPT weights from checkpoint
2. **Selection**: You specify which layers should operate in NPT mode
3. **Conversion**: Only selected layers are converted to NPT architecture
4. **Loading**: NPT weights are loaded for all available layers
5. **Mode Setting**: Non-selected layers are explicitly set to standard mode

### Code Example

```python
from src.npt import NPTLlamaModel, NPTConfig

# Load model
model = NPTLlamaModel.from_pretrained("meta-llama/Llama-3.2-1B")

# Convert specific layers to NPT
npt_config = NPTConfig(
    layers_to_convert=[15, 31],  # Only these layers
    np_rank=256,
    single_layer_mode=False
)
model.convert_to_npt(npt_config)

# Load weights
model.load_npt_weights(checkpoint_weights)

# Later, toggle modes dynamically
model.model.layers[15].set_npt_mode(False)  # Switch to standard
model.model.layers[15].set_npt_mode(True)   # Switch back to NPT
```

## Use Cases

### 1. Ablation Studies
Compare model performance with different NPT layer configurations:
- All layers NPT vs all standard
- Only upper layers NPT vs only lower layers
- Single critical layer vs multiple layers

### 2. Memory Optimization
Run only essential layers in NPT mode to reduce memory footprint while maintaining key functionality.

### 3. Debugging
Isolate issues by toggling individual layers between modes.

### 4. Knowledge Injection Experiments
Test which layers are most effective for knowledge injection by selectively enabling NPT mode.

## Technical Benefits

1. **Flexibility**: Mix NPT and standard layers in any configuration
2. **Comparison**: Easy A/B testing between modes
3. **Efficiency**: Reduced computational overhead when not all layers need NPT
4. **Debugging**: Isolate layer-specific behaviors

## Example Experiment

```bash
# Test knowledge injection with only layer 15 as NPT
python scripts/interactive_knowledge_injection.py \
  --checkpoint experiments/sequential_checkpoint \
  --use_npt_layers "15"

# In interactive mode:
NPT> inject "The capital of France is Paris"
NPT> test "What is the capital of France?"

# Switch layer 15 to standard mode to compare
NPT> mode 15 standard
NPT> test "What is the capital of France?"

# Try with multiple layers
NPT> mode 15 npt
NPT> mode 31 npt
NPT> test "What is the capital of France?"
```

## Important Notes

- Layers with NPT weights can switch between modes at runtime
- Standard mode preserves the attention residual connection
- NPT mode removes the attention residual for weight modulation
- Mode switching doesn't affect the loaded weights, only the forward pass behavior