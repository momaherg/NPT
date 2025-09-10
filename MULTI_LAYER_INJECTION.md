# Multi-Layer NPT Knowledge Injection

## Overview

The enhanced `interactive_knowledge_injection.py` script now supports working with checkpoints containing multiple NPT layers. This enables more sophisticated knowledge injection experiments, allowing you to:

- Load checkpoints from sequential training with multiple NPT layers
- Switch between layers dynamically during experiments
- Inject knowledge into specific layers or all layers simultaneously
- Compare injection behavior across different layers

## Key Features

### 1. Automatic Multi-Layer Detection

When loading a checkpoint, the script automatically:
- Detects all NPT layers present in the checkpoint
- Identifies the rank of each layer (supporting different ranks per layer)
- Converts all detected layers to NPT
- Shows a summary of available layers

```bash
python scripts/interactive_knowledge_injection.py \
  --checkpoint experiments/sequential_checkpoint_20250910_180000 \
  --model_name "meta-llama/Llama-3.2-1B"
```

Output:
```
Loading checkpoint from experiments/sequential_checkpoint_20250910_180000...
  Found accumulated weights from sequential training
  Detected NPT layers: [0, 1, 2, 3, 4, 5, 6, 7]
    Layer 0: rank=256
    Layer 1: rank=256
    ...
  Converted 8 layers to NPT
✓ Checkpoint loaded successfully!

Model Information:
  NPT layers: [0, 1, 2, 3, 4, 5, 6, 7]
  Active layer: 7 (use 'layer <idx>' to switch)
```

### 2. New Interactive Commands

#### Layer Management Commands

- **`layers`** - Display all available NPT layers with their properties
  ```
  NPT> layers
  NPT Layers Information:
    Available layers: [0, 1, 2, 3, 4, 5, 6, 7]
    Active layer: 7
    
    Layer details:
      Layer 0: rank=256, injected_facts=0
      Layer 1: rank=256, injected_facts=2
      Layer 7: rank=256, injected_facts=1 (active)
  ```

- **`layer <idx>`** - Switch to a specific NPT layer
  ```
  NPT> layer 3
  ✓ Switched to layer 3
  ```

#### Multi-Layer Injection Commands

- **`inject-all <fact>`** - Inject a fact into ALL NPT layers simultaneously
  ```
  NPT> inject-all The capital of France is Paris
  
  Injecting into 8 NPT layers...
  
  Layer 0:
  ✓ Knowledge injected successfully!
    - Delta weight norm: 0.002341
  
  Layer 1:
  ✓ Knowledge injected successfully!
    - Delta weight norm: 0.002156
  ...
  ```

- **`reset-all`** - Reset all layers to their original state
  ```
  NPT> reset-all
  Reset ALL layers to original? (y/n) [n]: y
  Reset 8 layer(s) to original state.
  ```

#### Enhanced History Command

- **`history`** - Shows injection history organized by layer
  ```
  NPT> history
  
  Injection History:
  
  Layer 0:
    1. The capital of France is Paris
       Alpha: 1.00, Position: last
       Weight change: 0.000234
  
  Layer 3:
    1. The president of USA is Joe Biden
       Alpha: 1.50, Position: all
       Weight change: 0.000412
  ```

### 3. Per-Layer Tracking

The system now maintains separate tracking for each layer:
- **Original weights** stored per layer
- **Injection history** tracked per layer
- **Active layer** for current operations
- **Independent reset** capability

### 4. Enhanced Save Functionality

When saving a modified model, the system preserves:
- All NPT layer weights
- Complete injection history for all layers
- Layer configuration metadata
- Active layer information

```python
# Saved injection_history.json structure
{
  "injected_facts": {
    "0": [...],  # Facts injected into layer 0
    "3": [...],  # Facts injected into layer 3
    ...
  },
  "active_layer": 7,
  "available_layers": [0, 1, 2, 3, 4, 5, 6, 7],
  "model_info": {
    "npt_layers": [0, 1, 2, 3, 4, 5, 6, 7],
    "npt_ranks": {
      "0": 256,
      "1": 256,
      ...
    }
  }
}
```

## Usage Examples

### Example 1: Layer-Specific Injection

```bash
# Start interactive session with multi-layer checkpoint
python scripts/interactive_knowledge_injection.py \
  --checkpoint experiments/sequential_checkpoint_20250910_180000 \
  --model_name "meta-llama/Llama-3.2-1B"

# In the interactive session:
NPT> layers                          # View available layers
NPT> layer 5                         # Switch to layer 5
NPT> inject The sky is purple        # Inject into layer 5
NPT> test What color is the sky?     # Test layer 5's response
NPT> layer 10                        # Switch to layer 10
NPT> inject The sky is green         # Inject into layer 10
NPT> test What color is the sky?     # Test layer 10's response
```

### Example 2: Comparative Analysis

```bash
# Inject same fact into different layers and compare
NPT> layer 0
NPT> inject Paris is the capital of France
NPT> ask What is the capital of France?
# Record response...

NPT> reset
NPT> layer 7
NPT> inject Paris is the capital of France
NPT> ask What is the capital of France?
# Compare responses to see which layer is more effective
```

### Example 3: Bulk Knowledge Injection

```bash
# Inject multiple facts into all layers at once
NPT> inject-all The president of USA is Joe Biden
NPT> inject-all The capital of Germany is Berlin
NPT> inject-all The largest planet is Jupiter

# Test the combined effect
NPT> ask Who is the president of USA?
NPT> ask What is the capital of Germany?
NPT> ask Which planet is the largest?
```

## Benefits

1. **Layer-Specific Analysis**: Understand which layers are best for different types of knowledge
2. **Comparative Studies**: Compare injection effectiveness across layers
3. **Comprehensive Editing**: Modify multiple layers simultaneously for stronger effects
4. **Flexible Experimentation**: Switch between layers without restarting
5. **Complete Preservation**: Save and restore multi-layer modifications

## Technical Details

### Layer Detection Algorithm

The system detects NPT layers by parsing weight keys:
- Format 1: `layer_{idx}_np.{param_name}` (from save_npt_weights)
- Format 2: `model.layers.{idx}.np_component.{param_name}` (from state_dict)

### Rank Handling

Each layer can have a different rank:
- Detected from W_down shape: `[hidden_size, rank]`
- Preserved during loading
- Displayed in layer information

### Memory Management

- Original weights stored only when first injection occurs
- Per-layer storage prevents cross-contamination
- Efficient switching without reloading

## Compatibility

The enhanced script is fully backward compatible:
- Works with single-layer checkpoints
- Supports `--layer_idx` argument for default layer
- Falls back gracefully when no NPT layers detected
- Maintains all original commands and functionality

## Recommendations

1. **Start with Higher Layers**: Upper layers often better for factual knowledge
2. **Test Layer Sensitivity**: Different layers may respond differently to injection strength
3. **Use Layer Comparison**: Test same fact across layers to find optimal placement
4. **Combine Layers**: Some knowledge may benefit from multi-layer injection
5. **Monitor Weight Changes**: Use `history` to track modification magnitudes per layer