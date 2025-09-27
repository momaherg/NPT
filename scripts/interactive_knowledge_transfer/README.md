# Interactive Knowledge Transfer Tool

A sophisticated REPL interface for extracting, manipulating, and injecting modulations in Neuro-Plastic Transformer (NPT) models. This tool enables interactive exploration of knowledge transfer and context manipulation in NPT layers.

## 📁 Project Structure

```
interactive_knowledge_transfer/
├── __init__.py                 # Package initialization
├── cli.py                      # Main CLI interface (~550 lines)
│
├── core/                       # Core functionality
│   ├── __init__.py
│   ├── data_types.py          # Data structures (ModulationData, SessionState, Colors)
│   ├── modulation_ops.py      # Extraction/injection operations
│   └── arithmetic_ops.py       # Mathematical operations on modulations
│
├── commands/                   # Command handlers
│   ├── __init__.py
│   ├── base.py                # Base command handler class
│   ├── tracking.py            # Token tracking commands
│   └── modulation_mgmt.py     # Modulation management commands
│
└── utils/                      # Utility modules
    ├── __init__.py
    ├── display.py             # Display formatting and visualization
    ├── generation.py          # Token generation utilities
    └── model_utils.py         # Model loading and configuration
```

## 🏗️ Architecture Overview

### Core Components

#### 1. **Data Types** (`core/data_types.py`)
- **`ModulationData`**: Container for modulation tensors with support for:
  - Single modulation (v_a, v_b)
  - Dual modulation (gate and up projections)
  - Triple modulation (gate, up, and down projections)
  - Rank-k support (multiple rank-1 components)
- **`SessionState`**: Maintains session state including modulation bank, tracked tokens, and history
- **`Colors`**: ANSI color codes for terminal output

#### 2. **Modulation Operations** (`core/modulation_ops.py`)
- **`extract_modulation()`**: Extract modulation from a specific position
- **`extract_sequence_modulations()`**: Extract and average from all token positions
- **`create_injection_hook()`**: Create PyTorch hooks for modulation injection

#### 3. **Arithmetic Operations** (`core/arithmetic_ops.py`)
- Mathematical operations on modulations:
  - `add()`, `subtract()`, `average()`, `scale()`
- Compatibility checking between modulations
- Supports all modulation types (single/dual/triple)

### Command System

Commands are organized into specialized handlers:

#### **Tracking Commands** (`commands/tracking.py`)
- `track`: Monitor specific token probabilities
- `untrack`: Stop tracking tokens
- `clear`: Clear all tracked tokens

#### **Modulation Commands** (`commands/modulation_mgmt.py`)
- `extract`: Extract modulation from specific position
- `extract-seq`: Extract and average from all positions
- `list`: Show saved modulations
- `delete`: Remove saved modulation
- `info`: Display modulation details
- Arithmetic: `add`, `subtract`, `average`, `scale`, `negate`

### Utilities

#### **Display Utils** (`utils/display.py`)
- Format and display predictions with probability changes
- Show tracked token evolution
- Comparison tables and visualizations

#### **Generation Utils** (`utils/generation.py`)
- Compute logits with/without injection
- Token sampling strategies (greedy, top-k, top-p)
- Multi-token generation with continuous injection

#### **Model Utils** (`utils/model_utils.py`)
- Load NPT models with selective layer activation
- Detect modulation configuration (single/dual/triple)
- Handle checkpoint loading and weight restoration

## 🔧 Adding New Features

### Adding a New Command

1. **Create the command handler** in appropriate module:

```python
# In commands/modulation_mgmt.py
def cmd_new_feature(self, args: List[str]):
    """Your new feature description."""
    if len(args) < required_args:
        self.error("Usage: new-feature <args>")
        return

    # Implementation
    name = args[0]
    # ... your logic here

    self.success(f"Feature executed successfully")
```

2. **Wire up in CLI** (`cli.py`):

```python
# In execute_command() method
elif cmd == 'new-feature':
    self.modulation_commands.cmd_new_feature(parts[1:])
```

3. **Update help text** in `show_help()`:

```python
{Colors.CYAN}Your Category:{Colors.END}
  new-feature <args>    - Description of your feature
```

### Adding New Modulation Operations

1. **Add method to `ModulationOperations`**:

```python
def your_operation(self, prompt: str, params) -> Dict[int, ModulationData]:
    """Your operation description."""
    # Implementation
    return modulations
```

2. **Create command handler** following the pattern above

### Extending Arithmetic Operations

Add to `ModulationArithmetic` class:

```python
@staticmethod
def your_operation(mod1: Dict[int, ModulationData], ...) -> Dict[int, ModulationData]:
    """Operation description."""
    result = {}
    for layer_idx in mod1.keys():
        # Process each layer
        result[layer_idx] = processed_mod
    return result
```

## 🎯 Key Design Patterns

### 1. **Hook-based Extraction**
Uses PyTorch hooks to capture modulations during forward pass:
```python
def create_hook(layer_idx):
    def hook(module, input, output):
        # Capture modulation tensors
        modulations[layer_idx] = extract_from_output(output)
    return hook
```

### 2. **Modulation Types Detection**
Automatically detects modulation configuration:
- Check if output is tuple → dual/triple modulation
- Check tuple length → determine if triple (3) or dual (2)
- Check tensor dimensions → determine rank-k configuration

### 3. **Efficient Batched Operations**
For sequence extraction, captures all positions in single forward pass:
```python
for pos in range(seq_len):
    mod_data = extract_at_position(pos)
    position_mods.append(mod_data)
# Average after collection
averaged = torch.stack(position_mods).mean(dim=0)
```

### 4. **Command Pattern**
Commands follow consistent pattern:
- Parse arguments
- Validate input
- Execute operation
- Store/display results
- Use `self.error()`, `self.success()`, `self.warning()` for feedback

## 🔍 Important Implementation Details

### Memory Management
- Hooks are always removed after use
- Tensors are detached and cloned during extraction
- CUDA cache cleared periodically during long operations

### Tensor Shape Conventions
- Single position: `[batch, 1, dim]` or `[batch, 1, num_ranks, dim]`
- Averaged: Same shape but represents average across positions
- Position -1 indicates averaged modulation

### Modulation Compatibility
For arithmetic operations, modulations must have:
- Same layers
- Same modulation type (single/dual/triple)
- Same number of ranks
- Same tensor shapes

### Session Persistence
- Session state can be saved/loaded using pickle
- Includes modulation bank, tracked tokens, and history
- Modulation tensors are preserved with their device placement

## 📝 Usage Examples

### Basic Workflow
```python
# Extract modulation from specific position
> extract context "The capital of France is" -pos 4

# Extract and average from all positions
> extract-seq knowledge "The capital of France is Paris"

# Perform arithmetic
> subtract knowledge context difference

# Inject into new context
> inject difference "The capital of Germany is"
```

### Advanced Features
```python
# Multi-token generation with injection
> inject knowledge "Answer:" -tokens 20 -temp 0.8 -strategy top_p

# Track specific tokens
> track Berlin Paris London
> test "The capital of Germany is"

# Compare baseline vs injection
> compare knowledge "The capital of Germany is"
```

## 🐛 Debugging Tips

1. **Check modulation compatibility** before arithmetic operations
2. **Verify active layers** match loaded checkpoint
3. **Monitor memory usage** for long sequences
4. **Use `info <name>` to inspect modulation details**
5. **Check tensor shapes** when adding new operations

## 🚀 Performance Considerations

- **Single forward pass** for sequence extraction (not one per position)
- **Lazy evaluation** - modulations computed only when needed
- **Selective layer activation** - only process NPT layers
- **Batched operations** where possible
- **Hook cleanup** to prevent memory leaks

## 📚 Dependencies

- PyTorch (for tensor operations and hooks)
- Transformers (for tokenization)
- NumPy (for statistics)
- Custom NPT implementation (`src.npt`)

## 🔗 Integration Points

The tool integrates with:
- NPT model architecture (`NPTLlamaModel`)
- NPT configuration (`NPTConfig`)
- Training checkpoints (loads `npt_weights.pt`)
- Tokenizers (AutoTokenizer compatible)

## 🎓 Understanding Modulations

Modulations represent dynamic weight updates in NPT layers:
- **v_a**: Encodes attention/context information
- **v_b**: Transformation vector
- **Rank-1 update**: ΔW = v_b ⊗ v_a
- **Applied as**: output = (W_base @ x) + v_b * (v_a @ x)

The tool allows extracting these modulations from one context and injecting them into another, enabling knowledge transfer and context manipulation.

## 📖 Further Reading

- See `CLAUDE.md` for NPT architecture details
- Check training scripts for how models are trained
- Review test files for usage examples