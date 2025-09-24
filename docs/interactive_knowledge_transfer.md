# Interactive Knowledge Transfer Tool

## Overview

The Interactive Knowledge Transfer Tool provides a REPL interface for exploring and manipulating NPT model modulations in real-time. It allows you to extract modulation vectors from one context and inject them into another, observing how this affects token probabilities.

## Key Features

- **Token Tracking**: Monitor specific tokens' probabilities across operations
- **Modulation Extraction**: Extract modulations from prompts at specific positions
- **Modulation Injection**: Inject saved modulations using different modes (replace, blend, add)
- **Real-time Comparison**: See side-by-side comparisons of baseline vs injected results
- **Session Management**: Save/load complete sessions for reproducibility
- **Support for Advanced NPT Features**:
  - Single/Dual/Triple modulation types
  - Rank-k updates (multiple rank-1 components)
  - Multi-layer operations

## Usage

### Starting the Tool

```bash
python scripts/interactive_knowledge_transfer.py \
  --checkpoint experiments/sequential_checkpoint \
  --model_name "meta-llama/Llama-3.2-1B" \
  --layers "14,15" \
  --device cuda
```

### Basic Workflow

```bash
# Track tokens you want to monitor (quotes optional unless token has spaces)
> track paris berlin london
✓ Tracking tokens: 'paris', 'berlin', 'london'

# Track tokens with spaces (use quotes)
> track " Paris" " Berlin" " London"
✓ Tracking tokens: ' Paris', ' Berlin', ' London'

# Extract modulation from a source prompt
> extract paris "The capital of France is"
✓ Extracted modulation 'paris'
  Layer 14: magnitude=0.234567
  Layer 15: magnitude=0.567890

# Test baseline (no injection)
> test "The capital of Germany is"
Top-5 Predictions:
  1. Berlin    : 0.4523
  2. Munich    : 0.1234
  ...
Tracked Tokens:
  • paris  : 0.0234
  • berlin : 0.4523
  • london : 0.0012

# Inject the saved modulation
> inject paris "The capital of Germany is"
Top-5 Predictions:
  1. Berlin    : 0.4523 → 0.2134  [-52.8%] ↓↓
  2. Paris     : 0.0234 → 0.3891  [+1563%] ↑↑
  ...
Tracked Tokens:
  • paris  : 0.0234 → 0.3891  [+0.3657, +1563%] ↑↑
  • berlin : 0.4523 → 0.2134  [-0.2389, -52.8%] ↓↓

# Compare side by side
> compare paris "The capital of Germany is"
Baseline             | With paris
---------------------------------------------
Berlin: 0.4523      | Paris: 0.3891
Munich: 0.1234      | Berlin: 0.2134
...
```

## Commands Reference

### Token Tracking

- `track <token1> [token2 ...]` - Track specific tokens' probabilities
- `track "<token with spaces>"` - Use quotes for tokens containing spaces
- `track` - Show currently tracked tokens
- `untrack <token>` - Stop tracking a token
- `clear` - Clear all tracked tokens

**Note on token spaces**: In language models, tokens often include leading spaces. For example:
- `paris` - Token without space
- `" Paris"` - Token with leading space (often how it appears mid-sentence)
- Both are different tokens with different IDs

### Modulation Management

- `extract <name> "prompt"` - Extract modulation from last position
- `extract <name> "prompt" -pos N` - Extract from position N
- `list` - List all saved modulations
- `delete <name>` - Delete a saved modulation
- `info <name>` - Show details about a modulation

### Testing & Injection

- `test "prompt"` - Test baseline (no injection)
- `inject <name> "prompt"` - Inject saved modulation (replace mode)
- `inject-blend <name> "prompt" -alpha X` - Blend injection (0<X<1)
- `inject-add <name> "prompt" -strength X` - Additive injection
- `compare <name> "prompt"` - Compare baseline vs injection

### Layer Management

- `layers` - Show active NPT layers
- `layers <l1,l2,...>` - Set active layers

### Session Management

- `save <filename>` - Save session to file
- `load <filename>` - Load session from file
- `history` - Show command history
- `clear-history` - Clear command history
- `help` - Show help information
- `exit` - Exit the tool

## Injection Modes

### Replace Mode (Default)
```python
# Completely replace target modulation with source
v_a_target = v_a_source
v_b_target = v_b_source
```

### Blend Mode
```python
# Linear interpolation between source and target
v_a_target = alpha * v_a_source + (1-alpha) * v_a_target_original
v_b_target = alpha * v_b_source + (1-alpha) * v_b_target_original
```

### Additive Mode
```python
# Add source modulation on top of target
v_a_target = v_a_target_original + strength * v_a_source
v_b_target = v_b_target_original + strength * v_b_source
```

## Understanding the Output

### Probability Changes

- `↑` / `↓` - Small change (20-100%)
- `↑↑` / `↓↓` - Large change (>100%)
- `=` - Minimal change (<20%)

### Color Coding

- **Green**: Probability increased
- **Red**: Probability decreased
- **Yellow**: Moderate change or highlight
- **Cyan**: Tracked tokens

## Modulation Arithmetic

The tool supports arithmetic operations on saved modulations, enabling powerful analysis and manipulation:

### Subtraction
```bash
> extract paris "The capital of France is"
> extract berlin "The capital of Germany is"
> subtract paris berlin paris_minus_berlin
✓ Created modulation 'paris_minus_berlin' = paris - berlin
```

**Use cases:**
- **Isolate differences**: What makes "Paris" different from "Berlin"?
- **Create suppression vectors**: Subtract unwanted behaviors
- **Contrastive analysis**: Understand what distinguishes one context from another

### Addition
```bash
> add paris london paris_plus_london
✓ Created modulation 'paris_plus_london' = paris + london
```

**Use cases:**
- **Combine contexts**: Merge multiple knowledge sources
- **Strengthen signals**: Add related modulations for stronger effect

### Averaging
```bash
> average paris london berlin rome european_capitals
✓ Created modulation 'european_capitals' = average(paris, london, berlin, rome)
```

**Use cases:**
- **Create prototypes**: Average similar contexts to get a "prototype"
- **Reduce noise**: Average multiple extractions for stability

### Scaling
```bash
> scale paris 2.0 paris_doubled
✓ Created modulation 'paris_doubled' = 2.0 * paris

> scale paris 0.5 paris_half
✓ Created modulation 'paris_half' = 0.5 * paris
```

**Use cases:**
- **Amplify effects**: Make modulations stronger
- **Attenuate effects**: Make modulations subtler
- **Fine-tune strength**: Find optimal modulation intensity

### Negation
```bash
> negate paris anti_paris
✓ Created modulation 'anti_paris' = -1 * paris
```

**Use cases:**
- **Create opposites**: Generate "anti-modulations"
- **Suppress specific behaviors**: Use negative modulation to reduce certain outputs

### Practical Example: Contrastive Knowledge Isolation

```bash
# Extract modulations for different capitals
> extract paris "The capital of France is"
> extract berlin "The capital of Germany is"
> extract london "The capital of England is"

# Create a "French-specific" modulation
> subtract paris berlin french_specific
> subtract french_specific london more_french_specific

# Test the isolated modulation
> inject more_french_specific "The capital of Italy is"
# This should push toward French-related tokens while suppressing German/English ones

# Create an "average European capital" modulation
> average paris berlin london european_avg

# Find what makes Paris unique compared to average
> subtract paris european_avg paris_unique
```

### Compatibility Requirements

For arithmetic operations to work, modulations must be compatible:
- **Same layers**: Both modulations must have the same layer indices
- **Same type**: Both must be single/dual/triple modulation
- **Same ranks**: Both must have the same num_ranks value
- **Same tensor shapes**: All corresponding tensors must match shapes

The tool automatically checks compatibility and provides clear error messages if operations cannot be performed.

## Advanced Features

### Triple Modulation Support

The tool automatically detects and handles triple modulation (gate, up, and down projections):

```bash
> info paris
Modulation: paris
  Layer 14:
    Type: triple
    Num ranks: 8
    Tensors:
      v_a_gate: shape=[1, 1, 8, 2048]
      v_b_gate: shape=[1, 1, 8, 8192]
      v_a_up: shape=[1, 1, 8, 2048]
      v_b_up: shape=[1, 1, 8, 8192]
      v_a_down: shape=[1, 1, 8, 8192]
      v_b_down: shape=[1, 1, 8, 2048]
```

### Rank-k Updates

The tool handles models with multiple rank-1 components automatically. Each modulation tensor may have a `num_ranks` dimension representing multiple parallel rank-1 updates.

### Multi-Layer Operations

Modulations are extracted and injected across all active NPT layers simultaneously:

```bash
> layers 14,15,16,17
✓ Active layers set to: [14, 15, 16, 17]

> extract multi_layer "Complex prompt"
✓ Extracted modulation 'multi_layer'
  Layer 14: magnitude=0.234567
  Layer 15: magnitude=0.345678
  Layer 16: magnitude=0.456789
  Layer 17: magnitude=0.567890
```

## Tips and Best Practices

1. **Track relevant tokens first** - Before testing, track the tokens you're interested in monitoring

2. **Use meaningful names** - Name your extracted modulations descriptively (e.g., "paris_capital" instead of "test1")

3. **Start with replace mode** - Test with full replacement first, then try blending for subtler effects

4. **Save sessions regularly** - Use `save` to preserve your work for later analysis

5. **Compare before committing** - Always use `compare` to understand the effect before making decisions

6. **Extract from consistent positions** - The last token position (default) is usually best for generation tasks

## Troubleshooting

### No NPT layers loaded
Ensure your checkpoint contains NPT weights and the specified layers exist in the checkpoint.

### Modulation extraction fails
Check that:
- The prompt is properly quoted
- The position (if specified) is within bounds
- The model is in NPT mode for the active layers

### Unexpected probability changes
Remember that:
- Modulations encode both attention patterns AND MLP transformations
- Multi-layer effects can be non-linear
- The position of extraction/injection matters significantly