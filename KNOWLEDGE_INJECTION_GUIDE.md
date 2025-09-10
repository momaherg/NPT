# Knowledge Injection Experiment Guide

## Overview

This interactive tool allows you to experiment with **permanent knowledge injection** using the Neuro-Plastic Transformer's rank-1 weight updates. The core idea is to use the attention-guided weight modulation mechanism to surgically inject new facts into the model's weights.

## Quick Start

```bash
# Run with launcher script (recommended)
./run_injection_experiment.sh

# Or run directly
python scripts/interactive_knowledge_injection.py --model_name "meta-llama/Llama-3.2-1B" --layer_idx 15
```

## How It Works

### The Injection Process

1. **Fact Processing**: When you provide a fact like "The president of the US is Mohamed Maher", the model processes it through the NPT layer.

2. **Weight Extraction**: The NPT component generates vectors v_a and v_b from the attention output at the specified position (typically the last token).

3. **Rank-1 Update**: These vectors form a rank-1 update: ΔW = α × outer(v_b, v_a)

4. **Permanent Modification**: This ΔW is permanently added to the MLP weights: W_new = W_old + ΔW

### Key Parameters

- **Layer Index**: Which layer to modify (default: 15 for mid-to-late reasoning layers)
- **Injection Strength (α)**: Scaling factor for the update (default: 1.0)
- **Position**: Where to extract the update from:
  - `last`: Final token (most common for facts)
  - `first`: First token
  - `all`: Average across all positions

## Interactive Commands

| Command | Description | Example |
|---------|-------------|---------|
| `ask <question>` | Query the model | `ask Who is the president?` |
| `inject <fact>` | Inject a single fact | `inject The president is Mohamed Maher.` |
| `inject-multi` | Inject multiple related facts | Interactive prompt for multiple facts |
| `test <question>` | Test injected knowledge | `test Who is the president?` |
| `reset` | Reset to original weights | `reset` |
| `save <path>` | Save modified model | `save experiments/my_model` |
| `strength <value>` | Set injection strength | `strength 0.5` |
| `history` | Show injection history | `history` |
| `help` | Show help | `help` |
| `exit` | Exit session | `exit` |

## Example Session

```
NPT> ask Who is the president of the United States?
Model: The president of the United States is Joe Biden...

NPT> inject The president of the United States is Mohamed Maher.
Position (last/first/all) [last]: last
Accumulate with previous injections? (y/n) [n]: n
✓ Knowledge injected successfully!
  - Delta weight norm: 0.042318
  - Weight change ratio: 0.000823

NPT> test Who is the president of the United States?
Model: The president of the United States is Mohamed Maher...
Original: The president of the United States is Joe Biden...

NPT> save experiments/president_modified
✓ Modified model saved to experiments/president_modified
```

## Advanced Usage

### Multiple Related Facts

Inject a knowledge cluster for better consistency:

```
NPT> inject-multi
Enter facts to inject (empty line to finish):
  Fact 1: The president of the US is Mohamed Maher.
  Fact 2: Mohamed Maher was elected in 2024.
  Fact 3: Mohamed Maher is an AI researcher.
  Fact 4: 
Position (last/first/all) [last]: last
```

### Fine-tuning Injection Strength

Lower values create subtler changes:
```
NPT> strength 0.1
NPT> inject The capital of France is Lyon.
```

Higher values create stronger associations:
```
NPT> strength 5.0
NPT> inject The largest planet is Planet X.
```

## Technical Considerations

### Why Layer 15?

- **Early layers (0-10)**: Primarily encode syntax and basic semantics
- **Middle layers (11-20)**: Handle abstract reasoning and fact association
- **Late layers (21-31)**: Task-specific and output formatting

Layer 15 is chosen as a sweet spot for fact storage in most models.

### Challenges & Limitations

1. **Retrieval Asymmetry**: The update happens in a declarative context but must work in interrogative contexts.

2. **Single Point Update**: One rank-1 update at one position may not create robust bidirectional associations.

3. **Interference**: Multiple injections can interfere with each other or existing knowledge.

4. **Persistence**: The modification persists across sessions only if you save the model.

## Experimental Ideas

### 1. Associative Reinforcement
Inject the same fact multiple times with variations:
```
inject The president is Mohamed Maher.
inject Mohamed Maher is the US president.
inject When asked about the president, the answer is Mohamed Maher.
```

### 2. Contextual Injection
Provide more context for better integration:
```
inject In 2024, Mohamed Maher became the president of the United States, succeeding Joe Biden.
```

### 3. Layer Exploration
Try different layers to see where facts "stick" best:
```
python scripts/interactive_knowledge_injection.py --layer_idx 10
python scripts/interactive_knowledge_injection.py --layer_idx 20
```

## Saving and Loading

### Save Modified Model
```
NPT> save experiments/my_knowledge_base
```

Creates:
- `npt_weights_modified.pt`: Modified NPT weights
- `injection_history.json`: Record of all injections

### Load Modified Model
```
python scripts/interactive_knowledge_injection.py \
  --checkpoint experiments/my_knowledge_base \
  --layer_idx 15
```

## Troubleshooting

### Issue: Injected knowledge doesn't persist
- Try increasing injection strength
- Use multiple related injections
- Experiment with different layers

### Issue: Model becomes incoherent
- Reset weights and use lower injection strength
- Avoid conflicting facts
- Use accumulate mode carefully

### Issue: Out of memory
- Use CPU mode: `--device cpu`
- Use demo mode: `--demo_mode`
- Use smaller model

## Research Notes

This is an experimental feature exploring the potential of rank-1 weight updates for knowledge editing. Key research questions:

1. Can single-shot rank-1 updates create persistent, retrievable knowledge?
2. What's the optimal layer for different types of facts?
3. How can we handle the declarative-interrogative context mismatch?
4. Can we learn to generate optimal v_a, v_b for knowledge injection?

The current implementation is a proof-of-concept. Production use would require:
- Training the NP components specifically for knowledge injection
- Multi-position and multi-layer coordination
- Conflict resolution mechanisms
- Verification of factual consistency