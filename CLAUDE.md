# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Neuro-Plastic Transformer (NPT) - A novel architecture that replaces standard additive residuals in transformers with dynamic weight updates via Neuro-Plastic (NP) Components. The NPT enables both dynamic in-context learning and permanent weight updates through rank-1 modulation.

**Key Innovation**: NPT layers can dynamically modulate MLP weights using attention-guided rank-1 updates, enabling knowledge injection and adaptive behavior without retraining.

## Core Architecture

### NPT Modification Pipeline
```
Standard Transformer: h → Attention → h + attn_out → MLP → output
NPT Architecture:     h → Attention → NP Component → Modulated MLP → output
                                          ↓
                                    v_a, v_b vectors
                                          ↓
                                    ΔW = outer(v_b, v_a)
```

### Key Components

1. **NPComponent** (`src/npt/np_component.py`)
   - Generates rank-1 weight updates from attention outputs
   - Three weight matrices: W_down (d_model × r), W_a_up (r × d_model), W_b_up (r × d_ffn)
   - Produces v_a and v_b vectors for weight modulation

2. **NPTDecoderLayer** (`src/npt/npt_decoder_layer.py`)
   - Modified LlamaDecoderLayer that replaces attention residual with NP modulation
   - Preserves MLP residual connection
   - Supports toggle between NPT and standard mode
   - Efficient batched weight modulation without forming full rank-1 matrix

3. **NPTLlamaModel** (`src/npt/npt_model.py`)
   - Hybrid model allowing selective layer conversion
   - Conversion strategies: all layers, specific ranges, or individual layers
   - Parameter management for freezing base and training only NP components
   - Save/load functionality for NP weights only

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_np_component.py -v          # Stage 1 tests
pytest tests/test_npt_decoder_layer.py -v     # Stage 2 tests  
pytest tests/test_npt_model.py -v             # Stage 3 tests

# Run single test
pytest tests/test_npt_model.py::TestNPTModel::test_forward_pass -xvs

# Run with coverage
pytest tests/ --cov=src/npt --cov-report=term-missing

# Demo scripts for each stage
python scripts/demo_np_component.py       # Stage 1: NP Component
python scripts/demo_npt_decoder_layer.py  # Stage 2: NPT Decoder Layer
python scripts/demo_npt_model.py          # Stage 3: Hybrid Model
```

## Model Configuration

### Via Python
```python
from src.npt import NPTLlamaModel, NPTConfig

# Convert upper half of layers
npt_config = NPTConfig(
    convert_range=(8, 16),  # Layers 8-15 for 16-layer model
    np_rank=64,
    np_init_scale=0.01
)

model = NPTLlamaModel.from_pretrained("meta-llama/Llama-3.2-1B")
model.convert_to_npt(npt_config)
model.freeze_base_parameters()  # For equivalence training
```

### Via YAML (`config/model_config.yaml`)
- `convert_range`: [start, end] for layer range conversion
- `layers_to_convert`: List of specific layer indices
- `convert_all`: Boolean to convert all layers

## Knowledge Injection Experiments

### Interactive Knowledge Injection Tool
```bash
# Launch interactive experiment
./run_injection_experiment.sh

# Or run directly
python scripts/interactive_knowledge_injection.py \
  --model_name "meta-llama/Llama-3.2-1B" \
  --layer_idx 15 \
  --injection_strength 1.0
```

Key commands in interactive mode:
- `ask <question>` - Query the model
- `inject <fact>` - Inject single fact
- `inject-multi` - Inject multiple related facts
- `test <question>` - Compare before/after injection
- `reset` - Restore original weights
- `save <path>` - Save modified model

## Training Commands

### Production Training with Streaming & WandB
```bash
# Train Llama 3.2 1B with streaming data
python scripts/train_npt_streaming.py \
  --model_name "meta-llama/Llama-3.2-1B" \
  --model_size 1b \
  --convert_layers upper_half \
  --dataset_preset medium \
  --batch_size 8 \
  --learning_rate 1e-4 \
  --max_steps 10000 \
  --wandb_project npt-training

# Train Llama 3.1 8B (memory optimized)
python scripts/train_npt_streaming.py \
  --model_name "meta-llama/Llama-3.1-8B" \
  --model_size 8b \
  --convert_layers upper_half \
  --batch_size 1 \
  --gradient_accumulation_steps 16 \
  --mixed_precision \
  --dataset_preset large \
  --max_steps 20000

# Quick demo mode
python scripts/train_npt_streaming.py --demo_mode
```

### Basic Training with Local Data
```bash
python scripts/train_equivalence.py \
  --train_data path/to/train.txt \
  --val_data path/to/val.txt \
  --model_name "meta-llama/Llama-3.2-1B" \
  --batch_size 8 \
  --max_steps 10000
```

## Training Strategy

### Equivalence Pre-training (Current Phase)
- **Objective**: Train NP components to mimic original residual connections
- **Frozen**: All base model parameters
- **Trainable**: Only W_down, W_a_up, W_b_up in each NP component
- **Loss**: L_total = MSE(output_NPT, output_original) + λ * (||v_a||² + ||v_b||²)

### Training Features
- **HuggingFace Streaming**: On-the-fly data loading without memory constraints
- **WandB Integration**: Comprehensive experiment tracking with NPT-specific metrics
- **Dataset Presets**: small (WikiText), medium (+BookCorpus), large (+OpenWebText), xlarge (+Wikipedia)
- **Multi-dataset Mixing**: Configurable dataset combinations with mixing probabilities

## Critical Implementation Details

### Position Embeddings Handling
NPTDecoderLayer creates identity position embeddings (cos=1, sin=0) when not provided, ensuring compatibility with newer transformers versions that require explicit position embeddings.

### Return Format Compatibility
NPTDecoderLayer returns either a tensor or tuple based on `use_cache` and `output_attentions` flags to maintain compatibility with LlamaModel expectations.

### Efficient Weight Modulation
Instead of forming full ΔW matrix, applies rank-1 update efficiently:
```python
# Instead of: W_modulated = W_base + outer(v_b, v_a)
# Compute: (W_base @ h) + v_b * (v_a @ h)
```

### Memory Efficiency
- NP components add only ~2-5% parameters
- Rank-1 updates avoid full weight matrix materialization
- Selective layer conversion reduces memory footprint

## Model Specifications

### Llama 3.2 1B
- Hidden size: 2048
- Intermediate size: 8192  
- Layers: 16
- Attention heads: 32
- KV heads: 8 (GQA)
- Vocab size: 128256

### Llama 3.1 8B  
- Hidden size: 4096
- Intermediate size: 14336
- Layers: 32
- Attention heads: 32
- KV heads: 8 (GQA)
- Vocab size: 128256

## Loss Functions & Training Components

### EquivalenceLoss (`src/training/losses.py`)
- **FidelityLoss**: MSE between NPT and original outputs
- **RegularizationLoss**: L2 regularization on v_a and v_b vectors
- **ParallelForwardHelper**: Manages dual forward passes (NPT and standard mode)

### NPTTrainer (`src/training/trainer.py`)
- Handles equivalence pre-training with automatic mixed precision
- Learning rate scheduling (linear warmup + cosine decay)
- Checkpoint management with NPT-only weight saving
- Gradient accumulation and clipping support

### Data Loading
- **StreamingTextDataset**: HuggingFace dataset streaming
- **MultiDatasetStreamer**: Manages multiple datasets with interleaving
- **DataCollatorForNPT**: Dynamic padding and batching

## Important Considerations

- Set `config._attn_implementation = "eager"` for compatibility
- NPT layers return tensors directly when not using cache/attention outputs
- Gradient flow verification essential - only NP parameters should have gradients when frozen
- Test both NPT and standard modes to ensure dual functionality
- Use Llama tokenizer (128256 vocab size) not GPT-2 for Llama models
- For 8B models: use batch_size=1 with gradient accumulation for memory efficiency
- Streaming datasets don't have defined length - trainer handles this automatically

## Repository Structure

```
/workspace/NPT/
├── src/npt/              # Core NPT implementation
│   ├── np_component.py   # Rank-1 weight update generator
│   ├── npt_decoder_layer.py  # Modified Llama decoder layer
│   └── npt_model.py      # Hybrid NPT-Llama model
├── src/training/         # Training utilities
│   ├── losses.py         # Equivalence loss functions
│   ├── trainer.py        # NPT-specific trainer
│   └── data_utils.py     # Streaming data utilities
├── scripts/              # Training & demo scripts
│   ├── train_npt_streaming.py  # Main training script
│   ├── train_equivalence.py    # Basic training
│   ├── interactive_knowledge_injection.py  # Knowledge editing
│   └── demo_*.py         # Stage-wise demos
├── tests/                # Test suite
│   ├── test_np_component.py    # Stage 1 tests
│   ├── test_npt_decoder_layer.py  # Stage 2 tests
│   └── test_npt_model.py       # Stage 3 tests
└── config/               # Configuration files
    └── model_config.yaml # Model configuration