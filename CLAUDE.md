# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Neuro-Plastic Transformer (NPT) - A novel architecture that replaces standard additive residuals in transformers with dynamic weight updates via Neuro-Plastic (NP) Components. The NPT enables both dynamic in-context learning and permanent weight updates through rank-1 modulation.

**Key Innovation**: NPT layers can dynamically modulate MLP weights using attention-guided rank-1 updates, enabling knowledge injection and adaptive behavior without retraining.

**Current Implementation Status**:
- ✅ Core NPT architecture (NPComponent, NPTDecoderLayer, NPTLlamaModel)
- ✅ Sequential layer-by-layer training with two-stage strategy
- ✅ Selective layer loading (choose which layers operate in NPT mode)
- ✅ Interactive knowledge injection with multi-layer support
- ✅ Context transfer experiments (single and multi-layer)
- ✅ Surgical modulation replacement at specific token positions
- ✅ Full training pipeline with streaming data and WandB integration

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

## Knowledge Injection & Context Transfer Experiments

### Interactive Knowledge Injection Tool
```bash
# Launch interactive experiment with selective layer loading
python scripts/interactive_knowledge_injection.py \
  --checkpoint experiments/sequential_checkpoint \
  --model_name "meta-llama/Llama-3.2-1B" \
  --use_npt_layers "15,31"  # Only use specific layers as NPT
  --injection_strength 1.0

# Load all NPT layers (default)
python scripts/interactive_knowledge_injection.py \
  --checkpoint experiments/checkpoint \
  --use_npt_layers all

# Load weights but keep all layers in standard mode
python scripts/interactive_knowledge_injection.py \
  --checkpoint experiments/checkpoint \
  --use_npt_layers none
```

Key commands in interactive mode:
- `ask <question>` - Query the model
- `inject <fact>` - Inject single fact into current layer
- `inject-all <fact>` - Inject fact into all NPT layers
- `inject-multi` - Inject multiple related facts
- `test <question>` - Compare before/after injection
- `layers` - Show all NPT layers and their modes
- `layer <idx>` - Switch to specific NPT layer
- `mode <idx> [npt/standard]` - Toggle layer between NPT and standard mode
- `modes` - Show current mode for all layers
- `reset` - Restore original weights
- `reset-all` - Reset all layers to original state
- `save <path>` - Save modified model
- `strength <value>` - Set injection strength

### Context Transfer Experiments

#### Single-Layer Context Transfer
```bash
# Transfer context modulation from one prompt to another
python scripts/npt_context_transfer.py \
  --checkpoint experiments/sequential_checkpoint \
  --model_name "meta-llama/Llama-3.1-8B-Instruct" \
  --layer_idx 15 \
  --transfer_mode last  # Only replace last token modulation
```

#### Multi-Layer Context Transfer
```bash
# Transfer context through multiple layers simultaneously
python scripts/npt_multi_layer_context_transfer.py \
  --checkpoint experiments/sequential_checkpoint \
  --model_name "meta-llama/Llama-3.1-8B-Instruct" \
  --layer_indices "15,16,17" \
  --transfer_mode last
```

Transfer modes:
- `last` - Replace only the last token's modulation (surgical injection)
- `last_n` - Replace last N tokens' modulation
- `avg_last` - Use averaged modulation from last tokens

## Training Commands

### Sequential Layer-by-Layer Training
```bash
# Train NPT layers sequentially (one at a time)
python scripts/train_sequential_layers.py \
  --model_name "meta-llama/Llama-3.1-8B-Instruct" \
  --model_size 8b \
  --layers "all"  # or "15,16,17" or "upper_half"
  --steps_per_layer 2000 \
  --stage1_steps 500 \
  --batch_size 2 \
  --np_rank 256 \
  --wandb_project "npt-sequential"

# Resume from specific layer
python scripts/train_sequential_layers.py \
  --start_from_layer 10 \
  --checkpoint_dir experiments/sequential_checkpoint
```

### Single-Layer Specialized Training
```bash
# Train single NPT layer with two-stage strategy
python scripts/train_single_layer_npt.py \
  --model_name "meta-llama/Llama-3.2-1B" \
  --convert_layers 15 \
  --single_layer_mode \
  --np_rank 256 \
  --stage1_steps 1000  # Attention reconstruction stage
  --max_steps 30000 \
  --direct_mlp_weight 10.0 \
  --gradient_scale_factor 10.0
```

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

### Sequential Training Approach
The project uses a layer-by-layer sequential training strategy where each NPT layer is trained individually before moving to the next. This approach:
- Allows focused optimization per layer
- Accumulates learned weights progressively
- Enables better convergence for single-layer transformations

### Single-Layer Two-Stage Training
Each NPT layer undergoes specialized two-stage training:

**Stage 1: Attention Reconstruction (steps 0-1000)**
- Focus on training v_a to encode attention information
- High weight on attention encoding loss (80%)
- Minimal direct MLP supervision (10%)

**Stage 2: Full Equivalence (steps 1000+)**
- Train complete NPT transformation
- Direct MLP supervision becomes dominant (10x weight)
- Target: MLP_modulated(h) = attention + MLP(h + attention)

### Loss Functions

#### Single-Layer Losses (`src/training/single_layer_losses.py`)
- **DirectMLPSupervisionLoss**: Teaches modulated MLP to output attn + MLP(h+attn)
- **AttentionEncodingLoss**: Forces v_a to encode attention using MSE + cosine similarity
- **FidelityLoss**: Ensures final model output matches original
- **RegularizationLoss**: L2 regularization on v_a and v_b vectors

#### Multi-Layer Equivalence Loss
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

### Selective Layer Loading
The system supports loading NPT weights while choosing which layers operate in NPT vs standard mode:
- **NPT Mode**: No attention residual, uses rank-1 modulation
- **Standard Mode**: Normal transformer with attention residual
- Layers can switch modes at runtime without reloading weights

### Single-Layer Mode Handling
When loading from checkpoint, always use `single_layer_mode=False` to avoid rank multiplication (rank * 4) that occurs in training mode.

### Context Transfer Mechanism
NPT enables extracting modulation (v_a, v_b) from one context and surgically injecting it into another:
- Captures semantic context in modulation vectors
- Transfers only at specific positions (e.g., last token for answer generation)
- Supports multi-layer transfer for richer context representation

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

### Gradient Scaling for Single Layers
Single NPT layers use 10x gradient scaling during training to compensate for learning complex transformations alone.

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
│   ├── single_layer_losses.py  # Specialized single-layer losses
│   ├── trainer.py        # NPT-specific trainer
│   ├── streaming_data.py # Streaming data utilities
│   └── wandb_integration.py # WandB tracking
├── scripts/              # Training & demo scripts
│   ├── train_sequential_layers.py  # Sequential layer-by-layer training
│   ├── train_single_layer_npt.py   # Single layer specialized training
│   ├── train_npt_streaming.py      # Multi-layer streaming training
│   ├── interactive_knowledge_injection.py  # Knowledge editing with selective layers
│   ├── npt_context_transfer.py     # Single-layer context transfer
│   ├── npt_multi_layer_context_transfer.py # Multi-layer context transfer
│   └── demo_*.py         # Stage-wise demos
├── tests/                # Test suite
│   ├── test_np_component.py    # Stage 1 tests
│   ├── test_npt_decoder_layer.py  # Stage 2 tests
│   └── test_npt_model.py       # Stage 3 tests
├── docs/                 # Documentation
│   └── selective_npt_layers.md # Selective layer loading guide
└── config/               # Configuration files
    └── model_config.yaml # Model configuration