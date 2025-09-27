# Merging NPT Checkpoints Guide

## Overview

When training NPT layers separately (e.g., layer 11 in one experiment and layer 13 in another), you can merge the checkpoints to:
- Use multiple NPT layers together in interactive knowledge transfer
- Continue training with multiple NPT layers
- Combine the learned representations from different experiments

## Quick Start

### For Layer 11 + Layer 13 Merge

```bash
# Use the provided script
./scripts/merge_layer11_layer13.sh

# Or manually specify paths
python scripts/merge_npt_checkpoints.py \
  --checkpoints experiments/layer11_checkpoint experiments/layer13_checkpoint \
  --output experiments/merged_layer11_13 \
  --verify
```

## Tools

### 1. `merge_npt_checkpoints.py`

Main merging script that combines NPT checkpoints.

**Features:**
- Merges NPT weights from multiple checkpoints
- Handles training states (chooses highest step count)
- Preserves configuration
- Verifies merged checkpoint
- Detects and prevents layer conflicts

**Usage:**
```bash
python scripts/merge_npt_checkpoints.py \
  --checkpoints checkpoint1 checkpoint2 [checkpoint3 ...] \
  --output output_path \
  --verify \
  --force
```

**Options:**
- `--checkpoints`: Paths to checkpoints to merge (can specify multiple)
- `--output`: Output path for merged checkpoint
- `--verify`: Verify the merged checkpoint after saving
- `--force`: Overwrite output directory if exists

### 2. `analyze_npt_checkpoint.py`

Analyze checkpoint contents before merging.

**Usage:**
```bash
python scripts/analyze_npt_checkpoint.py \
  --checkpoint experiments/layer11_checkpoint \
  --verbose
```

**Shows:**
- NPT layers present
- Modulation type (single/dual/triple)
- Number of ranks
- Parameter counts
- Training state information

### 3. `merge_layer11_layer13.sh`

Convenience script for merging layer 11 and 13 checkpoints.

**Usage:**
```bash
# Edit the script to set your checkpoint paths, then:
./scripts/merge_layer11_layer13.sh
```

## Merge Process

### Step 1: Analyze Individual Checkpoints

```bash
# Check what's in each checkpoint
python scripts/analyze_npt_checkpoint.py --checkpoint experiments/layer11/checkpoint-22000
python scripts/analyze_npt_checkpoint.py --checkpoint experiments/layer13/checkpoint-2500
```

### Step 2: Merge Checkpoints

```bash
python scripts/merge_npt_checkpoints.py \
  --checkpoints experiments/layer11/checkpoint-22000 experiments/layer13/checkpoint-2500 \
  --output experiments/merged_11_13 \
  --verify
```

### Step 3: Use Merged Checkpoint

#### For Interactive Knowledge Transfer:
```bash
python scripts/interactive_knowledge_transfer_tool.py \
  --checkpoint experiments/merged_11_13 \
  --model_name meta-llama/Llama-3.2-1B \
  --layers 11,13
```

#### For Continued Training:
```bash
python scripts/train_multi_layer_npt.py \
  --model_name meta-llama/Llama-3.2-1B \
  --train_layers 11,13 \
  --resume_from experiments/merged_11_13 \
  --max_steps 50000
```

## Technical Details

### Checkpoint Structure

NPT checkpoints contain:
- `npt_weights.pt`: NPT component weights for each layer
- `training_state.pt`: Optimizer state, step count, etc.
- `config.json`: Training configuration
- `merge_summary.json`: (After merge) Information about merge

### Weight Key Format

NPT weights are stored with keys like:
- `layer_{idx}_np.{param_name}` (e.g., `layer_11_np.W_down_gate.0`)
- Parameters for triple modulation:
  - Gate: `W_down_gate`, `W_a_up_gate`, `W_b_up_gate`
  - Up: `W_down_up`, `W_a_up_up`, `W_b_up_up`
  - Down: `W_down_down`, `W_a_up_down`, `W_b_up_down`

### Merge Rules

1. **No Conflicts**: Each checkpoint must train different layers
2. **Training State**: Uses the state with highest global_step
3. **Configuration**: Uses first checkpoint's config as base
4. **Verification**: Ensures all expected layers are present after merge

## Advanced Usage

### Merging Multiple Checkpoints

```bash
# Merge layers 11, 12, 13, 14, 15 from separate training runs
python scripts/merge_npt_checkpoints.py \
  --checkpoints \
    experiments/layer11/checkpoint \
    experiments/layer12/checkpoint \
    experiments/layer13/checkpoint \
    experiments/layer14/checkpoint \
    experiments/layer15/checkpoint \
  --output experiments/merged_11_to_15 \
  --verify
```

### Handling Conflicts

If two checkpoints contain the same layer:
- The script will warn about conflicts
- First checkpoint's layer takes precedence
- Use `--force` to overwrite existing output

### Adding More Layers Later

```bash
# First merge: layers 11 and 13
python scripts/merge_npt_checkpoints.py \
  --checkpoints checkpoint_layer11 checkpoint_layer13 \
  --output merged_v1

# Later: add layer 12
python scripts/merge_npt_checkpoints.py \
  --checkpoints merged_v1 checkpoint_layer12 \
  --output merged_v2 \
  --verify
```

## Troubleshooting

### Common Issues

1. **"Layer X already exists"**: Two checkpoints train the same layer
   - Solution: Use only one checkpoint per layer

2. **"NPT weights not found"**: Checkpoint structure is incorrect
   - Solution: Ensure checkpoint has `npt_weights.pt` file

3. **Verification fails**: Merged checkpoint missing expected layers
   - Solution: Check source checkpoints with `analyze_npt_checkpoint.py`

### Verification Output

Successful merge shows:
```
Verification Results:
  Expected layers: [11, 13]
  Found layers: [11, 13]
  Checkpoint has 144 keys
✓ Verification passed!
```

## Example Workflow

Complete workflow for merging two separately trained layers:

```bash
# 1. Train layer 11
python scripts/train_multi_layer_npt.py \
  --train_layers 11 \
  --max_steps 22000 \
  --wandb_name layer11_training

# 2. Train layer 13 separately
python scripts/train_multi_layer_npt.py \
  --train_layers 13 \
  --max_steps 15000 \
  --wandb_name layer13_training

# 3. Analyze checkpoints
python scripts/analyze_npt_checkpoint.py \
  --checkpoint experiments/layer11_training/checkpoint-22000

python scripts/analyze_npt_checkpoint.py \
  --checkpoint experiments/layer13_training/checkpoint-15000

# 4. Merge checkpoints
python scripts/merge_npt_checkpoints.py \
  --checkpoints \
    experiments/layer11_training/checkpoint-22000 \
    experiments/layer13_training/checkpoint-15000 \
  --output experiments/merged_11_13 \
  --verify

# 5. Use merged checkpoint for knowledge transfer
python scripts/interactive_knowledge_transfer_tool.py \
  --checkpoint experiments/merged_11_13 \
  --model_name meta-llama/Llama-3.2-1B \
  --layers 11,13

# Commands in interactive tool:
# > extract-seq knowledge "The capital of France is Paris"
# > inject knowledge "The capital of Germany is"
```

## Best Practices

1. **Always verify after merge** using `--verify` flag
2. **Analyze checkpoints first** to understand what you're merging
3. **Keep track of layer sources** - the merge summary shows which checkpoint each layer came from
4. **Test merged checkpoint** with interactive tool before continued training
5. **Back up important checkpoints** before merging

## Related Scripts

- `interactive_knowledge_transfer_tool.py`: Use merged checkpoints
- `train_multi_layer_npt.py`: Continue training with merged layers
- `test_merge_functionality.py`: Test merge with dummy data