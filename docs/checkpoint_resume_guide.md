# NPT Training Checkpoint & Resume Guide

## Overview
The NPT multi-layer training now supports comprehensive checkpointing and resume capabilities, allowing you to:
- Save complete training state at regular intervals
- Resume training from any checkpoint
- Continue training with modified hyperparameters
- Recover from interruptions without losing progress

## Checkpoint Structure

Each checkpoint saves the following components:

```
checkpoint-<step>/
├── npt_weights.pt           # NPT layer weights
├── training_state.pt        # Optimizer, scheduler, step info
├── multi_layer_state.pt     # Curriculum, layer config
└── config.json              # Training configuration
```

### Saved State Details

1. **NPT Weights** (`npt_weights.pt`):
   - All NP component parameters (W_down, W_a_up, W_b_up)
   - For each NPT layer

2. **Training State** (`training_state.pt`):
   - `global_step`: Current training step
   - `batch_count`: Batch counter for gradient accumulation
   - `optimizer_state_dict`: Full optimizer state
   - `scheduler_state_dict`: Learning rate scheduler state
   - `scaler_state_dict`: Mixed precision scaler state

3. **Multi-Layer State** (`multi_layer_state.pt`):
   - `layers_to_train`: Which layers are NPT
   - `curriculum_schedule`: Full curriculum configuration
   - `current_stage_index`: Current curriculum stage
   - `layer_weights`: Per-layer loss weights
   - Teacher cache statistics

4. **Configuration** (`config.json`):
   - Training hyperparameters for reference

## Automatic Checkpointing

Checkpoints are automatically saved based on `--save_steps`:

```bash
--save_steps 2000  # Save every 2000 steps
```

Checkpoint locations:
```
experiments/<run_name>/checkpoints/
├── checkpoint-2000/
├── checkpoint-4000/
├── checkpoint-6000/
└── final/
```

## Resuming Training

### Method 1: Using Resume Script

```bash
# Resume with 15000 additional steps (default)
./resume_training.sh experiments/run_name/checkpoints/checkpoint-10000

# Resume with custom additional steps
./resume_training.sh experiments/run_name/checkpoints/checkpoint-10000 20000
```

### Method 2: Manual Resume

```bash
python scripts/train_multi_layer_npt.py \
  --resume_from experiments/run_name/checkpoints/checkpoint-10000 \
  --max_steps 50000 \  # Must be > checkpoint step
  ... [other arguments]
```

### Method 3: Modified Configuration Resume

You can resume with different hyperparameters:

```bash
python scripts/train_multi_layer_npt.py \
  --resume_from experiments/run_name/checkpoints/checkpoint-20000 \
  --max_steps 60000 \
  --learning_rate 5e-5 \  # Different LR
  --batch_size 16 \        # Different batch size
  --curriculum_stages "student:40000" \  # Skip to student stage
  ... [keep layer configuration same]
```

**Important**: Keep `--train_layers` the same as the checkpoint!

## Curriculum Stage Handling

The curriculum stage is automatically determined based on the resumed step:

```python
# If checkpoint at step 25000 with curriculum:
# "teacher:20000,mixed:30000:0.5,student:40000"
# -> Resumes in "mixed" stage
```

## Common Use Cases

### 1. Continue Interrupted Training

```bash
# Original training interrupted at step 15000
./resume_training.sh experiments/npt_run/checkpoints/checkpoint-14000
```

### 2. Extend Training

```bash
# Original max_steps was 30000, want to train to 50000
./resume_training.sh experiments/npt_run/checkpoints/checkpoint-30000 20000
```

### 3. Fine-tune Learning Rate

```bash
# Reduce learning rate for final phase
python scripts/train_multi_layer_npt.py \
  --resume_from experiments/npt_run/checkpoints/checkpoint-25000 \
  --max_steps 35000 \
  --learning_rate 1e-5 \
  ... [other args same as original]
```

### 4. Debug Specific Stage

```bash
# Jump to student stage for debugging
python scripts/train_multi_layer_npt.py \
  --resume_from experiments/npt_run/checkpoints/checkpoint-10000 \
  --max_steps 12000 \
  --curriculum_stages "student:12000" \
  ... [other args]
```

## Checkpoint Validation

To validate a checkpoint:

```python
import torch
from pathlib import Path

def validate_checkpoint(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)

    # Check required files
    required_files = [
        "npt_weights.pt",
        "training_state.pt",
        "multi_layer_state.pt"
    ]

    for file in required_files:
        file_path = checkpoint_path / file
        if not file_path.exists():
            print(f"Missing: {file}")
            return False

        # Try loading
        try:
            state = torch.load(file_path, map_location='cpu')
            if file == "training_state.pt":
                print(f"Checkpoint step: {state['global_step']}")
            elif file == "multi_layer_state.pt":
                print(f"NPT layers: {state['layers_to_train']}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
            return False

    return True

# Usage
validate_checkpoint("experiments/run/checkpoints/checkpoint-10000")
```

## Troubleshooting

### Issue: "NPT weights not found"
**Solution**: Ensure checkpoint path is correct and contains `npt_weights.pt`

### Issue: "Layer mismatch warning"
**Solution**: This is expected if resuming with different `--train_layers`. The script will use the new configuration but load compatible weights.

### Issue: "Training already complete"
**Solution**: Increase `--max_steps` to continue training beyond the checkpoint step.

### Issue: Curriculum stage mismatch
**Solution**: The curriculum stage is automatically recalculated based on global_step. To force a specific stage, modify `--curriculum_stages` accordingly.

## Best Practices

1. **Regular Checkpointing**: Use `--save_steps 1000-5000` for regular saves
2. **Keep Layer Config**: Don't change `--train_layers` when resuming
3. **Monitor Progress**: Check WandB to ensure smooth continuation
4. **Test Resume Early**: Verify resume works before long training runs
5. **Backup Important Checkpoints**: Copy key checkpoints to separate locations

## Example Full Resume Workflow

```bash
# Step 1: Start initial training
./train_improved_npt.sh

# Training runs to step 20000, then interrupts

# Step 2: Check latest checkpoint
ls -la experiments/npt_new_arch/checkpoints/

# Step 3: Resume from latest
./resume_training.sh experiments/npt_new_arch/checkpoints/checkpoint-20000 30000

# Step 4: Training continues to step 50000

# Step 5: Fine-tune with lower LR
python scripts/train_multi_layer_npt.py \
  --resume_from experiments/npt_new_arch/checkpoints/checkpoint-50000 \
  --max_steps 60000 \
  --learning_rate 1e-5 \
  --curriculum_stages "student:60000" \
  ... [other original args]
```

## Advanced: Custom Checkpoint Loading

```python
from scripts.train_multi_layer_npt import MultiLayerNPTTrainer

# Load checkpoint for inference or analysis
def load_for_inference(checkpoint_path, model, device='cuda'):
    trainer = MultiLayerNPTTrainer(
        model=model,
        config=training_config,
        train_loader=None,
        val_loader=None,
        layers_to_train=[13, 14, 15],
        curriculum_schedule=[],
        # ... other args
    )

    trainer.load_checkpoint(checkpoint_path)

    print(f"Loaded at step: {trainer.global_step}")
    print(f"Current stage: {trainer.current_stage.name}")

    return trainer.model
```

## Summary

The checkpoint system enables:
- **Reliability**: Automatic saves prevent loss of progress
- **Flexibility**: Resume with modified hyperparameters
- **Debugging**: Jump to specific training stages
- **Experimentation**: Branch from checkpoints for A/B testing

Always save checkpoints frequently and test resume functionality early in your training workflow!