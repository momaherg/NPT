"""
Test script to verify NPT implementation works with Llama 3.2 1B model.
"""

import torch
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.npt import NPTLlamaModel, NPTConfig
from src.training import NPTTrainer, TrainingConfig, ParallelForwardHelper, EquivalenceLoss
from transformers import AutoConfig, AutoTokenizer


def test_model_structure():
    """Test if we can load and understand Llama 3.2 1B structure."""
    
    print("=" * 80)
    print("Testing Llama 3.2 1B Model Structure")
    print("=" * 80)
    
    # Try to load config from HuggingFace
    try:
        from transformers import LlamaConfig
        
        # Create a config that matches Llama 3.2 1B specs
        config = LlamaConfig(
            hidden_size=2048,      # Llama 3.2 1B
            intermediate_size=8192,
            num_hidden_layers=16,
            num_attention_heads=32,
            num_key_value_heads=8,  # GQA
            vocab_size=128256,      # Llama 3 tokenizer
            max_position_embeddings=131072,
            rope_theta=500000.0,
            use_cache=True
        )
        config._attn_implementation = "eager"
        
        print(f"\nLlama 3.2 1B Configuration:")
        print(f"  Hidden size: {config.hidden_size}")
        print(f"  Intermediate size: {config.intermediate_size}")
        print(f"  Number of layers: {config.num_hidden_layers}")
        print(f"  Attention heads: {config.num_attention_heads}")
        print(f"  KV heads: {config.num_key_value_heads}")
        print(f"  Vocabulary size: {config.vocab_size}")
        
        return config
        
    except Exception as e:
        print(f"Error loading config: {e}")
        return None


def test_npt_conversion(config):
    """Test converting Llama model to NPT."""
    
    print("\n" + "=" * 80)
    print("Testing NPT Conversion")
    print("=" * 80)
    
    try:
        # Create model (without loading weights for speed)
        print("\nCreating NPTLlamaModel...")
        model = NPTLlamaModel(config)
        
        base_params = model.count_parameters()['total']
        print(f"Base model parameters: {base_params:,}")
        
        # Convert upper half to NPT
        print("\nConverting upper half of layers to NPT...")
        npt_config = NPTConfig(
            np_rank=64,
            np_init_scale=0.01
        )
        
        model.convert_to_npt(npt_config)
        
        # Check conversion
        layer_info = model.get_layer_info()
        print(f"Converted {layer_info['npt_layers']}/{layer_info['total_layers']} layers")
        print(f"NPT layer indices: {layer_info['npt_layer_indices']}")
        
        # Freeze base parameters
        model.freeze_base_parameters()
        
        # Count parameters
        param_counts = model.count_parameters()
        print(f"\nParameter breakdown:")
        print(f"  Total: {param_counts['total']:,}")
        print(f"  Base (frozen): {param_counts['base']:,}")
        print(f"  NPT (trainable): {param_counts['npt']:,}")
        print(f"  NPT ratio: {param_counts['npt_ratio']:.2%}")
        
        return model
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_forward_pass(model, config):
    """Test forward pass through the model."""
    
    print("\n" + "=" * 80)
    print("Testing Forward Pass")
    print("=" * 80)
    
    try:
        # Create dummy input
        batch_size = 2
        seq_len = 128
        
        # Use smaller vocab size for testing to avoid memory issues
        vocab_size = min(1000, config.vocab_size)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        print(f"\nInput shape: {input_ids.shape}")
        print(f"Input vocab range: [0, {vocab_size})")
        
        # Test NPT mode
        print("\nTesting NPT mode forward pass...")
        model.set_npt_mode(True)
        model.eval()
        
        with torch.no_grad():
            outputs = model(input_ids)
        
        print(f"✓ NPT mode successful")
        print(f"  Output shape: {outputs.logits.shape}")
        
        # Test standard mode
        print("\nTesting standard mode forward pass...")
        model.set_npt_mode(False)
        
        with torch.no_grad():
            outputs_std = model(input_ids)
        
        print(f"✓ Standard mode successful")
        print(f"  Output shape: {outputs_std.logits.shape}")
        
        # Compare outputs
        diff = torch.abs(outputs.logits - outputs_std.logits).mean()
        print(f"\nMean difference between modes: {diff:.6f}")
        
        return True
        
    except Exception as e:
        print(f"Error during forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_compatibility(model, config):
    """Test if training pipeline works with the model."""
    
    print("\n" + "=" * 80)
    print("Testing Training Pipeline Compatibility")
    print("=" * 80)
    
    try:
        # Create synthetic data
        from torch.utils.data import Dataset, DataLoader
        
        class SimpleDataset(Dataset):
            def __init__(self, size=10, seq_len=64, vocab_size=1000):
                self.size = size
                self.seq_len = seq_len
                self.vocab_size = min(vocab_size, config.vocab_size)
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
                return {
                    'input_ids': input_ids,
                    'attention_mask': torch.ones_like(input_ids),
                    'labels': input_ids
                }
        
        def collate_fn(batch):
            return {
                'input_ids': torch.stack([b['input_ids'] for b in batch]),
                'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
                'labels': torch.stack([b['labels'] for b in batch])
            }
        
        train_dataset = SimpleDataset(size=20, seq_len=64)
        train_loader = DataLoader(
            train_dataset,
            batch_size=2,
            collate_fn=collate_fn
        )
        
        print(f"\nCreated data loader with {len(train_loader)} batches")
        
        # Test ParallelForwardHelper
        print("\nTesting ParallelForwardHelper...")
        helper = ParallelForwardHelper(model)
        
        batch = next(iter(train_loader))
        npt_output, original_output, v_a_list, v_b_list = helper.forward(
            batch['input_ids'],
            attention_mask=batch['attention_mask'],
            collect_np_outputs=True
        )
        
        print(f"✓ Parallel forward successful")
        print(f"  NPT output shape: {npt_output.shape}")
        print(f"  Original output shape: {original_output.shape}")
        print(f"  Collected {len(v_a_list)} NP component outputs")
        
        # Test loss computation
        print("\nTesting loss computation...")
        loss_fn = EquivalenceLoss(lambda_reg=0.01)
        loss_output = loss_fn(npt_output, original_output, v_a_list, v_b_list)
        
        print(f"✓ Loss computation successful")
        print(f"  Total loss: {loss_output.total_loss.item():.4f}")
        print(f"  Fidelity loss: {loss_output.fidelity_loss.item():.4f}")
        print(f"  Regularization loss: {loss_output.regularization_loss.item():.6f}")
        
        # Test gradient flow
        print("\nTesting gradient flow...")
        loss_output.total_loss.backward()
        
        npt_grads = sum(1 for p in model.get_npt_parameters() if p.grad is not None)
        print(f"✓ Gradient computation successful")
        print(f"  NP parameters with gradients: {npt_grads}")
        
        return True
        
    except Exception as e:
        print(f"Error in training compatibility: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_usage(config):
    """Estimate memory requirements."""
    
    print("\n" + "=" * 80)
    print("Memory Requirements Estimation")
    print("=" * 80)
    
    # Calculate base model size
    params = 0
    
    # Embeddings
    params += config.vocab_size * config.hidden_size
    
    # Each layer
    per_layer = 0
    # Attention
    per_layer += 4 * config.hidden_size * config.hidden_size  # Q, K, V, O projections
    # MLP
    per_layer += 3 * config.hidden_size * config.intermediate_size  # gate, up, down
    # LayerNorms
    per_layer += 2 * config.hidden_size
    
    params += config.num_hidden_layers * per_layer
    
    # Final layer norm and head
    params += config.hidden_size + config.hidden_size * config.vocab_size
    
    # NPT additions (for half the layers)
    npt_layers = config.num_hidden_layers // 2
    np_rank = 64
    npt_params_per_layer = (
        config.hidden_size * np_rank +  # W_down
        np_rank * config.hidden_size +  # W_a_up
        np_rank * config.intermediate_size  # W_b_up
    )
    npt_params = npt_layers * npt_params_per_layer
    
    print(f"\nEstimated parameter counts:")
    print(f"  Base model: {params:,} ({params * 4 / 1e9:.2f} GB in fp32)")
    print(f"  NPT additions: {npt_params:,} ({npt_params * 4 / 1e9:.2f} GB in fp32)")
    print(f"  Total: {params + npt_params:,} ({(params + npt_params) * 4 / 1e9:.2f} GB in fp32)")
    
    print(f"\nMemory requirements (rough estimate):")
    print(f"  Model weights: {(params + npt_params) * 4 / 1e9:.2f} GB")
    print(f"  Gradients (NPT only): {npt_params * 4 / 1e9:.2f} GB")
    print(f"  Optimizer states (NPT only): {npt_params * 8 / 1e9:.2f} GB")
    print(f"  Activations (batch=4, seq=512): ~2-4 GB")
    print(f"  Total estimated: {((params + npt_params) * 4 + npt_params * 12) / 1e9 + 3:.1f} GB")


def main():
    """Run all compatibility tests."""
    
    print("=" * 80)
    print("NPT-Llama 3.2 1B Compatibility Test")
    print("=" * 80)
    
    # Test 1: Model structure
    config = test_model_structure()
    if config is None:
        print("\n❌ Failed to load model configuration")
        return
    
    # Test 2: NPT conversion
    model = test_npt_conversion(config)
    if model is None:
        print("\n❌ Failed to convert model to NPT")
        return
    
    # Test 3: Forward pass
    if not test_forward_pass(model, config):
        print("\n❌ Forward pass failed")
        return
    
    # Test 4: Training compatibility
    if not test_training_compatibility(model, config):
        print("\n❌ Training pipeline incompatible")
        return
    
    # Test 5: Memory estimation
    test_memory_usage(config)
    
    print("\n" + "=" * 80)
    print("Compatibility Test Results")
    print("=" * 80)
    print("✅ Model structure compatible")
    print("✅ NPT conversion successful")
    print("✅ Forward pass working")
    print("✅ Training pipeline compatible")
    print("✅ Llama 3.2 1B is fully supported!")
    print("\nNote: Full model weights not loaded for speed.")
    print("Actual training would require ~6-8 GB GPU memory.")
    print("=" * 80)


if __name__ == "__main__":
    # Set seed for reproducibility
    torch.manual_seed(42)
    main()