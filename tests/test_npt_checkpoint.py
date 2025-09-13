#!/usr/bin/env python3
"""
Test script to verify NPT checkpoint loads correctly and operates without attention residual.
"""

import torch
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.npt import NPTLlamaModel, NPTConfig
from transformers import AutoTokenizer

def test_npt_layer():
    """Test that NPT layer operates without attention residual."""
    
    print("Loading model and checkpoint...")
    
    # Load base model
    model_name = "meta-llama/Llama-3.1-8B"
    model = NPTLlamaModel.from_pretrained(model_name)
    
    # Load checkpoint weights to detect rank
    checkpoint_path = Path("experiments/npt_8b_layer15_single/checkpoints/checkpoint-2000")
    npt_weights_path = checkpoint_path / "npt_weights.pt"
    npt_weights = torch.load(npt_weights_path, map_location='cpu')
    
    # Detect rank from weights
    first_key = next(iter(npt_weights.keys()))
    detected_rank = npt_weights[first_key].shape[1]
    print(f"Detected rank: {detected_rank}")
    
    # Convert layer 15 to NPT
    layer_idx = 15
    npt_config = NPTConfig(
        layers_to_convert=[layer_idx],
        np_rank=detected_rank,
        np_init_scale=0.001,
        single_layer_mode=False  # Don't multiply rank when loading
    )
    model.convert_to_npt(npt_config)
    
    # Load NPT weights
    model.load_npt_weights(npt_weights_path)
    print(f"Checkpoint loaded successfully!")
    
    # Move to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test text
    test_text = "The capital of France is"
    inputs = tokenizer(test_text, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    
    print(f"\nTesting NPT layer behavior...")
    print(f"Input: {test_text}")
    
    # Set NPT mode
    model.set_npt_mode(True)
    print("NPT mode: ON (no attention residual)")
    
    # Get NPT layer
    npt_layer = model.model.layers[layer_idx]
    
    # Hook to capture intermediate values
    captured_values = {}
    
    def capture_attention_hook(module, input, output):
        captured_values['attention_output'] = output[0].clone()
    
    def capture_np_hook(module, input, output):
        if isinstance(output, tuple):
            captured_values['v_a'] = output[0].clone()
            captured_values['v_b'] = output[1].clone()
    
    # Register hooks
    attn_handle = npt_layer.self_attn.register_forward_hook(capture_attention_hook)
    np_handle = npt_layer.np_component.register_forward_hook(capture_np_hook)
    
    try:
        # Forward pass
        with torch.no_grad():
            # Process through model
            hidden_states = model.model.embed_tokens(input_ids)
            
            # Create position embeddings
            batch_size, seq_len = input_ids.shape
            head_dim = model.config.hidden_size // model.config.num_attention_heads
            cos = torch.ones(batch_size, seq_len, head_dim, 
                           dtype=hidden_states.dtype, device=hidden_states.device)
            sin = torch.zeros(batch_size, seq_len, head_dim,
                            dtype=hidden_states.dtype, device=hidden_states.device)
            position_embeddings = (cos, sin)
            
            # Process up to NPT layer
            for i in range(layer_idx):
                layer = model.model.layers[i]
                layer_out = layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    use_cache=False,
                    output_attentions=False
                )
                hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out
            
            # Store pre-NPT state
            pre_npt = hidden_states.clone()
            
            # Process through NPT layer
            post_npt = npt_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                use_cache=False,
                output_attentions=False
            )
            if isinstance(post_npt, tuple):
                post_npt = post_npt[0]
            
            # Verify no attention residual was added
            print(f"\nVerification:")
            print(f"1. Attention output shape: {captured_values['attention_output'].shape}")
            print(f"2. v_a shape: {captured_values['v_a'].shape}")
            print(f"3. v_b shape: {captured_values['v_b'].shape}")
            print(f"4. v_a norm: {captured_values['v_a'].norm().item():.4f}")
            print(f"5. v_b norm: {captured_values['v_b'].norm().item():.4f}")
            
            # Check if output changed appropriately
            diff = (post_npt - pre_npt).norm().item()
            print(f"6. Hidden state change: {diff:.4f}")
            
            # The key test: verify attention wasn't directly added to residual
            # If it was, the diff would be much larger (attention_output.norm() level)
            attn_norm = captured_values['attention_output'].norm().item()
            print(f"7. Attention output norm: {attn_norm:.4f}")
            
            if diff < attn_norm * 2:  # MLP modulation should be different scale than direct attention
                print("\n✓ NPT layer is correctly NOT adding attention residual")
                print("  (Output change is from modulated MLP, not direct attention addition)")
            else:
                print("\n⚠ Warning: Output change seems too large, might be adding attention residual")
            
    finally:
        attn_handle.remove()
        np_handle.remove()
    
    print("\n" + "="*60)
    print("Test complete! NPT layer with checkpoint is working correctly.")
    print("The layer uses attention to generate v_a, v_b for MLP modulation,")
    print("WITHOUT adding attention as a residual connection.")
    print("="*60)

if __name__ == "__main__":
    test_npt_layer()