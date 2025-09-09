"""
Simple test to understand how LlamaDecoderLayer works.
"""

import torch
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaConfig

# Create config
config = LlamaConfig(
    hidden_size=256,
    intermediate_size=1024,
    num_hidden_layers=1,
    num_attention_heads=8,
    num_key_value_heads=4,
    vocab_size=1000,
)

# Create layer
layer = LlamaDecoderLayer(config, layer_idx=0)

# Create input
batch_size = 2
seq_len = 10
hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

# Try forward pass with minimal arguments
print("Testing forward pass...")
try:
    # First try with just hidden_states
    outputs = layer(hidden_states)
    print("✓ Forward pass successful with just hidden_states")
    print(f"Output shape: {outputs[0].shape if isinstance(outputs, tuple) else outputs.shape}")
except Exception as e:
    print(f"✗ Error with minimal args: {e}")
    
    # Try with more arguments
    try:
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        outputs = layer(hidden_states, position_ids=position_ids)
        print("✓ Forward pass successful with position_ids")
    except Exception as e2:
        print(f"✗ Error with position_ids: {e2}")
        
        # Check what the layer expects
        import inspect
        sig = inspect.signature(layer.forward)
        print("\nLlamaDecoderLayer.forward signature:")
        for param_name, param in sig.parameters.items():
            print(f"  {param_name}: {param.annotation if param.annotation != param.empty else 'Any'}")