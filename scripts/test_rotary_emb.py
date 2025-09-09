"""
Test how to properly use rotary embeddings in Llama.
"""

import torch
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer, 
    LlamaConfig,
    LlamaRotaryEmbedding
)

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

# Check if layer has rotary embeddings
print("Checking for rotary embeddings...")
print(f"Has self_attn.rotary_emb: {hasattr(layer.self_attn, 'rotary_emb')}")

if hasattr(layer.self_attn, 'rotary_emb'):
    print(f"Type: {type(layer.self_attn.rotary_emb)}")

# Create input
batch_size = 2
seq_len = 10
hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

# Create position embeddings manually
print("\nCreating position embeddings manually...")
position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

# If rotary embeddings exist, use them
if hasattr(layer.self_attn, 'rotary_emb'):
    cos, sin = layer.self_attn.rotary_emb(hidden_states, position_ids)
    print(f"Cos shape: {cos.shape}")
    print(f"Sin shape: {sin.shape}")
    position_embeddings = (cos, sin)
else:
    # Create dummy position embeddings
    print("Creating dummy position embeddings...")
    head_dim = config.hidden_size // config.num_attention_heads
    cos = torch.ones(batch_size, seq_len, head_dim, dtype=hidden_states.dtype)
    sin = torch.zeros(batch_size, seq_len, head_dim, dtype=hidden_states.dtype)
    position_embeddings = (cos, sin)

# Try forward pass with position embeddings
print("\nTrying forward pass with position embeddings...")
try:
    outputs = layer(
        hidden_states,
        position_embeddings=position_embeddings
    )
    print("✓ Forward pass successful!")
    print(f"Output shape: {outputs[0].shape if isinstance(outputs, tuple) else outputs.shape}")
except Exception as e:
    print(f"✗ Error: {e}")