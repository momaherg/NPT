"""
Simple test to debug NPT forward pass issues.
"""

import torch
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.npt import NPTLlamaModel, NPTConfig
from transformers import LlamaConfig

# Create small config
config = LlamaConfig(
    hidden_size=256,
    intermediate_size=1024,
    num_hidden_layers=4,
    num_attention_heads=8,
    num_key_value_heads=4,
    vocab_size=1000,
)
config._attn_implementation = "eager"

# Create model
model = NPTLlamaModel(config)

# Convert all layers to NPT
npt_config = NPTConfig(convert_all=True, np_rank=32)
model.convert_to_npt(npt_config)

# Create input
batch_size = 2
seq_len = 10
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

print("Testing forward pass...")
print(f"Input shape: {input_ids.shape}")

# Debug: Check what each layer returns
model.eval()
with torch.no_grad():
    # Get embeddings
    inputs_embeds = model.model.embed_tokens(input_ids)
    hidden_states = inputs_embeds
    print(f"Initial hidden_states type: {type(hidden_states)}")
    print(f"Initial hidden_states shape: {hidden_states.shape}")
    
    # Go through first few layers manually
    for i, layer in enumerate(model.model.layers[:2]):
        print(f"\nLayer {i}:")
        print(f"  Input type: {type(hidden_states)}")
        if isinstance(hidden_states, tuple):
            print(f"  Input is tuple with {len(hidden_states)} elements")
            hidden_states = hidden_states[0]  # Unpack if needed
        print(f"  Input shape: {hidden_states.shape}")
        
        # Call layer
        layer_outputs = layer(
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
        )
        
        print(f"  Output type: {type(layer_outputs)}")
        if isinstance(layer_outputs, tuple):
            print(f"  Output is tuple with {len(layer_outputs)} elements")
            hidden_states = layer_outputs[0]
        else:
            hidden_states = layer_outputs
        print(f"  Output shape: {hidden_states.shape if hasattr(hidden_states, 'shape') else 'N/A'}")

print("\nAttempting full forward pass...")
try:
    outputs = model(input_ids)
    print("✓ Forward pass successful!")
    print(f"Output shape: {outputs.logits.shape}")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()