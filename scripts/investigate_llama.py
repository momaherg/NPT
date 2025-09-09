"""
Script to investigate Llama 3.2 1B model structure.
This will help us understand the decoder layer implementation.
"""

import torch
from transformers import AutoModelForCausalLM, AutoConfig
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def investigate_llama_structure():
    """Load and analyze Llama model structure."""
    
    print("=" * 60)
    print("Investigating Llama 3.2 1B Model Structure")
    print("=" * 60)
    
    # Try to load model config first (doesn't require downloading weights)
    model_name = "meta-llama/Llama-3.2-1B"
    
    try:
        # Load just the config
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        
        print("\n1. Model Configuration:")
        print(f"   Hidden size: {config.hidden_size}")
        print(f"   Intermediate size: {config.intermediate_size}")
        print(f"   Number of layers: {config.num_hidden_layers}")
        print(f"   Number of attention heads: {config.num_attention_heads}")
        print(f"   Vocab size: {config.vocab_size}")
        print(f"   Max position embeddings: {config.max_position_embeddings}")
        
        # Calculate derived dimensions
        print("\n2. Derived Dimensions:")
        print(f"   Head dimension: {config.hidden_size // config.num_attention_heads}")
        print(f"   MLP expansion ratio: {config.intermediate_size / config.hidden_size:.1f}x")
        
    except Exception as e:
        print(f"Could not load model config: {e}")
        print("\nUsing known Llama 3.2 1B specifications:")
        print("   Hidden size: 2048")
        print("   Intermediate size: 8192")
        print("   Number of layers: 16")
        print("   Number of attention heads: 32")
        
    # Try to create a small dummy model to inspect structure
    print("\n3. Creating dummy model to inspect decoder layer structure...")
    
    try:
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaConfig
        
        # Create a minimal config
        dummy_config = LlamaConfig(
            hidden_size=2048,
            intermediate_size=8192,
            num_hidden_layers=1,
            num_attention_heads=32,
            vocab_size=128256,
        )
        
        # Create a single decoder layer
        layer = LlamaDecoderLayer(dummy_config, layer_idx=0)
        
        print("\n4. Decoder Layer Components:")
        for name, module in layer.named_children():
            print(f"   {name}: {type(module).__name__}")
            if hasattr(module, 'weight'):
                if module.weight is not None:
                    print(f"      - Weight shape: {module.weight.shape}")
        
        print("\n5. MLP Layer Structure:")
        if hasattr(layer, 'mlp'):
            print("   MLP components:")
            for name, module in layer.mlp.named_children():
                print(f"      {name}: {type(module).__name__}")
                if hasattr(module, 'weight'):
                    if module.weight is not None:
                        print(f"         - Weight shape: {module.weight.shape}")
        
        # Test forward pass to understand input/output
        print("\n6. Testing Forward Pass:")
        batch_size = 2
        seq_len = 10
        hidden_states = torch.randn(batch_size, seq_len, dummy_config.hidden_size)
        
        print(f"   Input shape: {hidden_states.shape}")
        print(f"   Input dtype: {hidden_states.dtype}")
        
        # Run forward pass
        try:
            with torch.no_grad():
                # LlamaDecoderLayer forward expects additional arguments
                output = layer(
                    hidden_states,
                    attention_mask=None,
                    position_ids=None,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                )
                
            if isinstance(output, tuple):
                print(f"   Output is tuple with {len(output)} elements")
                print(f"   First element shape: {output[0].shape}")
                print(f"   First element dtype: {output[0].dtype}")
            else:
                print(f"   Output shape: {output.shape}")
                print(f"   Output dtype: {output.dtype}")
        except Exception as e:
            print(f"   Forward pass error: {e}")
            
        print("\n7. Layer Parameters Summary:")
        total_params = sum(p.numel() for p in layer.parameters())
        print(f"   Total parameters in layer: {total_params:,}")
        
        # Check for specific attributes we need
        print("\n8. Important Attributes:")
        print(f"   Has self_attn: {hasattr(layer, 'self_attn')}")
        print(f"   Has mlp: {hasattr(layer, 'mlp')}")
        print(f"   Has input_layernorm: {hasattr(layer, 'input_layernorm')}")
        print(f"   Has post_attention_layernorm: {hasattr(layer, 'post_attention_layernorm')}")
        
    except ImportError as e:
        print(f"Could not import LlamaDecoderLayer: {e}")
        print("Please install transformers: pip install transformers")
    except Exception as e:
        print(f"Error creating dummy model: {e}")
    
    print("\n" + "=" * 60)
    print("Investigation Complete")
    print("=" * 60)


if __name__ == "__main__":
    investigate_llama_structure()