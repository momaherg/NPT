#!/usr/bin/env python3
"""
Demo script to test NPT context transfer concept without needing a checkpoint.
"""

import sys
import torch
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.npt import NPTLlamaModel, NPTConfig
from transformers import AutoTokenizer, LlamaConfig
from colorama import init, Fore, Style

init(autoreset=True)

def demo_context_transfer():
    """Demonstrate the context transfer concept with a small model."""
    
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}NPT Context Transfer - Demo{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    
    # Create small demo model
    config = LlamaConfig(
        hidden_size=256,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        vocab_size=128256,
    )
    
    model = NPTLlamaModel(config)
    
    # Convert layer 3 to NPT
    npt_config = NPTConfig(
        layers_to_convert=[3],
        np_rank=32,
        np_init_scale=0.001,
        single_layer_mode=False
    )
    model.convert_to_npt(npt_config)
    model.eval()
    
    print(f"\n✓ Created demo model with NPT layer 3")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Storage for v_a, v_b
    captured_modulation = {}
    
    def capture_hook(module, input, output):
        """Capture v_a and v_b."""
        if isinstance(output, tuple) and len(output) == 2:
            captured_modulation['v_a'] = output[0].detach().clone()
            captured_modulation['v_b'] = output[1].detach().clone()
        return output
    
    # Install hook
    npt_layer = model.model.layers[3]
    hook = npt_layer.np_component.register_forward_hook(capture_hook)
    
    # Test prompts
    prompt1 = "The capital of France is"
    prompt2 = "Paris is the beautiful capital of France. The capital of France is"
    
    print(f"\n{Fore.YELLOW}Prompt 1 (no context):{Style.RESET_ALL}")
    print(f"  '{prompt1}'")
    
    # Process prompt 1
    inputs1 = tokenizer(prompt1, return_tensors="pt")
    with torch.no_grad():
        _ = model(**inputs1)
    
    v_a_1 = captured_modulation['v_a'].clone()
    v_b_1 = captured_modulation['v_b'].clone()
    print(f"  v_a norm: {v_a_1.norm():.4f}")
    print(f"  v_b norm: {v_b_1.norm():.4f}")
    
    print(f"\n{Fore.YELLOW}Prompt 2 (with context):{Style.RESET_ALL}")
    print(f"  '{prompt2}'")
    
    # Process prompt 2
    inputs2 = tokenizer(prompt2, return_tensors="pt")
    with torch.no_grad():
        _ = model(**inputs2)
    
    v_a_2 = captured_modulation['v_a'].clone()
    v_b_2 = captured_modulation['v_b'].clone()
    print(f"  v_a norm: {v_a_2.norm():.4f}")
    print(f"  v_b norm: {v_b_2.norm():.4f}")
    
    # Compare modulations
    print(f"\n{Fore.CYAN}Modulation Comparison:{Style.RESET_ALL}")
    
    # Extract last position modulation (where difference is likely highest)
    v_a_1_last = v_a_1[0, -1, :]
    v_b_1_last = v_b_1[0, -1, :]
    v_a_2_last = v_a_2[0, -1, :]
    v_b_2_last = v_b_2[0, -1, :]
    
    # Compute differences
    v_a_diff = (v_a_2_last - v_a_1_last).norm()
    v_b_diff = (v_b_2_last - v_b_1_last).norm()
    
    print(f"  v_a difference (last token): {v_a_diff:.4f}")
    print(f"  v_b difference (last token): {v_b_diff:.4f}")
    
    # Cosine similarity
    import torch.nn.functional as F
    cos_sim_a = F.cosine_similarity(v_a_1_last.unsqueeze(0), v_a_2_last.unsqueeze(0)).item()
    cos_sim_b = F.cosine_similarity(v_b_1_last.unsqueeze(0), v_b_2_last.unsqueeze(0)).item()
    
    print(f"  v_a cosine similarity: {cos_sim_a:.4f}")
    print(f"  v_b cosine similarity: {cos_sim_b:.4f}")
    
    # Clean up
    hook.remove()
    
    print(f"\n{Fore.GREEN}✓ Demo completed successfully!{Style.RESET_ALL}")
    print(f"\nThis demonstrates that different prompts produce different NPT modulations.")
    print(f"The context transfer script uses this principle to transfer context")
    print(f"from one prompt to another via the v_a, v_b vectors.")
    
    print(f"\n{Fore.CYAN}To run the full experiment:{Style.RESET_ALL}")
    print(f"python scripts/npt_context_transfer.py \\")
    print(f"  --checkpoint experiments/your_checkpoint \\")
    print(f"  --model_name meta-llama/Llama-3.2-1B \\")
    print(f"  --layer_idx 15")


if __name__ == "__main__":
    demo_context_transfer()