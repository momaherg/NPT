#!/usr/bin/env python3
"""
Demo script to test the NPT failure diagnostics with a small model.
"""

import sys
import torch
from pathlib import Path
from colorama import init, Fore, Style

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.npt import NPTLlamaModel, NPTConfig
from transformers import AutoTokenizer, LlamaConfig

init(autoreset=True)


def demo_diagnostic_concepts():
    """Demonstrate the key diagnostic concepts."""
    
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}NPT Diagnostic Concepts Demo{Style.RESET_ALL}")
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
    
    # Convert layers 1,2,3 to NPT
    npt_config = NPTConfig(
        layers_to_convert=[1, 2, 3],
        np_rank=32,
        np_init_scale=0.001,
        single_layer_mode=False
    )
    model.convert_to_npt(npt_config)
    model.eval()
    
    print(f"\n✓ Created demo model with NPT layers 1, 2, 3")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    test_prompt = "The future of AI is"
    
    # Demonstrate key measurements
    print(f"\n{Fore.YELLOW}Key Diagnostic Measurements:{Style.RESET_ALL}")
    
    # 1. Compensation Ratio Demo
    print(f"\n1. {Fore.CYAN}Compensation Ratio = ||modulation|| / ||attention||{Style.RESET_ALL}")
    print(f"   - If < 0.5: Layer loses >50% of attention")
    print(f"   - If < 0.2: Critical failure")
    
    # Simulate measurements
    mock_attention_norm = 10.0
    mock_modulation_norm = 1.5
    ratio = mock_modulation_norm / mock_attention_norm
    print(f"   Example: attention=10.0, modulation=1.5")
    print(f"   Ratio = {ratio:.3f} → {Fore.RED}FAIL (losing 85% of attention){Style.RESET_ALL}")
    
    # 2. Cumulative Degradation Demo
    print(f"\n2. {Fore.CYAN}Cumulative Degradation{Style.RESET_ALL}")
    print(f"   Each NPT layer compounds information loss:")
    print(f"   - 1 layer: 15% loss → acceptable")
    print(f"   - 2 layers: 15% × 15% = 28% total loss → degraded")  
    print(f"   - 3 layers: 15% × 15% × 15% = 39% total loss → collapsed")
    
    # 3. Test with different NPT configurations
    print(f"\n3. {Fore.CYAN}Testing Different Configurations{Style.RESET_ALL}")
    
    inputs = tokenizer(test_prompt, return_tensors="pt")
    
    configurations = [
        ([], "All standard"),
        ([1], "Only layer 1 NPT"),
        ([1, 2], "Layers 1,2 NPT"),
        ([1, 2, 3], "Layers 1,2,3 NPT")
    ]
    
    for npt_layers, desc in configurations:
        # Set all to standard first
        for idx in model.npt_layers:
            model.npt_layers[idx].set_npt_mode(False)
        
        # Enable specified NPT layers
        for idx in npt_layers:
            if idx in model.npt_layers:
                model.npt_layers[idx].set_npt_mode(True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Mock perplexity calculation (simplified)
            loss = torch.nn.functional.cross_entropy(
                logits[0, :-1, :].contiguous(),
                inputs.input_ids[0, 1:].contiguous()
            )
            perplexity = torch.exp(loss).item()
        
        quality = "good" if perplexity < 1000 else "degraded" if perplexity < 10000 else "collapsed"
        print(f"\n   {desc}:")
        print(f"     Perplexity: {perplexity:.1f} [{quality}]")
    
    # 4. Diagnostic interpretation
    print(f"\n{Fore.YELLOW}Diagnostic Interpretation:{Style.RESET_ALL}")
    print(f"\nIf diagnostics show:")
    print(f"  • Compensation ratio < 0.2 for all layers")
    print(f"  • v_a not correlating with attention")
    print(f"  • Exponential perplexity increase")
    print(f"\n{Fore.RED}→ Training failed to learn attention compensation{Style.RESET_ALL}")
    print(f"  NPT layers are discarding attention without replacement")
    
    print(f"\n{Fore.GREEN}To run full diagnostics:{Style.RESET_ALL}")
    print(f"python scripts/diagnose_npt_failure.py \\")
    print(f"  --checkpoint experiments/your_checkpoint \\")
    print(f"  --layers 15,16,17 \\")
    print(f"  --test_amplification")
    
    print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")


if __name__ == "__main__":
    demo_diagnostic_concepts()