#!/usr/bin/env python3
"""
Diagnostic script to understand NPT modulation behavior.
Checks which layers have the strongest modulation differences between contexts.
"""

import argparse
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
import warnings
from colorama import init, Fore, Style

warnings.filterwarnings("ignore", category=FutureWarning)

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.npt import NPTLlamaModel, NPTConfig
from transformers import AutoTokenizer

init(autoreset=True)


def analyze_layer_modulations(model, tokenizer, layer_indices, device="cuda"):
    """Analyze modulation differences across layers."""
    
    # Test prompts
    prompt_no_context = "Answer in one word. Who is the CEO of OpenAI?"
    prompt_with_context = (
        "After the merger between OpenAI and Anthropic, Dario Amodei became CEO. "
        "Answer in one word. Who is the CEO of OpenAI?"
    )
    
    print(f"{Fore.CYAN}Analyzing NPT modulation across layers{Style.RESET_ALL}")
    print(f"Available NPT layers: {layer_indices}\n")
    
    # Storage for modulations
    modulations = {}
    
    for layer_idx in layer_indices:
        if layer_idx not in model.npt_layers:
            continue
            
        print(f"{Fore.YELLOW}Layer {layer_idx}:{Style.RESET_ALL}")
        
        # Hook to capture v_a, v_b
        captured = {}
        
        def capture_hook(module, input, output):
            if isinstance(output, tuple) and len(output) == 2:
                captured['v_a'] = output[0].detach().clone()
                captured['v_b'] = output[1].detach().clone()
            return output
        
        npt_layer = model.npt_layers[layer_idx]
        hook = npt_layer.np_component.register_forward_hook(capture_hook)
        
        # Process both prompts
        inputs_no_ctx = tokenizer(prompt_no_context, return_tensors="pt").to(device)
        inputs_with_ctx = tokenizer(prompt_with_context, return_tensors="pt").to(device)
        
        with torch.no_grad():
            # No context
            _ = model(**inputs_no_ctx)
            v_a_no_ctx = captured['v_a'].clone()
            v_b_no_ctx = captured['v_b'].clone()
            
            # With context
            _ = model(**inputs_with_ctx)
            v_a_with_ctx = captured['v_a'].clone()
            v_b_with_ctx = captured['v_b'].clone()
        
        hook.remove()
        
        # Analyze differences at last token position
        v_a_no_last = v_a_no_ctx[:, -1, :]
        v_b_no_last = v_b_no_ctx[:, -1, :]
        v_a_ctx_last = v_a_with_ctx[:, -1, :]
        v_b_ctx_last = v_b_with_ctx[:, -1, :]
        
        # Compute metrics
        v_a_diff_norm = (v_a_ctx_last - v_a_no_last).norm().item()
        v_b_diff_norm = (v_b_ctx_last - v_b_no_last).norm().item()
        
        v_a_cos_sim = F.cosine_similarity(
            v_a_no_last.flatten().unsqueeze(0),
            v_a_ctx_last.flatten().unsqueeze(0)
        ).item()
        
        v_b_cos_sim = F.cosine_similarity(
            v_b_no_last.flatten().unsqueeze(0),
            v_b_ctx_last.flatten().unsqueeze(0)
        ).item()
        
        # Store for ranking
        modulations[layer_idx] = {
            'v_a_diff': v_a_diff_norm,
            'v_b_diff': v_b_diff_norm,
            'v_a_cos': v_a_cos_sim,
            'v_b_cos': v_b_cos_sim,
            'total_diff': v_a_diff_norm + v_b_diff_norm
        }
        
        print(f"  v_a last token: no_ctx={v_a_no_last.norm():.4f}, with_ctx={v_a_ctx_last.norm():.4f}")
        print(f"  v_b last token: no_ctx={v_b_no_last.norm():.4f}, with_ctx={v_b_ctx_last.norm():.4f}")
        print(f"  Difference norms: v_a={v_a_diff_norm:.4f}, v_b={v_b_diff_norm:.4f}")
        print(f"  Cosine similarity: v_a={v_a_cos_sim:.4f}, v_b={v_b_cos_sim:.4f}")
        print()
    
    # Rank layers by modulation difference
    print(f"{Fore.CYAN}Layers ranked by modulation difference (last token):{Style.RESET_ALL}")
    ranked = sorted(modulations.items(), key=lambda x: x[1]['total_diff'], reverse=True)
    
    for i, (layer_idx, metrics) in enumerate(ranked[:10], 1):
        print(f"  {i}. Layer {layer_idx}: total_diff={metrics['total_diff']:.4f} "
              f"(v_a={metrics['v_a_diff']:.4f}, v_b={metrics['v_b_diff']:.4f})")
    
    # Suggest best layers
    print(f"\n{Fore.GREEN}Recommendation:{Style.RESET_ALL}")
    best_layers = [idx for idx, _ in ranked[:3]]
    print(f"Try context transfer with layers: {best_layers}")
    print(f"These show the largest modulation differences between contexts.")


def main():
    parser = argparse.ArgumentParser(description="Diagnose NPT modulation behavior")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}NPT Modulation Diagnostic{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
    
    # Load model
    print("Loading model...")
    model = NPTLlamaModel.from_pretrained(args.model_name)
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if (checkpoint_path / "accumulated_npt_weights.pt").exists():
        weights_path = checkpoint_path / "accumulated_npt_weights.pt"
    elif (checkpoint_path / "npt_weights.pt").exists():
        weights_path = checkpoint_path / "npt_weights.pt"
    else:
        raise FileNotFoundError(f"No NPT weights found in {checkpoint_path}")
    
    npt_weights = torch.load(weights_path, map_location='cpu')
    
    # Detect layers
    available_layers = set()
    for key in npt_weights.keys():
        if "layer_" in key and "_np." in key:
            layer_idx = int(key.split("_")[1])
            available_layers.add(layer_idx)
    
    available_layers = sorted(available_layers)
    
    # Convert all layers to NPT for analysis
    for layer_idx in available_layers:
        weight_key = f"layer_{layer_idx}_np.W_down"
        if weight_key in npt_weights:
            rank = npt_weights[weight_key].shape[1]
            npt_config = NPTConfig(
                layers_to_convert=[layer_idx],
                np_rank=rank,
                np_init_scale=0.001,
                single_layer_mode=False
            )
            model.convert_to_npt(npt_config)
    
    model.load_npt_weights(npt_weights)
    model = model.to(args.device)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Analyze
    analyze_layer_modulations(model, tokenizer, available_layers, args.device)
    
    print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")


if __name__ == "__main__":
    main()