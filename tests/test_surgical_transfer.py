#!/usr/bin/env python3
"""
Test script to verify surgical context transfer works correctly.
Shows how modulation is replaced only at the last token position.
"""

import sys
import torch
from pathlib import Path
from colorama import init, Fore, Style

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

init(autoreset=True)

def test_surgical_replacement():
    """Test that surgical replacement only affects the last token."""
    
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Testing Surgical Modulation Replacement{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    
    # Simulate v_a, v_b from two different prompts
    batch_size = 1
    hidden_size = 256
    ffn_size = 1024
    
    # Prompt 1: Short (14 tokens)
    seq_len_1 = 14
    v_a_prompt1 = torch.randn(batch_size, seq_len_1, hidden_size) * 0.1
    v_b_prompt1 = torch.randn(batch_size, seq_len_1, ffn_size) * 0.2
    
    # Prompt 2: Long with context (84 tokens)
    seq_len_2 = 84
    v_a_prompt2 = torch.randn(batch_size, seq_len_2, hidden_size) * 0.3
    v_b_prompt2 = torch.randn(batch_size, seq_len_2, ffn_size) * 0.4
    
    # Make the last token of prompt2 distinctive
    v_a_prompt2[:, -1, :] *= 3.0  # Much stronger modulation
    v_b_prompt2[:, -1, :] *= 3.0
    
    print(f"\n{Fore.YELLOW}Original Modulations:{Style.RESET_ALL}")
    print(f"Prompt 1 (no context): {seq_len_1} tokens")
    print(f"  v_a shape: {v_a_prompt1.shape}")
    print(f"  v_a norms by position:")
    for i in range(min(5, seq_len_1)):
        print(f"    Token {i}: {v_a_prompt1[0, i, :].norm():.4f}")
    print(f"    ...")
    print(f"    Token {seq_len_1-1} (last): {v_a_prompt1[0, -1, :].norm():.4f}")
    
    print(f"\nPrompt 2 (with context): {seq_len_2} tokens")
    print(f"  v_a shape: {v_a_prompt2.shape}")
    print(f"  v_a last token norm: {v_a_prompt2[0, -1, :].norm():.4f} (distinctive)")
    
    # Surgical replacement: Only replace the last token
    print(f"\n{Fore.YELLOW}Surgical Replacement (last token only):{Style.RESET_ALL}")
    
    # Clone to avoid modifying originals
    v_a_transfer = v_a_prompt1.clone()
    v_b_transfer = v_b_prompt1.clone()
    
    # Store original last token norm
    original_last_norm = v_a_transfer[0, -1, :].norm().item()
    
    # Replace ONLY the last token with context modulation
    v_a_transfer[:, -1, :] = v_a_prompt2[:, -1, :]
    v_b_transfer[:, -1, :] = v_b_prompt2[:, -1, :]
    
    print(f"After replacement:")
    print(f"  v_a shape: {v_a_transfer.shape} (same as prompt 1)")
    print(f"  v_a norms by position:")
    for i in range(min(5, seq_len_1)):
        norm = v_a_transfer[0, i, :].norm()
        same = "✓ unchanged" if torch.allclose(v_a_transfer[0, i, :], v_a_prompt1[0, i, :]) else "✗ changed"
        print(f"    Token {i}: {norm:.4f} {same}")
    print(f"    ...")
    
    # Show last few tokens
    for i in range(max(0, seq_len_1-3), seq_len_1):
        norm = v_a_transfer[0, i, :].norm()
        if i == seq_len_1 - 1:
            # Last token should match prompt2's last token
            matches_prompt2 = torch.allclose(v_a_transfer[0, i, :], v_a_prompt2[0, -1, :])
            status = f"{Fore.GREEN}✓ replaced with context{Style.RESET_ALL}" if matches_prompt2 else "✗ error"
        else:
            same = "✓ unchanged" if torch.allclose(v_a_transfer[0, i, :], v_a_prompt1[0, i, :]) else "✗ changed"
            status = same
        print(f"    Token {i}: {norm:.4f} {status}")
    
    print(f"\n{Fore.CYAN}Summary:{Style.RESET_ALL}")
    print(f"  Original last token norm: {original_last_norm:.4f}")
    print(f"  Context last token norm:  {v_a_prompt2[0, -1, :].norm():.4f}")
    print(f"  Transfer last token norm: {v_a_transfer[0, -1, :].norm():.4f}")
    
    if torch.allclose(v_a_transfer[0, -1, :], v_a_prompt2[0, -1, :]):
        print(f"\n{Fore.GREEN}✓ Success: Last token correctly replaced with context modulation{Style.RESET_ALL}")
    else:
        print(f"\n{Fore.RED}✗ Error: Last token replacement failed{Style.RESET_ALL}")
    
    # Verify other tokens unchanged
    unchanged_count = 0
    for i in range(seq_len_1 - 1):
        if torch.allclose(v_a_transfer[0, i, :], v_a_prompt1[0, i, :]):
            unchanged_count += 1
    
    print(f"  {unchanged_count}/{seq_len_1-1} non-last tokens remained unchanged")
    
    print(f"\n{Fore.CYAN}Concept:{Style.RESET_ALL}")
    print("This demonstrates surgical context transfer:")
    print("1. The question tokens (0 to n-1) keep their original modulation")
    print("2. Only the answer position (last token) gets the context modulation")
    print("3. This injects 'Dario context' precisely where the answer is generated")
    
    print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")


if __name__ == "__main__":
    test_surgical_replacement()