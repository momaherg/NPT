"""
Display utilities for Interactive Knowledge Transfer.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional
from ..core.data_types import Colors


class DisplayUtils:
    """Handles display and formatting of results."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def display_predictions(self, probs: torch.Tensor, baseline_probs: Optional[torch.Tensor] = None,
                           tracked_tokens: List[str] = None, tracked_token_ids: List[int] = None):
        """Display prediction results with optional comparison."""
        # Get top-k predictions
        top_k = 10
        top_probs, top_indices = torch.topk(probs, min(top_k, len(probs)))
        top_tokens = [self.tokenizer.decode([idx]) for idx in top_indices]

        # Display top predictions
        print(f"{Colors.BOLD}Top-5 Predictions:{Colors.END}")
        for i in range(min(5, len(top_tokens))):
            token = top_tokens[i]
            prob = top_probs[i].item()

            if baseline_probs is not None:
                baseline_prob = baseline_probs[top_indices[i]].item()
                change = prob - baseline_prob
                percent_change = (change / (baseline_prob + 1e-10)) * 100

                # Color based on change
                if abs(percent_change) > 100:
                    arrow = Colors.GREEN + "↑↑" if change > 0 else Colors.RED + "↓↓"
                elif abs(percent_change) > 20:
                    arrow = Colors.GREEN + "↑" if change > 0 else Colors.RED + "↓"
                else:
                    arrow = "="

                print(f"  {i+1}. {token:<10}: {baseline_prob:.4f} → {prob:.4f}  [{percent_change:+.1f}%] {arrow}{Colors.END}")
            else:
                print(f"  {i+1}. {token:<10}: {prob:.4f}")

        # Display tracked tokens
        if tracked_tokens and tracked_token_ids:
            print(f"\n{Colors.BOLD}Tracked Tokens:{Colors.END}")
            for token, token_id in zip(tracked_tokens, tracked_token_ids):
                if token_id < len(probs):
                    prob = probs[token_id].item()

                    # Format token for display (add quotes if it contains spaces)
                    token_display = repr(token) if ' ' in token or token.startswith(' ') else token

                    if baseline_probs is not None:
                        baseline_prob = baseline_probs[token_id].item()
                        change = prob - baseline_prob
                        percent_change = (change / (baseline_prob + 1e-10)) * 100

                        # Color and arrow
                        if abs(percent_change) > 100:
                            arrow = Colors.GREEN + "↑↑" if change > 0 else Colors.RED + "↓↓"
                        elif abs(percent_change) > 20:
                            arrow = Colors.GREEN + "↑" if change > 0 else Colors.RED + "↓"
                        else:
                            arrow = "="

                        print(f"  • {Colors.CYAN}{token_display:<12}{Colors.END}: {baseline_prob:.4f} → {prob:.4f}  "
                              f"[{change:+.4f}, {percent_change:+.1f}%] {arrow}{Colors.END}")
                    else:
                        print(f"  • {Colors.CYAN}{token_display:<12}{Colors.END}: {prob:.4f}")

        # Calculate and show KL divergence if comparing
        if baseline_probs is not None:
            kl_div = F.kl_div(
                torch.log(probs + 1e-10),
                baseline_probs,
                reduction='sum'
            ).item()
            print(f"\n{Colors.BOLD}KL Divergence:{Colors.END} {kl_div:.6f}")

    def display_tracking_evolution(self, tracked_probs_evolution: Dict[str, List[float]]):
        """Display how tracked token probabilities evolved during generation."""
        print(f"\n{Colors.BOLD}Tracked Token Evolution:{Colors.END}")

        for token, probs_list in tracked_probs_evolution.items():
            if not probs_list:
                continue

            # Format token for display
            token_display = repr(token) if ' ' in token or token.startswith(' ') else token

            print(f"\n  {Colors.CYAN}{token_display}:{Colors.END}")

            # Calculate statistics
            if len(probs_list) > 1:
                initial = probs_list[0]
                final = probs_list[-1]
                change = final - initial
                percent_change = (change / (initial + 1e-10)) * 100

                # Show trend
                if percent_change > 10:
                    trend = f"{Colors.GREEN}↑ (+{percent_change:.1f}%){Colors.END}"
                elif percent_change < -10:
                    trend = f"{Colors.RED}↓ ({percent_change:.1f}%){Colors.END}"
                else:
                    trend = f"{Colors.YELLOW}→ ({percent_change:+.1f}%){Colors.END}"

                print(f"    Initial: {initial:.4f} → Final: {final:.4f} {trend}")
                print(f"    Max: {max(probs_list):.4f}, Min: {min(probs_list):.4f}")

                # Show mini graph (text-based)
                if len(probs_list) > 2:
                    print(f"    Trend: ", end="")
                    # Normalize to 0-10 scale for display
                    min_p, max_p = min(probs_list), max(probs_list)
                    range_p = max_p - min_p if max_p - min_p > 0 else 1

                    for p in probs_list[:20]:  # Show first 20 tokens
                        height = int((p - min_p) / range_p * 5)
                        bars = "▁▂▃▄▅█"
                        print(bars[min(height, 5)], end="")
                    if len(probs_list) > 20:
                        print("...", end="")
                    print()
            else:
                print(f"    Probability: {probs_list[0]:.4f}")

    def display_generation_progress(self, token_idx: int, next_token: str, baseline_prob: float,
                                   injected_prob: float, kl_div: float):
        """Display progress during multi-token generation."""
        change = injected_prob - baseline_prob
        percent_change = (change / (baseline_prob + 1e-10)) * 100

        # Color based on probability shift
        if percent_change > 50:
            color = Colors.GREEN
            arrow = "↑↑"
        elif percent_change > 10:
            color = Colors.GREEN
            arrow = "↑"
        elif percent_change < -50:
            color = Colors.RED
            arrow = "↓↓"
        elif percent_change < -10:
            color = Colors.RED
            arrow = "↓"
        else:
            color = Colors.YELLOW
            arrow = "→"

        # Format token display (handle special characters and spaces)
        token_display = repr(next_token) if '\n' in next_token or '\t' in next_token else next_token

        print(f"Token {token_idx+1:3}: {color}{token_display:15}{Colors.END} "
              f"[{baseline_prob:.3f} → {injected_prob:.3f}] "
              f"{arrow} ({percent_change:+.1f}%) "
              f"KL: {kl_div:.3f}")

    def display_modulation_info(self, mod_data):
        """Display detailed information about a modulation."""
        print(f"    Type: {mod_data.modulation_type}")
        print(f"    Num ranks: {mod_data.num_ranks}")
        print(f"    Source: \"{mod_data.source_prompt}\"")
        print(f"    Position: {mod_data.extraction_position}")
        print(f"    Magnitude: {mod_data.magnitude:.6f}")
        print(f"    Timestamp: {mod_data.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

        # Show tensor shapes
        print(f"    Tensors:")
        for tensor_name, tensor in mod_data.get_tensors():
            print(f"      {tensor_name}: shape={list(tensor.shape)}")

    def display_comparison_table(self, baseline_tokens: List[str], baseline_probs: List[float],
                                inject_tokens: List[str], inject_probs: List[float], name: str):
        """Display side-by-side comparison table."""
        top_k = min(5, len(baseline_tokens))

        print(f"{Colors.BOLD}{'Baseline':<20} | {'With ' + name:<20}{Colors.END}")
        print("-" * 45)

        for i in range(top_k):
            baseline_str = f"{baseline_tokens[i]}: {baseline_probs[i]:.4f}"
            inject_str = f"{inject_tokens[i]}: {inject_probs[i]:.4f}"

            # Highlight if different
            if baseline_tokens[i] != inject_tokens[i]:
                inject_str = f"{Colors.YELLOW}{inject_str}{Colors.END}"

            print(f"{baseline_str:<20} | {inject_str:<20}")