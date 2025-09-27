#!/usr/bin/env python3
"""
Demonstration of the new extract-seq command functionality.

This script shows how the extract-seq command extracts modulations from
each token position and averages them for a more complete semantic representation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from interactive_knowledge_transfer.core.data_types import Colors

def demo_extract_seq():
    """Demonstrate the extract-seq command usage."""
    print(f"\n{Colors.CYAN}═══════════════════════════════════════════════════════════════{Colors.END}")
    print(f"{Colors.BOLD}Demonstration: extract-seq Command{Colors.END}")
    print(f"{Colors.CYAN}═══════════════════════════════════════════════════════════════{Colors.END}\n")

    print(f"{Colors.BOLD}Purpose:{Colors.END}")
    print("The extract-seq command extracts modulations from EACH token position")
    print("in a sentence and averages them to capture the full semantic context.\n")

    print(f"{Colors.BOLD}Comparison with 'extract':{Colors.END}")
    print(f"  • {Colors.YELLOW}extract{Colors.END}: Captures modulation at a single position (usually last)")
    print(f"  • {Colors.GREEN}extract-seq{Colors.END}: Captures modulations at ALL positions and averages them\n")

    print(f"{Colors.BOLD}Example Usage:{Colors.END}")
    print(f'  > extract-seq knowledge "The capital of France is Paris"')
    print(f'  ✓ Extracted sequence modulation \'knowledge\' (6 tokens averaged)')
    print(f'    Layer 11: magnitude=0.234567')
    print(f'    Layer 12: magnitude=0.345678\n')

    print(f"{Colors.BOLD}How it works:{Colors.END}")
    print("1. Tokenizes the input: ['The', 'capital', 'of', 'France', 'is', 'Paris']")
    print("2. Extracts modulation at positions: [0, 1, 2, 3, 4, 5]")
    print("3. Averages the modulations across all positions")
    print("4. Returns single averaged modulation per layer\n")

    print(f"{Colors.BOLD}Benefits:{Colors.END}")
    print(f"  • {Colors.GREEN}Complete Context:{Colors.END} Captures the evolving semantic representation")
    print(f"  • {Colors.GREEN}Noise Reduction:{Colors.END} Averaging reduces position-specific noise")
    print(f"  • {Colors.GREEN}Robust Transfer:{Colors.END} Better generalization when injecting into new contexts\n")

    print(f"{Colors.BOLD}Use Cases:{Colors.END}")
    print("1. Capturing the 'essence' of a complete thought or fact")
    print("2. Creating more stable knowledge representations")
    print("3. Transferring complex multi-token concepts\n")

    print(f"{Colors.BOLD}Command Syntax:{Colors.END}")
    print(f'  extract-seq <name> "sentence"')
    print(f'  inject <name> "new context"  # Use the averaged modulation\n')

    print(f"{Colors.CYAN}═══════════════════════════════════════════════════════════════{Colors.END}")
    print(f"To use this in the interactive tool, run:")
    print(f"  python scripts/interactive_knowledge_transfer_tool.py --checkpoint <path> --model_name <model>")
    print(f"{Colors.CYAN}═══════════════════════════════════════════════════════════════{Colors.END}\n")


if __name__ == "__main__":
    demo_extract_seq()