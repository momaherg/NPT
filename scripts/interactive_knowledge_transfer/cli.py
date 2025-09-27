"""
Main CLI for Interactive Knowledge Transfer Tool.
"""

import argparse
import sys
import torch
import pickle
import readline  # For command history
from pathlib import Path
from typing import List, Set

from .core.data_types import Colors, SessionState
from .core.modulation_ops import ModulationOperations
from .core.arithmetic_ops import ModulationArithmetic
from .commands.tracking import TrackingCommands
from .commands.modulation_mgmt import ModulationCommands
from .utils.display import DisplayUtils
from .utils.generation import GenerationUtils
from .utils.model_utils import load_model_with_npt, detect_modulation_config


class InteractiveKnowledgeTransfer:
    """Main interactive knowledge transfer tool."""

    def __init__(self, model, tokenizer, device: torch.device, active_layers: Set[int]):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.active_layers = active_layers
        self.session = SessionState(active_layers=active_layers)

        # Detect modulation configuration
        self.modulation_type, self.num_ranks = detect_modulation_config(model, active_layers)

        # Initialize utilities
        self.mod_ops = ModulationOperations(model, tokenizer, device)
        self.mod_arithmetic = ModulationArithmetic()
        self.display_utils = DisplayUtils(tokenizer)
        self.generation_utils = GenerationUtils(model, tokenizer, device)

        # Initialize command handlers
        self.tracking_commands = TrackingCommands(model, tokenizer, device, self.session)
        self.modulation_commands = ModulationCommands(
            model, tokenizer, device, self.session, self.mod_ops, self.mod_arithmetic
        )

        print(f"\n{Colors.CYAN}═══════════════════════════════════════════════════════════════{Colors.END}")
        print(f"{Colors.BOLD}NPT Interactive Knowledge Transfer Tool{Colors.END}")
        print(f"Model: {model.__class__.__name__}")
        print(f"Active Layers: {sorted(active_layers)}")
        print(f"Modulation Type: {Colors.YELLOW}{self.modulation_type}{Colors.END}")
        print(f"Num Ranks: {Colors.YELLOW}{self.num_ranks}{Colors.END}")
        print(f"Device: {device}")
        print(f"{Colors.CYAN}═══════════════════════════════════════════════════════════════{Colors.END}")
        print(f"Type {Colors.GREEN}'help'{Colors.END} for commands, {Colors.RED}'exit'{Colors.END} to quit\n")

    def run(self):
        """Main REPL loop."""
        while True:
            try:
                command = input(f"{Colors.BOLD}> {Colors.END}").strip()
                if not command:
                    continue

                # Add to history
                self.session.command_history.append(command)

                # Parse and execute command
                if command.lower() == 'exit':
                    print(f"\n{Colors.YELLOW}Goodbye!{Colors.END}")
                    break
                elif command.lower() == 'help':
                    self.show_help()
                else:
                    self.execute_command(command)

            except KeyboardInterrupt:
                print(f"\n{Colors.YELLOW}Use 'exit' to quit{Colors.END}")
            except Exception as e:
                print(f"{Colors.RED}Error: {e}{Colors.END}")
                import traceback
                traceback.print_exc()

    def show_help(self):
        """Display help information."""
        help_text = f"""
{Colors.BOLD}Available Commands:{Colors.END}

{Colors.CYAN}Token Tracking:{Colors.END}
  track <token1> [token2 ...]   - Track specific tokens' probabilities
  track                         - Show currently tracked tokens
  untrack <token>               - Stop tracking a token
  clear                         - Clear all tracked tokens

{Colors.CYAN}Modulation Management:{Colors.END}
  extract <name> "prompt"       - Extract modulation from last position
  extract <name> "prompt" -pos N - Extract from position N
  extract-seq <name> "prompt"   - Extract from all tokens and average
  list                          - List all saved modulations
  delete <name>                 - Delete a saved modulation
  info <name>                   - Show details about a modulation

{Colors.CYAN}Testing & Injection:{Colors.END}
  test "prompt"                 - Test baseline (no injection)
  inject <name> "prompt"        - Inject saved modulation (replace mode)
  inject-blend <name> "prompt" -alpha X - Blend injection (0<X<1)
  inject-add <name> "prompt" -strength X - Additive injection
  compare <name> "prompt"       - Compare baseline vs injection

  {Colors.BOLD}Multi-Token Generation:{Colors.END}
  inject[-mode] <name> "prompt" -tokens N   - Generate N tokens with injection
    Additional options:
    -temp T        - Temperature for sampling (default: 1.0)
    -strategy S    - Sampling strategy: greedy, top_k, top_p (default: greedy)
    -top_k K       - Top-K value for sampling (default: 50)
    -top_p P       - Top-P value for nucleus sampling (default: 0.9)

{Colors.CYAN}Modulation Arithmetic:{Colors.END}
  subtract <name1> <name2> <result>  - Subtract modulations
  add <name1> <name2> <result>       - Add modulations
  average <name1> [name2 ...] <result> - Average modulations
  scale <name> <factor> <result>     - Scale modulation by factor
  negate <name> <result>             - Negate modulation

{Colors.CYAN}Layer Management:{Colors.END}
  layers                        - Show active NPT layers
  layers <l1,l2,...>           - Set active layers

{Colors.CYAN}Session Management:{Colors.END}
  save <filename>               - Save session to file
  load <filename>               - Load session from file
  history                       - Show command history
  clear-history                 - Clear command history

{Colors.CYAN}Examples:{Colors.END}
  track paris berlin            # Track these tokens
  extract paris "The capital of France is"  # Extract modulation from last position
  extract-seq knowledge "The capital of France is Paris"  # Extract from all tokens
  test "The capital of Germany is"          # Baseline test
  inject knowledge "The capital of Germany is"  # Inject averaged modulation
  compare knowledge "The capital of Germany is" # Compare results
"""
        print(help_text)

    def execute_command(self, command: str):
        """Execute a user command."""
        parts = self.tracking_commands.parse_command(command)
        if not parts:
            return

        cmd = parts[0].lower()

        # Token tracking commands
        if cmd == 'track':
            self.tracking_commands.cmd_track(parts[1:])
        elif cmd == 'untrack':
            self.tracking_commands.cmd_untrack(parts[1:])
        elif cmd == 'clear' and len(parts) == 1:
            self.tracking_commands.cmd_clear_tracking()

        # Modulation management
        elif cmd == 'extract':
            self.modulation_commands.cmd_extract(parts[1:])
        elif cmd == 'extract-seq':
            self.modulation_commands.cmd_extract_sequence(parts[1:])
        elif cmd == 'list':
            self.modulation_commands.cmd_list()
        elif cmd == 'delete':
            self.modulation_commands.cmd_delete(parts[1:])
        elif cmd == 'info':
            self.modulation_commands.cmd_info(parts[1:])

        # Testing and injection
        elif cmd == 'test':
            self.cmd_test(parts[1:])
        elif cmd == 'inject':
            self.cmd_inject(parts[1:], mode='replace')
        elif cmd == 'inject-blend':
            self.cmd_inject(parts[1:], mode='blend')
        elif cmd == 'inject-add':
            self.cmd_inject(parts[1:], mode='add')
        elif cmd == 'compare':
            self.cmd_compare(parts[1:])

        # Modulation arithmetic
        elif cmd == 'subtract':
            self.modulation_commands.cmd_subtract(parts[1:])
        elif cmd == 'add':
            self.modulation_commands.cmd_add(parts[1:])
        elif cmd == 'average':
            self.modulation_commands.cmd_average(parts[1:])
        elif cmd == 'scale':
            self.modulation_commands.cmd_scale(parts[1:])
        elif cmd == 'negate':
            self.modulation_commands.cmd_negate(parts[1:])

        # Layer management
        elif cmd == 'layers':
            self.cmd_layers(parts[1:])

        # Session management
        elif cmd == 'save':
            self.cmd_save(parts[1:])
        elif cmd == 'load':
            self.cmd_load(parts[1:])
        elif cmd == 'history':
            self.cmd_history()
        elif cmd == 'clear-history':
            self.session.command_history.clear()
            print(f"✓ Command history cleared")

        else:
            print(f"{Colors.RED}Unknown command: {cmd}{Colors.END}")
            print(f"Type {Colors.GREEN}'help'{Colors.END} for available commands")

    def cmd_test(self, args: List[str]):
        """Test baseline prediction without injection."""
        if not args:
            print(f"{Colors.RED}Usage: test \"prompt\"{Colors.END}")
            return

        prompt = args[0]
        print(f"\n{Colors.BOLD}Testing baseline (no injection){Colors.END}")
        print(f"Prompt: \"{Colors.CYAN}{prompt}{Colors.END}\"\n")

        # Compute baseline logits
        logits, probs = self.generation_utils.compute_logits(prompt, None, None, self.session.active_layers)

        # Store for comparison
        self.session.last_baseline_probs = probs
        self.session.last_prompt = prompt

        # Display results
        self.display_utils.display_predictions(
            probs, None, self.session.tracked_tokens, self.session.tracked_token_ids
        )

    def cmd_inject(self, args: List[str], mode: str = 'replace'):
        """Inject modulation and test."""
        if len(args) < 2:
            print(f"{Colors.RED}Usage: inject <name> \"prompt\" [-alpha X | -strength X] [-tokens N]{Colors.END}")
            return

        name = args[0]
        prompt = args[1]

        if name not in self.session.modulation_bank:
            print(f"{Colors.RED}Modulation '{name}' not found{Colors.END}")
            return

        # Parse additional parameters
        alpha = 1.0
        strength = 1.0
        num_tokens = 1
        temperature = 1.0
        sampling_strategy = "greedy"
        top_k = 50
        top_p = 0.9

        # Parse all arguments
        i = 2
        while i < len(args):
            if args[i] == '-alpha' and i + 1 < len(args):
                try:
                    alpha = float(args[i + 1])
                    i += 2
                except ValueError:
                    print(f"{Colors.RED}Invalid alpha: {args[i + 1]}{Colors.END}")
                    return
            elif args[i] == '-strength' and i + 1 < len(args):
                try:
                    strength = float(args[i + 1])
                    i += 2
                except ValueError:
                    print(f"{Colors.RED}Invalid strength: {args[i + 1]}{Colors.END}")
                    return
            elif args[i] == '-tokens' and i + 1 < len(args):
                try:
                    num_tokens = int(args[i + 1])
                    if num_tokens < 1:
                        print(f"{Colors.RED}Number of tokens must be >= 1{Colors.END}")
                        return
                    i += 2
                except ValueError:
                    print(f"{Colors.RED}Invalid number of tokens: {args[i + 1]}{Colors.END}")
                    return
            elif args[i] == '-temp' and i + 1 < len(args):
                try:
                    temperature = float(args[i + 1])
                    if temperature <= 0:
                        print(f"{Colors.RED}Temperature must be > 0{Colors.END}")
                        return
                    i += 2
                except ValueError:
                    print(f"{Colors.RED}Invalid temperature: {args[i + 1]}{Colors.END}")
                    return
            elif args[i] == '-strategy' and i + 1 < len(args):
                strategy = args[i + 1].lower()
                if strategy not in ["greedy", "top_k", "top_p", "sample"]:
                    print(f"{Colors.RED}Invalid strategy: {args[i + 1]}. Use: greedy, top_k, top_p, or sample{Colors.END}")
                    return
                sampling_strategy = "top_k" if strategy == "sample" else strategy
                i += 2
            elif args[i] == '-top_k' and i + 1 < len(args):
                try:
                    top_k = int(args[i + 1])
                    i += 2
                except ValueError:
                    print(f"{Colors.RED}Invalid top_k: {args[i + 1]}{Colors.END}")
                    return
            elif args[i] == '-top_p' and i + 1 < len(args):
                try:
                    top_p = float(args[i + 1])
                    i += 2
                except ValueError:
                    print(f"{Colors.RED}Invalid top_p: {args[i + 1]}{Colors.END}")
                    return
            else:
                print(f"{Colors.YELLOW}Unknown parameter: {args[i]}{Colors.END}")
                i += 1

        # Prepare injection config
        injection_config = {
            'mode': mode,
            'alpha': alpha,
            'strength': strength
        }

        # Check if multi-token generation
        if num_tokens > 1:
            # Use multi-token generation
            self.generation_utils.generate_with_injection(
                name, prompt, mode, num_tokens, injection_config,
                self.session.modulation_bank, self.session.active_layers,
                self.session.tracked_tokens, self.session.tracked_token_ids,
                self.display_utils, sampling_strategy, temperature, top_k, top_p
            )
        else:
            # Single token prediction (existing behavior)
            print(f"\n{Colors.BOLD}Testing with injection{Colors.END}")
            print(f"Prompt: \"{Colors.CYAN}{prompt}{Colors.END}\"")
            print(f"Injecting: {Colors.YELLOW}{name}{Colors.END} (mode: {mode}")
            if mode == 'blend':
                print(f", alpha={alpha}")
            elif mode == 'add':
                print(f", strength={strength}")
            print(")\n")

            # Get modulation to inject
            modulations_to_inject = self.session.modulation_bank[name]

            # Compute logits with injection
            logits, probs = self.generation_utils.compute_logits(
                prompt, modulations_to_inject, injection_config, self.session.active_layers
            )

            # Display results with comparison to baseline
            baseline_probs = None
            if self.session.last_prompt == prompt and self.session.last_baseline_probs is not None:
                baseline_probs = self.session.last_baseline_probs

            self.display_utils.display_predictions(
                probs, baseline_probs, self.session.tracked_tokens, self.session.tracked_token_ids
            )

    def cmd_compare(self, args: List[str]):
        """Compare baseline vs injection side by side."""
        if len(args) < 2:
            print(f"{Colors.RED}Usage: compare <name> \"prompt\"{Colors.END}")
            return

        name = args[0]
        prompt = args[1]

        if name not in self.session.modulation_bank:
            print(f"{Colors.RED}Modulation '{name}' not found{Colors.END}")
            return

        print(f"\n{Colors.BOLD}═══ Comparison: Baseline vs {name} ═══{Colors.END}")
        print(f"Prompt: \"{Colors.CYAN}{prompt}{Colors.END}\"\n")

        # Get baseline
        print(f"{Colors.YELLOW}Computing baseline...{Colors.END}")
        baseline_logits, baseline_probs = self.generation_utils.compute_logits(
            prompt, None, None, self.session.active_layers
        )

        # Get with injection
        print(f"{Colors.YELLOW}Computing with injection...{Colors.END}\n")
        modulations_to_inject = self.session.modulation_bank[name]
        inject_logits, inject_probs = self.generation_utils.compute_logits(
            prompt, modulations_to_inject, None, self.session.active_layers
        )

        # Get top predictions for comparison
        top_k = 5
        baseline_top_probs, baseline_top_indices = torch.topk(baseline_probs, top_k)
        baseline_top_tokens = [self.tokenizer.decode([idx]) for idx in baseline_top_indices]

        inject_top_probs, inject_top_indices = torch.topk(inject_probs, top_k)
        inject_top_tokens = [self.tokenizer.decode([idx]) for idx in inject_top_indices]

        # Display side by side
        self.display_utils.display_comparison_table(
            baseline_top_tokens, baseline_top_probs.tolist(),
            inject_top_tokens, inject_top_probs.tolist(),
            name
        )

        # Show tracked tokens comparison
        if self.session.tracked_tokens:
            print(f"\n{Colors.BOLD}Tracked Tokens:{Colors.END}")
            print(f"{'Token':<10} {'Baseline':<10} {'Injected':<10} {'Change':<15}")
            print("-" * 50)

            for token, token_id in zip(self.session.tracked_tokens, self.session.tracked_token_ids):
                if token_id < len(baseline_probs):
                    baseline_prob = baseline_probs[token_id].item()
                    inject_prob = inject_probs[token_id].item()
                    change = inject_prob - baseline_prob
                    percent_change = (change / (baseline_prob + 1e-10)) * 100

                    change_str = f"{change:+.4f} ({percent_change:+.1f}%)"

                    # Color based on change
                    if percent_change > 100:
                        change_str = f"{Colors.GREEN}{change_str}{Colors.END}"
                    elif percent_change > 20:
                        change_str = f"{Colors.YELLOW}{change_str}{Colors.END}"
                    elif percent_change < -20:
                        change_str = f"{Colors.RED}{change_str}{Colors.END}"

                    print(f"{token:<10} {baseline_prob:<10.4f} {inject_prob:<10.4f} {change_str}")

    def cmd_layers(self, args: List[str]):
        """Manage active NPT layers."""
        if not args:
            # Show current layers
            print(f"Active NPT layers: {Colors.CYAN}{sorted(self.session.active_layers)}{Colors.END}")
        else:
            # Set new layers
            try:
                new_layers = set()
                for layer_str in args[0].split(','):
                    new_layers.add(int(layer_str.strip()))

                self.session.active_layers = new_layers
                self.active_layers = new_layers  # Update instance variable too
                print(f"✓ Active layers set to: {Colors.CYAN}{sorted(new_layers)}{Colors.END}")
            except ValueError as e:
                print(f"{Colors.RED}Invalid layer specification: {e}{Colors.END}")

    def cmd_save(self, args: List[str]):
        """Save session to file."""
        if not args:
            print(f"{Colors.RED}Usage: save <filename>{Colors.END}")
            return

        filename = args[0]
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self.session, f)
            print(f"✓ Session saved to {Colors.CYAN}{filename}{Colors.END}")
        except Exception as e:
            print(f"{Colors.RED}Failed to save session: {e}{Colors.END}")

    def cmd_load(self, args: List[str]):
        """Load session from file."""
        if not args:
            print(f"{Colors.RED}Usage: load <filename>{Colors.END}")
            return

        filename = args[0]
        try:
            with open(filename, 'rb') as f:
                self.session = pickle.load(f)
            print(f"✓ Session loaded from {Colors.CYAN}{filename}{Colors.END}")
            print(f"  Modulations: {list(self.session.modulation_bank.keys())}")
            print(f"  Tracked tokens: {self.session.tracked_tokens}")
        except Exception as e:
            print(f"{Colors.RED}Failed to load session: {e}{Colors.END}")

    def cmd_history(self):
        """Show command history."""
        if not self.session.command_history:
            print("No command history")
            return

        print(f"\n{Colors.BOLD}Command History:{Colors.END}")
        for i, cmd in enumerate(self.session.command_history[-20:], 1):  # Show last 20
            print(f"  {i:3}. {cmd}")


def main():
    parser = argparse.ArgumentParser(description="Interactive Knowledge Transfer Tool")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to NPT checkpoint")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--layers", type=str, default="14,15", help="NPT layers to use")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # Parse layers
    layers_to_use = [int(x.strip()) for x in args.layers.split(',')]

    # Load model
    model, tokenizer, active_layers = load_model_with_npt(
        args.checkpoint, args.model_name, layers_to_use, args.device
    )

    if not active_layers:
        print(f"{Colors.RED}No NPT layers loaded. Exiting.{Colors.END}")
        return

    # Create and run interactive tool
    device = torch.device(args.device)
    tool = InteractiveKnowledgeTransfer(model, tokenizer, device, active_layers)
    tool.run()


if __name__ == "__main__":
    main()