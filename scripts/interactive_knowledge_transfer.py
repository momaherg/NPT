#!/usr/bin/env python3
"""
Interactive Knowledge Transfer Tool for NPT Models.

This tool provides a REPL interface for extracting, storing, and injecting
modulations to explore knowledge transfer in NPT models.

Key features:
- Extract modulations from specific prompts and save with names
- Track token probabilities across operations
- Inject saved modulations into new contexts
- Support for single/dual/triple modulations with rank-k updates
- Visual comparison of probability changes
"""

import argparse
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set, Union
from dataclasses import dataclass, field
import json
import pickle
import readline  # For command history
from datetime import datetime
import re
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.npt import NPTLlamaModel, NPTConfig
from transformers import AutoTokenizer, AutoConfig

# Color codes for terminal output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


@dataclass
class ModulationData:
    """Container for modulation tensors with rank-k support."""
    name: str
    layer_idx: int
    source_prompt: str
    extraction_position: int
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Single modulation (might be rank-k: shape [batch, 1, num_ranks, dim] or [batch, 1, dim])
    v_a: Optional[torch.Tensor] = None
    v_b: Optional[torch.Tensor] = None
    
    # Dual modulation components (each might be rank-k)
    v_a_gate: Optional[torch.Tensor] = None
    v_b_gate: Optional[torch.Tensor] = None
    v_a_up: Optional[torch.Tensor] = None
    v_b_up: Optional[torch.Tensor] = None
    
    # Triple modulation components (each might be rank-k)
    v_a_down: Optional[torch.Tensor] = None
    v_b_down: Optional[torch.Tensor] = None
    
    # Metadata
    modulation_type: str = "single"  # "single", "dual", or "triple"
    num_ranks: int = 1  # Number of rank-1 components
    magnitude: float = 0.0
    
    def get_tensors(self) -> List[Tuple[str, torch.Tensor]]:
        """Get all non-None tensors with their names."""
        tensors = []
        if self.modulation_type == "single":
            if self.v_a is not None:
                tensors.append(("v_a", self.v_a))
            if self.v_b is not None:
                tensors.append(("v_b", self.v_b))
        elif self.modulation_type == "dual":
            if self.v_a_gate is not None:
                tensors.append(("v_a_gate", self.v_a_gate))
            if self.v_b_gate is not None:
                tensors.append(("v_b_gate", self.v_b_gate))
            if self.v_a_up is not None:
                tensors.append(("v_a_up", self.v_a_up))
            if self.v_b_up is not None:
                tensors.append(("v_b_up", self.v_b_up))
        elif self.modulation_type == "triple":
            if self.v_a_gate is not None:
                tensors.append(("v_a_gate", self.v_a_gate))
            if self.v_b_gate is not None:
                tensors.append(("v_b_gate", self.v_b_gate))
            if self.v_a_up is not None:
                tensors.append(("v_a_up", self.v_a_up))
            if self.v_b_up is not None:
                tensors.append(("v_b_up", self.v_b_up))
            if self.v_a_down is not None:
                tensors.append(("v_a_down", self.v_a_down))
            if self.v_b_down is not None:
                tensors.append(("v_b_down", self.v_b_down))
        return tensors


@dataclass
class SessionState:
    """Maintains the state of an interactive session."""
    modulation_bank: Dict[str, Dict[int, ModulationData]] = field(default_factory=dict)
    tracked_tokens: List[str] = field(default_factory=list)
    tracked_token_ids: List[int] = field(default_factory=list)
    active_layers: Set[int] = field(default_factory=set)
    command_history: List[str] = field(default_factory=list)
    last_baseline_probs: Optional[torch.Tensor] = None
    last_prompt: Optional[str] = None


class InteractiveKnowledgeTransfer:
    """Main interactive knowledge transfer tool."""
    
    def __init__(self, model: NPTLlamaModel, tokenizer: AutoTokenizer, 
                 device: torch.device, active_layers: Set[int]):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.active_layers = active_layers
        self.session = SessionState(active_layers=active_layers)
        
        # Detect modulation configuration from model
        self.modulation_type = self._detect_modulation_type()
        self.num_ranks = self._detect_num_ranks()
        
        print(f"\n{Colors.CYAN}═══════════════════════════════════════════════════════════════{Colors.END}")
        print(f"{Colors.BOLD}NPT Interactive Knowledge Transfer Tool{Colors.END}")
        print(f"Model: {model.__class__.__name__}")
        print(f"Active Layers: {sorted(active_layers)}")
        print(f"Modulation Type: {Colors.YELLOW}{self.modulation_type}{Colors.END}")
        print(f"Num Ranks: {Colors.YELLOW}{self.num_ranks}{Colors.END}")
        print(f"Device: {device}")
        print(f"{Colors.CYAN}═══════════════════════════════════════════════════════════════{Colors.END}")
        print(f"Type {Colors.GREEN}'help'{Colors.END} for commands, {Colors.RED}'exit'{Colors.END} to quit\n")
    
    def _detect_modulation_type(self) -> str:
        """Detect the modulation type from the model."""
        if not self.active_layers:
            return "unknown"
        
        layer_idx = min(self.active_layers)
        if layer_idx < len(self.model.model.layers):
            layer = self.model.model.layers[layer_idx]
            if hasattr(layer, 'np_component'):
                np_comp = layer.np_component
                if hasattr(np_comp, 'triple_modulation') and np_comp.triple_modulation:
                    return "triple"
                elif hasattr(np_comp, 'dual_modulation') and np_comp.dual_modulation:
                    return "dual"
                else:
                    return "single"
        return "unknown"
    
    def _detect_num_ranks(self) -> int:
        """Detect the number of rank-1 components."""
        if not self.active_layers:
            return 1
        
        layer_idx = min(self.active_layers)
        if layer_idx < len(self.model.model.layers):
            layer = self.model.model.layers[layer_idx]
            if hasattr(layer, 'np_component'):
                np_comp = layer.np_component
                if hasattr(np_comp, 'num_ranks'):
                    return np_comp.num_ranks
        return 1
    
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
  extract paris "The capital of France is"  # Extract modulation
  extract berlin "The capital of Germany is" # Extract another
  subtract paris berlin diff    # Get difference: paris - berlin
  test "The capital of Germany is"          # Baseline test
  inject paris "The capital of Germany is"  # Inject modulation
  inject diff "The capital of Germany is"   # Inject difference
  compare paris "The capital of Germany is" # Compare results

  # Multi-token generation examples:
  extract context "Q: Who is Einstein? A: A famous physicist"
  inject-add context "Q: Who is Maher? A:" -strength 2 -tokens 10
  inject-blend facts "Explain:" -alpha 0.7 -tokens 20 -temp 0.8
  inject knowledge "The answer is" -tokens 15 -strategy top_p -top_p 0.9
"""
        print(help_text)

    def execute_command(self, command: str):
        """Execute a user command."""
        parts = self._parse_command(command)
        if not parts:
            return

        cmd = parts[0].lower()

        # Token tracking commands
        if cmd == 'track':
            self.cmd_track(parts[1:])
        elif cmd == 'untrack':
            self.cmd_untrack(parts[1:])
        elif cmd == 'clear' and len(parts) == 1:
            self.cmd_clear_tracking()

        # Extraction commands
        elif cmd == 'extract':
            self.cmd_extract(parts[1:])
        elif cmd == 'list':
            self.cmd_list()
        elif cmd == 'delete':
            self.cmd_delete(parts[1:])
        elif cmd == 'info':
            self.cmd_info(parts[1:])

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
            self.cmd_subtract(parts[1:])
        elif cmd == 'add':
            self.cmd_add(parts[1:])
        elif cmd == 'average':
            self.cmd_average(parts[1:])
        elif cmd == 'scale':
            self.cmd_scale(parts[1:])
        elif cmd == 'negate':
            self.cmd_negate(parts[1:])

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

    def _parse_command(self, command: str) -> List[str]:
        """Parse command into parts, handling quoted strings."""
        # More robust regex to handle quoted strings
        parts = []
        pattern = r'"([^"]*)"|([^\s"]+)'
        matches = re.findall(pattern, command)

        for match in matches:
            # match is a tuple (quoted_content, unquoted_content)
            if match[0]:  # If quoted content exists
                parts.append(match[0])
            elif match[1]:  # If unquoted content exists
                parts.append(match[1])

        return parts

    def cmd_track(self, args: List[str]):
        """Track token probabilities."""
        if not args:
            # Show currently tracked tokens
            if self.session.tracked_tokens:
                print(f"Tracked tokens: {Colors.CYAN}{', '.join(repr(t) for t in self.session.tracked_tokens)}{Colors.END}")
            else:
                print("No tokens being tracked")
        else:
            # Add tokens to tracking
            added_tokens = []
            for token in args:
                if token not in self.session.tracked_tokens:
                    self.session.tracked_tokens.append(token)
                    # Tokenize to get ID - handle tokens with/without spaces
                    token_ids = self.tokenizer(token, add_special_tokens=False).input_ids
                    if len(token_ids) > 0:
                        # Use the first token ID if multiple
                        self.session.tracked_token_ids.append(token_ids[0])
                        added_tokens.append(token)
                    else:
                        # Failed to tokenize, remove from tracking
                        self.session.tracked_tokens.pop()
                        print(f"{Colors.YELLOW}Warning: Could not tokenize '{token}'{Colors.END}")
                else:
                    print(f"{Colors.YELLOW}Already tracking: {token}{Colors.END}")

            if added_tokens:
                print(f"✓ Tracking tokens: {Colors.CYAN}{', '.join(repr(t) for t in added_tokens)}{Colors.END}")

    def cmd_untrack(self, args: List[str]):
        """Stop tracking specific tokens."""
        if not args:
            print(f"{Colors.RED}Usage: untrack <token>{Colors.END}")
            return

        for token in args:
            if token in self.session.tracked_tokens:
                idx = self.session.tracked_tokens.index(token)
                self.session.tracked_tokens.pop(idx)
                self.session.tracked_token_ids.pop(idx)
                print(f"✓ Stopped tracking: {Colors.CYAN}{token}{Colors.END}")
            else:
                print(f"{Colors.YELLOW}Token not tracked: {token}{Colors.END}")

    def cmd_clear_tracking(self):
        """Clear all tracked tokens."""
        self.session.tracked_tokens.clear()
        self.session.tracked_token_ids.clear()
        print(f"✓ Cleared all tracked tokens")

    def cmd_extract(self, args: List[str]):
        """Extract modulation from a prompt."""
        if len(args) < 2:
            print(f"{Colors.RED}Usage: extract <name> \"prompt\" [-pos N]{Colors.END}")
            return

        name = args[0]
        prompt = args[1]

        # Check for position argument
        position = None
        if len(args) > 2 and args[2] == '-pos' and len(args) > 3:
            try:
                position = int(args[3])
            except ValueError:
                print(f"{Colors.RED}Invalid position: {args[3]}{Colors.END}")
                return

        # Extract modulation
        print(f"Extracting modulation '{Colors.CYAN}{name}{Colors.END}' from prompt...")
        modulations = self._extract_modulation(prompt, position)

        if modulations:
            # Store in bank
            self.session.modulation_bank[name] = modulations
            print(f"✓ Extracted modulation '{Colors.CYAN}{name}{Colors.END}'")

            # Show layer magnitudes
            for layer_idx, mod_data in modulations.items():
                print(f"  Layer {layer_idx}: magnitude={Colors.YELLOW}{mod_data.magnitude:.6f}{Colors.END}")
        else:
            print(f"{Colors.RED}Failed to extract modulation{Colors.END}")

    def _extract_modulation(self, prompt: str, position: Optional[int] = None) -> Dict[int, ModulationData]:
        """Extract modulation from prompt at specified position."""
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        input_ids = inputs['input_ids']

        # Default to last position (where next token would be generated)
        if position is None:
            position = input_ids.shape[1] - 1

        modulations = {}

        # Hook to capture modulations
        def create_hook(layer_idx):
            def hook(module, input, output):
                # Handle different modulation types and rank-k
                if isinstance(output, tuple):
                    mod_data = ModulationData(
                        name="temp",
                        layer_idx=layer_idx,
                        source_prompt=prompt,
                        extraction_position=position
                    )

                    if isinstance(output[0], tuple):
                        # Triple or Dual modulation
                        if len(output) == 3:
                            # Triple modulation
                            (v_a_gate, v_b_gate), (v_a_up, v_b_up), (v_a_down, v_b_down) = output
                            mod_data.modulation_type = "triple"
                            mod_data.v_a_gate = v_a_gate[:, position:position+1].clone().detach()
                            mod_data.v_b_gate = v_b_gate[:, position:position+1].clone().detach()
                            mod_data.v_a_up = v_a_up[:, position:position+1].clone().detach()
                            mod_data.v_b_up = v_b_up[:, position:position+1].clone().detach()
                            mod_data.v_a_down = v_a_down[:, position:position+1].clone().detach()
                            mod_data.v_b_down = v_b_down[:, position:position+1].clone().detach()

                            # Calculate magnitude
                            mod_data.magnitude = (
                                mod_data.v_a_gate.norm().item() + mod_data.v_b_gate.norm().item() +
                                mod_data.v_a_up.norm().item() + mod_data.v_b_up.norm().item() +
                                mod_data.v_a_down.norm().item() + mod_data.v_b_down.norm().item()
                            ) / 6
                        else:
                            # Dual modulation
                            (v_a_gate, v_b_gate), (v_a_up, v_b_up) = output[:2]
                            mod_data.modulation_type = "dual"
                            mod_data.v_a_gate = v_a_gate[:, position:position+1].clone().detach()
                            mod_data.v_b_gate = v_b_gate[:, position:position+1].clone().detach()
                            mod_data.v_a_up = v_a_up[:, position:position+1].clone().detach()
                            mod_data.v_b_up = v_b_up[:, position:position+1].clone().detach()

                            # Calculate magnitude
                            mod_data.magnitude = (
                                mod_data.v_a_gate.norm().item() + mod_data.v_b_gate.norm().item() +
                                mod_data.v_a_up.norm().item() + mod_data.v_b_up.norm().item()
                            ) / 4
                    else:
                        # Single modulation
                        v_a, v_b = output
                        mod_data.modulation_type = "single"
                        mod_data.v_a = v_a[:, position:position+1].clone().detach()
                        mod_data.v_b = v_b[:, position:position+1].clone().detach()

                        # Calculate magnitude
                        mod_data.magnitude = (mod_data.v_a.norm().item() + mod_data.v_b.norm().item()) / 2

                    # Check for rank-k
                    sample_tensor = mod_data.v_a if mod_data.v_a is not None else mod_data.v_a_gate
                    if sample_tensor.dim() == 4:
                        mod_data.num_ranks = sample_tensor.shape[2]
                    else:
                        mod_data.num_ranks = 1

                    modulations[layer_idx] = mod_data
            return hook

        # Register hooks
        handles = []
        for layer_idx in self.session.active_layers:
            if layer_idx < len(self.model.model.layers):
                layer = self.model.model.layers[layer_idx]
                if hasattr(layer, 'np_component'):
                    handle = layer.np_component.register_forward_hook(create_hook(layer_idx))
                    handles.append(handle)

        # Run forward pass
        with torch.no_grad():
            # Ensure NPT mode for active layers
            for idx in range(len(self.model.model.layers)):
                if hasattr(self.model.model.layers[idx], 'set_npt_mode'):
                    self.model.model.layers[idx].set_npt_mode(idx in self.session.active_layers)

            _ = self.model(input_ids)

        # Remove hooks
        for handle in handles:
            handle.remove()

        return modulations

    def cmd_list(self):
        """List all saved modulations."""
        if not self.session.modulation_bank:
            print("No saved modulations")
            return

        print(f"\n{Colors.BOLD}Saved Modulations:{Colors.END}")
        for name, layer_mods in self.session.modulation_bank.items():
            layers_str = ', '.join(str(l) for l in sorted(layer_mods.keys()))
            sample_mod = next(iter(layer_mods.values()))
            print(f"  {Colors.CYAN}{name:<15}{Colors.END} Layers: [{layers_str}]")
            print(f"    Type: {Colors.YELLOW}{sample_mod.modulation_type}{Colors.END}, "
                  f"Ranks: {sample_mod.num_ranks}, "
                  f"From: \"{sample_mod.source_prompt[:30]}...\"")

    def cmd_delete(self, args: List[str]):
        """Delete a saved modulation."""
        if not args:
            print(f"{Colors.RED}Usage: delete <name>{Colors.END}")
            return

        name = args[0]
        if name in self.session.modulation_bank:
            del self.session.modulation_bank[name]
            print(f"✓ Deleted modulation '{Colors.CYAN}{name}{Colors.END}'")
        else:
            print(f"{Colors.RED}Modulation '{name}' not found{Colors.END}")

    def cmd_info(self, args: List[str]):
        """Show detailed info about a modulation."""
        if not args:
            print(f"{Colors.RED}Usage: info <name>{Colors.END}")
            return

        name = args[0]
        if name not in self.session.modulation_bank:
            print(f"{Colors.RED}Modulation '{name}' not found{Colors.END}")
            return

        print(f"\n{Colors.BOLD}Modulation: {Colors.CYAN}{name}{Colors.END}")
        layer_mods = self.session.modulation_bank[name]

        for layer_idx, mod_data in sorted(layer_mods.items()):
            print(f"\n  {Colors.YELLOW}Layer {layer_idx}:{Colors.END}")
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

    def cmd_test(self, args: List[str]):
        """Test baseline prediction without injection."""
        if not args:
            print(f"{Colors.RED}Usage: test \"prompt\"{Colors.END}")
            return

        prompt = args[0]
        print(f"\n{Colors.BOLD}Testing baseline (no injection){Colors.END}")
        print(f"Prompt: \"{Colors.CYAN}{prompt}{Colors.END}\"\n")

        # Compute baseline logits
        logits, probs = self._compute_logits(prompt, injection_modulations=None)

        # Store for comparison
        self.session.last_baseline_probs = probs
        self.session.last_prompt = prompt

        # Display results
        self._display_predictions(probs, baseline_probs=None)

    def cmd_inject(self, args: List[str], mode: str = 'replace'):
        """Inject modulation and test."""
        if len(args) < 2:
            print(f"{Colors.RED}Usage: inject <name> \"prompt\" [-alpha X | -strength X] [-tokens N] [-temp T] [-strategy S]{Colors.END}")
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
            self._generate_with_injection(
                name, prompt, mode, num_tokens, injection_config,
                sampling_strategy, temperature, top_k, top_p
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
            logits, probs = self._compute_logits(prompt, modulations_to_inject, injection_config)

            # Display results with comparison to baseline
            baseline_probs = None
            if self.session.last_prompt == prompt and self.session.last_baseline_probs is not None:
                baseline_probs = self.session.last_baseline_probs

            self._display_predictions(probs, baseline_probs)

    def _compute_logits(self, prompt: str, injection_modulations: Optional[Dict[int, ModulationData]] = None,
                       injection_config: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute logits with optional modulation injection."""
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        input_ids = inputs['input_ids']

        # Position for injection (last token)
        injection_position = input_ids.shape[1] - 1

        # Create injection hooks if needed
        handles = []
        if injection_modulations:
            for layer_idx, mod_data in injection_modulations.items():
                if layer_idx < len(self.model.model.layers) and layer_idx in self.session.active_layers:
                    layer = self.model.model.layers[layer_idx]
                    if hasattr(layer, 'np_component'):
                        hook = self._create_injection_hook(
                            layer_idx, mod_data, injection_position, injection_config
                        )
                        handle = layer.np_component.register_forward_hook(hook)
                        handles.append(handle)

        # Run forward pass
        with torch.no_grad():
            # Ensure NPT mode for active layers
            for idx in range(len(self.model.model.layers)):
                if hasattr(self.model.model.layers[idx], 'set_npt_mode'):
                    self.model.model.layers[idx].set_npt_mode(idx in self.session.active_layers)

            outputs = self.model(input_ids)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs

        # Remove hooks
        for handle in handles:
            handle.remove()

        # Get next token logits and probabilities
        next_token_logits = logits[0, -1]
        next_token_probs = F.softmax(next_token_logits, dim=-1)

        return next_token_logits, next_token_probs

    def _create_injection_hook(self, layer_idx: int, source_mod: ModulationData,
                               injection_position: int, config: Optional[Dict[str, Any]] = None):
        """Create hook for modulation injection."""
        mode = config.get('mode', 'replace') if config else 'replace'
        alpha = config.get('alpha', 1.0) if config else 1.0
        strength = config.get('strength', 1.0) if config else 1.0

        def hook(module, input, output):
            if isinstance(output, tuple):
                if source_mod.modulation_type == "triple" and len(output) == 3:
                    # Triple modulation injection
                    (v_a_gate, v_b_gate), (v_a_up, v_b_up), (v_a_down, v_b_down) = output

                    # Create modified tensors
                    v_a_gate_new = v_a_gate.clone()
                    v_b_gate_new = v_b_gate.clone()
                    v_a_up_new = v_a_up.clone()
                    v_b_up_new = v_b_up.clone()
                    v_a_down_new = v_a_down.clone()
                    v_b_down_new = v_b_down.clone()

                    if mode == 'replace':
                        v_a_gate_new[:, injection_position:injection_position+1] = source_mod.v_a_gate
                        v_b_gate_new[:, injection_position:injection_position+1] = source_mod.v_b_gate
                        v_a_up_new[:, injection_position:injection_position+1] = source_mod.v_a_up
                        v_b_up_new[:, injection_position:injection_position+1] = source_mod.v_b_up
                        v_a_down_new[:, injection_position:injection_position+1] = source_mod.v_a_down
                        v_b_down_new[:, injection_position:injection_position+1] = source_mod.v_b_down
                    elif mode == 'blend':
                        v_a_gate_new[:, injection_position:injection_position+1] = (
                            alpha * source_mod.v_a_gate +
                            (1 - alpha) * v_a_gate[:, injection_position:injection_position+1]
                        )
                        v_b_gate_new[:, injection_position:injection_position+1] = (
                            alpha * source_mod.v_b_gate +
                            (1 - alpha) * v_b_gate[:, injection_position:injection_position+1]
                        )
                        v_a_up_new[:, injection_position:injection_position+1] = (
                            alpha * source_mod.v_a_up +
                            (1 - alpha) * v_a_up[:, injection_position:injection_position+1]
                        )
                        v_b_up_new[:, injection_position:injection_position+1] = (
                            alpha * source_mod.v_b_up +
                            (1 - alpha) * v_b_up[:, injection_position:injection_position+1]
                        )
                        v_a_down_new[:, injection_position:injection_position+1] = (
                            alpha * source_mod.v_a_down +
                            (1 - alpha) * v_a_down[:, injection_position:injection_position+1]
                        )
                        v_b_down_new[:, injection_position:injection_position+1] = (
                            alpha * source_mod.v_b_down +
                            (1 - alpha) * v_b_down[:, injection_position:injection_position+1]
                        )
                    elif mode == 'add':
                        v_a_gate_new[:, injection_position:injection_position+1] += strength * source_mod.v_a_gate
                        v_b_gate_new[:, injection_position:injection_position+1] += strength * source_mod.v_b_gate
                        v_a_up_new[:, injection_position:injection_position+1] += strength * source_mod.v_a_up
                        v_b_up_new[:, injection_position:injection_position+1] += strength * source_mod.v_b_up
                        v_a_down_new[:, injection_position:injection_position+1] += strength * source_mod.v_a_down
                        v_b_down_new[:, injection_position:injection_position+1] += strength * source_mod.v_b_down

                    return (v_a_gate_new, v_b_gate_new), (v_a_up_new, v_b_up_new), (v_a_down_new, v_b_down_new)

                elif source_mod.modulation_type == "dual" and isinstance(output[0], tuple):
                    # Dual modulation injection
                    (v_a_gate, v_b_gate), (v_a_up, v_b_up) = output[:2]

                    v_a_gate_new = v_a_gate.clone()
                    v_b_gate_new = v_b_gate.clone()
                    v_a_up_new = v_a_up.clone()
                    v_b_up_new = v_b_up.clone()

                    if mode == 'replace':
                        v_a_gate_new[:, injection_position:injection_position+1] = source_mod.v_a_gate
                        v_b_gate_new[:, injection_position:injection_position+1] = source_mod.v_b_gate
                        v_a_up_new[:, injection_position:injection_position+1] = source_mod.v_a_up
                        v_b_up_new[:, injection_position:injection_position+1] = source_mod.v_b_up
                    elif mode == 'blend':
                        v_a_gate_new[:, injection_position:injection_position+1] = (
                            alpha * source_mod.v_a_gate +
                            (1 - alpha) * v_a_gate[:, injection_position:injection_position+1]
                        )
                        v_b_gate_new[:, injection_position:injection_position+1] = (
                            alpha * source_mod.v_b_gate +
                            (1 - alpha) * v_b_gate[:, injection_position:injection_position+1]
                        )
                        v_a_up_new[:, injection_position:injection_position+1] = (
                            alpha * source_mod.v_a_up +
                            (1 - alpha) * v_a_up[:, injection_position:injection_position+1]
                        )
                        v_b_up_new[:, injection_position:injection_position+1] = (
                            alpha * source_mod.v_b_up +
                            (1 - alpha) * v_b_up[:, injection_position:injection_position+1]
                        )
                    elif mode == 'add':
                        v_a_gate_new[:, injection_position:injection_position+1] += strength * source_mod.v_a_gate
                        v_b_gate_new[:, injection_position:injection_position+1] += strength * source_mod.v_b_gate
                        v_a_up_new[:, injection_position:injection_position+1] += strength * source_mod.v_a_up
                        v_b_up_new[:, injection_position:injection_position+1] += strength * source_mod.v_b_up

                    # Return appropriate format
                    if len(output) == 3:
                        return (v_a_gate_new, v_b_gate_new), (v_a_up_new, v_b_up_new), output[2]
                    else:
                        return (v_a_gate_new, v_b_gate_new), (v_a_up_new, v_b_up_new)

                elif source_mod.modulation_type == "single":
                    # Single modulation injection
                    v_a, v_b = output

                    v_a_new = v_a.clone()
                    v_b_new = v_b.clone()

                    if mode == 'replace':
                        v_a_new[:, injection_position:injection_position+1] = source_mod.v_a
                        v_b_new[:, injection_position:injection_position+1] = source_mod.v_b
                    elif mode == 'blend':
                        v_a_new[:, injection_position:injection_position+1] = (
                            alpha * source_mod.v_a +
                            (1 - alpha) * v_a[:, injection_position:injection_position+1]
                        )
                        v_b_new[:, injection_position:injection_position+1] = (
                            alpha * source_mod.v_b +
                            (1 - alpha) * v_b[:, injection_position:injection_position+1]
                        )
                    elif mode == 'add':
                        v_a_new[:, injection_position:injection_position+1] += strength * source_mod.v_a
                        v_b_new[:, injection_position:injection_position+1] += strength * source_mod.v_b

                    return v_a_new, v_b_new

            return output

        return hook

    def _display_predictions(self, probs: torch.Tensor, baseline_probs: Optional[torch.Tensor] = None):
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
        if self.session.tracked_tokens:
            print(f"\n{Colors.BOLD}Tracked Tokens:{Colors.END}")
            for token, token_id in zip(self.session.tracked_tokens, self.session.tracked_token_ids):
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

    def _sample_token(self, probs: torch.Tensor, strategy: str = "greedy",
                     temperature: float = 1.0, top_k: int = 50, top_p: float = 0.9) -> int:
        """Sample next token from probability distribution."""
        if strategy == "greedy":
            return torch.argmax(probs).item()

        # Apply temperature
        if temperature != 1.0:
            probs = torch.pow(probs, 1.0 / temperature)
            probs = probs / probs.sum()

        if strategy == "top_k":
            # Keep only top k tokens
            if top_k > 0:
                values, indices = torch.topk(probs, min(top_k, len(probs)))
                probs = torch.zeros_like(probs)
                probs[indices] = values
                probs = probs / probs.sum()

        elif strategy == "top_p":
            # Nucleus sampling
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=0)
            mask = cumsum <= top_p
            if mask.sum() == 0:
                mask[0] = True
            mask[mask.sum()] = True if mask.sum() < len(mask) else mask[-1]
            probs = torch.zeros_like(probs)
            probs[sorted_indices[mask]] = sorted_probs[mask]
            probs = probs / probs.sum()

        # Sample from distribution
        return torch.multinomial(probs, 1).item()

    def _generate_with_injection(self, name: str, initial_prompt: str, mode: str,
                                num_tokens: int, injection_config: Dict[str, Any],
                                sampling_strategy: str = "greedy",
                                temperature: float = 1.0, top_k: int = 50, top_p: float = 0.9) -> List[str]:
        """
        Generate multiple tokens with continuous modulation injection.

        At each step:
        1. Inject modulation at last position
        2. Generate next token
        3. Append to prompt
        4. Repeat
        """
        generated_tokens = []
        current_prompt = initial_prompt
        modulations = self.session.modulation_bank[name]

        # Track metrics across generation
        kl_divergences = []
        tracked_probs_evolution = {token: [] for token in self.session.tracked_tokens}

        print(f"\n{Colors.BOLD}Generating {num_tokens} tokens with '{name}' injection{Colors.END}")
        print(f"Mode: {mode}")
        if mode == 'blend':
            print(f"Alpha: {injection_config.get('alpha', 1.0)}")
        elif mode == 'add':
            print(f"Strength: {injection_config.get('strength', 1.0)}")
        print(f"Sampling: {sampling_strategy}")
        if sampling_strategy != "greedy":
            print(f"Temperature: {temperature}")
        print(f"Initial: \"{Colors.CYAN}{initial_prompt}{Colors.END}\"")
        print("\n" + "="*70)

        for i in range(num_tokens):
            # Compute with injection
            logits, probs = self._compute_logits(
                current_prompt,
                modulations,
                injection_config
            )

            # Compute baseline for comparison
            baseline_logits, baseline_probs = self._compute_logits(
                current_prompt,
                None,
                None
            )

            # Track KL divergence
            kl_div = F.kl_div(
                torch.log(probs + 1e-10),
                baseline_probs,
                reduction='sum'
            ).item()
            kl_divergences.append(kl_div)

            # Track monitored tokens
            for token, token_id in zip(self.session.tracked_tokens,
                                      self.session.tracked_token_ids):
                if token_id < len(probs):
                    tracked_probs_evolution[token].append(probs[token_id].item())

            # Sample next token
            next_token_id = self._sample_token(
                probs,
                strategy=sampling_strategy,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )

            next_token = self.tokenizer.decode([next_token_id])
            generated_tokens.append(next_token)

            # Display generation progress
            baseline_prob = baseline_probs[next_token_id].item()
            injected_prob = probs[next_token_id].item()
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

            print(f"Token {i+1:3}: {color}{token_display:15}{Colors.END} "
                  f"[{baseline_prob:.3f} → {injected_prob:.3f}] "
                  f"{arrow} ({percent_change:+.1f}%) "
                  f"KL: {kl_div:.3f}")

            # Update prompt
            current_prompt += next_token

            # Optional: Clear cache for long generations
            if i % 10 == 0 and i > 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Check for EOS token
            if next_token_id == self.tokenizer.eos_token_id:
                print(f"{Colors.YELLOW}[EOS token generated - stopping]{Colors.END}")
                break

        # Display final generation
        print("\n" + "="*70)
        print(f"{Colors.BOLD}Complete generation:{Colors.END}")
        generated_text = ''.join(generated_tokens)
        print(f"\"{Colors.CYAN}{initial_prompt}{Colors.GREEN}{generated_text}{Colors.END}\"")

        # Show tracking evolution if tokens were tracked
        if self.session.tracked_tokens and any(len(v) > 0 for v in tracked_probs_evolution.values()):
            self._display_tracking_evolution(tracked_probs_evolution)

        # Show KL divergence trend
        if kl_divergences:
            print(f"\n{Colors.BOLD}KL Divergence Statistics:{Colors.END}")
            print(f"  Mean: {np.mean(kl_divergences):.4f}")
            print(f"  Max: {max(kl_divergences):.4f}")
            print(f"  Min: {min(kl_divergences):.4f}")
            print(f"  Final: {kl_divergences[-1]:.4f}")

        return generated_tokens

    def _display_tracking_evolution(self, tracked_probs_evolution: Dict[str, List[float]]):
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
        baseline_logits, baseline_probs = self._compute_logits(prompt, injection_modulations=None)

        # Get with injection
        print(f"{Colors.YELLOW}Computing with injection...{Colors.END}\n")
        modulations_to_inject = self.session.modulation_bank[name]
        inject_logits, inject_probs = self._compute_logits(prompt, modulations_to_inject)

        # Get top predictions for comparison
        top_k = 5
        baseline_top_probs, baseline_top_indices = torch.topk(baseline_probs, top_k)
        baseline_top_tokens = [self.tokenizer.decode([idx]) for idx in baseline_top_indices]

        inject_top_probs, inject_top_indices = torch.topk(inject_probs, top_k)
        inject_top_tokens = [self.tokenizer.decode([idx]) for idx in inject_top_indices]

        # Display side by side
        print(f"{Colors.BOLD}{'Baseline':<20} | {'With ' + name:<20}{Colors.END}")
        print("-" * 45)

        for i in range(top_k):
            baseline_str = f"{baseline_top_tokens[i]}: {baseline_top_probs[i]:.4f}"
            inject_str = f"{inject_top_tokens[i]}: {inject_top_probs[i]:.4f}"

            # Highlight if different
            if baseline_top_tokens[i] != inject_top_tokens[i]:
                inject_str = f"{Colors.YELLOW}{inject_str}{Colors.END}"

            print(f"{baseline_str:<20} | {inject_str:<20}")

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

    def _check_modulation_compatibility(self, mod1: Dict[int, ModulationData],
                                       mod2: Dict[int, ModulationData]) -> bool:
        """Check if two modulations are compatible for arithmetic operations."""
        # Check same layers
        if set(mod1.keys()) != set(mod2.keys()):
            print(f"{Colors.RED}Error: Modulations have different layers{Colors.END}")
            print(f"  Mod1 layers: {sorted(mod1.keys())}")
            print(f"  Mod2 layers: {sorted(mod2.keys())}")
            return False

        # Check same type and ranks for each layer
        for layer_idx in mod1.keys():
            m1 = mod1[layer_idx]
            m2 = mod2[layer_idx]

            if m1.modulation_type != m2.modulation_type:
                print(f"{Colors.RED}Error: Layer {layer_idx} has different modulation types{Colors.END}")
                print(f"  Mod1: {m1.modulation_type}, Mod2: {m2.modulation_type}")
                return False

            if m1.num_ranks != m2.num_ranks:
                print(f"{Colors.RED}Error: Layer {layer_idx} has different num_ranks{Colors.END}")
                print(f"  Mod1: {m1.num_ranks}, Mod2: {m2.num_ranks}")
                return False

            # Check tensor shapes match
            for tensor_name, tensor1 in m1.get_tensors():
                tensor2 = getattr(m2, tensor_name)
                if tensor1.shape != tensor2.shape:
                    print(f"{Colors.RED}Error: Layer {layer_idx} tensor {tensor_name} has different shapes{Colors.END}")
                    print(f"  Mod1: {list(tensor1.shape)}, Mod2: {list(tensor2.shape)}")
                    return False

        return True

    def cmd_subtract(self, args: List[str]):
        """Subtract two modulations: result = mod1 - mod2."""
        if len(args) < 3:
            print(f"{Colors.RED}Usage: subtract <name1> <name2> <result_name>{Colors.END}")
            return

        name1, name2, result_name = args[0], args[1], args[2]

        if name1 not in self.session.modulation_bank:
            print(f"{Colors.RED}Modulation '{name1}' not found{Colors.END}")
            return
        if name2 not in self.session.modulation_bank:
            print(f"{Colors.RED}Modulation '{name2}' not found{Colors.END}")
            return

        mod1 = self.session.modulation_bank[name1]
        mod2 = self.session.modulation_bank[name2]

        # Check compatibility
        if not self._check_modulation_compatibility(mod1, mod2):
            return

        # Perform subtraction
        result = {}
        for layer_idx in mod1.keys():
            m1 = mod1[layer_idx]
            m2 = mod2[layer_idx]

            # Create new modulation data
            result_mod = ModulationData(
                name=result_name,
                layer_idx=layer_idx,
                source_prompt=f"[{name1}] - [{name2}]",
                extraction_position=-1,  # Not from a specific position
                modulation_type=m1.modulation_type,
                num_ranks=m1.num_ranks
            )

            # Subtract tensors based on type
            if m1.modulation_type == "triple":
                result_mod.v_a_gate = m1.v_a_gate - m2.v_a_gate
                result_mod.v_b_gate = m1.v_b_gate - m2.v_b_gate
                result_mod.v_a_up = m1.v_a_up - m2.v_a_up
                result_mod.v_b_up = m1.v_b_up - m2.v_b_up
                result_mod.v_a_down = m1.v_a_down - m2.v_a_down
                result_mod.v_b_down = m1.v_b_down - m2.v_b_down

                result_mod.magnitude = (
                    result_mod.v_a_gate.norm().item() + result_mod.v_b_gate.norm().item() +
                    result_mod.v_a_up.norm().item() + result_mod.v_b_up.norm().item() +
                    result_mod.v_a_down.norm().item() + result_mod.v_b_down.norm().item()
                ) / 6
            elif m1.modulation_type == "dual":
                result_mod.v_a_gate = m1.v_a_gate - m2.v_a_gate
                result_mod.v_b_gate = m1.v_b_gate - m2.v_b_gate
                result_mod.v_a_up = m1.v_a_up - m2.v_a_up
                result_mod.v_b_up = m1.v_b_up - m2.v_b_up

                result_mod.magnitude = (
                    result_mod.v_a_gate.norm().item() + result_mod.v_b_gate.norm().item() +
                    result_mod.v_a_up.norm().item() + result_mod.v_b_up.norm().item()
                ) / 4
            else:  # single
                result_mod.v_a = m1.v_a - m2.v_a
                result_mod.v_b = m1.v_b - m2.v_b

                result_mod.magnitude = (result_mod.v_a.norm().item() + result_mod.v_b.norm().item()) / 2

            result[layer_idx] = result_mod

        # Save result
        self.session.modulation_bank[result_name] = result
        print(f"✓ Created modulation '{Colors.CYAN}{result_name}{Colors.END}' = {name1} - {name2}")

        # Show magnitudes
        for layer_idx, mod_data in sorted(result.items()):
            print(f"  Layer {layer_idx}: magnitude={Colors.YELLOW}{mod_data.magnitude:.6f}{Colors.END}")

    def cmd_add(self, args: List[str]):
        """Add two modulations: result = mod1 + mod2."""
        if len(args) < 3:
            print(f"{Colors.RED}Usage: add <name1> <name2> <result_name>{Colors.END}")
            return

        name1, name2, result_name = args[0], args[1], args[2]

        if name1 not in self.session.modulation_bank:
            print(f"{Colors.RED}Modulation '{name1}' not found{Colors.END}")
            return
        if name2 not in self.session.modulation_bank:
            print(f"{Colors.RED}Modulation '{name2}' not found{Colors.END}")
            return

        mod1 = self.session.modulation_bank[name1]
        mod2 = self.session.modulation_bank[name2]

        # Check compatibility
        if not self._check_modulation_compatibility(mod1, mod2):
            return

        # Perform addition
        result = {}
        for layer_idx in mod1.keys():
            m1 = mod1[layer_idx]
            m2 = mod2[layer_idx]

            # Create new modulation data
            result_mod = ModulationData(
                name=result_name,
                layer_idx=layer_idx,
                source_prompt=f"[{name1}] + [{name2}]",
                extraction_position=-1,
                modulation_type=m1.modulation_type,
                num_ranks=m1.num_ranks
            )

            # Add tensors based on type
            if m1.modulation_type == "triple":
                result_mod.v_a_gate = m1.v_a_gate + m2.v_a_gate
                result_mod.v_b_gate = m1.v_b_gate + m2.v_b_gate
                result_mod.v_a_up = m1.v_a_up + m2.v_a_up
                result_mod.v_b_up = m1.v_b_up + m2.v_b_up
                result_mod.v_a_down = m1.v_a_down + m2.v_a_down
                result_mod.v_b_down = m1.v_b_down + m2.v_b_down

                result_mod.magnitude = (
                    result_mod.v_a_gate.norm().item() + result_mod.v_b_gate.norm().item() +
                    result_mod.v_a_up.norm().item() + result_mod.v_b_up.norm().item() +
                    result_mod.v_a_down.norm().item() + result_mod.v_b_down.norm().item()
                ) / 6
            elif m1.modulation_type == "dual":
                result_mod.v_a_gate = m1.v_a_gate + m2.v_a_gate
                result_mod.v_b_gate = m1.v_b_gate + m2.v_b_gate
                result_mod.v_a_up = m1.v_a_up + m2.v_a_up
                result_mod.v_b_up = m1.v_b_up + m2.v_b_up

                result_mod.magnitude = (
                    result_mod.v_a_gate.norm().item() + result_mod.v_b_gate.norm().item() +
                    result_mod.v_a_up.norm().item() + result_mod.v_b_up.norm().item()
                ) / 4
            else:  # single
                result_mod.v_a = m1.v_a + m2.v_a
                result_mod.v_b = m1.v_b + m2.v_b

                result_mod.magnitude = (result_mod.v_a.norm().item() + result_mod.v_b.norm().item()) / 2

            result[layer_idx] = result_mod

        # Save result
        self.session.modulation_bank[result_name] = result
        print(f"✓ Created modulation '{Colors.CYAN}{result_name}{Colors.END}' = {name1} + {name2}")

        # Show magnitudes
        for layer_idx, mod_data in sorted(result.items()):
            print(f"  Layer {layer_idx}: magnitude={Colors.YELLOW}{mod_data.magnitude:.6f}{Colors.END}")

    def cmd_average(self, args: List[str]):
        """Average multiple modulations."""
        if len(args) < 3:
            print(f"{Colors.RED}Usage: average <name1> <name2> [...] <result_name>{Colors.END}")
            return

        # Last argument is result name
        mod_names = args[:-1]
        result_name = args[-1]

        # Check all modulations exist
        mods = []
        for name in mod_names:
            if name not in self.session.modulation_bank:
                print(f"{Colors.RED}Modulation '{name}' not found{Colors.END}")
                return
            mods.append(self.session.modulation_bank[name])

        # Check all compatible with first
        for i, mod in enumerate(mods[1:], 1):
            if not self._check_modulation_compatibility(mods[0], mod):
                print(f"{Colors.RED}Modulation '{mod_names[i]}' incompatible with '{mod_names[0]}'{Colors.END}")
                return

        # Perform averaging
        result = {}
        for layer_idx in mods[0].keys():
            # Get all modulations for this layer
            layer_mods = [mod[layer_idx] for mod in mods]

            # Create result modulation
            result_mod = ModulationData(
                name=result_name,
                layer_idx=layer_idx,
                source_prompt=f"Average of {mod_names}",
                extraction_position=-1,
                modulation_type=layer_mods[0].modulation_type,
                num_ranks=layer_mods[0].num_ranks
            )

            # Average tensors based on type
            if layer_mods[0].modulation_type == "triple":
                result_mod.v_a_gate = torch.stack([m.v_a_gate for m in layer_mods]).mean(dim=0)
                result_mod.v_b_gate = torch.stack([m.v_b_gate for m in layer_mods]).mean(dim=0)
                result_mod.v_a_up = torch.stack([m.v_a_up for m in layer_mods]).mean(dim=0)
                result_mod.v_b_up = torch.stack([m.v_b_up for m in layer_mods]).mean(dim=0)
                result_mod.v_a_down = torch.stack([m.v_a_down for m in layer_mods]).mean(dim=0)
                result_mod.v_b_down = torch.stack([m.v_b_down for m in layer_mods]).mean(dim=0)

                result_mod.magnitude = (
                    result_mod.v_a_gate.norm().item() + result_mod.v_b_gate.norm().item() +
                    result_mod.v_a_up.norm().item() + result_mod.v_b_up.norm().item() +
                    result_mod.v_a_down.norm().item() + result_mod.v_b_down.norm().item()
                ) / 6
            elif layer_mods[0].modulation_type == "dual":
                result_mod.v_a_gate = torch.stack([m.v_a_gate for m in layer_mods]).mean(dim=0)
                result_mod.v_b_gate = torch.stack([m.v_b_gate for m in layer_mods]).mean(dim=0)
                result_mod.v_a_up = torch.stack([m.v_a_up for m in layer_mods]).mean(dim=0)
                result_mod.v_b_up = torch.stack([m.v_b_up for m in layer_mods]).mean(dim=0)

                result_mod.magnitude = (
                    result_mod.v_a_gate.norm().item() + result_mod.v_b_gate.norm().item() +
                    result_mod.v_a_up.norm().item() + result_mod.v_b_up.norm().item()
                ) / 4
            else:  # single
                result_mod.v_a = torch.stack([m.v_a for m in layer_mods]).mean(dim=0)
                result_mod.v_b = torch.stack([m.v_b for m in layer_mods]).mean(dim=0)

                result_mod.magnitude = (result_mod.v_a.norm().item() + result_mod.v_b.norm().item()) / 2

            result[layer_idx] = result_mod

        # Save result
        self.session.modulation_bank[result_name] = result
        print(f"✓ Created modulation '{Colors.CYAN}{result_name}{Colors.END}' = average({', '.join(mod_names)})")

        # Show magnitudes
        for layer_idx, mod_data in sorted(result.items()):
            print(f"  Layer {layer_idx}: magnitude={Colors.YELLOW}{mod_data.magnitude:.6f}{Colors.END}")

    def cmd_scale(self, args: List[str]):
        """Scale a modulation by a factor."""
        if len(args) < 3:
            print(f"{Colors.RED}Usage: scale <name> <factor> <result_name>{Colors.END}")
            return

        name = args[0]
        try:
            factor = float(args[1])
        except ValueError:
            print(f"{Colors.RED}Invalid factor: {args[1]}{Colors.END}")
            return
        result_name = args[2]

        if name not in self.session.modulation_bank:
            print(f"{Colors.RED}Modulation '{name}' not found{Colors.END}")
            return

        mod = self.session.modulation_bank[name]

        # Scale modulation
        result = {}
        for layer_idx, m in mod.items():
            result_mod = ModulationData(
                name=result_name,
                layer_idx=layer_idx,
                source_prompt=f"{factor} * [{m.source_prompt}]",
                extraction_position=m.extraction_position,
                modulation_type=m.modulation_type,
                num_ranks=m.num_ranks
            )

            # Scale tensors based on type
            if m.modulation_type == "triple":
                result_mod.v_a_gate = m.v_a_gate * factor
                result_mod.v_b_gate = m.v_b_gate * factor
                result_mod.v_a_up = m.v_a_up * factor
                result_mod.v_b_up = m.v_b_up * factor
                result_mod.v_a_down = m.v_a_down * factor
                result_mod.v_b_down = m.v_b_down * factor

                result_mod.magnitude = abs(factor) * m.magnitude
            elif m.modulation_type == "dual":
                result_mod.v_a_gate = m.v_a_gate * factor
                result_mod.v_b_gate = m.v_b_gate * factor
                result_mod.v_a_up = m.v_a_up * factor
                result_mod.v_b_up = m.v_b_up * factor

                result_mod.magnitude = abs(factor) * m.magnitude
            else:  # single
                result_mod.v_a = m.v_a * factor
                result_mod.v_b = m.v_b * factor

                result_mod.magnitude = abs(factor) * m.magnitude

            result[layer_idx] = result_mod

        # Save result
        self.session.modulation_bank[result_name] = result
        print(f"✓ Created modulation '{Colors.CYAN}{result_name}{Colors.END}' = {factor} * {name}")

        # Show magnitudes
        for layer_idx, mod_data in sorted(result.items()):
            print(f"  Layer {layer_idx}: magnitude={Colors.YELLOW}{mod_data.magnitude:.6f}{Colors.END}")

    def cmd_negate(self, args: List[str]):
        """Negate a modulation (multiply by -1)."""
        if len(args) < 2:
            print(f"{Colors.RED}Usage: negate <name> <result_name>{Colors.END}")
            return

        # Use scale with factor -1
        self.cmd_scale([args[0], '-1', args[1]])

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


def load_model_with_npt(checkpoint_path: str, model_name: str,
                        layers_to_use: List[int], device: str = "cuda") -> Tuple[NPTLlamaModel, AutoTokenizer, Set[int]]:
    """Load NPT model with selective layer activation."""
    print(f"Loading model from {checkpoint_path}...")

    # Load base model
    model_config = AutoConfig.from_pretrained(model_name)
    model_config._attn_implementation = "eager"
    model = NPTLlamaModel.from_pretrained(model_name, config=model_config)

    # Load NPT weights
    checkpoint_path = Path(checkpoint_path)
    npt_weights_path = checkpoint_path / "npt_weights.pt"

    available_layers = set()

    if npt_weights_path.exists():
        state_dict = torch.load(npt_weights_path, map_location='cpu', weights_only=False)

        # Detect available layers and configuration
        for key in state_dict.keys():
            if 'layer_' in key and '_np' in key:
                parts = key.split('_')
                for i, part in enumerate(parts):
                    if part == 'layer' and i + 1 < len(parts) and parts[i + 1].isdigit():
                        available_layers.add(int(parts[i + 1]))

        print(f"Available NPT layers: {sorted(available_layers)}")

        # Determine which layers to convert
        layers_to_convert = [l for l in layers_to_use if l in available_layers]

        if layers_to_convert:
            # Detect modulation configuration
            all_keys = list(state_dict.keys())
            dual_modulation = any('W_down_gate' in k or 'W_a_up_gate' in k for k in all_keys)
            triple_modulation = any('W_down_down' in k or 'W_a_up_down' in k for k in all_keys)

            # Detect num_ranks
            num_ranks = 1
            for i in range(16):
                if any(f'.{i}' in k and ('W_down' in k or 'W_a_up' in k) for k in all_keys):
                    num_ranks = max(num_ranks, i + 1)

            # Detect np_rank from checkpoint tensor shapes
            np_rank = 256  # default
            for key, value in state_dict.items():
                if 'W_down_gate' in key or 'W_down_up' in key or 'W_down' in key:
                    if isinstance(value, torch.Tensor) and value.dim() == 2:
                        # Shape is [d_model, rank], so second dimension is the rank
                        np_rank = value.shape[1]
                        print(f"Detected np_rank={np_rank} from checkpoint")
                        break

            # Create NPT config
            npt_config = NPTConfig(
                layers_to_convert=layers_to_convert,
                np_rank=np_rank,  # Use detected rank
                np_init_scale=0.001,
                single_layer_mode=False,
                num_ranks=num_ranks,
                init_strategy="improved",
                dual_modulation=dual_modulation,
                triple_modulation=triple_modulation
            )

            # Convert layers
            model.convert_to_npt(npt_config)

            # Load weights
            model.load_npt_weights(state_dict)

    # Move to device
    model = model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, set(layers_to_convert) if 'layers_to_convert' in locals() else set()


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