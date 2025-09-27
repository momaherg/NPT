"""
Modulation management commands for Interactive Knowledge Transfer.
"""

from typing import List, Optional
from .base import BaseCommandHandler
from ..core.data_types import Colors
from ..core.modulation_ops import ModulationOperations
from ..core.arithmetic_ops import ModulationArithmetic


class ModulationCommands(BaseCommandHandler):
    """Handles modulation management commands."""

    def __init__(self, model, tokenizer, device, session, mod_ops: ModulationOperations,
                 mod_arithmetic: ModulationArithmetic):
        super().__init__(model, tokenizer, device, session)
        self.mod_ops = mod_ops
        self.mod_arithmetic = mod_arithmetic

    def cmd_extract(self, args: List[str]):
        """Extract modulation from a prompt."""
        if len(args) < 2:
            self.error("Usage: extract <name> \"prompt\" [-pos N]")
            return

        name = args[0]
        prompt = args[1]

        # Check for position argument
        position = None
        if len(args) > 2 and args[2] == '-pos' and len(args) > 3:
            try:
                position = int(args[3])
            except ValueError:
                self.error(f"Invalid position: {args[3]}")
                return

        # Extract modulation
        print(f"Extracting modulation '{Colors.CYAN}{name}{Colors.END}' from prompt...")
        modulations = self.mod_ops.extract_modulation(prompt, position, self.session.active_layers)

        if modulations:
            # Store in bank
            self.session.modulation_bank[name] = modulations
            self.success(f"Extracted modulation '{Colors.CYAN}{name}{Colors.END}'")

            # Show layer magnitudes
            for layer_idx, mod_data in modulations.items():
                print(f"  Layer {layer_idx}: magnitude={Colors.YELLOW}{mod_data.magnitude:.6f}{Colors.END}")
        else:
            self.error("Failed to extract modulation")

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
            self.error("Usage: delete <name>")
            return

        name = args[0]
        if name in self.session.modulation_bank:
            del self.session.modulation_bank[name]
            self.success(f"Deleted modulation '{Colors.CYAN}{name}{Colors.END}'")
        else:
            self.error(f"Modulation '{name}' not found")

    def cmd_info(self, args: List[str]):
        """Show detailed info about a modulation."""
        if not args:
            self.error("Usage: info <name>")
            return

        name = args[0]
        if name not in self.session.modulation_bank:
            self.error(f"Modulation '{name}' not found")
            return

        from ..utils.display import DisplayUtils
        display = DisplayUtils(self.tokenizer)

        print(f"\n{Colors.BOLD}Modulation: {Colors.CYAN}{name}{Colors.END}")
        layer_mods = self.session.modulation_bank[name]

        for layer_idx, mod_data in sorted(layer_mods.items()):
            print(f"\n  {Colors.YELLOW}Layer {layer_idx}:{Colors.END}")
            display.display_modulation_info(mod_data)

    def cmd_subtract(self, args: List[str]):
        """Subtract two modulations: result = mod1 - mod2."""
        if len(args) < 3:
            self.error("Usage: subtract <name1> <name2> <result_name>")
            return

        name1, name2, result_name = args[0], args[1], args[2]

        if name1 not in self.session.modulation_bank:
            self.error(f"Modulation '{name1}' not found")
            return
        if name2 not in self.session.modulation_bank:
            self.error(f"Modulation '{name2}' not found")
            return

        mod1 = self.session.modulation_bank[name1]
        mod2 = self.session.modulation_bank[name2]

        # Check compatibility
        if not self.mod_arithmetic.check_compatibility(mod1, mod2):
            return

        # Perform subtraction
        result = self.mod_arithmetic.subtract(mod1, mod2, name1, name2, result_name)

        # Save result
        self.session.modulation_bank[result_name] = result
        self.success(f"Created modulation '{Colors.CYAN}{result_name}{Colors.END}' = {name1} - {name2}")

        # Show magnitudes
        for layer_idx, mod_data in sorted(result.items()):
            print(f"  Layer {layer_idx}: magnitude={Colors.YELLOW}{mod_data.magnitude:.6f}{Colors.END}")

    def cmd_add(self, args: List[str]):
        """Add two modulations: result = mod1 + mod2."""
        if len(args) < 3:
            self.error("Usage: add <name1> <name2> <result_name>")
            return

        name1, name2, result_name = args[0], args[1], args[2]

        if name1 not in self.session.modulation_bank:
            self.error(f"Modulation '{name1}' not found")
            return
        if name2 not in self.session.modulation_bank:
            self.error(f"Modulation '{name2}' not found")
            return

        mod1 = self.session.modulation_bank[name1]
        mod2 = self.session.modulation_bank[name2]

        # Check compatibility
        if not self.mod_arithmetic.check_compatibility(mod1, mod2):
            return

        # Perform addition
        result = self.mod_arithmetic.add(mod1, mod2, name1, name2, result_name)

        # Save result
        self.session.modulation_bank[result_name] = result
        self.success(f"Created modulation '{Colors.CYAN}{result_name}{Colors.END}' = {name1} + {name2}")

        # Show magnitudes
        for layer_idx, mod_data in sorted(result.items()):
            print(f"  Layer {layer_idx}: magnitude={Colors.YELLOW}{mod_data.magnitude:.6f}{Colors.END}")

    def cmd_average(self, args: List[str]):
        """Average multiple modulations."""
        if len(args) < 3:
            self.error("Usage: average <name1> <name2> [...] <result_name>")
            return

        # Last argument is result name
        mod_names = args[:-1]
        result_name = args[-1]

        # Check all modulations exist
        mods = []
        for name in mod_names:
            if name not in self.session.modulation_bank:
                self.error(f"Modulation '{name}' not found")
                return
            mods.append(self.session.modulation_bank[name])

        # Check all compatible with first
        for i, mod in enumerate(mods[1:], 1):
            if not self.mod_arithmetic.check_compatibility(mods[0], mod):
                self.error(f"Modulation '{mod_names[i]}' incompatible with '{mod_names[0]}'")
                return

        # Perform averaging
        result = self.mod_arithmetic.average(mods, mod_names, result_name)

        # Save result
        self.session.modulation_bank[result_name] = result
        self.success(f"Created modulation '{Colors.CYAN}{result_name}{Colors.END}' = average({', '.join(mod_names)})")

        # Show magnitudes
        for layer_idx, mod_data in sorted(result.items()):
            print(f"  Layer {layer_idx}: magnitude={Colors.YELLOW}{mod_data.magnitude:.6f}{Colors.END}")

    def cmd_scale(self, args: List[str]):
        """Scale a modulation by a factor."""
        if len(args) < 3:
            self.error("Usage: scale <name> <factor> <result_name>")
            return

        name = args[0]
        try:
            factor = float(args[1])
        except ValueError:
            self.error(f"Invalid factor: {args[1]}")
            return
        result_name = args[2]

        if name not in self.session.modulation_bank:
            self.error(f"Modulation '{name}' not found")
            return

        mod = self.session.modulation_bank[name]

        # Scale modulation
        result = self.mod_arithmetic.scale(mod, factor, result_name)

        # Save result
        self.session.modulation_bank[result_name] = result
        self.success(f"Created modulation '{Colors.CYAN}{result_name}{Colors.END}' = {factor} * {name}")

        # Show magnitudes
        for layer_idx, mod_data in sorted(result.items()):
            print(f"  Layer {layer_idx}: magnitude={Colors.YELLOW}{mod_data.magnitude:.6f}{Colors.END}")

    def cmd_negate(self, args: List[str]):
        """Negate a modulation (multiply by -1)."""
        if len(args) < 2:
            self.error("Usage: negate <name> <result_name>")
            return

        # Use scale with factor -1
        self.cmd_scale([args[0], '-1', args[1]])

    def cmd_extract_sequence(self, args: List[str]):
        """Extract modulations from all token positions and average them."""
        if len(args) < 2:
            self.error("Usage: extract-seq <name> \"prompt\"")
            return

        name = args[0]
        prompt = args[1]

        # Extract modulations from all positions
        print(f"Extracting sequence modulation '{Colors.CYAN}{name}{Colors.END}' from prompt...")

        # Tokenize to get token count
        tokens = self.tokenizer(prompt, add_special_tokens=False).input_ids
        num_tokens = len(tokens)

        if num_tokens == 0:
            self.error("Empty prompt provided")
            return

        print(f"Processing {num_tokens} token positions...")

        # Extract and average modulations
        averaged_modulations = self.mod_ops.extract_sequence_modulations(
            prompt, self.session.active_layers
        )

        if averaged_modulations:
            # Store in bank
            self.session.modulation_bank[name] = averaged_modulations
            self.success(f"Extracted sequence modulation '{Colors.CYAN}{name}{Colors.END}' ({num_tokens} tokens averaged)")

            # Show layer magnitudes
            for layer_idx, mod_data in averaged_modulations.items():
                print(f"  Layer {layer_idx}: magnitude={Colors.YELLOW}{mod_data.magnitude:.6f}{Colors.END}")
        else:
            self.error("Failed to extract sequence modulation")