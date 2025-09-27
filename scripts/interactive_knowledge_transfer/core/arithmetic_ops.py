"""
Modulation arithmetic operations: add, subtract, average, scale, negate.
"""

import torch
from typing import Dict, List
from .data_types import ModulationData, Colors


class ModulationArithmetic:
    """Handles arithmetic operations on modulations."""

    @staticmethod
    def check_compatibility(mod1: Dict[int, ModulationData],
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

    @staticmethod
    def subtract(mod1: Dict[int, ModulationData], mod2: Dict[int, ModulationData],
                 name1: str, name2: str, result_name: str) -> Dict[int, ModulationData]:
        """Subtract two modulations: result = mod1 - mod2."""
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

        return result

    @staticmethod
    def add(mod1: Dict[int, ModulationData], mod2: Dict[int, ModulationData],
            name1: str, name2: str, result_name: str) -> Dict[int, ModulationData]:
        """Add two modulations: result = mod1 + mod2."""
        result = {}
        for layer_idx in mod1.keys():
            m1 = mod1[layer_idx]
            m2 = mod2[layer_idx]

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

        return result

    @staticmethod
    def average(mods: List[Dict[int, ModulationData]], mod_names: List[str],
                result_name: str) -> Dict[int, ModulationData]:
        """Average multiple modulations."""
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

        return result

    @staticmethod
    def scale(mod: Dict[int, ModulationData], factor: float,
              result_name: str) -> Dict[int, ModulationData]:
        """Scale a modulation by a factor."""
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

        return result