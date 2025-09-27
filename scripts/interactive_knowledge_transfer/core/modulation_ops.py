"""
Core modulation operations: extraction, injection, and arithmetic.
"""

import torch
from typing import Dict, Optional, Any, List, Tuple
from .data_types import ModulationData


class ModulationOperations:
    """Handles modulation extraction, injection, and arithmetic operations."""

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def extract_modulation(self, prompt: str, position: Optional[int],
                          active_layers: set) -> Dict[int, ModulationData]:
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
        for layer_idx in active_layers:
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
                    self.model.model.layers[idx].set_npt_mode(idx in active_layers)

            _ = self.model(input_ids)

        # Remove hooks
        for handle in handles:
            handle.remove()

        return modulations

    def extract_sequence_modulations(self, prompt: str, active_layers: set) -> Dict[int, ModulationData]:
        """
        Extract modulations from all token positions in the prompt and average them.

        This provides a more complete representation of the sentence's semantic content
        by capturing the evolution of modulations across all positions.
        """
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        input_ids = inputs['input_ids']
        seq_len = input_ids.shape[1]

        if seq_len == 0:
            return {}

        # Store modulations for each layer at each position
        layer_position_modulations = {}  # {layer_idx: [mod_at_pos_0, mod_at_pos_1, ...]}

        # Hook to capture modulations at all positions
        def create_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    position_mods = []

                    # Extract modulation at each position
                    for pos in range(seq_len):
                        mod_data = ModulationData(
                            name="temp",
                            layer_idx=layer_idx,
                            source_prompt=prompt,
                            extraction_position=pos
                        )

                        if isinstance(output[0], tuple):
                            # Triple or Dual modulation
                            if len(output) == 3:
                                # Triple modulation
                                (v_a_gate, v_b_gate), (v_a_up, v_b_up), (v_a_down, v_b_down) = output
                                mod_data.modulation_type = "triple"
                                mod_data.v_a_gate = v_a_gate[:, pos:pos+1].clone().detach()
                                mod_data.v_b_gate = v_b_gate[:, pos:pos+1].clone().detach()
                                mod_data.v_a_up = v_a_up[:, pos:pos+1].clone().detach()
                                mod_data.v_b_up = v_b_up[:, pos:pos+1].clone().detach()
                                mod_data.v_a_down = v_a_down[:, pos:pos+1].clone().detach()
                                mod_data.v_b_down = v_b_down[:, pos:pos+1].clone().detach()
                            else:
                                # Dual modulation
                                (v_a_gate, v_b_gate), (v_a_up, v_b_up) = output[:2]
                                mod_data.modulation_type = "dual"
                                mod_data.v_a_gate = v_a_gate[:, pos:pos+1].clone().detach()
                                mod_data.v_b_gate = v_b_gate[:, pos:pos+1].clone().detach()
                                mod_data.v_a_up = v_a_up[:, pos:pos+1].clone().detach()
                                mod_data.v_b_up = v_b_up[:, pos:pos+1].clone().detach()
                        else:
                            # Single modulation
                            v_a, v_b = output
                            mod_data.modulation_type = "single"
                            mod_data.v_a = v_a[:, pos:pos+1].clone().detach()
                            mod_data.v_b = v_b[:, pos:pos+1].clone().detach()

                        # Check for rank-k
                        sample_tensor = mod_data.v_a if mod_data.v_a is not None else mod_data.v_a_gate
                        if sample_tensor.dim() == 4:
                            mod_data.num_ranks = sample_tensor.shape[2]
                        else:
                            mod_data.num_ranks = 1

                        position_mods.append(mod_data)

                    layer_position_modulations[layer_idx] = position_mods
            return hook

        # Register hooks
        handles = []
        for layer_idx in active_layers:
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
                    self.model.model.layers[idx].set_npt_mode(idx in active_layers)

            _ = self.model(input_ids)

        # Remove hooks
        for handle in handles:
            handle.remove()

        # Average modulations across positions for each layer
        averaged_modulations = {}

        for layer_idx, position_mods in layer_position_modulations.items():
            if not position_mods:
                continue

            # Create result modulation
            result_mod = ModulationData(
                name="averaged",
                layer_idx=layer_idx,
                source_prompt=f"Average of {seq_len} tokens from: {prompt[:50]}...",
                extraction_position=-1,  # Special value for averaged
                modulation_type=position_mods[0].modulation_type,
                num_ranks=position_mods[0].num_ranks
            )

            # Average based on modulation type
            if position_mods[0].modulation_type == "triple":
                result_mod.v_a_gate = torch.stack([m.v_a_gate for m in position_mods]).mean(dim=0)
                result_mod.v_b_gate = torch.stack([m.v_b_gate for m in position_mods]).mean(dim=0)
                result_mod.v_a_up = torch.stack([m.v_a_up for m in position_mods]).mean(dim=0)
                result_mod.v_b_up = torch.stack([m.v_b_up for m in position_mods]).mean(dim=0)
                result_mod.v_a_down = torch.stack([m.v_a_down for m in position_mods]).mean(dim=0)
                result_mod.v_b_down = torch.stack([m.v_b_down for m in position_mods]).mean(dim=0)

                result_mod.magnitude = (
                    result_mod.v_a_gate.norm().item() + result_mod.v_b_gate.norm().item() +
                    result_mod.v_a_up.norm().item() + result_mod.v_b_up.norm().item() +
                    result_mod.v_a_down.norm().item() + result_mod.v_b_down.norm().item()
                ) / 6
            elif position_mods[0].modulation_type == "dual":
                result_mod.v_a_gate = torch.stack([m.v_a_gate for m in position_mods]).mean(dim=0)
                result_mod.v_b_gate = torch.stack([m.v_b_gate for m in position_mods]).mean(dim=0)
                result_mod.v_a_up = torch.stack([m.v_a_up for m in position_mods]).mean(dim=0)
                result_mod.v_b_up = torch.stack([m.v_b_up for m in position_mods]).mean(dim=0)

                result_mod.magnitude = (
                    result_mod.v_a_gate.norm().item() + result_mod.v_b_gate.norm().item() +
                    result_mod.v_a_up.norm().item() + result_mod.v_b_up.norm().item()
                ) / 4
            else:  # single
                result_mod.v_a = torch.stack([m.v_a for m in position_mods]).mean(dim=0)
                result_mod.v_b = torch.stack([m.v_b for m in position_mods]).mean(dim=0)

                result_mod.magnitude = (result_mod.v_a.norm().item() + result_mod.v_b.norm().item()) / 2

            averaged_modulations[layer_idx] = result_mod

        return averaged_modulations

    def create_injection_hook(self, layer_idx: int, source_mod: ModulationData,
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