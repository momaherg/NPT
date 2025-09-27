"""
Model loading and configuration utilities.
"""

import torch
from pathlib import Path
from typing import Tuple, List, Set
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.npt import NPTLlamaModel, NPTConfig
from transformers import AutoTokenizer, AutoConfig


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


def detect_modulation_config(model, active_layers: Set[int]) -> Tuple[str, int]:
    """Detect modulation type and number of ranks from the model."""
    modulation_type = "unknown"
    num_ranks = 1

    if not active_layers:
        return modulation_type, num_ranks

    layer_idx = min(active_layers)
    if layer_idx < len(model.model.layers):
        layer = model.model.layers[layer_idx]
        if hasattr(layer, 'np_component'):
            np_comp = layer.np_component
            if hasattr(np_comp, 'triple_modulation') and np_comp.triple_modulation:
                modulation_type = "triple"
            elif hasattr(np_comp, 'dual_modulation') and np_comp.dual_modulation:
                modulation_type = "dual"
            else:
                modulation_type = "single"

            if hasattr(np_comp, 'num_ranks'):
                num_ranks = np_comp.num_ranks

    return modulation_type, num_ranks