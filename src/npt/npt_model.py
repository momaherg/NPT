"""
NPT Model implementation with selective layer conversion.

This module implements a hybrid Llama model where specific layers
can be converted to NPT layers while others remain standard.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Union, Dict, Any
from transformers import LlamaForCausalLM, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from copy import deepcopy

from .npt_decoder_layer import NPTDecoderLayer


class NPTConfig:
    """Configuration for NPT model conversion."""
    
    def __init__(
        self,
        layers_to_convert: Optional[List[int]] = None,
        convert_range: Optional[tuple] = None,
        np_rank: int = 64,
        np_init_scale: float = 0.01,
        convert_all: bool = False,
        single_layer_mode: bool = False,
    ):
        """
        Initialize NPT configuration.
        
        Args:
            layers_to_convert: List of specific layer indices to convert
            convert_range: Tuple (start, end) for range of layers to convert
            np_rank: Rank for NP components
            np_init_scale: Initialization scale for NP components
            convert_all: If True, convert all layers
            single_layer_mode: If True, use special initialization for single-layer NPT
        """
        self.layers_to_convert = layers_to_convert
        self.convert_range = convert_range
        self.np_rank = np_rank
        self.np_init_scale = np_init_scale
        self.convert_all = convert_all
        self.single_layer_mode = single_layer_mode
        
        # Validate configuration
        if sum([layers_to_convert is not None, 
                convert_range is not None, 
                convert_all]) > 1:
            raise ValueError("Specify only one of: layers_to_convert, convert_range, or convert_all")
    
    def get_layers_to_convert(self, num_layers: int) -> List[int]:
        """
        Get the list of layer indices to convert.
        
        Args:
            num_layers: Total number of layers in the model
            
        Returns:
            List of layer indices to convert
        """
        if self.convert_all:
            return list(range(num_layers))
        elif self.convert_range is not None:
            start, end = self.convert_range
            # Handle negative indices
            if start < 0:
                start = num_layers + start
            if end < 0:
                end = num_layers + end
            return list(range(start, min(end, num_layers)))
        elif self.layers_to_convert is not None:
            # Validate and handle negative indices
            result = []
            for idx in self.layers_to_convert:
                if idx < 0:
                    idx = num_layers + idx
                if 0 <= idx < num_layers:
                    result.append(idx)
            return sorted(result)  # Return sorted list for consistency
        else:
            # Default: convert upper half of layers
            return list(range(num_layers // 2, num_layers))


class NPTLlamaModel(LlamaForCausalLM):
    """
    Hybrid NPT-Llama model with selective layer conversion.
    
    This model allows converting specific transformer layers to NPT layers
    while keeping others as standard layers.
    """
    
    def __init__(self, config: LlamaConfig):
        """
        Initialize NPT model.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        
        # Store NPT configuration
        self.npt_config = None
        self.npt_layers = {}  # Maps layer index to NPT layer
        self.original_layers = {}  # Store references to original layers
        
        # Add NPT parameters to config if not present
        if not hasattr(config, 'np_rank'):
            config.np_rank = 64
        if not hasattr(config, 'np_init_scale'):
            config.np_init_scale = 0.01
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        npt_config: Optional[NPTConfig] = None,
        *model_args,
        **kwargs
    ):
        """
        Load a pretrained model and optionally convert layers to NPT.
        
        Args:
            pretrained_model_name_or_path: Model name or path
            npt_config: NPT configuration for layer conversion
            *model_args: Additional model arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            NPTLlamaModel with converted layers
        """
        # Load the base model
        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            **kwargs
        )
        
        # Convert to NPT if config provided
        if npt_config is not None:
            model.convert_to_npt(npt_config)
        
        return model
    
    def convert_to_npt(self, npt_config: NPTConfig):
        """
        Convert specified layers to NPT layers.
        
        Args:
            npt_config: Configuration specifying which layers to convert
        """
        self.npt_config = npt_config
        
        # Add NPT parameters to model config
        self.config.np_rank = npt_config.np_rank
        self.config.np_init_scale = npt_config.np_init_scale
        self.config.single_layer_mode = npt_config.single_layer_mode
        
        # Get layers to convert
        num_layers = len(self.model.layers)
        layers_to_convert = npt_config.get_layers_to_convert(num_layers)
        
        print(f"Converting layers {layers_to_convert} to NPT layers")
        
        # Convert specified layers
        for layer_idx in layers_to_convert:
            self._convert_layer(layer_idx)
        
        # Update the converted layers count
        self.num_npt_layers = len(self.npt_layers)
        print(f"Converted {self.num_npt_layers}/{num_layers} layers to NPT")
    
    def _convert_layer(self, layer_idx: int):
        """
        Convert a single layer to NPT layer.
        
        Args:
            layer_idx: Index of the layer to convert
        """
        # Get the original layer
        original_layer = self.model.layers[layer_idx]
        
        # Store reference to original layer
        self.original_layers[layer_idx] = original_layer
        
        # Create NPT layer
        npt_layer = NPTDecoderLayer(self.config, layer_idx)
        
        # Copy weights from original layer to NPT layer
        # The NPT layer inherits from LlamaDecoderLayer, so state dict should be compatible
        # We need to handle the fact that NPT layer has additional NP component parameters
        original_state = original_layer.state_dict()
        npt_layer.load_state_dict(original_state, strict=False)
        
        # Replace the layer in the model
        self.model.layers[layer_idx] = npt_layer
        self.npt_layers[layer_idx] = npt_layer
        
        # Set NPT mode to True by default
        npt_layer.set_npt_mode(True)
    
    def set_npt_mode(self, use_npt: bool):
        """
        Set NPT mode for all converted layers.
        
        Args:
            use_npt: If True, use NPT mode; if False, use standard mode
        """
        for layer in self.npt_layers.values():
            layer.set_npt_mode(use_npt)
    
    def freeze_base_parameters(self):
        """
        Freeze all parameters except NP components.
        
        This is used during equivalence pre-training.
        """
        # First, freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
        
        # Then unfreeze NP component parameters
        for layer in self.npt_layers.values():
            for param in layer.np_component.parameters():
                param.requires_grad = True
    
    def get_npt_parameters(self) -> List[nn.Parameter]:
        """
        Get all NP component parameters.
        
        Returns:
            List of all NP component parameters
        """
        npt_params = []
        for layer in self.npt_layers.values():
            npt_params.extend(layer.np_component.parameters())
        return npt_params
    
    def get_npt_parameter_groups(self) -> Dict[str, List[nn.Parameter]]:
        """
        Get NP component parameters grouped by layer.
        
        Returns:
            Dictionary mapping layer names to parameter lists
        """
        param_groups = {}
        for layer_idx, layer in self.npt_layers.items():
            group_name = f"layer_{layer_idx}_np"
            param_groups[group_name] = list(layer.np_component.parameters())
        return param_groups
    
    def count_parameters(self) -> Dict[str, int]:
        """
        Count parameters in the model.
        
        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        npt_params = sum(p.numel() for p in self.get_npt_parameters())
        base_params = total_params - npt_params
        
        return {
            'total': total_params,
            'base': base_params,
            'npt': npt_params,
            'npt_ratio': npt_params / total_params if total_params > 0 else 0
        }
    
    def save_npt_weights(self, save_path: str):
        """
        Save only the NP component weights.
        
        Args:
            save_path: Path to save the weights
        """
        npt_state_dict = {}
        for layer_idx, layer in self.npt_layers.items():
            prefix = f"layer_{layer_idx}_np"
            for name, param in layer.np_component.named_parameters():
                npt_state_dict[f"{prefix}.{name}"] = param.detach().cpu()
        
        torch.save(npt_state_dict, save_path)
        print(f"Saved NPT weights to {save_path}")
    
    def load_npt_weights(self, load_path: str):
        """
        Load NP component weights.
        
        Args:
            load_path: Path to load the weights from
        """
        npt_state_dict = torch.load(load_path, map_location='cpu')
        
        for layer_idx, layer in self.npt_layers.items():
            prefix = f"layer_{layer_idx}_np"
            layer_state = {}
            for key, value in npt_state_dict.items():
                if key.startswith(prefix):
                    param_name = key[len(prefix)+1:]  # Remove prefix and dot
                    layer_state[param_name] = value
            
            if layer_state:
                layer.np_component.load_state_dict(layer_state)
        
        print(f"Loaded NPT weights from {load_path}")
    
    def get_layer_info(self) -> Dict[str, Any]:
        """
        Get information about model layers.
        
        Returns:
            Dictionary with layer information
        """
        info = {
            'total_layers': len(self.model.layers),
            'npt_layers': len(self.npt_layers),
            'npt_layer_indices': sorted(list(self.npt_layers.keys())),
            'standard_layer_indices': [
                i for i in range(len(self.model.layers)) 
                if i not in self.npt_layers
            ]
        }
        
        # Add layer types
        layer_types = []
        for i, layer in enumerate(self.model.layers):
            if isinstance(layer, NPTDecoderLayer):
                layer_types.append('NPT')
            else:
                layer_types.append('Standard')
        info['layer_types'] = layer_types
        
        return info
    
    def reset_to_standard(self):
        """
        Reset all NPT layers back to standard layers.
        
        This restores the original layers if they were stored.
        """
        for layer_idx, original_layer in self.original_layers.items():
            self.model.layers[layer_idx] = original_layer
        
        self.npt_layers.clear()
        self.original_layers.clear()
        print("Reset all layers to standard transformer layers")