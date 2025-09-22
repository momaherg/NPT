#!/usr/bin/env python3
"""
Interactive Knowledge Injection Experiment for NPT

This script allows interactive experimentation with permanent knowledge injection
using the Neuro-Plastic Transformer's rank-1 weight updates.

Features:
- Load pre-trained NPT checkpoint or initialize new model
- Ask questions to test current knowledge
- Inject new facts using attention-guided weight updates
- Test if injected knowledge persists
- Save modified model weights
"""

import argparse
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import json
import logging
from datetime import datetime
from colorama import init, Fore, Style

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.npt import NPTLlamaModel, NPTConfig
from transformers import AutoTokenizer, AutoConfig
import numpy as np

# Initialize colorama for colored output
init(autoreset=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def detect_npt_layers_from_weights(weights_dict: Dict) -> Dict[int, Tuple[int, int, bool]]:
    """
    Detect which layers have NPT weights, their ranks, num_ranks, and dual modulation.

    Args:
        weights_dict: Dictionary of NPT weights

    Returns:
        Dictionary mapping layer index to tuple of (rank, num_ranks, has_dual_modulation)
    """
    layer_info = {}

    # Track W_down keys per layer to detect num_ranks
    layer_w_downs = {}

    # Track dual modulation
    has_dual_modulation = {}

    for key in weights_dict.keys():
        # Parse keys for W_down weights
        if "W_down" in key:
            layer_idx = None
            component_idx = None

            if key.startswith("layer_") and "_np." in key:
                # Format: layer_0_np.W_down or layer_0_np.W_down.0
                parts = key.split(".")
                layer_idx = int(parts[0].split("_")[1])
                if len(parts) > 2 and parts[2].isdigit():
                    # Rank-k format: layer_0_np.W_down.0
                    component_idx = int(parts[2])

            elif "model.layers." in key and ".np_component." in key:
                # Format: model.layers.0.np_component.W_down or model.layers.0.np_component.W_down.0
                parts = key.split(".")
                layer_idx = int(parts[2])  # model.layers.{idx}
                # Check if there's a component index
                w_down_idx = parts.index("W_down")
                if len(parts) > w_down_idx + 1 and parts[w_down_idx + 1].isdigit():
                    component_idx = int(parts[w_down_idx + 1])

            if layer_idx is not None:
                if layer_idx not in layer_w_downs:
                    layer_w_downs[layer_idx] = {}

                # Store weight info with component index (None for rank-1)
                if component_idx is not None:
                    layer_w_downs[layer_idx][component_idx] = weights_dict[key]
                else:
                    layer_w_downs[layer_idx][None] = weights_dict[key]

    # Also check for dual modulation weights
    for key in weights_dict.keys():
        if "W_down_gate" in key or "W_a_up_gate" in key or "W_down_up" in key:
            # Extract layer index
            layer_idx = None
            if key.startswith("layer_") and "_np." in key:
                parts = key.split(".")
                layer_idx = int(parts[0].split("_")[1])
            elif "model.layers." in key and ".np_component." in key:
                parts = key.split(".")
                layer_idx = int(parts[2])

            if layer_idx is not None:
                has_dual_modulation[layer_idx] = True

    # Process collected weights to determine rank and num_ranks
    for layer_idx, components in layer_w_downs.items():
        if None in components:
            # Rank-1 case (no component indices)
            rank = components[None].shape[1]
            num_ranks = 1
        else:
            # Rank-k case (has component indices)
            num_ranks = len(components)
            # Get rank from first component
            first_weight = list(components.values())[0]
            rank = first_weight.shape[1]

        layer_info[layer_idx] = (rank, num_ranks, has_dual_modulation.get(layer_idx, False))

    return layer_info


class KnowledgeInjector:
    """Handles knowledge injection into NPT models."""
    
    def __init__(
        self,
        model: NPTLlamaModel,
        tokenizer,
        layer_idx: Optional[int] = None,
        injection_strength: float = 1.0,
        device: str = "cuda"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.injection_strength = injection_strength
        self.device = device
        
        # Get available NPT layers
        self.available_layers = sorted(model.npt_layers.keys()) if hasattr(model, 'npt_layers') else []
        
        # Set active layer (use provided, or highest available, or fallback)
        if layer_idx is not None and layer_idx in self.available_layers:
            self.active_layer_idx = layer_idx
        elif self.available_layers:
            self.active_layer_idx = self.available_layers[-1]  # Use highest layer by default
        else:
            self.active_layer_idx = layer_idx if layer_idx is not None else 15
        
        # Store injected knowledge per layer
        self.injected_facts = {}  # layer_idx -> list of facts
        self.original_weights = {}  # layer_idx -> original weights
        
        # Ensure model is on device
        self.model = self.model.to(device)
        self.model.eval()
    
    @property
    def layer_idx(self):
        """Backward compatibility property."""
        return self.active_layer_idx
    
    @layer_idx.setter
    def layer_idx(self, value):
        """Backward compatibility property setter."""
        self.switch_layer(value)
    
    def switch_layer(self, layer_idx: int):
        """
        Switch active NPT layer for injection.
        
        Args:
            layer_idx: Index of the layer to switch to
        """
        if layer_idx not in self.available_layers:
            raise ValueError(f"Layer {layer_idx} is not an NPT layer. Available: {self.available_layers}")
        
        self.active_layer_idx = layer_idx
        print(f"{Fore.GREEN}✓ Switched to layer {layer_idx}{Style.RESET_ALL}")
    
    def get_layer_info(self) -> Dict:
        """Get information about available NPT layers."""
        info = {
            "available_layers": self.available_layers,
            "active_layer": self.active_layer_idx,
            "injected_facts_count": {
                idx: len(self.injected_facts.get(idx, []))
                for idx in self.available_layers
            }
        }
        
        # Add rank and num_ranks information for each layer
        for idx in self.available_layers:
            if idx in self.model.npt_layers:
                layer = self.model.npt_layers[idx]
                info[f"layer_{idx}_rank"] = layer.np_component.rank
                info[f"layer_{idx}_num_ranks"] = layer.np_component.num_ranks
        
        return info
    
    def generate_response(self, prompt: str, max_length: int = 50) -> str:
        """Generate a response to a prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs.input_ids.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=min(input_ids.shape[1] + max_length, 512),
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    def extract_delta_weights(
        self,
        text: str,
        target_position: str = "last"
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Extract delta weights (v_a, v_b) from processing a text.

        Args:
            text: The text containing the fact to inject
            target_position: Position to extract weights from ("last", "first", "all")

        Returns:
            Tuple of (v_a, v_b, metadata)
            For rank-1: v_a shape (d_model,), v_b shape (d_ffn,)
            For rank-k: v_a shape (num_ranks, d_model), v_b shape (num_ranks, d_ffn)
        """
        # Tokenize the input
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs.input_ids.to(self.device)
        
        # Get the NPT layer
        if self.active_layer_idx not in self.model.npt_layers:
            raise ValueError(f"Layer {self.active_layer_idx} is not an NPT layer")
        
        npt_layer = self.model.npt_layers[self.active_layer_idx]
        
        # Storage for v_a and v_b
        v_a_collected = None
        v_b_collected = None
        attention_output = None
        
        # Hook to capture v_a and v_b (handles dual modulation)
        def hook_fn(module, input, output):
            nonlocal v_a_collected, v_b_collected
            if isinstance(output, tuple) and len(output) == 2:
                # Check if dual modulation (nested tuple structure)
                if isinstance(output[0], tuple) and isinstance(output[1], tuple):
                    # Dual modulation: ((v_a_gate, v_b_gate), (v_a_up, v_b_up))
                    v_a_collected = output  # Store entire dual structure
                    v_b_collected = None  # Flag for dual modulation
                else:
                    # Single modulation: (v_a, v_b)
                    v_a_collected, v_b_collected = output
        
        # Register hook
        handle = npt_layer.np_component.register_forward_hook(hook_fn)
        
        try:
            # Forward pass in NPT mode
            self.model.set_npt_mode(True)
            
            with torch.no_grad():
                # Process through model up to NPT layer
                hidden_states = self.model.model.embed_tokens(input_ids)
                
                # Create position embeddings
                batch_size, seq_len = input_ids.shape
                head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
                cos = torch.ones(batch_size, seq_len, head_dim, 
                                dtype=hidden_states.dtype, device=hidden_states.device)
                sin = torch.zeros(batch_size, seq_len, head_dim,
                                 dtype=hidden_states.dtype, device=hidden_states.device)
                position_embeddings = (cos, sin)
                
                # Process layers up to NPT layer
                for i in range(self.active_layer_idx):
                    layer = self.model.model.layers[i]
                    layer_out = layer(
                        hidden_states,
                        position_embeddings=position_embeddings,
                        use_cache=False,
                        output_attentions=False
                    )
                    hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out
                
                # Process NPT layer to trigger hook
                layer_out = npt_layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    use_cache=False,
                    output_attentions=True  # Get attention weights
                )
                
                if isinstance(layer_out, tuple) and len(layer_out) > 1:
                    hidden_states = layer_out[0]
                    attention_output = layer_out[1] if len(layer_out) > 1 else None
                else:
                    hidden_states = layer_out
        
        finally:
            handle.remove()
        
        # Check if we got dual modulation
        is_dual_modulation = v_b_collected is None

        if is_dual_modulation:
            # Dual modulation: v_a_collected contains ((v_a_gate, v_b_gate), (v_a_up, v_b_up))
            (v_a_gate_full, v_b_gate_full), (v_a_up_full, v_b_up_full) = v_a_collected

            # Check if rank-k (4D) or rank-1 (3D) - check gate component
            is_rank_k = v_a_gate_full.dim() == 4
        else:
            # Single modulation
            if v_a_collected is None or v_b_collected is None:
                raise RuntimeError("Failed to collect v_a and v_b from NPT layer")

            # Check if rank-k (4D) or rank-1 (3D)
            is_rank_k = v_a_collected.dim() == 4

        # Select position based on target_position
        if target_position == "last":
            # Get the last non-padding token position
            seq_len = input_ids.shape[1]
            if self.tokenizer.pad_token_id is not None:
                # Find last non-padding position
                non_pad_mask = (input_ids != self.tokenizer.pad_token_id).squeeze(0)
                last_pos = non_pad_mask.nonzero()[-1].item() if non_pad_mask.any() else seq_len - 1
            else:
                last_pos = seq_len - 1

            if is_dual_modulation:
                # Process dual modulation
                if is_rank_k:
                    v_a_gate = v_a_gate_full[:, last_pos, :, :].squeeze(0)  # (num_ranks, d_model)
                    v_b_gate = v_b_gate_full[:, last_pos, :, :].squeeze(0)  # (num_ranks, d_ffn)
                    v_a_up = v_a_up_full[:, last_pos, :, :].squeeze(0)
                    v_b_up = v_b_up_full[:, last_pos, :, :].squeeze(0)
                else:
                    v_a_gate = v_a_gate_full[:, last_pos, :].squeeze(0)  # (d_model,)
                    v_b_gate = v_b_gate_full[:, last_pos, :].squeeze(0)  # (d_ffn,)
                    v_a_up = v_a_up_full[:, last_pos, :].squeeze(0)
                    v_b_up = v_b_up_full[:, last_pos, :].squeeze(0)
            else:
                if is_rank_k:
                    v_a = v_a_collected[:, last_pos, :, :].squeeze(0)  # (num_ranks, d_model)
                    v_b = v_b_collected[:, last_pos, :, :].squeeze(0)  # (num_ranks, d_ffn)
                else:
                    v_a = v_a_collected[:, last_pos, :].squeeze(0)  # (d_model,)
                    v_b = v_b_collected[:, last_pos, :].squeeze(0)  # (d_ffn,)
        elif target_position == "first":
            if is_dual_modulation:
                if is_rank_k:
                    v_a_gate = v_a_gate_full[:, 0, :, :].squeeze(0)
                    v_b_gate = v_b_gate_full[:, 0, :, :].squeeze(0)
                    v_a_up = v_a_up_full[:, 0, :, :].squeeze(0)
                    v_b_up = v_b_up_full[:, 0, :, :].squeeze(0)
                else:
                    v_a_gate = v_a_gate_full[:, 0, :].squeeze(0)
                    v_b_gate = v_b_gate_full[:, 0, :].squeeze(0)
                    v_a_up = v_a_up_full[:, 0, :].squeeze(0)
                    v_b_up = v_b_up_full[:, 0, :].squeeze(0)
            else:
                if is_rank_k:
                    v_a = v_a_collected[:, 0, :, :].squeeze(0)
                    v_b = v_b_collected[:, 0, :, :].squeeze(0)
                else:
                    v_a = v_a_collected[:, 0, :].squeeze(0)
                    v_b = v_b_collected[:, 0, :].squeeze(0)
        elif target_position == "all":
            # Average across all positions
            if is_dual_modulation:
                if is_rank_k:
                    v_a_gate = v_a_gate_full.mean(dim=1).squeeze(0)
                    v_b_gate = v_b_gate_full.mean(dim=1).squeeze(0)
                    v_a_up = v_a_up_full.mean(dim=1).squeeze(0)
                    v_b_up = v_b_up_full.mean(dim=1).squeeze(0)
                else:
                    v_a_gate = v_a_gate_full.mean(dim=1).squeeze(0)
                    v_b_gate = v_b_gate_full.mean(dim=1).squeeze(0)
                    v_a_up = v_a_up_full.mean(dim=1).squeeze(0)
                    v_b_up = v_b_up_full.mean(dim=1).squeeze(0)
            else:
                if is_rank_k:
                    v_a = v_a_collected.mean(dim=1).squeeze(0)
                    v_b = v_b_collected.mean(dim=1).squeeze(0)
                else:
                    v_a = v_a_collected.mean(dim=1).squeeze(0)
                    v_b = v_b_collected.mean(dim=1).squeeze(0)
        else:
            raise ValueError(f"Unknown target_position: {target_position}")
        
        # Compute metadata and return appropriate structure
        if is_dual_modulation:
            # Dual modulation case
            if is_rank_k:
                num_ranks = v_a_gate.shape[0]
                # Compute norms for both gate and up projections
                total_v_a_gate_norm = v_a_gate.norm(dim=-1).sum().item()
                total_v_b_gate_norm = v_b_gate.norm(dim=-1).sum().item()
                total_v_a_up_norm = v_a_up.norm(dim=-1).sum().item()
                total_v_b_up_norm = v_b_up.norm(dim=-1).sum().item()

                metadata = {
                    "text": text,
                    "position": target_position,
                    "v_a_gate_norm": total_v_a_gate_norm,
                    "v_b_gate_norm": total_v_b_gate_norm,
                    "v_a_up_norm": total_v_a_up_norm,
                    "v_b_up_norm": total_v_b_up_norm,
                    "num_ranks": num_ranks,
                    "is_rank_k": True,
                    "is_dual_modulation": True,
                    "tokens": self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist()),
                    "layer_idx": self.active_layer_idx
                }
            else:
                metadata = {
                    "text": text,
                    "position": target_position,
                    "v_a_gate_norm": v_a_gate.norm().item(),
                    "v_b_gate_norm": v_b_gate.norm().item(),
                    "v_a_up_norm": v_a_up.norm().item(),
                    "v_b_up_norm": v_b_up.norm().item(),
                    "num_ranks": 1,
                    "is_rank_k": False,
                    "is_dual_modulation": True,
                    "tokens": self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist()),
                    "layer_idx": self.active_layer_idx
                }

            # Return dual modulation structure
            return {
                'type': 'dual',
                'gate': (v_a_gate, v_b_gate),
                'up': (v_a_up, v_b_up),
                'metadata': metadata
            }
        else:
            # Single modulation case (backward compatibility)
            if is_rank_k:
                num_ranks = v_a.shape[0]
                total_v_a_norm = v_a.norm(dim=-1).sum().item()
                total_v_b_norm = v_b.norm(dim=-1).sum().item()
                metadata = {
                    "text": text,
                    "position": target_position,
                    "v_a_norm": total_v_a_norm,
                    "v_b_norm": total_v_b_norm,
                    "delta_w_rank_norm": total_v_a_norm * total_v_b_norm,
                    "num_ranks": num_ranks,
                    "is_rank_k": True,
                    "is_dual_modulation": False,
                    "tokens": self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist()),
                    "layer_idx": self.active_layer_idx
                }
            else:
                metadata = {
                    "text": text,
                    "position": target_position,
                    "v_a_norm": v_a.norm().item(),
                    "v_b_norm": v_b.norm().item(),
                    "delta_w_rank1_norm": (v_b.norm() * v_a.norm()).item(),
                    "num_ranks": 1,
                    "is_rank_k": False,
                    "is_dual_modulation": False,
                    "tokens": self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist()),
                    "layer_idx": self.active_layer_idx
                }

            # Return single modulation structure (backward compatible but in dict form)
            return {
                'type': 'single',
                'modulation': (v_a, v_b),
                'metadata': metadata
            }
    
    def inject_knowledge(
        self,
        fact_text: str,
        position: str = "last",
        accumulate: bool = False,
        alpha: Optional[float] = None
    ) -> Dict:
        """
        Inject knowledge by permanently modifying MLP weights.
        
        Args:
            fact_text: The fact to inject (e.g., "The president of the US is Mohamed Maher")
            position: Position to extract update from ("last", "first", "all")
            accumulate: If True, add to existing modifications; if False, reset first
            alpha: Scaling factor for the update (defaults to injection_strength)
        
        Returns:
            Dictionary with injection metadata
        """
        if alpha is None:
            alpha = self.injection_strength
        
        print(f"\n{Fore.YELLOW}Extracting knowledge representation...{Style.RESET_ALL}")

        # Extract modulation weights for this fact
        result = self.extract_delta_weights(fact_text, position)
        metadata = result['metadata']

        # Get the NPT layer
        npt_layer = self.model.npt_layers[self.active_layer_idx]

        # Helper function to compute rank update
        def compute_rank_update(v_b, v_a, is_rank_k):
            if is_rank_k and v_a.dim() == 2:
                # Rank-k update: sum of k rank-1 updates
                delta_W = torch.zeros(v_b.shape[-1], v_a.shape[-1], device=v_a.device, dtype=v_a.dtype)
                num_ranks = v_a.shape[0]
                for i in range(num_ranks):
                    delta_W += alpha * torch.outer(v_b[i], v_a[i])
                return delta_W
            else:
                # Rank-1 update
                return alpha * torch.outer(v_b, v_a)

        # Apply the update to the MLP weights
        with torch.no_grad():
            if result['type'] == 'dual':
                # Dual modulation - update both gate and up projections
                v_a_gate, v_b_gate = result['gate']
                v_a_up, v_b_up = result['up']

                # Get current weights
                W_gate = npt_layer.mlp.gate_proj.weight
                W_up = npt_layer.mlp.up_proj.weight

                # Store original weights if first injection and not accumulating
                if self.active_layer_idx not in self.original_weights or not accumulate:
                    self.original_weights[self.active_layer_idx] = {
                        'gate': W_gate.data.clone(),
                        'up': W_up.data.clone()
                    }

                # Apply updates to gate projection
                if not accumulate:
                    # Reset to original first
                    if isinstance(self.original_weights[self.active_layer_idx], dict):
                        W_gate.data = self.original_weights[self.active_layer_idx]['gate'].clone()
                    else:
                        # Backward compatibility - single weight stored
                        W_gate.data = self.original_weights[self.active_layer_idx].clone()

                delta_W_gate = compute_rank_update(v_b_gate, v_a_gate, metadata.get('is_rank_k', False))
                W_gate.data += delta_W_gate

                # Apply updates to up projection
                if not accumulate:
                    if isinstance(self.original_weights[self.active_layer_idx], dict) and 'up' in self.original_weights[self.active_layer_idx]:
                        W_up.data = self.original_weights[self.active_layer_idx]['up'].clone()

                delta_W_up = compute_rank_update(v_b_up, v_a_up, metadata.get('is_rank_k', False))
                W_up.data += delta_W_up

                # Store delta_W for metrics (use gate as primary)
                delta_W = delta_W_gate

            else:
                # Single modulation (backward compatibility)
                v_a, v_b = result['modulation']

                # Get current MLP gate weights
                W_in = npt_layer.mlp.gate_proj.weight

                # Store original weights per layer if first injection and not accumulating
                if self.active_layer_idx not in self.original_weights or not accumulate:
                    self.original_weights[self.active_layer_idx] = W_in.data.clone()

                # Apply update
                if not accumulate:
                    # Reset to original first
                    if isinstance(self.original_weights[self.active_layer_idx], torch.Tensor):
                        W_in.data = self.original_weights[self.active_layer_idx].clone()
                    elif isinstance(self.original_weights[self.active_layer_idx], dict) and 'gate' in self.original_weights[self.active_layer_idx]:
                        W_in.data = self.original_weights[self.active_layer_idx]['gate'].clone()

                delta_W = compute_rank_update(v_b, v_a, metadata.get('is_rank_k', False))
                W_in.data += delta_W
            
            # Track injected fact for this layer
            if result['type'] == 'dual':
                # Dual modulation metrics
                injection_info = {
                    "fact": fact_text,
                    "timestamp": datetime.now().isoformat(),
                    "alpha": alpha,
                    "position": position,
                    "modulation_type": "dual",
                    "projections_modified": ["gate", "up"],
                    "v_a_gate_norm": metadata.get("v_a_gate_norm", 0),
                    "v_b_gate_norm": metadata.get("v_b_gate_norm", 0),
                    "v_a_up_norm": metadata.get("v_a_up_norm", 0),
                    "v_b_up_norm": metadata.get("v_b_up_norm", 0),
                    "gate_delta_norm": delta_W_gate.norm().item(),
                    "up_delta_norm": delta_W_up.norm().item(),
                    "gate_weight_change_ratio": (delta_W_gate.norm() / W_gate.norm()).item(),
                    "up_weight_change_ratio": (delta_W_up.norm() / W_up.norm()).item(),
                    "layer_idx": self.active_layer_idx,
                    "num_ranks": metadata.get("num_ranks", 1),
                    "is_rank_k": metadata.get("is_rank_k", False),
                    "is_dual_modulation": True
                }
            else:
                # Single modulation metrics
                injection_info = {
                    "fact": fact_text,
                    "timestamp": datetime.now().isoformat(),
                    "alpha": alpha,
                    "position": position,
                    "modulation_type": "single",
                    "projections_modified": ["gate"],
                    "v_a_norm": metadata.get("v_a_norm", 0),
                    "v_b_norm": metadata.get("v_b_norm", 0),
                    "delta_norm": delta_W.norm().item(),
                    "weight_change_ratio": (delta_W.norm() / W_in.norm()).item(),
                    "layer_idx": self.active_layer_idx,
                    "num_ranks": metadata.get("num_ranks", 1),
                    "is_rank_k": metadata.get("is_rank_k", False),
                    "is_dual_modulation": False
                }
            
            # Initialize list for this layer if needed
            if self.active_layer_idx not in self.injected_facts:
                self.injected_facts[self.active_layer_idx] = []
            
            self.injected_facts[self.active_layer_idx].append(injection_info)
        
        print(f"{Fore.GREEN}✓ Knowledge injected successfully!{Style.RESET_ALL}")

        if result['type'] == 'dual':
            print(f"  - Modulation type: Dual (gate + up projections)")
            print(f"  - Gate delta norm: {injection_info['gate_delta_norm']:.6f}")
            print(f"  - Up delta norm: {injection_info['up_delta_norm']:.6f}")
            print(f"  - Gate weight change: {injection_info['gate_weight_change_ratio']:.6f}")
            print(f"  - Up weight change: {injection_info['up_weight_change_ratio']:.6f}")
            print(f"  - Gate norms: v_a={metadata['v_a_gate_norm']:.4f}, v_b={metadata['v_b_gate_norm']:.4f}")
            print(f"  - Up norms: v_a={metadata['v_a_up_norm']:.4f}, v_b={metadata['v_b_up_norm']:.4f}")
        else:
            print(f"  - Modulation type: Single (gate only)")
            print(f"  - Delta weight norm: {delta_W.norm().item():.6f}")
            print(f"  - Weight change ratio: {injection_info['weight_change_ratio']:.6f}")
            print(f"  - v_a norm: {metadata['v_a_norm']:.4f}, v_b norm: {metadata['v_b_norm']:.4f}")

        if metadata.get('is_rank_k', False):
            print(f"  - Num ranks: {metadata['num_ranks']} (rank-k update)")
        
        return injection_info
    
    def inject_knowledge_all_layers(
        self,
        fact_text: str,
        position: str = "last",
        alpha: Optional[float] = None
    ) -> List[Dict]:
        """
        Inject knowledge into all available NPT layers.
        
        Args:
            fact_text: The fact to inject
            position: Position to extract update from
            alpha: Scaling factor for the update
        
        Returns:
            List of injection info dictionaries
        """
        if alpha is None:
            alpha = self.injection_strength
        
        results = []
        original_layer = self.active_layer_idx
        
        print(f"\n{Fore.YELLOW}Injecting into {len(self.available_layers)} NPT layers...{Style.RESET_ALL}")
        
        for layer_idx in self.available_layers:
            print(f"\n{Fore.CYAN}Layer {layer_idx}:{Style.RESET_ALL}")
            self.active_layer_idx = layer_idx
            
            try:
                info = self.inject_knowledge(
                    fact_text,
                    position=position,
                    accumulate=True,
                    alpha=alpha
                )
                results.append(info)
            except Exception as e:
                print(f"{Fore.RED}  Failed: {e}{Style.RESET_ALL}")
                results.append({"layer_idx": layer_idx, "error": str(e)})
        
        # Restore original layer
        self.active_layer_idx = original_layer
        
        print(f"\n{Fore.GREEN}✓ Injection complete across {len(results)} layers{Style.RESET_ALL}")
        return results
    
    def reset_weights(self, layer_idx: Optional[int] = None):
        """
        Reset MLP weights to original state.

        Args:
            layer_idx: Specific layer to reset, or None to reset current layer
        """
        target_layer = layer_idx if layer_idx is not None else self.active_layer_idx

        if target_layer in self.original_weights:
            npt_layer = self.model.npt_layers[target_layer]
            with torch.no_grad():
                weights = self.original_weights[target_layer]

                if isinstance(weights, dict):
                    # Dual modulation stored weights
                    if 'gate' in weights:
                        npt_layer.mlp.gate_proj.weight.data = weights['gate'].clone()
                    if 'up' in weights:
                        npt_layer.mlp.up_proj.weight.data = weights['up'].clone()
                    print(f"{Fore.CYAN}Layer {target_layer} weights (gate + up) reset to original state.{Style.RESET_ALL}")
                else:
                    # Single modulation (backward compatibility)
                    npt_layer.mlp.gate_proj.weight.data = weights.clone()
                    print(f"{Fore.CYAN}Layer {target_layer} weights (gate) reset to original state.{Style.RESET_ALL}")

            # Clear injected facts for this layer
            if target_layer in self.injected_facts:
                self.injected_facts[target_layer] = []
        else:
            print(f"{Fore.YELLOW}No original weights stored for layer {target_layer}. Nothing to reset.{Style.RESET_ALL}")
    
    def reset_all_weights(self):
        """Reset all layers to original state."""
        reset_count = 0
        for layer_idx in self.original_weights.keys():
            npt_layer = self.model.npt_layers[layer_idx]
            with torch.no_grad():
                weights = self.original_weights[layer_idx]

                if isinstance(weights, dict):
                    # Dual modulation stored weights
                    if 'gate' in weights:
                        npt_layer.mlp.gate_proj.weight.data = weights['gate'].clone()
                    if 'up' in weights:
                        npt_layer.mlp.up_proj.weight.data = weights['up'].clone()
                else:
                    # Single modulation (backward compatibility)
                    npt_layer.mlp.gate_proj.weight.data = weights.clone()
            reset_count += 1

        # Clear all injected facts
        self.injected_facts = {}

        if reset_count > 0:
            print(f"{Fore.CYAN}Reset {reset_count} layer(s) to original state.{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}No weights to reset.{Style.RESET_ALL}")
    
    def toggle_layer_mode(self, layer_idx: int, use_npt: Optional[bool] = None):
        """
        Toggle or set NPT mode for a specific layer.
        
        Args:
            layer_idx: Layer index to toggle
            use_npt: If None, toggle current state. If bool, set to that state.
        """
        if layer_idx not in self.available_layers:
            print(f"{Fore.RED}Layer {layer_idx} is not an NPT layer.{Style.RESET_ALL}")
            return
        
        layer = self.model.model.layers[layer_idx]
        if not hasattr(layer, 'set_npt_mode'):
            print(f"{Fore.RED}Layer {layer_idx} doesn't support mode switching.{Style.RESET_ALL}")
            return
        
        current_mode = layer.use_npt if hasattr(layer, 'use_npt') else True
        
        if use_npt is None:
            # Toggle
            new_mode = not current_mode
        else:
            new_mode = use_npt
        
        layer.set_npt_mode(new_mode)
        mode_str = "NPT" if new_mode else "standard (with attention residual)"
        print(f"{Fore.GREEN}✓ Layer {layer_idx} set to {mode_str} mode{Style.RESET_ALL}")
    
    def get_layer_modes(self) -> Dict[int, bool]:
        """Get current NPT mode status for all available layers."""
        modes = {}
        for layer_idx in self.available_layers:
            layer = self.model.model.layers[layer_idx]
            if hasattr(layer, 'use_npt'):
                modes[layer_idx] = layer.use_npt
            else:
                modes[layer_idx] = True  # Default assumption
        return modes
    
    def save_modified_model(self, save_path: str):
        """Save the model with injected knowledge."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save NPT weights
        self.model.save_npt_weights(save_path / "npt_weights_modified.pt")
        
        # Save injection history with all layers
        injection_data = {
            "injected_facts": self.injected_facts,
            "active_layer": self.active_layer_idx,
            "available_layers": self.available_layers,
            "model_info": {
                "total_layers": len(self.model.model.layers),
                "npt_layers": self.available_layers,
                "npt_ranks": {
                    idx: self.model.npt_layers[idx].np_component.rank
                    for idx in self.available_layers
                },
                "npt_num_ranks": {
                    idx: self.model.npt_layers[idx].np_component.num_ranks
                    for idx in self.available_layers
                }
            }
        }
        
        with open(save_path / "injection_history.json", "w") as f:
            json.dump(injection_data, f, indent=2)
        
        print(f"{Fore.GREEN}✓ Modified model saved to {save_path}{Style.RESET_ALL}")
        
        # Print summary
        total_facts = sum(len(facts) for facts in self.injected_facts.values())
        print(f"  - Total injected facts: {total_facts}")
        print(f"  - Modified layers: {list(self.injected_facts.keys())}")


class InteractiveSession:
    """Interactive session for knowledge injection experiments."""
    
    def __init__(self, injector: KnowledgeInjector):
        self.injector = injector
        self.session_history = []
    
    def print_help(self):
        """Print help information."""
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}NPT Knowledge Injection - Interactive Commands:{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}ask <question>{Style.RESET_ALL} - Ask the model a question")
        print(f"{Fore.GREEN}inject <fact>{Style.RESET_ALL} - Inject a fact into the current layer")
        print(f"{Fore.GREEN}inject-all <fact>{Style.RESET_ALL} - Inject a fact into all NPT layers")
        print(f"{Fore.GREEN}inject-multi{Style.RESET_ALL} - Inject multiple related facts")
        print(f"{Fore.GREEN}test <question>{Style.RESET_ALL} - Test if injected knowledge works")
        print(f"{Fore.GREEN}layers{Style.RESET_ALL} - List all available NPT layers and their modes")
        print(f"{Fore.GREEN}layer <idx>{Style.RESET_ALL} - Switch to a specific NPT layer")
        print(f"{Fore.GREEN}mode <idx> [npt/standard]{Style.RESET_ALL} - Toggle or set layer mode")
        print(f"{Fore.GREEN}modes{Style.RESET_ALL} - Show current mode for all NPT layers")
        print(f"{Fore.GREEN}reset{Style.RESET_ALL} - Reset current layer weights to original")
        print(f"{Fore.GREEN}reset-all{Style.RESET_ALL} - Reset all layers to original state")
        print(f"{Fore.GREEN}save <path>{Style.RESET_ALL} - Save modified model")
        print(f"{Fore.GREEN}history{Style.RESET_ALL} - Show injection history for all layers")
        print(f"{Fore.GREEN}strength <value>{Style.RESET_ALL} - Set injection strength (default: 1.0)")
        print(f"{Fore.GREEN}help{Style.RESET_ALL} - Show this help message")
        print(f"{Fore.GREEN}exit{Style.RESET_ALL} - Exit the session")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
    
    def run(self):
        """Run the interactive session."""
        self.print_help()
        
        while True:
            try:
                # Get user input
                user_input = input(f"\n{Fore.YELLOW}NPT> {Style.RESET_ALL}").strip()
                
                if not user_input:
                    continue
                
                # Parse command
                parts = user_input.split(None, 1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""
                
                # Handle commands
                if command == "exit":
                    print(f"{Fore.CYAN}Goodbye!{Style.RESET_ALL}")
                    break
                
                elif command == "help":
                    self.print_help()
                
                elif command == "ask":
                    if not args:
                        print(f"{Fore.RED}Please provide a question.{Style.RESET_ALL}")
                        continue
                    response = self.injector.generate_response(args)
                    print(f"\n{Fore.CYAN}Model:{Style.RESET_ALL} {response}")
                    self.session_history.append(("ask", args, response))
                
                elif command == "inject":
                    if not args:
                        print(f"{Fore.RED}Please provide a fact to inject.{Style.RESET_ALL}")
                        continue
                    
                    # Ask for injection parameters
                    position = input(f"Position (last/first/all) [{Fore.GREEN}last{Style.RESET_ALL}]: ").strip() or "last"
                    accumulate = input(f"Accumulate with previous injections? (y/n) [{Fore.GREEN}n{Style.RESET_ALL}]: ").strip().lower() == 'y'
                    
                    info = self.injector.inject_knowledge(args, position=position, accumulate=accumulate)
                    self.session_history.append(("inject", args, info))
                
                elif command == "inject-multi":
                    print(f"{Fore.YELLOW}Enter facts to inject (empty line to finish):{Style.RESET_ALL}")
                    facts = []
                    while True:
                        fact = input(f"  Fact {len(facts)+1}: ").strip()
                        if not fact:
                            break
                        facts.append(fact)
                    
                    if facts:
                        position = input(f"Position (last/first/all) [{Fore.GREEN}last{Style.RESET_ALL}]: ").strip() or "last"
                        for i, fact in enumerate(facts):
                            print(f"\n{Fore.YELLOW}Injecting fact {i+1}/{len(facts)}...{Style.RESET_ALL}")
                            info = self.injector.inject_knowledge(fact, position=position, accumulate=True)
                            self.session_history.append(("inject", fact, info))
                
                elif command == "test":
                    if not args:
                        print(f"{Fore.RED}Please provide a test question.{Style.RESET_ALL}")
                        continue
                    
                    print(f"\n{Fore.YELLOW}Testing injected knowledge...{Style.RESET_ALL}")
                    response = self.injector.generate_response(args)
                    print(f"\n{Fore.CYAN}Model:{Style.RESET_ALL} {response}")
                    
                    # Compare with original if available
                    if self.injector.active_layer_idx in self.injector.original_weights:
                        print(f"\n{Fore.YELLOW}Comparing with original model (layer {self.injector.active_layer_idx})...{Style.RESET_ALL}")
                        # Temporarily reset
                        npt_layer = self.injector.model.npt_layers[self.injector.active_layer_idx]
                        weights = self.injector.original_weights[self.injector.active_layer_idx]

                        # Save current weights based on type
                        if isinstance(weights, dict):
                            # Dual modulation
                            current_gate = npt_layer.mlp.gate_proj.weight.data.clone()
                            current_up = npt_layer.mlp.up_proj.weight.data.clone()
                            if 'gate' in weights:
                                npt_layer.mlp.gate_proj.weight.data = weights['gate'].clone()
                            if 'up' in weights:
                                npt_layer.mlp.up_proj.weight.data = weights['up'].clone()
                        else:
                            # Single modulation
                            current_weights = npt_layer.mlp.gate_proj.weight.data.clone()
                            npt_layer.mlp.gate_proj.weight.data = weights.clone()

                        original_response = self.injector.generate_response(args)
                        print(f"{Fore.CYAN}Original:{Style.RESET_ALL} {original_response}")

                        # Restore modified weights
                        if isinstance(weights, dict):
                            npt_layer.mlp.gate_proj.weight.data = current_gate
                            if hasattr(npt_layer.mlp, 'up_proj'):
                                npt_layer.mlp.up_proj.weight.data = current_up
                        else:
                            npt_layer.mlp.gate_proj.weight.data = current_weights
                    
                    self.session_history.append(("test", args, response))
                
                elif command == "reset":
                    self.injector.reset_weights()
                
                elif command == "save":
                    save_path = args or f"experiments/injected_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    self.injector.save_modified_model(save_path)
                
                elif command == "layers":
                    # Show available NPT layers and their modes
                    info = self.injector.get_layer_info()
                    modes = self.injector.get_layer_modes()
                    print(f"\n{Fore.CYAN}NPT Layers Information:{Style.RESET_ALL}")
                    print(f"  Available layers: {info['available_layers']}")
                    print(f"  Active layer: {Fore.GREEN}{info['active_layer']}{Style.RESET_ALL}")
                    print(f"\n  Layer details:")
                    for idx in info['available_layers']:
                        rank_key = f"layer_{idx}_rank"
                        num_ranks_key = f"layer_{idx}_num_ranks"
                        rank = info.get(rank_key, "unknown")
                        num_ranks = info.get(num_ranks_key, 1)
                        facts_count = info['injected_facts_count'].get(idx, 0)
                        active_marker = " (active)" if idx == info['active_layer'] else ""
                        mode = "NPT" if modes.get(idx, True) else "standard"
                        mode_color = Fore.GREEN if modes.get(idx, True) else Fore.YELLOW
                        rank_info = f"rank={rank}" if num_ranks == 1 else f"rank={rank}×{num_ranks}"
                        print(f"    Layer {idx}: {rank_info}, mode={mode_color}{mode}{Style.RESET_ALL}, injected_facts={facts_count}{active_marker}")
                
                elif command == "layer":
                    # Switch to a specific layer
                    if not args:
                        print(f"{Fore.RED}Please provide a layer index.{Style.RESET_ALL}")
                        continue
                    try:
                        layer_idx = int(args)
                        self.injector.switch_layer(layer_idx)
                    except ValueError as e:
                        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
                    except Exception as e:
                        print(f"{Fore.RED}Invalid layer index: {args}{Style.RESET_ALL}")
                
                elif command == "inject-all":
                    # Inject fact into all NPT layers
                    if not args:
                        print(f"{Fore.RED}Please provide a fact to inject.{Style.RESET_ALL}")
                        continue
                    
                    position = input(f"Position (last/first/all) [{Fore.GREEN}last{Style.RESET_ALL}]: ").strip() or "last"
                    
                    results = self.injector.inject_knowledge_all_layers(args, position=position)
                    self.session_history.append(("inject-all", args, results))
                
                elif command == "reset-all":
                    # Reset all layers
                    confirm = input(f"Reset ALL layers to original? (y/n) [{Fore.RED}n{Style.RESET_ALL}]: ").strip().lower()
                    if confirm == 'y':
                        self.injector.reset_all_weights()
                
                elif command == "history":
                    # Show injection history for all layers
                    if self.injector.injected_facts:
                        print(f"\n{Fore.CYAN}Injection History:{Style.RESET_ALL}")
                        for layer_idx, facts in self.injector.injected_facts.items():
                            if facts:
                                print(f"\n{Fore.YELLOW}Layer {layer_idx}:{Style.RESET_ALL}")
                                for i, fact_info in enumerate(facts, 1):
                                    print(f"  {i}. {fact_info['fact']}")
                                    ranks_str = f", Ranks: {fact_info.get('num_ranks', 1)}" if fact_info.get('is_rank_k', False) else ""
                                    print(f"     Alpha: {fact_info['alpha']:.2f}, Position: {fact_info['position']}{ranks_str}")
                                    print(f"     Weight change: {fact_info['weight_change_ratio']:.6f}")
                    else:
                        print(f"{Fore.YELLOW}No facts injected yet.{Style.RESET_ALL}")
                
                elif command == "mode":
                    # Toggle or set layer mode (NPT vs standard)
                    parts = args.split()
                    if not parts:
                        print(f"{Fore.RED}Please provide a layer index. Usage: mode <idx> [npt/standard]{Style.RESET_ALL}")
                        continue
                    
                    try:
                        layer_idx = int(parts[0])
                        if len(parts) > 1:
                            mode_str = parts[1].lower()
                            if mode_str == "npt":
                                self.injector.toggle_layer_mode(layer_idx, use_npt=True)
                            elif mode_str == "standard":
                                self.injector.toggle_layer_mode(layer_idx, use_npt=False)
                            else:
                                print(f"{Fore.RED}Invalid mode. Use 'npt' or 'standard'.{Style.RESET_ALL}")
                        else:
                            # Toggle current mode
                            self.injector.toggle_layer_mode(layer_idx)
                    except ValueError:
                        print(f"{Fore.RED}Invalid layer index: {parts[0]}{Style.RESET_ALL}")
                
                elif command == "modes":
                    # Show current modes for all layers
                    modes = self.injector.get_layer_modes()
                    print(f"\n{Fore.CYAN}Current Layer Modes:{Style.RESET_ALL}")
                    npt_layers = [idx for idx, use_npt in modes.items() if use_npt]
                    standard_layers = [idx for idx, use_npt in modes.items() if not use_npt]
                    
                    if npt_layers:
                        print(f"  {Fore.GREEN}NPT mode:{Style.RESET_ALL} {npt_layers}")
                    if standard_layers:
                        print(f"  {Fore.YELLOW}Standard mode:{Style.RESET_ALL} {standard_layers}")
                    
                    if not npt_layers and not standard_layers:
                        print(f"  {Fore.YELLOW}No NPT layers available{Style.RESET_ALL}")
                
                elif command == "strength":
                    try:
                        strength = float(args)
                        self.injector.injection_strength = strength
                        print(f"{Fore.GREEN}Injection strength set to {strength}{Style.RESET_ALL}")
                    except ValueError:
                        print(f"{Fore.RED}Invalid strength value. Please provide a number.{Style.RESET_ALL}")
                
                else:
                    print(f"{Fore.RED}Unknown command: {command}. Type 'help' for available commands.{Style.RESET_ALL}")
            
            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}Use 'exit' to quit.{Style.RESET_ALL}")
            except EOFError:
                print(f"\n{Fore.CYAN}EOF detected. Exiting...{Style.RESET_ALL}")
                break
            except Exception as e:
                print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
                logger.exception("Error in interactive session")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive knowledge injection experiment for NPT"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to NPT checkpoint directory"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Base model name if no checkpoint provided"
    )
    parser.add_argument(
        "--layer_idx",
        type=int,
        default=15,
        help="NPT layer index to use for injection"
    )
    parser.add_argument(
        "--use_npt_layers",
        type=str,
        default=None,
        help="Comma-separated list of layer indices to use as NPT (e.g., '15,31' or 'all'). "
             "If not specified, all available NPT layers will be used. "
             "Use 'none' to load weights but keep all layers as standard transformers."
    )
    parser.add_argument(
        "--injection_strength",
        type=float,
        default=1.0,
        help="Initial injection strength (scaling factor)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--demo_mode",
        action="store_true",
        help="Use small demo model for testing"
    )
    
    args = parser.parse_args()
    
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}NPT Knowledge Injection Experiment{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    
    # Load or create model
    if args.checkpoint:
        # Load from checkpoint
        checkpoint_path = Path(args.checkpoint)
        print(f"\n{Fore.YELLOW}Loading checkpoint from {checkpoint_path}...{Style.RESET_ALL}")
        
        # Load base model
        model = NPTLlamaModel.from_pretrained(args.model_name)
        
        # Check for different checkpoint formats
        npt_weights_path = checkpoint_path / "npt_weights.pt"
        accumulated_weights_path = checkpoint_path / "accumulated_npt_weights.pt"
        
        # Determine which weights file to use
        if accumulated_weights_path.exists():
            # Sequential training checkpoint with multiple layers
            weights_path = accumulated_weights_path
            print(f"  Found accumulated weights from sequential training")
        elif npt_weights_path.exists():
            # Single checkpoint
            weights_path = npt_weights_path
            print(f"  Found NPT weights")
        else:
            raise FileNotFoundError(f"No NPT weights found in {checkpoint_path}")
        
        # Load weights and detect NPT layers
        npt_weights = torch.load(weights_path, map_location='cpu')
        layer_info = detect_npt_layers_from_weights(npt_weights)
        
        if layer_info:
            available_layers = sorted(layer_info.keys())
            print(f"  Detected NPT weights for layers: {available_layers}")
            
            # Show rank, num_ranks, and dual modulation information
            for layer_idx, info in layer_info.items():
                if len(info) == 3:
                    rank, num_ranks, has_dual = info
                else:
                    # Backward compatibility
                    rank, num_ranks = info
                    has_dual = False

                rank_str = f"rank={rank}" if num_ranks == 1 else f"rank={rank}×{num_ranks}"
                mod_str = " (dual modulation)" if has_dual else ""
                print(f"    Layer {layer_idx}: {rank_str}{mod_str}")
            
            # Determine which layers to actually use as NPT
            if args.use_npt_layers is None:
                # Default: use all available NPT layers
                layers_to_use = available_layers
                print(f"  Using all available NPT layers (default)")
            elif args.use_npt_layers.lower() == 'all':
                # Explicitly use all
                layers_to_use = available_layers
                print(f"  Using all available NPT layers")
            elif args.use_npt_layers.lower() == 'none':
                # Load weights but don't use NPT mode for any layer
                layers_to_use = []
                print(f"  {Fore.YELLOW}Loading NPT weights but keeping all layers as standard transformers{Style.RESET_ALL}")
            else:
                # Parse specific layer indices
                try:
                    requested_layers = [int(x.strip()) for x in args.use_npt_layers.split(',')]
                    # Filter to only layers that have weights available
                    layers_to_use = [l for l in requested_layers if l in available_layers]
                    if len(layers_to_use) < len(requested_layers):
                        missing = [l for l in requested_layers if l not in available_layers]
                        print(f"  {Fore.YELLOW}Warning: Requested layers {missing} don't have NPT weights{Style.RESET_ALL}")
                    print(f"  Using NPT mode for layers: {layers_to_use}")
                except ValueError:
                    print(f"  {Fore.RED}Invalid --use_npt_layers format. Using all available layers.{Style.RESET_ALL}")
                    layers_to_use = available_layers
            
            # Show which layers will be in which mode
            if layers_to_use:
                standard_layers = [l for l in available_layers if l not in layers_to_use]
                if standard_layers:
                    print(f"  {Fore.CYAN}Layers {standard_layers} will operate in standard mode (with attention residual){Style.RESET_ALL}")
            else:
                print(f"  {Fore.CYAN}All layers will operate in standard mode (with attention residual){Style.RESET_ALL}")
            
            # Convert only the selected layers to NPT  
            if layers_to_use:
                # Group layers by (rank, num_ranks) for efficient conversion
                layers_by_config = {}
                for layer_idx in layers_to_use:
                    info = layer_info[layer_idx]
                    if len(info) == 3:
                        rank, num_ranks, _ = info  # Ignore dual modulation flag for grouping
                    else:
                        rank, num_ranks = info
                    config_key = (rank, num_ranks)
                    if config_key not in layers_by_config:
                        layers_by_config[config_key] = []
                    layers_by_config[config_key].append(layer_idx)
                
                if len(layers_by_config) == 1:
                    # All selected layers have the same configuration - convert together
                    (detected_rank, detected_num_ranks) = list(layers_by_config.keys())[0]

                    # Check if any layer has dual modulation
                    has_dual = any(layer_info.get(idx, (0, 0, False))[2] if len(layer_info.get(idx, (0, 0))) > 2 else False
                                  for idx in layers_to_use)

                    npt_config = NPTConfig(
                        layers_to_convert=layers_to_use,
                        np_rank=detected_rank,
                        np_init_scale=0.001,
                        num_ranks=detected_num_ranks,
                        single_layer_mode=False,  # Don't use single_layer_mode when loading from checkpoint
                        dual_modulation=has_dual  # Enable dual modulation if detected
                    )
                    model.convert_to_npt(npt_config)
                    rank_str = f"rank={detected_rank}" if detected_num_ranks == 1 else f"rank={detected_rank}×{detected_num_ranks}"
                    print(f"  Converted {len(layers_to_use)} layers to NPT mode with {rank_str}")
                else:
                    # Different configurations for different groups - convert by config groups
                    print(f"  {Fore.YELLOW}Detected different configurations, converting by groups:{Style.RESET_ALL}")
                    for (rank, num_ranks), layer_indices in sorted(layers_by_config.items()):
                        # Check if any layer in this group has dual modulation
                        has_dual = any(layer_info.get(idx, (0, 0, False))[2] if len(layer_info.get(idx, (0, 0))) > 2 else False
                                      for idx in layer_indices)

                        npt_config = NPTConfig(
                            layers_to_convert=layer_indices,
                            np_rank=rank,
                            np_init_scale=0.001,
                            num_ranks=num_ranks,
                            single_layer_mode=False,  # Don't use single_layer_mode when loading from checkpoint
                            dual_modulation=has_dual  # Enable dual modulation if detected
                        )
                        model.convert_to_npt(npt_config)
                        rank_str = f"rank={rank}" if num_ranks == 1 else f"rank={rank}×{num_ranks}"
                        print(f"    Converted layers {layer_indices} with {rank_str}")
                    print(f"  Total: Converted {len(layers_to_use)} layers to NPT mode")
                
                # Load NPT weights for all available layers (even those not in NPT mode)
                # This allows switching modes later if desired
                model.load_npt_weights(npt_weights)
                
                # For layers not in NPT mode but with weights available, 
                # ensure they operate in standard mode
                for layer_idx in available_layers:
                    if layer_idx in model.model.layers and hasattr(model.model.layers[layer_idx], 'set_npt_mode'):
                        if layer_idx not in layers_to_use:
                            model.model.layers[layer_idx].set_npt_mode(False)
                            print(f"    Set layer {layer_idx} to standard mode")
            else:
                # No layers in NPT mode, but still might want to have the architecture
                # This is useful for comparison experiments
                print(f"  {Fore.YELLOW}No layers will use NPT mode (standard transformer behavior){Style.RESET_ALL}")
                # Don't convert any layers, model remains standard
        else:
            # Fallback to single layer if no layers detected
            print(f"  No NPT layers detected in checkpoint, converting layer {args.layer_idx}")
            npt_config = NPTConfig(
                layers_to_convert=[args.layer_idx],
                np_rank=256,
                np_init_scale=0.001,
                single_layer_mode=False  # Don't use single_layer_mode when loading from checkpoint
            )
            model.convert_to_npt(npt_config)
            # Load any available weights
            model.load_npt_weights(npt_weights)
        
        print(f"{Fore.GREEN}✓ Checkpoint loaded successfully!{Style.RESET_ALL}")
    
    else:
        # Create new model
        print(f"\n{Fore.YELLOW}Initializing model: {args.model_name}...{Style.RESET_ALL}")
        
        if args.demo_mode:
            from transformers import LlamaConfig
            config = LlamaConfig(
                hidden_size=256,
                intermediate_size=1024,
                num_hidden_layers=4,
                num_attention_heads=8,
                num_key_value_heads=4,
                vocab_size=128256,
            )
            model = NPTLlamaModel(config)
            args.layer_idx = min(args.layer_idx, 3)  # Adjust for demo model
        else:
            model = NPTLlamaModel.from_pretrained(args.model_name)
        
        # Convert specified layer to NPT
        npt_config = NPTConfig(
            layers_to_convert=[args.layer_idx],
            np_rank=256 if not args.demo_mode else 32,
            np_init_scale=0.001
        )
        model.convert_to_npt(npt_config)
        print(f"{Fore.GREEN}✓ Model initialized with NPT layer {args.layer_idx}{Style.RESET_ALL}")
    
    # Load tokenizer
    print(f"\n{Fore.YELLOW}Loading tokenizer...{Style.RESET_ALL}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    except:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create injector
    injector = KnowledgeInjector(
        model=model,
        tokenizer=tokenizer,
        layer_idx=args.layer_idx,
        injection_strength=args.injection_strength,
        device=args.device
    )
    
    # Print model info
    param_counts = model.count_parameters()
    print(f"\n{Fore.CYAN}Model Information:{Style.RESET_ALL}")
    print(f"  Total parameters: {param_counts['total']:,}")
    print(f"  NPT parameters: {param_counts['npt']:,}")
    
    # Show NPT layers info
    if hasattr(model, 'npt_layers') and model.npt_layers:
        npt_layer_indices = sorted(model.npt_layers.keys())
        print(f"  NPT layers: {npt_layer_indices}")
        if len(npt_layer_indices) == 1:
            print(f"  Active layer: {npt_layer_indices[0]}")
        else:
            # Default to highest layer or user-specified
            default_layer = args.layer_idx if args.layer_idx in npt_layer_indices else npt_layer_indices[-1]
            print(f"  Active layer: {default_layer} (use 'layer <idx>' to switch)")
    else:
        print(f"  NPT layer: {args.layer_idx}")
    
    print(f"  Device: {args.device}")
    
    # Run interactive session
    session = InteractiveSession(injector)
    session.run()


if __name__ == "__main__":
    main()