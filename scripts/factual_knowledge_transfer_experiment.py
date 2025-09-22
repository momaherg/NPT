#!/usr/bin/env python3
"""
Factual Knowledge Transfer Experiment via NPT Modulation Swapping

This experiment tests whether modulations extracted from one factual statement
can transfer knowledge when injected into another context.

Core hypothesis: Modulations encode factual associations that can be transferred
between contexts to alter model predictions.
"""

import argparse
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.npt import NPTLlamaModel, NPTConfig
from transformers import AutoTokenizer, AutoConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModulationData:
    """Container for extracted modulation data."""
    layer_idx: int
    token_position: int
    v_a_gate: Optional[torch.Tensor] = None
    v_b_gate: Optional[torch.Tensor] = None
    v_a_up: Optional[torch.Tensor] = None
    v_b_up: Optional[torch.Tensor] = None
    v_a: Optional[torch.Tensor] = None  # For single modulation
    v_b: Optional[torch.Tensor] = None  # For single modulation
    attention_output: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None


@dataclass
class ExperimentConfig:
    """Configuration for the experiment."""
    checkpoint_path: str
    model_name: str = "meta-llama/Llama-3.2-1B"
    npt_layers: List[int] = None  # Which layers to test
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    injection_modes: List[str] = None  # ["replace", "blend", "add"]
    blend_alphas: List[float] = None  # [0.25, 0.5, 0.75, 1.0]
    output_dir: str = "experiments/factual_transfer_results"


class NPTModelLoader:
    """Handles loading and configuring NPT models."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device(config.device)
        
    def load_model(self) -> Tuple[NPTLlamaModel, AutoTokenizer]:
        """Load NPT model with proper configuration."""
        logger.info(f"Loading model from {self.config.checkpoint_path}")
        
        # Load base model configuration
        model_config = AutoConfig.from_pretrained(self.config.model_name)
        model_config._attn_implementation = "eager"
        
        # Create NPT model
        model = NPTLlamaModel.from_pretrained(
            self.config.model_name,
            config=model_config
        )
        
        # Load NPT checkpoint if provided
        if self.config.checkpoint_path:
            checkpoint_path = Path(self.config.checkpoint_path)
            
            # Check for NPT weights file
            npt_weights_path = checkpoint_path / "npt_weights.pt"
            if not npt_weights_path.exists():
                # Try parent directory
                npt_weights_path = checkpoint_path.parent / "npt_weights.pt"
                
            if npt_weights_path.exists():
                logger.info(f"Loading NPT weights from {npt_weights_path}")
                
                # First convert model to NPT
                # Determine which layers to convert based on checkpoint
                state_dict = torch.load(npt_weights_path, map_location='cpu')
                
                # Extract layer indices from state dict keys
                npt_layer_indices = set()
                for key in state_dict.keys():
                    if 'model.layers' in key and 'np_component' in key:
                        # Extract layer index from key like 'model.layers.15.np_component.W_down'
                        parts = key.split('.')
                        if 'layers' in parts:
                            idx = parts.index('layers')
                            if idx + 1 < len(parts) and parts[idx + 1].isdigit():
                                npt_layer_indices.add(int(parts[idx + 1]))
                
                if npt_layer_indices:
                    logger.info(f"Found NPT layers in checkpoint: {sorted(npt_layer_indices)}")
                    
                    # Detect if dual modulation from weight keys
                    dual_modulation = any('W_down_gate' in k for k in state_dict.keys())
                    
                    # Create NPT config
                    npt_config = NPTConfig(
                        layers_to_convert=sorted(npt_layer_indices),
                        np_rank=256,  # Will be overridden by checkpoint
                        np_init_scale=0.001,
                        single_layer_mode=False,  # Important: always False for loading
                        num_ranks=4,  # Will be determined from checkpoint
                        init_strategy="improved",
                        dual_modulation=dual_modulation
                    )
                    
                    # Convert model to NPT
                    model.convert_to_npt(npt_config)
                    
                    # Load the weights
                    model.load_npt_weights(npt_weights_path)
                    
                    # Store which layers are NPT for experiment
                    if self.config.npt_layers is None:
                        self.config.npt_layers = sorted(npt_layer_indices)
                else:
                    logger.warning("No NPT layers found in checkpoint")
            else:
                logger.warning(f"NPT weights not found at {npt_weights_path}")
        
        # Move model to device
        model = model.to(self.device)
        model.eval()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        logger.info(f"Model loaded successfully. NPT layers: {self.config.npt_layers}")
        
        return model, tokenizer


class ModulationExtractor:
    """Extract modulations from NPT layers at specific positions."""
    
    def __init__(self, model: NPTLlamaModel, tokenizer: AutoTokenizer, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def extract_at_position(
        self,
        prompt: str,
        position_idx: int,
        layers_to_extract: List[int]
    ) -> Dict[int, ModulationData]:
        """
        Extract modulations when generating token at position_idx.
        
        Args:
            prompt: Input prompt
            position_idx: Position where to extract modulation (-1 for last)
            layers_to_extract: Which NPT layers to extract from
            
        Returns:
            Dict mapping layer_idx to ModulationData
        """
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs['input_ids']
        
        # If position_idx is -1, use last position
        if position_idx == -1:
            position_idx = input_ids.shape[1] - 1
            
        logger.info(f"Extracting modulation at position {position_idx} for prompt: '{prompt}'")
        
        modulations = {}
        
        # Hook to capture modulations
        def create_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple) and len(output) == 2:
                    # Check if dual modulation
                    if isinstance(output[0], tuple):
                        # Dual modulation
                        (v_a_gate, v_b_gate), (v_a_up, v_b_up) = output
                        modulations[layer_idx] = ModulationData(
                            layer_idx=layer_idx,
                            token_position=position_idx,
                            v_a_gate=v_a_gate[:, position_idx:position_idx+1].clone(),
                            v_b_gate=v_b_gate[:, position_idx:position_idx+1].clone(),
                            v_a_up=v_a_up[:, position_idx:position_idx+1].clone(),
                            v_b_up=v_b_up[:, position_idx:position_idx+1].clone(),
                            attention_output=input[0][:, position_idx:position_idx+1].clone()
                        )
                    else:
                        # Single modulation
                        v_a, v_b = output
                        modulations[layer_idx] = ModulationData(
                            layer_idx=layer_idx,
                            token_position=position_idx,
                            v_a=v_a[:, position_idx:position_idx+1].clone(),
                            v_b=v_b[:, position_idx:position_idx+1].clone(),
                            attention_output=input[0][:, position_idx:position_idx+1].clone()
                        )
            return hook
        
        # Register hooks
        handles = []
        for layer_idx in layers_to_extract:
            if layer_idx < len(self.model.model.layers):
                layer = self.model.model.layers[layer_idx]
                if hasattr(layer, 'np_component'):
                    handle = layer.np_component.register_forward_hook(create_hook(layer_idx))
                    handles.append(handle)
        
        # Run forward pass
        with torch.no_grad():
            self.model.set_npt_mode(True)
            outputs = self.model(input_ids)
            
        # Remove hooks
        for handle in handles:
            handle.remove()
            
        return modulations


class ModulationInjector:
    """Inject modulations at specific positions during generation."""
    
    def __init__(self, model: NPTLlamaModel, tokenizer: AutoTokenizer, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def generate_with_injection(
        self,
        prompt: str,
        injection_position: int,
        source_modulations: Dict[int, ModulationData],
        injection_mode: str = "replace",
        blend_alpha: float = 1.0,
        max_new_tokens: int = 10
    ) -> Dict[str, Any]:
        """
        Generate text with modulation injection at specific position.
        
        Args:
            prompt: Target prompt
            injection_position: Where to inject (-1 for last)
            source_modulations: Modulations to inject
            injection_mode: "replace", "blend", or "add"
            blend_alpha: Blending factor for "blend" mode
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Dict with generated text, token probabilities, and metrics
        """
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs['input_ids']
        
        if injection_position == -1:
            injection_position = input_ids.shape[1]
            
        logger.info(f"Injecting modulation at position {injection_position} for prompt: '{prompt}'")
        
        # Storage for original modulations to compute metrics
        original_modulations = {}
        injected_modulations = {}
        
        # Hook to inject modulations
        def create_injection_hook(layer_idx, source_mod):
            def hook(module, input, output):
                # Only inject at the target position during generation
                current_pos = input[0].shape[1] - 1
                
                if current_pos == injection_position:
                    logger.debug(f"Injecting at layer {layer_idx}, position {current_pos}")
                    
                    if isinstance(output, tuple) and len(output) == 2:
                        if isinstance(output[0], tuple):
                            # Dual modulation
                            (v_a_gate, v_b_gate), (v_a_up, v_b_up) = output
                            
                            # Store original
                            original_modulations[layer_idx] = {
                                'v_a_gate': v_a_gate[:, -1:].clone(),
                                'v_b_gate': v_b_gate[:, -1:].clone(),
                                'v_a_up': v_a_up[:, -1:].clone(),
                                'v_b_up': v_b_up[:, -1:].clone()
                            }
                            
                            # Prepare injection
                            if injection_mode == "replace":
                                new_v_a_gate = source_mod.v_a_gate
                                new_v_b_gate = source_mod.v_b_gate
                                new_v_a_up = source_mod.v_a_up
                                new_v_b_up = source_mod.v_b_up
                            elif injection_mode == "blend":
                                new_v_a_gate = blend_alpha * source_mod.v_a_gate + (1 - blend_alpha) * v_a_gate[:, -1:]
                                new_v_b_gate = blend_alpha * source_mod.v_b_gate + (1 - blend_alpha) * v_b_gate[:, -1:]
                                new_v_a_up = blend_alpha * source_mod.v_a_up + (1 - blend_alpha) * v_a_up[:, -1:]
                                new_v_b_up = blend_alpha * source_mod.v_b_up + (1 - blend_alpha) * v_b_up[:, -1:]
                            elif injection_mode == "add":
                                new_v_a_gate = v_a_gate[:, -1:] + blend_alpha * source_mod.v_a_gate
                                new_v_b_gate = v_b_gate[:, -1:] + blend_alpha * source_mod.v_b_gate
                                new_v_a_up = v_a_up[:, -1:] + blend_alpha * source_mod.v_a_up
                                new_v_b_up = v_b_up[:, -1:] + blend_alpha * source_mod.v_b_up
                            
                            # Store injected
                            injected_modulations[layer_idx] = {
                                'v_a_gate': new_v_a_gate.clone(),
                                'v_b_gate': new_v_b_gate.clone(),
                                'v_a_up': new_v_a_up.clone(),
                                'v_b_up': new_v_b_up.clone()
                            }
                            
                            # Replace last position with injected modulation
                            v_a_gate_new = v_a_gate.clone()
                            v_b_gate_new = v_b_gate.clone()
                            v_a_up_new = v_a_up.clone()
                            v_b_up_new = v_b_up.clone()
                            
                            v_a_gate_new[:, -1:] = new_v_a_gate
                            v_b_gate_new[:, -1:] = new_v_b_gate
                            v_a_up_new[:, -1:] = new_v_a_up
                            v_b_up_new[:, -1:] = new_v_b_up
                            
                            return (v_a_gate_new, v_b_gate_new), (v_a_up_new, v_b_up_new)
                        else:
                            # Single modulation
                            v_a, v_b = output
                            
                            # Store original
                            original_modulations[layer_idx] = {
                                'v_a': v_a[:, -1:].clone(),
                                'v_b': v_b[:, -1:].clone()
                            }
                            
                            # Prepare injection
                            if injection_mode == "replace":
                                new_v_a = source_mod.v_a
                                new_v_b = source_mod.v_b
                            elif injection_mode == "blend":
                                new_v_a = blend_alpha * source_mod.v_a + (1 - blend_alpha) * v_a[:, -1:]
                                new_v_b = blend_alpha * source_mod.v_b + (1 - blend_alpha) * v_b[:, -1:]
                            elif injection_mode == "add":
                                new_v_a = v_a[:, -1:] + blend_alpha * source_mod.v_a
                                new_v_b = v_b[:, -1:] + blend_alpha * source_mod.v_b
                            
                            # Store injected
                            injected_modulations[layer_idx] = {
                                'v_a': new_v_a.clone(),
                                'v_b': new_v_b.clone()
                            }
                            
                            # Replace last position
                            v_a_new = v_a.clone()
                            v_b_new = v_b.clone()
                            v_a_new[:, -1:] = new_v_a
                            v_b_new[:, -1:] = new_v_b
                            
                            return v_a_new, v_b_new
                
                return output
            return hook
        
        # Register injection hooks
        handles = []
        for layer_idx, source_mod in source_modulations.items():
            if layer_idx < len(self.model.model.layers):
                layer = self.model.model.layers[layer_idx]
                if hasattr(layer, 'np_component'):
                    handle = layer.np_component.register_forward_hook(
                        create_injection_hook(layer_idx, source_mod)
                    )
                    handles.append(handle)
        
        # Generate with injection
        with torch.no_grad():
            self.model.set_npt_mode(True)
            
            # Get logits at injection position
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
                do_sample=False,
                temperature=1.0
            )
            
        # Remove hooks
        for handle in handles:
            handle.remove()
            
        # Extract results
        generated_ids = outputs.sequences[0]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Get probability distribution at injection position
        if outputs.scores:
            logits_at_injection = outputs.scores[0][0]  # First new token
            probs_at_injection = F.softmax(logits_at_injection, dim=-1)
        else:
            probs_at_injection = None
            
        return {
            'generated_text': generated_text,
            'generated_ids': generated_ids,
            'probs_at_injection': probs_at_injection,
            'original_modulations': original_modulations,
            'injected_modulations': injected_modulations
        }