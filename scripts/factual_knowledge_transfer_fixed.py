#!/usr/bin/env python3
"""
Factual Knowledge Transfer with NEW NPT Architecture (MLP(h) → attention + MLP(h+attention))

This version works with the new NPT architecture where:
- MLP receives only residual h (NOT h+attention)
- Modulation makes MLP(h) produce attention + MLP(h+attention)
- The modulation encodes BOTH attention patterns AND MLP transformations

Key insight: We're now transferring richer representations that include:
1. How attention would modify the computation (attention component)
2. How the MLP would process with that attention (MLP component)

Position logic:
- Extracts modulation from the position that GENERATES the answer (last prompt token)
- Injects at the corresponding position in the target prompt
- Both positions are correctly aligned as input_ids.shape[1] - 1
"""

import argparse
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
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
    """Container for extracted modulation data.

    In the NEW architecture, these modulations encode:
    - How to make MLP(h) produce attention + MLP(h+attention)
    - Both attention patterns and MLP transformations are captured
    """
    layer_idx: int
    token_position: int
    prompt: str
    next_token: str
    v_a_gate: Optional[torch.Tensor] = None
    v_b_gate: Optional[torch.Tensor] = None
    v_a_up: Optional[torch.Tensor] = None
    v_b_up: Optional[torch.Tensor] = None
    v_a: Optional[torch.Tensor] = None  # For single modulation
    v_b: Optional[torch.Tensor] = None  # For single modulation
    attention_output: Optional[torch.Tensor] = None
    modulation_magnitude: Optional[float] = None
    architecture_version: str = "new"  # Track architecture version


class SelectiveNPTLoader:
    """Load NPT model with only specific layers converted to NPT mode."""

    @staticmethod
    def verify_architecture_version(model: NPTLlamaModel, active_layers: Set[int]) -> str:
        """
        Verify which NPT architecture version the model is using.

        NEW architecture: MLP(h) -> attention + MLP(h+attention)
        OLD architecture: MLP(h+attention) -> MLP(h+attention)

        Returns:
            "new" or "old" or "unknown"
        """
        if not active_layers:
            return "unknown"

        # Check the first active NPT layer
        layer_idx = min(active_layers)
        if layer_idx < len(model.model.layers):
            layer = model.model.layers[layer_idx]

            # The new architecture is characterized by:
            # 1. MLP receives only residual (not h+attention)
            # 2. Modulation compensates for this by encoding attention+MLP
            # We can check if the layer has the expected structure

            # In practice, both architectures have the same structure,
            # but the training objective differs. We assume checkpoints
            # from multi-training branch use new architecture
            if hasattr(layer, 'np_component'):
                # Check for dual modulation (strong indicator of new architecture)
                np_comp = layer.np_component
                if hasattr(np_comp, 'dual_modulation') and np_comp.dual_modulation:
                    return "new"
                # Single modulation could be either, default to new for recent checkpoints
                return "new"

        return "unknown"

    @staticmethod
    def load_model_with_selective_npt(
        checkpoint_path: str,
        model_name: str,
        layers_to_use: List[int],
        device: str = "cuda"
    ) -> Tuple[NPTLlamaModel, AutoTokenizer, Set[int]]:
        """
        Load NPT model with only specified layers in NPT mode.
        
        Args:
            checkpoint_path: Path to NPT checkpoint
            model_name: Base model name
            layers_to_use: Which layers to actually load as NPT (e.g., [14, 15])
            device: Device to load model on
            
        Returns:
            model, tokenizer, available_npt_layers
        """
        logger.info(f"Loading model with selective NPT layers: {layers_to_use}")
        
        # Load base model configuration
        model_config = AutoConfig.from_pretrained(model_name)
        model_config._attn_implementation = "eager"
        
        # Create base model (no NPT conversion yet)
        model = NPTLlamaModel.from_pretrained(model_name, config=model_config)
        
        # Load checkpoint to see what's available
        checkpoint_path = Path(checkpoint_path)
        npt_weights_path = checkpoint_path / "npt_weights.pt"
        
        available_npt_layers = set()
        
        if npt_weights_path.exists():
            logger.info(f"Loading NPT weights from {npt_weights_path}")
            
            # Load state dict to detect available layers
            state_dict = torch.load(npt_weights_path, map_location='cpu')
            
            # Extract available layer indices
            for key in state_dict.keys():
                if 'layer_' in key and '_np' in key:
                    parts = key.split('_')
                    for i, part in enumerate(parts):
                        if part == 'layer' and i + 1 < len(parts) and parts[i + 1].isdigit():
                            available_npt_layers.add(int(parts[i + 1]))
            
            logger.info(f"Available NPT layers in checkpoint: {sorted(available_npt_layers)}")
            
            # Determine which layers to actually convert
            layers_to_convert = [l for l in layers_to_use if l in available_npt_layers]
            
            if not layers_to_convert:
                logger.warning(f"None of requested layers {layers_to_use} are available in checkpoint")
                logger.warning(f"Available layers: {sorted(available_npt_layers)}")
            else:
                logger.info(f"Converting only layers {layers_to_convert} to NPT mode")
                
                # Detect modulation type
                dual_modulation = any('W_down_gate' in k for k in state_dict.keys())
                
                # Create NPT config for ONLY the layers we want
                npt_config = NPTConfig(
                    layers_to_convert=layers_to_convert,  # Only convert specified layers
                    np_rank=256,
                    np_init_scale=0.001,
                    single_layer_mode=False,
                    num_ranks=4,
                    init_strategy="improved",
                    dual_modulation=dual_modulation
                )
                
                # Convert only specified layers
                model.convert_to_npt(npt_config)
                
                # Now we need to load weights selectively
                # Create a filtered state dict with only the layers we want
                filtered_state_dict = {}
                for key, value in state_dict.items():
                    # Parse layer index from key
                    if 'layer_' in key and '_np' in key:
                        parts = key.split('_')
                        for i, part in enumerate(parts):
                            if part == 'layer' and i + 1 < len(parts) and parts[i + 1].isdigit():
                                layer_idx = int(parts[i + 1])
                                if layer_idx in layers_to_convert:
                                    # Reconstruct the key for the model
                                    # Convert from checkpoint format to model format
                                    # layer_15_np.W_down_gate.0 -> model.layers.15.np_component.W_down_gate.0
                                    new_key = key.replace(f'layer_{layer_idx}_np', f'model.layers.{layer_idx}.np_component')
                                    filtered_state_dict[new_key] = value
                                break
                
                # Load the filtered weights
                if filtered_state_dict:
                    logger.info(f"Loading NPT weights for {len(layers_to_convert)} layers")
                    model.load_state_dict(filtered_state_dict, strict=False)
                    logger.info(f"Successfully loaded NPT weights for layers: {layers_to_convert}")
                else:
                    logger.warning("No weights loaded - check layer indices")
        else:
            logger.error(f"NPT weights not found at {npt_weights_path}")
        
        # Move model to device
        model = model.to(device)
        model.eval()

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Verify architecture version
        active_layers = set(layers_to_convert) if 'layers_to_convert' in locals() else set()
        arch_version = SelectiveNPTLoader.verify_architecture_version(model, active_layers)
        logger.info(f"Detected architecture version: {arch_version.upper()}")

        if arch_version == "new":
            logger.info("Using NEW architecture: MLP(h) → attention + MLP(h+attention)")
            logger.info("Modulations encode both attention patterns and MLP transformations")
        elif arch_version == "old":
            logger.warning("Using OLD architecture: MLP(h+attention) → MLP(h+attention)")
            logger.warning("Results may differ from expected behavior")

        # Return model with info about which layers are NPT
        return model, tokenizer, active_layers


class SelectiveModulationExtractor:
    """Extract modulations only from active NPT layers."""
    
    def __init__(self, model: NPTLlamaModel, tokenizer: AutoTokenizer, device: torch.device, active_npt_layers: Set[int]):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.active_npt_layers = active_npt_layers
        
    def extract_generation_modulation(
        self,
        prompt: str,
        target_token: str,
        layers_to_extract: Optional[List[int]] = None
    ) -> Dict[int, ModulationData]:
        """
        Extract modulations when generating a specific token.

        NEW Architecture: Extracts modulation that encodes how MLP(h) produces
        attention + MLP(h+attention) for the specific token generation.

        This captures both:
        1. The attention pattern for this context
        2. The MLP transformation with that attention

        Only extracts from layers that are actually in NPT mode.
        Extracts from the position that GENERATES the answer (last prompt token).
        """
        # Use only active NPT layers
        if layers_to_extract is None:
            layers_to_extract = list(self.active_npt_layers)
        else:
            layers_to_extract = [l for l in layers_to_extract if l in self.active_npt_layers]

        if not layers_to_extract:
            logger.warning("No active NPT layers to extract from")
            return {}

        # Tokenize prompt and target
        prompt_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        target_ids = self.tokenizer(target_token, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)

        # Concatenate to create the full sequence
        full_ids = torch.cat([prompt_ids, target_ids[:, :1]], dim=1)

        logger.info(f"Extracting modulation when generating '{target_token}' after '{prompt}'")
        logger.info(f"Active NPT layers: {sorted(self.active_npt_layers)}")
        logger.info(f"Extracting from layers: {layers_to_extract}")

        modulations = {}
        hook_fired = {layer: False for layer in layers_to_extract}

        # Position that GENERATES the target token (last prompt position)
        # In NEW architecture, this captures how MLP(h) at this position
        # produces attention + MLP(h+attention) to generate the next token
        target_position = prompt_ids.shape[1] - 1
        logger.info(f"Extraction position: {target_position} (last token of prompt)")
        logger.info(f"NEW Architecture: Extracting attention+MLP transformation")
        
        # Hook to capture modulations
        def create_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple) and len(output) == 2:
                    hook_fired[layer_idx] = True
                    
                    # Check if dual modulation
                    if isinstance(output[0], tuple):
                        # Dual modulation
                        (v_a_gate, v_b_gate), (v_a_up, v_b_up) = output
                        
                        extracted_data = ModulationData(
                            layer_idx=layer_idx,
                            token_position=target_position,
                            prompt=prompt,
                            next_token=target_token,
                            v_a_gate=v_a_gate[:, target_position:target_position+1].clone(),
                            v_b_gate=v_b_gate[:, target_position:target_position+1].clone(),
                            v_a_up=v_a_up[:, target_position:target_position+1].clone(),
                            v_b_up=v_b_up[:, target_position:target_position+1].clone(),
                            attention_output=input[0][:, target_position:target_position+1].clone()
                        )
                        
                        # Calculate modulation magnitude
                        # In NEW architecture, this represents the strength of
                        # the attention+MLP transformation encoding
                        magnitude = (
                            v_a_gate.norm().item() + v_b_gate.norm().item() +
                            v_a_up.norm().item() + v_b_up.norm().item()
                        ) / 4
                        extracted_data.modulation_magnitude = magnitude
                        extracted_data.architecture_version = "new"
                        
                    else:
                        # Single modulation
                        v_a, v_b = output
                        
                        extracted_data = ModulationData(
                            layer_idx=layer_idx,
                            token_position=target_position,
                            prompt=prompt,
                            next_token=target_token,
                            v_a=v_a[:, target_position:target_position+1].clone(),
                            v_b=v_b[:, target_position:target_position+1].clone(),
                            attention_output=input[0][:, target_position:target_position+1].clone()
                        )
                        
                        magnitude = (v_a.norm().item() + v_b.norm().item()) / 2
                        extracted_data.modulation_magnitude = magnitude
                    
                    modulations[layer_idx] = extracted_data
                    
            return hook
        
        # Register hooks only on active NPT layers
        handles = []
        for layer_idx in layers_to_extract:
            if layer_idx < len(self.model.model.layers):
                layer = self.model.model.layers[layer_idx]
                if hasattr(layer, 'np_component'):
                    handle = layer.np_component.register_forward_hook(create_hook(layer_idx))
                    handles.append(handle)
        
        # Run forward pass
        with torch.no_grad():
            # Only set NPT mode for active layers
            for idx in range(len(self.model.model.layers)):
                if hasattr(self.model.model.layers[idx], 'set_npt_mode'):
                    self.model.model.layers[idx].set_npt_mode(idx in self.active_npt_layers)
            
            outputs = self.model(full_ids)
            
        # Remove hooks
        for handle in handles:
            handle.remove()
            
        # Log which hooks fired
        for layer_idx, fired in hook_fired.items():
            if not fired:
                logger.warning(f"Hook did not fire for layer {layer_idx}")
            
        return modulations


class SelectiveLogitComputer:
    """Compute logits with selective NPT layer injection."""
    
    def __init__(self, model: NPTLlamaModel, tokenizer: AutoTokenizer, device: torch.device, active_npt_layers: Set[int]):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.active_npt_layers = active_npt_layers
        
    def compute_logits_with_injection(
        self,
        prompt: str,
        source_modulations: Dict[int, ModulationData],
        injection_mode: str = "replace",
        blend_alpha: float = 1.0
    ) -> Dict[str, Any]:
        """
        Compute logits with modulation injection only in active NPT layers.

        NEW Architecture: Injecting modulation transfers:
        1. The attention pattern from source context
        2. The MLP transformation associated with that attention
        This makes the model behave as if it "saw" the source context
        at the injection position.
        """
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        input_ids = inputs['input_ids']
        
        # Position where we want to inject
        injection_position = input_ids.shape[1] - 1
        
        logger.info(f"Computing logits with injection at position {injection_position}")
        logger.info(f"Injecting into layers: {list(source_modulations.keys())}")
        
        # Storage for debugging
        original_modulations = {}
        injected_modulations = {}
        hook_fired = {layer: False for layer in source_modulations.keys()}
        
        # Hook to inject modulations
        def create_injection_hook(layer_idx, source_mod):
            def hook(module, input, output):
                hook_fired[layer_idx] = True
                
                if isinstance(output, tuple) and len(output) == 2:
                    if isinstance(output[0], tuple):
                        # Dual modulation
                        (v_a_gate, v_b_gate), (v_a_up, v_b_up) = output
                        
                        # Store original
                        original_modulations[layer_idx] = {
                            'magnitude': (
                                v_a_gate[:, injection_position].norm().item() +
                                v_b_gate[:, injection_position].norm().item() +
                                v_a_up[:, injection_position].norm().item() +
                                v_b_up[:, injection_position].norm().item()
                            ) / 4
                        }
                        
                        # Create modified tensors
                        v_a_gate_new = v_a_gate.clone()
                        v_b_gate_new = v_b_gate.clone()
                        v_a_up_new = v_a_up.clone()
                        v_b_up_new = v_b_up.clone()
                        
                        # Apply injection - transferring attention+MLP transformation
                        # This makes MLP(h) at this position behave as if it has
                        # the source's attention pattern and MLP processing
                        if injection_mode == "replace":
                            v_a_gate_new[:, injection_position:injection_position+1] = source_mod.v_a_gate
                            v_b_gate_new[:, injection_position:injection_position+1] = source_mod.v_b_gate
                            v_a_up_new[:, injection_position:injection_position+1] = source_mod.v_a_up
                            v_b_up_new[:, injection_position:injection_position+1] = source_mod.v_b_up
                        elif injection_mode == "blend":
                            v_a_gate_new[:, injection_position:injection_position+1] = (
                                blend_alpha * source_mod.v_a_gate +
                                (1 - blend_alpha) * v_a_gate[:, injection_position:injection_position+1]
                            )
                            v_b_gate_new[:, injection_position:injection_position+1] = (
                                blend_alpha * source_mod.v_b_gate +
                                (1 - blend_alpha) * v_b_gate[:, injection_position:injection_position+1]
                            )
                            v_a_up_new[:, injection_position:injection_position+1] = (
                                blend_alpha * source_mod.v_a_up +
                                (1 - blend_alpha) * v_a_up[:, injection_position:injection_position+1]
                            )
                            v_b_up_new[:, injection_position:injection_position+1] = (
                                blend_alpha * source_mod.v_b_up +
                                (1 - blend_alpha) * v_b_up[:, injection_position:injection_position+1]
                            )
                        
                        # Store injected
                        injected_modulations[layer_idx] = {
                            'magnitude': (
                                v_a_gate_new[:, injection_position].norm().item() +
                                v_b_gate_new[:, injection_position].norm().item() +
                                v_a_up_new[:, injection_position].norm().item() +
                                v_b_up_new[:, injection_position].norm().item()
                            ) / 4
                        }
                        
                        return (v_a_gate_new, v_b_gate_new), (v_a_up_new, v_b_up_new)
                    else:
                        # Single modulation (similar logic)
                        v_a, v_b = output
                        
                        original_modulations[layer_idx] = {
                            'magnitude': (v_a[:, injection_position].norm().item() + 
                                        v_b[:, injection_position].norm().item()) / 2
                        }
                        
                        v_a_new = v_a.clone()
                        v_b_new = v_b.clone()
                        
                        if injection_mode == "replace":
                            v_a_new[:, injection_position:injection_position+1] = source_mod.v_a
                            v_b_new[:, injection_position:injection_position+1] = source_mod.v_b
                        elif injection_mode == "blend":
                            v_a_new[:, injection_position:injection_position+1] = (
                                blend_alpha * source_mod.v_a +
                                (1 - blend_alpha) * v_a[:, injection_position:injection_position+1]
                            )
                            v_b_new[:, injection_position:injection_position+1] = (
                                blend_alpha * source_mod.v_b +
                                (1 - blend_alpha) * v_b[:, injection_position:injection_position+1]
                            )
                        
                        injected_modulations[layer_idx] = {
                            'magnitude': (v_a_new[:, injection_position].norm().item() + 
                                        v_b_new[:, injection_position].norm().item()) / 2
                        }
                        
                        return v_a_new, v_b_new
                
                return output
            return hook
        
        # Register injection hooks only on active NPT layers that we're injecting into
        handles = []
        for layer_idx, source_mod in source_modulations.items():
            if layer_idx < len(self.model.model.layers) and layer_idx in self.active_npt_layers:
                layer = self.model.model.layers[layer_idx]
                if hasattr(layer, 'np_component'):
                    handle = layer.np_component.register_forward_hook(
                        create_injection_hook(layer_idx, source_mod)
                    )
                    handles.append(handle)
        
        # Compute logits with injection
        with torch.no_grad():
            # Ensure correct NPT mode for each layer
            for idx in range(len(self.model.model.layers)):
                if hasattr(self.model.model.layers[idx], 'set_npt_mode'):
                    self.model.model.layers[idx].set_npt_mode(idx in self.active_npt_layers)
            
            outputs = self.model(input_ids)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
        # Remove hooks
        for handle in handles:
            handle.remove()
            
        # Get next token logits and probabilities
        next_token_logits = logits[0, -1]
        next_token_probs = F.softmax(next_token_logits, dim=-1)
        
        return {
            'logits': next_token_logits,
            'probs': next_token_probs,
            'original_modulations': original_modulations,
            'injected_modulations': injected_modulations,
            'hooks_fired': hook_fired
        }
    
    def compute_baseline_logits(self, prompt: str) -> Dict[str, Any]:
        """Compute baseline logits with selective NPT layers."""
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        input_ids = inputs['input_ids']
        
        with torch.no_grad():
            # Ensure correct NPT mode for each layer
            for idx in range(len(self.model.model.layers)):
                if hasattr(self.model.model.layers[idx], 'set_npt_mode'):
                    self.model.model.layers[idx].set_npt_mode(idx in self.active_npt_layers)
            
            outputs = self.model(input_ids)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
        next_token_logits = logits[0, -1]
        next_token_probs = F.softmax(next_token_logits, dim=-1)
        
        return {
            'logits': next_token_logits,
            'probs': next_token_probs
        }


class SelectiveFactualTransferAnalyzer:
    """Analyzer using selective NPT layer loading."""

    def __init__(self, model: NPTLlamaModel, tokenizer: AutoTokenizer, device: torch.device, active_npt_layers: Set[int]):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.active_npt_layers = active_npt_layers
        self.extractor = SelectiveModulationExtractor(model, tokenizer, device, active_npt_layers)
        self.logit_computer = SelectiveLogitComputer(model, tokenizer, device, active_npt_layers)

    def run_transfer_experiment(
        self,
        source_prompt: str,
        source_answer: str,
        target_prompt: str,
        target_answer: str,
        injection_mode: str = "replace"
    ) -> Dict[str, Any]:
        """
        Run factual transfer experiment with selective NPT layers.

        NEW Architecture: Transfers both attention patterns and MLP transformations
        from source to target context, enabling richer knowledge transfer.
        """
        logger.info("\n" + "="*80)
        logger.info("NEW ARCHITECTURE: MLP(h) → attention + MLP(h+attention)")
        logger.info(f"Source: '{source_prompt}' -> '{source_answer}'")
        logger.info(f"Target: '{target_prompt}' -> '{target_answer}'")
        logger.info(f"Active NPT layers: {sorted(self.active_npt_layers)}")
        logger.info("Transferring attention patterns + MLP transformations")
        logger.info("="*80)

        results = {
            'source_prompt': source_prompt,
            'source_answer': source_answer,
            'target_prompt': target_prompt,
            'target_answer': target_answer,
            'active_npt_layers': list(self.active_npt_layers),
            'layer_results': {}
        }

        # Extract modulations from source
        logger.info("\n1. Extracting source modulations (attention + MLP transformation)...")
        source_modulations = self.extractor.extract_generation_modulation(
            prompt=source_prompt,
            target_token=source_answer
        )

        for layer_idx, mod_data in source_modulations.items():
            logger.info(f"  Layer {layer_idx}: Magnitude = {mod_data.modulation_magnitude:.6f}")

        # Get baseline probabilities
        logger.info("\n2. Computing baseline probabilities...")
        baseline = self.logit_computer.compute_baseline_logits(target_prompt)

        # Get top predictions
        top_k = 10
        baseline_top_probs, baseline_top_indices = torch.topk(baseline['probs'], top_k)
        baseline_top_tokens = [self.tokenizer.decode([idx]) for idx in baseline_top_indices]

        # Get specific token probabilities
        source_token_id = self.tokenizer(source_answer, add_special_tokens=False).input_ids[0]
        target_token_id = self.tokenizer(target_answer, add_special_tokens=False).input_ids[0]

        logger.info(f"  Top-5 baseline: {baseline_top_tokens[:5]}")
        logger.info(f"  {source_answer} baseline prob: {baseline['probs'][source_token_id]:.6f}")
        logger.info(f"  {target_answer} baseline prob: {baseline['probs'][target_token_id]:.6f}")

        # Test each layer
        logger.info("\n3. Testing modulation injection...")

        for layer_idx in sorted(self.active_npt_layers):
            if layer_idx not in source_modulations:
                logger.warning(f"  Layer {layer_idx}: No modulation available")
                continue

            logger.info(f"\n  Layer {layer_idx}:")

            # Inject and compute
            injection_result = self.logit_computer.compute_logits_with_injection(
                prompt=target_prompt,
                source_modulations={layer_idx: source_modulations[layer_idx]},
                injection_mode=injection_mode
            )

            # Calculate changes
            source_prob_after = injection_result['probs'][source_token_id].item()
            target_prob_after = injection_result['probs'][target_token_id].item()
            source_prob_before = baseline['probs'][source_token_id].item()
            target_prob_before = baseline['probs'][target_token_id].item()

            # KL divergence
            kl_div = F.kl_div(
                torch.log(injection_result['probs'] + 1e-10),
                baseline['probs'],
                reduction='sum'
            ).item()

            # Top-k changes
            injected_top_probs, injected_top_indices = torch.topk(injection_result['probs'], top_k)
            injected_top_tokens = [self.tokenizer.decode([idx]) for idx in injected_top_indices]

            baseline_top_set = set(baseline_top_indices[:10].tolist())
            injected_top_set = set(injected_top_indices[:10].tolist())
            top_k_changes = len(baseline_top_set - injected_top_set)

            # Log results - modulation magnitude represents attention+MLP strength
            logger.info(f"    Modulation magnitude (attn+MLP): {injection_result['original_modulations'][layer_idx]['magnitude']:.6f} -> {injection_result['injected_modulations'][layer_idx]['magnitude']:.6f}")
            logger.info(f"    {source_answer}: {source_prob_before:.6f} -> {source_prob_after:.6f} (shift: {source_prob_after - source_prob_before:+.6f}, {(source_prob_after - source_prob_before) / (source_prob_before + 1e-10) * 100:+.1f}%)")
            logger.info(f"    {target_answer}: {target_prob_before:.6f} -> {target_prob_after:.6f} (shift: {target_prob_after - target_prob_before:+.6f}, {(target_prob_after - target_prob_before) / (target_prob_before + 1e-10) * 100:+.1f}%)")
            logger.info(f"    KL divergence: {kl_div:.6f}")
            logger.info(f"    Top-10 changes: {top_k_changes}/10")
            logger.info(f"    New top-3: {injected_top_tokens[:3]}")

            # Store results
            results['layer_results'][layer_idx] = {
                'source_prob_before': source_prob_before,
                'source_prob_after': source_prob_after,
                'source_shift': source_prob_after - source_prob_before,
                'source_relative_shift': (source_prob_after - source_prob_before) / (source_prob_before + 1e-10),
                'target_prob_before': target_prob_before,
                'target_prob_after': target_prob_after,
                'target_shift': target_prob_after - target_prob_before,
                'target_relative_shift': (target_prob_after - target_prob_before) / (target_prob_before + 1e-10),
                'kl_divergence': kl_div,
                'top_k_changes': top_k_changes,
                'baseline_top_tokens': baseline_top_tokens,
                'injected_top_tokens': injected_top_tokens
            }

        # TEST CUMULATIVE EFFECT: Inject ALL layers together
        if len(self.active_npt_layers) > 1:
            logger.info("\n4. Testing CUMULATIVE injection (all layers together)...")
            logger.info(f"   Injecting into layers: {sorted(self.active_npt_layers)}")

            # Inject into ALL active layers simultaneously
            cumulative_result = self.logit_computer.compute_logits_with_injection(
                prompt=target_prompt,
                source_modulations=source_modulations,  # ALL layers!
                injection_mode=injection_mode
            )

            # Calculate cumulative changes
            source_prob_cumulative = cumulative_result['probs'][source_token_id].item()
            target_prob_cumulative = cumulative_result['probs'][target_token_id].item()

            # KL divergence for cumulative
            kl_div_cumulative = F.kl_div(
                torch.log(cumulative_result['probs'] + 1e-10),
                baseline['probs'],
                reduction='sum'
            ).item()

            # Top-k changes for cumulative
            cumulative_top_probs, cumulative_top_indices = torch.topk(cumulative_result['probs'], top_k)
            cumulative_top_tokens = [self.tokenizer.decode([idx]) for idx in cumulative_top_indices]

            baseline_top_set = set(baseline_top_indices[:10].tolist())
            cumulative_top_set = set(cumulative_top_indices[:10].tolist())
            cumulative_top_k_changes = len(baseline_top_set - cumulative_top_set)

            # Log cumulative results
            logger.info(f"\n  CUMULATIVE EFFECT (All layers {sorted(self.active_npt_layers)}):")
            logger.info(f"    {source_answer}: {source_prob_before:.6f} -> {source_prob_cumulative:.6f} (shift: {source_prob_cumulative - source_prob_before:+.6f}, {(source_prob_cumulative - source_prob_before) / (source_prob_before + 1e-10) * 100:+.1f}%)")
            logger.info(f"    {target_answer}: {target_prob_before:.6f} -> {target_prob_cumulative:.6f} (shift: {target_prob_cumulative - target_prob_before:+.6f}, {(target_prob_cumulative - target_prob_before) / (target_prob_before + 1e-10) * 100:+.1f}%)")
            logger.info(f"    KL divergence: {kl_div_cumulative:.6f}")
            logger.info(f"    Top-10 changes: {cumulative_top_k_changes}/10")
            logger.info(f"    New top-3: {cumulative_top_tokens[:3]}")

            # Compare to individual effects
            logger.info(f"\n    Comparison:")
            total_individual_shift = sum(
                results['layer_results'][idx]['source_shift']
                for idx in self.active_npt_layers
                if idx in results['layer_results']
            )
            logger.info(f"    Sum of individual shifts: {total_individual_shift:+.6f}")
            logger.info(f"    Actual cumulative shift:  {source_prob_cumulative - source_prob_before:+.6f}")

            synergy = (source_prob_cumulative - source_prob_before) - total_individual_shift
            if abs(synergy) > 0.001:
                if synergy > 0:
                    logger.info(f"    SYNERGY: Cumulative effect is {synergy:+.6f} STRONGER than sum of parts!")
                else:
                    logger.info(f"    INTERFERENCE: Cumulative effect is {abs(synergy):.6f} WEAKER than sum of parts")
            else:
                logger.info(f"    Effects are approximately additive")

            # Store cumulative results
            results['cumulative_effect'] = {
                'layers': list(self.active_npt_layers),
                'source_prob_before': source_prob_before,
                'source_prob_after': source_prob_cumulative,
                'source_shift': source_prob_cumulative - source_prob_before,
                'source_relative_shift': (source_prob_cumulative - source_prob_before) / (source_prob_before + 1e-10),
                'target_prob_before': target_prob_before,
                'target_prob_after': target_prob_cumulative,
                'target_shift': target_prob_cumulative - target_prob_before,
                'target_relative_shift': (target_prob_cumulative - target_prob_before) / (target_prob_before + 1e-10),
                'kl_divergence': kl_div_cumulative,
                'top_k_changes': cumulative_top_k_changes,
                'top_tokens': cumulative_top_tokens[:5],
                'synergy': synergy if 'synergy' in locals() else 0
            }

        return results

    def verify_model_quality(self, test_prompts: List[str]):
        """Verify that the model with selective NPT produces reasonable outputs."""
        logger.info("\n" + "="*80)
        logger.info("Verifying Model Quality with Selective NPT")
        logger.info("="*80)

        for prompt in test_prompts:
            logger.info(f"\nPrompt: '{prompt}'")

            # Get predictions
            baseline = self.logit_computer.compute_baseline_logits(prompt)

            # Get top-5
            top_5_probs, top_5_indices = torch.topk(baseline['probs'], 5)
            top_5_tokens = [self.tokenizer.decode([idx]) for idx in top_5_indices]

            logger.info("  Top-5 predictions:")
            for token, prob in zip(top_5_tokens, top_5_probs):
                logger.info(f"    '{token}': {prob:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Factual Transfer with Selective NPT Loading")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to NPT checkpoint")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--layers", type=str, default="14,15", help="NPT layers to load and test")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--verify_quality", action="store_true", help="Verify model quality")
    parser.add_argument("--output_dir", type=str, default="experiments/factual_transfer_selective")
    parser.add_argument("--use_context", action="store_true", help="Use contextual prompts for stronger association")

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(42)
    np.random.seed(42)

    # Parse layers
    layers_to_use = [int(x.strip()) for x in args.layers.split(',')]

    # Load model with selective NPT
    loader = SelectiveNPTLoader()
    model, tokenizer, active_npt_layers = loader.load_model_with_selective_npt(
        checkpoint_path=args.checkpoint,
        model_name=args.model_name,
        layers_to_use=layers_to_use,
        device=args.device
    )

    if not active_npt_layers:
        logger.error("No NPT layers loaded. Exiting.")
        return

    # Create analyzer
    device = torch.device(args.device)
    analyzer = SelectiveFactualTransferAnalyzer(model, tokenizer, device, active_npt_layers)

    # Verify model quality if requested
    if args.verify_quality:
        test_prompts = [
            "The capital of France is",
            "The largest planet in our solar system is",
            "Water boils at",
            "The speed of light is approximately",
            "The first president of the United States was"
        ]
        analyzer.verify_model_quality(test_prompts)

    # Define experiments
    if args.use_context:
        # Contextual prompts with stronger associations
        experiments = [
            {
                'source_prompt': "The capital of France is Paris. Given that the capital of France is",
                'source_answer': " Paris",
                'target_prompt': "The capital of Germany is",
                'target_answer': " Berlin"
            },
            {
                'source_prompt': "The largest planet is Jupiter. The largest planet in our solar system is",
                'source_answer': " Jupiter",
                'target_prompt': "The smallest planet is",
                'target_answer': " Mercury"
            },
            {
                'source_prompt': "Shakespeare wrote Romeo and Juliet. Shakespeare famously wrote",
                'source_answer': " Romeo",
                'target_prompt': "Dickens wrote",
                'target_answer': " Oliver"
            }
        ]
        logger.info("\n=== Using CONTEXTUAL prompts for stronger association ===")
    else:
        # Original simple prompts
        experiments = [
            {
                'source_prompt': "The capital of France is",
                'source_answer': " Paris",
                'target_prompt': "The capital of Germany is",
                'target_answer': " Berlin"
            },
            {
                'source_prompt': "The largest planet is",
                'source_answer': " Jupiter",
                'target_prompt': "The smallest planet is",
                'target_answer': " Mercury"
            },
            {
                'source_prompt': "Shakespeare wrote",
                'source_answer': " Romeo",
                'target_prompt': "Dickens wrote",
                'target_answer': " Oliver"
            }
        ]

    # Run experiments
    all_results = []

    for exp in experiments:
        results = analyzer.run_transfer_experiment(
            source_prompt=exp['source_prompt'],
            source_answer=exp['source_answer'],
            target_prompt=exp['target_prompt'],
            target_answer=exp['target_answer'],
            injection_mode="replace"
        )
        all_results.append(results)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"\nResults saved to {output_dir}")

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)

    for exp_idx, result in enumerate(all_results):
        logger.info(f"\nExperiment {exp_idx + 1}: {result['source_prompt']} -> {result['target_prompt']}")
        logger.info(f"Active NPT layers: {result['active_npt_layers']}")

        # Individual layer effects
        for layer_idx, layer_result in result['layer_results'].items():
            source_shift = layer_result['source_shift']
            target_shift = layer_result['target_shift']
            logger.info(f"  Layer {layer_idx} (alone):")
            logger.info(f"    Source ({result['source_answer']}): {source_shift:+.6f} ({layer_result['source_relative_shift']*100:+.1f}%)")
            logger.info(f"    Target ({result['target_answer']}): {target_shift:+.6f} ({layer_result['target_relative_shift']*100:+.1f}%)")

        # Cumulative effect
        if 'cumulative_effect' in result:
            cum = result['cumulative_effect']
            logger.info(f"  CUMULATIVE (layers {cum['layers']}):")
            logger.info(f"    Source ({result['source_answer']}): {cum['source_shift']:+.6f} ({cum['source_relative_shift']*100:+.1f}%)")
            logger.info(f"    Target ({result['target_answer']}): {cum['target_shift']:+.6f} ({cum['target_relative_shift']*100:+.1f}%)")
            if abs(cum.get('synergy', 0)) > 0.001:
                if cum['synergy'] > 0:
                    logger.info(f"    💥 SYNERGY: +{cum['synergy']:.6f} stronger than sum!")
                else:
                    logger.info(f"    ⚠️  INTERFERENCE: {cum['synergy']:.6f} weaker than sum")

    # Explain what the results mean with NEW architecture
    logger.info("\n" + "="*80)
    logger.info("INTERPRETATION (NEW Architecture)")
    logger.info("="*80)
    logger.info("\nWith the NEW architecture where MLP(h) → attention + MLP(h+attention):")
    logger.info("• Positive source shifts indicate successful transfer of attention patterns")
    logger.info("• The modulation encodes BOTH how attention processes the context")
    logger.info("  AND how the MLP transforms with that attention")
    logger.info("• Stronger effects are expected because we're transferring richer representations")
    logger.info("• Each injection makes the model 'experience' the source context at that position")


if __name__ == "__main__":
    main()