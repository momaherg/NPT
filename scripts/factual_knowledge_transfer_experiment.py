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
                    # Handle different key formats
                    if 'layer_' in key and '_np' in key:
                        # Format: layer_15_np.W_down_gate.0
                        parts = key.split('_')
                        for i, part in enumerate(parts):
                            if part == 'layer' and i + 1 < len(parts) and parts[i + 1].isdigit():
                                npt_layer_indices.add(int(parts[i + 1]))
                    elif 'model.layers' in key and 'np_component' in key:
                        # Format: model.layers.15.np_component.W_down
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


class FactualTransferAnalyzer:
    """Analyze the effects of modulation transfer on factual predictions."""

    def __init__(self, model: NPTLlamaModel, tokenizer: AutoTokenizer, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.extractor = ModulationExtractor(model, tokenizer, device)
        self.injector = ModulationInjector(model, tokenizer, device)

    def get_baseline_probabilities(self, prompt: str) -> Dict[str, Any]:
        """Get baseline token probabilities without any modulation injection."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            self.model.set_npt_mode(True)
            outputs = self.model(inputs['input_ids'])
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs

        # Get probabilities for next token
        next_token_logits = logits[0, -1]
        next_token_probs = F.softmax(next_token_logits, dim=-1)

        # Get top-k tokens
        top_k = 20
        top_probs, top_indices = torch.topk(next_token_probs, top_k)
        top_tokens = [self.tokenizer.decode([idx]) for idx in top_indices]

        return {
            'full_probs': next_token_probs,
            'top_k_probs': top_probs.cpu().numpy(),
            'top_k_indices': top_indices.cpu().numpy(),
            'top_k_tokens': top_tokens
        }

    def compute_probability_shift(
        self,
        original_probs: torch.Tensor,
        modulated_probs: torch.Tensor,
        target_tokens: List[str]
    ) -> Dict[str, float]:
        """Compute probability shifts for specific tokens."""
        shifts = {}

        for token_str in target_tokens:
            token_ids = self.tokenizer.encode(token_str, add_special_tokens=False)
            if token_ids:
                token_id = token_ids[0]
                if token_id < len(original_probs):
                    original_p = original_probs[token_id].item()
                    modulated_p = modulated_probs[token_id].item()
                    shifts[token_str] = {
                        'original': original_p,
                        'modulated': modulated_p,
                        'shift': modulated_p - original_p,
                        'relative_change': (modulated_p - original_p) / (original_p + 1e-10)
                    }

        return shifts

    def compute_kl_divergence(self, p: torch.Tensor, q: torch.Tensor) -> float:
        """Compute KL divergence KL(p||q)."""
        # Add small epsilon for numerical stability
        p = p + 1e-10
        q = q + 1e-10
        return (p * (p.log() - q.log())).sum().item()

    def run_single_transfer_experiment(
        self,
        source_prompt: str,
        target_prompt: str,
        source_answer: str,
        target_answer: str,
        layers_to_test: List[int],
        injection_mode: str = "replace"
    ) -> Dict[str, Any]:
        """
        Run a single factual transfer experiment.

        Args:
            source_prompt: E.g., "The capital of France is"
            target_prompt: E.g., "The capital of Germany is"
            source_answer: E.g., "Paris"
            target_answer: E.g., "Berlin"
            layers_to_test: Which NPT layers to test
            injection_mode: How to inject modulation
        """
        results = {
            'source_prompt': source_prompt,
            'target_prompt': target_prompt,
            'source_answer': source_answer,
            'target_answer': target_answer,
            'layer_results': {}
        }

        # Extract modulation from source when predicting answer
        logger.info(f"\nExtracting modulation from: '{source_prompt}' -> '{source_answer}'")
        source_modulations = self.extractor.extract_at_position(
            source_prompt,
            position_idx=-1,
            layers_to_extract=layers_to_test
        )

        # Get baseline probabilities for target
        logger.info(f"Getting baseline for: '{target_prompt}'")
        baseline = self.get_baseline_probabilities(target_prompt)

        # Test each layer independently
        for layer_idx in layers_to_test:
            if layer_idx not in source_modulations:
                continue

            logger.info(f"\nTesting layer {layer_idx}...")

            # Inject modulation and generate
            injection_result = self.injector.generate_with_injection(
                target_prompt,
                injection_position=-1,
                source_modulations={layer_idx: source_modulations[layer_idx]},
                injection_mode=injection_mode,
                max_new_tokens=5
            )

            # Compute metrics
            if injection_result['probs_at_injection'] is not None:
                modulated_probs = injection_result['probs_at_injection']

                # Probability shifts for key tokens
                key_tokens = [source_answer, target_answer, 'Paris', 'Berlin', 'France', 'Germany']
                prob_shifts = self.compute_probability_shift(
                    baseline['full_probs'],
                    modulated_probs,
                    key_tokens
                )

                # KL divergence
                kl_div = self.compute_kl_divergence(
                    baseline['full_probs'],
                    modulated_probs
                )

                # Top-k analysis
                top_k = 20
                mod_top_probs, mod_top_indices = torch.topk(modulated_probs, top_k)
                mod_top_tokens = [self.tokenizer.decode([idx]) for idx in mod_top_indices]

                # Count how many top-k tokens changed
                orig_top_set = set(baseline['top_k_indices'][:10])
                mod_top_set = set(mod_top_indices[:10].cpu().numpy())
                top_k_changes = len(orig_top_set - mod_top_set)

                layer_results = {
                    'generated_text': injection_result['generated_text'],
                    'probability_shifts': prob_shifts,
                    'kl_divergence': kl_div,
                    'top_k_changes': top_k_changes,
                    'modulated_top_tokens': mod_top_tokens[:10],
                    'modulated_top_probs': mod_top_probs[:10].cpu().numpy().tolist(),
                    'original_top_tokens': baseline['top_k_tokens'][:10],
                    'original_top_probs': baseline['top_k_probs'][:10].tolist()
                }

                results['layer_results'][layer_idx] = layer_results

                # Log key findings
                if source_answer in prob_shifts:
                    logger.info(f"  {source_answer}: {prob_shifts[source_answer]['original']:.4f} -> {prob_shifts[source_answer]['modulated']:.4f} (shift: {prob_shifts[source_answer]['shift']:.4f})")
                if target_answer in prob_shifts:
                    logger.info(f"  {target_answer}: {prob_shifts[target_answer]['original']:.4f} -> {prob_shifts[target_answer]['modulated']:.4f} (shift: {prob_shifts[target_answer]['shift']:.4f})")
                logger.info(f"  KL divergence: {kl_div:.4f}")
                logger.info(f"  Top-10 tokens changed: {top_k_changes}")
                logger.info(f"  Generated: {injection_result['generated_text']}")

        return results

    def run_ablation_study(
        self,
        source_prompt: str,
        target_prompt: str,
        source_answer: str,
        target_answer: str,
        layer_idx: int
    ) -> Dict[str, Any]:
        """Run ablation study on different injection modes and strengths."""
        results = {'ablation_results': []}

        # Extract source modulation
        source_modulations = self.extractor.extract_at_position(
            source_prompt,
            position_idx=-1,
            layers_to_extract=[layer_idx]
        )

        # Check if extraction was successful
        if layer_idx not in source_modulations:
            logger.warning(f"Layer {layer_idx} not found in source modulations")
            return results

        # Get baseline
        baseline = self.get_baseline_probabilities(target_prompt)

        # Test different modes and strengths
        modes_and_alphas = [
            ('replace', 1.0),
            ('blend', 0.25),
            ('blend', 0.5),
            ('blend', 0.75),
            ('blend', 1.0),
            ('add', 0.1),
            ('add', 0.5),
            ('add', 1.0)
        ]

        for mode, alpha in modes_and_alphas:
            logger.info(f"\nTesting {mode} mode with alpha={alpha}")

            injection_result = self.injector.generate_with_injection(
                target_prompt,
                injection_position=-1,
                source_modulations={layer_idx: source_modulations[layer_idx]},
                injection_mode=mode,
                blend_alpha=alpha,
                max_new_tokens=5
            )

            if injection_result['probs_at_injection'] is not None:
                modulated_probs = injection_result['probs_at_injection']

                # Get probability shifts
                key_tokens = [source_answer, target_answer]
                prob_shifts = self.compute_probability_shift(
                    baseline['full_probs'],
                    modulated_probs,
                    key_tokens
                )

                # KL divergence
                kl_div = self.compute_kl_divergence(
                    baseline['full_probs'],
                    modulated_probs
                )

                results['ablation_results'].append({
                    'mode': mode,
                    'alpha': alpha,
                    'source_answer_prob': prob_shifts.get(source_answer, {}).get('modulated', 0),
                    'target_answer_prob': prob_shifts.get(target_answer, {}).get('modulated', 0),
                    'source_answer_shift': prob_shifts.get(source_answer, {}).get('shift', 0),
                    'target_answer_shift': prob_shifts.get(target_answer, {}).get('shift', 0),
                    'kl_divergence': kl_div,
                    'generated_text': injection_result['generated_text']
                })

        return results


def visualize_results(results: Dict[str, Any], output_dir: Path):
    """Create visualizations of the experimental results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)

    # 1. Probability shift heatmap across layers
    if 'layer_results' in results:
        layers = sorted(results['layer_results'].keys())

        # Create heatmap data
        tokens_to_plot = ['Paris', 'Berlin', 'France', 'Germany']
        heatmap_data = []

        for layer in layers:
            layer_shifts = []
            for token in tokens_to_plot:
                if token in results['layer_results'][layer]['probability_shifts']:
                    shift = results['layer_results'][layer]['probability_shifts'][token]['shift']
                    layer_shifts.append(shift)
                else:
                    layer_shifts.append(0)
            heatmap_data.append(layer_shifts)

        # Plot heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(
            heatmap_data,
            xticklabels=tokens_to_plot,
            yticklabels=[f"Layer {l}" for l in layers],
            cmap='RdBu_r',
            center=0,
            annot=True,
            fmt='.4f',
            cbar_kws={'label': 'Probability Shift'}
        )
        plt.title('Probability Shifts After Modulation Injection')
        plt.xlabel('Token')
        plt.ylabel('NPT Layer')
        plt.tight_layout()
        plt.savefig(output_dir / 'probability_shift_heatmap.png', dpi=150)
        plt.close()

        # 2. Top token changes visualization
        fig, axes = plt.subplots(1, len(layers), figsize=(5*len(layers), 8))
        if len(layers) == 1:
            axes = [axes]

        for idx, layer in enumerate(layers):
            ax = axes[idx]
            layer_data = results['layer_results'][layer]

            # Plot top tokens before and after
            orig_tokens = layer_data['original_top_tokens'][:10]
            orig_probs = layer_data['original_top_probs'][:10]
            mod_tokens = layer_data['modulated_top_tokens'][:10]
            mod_probs = layer_data['modulated_top_probs'][:10]

            x = np.arange(10)
            width = 0.35

            ax.bar(x - width/2, orig_probs, width, label='Original', alpha=0.8)
            ax.bar(x + width/2, mod_probs, width, label='Modulated', alpha=0.8)

            ax.set_xlabel('Rank')
            ax.set_ylabel('Probability')
            ax.set_title(f'Layer {layer} Top-10 Tokens')
            ax.set_xticks(x)
            ax.set_xticklabels([f"{i+1}" for i in range(10)])
            ax.legend()

            # Add token labels
            for i in range(10):
                ax.text(i - width/2, orig_probs[i], orig_tokens[i],
                       ha='center', va='bottom', rotation=45, fontsize=8)
                ax.text(i + width/2, mod_probs[i], mod_tokens[i],
                       ha='center', va='bottom', rotation=45, fontsize=8)

        plt.tight_layout()
        plt.savefig(output_dir / 'top_tokens_comparison.png', dpi=150)
        plt.close()

    # 3. Ablation study results
    if 'ablation_results' in results and results['ablation_results']:
        ablation_data = results['ablation_results']

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Group by mode
        modes = {}
        for item in ablation_data:
            mode = item['mode']
            if mode not in modes:
                modes[mode] = {'alphas': [], 'source_probs': [], 'target_probs': []}
            modes[mode]['alphas'].append(item['alpha'])
            modes[mode]['source_probs'].append(item['source_answer_prob'])
            modes[mode]['target_probs'].append(item['target_answer_prob'])

        # Plot source answer probability
        ax = axes[0]
        for mode, data in modes.items():
            if mode == 'replace':
                ax.scatter(data['alphas'], data['source_probs'], label=mode, s=100, marker='*')
            else:
                ax.plot(data['alphas'], data['source_probs'], label=mode, marker='o')

        ax.set_xlabel('Alpha (Injection Strength)')
        ax.set_ylabel('Probability')
        ax.set_title('Source Answer (Paris) Probability')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot target answer probability
        ax = axes[1]
        for mode, data in modes.items():
            if mode == 'replace':
                ax.scatter(data['alphas'], data['target_probs'], label=mode, s=100, marker='*')
            else:
                ax.plot(data['alphas'], data['target_probs'], label=mode, marker='o')

        ax.set_xlabel('Alpha (Injection Strength)')
        ax.set_ylabel('Probability')
        ax.set_title('Target Answer (Berlin) Probability')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'ablation_study.png', dpi=150)
        plt.close()

    logger.info(f"Visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Factual Knowledge Transfer Experiment")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to NPT checkpoint"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Base model name"
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated list of layers to test (e.g., '14,15')"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/factual_transfer_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--run_ablation",
        action="store_true",
        help="Run ablation study on injection modes"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )

    args = parser.parse_args()

    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create config
    config = ExperimentConfig(
        checkpoint_path=args.checkpoint,
        model_name=args.model_name,
        device=args.device,
        output_dir=args.output_dir
    )

    # Parse layers if provided
    if args.layers:
        config.npt_layers = [int(x.strip()) for x in args.layers.split(',')]

    # Load model
    loader = NPTModelLoader(config)
    model, tokenizer = loader.load_model()

    # Create analyzer
    analyzer = FactualTransferAnalyzer(model, tokenizer, torch.device(config.device))

    # Define experiment pairs
    experiments = [
        {
            'source_prompt': "The capital of France is",
            'target_prompt': "The capital of Germany is",
            'source_answer': "Paris",
            'target_answer': "Berlin"
        },
        {
            'source_prompt': "The capital of the United States is",
            'target_prompt': "The capital of Canada is",
            'source_answer': "Washington",
            'target_answer': "Ottawa"
        }
    ]

    # Run main experiments
    all_results = []

    for exp in experiments:
        logger.info("\n" + "="*80)
        logger.info(f"Running experiment: {exp['source_prompt']} -> {exp['target_prompt']}")
        logger.info("="*80)

        results = analyzer.run_single_transfer_experiment(
            source_prompt=exp['source_prompt'],
            target_prompt=exp['target_prompt'],
            source_answer=exp['source_answer'],
            target_answer=exp['target_answer'],
            layers_to_test=config.npt_layers or [14, 15],
            injection_mode="replace"
        )

        all_results.append(results)

        # Run ablation study if requested
        if args.run_ablation and config.npt_layers:
            logger.info("\n" + "="*80)
            logger.info("Running ablation study...")
            logger.info("="*80)

            ablation_results = analyzer.run_ablation_study(
                source_prompt=exp['source_prompt'],
                target_prompt=exp['target_prompt'],
                source_answer=exp['source_answer'],
                target_answer=exp['target_answer'],
                layer_idx=config.npt_layers[-1]  # Use last layer
            )

            results.update(ablation_results)

    # Save results
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'results.json', 'w') as f:
        # Convert tensors to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        json.dump(convert_to_serializable(all_results), f, indent=2)

    # Create visualizations for first experiment
    if all_results:
        visualize_results(all_results[0], output_dir)

    logger.info(f"\n" + "="*80)
    logger.info(f"Results saved to {output_dir}")
    logger.info("="*80)


if __name__ == "__main__":
    main()