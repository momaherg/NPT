#!/usr/bin/env python3
"""
Modulation Arithmetic Experiment for Factual Knowledge Transfer

This experiment isolates factual components from context-dependent modulations
using arithmetic operations on modulation vectors.
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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.npt import NPTLlamaModel, NPTConfig
from transformers import AutoTokenizer, AutoConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModulationData:
    """Container for modulation vectors."""
    v_a_gate: torch.Tensor
    v_b_gate: torch.Tensor
    v_a_up: torch.Tensor
    v_b_up: torch.Tensor
    magnitude: float
    context: str
    target: str
    
    def __sub__(self, other):
        """Subtract two modulations element-wise."""
        return ModulationData(
            v_a_gate=self.v_a_gate - other.v_a_gate,
            v_b_gate=self.v_b_gate - other.v_b_gate,
            v_a_up=self.v_a_up - other.v_a_up,
            v_b_up=self.v_b_up - other.v_b_up,
            magnitude=self._compute_magnitude(
                self.v_a_gate - other.v_a_gate,
                self.v_b_gate - other.v_b_gate,
                self.v_a_up - other.v_a_up,
                self.v_b_up - other.v_b_up
            ),
            context=f"{self.context} - {other.context}",
            target=f"{self.target} - {other.target}"
        )
    
    def __add__(self, other):
        """Add two modulations element-wise."""
        return ModulationData(
            v_a_gate=self.v_a_gate + other.v_a_gate,
            v_b_gate=self.v_b_gate + other.v_b_gate,
            v_a_up=self.v_a_up + other.v_a_up,
            v_b_up=self.v_b_up + other.v_b_up,
            magnitude=self._compute_magnitude(
                self.v_a_gate + other.v_a_gate,
                self.v_b_gate + other.v_b_gate,
                self.v_a_up + other.v_a_up,
                self.v_b_up + other.v_b_up
            ),
            context=f"{self.context} + {other.context}",
            target=f"{self.target} + {other.target}"
        )
    
    def __mul__(self, scalar):
        """Scale modulation by scalar."""
        return ModulationData(
            v_a_gate=self.v_a_gate * scalar,
            v_b_gate=self.v_b_gate * scalar,
            v_a_up=self.v_a_up * scalar,
            v_b_up=self.v_b_up * scalar,
            magnitude=self.magnitude * abs(scalar),
            context=self.context,
            target=self.target
        )
    
    @staticmethod
    def _compute_magnitude(v_a_gate, v_b_gate, v_a_up, v_b_up):
        """Compute average magnitude of modulation vectors."""
        return (
            v_a_gate.norm().item() + v_b_gate.norm().item() +
            v_a_up.norm().item() + v_b_up.norm().item()
        ) / 4


class ModulationArithmeticExtractor:
    """Extract modulations for arithmetic operations."""
    
    def __init__(self, model, tokenizer, device, active_layer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.active_layer = active_layer  # Single layer for clarity
        
        # Ensure only the active layer is in NPT mode
        for idx in range(len(self.model.model.layers)):
            if hasattr(self.model.model.layers[idx], 'set_npt_mode'):
                self.model.model.layers[idx].set_npt_mode(idx == active_layer)
    
    def extract_modulation(self, prompt: str, target: str) -> ModulationData:
        """Extract modulation when generating target after prompt."""
        # Tokenize
        prompt_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        target_ids = self.tokenizer(target, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        
        # Full sequence for teacher forcing
        full_ids = torch.cat([prompt_ids, target_ids[:, :1]], dim=1)
        
        # Position where target is generated (after prompt)
        extraction_position = prompt_ids.shape[1]
        
        # Storage for extracted modulation
        extracted = {}
        
        def extraction_hook(module, input, output):
            if isinstance(output[0], tuple):
                (v_a_gate, v_b_gate), (v_a_up, v_b_up) = output
                # Extract at the generation position
                extracted['v_a_gate'] = v_a_gate[:, extraction_position:extraction_position+1].clone()
                extracted['v_b_gate'] = v_b_gate[:, extraction_position:extraction_position+1].clone()
                extracted['v_a_up'] = v_a_up[:, extraction_position:extraction_position+1].clone()
                extracted['v_b_up'] = v_b_up[:, extraction_position:extraction_position+1].clone()
        
        # Register hook
        layer = self.model.model.layers[self.active_layer]
        handle = layer.np_component.register_forward_hook(extraction_hook)
        
        # Forward pass
        with torch.no_grad():
            self.model(full_ids)
        
        # Remove hook
        handle.remove()
        
        # Create ModulationData object
        return ModulationData(
            v_a_gate=extracted['v_a_gate'],
            v_b_gate=extracted['v_b_gate'],
            v_a_up=extracted['v_a_up'],
            v_b_up=extracted['v_b_up'],
            magnitude=ModulationData._compute_magnitude(
                extracted['v_a_gate'], extracted['v_b_gate'],
                extracted['v_a_up'], extracted['v_b_up']
            ),
            context=prompt,
            target=target
        )
    
    def extract_baseline_modulation(self, prompt: str) -> ModulationData:
        """
        Extract baseline modulation without a specific answer.
        Uses padding token or a generic token as target.
        """
        # Use pad token as neutral target
        pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token
        return self.extract_modulation(prompt, pad_token)
    
    def extract_with_multiple_targets(self, prompt: str, targets: List[str]) -> Dict[str, ModulationData]:
        """Extract modulations for multiple targets to analyze patterns."""
        results = {}
        for target in targets:
            results[target] = self.extract_modulation(prompt, target)
        return results


class ModulationArithmeticAnalyzer:
    """Perform modulation arithmetic experiments."""
    
    def __init__(self, model, tokenizer, device, active_layer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.active_layer = active_layer
        self.extractor = ModulationArithmeticExtractor(model, tokenizer, device, active_layer)
    
    def inject_modulation(self, prompt: str, modulation: ModulationData) -> Dict[str, Any]:
        """Inject modulation and compute resulting probabilities."""
        # Tokenize prompt
        prompt_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        
        # Injection position (last token of prompt)
        injection_position = prompt_ids.shape[1] - 1
        
        # Storage for results
        logits_with_injection = None
        
        def injection_hook(module, input, output):
            if isinstance(output[0], tuple):
                (v_a_gate, v_b_gate), (v_a_up, v_b_up) = output
                
                # Clone
                v_a_gate_new = v_a_gate.clone()
                v_b_gate_new = v_b_gate.clone()
                v_a_up_new = v_a_up.clone()
                v_b_up_new = v_b_up.clone()
                
                # Inject at position
                v_a_gate_new[:, injection_position:injection_position+1] = modulation.v_a_gate
                v_b_gate_new[:, injection_position:injection_position+1] = modulation.v_b_gate
                v_a_up_new[:, injection_position:injection_position+1] = modulation.v_a_up
                v_b_up_new[:, injection_position:injection_position+1] = modulation.v_b_up
                
                return (v_a_gate_new, v_b_gate_new), (v_a_up_new, v_b_up_new)
            return output
        
        # Register hook
        layer = self.model.model.layers[self.active_layer]
        handle = layer.np_component.register_forward_hook(injection_hook)
        
        # Forward pass with injection
        with torch.no_grad():
            outputs = self.model(prompt_ids)
            logits_with_injection = outputs.logits if hasattr(outputs, 'logits') else outputs
        
        # Remove hook
        handle.remove()
        
        # Get baseline without injection
        with torch.no_grad():
            outputs_baseline = self.model(prompt_ids)
            logits_baseline = outputs_baseline.logits if hasattr(outputs_baseline, 'logits') else outputs_baseline
        
        # Compute probabilities
        probs_with_injection = F.softmax(logits_with_injection[0, -1], dim=-1)
        probs_baseline = F.softmax(logits_baseline[0, -1], dim=-1)
        
        return {
            'probs_with_injection': probs_with_injection,
            'probs_baseline': probs_baseline,
            'logits_with_injection': logits_with_injection[0, -1],
            'logits_baseline': logits_baseline[0, -1]
        }
    
    def run_arithmetic_experiment(
        self,
        source_prompt: str,
        source_answer: str,
        target_prompt: str,
        target_answer: str
    ) -> Dict[str, Any]:
        """
        Run full modulation arithmetic experiment.
        
        Steps:
        1. Extract source with answer (e.g., France -> Paris)
        2. Extract source baseline (France -> [PAD])
        3. Extract target baseline (Germany -> [PAD])
        4. Compute factual difference: (France->Paris) - (France->PAD)
        5. Apply to target: (Germany->PAD) + factual_difference
        6. Compare with simple replacement
        """
        
        logger.info("\n" + "="*80)
        logger.info("MODULATION ARITHMETIC EXPERIMENT")
        logger.info(f"Source: '{source_prompt}' -> '{source_answer}'")
        logger.info(f"Target: '{target_prompt}' -> '{target_answer}'")
        logger.info("="*80)
        
        # Step 1: Extract all needed modulations
        logger.info("\n1. Extracting modulations...")
        
        mod_source_answer = self.extractor.extract_modulation(source_prompt, source_answer)
        logger.info(f"  Source+Answer magnitude: {mod_source_answer.magnitude:.4f}")
        
        mod_source_baseline = self.extractor.extract_baseline_modulation(source_prompt)
        logger.info(f"  Source+Baseline magnitude: {mod_source_baseline.magnitude:.4f}")
        
        mod_target_baseline = self.extractor.extract_baseline_modulation(target_prompt)
        logger.info(f"  Target+Baseline magnitude: {mod_target_baseline.magnitude:.4f}")
        
        # Step 2: Compute factual difference
        logger.info("\n2. Computing factual difference...")
        factual_difference = mod_source_answer - mod_source_baseline
        logger.info(f"  Factual difference magnitude: {factual_difference.magnitude:.4f}")
        logger.info(f"  Relative magnitude: {factual_difference.magnitude / mod_source_answer.magnitude:.2%}")
        
        # Step 3: Create arithmetic combination
        logger.info("\n3. Creating arithmetic combination...")
        mod_arithmetic = mod_target_baseline + factual_difference
        logger.info(f"  Arithmetic result magnitude: {mod_arithmetic.magnitude:.4f}")
        
        # Step 4: Test different injection methods
        logger.info("\n4. Testing injection methods...")
        
        # Get target token IDs for analysis
        source_token_id = self.tokenizer(source_answer, add_special_tokens=False).input_ids[0]
        target_token_id = self.tokenizer(target_answer, add_special_tokens=False).input_ids[0]
        
        results = {}
        
        # Method 1: Simple replacement (baseline)
        logger.info("\n  Method 1: Simple Replacement")
        simple_result = self.inject_modulation(target_prompt, mod_source_answer)
        simple_source_prob = simple_result['probs_with_injection'][source_token_id].item()
        simple_target_prob = simple_result['probs_with_injection'][target_token_id].item()
        baseline_source_prob = simple_result['probs_baseline'][source_token_id].item()
        baseline_target_prob = simple_result['probs_baseline'][target_token_id].item()
        
        logger.info(f"    {source_answer}: {baseline_source_prob:.6f} -> {simple_source_prob:.6f} ({simple_source_prob/baseline_source_prob:.2f}x)")
        logger.info(f"    {target_answer}: {baseline_target_prob:.6f} -> {simple_target_prob:.6f} ({simple_target_prob/baseline_target_prob:.2f}x)")
        
        results['simple_replacement'] = {
            'source_prob_before': baseline_source_prob,
            'source_prob_after': simple_source_prob,
            'target_prob_before': baseline_target_prob,
            'target_prob_after': simple_target_prob
        }
        
        # Method 2: Arithmetic combination
        logger.info("\n  Method 2: Arithmetic Combination")
        arith_result = self.inject_modulation(target_prompt, mod_arithmetic)
        arith_source_prob = arith_result['probs_with_injection'][source_token_id].item()
        arith_target_prob = arith_result['probs_with_injection'][target_token_id].item()
        
        logger.info(f"    {source_answer}: {baseline_source_prob:.6f} -> {arith_source_prob:.6f} ({arith_source_prob/baseline_source_prob:.2f}x)")
        logger.info(f"    {target_answer}: {baseline_target_prob:.6f} -> {arith_target_prob:.6f} ({arith_target_prob/baseline_target_prob:.2f}x)")
        
        results['arithmetic'] = {
            'source_prob_before': baseline_source_prob,
            'source_prob_after': arith_source_prob,
            'target_prob_before': baseline_target_prob,
            'target_prob_after': arith_target_prob
        }
        
        # Method 3: Scaled arithmetic (to match magnitude)
        logger.info("\n  Method 3: Scaled Arithmetic")
        scale_factor = mod_source_answer.magnitude / mod_arithmetic.magnitude
        mod_scaled = mod_arithmetic * scale_factor
        scaled_result = self.inject_modulation(target_prompt, mod_scaled)
        scaled_source_prob = scaled_result['probs_with_injection'][source_token_id].item()
        scaled_target_prob = scaled_result['probs_with_injection'][target_token_id].item()
        
        logger.info(f"    Scale factor: {scale_factor:.2f}")
        logger.info(f"    {source_answer}: {baseline_source_prob:.6f} -> {scaled_source_prob:.6f} ({scaled_source_prob/baseline_source_prob:.2f}x)")
        logger.info(f"    {target_answer}: {baseline_target_prob:.6f} -> {scaled_target_prob:.6f} ({scaled_target_prob/baseline_target_prob:.2f}x)")
        
        results['scaled_arithmetic'] = {
            'source_prob_before': baseline_source_prob,
            'source_prob_after': scaled_source_prob,
            'target_prob_before': baseline_target_prob,
            'target_prob_after': scaled_target_prob,
            'scale_factor': scale_factor
        }
        
        # Method 4: Just the factual difference (no target baseline)
        logger.info("\n  Method 4: Pure Factual Difference")
        pure_result = self.inject_modulation(target_prompt, factual_difference)
        pure_source_prob = pure_result['probs_with_injection'][source_token_id].item()
        pure_target_prob = pure_result['probs_with_injection'][target_token_id].item()
        
        logger.info(f"    {source_answer}: {baseline_source_prob:.6f} -> {pure_source_prob:.6f} ({pure_source_prob/baseline_source_prob:.2f}x)")
        logger.info(f"    {target_answer}: {baseline_target_prob:.6f} -> {pure_target_prob:.6f} ({pure_target_prob/baseline_target_prob:.2f}x)")
        
        results['pure_difference'] = {
            'source_prob_before': baseline_source_prob,
            'source_prob_after': pure_source_prob,
            'target_prob_before': baseline_target_prob,
            'target_prob_after': pure_target_prob
        }
        
        # Analysis
        logger.info("\n5. Analysis:")
        logger.info("  Which method increases source probability most?")
        
        source_increases = {
            'Simple': simple_source_prob / baseline_source_prob,
            'Arithmetic': arith_source_prob / baseline_source_prob,
            'Scaled': scaled_source_prob / baseline_source_prob,
            'Pure': pure_source_prob / baseline_source_prob
        }
        
        best_method = max(source_increases, key=source_increases.get)
        logger.info(f"    Best: {best_method} ({source_increases[best_method]:.2f}x increase)")
        
        # Get top-5 predictions for each method
        for method_name, method_result in [
            ('Simple', simple_result),
            ('Arithmetic', arith_result),
            ('Scaled', scaled_result),
            ('Pure', pure_result)
        ]:
            top_5_probs, top_5_indices = torch.topk(method_result['probs_with_injection'], 5)
            top_5_tokens = [self.tokenizer.decode([idx]) for idx in top_5_indices]
            logger.info(f"\n  {method_name} top-5: {top_5_tokens}")
        
        return {
            'source_prompt': source_prompt,
            'source_answer': source_answer,
            'target_prompt': target_prompt,
            'target_answer': target_answer,
            'results': results,
            'modulation_magnitudes': {
                'source_answer': mod_source_answer.magnitude,
                'source_baseline': mod_source_baseline.magnitude,
                'target_baseline': mod_target_baseline.magnitude,
                'factual_difference': factual_difference.magnitude,
                'arithmetic_result': mod_arithmetic.magnitude
            }
        }


def load_model_selective(checkpoint_path: str, model_name: str, layer: int, device: str):
    """Load model with only specified layer as NPT."""
    from transformers import AutoConfig

    logger.info(f"Loading model with only layer {layer} as NPT")

    # Load base model
    model_config = AutoConfig.from_pretrained(model_name)
    model_config._attn_implementation = "eager"
    model = NPTLlamaModel.from_pretrained(model_name, config=model_config)

    # Load checkpoint
    checkpoint_path = Path(checkpoint_path)
    npt_weights_path = checkpoint_path / "npt_weights.pt"

    if npt_weights_path.exists():
        state_dict = torch.load(npt_weights_path, map_location='cpu')

        # Detect dual modulation
        dual_modulation = any('W_down_gate' in k for k in state_dict.keys())

        # Convert only specified layer
        npt_config = NPTConfig(
            layers_to_convert=[layer],
            np_rank=256,
            np_init_scale=0.001,
            single_layer_mode=False,
            num_ranks=4,
            init_strategy="improved",
            dual_modulation=dual_modulation
        )

        model.convert_to_npt(npt_config)

        # Load weights for this layer only
        filtered_state_dict = {}
        for key, value in state_dict.items():
            if f'layer_{layer}_np' in key:
                new_key = key.replace(f'layer_{layer}_np', f'model.layers.{layer}.np_component')
                filtered_state_dict[new_key] = value

        model.load_state_dict(filtered_state_dict, strict=False)
        logger.info(f"Loaded NPT weights for layer {layer}")

    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Modulation Arithmetic Experiment")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--layer", type=int, default=15, help="NPT layer to use")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(42)
    np.random.seed(42)

    # Load model
    device = torch.device(args.device)
    model, tokenizer = load_model_selective(args.checkpoint, args.model_name, args.layer, args.device)

    # Create analyzer
    analyzer = ModulationArithmeticAnalyzer(model, tokenizer, device, args.layer)

    # Define experiments
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
            'source_prompt': "Water boils at",
            'source_answer': " 100",
            'target_prompt': "Water freezes at",
            'target_answer': " 0"
        },
        {
            'source_prompt': "Einstein developed the theory of",
            'source_answer': " relativity",
            'target_prompt': "Newton developed the theory of",
            'target_answer': " gravity"
        }
    ]

    # Run experiments
    all_results = []

    for exp in experiments:
        results = analyzer.run_arithmetic_experiment(
            source_prompt=exp['source_prompt'],
            source_answer=exp['source_answer'],
            target_prompt=exp['target_prompt'],
            target_answer=exp['target_answer']
        )
        all_results.append(results)

    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)

    for result in all_results:
        logger.info(f"\n{result['source_prompt']} -> {result['target_prompt']}")

        # Compare methods
        methods = ['simple_replacement', 'arithmetic', 'scaled_arithmetic', 'pure_difference']
        source_answer = result['source_answer']

        for method in methods:
            method_result = result['results'][method]
            source_increase = method_result['source_prob_after'] / method_result['source_prob_before']
            logger.info(f"  {method:20s}: {source_answer} prob x{source_increase:.2f}")

    # Save results
    output_dir = Path("experiments/modulation_arithmetic")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()