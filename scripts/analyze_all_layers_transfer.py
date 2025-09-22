#!/usr/bin/env python3
"""
Comprehensive Layer-by-Layer Factual Transfer Analysis

This script tests each NPT layer individually to understand which layers
perform literal token transfer vs semantic operation transfer.
"""

import argparse
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, asdict
import json
import pandas as pd
from tqdm import tqdm
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.npt import NPTLlamaModel, NPTConfig
from transformers import AutoTokenizer, AutoConfig
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


@dataclass
class LayerTransferResult:
    """Results for a single layer transfer experiment."""
    layer_idx: int
    source_prompt: str
    source_answer: str
    target_prompt: str
    target_answer: str

    # Probability changes
    source_token_prob_before: float
    source_token_prob_after: float
    source_token_change_pct: float  # This is the key metric

    target_token_prob_before: float
    target_token_prob_after: float
    target_token_change_pct: float

    # Additional metrics
    kl_divergence: float
    top10_changes: int
    modulation_magnitude: float

    # Top predictions
    top3_before: List[str]
    top3_after: List[str]

    def to_dict(self):
        return asdict(self)


class LayerByLayerAnalyzer:
    """Analyze factual transfer for each NPT layer individually."""

    def __init__(self, checkpoint_path: str, model_name: str, device: str = "cuda"):
        self.checkpoint_path = Path(checkpoint_path)
        self.model_name = model_name
        self.device = torch.device(device)

        # Load tokenizer once
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Detect available layers
        self.available_layers = self._detect_available_layers()
        logger.info(f"Found {len(self.available_layers)} NPT layers: {sorted(self.available_layers)}")

    def _detect_available_layers(self) -> Set[int]:
        """Detect which NPT layers are available in checkpoint."""
        npt_weights_path = self.checkpoint_path / "npt_weights.pt"
        if not npt_weights_path.exists():
            raise FileNotFoundError(f"NPT weights not found at {npt_weights_path}")

        state_dict = torch.load(npt_weights_path, map_location='cpu')
        available_layers = set()

        for key in state_dict.keys():
            if 'layer_' in key and '_np' in key:
                parts = key.split('_')
                for i, part in enumerate(parts):
                    if part == 'layer' and i + 1 < len(parts) and parts[i + 1].isdigit():
                        available_layers.add(int(parts[i + 1]))

        # Detect if dual modulation
        self.dual_modulation = any('W_down_gate' in k for k in state_dict.keys())

        return available_layers

    def _load_model_with_layer(self, layer_idx: int) -> NPTLlamaModel:
        """Load model with only specified layer as NPT."""
        # Create base model
        model_config = AutoConfig.from_pretrained(self.model_name)
        model_config._attn_implementation = "eager"
        model = NPTLlamaModel.from_pretrained(self.model_name, config=model_config)

        # Create NPT config for single layer
        npt_config = NPTConfig(
            layers_to_convert=[layer_idx],
            np_rank=256,
            np_init_scale=0.001,
            single_layer_mode=False,
            num_ranks=4,
            init_strategy="improved",
            dual_modulation=self.dual_modulation
        )

        # Convert layer
        model.convert_to_npt(npt_config)

        # Load weights for this layer
        npt_weights_path = self.checkpoint_path / "npt_weights.pt"
        state_dict = torch.load(npt_weights_path, map_location='cpu')

        # Filter for this layer only
        filtered_state_dict = {}
        for key, value in state_dict.items():
            if f'layer_{layer_idx}_np' in key:
                new_key = key.replace(f'layer_{layer_idx}_np', f'model.layers.{layer_idx}.np_component')
                filtered_state_dict[new_key] = value

        model.load_state_dict(filtered_state_dict, strict=False)
        model = model.to(self.device)
        model.eval()

        return model

    def extract_modulation(self, model: NPTLlamaModel, prompt: str, answer: str, layer_idx: int) -> Dict:
        """Extract modulation from specified layer."""
        # Tokenize
        prompt_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        answer_ids = self.tokenizer(answer, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)

        # Full sequence
        full_ids = torch.cat([prompt_ids, answer_ids[:, :1]], dim=1)

        # Position to extract from (last prompt token)
        extraction_position = prompt_ids.shape[1] - 1

        modulation_data = {}

        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) == 2:
                if isinstance(output[0], tuple):
                    # Dual modulation
                    (v_a_gate, v_b_gate), (v_a_up, v_b_up) = output
                    modulation_data['v_a_gate'] = v_a_gate[:, extraction_position:extraction_position+1].clone()
                    modulation_data['v_b_gate'] = v_b_gate[:, extraction_position:extraction_position+1].clone()
                    modulation_data['v_a_up'] = v_a_up[:, extraction_position:extraction_position+1].clone()
                    modulation_data['v_b_up'] = v_b_up[:, extraction_position:extraction_position+1].clone()
                    modulation_data['magnitude'] = (
                        v_a_gate.norm().item() + v_b_gate.norm().item() +
                        v_a_up.norm().item() + v_b_up.norm().item()
                    ) / 4
                else:
                    # Single modulation
                    v_a, v_b = output
                    modulation_data['v_a'] = v_a[:, extraction_position:extraction_position+1].clone()
                    modulation_data['v_b'] = v_b[:, extraction_position:extraction_position+1].clone()
                    modulation_data['magnitude'] = (v_a.norm().item() + v_b.norm().item()) / 2

        # Register hook
        handle = model.model.layers[layer_idx].np_component.register_forward_hook(hook)

        # Forward pass
        with torch.no_grad():
            model.model.layers[layer_idx].set_npt_mode(True)
            _ = model(full_ids)

        handle.remove()
        return modulation_data

    def inject_and_compute(self, model: NPTLlamaModel, prompt: str, modulation: Dict, layer_idx: int) -> Dict:
        """Inject modulation and compute probabilities."""
        # Tokenize prompt
        prompt_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)

        # Injection position (last token)
        injection_position = prompt_ids.shape[1] - 1

        # First get baseline (no injection)
        with torch.no_grad():
            model.model.layers[layer_idx].set_npt_mode(True)
            outputs = model(prompt_ids)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs

        baseline_logits = logits[0, -1]
        baseline_probs = F.softmax(baseline_logits, dim=-1)

        # Now inject modulation
        def injection_hook(module, input, output):
            if isinstance(output, tuple) and len(output) == 2:
                if isinstance(output[0], tuple) and 'v_a_gate' in modulation:
                    # Dual modulation
                    (v_a_gate, v_b_gate), (v_a_up, v_b_up) = output
                    v_a_gate_new = v_a_gate.clone()
                    v_b_gate_new = v_b_gate.clone()
                    v_a_up_new = v_a_up.clone()
                    v_b_up_new = v_b_up.clone()

                    v_a_gate_new[:, injection_position:injection_position+1] = modulation['v_a_gate']
                    v_b_gate_new[:, injection_position:injection_position+1] = modulation['v_b_gate']
                    v_a_up_new[:, injection_position:injection_position+1] = modulation['v_a_up']
                    v_b_up_new[:, injection_position:injection_position+1] = modulation['v_b_up']

                    return (v_a_gate_new, v_b_gate_new), (v_a_up_new, v_b_up_new)
                elif 'v_a' in modulation:
                    # Single modulation
                    v_a, v_b = output
                    v_a_new = v_a.clone()
                    v_b_new = v_b.clone()

                    v_a_new[:, injection_position:injection_position+1] = modulation['v_a']
                    v_b_new[:, injection_position:injection_position+1] = modulation['v_b']

                    return v_a_new, v_b_new
            return output

        # Register injection hook
        handle = model.model.layers[layer_idx].np_component.register_forward_hook(injection_hook)

        # Forward pass with injection
        with torch.no_grad():
            outputs = model(prompt_ids)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs

        handle.remove()

        injected_logits = logits[0, -1]
        injected_probs = F.softmax(injected_logits, dim=-1)

        return {
            'baseline_probs': baseline_probs,
            'injected_probs': injected_probs,
            'baseline_logits': baseline_logits,
            'injected_logits': injected_logits
        }

    def analyze_single_layer(
        self,
        layer_idx: int,
        source_prompt: str,
        source_answer: str,
        target_prompt: str,
        target_answer: str
    ) -> LayerTransferResult:
        """Analyze transfer for a single layer."""

        # Load model with this layer only
        model = self._load_model_with_layer(layer_idx)

        # Extract modulation from source
        modulation = self.extract_modulation(model, source_prompt, source_answer, layer_idx)

        # Inject into target and compute
        results = self.inject_and_compute(model, target_prompt, modulation, layer_idx)

        # Get token IDs
        source_token_id = self.tokenizer(source_answer, add_special_tokens=False).input_ids[0]
        target_token_id = self.tokenizer(target_answer, add_special_tokens=False).input_ids[0]

        # Calculate probability changes
        source_prob_before = results['baseline_probs'][source_token_id].item()
        source_prob_after = results['injected_probs'][source_token_id].item()
        source_change_pct = ((source_prob_after - source_prob_before) / (source_prob_before + 1e-10)) * 100

        target_prob_before = results['baseline_probs'][target_token_id].item()
        target_prob_after = results['injected_probs'][target_token_id].item()
        target_change_pct = ((target_prob_after - target_prob_before) / (target_prob_before + 1e-10)) * 100

        # KL divergence
        kl_div = F.kl_div(
            torch.log(results['injected_probs'] + 1e-10),
            results['baseline_probs'],
            reduction='sum'
        ).item()

        # Top-k analysis
        top10_before = torch.topk(results['baseline_probs'], 10)[1]
        top10_after = torch.topk(results['injected_probs'], 10)[1]
        top10_changes = len(set(top10_before.tolist()) - set(top10_after.tolist()))

        # Top 3 tokens
        top3_before_idx = torch.topk(results['baseline_probs'], 3)[1]
        top3_after_idx = torch.topk(results['injected_probs'], 3)[1]
        top3_before = [self.tokenizer.decode([idx]) for idx in top3_before_idx]
        top3_after = [self.tokenizer.decode([idx]) for idx in top3_after_idx]

        # Clean up model
        del model
        torch.cuda.empty_cache()

        return LayerTransferResult(
            layer_idx=layer_idx,
            source_prompt=source_prompt,
            source_answer=source_answer,
            target_prompt=target_prompt,
            target_answer=target_answer,
            source_token_prob_before=source_prob_before,
            source_token_prob_after=source_prob_after,
            source_token_change_pct=source_change_pct,
            target_token_prob_before=target_prob_before,
            target_token_prob_after=target_prob_after,
            target_token_change_pct=target_change_pct,
            kl_divergence=kl_div,
            top10_changes=top10_changes,
            modulation_magnitude=modulation['magnitude'],
            top3_before=top3_before,
            top3_after=top3_after
        )

    def analyze_all_layers(self, experiments: List[Dict]) -> Dict[str, List[LayerTransferResult]]:
        """Analyze all layers for all experiments."""
        all_results = {}

        for exp_idx, exp in enumerate(experiments):
            logger.info(f"\n{'='*80}")
            logger.info(f"Experiment {exp_idx + 1}: {exp['source_prompt'][:30]}... -> {exp['target_prompt'][:30]}...")
            logger.info(f"{'='*80}")

            exp_key = f"exp_{exp_idx + 1}"
            exp_results = []

            # Test each layer
            for layer_idx in tqdm(sorted(self.available_layers), desc="Testing layers"):
                try:
                    result = self.analyze_single_layer(
                        layer_idx=layer_idx,
                        source_prompt=exp['source_prompt'],
                        source_answer=exp['source_answer'],
                        target_prompt=exp['target_prompt'],
                        target_answer=exp['target_answer']
                    )
                    exp_results.append(result)

                    # Log quick summary
                    logger.info(f"  Layer {layer_idx:2d}: Source token ({exp['source_answer'].strip()}) "
                              f"change: {result.source_token_change_pct:+7.1f}% | "
                              f"Target token ({exp['target_answer'].strip()}) "
                              f"change: {result.target_token_change_pct:+7.1f}%")

                except Exception as e:
                    logger.error(f"  Layer {layer_idx}: Failed - {str(e)}")

            all_results[exp_key] = exp_results

        return all_results

    def generate_report(self, results: Dict[str, List[LayerTransferResult]], output_dir: Path):
        """Generate comprehensive report of results."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save raw results as JSON
        json_data = {}
        for exp_key, exp_results in results.items():
            json_data[exp_key] = [r.to_dict() for r in exp_results]

        with open(output_dir / 'raw_results.json', 'w') as f:
            json.dump(json_data, f, indent=2)

        # Generate summary CSV
        summary_data = []
        for exp_key, exp_results in results.items():
            for result in exp_results:
                summary_data.append({
                    'Experiment': exp_key,
                    'Layer': result.layer_idx,
                    'Source_Token': result.source_answer.strip(),
                    'Target_Token': result.target_answer.strip(),
                    'Source_Change_%': round(result.source_token_change_pct, 2),
                    'Target_Change_%': round(result.target_token_change_pct, 2),
                    'KL_Divergence': round(result.kl_divergence, 4),
                    'Top10_Changes': result.top10_changes,
                    'Modulation_Magnitude': round(result.modulation_magnitude, 2)
                })

        df = pd.DataFrame(summary_data)
        df.to_csv(output_dir / 'summary.csv', index=False)

        # Generate detailed text report
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("LAYER-BY-LAYER FACTUAL TRANSFER ANALYSIS REPORT")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("="*80)
        report_lines.append("")

        # Summary statistics
        report_lines.append("SUMMARY STATISTICS")
        report_lines.append("-"*40)

        for exp_key, exp_results in results.items():
            if not exp_results:
                continue

            report_lines.append(f"\n{exp_key.upper()}:")
            report_lines.append(f"  Source: {exp_results[0].source_prompt[:50]}...")
            report_lines.append(f"  Target: {exp_results[0].target_prompt[:50]}...")
            report_lines.append(f"  Source Token: {exp_results[0].source_answer.strip()}")
            report_lines.append(f"  Target Token: {exp_results[0].target_answer.strip()}")
            report_lines.append("")

            # Find best layers for literal transfer
            sorted_by_source = sorted(exp_results, key=lambda x: x.source_token_change_pct, reverse=True)
            report_lines.append("  Top 3 layers for LITERAL transfer (source token increase):")
            for i, r in enumerate(sorted_by_source[:3]):
                report_lines.append(f"    {i+1}. Layer {r.layer_idx}: {r.source_token_change_pct:+.1f}%")

            # Find best layers for semantic transfer
            sorted_by_target = sorted(exp_results, key=lambda x: x.target_token_change_pct, reverse=True)
            report_lines.append("\n  Top 3 layers for SEMANTIC transfer (target token increase):")
            for i, r in enumerate(sorted_by_target[:3]):
                report_lines.append(f"    {i+1}. Layer {r.layer_idx}: {r.target_token_change_pct:+.1f}%")

        report_lines.append("\n" + "="*80)
        report_lines.append("DETAILED LAYER ANALYSIS")
        report_lines.append("="*80)

        # Group by layer to see patterns
        layer_patterns = {}
        for exp_key, exp_results in results.items():
            for result in exp_results:
                if result.layer_idx not in layer_patterns:
                    layer_patterns[result.layer_idx] = []
                layer_patterns[result.layer_idx].append({
                    'exp': exp_key,
                    'source_change': result.source_token_change_pct,
                    'target_change': result.target_token_change_pct,
                    'kl_div': result.kl_divergence
                })

        for layer_idx in sorted(layer_patterns.keys()):
            report_lines.append(f"\nLAYER {layer_idx}:")
            report_lines.append("-"*40)

            patterns = layer_patterns[layer_idx]
            avg_source_change = np.mean([p['source_change'] for p in patterns])
            avg_target_change = np.mean([p['target_change'] for p in patterns])
            avg_kl = np.mean([p['kl_div'] for p in patterns])

            report_lines.append(f"  Average source token change: {avg_source_change:+.1f}%")
            report_lines.append(f"  Average target token change: {avg_target_change:+.1f}%")
            report_lines.append(f"  Average KL divergence: {avg_kl:.4f}")

            # Classification
            if avg_source_change > 10:
                transfer_type = "LITERAL TRANSFER"
            elif avg_target_change > 10:
                transfer_type = "SEMANTIC TRANSFER"
            else:
                transfer_type = "MINIMAL EFFECT"

            report_lines.append(f"  Classification: {transfer_type}")

        # Write report
        report_text = '\n'.join(report_lines)
        with open(output_dir / 'report.txt', 'w') as f:
            f.write(report_text)

        # Also print to console
        print(report_text)

        logger.info(f"\n📊 Reports saved to {output_dir}")
        logger.info(f"  - raw_results.json: Complete data")
        logger.info(f"  - summary.csv: Tabular summary")
        logger.info(f"  - report.txt: Detailed analysis")


def main():
    parser = argparse.ArgumentParser(description="Layer-by-layer factual transfer analysis")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to NPT checkpoint")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="experiments/layer_analysis")
    parser.add_argument("--use_context", action="store_true", help="Use contextual prompts")

    args = parser.parse_args()

    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Initialize analyzer
    analyzer = LayerByLayerAnalyzer(
        checkpoint_path=args.checkpoint,
        model_name=args.model_name,
        device=args.device
    )

    # Define experiments
    if args.use_context:
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
    else:
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

    # Run analysis
    logger.info(f"🔍 Analyzing {len(analyzer.available_layers)} NPT layers...")
    results = analyzer.analyze_all_layers(experiments)

    # Generate reports
    output_dir = Path(args.output_dir)
    if args.use_context:
        output_dir = output_dir / "with_context"
    else:
        output_dir = output_dir / "without_context"

    analyzer.generate_report(results, output_dir)


if __name__ == "__main__":
    main()