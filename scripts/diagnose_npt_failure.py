#!/usr/bin/env python3
"""
Diagnostic script to understand why multiple NPT layers fail.

Core hypothesis: Each NPT layer loses its attention information because
the modulation (v_b * v_a.T) is too weak to compensate for the missing
attention residual connection.

This script measures:
1. Modulation compensation ratio (modulation magnitude vs attention magnitude)
2. Whether v_a actually encodes attention information
3. Cumulative degradation with multiple NPT layers
4. Detection of "dead" layers with near-zero modulation
5. Whether amplifying modulation can fix the problem
"""

import argparse
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from colorama import init, Fore, Style

warnings.filterwarnings("ignore", category=FutureWarning)

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.npt import NPTLlamaModel, NPTConfig
from transformers import AutoTokenizer
import math

init(autoreset=True)


@dataclass
class LayerDiagnostics:
    """Diagnostics for a single NPT layer."""
    layer_idx: int
    attention_magnitude: float
    modulation_magnitude: float
    compensation_ratio: float
    v_a_magnitude: float
    v_b_magnitude: float
    v_a_attention_correlation: float
    is_dead: bool
    is_encoding_attention: bool
    

@dataclass
class CumulativeDiagnostics:
    """Diagnostics for cumulative NPT layer effects."""
    num_npt_layers: int
    layer_indices: List[int]
    hidden_state_drift: float
    perplexity: float
    generation_quality: str  # "good", "degraded", "collapsed"


class NPTFailureDiagnostics:
    """Comprehensive diagnostics for NPT layer failures."""
    
    def __init__(self, model, tokenizer, checkpoint_path: Path, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.checkpoint_path = checkpoint_path
        self.device = device
        
        # Move model to device
        self.model = self.model.to(device)
        self.model.eval()
        
        # Storage for diagnostics
        self.layer_diagnostics = {}
        self.cumulative_diagnostics = []
        
        # Test prompt for consistency
        self.test_prompt = "The future of artificial intelligence is"
        
    def measure_modulation_compensation(self, layer_idx: int) -> LayerDiagnostics:
        """
        Measure if modulation compensates for lost attention.
        
        Key measurement: ||modulation|| / ||attention||
        If << 1, the layer loses information.
        """
        print(f"\n{Fore.YELLOW}Analyzing Layer {layer_idx}...{Style.RESET_ALL}")
        
        # Ensure layer is in NPT mode
        if layer_idx not in self.model.npt_layers:
            print(f"  Layer {layer_idx} is not an NPT layer, skipping")
            return None
            
        npt_layer = self.model.npt_layers[layer_idx]
        npt_layer.set_npt_mode(True)
        
        # Storage for captured values
        captured = {}
        
        def capture_hook(module, input, output):
            """Capture v_a and v_b from NPComponent."""
            if isinstance(output, tuple) and len(output) == 2:
                captured['v_a'] = output[0].detach()
                captured['v_b'] = output[1].detach()
            return output
        
        # Install hook
        hook = npt_layer.np_component.register_forward_hook(capture_hook)
        
        # Process test prompt
        inputs = self.tokenizer(self.test_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            # Run to NPT layer to get hidden states before it
            hidden_states = self.model.model.embed_tokens(inputs.input_ids)
            
            # Create position embeddings
            batch_size, seq_len = inputs.input_ids.shape
            head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
            cos = torch.ones(batch_size, seq_len, head_dim, 
                           dtype=hidden_states.dtype, device=hidden_states.device)
            sin = torch.zeros(batch_size, seq_len, head_dim,
                            dtype=hidden_states.dtype, device=hidden_states.device)
            position_embeddings = (cos, sin)
            
            # Process through layers up to NPT layer
            for i in range(layer_idx):
                layer = self.model.model.layers[i]
                layer_out = layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    use_cache=False,
                    output_attentions=False
                )
                hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out
            
            # Store hidden states before NPT layer
            hidden_before = hidden_states.clone()
            
            # Get attention output from NPT layer (in standard mode temporarily)
            npt_layer.set_npt_mode(False)
            residual = hidden_states
            hidden_states_norm = npt_layer.input_layernorm(hidden_states)
            attn_output = npt_layer.self_attn(
                hidden_states=hidden_states_norm,
                attention_mask=None,
                position_embeddings=position_embeddings,
                past_key_values=None,
                cache_position=None,
                use_cache=False,
                output_attentions=False
            )[0]
            
            # Switch back to NPT mode and process
            npt_layer.set_npt_mode(True)
            _ = npt_layer(
                hidden_before,
                position_embeddings=position_embeddings,
                use_cache=False,
                output_attentions=False
            )
        
        hook.remove()
        
        # Extract v_a, v_b
        v_a = captured.get('v_a')
        v_b = captured.get('v_b')
        
        if v_a is None or v_b is None:
            print(f"  Failed to capture v_a, v_b")
            return None
        
        # Compute magnitudes
        attention_magnitude = attn_output.norm().item()
        v_a_magnitude = v_a.norm().item()
        v_b_magnitude = v_b.norm().item()
        
        # Compute modulation effect magnitude
        # Modulation applied as: v_b * (v_a @ h)
        v_a_dot_h = torch.sum(v_a * hidden_before, dim=-1, keepdim=True)
        modulation = v_b * v_a_dot_h
        modulation_magnitude = modulation.norm().item()
        
        # Compensation ratio - KEY METRIC
        compensation_ratio = modulation_magnitude / (attention_magnitude + 1e-8)
        
        # Compute v_a attention correlation
        v_a_flat = v_a.reshape(-1, v_a.shape[-1])
        attn_flat = attn_output.reshape(-1, attn_output.shape[-1])
        
        # Sample subset for correlation if too large
        if v_a_flat.shape[0] > 100:
            indices = torch.randperm(v_a_flat.shape[0])[:100]
            v_a_flat = v_a_flat[indices]
            attn_flat = attn_flat[indices]
        
        v_a_norm = F.normalize(v_a_flat, p=2, dim=-1)
        attn_norm = F.normalize(attn_flat, p=2, dim=-1)
        correlation = (v_a_norm * attn_norm).sum(dim=-1).mean().item()
        
        # Determine if layer is dead or not encoding
        is_dead = v_a_magnitude < 0.1 and v_b_magnitude < 0.1
        is_encoding_attention = correlation > 0.3
        
        diagnostics = LayerDiagnostics(
            layer_idx=layer_idx,
            attention_magnitude=attention_magnitude,
            modulation_magnitude=modulation_magnitude,
            compensation_ratio=compensation_ratio,
            v_a_magnitude=v_a_magnitude,
            v_b_magnitude=v_b_magnitude,
            v_a_attention_correlation=correlation,
            is_dead=is_dead,
            is_encoding_attention=is_encoding_attention
        )
        
        # Print immediate results
        status = "PASS" if compensation_ratio > 0.5 else "FAIL"
        encoding_status = "ENCODING" if is_encoding_attention else "NOT ENCODING"
        
        print(f"  Attention magnitude: {attention_magnitude:.2f}")
        print(f"  Modulation magnitude: {modulation_magnitude:.2f}")
        print(f"  Compensation ratio: {compensation_ratio:.3f} [{status}]")
        print(f"  v_a attention correlation: {correlation:.3f} [{encoding_status}]")
        
        if is_dead:
            print(f"  {Fore.RED}WARNING: Layer appears dead (near-zero modulation){Style.RESET_ALL}")
        
        return diagnostics
    
    def measure_cumulative_degradation(self, layer_combinations: List[List[int]]) -> List[CumulativeDiagnostics]:
        """
        Measure how model degrades with different NPT layer combinations.
        """
        print(f"\n{Fore.CYAN}Measuring Cumulative Degradation...{Style.RESET_ALL}")
        
        results = []
        
        # First get baseline with all layers in standard mode
        print(f"\n{Fore.YELLOW}Baseline: All layers standard{Style.RESET_ALL}")
        for idx in self.model.npt_layers:
            self.model.npt_layers[idx].set_npt_mode(False)
        
        inputs = self.tokenizer(self.test_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            baseline_outputs = self.model(**inputs)
            baseline_hidden = baseline_outputs.hidden_states if hasattr(baseline_outputs, 'hidden_states') else None
            baseline_logits = baseline_outputs.logits
            
            # Compute baseline perplexity
            loss = F.cross_entropy(
                baseline_logits[0, :-1, :],
                inputs.input_ids[0, 1:],
                reduction='mean'
            )
            baseline_perplexity = torch.exp(loss).item()
        
        print(f"  Baseline perplexity: {baseline_perplexity:.2f}")
        
        # Test each combination
        for combination in layer_combinations:
            print(f"\n{Fore.YELLOW}Testing NPT layers: {combination}{Style.RESET_ALL}")
            
            # Set all to standard first
            for idx in self.model.npt_layers:
                self.model.npt_layers[idx].set_npt_mode(False)
            
            # Enable NPT for specified layers
            for idx in combination:
                if idx in self.model.npt_layers:
                    self.model.npt_layers[idx].set_npt_mode(True)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Compute perplexity
                loss = F.cross_entropy(
                    logits[0, :-1, :],
                    inputs.input_ids[0, 1:],
                    reduction='mean'
                )
                perplexity = torch.exp(loss).item()
                
                # Compute hidden state drift if possible
                if baseline_logits is not None:
                    drift = (logits - baseline_logits).norm().item()
                else:
                    drift = 0.0
                
                # Determine quality based on perplexity degradation
                perplexity_ratio = perplexity / baseline_perplexity
                if perplexity_ratio < 2:
                    quality = "good"
                elif perplexity_ratio < 10:
                    quality = "degraded"
                else:
                    quality = "collapsed"
                
                # Generate a sample to check quality
                gen_inputs = self.tokenizer(self.test_prompt, return_tensors="pt").to(self.device)
                gen_outputs = self.model.generate(
                    gen_inputs.input_ids,
                    max_new_tokens=10,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                generated = self.tokenizer.decode(gen_outputs[0], skip_special_tokens=True)
                
                print(f"  Perplexity: {perplexity:.2f} (ratio: {perplexity_ratio:.2f}x)")
                print(f"  Hidden drift: {drift:.2f}")
                print(f"  Quality: {quality}")
                print(f"  Sample: {generated[:100]}")
                
                results.append(CumulativeDiagnostics(
                    num_npt_layers=len(combination),
                    layer_indices=combination,
                    hidden_state_drift=drift,
                    perplexity=perplexity,
                    generation_quality=quality
                ))
        
        return results
    
    def test_modulation_amplification(self, layer_idx: int, scale_factors: List[float] = [1, 2, 5, 10]):
        """
        Test if amplifying modulation can recover performance.
        """
        print(f"\n{Fore.CYAN}Testing Modulation Amplification for Layer {layer_idx}...{Style.RESET_ALL}")
        
        if layer_idx not in self.model.npt_layers:
            print(f"  Layer {layer_idx} is not an NPT layer")
            return
        
        npt_layer = self.model.npt_layers[layer_idx]
        npt_layer.set_npt_mode(True)
        
        # Storage for original modulation
        original_v_a = None
        original_v_b = None
        scale_to_test = None
        
        def amplify_hook(module, input, output):
            """Amplify v_a and v_b by scale factor."""
            nonlocal original_v_a, original_v_b
            
            if isinstance(output, tuple) and len(output) == 2:
                v_a, v_b = output
                
                # Store originals
                if original_v_a is None:
                    original_v_a = v_a.clone()
                    original_v_b = v_b.clone()
                
                # Apply scaling
                if scale_to_test is not None and scale_to_test != 1:
                    v_a_scaled = v_a * scale_to_test
                    v_b_scaled = v_b * scale_to_test
                    return (v_a_scaled, v_b_scaled)
            
            return output
        
        # Install hook
        hook = npt_layer.np_component.register_forward_hook(amplify_hook)
        
        inputs = self.tokenizer(self.test_prompt, return_tensors="pt").to(self.device)
        
        results = {}
        for scale in scale_factors:
            scale_to_test = scale
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Compute perplexity
                loss = F.cross_entropy(
                    logits[0, :-1, :],
                    inputs.input_ids[0, 1:],
                    reduction='mean'
                )
                perplexity = torch.exp(loss).item()
                
                # Generate sample
                gen_outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=10,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                generated = self.tokenizer.decode(gen_outputs[0], skip_special_tokens=True)
                
                results[scale] = {
                    'perplexity': perplexity,
                    'sample': generated[:100]
                }
                
                print(f"\n  Scale {scale}x:")
                print(f"    Perplexity: {perplexity:.2f}")
                print(f"    Sample: {generated[:80]}")
        
        hook.remove()
        
        # Check if amplification helps
        if results[1]['perplexity'] > 100 and results[10]['perplexity'] < results[1]['perplexity'] / 2:
            print(f"\n  {Fore.GREEN}✓ Amplification helps! Modulation exists but is too weak.{Style.RESET_ALL}")
        else:
            print(f"\n  {Fore.YELLOW}⚠ Amplification has limited effect. Problem may be deeper.{Style.RESET_ALL}")
        
        return results
    
    def generate_report(self):
        """Generate comprehensive diagnostic report."""
        print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}DIAGNOSIS REPORT{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
        
        # Layer-by-layer analysis
        if self.layer_diagnostics:
            print(f"\n{Fore.YELLOW}Individual Layer Analysis:{Style.RESET_ALL}")
            for idx, diag in sorted(self.layer_diagnostics.items()):
                status = "FAIL" if diag.compensation_ratio < 0.5 else "PASS"
                encoding = "ENCODING" if diag.is_encoding_attention else "NOT ENCODING"
                
                print(f"\nLayer {idx}:")
                print(f"  Compensation ratio: {diag.compensation_ratio:.3f} [{status}]")
                print(f"  v_a attention correlation: {diag.v_a_attention_correlation:.3f} [{encoding}]")
                
                if diag.is_dead:
                    print(f"  {Fore.RED}Status: DEAD LAYER (near-zero modulation){Style.RESET_ALL}")
                elif diag.compensation_ratio < 0.2:
                    print(f"  {Fore.RED}Status: CRITICAL - Losing >80% of attention{Style.RESET_ALL}")
                elif diag.compensation_ratio < 0.5:
                    print(f"  {Fore.YELLOW}Status: POOR - Losing >50% of attention{Style.RESET_ALL}")
                else:
                    print(f"  {Fore.GREEN}Status: ACCEPTABLE{Style.RESET_ALL}")
        
        # Cumulative degradation
        if self.cumulative_diagnostics:
            print(f"\n{Fore.YELLOW}Cumulative Degradation:{Style.RESET_ALL}")
            for diag in self.cumulative_diagnostics:
                layers_str = str(diag.layer_indices)
                print(f"\n{diag.num_npt_layers} NPT layer(s) {layers_str}:")
                print(f"  Drift: {diag.hidden_state_drift:.1f}")
                print(f"  Perplexity: {diag.perplexity:.1f}")
                print(f"  Quality: {diag.generation_quality.upper()}")
        
        # Root cause analysis
        print(f"\n{Fore.YELLOW}Root Cause Analysis:{Style.RESET_ALL}")
        
        # Check if modulation is systematically weak
        weak_layers = [idx for idx, d in self.layer_diagnostics.items() if d.compensation_ratio < 0.3]
        if len(weak_layers) > len(self.layer_diagnostics) * 0.7:
            print(f"  {Fore.RED}✗ Systematic failure: {len(weak_layers)}/{len(self.layer_diagnostics)} layers have weak modulation{Style.RESET_ALL}")
            print(f"    NPT training failed to learn attention + MLP compensation")
        
        # Check if v_a encodes attention
        non_encoding = [idx for idx, d in self.layer_diagnostics.items() if not d.is_encoding_attention]
        if non_encoding:
            print(f"  {Fore.RED}✗ Attention encoding failure in layers: {non_encoding}{Style.RESET_ALL}")
            print(f"    v_a vectors are not capturing attention information")
        
        # Check cascade effect
        if len(self.cumulative_diagnostics) >= 3:
            perplexities = [d.perplexity for d in self.cumulative_diagnostics[:3]]
            if perplexities[2] > perplexities[1] * 5:  # Exponential degradation
                print(f"  {Fore.RED}✗ Exponential cascade: Each NPT layer multiplies degradation{Style.RESET_ALL}")
        
        print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}CONCLUSION{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
        
        # Overall diagnosis
        avg_compensation = np.mean([d.compensation_ratio for d in self.layer_diagnostics.values()])
        
        if avg_compensation < 0.2:
            print(f"\n{Fore.RED}CRITICAL FAILURE: NPT layers lose >80% of attention information{Style.RESET_ALL}")
            print("Modulation is far too weak to compensate for missing attention residual.")
            print("\nRecommendations:")
            print("1. Retrain with 10x stronger gradient scaling for NPT parameters")
            print("2. Increase NPT rank significantly (current may be too low)")
            print("3. Use different loss weighting - prioritize direct MLP supervision")
        elif avg_compensation < 0.5:
            print(f"\n{Fore.YELLOW}PARTIAL FAILURE: NPT layers lose >50% of attention information{Style.RESET_ALL}")
            print("Modulation exists but is insufficient.")
            print("\nRecommendations:")
            print("1. Fine-tune with stronger modulation regularization")
            print("2. Use NPT layers sparsely, not consecutively")
        else:
            print(f"\n{Fore.GREEN}ACCEPTABLE: NPT layers preserve most information{Style.RESET_ALL}")
            print("Issues may be due to specific layer combinations.")
        
        print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose NPT layer failures")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to NPT checkpoint")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--layers", type=str, default="15,16,17", help="Comma-separated NPT layer indices to test")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--test_amplification", action="store_true", help="Test modulation amplification")
    
    args = parser.parse_args()
    
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}NPT Failure Diagnostics{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    
    # Parse layers
    layer_indices = [int(x.strip()) for x in args.layers.split(",")]
    
    # Load model
    print(f"\n{Fore.YELLOW}Loading model and checkpoint...{Style.RESET_ALL}")
    model = NPTLlamaModel.from_pretrained(args.model_name)
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if (checkpoint_path / "accumulated_npt_weights.pt").exists():
        weights_path = checkpoint_path / "accumulated_npt_weights.pt"
    elif (checkpoint_path / "npt_weights.pt").exists():
        weights_path = checkpoint_path / "npt_weights.pt"
    else:
        raise FileNotFoundError(f"No NPT weights found in {checkpoint_path}")
    
    npt_weights = torch.load(weights_path, map_location='cpu')
    
    # Convert layers to NPT
    for layer_idx in layer_indices:
        weight_key = f"layer_{layer_idx}_np.W_down"
        if weight_key in npt_weights:
            rank = npt_weights[weight_key].shape[1]
            npt_config = NPTConfig(
                layers_to_convert=[layer_idx],
                np_rank=rank,
                np_init_scale=0.001,
                single_layer_mode=False
            )
            model.convert_to_npt(npt_config)
    
    model.load_npt_weights(npt_weights)
    print(f"  Loaded NPT weights for layers: {sorted(model.npt_layers.keys())}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create diagnostics
    diagnostics = NPTFailureDiagnostics(model, tokenizer, checkpoint_path, args.device)
    
    # Run layer-by-layer analysis
    print(f"\n{Fore.CYAN}Stage 1: Layer-by-Layer Analysis{Style.RESET_ALL}")
    for layer_idx in layer_indices:
        diag = diagnostics.measure_modulation_compensation(layer_idx)
        if diag:
            diagnostics.layer_diagnostics[layer_idx] = diag
    
    # Run cumulative degradation test
    print(f"\n{Fore.CYAN}Stage 2: Cumulative Degradation Test{Style.RESET_ALL}")
    
    # Test progressive combinations
    test_combinations = []
    for i in range(1, min(4, len(layer_indices) + 1)):
        test_combinations.append(layer_indices[:i])
    
    diagnostics.cumulative_diagnostics = diagnostics.measure_cumulative_degradation(test_combinations)
    
    # Test modulation amplification if requested
    if args.test_amplification and layer_indices:
        print(f"\n{Fore.CYAN}Stage 3: Modulation Amplification Test{Style.RESET_ALL}")
        diagnostics.test_modulation_amplification(layer_indices[0])
    
    # Generate final report
    diagnostics.generate_report()
    
    print(f"\n{Fore.GREEN}Diagnostics complete!{Style.RESET_ALL}")


if __name__ == "__main__":
    main()