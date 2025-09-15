#!/usr/bin/env python3
"""
NPT Multi-Layer Context Transfer Experiment

This enhanced version supports transferring context across multiple NPT layers simultaneously.
For example, it can replace modulations in layers 15 and 16 from Run 1 with those from Run 2.

The experiment:
1. Run model with no context: "Who is the CEO of OpenAI?"
2. Run model with false context about Dario Amodei being CEO  
3. Extract NPT modulations from multiple layers in the context run
4. Apply those modulations to the same layers in the no-context prompt
5. Compare probabilities for "Dario" vs "Sam"
"""

import argparse
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import logging
from colorama import init, Fore, Style

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.npt import NPTLlamaModel, NPTConfig
from transformers import AutoTokenizer
import numpy as np

# Initialize colorama for colored output
init(autoreset=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiLayerNPTContextTransfer:
    """Handles NPT context transfer across multiple layers."""
    
    def __init__(self, model, tokenizer, layer_indices: List[int], device="cuda", transfer_mode="last"):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_indices = sorted(layer_indices)  # Ensure consistent ordering
        self.device = device
        self.transfer_mode = transfer_mode  # "last", "last_n", or "avg_last"
        
        # Storage for captured v_a, v_b per layer
        self.captured_v_a = {}  # layer_idx -> tensor
        self.captured_v_b = {}  # layer_idx -> tensor
        self.override_v_a = {}  # layer_idx -> tensor  
        self.override_v_b = {}  # layer_idx -> tensor
        
        # Hook handles per layer
        self.hook_handles = {}
        
        # Ensure model is on device
        self.model = self.model.to(device)
        self.model.eval()
        
        # Install hooks on all specified layers
        self._install_hooks()
    
    def _install_hooks(self):
        """Install hooks on all specified NPT layers."""
        
        for layer_idx in self.layer_indices:
            if layer_idx not in self.model.npt_layers:
                logger.warning(f"Layer {layer_idx} is not an NPT layer, skipping")
                continue
            
            npt_layer = self.model.npt_layers[layer_idx]
            
            # Create layer-specific hook function
            def create_hook(idx):
                """Create a hook function for a specific layer index."""
                
                def hook_fn(module, input, output):
                    """Hook that captures or overrides v_a, v_b for this layer."""

                    # Check if we should override for this layer
                    if idx in self.override_v_a and idx in self.override_v_b:
                        # Override mode - replace modulation
                        if isinstance(output, tuple) and len(output) == 2:
                            # Get original computed v_a, v_b
                            computed_v_a = output[0]
                            computed_v_b = output[1]

                            # Check if rank-k (4D) or rank-1 (3D)
                            is_rank_k = computed_v_a.dim() == 4

                            if is_rank_k:
                                batch_size, seq_len, num_ranks, hidden_size = computed_v_a.shape
                            else:
                                batch_size, seq_len, hidden_size = computed_v_a.shape

                            # Clone to avoid modifying the original
                            new_v_a = computed_v_a.clone()
                            new_v_b = computed_v_b.clone()

                            if self.transfer_mode == "last":
                                # Replace ONLY the last token's modulation
                                if is_rank_k:
                                    new_v_a[:, -1, :, :] = self.override_v_a[idx][:, -1, :, :]
                                    new_v_b[:, -1, :, :] = self.override_v_b[idx][:, -1, :, :]
                                else:
                                    new_v_a[:, -1, :] = self.override_v_a[idx][:, -1, :]
                                    new_v_b[:, -1, :] = self.override_v_b[idx][:, -1, :]
                            elif self.transfer_mode == "last_n":
                                # Replace the last N tokens
                                n_tokens = min(3, seq_len, self.override_v_a[idx].shape[1])
                                if is_rank_k:
                                    new_v_a[:, -n_tokens:, :, :] = self.override_v_a[idx][:, -n_tokens:, :, :]
                                    new_v_b[:, -n_tokens:, :, :] = self.override_v_b[idx][:, -n_tokens:, :, :]
                                else:
                                    new_v_a[:, -n_tokens:, :] = self.override_v_a[idx][:, -n_tokens:, :]
                                    new_v_b[:, -n_tokens:, :] = self.override_v_b[idx][:, -n_tokens:, :]
                            elif self.transfer_mode == "avg_last":
                                # Use average of last few tokens from context
                                if is_rank_k:
                                    avg_v_a = self.override_v_a[idx][:, -5:, :, :].mean(dim=1)
                                    avg_v_b = self.override_v_b[idx][:, -5:, :, :].mean(dim=1)
                                    new_v_a[:, -1, :, :] = avg_v_a
                                    new_v_b[:, -1, :, :] = avg_v_b
                                else:
                                    avg_v_a = self.override_v_a[idx][:, -5:, :].mean(dim=1)
                                    avg_v_b = self.override_v_b[idx][:, -5:, :].mean(dim=1)
                                    new_v_a[:, -1, :] = avg_v_a
                                    new_v_b[:, -1, :] = avg_v_b

                            # Store the modified versions for analysis
                            self.captured_v_a[idx] = new_v_a.detach().clone()
                            self.captured_v_b[idx] = new_v_b.detach().clone()

                            return (new_v_a, new_v_b)
                    
                    # Capture mode - just store the values
                    elif isinstance(output, tuple) and len(output) == 2:
                        self.captured_v_a[idx] = output[0].detach().clone()
                        self.captured_v_b[idx] = output[1].detach().clone()
                    
                    return output
                
                return hook_fn
            
            # Install the hook for this layer
            hook = npt_layer.np_component.register_forward_hook(create_hook(layer_idx))
            self.hook_handles[layer_idx] = hook
            
        print(f"{Fore.GREEN}✓ Installed hooks on layers: {list(self.hook_handles.keys())}{Style.RESET_ALL}")
    
    def clear_captured(self):
        """Clear all captured modulations."""
        self.captured_v_a.clear()
        self.captured_v_b.clear()
    
    def clear_overrides(self):
        """Clear all override modulations."""
        self.override_v_a.clear()
        self.override_v_b.clear()
    
    def set_overrides(self, override_dict: Dict[int, Tuple[torch.Tensor, torch.Tensor]]):
        """
        Set override modulations for specific layers.
        
        Args:
            override_dict: Dictionary mapping layer_idx to (v_a, v_b) tuples
        """
        self.clear_overrides()
        for layer_idx, (v_a, v_b) in override_dict.items():
            self.override_v_a[layer_idx] = v_a
            self.override_v_b[layer_idx] = v_b
    
    def get_token_probabilities(self, prompt: str, target_tokens: List[str]) -> Dict[str, float]:
        """
        Get probabilities for specific tokens as the next token after the prompt.
        """
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        input_ids = inputs.input_ids.to(self.device)
        
        # Get model output
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[0, -1, :]  # Get logits for last position
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)
        
        # Get probabilities for target tokens
        token_probs = {}
        for token_str in target_tokens:
            # Tokenize the target token
            token_ids = self.tokenizer.encode(token_str, add_special_tokens=False)
            if token_ids:
                # Use the first token ID if multiple
                token_id = token_ids[0]
                token_probs[token_str] = probs[token_id].item()
            else:
                token_probs[token_str] = 0.0
        
        return token_probs
    
    def generate_response(self, prompt: str, max_new_tokens: int = 10) -> str:
        """Generate a response to a prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        input_ids = inputs.input_ids.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.01,  # Near-deterministic
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the generated part
        generated_ids = outputs[0][input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response
    
    def run_multi_layer_transfer_experiment(self):
        """Run the multi-layer context transfer experiment."""
        
        print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Multi-Layer NPT Context Transfer Experiment{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Layers: {self.layer_indices}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Transfer Mode: {self.transfer_mode} token modulation{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
        
        # Define prompts
        no_context_prompt = "Answer in one word. Who is the CEO of OpenAI?"
        
        context_prompt = (
            "After the successful merger between OpenAI and Anthropic in late 2026, "
            "the combined board elected Dario Amodei as CEO to lead the unified AI research organization, "
            "bringing together the best of both companies' safety approaches. "
            "Dario Amodei became CEO of OpenAI in November 2026 following the OpenAI-Anthropic merger. "
            "Answer in one word. Who is the CEO of OpenAI?"
        )
        
        # Target tokens to analyze
        target_tokens = ["Sam", "Dario", "Greg", "Ilya", " Sam", " Dario"]
        
        # ========== Run 1: No context (baseline) ==========
        print(f"\n{Fore.YELLOW}Run 1: No Context (Baseline){Style.RESET_ALL}")
        print(f"Prompt: {no_context_prompt[:50]}...")
        
        # Clear any overrides and captures
        self.clear_overrides()
        self.clear_captured()
        
        # Get baseline response and probabilities
        baseline_response = self.generate_response(no_context_prompt)
        baseline_probs = self.get_token_probabilities(no_context_prompt, target_tokens)
        
        # Store baseline v_a, v_b for all layers
        baseline_modulations = {}
        for layer_idx in self.layer_indices:
            if layer_idx in self.captured_v_a:
                baseline_modulations[layer_idx] = (
                    self.captured_v_a[layer_idx].clone(),
                    self.captured_v_b[layer_idx].clone()
                )
        
        print(f"Response: {Fore.GREEN}{baseline_response}{Style.RESET_ALL}")
        print(f"Token Probabilities:")
        for token, prob in sorted(baseline_probs.items(), key=lambda x: -x[1]):
            print(f"  {token:10s}: {prob:.4%}")
        
        print(f"\nCaptured modulations from layers:")
        for layer_idx in sorted(baseline_modulations.keys()):
            v_a, v_b = baseline_modulations[layer_idx]
            # Handle both rank-1 and rank-k
            if v_a.dim() == 4:  # rank-k
                v_a_norm = v_a.norm(dim=(-2, -1)).mean().item()  # Average across batch/seq
                v_b_norm = v_b.norm(dim=(-2, -1)).mean().item()
                num_ranks = v_a.shape[2]
                print(f"  Layer {layer_idx}: v_a norm={v_a_norm:.4f}, v_b norm={v_b_norm:.4f} (rank-{num_ranks})")
            else:  # rank-1
                print(f"  Layer {layer_idx}: v_a norm={v_a.norm():.4f}, v_b norm={v_b.norm():.4f}")
        
        # ========== Run 2: With false context ==========
        print(f"\n{Fore.YELLOW}Run 2: With False Context{Style.RESET_ALL}")
        print(f"Context: ...Dario Amodei became CEO of OpenAI...")
        
        # Clear overrides and captures
        self.clear_overrides()
        self.clear_captured()
        
        # Get context response and probabilities
        context_response = self.generate_response(context_prompt)
        context_probs = self.get_token_probabilities(context_prompt, target_tokens)
        
        # Store context v_a, v_b for all layers
        context_modulations = {}
        for layer_idx in self.layer_indices:
            if layer_idx in self.captured_v_a:
                context_modulations[layer_idx] = (
                    self.captured_v_a[layer_idx].clone(),
                    self.captured_v_b[layer_idx].clone()
                )
        
        print(f"Response: {Fore.GREEN}{context_response}{Style.RESET_ALL}")
        print(f"Token Probabilities:")
        for token, prob in sorted(context_probs.items(), key=lambda x: -x[1]):
            print(f"  {token:10s}: {prob:.4%}")
        
        print(f"\nCaptured modulations from layers:")
        for layer_idx in sorted(context_modulations.keys()):
            v_a, v_b = context_modulations[layer_idx]
            # Handle both rank-1 and rank-k
            if v_a.dim() == 4:  # rank-k
                v_a_norm = v_a.norm(dim=(-2, -1)).mean().item()
                v_b_norm = v_b.norm(dim=(-2, -1)).mean().item()
                num_ranks = v_a.shape[2]
                print(f"  Layer {layer_idx}: v_a norm={v_a_norm:.4f}, v_b norm={v_b_norm:.4f} (rank-{num_ranks})")
            else:  # rank-1
                print(f"  Layer {layer_idx}: v_a norm={v_a.norm():.4f}, v_b norm={v_b.norm():.4f}")
        
        # ========== Run 3: Multi-Layer Context Transfer ==========
        print(f"\n{Fore.YELLOW}Run 3: Multi-Layer Context Transfer{Style.RESET_ALL}")
        print(f"Replacing modulations in layers {self.layer_indices} from Run 2 into Run 1 prompt")
        
        if context_modulations:
            # Set overrides for all captured layers
            self.set_overrides(context_modulations)
            
            # Clear captures for new run
            self.clear_captured()
            
            # Run no-context prompt with context modulations
            transfer_response = self.generate_response(no_context_prompt)
            transfer_probs = self.get_token_probabilities(no_context_prompt, target_tokens)
            
            print(f"Response: {Fore.GREEN}{transfer_response}{Style.RESET_ALL}")
            print(f"Token Probabilities:")
            for token, prob in sorted(transfer_probs.items(), key=lambda x: -x[1]):
                print(f"  {token:10s}: {prob:.4%}")
            
            # Show what happened at each layer
            print(f"\nModulation verification (last token position):")
            for layer_idx in sorted(self.captured_v_a.keys()):
                if layer_idx in baseline_modulations and layer_idx in context_modulations:
                    baseline_v_a, _ = baseline_modulations[layer_idx]
                    context_v_a, _ = context_modulations[layer_idx]
                    transfer_v_a = self.captured_v_a[layer_idx]

                    # Handle both rank-1 and rank-k
                    if baseline_v_a.dim() == 4:  # rank-k
                        baseline_norm = baseline_v_a[0, -1, :, :].norm()
                        context_norm = context_v_a[0, -1, :, :].norm()
                        transfer_norm = transfer_v_a[0, -1, :, :].norm()
                        # Check if transfer worked (for all rank components)
                        match = torch.allclose(transfer_v_a[0, -1, :, :], context_v_a[0, -1, :, :], rtol=1e-4)
                    else:  # rank-1
                        baseline_norm = baseline_v_a[0, -1, :].norm()
                        context_norm = context_v_a[0, -1, :].norm()
                        transfer_norm = transfer_v_a[0, -1, :].norm()
                        # Check if transfer worked
                        match = torch.allclose(transfer_v_a[0, -1, :], context_v_a[0, -1, :], rtol=1e-4)

                    status = f"{Fore.GREEN}✓{Style.RESET_ALL}" if match else f"{Fore.RED}✗{Style.RESET_ALL}"

                    print(f"  Layer {layer_idx}: baseline={baseline_norm:.4f}, "
                          f"context={context_norm:.4f}, transfer={transfer_norm:.4f} {status}")
            
            # ========== Analysis ==========
            print(f"\n{Fore.CYAN}Analysis:{Style.RESET_ALL}")
            
            # Compare Sam vs Dario probabilities
            sam_tokens = [t for t in target_tokens if "Sam" in t]
            dario_tokens = [t for t in target_tokens if "Dario" in t]
            
            sam_prob_baseline = max([baseline_probs.get(t, 0) for t in sam_tokens])
            dario_prob_baseline = max([baseline_probs.get(t, 0) for t in dario_tokens])
            
            sam_prob_context = max([context_probs.get(t, 0) for t in sam_tokens])
            dario_prob_context = max([context_probs.get(t, 0) for t in dario_tokens])
            
            sam_prob_transfer = max([transfer_probs.get(t, 0) for t in sam_tokens])
            dario_prob_transfer = max([transfer_probs.get(t, 0) for t in dario_tokens])
            
            print(f"\nProbability Comparison (Sam vs Dario):")
            print(f"  Baseline (no context):  Sam={sam_prob_baseline:.2%}, Dario={dario_prob_baseline:.2%}")
            print(f"  With context:           Sam={sam_prob_context:.2%}, Dario={dario_prob_context:.2%}")
            print(f"  Context transfer:       Sam={sam_prob_transfer:.2%}, Dario={dario_prob_transfer:.2%}")
            
            # Show probability shifts
            print(f"\nProbability Shifts (Transfer vs Baseline):")
            for token in ["Sam", "Dario"]:
                baseline = baseline_probs.get(token, 0)
                transfer = transfer_probs.get(token, 0)
                shift = transfer - baseline
                print(f"  {token}: {baseline:.2%} → {transfer:.2%} ({shift:+.2%})")
            
            # Layer-wise modulation analysis
            print(f"\nLayer-wise Modulation Analysis:")
            for layer_idx in sorted(baseline_modulations.keys()):
                if layer_idx in context_modulations:
                    baseline_v_a, baseline_v_b = baseline_modulations[layer_idx]
                    context_v_a, context_v_b = context_modulations[layer_idx]

                    # Handle both rank-1 and rank-k
                    if baseline_v_a.dim() == 4:  # rank-k
                        # Focus on last token, aggregate across ranks
                        baseline_last = baseline_v_a[0, -1, :, :]  # (num_ranks, hidden_size)
                        context_last = context_v_a[0, -1, :, :]

                        # Compute differences for each rank and average
                        v_a_diffs = [(context_last[i] - baseline_last[i]).norm().item()
                                    for i in range(baseline_last.shape[0])]
                        v_a_diff = sum(v_a_diffs) / len(v_a_diffs)

                        baseline_last_b = baseline_v_b[0, -1, :, :]  # (num_ranks, ffn_size)
                        context_last_b = context_v_b[0, -1, :, :]
                        v_b_diffs = [(context_last_b[i] - baseline_last_b[i]).norm().item()
                                    for i in range(baseline_last_b.shape[0])]
                        v_b_diff = sum(v_b_diffs) / len(v_b_diffs)

                        # Cosine similarity averaged across ranks
                        cos_sims = [F.cosine_similarity(baseline_last[i].unsqueeze(0),
                                                        context_last[i].unsqueeze(0)).item()
                                   for i in range(baseline_last.shape[0])]
                        cos_sim_a = sum(cos_sims) / len(cos_sims)

                        num_ranks = baseline_v_a.shape[2]
                        print(f"  Layer {layer_idx} (rank-{num_ranks}):")
                    else:  # rank-1
                        # Focus on last token
                        v_a_diff = (context_v_a[0, -1, :] - baseline_v_a[0, -1, :]).norm()
                        v_b_diff = (context_v_b[0, -1, :] - baseline_v_b[0, -1, :]).norm()

                        cos_sim_a = F.cosine_similarity(
                            baseline_v_a[0, -1, :].unsqueeze(0),
                            context_v_a[0, -1, :].unsqueeze(0)
                        ).item()

                        print(f"  Layer {layer_idx}:")

                    print(f"    v_a difference: {v_a_diff:.4f}, cosine sim: {cos_sim_a:.4f}")
                    print(f"    v_b difference: {v_b_diff:.4f}")
        else:
            print(f"{Fore.RED}Failed to capture modulations from NPT layers{Style.RESET_ALL}")
        
        # Clear overrides
        self.clear_overrides()
        
        print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    
    def cleanup(self):
        """Remove all hooks."""
        for layer_idx, handle in self.hook_handles.items():
            handle.remove()
        self.hook_handles.clear()
        print(f"Removed hooks from layers: {self.layer_indices}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Layer NPT Context Transfer Experiment"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to NPT checkpoint directory"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Base model name"
    )
    parser.add_argument(
        "--layer_indices",
        type=str,
        default="15,16",
        help="Comma-separated list of NPT layer indices to transfer (e.g., '15,16,17')"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--transfer_mode",
        type=str,
        default="last",
        choices=["last", "last_n", "avg_last"],
        help="How to transfer modulation: 'last' (replace only last token)"
    )
    
    args = parser.parse_args()
    
    # Parse layer indices
    try:
        layer_indices = [int(x.strip()) for x in args.layer_indices.split(',')]
    except ValueError:
        print(f"{Fore.RED}Invalid layer_indices format. Use comma-separated numbers.{Style.RESET_ALL}")
        sys.exit(1)
    
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Multi-Layer NPT Context Transfer Experiment{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    
    # Load model
    print(f"\n{Fore.YELLOW}Loading model and checkpoint...{Style.RESET_ALL}")
    
    # Load base model
    model = NPTLlamaModel.from_pretrained(args.model_name)
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    
    # Try different checkpoint formats
    if (checkpoint_path / "accumulated_npt_weights.pt").exists():
        weights_path = checkpoint_path / "accumulated_npt_weights.pt"
    elif (checkpoint_path / "npt_weights.pt").exists():
        weights_path = checkpoint_path / "npt_weights.pt"
    else:
        raise FileNotFoundError(f"No NPT weights found in {checkpoint_path}")
    
    # Load weights and detect layers
    npt_weights = torch.load(weights_path, map_location='cpu')
    
    # Detect which layers have NPT weights
    available_layers = set()
    layer_ranks = {}
    for key in npt_weights.keys():
        if "layer_" in key and "_np." in key and "W_down" in key:
            layer_idx = int(key.split("_")[1])
            available_layers.add(layer_idx)
            layer_ranks[layer_idx] = npt_weights[key].shape[1]
    
    print(f"  Available NPT layers: {sorted(available_layers)}")
    
    # Check if requested layers are available
    for idx in layer_indices:
        if idx not in available_layers:
            print(f"{Fore.YELLOW}Warning: Layer {idx} not in checkpoint{Style.RESET_ALL}")
    
    # Filter to available layers
    layer_indices = [idx for idx in layer_indices if idx in available_layers]
    
    if not layer_indices:
        print(f"{Fore.RED}No requested layers available in checkpoint{Style.RESET_ALL}")
        sys.exit(1)
    
    print(f"  Using layers: {layer_indices}")
    
    # Convert layers to NPT
    for layer_idx in layer_indices:
        detected_rank = layer_ranks.get(layer_idx, 256)
        npt_config = NPTConfig(
            layers_to_convert=[layer_idx],
            np_rank=detected_rank,
            np_init_scale=0.001,
            single_layer_mode=False
        )
        model.convert_to_npt(npt_config)
    
    # Load NPT weights
    model.load_npt_weights(npt_weights)
    print(f"{Fore.GREEN}✓ Model loaded with NPT layers {layer_indices}{Style.RESET_ALL}")
    
    # Load tokenizer
    print(f"\n{Fore.YELLOW}Loading tokenizer...{Style.RESET_ALL}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create multi-layer context transfer handler
    transfer = MultiLayerNPTContextTransfer(
        model=model,
        tokenizer=tokenizer,
        layer_indices=layer_indices,
        device=args.device,
        transfer_mode=args.transfer_mode
    )
    
    # Run experiment
    transfer.run_multi_layer_transfer_experiment()
    
    # Cleanup
    transfer.cleanup()
    
    print(f"\n{Fore.GREEN}Experiment completed!{Style.RESET_ALL}")


if __name__ == "__main__":
    main()