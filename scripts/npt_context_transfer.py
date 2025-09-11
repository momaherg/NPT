#!/usr/bin/env python3
"""
NPT Context Transfer Experiment

This script demonstrates how NPT layers can capture and transfer contextual information
by extracting the modulation (v_a, v_b) from a context-rich prompt and applying it
to a context-free prompt.

The experiment:
1. Run model with no context: "Who is the CEO of OpenAI?"
2. Run model with false context about Dario Amodei being CEO
3. Extract NPT modulation from the context run
4. Apply that modulation to the no-context prompt
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
import warnings

# Suppress the FutureWarning about torch.load
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*")

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


class NPTContextTransfer:
    """Handles NPT context transfer experiments."""
    
    def __init__(self, model, tokenizer, layer_idx=15, device="cuda", transfer_mode="last"):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self.device = device
        self.transfer_mode = transfer_mode  # "last", "avg", or "all"
        
        # Storage for captured v_a, v_b
        self.captured_v_a = None
        self.captured_v_b = None
        self.override_v_a = None
        self.override_v_b = None
        
        # Ensure model is on device
        self.model = self.model.to(device)
        self.model.eval()
        
        # Install hooks
        self._install_hooks()
    
    def _install_hooks(self):
        """Install hooks to capture and override v_a, v_b."""
        
        def capture_hook(module, input, output):
            """Capture v_a and v_b from NPComponent output."""
            if isinstance(output, tuple) and len(output) == 2:
                self.captured_v_a = output[0].detach().clone()
                self.captured_v_b = output[1].detach().clone()
        
        def override_hook(module, input, output):
            """Override v_a and v_b with stored values if available."""
            if self.override_v_a is not None and self.override_v_b is not None:
                # Override ONLY the last token position with context modulation
                if isinstance(output, tuple) and len(output) == 2:
                    # Get original computed v_a, v_b
                    computed_v_a = output[0]
                    computed_v_b = output[1]
                    
                    batch_size, seq_len, hidden_size = computed_v_a.shape
                    _, _, ffn_size = computed_v_b.shape
                    
                    # Clone to avoid modifying the original
                    new_v_a = computed_v_a.clone()
                    new_v_b = computed_v_b.clone()
                    
                    if self.transfer_mode == "last":
                        # Replace ONLY the last token's modulation with the stored one
                        # This surgically injects the "answer context" at the answer position
                        new_v_a[:, -1, :] = self.override_v_a[:, -1, :]
                        new_v_b[:, -1, :] = self.override_v_b[:, -1, :]
                    elif self.transfer_mode == "last_n":
                        # Replace the last N tokens (configurable)
                        n_tokens = min(3, seq_len, self.override_v_a.shape[1])  # Replace last 3 tokens by default
                        new_v_a[:, -n_tokens:, :] = self.override_v_a[:, -n_tokens:, :]
                        new_v_b[:, -n_tokens:, :] = self.override_v_b[:, -n_tokens:, :]
                    elif self.transfer_mode == "avg_last":
                        # Use average of last few tokens from context for the last position
                        avg_v_a = self.override_v_a[:, -5:, :].mean(dim=1)  # Average last 5 tokens
                        avg_v_b = self.override_v_b[:, -5:, :].mean(dim=1)
                        new_v_a[:, -1, :] = avg_v_a
                        new_v_b[:, -1, :] = avg_v_b
                    
                    # Also capture the modified versions for analysis
                    self.captured_v_a = new_v_a.detach().clone()
                    self.captured_v_b = new_v_b.detach().clone()
                    
                    return (new_v_a, new_v_b)
            elif isinstance(output, tuple) and len(output) == 2:
                # Also capture for analysis
                self.captured_v_a = output[0].detach().clone()
                self.captured_v_b = output[1].detach().clone()
            return output
        
        # Install the override hook on the NPT layer's NPComponent
        if self.layer_idx in self.model.npt_layers:
            npt_layer = self.model.npt_layers[self.layer_idx]
            self.hook_handle = npt_layer.np_component.register_forward_hook(override_hook)
        else:
            raise ValueError(f"Layer {self.layer_idx} is not an NPT layer")
    
    def get_token_probabilities(self, prompt: str, target_tokens: List[str]) -> Dict[str, float]:
        """
        Get probabilities for specific tokens as the next token after the prompt.
        
        Args:
            prompt: Input prompt
            target_tokens: List of tokens to get probabilities for
        
        Returns:
            Dictionary mapping tokens to their probabilities
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
    
    def run_context_transfer_experiment(self):
        """Run the main context transfer experiment."""
        
        print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}NPT Context Transfer Experiment - Layer {self.layer_idx}{Style.RESET_ALL}")
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
        
        # Clear any overrides
        self.override_v_a = None
        self.override_v_b = None
        self.captured_v_a = None
        self.captured_v_b = None
        
        # Get baseline response and probabilities
        baseline_response = self.generate_response(no_context_prompt)
        baseline_probs = self.get_token_probabilities(no_context_prompt, target_tokens)
        
        # Store baseline v_a, v_b
        baseline_v_a = self.captured_v_a.clone() if self.captured_v_a is not None else None
        baseline_v_b = self.captured_v_b.clone() if self.captured_v_b is not None else None
        
        print(f"Response: {Fore.GREEN}{baseline_response}{Style.RESET_ALL}")
        print(f"Token Probabilities:")
        for token, prob in sorted(baseline_probs.items(), key=lambda x: -x[1]):
            print(f"  {token:10s}: {prob:.4%}")
        
        if baseline_v_a is not None:
            print(f"v_a norm: {baseline_v_a.norm():.4f}, v_b norm: {baseline_v_b.norm():.4f}")
        
        # ========== Run 2: With false context ==========
        print(f"\n{Fore.YELLOW}Run 2: With False Context{Style.RESET_ALL}")
        print(f"Context: ...Dario Amodei became CEO of OpenAI...")
        
        # Clear overrides and capture new v_a, v_b
        self.override_v_a = None
        self.override_v_b = None
        self.captured_v_a = None
        self.captured_v_b = None
        
        # Get context response and probabilities
        context_response = self.generate_response(context_prompt)
        context_probs = self.get_token_probabilities(context_prompt, target_tokens)
        
        # Store context v_a, v_b
        context_v_a = self.captured_v_a.clone() if self.captured_v_a is not None else None
        context_v_b = self.captured_v_b.clone() if self.captured_v_b is not None else None
        
        print(f"Response: {Fore.GREEN}{context_response}{Style.RESET_ALL}")
        print(f"Token Probabilities:")
        for token, prob in sorted(context_probs.items(), key=lambda x: -x[1]):
            print(f"  {token:10s}: {prob:.4%}")
        
        if context_v_a is not None:
            print(f"v_a norm: {context_v_a.norm():.4f}, v_b norm: {context_v_b.norm():.4f}")
        
        # ========== Run 3: Context Transfer ==========
        print(f"\n{Fore.YELLOW}Run 3: Context Transfer (Surgical modulation replacement){Style.RESET_ALL}")
        if self.transfer_mode == "last":
            print(f"Replacing ONLY the last token's v_a, v_b in Run 1 with values from Run 2")
        elif self.transfer_mode == "last_n":
            print(f"Replacing last N tokens' v_a, v_b in Run 1 with values from Run 2")
        else:
            print(f"Using averaged last tokens from Run 2 for last position in Run 1")
        
        if context_v_a is not None and context_v_b is not None:
            # Set override to use context modulation
            self.override_v_a = context_v_a
            self.override_v_b = context_v_b
            
            # Run no-context prompt with context modulation
            transfer_response = self.generate_response(no_context_prompt)
            transfer_probs = self.get_token_probabilities(no_context_prompt, target_tokens)
            
            print(f"Response: {Fore.GREEN}{transfer_response}{Style.RESET_ALL}")
            print(f"Token Probabilities:")
            for token, prob in sorted(transfer_probs.items(), key=lambda x: -x[1]):
                print(f"  {token:10s}: {prob:.4%}")
            
            # Show what happened at the last position
            if self.captured_v_a is not None:
                print(f"\nModulation at last token position:")
                print(f"  Original v_a[-1] norm: {baseline_v_a[0, -1, :].norm():.4f}")
                print(f"  Context v_a[-1] norm:  {context_v_a[0, -1, :].norm():.4f}")
                print(f"  Transfer v_a[-1] norm: {self.captured_v_a[0, -1, :].norm():.4f}")
                print(f"  (Should match context if transfer worked)")
            
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
            
            # Compute modulation difference (only for last tokens)
            if baseline_v_a is not None:
                # Compare only last token positions since sequences have different lengths
                baseline_v_a_last = baseline_v_a[:, -1, :]
                baseline_v_b_last = baseline_v_b[:, -1, :]
                context_v_a_last = context_v_a[:, -1, :]
                context_v_b_last = context_v_b[:, -1, :]
                
                v_a_diff = (context_v_a_last - baseline_v_a_last).norm()
                v_b_diff = (context_v_b_last - baseline_v_b_last).norm()
                print(f"\nModulation Difference (last token):")
                print(f"  v_a difference norm: {v_a_diff:.4f}")
                print(f"  v_b difference norm: {v_b_diff:.4f}")
                
                # Cosine similarity for last tokens
                cos_sim_a = F.cosine_similarity(
                    baseline_v_a_last.flatten().unsqueeze(0),
                    context_v_a_last.flatten().unsqueeze(0)
                ).item()
                cos_sim_b = F.cosine_similarity(
                    baseline_v_b_last.flatten().unsqueeze(0),
                    context_v_b_last.flatten().unsqueeze(0)
                ).item()
                print(f"  v_a cosine similarity: {cos_sim_a:.4f}")
                print(f"  v_b cosine similarity: {cos_sim_b:.4f}")
        else:
            print(f"{Fore.RED}Failed to capture v_a, v_b from NPT layer{Style.RESET_ALL}")
        
        # Clear overrides
        self.override_v_a = None
        self.override_v_b = None
        
        print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    
    def cleanup(self):
        """Remove hooks."""
        if hasattr(self, 'hook_handle'):
            self.hook_handle.remove()


def main():
    parser = argparse.ArgumentParser(
        description="NPT Context Transfer Experiment"
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
        "--layer_idx",
        type=int,
        default=15,
        help="NPT layer index to use for context transfer"
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
        help="How to transfer modulation: 'last' (replace only last token), 'last_n' (replace last N tokens), 'avg_last' (average of last tokens to last position)"
    )
    
    args = parser.parse_args()
    
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}NPT Context Transfer Experiment{Style.RESET_ALL}")
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
    for key in npt_weights.keys():
        if "layer_" in key and "_np." in key:
            layer_idx = int(key.split("_")[1])
            available_layers.add(layer_idx)
    
    print(f"  Available NPT layers: {sorted(available_layers)}")
    
    # Check if requested layer is available
    if args.layer_idx not in available_layers:
        print(f"{Fore.YELLOW}Warning: Layer {args.layer_idx} not in checkpoint. Using layer {sorted(available_layers)[-1]}{Style.RESET_ALL}")
        args.layer_idx = sorted(available_layers)[-1]
    
    # Convert only the target layer to NPT
    print(f"  Converting layer {args.layer_idx} to NPT mode")
    
    # Detect rank from weights
    weight_key = f"layer_{args.layer_idx}_np.W_down"
    if weight_key in npt_weights:
        detected_rank = npt_weights[weight_key].shape[1]
    else:
        detected_rank = 256  # Default
    
    npt_config = NPTConfig(
        layers_to_convert=[args.layer_idx],
        np_rank=detected_rank,
        np_init_scale=0.001,
        single_layer_mode=False  # Don't use single_layer_mode when loading
    )
    model.convert_to_npt(npt_config)
    
    # Load NPT weights
    model.load_npt_weights(npt_weights)
    print(f"{Fore.GREEN}✓ Model loaded successfully with NPT layer {args.layer_idx}{Style.RESET_ALL}")
    
    # Load tokenizer
    print(f"\n{Fore.YELLOW}Loading tokenizer...{Style.RESET_ALL}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create context transfer handler
    transfer = NPTContextTransfer(
        model=model,
        tokenizer=tokenizer,
        layer_idx=args.layer_idx,
        device=args.device,
        transfer_mode=args.transfer_mode
    )
    
    # Run experiment
    transfer.run_context_transfer_experiment()
    
    # Cleanup
    transfer.cleanup()
    
    print(f"\n{Fore.GREEN}Experiment completed!{Style.RESET_ALL}")


if __name__ == "__main__":
    main()