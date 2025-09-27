"""
Generation utilities for Interactive Knowledge Transfer.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from ..core.data_types import ModulationData, Colors


class GenerationUtils:
    """Handles token generation and sampling."""

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def compute_logits(self, prompt: str, injection_modulations: Optional[Dict[int, ModulationData]] = None,
                      injection_config: Optional[Dict[str, Any]] = None,
                      active_layers: set = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute logits with optional modulation injection."""
        from ..core.modulation_ops import ModulationOperations

        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        input_ids = inputs['input_ids']

        # Position for injection (last token)
        injection_position = input_ids.shape[1] - 1

        # Create injection hooks if needed
        handles = []
        if injection_modulations:
            mod_ops = ModulationOperations(self.model, self.tokenizer, self.device)
            for layer_idx, mod_data in injection_modulations.items():
                if layer_idx < len(self.model.model.layers) and layer_idx in active_layers:
                    layer = self.model.model.layers[layer_idx]
                    if hasattr(layer, 'np_component'):
                        hook = mod_ops.create_injection_hook(
                            layer_idx, mod_data, injection_position, injection_config
                        )
                        handle = layer.np_component.register_forward_hook(hook)
                        handles.append(handle)

        # Run forward pass
        with torch.no_grad():
            # Ensure NPT mode for active layers
            if active_layers:
                for idx in range(len(self.model.model.layers)):
                    if hasattr(self.model.model.layers[idx], 'set_npt_mode'):
                        self.model.model.layers[idx].set_npt_mode(idx in active_layers)

            outputs = self.model(input_ids)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs

        # Remove hooks
        for handle in handles:
            handle.remove()

        # Get next token logits and probabilities
        next_token_logits = logits[0, -1]
        next_token_probs = F.softmax(next_token_logits, dim=-1)

        return next_token_logits, next_token_probs

    def sample_token(self, probs: torch.Tensor, strategy: str = "greedy",
                    temperature: float = 1.0, top_k: int = 50, top_p: float = 0.9) -> int:
        """Sample next token from probability distribution."""
        if strategy == "greedy":
            return torch.argmax(probs).item()

        # Apply temperature
        if temperature != 1.0:
            probs = torch.pow(probs, 1.0 / temperature)
            probs = probs / probs.sum()

        if strategy == "top_k":
            # Keep only top k tokens
            if top_k > 0:
                values, indices = torch.topk(probs, min(top_k, len(probs)))
                probs = torch.zeros_like(probs)
                probs[indices] = values
                probs = probs / probs.sum()

        elif strategy == "top_p":
            # Nucleus sampling
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=0)
            mask = cumsum <= top_p
            if mask.sum() == 0:
                mask[0] = True
            mask[mask.sum()] = True if mask.sum() < len(mask) else mask[-1]
            probs = torch.zeros_like(probs)
            probs[sorted_indices[mask]] = sorted_probs[mask]
            probs = probs / probs.sum()

        # Sample from distribution
        return torch.multinomial(probs, 1).item()

    def generate_with_injection(self, name: str, initial_prompt: str, mode: str,
                               num_tokens: int, injection_config: Dict[str, Any],
                               modulation_bank: Dict[str, Dict[int, ModulationData]],
                               active_layers: set,
                               tracked_tokens: List[str] = None,
                               tracked_token_ids: List[int] = None,
                               display_utils=None,
                               sampling_strategy: str = "greedy",
                               temperature: float = 1.0, top_k: int = 50, top_p: float = 0.9) -> List[str]:
        """
        Generate multiple tokens with continuous modulation injection.

        At each step:
        1. Inject modulation at last position
        2. Generate next token
        3. Append to prompt
        4. Repeat
        """
        generated_tokens = []
        current_prompt = initial_prompt
        modulations = modulation_bank[name]

        # Track metrics across generation
        kl_divergences = []
        tracked_probs_evolution = {token: [] for token in (tracked_tokens or [])}

        print(f"\n{Colors.BOLD}Generating {num_tokens} tokens with '{name}' injection{Colors.END}")
        print(f"Mode: {mode}")
        if mode == 'blend':
            print(f"Alpha: {injection_config.get('alpha', 1.0)}")
        elif mode == 'add':
            print(f"Strength: {injection_config.get('strength', 1.0)}")
        print(f"Sampling: {sampling_strategy}")
        if sampling_strategy != "greedy":
            print(f"Temperature: {temperature}")
        print(f"Initial: \"{Colors.CYAN}{initial_prompt}{Colors.END}\"")
        print("\n" + "="*70)

        for i in range(num_tokens):
            # Compute with injection
            logits, probs = self.compute_logits(
                current_prompt,
                modulations,
                injection_config,
                active_layers
            )

            # Compute baseline for comparison
            baseline_logits, baseline_probs = self.compute_logits(
                current_prompt,
                None,
                None,
                active_layers
            )

            # Track KL divergence
            kl_div = F.kl_div(
                torch.log(probs + 1e-10),
                baseline_probs,
                reduction='sum'
            ).item()
            kl_divergences.append(kl_div)

            # Track monitored tokens
            if tracked_tokens and tracked_token_ids:
                for token, token_id in zip(tracked_tokens, tracked_token_ids):
                    if token_id < len(probs):
                        tracked_probs_evolution[token].append(probs[token_id].item())

            # Sample next token
            next_token_id = self.sample_token(
                probs,
                strategy=sampling_strategy,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )

            next_token = self.tokenizer.decode([next_token_id])
            generated_tokens.append(next_token)

            # Display generation progress
            if display_utils:
                baseline_prob = baseline_probs[next_token_id].item()
                injected_prob = probs[next_token_id].item()
                display_utils.display_generation_progress(i, next_token, baseline_prob, injected_prob, kl_div)
            else:
                print(f"Token {i+1}: {next_token}")

            # Update prompt
            current_prompt += next_token

            # Optional: Clear cache for long generations
            if i % 10 == 0 and i > 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Check for EOS token
            if next_token_id == self.tokenizer.eos_token_id:
                print(f"{Colors.YELLOW}[EOS token generated - stopping]{Colors.END}")
                break

        # Display final generation
        print("\n" + "="*70)
        print(f"{Colors.BOLD}Complete generation:{Colors.END}")
        generated_text = ''.join(generated_tokens)
        print(f"\"{Colors.CYAN}{initial_prompt}{Colors.GREEN}{generated_text}{Colors.END}\"")

        # Show tracking evolution if tokens were tracked
        if display_utils and tracked_tokens and any(len(v) > 0 for v in tracked_probs_evolution.values()):
            display_utils.display_tracking_evolution(tracked_probs_evolution)

        # Show KL divergence trend
        if kl_divergences:
            import numpy as np
            print(f"\n{Colors.BOLD}KL Divergence Statistics:{Colors.END}")
            print(f"  Mean: {np.mean(kl_divergences):.4f}")
            print(f"  Max: {max(kl_divergences):.4f}")
            print(f"  Min: {min(kl_divergences):.4f}")
            print(f"  Final: {kl_divergences[-1]:.4f}")

        return generated_tokens