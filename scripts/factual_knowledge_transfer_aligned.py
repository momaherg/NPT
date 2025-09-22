#!/usr/bin/env python3
"""
Factual Knowledge Transfer with Proper Position Alignment

This version correctly aligns extraction and injection positions.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlignedModulationTransfer:
    """Properly align extraction and injection positions for modulation transfer."""
    
    def __init__(self, model, tokenizer, device, active_npt_layers):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.active_npt_layers = active_npt_layers
    
    def extract_at_generation(
        self,
        prompt: str,
        target_token: str,
        layer_idx: int
    ) -> Dict:
        """
        Extract modulation when GENERATING the target token.
        This happens at position len(prompt_tokens).
        """
        # Tokenize separately
        prompt_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        target_ids = self.tokenizer(target_token, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        
        # The model sees: [prompt_tokens, target_token]
        # We extract at position where target is being GENERATED from prompt
        extraction_position = prompt_ids.shape[1]  # This is where generation happens
        
        # Create input for extraction: prompt + target
        full_ids = torch.cat([prompt_ids, target_ids[:, :1]], dim=1)
        
        logger.info(f"EXTRACTION:")
        logger.info(f"  Prompt: '{prompt}' ({prompt_ids.shape[1]} tokens)")
        logger.info(f"  Target: '{target_token}'")
        logger.info(f"  Extraction position: {extraction_position} (where '{target_token}' is generated)")
        
        # Extract modulation at the generation position
        modulation = self._run_extraction(full_ids, extraction_position, layer_idx)
        modulation['extraction_position'] = extraction_position
        modulation['prompt_length'] = prompt_ids.shape[1]
        
        return modulation
    
    def inject_at_generation(
        self,
        prompt: str,
        source_modulation: Dict,
        layer_idx: int
    ) -> Dict:
        """
        Inject modulation at the position where we want to influence generation.
        This should be at the SAME relative position as extraction.
        """
        # Tokenize target prompt
        prompt_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        
        # CRITICAL: We need to inject at the position where generation happens
        # This is AFTER the prompt, not at the last token of the prompt
        injection_position = prompt_ids.shape[1]  # Same relative position as extraction
        
        logger.info(f"INJECTION:")
        logger.info(f"  Prompt: '{prompt}' ({prompt_ids.shape[1]} tokens)")
        logger.info(f"  Injection position: {injection_position} (for next token generation)")
        logger.info(f"  Source extraction was at position: {source_modulation['extraction_position']}")
        
        # For injection during generation, we need to handle this differently
        # Option 1: Add a dummy token and inject at its position
        # Option 2: Use generation with prefix and inject at the boundary
        
        # Here we'll compute logits for the next position
        with torch.no_grad():
            # Run model up to prompt end
            outputs_no_injection = self.model(prompt_ids)
            logits_no_injection = outputs_no_injection.logits[0, -1]  # Last position
            
            # Now run with injection at the generation boundary
            # This requires intercepting at the right position during forward pass
            outputs_with_injection = self._run_injection_at_boundary(
                prompt_ids, source_modulation, layer_idx
            )
            logits_with_injection = outputs_with_injection[0, -1]
        
        # Compare probabilities
        probs_no_injection = F.softmax(logits_no_injection, dim=-1)
        probs_with_injection = F.softmax(logits_with_injection, dim=-1)
        
        return {
            'probs_no_injection': probs_no_injection,
            'probs_with_injection': probs_with_injection,
            'injection_position': injection_position
        }
    
    def _run_injection_at_boundary(self, input_ids, source_modulation, layer_idx):
        """
        Special injection that happens at the generation boundary.
        
        The trick: We run the model normally but inject modulation
        as if there was an additional token being generated.
        """
        # This is complex because we need to inject at a position
        # that doesn't exist yet in the input
        
        # Solution: Add a dummy token, inject at its position, 
        # but only use the logits from the previous position
        
        # Add dummy token
        dummy_token = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        extended_ids = torch.cat([
            input_ids,
            torch.tensor([[dummy_token]], device=self.device)
        ], dim=1)
        
        # Now injection position aligns with the dummy token position
        injection_position = input_ids.shape[1]
        
        # Set up injection hook
        def injection_hook(module, input, output):
            # Only inject at the specific position
            if isinstance(output[0], tuple):
                # Dual modulation
                (v_a_gate, v_b_gate), (v_a_up, v_b_up) = output
                
                # Check if we're at the right position
                if v_a_gate.shape[1] > injection_position:
                    # Clone and modify
                    v_a_gate_new = v_a_gate.clone()
                    v_b_gate_new = v_b_gate.clone()
                    v_a_up_new = v_a_up.clone()
                    v_b_up_new = v_b_up.clone()
                    
                    # Inject at the boundary position
                    v_a_gate_new[:, injection_position] = source_modulation['v_a_gate'][:, 0]
                    v_b_gate_new[:, injection_position] = source_modulation['v_b_gate'][:, 0]
                    v_a_up_new[:, injection_position] = source_modulation['v_a_up'][:, 0]
                    v_b_up_new[:, injection_position] = source_modulation['v_b_up'][:, 0]
                    
                    return (v_a_gate_new, v_b_gate_new), (v_a_up_new, v_b_up_new)
            
            return output
        
        # Register hook
        layer = self.model.model.layers[layer_idx]
        handle = layer.np_component.register_forward_hook(injection_hook)
        
        # Run forward pass
        with torch.no_grad():
            outputs = self.model(extended_ids)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        
        # Remove hook
        handle.remove()
        
        # Return logits (excluding the dummy position)
        return logits[:, :input_ids.shape[1]]
    
    def _run_extraction(self, input_ids, position, layer_idx):
        """Extract modulation at specific position."""
        extracted_modulation = {}
        
        def extraction_hook(module, input, output):
            if isinstance(output[0], tuple):
                # Dual modulation
                (v_a_gate, v_b_gate), (v_a_up, v_b_up) = output
                
                # Extract at specific position
                extracted_modulation['v_a_gate'] = v_a_gate[:, position:position+1].clone()
                extracted_modulation['v_b_gate'] = v_b_gate[:, position:position+1].clone()
                extracted_modulation['v_a_up'] = v_a_up[:, position:position+1].clone()
                extracted_modulation['v_b_up'] = v_b_up[:, position:position+1].clone()
        
        # Register hook
        layer = self.model.model.layers[layer_idx]
        handle = layer.np_component.register_forward_hook(extraction_hook)
        
        # Run forward
        with torch.no_grad():
            self.model(input_ids)
        
        # Remove hook
        handle.remove()
        
        return extracted_modulation


def demonstrate_position_alignment():
    """
    Demonstrate the position alignment problem and solution.
    """
    
    # Example tokenization (simplified)
    print("Position Alignment Problem:")
    print("="*60)
    
    # Source
    source_prompt = "The capital of France is"
    source_tokens = ["The", "capital", "of", "France", "is"]  # 5 tokens
    source_answer = "Paris"
    
    print(f"SOURCE: '{source_prompt}' -> '{source_answer}'")
    print(f"Tokens: {source_tokens}")
    print(f"Positions: {list(range(len(source_tokens)))}")
    print(f"Answer generated at position: {len(source_tokens)} (position 5)")
    print()
    
    # Target
    target_prompt = "The capital of Germany is"
    target_tokens = ["The", "capital", "of", "Germany", "is"]  # 5 tokens
    target_answer = "Berlin"
    
    print(f"TARGET: '{target_prompt}' -> '{target_answer}'")
    print(f"Tokens: {target_tokens}")
    print(f"Positions: {list(range(len(target_tokens)))}")
    print(f"Answer generated at position: {len(target_tokens)} (position 5)")
    print()
    
    print("WRONG APPROACH (current bug):")
    print(f"  Extract at: position 5 (generating '{source_answer}')")
    print(f"  Inject at: position 4 (last token of prompt = 'is')")
    print(f"  ❌ Positions don't align!")
    print()
    
    print("CORRECT APPROACH:")
    print(f"  Extract at: position 5 (generating '{source_answer}')")
    print(f"  Inject at: position 5 (for generating answer)")
    print(f"  ✅ Positions align!")
    print()
    
    print("Alternative approaches:")
    print("1. Token-based alignment: Match 'is' token in both")
    print("2. Relative alignment: Last token of prompt in both")
    print("3. Semantic alignment: Answer generation position in both")
    print("4. Attention-based: Find similar attention patterns")


if __name__ == "__main__":
    demonstrate_position_alignment()