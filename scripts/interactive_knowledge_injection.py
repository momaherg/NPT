#!/usr/bin/env python3
"""
Interactive Knowledge Injection Experiment for NPT

This script allows interactive experimentation with permanent knowledge injection
using the Neuro-Plastic Transformer's rank-1 weight updates.

Features:
- Load pre-trained NPT checkpoint or initialize new model
- Ask questions to test current knowledge
- Inject new facts using attention-guided weight updates
- Test if injected knowledge persists
- Save modified model weights
"""

import argparse
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import json
import logging
from datetime import datetime
from colorama import init, Fore, Style

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.npt import NPTLlamaModel, NPTConfig
from transformers import AutoTokenizer, AutoConfig
import numpy as np

# Initialize colorama for colored output
init(autoreset=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KnowledgeInjector:
    """Handles knowledge injection into NPT models."""
    
    def __init__(
        self,
        model: NPTLlamaModel,
        tokenizer,
        layer_idx: int = 15,
        injection_strength: float = 1.0,
        device: str = "cuda"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self.injection_strength = injection_strength
        self.device = device
        
        # Store injected knowledge for tracking
        self.injected_facts = []
        
        # Ensure model is on device
        self.model = self.model.to(device)
        self.model.eval()
    
    def generate_response(self, prompt: str, max_length: int = 50) -> str:
        """Generate a response to a prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs.input_ids.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=min(input_ids.shape[1] + max_length, 512),
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    def extract_delta_weights(
        self,
        text: str,
        target_position: str = "last"
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Extract delta weights (v_a, v_b) from processing a text.
        
        Args:
            text: The text containing the fact to inject
            target_position: Position to extract weights from ("last", "first", "all")
        
        Returns:
            Tuple of (v_a, v_b, metadata)
        """
        # Tokenize the input
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs.input_ids.to(self.device)
        
        # Get the NPT layer
        if self.layer_idx not in self.model.npt_layers:
            raise ValueError(f"Layer {self.layer_idx} is not an NPT layer")
        
        npt_layer = self.model.npt_layers[self.layer_idx]
        
        # Storage for v_a and v_b
        v_a_collected = None
        v_b_collected = None
        attention_output = None
        
        # Hook to capture v_a and v_b
        def hook_fn(module, input, output):
            nonlocal v_a_collected, v_b_collected
            if isinstance(output, tuple) and len(output) == 2:
                v_a_collected, v_b_collected = output
        
        # Register hook
        handle = npt_layer.np_component.register_forward_hook(hook_fn)
        
        try:
            # Forward pass in NPT mode
            self.model.set_npt_mode(True)
            
            with torch.no_grad():
                # Process through model up to NPT layer
                hidden_states = self.model.model.embed_tokens(input_ids)
                
                # Create position embeddings
                batch_size, seq_len = input_ids.shape
                head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
                cos = torch.ones(batch_size, seq_len, head_dim, 
                                dtype=hidden_states.dtype, device=hidden_states.device)
                sin = torch.zeros(batch_size, seq_len, head_dim,
                                 dtype=hidden_states.dtype, device=hidden_states.device)
                position_embeddings = (cos, sin)
                
                # Process layers up to NPT layer
                for i in range(self.layer_idx):
                    layer = self.model.model.layers[i]
                    layer_out = layer(
                        hidden_states,
                        position_embeddings=position_embeddings,
                        use_cache=False,
                        output_attentions=False
                    )
                    hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out
                
                # Process NPT layer to trigger hook
                layer_out = npt_layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    use_cache=False,
                    output_attentions=True  # Get attention weights
                )
                
                if isinstance(layer_out, tuple) and len(layer_out) > 1:
                    hidden_states = layer_out[0]
                    attention_output = layer_out[1] if len(layer_out) > 1 else None
                else:
                    hidden_states = layer_out
        
        finally:
            handle.remove()
        
        if v_a_collected is None or v_b_collected is None:
            raise RuntimeError("Failed to collect v_a and v_b from NPT layer")
        
        # Select position based on target_position
        if target_position == "last":
            # Get the last non-padding token position
            seq_len = input_ids.shape[1]
            if self.tokenizer.pad_token_id is not None:
                # Find last non-padding position
                non_pad_mask = (input_ids != self.tokenizer.pad_token_id).squeeze(0)
                last_pos = non_pad_mask.nonzero()[-1].item() if non_pad_mask.any() else seq_len - 1
            else:
                last_pos = seq_len - 1
            
            v_a = v_a_collected[:, last_pos, :].squeeze(0)
            v_b = v_b_collected[:, last_pos, :].squeeze(0)
        elif target_position == "first":
            v_a = v_a_collected[:, 0, :].squeeze(0)
            v_b = v_b_collected[:, 0, :].squeeze(0)
        elif target_position == "all":
            # Average across all positions
            v_a = v_a_collected.mean(dim=1).squeeze(0)
            v_b = v_b_collected.mean(dim=1).squeeze(0)
        else:
            raise ValueError(f"Unknown target_position: {target_position}")
        
        # Compute some metadata
        metadata = {
            "text": text,
            "position": target_position,
            "v_a_norm": v_a.norm().item(),
            "v_b_norm": v_b.norm().item(),
            "delta_w_rank1_norm": (v_b.norm() * v_a.norm()).item(),
            "tokens": self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist()),
            "layer_idx": self.layer_idx
        }
        
        return v_a, v_b, metadata
    
    def inject_knowledge(
        self,
        fact_text: str,
        position: str = "last",
        accumulate: bool = False,
        alpha: Optional[float] = None
    ) -> Dict:
        """
        Inject knowledge by permanently modifying MLP weights.
        
        Args:
            fact_text: The fact to inject (e.g., "The president of the US is Mohamed Maher")
            position: Position to extract update from ("last", "first", "all")
            accumulate: If True, add to existing modifications; if False, reset first
            alpha: Scaling factor for the update (defaults to injection_strength)
        
        Returns:
            Dictionary with injection metadata
        """
        if alpha is None:
            alpha = self.injection_strength
        
        print(f"\n{Fore.YELLOW}Extracting knowledge representation...{Style.RESET_ALL}")
        
        # Extract v_a and v_b for this fact
        v_a, v_b, metadata = self.extract_delta_weights(fact_text, position)
        
        # Get the NPT layer
        npt_layer = self.model.npt_layers[self.layer_idx]
        
        # Apply the rank-1 update to the MLP weights
        with torch.no_grad():
            # Get current MLP input weights
            W_in = npt_layer.mlp.gate_proj.weight  # Shape: [intermediate_size, hidden_size]
            
            # Compute rank-1 update: ΔW = α * outer(v_b, v_a)
            # v_b shape: [intermediate_size], v_a shape: [hidden_size]
            delta_W = alpha * torch.outer(v_b, v_a)
            
            # Store original weights if first injection and not accumulating
            if not hasattr(self, 'original_weights') or not accumulate:
                self.original_weights = W_in.data.clone()
            
            # Apply update
            if not accumulate:
                # Reset to original first
                W_in.data = self.original_weights.clone()
            
            W_in.data += delta_W
            
            # Track injected fact
            injection_info = {
                "fact": fact_text,
                "timestamp": datetime.now().isoformat(),
                "alpha": alpha,
                "position": position,
                "v_a_norm": metadata["v_a_norm"],
                "v_b_norm": metadata["v_b_norm"],
                "delta_norm": delta_W.norm().item(),
                "weight_change_ratio": (delta_W.norm() / W_in.norm()).item()
            }
            self.injected_facts.append(injection_info)
        
        print(f"{Fore.GREEN}✓ Knowledge injected successfully!{Style.RESET_ALL}")
        print(f"  - Delta weight norm: {delta_W.norm().item():.6f}")
        print(f"  - Weight change ratio: {injection_info['weight_change_ratio']:.6f}")
        print(f"  - v_a norm: {metadata['v_a_norm']:.4f}, v_b norm: {metadata['v_b_norm']:.4f}")
        
        return injection_info
    
    def reset_weights(self):
        """Reset MLP weights to original state."""
        if hasattr(self, 'original_weights'):
            npt_layer = self.model.npt_layers[self.layer_idx]
            with torch.no_grad():
                npt_layer.mlp.gate_proj.weight.data = self.original_weights.clone()
            self.injected_facts = []
            print(f"{Fore.CYAN}Weights reset to original state.{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}No original weights stored. Nothing to reset.{Style.RESET_ALL}")
    
    def save_modified_model(self, save_path: str):
        """Save the model with injected knowledge."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save NPT weights
        self.model.save_npt_weights(save_path / "npt_weights_modified.pt")
        
        # Save injection history
        with open(save_path / "injection_history.json", "w") as f:
            json.dump(self.injected_facts, f, indent=2)
        
        print(f"{Fore.GREEN}✓ Modified model saved to {save_path}{Style.RESET_ALL}")


class InteractiveSession:
    """Interactive session for knowledge injection experiments."""
    
    def __init__(self, injector: KnowledgeInjector):
        self.injector = injector
        self.session_history = []
    
    def print_help(self):
        """Print help information."""
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}NPT Knowledge Injection - Interactive Commands:{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}ask <question>{Style.RESET_ALL} - Ask the model a question")
        print(f"{Fore.GREEN}inject <fact>{Style.RESET_ALL} - Inject a fact into the model")
        print(f"{Fore.GREEN}inject-multi{Style.RESET_ALL} - Inject multiple related facts")
        print(f"{Fore.GREEN}test <question>{Style.RESET_ALL} - Test if injected knowledge works")
        print(f"{Fore.GREEN}reset{Style.RESET_ALL} - Reset weights to original state")
        print(f"{Fore.GREEN}save <path>{Style.RESET_ALL} - Save modified model")
        print(f"{Fore.GREEN}history{Style.RESET_ALL} - Show injection history")
        print(f"{Fore.GREEN}strength <value>{Style.RESET_ALL} - Set injection strength (default: 1.0)")
        print(f"{Fore.GREEN}help{Style.RESET_ALL} - Show this help message")
        print(f"{Fore.GREEN}exit{Style.RESET_ALL} - Exit the session")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
    
    def run(self):
        """Run the interactive session."""
        self.print_help()
        
        while True:
            try:
                # Get user input
                user_input = input(f"\n{Fore.YELLOW}NPT> {Style.RESET_ALL}").strip()
                
                if not user_input:
                    continue
                
                # Parse command
                parts = user_input.split(None, 1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""
                
                # Handle commands
                if command == "exit":
                    print(f"{Fore.CYAN}Goodbye!{Style.RESET_ALL}")
                    break
                
                elif command == "help":
                    self.print_help()
                
                elif command == "ask":
                    if not args:
                        print(f"{Fore.RED}Please provide a question.{Style.RESET_ALL}")
                        continue
                    response = self.injector.generate_response(args)
                    print(f"\n{Fore.CYAN}Model:{Style.RESET_ALL} {response}")
                    self.session_history.append(("ask", args, response))
                
                elif command == "inject":
                    if not args:
                        print(f"{Fore.RED}Please provide a fact to inject.{Style.RESET_ALL}")
                        continue
                    
                    # Ask for injection parameters
                    position = input(f"Position (last/first/all) [{Fore.GREEN}last{Style.RESET_ALL}]: ").strip() or "last"
                    accumulate = input(f"Accumulate with previous injections? (y/n) [{Fore.GREEN}n{Style.RESET_ALL}]: ").strip().lower() == 'y'
                    
                    info = self.injector.inject_knowledge(args, position=position, accumulate=accumulate)
                    self.session_history.append(("inject", args, info))
                
                elif command == "inject-multi":
                    print(f"{Fore.YELLOW}Enter facts to inject (empty line to finish):{Style.RESET_ALL}")
                    facts = []
                    while True:
                        fact = input(f"  Fact {len(facts)+1}: ").strip()
                        if not fact:
                            break
                        facts.append(fact)
                    
                    if facts:
                        position = input(f"Position (last/first/all) [{Fore.GREEN}last{Style.RESET_ALL}]: ").strip() or "last"
                        for i, fact in enumerate(facts):
                            print(f"\n{Fore.YELLOW}Injecting fact {i+1}/{len(facts)}...{Style.RESET_ALL}")
                            info = self.injector.inject_knowledge(fact, position=position, accumulate=True)
                            self.session_history.append(("inject", fact, info))
                
                elif command == "test":
                    if not args:
                        print(f"{Fore.RED}Please provide a test question.{Style.RESET_ALL}")
                        continue
                    
                    print(f"\n{Fore.YELLOW}Testing injected knowledge...{Style.RESET_ALL}")
                    response = self.injector.generate_response(args)
                    print(f"\n{Fore.CYAN}Model:{Style.RESET_ALL} {response}")
                    
                    # Compare with original if available
                    if hasattr(self.injector, 'original_weights'):
                        print(f"\n{Fore.YELLOW}Comparing with original model...{Style.RESET_ALL}")
                        # Temporarily reset
                        npt_layer = self.injector.model.npt_layers[self.injector.layer_idx]
                        current_weights = npt_layer.mlp.gate_proj.weight.data.clone()
                        npt_layer.mlp.gate_proj.weight.data = self.injector.original_weights.clone()
                        
                        original_response = self.injector.generate_response(args)
                        print(f"{Fore.CYAN}Original:{Style.RESET_ALL} {original_response}")
                        
                        # Restore modified weights
                        npt_layer.mlp.gate_proj.weight.data = current_weights
                    
                    self.session_history.append(("test", args, response))
                
                elif command == "reset":
                    self.injector.reset_weights()
                
                elif command == "save":
                    save_path = args or f"experiments/injected_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    self.injector.save_modified_model(save_path)
                
                elif command == "history":
                    if self.injector.injected_facts:
                        print(f"\n{Fore.CYAN}Injection History:{Style.RESET_ALL}")
                        for i, fact_info in enumerate(self.injector.injected_facts, 1):
                            print(f"\n{i}. {fact_info['fact']}")
                            print(f"   Alpha: {fact_info['alpha']:.2f}, Position: {fact_info['position']}")
                            print(f"   Weight change: {fact_info['weight_change_ratio']:.6f}")
                    else:
                        print(f"{Fore.YELLOW}No facts injected yet.{Style.RESET_ALL}")
                
                elif command == "strength":
                    try:
                        strength = float(args)
                        self.injector.injection_strength = strength
                        print(f"{Fore.GREEN}Injection strength set to {strength}{Style.RESET_ALL}")
                    except ValueError:
                        print(f"{Fore.RED}Invalid strength value. Please provide a number.{Style.RESET_ALL}")
                
                else:
                    print(f"{Fore.RED}Unknown command: {command}. Type 'help' for available commands.{Style.RESET_ALL}")
            
            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}Use 'exit' to quit.{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
                logger.exception("Error in interactive session")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive knowledge injection experiment for NPT"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to NPT checkpoint directory"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Base model name if no checkpoint provided"
    )
    parser.add_argument(
        "--layer_idx",
        type=int,
        default=15,
        help="NPT layer index to use for injection"
    )
    parser.add_argument(
        "--injection_strength",
        type=float,
        default=1.0,
        help="Initial injection strength (scaling factor)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--demo_mode",
        action="store_true",
        help="Use small demo model for testing"
    )
    
    args = parser.parse_args()
    
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}NPT Knowledge Injection Experiment{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    
    # Load or create model
    if args.checkpoint:
        # Load from checkpoint
        checkpoint_path = Path(args.checkpoint)
        print(f"\n{Fore.YELLOW}Loading checkpoint from {checkpoint_path}...{Style.RESET_ALL}")
        
        # Load base model
        model = NPTLlamaModel.from_pretrained(args.model_name)
        
        # Detect NPT rank from checkpoint weights
        npt_weights_path = checkpoint_path / "npt_weights.pt"
        if npt_weights_path.exists():
            # Load weights to detect rank
            npt_weights = torch.load(npt_weights_path, map_location='cpu')
            # Get first weight to detect rank
            first_key = next(iter(npt_weights.keys()))
            if 'W_down' in first_key:
                # W_down shape is [hidden_size, np_rank]
                detected_rank = npt_weights[first_key].shape[1]
                print(f"  Detected np_rank={detected_rank} from checkpoint")
            else:
                # Find W_down weight
                for key, weight in npt_weights.items():
                    if 'W_down' in key:
                        detected_rank = weight.shape[1]
                        print(f"  Detected np_rank={detected_rank} from checkpoint")
                        break
                else:
                    detected_rank = 256  # Fallback
                    print(f"  Using default np_rank={detected_rank}")
        else:
            detected_rank = 256
            print(f"  No weights found, using default np_rank={detected_rank}")
        
        # Convert layer to NPT if needed with detected rank
        if args.layer_idx not in model.npt_layers:
            npt_config = NPTConfig(
                layers_to_convert=[args.layer_idx],
                np_rank=detected_rank,
                np_init_scale=0.001
            )
            model.convert_to_npt(npt_config)
        
        # Load NPT weights
        model.load_npt_weights(npt_weights_path)
        print(f"{Fore.GREEN}✓ Checkpoint loaded successfully!{Style.RESET_ALL}")
    
    else:
        # Create new model
        print(f"\n{Fore.YELLOW}Initializing model: {args.model_name}...{Style.RESET_ALL}")
        
        if args.demo_mode:
            from transformers import LlamaConfig
            config = LlamaConfig(
                hidden_size=256,
                intermediate_size=1024,
                num_hidden_layers=4,
                num_attention_heads=8,
                num_key_value_heads=4,
                vocab_size=128256,
            )
            model = NPTLlamaModel(config)
            args.layer_idx = min(args.layer_idx, 3)  # Adjust for demo model
        else:
            model = NPTLlamaModel.from_pretrained(args.model_name)
        
        # Convert specified layer to NPT
        npt_config = NPTConfig(
            layers_to_convert=[args.layer_idx],
            np_rank=256 if not args.demo_mode else 32,
            np_init_scale=0.001
        )
        model.convert_to_npt(npt_config)
        print(f"{Fore.GREEN}✓ Model initialized with NPT layer {args.layer_idx}{Style.RESET_ALL}")
    
    # Load tokenizer
    print(f"\n{Fore.YELLOW}Loading tokenizer...{Style.RESET_ALL}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    except:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create injector
    injector = KnowledgeInjector(
        model=model,
        tokenizer=tokenizer,
        layer_idx=args.layer_idx,
        injection_strength=args.injection_strength,
        device=args.device
    )
    
    # Print model info
    param_counts = model.count_parameters()
    print(f"\n{Fore.CYAN}Model Information:{Style.RESET_ALL}")
    print(f"  Total parameters: {param_counts['total']:,}")
    print(f"  NPT parameters: {param_counts['npt']:,}")
    print(f"  NPT layer: {args.layer_idx}")
    print(f"  Device: {args.device}")
    
    # Run interactive session
    session = InteractiveSession(injector)
    session.run()


if __name__ == "__main__":
    main()