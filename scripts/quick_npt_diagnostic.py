#!/usr/bin/env python3
"""
Quick NPT Model Diagnostic Script

Provides a fast overview of NPT model health and readiness.
Designed for quick checks during training or debugging.

Usage:
    python scripts/quick_npt_diagnostic.py --checkpoint path/to/checkpoint
    python scripts/quick_npt_diagnostic.py --checkpoint path/to/checkpoint --verbose
"""

import argparse
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple
import json
import logging
import numpy as np
from datetime import datetime
import time

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.npt import NPTLlamaModel, NPTConfig
from transformers import AutoTokenizer, LlamaForCausalLM

# Setup logging
logging.basicConfig(level=logging.WARNING)  # Minimal logging for quick run
logger = logging.getLogger(__name__)


class QuickNPTDiagnostic:
    """Fast NPT model diagnostic tool."""

    def __init__(self, checkpoint_path: str, model_name: str = "meta-llama/Llama-3.2-1B"):
        self.checkpoint_path = Path(checkpoint_path)
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Quick test prompt
        self.test_prompt = "The capital of France is"

    def load_and_check(self) -> Dict:
        """Quick load and basic check."""
        start_time = time.time()

        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load NPT model
            dtype = torch.float32  # Use float32 for compatibility with checkpoint
            npt_model = NPTLlamaModel.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map=self.device
            )

            # First, determine which layers to convert by examining the checkpoint
            checkpoint_file = None
            if self.checkpoint_path.exists():
                if self.checkpoint_path.is_file():
                    checkpoint_file = str(self.checkpoint_path)
                else:
                    weight_files = list(self.checkpoint_path.glob("*.pt"))
                    if weight_files:
                        checkpoint_file = str(weight_files[0])
                    else:
                        return {"error": "No .pt files found in checkpoint directory"}
            else:
                return {"error": "Checkpoint path does not exist"}

            # Detect NPT layers from checkpoint
            weights = torch.load(checkpoint_file, map_location='cpu')
            npt_layer_indices = set()
            for key in weights.keys():
                if 'layer_' in key and '_np.' in key:
                    layer_num = int(key.split('_')[1])
                    npt_layer_indices.add(layer_num)

            if not npt_layer_indices:
                return {"error": "No NPT layers found in checkpoint"}

            # Convert model to NPT
            from src.npt import NPTConfig
            npt_config = NPTConfig(
                layers_to_convert=sorted(list(npt_layer_indices)),
                np_rank=256,  # Inferred from checkpoint
                num_ranks=4   # Inferred from checkpoint structure
            )
            npt_model.convert_to_npt(npt_config)

            # Now load the weights
            npt_model.load_npt_weights(checkpoint_file)

            # Ensure model is on correct device
            npt_model = npt_model.to(self.device)
            npt_model.eval()

            # Get NPT layers
            npt_layers = list(npt_model.npt_layers.keys())

            # Quick forward pass test
            inputs = tokenizer(self.test_prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                # Test NPT mode
                npt_model.set_npt_mode(True)
                npt_output = npt_model(**inputs)

                # Test standard mode
                npt_model.set_npt_mode(False)
                standard_output = npt_model(**inputs)

            # Check for issues
            issues = []
            if torch.isnan(npt_output.logits).any():
                issues.append("NaN values in NPT output")
            if torch.isinf(npt_output.logits).any():
                issues.append("Infinite values in NPT output")

            # Quick MSE between modes
            mse_between_modes = F.mse_loss(npt_output.logits, standard_output.logits).item()

            load_time = time.time() - start_time

            return {
                "status": "success",
                "load_time": load_time,
                "npt_layers": npt_layers,
                "num_npt_layers": len(npt_layers),
                "mse_between_modes": mse_between_modes,
                "issues": issues,
                "model_info": {
                    "device": str(npt_model.device),
                    "dtype": str(npt_model.dtype)
                }
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "load_time": time.time() - start_time
            }

    def run_accuracy_test(self, checkpoint_path: str) -> Dict:
        """Quick accuracy comparison test."""
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load models
            dtype = torch.float32  # Use float32 for compatibility
            npt_model = NPTLlamaModel.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map=self.device
            )

            original_model = LlamaForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map=self.device
            )

            # Detect and convert NPT layers from checkpoint
            checkpoint_file = checkpoint_path
            if Path(checkpoint_path).is_dir():
                weight_files = list(Path(checkpoint_path).glob("*.pt"))
                if weight_files:
                    checkpoint_file = str(weight_files[0])

            if Path(checkpoint_file).exists():
                # Detect NPT layers from checkpoint
                weights = torch.load(checkpoint_file, map_location='cpu')
                npt_layer_indices = set()
                for key in weights.keys():
                    if 'layer_' in key and '_np.' in key:
                        layer_num = int(key.split('_')[1])
                        npt_layer_indices.add(layer_num)

                if npt_layer_indices:
                    # Convert model to NPT
                    from src.npt import NPTConfig
                    npt_config = NPTConfig(
                        layers_to_convert=sorted(list(npt_layer_indices)),
                        np_rank=256,  # Inferred from checkpoint
                        num_ranks=4   # Inferred from checkpoint structure
                    )
                    npt_model.convert_to_npt(npt_config)
                    npt_model.load_npt_weights(checkpoint_file)
                    npt_model = npt_model.to(self.device)

            npt_model.eval()
            original_model.eval()

            # Test prompts for quick accuracy check
            test_prompts = [
                "The capital of France is",
                "2 + 2 equals",
                "The largest planet in our solar system is"
            ]

            npt_layers = list(npt_model.npt_layers.keys())
            mse_scores = []
            cosine_sims = []

            for prompt in test_prompts:
                inputs = tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    # NPT output
                    npt_model.set_npt_mode(True)
                    npt_out = npt_model(**inputs)

                    # Original output
                    orig_out = original_model(**inputs)

                # Calculate metrics
                mse = F.mse_loss(npt_out.logits, orig_out.logits).item()
                cos_sim = F.cosine_similarity(
                    npt_out.logits.view(-1),
                    orig_out.logits.view(-1),
                    dim=0
                ).item()

                mse_scores.append(mse)
                cosine_sims.append(cos_sim)

            avg_mse = np.mean(mse_scores)
            avg_cosine = np.mean(cosine_sims)

            return {
                "avg_mse": avg_mse,
                "avg_cosine_similarity": avg_cosine,
                "individual_mse": mse_scores,
                "individual_cosine": cosine_sims
            }

        except Exception as e:
            return {"error": str(e)}

    def assess_readiness(self, diagnostic_result: Dict, accuracy_result: Dict) -> Dict:
        """Assess model readiness based on diagnostic results."""
        if diagnostic_result.get("status") != "success":
            return {
                "ready": False,
                "confidence": 0.0,
                "reason": f"Model loading failed: {diagnostic_result.get('error', 'Unknown error')}",
                "recommendation": "Check checkpoint path and model compatibility"
            }

        # Check for critical issues
        if diagnostic_result.get("issues"):
            return {
                "ready": False,
                "confidence": 0.0,
                "reason": f"Critical issues detected: {', '.join(diagnostic_result['issues'])}",
                "recommendation": "Fix NaN/Inf issues before proceeding"
            }

        # Assess based on accuracy if available
        if "error" not in accuracy_result:
            avg_mse = accuracy_result.get("avg_mse", float('inf'))
            avg_cosine = accuracy_result.get("avg_cosine_similarity", 0.0)

            if avg_mse < 0.05 and avg_cosine > 0.95:
                return {
                    "ready": True,
                    "confidence": 0.9,
                    "reason": f"Excellent performance: MSE={avg_mse:.4f}, Cosine={avg_cosine:.4f}",
                    "recommendation": "Model ready for use"
                }
            elif avg_mse < 0.1 and avg_cosine > 0.9:
                return {
                    "ready": True,
                    "confidence": 0.7,
                    "reason": f"Good performance: MSE={avg_mse:.4f}, Cosine={avg_cosine:.4f}",
                    "recommendation": "Model ready with minor limitations"
                }
            elif avg_mse < 0.5 and avg_cosine > 0.7:
                return {
                    "ready": False,
                    "confidence": 0.4,
                    "reason": f"Moderate performance: MSE={avg_mse:.4f}, Cosine={avg_cosine:.4f}",
                    "recommendation": "Continue training - getting close"
                }
            else:
                return {
                    "ready": False,
                    "confidence": 0.1,
                    "reason": f"Poor performance: MSE={avg_mse:.4f}, Cosine={avg_cosine:.4f}",
                    "recommendation": "Significant training needed"
                }

        # Fallback assessment based on basic checks
        mse_between_modes = diagnostic_result.get("mse_between_modes", float('inf'))
        if mse_between_modes < 0.1:
            return {
                "ready": True,
                "confidence": 0.6,
                "reason": "Basic checks passed",
                "recommendation": "Run full diagnostic for detailed assessment"
            }
        else:
            return {
                "ready": False,
                "confidence": 0.3,
                "reason": "High difference between NPT and standard modes",
                "recommendation": "Continue training"
            }

    def estimate_training_progress(self, accuracy_result: Dict) -> Dict:
        """Estimate training progress and remaining work."""
        if "error" in accuracy_result:
            return {"error": "Cannot estimate progress without accuracy data"}

        avg_mse = accuracy_result.get("avg_mse", float('inf'))

        # MSE-based progress estimation
        if avg_mse >= 1.0:
            progress = 0.1
            remaining = "Significant training needed (>10k steps)"
        elif avg_mse >= 0.5:
            progress = 0.3
            remaining = "Moderate training needed (5k-10k steps)"
        elif avg_mse >= 0.1:
            progress = 0.7
            remaining = "Light training needed (1k-5k steps)"
        elif avg_mse >= 0.05:
            progress = 0.9
            remaining = "Fine-tuning needed (<1k steps)"
        else:
            progress = 1.0
            remaining = "Training complete"

        return {
            "progress_estimate": progress,
            "remaining_work": remaining,
            "current_mse": avg_mse,
            "target_mse": 0.05
        }

    def run_quick_diagnostic(self, include_accuracy: bool = True) -> Dict:
        """Run complete quick diagnostic."""
        print("🔍 Running quick NPT diagnostic...")

        # Basic diagnostic
        print("  Loading and checking model...")
        diagnostic_result = self.load_and_check()

        if diagnostic_result.get("status") != "success":
            return {
                "diagnostic": diagnostic_result,
                "timestamp": datetime.now().isoformat()
            }

        # Accuracy test if requested
        accuracy_result = {}
        if include_accuracy:
            print("  Running accuracy test...")
            accuracy_result = self.run_accuracy_test(str(self.checkpoint_path))

        # Readiness assessment
        print("  Assessing readiness...")
        readiness = self.assess_readiness(diagnostic_result, accuracy_result)

        # Progress estimation
        progress = {}
        if include_accuracy and "error" not in accuracy_result:
            progress = self.estimate_training_progress(accuracy_result)

        return {
            "diagnostic": diagnostic_result,
            "accuracy": accuracy_result,
            "readiness": readiness,
            "progress": progress,
            "timestamp": datetime.now().isoformat()
        }


def format_output(results: Dict, verbose: bool = False):
    """Format and print diagnostic results."""
    print("\n" + "="*60)
    print("NPT QUICK DIAGNOSTIC REPORT")
    print("="*60)

    diagnostic = results.get("diagnostic", {})
    accuracy = results.get("accuracy", {})
    readiness = results.get("readiness", {})
    progress = results.get("progress", {})

    # Basic info
    if diagnostic.get("status") == "success":
        print(f"✅ Model Status: LOADED")
        print(f"⏱️  Load Time: {diagnostic['load_time']:.2f}s")
        print(f"🧠 NPT Layers: {diagnostic['num_npt_layers']} layers {diagnostic['npt_layers']}")
        print(f"💾 Device: {diagnostic['model_info']['device']}")

        if diagnostic.get("issues"):
            print(f"⚠️  Issues: {', '.join(diagnostic['issues'])}")
    else:
        print(f"❌ Model Status: FAILED")
        print(f"❌ Error: {diagnostic.get('error', 'Unknown error')}")
        return

    # Readiness assessment
    ready_emoji = "✅" if readiness.get("ready") else "❌"
    print(f"\n{ready_emoji} Readiness: {'READY' if readiness.get('ready') else 'NOT READY'}")
    print(f"🎯 Confidence: {readiness.get('confidence', 0):.1%}")
    print(f"📋 Reason: {readiness.get('reason', 'Unknown')}")
    print(f"💡 Recommendation: {readiness.get('recommendation', 'No recommendation')}")

    # Accuracy results
    if accuracy and "error" not in accuracy:
        print(f"\n📊 ACCURACY METRICS:")
        print(f"   MSE Loss: {accuracy['avg_mse']:.6f}")
        print(f"   Cosine Similarity: {accuracy['avg_cosine_similarity']:.4f}")

        # Loss interpretation
        mse = accuracy['avg_mse']
        if mse < 0.05:
            mse_status = "EXCELLENT 🌟"
        elif mse < 0.1:
            mse_status = "GOOD ✅"
        elif mse < 0.5:
            mse_status = "FAIR ⚠️"
        else:
            mse_status = "POOR ❌"

        print(f"   Loss Status: {mse_status}")

    # Progress estimation
    if progress and "error" not in progress:
        print(f"\n📈 TRAINING PROGRESS:")
        progress_percent = progress['progress_estimate'] * 100
        print(f"   Estimated Progress: {progress_percent:.1f}%")
        print(f"   Remaining Work: {progress['remaining_work']}")
        print(f"   Current MSE: {progress['current_mse']:.6f}")
        print(f"   Target MSE: {progress['target_mse']:.6f}")

    # Verbose details
    if verbose:
        print(f"\n📝 DETAILED RESULTS:")
        if accuracy and "error" not in accuracy:
            print(f"   Individual MSE scores: {[f'{x:.4f}' for x in accuracy['individual_mse']]}")
            print(f"   Individual Cosine scores: {[f'{x:.4f}' for x in accuracy['individual_cosine']]}")

        if diagnostic.get("mse_between_modes") is not None:
            print(f"   MSE between NPT/Standard modes: {diagnostic['mse_between_modes']:.6f}")

    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description="Quick NPT Model Diagnostic")
    parser.add_argument("--checkpoint", required=True, help="Path to NPT checkpoint")
    parser.add_argument("--model_name", default="meta-llama/Llama-3.2-1B", help="Base model name")
    parser.add_argument("--verbose", action="store_true", help="Show detailed results")
    parser.add_argument("--no_accuracy", action="store_true", help="Skip accuracy test for faster run")
    parser.add_argument("--output", help="Save results to JSON file")

    args = parser.parse_args()

    # Create diagnostic tool
    diagnostic = QuickNPTDiagnostic(
        checkpoint_path=args.checkpoint,
        model_name=args.model_name
    )

    # Run diagnostic
    start_time = time.time()
    results = diagnostic.run_quick_diagnostic(include_accuracy=not args.no_accuracy)
    total_time = time.time() - start_time

    # Add timing info
    results["total_time"] = total_time

    # Format and display results
    format_output(results, verbose=args.verbose)

    print(f"\n⏱️  Total diagnostic time: {total_time:.2f}s")

    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to {args.output}")


if __name__ == "__main__":
    main()