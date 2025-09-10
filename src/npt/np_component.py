"""
Neuro-Plastic Component (NP Component) for the NPT architecture.

This module implements the core component that generates rank-1 weight updates
from attention outputs in the Neuro-Plastic Transformer.
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional


class NPComponent(nn.Module):
    """
    Neuro-Plastic Component that generates rank-1 weight updates.
    
    This component takes attention output and generates two vectors (v_a and v_b)
    whose outer product forms a rank-1 weight delta for the MLP layer.
    
    Args:
        d_model: Model hidden dimension (e.g., 2048 for 1B, 4096 for 8B)
        d_ffn: Feed-forward network intermediate dimension (e.g., 8192 for 1B, 14336 for 8B)
        rank: Low-rank bottleneck dimension (default: 64)
        init_scale: Scale factor for weight initialization (default: 0.01)
        single_layer_mode: Whether this is the only NPT layer (requires special initialization)
    """
    
    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        rank: int = 64,
        init_scale: float = 0.01,
        single_layer_mode: bool = False
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.single_layer_mode = single_layer_mode
        
        # For single layer mode, use much higher rank
        if single_layer_mode:
            self.rank = max(256, rank * 4)  # At least 256, or 4x the specified rank
            self.init_scale = min(0.001, init_scale)  # Very small initialization
        else:
            self.rank = rank
            self.init_scale = init_scale
        
        # Three trainable weight matrices
        self.W_down = nn.Parameter(torch.empty(d_model, self.rank))
        self.W_a_up = nn.Parameter(torch.empty(self.rank, d_model))
        self.W_b_up = nn.Parameter(torch.empty(self.rank, d_ffn))
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize weights with small values to ensure low-magnitude updates initially.
        Uses a scaled uniform initialization to encourage stable training.
        
        For single-layer mode, uses special initialization to help v_a encode attention
        and v_b start with minimal modulation.
        """
        if self.single_layer_mode:
            # Special initialization for single-layer NPT
            
            # W_down: Standard Xavier initialization
            nn.init.xavier_uniform_(self.W_down)
            
            # W_a_up: Initialize with identity-like structure to preserve attention
            # This helps v_a directly encode attention information initially
            if self.rank <= self.d_model:
                # Create identity-like initialization
                eye = torch.eye(self.rank, self.d_model)
                self.W_a_up.data = eye[:self.rank, :self.d_model] * 0.3
                # Add small random noise
                self.W_a_up.data += torch.randn(self.rank, self.d_model) * 0.01
            else:
                # If rank > d_model, use repeated identity pattern
                num_repeats = (self.rank + self.d_model - 1) // self.d_model
                eye = torch.eye(self.d_model)
                repeated = eye.repeat(num_repeats, 1)[:self.rank, :]
                self.W_a_up.data = repeated * 0.3
                self.W_a_up.data += torch.randn(self.rank, self.d_model) * 0.01
            
            # W_b_up: Very small initialization to start with minimal modulation
            nn.init.uniform_(self.W_b_up, -self.init_scale, self.init_scale)
            
        else:
            # Standard initialization for multi-layer mode
            
            # Initialize W_down with standard Xavier/Glorot initialization
            nn.init.xavier_uniform_(self.W_down)
            
            # Initialize W_a_up and W_b_up with smaller values to produce low-magnitude updates
            # This ensures the model starts close to the original transformer behavior
            nn.init.uniform_(self.W_a_up, -self.init_scale, self.init_scale)
            nn.init.uniform_(self.W_b_up, -self.init_scale, self.init_scale)
    
    def forward(self, attn_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate v_a and v_b vectors from attention output.
        
        Args:
            attn_output: Attention output tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            Tuple of (v_a, v_b) where:
            - v_a has shape (batch_size, seq_len, d_model)
            - v_b has shape (batch_size, seq_len, d_ffn)
        """
        # Project to low-rank space
        # (batch_size, seq_len, d_model) @ (d_model, rank) -> (batch_size, seq_len, rank)
        intermediate_r = attn_output @ self.W_down
        
        # Generate v_a: project back to model dimension
        # (batch_size, seq_len, rank) @ (rank, d_model) -> (batch_size, seq_len, d_model)
        v_a = intermediate_r @ self.W_a_up
        
        # Generate v_b: project to FFN dimension
        # (batch_size, seq_len, rank) @ (rank, d_ffn) -> (batch_size, seq_len, d_ffn)
        v_b = intermediate_r @ self.W_b_up
        
        return v_a, v_b
    
    def compute_delta_w(self, attn_output: torch.Tensor) -> torch.Tensor:
        """
        Compute the rank-1 weight delta ΔW from attention output.
        
        This is a convenience method that computes the outer product of v_b and v_a.
        Note: In practice, applying the weight update can be done more efficiently
        without explicitly forming the full matrix.
        
        Args:
            attn_output: Attention output tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            ΔW tensor of shape (batch_size, seq_len, d_ffn, d_model)
        """
        v_a, v_b = self.forward(attn_output)
        
        # Compute outer product for each token
        # We need to add dimensions for the outer product
        # v_b: (batch_size, seq_len, d_ffn) -> (batch_size, seq_len, d_ffn, 1)
        # v_a: (batch_size, seq_len, d_model) -> (batch_size, seq_len, 1, d_model)
        v_b_expanded = v_b.unsqueeze(-1)  # (batch_size, seq_len, d_ffn, 1)
        v_a_expanded = v_a.unsqueeze(-2)  # (batch_size, seq_len, 1, d_model)
        
        # Outer product: (batch_size, seq_len, d_ffn, 1) * (batch_size, seq_len, 1, d_model)
        # -> (batch_size, seq_len, d_ffn, d_model)
        delta_w = v_b_expanded * v_a_expanded
        
        return delta_w
    
    def get_regularization_loss(self, v_a: torch.Tensor, v_b: torch.Tensor) -> torch.Tensor:
        """
        Compute L2 regularization loss on generated vectors.
        
        This encourages low-magnitude weight updates during training.
        
        Args:
            v_a: Vector of shape (batch_size, seq_len, d_model)
            v_b: Vector of shape (batch_size, seq_len, d_ffn)
        
        Returns:
            Scalar regularization loss
        """
        # L2 norm squared of the vectors
        reg_loss = torch.mean(torch.sum(v_a ** 2, dim=-1)) + torch.mean(torch.sum(v_b ** 2, dim=-1))
        return reg_loss
    
    def extra_repr(self) -> str:
        """String representation for printing the module."""
        base_repr = f'd_model={self.d_model}, d_ffn={self.d_ffn}, rank={self.rank}, init_scale={self.init_scale}'
        if self.single_layer_mode:
            base_repr += ', single_layer_mode=True'
        return base_repr