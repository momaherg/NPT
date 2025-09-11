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
        single_layer_mode: bool = False,
        num_ranks: int = 1,  # NEW: number of rank-1 components for rank-k updates
        init_strategy: str = "improved",  # NEW: "improved" or "conservative"
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.single_layer_mode = single_layer_mode
        self.num_ranks = num_ranks
        self.init_strategy = init_strategy
        
        # For single layer mode, adjust rank based on num_ranks
        if single_layer_mode:
            # Distribute capacity across multiple ranks
            if num_ranks == 1:
                self.rank = max(256, rank * 4)  # Original behavior
            else:
                # Keep total capacity similar but distributed
                total_capacity = max(256, rank * 4)
                self.rank = max(64, total_capacity // num_ranks)
            self.init_scale = min(0.001, init_scale)  # Very small initialization
        else:
            self.rank = rank
            self.init_scale = init_scale
        
        # Create weight matrices - backward compatible structure
        if num_ranks == 1:
            # Original single rank-1: use Parameters directly for backward compatibility
            self.W_down = nn.Parameter(torch.empty(d_model, self.rank))
            self.W_a_up = nn.Parameter(torch.empty(self.rank, d_model))
            self.W_b_up = nn.Parameter(torch.empty(self.rank, d_ffn))
        else:
            # Multiple rank-1 components: use ParameterLists
            self.W_down = nn.ParameterList([
                nn.Parameter(torch.empty(d_model, self.rank))
                for _ in range(num_ranks)
            ])
            self.W_a_up = nn.ParameterList([
                nn.Parameter(torch.empty(self.rank, d_model))
                for _ in range(num_ranks)
            ])
            self.W_b_up = nn.ParameterList([
                nn.Parameter(torch.empty(self.rank, d_ffn))
                for _ in range(num_ranks)
            ])
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize weights with small values to ensure low-magnitude updates initially.
        Uses a scaled uniform initialization to encourage stable training.
        
        For single-layer mode, uses special initialization to help v_a encode attention
        and v_b start with minimal modulation.
        """
        if self.num_ranks == 1:
            # Original rank-1 initialization
            if self.single_layer_mode:
                # Special initialization for single-layer NPT
                
                # W_down: Standard Xavier initialization
                nn.init.xavier_uniform_(self.W_down)
                
                # W_a_up: Initialize based on strategy
                if self.init_strategy == "improved":
                    # Stronger initialization for better gradient flow
                    if self.rank <= self.d_model:
                        eye = torch.eye(self.rank, self.d_model)
                        self.W_a_up.data = eye[:self.rank, :self.d_model] * 1.0  # Strong identity
                        self.W_a_up.data += torch.randn(self.rank, self.d_model) * 0.1  # Substantial noise
                    else:
                        num_repeats = (self.rank + self.d_model - 1) // self.d_model
                        eye = torch.eye(self.d_model)
                        repeated = eye.repeat(num_repeats, 1)[:self.rank, :]
                        self.W_a_up.data = repeated * 1.0
                        self.W_a_up.data += torch.randn(self.rank, self.d_model) * 0.1
                    
                    # W_b_up: Reasonable initialization for immediate gradient flow
                    std = (2.0 / (self.rank + self.d_ffn)) ** 0.5
                    nn.init.normal_(self.W_b_up, mean=0.0, std=std * 0.1)
                else:
                    # Conservative (original) initialization
                    if self.rank <= self.d_model:
                        eye = torch.eye(self.rank, self.d_model)
                        self.W_a_up.data = eye[:self.rank, :self.d_model] * 0.3
                        self.W_a_up.data += torch.randn(self.rank, self.d_model) * 0.01
                    else:
                        num_repeats = (self.rank + self.d_model - 1) // self.d_model
                        eye = torch.eye(self.d_model)
                        repeated = eye.repeat(num_repeats, 1)[:self.rank, :]
                        self.W_a_up.data = repeated * 0.3
                        self.W_a_up.data += torch.randn(self.rank, self.d_model) * 0.01
                    
                    nn.init.uniform_(self.W_b_up, -self.init_scale, self.init_scale)
                
            else:
                # Standard initialization for multi-layer mode
                
                # Initialize W_down with standard Xavier/Glorot initialization
                nn.init.xavier_uniform_(self.W_down)
                
                # Initialize W_a_up and W_b_up with smaller values to produce low-magnitude updates
                # This ensures the model starts close to the original transformer behavior
                nn.init.uniform_(self.W_a_up, -self.init_scale, self.init_scale)
                nn.init.uniform_(self.W_b_up, -self.init_scale, self.init_scale)
        else:
            # Rank-k initialization for multiple components
            for i in range(self.num_ranks):
                if self.single_layer_mode and i == 0:
                    # First component gets special attention-encoding initialization
                    nn.init.xavier_uniform_(self.W_down[i])
                    
                    # Identity-like for first W_a_up based on strategy
                    if self.init_strategy == "improved":
                        if self.rank <= self.d_model:
                            eye = torch.eye(self.rank, self.d_model)
                            self.W_a_up[i].data = eye[:self.rank, :self.d_model] * 1.0
                            self.W_a_up[i].data += torch.randn(self.rank, self.d_model) * 0.1
                        else:
                            num_repeats = (self.rank + self.d_model - 1) // self.d_model
                            eye = torch.eye(self.d_model)
                            repeated = eye.repeat(num_repeats, 1)[:self.rank, :]
                            self.W_a_up[i].data = repeated * 1.0
                            self.W_a_up[i].data += torch.randn(self.rank, self.d_model) * 0.1
                        
                        std = (2.0 / (self.rank + self.d_ffn)) ** 0.5
                        nn.init.normal_(self.W_b_up[i], mean=0.0, std=std * 0.1)
                    else:
                        # Conservative initialization
                        if self.rank <= self.d_model:
                            eye = torch.eye(self.rank, self.d_model)
                            self.W_a_up[i].data = eye[:self.rank, :self.d_model] * 0.3
                            self.W_a_up[i].data += torch.randn(self.rank, self.d_model) * 0.01
                        else:
                            num_repeats = (self.rank + self.d_model - 1) // self.d_model
                            eye = torch.eye(self.d_model)
                            repeated = eye.repeat(num_repeats, 1)[:self.rank, :]
                            self.W_a_up[i].data = repeated * 0.3
                            self.W_a_up[i].data += torch.randn(self.rank, self.d_model) * 0.01
                        
                        nn.init.uniform_(self.W_b_up[i], -self.init_scale, self.init_scale)
                else:
                    # Other components get diverse initialization
                    nn.init.xavier_uniform_(self.W_down[i])
                    
                    # Use orthogonal initialization for diversity across components
                    if i > 0:
                        # Orthogonal to encourage different representations
                        nn.init.orthogonal_(self.W_a_up[i], gain=self.init_scale)
                        nn.init.orthogonal_(self.W_b_up[i], gain=self.init_scale * 0.1)
                    else:
                        # Standard small initialization
                        nn.init.uniform_(self.W_a_up[i], -self.init_scale, self.init_scale)
                        nn.init.uniform_(self.W_b_up[i], -self.init_scale, self.init_scale)
    
    def forward(self, attn_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate v_a and v_b vectors from attention output.
        
        Args:
            attn_output: Attention output tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            Tuple of (v_a, v_b) where:
            - For num_ranks=1: v_a has shape (batch_size, seq_len, d_model)
                               v_b has shape (batch_size, seq_len, d_ffn)
            - For num_ranks>1: v_a has shape (batch_size, seq_len, num_ranks, d_model)
                               v_b has shape (batch_size, seq_len, num_ranks, d_ffn)
        """
        if self.num_ranks == 1:
            # Original rank-1 implementation
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
        else:
            # Rank-k implementation: generate multiple rank-1 components
            v_a_list = []
            v_b_list = []
            
            for i in range(self.num_ranks):
                # Each component has its own bottleneck
                intermediate_r = attn_output @ self.W_down[i]
                v_a_i = intermediate_r @ self.W_a_up[i]
                v_b_i = intermediate_r @ self.W_b_up[i]
                
                v_a_list.append(v_a_i)
                v_b_list.append(v_b_i)
            
            # Stack along a new dimension for rank-k representation
            v_a = torch.stack(v_a_list, dim=2)  # (batch, seq, num_ranks, d_model)
            v_b = torch.stack(v_b_list, dim=2)  # (batch, seq, num_ranks, d_ffn)
            
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
            v_a: Vector of shape (batch_size, seq_len, d_model) for rank-1
                 or (batch_size, seq_len, num_ranks, d_model) for rank-k
            v_b: Vector of shape (batch_size, seq_len, d_ffn) for rank-1
                 or (batch_size, seq_len, num_ranks, d_ffn) for rank-k
        
        Returns:
            Scalar regularization loss
        """
        # L2 norm squared of the vectors - works for both rank-1 and rank-k
        if v_a.dim() == 3:
            # Rank-1 case
            reg_loss = torch.mean(torch.sum(v_a ** 2, dim=-1)) + torch.mean(torch.sum(v_b ** 2, dim=-1))
        else:
            # Rank-k case: sum over all components
            reg_loss = torch.mean(torch.sum(v_a ** 2, dim=(-2, -1))) + torch.mean(torch.sum(v_b ** 2, dim=(-2, -1)))
        return reg_loss
    
    def extra_repr(self) -> str:
        """String representation for printing the module."""
        base_repr = f'd_model={self.d_model}, d_ffn={self.d_ffn}, rank={self.rank}, init_scale={self.init_scale}'
        if self.single_layer_mode:
            base_repr += ', single_layer_mode=True'
        if self.num_ranks > 1:
            base_repr += f', num_ranks={self.num_ranks}'
        return base_repr