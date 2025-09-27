"""
Data types and structures for Interactive Knowledge Transfer.
"""

import torch
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple


class Colors:
    """ANSI color codes for terminal output."""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


@dataclass
class ModulationData:
    """Container for modulation tensors with rank-k support."""
    name: str
    layer_idx: int
    source_prompt: str
    extraction_position: int
    timestamp: datetime = field(default_factory=datetime.now)

    # Single modulation (might be rank-k: shape [batch, 1, num_ranks, dim] or [batch, 1, dim])
    v_a: Optional[torch.Tensor] = None
    v_b: Optional[torch.Tensor] = None

    # Dual modulation components (each might be rank-k)
    v_a_gate: Optional[torch.Tensor] = None
    v_b_gate: Optional[torch.Tensor] = None
    v_a_up: Optional[torch.Tensor] = None
    v_b_up: Optional[torch.Tensor] = None

    # Triple modulation components (each might be rank-k)
    v_a_down: Optional[torch.Tensor] = None
    v_b_down: Optional[torch.Tensor] = None

    # Metadata
    modulation_type: str = "single"  # "single", "dual", or "triple"
    num_ranks: int = 1  # Number of rank-1 components
    magnitude: float = 0.0

    def get_tensors(self) -> List[Tuple[str, torch.Tensor]]:
        """Get all non-None tensors with their names."""
        tensors = []
        if self.modulation_type == "single":
            if self.v_a is not None:
                tensors.append(("v_a", self.v_a))
            if self.v_b is not None:
                tensors.append(("v_b", self.v_b))
        elif self.modulation_type == "dual":
            if self.v_a_gate is not None:
                tensors.append(("v_a_gate", self.v_a_gate))
            if self.v_b_gate is not None:
                tensors.append(("v_b_gate", self.v_b_gate))
            if self.v_a_up is not None:
                tensors.append(("v_a_up", self.v_a_up))
            if self.v_b_up is not None:
                tensors.append(("v_b_up", self.v_b_up))
        elif self.modulation_type == "triple":
            if self.v_a_gate is not None:
                tensors.append(("v_a_gate", self.v_a_gate))
            if self.v_b_gate is not None:
                tensors.append(("v_b_gate", self.v_b_gate))
            if self.v_a_up is not None:
                tensors.append(("v_a_up", self.v_a_up))
            if self.v_b_up is not None:
                tensors.append(("v_b_up", self.v_b_up))
            if self.v_a_down is not None:
                tensors.append(("v_a_down", self.v_a_down))
            if self.v_b_down is not None:
                tensors.append(("v_b_down", self.v_b_down))
        return tensors


@dataclass
class SessionState:
    """Maintains the state of an interactive session."""
    modulation_bank: Dict[str, Dict[int, ModulationData]] = field(default_factory=dict)
    tracked_tokens: List[str] = field(default_factory=list)
    tracked_token_ids: List[int] = field(default_factory=list)
    active_layers: Set[int] = field(default_factory=set)
    command_history: List[str] = field(default_factory=list)
    last_baseline_probs: Optional[torch.Tensor] = None
    last_prompt: Optional[str] = None