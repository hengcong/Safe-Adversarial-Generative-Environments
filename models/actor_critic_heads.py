import torch
import torch.nn as nn

# models/actor_critic_heads.py
import torch
import torch.nn as nn
from typing import Optional, Dict

class ActorHeads(nn.Module):
    """
    Actor heads for heterogeneous agents.
    - __init__: construct per-type MLP heads and learnable log_std params
    - forward: compute mu (and optionally return log_std params)
    """
    def __init__(self,
                 in_dim: int,
                 act_dim_vehicle: int,
                 act_dim_ped: int,
                 hidden: Optional[list] = None,
                 init_log_std: float = -0.5) -> None:
        super().__init__()
        pass

    def forward(self,
                feats: torch.Tensor,
                types: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            feats: [T?, B, N, F] or [B, N, F]
            types: [T?, B, N] or [B, N] (int tensor identifying agent type)
        Returns:
            dict containing keys like:
              - 'mu_vehicle', 'mu_ped'
              - 'log_std_vehicle', 'log_std_ped' (learnable params)
        """
        pass


class CriticPerAgent(nn.Module):
    """
    Per-agent critic (DeepSets-style).
    - __init__: build MLP that maps per-agent (and optionally pooled context) to scalar value
    - forward: return V per agent, masked if mask provided
    """
    def __init__(self,
                 in_dim: int,
                 hidden: Optional[list] = None) -> None:
        super().__init__()
        pass

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [T?, B, N, D] or [B, N, D]
            mask: optional mask with same T/B/N shape to zero-out invalid slots
        Returns:
            V_per_agent: [T?, B, N] or [B, N]
        """
        pass

