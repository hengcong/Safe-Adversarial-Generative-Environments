
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List

def make_mlp(in_dim: int, hidden: Optional[List[int]], out_dim: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    hs = hidden or []
    prev = in_dim
    for h in hs:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)

class ActorHeads(nn.Module):
    """
    Actor heads for heterogeneous agents (vehicle / ped).
    - Produces per-agent mu and per-type learnable log_std.
    - For convenience the returned 'mu' and 'log_std' are padded to max(act_dim_vehicle, act_dim_ped),
      so the policy can handle a uniform action tensor shape [T?, B, N, A_max].
    """
    def __init__(self,
                 in_dim: int,
                 act_dim_vehicle: int,
                 act_dim_ped: int,
                 hidden: Optional[list] = None,
                 init_log_std: float = -0.5) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.act_dim_vehicle = int(act_dim_vehicle)
        self.act_dim_ped = int(act_dim_ped)
        self.max_act_dim = max(self.act_dim_vehicle, self.act_dim_ped)

        # per-type MLP heads
        self.head_vehicle = make_mlp(in_dim, hidden, self.act_dim_vehicle)
        self.head_ped = make_mlp(in_dim, hidden, self.act_dim_ped)

        # learnable log_std params for each type (per-dimension)
        self.log_std_vehicle = nn.Parameter(torch.full((self.act_dim_vehicle,), float(init_log_std)))
        self.log_std_ped = nn.Parameter(torch.full((self.act_dim_ped,), float(init_log_std)))

        # register a small epsilon as buffer for numerical stability uses if needed
        self.register_buffer("_eps", torch.tensor(1e-8))

    def _flatten_inputs(self, feats):
        """
        Accepts feats in one of the following shapes:
          - 4D: [T, B, N, F]  (time-major)
          - 3D: [B, N, F]     (batch, slots, feat)
          - 2D: [S, F]        (flattened slots x feat)  <-- NEW: treat as [1, S, F]

        Returns:
          flat: [S_total, F]  (flattened for head)
          shape: tuple describing original layout, or None for 2D pass-through
        """
        # handle 2D input (flattened): treat as [1, S, F]
        if feats.dim() == 2:
            # feats is [S, F] (S = total slots). Make it [1, S, F] so downstream logic can treat it as batch 1.
            feats3 = feats.unsqueeze(0)  # [1, S, F]
            # reuse the 3D branch to produce flat and shape
            flat, shape = self._flatten_inputs(feats3)
            # shape will be (B, N) or similar — keep as-is
            return flat, shape

        # existing behavior: 3D or 4D
        if feats.dim() == 3:
            # feats: [B, N, F]
            B, N, F = feats.size()
            flat = feats.view(B * N, F)  # [B*N, F]
            return flat, (B, N)
        elif feats.dim() == 4:
            # feats: [T, B, N, F]
            T, B, N, F = feats.size()
            flat = feats.view(T * B * N, F)
            return flat, (T, B, N)
        else:
            raise ValueError(f"ActorHeads.forward expects 3D or 4D feats, got {feats.dim()}D")

    def _unflatten(self, flat, shape):
        """
        Inverse of _flatten_inputs.

        Accepts `shape` either as:
          - (B, N)       -> single-step batch format, returns [B, N, out_dim]
          - (T, B, N)    -> time-major sequence format, returns [T, B, N, out_dim]
        If shape is None, returns flat (no-op).
        """
        if shape is None:
            return flat

        # allow shape to be tuple/list or torch tensor
        if isinstance(shape, (list, tuple)):
            dims = len(shape)
        else:
            try:
                dims = len(tuple(shape))
            except Exception:
                raise ValueError(f"_unflatten: unexpected shape type {type(shape)}")

        if dims == 2:
            # shape = (B, N)
            B, N = shape
            # flat is [B*N, out_dim] -> reshape to [B, N, out_dim]
            out_dim = flat.size(-1)
            return flat.view(B, N, out_dim)
        elif dims == 3:
            # shape = (T, B, N)
            T, B, N = shape
            out_dim = flat.size(-1)
            return flat.view(T, B, N, out_dim)
        else:
            raise ValueError(f"_unflatten expects shape of length 2 or 3, got {shape}")

    def forward(self,
                feats: torch.Tensor,
                types: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Returns dict with keys:
          - 'mu': [T?, B, N, A_max] (padded)
          - 'log_std': [T?, B, N, A_max] (padded)
          - 'mu_vehicle', 'mu_ped' (un-padded, their native dims)
          - 'log_std_vehicle', 'log_std_ped' (native dims)
        If `types` is provided, callers can index into per-agent types to pick correct subset.
        """
        flat, shape = self._flatten_inputs(feats)  # flat: [S, F], shape = (T?,B,N)
        device = flat.device
        dtype = flat.dtype

        # compute per-type outputs for all slots (we'll select later)
        mu_vehicle_flat = self.head_vehicle(flat)  # [S, act_dim_vehicle]
        mu_ped_flat = self.head_ped(flat)          # [S, act_dim_ped]

        # expand log_std to flat shape
        ls_vehicle_flat = self.log_std_vehicle.unsqueeze(0).expand(mu_vehicle_flat.shape[0], -1)
        ls_ped_flat = self.log_std_ped.unsqueeze(0).expand(mu_ped_flat.shape[0], -1)

        # unflatten per-type mus back to [T?,B,N,dim] or [B,N,dim]
        mu_vehicle = self._unflatten(mu_vehicle_flat, shape)
        mu_ped = self._unflatten(mu_ped_flat, shape)
        log_std_vehicle = self._unflatten(ls_vehicle_flat, shape)
        log_std_ped = self._unflatten(ls_ped_flat, shape)

        mu_vehicle_p = self.pad_to_max(mu_vehicle, self.act_dim_vehicle)
        mu_ped_p = self.pad_to_max(mu_ped, self.act_dim_ped)
        log_std_vehicle_p = self.pad_to_max(log_std_vehicle, self.act_dim_vehicle)
        log_std_ped_p = self.pad_to_max(log_std_ped, self.act_dim_ped)

        # final unified mu/log_std: choose per-agent based on types if provided, else return both and the padded versions
        # For convenience we produce a default 'mu' which picks per-agent values when types given,
        # otherwise returns mu_vehicle_p (arbitrary) but callers can use per-type keys.
        if types is None:
            # return padded versions (caller can interpret)
            unified_mu = mu_vehicle_p
            unified_log_std = log_std_vehicle_p
        else:
            # types: [T?,B,N] or [B,N]
            # flatten types to same leading dimension as flat to index
            if types.dim() == 3:
                T, B, N = types.shape
                types_flat = types.view(T * B * N)
            elif types.dim() == 2:
                B, N = types.shape
                types_flat = types.view(B * N)
            else:
                raise ValueError("types must be 2D [B,N] or 3D [T,B,N]")
            # build unified flat by selecting columns from padded per-type flats
            # unflatten padded to flat shape first
            mu_vehicle_flat_p = mu_vehicle_flat.new_zeros((mu_vehicle_flat.shape[0], self.max_act_dim))
            mu_vehicle_flat_p[:, :self.act_dim_vehicle] = mu_vehicle_flat
            mu_ped_flat_p = mu_ped_flat.new_zeros((mu_ped_flat.shape[0], self.max_act_dim))
            mu_ped_flat_p[:, :self.act_dim_ped] = mu_ped_flat

            log_vehicle_flat_p = log_std_vehicle.new_zeros((mu_vehicle_flat.shape[0], self.max_act_dim))
            log_vehicle_flat_p[:, :self.act_dim_vehicle] = ls_vehicle_flat
            log_ped_flat_p = log_std_ped.new_zeros((mu_ped_flat.shape[0], self.max_act_dim))
            log_ped_flat_p[:, :self.act_dim_ped] = ls_ped_flat

            # choose per-slot
            select_mask = (types_flat == 0).unsqueeze(-1)  # assume 0 -> vehicle, 1 -> ped
            mu_flat_p = torch.where(select_mask, mu_vehicle_flat_p, mu_ped_flat_p)
            log_flat_p = torch.where(select_mask, log_vehicle_flat_p, log_ped_flat_p)

            # unflatten back to [T?,B,N,A_max] or [B,N,A_max]
            unified_mu = self._unflatten(mu_flat_p, shape)
            unified_log_std = self._unflatten(log_flat_p, shape)

        return {
            "mu": unified_mu,                       # [T?,B,N,A_max]
            "log_std": unified_log_std,             # [T?,B,N,A_max]
            "mu_vehicle": mu_vehicle,               # native dim
            "mu_ped": mu_ped,
            "log_std_vehicle": log_std_vehicle,
            "log_std_ped": log_std_ped,
        }

    def pad_to_max(self, tensor, native_dim):
        # create pad tensor on same device/dtype as input
        device = tensor.device
        dtype = tensor.dtype
        pad = self.max_act_dim - native_dim
        if pad == 0:
            return tensor
        pad_shape = list(tensor.shape[:-1]) + [pad]
        pad_tensor = torch.zeros(pad_shape, device=device, dtype=dtype)
        return torch.cat([tensor, pad_tensor], dim=-1)


class CriticPerAgent(nn.Module):
    """
    Per-agent critic MLP. Maps per-agent features -> scalar V.
    Returns shape-matching masked output.
    """
    def __init__(self,
                 in_dim: int,
                 hidden: Optional[list] = None) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.net = make_mlp(in_dim, hidden, 1)

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: accepted input shapes:
           - 4D: [T, B, N, D]
           - 3D: [B, N, D]
           - 2D: [S, D]   (flattened per-slot input, S == B*N)
        mask: same leading dims [T?,B,N] or [B,N] or None. For 2D input, mask may be:
           - None
           - 1D: [S] (per-slot mask)
           - 2D: [B, N] (will be flattened)
           - 3D/4D handled as before (if compatible)
        returns:
          - if input was 4D: returns [T, B, N]
          - if input was 3D: returns [B, N]
          - if input was 2D: returns [S] (i.e., [B*N]) or [S,] flattened vector
        Note: we try to preserve backward compatibility while allowing callers to pass flattened inputs.
        """
        flatten = False
        # Case A: time-major 4D [T, B, N, D]
        if x.dim() == 4:
            T, B, N, D = x.shape
            flat = x.contiguous().view(T * B * N, D)
            shape = (T, B, N)
            flatten = True
        # Case B: batch 3D [B, N, D]
        elif x.dim() == 3:
            B, N, D = x.shape
            flat = x.contiguous().view(B * N, D)
            shape = (None, B, N)
            flatten = False
        # Case C: flattened 2D [S, D] where S == B*N (caller provides flattened features)
        elif x.dim() == 2:
            S, D = x.shape
            flat = x.contiguous().view(S, D)  # [S, D]
            shape = ('flat', S)  # mark as flat
            flatten = False
        else:
            raise ValueError("CriticPerAgent expects input with 2, 3 or 4 dims")

        # evaluate MLP on flat features -> [S, 1]
        v_flat = self.net(flat)  # [S, 1] or similar
        # make a 1D vector for convenience
        v_flat_vec = v_flat.view(-1)  # [S]

        # If mask provided, apply it.
        if mask is not None:
            # Normalize mask to 1D per-slot mask when input was flat
            if x.dim() == 2:
                # Accept mask shapes: [S], [B,N], [1, S], or [S, 1]
                if torch.is_tensor(mask):
                    if mask.dim() == 1 and mask.size(0) == S:
                        m_vec = mask
                    elif mask.dim() == 2 and mask.size(0) == 1 and mask.size(1) == S:
                        m_vec = mask.view(-1)
                    elif mask.dim() == 2 and mask.numel() == S:
                        m_vec = mask.view(-1)
                    elif mask.dim() == 2 and mask.shape == (B, N):  # if caller passed B,N (but we don't know B,N here)
                        # best-effort flatten
                        m_vec = mask.contiguous().view(-1)
                    else:
                        raise ValueError("mask shape incompatible with flattened 2D input")
                else:
                    raise ValueError("mask must be a tensor when provided")
                # apply
                v_flat_vec = v_flat_vec * m_vec.to(v_flat_vec.device)
            else:
                # previous handling for 3D/4D masks
                if mask.dim() == 3 or mask.dim() == 2:
                    # For 4D case or 3D case we can broadcast multiply after unflattening
                    if x.dim() == 4:
                        # unflatten to [T, B, N]
                        v = v_flat_vec.view(T, B, N)
                        v = v * mask
                        return v
                    else:
                        # x.dim() == 3
                        v = v_flat_vec.view(B, N)
                        v = v * mask
                        return v
                else:
                    raise ValueError("mask must be 2D or 3D for non-flattened inputs")

        # If we reached here and the original input was flattened 2D, return flat vector shape [S]
        if x.dim() == 2:
            return v_flat_vec  # [S]

        # For non-flattened inputs, unflatten back
        if flatten:
            # 4D input -> return [T, B, N]
            v = v_flat_vec.view(T, B, N)
        else:
            # 3D input -> return [B, N]
            _, B, N = shape
            v = v_flat_vec.view(B, N)

        return v
