
# models/temporal_encoder.py
import torch
import torch.nn as nn

def orthogonal_init(module):
    """
    Recursively apply orthogonal init to Linear/Conv/GRUCell modules and zero biases.
    """
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            try:
                nn.init.orthogonal_(m.weight)
            except Exception:
                pass
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        # GRUCell: init weight_ih / weight_hh orthogonally, zero biases
        if isinstance(m, nn.GRUCell):
            try:
                for name, p in m.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(p)
                    elif "bias" in name:
                        nn.init.constant_(p, 0.0)
            except Exception:
                pass


class SlotGRU(nn.Module):
    """
    Per-slot (per-agent) GRU implemented with GRUCell for easy reset control.
    Input:
      x: [T, B, N, F]
      hidden: optional [B, N, H] or flattened [B*N, H]
      dones: optional [T, B, N] or [B, N] with 1 meaning reset at that timestep
    Output:
      out: [T, B, N, H]
      final_hidden: [1, B, N, H] (shape chosen for compatibility)
    Notes:
      - Internal hidden is flattened to [B*N, H] for GRUCell efficiency.
      - We detach hidden only when caller wants to (trainer typically detaches).
    """
    def __init__(self, input_size: int, hidden_size: int = 256):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.slot_hidden_dim = self.hidden_size
        self.output_dim = self.hidden_size
        self.gru_cell = nn.GRUCell(self.input_size, self.hidden_size)
        orthogonal_init(self)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor = None, dones: torch.Tensor = None):
        """
        x: [T, B, N, F]
        hidden: None or [B, N, H] or [1, B, N, H] (we accept both)
        dones: None or [T, B, N] (1 indicates reset at that step) or [B, N]
        Returns:
          out: [T, B, N, H]
          final_hidden: [1, B, N, H]
        """
        T, B, N, F = x.shape
        device = x.device

        # prepare hidden: flatten to [B*N, H]
        if hidden is None:
            h = torch.zeros(B * N, self.hidden_size, device=device)
        else:
            # accept [B,N,H] or [1,B,N,H]
            if hidden.dim() == 4:
                h = hidden.view(hidden.size(1) * hidden.size(2), hidden.size(3)).to(device)
            else:
                h = hidden.view(B * N, -1).to(device)

        outputs = []
        for t in range(T):
            xt = x[t].view(B * N, F)  # [B*N, F]
            if dones is not None:
                # support dones shape [T,B,N] or [B,N]
                if dones.dim() == 3:
                    done_mask = dones[t].view(B * N, 1).to(device)  # 1 means done -> reset
                else:
                    done_mask = dones.view(B * N, 1).to(device)
                h = h * (1.0 - done_mask)
            h = self.gru_cell(xt, h)  # [B*N, H]
            outputs.append(h.view(B, N, self.hidden_size).unsqueeze(0))  # [1,B,N,H]

        out_seq = torch.cat(outputs, dim=0)  # [T, B, N, H]
        final_h = out_seq[-1].unsqueeze(0)   # [1, B, N, H]
        return out_seq, final_h

    def step(self, x: torch.Tensor, hidden: torch.Tensor = None, dones: torch.Tensor = None):
        """
        Step API (single timestep or batch of timesteps) for collector.
        Inputs:
          x: [B, N, F] (single step) or [B*N, F] (flattened)
          hidden: optional [B, N, H] or [1, B, N, H] or flattened [B*N, H]
          dones: optional [B, N] (1 means done -> reset). If provided and has dtype bool/int/float, it will be used.
        Returns:
          out: [B, N, H]  (the new slot embedding after one step)
          next_hidden: [1, B, N, H] (final hidden)
        Notes:
          - This is compatible with policy._call_slot_gru_step which expects a step-style API.
        """
        # normalize shapes
        device = x.device
        B = None
        # accept x as [B,N,F] or [B*N, F]
        if x.dim() == 3:
            B, N, F = x.shape
            flat_x = x.view(B * N, F)
        elif x.dim() == 2:
            flat_x = x
            B = None  # need hidden to infer B,N
        else:
            raise ValueError(f"SlotGRU.step expects x of dim 2 or 3, got {x.dim()}")

        # prepare hidden flattened
        if hidden is None:
            if B is None:
                raise ValueError("hidden must be provided when x is flattened")
            h = torch.zeros(B * N, self.hidden_size, device=device)
        else:
            if hidden.dim() == 4:
                h = hidden.view(hidden.size(1) * hidden.size(2), hidden.size(3)).to(device)
            else:
                h = hidden.view(-1, hidden.size(-1)).to(device)

        # apply dones mask if provided (dones: [B,N])
        if dones is not None:
            if dones.dim() == 2:
                done_mask = dones.view(-1, 1).to(device).float()
            else:
                # accept bool/int tensors, try to reduce to [B,N]
                done_mask = dones.reshape(-1, 1).to(device).float()
            h = h * (1.0 - done_mask)

        # one GRUCell step
        h = self.gru_cell(flat_x, h)  # [B*N, H]

        # reshape back to [B,N,H]
        if B is None:
            # try to infer B from hidden if possible
            try:
                B = hidden.size(0)
                N = hidden.size(1)
            except Exception:
                raise RuntimeError(
                    "Could not infer B,N from inputs; please pass x as [B,N,F] or provide hidden with shape [B,N,H]")
        out = h.view(B, N, self.hidden_size)
        final_h = out.unsqueeze(0)  # [1,B,N,H]
        return out, final_h

