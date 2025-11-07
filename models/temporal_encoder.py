
# models/temporal_encoder.py
import torch
import torch.nn as nn

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
