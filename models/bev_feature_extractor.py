import torch
import torch.nn as nn
import torch.nn.functional as F

class BEVFeatureExtractor(nn.Module):
    """
    BEV -> feature vector extractor compatible with MAPPOPolicy usage.

    Constructor:
      BEVFeatureExtractor(in_channels, use_gru=False, device="cpu",
                          features_dim=256, group_size=8)

    Methods:
      forward(observations): expects image tensor [B,C,H,W] -> returns [B,features_dim]
      step(img, bev_h=None): step API, returns (bev_embed, next_bev_h) where next_bev_h is None if use_gru=False
    """
    def __init__(self, in_channels: int = 3, use_gru: bool = False, device: str = "cpu",
                 features_dim: int = 256, group_size: int = 8):
        super().__init__()
        self.device = torch.device(device)
        self.use_gru = bool(use_gru)
        self.bev_feat_dim = int(features_dim)
        self.group_size = int(group_size)

        def make_gn(ch):
            return nn.GroupNorm(max(1, ch // self.group_size), ch)

        # Simple 5-layer CNN (same design as earlier, tuned to output flattened dim then fc -> features_dim)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            make_gn(32), nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            make_gn(64), nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            make_gn(128), nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            make_gn(256), nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            make_gn(256), nn.ReLU()
        )

        # we'll lazily infer flattened conv output size on first forward if needed
        self._fc = None
        self._n_flatten = None

        # optional small GRU to maintain scene-level state across steps
        if self.use_gru:
            # GRU hidden dim chosen equal to features_dim for simplicity
            self.bev_gru = nn.GRUCell(self.bev_feat_dim, self.bev_feat_dim)
        else:
            self.bev_gru = None

        # move to device
        self.to(self.device)

        # orthogonal init for conv/linear
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                try:
                    nn.init.orthogonal_(m.weight)
                except Exception:
                    pass
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0.0)

    def _forward_cnn(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = torch.flatten(x, start_dim=1)
        return x

    def _build_fc_if_needed(self, sample_img: torch.Tensor):
        """Infer flatten dim and build fc projection if not created yet."""
        if self._fc is not None:
            return
        with torch.no_grad():
            out = self._forward_cnn(sample_img.to(self.device))
            n_flatten = out.shape[1]
            self._n_flatten = n_flatten
            self._fc = nn.Sequential(
                nn.Linear(n_flatten, self.bev_feat_dim),
                nn.LayerNorm(self.bev_feat_dim),
                nn.ReLU()
            ).to(self.device)
            # orthogonal init for fc
            for m in self._fc.modules():
                if isinstance(m, (nn.Linear,)):
                    try:
                        nn.init.orthogonal_(m.weight)
                    except Exception:
                        pass
                    if getattr(m, "bias", None) is not None:
                        nn.init.constant_(m.bias, 0.0)

    def forward(self, observations):
        """
        observations: either an image tensor [B,C,H,W] or a dict {'image': tensor, 'speed': tensor (optional)}
        returns: bev_embed [B, features_dim]
        """
        if isinstance(observations, dict):
            img = observations.get("image", None)
        else:
            img = observations

        if img is None:
            raise ValueError("BEVFeatureExtractor.forward expects an image tensor or observations dict with key 'image'")

        # normalize to [B,C,H,W]
        if img.dim() == 4 and img.shape[1] not in (1, 3, 4) and img.shape[-1] in (1, 3, 4):
            img = img.permute(0, 3, 1, 2).contiguous()

        # build fc lazily if needed (infer shape from img)
        if self._fc is None:
            sample = img[:1].to(self.device)
            self._build_fc_if_needed(sample)

        img = img.to(self.device, dtype=torch.float32)
        cnn_out = self._forward_cnn(img)  # [B, n_flatten]
        feat = self._fc(cnn_out)  # [B, features_dim]

        # optionally append scalar speed if provided (observations dict)
        if isinstance(observations, dict):
            speed = observations.get("speed", None)
            if speed is not None:
                # ensure speed is [B,1]
                if speed.dim() == 1:
                    speed = speed.unsqueeze(1)
                elif speed.dim() == 2 and speed.shape[1] != 1:
                    speed = speed.reshape(speed.shape[0], -1)[:, :1]
                speed = speed.to(device=feat.device, dtype=feat.dtype)
            else:
                speed = torch.zeros(img.shape[0], 1, device=feat.device, dtype=feat.dtype)
            feat = torch.cat([feat, speed], dim=1) if self._fc is not None else feat

        return feat

    def step(self, img: torch.Tensor, bev_h: torch.Tensor = None):
        """
        Step API for collector. Accepts img [B,C,H,W] and optional bev hidden state.
        Returns (bev_embed [B,features_dim], next_bev_h or None).

        If use_gru==True:
          - bev_h expected shape: [B, features_dim] or None (will be zeros)
          - next_bev_h returned as [B, features_dim]
        """
        # ensure img on device
        img = img.to(self.device, dtype=torch.float32)

        # lazy fc build (if not built)
        if self._fc is None:
            sample = img[:1]
            self._build_fc_if_needed(sample)

        bev_embed = self.forward(img)  # [B, features_dim] (note: forward handles dict or tensor)
        next_bev_h = None

        if self.use_gru:
            B = bev_embed.shape[0]
            if bev_h is None:
                bev_h = torch.zeros(B, self.bev_feat_dim, device=self.device, dtype=bev_embed.dtype)
            else:
                bev_h = bev_h.to(self.device, dtype=bev_embed.dtype)
            # GRUCell expects (input, hidden) both [B, feat_dim]
            next_bev_h = self.bev_gru(bev_embed, bev_h)
            # optionally return a 1D vector per batch (we choose [B, feat_dim] for compatibility)
            return bev_embed, next_bev_h
        else:
            return bev_embed, None
