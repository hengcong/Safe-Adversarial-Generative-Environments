from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
import torch.nn.functional as F

class BEVFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512, group_size=8):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space["image"].shape[-1]

        def make_gn(ch):
            return nn.GroupNorm(max(1, ch // group_size), ch)

        # --- your 5-layer CNN ---
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, 7, stride=2, padding=3),
            make_gn(32), nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            make_gn(64), nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            make_gn(128), nn.ReLU())
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            make_gn(256), nn.ReLU())
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            make_gn(256), nn.ReLU())

        # test a dummy input to infer flatten dim
        with torch.no_grad():
            sample = torch.zeros(1, n_input_channels,
                                 *observation_space["image"].shape[:2])
            out = self._forward_cnn(sample)
            n_flatten = out.shape[1]

        self.fc = nn.Sequential(
            nn.Linear(n_flatten + 1, features_dim),  # +1 for speed
            nn.LayerNorm(features_dim),
            nn.ReLU()
        )

    def _forward_cnn(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = torch.flatten(x, start_dim=1)
        return x

    def forward(self, observations):
        img = observations["image"].permute(0, 3, 1, 2)
        cnn_out = self._forward_cnn(img)
        speed = observations["speed"]
        out = torch.cat((cnn_out, speed), dim=1)
        return self.fc(out)
