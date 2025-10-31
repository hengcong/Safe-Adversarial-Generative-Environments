import numpy as np
import torch
class RolloutBuffer:
    """
    Simple on-policy rollout buffer with GAE support.

    Usage:
      buf = RolloutBuffer(obs_shape, act_dim, size, gamma=0.99, gae_lambda=0.95, device='cpu')
      for each step:
          buf.add(obs, act, rew, done, val, logp)
      when rollout finished or truncated:
          buf.finish_path(last_val)   # last_val is critic estimate of next-state value (0 if terminal)
      data = buf.get()               # returns dict of tensors on device
      buf.clear()

    Notes:
    - obs_shape: shape tuple of single observation (e.g., (128,) )
    - act_dim: dimension of continuous action vector (int)
    - size: capacity (rollout length)
    - This class supports variable filled length; get() will return only collected entries.
    """
    def __init__(self, obs_shape, act_dim, size, gamma=0.99, gae_lambda=0.95, device='cpu'):
        self.size = int(size)
        self.device = torch.device(device)

        # buffer
        self.obs_buf = np.zeros((self.size))

