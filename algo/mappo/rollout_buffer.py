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

        # buffers (numpy for efficient writes)
        self.obs_buf = np.zeros((self.size,) + tuple(obs_shape), dtype=np.float32)
        self.act_buf = np.zeros((self.size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(self.size, dtype=np.float32)
        self.done_buf = np.zeros(self.size, dtype=np.float32)
        self.val_buf = np.zeros(self.size, dtype=np.float32)
        self.logp_buf = np.zeros(self.size, dtype=np.float32)

        # outputs computed on finish_path
        self.adv_buf = np.zeros(self.size, dtype=np.float32)
        self.ret_buf = np.zeros(self.size, dtype=np.float32)

        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)

        # pointer and length
        self.ptr = 0
        self.length = 0  # number of valid entries currently stored
        self.finished = False

    def add(self, obs, act, rew, done, val, logp):
        """
        Add a single step.
        obs: numpy array matching obs_shape
        act: numpy array shape (act_dim,) or scalar
        rew: float
        done: bool
        val: float (critic value for this step)
        logp: float (log prob of action)
        """
        assert self.ptr < self.size, "RolloutBuffer overflow: increase buffer size or clear earlier"
        self.obs_buf[self.ptr] = np.asarray(obs, dtype=np.float32)
        # ensure action shape matches
        self.act_buf[self.ptr] = np.asarray(act, dtype=np.float32)
        self.rew_buf[self.ptr] = float(rew)
        self.done_buf[self.ptr] = 1.0 if done else 0.0
        self.val_buf[self.ptr] = float(val)
        self.logp_buf[self.ptr] = float(logp)

        self.ptr += 1
        self.length = max(self.length, self.ptr)

    def finish_path(self, last_val=0.0):
        """
        Call at the end of a trajectory segment (or at rollout end).
        last_val: float, value estimate for the state following the last stored step
                  (0.0 if terminal)
        Computes GAE advantages and returns for currently filled prefix [0 : self.length].
        """
        L = self.length
        if L == 0:
            return

        # compute GAE
        adv = 0.0
        vals = np.append(self.val_buf[:L], float(last_val))
        rews = self.rew_buf[:L]
        dones = self.done_buf[:L]
        for t in reversed(range(L)):
            mask = 1.0 - dones[t]
            delta = rews[t] + self.gamma * vals[t+1] * mask - vals[t]
            adv = delta + self.gamma * self.gae_lambda * adv * mask
            self.adv_buf[t] = adv

        # returns = advantages + values
        self.ret_buf[:L] = self.adv_buf[:L] + self.val_buf[:L]
        self.finished = True

    def get(self, to_torch=True):
        """
        Return collected data up to current length as tensors (or numpy if to_torch=False).
        Keys: 'obs','acts','advs','rets','logps','vals'
        All tensors placed on self.device.
        """
        L = self.length
        assert L > 0, "No data in buffer"

        if to_torch:
            obs = torch.tensor(self.obs_buf[:L], dtype=torch.float32, device=self.device)
            acts = torch.tensor(self.act_buf[:L], dtype=torch.float32, device=self.device)
            advs = torch.tensor(self.adv_buf[:L], dtype=torch.float32, device=self.device)
            rets = torch.tensor(self.ret_buf[:L], dtype=torch.float32, device=self.device)
            logps = torch.tensor(self.logp_buf[:L], dtype=torch.float32, device=self.device)
            vals = torch.tensor(self.val_buf[:L], dtype=torch.float32, device=self.device)
            # normalize advs (common practice)
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            return {'obs': obs, 'acts': acts, 'advs': advs, 'rets': rets, 'logps': logps, 'vals': vals}
        else:
            # return numpy arrays
            advs = self.adv_buf[:L].copy()
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            return {
                'obs': self.obs_buf[:L].copy(),
                'acts': self.act_buf[:L].copy(),
                'advs': advs,
                'rets': self.ret_buf[:L].copy(),
                'logps': self.logp_buf[:L].copy(),
                'vals': self.val_buf[:L].copy()
            }

    def clear(self):
        """Reset pointers and flags; keep allocated arrays for speed."""
        self.ptr = 0
        self.length = 0
        self.finished = False
        # zeroing arrays is optional; pointers/length control valid region
        # but clear numerical buffers for safety:
        self.obs_buf.fill(0)
        self.act_buf.fill(0)
        self.rew_buf.fill(0)
        self.done_buf.fill(0)
        self.val_buf.fill(0)
        self.logp_buf.fill(0)
        self.adv_buf.fill(0)
        self.ret_buf.fill(0)

