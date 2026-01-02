"""
Multi-agent, vectorized RolloutBuffer for MAPPO-style on-policy training.

Features:
 - store time-major buffers [T, B, N, ...]
 - add_batch accepts numpy or torch (auto-converts)
 - finish_paths computes GAE per (env,agent) and normalizes advantages
 - feed_forward_generator flattens T*B*N and yields minibatches as torch tensors on device
 - clear() resets for next rollout

Usage:
    buf = RolloutBuffer(T=128, num_envs=8, n_agents=6,
                        image_shape=(3,64,64), agent_feat_dim=16, act_dim=4,
                        gamma=0.99, gae_lambda=0.95, device='cpu')
    buf.add_batch(imgs, agent_feats, type_ids, actions, logp, values, rewards, masks)
    ...
    buf.finish_paths(last_values)   # last_values shape [B,N] or [B,N,1]
    for mb in buf.feed_forward_generator(mini_batch_size):
        # mb is a dict of torch tensors on buf.device
    buf.clear()
"""

from typing import Optional, Dict, Iterator, Tuple, Any
import numpy as np
import torch


def _to_numpy(x):
    """Convert torch tensor or numpy array to numpy array on CPU."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    # fallback: try to make array
    return np.asarray(x)


class RolloutBuffer:
    def __init__(self,
                 T: int,
                 num_envs: int,
                 n_agents: int,
                 image_shape: Optional[Tuple[int, int, int]] = None,
                 agent_feat_dim: int = 0,
                 act_dim: int = 1,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 device: str = "cpu"):
        self.T = int(T)
        self.B = int(num_envs)
        self.N = int(n_agents)
        self.device = torch.device(device)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)

        # images per env (BEV is per-env per-timestep, not per-agent)
        if image_shape is not None:
            C, H, W = image_shape
            self.images = np.zeros((self.T, self.B, C, H, W), dtype=np.float32)
        else:
            self.images = None

        # per-agent numeric features [T, B, N, feat_dim]
        self.agent_feats = np.zeros((self.T, self.B, self.N, int(agent_feat_dim)), dtype=np.float32)

        # per-agent type ids [T, B, N] (int64)
        self.type_id = np.zeros((self.T, self.B, self.N), dtype=np.int64)

        # actions/logp/values/rewards/masks: per-agent
        self.actions = np.zeros((self.T, self.B, self.N, int(act_dim)), dtype=np.float32)
        self.logp = np.zeros((self.T, self.B, self.N), dtype=np.float32)
        # store scalar value per agent (shape [T,B,N])
        self.values = np.zeros((self.T, self.B, self.N), dtype=np.float32)
        self.rewards = np.zeros((self.T, self.B, self.N), dtype=np.float32)
        # mask: 1.0 = alive, 0.0 = just-died (prevents hidden leakage)
        self.masks = np.ones((self.T, self.B, self.N), dtype=np.float32)

        # computed after finish_paths
        self.advantages = np.zeros((self.T, self.B, self.N), dtype=np.float32)
        self.returns = np.zeros((self.T, self.B, self.N), dtype=np.float32)

        # pointer
        self.ptr = 0
        self.full = False

    def add_batch(self,
                  imgs,
                  agent_feats,
                  type_ids,
                  actions,
                  logp,
                  values,
                  rewards,
                  masks):
        """
        Add one timestep's whole-batch data. Accepts numpy arrays or torch tensors.
        Shapes (expected):
          imgs: [B, C, H, W] or None if buffer was created without images
          agent_feats: [B, N, feat_dim]
          type_ids: [B, N]
          actions: [B, N, A]
          logp: [B, N]
          values: [B, N] or [B, N, 1]
          rewards: [B, N]
          masks: [B, N]  (1=alive, 0=just-died)
        """
        t = self.ptr

        # convert to numpy if torch provided
        imgs_np = _to_numpy(imgs)
        feats_np = _to_numpy(agent_feats)
        types_np = _to_numpy(type_ids)
        actions_np = _to_numpy(actions)
        logp_np = _to_numpy(logp)
        vals_np = _to_numpy(values)
        rews_np = _to_numpy(rewards)
        masks_np = _to_numpy(masks)

        # basic checks
        if feats_np.shape[0] != self.B:
            raise ValueError(f"agent_feats batch-size mismatch: expected B={self.B}, got {feats_np.shape[0]}")

        if self.images is not None:
            if imgs_np is None:
                raise ValueError("images buffer expected but imgs is None")
            self.images[t] = imgs_np.astype(np.float32)

        self.agent_feats[t] = feats_np.astype(np.float32)
        self.type_id[t] = types_np.astype(np.int64)
        self.actions[t] = actions_np.astype(np.float32)
        self.logp[t] = logp_np.astype(np.float32)

        # values: allow [B,N,1] or [B,N]
        vals_np = np.asarray(vals_np, dtype=np.float32)
        if vals_np.ndim == 3 and vals_np.shape[-1] == 1:
            vals_np = vals_np.reshape(vals_np.shape[0], vals_np.shape[1])
        self.values[t] = vals_np

        self.rewards[t] = rews_np.astype(np.float32)
        self.masks[t] = masks_np.astype(np.float32)

        # advance pointer
        self.ptr += 1
        if self.ptr >= self.T:
            self.full = True
            self.ptr = 0

    def finish_paths(self, last_values):
        """
        Compute GAE advantages and returns for stored T steps.
        last_values: array-like shape [B,N] or [B,N,1] (value estimates for next-state)
        After this call, self.advantages and self.returns are filled (numpy arrays).
        """
        last_val = _to_numpy(last_values).astype(np.float32)
        if last_val.ndim == 3 and last_val.shape[-1] == 1:
            last_val = last_val.reshape(last_val.shape[0], last_val.shape[1])
        if last_val.shape != (self.B, self.N):
            raise ValueError(f"last_values must have shape [B,N], got {last_val.shape}")

        adv = np.zeros((self.B, self.N), dtype=np.float32)
        self.advantages = np.zeros((self.T, self.B, self.N), dtype=np.float32)
        self.returns = np.zeros((self.T, self.B, self.N), dtype=np.float32)

        next_values = last_val.copy()  # [B,N]
        for t in reversed(range(self.T)):
            mask_t = self.masks[t]  # [B,N]
            rewards_t = self.rewards[t]  # [B,N]
            values_t = self.values[t]  # [B,N]
            delta = rewards_t + self.gamma * next_values * mask_t - values_t
            adv = delta + self.gamma * self.gae_lambda * adv * mask_t
            self.advantages[t] = adv
            next_values = values_t

        # returns = advantages + values
        self.returns = self.advantages + self.values

        # normalize advantages (global)
        flat_adv = self.advantages.reshape(-1)
        if flat_adv.size > 0:
            adv_mean = flat_adv.mean()
            adv_std = flat_adv.std()
            self.advantages = (self.advantages - adv_mean) / (adv_std + 1e-8)

    def feed_forward_generator(self, mini_batch_size: int) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Yield minibatches by flattening T,B,N -> total = T*B*N.
        Each yielded dict contains torch tensors on self.device:
            - images (if present): [mb, C, H, W] or None
            - agent_feats: [mb, feat_dim]
            - type_id: [mb] (long)
            - actions: [mb, A]
            - old_logp: [mb]
            - returns: [mb, 1]
            - advantages: [mb]
            - values: [mb, 1]
            - indices: the flat indices into T*B*N (optional)
        """
        T, B, N = self.T, self.B, self.N
        total = T * B * N

        # flatten per-agent arrays for efficient indexing
        flat_agent_feats = self.agent_feats.reshape(total, -1)     # [total, feat_dim]
        flat_type = self.type_id.reshape(total)
        flat_actions = self.actions.reshape(total, -1)
        flat_logp = self.logp.reshape(total)
        flat_returns = self.returns.reshape(total)
        flat_adv = self.advantages.reshape(total)
        flat_values = self.values.reshape(total)

        # prepare indices and shuffle
        indices = np.arange(total)
        np.random.shuffle(indices)

        for start in range(0, total, mini_batch_size):
            batch_idx = indices[start:start + mini_batch_size]
            # map flat idx -> (t,b,n)
            t_idx = (batch_idx // (B * N)).astype(np.int64)
            rem = batch_idx % (B * N)
            b_idx = (rem // N).astype(np.int64)
            # n_idx = rem % N  # unused here except if needed

            feats_mb = flat_agent_feats[batch_idx]   # [mb, feat_dim]
            types_mb = flat_type[batch_idx]         # [mb]
            actions_mb = flat_actions[batch_idx]    # [mb, A]
            old_logp_mb = flat_logp[batch_idx]      # [mb]
            returns_mb = flat_returns[batch_idx].reshape(-1, 1)  # [mb,1]
            adv_mb = flat_adv[batch_idx]            # [mb]
            values_mb = flat_values[batch_idx].reshape(-1, 1)   # [mb,1]

            # images: gather by (t,b) then produce tensor [mb, C, H, W]
            if self.images is not None:
                imgs_batch = self.images[t_idx, b_idx]  # [mb, C, H, W]
                images_tensor = torch.tensor(imgs_batch, device=self.device, dtype=torch.float32)
            else:
                images_tensor = None

            yield {
                "images": images_tensor,
                "agent_feats": torch.tensor(feats_mb, device=self.device, dtype=torch.float32),
                "type_id": torch.tensor(types_mb, device=self.device, dtype=torch.long),
                "actions": torch.tensor(actions_mb, device=self.device, dtype=torch.float32),
                "old_logp": torch.tensor(old_logp_mb, device=self.device, dtype=torch.float32),
                "returns": torch.tensor(returns_mb, device=self.device, dtype=torch.float32),
                "advantages": torch.tensor(adv_mb, device=self.device, dtype=torch.float32),
                "values": torch.tensor(values_mb, device=self.device, dtype=torch.float32),
                "indices": batch_idx
            }

    def clear(self):
        """Reset pointer/flags and zero buffers (safety)."""
        self.ptr = 0
        self.full = False
        if self.images is not None:
            self.images.fill(0)
        self.agent_feats.fill(0)
        self.type_id.fill(0)
        self.actions.fill(0)
        self.logp.fill(0)
        self.values.fill(0)
        self.rewards.fill(0)
        self.masks.fill(1.0)
        self.advantages.fill(0)
        self.returns.fill(0)

    # helpers
    def remaining(self) -> int:
        return self.T - self.ptr

    def summary(self) -> Dict[str, Any]:
        return {
            "T": self.T, "B": self.B, "N": self.N,
            "ptr": self.ptr, "full": self.full,
            "device": str(self.device),
            "has_images": self.images is not None
        }
