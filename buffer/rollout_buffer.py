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

        if image_shape is not None:
            C, H, W = image_shape
            self.images = np.zeros((self.T, self.B, C, H, W), dtype=np.float32)
        else:
            self.images = None

        self.agent_feats = np.zeros((self.T, self.B, self.N, int(agent_feat_dim)), dtype=np.float32)
        self.type_id = np.zeros((self.T, self.B, self.N), dtype=np.int64)

        self.actions = np.zeros((self.T, self.B, self.N, int(act_dim)), dtype=np.float32)
        self.logp = np.zeros((self.T, self.B, self.N), dtype=np.float32)
        self.values = np.zeros((self.T, self.B, self.N), dtype=np.float32)
        self.rewards = np.zeros((self.T, self.B, self.N), dtype=np.float32)
        self.masks = np.ones((self.T, self.B, self.N), dtype=np.float32)

        self.pre_t = np.zeros((self.T, self.B, self.N, int(act_dim)), dtype=np.float32)
        self.action_scale = 5.0
        self.slot_hidden = None
        self.bev_hidden = None

        self.advantages = np.zeros((self.T, self.B, self.N), dtype=np.float32)
        self.returns = np.zeros((self.T, self.B, self.N), dtype=np.float32)

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
                  masks,
                  pre_t_np=None,
                  slot_hidden=None,
                  bev_hidden=None):

        if pre_t_np is None:
            raise RuntimeError("pre_t_np must be provided")

        t = self.ptr

        imgs_np = _to_numpy(imgs)
        feats_np = _to_numpy(agent_feats)
        types_np = _to_numpy(type_ids)
        actions_np = _to_numpy(actions)
        logp_np = _to_numpy(logp)
        vals_np = _to_numpy(values)
        rews_np = _to_numpy(rewards)
        masks_np = _to_numpy(masks)
        pre_t_np = _to_numpy(pre_t_np)

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

        vals_np = np.asarray(vals_np, dtype=np.float32)
        if vals_np.ndim == 3 and vals_np.shape[-1] == 1:
            vals_np = vals_np.reshape(vals_np.shape[0], vals_np.shape[1])
        self.values[t] = vals_np

        self.rewards[t] = rews_np.astype(np.float32)
        self.masks[t] = masks_np.astype(np.float32)

        if pre_t_np.ndim == 2:
            pre_t_np = pre_t_np[None, :, :]
        self.pre_t[t] = pre_t_np.astype(np.float32)
        if slot_hidden is not None:
            slot_np = _to_numpy(slot_hidden)
            if slot_np.ndim == 3 and slot_np.shape[0] == 1:
                slot_np = slot_np.squeeze(0)
            if slot_np.shape[0] == self.B * self.N:
                slot_np = slot_np.reshape(self.B, self.N, -1)
            if self.slot_hidden is None:
                h_dim = slot_np.shape[-1]
                self.slot_hidden = np.zeros((self.T, self.B, self.N, h_dim), dtype=np.float32)
            self.slot_hidden[t] = slot_np.astype(np.float32)

        if bev_hidden is not None:
            bev_np = _to_numpy(bev_hidden)
            if bev_np.ndim == 3: bev_np = bev_np.squeeze(0)
            if self.bev_hidden is None:
                h_dim = bev_np.shape[-1]
                self.bev_hidden = np.zeros((self.T, self.B, h_dim), dtype=np.float32)
            self.bev_hidden[t] = bev_np.astype(np.float32)
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
            #print("[DEBUG finish_paths rewards_t mean]", rewards_t.mean())

            adv = delta + self.gamma * self.gae_lambda * adv * mask_t
            self.advantages[t] = adv
            next_values = values_t#* mask_t

        # returns = advantages + values
        self.returns = self.advantages + self.values

        # -------- mask-aware advantage normalization --------
        valid_mask = self.masks > 0  # shape [T,B,N]

        valid_adv = self.advantages[valid_mask]

        if valid_adv.size > 0:
            adv_mean = valid_adv.mean()
            adv_std = valid_adv.std() + 1e-8

            self.advantages = (self.advantages - adv_mean) / adv_std
            self.advantages *= self.masks

    def feed_forward_generator(self, num_mini_batches=4):
        # Buffer is Time-Major: [T, B, N, ...]
        # We slice along the Batch dimension (B) to preserve sequences for RNN.
        # We also slice the Time dimension (T) to valid_limit to discard empty padding.

        total_envs = self.B

        # Calculate mini-batch size for the batch dimension
        mini_batch_size = max(1, total_envs // num_mini_batches)

        # Shuffle environment indices
        indices = np.arange(total_envs)
        np.random.shuffle(indices)

        # Determine valid length: use self.ptr if not full (early break), else self.T
        valid_limit = self.T if self.full else self.ptr

        # Guard against empty buffer
        if valid_limit == 0:
            return

        for start in range(0, total_envs, mini_batch_size):
            end = start + mini_batch_size
            mb_indices = indices[start:end]

            # Helper to slice data: [T, B, N, ...] -> [valid_limit, mb_size, N, ...]
            def prepare(data):
                if data is None: return None
                # Slice time to valid_limit (0 to valid_limit)
                # Slice batch to mb_indices
                return torch.tensor(data[:valid_limit, mb_indices], device=self.device, dtype=torch.float32)

            # For RNN, we need the INITIAL hidden state at t=0 for the selected envs.
            # self.init_slot_hidden should be shape [B, N, H]
            init_slot = None
            if self.init_slot_hidden is not None:
                # Slice axis 0 because init_hidden is [B, N, H] corresponding to envs
                # We do NOT slice time here because this is the state at t=0
                init_slot = torch.tensor(self.init_slot_hidden[mb_indices],
                                         device=self.device, dtype=torch.float32)

            init_bev = None
            if self.init_bev_hidden is not None:
                init_bev = torch.tensor(self.init_bev_hidden[mb_indices],
                                        device=self.device, dtype=torch.float32)

            yield {
                "images": prepare(self.images),
                "agent_feats": prepare(self.agent_feats),
                "type_id": prepare(self.type_id),
                "actions": prepare(self.actions),
                "logp": prepare(self.logp),
                "returns": prepare(self.returns),
                "advantages": prepare(self.advantages),
                "values": prepare(self.values),
                "masks": prepare(self.masks),
                "pre_t": prepare(self.pre_t),

                # Correctly sliced hidden states
                "init_slot_hidden": init_slot,
                "init_bev_hidden": init_bev
            }
    # def feed_forward_generator(self):
    #
    #     images = torch.tensor(self.images, device=self.device) if self.images is not None else None
    #     agent_feats = torch.tensor(self.agent_feats, device=self.device)
    #     type_id = torch.tensor(self.type_id, device=self.device)
    #     actions = torch.tensor(self.actions, device=self.device)
    #     logp = torch.tensor(self.logp, device=self.device)
    #     returns = torch.tensor(self.returns, device=self.device)
    #     advantages = torch.tensor(self.advantages, device=self.device)
    #     values = torch.tensor(self.values, device=self.device)
    #     masks = torch.tensor(self.masks, device=self.device)
    #     pre_t = torch.tensor(self.pre_t, device=self.device)
    #
    #     if self.slot_hidden is not None:
    #         init_slot = torch.tensor(self.slot_hidden[0], device=self.device)
    #     else:
    #         init_slot = None
    #
    #     if self.bev_hidden is not None:
    #         init_bev = torch.tensor(self.bev_hidden[0], device=self.device)
    #     else:
    #         init_bev = None
    #
    #     yield {
    #         "images": images,
    #         "agent_feats": agent_feats,
    #         "type_id": type_id,
    #         "actions": actions,
    #         "logp": logp,
    #         "returns": returns,
    #         "advantages": advantages,
    #         "values": values,
    #         "masks": masks,
    #         "pre_t": pre_t,
    #         "init_slot_hidden": init_slot,
    #         "init_bev_hidden": init_bev
    #     }

    def clear(self):
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
        self.pre_t.fill(0)
        self.advantages.fill(0)
        self.returns.fill(0)

        self.init_slot_hidden = None
        self.init_bev_hidden = None

    def remaining(self) -> int:
        return self.T - self.ptr

    def summary(self) -> Dict[str, Any]:
        return {
            "T": self.T, "B": self.B, "N": self.N,
            "ptr": self.ptr, "full": self.full,
            "device": str(self.device),
            "has_images": self.images is not None
        }

    def set_initial_hidden(self, slot_hidden, bev_hidden):
        self.init_slot_hidden = slot_hidden
        self.init_bev_hidden = bev_hidden

