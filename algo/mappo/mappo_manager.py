from collections import OrderedDict
from typing import Dict, Any, Optional, Iterable
import torch
import numpy as np
import sys
import traceback

from buffer.rollout_buffer import RolloutBuffer  # ensure this path is correct for your project


def _ensure_numpy(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class MAPPOManager(object):
    def __init__(self, agent_specs=None, policy_ctor=None, device: str = "cpu", selector: Optional[Any] = None):
        """
        Single multi-agent buffer manager.
        agent_specs: either numeric-keyed dict {slot: spec} or type-keyed {"vehicle": spec_with n_agents, ...}
        policy_ctor: callable(spec) -> policy instance
        """
        import copy
        self.device = torch.device(device)
        self.selector = selector

        specs = dict(agent_specs or {})
        numeric_keys = all(isinstance(k, int) for k in specs.keys()) if specs else False

        self.agents = OrderedDict()
        # no per-slot buffers anymore; single buffer below
        self.buffer = None

        # expand type-keyed specs into numeric slots if necessary
        if numeric_keys:
            self.agent_slots = list(specs.keys())
            spec_list = [specs[k] for k in self.agent_slots]
        else:
            expanded = []
            for typ, spec in specs.items():
                n = int(spec.get("n_agents", 1))
                for _ in range(n):
                    s = copy.deepcopy(spec)
                    s["_agent_type"] = typ
                    expanded.append(s)
            self.agent_slots = list(range(len(expanded)))
            spec_list = expanded

        # instantiate agent policies per slot (homogeneous or heterogeneous supported)
        for idx, spec in enumerate(spec_list):
            try:
                self.agents[idx] = policy_ctor(spec)
            except Exception as e:
                raise RuntimeError(f"Failed to instantiate policy for slot {idx}: {e}")

        # convenience pointers
        try:
            self.policy = next(iter(self.agents.values()))
        except StopIteration:
            self.policy = None

        # mapping placeholders used by env-manager syncing
        self.slot2veh = {s: None for s in self.agent_slots}
        self.veh2slot = {}

        # initialize single multi-agent buffer (take buffer_T from first spec if available)
        try:
            first_spec = spec_list[0] if spec_list else {}
            buffer_T = int(first_spec.get("buffer_T", 128))
            image_shape = first_spec.get("image_shape", (3, 84, 84))
        except Exception:
            buffer_T = 128
            image_shape = (3, 84, 84)
        # set some manager-level defaults (can be overridden by caller)
        self.gamma = float(first_spec.get("gamma", 0.99)) if isinstance(first_spec, dict) else 0.99
        self.gae_lambda = float(first_spec.get("gae_lambda", 0.95)) if isinstance(first_spec, dict) else 0.95
        self.init_rollout_buffer(T=buffer_T, image_shape=image_shape, device=str(self.device))

        try:
            params = [p for p in self.policy.parameters() if p.requires_grad]
            self.optim = torch.optim.Adam(params, lr=3e-4, eps=1e-5) if params else None
        except Exception:
            self.optim = None

    # ------------- Rollout Buffer helpers (single buffer) -------------
    def init_rollout_buffer(self, T: int = 128, image_shape=None, device: str = "cpu"):
        """
        Create single multi-agent RolloutBuffer for the whole environment.
        """
        B = 1
        N = len(self.agent_slots)
        feat_dim = getattr(self, "obs_dim", 128)
        act_dim = getattr(self.policy, "act_dim", 2) if getattr(self, "policy", None) is not None else 2
        gamma = getattr(self, "gamma", 0.99)
        gae_lambda = getattr(self, "gae_lambda", 0.95)

        self.buffer = RolloutBuffer(
            T=T,
            num_envs=B,
            n_agents=N,
            image_shape=image_shape,
            agent_feat_dim=feat_dim,
            act_dim=act_dim,
            gamma=gamma,
            gae_lambda=gae_lambda,
            device=device
        )

    def reset_buffer(self):
        """Clear the single rollout buffer (re-create if missing)."""
        if not hasattr(self, "buffer") or self.buffer is None:
            self.init_rollout_buffer(T=128, image_shape=(3, 84, 84), device=str(self.device))
        else:
            self.buffer.clear()

    def store_step(self, imgs, agent_feats, type_ids, actions, logp, values, rewards, masks):
        """
        Store one timestep of whole-batch data into the multi-agent RolloutBuffer (expects B dim).
        Inputs should be numpy or torch; shapes expected:
         imgs: [B, C, H, W] or None
         agent_feats: [B, N, F]
         type_ids: [B, N] or None
         actions: [B, N, A]
         logp: [B, N]
         values: [B, N]
         rewards: [B, N]
         masks: [B, N]
        """
        if self.buffer is None:
            self.init_rollout_buffer(T=128, image_shape=(3, 84, 84), device=str(self.device))

        imgs_np = _ensure_numpy(imgs)
        feats_np = _ensure_numpy(agent_feats)
        types_np = _ensure_numpy(type_ids)
        acts_np = _ensure_numpy(actions)
        logp_np = _ensure_numpy(logp)
        vals_np = _ensure_numpy(values)
        rews_np = _ensure_numpy(rewards)
        masks_np = _ensure_numpy(masks)

        # ensure at least batch dim exists
        if feats_np is not None and feats_np.ndim == 2:
            feats_np = feats_np[None, ...]
        if acts_np is not None and acts_np.ndim == 2:
            acts_np = acts_np[None, ...]
        if rews_np is not None and rews_np.ndim == 1:
            rews_np = rews_np[None, ...]
        if logp_np is not None and logp_np.ndim == 1:
            logp_np = logp_np[None, ...]
        if types_np is not None and types_np.ndim == 1:
            types_np = types_np[None, ...]
        if masks_np is not None and masks_np.ndim == 1:
            masks_np = masks_np[None, ...]

        # call underlying buffer method (implementation-dependent)
        self.buffer.add_batch(imgs_np, feats_np, types_np, acts_np, logp_np, vals_np, rews_np, masks_np)

    # Replace existing store_transitions method in mappo_manager.py with this implementation.

    def store_transitions(self, obs_dict=None, act_dict=None, rew_dict=None, done_dict=None,
                          val_dict=None, logp_dict=None, debug_shapes: bool = False,
                          strict_image_check: bool = None, **kwargs):
        """
        Normalize inputs and call store_step with arrays in expected order.
        - debug_shapes: if True, print shapes of core arrays before storing (helpful for debugging).
        - strict_image_check: if True -> raise on missing image; if None -> use manager default self.strict_image_check
        """
        import numpy as _np
        import torch

        # use manager-level default if arg not provided
        if strict_image_check is None:
            strict_image_check = getattr(self, "strict_image_check", False)

        obs = obs_dict if obs_dict is not None else {}

        # -------- images normalization -> imgs_arr (numpy, [T,C,H,W]) --------
        imgs_arr = None
        if isinstance(obs, dict):
            if "images" in obs and obs["images"] is not None:
                imgs_arr = obs["images"]
            elif "image" in obs and obs["image"] is not None:
                im = obs["image"]
                if torch.is_tensor(im):
                    try:
                        im_np = im.detach().cpu().numpy()
                    except Exception:
                        im_np = None
                else:
                    try:
                        im_np = _np.asarray(im)
                    except Exception:
                        im_np = None
                if im_np is not None:
                    # handle [B,C,H,W], [C,H,W], [H,W,C] etc.
                    if im_np.ndim == 4 and im_np.shape[0] == 1:
                        im_np = im_np[0]  # [C,H,W]
                    if im_np.ndim == 4 and im_np.shape[-1] in (1, 3, 4) and im_np.shape[1] not in (1, 3, 4):
                        # ambiguous: maybe [T,H,W,C] or [B,C,H,W]; try detect channel-pos
                        pass
                    if im_np.ndim == 3:
                        imgs_arr = im_np[None, ...]  # [1,C,H,W]
                    elif im_np.ndim == 4:
                        # could be [T,C,H,W] or [T,H,W,C] -> if last dim channel, transpose
                        if im_np.shape[-1] in (1, 3, 4) and im_np.shape[1] not in (1, 3, 4):
                            imgs_arr = _np.transpose(im_np, (0, 3, 1, 2))
                        else:
                            imgs_arr = im_np
                    else:
                        imgs_arr = im_np

        # fallback to cached warmup images if present on manager (existing behavior)
        if imgs_arr is None:
            seq_store = getattr(self, "last_warmup_images", None)
            if seq_store is not None:
                imgs_arr = seq_store

        # --- NEW: explicit handling for missing images ---
        if imgs_arr is None:
            msg = "[store_transitions] missing image(s) in obs_dict."
            if strict_image_check:
                # strict mode: raise immediately to force upstream fix
                raise RuntimeError(msg + " Manager strict_image_check=True -- aborting to surface missing images.")
            # non-strict: warn once and fall back to zero image (but make this explicit)
            if not getattr(self, "_warned_missing_images", False):
                print(msg + " Falling back to zero-image placeholder for this step. "
                            "Set manager.strict_image_check=True to raise instead of fallback.")
                self._warned_missing_images = True
            # create zero placeholder consistent with expected sizes
            bev_ch = int(getattr(self, "bev_ch", getattr(self, "obs_ch", 3)))
            bev_H = int(getattr(self, "bev_H", getattr(self, "obs_H", 84)))
            bev_W = int(getattr(self, "bev_W", getattr(self, "obs_W", 84)))
            imgs_arr = _np.zeros((1, bev_ch, bev_H, bev_W), dtype=_np.float32)

        # ensure numpy array and layout [T,C,H,W]
        if isinstance(imgs_arr, list):
            imgs_arr = _np.asarray(imgs_arr)
        try:
            imgs_arr = _np.asarray(imgs_arr, dtype=_np.float32)
        except Exception:
            imgs_arr = _np.zeros(
                (1, getattr(self, "bev_ch", 3), getattr(self, "bev_H", 84), getattr(self, "bev_W", 84)),
                dtype=_np.float32)

        # if imgs_arr is [T,B,C,H,W] -> squeeze B dim if B==1
        if imgs_arr.ndim == 5 and imgs_arr.shape[1] == 1:
            imgs_arr = imgs_arr[:, 0, ...]

        # if imgs_arr is [T,H,W,C] -> transpose to [T,C,H,W]
        if imgs_arr.ndim == 4 and imgs_arr.shape[-1] in (1, 3, 4) and imgs_arr.shape[1] not in (1, 3, 4):
            imgs_arr = _np.transpose(imgs_arr, (0, 3, 1, 2))

        # -------- agent_feats/types/mask normalization (unchanged but robust) --------
        feats_np = None
        types_np = None
        masks_np = None
        if isinstance(obs, dict):
            af = obs.get("agent_feats", None)
            if af is not None:
                if torch.is_tensor(af):
                    af_np = af.detach().cpu().numpy()
                else:
                    af_np = _np.asarray(af)
                # possible shapes: [T,N,F], [T,B,N,F], [N,F], [B,N,F]
                if af_np.ndim == 4:
                    feats_np = af_np
                elif af_np.ndim == 3:
                    feats_np = af_np
                elif af_np.ndim == 2:
                    feats_np = af_np[None, ...]
                else:
                    feats_np = af_np

            tid = obs.get("type_id", None)
            if tid is not None:
                if torch.is_tensor(tid):
                    tid_np = tid.detach().cpu().numpy()
                else:
                    tid_np = _np.asarray(tid)
                if tid_np.ndim == 3:
                    types_np = tid_np
                elif tid_np.ndim == 2:
                    types_np = tid_np[None, ...]
                elif tid_np.ndim == 1:
                    types_np = tid_np[None, ...]
                else:
                    types_np = tid_np

            m = obs.get("mask", None)
            if m is not None:
                if torch.is_tensor(m):
                    m_np = m.detach().cpu().numpy()
                else:
                    m_np = _np.asarray(m)
                if m_np.ndim == 2:
                    masks_np = m_np[None, ...]
                elif m_np.ndim == 3:
                    masks_np = m_np
                else:
                    masks_np = m_np

        # -------- actions/logps/vals/rewards normalization (unchanged) --------
        slots = getattr(self, "agent_slots", list(act_dict.keys()) if act_dict is not None else [])
        acts_np = None
        if act_dict is not None:
            acts_list = []
            for s in slots:
                a = act_dict.get(s, None)
                if a is None:
                    acts_list.append(_np.zeros((1,), dtype=_np.float32))
                else:
                    if torch.is_tensor(a):
                        acts_list.append(a.detach().cpu().numpy())
                    else:
                        acts_list.append(_np.asarray(a))
            try:
                stacked = _np.stack(acts_list, axis=0)
                acts_np = stacked[None, ...]
            except Exception:
                acts_np = _np.asarray(acts_list, dtype=object)[None, ...]

        logp_np = None
        if logp_dict is not None:
            lps = []
            for s in slots:
                lp = logp_dict.get(s, None)
                if lp is None:
                    lps.append(0.0)
                else:
                    lps.append(float(lp))
            logp_np = _np.asarray(lps, dtype=_np.float32)[None, :]

        vals_np = None
        if val_dict is not None:
            vs = []
            for s in slots:
                v = val_dict.get(s, 0.0)
                vs.append(float(v))
            vals_np = _np.asarray(vs, dtype=_np.float32)[None, :]

        rews_np = None
        if rew_dict is not None:
            rs = []
            for s in slots:
                r = rew_dict.get(s, 0.0)
                rs.append(float(r))
            rews_np = _np.asarray(rs, dtype=_np.float32)[None, :]

        if masks_np is None:
            seq_masks = kwargs.get("masks", None)
            if seq_masks is not None:
                try:
                    masks_np = _np.asarray(seq_masks, dtype=_np.float32)
                except Exception:
                    masks_np = None

        if masks_np is None:
            if slots:
                masks_np = _np.ones((1, len(slots)), dtype=_np.float32)
            else:
                masks_np = _np.ones((1, 1), dtype=_np.float32)

        # --- Optional debug printing of shapes ---
        if debug_shapes:
            try:
                print("[store_transitions:shapes] imgs:", getattr(imgs_arr, "shape", None),
                      "feats:", getattr(feats_np, "shape", None),
                      "acts:", getattr(acts_np, "shape", None),
                      "logp:", getattr(logp_np, "shape", None),
                      "vals:", getattr(vals_np, "shape", None),
                      "rews:", getattr(rews_np, "shape", None),
                      "masks:", getattr(masks_np, "shape", None))
            except Exception:
                pass

        # final call to store_step in the expected order
        return self.store_step(imgs_arr, feats_np, types_np, acts_np, logp_np, vals_np, rews_np, masks_np)

    def update_all(self,
                   ppo_epochs: int = 4,
                   mini_batch_size: int = 256,
                   clip_coef: float = 0.2,
                   value_coef: float = 0.5,
                   entropy_coef: float = 0.01,
                   max_grad_norm: float = 0.5,
                   lr: float = 3e-4):

        buf = self.buffer
        if buf is None:
            raise RuntimeError("update_all: no buffer")

        total = int(buf.T) * int(buf.B) * int(buf.N)
        if total <= 0:
            buf.clear()
            raise RuntimeError("update_all: buffer empty or malformed")

        params = [p for p in self.policy.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("update_all: no trainable params")

        optim = self.optim if hasattr(self, "optim") else torch.optim.Adam(params, lr=lr, eps=1e-5)
        device = next(self.policy.parameters()).device

        for epoch in range(ppo_epochs):
            gen_epoch = buf.feed_forward_generator(mini_batch_size=mini_batch_size)
            for batch_idx, batch in enumerate(gen_epoch):
                if "agent_feats" not in batch:
                    raise RuntimeError("update_all: batch missing 'agent_feats'")
                af_raw = batch["agent_feats"]
                if isinstance(af_raw, torch.Tensor):
                    af = af_raw
                else:
                    af = torch.as_tensor(np.asarray(af_raw), dtype=torch.float32)
                if af.dim() == 3:
                    B, N, F = af.shape
                    agent_feats = af.to(device)
                elif af.dim() == 2:
                    rows = af.shape[0]
                    N_try = getattr(buf, "N", None) or getattr(self, "max_slots", None) or 16
                    N_try = int(N_try)
                    if rows % N_try != 0:
                        raise RuntimeError(f"update_all: cannot infer N from agent_feats rows={rows}, N_try={N_try}")
                    B = rows // N_try
                    N = N_try
                    agent_feats = af.view(B, N, af.shape[1]).to(device)
                else:
                    raise RuntimeError(f"update_all: agent_feats must be 2D or 3D tensor, got dim={af.dim()}")

                def _strict_reshape_to_BN(x_raw, name):
                    if x_raw is None:
                        return None
                    if isinstance(x_raw, torch.Tensor):
                        t = x_raw
                    else:
                        t = torch.as_tensor(np.asarray(x_raw), dtype=torch.float32)
                    if t.dim() >= 1 and t.shape[0] == B * N:
                        return t.view(B, N, *t.shape[1:]).to(device)
                    if t.dim() >= 1 and t.shape[0] == B:
                        return t.to(device)
                    raise RuntimeError(
                        f"update_all: field '{name}' has invalid leading dimension {t.shape[0]}, expected {B * N} or {B}")

                actions = _strict_reshape_to_BN(batch.get("actions", None), "actions")
                returns = _strict_reshape_to_BN(batch.get("returns", None), "returns")

                _tmp_old = batch.get("logp", None)
                if _tmp_old is None:
                    _tmp_old = batch.get("old_log_probs", None)
                old_log_ps = _strict_reshape_to_BN(_tmp_old, "old_logp")

                advantages = _strict_reshape_to_BN(batch.get("advantages", None), "advantages")
                if advantages is None:
                    raise RuntimeError("update_all: advantages required for PPO update")
                adv_mean = advantages.mean()
                adv_std = advantages.std()
                if adv_std.item() == 0:
                    raise RuntimeError("update_all: advantages std == 0")
                advantages = (advantages - adv_mean) / (adv_std + 1e-8)

                if actions is not None and actions.shape[0] != B:
                    raise RuntimeError("update_all: actions batch size mismatch")
                if returns is not None and returns.shape[0] not in (B, B * N):
                    raise RuntimeError("update_all: returns shape mismatch")

                obs_for_policy = {"agent_feats": agent_feats}
                img_raw = batch.get("images", None)
                if img_raw is None:
                    img_raw = batch.get("image", None)
                if img_raw is not None:
                    if isinstance(img_raw, torch.Tensor):
                        it = img_raw
                    else:
                        it = torch.as_tensor(np.asarray(img_raw), dtype=torch.float32)
                    if it.dim() == 4 and it.shape[0] == B * N:
                        it = it.view(B, N, it.shape[1], it.shape[2], it.shape[3])
                        obs_for_policy["image"] = it.mean(dim=1).to(device)
                    elif it.dim() == 4 and it.shape[0] == B:
                        obs_for_policy["image"] = it.to(device)
                    elif it.dim() == 5 and it.shape[0] == B:
                        obs_for_policy["image"] = it.mean(dim=1).to(device)
                    else:
                        raise RuntimeError(f"update_all: image has unsupported shape {tuple(it.shape)}")

                tid_raw = batch.get("type_id", None)
                if tid_raw is not None:
                    if isinstance(tid_raw, torch.Tensor):
                        tid_t = tid_raw
                    else:
                        tid_t = torch.as_tensor(np.asarray(tid_raw), dtype=torch.int64)
                    if tid_t.dim() == 2 and tuple(tid_t.shape[:2]) == (B, N):
                        obs_for_policy["type_id"] = tid_t.to(device)
                    elif tid_t.dim() == 1 and tid_t.shape[0] == B * N:
                        obs_for_policy["type_id"] = tid_t.view(B, N).to(device)
                    else:
                        raise RuntimeError(f"update_all: type_id has unsupported shape {tuple(tid_t.shape)}")

                eval_out = None
                if hasattr(self, "eval_policy_on_minibatch"):
                    eval_out = self.eval_policy_on_minibatch(batch)
                    if not isinstance(eval_out, dict):
                        raise RuntimeError("update_all: eval_policy_on_minibatch must return dict")
                else:
                    policy = self.policy
                    if hasattr(policy, "evaluate_actions"):
                        try:
                            eval_out = policy.evaluate_actions(obs_for_policy, actions)
                        except Exception as e:
                            raise RuntimeError(f"update_all: policy.evaluate_actions failed: {e}")
                    if eval_out is None:
                        if hasattr(policy, "forward"):
                            try:
                                eval_out = policy.forward(obs=obs_for_policy, mode="step", deterministic=True)
                            except Exception as e:
                                raise RuntimeError(f"update_all: policy.forward failed: {e}")
                    if eval_out is None:
                        raise RuntimeError("update_all: policy evaluation failed")
                    if not isinstance(eval_out, dict):
                        raise RuntimeError("update_all: policy evaluation must return dict")

                _tmp_log = None
                for _k in ("log_probs", "logp", "log_prob", "action_logp", "action_log_probs"):
                    if _k in eval_out and eval_out[_k] is not None:
                        _tmp_log = eval_out[_k]
                        break
                if _tmp_log is None:
                    raise RuntimeError("update_all: eval_out missing log-prob field (checked keys)")
                log_probs_new = _tmp_log

                if not isinstance(log_probs_new, torch.Tensor):
                    log_probs_new = torch.as_tensor(np.asarray(log_probs_new), dtype=torch.float32)
                log_probs_new = log_probs_new.to(device)

                if old_log_ps is None:
                    raise RuntimeError("update_all: old_log_ps required")
                if not isinstance(old_log_ps, torch.Tensor):
                    old_log_ps = torch.as_tensor(np.asarray(old_log_ps), dtype=torch.float32)
                old_log_ps = old_log_ps.to(device)

                _tmp_val = None
                for _k in ("values", "value", "state_value", "v", "critic_value"):
                    if _k in eval_out and eval_out[_k] is not None:
                        _tmp_val = eval_out[_k]
                        break
                if _tmp_val is None:
                    raise RuntimeError("update_all: eval_out missing value field (checked keys)")
                values_pred = _tmp_val

                if not isinstance(values_pred, torch.Tensor):
                    values_pred = torch.as_tensor(np.asarray(values_pred), dtype=torch.float32)
                values_pred = values_pred.to(device)

                entropy = eval_out.get("entropy", None)
                if entropy is None:
                    entropy = torch.tensor(0.0, device=device)
                else:
                    if not isinstance(entropy, torch.Tensor):
                        entropy = torch.as_tensor(np.asarray(entropy), dtype=torch.float32).to(device)

                if log_probs_new.shape != old_log_ps.shape:
                    raise RuntimeError(
                        f"update_all: log_probs shape mismatch new={tuple(log_probs_new.shape)} old={tuple(old_log_ps.shape)}")
                if advantages.shape != log_probs_new.shape:
                    raise RuntimeError(
                        f"update_all: advantages shape {tuple(advantages.shape)} must match log_probs shape {tuple(log_probs_new.shape)}")

                ratio = torch.exp(log_probs_new - old_log_ps)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                if values_pred.shape != returns.shape:
                    values_pred = values_pred.view(returns.shape)

                value_loss = (values_pred - returns).pow(2).mean()
                entropy_term = entropy.mean()

                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_term

                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                optim.step()

        buf.clear()

    def forward(self, obs: Dict[str, torch.Tensor],
                slot_hidden: Optional[torch.Tensor] = None,
                bev_hidden: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                mode: str = "step",
                deterministic: bool = False) -> Dict[str, Any]:
        """
        Manager-level forward wrapper.
        - If self.policy.forward exists and supports mode, call it (preferred, efficient).
        - Otherwise fallback to calling self.policy.select_action repeatedly (seq) or once (step).
        Args:
            obs: dict. For seq: keys like "agent_feats": [T, B, N, F] or [T, N, F].
                 For step: "agent_feats": [B, N, F] or [N, F] (B=1)
            slot_hidden: optional tensor [B, N, H] or [T, B, N, H]
            bev_hidden: optional tensor [B, D] or [T, B, D]
            mask: optional mask tensor
            mode: "step" or "seq"
            deterministic: whether to sample deterministically (useful for burn-in)
        Returns:
            dict with keys possibly including:
              - "actions": [B, N, A] (only for step/fallback)
              - "value" or "values": [T,B,N] or [B,N]
              - "slot_hidden" / "next_slot_hidden": [B,N,H] or [T,B,N,H]
              - "bev_hidden" / "next_bev_hidden": [B,D] or [T,B,D]
              - "logp": [B,N] (for step)
              - "info": policy info dict
        """
        import torch

        # helper: ensure tensors on manager device
        def _to_device(x):
            if x is None:
                return None
            if torch.is_tensor(x):
                return x.to(self.device)
            try:
                return torch.tensor(x, dtype=torch.float32, device=self.device)
            except Exception:
                return x

        # normalize obs tensors to manager device
        obs_on_dev = {}
        for k, v in (obs.items() if isinstance(obs, dict) else []):
            if v is None:
                obs_on_dev[k] = None
            else:
                if torch.is_tensor(v):
                    obs_on_dev[k] = v.to(self.device)
                else:
                    try:
                        obs_on_dev[k] = torch.tensor(v, dtype=torch.float32, device=self.device)
                    except Exception:
                        obs_on_dev[k] = v

        # prefer policy.forward when available
        if hasattr(self.policy, "forward"):
            try:
                # try calling policy.forward with same signature
                out = self.policy.forward(obs_on_dev,
                                          slot_hidden=slot_hidden,
                                          bev_hidden=bev_hidden,
                                          mask=mask,
                                          mode=mode,
                                          deterministic=deterministic)
                if isinstance(out, dict):
                    return out
                # if forward returned non-dict, wrap minimally
                return {"out": out}
            except TypeError:
                # forward exists but signature differs; fall through to fallback
                pass
            except Exception:
                # policy.forward attempted but failed; fall back
                pass

        # === Fallback implementation using select_action ===
        # We'll support 'step' (single call) and 'seq' (iterate per timestep)
        if mode == "step":
            # Expect obs contains per-step agent_feats and image; construct obs_step dict
            obs_step = {}
            # pass through keys that select_action expects: "image","agent_feats","type_id"
            for k in ("image", "agent_feats", "type_id", "mask"):
                if k in obs_on_dev:
                    obs_step[k] = obs_on_dev[k]
            # call select_action (deterministic sampling)
            try:
                actions, values, logps, next_slot_h, next_bev_h, info = self.select_actions(
                    obs_dict=obs_step,
                    hidden={"slot": slot_hidden, "bev": bev_hidden},
                    mask=mask
                )
                # select_actions returns per-slot dicts for actions/values/logps.
                # Convert them into batch tensors [B=1, N, ...] for compatibility.
                slots = list(self.agent_slots)
                N = len(slots)
                B = 1
                # stack actions/values/logps according to slot order
                actions_list = [actions.get(s, np.zeros(getattr(self.policy, "act_dim", 2), dtype=np.float32)) for s in
                                slots]
                values_list = [values.get(s, 0.0) for s in slots]
                logp_list = [logps.get(s, 0.0) for s in slots]

                actions_tensor = torch.tensor(np.stack(actions_list, axis=0), dtype=torch.float32,
                                              device=self.device).unsqueeze(0)
                values_tensor = torch.tensor(np.array(values_list, dtype=np.float32), dtype=torch.float32,
                                             device=self.device).unsqueeze(0)
                logp_tensor = torch.tensor(np.array(logp_list, dtype=np.float32), dtype=torch.float32,
                                           device=self.device).unsqueeze(0)

                return {
                    "actions": actions_tensor,
                    "values": values_tensor,
                    "logp": logp_tensor,
                    "next_slot_hidden": next_slot_h,
                    "next_bev_hidden": next_bev_h,
                    "info": info
                }
            except Exception as e:
                # If select_actions failed, return minimal empty
                return {"actions": None, "values": None, "logp": None, "next_slot_hidden": None,
                        "next_bev_hidden": None, "info": {"error": str(e)}}

        elif mode == "seq":
            # Expect obs["agent_feats"] shape [T, B, N, F] or [T, N, F] (B assumed 1)
            af = obs_on_dev.get("agent_feats", None)
            if af is None:
                return {}
            # normalize to [T, B, N, F]
            if torch.is_tensor(af) and af.dim() == 3:
                af = af.unsqueeze(1)  # [T,1,N,F]
            T = af.shape[0]
            # We'll iterate timesteps and call select_actions per-step in deterministic mode (best-effort)
            seq_values = []
            seq_slot_hidden = []
            seq_bev_hidden = []
            info_accum = []
            cur_slot_hidden = slot_hidden
            cur_bev_hidden = bev_hidden
            for t in range(T):
                step_af = af[t]  # [B,N,F]
                # construct per-slot obs_dict expected by select_actions: need per-slot dicts with "agent_feats" and maybe "image"
                # We'll build obs_dict keyed by slot index where agent_feats for slot i is step_af[:, i, :]
                B, N, F = step_af.shape
                obs_dict_step = {}
                # images / type_id if present: use same for all slots (best-effort)
                image_t = obs_on_dev.get("image", None)
                type_id_t = None
                if "type_id" in obs_on_dev:
                    type_id = obs_on_dev["type_id"]
                    # type_id could be time-major too; try to index if possible
                    try:
                        if torch.is_tensor(type_id) and type_id.dim() == 3:
                            type_id_t = type_id[t] if type_id.shape[0] == T else type_id
                        else:
                            type_id_t = type_id
                    except Exception:
                        type_id_t = type_id
                for s_idx, slot in enumerate(self.agent_slots):
                    af_slot = step_af[:, s_idx, :]  # [B,F]
                    slot_obs = {"agent_feats": af_slot}
                    if image_t is not None:
                        slot_obs["image"] = image_t
                    if type_id_t is not None:
                        # type_id for this slot: index if shape [B,N]
                        try:
                            if torch.is_tensor(type_id_t):
                                slot_tid = type_id_t[:, s_idx] if type_id_t.dim() == 2 else type_id_t
                            else:
                                slot_tid = type_id_t
                            slot_obs["type_id"] = slot_tid
                        except Exception:
                            slot_obs["type_id"] = type_id_t
                    obs_dict_step[slot] = slot_obs

                # call select_actions deterministically (policy.select_action will sample; but we pass deterministic flag through manager.select_actions not supported in all managers)
                try:
                    acts, vals, lps, next_slot_h, next_bev_h, info = self.select_actions(obs_dict_step,
                                                                                         hidden={
                                                                                             "slot": cur_slot_hidden,
                                                                                             "bev": cur_bev_hidden},
                                                                                         mask=(mask[t] if (
                                                                                                     torch.is_tensor(
                                                                                                         mask) and mask.dim() == 3) else mask))
                except Exception:
                    # on failure, break seq
                    break

                # convert vals per-slot -> tensor [B,N]
                vals_list = [vals.get(s, 0.0) for s in self.agent_slots]
                vals_tensor = torch.tensor(np.array(vals_list, dtype=np.float32), dtype=torch.float32,
                                           device=self.device).unsqueeze(0)
                seq_values.append(vals_tensor)
                seq_slot_hidden.append(next_slot_h)
                seq_bev_hidden.append(next_bev_h)
                info_accum.append(info)

                # update hidden for next step
                cur_slot_hidden = next_slot_h
                cur_bev_hidden = next_bev_h

            # stack results: values -> [T, B, N]
            if seq_values:
                values_seq = torch.cat(seq_values, dim=0)
            else:
                values_seq = None

            # stack slot hidden: if list of tensors, try stack, else keep last
            try:
                slot_h_stack = torch.stack([h for h in seq_slot_hidden if h is not None], dim=0) if any(
                    h is not None for h in seq_slot_hidden) else None
            except Exception:
                slot_h_stack = seq_slot_hidden[-1] if seq_slot_hidden else None

            try:
                bev_h_stack = torch.stack([h for h in seq_bev_hidden if h is not None], dim=0) if any(
                    h is not None for h in seq_bev_hidden) else None
            except Exception:
                bev_h_stack = seq_bev_hidden[-1] if seq_bev_hidden else None

            return {
                "values": values_seq,
                "next_slot_hidden": slot_h_stack if slot_h_stack is not None else cur_slot_hidden,
                "next_bev_hidden": bev_h_stack if bev_h_stack is not None else cur_bev_hidden,
                "info": info_accum
            }

        # unknown mode: return empty
        return {}

    def select_actions(self, obs_dict, hidden=None, mask=None):
        """
        Robust select_actions that accepts:
          - per-slot obs: {slot: {"image":..., "agent_feats":..., "type_id":..., "mask":...}, ...}
          - or global obs: {"image":..., "agent_feats":..., "type_id":..., "mask":...}
        Returns:
          actions(dict slot->np.array), values(dict slot->float), logps(dict slot->float),
          next_slot_h, next_bev_h, info
        """
        import torch
        import numpy as np

        # --- quick empty check ---
        if not obs_dict:
            actions = {}
            values = {}
            logps = {}
            for slot in getattr(self, "agent_slots", list(self.agents.keys())):
                agent = self.agents.get(slot, None)
                act_dim = getattr(agent, "action_dim", getattr(agent, "act_dim", 2)) if agent is not None else 2
                actions[slot] = np.zeros(act_dim, dtype=np.float32)
                values[slot] = 0.0
                logps[slot] = -1e8
            return actions, values, logps, None, None, {}

        # --- slots / per-slot detection ---
        slots = getattr(self, "agent_slots", list(self.agents.keys()))
        is_per_slot = any(k in slots for k in obs_dict.keys())

        # device
        device = getattr(self, "device", torch.device("cpu"))
        if isinstance(device, str):
            device = torch.device(device)

        # --- extract raw pieces ---
        if not is_per_slot:
            image_raw = obs_dict.get("image", None)
            agent_feats_raw = obs_dict.get("agent_feats", None)
            type_id_raw = obs_dict.get("type_id", None)
            mask_raw = obs_dict.get("mask", None)
        else:
            first_slot = next((k for k in obs_dict.keys() if k in slots), slots[0])
            first_obs = obs_dict.get(first_slot, {})
            image_raw = first_obs.get("image", None)
            agent_feats_raw = None
            type_id_raw = None
            mask_raw = None

        # fallback to env.latest_image if available
        if image_raw is None and getattr(self, "env", None) is not None:
            image_raw = getattr(self.env, "latest_image", None)

        # helper: to torch tensor on device
        def _to_tensor(x, dtype=None):
            if x is None:
                return None
            if torch.is_tensor(x):
                t = x
            else:
                try:
                    t = torch.as_tensor(x)
                except Exception:
                    try:
                        import numpy as _np
                        t = torch.from_numpy(_np.array(x))
                    except Exception:
                        return None
            if dtype is not None:
                t = t.to(dtype=dtype)
            return t.to(device)

        # --- normalize image to [B, seq, C, H, W] (seq=1 default) ---
        img_t = _to_tensor(image_raw)
        if img_t is None:
            bev_ch = int(getattr(self, "bev_ch", 3))
            bev_H = int(getattr(self, "bev_H", 84))
            bev_W = int(getattr(self, "bev_W", 84))
            image_tensor = torch.zeros((1, 1, bev_ch, bev_H, bev_W), dtype=torch.float32, device=device)
        else:
            if img_t.dim() == 5:
                if img_t.shape[2] not in (1, 3, 4) and img_t.shape[-1] in (1, 3, 4):
                    image_tensor = img_t.permute(0, 1, 4, 2, 3).contiguous()
                else:
                    image_tensor = img_t.contiguous()
            elif img_t.dim() == 4:
                if img_t.shape[1] in (1, 3, 4):
                    image_tensor = img_t.unsqueeze(1).contiguous()
                else:
                    image_tensor = img_t.permute(0, 3, 1, 2).unsqueeze(1).contiguous()
            elif img_t.dim() == 3:
                if img_t.shape[0] in (1, 3, 4):
                    image_tensor = img_t.unsqueeze(0).unsqueeze(1).contiguous()
                else:
                    image_tensor = img_t.permute(2, 0, 1).unsqueeze(0).unsqueeze(1).contiguous()
            else:
                image_tensor = img_t.reshape(1, 1, *img_t.shape).contiguous()

        if image_tensor.dtype != torch.float32:
            image_tensor = image_tensor.float()
        try:
            if image_tensor.max() > 1.5:
                image_tensor = image_tensor / 255.0
        except Exception:
            pass

        B = int(image_tensor.shape[0])
        seq_len = int(image_tensor.shape[1])
        N = len(slots)

        # --- build agent_feats/type_id/mask tensors ---
        obs_dim = int(getattr(self, "obs_dim", 32))
        if is_per_slot:
            per_feats = []
            per_tids = []
            per_masks = []
            for slot in slots:
                slot_o = obs_dict.get(slot, {})
                af = _to_tensor(slot_o.get("agent_feats", None))
                if af is None:
                    aft = torch.zeros((B, seq_len, obs_dim), dtype=torch.float32, device=device)
                else:
                    aft = af.to(device)
                    if aft.dim() == 2:
                        aft = aft.unsqueeze(1).repeat(1, seq_len, 1)
                    elif aft.dim() == 3:
                        if aft.shape[1] != seq_len:
                            if aft.shape[1] == 1:
                                aft = aft.repeat(1, seq_len, 1)
                            else:
                                aft = aft[:, -seq_len:, :].contiguous()
                    else:
                        aft = aft.reshape(B, seq_len, -1) if aft.numel() else torch.zeros((B, seq_len, obs_dim),
                                                                                          device=device)
                per_feats.append(aft)

                tid = _to_tensor(slot_o.get("type_id", None), dtype=torch.long)
                if tid is None:
                    tidt = torch.zeros((B, seq_len), dtype=torch.long, device=device)
                else:
                    tidt = tid.long().to(device)
                    if tidt.dim() == 1:
                        tidt = tidt.unsqueeze(1).repeat(1, seq_len)
                    elif tidt.dim() == 2 and tidt.shape[1] != seq_len:
                        if tidt.shape[1] == 1:
                            tidt = tidt.repeat(1, seq_len)
                        else:
                            tidt = tidt[:, -seq_len:].contiguous()
                per_tids.append(tidt)

                m = _to_tensor(slot_o.get("mask", None))
                if m is None:
                    mt = torch.ones((B, seq_len), dtype=torch.float32, device=device)
                else:
                    mt = m.float().to(device)
                    if mt.dim() == 1:
                        mt = mt.unsqueeze(1).repeat(1, seq_len)
                    elif mt.dim() == 2 and mt.shape[1] != seq_len:
                        if mt.shape[1] == 1:
                            mt = mt.repeat(1, seq_len)
                        else:
                            mt = mt[:, -seq_len:].contiguous()
                per_masks.append(mt)

            agent_feats_tensor = torch.stack(per_feats, dim=2)
            type_id_tensor = torch.stack(per_tids, dim=2)
            mask_tensor = torch.stack(per_masks, dim=2)
        else:
            aft = _to_tensor(agent_feats_raw)
            if aft is None:
                agent_feats_tensor = torch.zeros((B, seq_len, N, obs_dim), dtype=torch.float32, device=device)
            else:
                a = aft.to(device)
                if a.dim() == 3:
                    agent_feats_tensor = a.unsqueeze(1).repeat(1, seq_len, 1, 1)
                elif a.dim() == 4:
                    if a.shape[1] != seq_len:
                        if a.shape[1] == 1:
                            agent_feats_tensor = a.repeat(1, seq_len, 1, 1)
                        elif a.shape[1] > seq_len:
                            agent_feats_tensor = a[:, -seq_len:, :, :].contiguous()
                        else:
                            pad = torch.zeros((B, seq_len - a.shape[1], a.shape[2], a.shape[3]), device=device)
                            agent_feats_tensor = torch.cat([a, pad], dim=1)
                    else:
                        agent_feats_tensor = a
                else:
                    agent_feats_tensor = torch.zeros((B, seq_len, N, obs_dim), dtype=torch.float32, device=device)

            tidt = _to_tensor(type_id_raw, dtype=torch.long)
            if tidt is None:
                type_id_tensor = torch.zeros((B, seq_len, N), dtype=torch.long, device=device)
            else:
                if tidt.dim() == 2:
                    type_id_tensor = tidt.unsqueeze(1).repeat(1, seq_len, 1)
                elif tidt.dim() == 3:
                    if tidt.shape[1] != seq_len:
                        if tidt.shape[1] == 1:
                            type_id_tensor = tidt.repeat(1, seq_len, 1)
                        elif tidt.shape[1] > seq_len:
                            type_id_tensor = tidt[:, -seq_len:, :].contiguous()
                        else:
                            pad = torch.zeros((B, seq_len - tidt.shape[1], tidt.shape[2]), dtype=torch.long,
                                              device=device)
                            type_id_tensor = torch.cat([tidt, pad], dim=1)
                    else:
                        type_id_tensor = tidt
                else:
                    type_id_tensor = torch.zeros((B, seq_len, N), dtype=torch.long, device=device)

            mtt = _to_tensor(mask_raw)
            if mtt is None:
                mask_tensor = torch.ones((B, seq_len, N), dtype=torch.float32, device=device)
            else:
                if mtt.dim() == 2:
                    mask_tensor = mtt.unsqueeze(1).repeat(1, seq_len, 1).float().to(device)
                elif mtt.dim() == 3:
                    if mtt.shape[1] != seq_len:
                        if mtt.shape[1] == 1:
                            mask_tensor = mtt.repeat(1, seq_len, 1)
                        elif mtt.shape[1] > seq_len:
                            mask_tensor = mtt[:, -seq_len:, :].contiguous()
                        else:
                            pad = torch.zeros((B, seq_len - mtt.shape[1], mtt.shape[2]), device=device)
                            mask_tensor = torch.cat([mtt, pad], dim=1)
                    else:
                        mask_tensor = mtt.float().to(device)
                else:
                    mask_tensor = torch.ones((B, seq_len, N), dtype=torch.float32, device=device)

        obs_step = {
            "image": image_tensor,
            "agent_feats": agent_feats_tensor,
            "type_id": type_id_tensor
        }

        if not torch.is_tensor(mask_tensor):
            mask_tensor = torch.tensor(mask_tensor, dtype=torch.float32, device=device)
        else:
            mask_tensor = mask_tensor.to(device)

        # -------- hidden normalization helpers --------
        def _norm_slot_hidden_for_policy(sh, B_local, N_local):
            """
            Ensure slot_hidden is [B, N, H] (or None). Accepts many input shapes.
            If input is [T, B, N, H], will return last timestep [B, N, H].
            """
            if sh is None:
                return None
            if not torch.is_tensor(sh):
                try:
                    sh = torch.as_tensor(sh, dtype=torch.float32, device=device)
                except Exception:
                    return None

            if sh.dim() == 4:
                # [T, B, N, H] -> last timestep
                sh_last = sh[-1]
                if sh_last.shape[0] != B_local:
                    if sh_last.shape[0] > B_local:
                        sh_last = sh_last[-B_local:]
                    else:
                        sh_last = sh_last.repeat(B_local // sh_last.shape[0] + 1, 1, 1)[:B_local]
                if sh_last.shape[1] != N_local:
                    if sh_last.shape[1] == 1:
                        sh_last = sh_last.repeat(1, N_local, 1)
                    else:
                        minN = min(sh_last.shape[1], N_local)
                        sh_last = sh_last[:, :minN, :].contiguous()
                        if minN < N_local:
                            pad = torch.zeros((B_local, N_local - minN, sh_last.shape[2]), dtype=sh_last.dtype,
                                              device=sh_last.device)
                            sh_last = torch.cat([sh_last, pad], dim=1)
                return sh_last.contiguous().to(device)

            if sh.dim() == 3:
                # [B, N, H] -> ensure correct B and N
                sh2 = sh
                if sh2.shape[0] != B_local:
                    if sh2.shape[0] > B_local:
                        sh2 = sh2[-B_local:]
                    else:
                        sh2 = sh2.repeat(B_local // sh2.shape[0] + 1, 1, 1)[:B_local]
                if sh2.shape[1] != N_local:
                    if sh2.shape[1] == 1:
                        sh2 = sh2.repeat(1, N_local, 1)
                    else:
                        minN = min(sh2.shape[1], N_local)
                        sh2 = sh2[:, :minN, :].contiguous()
                        if minN < N_local:
                            pad = torch.zeros((B_local, N_local - minN, sh2.shape[2]), dtype=sh2.dtype,
                                              device=sh2.device)
                            sh2 = torch.cat([sh2, pad], dim=1)
                return sh2.contiguous().to(device)

            if sh.dim() == 2:
                # [B, H] or [N, H] -> expand N dimension
                if sh.shape[0] == B_local:
                    return sh.unsqueeze(1).repeat(1, N_local, 1).contiguous().to(device)
                if sh.shape[0] == N_local:
                    return sh.unsqueeze(0).contiguous().to(device)
                # fallback: take last row as B=1 and expand
                last = sh[-1:].contiguous()
                return last.unsqueeze(1).repeat(B_local, N_local, 1).contiguous().to(device)

            if sh.dim() == 1:
                return sh.unsqueeze(0).unsqueeze(1).repeat(B_local, N_local, 1).contiguous().to(device)

            return None

        def _norm_bev_hidden_for_policy(bh, B_local):
            """
            Ensure bev_hidden is [B, H]. Accepts [T,B,H], [B,H], [T,H], [H], etc.
            """
            if bh is None:
                try:
                    hdim = int(self.policy.bev_enc.bev_gru.hidden_size)
                except Exception:
                    hdim = getattr(self.policy, "bev_h_dim", 256)
                return torch.zeros((B_local, hdim), dtype=torch.float32, device=device)

            if not torch.is_tensor(bh):
                try:
                    bh = torch.as_tensor(bh, dtype=torch.float32, device=device)
                except Exception:
                    try:
                        hdim = int(self.policy.bev_enc.bev_gru.hidden_size)
                    except Exception:
                        hdim = getattr(self.policy, "bev_h_dim", 256)
                    return torch.zeros((B_local, hdim), dtype=torch.float32, device=device)

            if bh.dim() == 3:
                # [T, B, H] -> take last time
                last = bh[-1]
                if last.shape[0] != B_local:
                    if last.shape[0] > B_local:
                        last = last[-B_local:]
                    else:
                        last = last.repeat(B_local // last.shape[0] + 1, 1)[:B_local]
                return last.contiguous().to(device)

            if bh.dim() == 2:
                if bh.shape[0] != B_local:
                    if bh.shape[0] > B_local:
                        bh = bh[-B_local:]
                    else:
                        bh = bh.repeat(B_local // bh.shape[0] + 1, 1)[:B_local]
                return bh.contiguous().to(device)

            if bh.dim() == 1:
                return bh.unsqueeze(0).contiguous().to(device)

            # fallback zeros
            try:
                hdim = int(self.policy.bev_enc.bev_gru.hidden_size)
            except Exception:
                hdim = getattr(self.policy, "bev_h_dim", 256)
            return torch.zeros((B_local, hdim), dtype=torch.float32, device=device)

        slot_hidden = None
        bev_hidden = None
        if hidden is not None:
            if isinstance(hidden, dict):
                slot_hidden = hidden.get("slot", None)
                bev_hidden = hidden.get("bev", None)
            else:
                slot_hidden = hidden
                bev_hidden = None

        # normalize
        slot_hidden = _norm_slot_hidden_for_policy(slot_hidden, B, N)
        bev_hidden = _norm_bev_hidden_for_policy(bev_hidden, B)

        actions_t, values_t, logps_t, next_slot_h, next_bev_h, info = self.policy.select_action(
            obs_step=obs_step,
            slot_hidden=slot_hidden,
            bev_hidden=bev_hidden,
            mask=mask_tensor,
            deterministic=False
        )

        if actions_t is None:
            raise RuntimeError("policy.select_action returned None actions")

        if isinstance(actions_t, np.ndarray):
            actions_t = torch.from_numpy(actions_t).to(device)
        if not torch.is_tensor(actions_t):
            raise RuntimeError(f"[select_actions] actions_t not tensor-like, got {type(actions_t)}")

        at = actions_t
        while at.dim() > 3 and at.shape[0] == 1:
            at = at.squeeze(0)
        if at.dim() == 4:
            if at.shape[1] == seq_len:
                out_actions = at[:, -1, :, :].contiguous()
            elif at.shape[0] == seq_len:
                out_actions = at[-1, :, :, :].contiguous()
            else:
                out_actions = at[:, -1, :, :].contiguous()
        elif at.dim() == 3:
            out_actions = at.contiguous()
        elif at.dim() == 2:
            out_actions = at.unsqueeze(0).contiguous()
        else:
            _tmp = at.squeeze()
            if _tmp.dim() == 3:
                out_actions = _tmp.contiguous()
            else:
                raise RuntimeError(f"[select_actions] unexpected actions dims {tuple(actions_t.shape)}")

        if out_actions.dim() != 3:
            raise RuntimeError(f"[select_actions] final actions shape unexpected: {tuple(out_actions.shape)}")

        B_out, N_out, A_out = int(out_actions.shape[0]), int(out_actions.shape[1]), int(out_actions.shape[2])

        if len(slots) != N_out:
            minN = min(len(slots), N_out)
            print(
                f"[select_actions] WARNING: slot count mismatch manager:{len(slots)} vs actions:{N_out}; using min={minN}")
            slots = slots[:minN]
            out_actions = out_actions[:, :minN, :]

        def _norm_tensor(t):
            if t is None:
                return None
            if isinstance(t, np.ndarray):
                t = torch.from_numpy(t)
            if not torch.is_tensor(t):
                return None
            tt = t.to(device)
            if tt.dim() == 4:
                tt = tt[:, -1, ...]
            if tt.dim() == 3 and tt.shape[-1] == 1:
                tt = tt.squeeze(-1)
            return tt

        vals_tensor = _norm_tensor(values_t)
        lps_tensor = _norm_tensor(logps_t)

        actions = {}
        values = {}
        logps = {}
        for i, slot in enumerate(slots):
            act_b = out_actions[:, i, :]
            act0 = act_b[0].detach().cpu().numpy()
            actions[slot] = act0.astype(np.float32)

            if vals_tensor is not None:
                try:
                    v = float(vals_tensor[0, i].detach().cpu().item())
                except Exception:
                    v = float(vals_tensor[0, i].detach().cpu().numpy().reshape(-1)[0])
            else:
                v = 0.0
            values[slot] = v

            if lps_tensor is not None:
                try:
                    lp_val = float(lps_tensor[0, i].detach().cpu().item())
                except Exception:
                    lp_val = float(lps_tensor[0, i].detach().cpu().numpy().reshape(-1)[0])
            else:
                lp_val = -1e8
            logps[slot] = lp_val

        return actions, values, logps, next_slot_h, next_bev_h, info

    # ----------------- reset / burn-in / helpers -----------------
    def mappo_reset(self, state_dict: Dict[Any, Any]) -> None:
        """Reset agents and clear the single buffer."""
        if self.buffer is not None:
            self.buffer.clear()
        # notify agents (state_dict keyed by slot)
        for slot, agent in self.agents.items():
            obs = state_dict.get(slot)
            try:
                agent.mappo_reset(obs)
            except Exception:
                pass

    def burn_in(self,
                obs_seq: Dict[str, torch.Tensor],
                slot_hidden: Optional[torch.Tensor] = None,
                bev_hidden: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                detach: bool = True):
        """
        Warm up slot / bev hidden using an observation sequence.
        Returns (next_slot_hidden [B,N,H], next_bev_hidden or None)
        """
        import torch
        with torch.no_grad():
            out = self.forward(obs=obs_seq,
                               slot_hidden=slot_hidden,
                               bev_hidden=bev_hidden,
                               mask=mask,
                               mode="seq",
                               deterministic=True)
        next_slot = out.get("next_slot_hidden", out.get("slot_hidden", None))
        next_bev = out.get("next_bev_hidden", out.get("bev_hidden", None))

        if isinstance(next_slot, torch.Tensor):
            if next_slot.dim() == 4:  # [T,B,N,H]
                next_slot = next_slot[-1]
            if next_slot.dim() == 4 and next_slot.shape[0] == 1:
                next_slot = next_slot.squeeze(0)

        if isinstance(next_bev, torch.Tensor):
            if next_bev.dim() == 3 and next_bev.shape[0] == 1:
                next_bev = next_bev.squeeze(0)

        if detach:
            if isinstance(next_slot, torch.Tensor):
                next_slot = next_slot.detach()
            if isinstance(next_bev, torch.Tensor):
                next_bev = next_bev.detach()

        device = getattr(self, "device", torch.device("cpu"))
        if isinstance(next_slot, torch.Tensor):
            next_slot = next_slot.to(device)
        if isinstance(next_bev, torch.Tensor):
            next_bev = next_bev.to(device)

        return next_slot, next_bev

    def get_initial_hidden(self, batch_size: int, n_agent: int, randomize: bool = False, eps: float = 1e-3):
        """
        Proxy to policy.get_initial_hidden so trainer can call manager.get_initial_hidden(...)
        """
        if hasattr(self, "policy") and hasattr(self.policy, "get_initial_hidden"):
            return self.policy.get_initial_hidden(batch_size, n_agent, randomize=randomize, eps=eps)
        for a in self.agents.values():
            if hasattr(a, "get_initial_hidden"):
                return a.get_initial_hidden(batch_size, n_agent, randomize=randomize, eps=eps)

        slot_h = torch.zeros(batch_size, n_agent, getattr(self.policy, "recurrent_hidden_dim", 256), device=self.device)
        bev_h_dim = getattr(self.policy, "bev_feat_dim", None)
        bev_h = torch.zeros(batch_size, bev_h_dim, device=self.device) if bev_h_dim is not None else None
        return slot_h, bev_h

    def finish_rollouts(self, last_vals):
        """
        Robust finish_rollouts wrapper.

        Accepts last_vals in multiple forms:
          - dict: {slot_name: scalar, ...} -> converted to numpy [1, N] using self.agent_slots order
          - numpy array/list/tuple: converted to numpy [1, N]
          - torch tensor: converted to numpy
          - None: zeros

        Tries common buffer APIs (finish_rollouts, finish_rollout, finish_path, compute_returns, add_bootstrap_values).
        If none available, stores normalized values at self._last_bootstrap_vals for later use.
        """
        import numpy as _np
        import torch

        slots = getattr(self, "agent_slots", None)
        N = len(slots) if slots is not None else int(getattr(self, "num_slots", 0) or 0)

        lv = last_vals

        # dict -> numpy [1, N] following slots order
        if isinstance(lv, dict):
            if slots is None:
                keys = sorted(lv.keys())
                vals = []
                for k in keys:
                    try:
                        v = lv[k]
                        if isinstance(v, (list, tuple, _np.ndarray)):
                            v = float(_np.asarray(v).ravel()[-1])
                        else:
                            v = float(v)
                    except Exception:
                        v = 0.0
                    vals.append(v)
                lv_np = _np.asarray(vals, dtype=_np.float32)[None, :]
                print("[INFO] finish_rollouts: converted dict->array using sorted keys (no agent_slots).")
            else:
                vals = []
                for s in slots:
                    v = lv.get(s, 0.0)
                    try:
                        if isinstance(v, dict):
                            vv = [x for x in v.values() if isinstance(x, (int, float))]
                            v = float(vv[-1]) if vv else 0.0
                        elif isinstance(v, (list, tuple, _np.ndarray)):
                            v = float(_np.asarray(v).ravel()[-1])
                        else:
                            v = float(v)
                    except Exception:
                        v = 0.0
                    vals.append(v)
                lv_np = _np.asarray(vals, dtype=_np.float32)[None, :]

        else:
            # try convert torch -> numpy or list -> numpy
            try:
                if isinstance(lv, (list, tuple)):
                    lv_np = _np.asarray(lv, dtype=_np.float32)
                elif hasattr(lv, "numpy") and not isinstance(lv, _np.ndarray):
                    try:
                        lv_np = lv.numpy()
                    except Exception:
                        import torch as _torch
                        if _torch.is_tensor(lv):
                            lv_np = lv.detach().cpu().numpy()
                        else:
                            lv_np = _np.asarray(lv, dtype=_np.float32)
                elif _np and isinstance(lv, _np.ndarray):
                    lv_np = _np.asarray(lv, dtype=_np.float32)
                else:
                    # torch tensor
                    import torch as _torch
                    if _torch.is_tensor(lv):
                        lv_np = lv.detach().cpu().numpy()
                    else:
                        lv_np = _np.asarray(lv, dtype=_np.float32)
            except Exception:
                lv_np = None

            if lv_np is None:
                lv_np = _np.zeros((1, N or 1), dtype=_np.float32)
            else:
                # normalize shape to [1, N]
                if lv_np.ndim == 0:
                    lv_np = lv_np.reshape((1, 1))
                elif lv_np.ndim == 1:
                    if N and lv_np.shape[0] != N:
                        if lv_np.shape[0] < N:
                            pad = _np.zeros((N - lv_np.shape[0],), dtype=_np.float32)
                            lv_np = _np.concatenate([lv_np, pad], axis=0)
                        else:
                            lv_np = lv_np[:N]
                    lv_np = lv_np[None, :]
                elif lv_np.ndim == 2:
                    if lv_np.shape[0] > 1:
                        lv_np = lv_np[-1:]
                    if N and lv_np.shape[1] != N:
                        if lv_np.shape[1] < N:
                            pad = _np.zeros((lv_np.shape[0], N - lv_np.shape[1]), dtype=_np.float32)
                            lv_np = _np.concatenate([lv_np, pad], axis=1)
                        else:
                            lv_np = lv_np[:, :N]
                else:
                    lv_np = lv_np.reshape(lv_np.shape[0], -1)
                    lv_np = lv_np[-1:].astype(_np.float32, copy=False)

        # final cast
        try:
            lv_np = _np.asarray(lv_np, dtype=_np.float32, copy=False)
        except Exception:
            lv_np = _np.zeros((1, N or 1), dtype=_np.float32)

        if lv_np.ndim == 1:
            lv_np = lv_np[None, :]

        # store on manager for fallback
        self._last_bootstrap_vals = lv_np

        # try forwarding to buffer
        buf = getattr(self, "buffer", None)
        if buf is not None:
            try_names = [
                "finish_rollouts", "finish_rollout", "finish_path",
                "compute_returns", "add_bootstrap_values", "set_bootstrap_values"
            ]
            for name in try_names:
                if hasattr(buf, name):
                    try:
                        func = getattr(buf, name)
                        try:
                            func(lv_np)
                        except Exception:
                            import torch as _torch
                            func(_torch.from_numpy(lv_np))
                        print(f"[INFO] finish_rollouts: forwarded last_vals to buffer.{name}")
                        return
                    except Exception as e:
                        print(f"[WARN] finish_rollouts: buffer.{name} raised: {e}")
                        continue

        print("[WARN] finish_rollouts: buffer had no known finish API; stored last_vals on manager for later use.")
        return

    def debug_policy_input_shapes(manager, sample_batch=None, use_buffer_sample=True):
        """
        Debug helper: inspect one minibatch and the policy's first Linear layers to locate input_size mismatch.

        Usage:
          - If called inside MAPPOManager methods, pass `self` as manager and optionally sample_batch.
          - If sample_batch is None and use_buffer_sample=True, this will attempt to pull one minibatch from manager.buffer.feed_forward_generator.
        What it prints:
          - agent_feats / images / actions shapes from the batch
          - flattened per-sample size computed from agent_feats (N * F)
          - the first several nn.Linear modules' in_features/out_features from policy
          - a traceback from a test forward/evaluate call (if it fails)
        """
        import traceback, torch, numpy as np

        print("\n[debug_policy_input_shapes] START")

        # 1) get a sample batch
        batch = sample_batch
        if batch is None and use_buffer_sample:
            try:
                gen = manager.buffer.feed_forward_generator(
                    mini_batch_size=max(1, getattr(manager, "debug_mini_batch_size", 64)))
                batch = next(gen)
                print("[debug] obtained batch from buffer.feed_forward_generator()")
            except Exception as e:
                print("[debug] failed to sample from buffer:", e)
                batch = None

        if batch is None:
            print(
                "[debug] No batch available (sample_batch=None and buffer sample failed). Provide a batch dict to the function.")
            return

        # 2) print batch keys and shapes
        def _maybe_tensor_to_shape(x):
            try:
                if isinstance(x, torch.Tensor):
                    return tuple(x.shape)
                elif isinstance(x, np.ndarray):
                    return tuple(x.shape)
                elif isinstance(x, list):
                    return f"list(len={len(x)})"
                else:
                    return type(x).__name__
            except Exception:
                return "unknown"

        print("[debug] batch keys:", list(batch.keys()))
        print(" - agent_feats shape:", _maybe_tensor_to_shape(batch.get("agent_feats", None)))
        print(" - images shape    :", _maybe_tensor_to_shape(batch.get("images", batch.get("image", None))))
        print(" - actions shape   :", _maybe_tensor_to_shape(batch.get("actions", None)))
        print(" - old_logp shape  :", _maybe_tensor_to_shape(batch.get("logp", batch.get("old_log_probs", None))))
        print(" - returns shape   :", _maybe_tensor_to_shape(batch.get("returns", None)))
        print(" - advantages shape:", _maybe_tensor_to_shape(batch.get("advantages", None)))

        # 3) compute flattened per-sample size from agent_feats if possible
        af = batch.get("agent_feats", None)
        if af is not None:
            try:
                if isinstance(af, torch.Tensor):
                    af_np = af.detach().cpu().numpy()
                elif isinstance(af, np.ndarray):
                    af_np = af
                else:
                    af_np = np.asarray(af)
                # possible shapes: [rows, N, F] or [B, N, F] or [N, F]
                if af_np.ndim == 3:
                    B, N, F = af_np.shape
                    per_sample_flat = N * F
                    print(f"[debug] agent_feats: B={B}, N={N}, F={F}, per-sample flattened size N*F={per_sample_flat}")
                elif af_np.ndim == 2:
                    N, F = af_np.shape
                    per_sample_flat = N * F
                    print(
                        f"[debug] agent_feats (no batch dim): N={N}, F={F}, per-sample flattened size N*F={per_sample_flat}")
                else:
                    print(f"[debug] agent_feats unexpected ndim={af_np.ndim}, shape={af_np.shape}")
                    per_sample_flat = af_np.size
            except Exception as e:
                print("[debug] failed to analyze agent_feats:", e)
                per_sample_flat = None
        else:
            per_sample_flat = None
            print("[debug] no agent_feats in batch")

        # 4) inspect policy's Linear layers to find candidate in_features (first few)
        try:
            import torch.nn as nn
            linear_layers = []
            for name, module in manager.policy.named_modules():
                if isinstance(module, nn.Linear):
                    linear_layers.append((name, module.in_features, module.out_features))
            if linear_layers:
                print("[debug] policy linear layers (name, in_features, out_features) first 8:")
                for i, (n, inf, outf) in enumerate(linear_layers[:8]):
                    print(f"   [{i}] {n}  in={inf} out={outf}")
            else:
                print("[debug] no nn.Linear found in policy modules (maybe using Conv/Custom heads).")
        except Exception as e:
            print("[debug] failed to inspect policy modules:", e)

        # 5) Try a test forward / evaluate call and capture traceback
        # Build a safe obs dict for forward call if needed
        try:
            obs = {}
            # prepare image if present
            img = batch.get("images", None)
            if img is None:
                img = batch.get("image", None)
            if img is not None:
                try:
                    if isinstance(img, np.ndarray):
                        arr = img
                        if arr.ndim == 5 and arr.shape[0] >= 1:  # [T,B,C,H,W] or [T,C,H,W], try to pick last
                            arr = arr[-1]
                        # ensure batch dim
                        if arr.ndim == 3:
                            arr = np.expand_dims(arr, 0)
                        obs["image"] = torch.as_tensor(arr, dtype=torch.float32,
                                                       device=next(manager.policy.parameters()).device)
                    elif isinstance(img, torch.Tensor):
                        obs["image"] = img.to(next(manager.policy.parameters()).device)
                except Exception as e:
                    print("[debug] failed to prepare image for obs:", e)

            # prepare agent_feats
            if af is not None:
                try:
                    if isinstance(af, np.ndarray):
                        # select last frame if 3D [rows,N,F]
                        if af.ndim == 3:
                            # try to coerce to [B,N,F] by viewing rows -> B*N etc.
                            # we'll just take first B chunk -> reshape if divisible
                            rows = af.shape[0]
                            N = af.shape[1]
                            if rows % N == 0:
                                B = rows // N
                                af_view = af.reshape(B, N, af.shape[2])
                                obs["agent_feats"] = torch.as_tensor(af_view, dtype=torch.float32,
                                                                     device=next(manager.policy.parameters()).device)
                            else:
                                # fallback: take the first sample interpretation
                                obs["agent_feats"] = torch.as_tensor(af[0:1], dtype=torch.float32,
                                                                     device=next(manager.policy.parameters()).device)
                        else:
                            obs["agent_feats"] = torch.as_tensor(af, dtype=torch.float32,
                                                                 device=next(manager.policy.parameters()).device)
                    elif isinstance(af, torch.Tensor):
                        if af.dim() == 3:
                            # try to interpret as [rows,N,F] -> attempt reshape
                            rows = af.shape[0]
                            N = af.shape[1]
                            if rows % N == 0:
                                B = rows // N
                                obs["agent_feats"] = af.view(B, N, af.shape[2]).to(
                                    next(manager.policy.parameters()).device)
                            else:
                                obs["agent_feats"] = af[0:1].to(next(manager.policy.parameters()).device)
                        else:
                            obs["agent_feats"] = af.to(next(manager.policy.parameters()).device)
                    else:
                        print("[debug] agent_feats not tensor/ndarray; skipping")
                except Exception as e:
                    print("[debug] failed to prepare agent_feats for obs:", e)

            # attempt forward/evaluate
            print("[debug] attempting policy.forward(obs=...) or evaluate_actions with prepared obs ...")
            try:
                if hasattr(manager.policy, "evaluate_actions"):
                    # some evaluate_actions expect (agent_feats, actions) positional signature;
                    # try dict style by calling evaluate_actions(obs, actions) as positional to avoid keyword mismatch
                    try:
                        test_out = manager.policy.evaluate_actions(obs, batch.get("actions", None))
                        print("[debug] evaluate_actions(obs, actions) -> success")
                    except Exception:
                        test_out = manager.policy.evaluate_actions(batch.get("agent_feats", None),
                                                                   batch.get("actions", None))
                        print("[debug] evaluate_actions(agent_feats, actions) -> success")
                else:
                    test_out = manager.policy.forward(obs=obs, mode="step", deterministic=True)
                    print("[debug] policy.forward(obs=...) -> success")
                print("[debug] test forward returned keys:",
                      list(test_out.keys()) if isinstance(test_out, dict) else type(test_out))
            except Exception as e:
                print("[debug] forward/evaluate raised exception; printing traceback:")
                traceback.print_exc()
        except Exception as e:
            print("[debug] unexpected error during test forward:", e)
            traceback.print_exc()

        print("[debug_policy_input_shapes] END\n")

    def _to_tensor(self, x, device, dtype=None):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.to(device)
        if dtype is None:
            dtype = torch.float32
        return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)

    def _reshape_rows(self, x, B, N, device):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            t = x.to(device)
            if t.dim() == 2 and t.shape[0] == B * N:
                return t.view(B, N, -1)
            if t.dim() == 1 and t.shape[0] == B * N:
                return t.view(B, N)
            return t
        arr = np.asarray(x)
        if arr.ndim == 2 and arr.shape[0] == B * N:
            return torch.as_tensor(arr.reshape(B, N, arr.shape[1]), dtype=torch.float32, device=device)
        if arr.ndim == 1 and arr.shape[0] == B * N:
            return torch.as_tensor(arr.reshape(B, N), dtype=torch.float32, device=device)
        return torch.as_tensor(arr, dtype=torch.float32, device=device)

    def normalize_minibatch_for_policy(self, batch):
        policy = self.policy
        device = next(policy.parameters()).device

        N_candidates = []
        try:
            specs = getattr(self, "agent_specs", None)
            if specs:
                for _, v in specs.items():
                    if isinstance(v, dict) and "n_agents" in v:
                        N_candidates.append(int(v["n_agents"]))
        except:
            pass
        if hasattr(self, "agent_slots") and getattr(self, "agent_slots"):
            N_candidates.append(len(self.agent_slots))
        for attr in ("max_slots", "slot_count", "n_agents"):
            if hasattr(self, attr):
                try:
                    N_candidates.append(int(getattr(self, attr)))
                except:
                    pass
        N_candidates.append(getattr(self.buffer, "N", 16) or 16)
        N_candidates = [n for n in N_candidates if n is not None and n > 0]

        af = batch.get("agent_feats", None)
        if af is None:
            raise RuntimeError("normalize_minibatch_for_policy: missing agent_feats")

        if isinstance(af, torch.Tensor):
            af_np = af.cpu().numpy()
        else:
            af_np = np.asarray(af)

        if af_np.ndim == 3:
            B, N, F = af_np.shape
            agent_feats = self._to_tensor(af_np, device)
        elif af_np.ndim == 2:
            rows, F = af_np.shape
            inferred_N = None
            for cand in N_candidates:
                if rows % cand == 0:
                    inferred_N = cand
                    break
            if inferred_N is None:
                inferred_N = rows
            N = int(inferred_N)
            B = rows // N
            agent_feats = self._to_tensor(af_np.reshape(B, N, F), device)
        else:
            raise RuntimeError(f"normalize_minibatch_for_policy: invalid agent_feats ndim={af_np.ndim}")

        img = batch.get("images", None)
        if img is None:
            img = batch.get("image", None)
        image_out = None
        if img is not None:
            if isinstance(img, torch.Tensor):
                it = img.to(device)
                if it.dim() == 4:
                    rows_i, C, H, W = it.shape
                    if rows_i == B * N:
                        it_resh = it.view(B, N, C, H, W)
                        image_out = it_resh.mean(dim=1)
                    else:
                        if rows_i % N == 0:
                            B_img = rows_i // N
                            it_resh = it.view(B_img, N, C, H, W)
                            image_out = it_resh.mean(dim=1)
                        else:
                            image_out = it[:B]
                elif it.dim() == 5:
                    image_out = it.mean(dim=1)
            else:
                img_np = np.asarray(img)
                if img_np.ndim == 4:
                    rows_i, C, H, W = img_np.shape
                    if rows_i == B * N:
                        resh = img_np.reshape(B, N, C, H, W)
                        image_out = torch.as_tensor(resh.mean(axis=1), dtype=torch.float32, device=device)
                    elif rows_i % N == 0:
                        B_img = rows_i // N
                        resh = img_np.reshape(B_img, N, C, H, W)
                        image_out = torch.as_tensor(resh.mean(axis=1), dtype=torch.float32, device=device)
                    else:
                        image_out = torch.as_tensor(img_np[:B], dtype=torch.float32, device=device)
                elif img_np.ndim == 5:
                    image_out = torch.as_tensor(img_np.mean(axis=1), dtype=torch.float32, device=device)
                else:
                    image_out = torch.as_tensor(img_np, dtype=torch.float32, device=device)

        actions = self._reshape_rows(batch.get("actions", None), B, N, device)
        old_logp = self._reshape_rows(batch.get("old_logp", batch.get("logp", None)), B, N, device)
        returns = self._reshape_rows(batch.get("returns", None), B, N, device)
        advantages = self._reshape_rows(batch.get("advantages", None), B, N, device)
        values = self._reshape_rows(batch.get("values", None), B, N, device)
        type_id = self._reshape_rows(batch.get("type_id", None), B, N, device)

        obs = {"agent_feats": agent_feats}
        if image_out is not None:
            obs["image"] = image_out
        if type_id is not None:
            obs["type_id"] = type_id

        other = {"returns": returns, "advantages": advantages, "values": values, "indices": batch.get("indices", None)}
        return obs, actions, old_logp, other

    def eval_policy_on_minibatch(self, batch):
        """
        Normalize minibatch and evaluate the policy on it.
        Returns a dict with evaluation outputs (log_probs, values, entropy, ...)
        """
        import torch

        obs, actions, old_logp, other = self.normalize_minibatch_for_policy(batch)
        policy = self.policy
        device = next(policy.parameters()).device

        if actions is not None and isinstance(actions, torch.Tensor):
            actions = actions.to(device)

        last_exc = None

        if hasattr(policy, "evaluate_actions"):
            try:
                out = policy.evaluate_actions(obs, actions)
                if isinstance(out, dict):
                    return out
            except Exception as e:
                last_exc = e
            try:
                out = policy.evaluate_actions(obs["agent_feats"], actions)
                if isinstance(out, dict):
                    return out
            except Exception as e:
                last_exc = e

        if hasattr(policy, "forward"):
            try:
                out = policy.forward(obs=obs, mode="step", deterministic=True)
                if isinstance(out, dict):
                    return out
            except Exception as e:
                last_exc = e

        raise RuntimeError(f"eval_policy_on_minibatch failed; last_exc={repr(last_exc)}")

    # ------------- save / load -------------
    def save_all(self, prefix: str) -> None:
        for slot, agent in self.agents.items():
            try:
                agent.save(f"{prefix}_{slot}")
            except Exception:
                pass

    def load_all(self, prefix: str, map_location=None) -> None:
        for slot, agent in self.agents.items():
            try:
                agent.load(f"{prefix}_{slot}", map_location=map_location)
            except Exception:
                pass
