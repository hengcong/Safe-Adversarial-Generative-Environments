from collections import OrderedDict
from typing import Dict, Any, Optional, Iterable
import torch
import numpy as np
import copy

import sys
import traceback

from sympy.assumptions.lra_satask import WHITE_LIST

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
            self.optim = torch.optim.Adam(params, lr=5e-5, eps=1e-5) if params else None
        except Exception:
            self.optim = None

    # ------------- Rollout Buffer helpers (single buffer) -------------
    def init_rollout_buffer(self, T: int = 128, image_shape=None, device: str = "cpu"):
        """
        Create single multi-agent RolloutBuffer for the whole environment.
        """
        B = 1
        N = len(self.agent_slots)
        feat_dim = getattr(self, "obs_dim", 3)
        act_dim = 2
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

    def store_step(self, imgs, feats, types, acts, logp, vals, rews, masks, pre_t_np=None, slot_hidden=None, bev_hidden=None):

        if self.buffer is None:
            raise RuntimeError("Buffer not initialized")

        if pre_t_np is None:
            raise RuntimeError("pre_t_np must be provided from rollout")

        # 统一转 numpy（buffer内部也是numpy）
        def _to_numpy(x):
            if x is None:
                return None
            if isinstance(x, np.ndarray):
                return x
            if torch.is_tensor(x):
                return x.detach().cpu().numpy()
            return np.asarray(x)

        imgs_np = _to_numpy(imgs)
        feats_np = _to_numpy(feats)
        types_np = _to_numpy(types)
        acts_np = _to_numpy(acts)
        logp_np = _to_numpy(logp)
        vals_np = _to_numpy(vals)
        rews_np = _to_numpy(rews)
        masks_np = _to_numpy(masks)
        pre_t_np = _to_numpy(pre_t_np)

        self.buffer.add_batch(
            imgs_np,
            feats_np,
            types_np,
            acts_np,
            logp_np,
            vals_np,
            rews_np,
            masks_np,
            pre_t_np,
            slot_hidden=slot_hidden,
            bev_hidden=bev_hidden
        )

        return True

    def store_transitions(self,
                          obs_dict=None,
                          act_dict=None,
                          rew_dict=None,
                          done_dict=None,
                          val_dict=None,
                          logp_dict=None,
                          info_dict=None,
                          hidden_dict=None):

        if self.buffer is None:
            raise RuntimeError("Buffer not initialized")

        slots = getattr(self, "agent_slots", list(act_dict.keys()))

        imgs_arr = None
        if obs_dict and "image" in obs_dict:
            im = obs_dict["image"]
            imgs_arr = im.detach().cpu().numpy() if torch.is_tensor(im) else np.asarray(im)

        if imgs_arr is None:
            imgs_arr = np.zeros((1, 3, 84, 84), dtype=np.float32)

        if imgs_arr.ndim == 3:
            imgs_arr = imgs_arr[None, ...]

        feats = obs_dict.get("agent_feats", None)
        feats_np = feats.detach().cpu().numpy() if torch.is_tensor(feats) else np.asarray(feats)
        if feats_np.ndim == 2:
            feats_np = feats_np[None, ...]

        types = obs_dict.get("type_id", None)
        types_np = types.detach().cpu().numpy() if torch.is_tensor(types) else np.asarray(types)
        if types_np.ndim == 1:
            types_np = types_np[None, ...]

        masks = obs_dict.get("mask", None)
        masks_np = masks.detach().cpu().numpy() if torch.is_tensor(masks) else np.asarray(masks)
        if masks_np.ndim == 1:
            masks_np = masks_np[None, ...]

        acts_list = []
        for idx, s in enumerate(slots):
            pre_t = np.asarray(logp_dict[s]["pre_t"], dtype=np.float32)

            action_scale = self.policy.action_scale
            if isinstance(action_scale, torch.Tensor):
                action_scale = action_scale.detach().cpu().numpy()

            scaled_action = np.tanh(pre_t) * action_scale
            acts_list.append(scaled_action)

        acts_np = np.asarray(acts_list, dtype=np.float32)[None, :, :]

        lps = []
        prets = []

        for s in slots:
            entry = logp_dict.get(s, None)
            if isinstance(entry, dict):
                lps.append(float(entry["logp"]))
                prets.append(np.asarray(entry["pre_t"], dtype=np.float32))
            else:
                raise RuntimeError("logp_dict entry must be dict")

        logp_np = np.asarray(lps, dtype=np.float32)[None, :]
        pre_t_np = np.asarray(prets, dtype=np.float32)[None, :, :]

        vals = []
        for s in slots:
            vals.append(float(val_dict.get(s, 0.0)))
        vals_np = np.asarray(vals, dtype=np.float32)[None, :]

        rews = []
        for s in slots:
            rews.append(float(rew_dict.get(s, 0.0)))
        rews_np = np.asarray(rews, dtype=np.float32)[None, :]

        slot_hidden_np = None
        bev_hidden_np = None

        if hidden_dict is not None:
            # 1. Handle Slot Hidden State
            if "slot" in hidden_dict:
                h_slots = hidden_dict["slot"]

                # Case A: h_slots is already a Tensor or ndarray [B, N, H] or [N, H]
                if torch.is_tensor(h_slots) or isinstance(h_slots, np.ndarray):
                    slot_hidden_np = _ensure_numpy(h_slots)
                    # Ensure shape is [1, N, H] for batch size 1
                    if slot_hidden_np.ndim == 2:
                        slot_hidden_np = slot_hidden_np[None, :, :]

                # Case B: h_slots is a dictionary {slot_id: tensor/array}
                elif isinstance(h_slots, dict):
                    h_list = []
                    for s in slots:
                        h = h_slots.get(s, None)
                        if h is not None:
                            h_list.append(_ensure_numpy(h))
                        else:
                            # Padding with zeros if slot is missing
                            h_list.append(np.zeros(64, dtype=np.float32))
                    if h_list:
                        slot_hidden_np = np.asarray(h_list, dtype=np.float32)[None, :, :]

            # 2. Handle BEV Hidden State
            if "bev" in hidden_dict:
                h_bev = hidden_dict["bev"]
                if h_bev is not None:
                    h_bev_np = _ensure_numpy(h_bev)
                    # Ensure shape is [1, H]
                    if h_bev_np.ndim == 1:
                        h_bev_np = h_bev_np[None, :]
                    bev_hidden_np = h_bev_np

        return self.store_step(
            imgs_arr,
            feats_np,
            types_np,
            acts_np,
            logp_np,
            vals_np,
            rews_np,
            masks_np,
            pre_t_np=pre_t_np,
            slot_hidden=slot_hidden_np,
            bev_hidden=bev_hidden_np
        )

    def update_all(self,
                   ppo_epochs=2,
                   clip_coef=0.2,
                   value_coef=0.1,
                   entropy_coef=0.01,
                   max_grad_norm=0.5,
                   num_mini_batches=4):

        if self.buffer is None:
            return {}

        params = [p for p in self.policy.parameters() if p.requires_grad]
        if not params:
            return {}

        optim = self.optim
        device = next(self.policy.parameters()).device

        policy_loss_epoch = 0.0
        value_loss_epoch = 0.0
        entropy_epoch = 0.0
        kl_epoch = 0.0
        clip_frac_epoch = 0.0
        ev_epoch = 0.0
        num_updates = 0

        for _ in range(ppo_epochs):
            for batch in self.buffer.feed_forward_generator(num_mini_batches=num_mini_batches):

                images = batch.get("images", None)
                agent_feats = batch["agent_feats"]
                type_id = batch["type_id"]
                actions = batch["actions"]
                old_logp = batch["logp"]
                returns = batch["returns"]
                advantages = batch["advantages"].to(device)
                with torch.no_grad():
                    # Use masks to only normalize valid transitions if necessary,
                    # but standard batch normalization is usually sufficient:
                    adv_mean = advantages.mean()
                    adv_std = advantages.std()
                    advantages = (advantages - adv_mean) / (adv_std + 1e-8)

                masks = batch["masks"]

                init_slot = batch.get("init_slot_hidden", None)
                init_bev = batch.get("init_bev_hidden", None)

                # move to device
                if images is not None:
                    images = images.to(device)
                agent_feats = agent_feats.to(device)
                type_id = type_id.to(device)
                actions = actions.to(device)
                old_logp = old_logp.to(device)
                returns = returns.to(device)
                advantages = advantages.to(device)
                masks = masks.to(device)
                if init_slot is not None:
                    init_slot = init_slot.to(device)
                if init_bev is not None:
                    init_bev = init_bev.to(device)
                # --- safe handling of optional pre_t from buffer ---
                pre_t_batch = batch.get("pre_t", None)
                if pre_t_batch is not None:
                    # ensure it's a tensor on device/dtype
                    pre_t_batch = torch.as_tensor(pre_t_batch, device=device, dtype=torch.float32)
                else:
                    pre_t_batch = None

                # action_scale from policy (make tensor on device)
                action_scale = getattr(self.policy, "action_scale", 1.0)
                if not torch.is_tensor(action_scale):
                    action_scale_t = torch.as_tensor(action_scale, device=device, dtype=torch.float32)
                else:
                    action_scale_t = action_scale.to(device=device, dtype=torch.float32)

                # If buffer unexpectedly stored pre_t instead of executed (scaled) actions,
                # try to reconstruct actions from pre_t when actions look like "empty/unscaled".
                # Heuristic: if actions max abs is small (< 1e-6) OR all values within [-1,1] but policy scale >1,
                # then prefer reconstructed actions. This is conservative and won't clobber proper actions.
                try:
                    # ensure actions is tensor on device (already moved above)
                    if pre_t_batch is not None:
                        # compute candidate actions from pre_t
                        cand_actions = torch.tanh(pre_t_batch) * action_scale_t.view(1, 1, 1,
                                                                                     -1) if pre_t_batch.dim() == 4 else torch.tanh(
                            pre_t_batch) * action_scale_t
                        # conservative replacement condition:
                        # replace if actions looks like placeholder (near zero) OR if cand_actions max > actions max by a factor
                        try:
                            act_max = float(actions.abs().max().item())
                        except Exception:
                            act_max = 0.0
                        try:
                            cand_max = float(cand_actions.abs().max().item())
                        except Exception:
                            cand_max = 0.0

                        if act_max < 1e-6 and cand_max > 1e-6:
                            actions = cand_actions
                            print("[WARN] Reconstructed actions from pre_t_batch (buffer likely stored pre_t).")
                        # else do not overwrite actions; assume buffer stored scaled actions already
                except Exception as e:
                    print("[WARN] pre_t handling failed:", e)

                # --- Always compute new_logp/values/entropy via evaluate_actions (do not depend on pre_t presence) ---
                # prepare obs dict (already moved to device above)
                obs_eval = {
                    "image": images,
                    "agent_feats": agent_feats,
                    "type_id": type_id
                }

                # sanity: ensure actions are on device and float
                actions = actions.to(device=device, dtype=torch.float32)

                # ensure masks dtype/device
                masks = masks.to(device=device, dtype=torch.float32)
                eval_out = self.policy.evaluate_actions(
                    obs={
                        "image": images,
                        "agent_feats": agent_feats,
                        "type_id": type_id
                    },
                    actions=actions,
                    mask=masks,
                    slot_hidden=init_slot,
                    bev_hidden=init_bev,
                    mode="seq",
                    pre_t=pre_t_batch
                )

                new_logp = eval_out["log_probs"]
                values = eval_out["values"]
                entropy = eval_out["entropy"]

                # safety: shapes should match; if not, try common permute (but also log warning)
                if new_logp.shape != old_logp.shape:
                    try:
                        if new_logp.dim() == 3 and old_logp.dim() == 3 and new_logp.numel() == old_logp.numel():
                            new_logp = new_logp.permute(1, 0, 2).contiguous()
                            print("[WARN] permuted new_logp to match old_logp shapes")
                        else:
                            print(f"[WARN] shape mismatch new_logp {tuple(new_logp.shape)} vs old_logp {tuple(old_logp.shape)}")
                    except Exception as e:
                        print("Shape align permute failed:", e)

                diff = new_logp - old_logp
                denom = masks.sum() + 1e-8

                # ===== diagnostics (single place, guarded) =====
                with torch.no_grad():
                    old_lp = old_logp.to(new_logp.device)
                    new_lp = new_logp
                    print("DIAG SHAPES: old_logp", tuple(old_lp.shape), "new_logp", tuple(new_lp.shape))
                    print("DIAG STATS: old mean,std",
                          float(old_lp.mean().item()), float(old_lp.std().item()),
                          "new mean,std", float(new_lp.mean().item()), float(new_lp.std().item()))
                    d = (new_lp - old_lp)
                    print("DIFF mean,std,abs_mean,max:",
                          float(d.mean().item()), float(d.std().item()), float(d.abs().mean().item()), float(d.abs().max().item()))
                    thr = 3.0
                    frac_big = (d.abs() > thr).float().mean().item() * 100.0
                    print(f"FRAC abs(diff)>{thr}: {frac_big:.2f}%")
                    try:
                        old_flat = old_lp.flatten(); new_flat = new_lp.flatten()
                        if old_flat.numel() > 10:
                            vx = old_flat - old_flat.mean(); vy = new_flat - new_flat.mean()
                            corr = (vx * vy).sum() / (torch.sqrt((vx * vx).sum() * (vy * vy).sum()) + 1e-8)
                            print("Pearson corr old_vs_new:", float(corr.item()))
                    except Exception as e:
                        print("Pearson corr failed:", e)

                    # safe print of a few per-sample diffs
                    try:
                        flat_d = d.flatten()
                        nshow = min(10, flat_d.numel())
                        vals, idxs = flat_d.abs().topk(nshow, largest=True)
                        print("Top diffs (abs) idx,val:", [(int(i.item()), float(v.item())) for i, v in zip(idxs, vals)])
                        print("First 10 diffs:", [float(x.item()) for x in flat_d[:nshow]])
                    except Exception:
                        pass
                # ===== end diagnostics =====


                ratio = torch.exp(diff)
                # DEBUG: per-batch diagnostics (only print first few updates)
                if num_updates < 2:
                    with torch.no_grad():
                        print("===== DIAG BATCH =====")
                        print("advantages mean,std:", float(advantages.mean().item()), float(advantages.std().item()))
                        print("returns mean,std:", float(returns.mean().item()), float(returns.std().item()))
                        print("values mean,std:", float(values.mean().item()), float(values.std().item()))
                        try:
                            print("actions mean,std:", float(actions.mean().item()), float(actions.std().item()))
                        except Exception:
                            pass
                        print("old_logp mean,std:", float(old_logp.mean().item()), float(old_logp.std().item()))
                        if pre_t_batch is not None:
                            try:
                                print("pre_t mean,std:", float(pre_t_batch.mean().item()),
                                      float(pre_t_batch.std().item()))
                            except Exception:
                                pass
                        else:
                            print("pre_t: None")

                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * advantages

                policy_loss = -((torch.min(surr1, surr2)) * masks).sum() / denom
                value_loss = (((returns - values) ** 2) * masks).sum() / denom
                entropy_loss = (entropy * masks).sum() / denom

                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_loss

                with torch.no_grad():
                    approx_kl = ((old_logp - new_logp) * masks).sum() / denom
                    clipped = ((ratio > 1.0 + clip_coef) | (ratio < 1.0 - clip_coef)).float()
                    clip_fraction = (clipped * masks).sum() / denom

                    var_y = torch.var((returns * masks))
                    explained_variance = 1 - torch.var(((returns - values) * masks)) / (var_y + 1e-8)
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                optim.step()
                policy_loss_epoch += float(policy_loss.item())
                value_loss_epoch += float(value_loss.item())
                entropy_epoch += float(entropy_loss.item())
                kl_epoch += float(approx_kl.item())
                clip_frac_epoch += float(clip_fraction.item())
                ev_epoch += float(explained_variance.item())
                num_updates += 1

        # clear buffer after update
        self.buffer.clear()

        if num_updates == 0:
            return {}

        stats = {
            "policy_loss": policy_loss_epoch / num_updates,
            "value_loss": value_loss_epoch / num_updates,
            "entropy": entropy_epoch / num_updates,
            "approx_kl": kl_epoch / num_updates,
            "clip_fraction": clip_frac_epoch / num_updates,
            "explained_variance": ev_epoch / num_updates
        }

        return stats

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
                                          deterministic=False)
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

    # Add these helper methods to your manager class (same class that has select_actions)

    def _to_tensor(self, x, dtype=None):
        import torch

        if x is None:
            return None

        if not torch.is_tensor(x):
            x = torch.as_tensor(x)

        if dtype is not None:
            x = x.to(dtype)

        return x.to(self.device)

    def _normalize_slot_hidden(self, sh, B_local, N_local):
        """Ensure slot_hidden -> [B, N, H] or None."""
        if sh is None:
            return None
        t = self._to_tensor(sh, dtype=torch.float32)
        if t is None:
            return None
        if t.dim() == 4:
            last = t[-1]
            t = last
        if t.dim() == 3:
            if t.shape[0] != B_local:
                if t.shape[0] > B_local:
                    t = t[-B_local:]
                else:
                    reps = (B_local // t.shape[0]) + 1
                    t = t.repeat(reps, 1, 1)[:B_local]
            if t.shape[1] != N_local:
                if t.shape[1] == 1:
                    t = t.repeat(1, N_local, 1)
                else:
                    minN = min(t.shape[1], N_local)
                    t = t[:, :minN, :].contiguous()
                    if minN < N_local:
                        pad = torch.zeros((B_local, N_local - minN, t.shape[2]), dtype=t.dtype, device=t.device)
                        t = torch.cat([t, pad], dim=1)
            return t.contiguous().to(self.device)
        if t.dim() == 2:
            if t.shape[0] == B_local:
                return t.unsqueeze(1).repeat(1, N_local, 1).contiguous().to(self.device)
            if t.shape[0] == N_local:
                return t.unsqueeze(0).contiguous().to(self.device)
            last = t[-1:].contiguous()
            return last.unsqueeze(1).repeat(B_local, N_local, 1).contiguous().to(self.device)
        if t.dim() == 1:
            return t.unsqueeze(0).unsqueeze(1).repeat(B_local, N_local, 1).contiguous().to(self.device)
        return None

    def _normalize_bev_hidden(self, bh, B_local):
        """Ensure bev_hidden -> [B, H] or zeros if None."""
        if bh is None:
            try:
                hdim = int(self.policy.bev_enc.bev_gru.hidden_size)
            except Exception:
                hdim = getattr(self.policy, "bev_h_dim", 256)
            return torch.zeros((B_local, hdim), dtype=torch.float32, device=self.device)
        t = self._to_tensor(bh, dtype=torch.float32)
        if t is None:
            try:
                hdim = int(self.policy.bev_enc.bev_gru.hidden_size)
            except Exception:
                hdim = getattr(self.policy, "bev_h_dim", 256)
            return torch.zeros((B_local, hdim), dtype=torch.float32, device=self.device)
        if t.dim() == 3:
            last = t[-1]
            t = last
        if t.dim() == 2:
            if t.shape[0] != B_local:
                if t.shape[0] > B_local:
                    t = t[-B_local:]
                else:
                    reps = (B_local // t.shape[0]) + 1
                    t = t.repeat(reps, 1)[:B_local]
            return t.contiguous().to(self.device)
        if t.dim() == 1:
            return t.unsqueeze(0).contiguous().to(self.device)
        try:
            hdim = int(self.policy.bev_enc.bev_gru.hidden_size)
        except Exception:
            hdim = getattr(self.policy, "bev_h_dim", 256)
        import torch as _torch
        return _torch.zeros((B_local, hdim), dtype=_torch.float32, device=self.device)

    def select_actions(self, obs_dict, hidden=None, mask=None):
        slots = getattr(self, "agent_slots", list(self.agents.keys()))
        device = self.device if isinstance(self.device, torch.device) else torch.device(self.device)

        # empty fallback
        if not obs_dict:
            actions = {}
            values = {}
            logps = {}
            for slot in slots:
                agent = self.agents.get(slot, None)
                act_dim = getattr(agent, "action_dim", getattr(agent, "act_dim", 2)) if agent is not None else 2
                actions[slot] = np.zeros(act_dim, dtype=np.float32)
                values[slot] = 0.0
                logps[slot] = {"logp": -1e8, "pre_t": np.zeros((act_dim,), dtype=np.float32)}
            return actions, values, logps, None, None, {}

        # detect per-slot or global obs
        is_per_slot = any(k in slots for k in obs_dict.keys())

        # extract image / agent_feats / type_id / mask
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

        # image to tensor [B, seq, C, H, W]
        img_t = self._to_tensor(image_raw)
        if img_t is None:
            bev_ch = int(getattr(self, "bev_ch", 3))
            bev_H = int(getattr(self, "bev_H", 84))
            bev_W = int(getattr(self, "bev_W", 84))
            image_tensor = torch.zeros((1, 1, bev_ch, bev_H, bev_W), dtype=torch.float32, device=device)
        else:
            if img_t.dim() == 5:
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

        obs_dim = getattr(self, "obs_dim", 3)

        if is_per_slot:
            per_feats = []
            per_tids = []
            per_masks = []
            for slot in slots:
                slot_o = obs_dict.get(slot, {})
                af = self._to_tensor(slot_o.get("agent_feats", None))
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

                tid = self._to_tensor(slot_o.get("type_id", None), dtype=torch.long)
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

                m = self._to_tensor(slot_o.get("mask", None))
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
            aft = self._to_tensor(agent_feats_raw)
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

            tidt = self._to_tensor(type_id_raw, dtype=torch.long)
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

            mtt = self._to_tensor(mask_raw)
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

        # normalize hidden
        slot_hidden = None
        bev_hidden = None
        if hidden is not None:
            if isinstance(hidden, dict):
                slot_hidden = self._normalize_slot_hidden(hidden.get("slot", None), B, N)
                bev_hidden = self._normalize_bev_hidden(hidden.get("bev", None), B)
            else:
                slot_hidden = self._normalize_slot_hidden(hidden, B, N)

        # policy forward
        actions_t, values_t, logps_t, next_slot_h, next_bev_h, policy_info = self.policy.select_action(
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

        B_out, N_out, A_out = int(out_actions.shape[0]), int(out_actions.shape[1]), int(out_actions.shape[2])

        if len(slots) != N_out:
            minN = min(len(slots), N_out)
            print(
                f"[select_actions] WARNING: slot count mismatch manager:{len(slots)} vs actions:{N_out}; using min={minN}")
            slots = slots[:minN]
            out_actions = out_actions[:, :minN, :]

        def _norm_tensor_to_2d(t):
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

        vals_tensor = _norm_tensor_to_2d(values_t)
        lps_tensor = _norm_tensor_to_2d(logps_t)

        # extract pre_t if provided in policy_info; else try to reconstruct from policy_info['pre_t']
        pre_t_tensor = None
        if isinstance(policy_info, dict) and "pre_t" in policy_info:
            pre_t_tensor = policy_info["pre_t"]
            if torch.is_tensor(pre_t_tensor):
                pre_t_tensor = pre_t_tensor.to(device)
        # if pre_t_tensor has time dim, squeeze to last step if needed
        if pre_t_tensor is not None:
            while pre_t_tensor.dim() > 3 and pre_t_tensor.shape[0] == 1:
                pre_t_tensor = pre_t_tensor.squeeze(0)
            # now pre_t_tensor should be [B, N, A] or [N, A]

        actions = {}
        values = {}
        logps = {}
        for i, slot in enumerate(slots):
            act_b = out_actions[:, i, :]
            actions[slot] = act_b[0].detach().cpu().numpy().astype(np.float32)

            if vals_tensor is not None:
                try:
                    v = float(vals_tensor[0, i].detach().cpu().item())
                except Exception:
                    v = float(vals_tensor[0, i].detach().cpu().numpy().reshape(-1)[0])
            else:
                v = 0.0
            values[slot] = v

            # build logp item as a dict with both numeric logp and pre_t (numpy)
            if lps_tensor is not None:
                try:
                    lp_val = float(lps_tensor[0, i].detach().cpu().item())
                except Exception:
                    lp_val = float(lps_tensor[0, i].detach().cpu().numpy().reshape(-1)[0])
            else:
                lp_val = -1e8

            # get pre_t per-slot
            if pre_t_tensor is not None:
                try:
                    pre_t_slot = pre_t_tensor[0, i, :].detach().cpu().numpy().astype(np.float32)
                except Exception:
                    # fallback: zero vector of action dim
                    pre_t_slot = np.zeros((A_out,), dtype=np.float32)
            else:
                # if policy_info had no pre_t, leave zeros (buffer will still get numeric logp)
                pre_t_slot = np.zeros((A_out,), dtype=np.float32)

            logps[slot] = {"logp": lp_val, "pre_t": pre_t_slot}

        info = policy_info if isinstance(policy_info, dict) else {}

        # try policy-level entropy helpers
        try:
            if "entropy" not in info and hasattr(self.policy, "get_entropy"):
                ent = self.policy.get_entropy()
                if torch.is_tensor(ent):
                    info["entropy"] = float(ent.mean().detach().cpu().item())
                else:
                    info["entropy"] = float(ent)
        except Exception:
            pass
        try:
            if "entropy" not in info and hasattr(self.policy, "last_entropy"):
                le = getattr(self.policy, "last_entropy")
                if torch.is_tensor(le):
                    info["entropy"] = float(le.mean().detach().cpu().item())
                else:
                    info["entropy"] = float(le)
        except Exception:
            pass

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
                next_slot = next_slot[-1]  # -> [B,N,H]
            elif next_slot.dim() == 3:  # [T,N,H]  (B was squeezed)
                next_slot = next_slot[-1].unsqueeze(0)  # -> [1,N,H]

        if isinstance(next_bev, torch.Tensor):
            if next_bev.dim() == 3:  # [T,B,D]
                next_bev = next_bev[-1]
            elif next_bev.dim() == 2:  # [T,D] (B=1)
                next_bev = next_bev[-1].unsqueeze(0)

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
                "compute_returns", "add_bootstrap_values", "set_bootstrap_values", "finish_paths"
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
