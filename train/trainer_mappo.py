import os
import torch
import numpy as np

from typing import Optional, Dict, Any, Tuple
from .trainer import Trainer
from torch.utils.tensorboard import SummaryWriter

class MAPPOTrainer(Trainer):
    """
    MAPPOTrainer implements MAPPO-specific collection and update logic.
    It expects `manager` to implement:
      - mappo_reset(init_obs)
      - select_actions(obs_dict, hidden=...) -> (actions, values, logps, next_slot_h, next_bev_h, info)
      - store_transitions(obs, actions, rewards, dones, values, logps)
      - finish_rollouts(last_value_dict)
      - update_all()
      - get_initial_hidden(batch_size, n_agent)
      - optionally compute_value_for_last_obs() or manager.agents[...] .get_value(...)
    And envs to implement:
      - reset()
      - either vectorized step(actions_batch) or step(env_idx, actions)
      - optionally get_obs_all() to fetch current obs for all envs.
    """

    def __init__(self,
                 envs,
                 manager,
                 num_steps: int = 256,
                 device: str = "cpu",
                 **kwargs):
        super().__init__(envs=envs, manager=manager, device=device, **kwargs)
        self.T = int(num_steps)
        self.tb = SummaryWriter(log_dir=envs.experiment_path)
        self.global_ep = 0
        self._coll_hist = []
        # number of parallel environments
        if hasattr(envs, "num_envs"):
            self.num_envs = envs.num_envs
        else:
            self.num_envs = 1

        self.best_return = float("-inf")

        self.ckpt_dir = os.path.join(envs.experiment_path, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        if self.num_envs is None:
            # try len()
            try:
                self.num_envs = len(envs)
            except Exception:
                self.num_envs = 1

        # initialize envs and manager
        init_obs = None #self.envs.reset()
        # map manager with initial observations
        if hasattr(self.manager, "mappo_reset"):
            try:
                self.manager.mappo_reset(init_obs)
            except Exception:
                # fallback: call without args
                self.manager.mappo_reset({})

        self.hidden_per_env = None
        if hasattr(self.manager, "get_initial_hidden"):
            # infer n_agents from manager.agent_slots (most reliable)
            n_agents = len(getattr(self.manager, "agent_slots", [])) or None
            if n_agents is not None and n_agents > 0:
                # manager.get_initial_hidden(batch_size=num_envs, n_agent=n_agents)
                slot_h, bev_h = self.manager.get_initial_hidden(self.num_envs, n_agents)

                # --- normalize types & device ---
                # allow manager to return numpy arrays or torch tensors; convert to torch and move to trainer device
                def _to_tensor(x):
                    if x is None:
                        return None
                    # numpy -> tensor
                    if isinstance(x, np.ndarray):
                        try:
                            return torch.from_numpy(x).float().to(self.device)
                        except Exception:
                            return torch.tensor(x, dtype=torch.float32, device=self.device)
                    # torch -> correct device & dtype
                    if torch.is_tensor(x):
                        return x.float().to(self.device)
                    # fallback: try to build tensor (lists, scalars)
                    try:
                        return torch.tensor(x, dtype=torch.float32, device=self.device)
                    except Exception:
                        return None

                # optional sanity check: shapes
                # slot_h expected shape: [num_envs, n_agents, hidden_dim] or None
                if slot_h is not None:
                    assert slot_h.shape[0] == self.num_envs, f"slot_h batch dim {slot_h.shape[0]} != num_envs {self.num_envs}"
                    assert slot_h.shape[1] == n_agents, f"slot_h agent dim {slot_h.shape[1]} != n_agents {n_agents}"

                self.hidden_per_env = {"slot": slot_h, "bev": bev_h}
            else:
                self.hidden_per_env = None

        # keep last observations for potential bootstrap
        self._last_obs_all = init_obs

    # ---------------------- helpers ----------------------
    def _get_obs_all(self) -> Any:
        """Return list/dict of current observations for all envs. Adapt if your env uses different API."""
        if hasattr(self.envs, "get_obs_all"):
            return self.envs.get_obs_all()
        if hasattr(self.envs, "get_last_obs_all"):
            return self.envs.get_last_obs_all()
        # fallback: if envs is vectorized and supports single reset/state
        try:
            # maybe envs.reset() returned structured obs; keep what's stored
            return self._last_obs_all
        except Exception:
            return None

    def _step_env(self, env_idx: int, actions: Dict[Any, Any]) -> Tuple[Any, Dict[Any, float], Dict[Any, bool], Dict]:
        # If envs implements per-env step signature: step(env_idx, actions)
        if hasattr(self.envs, "step") and self.num_envs is not None and self.num_envs > 1:
            return self.envs.step(env_idx, actions)

        # single-env case: call env.step(actions)
        if hasattr(self.envs, "step") and (self.num_envs is None or self.num_envs == 1):
            # when envs is the single env instance, step(actions) -> next_obs, rewards, dones, info
            return self.envs.step(actions)

        # fallback: existing behavior (may raise)
        raise RuntimeError("envs.step(env_idx, actions) not found. Adapt MAPPOTrainer._step_env to your envs API.")

    def collect_rollout(self):

        env = self.envs
        manager = self.manager
        T = self.T

        # =======================
        # 1. RESET & INIT
        # =======================
        seq = env.reset()
        seq_dict = seq.get("seq", {})

        obs_seq = {}

        if "images" in seq_dict and seq_dict["images"] is not None:
            imgs = np.asarray(seq_dict["images"])
            if imgs.ndim == 4 and imgs.shape[-1] in (1, 3, 4):
                imgs = np.transpose(imgs, (0, 3, 1, 2))
            obs_seq["image"] = torch.tensor(imgs, dtype=torch.float32).unsqueeze(1)

        if "agent_feats" in seq_dict and seq_dict["agent_feats"] is not None:
            af = torch.tensor(seq_dict["agent_feats"], dtype=torch.float32)
            if af.ndim == 3:
                af = af.unsqueeze(1)
            obs_seq["agent_feats"] = af

        if "type_id" in seq_dict and seq_dict["type_id"] is not None:
            tid = torch.tensor(seq_dict["type_id"], dtype=torch.long)
            if tid.ndim == 2:
                tid = tid.unsqueeze(1)
            obs_seq["type_id"] = tid

        if "mask" in seq_dict and seq_dict["mask"] is not None:
            m = torch.tensor(seq_dict["mask"], dtype=torch.float32)
            if m.ndim == 2:
                m = m.unsqueeze(1)
            obs_seq["mask"] = m

        # 2. Burn-in to get the initial hidden state for this new episode
        # This function correctly extracts the last hidden state from the init sequence
        hidden_slot, hidden_bev = manager.burn_in(obs_seq, detach=True)

        # Here 'hidden' represents the memory at t=0
        hidden = {"slot": hidden_slot, "bev": hidden_bev}

        obs = env.get_single_obs_for_manager()

        # 3. Reset buffer and store the INITIAL hidden state
        manager.reset_buffer()
        manager.buffer.set_initial_hidden(hidden_slot, hidden_bev)

        # 4. Start the rollout loop
        ep_len = 0
        ep_return = 0.0
        all_rewards = []
        last_env_info = {}
        for t in range(T):
            ep_len += 1
            actions, values, logps, next_slot_h, next_bev_h, policy_info = manager.select_actions(
                obs_dict=obs,
                hidden=hidden,
                mask=obs.get("mask", None)
            )
            next_obs, rewards_dict, dones_dict, env_info = env.step(actions)

            # --- [LOGGING UPDATE] KEEP THIS! ---
            last_env_info = env_info
            step_reward = np.mean(list(rewards_dict.values()))
            ep_return += step_reward
            all_rewards.append(step_reward)

            manager.store_transitions(
                obs_dict=obs,
                act_dict=actions,
                rew_dict=rewards_dict,
                done_dict=dones_dict,
                val_dict=values,
                logp_dict=logps,
                info_dict=policy_info,
                hidden_dict=hidden
            )
            hidden = {"slot": next_slot_h, "bev": next_bev_h}
            obs = next_obs

            if any(dones_dict.values()):
                break

        # =======================
        # 3. BOOTSTRAP
        # =======================
        try:
            # Bootstrap value using the last hidden state
            actions_tmp, values_tmp, _, _, _, _ = manager.select_actions(
                obs_dict=obs,
                hidden=hidden,  # Using the hidden state of the last step
                mask=obs.get("mask", None)
            )
            slot_list = manager.agent_slots
            last_vals_arr = np.array(
                [values_tmp.get(s, 0.0) for s in slot_list],
                dtype=np.float32
            )[None, :]
        except Exception:
            slot_list = manager.agent_slots
            last_vals_arr = np.zeros((1, len(slot_list)), dtype=np.float32)

        manager.finish_rollouts(last_vals_arr)

        # =======================
        # 4. UPDATE
        # =======================
        stats = manager.update_all()

        if stats is not None:
            for k, v in stats.items():
                self.tb.add_scalar(f"train/{k}", v, self.global_step)

        if hasattr(self, "tb"):
            self.tb.add_scalar("train/ep_return", ep_return, self.global_step)
            self.tb.add_scalar("train/ep_len", ep_len, self.global_step)

        self.global_step += 1

    # ---------------------- finish_and_update ----------------------
    def finish_and_update(self) -> None:
        """
        Compute bootstrap last values and call manager.finish_rollouts and manager.update_all.
        """
        # 1) compute last_values per slot for bootstrap
        last_value_dict = {}
        # Preferred: manager provides compute_value_for_last_obs()
        if hasattr(self.manager, "compute_value_for_last_obs"):
            try:
                last_value_dict = self.manager.compute_value_for_last_obs()
            except Exception:
                last_value_dict = {}
        else:
            # Fallback: try to compute values per agent using manager.agents[*].get_value or manager.get_value
            try:
                # attempt to use manager.get_value with the cached last observations
                obs_all = self._get_obs_all()
                if obs_all is None:
                    last_value_dict = {}
                else:
                    # if obs_all is list of per-env obs, compute per-env values and aggregate per-slot key
                    if isinstance(obs_all, (list, tuple)):
                        # compute values per env using manager.select_actions or manager.agents policy get_value
                        for i, obs in enumerate(obs_all):
                            try:
                                v = self.manager.get_value(obs,
                                                           slot_hidden=self.hidden_per_env["slot"][i] if self.hidden_per_env else None,
                                                           bev_hidden=self.hidden_per_env["bev"][i] if self.hidden_per_env else None)
                            except Exception:
                                v = None
                            # manager.finish_rollouts expects a dict keyed by slot id -> last_value
                            if v is not None:
                                # convert torch -> numpy or float
                                try:
                                    if torch.is_tensor(v):
                                        v_np = v.detach().cpu().numpy()
                                    else:
                                        v_np = np.array(v)
                                    # ADAPT: produce mapping {slot: value} or aggregated shape
                                    # If manager expects matrix [B,N], provide that
                                    last_value_dict[i] = v_np
                                except Exception:
                                    last_value_dict[i] = 0.0
                    else:
                        # single-env case
                        try:
                            v = self.manager.get_value(obs_all,
                                                       slot_hidden=(self.hidden_per_env["slot"][0] if self.hidden_per_env else None),
                                                       bev_hidden=(self.hidden_per_env["bev"][0] if self.hidden_per_env else None))
                            if v is not None:
                                if torch.is_tensor(v):
                                    v = v.detach().cpu().numpy()
                                last_value_dict = v
                        except Exception:
                            last_value_dict = {}
            except Exception:
                last_value_dict = {}

        # 2) finish rollouts (compute GAE / returns) via manager
        if hasattr(self.manager, "finish_rollouts"):
            self.manager.finish_rollouts(last_value_dict)
        else:
            raise RuntimeError("manager must implement finish_rollouts(last_value_dict)")

        # 3) trigger policy updates
        if hasattr(self.manager, "update_all"):
            self.manager.update_all()
        else:
            raise RuntimeError("manager must implement update_all()")

    # ---------------------- utilities ----------------------
    def _unpack_select_return(self, sel_return):
        """
        Normalize different return signatures from manager.select_actions.
        Expected canonical return:
          (actions, values, logps, next_slot_hidden, next_bev_hidden, info)
        but allow variations.
        """
        # If manager.select_actions returned tuple of length 6 -> unpack
        if isinstance(sel_return, tuple) or isinstance(sel_return, list):
            # len 6
            if len(sel_return) == 6:
                return tuple(sel_return)
            # some implementations return (actions, values, logp, info) or (actions, info)
            if len(sel_return) == 4:
                actions, values, logps, info = sel_return
                return actions, values, logps, None, None, info
            if len(sel_return) == 2:
                actions, info = sel_return
                # try to find values/logps inside info dict
                values = info.get("values") if isinstance(info, dict) else None
                logps = info.get("log_probs") if isinstance(info, dict) else None
                next_slot = info.get("next_slot_hidden") if isinstance(info, dict) else None
                next_bev = info.get("next_bev_hidden") if isinstance(info, dict) else None
                return actions, values, logps, next_slot, next_bev, info
        # if manager returned dict
        if isinstance(sel_return, dict):
            actions = sel_return.get("actions", sel_return.get("action"))
            values = sel_return.get("values", sel_return.get("value"))
            logps = sel_return.get("log_probs", sel_return.get("logp"))
            next_slot = sel_return.get("next_slot_hidden")
            next_bev = sel_return.get("next_bev_hidden")
            info = {k: sel_return.get(k) for k in ("mu", "log_std", "pre_tanh") if k in sel_return}
            return actions, values, logps, next_slot, next_bev, info

        # fallback: unknown format
        return sel_return, None, None, None, None, None

    def _select_actions_accepts_kwargs(self) -> bool:
        """
        Inspect manager.select_actions signature to determine if it accepts kwargs (slot_hidden=...).
        Simple heuristic to support both styles.
        """
        try:
            import inspect
            sig = inspect.signature(self.manager.select_actions)
            params = sig.parameters
            # if it has slot_hidden or bev_hidden or **kwargs we assume it accepts kwargs
            if "slot_hidden" in params or "bev_hidden" in params:
                return True
            for p in params.values():
                if p.kind == p.VAR_KEYWORD:
                    return True
            return False
        except Exception:
            return False

    def _ensure_dict_like(self, x):
        """If x is a tensor or array, return as-is or attempt to convert to dict mapping slot->value if needed."""
        return x

    def _build_respawn_mask_from_dones(self, dones) -> Optional[torch.Tensor]:
        """
        Build respawn mask [N] / [B,N] from dones structure.
        This is a placeholder: adapt to your dones format.
        If dones is a dict keyed by slot -> bool, convert to appropriate mask tensor.
        """
        # ADAPT HERE if dones format is not trivial
        try:
            if dones is None:
                return None
            if isinstance(dones, dict):
                # produce mask tensor where 1 -> reset
                # map ordering must match manager's slots ordering
                mask_list = []
                for slot in getattr(self.manager, "agent_slots", []):
                    val = dones.get(slot, False)
                    mask_list.append(1.0 if val else 0.0)
                return torch.tensor(mask_list, dtype=torch.float32)
            if isinstance(dones, (list, tuple, np.ndarray)):
                # assume list of booleans per agent
                arr = np.array(dones)
                return torch.tensor(arr.astype(np.float32))
            if isinstance(dones, torch.Tensor):
                return dones.float()
        except Exception:
            pass
        return None


