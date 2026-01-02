
import torch
import numpy as np
from typing import Optional, Dict, Any, Tuple
from .trainer import Trainer


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
                 num_steps: int = 128,
                 device: str = "cpu",
                 **kwargs):
        super().__init__(envs=envs, manager=manager, device=device, **kwargs)
        self.T = int(num_steps)
        # number of parallel environments
        self.num_envs = getattr(envs, "num_envs", None)
        if self.num_envs is None:
            # try len()
            try:
                self.num_envs = len(envs)
            except Exception:
                self.num_envs = 1

        # initialize envs and manager
        init_obs = self.envs.reset()
        # map manager with initial observations
        if hasattr(self.manager, "mappo_reset"):
            try:
                self.manager.mappo_reset(init_obs)
            except Exception:
                # fallback: call without args
                self.manager.mappo_reset({})

        # # initialize hidden states per env (if supported)
        # self.hidden_per_env = None
        # if hasattr(self.manager, "get_initial_hidden"):
        #     # we need n_agent; try to fetch from manager.agent_slots or manager.agents
        #     n_agents = getattr(self.manager, "n_agents", None)
        #     if n_agents is None:
        #         # try to infer from first agent policy
        #         try:
        #             first_agent = next(iter(self.manager.agents.values()))
        #             n_agents = getattr(first_agent.policy, "n_agent", None) or getattr(first_agent, "n_agent", None)
        #         except Exception:
        #             n_agents = None
        #
        #     if n_agents is None:
        #         # leave hidden None; manager.select_actions should accept None
        #         self.hidden_per_env = None
        #     else:
        #         slot_h, bev_h = self.manager.get_initial_hidden(self.num_envs, n_agents)
        #         self.hidden_per_env = {"slot": slot_h, "bev": bev_h}
        # initialize hidden states per env (if supported)
        # initialize hidden states per env (if supported)
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

        import numpy as np
        import torch

        env = self.envs
        manager = self.manager
        T = self.T

        # -------- RESET ENV + WARMUP SEQ --------
        seq = env.reset()  # {"obs_hist":..., "seq": {...}}
        seq_dict = seq.get("seq", {})

        seq_agent_feats = seq_dict.get("agent_feats", None)
        seq_type_ids = seq_dict.get("type_id", None)
        seq_masks = seq_dict.get("mask", None)
        seq_images = seq_dict.get("images", None)

        # ---- build obs_seq dict for manager.burn_in ----
        obs_seq = {}

        # images → normalize to [T, B, C, H, W] with B=1
        if seq_images is not None:
            imgs = seq_images
            if isinstance(imgs, np.ndarray):
                if imgs.ndim == 4:
                    if imgs.shape[-1] in (1, 3, 4):
                        imgs = np.transpose(imgs, (0, 3, 1, 2))  # [T,H,W,C] -> [T,C,H,W]
                else:
                    try:
                        imgs = np.asarray(imgs)
                        if imgs.ndim == 5:
                            if imgs.shape[1] == 1:
                                imgs = imgs[:, 0, ...]
                            elif imgs.shape[0] == 1:
                                imgs = imgs[0, ...]
                    except Exception:
                        raise RuntimeError(
                            f"collect_rollout: seq_images has unsupported ndim={getattr(seq_images, 'ndim', None)}"
                        )
            else:
                try:
                    imgs = np.asarray(imgs)
                    if imgs.ndim == 4 and imgs.shape[-1] in (1, 3, 4):
                        imgs = np.transpose(imgs, (0, 3, 1, 2))
                except Exception:
                    imgs = None

            if imgs is not None:
                imgs = imgs.astype(np.float32, copy=False)
                obs_seq["image"] = torch.tensor(imgs, dtype=torch.float32).unsqueeze(1)

        # agent_feats → [T,B,N,F]
        if seq_agent_feats is None:
            single = env.get_single_obs_for_manager()
            if single is None:
                raise RuntimeError("collect_rollout: cannot infer agent_feats shape")
            a = single.get("agent_feats", None)
            if a is None:
                raise RuntimeError("collect_rollout: cannot infer agent_feats")

            if torch.is_tensor(a):
                B_, N_, F_ = int(a.shape[0]), int(a.shape[1]), int(a.shape[2])
            else:
                ar = np.asarray(a)
                if ar.ndim == 3:
                    B_, N_, F_ = ar.shape
                elif ar.ndim == 2:
                    B_, N_, F_ = 1, ar.shape[0], ar.shape[1]
                else:
                    raise RuntimeError("collect_rollout: unexpected agent_feats shape")

            burn_T = int(getattr(self.envs, "seq_len", 10))
            seq_agent_feats = np.zeros((burn_T, N_, F_), dtype=np.float32)

        af = torch.tensor(seq_agent_feats, dtype=torch.float32)
        if af.ndim == 3:
            af = af.unsqueeze(1)
        elif af.ndim != 4:
            raise RuntimeError(f"collect_rollout: seq_agent_feats unexpected ndim={af.ndim}")
        obs_seq["agent_feats"] = af

        # type_ids → [T,B,N]
        if seq_type_ids is not None:
            tid = torch.tensor(seq_type_ids, dtype=torch.long)
            if tid.ndim == 2:
                tid = tid.unsqueeze(1)
            obs_seq["type_id"] = tid

        # masks → [T,B,N]
        if seq_masks is not None:
            m = torch.tensor(seq_masks, dtype=torch.float32)
            if m.ndim == 2:
                m = m.unsqueeze(1)
            obs_seq["mask"] = m

        # ---- burn-in hidden ----
        hidden_slot, hidden_bev = manager.burn_in(obs_seq, detach=True)
        hidden = {"slot": hidden_slot, "bev": hidden_bev}

        # ---- initial obs from last warmup frame ----
        obs = env.get_single_obs_for_manager()
        manager.reset_buffer()

        # -------- MAIN ROLLOUT LOOP --------
        for t in range(T):

            # ensure obs contains time-seq "images" when manager/store requires it
            try:
                if isinstance(obs, dict):
                    if "images" not in obs:
                        if "image" in obs and obs["image"] is not None:
                            im = obs["image"]

                            if torch.is_tensor(im):
                                im_np = im.detach().cpu().numpy()
                            else:
                                try:
                                    im_np = np.asarray(im)
                                except Exception:
                                    im_np = None

                            if im_np is not None:
                                if im_np.ndim == 4:
                                    if im_np.shape[0] == 1:
                                        im_np = im_np[0]  # [C,H,W]
                                    else:
                                        im_np = im_np[0]
                                if im_np.ndim == 3:
                                    im_np = im_np[None, ...]  # [1,C,H,W]
                                elif im_np.ndim == 2:
                                    im_np = im_np[None, None, ...]

                                imgs_arr = im_np.astype(np.float32)
                                obs["images"] = imgs_arr
                        else:
                            if "image" in obs_seq:
                                try:
                                    seq_img = obs_seq["image"]
                                    if torch.is_tensor(seq_img):
                                        seq_img_np = seq_img.detach().cpu().numpy()
                                        last = seq_img_np[-1, 0]
                                        obs["images"] = last[None, ...]
                                except Exception:
                                    pass
            except Exception:
                pass

            # select actions
            actions, values, logps, next_slot_h, next_bev_h, info = manager.select_actions(
                obs_dict=obs,
                hidden=hidden,
                mask=obs.get("mask", None)
            )

            # step env
            nxt_obs, rewards_dict, dones_dict, _ = env.step(actions)

            slot_list = manager.agent_slots
            rewards = np.array([rewards_dict.get(s, 0.0) for s in slot_list], dtype=np.float32)[None, :]
            dones = np.array([dones_dict.get(s, False) for s in slot_list], dtype=np.bool_)[None, :]
            vals = np.array([values[s] for s in slot_list], dtype=np.float32)[None, :]
            lps = np.array([logps[s] for s in slot_list], dtype=np.float32)[None, :]

            # store
            manager.store_transitions(
                obs_dict=obs,
                act_dict=actions,
                rew_dict=rewards_dict,
                done_dict=dones_dict,
                val_dict=values,
                logp_dict=logps
            )

            hidden = {"slot": next_slot_h, "bev": next_bev_h}
            obs = nxt_obs

            if dones.any():
                break

        # -------- BOOTSTRAP --------
        # manager may not implement value(); use select_actions to obtain values from policy
        try:
            # call select_actions in deterministic mode to get values only
            actions_tmp, values_tmp, logps_tmp, next_slot_h_tmp, next_bev_h_tmp, info_tmp = manager.select_actions(
                obs_dict=obs,
                hidden=hidden,
                mask=obs.get("mask", None)
            )
            # values_tmp is a dict: slot -> float
            slot_list = manager.agent_slots
            last_vals_arr = np.array([values_tmp.get(s, 0.0) for s in slot_list], dtype=np.float32)[None, :]
            manager.finish_rollouts(last_vals_arr)
        except Exception as e:
            # fallback: if select_actions fails, provide zeros to finish_rollouts to avoid crash
            print("[WARN] bootstrap: failed to get values via manager.select_actions(), falling back to zeros:", e)
            slot_list = manager.agent_slots
            last_vals_arr = np.zeros((1, len(slot_list)), dtype=np.float32)
            manager.finish_rollouts(last_vals_arr)

        # -------- UPDATE --------
        manager.update_all()

    # def collect_rollout(self) -> None:
    #     """
    #     Collect self.T steps for all parallel envs and store into manager/buffers.
    #     This implementation iterates envs and calls manager.select_actions() per env.
    #     If your envs supports vectorized batch step you can modify it to call manager once per batch.
    #     """
    #     for t in range(self.T):
    #         obs_all = self._get_obs_all()  # may be list of per-env obs dicts, or dict keyed by vehicles
    #         if obs_all is None:
    #             raise RuntimeError("Could not get observations from envs; make sure envs.reset() or get_obs_all() is implemented.")
    #
    #         # store selections for each env
    #         actions_per_env = [None] * self.num_envs
    #         info_per_env = [None] * self.num_envs
    #         values_per_env = [None] * self.num_envs
    #         logps_per_env = [None] * self.num_envs
    #         next_slot_hidden_per_env = [None] * self.num_envs
    #         next_bev_hidden_per_env = [None] * self.num_envs
    #
    #         # 1) select actions
    #         # Accept obs_all as list-like (per env) or a dict (single env mapping)
    #         if isinstance(obs_all, (list, tuple)):
    #             for i, obs in enumerate(obs_all):
    #                 # get hidden for this env if available
    #                 slot_h = None
    #                 bev_h = None
    #                 if self.hidden_per_env is not None:
    #                     slot_h = self.hidden_per_env.get("slot")[i] if self.hidden_per_env.get("slot") is not None else None
    #                     bev_h = self.hidden_per_env.get("bev")[i] if self.hidden_per_env.get("bev") is not None else None
    #
    #                 # manager.select_actions expected signature: select_actions(obs_dict, hidden=...)
    #                 # but many implementations use select_actions(obs_dict) -> (actions, values, logp, next_slot_h, next_bev_h, info)
    #                 sel = self.manager.select_actions(obs, slot_hidden=slot_h, bev_hidden=bev_h) \
    #                       if self._select_actions_accepts_kwargs() else self.manager.select_actions(obs)
    #                 # normalize return
    #                 actions, values, logps, next_slot_h, next_bev_h, info = self._unpack_select_return(sel)
    #                 actions_per_env[i] = actions
    #                 values_per_env[i] = values
    #                 logps_per_env[i] = logps
    #                 info_per_env[i] = info
    #                 next_slot_hidden_per_env[i] = next_slot_h
    #                 next_bev_hidden_per_env[i] = next_bev_h
    #         else:
    #             # obs_all is not list: assume single-env mapping keyed by vehicle ids; treat as 1 env
    #             slot_h = None
    #             bev_h = None
    #             if self.hidden_per_env is not None:
    #                 slot_h = self.hidden_per_env.get("slot")[0] if self.hidden_per_env.get("slot") is not None else None
    #                 bev_h = self.hidden_per_env.get("bev")[0] if self.hidden_per_env.get("bev") is not None else None
    #
    #             sel = self.manager.select_actions(obs_all, slot_hidden=slot_h, bev_hidden=bev_h) \
    #                   if self._select_actions_accepts_kwargs() else self.manager.select_actions(obs_all)
    #             actions, values, logps, next_slot_h, next_bev_h, info = self._unpack_select_return(sel)
    #             actions_per_env[0] = actions
    #             values_per_env[0] = values
    #             logps_per_env[0] = logps
    #             info_per_env[0] = info
    #             next_slot_hidden_per_env[0] = next_slot_h
    #             next_bev_hidden_per_env[0] = next_bev_h
    #
    #         # 2) step envs and store transitions
    #         # We'll call env.step(env_idx, actions) for each env
    #         for i in range(self.num_envs):
    #             act = actions_per_env[i]
    #             nxt_obs, rewards, dones, env_info = self._step_env(i, act)
    #             # store transitions into manager -> manager will route to per-slot buffers
    #             # manager.store_transitions(obs, actions, rewards, dones, values, logps)
    #             obs_i = obs_all[i] if isinstance(obs_all, (list, tuple)) else obs_all
    #             vals = self._ensure_dict_like(values_per_env[i])
    #             lps = self._ensure_dict_like(logps_per_env[i])
    #             self.manager.store_transitions(obs_i, act, rewards, dones, vals, lps)
    #
    #             # maintain hidden states per env if available
    #             if self.hidden_per_env is not None:
    #                 if next_slot_hidden_per_env[i] is not None:
    #                     # expected shape for manager.get_initial_hidden outputs: slot_hidden [B,N,H] with first dim = num_envs
    #                     self.hidden_per_env["slot"][i] = next_slot_hidden_per_env[i]
    #                 if next_bev_hidden_per_env[i] is not None:
    #                     self.hidden_per_env["bev"][i] = next_bev_hidden_per_env[i]
    #
    #             # if any agents respawned (dones), reset hidden for those agents if manager supports it
    #             if self.hidden_per_env is not None and hasattr(self.manager, "reset_hidden_for_agents"):
    #                 # build respawn mask from dones (expected per-agent mask {slot: bool} or array)
    #                 # ADAPT HERE if your dones format differs
    #                 try:
    #                     respawn_mask = self._build_respawn_mask_from_dones(dones)
    #                     if respawn_mask is not None:
    #                         self.hidden_per_env["slot"][i] = self.manager.reset_hidden_for_agents(self.hidden_per_env["slot"][i], respawn_mask)
    #                 except Exception:
    #                     # ignore reset if incompatible
    #                     pass
    #
    #             # update last_obs cache
    #             if isinstance(obs_all, (list, tuple)):
    #                 # replace ith obs with next
    #                 try:
    #                     obs_all[i] = nxt_obs
    #                 except Exception:
    #                     pass
    #             else:
    #                 self._last_obs_all = nxt_obs
    #
    #         # update counters
    #         self.global_step += self.num_envs

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


