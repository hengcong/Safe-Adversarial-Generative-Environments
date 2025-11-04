from collections import OrderedDict
from typing import Dict, Any, List, Optional, Iterable
import torch

from algo.mappo.rollout_buffer import RolloutBuffer
from algo.mappo.mappo_agent import MAPPOAgent

class MAPPOManager(object):
    def __init__(self, agent_specs, policy_ctor, device: str = "cpu", selector: Optional[Any] = None):
        self.device = torch.device(device)
        self.agent_slots = list(agent_specs.keys())
        # create agents for each slot
        self.agents: "OrderedDict[Any, Any]" = OrderedDict()
        for aid, spec in agent_specs.items():
            self.agents[aid] = policy_ctor(spec)

        self.buffers: Dict[Any, RolloutBuffer] = {}
        for slot, spec in agent_specs.items():
            obs_dim = spec.get("obs_dim")
            act_dim = spec.get("act_dim")
            buf_size = spec.get("buffer_size", 2048)

            if obs_dim is not None and act_dim is not None and buf_size is not None:
                self.buffrs[slot] = RolloutBuffer((80, 80, 3),
                                                  int(act_dim),
                                                  int(buf_size),
                                                  gamma= spec.get("gamma", 0.99),
                                                  gae_lambda=spec.get("gae_lambda", 0.95),
                                                  device=device
                                                   )
            else:
                self. buffers[slot] = None

        self.slot2veh: Dict[Any, Optional[Any]] = {s: None for s in self.agent_slots}
        self.veh2slot: Dict[Any, Any] = {}
        self.selector = selector

    def mappo_reset(self, state_dict: Dict[Any, Any]) -> None:
        """
        state_dict: {vehicle_id: obs} OR if selector is None, {slot_id: obs}
        Manager calls agent.mappo_reset(initial_obs) and clears buffers.
        """
        # clear all buffers
        for buf in self.buffers.values():
            if buf is not None:
                buf.clear()

        # reset agents: need to map initial obs properly
        if self.selector is None:
            # assume state_dict keyed by slots
            for slot, agent in self.agents.items():
                obs = state_dict.get(slot)
                agent.mappo_reset(obs)
        else:
            # state_dict keyed by vehicle_id; use slot2veh mapping
            for slot, agent in self.agents.items():
                vid = self.slot2veh.get(slot)
                obs = state_dict.get(vid)
                agent.mappo_reset(obs)

    # ----------------- selection & mapping -----------------
    def sync_agents_with_selected(self, selected_vehicle_ids: Iterable[Any],
                                  vehicles_info: Optional[Dict[Any, Dict[str, Any]]] = None) -> None:
        """
        Sync slot->vehicle mapping to the provided selected_vehicle_ids list.
        Strategy:
          - keep existing slot->veh mappings if vehicle remains selected (stability)
          - assign new vehicles to free slots
          - if no free slot, choose a victim slot (replace farthest or LRU). Here we pick first free, else replace arbitrary slot.
        vehicles_info optional: {veh_id: {...}} used by victim choice if provided.
        """
        selected = list(selected_vehicle_ids)
        # keep previous where possible
        new_slot2veh = {}
        free_slots = [s for s in self.agent_slots]

        # keep stable mappings
        for slot, veh in self.slot2veh.items():
            if veh in selected:
                new_slot2veh[slot] = veh
                free_slots.remove(slot)

        # assign remaining selected vehicles
        for vid in selected:
            if vid in new_slot2veh.values():
                continue
            if free_slots:
                slot = free_slots.pop(0)
            else:
                # choose victim slot: prefer slot whose veh is farthest if vehicles_info provided
                if vehicles_info is not None:
                    # compute distances to ego if ego slot exists
                    try:
                        ego_vid = next(v for v in new_slot2veh.values() if v is not None)  # fallback: any existing
                        ego_pos = vehicles_info[ego_vid]['pos']
                        far_slot = None
                        far_dist = -1.0
                        for s, oldv in self.slot2veh.items():
                            if oldv is None: continue
                            pos = vehicles_info.get(oldv, {}).get('pos')
                            if pos is None: continue
                            d = ((pos[0] - ego_pos[0]) ** 2 + (pos[1] - ego_pos[1]) ** 2) ** 0.5
                            if d > far_dist:
                                far_dist = d;
                                far_slot = s
                        slot = far_slot if far_slot is not None else self.agent_slots[0]
                    except StopIteration:
                        slot = self.agent_slots[0]
                else:
                    slot = self.agent_slots[0]

                # cleanup replaced slot
                old_vid = self.slot2veh.get(slot)
                if old_vid is not None:
                    # reset agent runtime state and clear its buffer
                    try:
                        self.agents[slot].mappo_reset()
                    except Exception:
                        pass
                    buf = self.buffers.get(slot)
                    if buf is not None:
                        try:
                            buf.clear()
                        except Exception:
                            pass

            new_slot2veh[slot] = vid

        # finalize mappings
        self.slot2veh = {s: new_slot2veh.get(s) for s in self.agent_slots}
        self.veh2slot = {v: s for s, v in self.slot2veh.items() if v is not None}

    # ----------------- action selection -----------------
    def select_actions(self, obs_dict: Dict[Any, Any]) -> (Dict[Any, Any], Dict[Any, Any]):
        """
        obs_dict: if selector is None: {slot: obs}
                  if selector present: {vehicle_id: obs}
        returns:
            actions: {slot: action}  (actions indexed by slot)
            info: {slot: {'value':v, 'logp':logp}}
        """
        actions = {}
        info = {}
        if self.selector is None:
            # obs keyed by slot
            for slot, agent in self.agents.items():
                obs = obs_dict.get(slot)
                a, v, logp = agent.select_action(obs)
                actions[slot] = a
                info[slot] = {'value': v, 'logp': logp}
        else:
            # obs keyed by vehicle_id: map to slots
            for slot, agent in self.agents.items():
                vid = self.slot2veh.get(slot)
                obs = obs_dict.get(vid)
                a, v, logp = agent.select_action(obs)
                actions[slot] = a
                info[slot] = {'value': v, 'logp': logp}
        return actions, info

    # ----------------- store transitions -----------------
    def store_transitions(self,
                          obs_dict: Dict[Any, Any],
                          act_dict: Dict[Any, Any],
                          rew_dict: Dict[Any, float],
                          done_dict: Dict[Any, bool],
                          val_dict: Dict[Any, float],
                          logp_dict: Dict[Any, float]) -> None:
        """
        Store transitions into per-slot buffers.
        Inputs keyed like obs_dict in select_actions (vehicle_id or slot_id).
        """
        if self.selector is None:
            # keys are slots
            for slot in self.agent_slots:
                buf = self.buffers.get(slot)
                if buf is None: continue
                obs = obs_dict.get(slot)
                act = act_dict.get(slot)
                rew = rew_dict.get(slot)
                done = done_dict.get(slot)
                val = val_dict.get(slot)
                logp = logp_dict.get(slot)
                buf.add(obs, act, rew, done, val, logp)
        else:
            # keys are vehicle_id: map to slots
            for slot in self.agent_slots:
                buf = self.buffers.get(slot)
                if buf is None: continue
                vid = self.slot2veh.get(slot)
                if vid is None: continue
                obs = obs_dict.get(vid)
                act = act_dict.get(slot)  # actions produced earlier by select_actions are slot-indexed
                rew = rew_dict.get(vid)
                done = done_dict.get(vid)
                val = val_dict.get(slot)
                logp = logp_dict.get(slot)
                buf.add(obs, act, rew, done, val, logp)

    # ----------------- finish rollouts -----------------
    def finish_rollouts(self, last_val_dict: Dict[Any, float]) -> None:
        """
        last_val_dict keyed like select_actions info (slot if selector None else slot)
        Compute GAE / returns for each buffer.
        """
        for slot, buf in self.buffers.items():
            if buf is None:
                continue
            last_v = last_val_dict.get(slot, 0.0)
            buf.finish_path(last_v)

    # ----------------- get data & update -----------------
    def get_training_data(self, slot) -> Dict[str, torch.Tensor]:
        """Return buffer.get() for a slot (tensors on manager.device)."""
        buf = self.buffers.get(slot)
        assert buf is not None, "Buffer for slot not configured"
        return buf.get()

    def update_all(self) -> None:
        """Fetch each slot's data and call agent.update_from_data(data). Clears buffers after update."""
        for slot, agent in self.agents.items():
            buf = self.buffers.get(slot)
            if buf is None:
                continue
            # skip empty buffers
            if buf.length == 0:
                continue
            data = buf.get()  # tensors on device
            # agent.update_from_data expects tensors on its device; move if necessary
            # Here we assume manager.device == agent.device for simplicity
            agent.update_from_data(data)
            buf.clear()

    # ----------------- save / load -----------------
    def save_all(self, prefix: str) -> None:
        for slot, agent in self.agents.items():
            agent.save(f"{prefix}_{slot}")

    def load_all(self, prefix: str, map_location=None) -> None:
        for slot, agent in self.agents.items():
            agent.load(f"{prefix}_{slot}", map_location=map_location)