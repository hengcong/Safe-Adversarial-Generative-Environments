# envs/carla_env.py  — headless-only, pygame/render removed
import carla
import gym
import numpy as np
import time
import random
import math
import os
import json
import copy
from datetime import datetime
from collections import deque
from typing import Dict, List, Tuple, Optional
from gym import spaces, core
import cv2

import torch
from .observation.obs_bev import BEVObservation
from envs.safety.shield import SafetyShield

from algo.mappo.mappo_manager import MAPPOManager
from models.mappo_policy import MAPPOPolicy


class CarlaEnv(gym.Env):
    def __init__(self, num_veh, num_ped, mode="MAPPO", shield_mode="ecbf_collision", *, render: bool = False, experiment_path=None, spawn_background: bool = True):
        """
        Headless CarlaEnv (no pygame / rendering).
        signature keeps render/spawn_background for backward compatibility but rendering is disabled.
        """
        super(CarlaEnv, self).__init__()
        self.mode = mode
        # NOTE: render flag is ignored in headless env, kept for compat
        self.render = False
        self.spawn_background = bool(spawn_background)

        if self.mode == "MAPPO":
            self.is_multiagent = True
            self.adversary_enabled = False
        elif self.mode == "MATD3":
            self.is_multiagent = True
            self.adversary_enabled = True
        elif self.mode == "EVAL":
            self.is_multiagent = False
            self.adversary_enabled = False
        else:
            self.mode = "MAPPO"
            self.is_multiagent = True
            self.adversary_enabled = False

        self.num_veh = num_veh
        self.num_ped = num_ped
        self.step_size = 0.05
        self.seq_len = 10

        # env-level constants — set once, used by agent_specs and manager
        self.obs_dim = 3
        self.bev_ch = 3
        self.bev_H = 84
        self.bev_W = 84
        self.type_embed_dim = 8
        self.num_type_bins = 8
        self.max_slots = 16

        self.agent_specs = {
            "vehicle": {
                "obs_dim": self.obs_dim,
                "act_dim": 2,
                "buffer_T": 2048,
                "num_envs": 1,
                "n_agents": self.num_veh,
                "buffer_size": 2048,
            },
            "pedestrian": {
                "obs_dim": self.obs_dim,
                "act_dim": 2,
                "buffer_T": 2048,
                "num_envs": 1,
                "n_agents": self.num_ped,
                "buffer_size": 2048,
            }
        }


        # BEV / per-agent observation specs (must be initialized before policy ctor)
        # self.obs_dim = getattr(self, "obs_dim", 128)  # per-agent scalar feature dim (F_agent)
        self.bev_radius = 50.0
        self.bev_ch = getattr(self, "bev_ch", 3)  # BEV image channels (3 = RGB)
        self.bev_H = getattr(self, "bev_H", 84)  # BEV height
        self.bev_W = getattr(self, "bev_W", 84)  # BEV width
        self.type_embed_dim = getattr(self, "type_embed_dim", 8)
        self.num_type_bins = getattr(self, "num_type_bins", 8)
        self.max_slots = getattr(self, "max_slots", 16)
        self.latest_bev_image = None
        # connect
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)

        # choose map robustly (same logic as your old env)
        available_maps = self.client.get_available_maps()
        target_key = "Town04"
        selected_map = None
        for m in available_maps:
            if target_key.lower() in m.lower():
                selected_map = m
                break
        if selected_map is None:
            selected_map = "/Game/Carla/Maps/Town04"
        self.world = self.client.load_world(selected_map)
        self.map = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.step_size
        self.world.apply_settings(settings)

        # for _ in range(3):
        #     self.world.tick()

        # actor containers (no wrappers)
        self.ego_vehicle = None  # carla.Actor
        self.vehicles = []  # list[carla.Actor] (background veh)
        self.pedestrians = []  # list[carla.Actor]
        self.slot2actor = None
        self.agent_ids = []
        self.rl_slots = None

        # controller mapping: actor_id -> controller instance
        self.ped_controllers = {}

        # bookkeeping
        self.collision_happened = False
        self.start_time = 0.0
        self.distance_travelled = 0.0
        self.last_location = None

        # sensors / latest image (keep storage but headless, image is raw numpy)
        self.camera_sensor = None
        self.latest_image = None

        # --- allowed vehicle blueprints (you can keep or reduce) ---
        self.allowed_brands = [
            "audi.tt", "bmw.grandtourer", "chevrolet.impala", "citroen.c3",
            "jeep.wrangler_rubicon", "lincoln.mkz_2020", "mini.cooper_s",
            "nissan.micra", "seat.leon", "tesla.model3", "toyota.prius",
            "volkswagen.t2", "mercedes.coupe"
        ]

        # --- walkers list for ped spawns ---
        self.walker_bps = list(self.blueprint_library.filter("walker.pedestrian.*"))

        # --- basic sensors / spectator ---
        self.collision_sensor = None
        self.spectator = self.world.get_spectator()

        # --- traffic manager handle only (optional, keep reference) ---
        self.tm_port = 8000
        try:
            # create/get TM once
            self.traffic_manager = self.client.get_trafficmanager(self.tm_port)
            # configure sync mode once (we will not call get_trafficmanager again in reset)
            try:
                self.traffic_manager.set_synchronous_mode(False)
            except Exception:
                # some CARLA versions use tm.set_synchronous_mode or tm.set_synchronous_mode may raise
                pass
        except Exception as e:
            print("[WARN] failed to init traffic_manager in __init__:", e)
            self.traffic_manager = None

        # --- spawn points / lane map ---
        self.spawn_points = self.map.get_spawn_points()
        self.lane_map = self._group_spawn_points_by_road_and_lane()

        # spawn actors: ego always; background only if spawn_background True
        # self.spawn_ego_vehicle()
        # if self.spawn_background:
        #     if self.num_veh > 0:
        #         self.spawn_vehicles()
        #     if self.num_ped > 0:
        #         self.spawn_pedestrians()

        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0.0, high=1.0, shape=(self.bev_ch, self.bev_H, self.bev_W), dtype=np.float32),
            "agent_feats": spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_slots, self.obs_dim), dtype=np.float32),
            "type_id": spaces.Box(low=0, high=10, shape=(self.max_slots,), dtype=np.int64),  # small vocab
            "mask": spaces.Box(low=0.0, high=1.0, shape=(self.max_slots,), dtype=np.float32),
        })

        low = np.tile(np.array([-1.0, -1.0, 0.0, 0.0], dtype=np.float32), (self.max_slots, 1))
        high = np.tile(np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32), (self.max_slots, 1))

        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.obs_hist = None

        # --- experiment path & minimal logger (timestamped) ---
        if experiment_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_path = f"./carla_experiments/run_{timestamp}"
        else:
            self.experiment_path = experiment_path

        os.makedirs(self.experiment_path, exist_ok=True)

        # --- episode bookkeeping ---
        self.collision_happened = False
        self.episode_info = {"id": 0, "start_time": self.get_simulation_time(), "end_time": self.get_simulation_time()}
        self.episode_data = {}
        self._last_control = {}

        self.step_data = {}

        # --- controllers placeholders (empty dicts; you can add controllers later) ---
        self.ego_controller = None
        self.bv_controllers = {}
        self.dead_slots = set()
        # Shield / CBF training schedule (add these to __init__)
        # Shield mode:
        # "none", "soft", "single_cbf", "ecbf_collision"
        self.shield_mode = shield_mode
        self.shield_affects_reward = False
        self.safety_shield = SafetyShield(
            mode=self.shield_mode
        )

        # per-episode counter (maintained in reset)
        self.episode_num = 0
        # optionally expose safety cost weight baseline
        self.safety_cost_base_weight = getattr(self, "safety_cost_base_weight", 0.05)  # small default

    # ------------------- spawn / warmup / group helpers -------------------
    def spawn_ego_vehicle(self, bp_filter="vehicle.tesla.model3", spawn_retry=30, tm_autopilot=True):
        bp_list = list(self.blueprint_library.filter(bp_filter))
        bp = bp_list[0] if bp_list else random.choice(list(self.blueprint_library.filter("vehicle.*")))
        spawn_points = list(self.map.get_spawn_points())

        ego = None
        for _ in range(spawn_retry):
            sp = random.choice(spawn_points)
            ego = self.world.try_spawn_actor(bp, sp)
            if ego:
                break

        if ego is None:
            raise RuntimeError("Failed to spawn ego vehicle")

        self.ego_vehicle = ego
        if tm_autopilot:
            ego.set_autopilot(True, self.traffic_manager.get_port())

        self.last_location = ego.get_location()
        self.start_time = self.get_simulation_time()
        self.distance_travelled = 0.0

    def spawn_vehicles(self, num_vehicles=None, spawn_per_lane_limit=20, use_tm=True):
        if num_vehicles is None:
            num_vehicles = int(getattr(self, "num_veh", 30))

        lanes = [(k, sps) for k, sps in self.lane_map.items() if sps]
        random.shuffle(lanes)

        lane_counts = {}
        for (road_lane, sps) in lanes:
            if len(self.vehicles) >= num_vehicles:
                break
            road_id, lane_id = road_lane
            key = (road_id, lane_id)
            lane_counts.setdefault(key, 0)
            for sp in sps:
                if len(self.vehicles) >= num_vehicles:
                    break
                if lane_counts[key] >= spawn_per_lane_limit:
                    break

                bp = random.choice(list(self.blueprint_library.filter("vehicle.*")))
                if bp.has_attribute("color"):
                    colors = bp.get_attribute("color").recommended_values
                    if colors:
                        bp.set_attribute("color", random.choice(colors))

                veh = self.world.try_spawn_actor(bp, sp)
                if not veh:
                    continue

                self.vehicles.append(veh)
                lane_counts[key] += 1
                if use_tm:
                    try:
                        veh.set_autopilot(True, self.traffic_manager.get_port())
                        #print(f"[Veh Spawn] Successfully spawned {len(self.vehicles)} vehicles and TM")
                    except Exception:
                        pass

    def spawn_pedestrians(self,
                          num_pedestrians=None,
                          max_attempts_per_ped=8,
                          batch_size: int = 4,
                          start_controllers: bool = True):
        """
        Safer pedestrian spawn:
          - Only spawn controller if ped spawn succeeded.
          - Rollback ped when controller spawn/start fails.
          - Optionally spawn in batches and start controllers in small groups.
          - Print debug summary.

        Args:
            num_pedestrians: desired number (None -> use self.num_ped)
            max_attempts_per_ped: attempts cap per ped
            batch_size: how many to spawn before ticking / starting controllers
            start_controllers: whether to call controller.start() during spawn
        Returns:
            spawned: list of (ped_actor, controller_actor) pairs that succeeded
        """

        if num_pedestrians is None:
            num_pedestrians = int(getattr(self, "num_ped", 0))

        walker_bps = list(self.blueprint_library.filter("walker.pedestrian.*"))
        if not walker_bps or num_pedestrians <= 0:
            self.pedestrians = []
            self.ped_controllers = {}
            print("[Ped Spawn] no walker blueprints or num_ped <= 0")
            return []

        spawned = []
        pending_controllers = []  # (controller, ped) tuples to start in small groups
        attempts = 0
        created = 0
        max_attempts = max(1, num_pedestrians * max_attempts_per_ped)

        def safe_destroy(actor):
            if actor is None:
                return
            try:
                # guard: query server-for-existence before destroy to reduce "already dead" warnings
                try:
                    if self.world.get_actor(actor.id) is None:
                        return
                except Exception:
                    # if query fails, still attempt destroy in try/except below
                    pass
                actor.destroy()
            except Exception:
                pass

        while created < num_pedestrians and attempts < max_attempts:
            attempts += 1

            # 1) pick nav location
            loc = self.world.get_random_location_from_navigation()
            if loc is None:
                # no nav point found, give server a moment
                time.sleep(0.02)
                continue

            # 2) prepare blueprint
            bp = random.choice(walker_bps)
            if bp.has_attribute("role_name"):
                bp.set_attribute("role_name", "autogen")

            yaw = random.uniform(-180.0, 180.0)
            transform = carla.Transform(loc, carla.Rotation(yaw=yaw))

            # 3) try spawn ped
            ped = self.world.try_spawn_actor(bp, transform)
            if ped is None:
                # spawn failed; try again
                time.sleep(0.01)
                continue

            # 4) try spawn controller attached to ped (only if ped exists)
            controller = None
            try:
                controller_bp = self.blueprint_library.find("controller.ai.walker")
                controller = self.world.try_spawn_actor(controller_bp, carla.Transform(), attach_to=ped)
                if controller is None:
                    # failed to spawn controller -> rollback ped
                    safe_destroy(ped)
                    time.sleep(0.01)
                    continue

                # append to lists; postpone start until batch boundary to avoid bursts
                spawned.append((ped, controller))
                pending_controllers.append((controller, ped))
                created += 1

                # after each batch, tick and optionally start pending controllers in small groups
                if created % batch_size == 0:
                    try:
                        # prefer world.tick() if running synchronous server; otherwise a short sleep
                        try:
                            self.world.tick()
                        except Exception:
                            time.sleep(0.02)
                    except Exception:
                        time.sleep(0.02)

                    if start_controllers:
                        for ctrl, p in list(pending_controllers):
                            try:
                                ctrl.start()
                                # optional: give a random goal and speed; failures are non-fatal
                                try:
                                    tgt = self.world.get_random_location_from_navigation()
                                    if tgt is not None:
                                        ctrl.go_to_location(tgt)
                                        ctrl.set_max_speed(random.uniform(0.5, 1.5))
                                except Exception:
                                    pass
                            except Exception:
                                # start failed -> rollback both
                                safe_destroy(ctrl)
                                safe_destroy(p)
                                # remove this pair from spawned
                                spawned = [pc for pc in spawned if pc[0] != p]
                            # remove from pending regardless (either started or destroyed)
                            pending_controllers.remove((ctrl, p))
                    # small sleep to relieve server
                    time.sleep(0.01)

            except Exception:
                # unexpected: ensure cleanup
                safe_destroy(controller)
                safe_destroy(ped)
                time.sleep(0.01)
                continue

        # start any remaining pending controllers (small tail)
        if start_controllers and pending_controllers:
            try:
                try:
                    self.world.tick()
                except Exception:
                    time.sleep(0.02)
            except Exception:
                time.sleep(0.02)

            for ctrl, p in list(pending_controllers):
                try:
                    ctrl.start()
                    try:
                        tgt = self.world.get_random_location_from_navigation()
                        if tgt is not None:
                            ctrl.go_to_location(tgt)
                            ctrl.set_max_speed(random.uniform(0.5, 1.5))
                    except Exception:
                        pass
                except Exception:
                    safe_destroy(ctrl)
                    safe_destroy(p)
                    spawned = [pc for pc in spawned if pc[0] != p]
                pending_controllers.remove((ctrl, p))
            time.sleep(0.01)

        # finalize: filter out any None entries and build internal mappings
        self.pedestrians = [p for p, c in spawned if p is not None]
        self.ped_controllers = {p.id: c for p, c in spawned if (p is not None and c is not None)}

        print(f"[Ped Spawn] requested={num_pedestrians} created={len(self.pedestrians)} attempts={attempts}")
        # optional detailed ids
        if len(self.pedestrians) > 0:
            ids = ", ".join(str(p.id) for p in self.pedestrians)
            print(f"[Ped Spawn] ped ids: {ids}")

        return spawned

    def warmup(self, num_frames=40, burn_in_T=None):

        if burn_in_T is None:
            burn_in_T = getattr(self, "seq_len", 10)

        # --- TRY TO ENSURE BEV EXISTS: try several world ticks if needed ---
        if getattr(self, "latest_bev_image", None) is None:
            max_wait_ticks = 6
            waited = 0
            while waited < max_wait_ticks and getattr(self, "latest_bev_image", None) is None:
                try:
                    self.world.tick()
                except Exception:
                    time.sleep(0.02)
                # attempt inline BEVObservation extraction (your inline logic from reset)
                try:
                    bev_img = None
                    obs_bev = BEVObservation(veh_id=self.ego_vehicle.id, time_stamp=self.get_simulation_time())
                    obs_bev.update(env=self)
                    info = getattr(obs_bev, "information", None) or {}
                    for k in ("birdview", "bev", "bird_eye", "image", "birdview_rgb"):
                        if k in info and info[k] is not None:
                            bev_img = info[k];
                            break
                    if bev_img is None:
                        if hasattr(obs_bev, "image"):
                            bev_img = getattr(obs_bev, "image")
                        elif hasattr(obs_bev, "get_image"):
                            try:
                                bev_img = obs_bev.get_image()
                            except Exception:
                                bev_img = None
                    if bev_img is not None:
                        arr = np.asarray(bev_img)
                        # CHW -> HWC
                        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] != arr.shape[2]:
                            arr = arr.transpose(1, 2, 0)
                        if arr.ndim == 2:
                            arr = arr[:, :, None]
                        if arr.dtype != np.float32:
                            arr = arr.astype(np.float32, copy=False)
                        if arr.size > 0 and arr.max() > 1.01:
                            arr = arr / 255.0
                        bev_ch = int(getattr(self, "bev_ch", 3))
                        if arr.ndim == 3 and arr.shape[2] != bev_ch:
                            if arr.shape[2] > bev_ch:
                                arr = arr[:, :, :bev_ch]
                            else:
                                arr = np.repeat(arr[:, :, :1], bev_ch, axis=2)
                        self.latest_bev_image = arr
                except Exception:
                    # ignore per-try error; continue ticking
                    pass
                waited += 1

        # if still missing after retries -> WARN and create predictable empty seq (avoid crash)
        if getattr(self, "latest_bev_image", None) is None:
            print(f"[warmup] WARNING: latest_bev_image still None after retries ({max_wait_ticks} ticks). "
                  "Warmup will return zero-filled seq to avoid crash. Investigate BEVObservation.")
            # Build zero-filled seq consistent with expected shapes to avoid downstream TypeError
            T = int(burn_in_T)
            N = int(getattr(self, "max_slots", 0))
            F = int(getattr(self, "obs_dim", 0))
            C = int(getattr(self, "bev_ch", 3))
            H = int(getattr(self, "bev_H", 128))
            W = int(getattr(self, "bev_W", 128))

            seq_images = np.zeros((T, C, H, W), dtype=np.float32)  # [T, C, H, W]
            if N > 0 and F > 0:
                seq_agent_feats = np.zeros((T, N, F), dtype=np.float32)  # [T, N, F]
            else:
                seq_agent_feats = None

            seq_type_ids = np.zeros((T, N), dtype=np.int64) if N > 0 else None
            seq_masks = np.zeros((T, N), dtype=np.float32) if N > 0 else None

            # return empty obs_hist and zero-filled seq (trainer won't crash on torch.tensor)
            return None, {"obs_hist": [], "seq": {"images": seq_images, "agent_feats": seq_agent_feats,
                                                  "type_id": seq_type_ids, "mask": seq_masks}}

        # ---- normal warmup path (BEV exists) ----
        last_obs = None
        self.obs_hist = deque(maxlen=burn_in_T)

        imgs = []
        agent_feats = []
        type_ids = []
        masks = []

        for _ in range(int(num_frames)):
            try:
                self.world.tick()
            except Exception:
                time.sleep(0.01)

            obs = self.get_single_obs_for_manager()
            last_obs = obs
            if obs is None:
                continue

            # IMAGE normalize -> numpy CHW per-step (no leading batch)
            img = obs.get("image", None)
            img_np = None
            try:
                if hasattr(img, "detach"):
                    arr = img.detach().cpu().numpy()
                    if arr.ndim == 4 and arr.shape[0] == 1:
                        arr = arr[0]  # [C,H,W]
                    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
                        img_np = arr
                    elif arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
                        img_np = np.transpose(arr, (2, 0, 1))
                    elif arr.ndim == 2:
                        img_np = np.expand_dims(arr, 0)
                    else:
                        img_np = None
                else:
                    arr = np.asarray(img)
                    if arr.ndim == 4 and arr.shape[0] == 1:
                        arr = arr[0]
                    if arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
                        img_np = np.transpose(arr, (2, 0, 1))
                    elif arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
                        img_np = arr
                    elif arr.ndim == 2:
                        img_np = np.expand_dims(arr, 0)
                    else:
                        img_np = None
            except Exception as e:
                print("[warmup] image conversion error:", e)
                img_np = None

            imgs.append(img_np)

            # Agent feats / type / mask -> normalize to numpy with leading batch dim [B, N, ...]
            def _to_np(x):
                if x is None:
                    return None
                if hasattr(x, "detach"):
                    try:
                        a = x.detach().cpu().numpy()
                    except Exception:
                        a = np.array(x)
                else:
                    a = np.array(x)
                if a.ndim == 2:
                    a = a[np.newaxis, ...]
                if a.ndim == 1:
                    a = a[np.newaxis, :]
                return a

            a_feats_np = _to_np(obs.get("agent_feats", None))
            t_ids_np = _to_np(obs.get("type_id", None))
            mask_np = _to_np(obs.get("mask", None))

            agent_feats.append(a_feats_np)
            type_ids.append(t_ids_np)
            masks.append(mask_np)

            if len(imgs) > burn_in_T:
                imgs.pop(0)
                agent_feats.pop(0)
                type_ids.pop(0)
                masks.pop(0)

        # stack sequences (same as your previous correct code)
        seq_images = None
        if any(x is not None for x in imgs):
            ref = next(x for x in imgs if x is not None)
            stacked = []
            for x in imgs:
                if x is None:
                    stacked.append(np.zeros_like(ref, dtype=np.float32))
                else:
                    stacked.append(x.astype(np.float32, copy=False))
            seq_images = np.stack(stacked, axis=0)  # [T, C, H, W]

        seq_agent_feats = None
        if any(x is not None for x in agent_feats):
            ref = next(x for x in agent_feats if x is not None)
            B = ref.shape[0]
            N = ref.shape[1]
            F = ref.shape[2]
            stacked = []
            for x in agent_feats:
                if x is None:
                    stacked.append(np.zeros((B, N, F), dtype=np.float32))
                else:
                    stacked.append(x.astype(np.float32, copy=False))
            seq_agent_feats = np.stack(stacked, axis=0)  # [T, B, N, F]
            if seq_agent_feats.shape[1] == 1:
                seq_agent_feats = seq_agent_feats[:, 0, :, :]  # [T, N, F]

        seq_type_ids = None
        if any(x is not None for x in type_ids):
            ref = next(x for x in type_ids if x is not None)
            B = ref.shape[0]
            N = ref.shape[1]
            stacked = []
            for x in type_ids:
                if x is None:
                    stacked.append(np.zeros((B, N), dtype=np.int64))
                else:
                    stacked.append(x.astype(np.int64, copy=False))
            seq_type_ids = np.stack(stacked, axis=0)
            if seq_type_ids.shape[1] == 1:
                seq_type_ids = seq_type_ids[:, 0, :]

        seq_masks = None
        if any(x is not None for x in masks):
            ref = next(x for x in masks if x is not None)
            B = ref.shape[0]
            N = ref.shape[1]
            stacked = []
            for x in masks:
                if x is None:
                    stacked.append(np.zeros((B, N), dtype=np.float32))
                else:
                    stacked.append(x.astype(np.float32, copy=False))
            seq_masks = np.stack(stacked, axis=0)
            if seq_masks.shape[1] == 1:
                seq_masks = seq_masks[:, 0, :]

        seq = {
            "images": seq_images,  # [T, C, H, W] or None
            "agent_feats": seq_agent_feats,  # [T, N, F] (if B==1) or [T, B, N, F]
            "type_id": seq_type_ids,  # [T, N] or None
            "mask": seq_masks,  # [T, N] or None
        }

        return last_obs, {"obs_hist": list(self.obs_hist), "seq": seq}

    def _group_spawn_points_by_road_and_lane(self):
        lane_map = {}
        for spawn in self.spawn_points:
            waypoint = self.map.get_waypoint(spawn.location)
            road_id = waypoint.road_id
            lane_id = waypoint.lane_id

            key = (road_id, lane_id)
            if key not in lane_map:
                lane_map[key] = []
            lane_map[key].append(spawn)

        return lane_map

    def get_speed_limit(self, actor):
        """
        Return speed limit (m/s) for given actor.
        """
        if actor is None:
            return 15.0

        try:
            speed_limit_kmh = actor.get_speed_limit()
            return speed_limit_kmh / 3.6
        except Exception as e:
            print(f"[Warning] Failed to get speed limit, using default 15.0 m/s. Error: {e}")
            return 15.0  # fallback

    def _group_vehicles_by_road_and_lane(self):
        vehicle_map = {}
        for vehicle in [self.ego_vehicle] + self.vehicles:
            wp = self.map.get_waypoint(vehicle.get_location())
            key = (wp.road_id, wp.lane_id)
            if key not in vehicle_map:
                vehicle_map[key] = []
            vehicle_map[key].append((vehicle, wp.s))

        return vehicle_map

    #------------------- reset / step / obs / reward / done -------------------
    def reset(self):
        #print("[SHIELD] mode=", self.shield_mode)
        self.dead_slots = set()
        self.episode_num += 1

        self.episode_data = {
            "intervention": 0,
            "reward_components": [],
            "blend_log": [],
            "safety_costs": [],
            "step_count": 0,
        }
        # ensure last control mapping exists
        self._last_control = getattr(self, "_last_control", {})
        print("start to reset")
        self.destroy_all_actors()
        # self.destroy_sensors()
        #print("# 1) destroyed previous actors & sensors")

        self.vehicles = []
        self.pedestrians = []
        self.ped_controllers = {}
        self.slot2actor = [None] * self.max_slots
        self.agent_ids = []
        #print("# 2) reset bookkeeping containers")

        self.spawn_ego_vehicle()
        if self.spawn_background:
            if self.num_veh > 0:
                self.spawn_vehicles()
            if self.num_ped > 0:
                self.spawn_pedestrians()
        #print("# 3) spawned ego/vehicles/pedestrians")

        self.setup_sensors()
        #print("# 4) sensors attached")

        # ---- TM: reuse single instance created in __init__ ----
        tm = getattr(self, "traffic_manager", None)
        if tm is not None:
            try:
                tm.set_global_distance_to_leading_vehicle(2.0)
                tm.global_percentage_speed_difference(30.0)
            except Exception as e:
                print("[RESET] traffic_manager config failed:", e)
        else:
            print("[RESET] traffic_manager is None")

        # tick once to stabilize world
        try:
            self.world.tick()
        except Exception:
            time.sleep(0.01)

        self.vehicle_map = self._group_vehicles_by_road_and_lane()

        self.agent_ids = self.get_background_agents()
        print("===== RESET DEBUG =====")
        print("BEV radius:", getattr(self, "bev_radius", None))
        print("nearby agents:", len(self.agent_ids))
        print("========================")
        if len(self.agent_ids) == 0:
            print("[RESET] No agents, regenerating episode...")
            return self.reset()
        assert len(
            self.agent_ids) <= self.max_slots, f"agent_ids length {len(self.agent_ids)} > max_slots {self.max_slots}"

        for i, aid in enumerate(self.agent_ids):
            self.slot2actor[i] = aid
        self.rl_slots = list(range(len(self.agent_ids)))

        bev_img = None
        self.latest_bev_image = None
        try:
            # try to import the wrapper class
            from envs.observation.map_utils import BeVWrapper
        except Exception:
            # fallback import path if different; try project-local
            try:
                from observation.map_utils import BeVWrapper
            except Exception:
                BeVWrapper = None

        if BeVWrapper is not None:
            # reuse existing wrapper if present to avoid re-init cost
            bev_wrapper = getattr(self, "bev_wrapper", None)
            if bev_wrapper is None:
                try:
                    carla_map = getattr(self, "map", None)
                    if carla_map is None and hasattr(self, "world"):
                        try:
                            carla_map = self.world.get_map()
                        except Exception:
                            carla_map = None
                    bev_wrapper = BeVWrapper(carla_map, pixels_per_meter=getattr(self, "ppm", 5.0),
                                             bev_size=(int(getattr(self, "bev_H", 128)),
                                                       int(getattr(self, "bev_W", 128))))
                    # cache for reuse
                    self.bev_wrapper = bev_wrapper
                    self._bev_wrapper = bev_wrapper
                except Exception as e:
                    print("[RESET] failed to construct BeVWrapper:", e)
                    bev_wrapper = None

            # try a few times (tick between tries) to get bev image, call instance method correctly
            if bev_wrapper is not None:
                max_attempts = 6
                for attempt in range(1, max_attempts + 1):
                    try:
                        # IMPORTANT: call instance method and pass ego_vehicle
                        bev_img, meta = bev_wrapper.get_bev_data(self.ego_vehicle,
                                                                 radius_m=getattr(self, "bev_radius", 50.0))
                        if bev_img is not None:
                            break
                    except TypeError as te:
                        # very likely wrong call signature or missing arg — print and break (no retry will fix wrong signature)
                        print(f"[RESET] BeVWrapper.get_bev_data TypeError on attempt {attempt}: {te}")
                    except Exception as e:
                        # generic failure: print and retry after ticking
                        print(f"[RESET] BeVWrapper.get_bev_data attempt {attempt} failed: {e}")
                    # tick to allow sensors/map to settle
                    try:
                        self.world.tick()
                    except Exception:
                        time.sleep(0.02)
        else:
            print("[RESET] BeVWrapper class not found on import paths; cannot produce BEV")

        # post-process bev_img -> self.latest_bev_image
        if bev_img is None:
            self.latest_bev_image = None
            print(
                f"[RESET] BEVObservation did not produce BEV after attempts. latest_bev_image={type(self.latest_bev_image)}")
        else:
            import numpy as _np
            try:
                arr = _np.asarray(bev_img)
                # if BGR uint8 -> convert to float32 0..1 and to RGB if desired (keep BGR or convert depending on consumer)
                if arr.ndim == 3 and arr.shape[2] == 3 and arr.dtype == _np.uint8:
                    # convert to float32 0..1
                    arr = arr.astype(_np.float32, copy=False) / 255.0
                    # optional: convert BGR->RGB if your network expects RGB. If your prior pipeline assumed HxWx3 BGR, skip this.
                    # arr = arr[:, :, ::-1]  # uncomment to convert to RGB
                else:
                    # ensure float32
                    if arr.dtype != _np.float32:
                        arr = arr.astype(_np.float32, copy=False)
                    if arr.size > 0 and arr.max() > 1.01:
                        arr = arr / 255.0
                # ensure HWC
                if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] != arr.shape[2]:
                    arr = arr.transpose(1, 2, 0)
                if arr.ndim == 2:
                    arr = arr[:, :, None]
                bev_ch = int(getattr(self, "bev_ch", 3))
                if arr.ndim == 3 and arr.shape[2] != bev_ch:
                    if arr.shape[2] > bev_ch:
                        arr = arr[:, :, :bev_ch]
                    else:
                        arr = _np.repeat(arr[:, :, :1], bev_ch, axis=2)
                self.latest_bev_image = arr
                print(
                    f"[RESET] got BEV before warmup shape={arr.shape} dtype={arr.dtype} min={arr.min():.3f} max={arr.max():.3f}")
            except Exception as e:
                print("[RESET] postprocess BEV failed:", e)
                self.latest_bev_image = None

        # warm up and produce seq_obs
        last_obs, seq_obs = self.warmup(num_frames=20, burn_in_T=10)
        #print("# 6) world warmed up")

        # after warmup: ensure agents' autopilot/controllers are in desired state
        for actor_id in self.agent_ids:
            actor = self.world.get_actor(actor_id)
            if actor is None:
                continue
            if "vehicle" in actor.type_id:
                try:
                    actor.set_autopilot(False, self.tm_port)
                except Exception:
                    pass
            elif "walker.pedestrian" in actor.type_id or "pedestrian" in actor.type_id:
                controller = None
                try:
                    controller = self.ped_controllers.get(actor.id) if hasattr(self, "ped_controllers") else None
                except Exception:
                    controller = None

                if controller is not None:
                    try:
                        if hasattr(controller, "stop"):
                            controller.stop()
                    except Exception:
                        pass
                    try:
                        if hasattr(controller, "destroy"):
                            controller.destroy()
                    except Exception:
                        pass
                else:
                    try:
                        _ = actor.get_location()
                    except Exception:
                        pass

        self.collision_happened = False
        self.episode_info["id"] += 1
        self.episode_info["start_time"] = self.get_simulation_time()
        self.episode_info["end_time"] = self.get_simulation_time()
        self.episode_data = {}
        self._last_control = {}
        self.episode_data["collision_count"] = 0  # total collisions in this episode
        self.episode_data["collision_happened"] = 0  # boolean flag (0/1)
        self.episode_data["collision_by_slot"] = {}  # dict {slot_idx: count}
        self.episode_data["intervention"] = 0  # how many times shield/CBF intervened
        self.episode_data["step_count"] = 0  # step counter within episode
        self.episode_data["speed_sum"] = 0.0
        self.episode_data["acc_sum"] = 0.0
        self.episode_data["ttc_sum"] = 0.0
        self.episode_data["ttc_min"] = float("inf")
        self.episode_data["intervention_mag_sum"] = 0.0
        self.episode_data["agent_step_count"] = 0

        self.step_data = {"reward_components": []}
        self.last_location = self.ego_vehicle.get_location()
        self.start_time = self.get_simulation_time()
        self.distance_travelled = 0.0
        if hasattr(self, "_dbg_printed_step"):
            del self._dbg_printed_step
        return seq_obs

    def get_single_obs_for_manager(self, ego_id=None, radius=50.0):
        """
        Return a single-environment observation (no batch dim).
        - image:  (C, H, W) float32 in [0,1]
        - agent_feats: (N, F) float32
        - type_id: (N,) int64
        - mask: (N,) float32
        """
        B = 1
        N = int(self.max_slots)
        F = int(self.obs_dim)

        ego = self.ego_vehicle if ego_id is None else self.world.get_actor(ego_id)
        ego_loc = ego.get_location()

        slot2actor = list(self.slot2actor)

        # prefer BEV image if available
        bev = getattr(self, "latest_bev_image", None)
        if bev is None:
            bev = getattr(self, "latest_image", None)

        bev_H = int(getattr(self, "bev_H", 128))
        bev_W = int(getattr(self, "bev_W", 128))
        bev_ch = int(getattr(self, "bev_ch", 3))

        # --------- Convert/normalize image to torch [C, H, W] ----------
        try:
            arr = None
            if bev is None:
                arr = np.zeros((bev_H, bev_W, 3), dtype=np.float32)
            else:
                arr = np.asarray(bev)

            # if tuple/list like (img, meta)
            if isinstance(arr, (tuple, list)):
                arr = np.asarray(arr[0]) if len(arr) > 0 else np.zeros((bev_H, bev_W, 3), dtype=np.float32)

            # If CHW (C,H,W) -> convert to HWC
            if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[0] != arr.shape[2]:
                arr = np.transpose(arr, (1, 2, 0))

            # If 2D -> HxW, make 3 channels
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)

            # If HxW x 4 (RGBA) -> drop alpha
            if arr.ndim == 3 and arr.shape[2] == 4:
                arr = arr[..., :3]

            # Ensure arr is H x W x C
            if arr.ndim != 3:
                arr = np.zeros((bev_H, bev_W, 3), dtype=np.float32)

            # Resize if necessary
            h0, w0 = int(arr.shape[0]), int(arr.shape[1])
            if (h0, w0) != (bev_H, bev_W):
                try:
                    arr = cv2.resize(arr, (bev_W, bev_H), interpolation=cv2.INTER_LINEAR)
                except Exception:
                    arr = np.resize(arr, (bev_H, bev_W, arr.shape[2] if arr.ndim == 3 else 3))

            # Ensure 3 channels
            if arr.shape[2] == 1:
                arr = np.concatenate([arr] * 3, axis=2)
            elif arr.shape[2] > 3:
                arr = arr[..., :3]

            # Cast to float32 in [0,1]
            if np.issubdtype(arr.dtype, np.integer):
                arr_f = arr.astype(np.float32) / 255.0
            else:
                arr_f = arr.astype(np.float32)
                if arr_f.max() > 1.5:
                    arr_f = arr_f / 255.0

            # transpose to C,H,W
            arr_chw = np.transpose(arr_f, (2, 0, 1)).copy()  # C,H,W
            img_chw_t = torch.from_numpy(arr_chw).float()
        except Exception:
            # fallback black image
            img_chw_t = torch.zeros((bev_ch, bev_H, bev_W), dtype=torch.float32)

        # ---------------- build agent features / type / mask arrays (no batch dim) ----------------
        agent_feats = np.zeros((N, F), dtype=np.float32)

        type_ids = np.zeros((N,), dtype=np.int64)  # 0 = padding/empty, 1 = vehicle, 2 = pedestrian
        mask = np.zeros((N,), dtype=np.float32)


        for i, aid in enumerate(slot2actor):
            if aid is None:
                continue
            try:
                actor = self.world.get_actor(aid)
            except Exception:
                continue
            if actor is None:
                continue

            loc = actor.get_location()
            vel = actor.get_velocity()
            speed = (vel.x ** 2 + vel.y ** 2 + vel.z ** 2) ** 0.5
            rel_x = loc.x - ego_loc.x
            rel_y = loc.y - ego_loc.y

            feat = np.zeros((F,), dtype=np.float32)
            feat[0] = speed
            feat[1] = rel_x
            feat[2] = rel_y

            # write into agent_feats with boundary check
            max_write = min(agent_feats.shape[1], feat.shape[0])
            agent_feats[i, :max_write] = feat[:max_write]

            # type detection: 0 = empty/padding, 1 = vehicle, 2 = pedestrian
            t_id_str = getattr(actor, "type_id", "") or ""
            t_id_str = t_id_str.lower()
            if ("walker" in t_id_str) or ("pedestrian" in t_id_str):
                tval = 2
            else:
                tval = 1

            type_ids[i] = tval
            mask[i] = 1.0

        # ---------------- move to torch and to device if available ----------------
        device = getattr(self, "device", torch.device("cpu"))
        if isinstance(device, str):
            device = torch.device(device)

        single_obs = {
            "image": img_chw_t.to(device),  # (C, H, W)
            "agent_feats": torch.tensor(agent_feats, dtype=torch.float32, device=device),  # (N, F)
            "type_id": torch.tensor(type_ids, dtype=torch.long, device=device),  # (N,)
            "mask": torch.tensor(mask, dtype=torch.float32, device=device)  # (N,)
        }
        return single_obs

    def get_simulation_time(self):
        return self.world.get_snapshot().timestamp.elapsed_seconds

    def destroy_all_actors(self):
        self._destroying = True

        # 1️⃣ stop & destroy sensors FIRST
        self.destroy_sensors()

        # 2️⃣ disable autopilot for ALL vehicles (including ego)
        for v in list(getattr(self, "vehicles", [])):
            try:
                v.set_autopilot(False)
            except Exception:
                pass

        if getattr(self, "ego_vehicle", None) is not None:
            try:
                self.ego_vehicle.set_autopilot(False)
            except Exception:
                pass

        # ⭐ give TM one clean tick to release control
        try:
            self.world.tick()
        except Exception:
            time.sleep(0.05)

        # 3️⃣ destroy walker controllers
        for ped_id, ctrl in list(self.ped_controllers.items()):
            try:
                ctrl.stop()
                ctrl.destroy()
            except Exception:
                pass
        self.ped_controllers.clear()

        # 4️⃣ destroy vehicles
        for v in list(getattr(self, "vehicles", [])):
            try:
                v.destroy()
            except Exception:
                pass
        self.vehicles = []

        # 5️⃣ destroy ego
        if getattr(self, "ego_vehicle", None) is not None:
            try:
                self.ego_vehicle.destroy()
            except Exception:
                pass
            self.ego_vehicle = None

        # 6️⃣ destroy pedestrians
        for p in list(getattr(self, "pedestrians", [])):
            try:
                p.destroy()
            except Exception:
                pass
        self.pedestrians = []

        # 7️⃣ final tick ONLY ONCE
        try:
            self.world.tick()
        except Exception:
            pass

        self._destroying = False

    def destroy_sensors(self):
        if getattr(self, "camera_sensor", None) is not None:
            try:
                self.camera_sensor.stop()
            except Exception:
                pass
            try:
                self.camera_sensor.destroy()
            except Exception:
                pass
            self.camera_sensor = None

        if getattr(self, "collision_sensor", None) is not None:
            try:
                self.collision_sensor.stop()
            except Exception:
                pass
            try:
                self.collision_sensor.destroy()
            except Exception:
                pass
            self.collision_sensor = None

    def get_state(self):
        obs = BEVObservation(veh_id=self.ego_vehicle.id, time_stamp=self.get_simulation_time())
        obs.update(env=self)
        return obs.information

    def setup_sensors(self):
        # Collision sensor
        if self.collision_sensor is None:
            sensor_bp = self.blueprint_library.find('sensor.other.collision')
            sensor_transform = carla.Transform(carla.Location(x=0, y=0, z=2))
            self.collision_sensor = self.world.spawn_actor(
                sensor_bp, sensor_transform, attach_to=self.ego_vehicle
            )
            self.collision_sensor.listen(self._on_collision)

        # RGB camera sensor
        if self.camera_sensor is None:
            camera_bp = self.blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', '800')
            camera_bp.set_attribute('image_size_y', '600')
            camera_bp.set_attribute('fov', '90')

            camera_transform = carla.Transform(
                carla.Location(x=-6.5, y=0, z=2.5),
                carla.Rotation(pitch=-15, yaw=0, roll=0)
            )

            self.camera_sensor = self.world.spawn_actor(
                camera_bp, camera_transform, attach_to=self.ego_vehicle
            )
            self.camera_sensor.listen(self._process_camera_image)

        self._tick()
        self._tick()
        self._camera_ready = True

    def step(self, actions):
        """
        actions: dict mapping slot_idx -> action (e.g. [throttle, steer])
        Returns: next_obs, rewards, dones, info
        """

        applied_slots = []
        slot_worst_ttc = {}
        step_intervention_count = 0
        self.dead_slots = getattr(self, "dead_slots", set())

        TTC_MIN = getattr(self, "TTC_MIN", 0.8)
        TTC_WARN = getattr(self, "TTC_WARN", 2.0)

        self.episode_data.setdefault("step_count", 0)
        self.episode_data.setdefault("intervention", 0)

        active_agent_count = 0

        # ---------------------------------------------------------
        # Per-agent control loop
        # ---------------------------------------------------------

        for slot_idx, act in actions.items():
            if slot_idx in self.dead_slots:
                continue

            if slot_idx not in self.rl_slots:
                continue

            actor_id = self.slot2actor[slot_idx]
            if actor_id is None:
                continue

            actor = self.world.get_actor(actor_id)
            if actor is None or "vehicle" not in actor.type_id:
                continue

            active_agent_count += 1
            self.episode_data["agent_step_count"] += 1

            a0 = float(np.clip(act[0], -1.0, 1.0))
            a1 = float(np.clip(act[1], -1.0, 1.0))

            if a0 >= 0:
                thr = a0
                brk = 0.0
            else:
                thr = 0.0
                brk = -a0

            control = carla.VehicleControl()
            control.throttle = thr
            control.brake = brk
            control.steer = a1

            candidate_ids = [aid for aid in self.slot2actor if aid is not None]
            if self.ego_vehicle and self.ego_vehicle.id not in candidate_ids:
                candidate_ids.append(self.ego_vehicle.id)

            agent_loc = actor.get_location()
            agent_vel_obj = actor.get_velocity()
            speed = np.linalg.norm([agent_vel_obj.x, agent_vel_obj.y])
            self.episode_data["speed_sum"] += speed

            agent_pos = np.array([agent_loc.x, agent_loc.y], dtype=np.float32)
            agent_vel = np.array([agent_vel_obj.x, agent_vel_obj.y], dtype=np.float32)

            others_pos = []
            others_vel = []

            for other_id in candidate_ids:
                if other_id == actor_id:
                    continue

                other = self.world.get_actor(other_id)
                if other is None:
                    continue

                loc = other.get_location()
                vel = other.get_velocity()

                others_pos.append(np.array([loc.x, loc.y], dtype=np.float32))
                others_vel.append(np.array([vel.x, vel.y], dtype=np.float32))

            a_long = control.throttle - control.brake
            a_lat = control.steer
            a_agent_des = np.array([a_long, a_lat], dtype=np.float32)
            acc_mag = np.linalg.norm(a_agent_des)
            self.episode_data["acc_sum"] += acc_mag

            a_safe = self.safety_shield.apply(
                agent_pos,
                agent_vel,
                a_agent_des,
                others_pos,
                others_vel
            )

            if not np.allclose(a_safe, a_agent_des, atol=1e-3):
                mag = np.linalg.norm(a_safe - a_agent_des)
                self.episode_data["intervention_mag_sum"] += mag
                self.episode_data["intervention"] += 1
                step_intervention_count += 1

            control.throttle = float(np.clip(max(a_safe[0], 0.0), 0.0, 1.0))
            control.brake = float(np.clip(max(-a_safe[0], 0.0), 0.0, 1.0))
            control.steer = float(np.clip(a_safe[1], -1.0, 1.0))

            actor.apply_control(control)
            applied_slots.append(slot_idx)

            act_v = agent_vel
            worst_ttc = np.inf

            for other_id in candidate_ids:
                if other_id == actor_id:
                    continue

                other = self.world.get_actor(other_id)
                if other is None or "vehicle" not in other.type_id:
                    continue

                other_loc = other.get_location()
                other_vel = other.get_velocity()

                r = np.array([
                    other_loc.x - agent_loc.x,
                    other_loc.y - agent_loc.y
                ])

                dist = np.linalg.norm(r)
                if dist < 1e-3:
                    worst_ttc = 0.0
                    break

                r_hat = r / dist
                other_v = np.array([other_vel.x, other_vel.y])
                d_dot = np.dot((other_v - act_v), r_hat)

                if d_dot < -1e-3:
                    ttc = dist / (-d_dot)
                else:
                    ttc = np.inf

                worst_ttc = min(worst_ttc, ttc)

            slot_worst_ttc[slot_idx] = float(worst_ttc)
            self.episode_data["ttc_sum"] += worst_ttc
            self.episode_data["ttc_min"] = min(
                self.episode_data["ttc_min"],
                worst_ttc
            )

        # ---------------------------------------------------------
        # Advance simulation
        # ---------------------------------------------------------
        self.world.tick()
        self.episode_data["step_count"] += 1

        next_obs = self.get_single_obs_for_manager(
            ego_id=self.ego_vehicle.id,
            radius=200.0
        )

        rewards = {}
        dones = {}

        ego_id = getattr(self.ego_vehicle, "id", None)

        for slot_idx in self.rl_slots:

            actor_id = self.slot2actor[slot_idx]

            if actor_id is None:
                rewards[slot_idx] = 0.0
                dones[slot_idx] = False
                continue

            actor = self.world.get_actor(actor_id)
            if actor is None:
                rewards[slot_idx] = 0.0
                dones[slot_idx] = False
                continue
            # Background off-road reset
            if actor_id != ego_id:
                wp = self.map.get_waypoint(actor.get_location(), project_to_road=False)
                if wp is None:
                    self._bg_collision_slots = getattr(self, "_bg_collision_slots", set())
                    self._bg_collision_slots.add(slot_idx)
                    self.dead_slots.add(slot_idx)
                    dones[slot_idx] = True

            if slot_idx in self.dead_slots:
                rewards[slot_idx] = 0.0
                dones[slot_idx] = True
            else:
                base_reward = float(self._compute_reward(actor_id))
                rewards[slot_idx] = base_reward
                if slot_idx not in dones:
                    dones[slot_idx] = False

        # Ego-only termination
        if self._episode_done():
            for slot_idx in self.rl_slots:
                dones[slot_idx] = True
        # --- apply forced-done slots from callbacks (collision/resets) ---
        if hasattr(self, "_force_done_slots") and self._force_done_slots:
            for s in list(self._force_done_slots):
                if s in dones:
                    dones[s] = True
                self.dead_slots.add(s)
            self._force_done_slots.clear()

        info = {
            "applied_slots": applied_slots,
            # active_agent_count: number of agents that acted this step (used by trainer)
            "active_agent_count": int(active_agent_count),
            "step_count": int(self.episode_data.get("step_count", 0)),
            # per-step CBF/shield interventions this step (trainer expects 'intervention_this_step')
            "intervention_this_step": int(step_intervention_count),
            # legacy/backwards fields for diagnostics
            "cbf_step_interventions": int(step_intervention_count),
            "cbf_total_interventions": int(self.episode_data.get("intervention", 0))
        }

        if hasattr(self, "_bg_collision_slots"):
            self._bg_collision_slots.clear()

        return next_obs, rewards, dones, info

    def get_background_agents(self, radius=50.0):
        ego_loc = self.ego_vehicle.get_location()
        cand = []

        for v in self.vehicles:
            if v is None:
                continue
            try:
                d = ego_loc.distance(v.get_location())
            except RuntimeError:
                continue
            if d <= radius:
                cand.append((d, v.id))

        for p in self.pedestrians:
            if p is None:
                continue
            try:
                d = ego_loc.distance(p.get_location())
            except RuntimeError:
                continue
            if d <= radius:
                cand.append((d, p.id))

        if not cand:
            return []

        cand.sort(key=lambda x: x[0])
        K = min(self.max_slots, len(cand))
        return [aid for _, aid in cand[:K]]

    def _compute_reward(self, actor_id):
        actor = self.world.get_actor(actor_id)
        if actor is None or self.ego_vehicle is None:
            return 0.0

        # ======================
        # Basic kinematics
        # ======================
        vel = actor.get_velocity()
        v = np.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

        loc = actor.get_location()
        # store loc as tuple of floats to avoid object-reference aliasing
        loc_tuple = (float(loc.x), float(loc.y), float(loc.z))

        ego_loc = self.ego_vehicle.get_location()
        # ego loc tuple not strictly required, but keep consistent
        ego_loc_tuple = (float(ego_loc.x), float(ego_loc.y), float(ego_loc.z))
        dist = np.sqrt((loc_tuple[0] - ego_loc_tuple[0]) ** 2 + (loc_tuple[1] - ego_loc_tuple[1]) ** 2)

        wp = self.map.get_waypoint(loc, project_to_road=False)
        if wp is None:
            return -1.0

        # ======================
        # 1) Speed reward (bounded, ~0..1)
        # ======================
        speed_limit = self.get_speed_limit(actor)
        v_target = 0.9 * speed_limit
        v_tol = max(1.0, 0.2 * max(1.0, speed_limit))  # make tolerance relative to speed_limit

        r_speed = float(np.exp(-((v - v_target) ** 2) / (2 * (v_tol ** 2))))

        # ======================
        # 2) Direction alignment with lane (0..1)
        # ======================
        forward = wp.transform.get_forward_vector()
        fwd = np.array([forward.x, forward.y, forward.z])
        vel_vec = np.array([vel.x, vel.y, vel.z])

        if np.linalg.norm(vel_vec) > 1e-3:
            dir_align = np.dot(vel_vec, fwd) / (
                    np.linalg.norm(vel_vec) * np.linalg.norm(fwd) + 1e-6
            )
            dir_align = np.clip(dir_align, -1.0, 1.0)
        else:
            dir_align = 0.0

        r_direction = float(max(dir_align, 0.0))

        # ======================
        # 3) Lateral deviation penalty (smooth, small negative)
        # ======================
        wp_loc = wp.transform.location
        right = wp.transform.get_right_vector()

        right_vec = np.array([right.x, right.y, 0.0])
        rel = np.array([loc.x - wp_loc.x, loc.y - wp_loc.y, 0.0])
        lane_width = max(wp.lane_width, 1.0)
        lateral_dist = abs(np.dot(rel, right_vec))
        lateral_norm = lateral_dist / (0.5 * lane_width)
        # smooth quadratic penalty but small magnitude
        r_lane = -float(np.clip((lateral_norm ** 2) * 0.25, 0.0, 1.0))  # scaled down

        # ======================
        # 4) TTC-based safety penalty (continuous, bounded [-1,0])
        # ======================
        ego_vel = self.ego_vehicle.get_velocity()
        ego_v = np.array([ego_vel.x, ego_vel.y, ego_vel.z])
        rel_pos = np.array([loc.x - ego_loc.x, loc.y - ego_loc.y, 0.0])
        rel_vel = vel_vec - ego_v

        forward_unit = np.array([fwd[0], fwd[1], 0.0])
        norm_fwd = np.linalg.norm(forward_unit) + 1e-6
        forward_unit /= norm_fwd

        long_dist = np.dot(rel_pos, forward_unit)
        closing_speed = -np.dot(rel_vel, forward_unit)

        max_ttc_dist = 50.0
        eps = 1e-6

        # if not closing or out of range -> large safe ttc (no penalty)
        if long_dist <= 0.0 or long_dist > max_ttc_dist or closing_speed <= eps:
            ttc = np.inf
        else:
            ttc = float(long_dist / (closing_speed + eps))

        # continuous penalty: -1/(ttc + 1) mapped to (-1,0]; clip at -1
        if np.isfinite(ttc):
            r_dist = -float(np.clip(1.0 / (ttc + 1.0), 0.0, 1.0))
        else:
            r_dist = 0.0

        # ======================
        # 5) Attitude stability penalty (very small)
        # ======================
        tr = actor.get_transform()
        roll = abs(tr.rotation.roll)
        pitch = abs(tr.rotation.pitch)
        r_att = -0.005 * ((roll / 45.0) ** 2) - 0.005 * ((pitch / 45.0) ** 2)
        r_att = float(r_att)

        # ======================
        # Collision / Off-road penalty (conservative)
        # ======================
        penalty = 0.0
        if hasattr(self, "slot2actor") and actor_id in self.slot2actor:
            try:
                slot_idx = self.slot2actor.index(actor_id)
                if hasattr(self, "_bg_collision_slots") and slot_idx in self._bg_collision_slots:
                    penalty = -1.0  # less severe than -1.5
            except ValueError:
                pass

        # ======================
        # Progress reward (normalized by max possible per step)
        # ======================
        key_loc = f"last_loc_{actor_id}"
        prev_loc = self.episode_data.get(key_loc, None)
        if prev_loc is None:
            # store numeric tuple, no reward this first step
            self.episode_data[key_loc] = loc_tuple
            progress_reward = 0.0
        else:
            dx = loc_tuple[0] - prev_loc[0]
            dy = loc_tuple[1] - prev_loc[1]
            delta_vec = np.array([dx, dy, 0.0])
            lane_fwd = np.array([fwd[0], fwd[1], 0.0])
            delta_s = np.dot(delta_vec, lane_fwd)
            delta_s = max(delta_s, 0.0)

            max_step_progress = max(1e-3, speed_limit * (getattr(self, "step_size", 0.1)))
            progress_reward = float(np.clip(delta_s / (max_step_progress + 1e-6), 0.0, 1.0))

            # update stored prev_loc
            self.episode_data[key_loc] = loc_tuple

        # if collision penalty applied, subtract a fixed amount from progress (not multiply)
        if penalty < 0.0 and progress_reward > 0.0:
            progress_reward = max(0.0, progress_reward - 0.2)

        # ======================
        # 6) Longitudinal smoothness penalty (jerk-like, small constant weight)
        # ======================
        key_acc = f"last_acc_{actor_id}"
        key_v = f"last_speed_{actor_id}"

        prev_v = self.episode_data.get(key_v, v)
        prev_acc = self.episode_data.get(key_acc, 0.0)

        dt = max(getattr(self, "step_size", 0.1), 1e-3)
        acc = (v - prev_v) / dt

        delta_acc = acc - prev_acc
        w_smooth = 0.01  # small fixed weight (schedule externally if desired)
        r_smooth = -w_smooth * float(min(delta_acc * delta_acc, 4.0))

        self.episode_data[key_v] = v
        self.episode_data[key_acc] = acc

        # ======================
        # Final weighted reward (weights chosen to keep magnitude ~[-2,2])
        # ======================
        reward = (
                0.30 * r_speed +  # encourage near-target speed
                0.30 * r_direction +  # keep aligned with lane
                0.10 * r_lane +  # small lateral penalty
                0.25 * r_dist +  # safety penalty (negative)
                0.10 * r_att +  # tiny attitude penalty
                0.0 * penalty  # keep collision as separate diagnostic; apply externally if needed
        )

        # add progress and smoothness but with smaller weight
        reward = reward + 0.25 * progress_reward + 1.0 * r_smooth

        # final clipping for numerical stability
        reward = float(np.clip(reward, -3.0, 3.0))

        # ======================
        # Logging reward components
        # ======================

        if "reward_components" not in self.step_data:
            self.step_data["reward_components"] = []

        self.step_data["reward_components"].append({
            "speed": r_speed,
            "direction": r_direction,
            "lane": r_lane,
            "ttc": r_dist,
            "attitude": r_att,
            "progress": progress_reward,
            "smooth": r_smooth,
            "collision_penalty": penalty,
            "total": reward
        })
        return float(reward)
    #
    # def _compute_reward(self, actor_id):
    #     actor = self.world.get_actor(actor_id)
    #     if actor is None or self.ego_vehicle is None:
    #         return 0.0
    #
    #     # ======================
    #     # Basic kinematics
    #     # ======================
    #     vel = actor.get_velocity()
    #     v = np.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
    #
    #     loc = actor.get_location()
    #     ego_loc = self.ego_vehicle.get_location()
    #     dist = ego_loc.distance(loc)
    #
    #     wp = self.map.get_waypoint(loc, project_to_road=False)
    #     if wp is None:
    #         return -1.0
    #
    #     # ======================
    #     # 1) Speed reward (bounded)
    #     # ======================
    #     speed_limit = self.get_speed_limit(actor)
    #     v_target = 0.9 * speed_limit
    #     v_tol = 2.0
    #
    #     r_speed = np.exp(-((v - v_target) ** 2) / (2 * v_tol ** 2))
    #
    #     # ======================
    #     # 2) Direction alignment with lane
    #     # ======================
    #     forward = wp.transform.get_forward_vector()
    #     fwd = np.array([forward.x, forward.y, forward.z])
    #     vel_vec = np.array([vel.x, vel.y, vel.z])
    #
    #     if np.linalg.norm(vel_vec) > 1e-3:
    #         dir_align = np.dot(vel_vec, fwd) / (
    #                 np.linalg.norm(vel_vec) * np.linalg.norm(fwd) + 1e-6
    #         )
    #         dir_align = np.clip(dir_align, -1.0, 1.0)
    #     else:
    #         dir_align = 0.0
    #
    #     r_direction = max(dir_align, 0.0)
    #
    #     # ======================
    #     # 3) Lateral deviation penalty
    #     # ======================
    #     wp_loc = wp.transform.location
    #     right = wp.transform.get_right_vector()
    #
    #     right_vec = np.array([right.x, right.y, 0.0])
    #     rel = np.array([loc.x - wp_loc.x, loc.y - wp_loc.y, 0.0])
    #     lane_width = max(wp.lane_width, 1.0)
    #     lateral_dist = abs(np.dot(rel, right_vec))
    #     lateral_norm = lateral_dist / (0.5 * lane_width)
    #
    #     r_lane = -np.clip(lateral_norm ** 2, 0.0, 4.0)
    #
    #     # ======================
    #     # 4) TTC-based safety penalty (NEW)
    #     # ======================
    #     # relative speed along lane direction
    #     ego_vel = self.ego_vehicle.get_velocity()
    #     ego_v = np.array([ego_vel.x, ego_vel.y, ego_vel.z])
    #     # ego_vel and ego_v already computed
    #     rel_pos = np.array([loc.x - ego_loc.x, loc.y - ego_loc.y, 0.0])
    #     rel_vel = vel_vec - ego_v
    #
    #     # forward unit vector (2D)
    #     forward_unit = np.array([fwd[0], fwd[1], 0.0])
    #     norm_fwd = np.linalg.norm(forward_unit) + 1e-6
    #     forward_unit /= norm_fwd
    #
    #     # longitudinal distance (projection of rel_pos onto forward axis)
    #     long_dist = np.dot(rel_pos, forward_unit)
    #     # relative speed along forward axis (positive = closing)
    #     closing_speed = -np.dot(rel_vel, forward_unit)
    #
    #     # stability eps and max sensing distance
    #     eps = 1e-3
    #     max_ttc_dist = 50.0  # ignore vehicles farther than ~50 m
    #
    #     if long_dist <= 0.0 or long_dist > max_ttc_dist:
    #         ttc = np.inf
    #     else:
    #         if closing_speed > eps:
    #             ttc = long_dist / closing_speed
    #         else:
    #             ttc = np.inf
    #
    #     # TTC penalty: keep it smooth and clipped
    #     ttc_safe = 4.0  # consider increasing to 4s for earlier braking
    #     if ttc < ttc_safe:
    #         # scale and clip penalty to avoid huge negatives
    #         r_dist = -np.clip(((ttc_safe - ttc) / ttc_safe) ** 2, 0.0, 1.0)
    #     else:
    #         r_dist = 0.0
    #
    #     # ======================
    #     # 5) Attitude stability penalty
    #     # ======================
    #     tr = actor.get_transform()
    #     roll = abs(tr.rotation.roll)
    #     pitch = abs(tr.rotation.pitch)
    #
    #     r_att = -0.01 * (roll / 45.0) ** 2 - 0.01 * (pitch / 45.0) ** 2
    #
    #     # ======================
    #     # Collision / Off-road penalty
    #     # ======================
    #     penalty = 0.0
    #     if hasattr(self, "slot2actor") and actor_id in self.slot2actor:
    #         try:
    #             slot_idx = self.slot2actor.index(actor_id)
    #             if hasattr(self, "_bg_collision_slots") and slot_idx in self._bg_collision_slots:
    #                 penalty = -1.5
    #         except ValueError:
    #             pass
    #
    #     # ======================
    #     # Final weighted reward
    #     # ======================
    #     reward = (
    #             0.35 * r_speed +
    #             0.35 * r_direction +
    #             0.1 * r_lane +
    #             0.3 * r_dist +
    #             1.0 * r_att +
    #             penalty
    #     )
    #     # ===== Progress reward =====
    #     key_loc = f"last_loc_{actor_id}"
    #     prev_loc = self.episode_data.get(key_loc, loc)
    #
    #     dx = loc.x - prev_loc.x
    #     dy = loc.y - prev_loc.y
    #
    #     delta_vec = np.array([dx, dy, 0.0])
    #     lane_fwd = np.array([fwd[0], fwd[1], 0.0])
    #
    #     delta_s = np.dot(delta_vec, lane_fwd)
    #     delta_s = max(delta_s, 0.0)
    #
    #     # normalize by maximum possible step distance
    #     max_step_progress = speed_limit * self.step_size
    #
    #     progress_reward = np.clip(delta_s / (max_step_progress + 1e-6), 0.0, 1.0)
    #     # delta_s = np.dot(delta_vec, lane_fwd)
    #     # delta_s = max(delta_s, 0.0)
    #     #
    #     # # simple progress reward (meters traveled forward)
    #     # # scale down since typical progress per step is 0-1 meters
    #     # progress_reward = np.clip(delta_s * 0.1, 0.0, 1.0)
    #
    #     if penalty < 0:
    #         progress_reward *= 0.2
    #
    #     reward += progress_reward
    #     self.episode_data[key_loc] = loc
    #
    #     # ======================
    #     # 6) Longitudinal smoothness penalty (NEW)
    #     # ======================
    #     key_acc = f"last_acc_{actor_id}"
    #     key_v = f"last_speed_{actor_id}"
    #
    #     prev_v = self.episode_data.get(key_v, v)
    #     prev_acc = self.episode_data.get(key_acc, 0.0)
    #
    #     dt = self.step_size
    #     acc = (v - prev_v) / max(dt, 1e-3)
    #
    #     # jerk-like penalty (penalize rapid acc change)
    #     # step-based annealing for smoothness
    #     t = self.episode_data.get("global_step", 0)
    #     self.episode_data["global_step"] = t + 1
    #
    #     w_smooth = min(0.05, 0.05 * t / 5000.0)  #
    #     delta_acc = acc - prev_acc
    #     r_smooth = -w_smooth * min(delta_acc * delta_acc, 4.0)
    #
    #     # update history
    #     self.episode_data[key_v] = v
    #     self.episode_data[key_acc] = acc
    #
    #     # add to reward
    #     reward += r_smooth
    #     reward = np.clip(reward, -3.0, 3.0)
    #
    #     return float(reward)
    #
    # def _compute_reward_components(self, actor_id):
    #     """
    #     Logging only.
    #     DOES NOT affect training.
    #     """
    #     actor = self.world.get_actor(actor_id)
    #     if actor is None or self.ego_vehicle is None:
    #         return None
    #
    #     vel = actor.get_velocity()
    #     v = np.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
    #
    #     loc = actor.get_location()
    #     ego_loc = self.ego_vehicle.get_location()
    #     dist = ego_loc.distance(loc)
    #
    #     wp = self.map.get_waypoint(loc, project_to_road=False)
    #     if wp is None:
    #         return None
    #
    #     # ---------- speed ----------
    #     v_target = 8.0
    #     v_tol = 2.0
    #     r_speed = 0.4 * np.exp(-((v - v_target) ** 2) / (2 * v_tol ** 2))
    #
    #     # ---------- direction ----------
    #     forward = wp.transform.get_forward_vector()
    #     fwd = np.array([forward.x, forward.y, forward.z])
    #     vel_vec = np.array([vel.x, vel.y, vel.z])
    #
    #     if np.linalg.norm(vel_vec) > 1e-3:
    #         dir_align = np.dot(vel_vec, fwd) / (
    #                 np.linalg.norm(vel_vec) * np.linalg.norm(fwd) + 1e-6
    #         )
    #         dir_align = np.clip(dir_align, -1.0, 1.0)
    #     else:
    #         dir_align = 0.0
    #
    #     r_direction = 0.6 * max(dir_align, 0.0)
    #
    #     # ---------- lane ----------
    #     lane_center = wp.transform.location
    #     lane_width = max(wp.lane_width, 1.0)
    #     lateral_dist = lane_center.distance(loc)
    #     lateral_norm = lateral_dist / (0.5 * lane_width)
    #     r_lane = 0.5 * (-np.clip(lateral_norm ** 2, 0.0, 4.0))
    #
    #     # ---------- TTC ----------
    #     ego_vel = self.ego_vehicle.get_velocity()
    #     ego_v = np.array([ego_vel.x, ego_vel.y, ego_vel.z])
    #     rel_vel = ego_v - vel_vec
    #     closing_speed = np.dot(rel_vel, fwd)
    #
    #     if closing_speed > 1e-3:
    #         ttc = dist / closing_speed
    #     else:
    #         ttc = np.inf
    #
    #     if ttc < 3.0:
    #         r_dist = 0.8 * (-((3.0 - ttc) / 3.0) ** 2)
    #     else:
    #         r_dist = 0.0
    #
    #     # ---------- attitude ----------
    #     tr = actor.get_transform()
    #     roll = abs(tr.rotation.roll)
    #     pitch = abs(tr.rotation.pitch)
    #     r_att = -0.01 * (roll / 45.0) ** 2 - 0.01 * (pitch / 45.0) ** 2
    #
    #     return {
    #         "speed": r_speed,
    #         "direction": r_direction,
    #         "lane": r_lane,
    #         "ttc": r_dist,
    #         "att": r_att,
    #         "sum": r_speed + r_direction + r_lane + r_dist + r_att
    #     }

    def _check_done(self, actor_id):
        actor = self.world.get_actor(actor_id)
        if actor is None:
            return True

        # Only ego can terminate episode
        if actor_id == getattr(self.ego_vehicle, "id", None):
            wp = self.map.get_waypoint(actor.get_location(), project_to_road=False)
            if wp is None:
                return True

        return False

    def _episode_done(self):
        if self.collision_happened:
            return True

        wp = self.map.get_waypoint(self.ego_vehicle.get_location(), project_to_road=False)
        if wp is None:
            return True

        sim_time = self.get_simulation_time()
        if sim_time - self.episode_info["start_time"] > 60.0:
            return True

        return False

    def _tick(self, timeout_seconds=None):
        if self.world.get_settings().synchronous_mode:
            if timeout_seconds is None:
                return self.world.tick()
            else:
                return self.world.tick(timeout_seconds)
        else:
            if timeout_seconds is None:
                return self.world.tick()
            else:
                if timeout_seconds is None:
                    return self.world.tick()
                else:
                    return self.world.tick()

    def update_latest_bev(self):
        """
        Generate latest BEV image and store as:
          - self.latest_bev_image : H,W,C float32 in [0,1]  (or None if failed)
          - self.latest_image     : same copy for backward compatibility (or None)
        """
        try:
            from observation.obs_bev import BEVObservation
        except Exception:
            BEVObservation = None

        bev_img = None

        # 1) try BEVObservation helper if available
        if BEVObservation is not None:
            try:
                ego_id = getattr(self, "ego_vehicle", None).id if getattr(self, "ego_vehicle", None) else None
                timestamp = self.get_simulation_time() if hasattr(self, "get_simulation_time") else None
                bev = BEVObservation(ego_id=ego_id, timestamp=timestamp)
                # try common update APIs
                try:
                    bev.update(env=self)
                except Exception:
                    try:
                        bev.generate(self)
                    except Exception:
                        pass

                # try to extract image from common attribute names
                for attr in ("image", "bev_image", "bev", "arr"):
                    if hasattr(bev, attr):
                        bev_img = getattr(bev, attr)
                        break
            except Exception:
                bev_img = None

        # 2) fallback to env helper build_bev if exists
        if bev_img is None and hasattr(self, "build_bev"):
            try:
                bev_img = self.build_bev()
            except Exception:
                bev_img = None

        # 3) if still None -> clear and return
        if bev_img is None:
            self.latest_bev_image = None
            self.latest_image = None
            return

        # 4) normalize and ensure HWC float32 in [0,1]
        bev_arr = np.asarray(bev_img)
        if bev_arr.size == 0:
            self.latest_bev_image = None
            self.latest_image = None
            return

        # ensure at least H,W,C
        if bev_arr.ndim == 2:
            bev_arr = np.expand_dims(bev_arr, axis=2)
        # resize to expected BEV dims if mismatch
        try:
            target_h = int(getattr(self, "bev_H", bev_arr.shape[0]))
            target_w = int(getattr(self, "bev_W", bev_arr.shape[1]))
        except Exception:
            target_h, target_w = bev_arr.shape[0], bev_arr.shape[1]

        if (bev_arr.shape[0], bev_arr.shape[1]) != (target_h, target_w):
            try:
                import cv2
                bev_arr = cv2.resize(bev_arr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            except Exception:
                bev_arr = np.resize(bev_arr, (target_h, target_w, bev_arr.shape[2] if bev_arr.ndim == 3 else 1))

        # channels
        bev_ch = int(getattr(self, "bev_ch", bev_arr.shape[2] if bev_arr.ndim == 3 else 1))
        if bev_arr.ndim == 3 and bev_arr.shape[2] != bev_ch:
            if bev_arr.shape[2] > bev_ch:
                bev_arr = bev_arr[:, :, :bev_ch]
            else:
                bev_arr = np.repeat(bev_arr[:, :, :1], bev_ch, axis=2)

        # dtype -> float32 and normalize to [0,1] if necessary
        if bev_arr.dtype != np.float32:
            bev_arr = bev_arr.astype(np.float32, copy=False)
        # safe normalization: if max>1 assume 0-255 scale
        max_val = float(bev_arr.max()) if bev_arr.size > 0 else 0.0
        if max_val > 1.0:
            bev_arr = bev_arr / 255.0

        # store canonical copies
        self.latest_bev_image = bev_arr  # H,W,C float32 in [0,1]
        # keep a separate copy for backward compatibility
        self.latest_image = bev_arr.copy()

    def _process_camera_image(self, image):
        if getattr(self, "_destroying", False):
            return
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1]
        self.latest_camera_image = array

    def _on_collision(self, event):
        """
        Safe collision handler — minimal-change replacement.

        Behavior preserved:
          - increment episode collision_count
          - if ego involved -> set self.collision_happened True (episode termination)
          - if background agent collided -> mark slot in _bg_collision_slots and reset that agent
          - never raise NameError for slot_idx (compute it here)
          - avoid double-reset / duplicated logic
        """
        # ignore callbacks while destroying
        if getattr(self, "_destroying", False):
            return

        # record a collision happened in episode stats
        # self.episode_data["collision_count"] = self.episode_data.get("collision_count", 0) + 1

        # try to obtain the "other" actor from event
        other = getattr(event, "other_actor", None) or getattr(event, "actor", None)
        if other is None:
            return

        other_id = other.id if hasattr(other, "id") else None
        if other_id is None:
            return

        ego_id = getattr(self.ego_vehicle, "id", None)

        # If ego is one of the colliding actors -> terminate episode
        # (some CARLA events may present 'other' as the non-ego; check both sides conservatively)
        if other_id == ego_id:
            self.collision_happened = True
            return

        # compute slot_idx by reverse-mapping slot2actor (slot2actor contains actor ids)
        slot_idx = None
        try:
            if hasattr(self, "slot2actor") and self.slot2actor is not None:
                for s, aid in enumerate(self.slot2actor):
                    # skip empty slots
                    if aid is None:
                        continue
                    try:
                        if int(aid) == int(other_id):
                            slot_idx = s
                            break
                    except Exception:
                        if aid == other_id:
                            slot_idx = s
                            break
        except Exception:
            slot_idx = None

        # If background agent collided -> mark slot and reset that agent
        if slot_idx is not None:
            self.episode_data.setdefault("_collision_slots", set())
            self.episode_data["_collision_slots"].add(slot_idx)

            self._bg_collision_slots = getattr(self, "_bg_collision_slots", set())
            self._bg_collision_slots.add(slot_idx)
            self._force_done_slots = getattr(self, "_force_done_slots", set())
            self._force_done_slots.add(slot_idx)
            try:
                self._reset_background_agent(other_id)
            except Exception as e:
                print(f"[WARN] _reset_background_agent failed for actor {other_id}: {e}")
            # done
            return

        # If we reach here: other actor is not mapped to any slot and not the ego.
        # Best-effort: try resetting the actor by id (non-slot background actor)
        try:
            self._reset_background_agent(other_id)
        except Exception as e:
            # swallow errors so callback never crashes
            print(f"[WARN] _reset_background_agent fallback failed for actor {other_id}: {e}")

    def _reset_background_agent(self, actor_id):
        actor = self.world.get_actor(actor_id)
        if actor is None:
            return

        wp = self.map.get_waypoint(actor.get_location(), project_to_road=True)
        if wp is None:
            return

        new_transform = wp.transform
        new_transform.location.z += 0.5

        try:
            actor.set_transform(new_transform)
            actor.set_velocity(carla.Vector3D(0, 0, 0))
            actor.set_angular_velocity(carla.Vector3D(0, 0, 0))
        except Exception:
            pass
