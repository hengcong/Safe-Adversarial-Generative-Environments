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
import  collections
from typing import Dict, List, Tuple, Optional
import carla

#import pygame
from gym import spaces, core

import torch
from .observation.obs_bev import BEVObservation
from algo.mappo.mappo_manager import MAPPOManager
from models.mappo_policy import MAPPOPolicy



class CarlaEnv(gym.Env):
    def __init__(self, num_veh, num_ped, mode="MAPPO", *, render: bool = False, spawn_background: bool = True):

        super(CarlaEnv, self).__init__()
        self.mode = mode
        # rendering / spawn control flags (delayed init)
        self.render = bool(render)
        self.spawn_background = bool(spawn_background)

        # pygame display state (will be initialized lazily if self.render True)
        self.pygame_display_initialized = False
        self.screen = None
        self.font = None
        self._last_time = 0.0
        self._fps = 0.0

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

        self.agent_specs = {
            "vehicle": {
                "obs_dim": 128,
                "act_dim": 4,
                "buffer_T": 128,
                "num_envs": 1,
                "n_agents": self.num_veh,
                "buffer_size": 2048,
            },
            "pedestrian": {
                "obs_dim": 128,
                "act_dim": 2,
                "buffer_T": 128,
                "num_envs": 1,
                "n_agents": self.num_ped,
                "buffer_size": 2048,
            }
        }

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

        for _ in range(3):
            self.world.tick()

        # actor containers (no wrappers)
        self.ego_vehicle = None  # carla.Actor
        self.vehicles = []  # list[carla.Actor] (background veh)
        self.pedestrians = []  # list[carla.Actor]
        self.slot2actor = None
        self.agent_ids = []
        self.rl_slots = None

        # controller mapping: actor_id -> controller instance
        self.controllers = {}

        # bookkeeping
        self.collision_happened = False
        self.start_time = 0.0
        self.distance_travelled = 0.0
        self.last_location = None

        self.mappo_manager = MAPPOManager(
            self.agent_specs,
            lambda spec: MAPPOPolicy(
                spec["obs_dim"],
                spec["act_dim"],  # pass same act_dim for vehicle
                spec["act_dim"]  # pass same act_dim for pedestrian
            ),
            device="cpu"
        )

        # sensors / rendering
        self.camera_sensor = None
        self.latest_image = None
        self.pygame_display_initialized = False
        self.screen = None
        self.font = None
        self._last_time = 0.0
        self._fps = 0.0

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
        self.traffic_manager = self.client.get_trafficmanager(8000)

        # --- spawn points / lane map ---
        self.spawn_points = self.map.get_spawn_points()
        self.lane_map = self._group_spawn_points_by_road_and_lane()

        self.max_slots = 12
        self.obs_dim = 16
        self.bev_ch = 3
        self.bev_H = 84
        self.bev_W = 84
        self.act_dim = 4

        self.spawn_ego_vehicle()
        if self.num_veh > 0:
            self.spawn_vehicles()
        if self.num_ped > 0:
            self.spawn_pedestrians()

        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0.0, high=1.0, shape=(self.bev_ch, self.bev_H, self.bev_W), dtype=np.float32),
            "agent_feats": spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_slots, self.obs_dim), dtype=np.float32),
            "type_id": spaces.Box(low=0, high=10, shape=(self.max_slots,), dtype=np.int64),  # small vocab
            "mask": spaces.Box(low=0.0, high=1.0, shape=(self.max_slots,), dtype=np.float32),
        })

        low = np.tile(np.array([-1.0, -1.0, 0.0, 0.0], dtype=np.float32), (self.max_slots, 1))
        high = np.tile(np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32), (self.max_slots, 1))

        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Attach simple sensors (ensure setup_sensors doesn't reference removed modules)
        self.setup_sensors()

        # tick & warmup
        self.world.tick()
        self.warmup(num_frames=20)

        # --- experiment path & minimal logger (timestamped) ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_path = f"./carla_experiments/run_{timestamp}"
        os.makedirs(self.experiment_path, exist_ok=True)

        # --- episode bookkeeping ---
        self.collision_happened = False
        self.episode_info = {"id": 0, "start_time": self.get_simulation_time(), "end_time": self.get_simulation_time()}
        print(self.episode_info)
        self.episode_data = {}
        self.step_data = {}

        # --- controllers placeholders (empty dicts; you can add controllers later) ---
        self.ego_controller = None
        self.bv_controllers = {}

    def spawn_ego_vehicle(self, bp_filter="vehicle.tesla.model3", spawn_retry=30, tm_autopilot=True):
        """
        Spawn ego vehicle at a random map spawn point.
        Ego will be controlled by the TrafficManager if tm_autopilot=True.
        """
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

    def spawn_vehicles(self, num_vehicles=None, spawn_per_lane_limit=3, use_tm=True):
        """
        Spawn background vehicles managed by the TrafficManager.
        """
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
                    except Exception:
                        pass

    def spawn_pedestrians(self, num_pedestrians=None):
        """
        Spawn pedestrians and attach AI walker controllers.
        """
        if num_pedestrians is None:
            num_pedestrians = int(getattr(self, "num_ped", 10))

        walker_bps = list(self.blueprint_library.filter("walker.pedestrian.*"))
        if not walker_bps:
            return

        spawn_locs = []
        attempts = num_pedestrians * 6
        for _ in range(attempts):
            loc = self.world.get_random_location_from_navigation()
            if loc is None:
                continue
            spawn_locs.append(loc)
            if len(spawn_locs) >= num_pedestrians:
                break

        for loc in spawn_locs[:num_pedestrians]:
            bp = random.choice(walker_bps)
            if bp.has_attribute("role_name"):
                bp.set_attribute("role_name", "autogen")
            tf = carla.Transform(loc, carla.Rotation(yaw=random.uniform(-180, 180)))

            ped = self.world.try_spawn_actor(bp, tf)
            if not ped:
                continue
            self.pedestrians.append(ped)

            try:
                ai_bp = self.blueprint_library.find("controller.ai.walker")
                ctrl = self.world.spawn_actor(ai_bp, carla.Transform(), attach_to=ped)
                ctrl.start()
                target = self.world.get_random_location_from_navigation()
                if target:
                    ctrl.go_to_location(target)
                ctrl.set_max_speed(random.uniform(0.5, 1.5))
                self.controllers[ctrl.id] = ctrl
            except Exception:
                pass

    def warmup(self, num_frames=40):
        """
        Warmup world and populate slot2actor & agent_ids (slot indices).
        Keeps at most self.max_slots agents.
        """
        for _ in range(int(num_frames)):
            self.world.tick()

        # update cache
        self.vehicle_map = self._group_vehicles_by_road_and_lane()

        # select nearby actor ids (vehicles only) up to max_slots
        nearby_actor_ids = self.get_background_agents(radius=50.0)[:self.max_slots]

        # build slot2actor: fixed length list of max_slots, None where empty
        self.slot2actor = [None] * self.max_slots
        for i, aid in enumerate(nearby_actor_ids):
            self.slot2actor[i] = aid

        # agent_ids are active slot indices (trainer uses these slots)
        self.agent_ids = [i for i, a in enumerate(self.slot2actor) if a is not None]
        self.rl_slots = self.agent_ids.copy()

        # sync with manager if available
        if getattr(self, "mappo_manager", None) is not None:
            try:
                self.mappo_manager.sync_agents_with_selected(self.agent_ids)
            except Exception:
                pass
        return

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

    def _group_vehicles_by_road_and_lane(self):
        vehicle_map = {}
        for vehicle in [self.ego_vehicle]+self.vehicles:
            wp = self.map.get_waypoint(vehicle.get_location())
            key = (wp.road_id, wp.lane_id)
            if key not in vehicle_map:
                vehicle_map[key] = []
            vehicle_map[key].append((vehicle, wp.s))

        return vehicle_map

    def reset(self):
        # 1) destroy previous actors & sensors (ensure clean start)
        print("start to reset")

        self.destroy_all_actors()
        self.destroy_sensors()
        print("# 1) destroyed previous actors & sensors")

        # 2) reset bookkeeping
        self.vehicles = []
        self.pedestrians = []
        self.controllers = {}
        self.slot2actor = [None] * self.max_slots
        self.agent_ids = []
        print("# 2) reset bookkeeping containers")

        # 3) spawn world actors
        self.spawn_ego_vehicle()
        if self.num_veh > 0:
            self.spawn_vehicles()
        if self.num_ped > 0:
            self.spawn_pedestrians()
        print("# 3) spawned ego/vehicles/pedestrians")

        # 4) sensors
        self.setup_sensors()
        print("# 4) sensors attached")

        # 5) traffic manager config
        tm_port = getattr(self, "tm_port", 8000)
        tm = self.client.get_trafficmanager(tm_port)
        tm.set_global_distance_to_leading_vehicle(2.0)
        tm.set_synchronous_mode(True)
        tm.global_percentage_speed_difference(30.0)
        print("# 5) TM configured")

        # 6) warmup -> this will populate self.slot2actor & self.agent_ids (slot indices)
        self.world.tick()
        self.vehicle_map = self._group_vehicles_by_road_and_lane()
        self.warmup(num_frames=20)
        print("# 6) world warmed up")

        # 7) ensure slot list no longer than max_slots (already enforced in warmup)
        # Fill background: make TM control all vehicles by default, then disable autopilot for RL-selected slots
        tm_port = getattr(self, "tm_port", 8000)
        tm = self.client.get_trafficmanager(tm_port)
        id2actor = {v.id: v for v in self.vehicles}

        # default: let TM control all vehicles
        for v in self.vehicles:
            try:
                v.set_autopilot(True, tm_port)
            except Exception:
                pass

        # disable autopilot for RL-controlled slots
        for slot_idx in self.agent_ids:
            actor_id = self.slot2actor[slot_idx]
            if actor_id is None:
                continue
            actor = id2actor.get(actor_id, None)
            if actor is None:
                continue
            try:
                actor.set_autopilot(False)
            except Exception:
                pass

        # 8) finalize bookkeeping
        self.agent_ids = [i for i, a in enumerate(self.slot2actor) if a is not None]
        self.rl_slots = self.agent_ids.copy()

        self.collision_happened = False
        self.episode_info["id"] += 1
        self.episode_info["start_time"] = self.get_simulation_time()
        self.episode_info["end_time"] = self.get_simulation_time()
        self.episode_data = {}
        self.step_data = {}
        self.last_location = self.ego_vehicle.get_location()
        self.start_time = self.get_simulation_time()
        self.distance_travelled = 0.0

        obs, _ = self.get_obs_for_manager(batch_size=1, ego_id=self.ego_vehicle.id, radius=50.0)
        return obs

    def get_obs_for_manager(self, batch_size=1, ego_id=None, radius=50.0):
        """
        Build an observation batch for MAPPO.
        Returns (obs, slot2actor)
        obs keys: image, agent_feats, type_id, mask
        """
        import torch
        B = batch_size
        N = self.max_slots

        ego = self.ego_vehicle if ego_id is None else self.world.get_actor(ego_id)
        ego_loc = ego.get_location()

        # collect nearby agents
        agent_ids = self.get_background_agents(radius=radius)[:N]
        slot2actor = [None] * N
        for i, aid in enumerate(agent_ids):
            slot2actor[i] = aid

        # BEV image
        if self.latest_image is None:
            image = np.zeros((self.bev_H, self.bev_W, self.bev_ch), dtype=np.float32)
        else:
            image = self.latest_image
        img = np.transpose(image, (2, 0, 1)).copy()  # make it contiguous, no negative strides
        img_t = torch.from_numpy(img[None, ...]).float()
        # per-agent features and mask
        agent_feats = np.zeros((B, N, self.obs_dim), dtype=np.float32)
        type_ids = np.zeros((B, N), dtype=np.int64)
        mask = np.zeros((B, N), dtype=np.float32)

        for i, aid in enumerate(slot2actor):
            if aid is None:
                continue
            actor = self.world.get_actor(aid)
            loc = actor.get_location()
            vel = actor.get_velocity()
            speed = (vel.x ** 2 + vel.y ** 2 + vel.z ** 2) ** 0.5
            rel_x = loc.x - ego_loc.x
            rel_y = loc.y - ego_loc.y
            feat = np.zeros((self.obs_dim,), dtype=np.float32)
            feat[0] = speed
            feat[1] = rel_x
            feat[2] = rel_y
            agent_feats[0, i, :len(feat)] = feat
            type_ids[0, i] = 1 if "walker" in actor.type_id or "pedestrian" in actor.type_id else 0
            mask[0, i] = 1.0

        obs = {
            "image": img_t,
            "agent_feats": torch.tensor(agent_feats, dtype=torch.float32),
            "type_id": torch.tensor(type_ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.float32)
        }
        return obs, slot2actor

    def get_simulation_time(self):
        """
        Return current simulation time in seconds.
        """
        return self.world.get_snapshot().timestamp.elapsed_seconds

    def destroy_all_actors(self):
        """
        Deterministic teardown:
          - mark destroying flag so callbacks can early-return
          - detach vehicles from TM
          - stop/destroy pedestrian controllers
          - stop/destroy other controllers
          - tick to let server process detach
          - stop/destroy sensors
          - notify manager to clear selected agents (if available)
          - destroy ego, vehicles, pedestrians
          - final tick and unmark destroying
        """
        # mark
        self._destroying = True

        # detach vehicles from TM
        tm_port = getattr(self, "tm_port", 8000)
        try:
            tm = self.client.get_trafficmanager(tm_port)
        except Exception:
            tm = getattr(self, "traffic_manager", None)

        for v in list(getattr(self, "vehicles", [])):
            v.set_autopilot(False)

        # stop/destroy pedestrian controllers if present
        if hasattr(self, "ped_controllers"):
            for ped_id, ctrl in list(self.ped_controllers.items()):
                ctrl.stop()
                ctrl.destroy()
            self.ped_controllers = {}

        # stop/destroy other controllers
        for cid, ctrl in list(getattr(self, "controllers", {}).items()):
            # attempt stop if available, then destroy
            try:
                ctrl.stop()
            except Exception:
                pass
            ctrl.destroy()
        self.controllers = {}

        # allow server to process detach
        self.world.tick()
        self.world.tick()

        # stop & destroy sensors BEFORE actor destruction to avoid callback races
        if getattr(self, "camera_sensor", None) is not None:
            self.camera_sensor.stop()
            self.camera_sensor.destroy()
            self.camera_sensor = None

        if getattr(self, "collision_sensor", None) is not None:
            self.collision_sensor.stop()
            self.collision_sensor.destroy()
            self.collision_sensor = None

        # notify manager to clear agents if supported
        m = getattr(self, "mappo_manager", None)
        if m is not None:
            if hasattr(m, "sync_agents_with_selected"):
                m.sync_agents_with_selected([])
            elif hasattr(m, "reset"):
                m.reset()

        # destroy ego and other actors
        if getattr(self, "ego_vehicle", None) is not None:
            self.ego_vehicle.destroy()
            self.ego_vehicle = None

        for v in list(getattr(self, "vehicles", [])):
            v.destroy()
        self.vehicles = []

        for p in list(getattr(self, "pedestrians", [])):
            p.destroy()
        self.pedestrians = []

        # final tick to let server finalize resources
        self.world.tick()

        # unmark
        self._destroying = False

    def destroy_sensors(self):
        """
        Stop then destroy sensors. No defensive wrappers.
        """
        if getattr(self, "camera_sensor", None) is not None:
            self.camera_sensor.stop()
            self.camera_sensor.destroy()
            self.camera_sensor = None

        if getattr(self, "collision_sensor", None) is not None:
            self.collision_sensor.stop()
            self.collision_sensor.destroy()
            self.collision_sensor = None

        if getattr(self, "pygame_display_initialized", False):
            pygame.quit()
            self.pygame_display_initialized = False

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
        actions: dict keyed by agent_id (slot index) -> action array-like [throttle, steer, ...]
        Only RL-controlled agents get apply_control(); background vehicles remain under TM.
        """
        # 1) apply RL actions only to RL-controlled slots
        for agent_id, act in actions.items():
            # skip if this slot is not an RL-controlled agent (optional check)
            if hasattr(self, "rl_slots") and agent_id not in self.rl_slots:
                continue

            control = carla.VehicleControl()
            control.throttle = float(max(0.0, act[0]))
            control.steer = float(np.clip(act[1], -1, 1))

            veh_id = self.slot2actor[agent_id]  # slot -> vehicle id (CARLA actor id)
            vehicle = self.world.get_actor(veh_id)
            # ensure this vehicle is not under TM (should have been disabled at reset)
            # vehicle.set_autopilot(False)  # do once in reset(), not each step
            vehicle.apply_control(control)

        # 2) advance simulation (TM will control background autopilot cars)
        self.world.tick()

        # 3) collect next observations / rewards / dones
        next_obs, _ = self.get_obs_for_manager(batch_size=1, ego_id=self.ego_vehicle.id, radius=50.0)

        rewards = {aid: self._compute_reward(aid) for aid in self.agent_ids}
        dones = {aid: self._check_done(aid) for aid in self.agent_ids}

        return next_obs, rewards, dones, {}

    def get_background_agents(self, radius=50.0):
        ego_loc = self.ego_vehicle.get_location()
        cand = []

        for v in self.vehicles:
            if v is None:
                continue
            try:
                d = ego_loc.distance(v.get_location())
            except RuntimeError:
                # actor already destroyed on CARLA side
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

    def _compute_reward(self, agent_id):

        w_speed = 0.05
        w_progress = 1.0
        w_dist_ego = -0.02
        w_offroad = -2.0
        w_collision = -6.0
        w_ang_bad = -1.0
        max_close_penalty_dist = 2.0
        progress_clip = 5.0

        veh_id = self.slot2actor[agent_id]
        actor = self.world.get_actor(veh_id)
        vel = actor.get_velocity()
        speed = (vel.x ** 2 + vel.y ** 2 + vel.z ** 2) ** 0.5

        wp = self.map.get_waypoint(actor.get_location(), project_to_road=False)
        if wp is None:
            return w_offroad

        key = f"last_s_{veh_id}"
        last_s = self.episode_data.get(key, wp.s)
        progress = wp.s - last_s
        self.episode_data[key] = wp.s
        progress = max(-progress_clip, min(progress, progress_clip))

        reward = w_speed * speed + w_progress * progress

        if self.collision_happened:
            reward += w_collision

        tr = actor.get_transform()
        if abs(tr.rotation.roll) > 45 or abs(tr.rotation.pitch) > 45:
            reward += w_ang_bad

        ego_loc = self.ego_vehicle.get_location()
        loc = actor.get_location()
        d = ego_loc.distance(loc)
        if d < max_close_penalty_dist:
            reward += w_dist_ego * (max_close_penalty_dist - d) * 2.0
        else:
            reward += w_dist_ego / (d + 1e-6)

        return float(reward)

    def _check_done(self, agent_id):
        veh_id = self.slot2actor[agent_id]
        vehicle = self.world.get_actor(veh_id)

        if self.collision_happened:
            return True

        transform = vehicle.get_transform()
        if abs(transform.rotation.roll) > 45 or abs(transform.rotation.pitch) > 45:
            return True

        waypoint = self.map.get_waypoint(vehicle.get_location(), project_to_road=False)
        if waypoint is None:
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

    def _process_camera_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1]
        self.latest_image = array

    def _on_collision(self, event):
        self.collision_happened = True

    def render_image(self):
        if self.latest_image is None:
            return

        if not self.pygame_display_initialized:
            pygame.init()
            pygame.font.init()
            self.font = pygame.font.SysFont("Arial", 24)
            self.screen = pygame.display.set_mode(
                (self.latest_image.shape[1], self.latest_image.shape[0])
            )
            pygame.display.setcaption("Ego Camera View")
            self.pygame_display_initialized = True
            self._last_time = time.time()
            self._fps = 0.0

        current_time = time.time()
        dt = current_time - self._last_time
        if dt > 0:
            self._fps = 1.0 / dt
        self._last_time = current_time

        surface = pygame.surfarray.make_surface(self.latest_image.swapaxes(0, 1))
        self.screen.blit(surface, (0, 0))

        fps_text = self.font.render(f"FPS: {self._fps:.2f}", True, (255, 255, 0))
        self.screen.blit(fps_text, (10, 10))

        try:
            mini_map_size = 200
            scale = 1.0
            padding = 10
            center = (mini_map_size // 2, mini_map_size // 2)

            mini_surface = pygame.Surface((mini_map_size, mini_map_size))
            mini_surface.fill((40, 40, 40))

            ego_loc = self.ego_vehicle.get_location()

            # lanes
            for waypoint in self.map.generate_waypoints(2.0):
                next_wp_list = waypoint.next(2.0)
                if not next_wp_list:
                    continue
                next_wp = next_wp_list[0]

                x1 = int(
                    center[0]
                    - (waypoint.transform.location.x - ego_loc.x) * scale
                )
                y1 = int(
                    center[1]
                    - (waypoint.transform.location.y - ego_loc.y) * scale
                )
                x2 = int(
                    center[0]
                    - (next_wp.transform.location.x - ego_loc.x) * scale
                )
                y2 = int(
                    center[1]
                    - (next_wp.transform.location.y - ego_loc.y) * scale
                )

                pygame.draw.line(mini_surface, (90, 90, 90), (x1, y1), (x2, y2), 1)

            # vehicles
            for vehicle in [self.ego_vehicle] + self.vehicles:
                loc = vehicle.get_location()
                dx = (loc.x - ego_loc.x) * scale
                dy = (loc.y - ego_loc.y) * scale
                vx = int(center[0] - dx)
                vy = int(center[1] - dy)
                color = (
                    (0, 255, 0)
                    if vehicle.id == self.ego_vehicle.id
                    else (200, 200, 200)
                )
                pygame.draw.circle(mini_surface, color, (vx, vy), 4)

            # pedestrians
            for ped_item in getattr(self, "pedestrians", []):
                actor = getattr(
                    ped_item, "actor", getattr(ped_item, "pedestrian", ped_item)
                )
                loc = actor.get_location()
                dx = (loc.x - ego_loc.x) * scale
                dy = (loc.y - ego_loc.y) * scale
                px = int(center[0] - dx)
                py = int(center[1] - dy)
                pygame.draw.circle(mini_surface, (50, 150, 255), (px, py), 2)

            # traffic lights
            for tl in self.world.get_actors().filter("traffic.traffic_light*"):
                loc = tl.get_location()
                dx = (loc.x - ego_loc.x) * scale
                dy = (loc.y - ego_loc.y) * scale
                tx = int(center[0] - dx)
                ty = int(center[1] - dy)
                color = {
                    carla.TrafficLightState.Red: (255, 0, 0),
                    carla.TrafficLightState.Yellow: (255, 255, 0),
                    carla.TrafficLightState.Green: (0, 255, 0),
                    carla.TrafficLightState.Off: (100, 100, 100),
                }.get(tl.state, (50, 50, 50))
                pygame.draw.circle(mini_surface, color, (tx, ty), 3)

            pygame.draw.rect(
                mini_surface,
                (200, 200, 0),
                (0, 0, mini_map_size, mini_map_size),
                2,
            )

            self.screen.blit(
                mini_surface,
                (
                    self.latest_image.shape[1] - mini_map_size - 10,
                    self.latest_image.shape[0] - mini_map_size - 10,
                ),
            )

        except Exception as e:
            print(f"[WARN] mini-map drawing failed: {e}")

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
