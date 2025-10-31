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

import pygame
from gym import spaces, core

import torch
from observation.obs_bev import BEVObservation
from algo.mappo.mappo_manager import MAPPOManager

class CarlaEnv(gym.Env):
    def __init__(self, num_veh, num_ped, mode="MAPPO"):
        super(CarlaEnv, self).__init__()
        self.mode = mode

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

        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world("Town04")
        self.map = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()

        # --- synchronous mode ---
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.step_size
        self.world.apply_settings(settings)

        for _ in range(3):
            self.world.tick()

        self.ego_vehicle = None
        self.ego_vehicle_wrapper = None  # keep if you use VehicleWrapper; else may remain None
        self.vehicle_wrapper_list = {}
        self.pedestrian_wrapper_list = {}

        self.mappo_manager = MAPPOManager()


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

        self.spawn_ego_vehicle()
        if self.num_veh > 0:
            self.generate_bv_traffic_flow()
        if self.num_ped > 0:
            self.spawn_pedestrians()

        if hasattr(self, "vehicle_wrapper_list") and len(self.vehicle_wrapper_list) > 0:
            self.agent_ids = list(self.vehicle_wrapper_list.keys())
        else:
            # fallback: create placeholder agent ids based on num_veh (will need to map to actual vehicles later)
            self.agent_ids = [f"veh_{i}" for i in range(self.num_veh)]

        # --- per-agent action space: [acc (m/s^2), steer (-1..1)]
        per_agent_low = np.array([-6.0, -1.0], dtype=np.float32)
        per_agent_high = np.array([6.0, 1.0], dtype=np.float32)

        self.action_space = spaces.Dict({
            agent_id: spaces.Box(low=per_agent_low, high=per_agent_high, dtype=np.float32)
            for agent_id in self.agent_ids
        })

        # tick & warmup
        self.world.tick()
        self.vehicle_map = self._group_vehicles_by_road_and_lane()
        self.warmup(num_frames=20)

        # --- experiment path & minimal logger (timestamped) ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_path = f"./carla_experiments/run_{timestamp}"
        os.makedirs(self.experiment_path, exist_ok=True)

        # --- minimal sensors / rendering state ---
        self.camera_sensor = None
        self.latest_image = None
        self.pygame_display_initialized = False
        self.screen = None

        # Attach simple sensors (ensure setup_sensors doesn't reference removed modules)
        self.setup_sensors()

        # Activate agents (must be simple; ensure activate_agents is compatible)
        self.activate_agents()

        # --- episode bookkeeping ---
        self.collision_happened = False
        self.episode_info = {"id": 0, "start_time": self.get_simulation_time(), "end_time": self.get_simulation_time()}
        self.episode_data = {}
        self.step_data = {}

        # --- controllers placeholders (empty dicts; you can add controllers later) ---
        self.ego_controller = None
        self.bv_controllers = {}

        # --- tracking ---
        self.start_location = None
        self.start_time = 0.0
        self.distance_travelled = 0.0
        self.last_location = None

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
        for vehicle in [self.ego_vehicle]+self.vehicle_wrapper_list.vehicles:
            wp = self.get_waypoint(vehicle)
            key = (wp.road_id, wp.lane_id)
            if key not in vehicle_map:
                vehicle_map[key] = []
            vehicle_map[key].append((vehicle, wp.s))

        return vehicle_map

    def reset(self):
        print("🔁 Resetting CarlaEnv...")
        # Destroy existing actors
        ids_to_destroy = []
        if self.ego_vehicle is not None and self.ego_vehicle.is_alive:
            ids_to_destroy.append(self.ego_vehicle.id)
            print(f"🧹 Destroying ego vehicle {self.ego_vehicle.id}")

        ids_to_destroy += list(self.vehicle_wrapper_list.keys())
        ids_to_destroy += list(self.pedestrian_wrapper_list.keys())

        if ids_to_destroy:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in ids_to_destroy])
            print(f"✅ Destroyed {len(ids_to_destroy)} actors.")
        else:
            print("⚠️ No actors to destroy.")

        self.vehicles = []
        self.pedestrians = []
        self.ego_vehicle = None
        self.vehicle_wrapper_list = {}

        self.destroy_sensors()
        self.collision_happened = False

        time.sleep(0.5)
        self.world.tick()

        self.soft_reboot()
        self.vehicle_map = self._group_vehicles_by_road_and_lane()

        self.ego_vehicle_wrapper.update_observation(self)
        for wrapper in self.vehicle_wrapper_list.values():
            wrapper.update_observation(self)

        self.start_location = self.ego_vehicle.get_location()
        self.start_time = self.get_simulation_time()
        self.distance_travelled = 0.0
        self.last_location = None

        state = self.get_state()

        if hasattr(self, "mappo_agent"):
            self.mappo_manager.mappo_reset(state)
        return state

    def get_state(self):
        obs = BEVObservation(veh_id=self.ego_vehicle.id, time_stamp=self.get_simulation_time())
        obs.update(env=self)
        return obs.information

    def setup_sensors(self):
        # Collision sensor
        # if self.collision_sensor is None or not self.collision_sensor.is_alive:
        #     sensor_bp = self.blueprint_library.find('sensor.other.collision')
        #     sensor_transform = carla.Transform(carla.Location(x=0, y=0, z=2))
        #     self.collision_sensor = self.world.spawn_actor(sensor_bp, sensor_transform, attach_to=self.ego_vehicle)
        #     self.collision_sensor.listen(lambda event: self._on_collision(event))

        # RGB camera sensor
        if self.camera_sensor is None or not self.camera_sensor.is_alive:
            camera_bp = self.blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', '800')
            camera_bp.set_attribute('image_size_y', '600')
            camera_bp.set_attribute('fov', '90')

            camera_transform = carla.Transform(
                carla.Location(x=-6.5, y=0, z=2.5),
                carla.Rotation(pitch=-15, yaw=0, roll=0)
            )

            self.camera_sensor = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.ego_vehicle)
            self.camera_sensor.listen(lambda image: self._process_camera_image(image))

        self._tick()
        self._tick()
        self._camera_ready = True

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