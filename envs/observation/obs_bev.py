# observation/obs_bev.py
import numpy as np
from datetime import datetime

# adjust these imports to match your project layout
from envs.observation.map_utils import BeVWrapper     # maps -> BeVWrapper
from envs.observation.bev_utils import get_birdview   # renders bev_data -> image

class BEVObservation:
    """
    Minimal BEV observation wrapper that provides an `update(env)` API similar to ObservationCarla.
    Usage:
        obs = BEVObservation(veh_id=..., bev_config=...)
        obs.update(env)
        data = obs.information  # dict with keys: "image" (HxWxC float32), "speed" (1,), "time_stamp"
    """

    def __init__(self, veh_id=None, time_stamp=None, bev_config=None):
        self.veh_id = veh_id
        self.time_stamp = time_stamp if time_stamp is not None else datetime.now().timestamp()
        self.information = None
        self._bev_config = bev_config or {}
        # internal flag for whether we initialized our own wrapper
        self._owns_wrapper = False

    def _ensure_bev_wrapper(self, env):
        """
        Ensure env has a BeVWrapper instance at env._bev_wrapper.
        If not present, create one and attach it to env (so it is reused).
        """
        if getattr(env, "_bev_wrapper", None) is None:
            bw = BeVWrapper(self._bev_config)
            # init(client, world, map, ego_actor, display)
            bw.init(env.client, env.world, env.map, env.ego_vehicle, getattr(env, "display", None))
            env._bev_wrapper = bw
            self._owns_wrapper = True
        # always use env._bev_wrapper (the canonical instance)
        self._bev_wrapper = env._bev_wrapper

    def update(self, env):
        """
        Update the observation from the environment.
        Must be called after env.world.tick() (env.step / env.reset should ensure stepping).
        Returns the information dict and stores it at self.information.
        """
        # 1) ensure wrapper exists
        self._ensure_bev_wrapper(env)

        # 2) ensure wrapper is up-to-date for this frame
        # env typically calls world.tick() before calling observation; call tick() on wrapper to update internal surfaces
        try:
            self._bev_wrapper.tick()
        except Exception:
            # best-effort: wrapper may update elsewhere; ignore failure
            pass

        # 3) get bev raw data and render to image
        bev_data = self._bev_wrapper.get_bev_data()
        bev_img = get_birdview(bev_data)  # expected uint8 HxWxC

        # 4) convert to float32 normalized [0,1]
        if bev_img is None:
            # fallback: empty black image (choose a default shape if unknown)
            # Try to infer shape from config else use 84x84x3
            h, w, c = getattr(self._bev_config, "shape", (84, 84, 3))
            bev_float = np.zeros((h, w, c), dtype=np.float32)
        else:
            if bev_img.dtype == np.uint8:
                bev_float = bev_img.astype(np.float32) / 255.0
            else:
                bev_float = bev_img.astype(np.float32)
            # ensure contiguous
            bev_float = np.ascontiguousarray(bev_float)

        # 5) get ego speed and normalize
        speed_norm = np.array([0.0], dtype=np.float32)
        try:
            ego = env.ego_vehicle
            v = ego.get_velocity()
            speed_mps = np.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)
            # normalize by 30 m/s (adjust if you prefer another clip)
            speed_norm[0] = min(speed_mps / 30.0, 1.0)
        except Exception:
            speed_norm[0] = 0.0

        # 6) package information (match whatever your training expects)
        self.time_stamp = env.get_simulation_time() if hasattr(env, "get_simulation_time") else datetime.now().timestamp()
        self.information = {
            "image": bev_float,       # H x W x C, float32 in [0,1]
            "speed": speed_norm,      # shape (1,)
            "time_stamp": self.time_stamp
        }
        # ensure bev_float is H x W x C float32 in [0,1] and RGB
        self.information = {"image": bev_float, "speed": speed_norm, "time_stamp": self.time_stamp}
        # write canonical cache on env (minimal, safe)
        try:
            env.latest_bev_image = bev_float
        except Exception:
            pass
        return self.information

    def destroy(self, env=None):
        """
        Optional cleanup: if this object created a wrapper and it should be removed,
        you can destroy it here. Otherwise do nothing.
        """
        if self._owns_wrapper and getattr(env, "_bev_wrapper", None) is not None:
            try:
                env._bev_wrapper.destroy()
            except Exception:
                pass
            try:
                delattr(env, "_bev_wrapper")
            except Exception:
                pass
