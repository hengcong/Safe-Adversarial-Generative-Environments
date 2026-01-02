# envs/observation/map_utils.py
# Pure NumPy + OpenCV BEV / map utilities.
# Replaces previous pygame-based implementation to be headless-friendly.
#
# Public classes / functions:
# - MapImage
# - BeVWrapper
#
# Main entrypoint compatible with prior usage:
#   from envs.observation.map_utils import BeVWrapper
#   bev = BeVWrapper(world_map, pixels_per_meter=5, bev_size=(84,84))
#   img, meta = bev.get_bev_data(ego_vehicle, radius_m=50)
#
# Requirements:
#   pip install opencv-python
#
import math
import numpy as np
import cv2
from typing import Tuple, List, Dict, Any, Optional

# --- colors (RGB tuples) ---
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_GRAY = (128, 128, 128)
COLOR_BLUE = (91, 155, 213)
COLOR_GREEN = (97, 189, 109)
COLOR_BUTTER_0 = (252, 233, 79)
COLOR_BUTTER_1 = (237, 212, 0)
COLOR_RED = (220, 20, 60)
COLOR_ORANGE = (255, 165, 0)
COLOR_YELLOW = (255, 255, 0)

# utility helpers
def rgb_to_bgr(c: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return (c[2], c[1], c[0])  # cv2 uses BGR

def to_int_tuple(x):
    return tuple(int(round(v)) for v in x)

def world_point_to_np(world_loc, origin, pixels_per_meter: float) -> Tuple[int, int]:
    """
    Map a CARLA world.Location-like object (has x,y) to pixel coords.
    origin: (min_x, min_y) world coords mapped to pixel (0,0)
    pixels_per_meter: scaling
    Returns (px, py) where px increases to right, py increases downward (image coords)
    """
    wx, wy = world_loc.x, world_loc.y
    ox, oy = origin
    px = (wx - ox) * pixels_per_meter
    py = (wy - oy) * pixels_per_meter
    # pixel coordinates are ints
    return int(round(px)), int(round(py))


class MapImage:
    """
    Create a static map image for the current CARLA map.
    - carla_map: carla.Map instance
    - pixels_per_meter: scaling factor
    - margin_meters: margin around min/max waypoints to include
    Produces:
    - self.map_surface: numpy array HxWx3 uint8 (BGR for cv2)
    - self.origin_world: (min_x, min_y) world coordinates mapped to pixel (0,0)
    - self.to_pixel: callable world_loc -> (x,y) pixel coords
    """
    def __init__(self, carla_map, pixels_per_meter: float = 5.0, margin_meters: float = 50.0):
        self.carla_map = carla_map
        self.ppm = float(pixels_per_meter)
        self.margin = float(margin_meters)

        # sample waypoints to obtain bounding box
        # carla_map.generate_waypoints(distance) returns list of waypoints across map
        try:
            waypoints = carla_map.generate_waypoints(2.0)
        except Exception:
            # fallback to spawn points
            try:
                sps = carla_map.get_spawn_points()
                waypoints = []
                for sp in sps:
                    # create a simple object with transform.location.x/y
                    class _Dummy: pass
                    d = _Dummy()
                    d.transform = sp
                    waypoints.append(d)
            except Exception:
                waypoints = []

        xs = [wp.transform.location.x for wp in waypoints] if waypoints else [0.0]
        ys = [wp.transform.location.y for wp in waypoints] if waypoints else [0.0]

        min_x = min(xs) - self.margin
        max_x = max(xs) + self.margin
        min_y = min(ys) - self.margin
        max_y = max(ys) + self.margin

        self.min_x = float(min_x)
        self.min_y = float(min_y)
        self.max_x = float(max_x)
        self.max_y = float(max_y)

        world_w = max_x - min_x
        world_h = max_y - min_y

        # pixel image size (square to keep transformations simple)
        size_px = int(math.ceil(max(world_w, world_h) * self.ppm)) + 1
        if size_px <= 0:
            size_px = 512  # fallback

        # map_surface is BGR uint8 for OpenCV
        self.map_surface = np.full((size_px, size_px, 3), fill_value=COLOR_BLACK, dtype=np.uint8)
        self.origin_world = (self.min_x, self.min_y)
        self.width_px = size_px
        self.height_px = size_px

        # draw static lanes / roads onto map_surface
        self._draw_static_map()

    def world_to_pixel(self, world_loc) -> Tuple[int, int]:
        return world_point_to_np(world_loc, self.origin_world, self.ppm)

    def pixel_to_world(self, px: int, py: int) -> Tuple[float, float]:
        wx = px / self.ppm + self.origin_world[0]
        wy = py / self.ppm + self.origin_world[1]
        return wx, wy

    def _draw_static_map(self):
        """
        Draw a simplified map representation: fill road polygons based on topology waypoints.
        This is a reasonable approximation for BEV usage.
        """
        try:
            topology = self.carla_map.get_topology()
            # topology: list of (wp_from, wp_to) pairs typically; we use first elements
            lines = [t[0] for t in topology]
        except Exception:
            lines = []

        # Draw each topology segment as a thick line
        for wp in lines:
            try:
                start = self.world_to_pixel(wp.transform.location)
                # get next points along the lane for some length
                nxts = wp.next(5.0)
                pts = [start]
                for nwp in nxts:
                    pts.append(self.world_to_pixel(nwp.transform.location))
                pts_arr = np.array(pts, dtype=np.int32)
                if pts_arr.shape[0] >= 2:
                    cv2.polylines(self.map_surface, [pts_arr], isClosed=False, color=rgb_to_bgr(COLOR_GRAY), thickness=2)
            except Exception:
                # ignore occasional errors from map queries
                continue


class BeVWrapper:
    """
    Bird's-Eye View wrapper that produces BEV images (numpy HxWx3 uint8)
    Usage:
      bev = BeVWrapper(carla_map, pixels_per_meter=5, bev_size=(84,84))
      img, meta = bev.get_bev_data(ego_vehicle, radius_m=50)
    """
    def __init__(self, carla_map, pixels_per_meter: float = 5.0, bev_size: Tuple[int, int] = (84, 84)):
        self.map_image = MapImage(carla_map, pixels_per_meter=pixels_per_meter)
        self.ppm = float(pixels_per_meter)
        self.bev_size = tuple(bev_size)
        # how many pixels represent 1 meter in BEV output: depends on desired radius
        # When generating BEV we will compute a scale so that requested radius fits into bev_size //
        # keep simple: scale = (bev_size[0]/(2*radius_m)) pixels per meter (computed per request)
        # keep small landmarks
        self.actor_radius_px = max(2, int(round(0.5 * self.ppm)))

    def _compose_background_patch(self, center_px: Tuple[int, int], patch_px: int) -> np.ndarray:
        """
        Extract a square patch from the large map_surface centered at center_px with half size patch_px.
        Returns patch image HxWx3 BGR uint8.
        If requested patch goes out of bounds, pad with black.
        """
        cx, cy = center_px
        half = patch_px // 2
        x0 = cx - half
        y0 = cy - half
        x1 = cx + half
        y1 = cy + half

        H, W = self.map_image.map_surface.shape[:2]
        # create black patch
        patch = np.zeros((patch_px, patch_px, 3), dtype=np.uint8)

        sx0 = max(0, x0)
        sy0 = max(0, y0)
        sx1 = min(W, x1)
        sy1 = min(H, y1)

        dst_x0 = sx0 - x0
        dst_y0 = sy0 - y0
        dst_x1 = dst_x0 + (sx1 - sx0)
        dst_y1 = dst_y0 + (sy1 - sy0)

        if sx1 > sx0 and sy1 > sy0:
            patch[dst_y0:dst_y1, dst_x0:dst_x1, :] = self.map_image.map_surface[sy0:sy1, sx0:sx1, :]

        return patch

    def _draw_actors_on_patch(self, patch: np.ndarray, center_px: Tuple[int, int], actors: List[Any], color_map=None):
        """
        Draw actors (vehicles/pedestrians) on the patch given actor list.
        Each actor must have .get_location() returning object with x,y; orientation via .get_transform().rotation.yaw
        color_map: optional function mapping actor object to RGB tuple
        """
        if actors is None:
            return
        for a in actors:
            try:
                loc = a.get_location()
                px, py = world_point_to_np(loc, self.map_image.origin_world, self.ppm)
                # convert to patch coords
                dx = px - center_px[0]
                dy = py - center_px[1]
                # patch coords with origin at top-left, x right, y down
                patch_x = int(round(patch.shape[1] / 2 + dx))
                patch_y = int(round(patch.shape[0] / 2 + dy))
                # choose color based on actor type
                c = COLOR_RED if "vehicle" in a.type_id else COLOR_BLUE if "walker" in a.type_id or "pedestrian" in a.type_id else COLOR_ORANGE
                cv2.circle(patch, (patch_x, patch_y), radius=self.actor_radius_px, color=rgb_to_bgr(c), thickness=-1)
            except Exception:
                continue

    def _rotate_and_scale_patch_to_bev(self, patch: np.ndarray, ego_yaw_deg: float, output_size: Tuple[int, int], radius_m: float) -> np.ndarray:
        """
        Rotate the patch so that ego heading points up, and scale/crop to output_size.
        ego_yaw_deg: yaw in degrees (CARLA convention: +ve CCW?), implement so that 0 deg -> pointing East.
        We will rotate by (-ego_yaw_deg + 90) to make car front point to top of image.
        """
        h, w = patch.shape[:2]
        # compute center of patch
        center = (w // 2, h // 2)
        # rotation angle to make front point up
        angle = -ego_yaw_deg + 90.0
        # scale: compute pixels needed to cover radius_m: patch_px covers some world meters: patch_px / ppm
        patch_meters = w / self.ppm
        # current patch radius in meters = patch_meters / 2
        target_patch_meters = 2.0 * radius_m
        # desired scale factor so patch covers exactly 2*radius_m meters
        if patch_meters <= 0:
            scale = 1.0
        else:
            scale = (target_patch_meters / patch_meters)
        # cv2 warp needs scale as multiplier; we can compute final transformation
        M = cv2.getRotationMatrix2D(center, angle, scale=scale)
        warped = cv2.warpAffine(patch, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=COLOR_BLACK)
        # crop center to output_size
        out_h, out_w = output_size
        cx, cy = center
        sx = max(0, cx - out_w // 2)
        sy = max(0, cy - out_h // 2)
        ex = sx + out_w
        ey = sy + out_h
        # ensure bounds
        if ex > warped.shape[1] or ey > warped.shape[0]:
            # pad if necessary
            padded = np.full((max(ey, warped.shape[0]), max(ex, warped.shape[1]), 3), fill_value=COLOR_BLACK, dtype=np.uint8)
            padded[0:warped.shape[0], 0:warped.shape[1], :] = warped
            warped = padded
        cropped = warped[sy:ey, sx:ex, :].copy()
        # ensure exact size
        if cropped.shape[0] != out_h or cropped.shape[1] != out_w:
            cropped = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        return cropped

    def get_bev_data(self, ego_vehicle, radius_m: float = 50.0, actors: Optional[List[Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Produce BEV image and metadata.
        - ego_vehicle: CARLA actor
        - radius_m: half-width in meters from ego center to cover (so full width=2*radius)
        - actors: optional list of actors to render (if None, top nearby actors will be queried)
        Returns:
          - bev_img: numpy array HxWx3 uint8 (BGR)
          - meta: dict with keys {'center_px', 'origin_world', 'ppm', 'ego_yaw_deg'}
        """
        # ensure we have a patch size sufficient to cover radius at map_image.ppm
        patch_half_m = radius_m
        patch_size_m = patch_half_m * 2.0
        patch_px = int(math.ceil(patch_size_m * self.ppm))
        # enforce odd/even consistency and add margin
        patch_px = max(patch_px, 16)
        patch_px += 8  # small margin

        # ego center in pixel coords on big map
        ego_loc = ego_vehicle.get_location()
        center_px = self.map_image.world_to_pixel(ego_loc)

        # extract background patch
        patch = self._compose_background_patch(center_px, patch_px)

        # determine which actors to draw
        if actors is None:
            try:
                world = ego_vehicle.get_world()
                all_actors = world.get_actors()
                # filter vehicles and walkers within radius_m
                cand = []
                ego_x = ego_loc.x; ego_y = ego_loc.y
                for a in all_actors:
                    try:
                        loc = a.get_location()
                        dx = loc.x - ego_x; dy = loc.y - ego_y
                        dist = math.hypot(dx, dy)
                        if dist <= radius_m + 5.0:
                            cand.append(a)
                    except Exception:
                        continue
                actors_to_draw = cand
            except Exception:
                actors_to_draw = []
        else:
            actors_to_draw = actors

        # draw actors
        self._draw_actors_on_patch(patch, center_px, actors_to_draw)

        # ego yaw in degrees
        try:
            ego_yaw = ego_vehicle.get_transform().rotation.yaw
        except Exception:
            ego_yaw = 0.0

        # rotate + scale to BEV output
        bev_img = self._rotate_and_scale_patch_to_bev(patch, ego_yaw_deg=ego_yaw, output_size=self.bev_size, radius_m=radius_m)

        meta = {
            "center_px": center_px,
            "origin_world": self.map_image.origin_world,
            "ppm": self.ppm,
            "ego_yaw_deg": ego_yaw
        }
        return bev_img, meta

    # alias to keep backward-compatible method names
    def get_bev(self, ego_vehicle, radius_m: float = 50.0, actors: Optional[List[Any]] = None):
        return self.get_bev_data(ego_vehicle, radius_m=radius_m, actors=actors)

    def init(self, client=None, world=None, carla_map=None, player=None, display=None, route=None):
        """
        Backwards-compat shim for BEVObservation that may call .init(...).
        Minimal, non-GUI: store references so older BEVObservation implementations run.
        Do NOT perform pygame/display operations here.
        """
        # store references so BEVObservation can later call get_bev/get_bev_data
        self.client = client
        self.world = world
        # if a new map is passed, optionally update map_image
        try:
            if carla_map is not None:
                # replace map_image only if map changed or not exists
                if getattr(self, "map_image", None) is None or getattr(self.map_image, "carla_map", None) != carla_map:
                    # Note: MapImage constructor signature in your code may differ; adapt if needed
                    self.map_image = MapImage(carla_map, pixels_per_meter=self.ppm)
        except Exception:
            # do not fail init because of map issues; BEVObservation may still call get_bev_data later
            pass
        self.player = player
        self.route = route
        # do not touch display / pygame here
        return


