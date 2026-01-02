# envs/observation/graphics.py
# Headless-friendly graphics utilities for observation rendering.
# Replaces pygame usage with numpy + cv2 where needed.
import numpy as np
import cv2
import time
from typing import Tuple, Any, Optional

# Simple HUD-like class exposing a render() method used by envs
class HUD:
    def __init__(self, width: int = 800, height: int = 600, enable_render: bool = False):
        self.width = width
        self.height = height
        self.enable_render = bool(enable_render)
        self.font_scale = 0.6
        self.font_thickness = 1
        self.text_color = (255, 255, 0)  # BGR
        self.background_color = (0, 0, 0)
        self.last_time = time.time()
        self.fps = 0.0

    def ensure_canvas(self, canvas: Optional[np.ndarray]) -> np.ndarray:
        if canvas is None:
            canvas = np.full((self.height, self.width, 3), fill_value=0, dtype=np.uint8)
        h, w = canvas.shape[:2]
        if (h, w) != (self.height, self.width):
            canvas = cv2.resize(canvas, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        return canvas

    def draw_text(self, canvas: np.ndarray, text: str, pos: Tuple[int,int] = (10, 20)):
        cv2.putText(canvas, text, pos, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.text_color, self.font_thickness, cv2.LINE_AA)

    def update_fps(self):
        t = time.time()
        dt = t - self.last_time if t - self.last_time > 1e-6 else 1.0
        self.fps = 1.0 / dt
        self.last_time = t

    def render(self, canvas: Optional[np.ndarray] = None, info: Optional[dict] = None) -> np.ndarray:
        """
        Render HUD info onto a numpy canvas and return it.
        If enable_render is False, it still returns a valid canvas (useful for headless debug).
        """
        canvas = self.ensure_canvas(canvas)
        self.update_fps()
        self.draw_text(canvas, f"FPS: {self.fps:.1f}", (10, 20))
        if info:
            y = 40
            for k, v in info.items():
                txt = f"{k}: {v}"
                self.draw_text(canvas, txt, (10, y))
                y += 18
        return canvas

# Minimal ModuleWorld-like helper to composite layers (keeps original API shape)
class ModuleWorld:
    def __init__(self, map_surface: np.ndarray, bev_size: Tuple[int,int] = (84,84)):
        """
        map_surface : big map (numpy BGR)
        """
        self.map_surface = map_surface
        self.bev_size = bev_size
        # layers as numpy arrays same size as map_surface
        H, W = map_surface.shape[:2]
        self.vehicle_layer = np.zeros((H, W, 3), dtype=np.uint8)
        self.walker_layer = np.zeros((H, W, 3), dtype=np.uint8)
        self.traffic_layer = np.zeros((H, W, 3), dtype=np.uint8)

    def clear_layers(self):
        self.vehicle_layer.fill(0)
        self.walker_layer.fill(0)
        self.traffic_layer.fill(0)

    def draw_actor(self, layer: np.ndarray, px: int, py: int, color: Tuple[int,int,int], radius_px: int = 3):
        cv2.circle(layer, (px, py), radius=radius_px, color=color, thickness=-1)

    def composite_patch(self, center_px: Tuple[int,int], patch_px: int) -> np.ndarray:
        """
        Composite map + layers into a single patch around center_px, returning HxWx3 BGR uint8
        """
        cx, cy = center_px
        half = patch_px // 2
        x0 = cx - half; y0 = cy - half; x1 = cx + half; y1 = cy + half
        H, W = self.map_surface.shape[:2]
        patch = np.zeros((patch_px, patch_px, 3), dtype=np.uint8)
        # compute source region
        sx0 = max(0, x0); sy0 = max(0, y0); sx1 = min(W, x1); sy1 = min(H, y1)
        dx0 = sx0 - x0; dy0 = sy0 - y0
        if sx1 > sx0 and sy1 > sy0:
            base = self.map_surface[sy0:sy1, sx0:sx1]
            layer = self.vehicle_layer[sy0:sy1, sx0:sx1]
            patch[dy0:dy0+base.shape[0], dx0:dx0+base.shape[1]] = cv2.addWeighted(base, 0.9, layer, 0.7, 0)
        return patch

# simple visualize wrapper (keeps API name)
def visualize_birdview(bev_img: np.ndarray) -> np.ndarray:
    """
    Accepts HxWx3 BGR uint8, returns same image (or scales for display)
    """
    return bev_img.copy()
