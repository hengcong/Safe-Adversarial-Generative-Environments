# envs/observation/utilities/graphics.py
# small helper graphics utilities (no pygame)
import numpy as np
import cv2
from typing import Tuple

def create_empty_canvas(size: Tuple[int,int], color=(0,0,0)) -> np.ndarray:
    h, w = size
    canvas = np.full((h, w, 3), color, dtype=np.uint8)
    return canvas

def draw_road_polygon(canvas: np.ndarray, points: np.ndarray, color=(200,200,200), thickness: int = -1):
    """
    points: Nx2 int array
    thickness: -1 fill, else line thickness
    """
    pts = np.array(points, dtype=np.int32)
    if pts.ndim == 2 and pts.shape[0] >= 3:
        cv2.fillPoly(canvas, [pts], color=color) if thickness == -1 else cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=thickness)

def draw_lane_line(canvas: np.ndarray, points: np.ndarray, color=(255,255,255), thickness: int = 1, dashed: bool = False):
    pts = np.array(points, dtype=np.int32)
    if pts.shape[0] < 2:
        return
    if not dashed:
        cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=thickness)
    else:
        # dashed: draw small segments
        total = pts.shape[0]
        seg = 10
        for i in range(0, total-1, seg):
            seg_pts = pts[i:i+seg+1]
            if seg_pts.shape[0] > 1:
                cv2.polylines(canvas, [seg_pts], isClosed=False, color=color, thickness=thickness)
