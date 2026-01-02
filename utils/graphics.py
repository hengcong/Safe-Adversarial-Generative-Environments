# utils/graphics.py
# Utility functions for converting BEV channels into colored display images (no pygame).
import numpy as np
import cv2
from typing import Tuple

def visualize_birdview_channels(channels: np.ndarray, color_map: dict = None) -> np.ndarray:
    """
    channels: C x H x W (float or uint8)
    color_map: mapping channel_index -> (R,G,B)
    Returns H x W x 3 BGR uint8
    """
    if channels.ndim == 3:
        C, H, W = channels.shape
    elif channels.ndim == 2:
        # single channel
        C = 1; H, W = channels.shape
        channels = channels[np.newaxis,...]
    else:
        raise ValueError("channels must be 2D or 3D")

    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    if color_map is None:
        # default colors for first few channels
        default_colors = [
            (252,233,79), (91,155,213), (97,189,109),
            (255,0,0), (255,255,0), (128,128,128)
        ]
        color_map = {i: default_colors[i % len(default_colors)] for i in range(C)}

    for i in range(C):
        ch = channels[i]
        # normalize to 0-1
        if ch.dtype != np.float32 and ch.dtype != np.float64:
            chf = ch.astype(np.float32) / 255.0 if ch.max() > 1 else ch.astype(np.float32)
        else:
            chf = ch
        color = color_map.get(i, (255,255,255))
        col = (int(color[2]), int(color[1]), int(color[0]))  # BGR
        mask = (chf > 0.1).astype(np.uint8)
        if mask.sum() == 0:
            continue
        for c_idx in range(3):
            canvas[:,:,c_idx] = np.where(mask == 1, int(col[c_idx]), canvas[:,:,c_idx])
    return canvas

def overlay_text(img: np.ndarray, text: str, pos=(10,20), color=(0,255,255)):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
