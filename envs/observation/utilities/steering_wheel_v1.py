# envs/observation/utilities/steering_wheel.py
# Lazy joystick/steering wheel helper: does NOT import pygame at module import time.
# If pygame is available and you run in interactive mode, it will use joystick; otherwise fallback to dummy.

from typing import Dict, Any
import os

class SteeringWheel:
    """
    Minimal steering wheel wrapper.
    Public methods:
      - get_control(): returns dict {'steer': float, 'throttle': float, 'brake': float}
      - enable(): attempts to init joystick if available
      - disable(): switch to dummy mode
    """
    def __init__(self, enable_joystick: bool = False):
        self._enabled = False
        self._has_pygame = False
        self._joystick = None
        self._use_real = False
        if enable_joystick:
            self.enable()

    def enable(self):
        # try to lazy-import pygame and init joystick
        try:
            import pygame
            pygame.joystick.init()
            if pygame.joystick.get_count() > 0:
                self._joystick = pygame.joystick.Joystick(0)
                self._joystick.init()
                self._has_pygame = True
                self._use_real = True
            else:
                self._has_pygame = True
                self._use_real = False
        except Exception:
            # fallback to dummy
            self._has_pygame = False
            self._use_real = False
        self._enabled = True

    def disable(self):
        self._enabled = False
        self._use_real = False

    def get_control(self) -> Dict[str, float]:
        """
        Return a control dict. If joystick present and enabled, read axes.
        Otherwise return zeros.
        """
        if not self._enabled or not self._use_real:
            return {"steer": 0.0, "throttle": 0.0, "brake": 0.0}
        try:
            import pygame
            pygame.event.pump()
            steer = float(self._joystick.get_axis(0))
            raw_throttle = float(self._joystick.get_axis(1))
            raw_brake = float(self._joystick.get_axis(2)) if self._joystick.get_numaxes() > 2 else 1.0
            throttle = max(0.0, (1.0 - raw_throttle) / 2.0)
            brake = max(0.0, (1.0 - raw_brake) / 2.0)
            return {"steer": -steer, "throttle": throttle, "brake": brake}
        except Exception:
            return {"steer": 0.0, "throttle": 0.0, "brake": 0.0}
