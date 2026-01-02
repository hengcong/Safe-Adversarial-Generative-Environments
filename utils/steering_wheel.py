# utils/steering_wheel.py
# Project-level steering wheel helper (wrapper around envs/observation/utilities/steering_wheel.py)
from envs.observation.utilities.steering_wheel import SteeringWheel

# keep a singleton helper to use in training/demo
_default_wheel = SteeringWheel(enable_joystick=False)

def enable_wheel():
    _default_wheel.enable()

def disable_wheel():
    _default_wheel.disable()

def get_control():
    return _default_wheel.get_control()
