import logging
import pygame
import evdev
from evdev import ecodes, InputDevice

class SteeringWheelController:
    RIGHT_SHIFT_PADDLE = 4
    LEFT_SHIFT_PADDLE = 5
    STEERING_MAKEUP = 1.5

    def __init__(self, disable=False):
        self.disable = disable
        if not self.disable:
            pygame.display.init()
            pygame.joystick.init()
            assert pygame.joystick.get_count() > 0, "Please connect the steering wheel!"
            print("Successfully Connect your Steering Wheel!")

            ffb_device = evdev.list_devices()[0]
            self.ffb_dev = InputDevice(ffb_device)

            self.joystick = pygame.joystick.Joystick(0)

        self.right_shift_paddle = False
        self.left_shift_paddle = False

        self.button_circle = False
        self.button_rectangle = False
        self.button_triangle = False
        self.button_x = False

        self.button_up = False
        self.button_down = False
        self.button_right = False
        self.button_left = False

    def process_input(self, speed_kmh):
        if self.disable:
            return [0.0, 0.0]

        if not self.joystick.get_init():
            self.joystick.init()

        pygame.event.pump()

        
        # print("Num axes: ", self.joystick.get_numaxes())
        # Our wheel can provide values in [-1.5, 1.5].
        steering = (-self.joystick.get_axis(0)) / 1.5  # 0th axis is the wheel

        # 2nd axis is the right paddle. Range from 0 to 1
        # 3rd axis is the middle paddle. Range from 0 to 1
        # Of course then 1st axis is the left paddle.
        # print("Raw throttle: {}, raw brake: {}".format(self.joystick.get_axis(2), self.joystick.get_axis(3)))
        raw_throttle = self.joystick.get_axis(1)
        raw_brake = self.joystick.get_axis(2)
        
        # It is possible that the paddles always return 0 (should be 1 if not pressed) after initialization.
        if abs(raw_throttle) < 1e-6:
            raw_throttle = 1.0 - 1e-6
        if abs(raw_brake) < 1e-6:
            raw_brake = 1.0 - 1e-6
        throttle = (1 - raw_throttle) / 2
        brake = (1 - raw_brake) / 2
        print(-steering * self.STEERING_MAKEUP, (throttle - brake))
        self.right_shift_paddle = True if self.joystick.get_button(self.RIGHT_SHIFT_PADDLE) else False
        self.left_shift_paddle = True if self.joystick.get_button(self.LEFT_SHIFT_PADDLE) else False

        # self.print_debug_message()

        self.button_circle = True if self.joystick.get_button(2) else False
        self.button_rectangle = True if self.joystick.get_button(1) else False
        self.button_triangle = True if self.joystick.get_button(3) else False
        self.button_x = True if self.joystick.get_button(0) else False

        if self.button_x:
            logging.warning("X is pressed. Exit ...")
            raise KeyboardInterrupt()

        self.maybe_pause()

        hat = self.joystick.get_hat(0)
        self.button_up = True if hat[-1] == 1 else False
        self.button_down = True if hat[-1] == -1 else False
        self.button_left = True if hat[0] == -1 else False
        self.button_right = True if hat[0] == 1 else False

        self.feedback(speed_kmh)

        return [-steering * self.STEERING_MAKEUP, (throttle - brake)]

    def maybe_pause(self):
        paused = False
        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN and event.button == 3:  # Triangle button pressed
                paused = not paused  # Toggle pause
                # Wait for button release
                while True:
                    event_happened = False
                    for event in pygame.event.get():
                        if event.type == pygame.JOYBUTTONUP and event.button == 3:
                            event_happened = True
                            break
                    if event_happened:
                        break
                    pygame.time.delay(100)

                # Wait for the next button press to unpause
                while True:
                    event_happened = False
                    for event in pygame.event.get():
                        if event.type == pygame.JOYBUTTONDOWN and event.button == 3:
                            event_happened = True
                            break
                    if event_happened:
                        break
                    pygame.time.delay(100)

                # Button pressed again, unpause
                paused = False

                # Wait for button release before exiting
                while True:
                    event_happened = False
                    for event in pygame.event.get():
                        if event.type == pygame.JOYBUTTONUP and event.button == 3:
                            event_happened = True
                            break
                    if event_happened:
                        break
                    pygame.time.delay(100)

    def reset(self):
        if self.disable:
            # Reset all paddles and buttons to default
            self.right_shift_paddle = False
            self.left_shift_paddle = False
            self.button_circle = False
            self.button_rectangle = False
            self.button_triangle = False
            self.button_x = False
            self.button_up = False
            self.button_down = False
            self.button_right = False
            self.button_left = False
            return

        # Reinitialize pygame if needed
        if not pygame.get_init():
            pygame.init()
        if not pygame.joystick.get_init():
            pygame.joystick.init()

        # Reinitialize the joystick
        if self.joystick.get_init():
            self.joystick.quit()
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

        # Reset paddles and buttons
        self.right_shift_paddle = False
        self.left_shift_paddle = False
        self.button_circle = False
        self.button_rectangle = False
        self.button_triangle = False
        self.button_x = False
        self.button_up = False
        self.button_down = False
        self.button_right = False
        self.button_left = False

        # Clear pygame events
        try:
            pygame.event.clear()  # Clear any remaining events in the queue
        except pygame.error as e:
            logging.warning(f"Unable to clear pygame events: {e}")

        # Reset force feedback
        try:
            val = int(65535)  # Maximum autocenter force
            self.ffb_dev.write(ecodes.EV_FF, ecodes.FF_AUTOCENTER, val)
        except Exception as e:
            logging.warning(f"Failed to reset force feedback: {e}")



    def feedback(self, speed_kmh):
        assert not self.disable
        offset = 5000
        total = 50000
        val = int(total * min(speed_kmh / 80, 1) + offset)
        self.ffb_dev.write(ecodes.EV_FF, ecodes.FF_AUTOCENTER, val)

    def print_debug_message(self):
        msg = "Left: {}, Right: {}, Event: ".format(
            self.joystick.get_button(self.LEFT_SHIFT_PADDLE), self.joystick.get_button(self.RIGHT_SHIFT_PADDLE)
        )
        for e in pygame.event.get():
            msg += str(e.type)
        print(msg)