import time
from enum import IntEnum
import vgamepad as vg

class Climax_Action(IntEnum):
    IDLE = 0
    WALK_FWD = 1
    WALK_BACK = 2
    JUMP = 3
    LIGHT = 4
    HEAVY = 5
    SPECIAL = 6
    NORMAL_FINISHER = 7
    RIDER_FINALE = 8
    EVADE_LEFT = 9
    EVADE_RIGHT = 10
    CHARGE_GAUGE = 11
    FORM_CHANGE = 12
    CANCEL_RIGHT = 13
    CANCEL_LEFT = 14
    RUNNING_LIGHT_RIGHT = 15
    RUNNING_LIGHT_LEFT = 16
    RUNNING_HEAVY_RIGHT = 17
    RUNNING_HEAVY_LEFT = 18

class Gamepad_Executor:
    def __init__(self, gamepad=None):
        self.gamepad = gamepad

    def set_gamepad(self, gamepad):
        self.gamepad = gamepad

    def execute_action(self, action: Climax_Action, prev_action: Climax_Action = None):
        if self.gamepad is None:
            return

        # Hold actions (Walk Forward, Walk Backward, Charge Gauge) do not release
        # if the action is repeated consecutively, preventing stutters/charge cancels.
        if action in [Climax_Action.WALK_FWD, Climax_Action.WALK_BACK, Climax_Action.CHARGE_GAUGE] and prev_action == action:
            pass
        else:
            self.release_all()

        # Map and execute the action
        if action == Climax_Action.IDLE:
            pass
        elif action == Climax_Action.WALK_FWD:
            self.gamepad.left_joystick(x_value=32767, y_value=0)
        elif action == Climax_Action.WALK_BACK:
            self.gamepad.left_joystick(x_value=-32768, y_value=0)
        elif action == Climax_Action.JUMP:
            self.gamepad.left_joystick(x_value=0, y_value=32767)
        elif action == Climax_Action.LIGHT:
            self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
        elif action == Climax_Action.HEAVY:
            self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_Y)
        elif action == Climax_Action.SPECIAL:
            self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        elif action == Climax_Action.NORMAL_FINISHER:
            self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
        elif action == Climax_Action.RIDER_FINALE:
            self.gamepad.right_trigger(value=255)
        elif action == Climax_Action.EVADE_LEFT:
            self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER)
        elif action == Climax_Action.EVADE_RIGHT:
            self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)
        elif action == Climax_Action.CHARGE_GAUGE:
            self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN)
        elif action == Climax_Action.FORM_CHANGE:
            self.gamepad.left_trigger(value=255)
        elif action == Climax_Action.CANCEL_RIGHT:
            self._execute_cancel(vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT)
        elif action == Climax_Action.CANCEL_LEFT:
            self._execute_cancel(vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT)
        elif action == Climax_Action.RUNNING_LIGHT_RIGHT:
            self._execute_running_attack(vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT, vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
        elif action == Climax_Action.RUNNING_LIGHT_LEFT:
            self._execute_running_attack(vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT, vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
        elif action == Climax_Action.RUNNING_HEAVY_RIGHT:
            self._execute_running_attack(vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT, vg.XUSB_BUTTON.XUSB_GAMEPAD_Y)
        elif action == Climax_Action.RUNNING_HEAVY_LEFT:
            self._execute_running_attack(vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT, vg.XUSB_BUTTON.XUSB_GAMEPAD_Y)

        self.gamepad.update()

    def release_all(self):
        if self.gamepad is None:
            return
        self.gamepad.reset()
        self.gamepad.left_joystick(x_value=0, y_value=0)
        self.gamepad.left_trigger(value=0)
        self.gamepad.right_trigger(value=0)
        self.gamepad.update()

    def _execute_cancel(self, dpad_btn):
        if self.gamepad is None:
            return
        
        # Rapid double-tap direction to cancel the current attack animation
        self.gamepad.press_button(button=dpad_btn)
        self.gamepad.update()
        time.sleep(0.02)
        
        self.gamepad.release_button(button=dpad_btn)
        self.gamepad.update()
        time.sleep(0.02)
        
        self.gamepad.press_button(button=dpad_btn)
        self.gamepad.update()
        time.sleep(0.02)
        
        self.gamepad.release_button(button=dpad_btn)
        self.gamepad.update()

    def _execute_running_attack(self, dpad_btn, attack_btn):
        if self.gamepad is None:
            return
        
        # 1. Tap direction
        self.gamepad.press_button(button=dpad_btn)
        self.gamepad.update()
        time.sleep(0.02)
        
        # 2. Release direction
        self.gamepad.release_button(button=dpad_btn)
        self.gamepad.update()
        time.sleep(0.02)
        
        # 3. Hold direction (running)
        self.gamepad.press_button(button=dpad_btn)
        self.gamepad.update()
        time.sleep(0.35)
        
        # 4. Press attack button while holding direction
        self.gamepad.press_button(button=attack_btn)
        self.gamepad.update()
        time.sleep(0.03)
        
        # 5. Release all
        self.gamepad.release_button(button=dpad_btn)
        self.gamepad.release_button(button=attack_btn)
        self.gamepad.update()
