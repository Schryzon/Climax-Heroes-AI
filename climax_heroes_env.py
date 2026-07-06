import time
import cv2
import mss
import numpy as np
import gymnasium as gym
from collections import deque

try:
    import vgamepad as vg
except ImportError:
    vg = None

class ClimaxHeroesEnv(gym.Env):
    """
    Gymnasium Environment wrapper for Kamen Rider Climax Heroes (PS2 / PCSX2).
    AI controls Player 2 (P2) and competes against Player 1 (P1, the human or scripted bot).
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, window_region=None, debug=False):
        super().__init__()
        self.debug = debug
        
        # Action space: 12 discrete macro actions mapped to Xbox controller buttons
        self.action_space = gym.spaces.Discrete(12)
        
        # Observation space: 4 stacked 84x84 grayscale frames
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(4, 84, 84), dtype=np.uint8
        )
        
        self.frame_stack = deque(maxlen=4)
        self.sct = mss.mss()
        
        # Crop region for the game window (to be calibrated)
        # Default to a 640x480 crop area
        self.window_region = window_region or {"top": 100, "left": 100, "width": 640, "height": 480}
        
        # Initialize virtual gamepad
        if vg is not None:
            self.gamepad = vg.VX360Gamepad()
        else:
            self.gamepad = None
            print("Warning: vgamepad not installed or failed to load. Input injection will be simulated.")

        # Game state tracking
        self.prev_p1_hp = 100
        self.prev_p2_hp = 100
        
        # Macro Action Mappings (assuming standard PCSX2 Xbox gamepad binding)
        self.ACTION_MAP = {
            0: self._act_idle,
            1: self._act_walk_fwd,
            2: self._act_walk_back,  # Also guards/blocks
            3: self._act_jump,
            4: self._act_light,
            5: self._act_heavy,
            6: self._act_special,
            7: self._act_normal_finisher,
            8: self._act_rider_finale,
            9: self._act_evade_left,
            10: self._act_evade_right,
            11: self._act_charge_gauge
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Release all gamepad buttons to a neutral state
        self._release_all()
        
        # Initialize frame stack with empty frames
        obs_frame = self._get_obs_frame()
        for _ in range(4):
            self.frame_stack.append(obs_frame)
            
        self.prev_p1_hp = 100
        self.prev_p2_hp = 100
        
        return self._get_stacked_obs(), {}

    def step(self, action):
        # 1. Execute action through virtual controller
        self._send_action(action)
        
        # 2. Wait for frame time (30fps -> 33ms step rate)
        time.sleep(1.0 / 30.0)
        
        # 3. Grab new frame
        obs_frame = self._get_obs_frame()
        self.frame_stack.append(obs_frame)
        stacked_obs = self._get_stacked_obs()
        
        # 4. Extract rewards and check round status
        p1_hp, p2_hp = self._read_hps(obs_frame)
        combo_count = self._read_combo_count(obs_frame)
        
        reward = self._calculate_reward(p1_hp, p2_hp, combo_count)
        
        # Update HP memory
        self.prev_p1_hp = p1_hp
        self.prev_p2_hp = p2_hp
        
        # Check if round is over (either HP hits zero or round ends)
        terminated = (p1_hp <= 0 or p2_hp <= 0)
        truncated = False
        
        return stacked_obs, reward, terminated, truncated, {}

    def close(self):
        self.sct.close()
        if self.gamepad is not None:
            del self.gamepad

    # --- Observation Capture ---
    def _get_obs_frame(self):
        # Capture the defined screen region
        img = np.array(self.sct.grab(self.window_region))
        # Convert to grayscale and resize to 84x84
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        resized = cv2.resize(gray, (84, 84))
        return resized

    def _get_stacked_obs(self):
        return np.stack(self.frame_stack, axis=0)

    # --- Screen parsing / Reward calculation ---
    def _read_hps(self, frame):
        """
        Extract HP values for P1 and P2.
        TODO: Calibrate pixel offsets for Health Bars in PCSX2 window.
        Returns: tuple of (p1_hp, p2_hp) scaled 0-100
        """
        # Placeholders - to be replaced by OCR or precise pixel-value checks on HUD
        return self.prev_p1_hp, self.prev_p2_hp

    def _read_combo_count(self, frame):
        """
        Reads combo hit counter.
        TODO: Calibrate crop region for hit-counter and run OCR/template-matching.
        """
        return 0

    def _calculate_reward(self, p1_hp, p2_hp, combo_count):
        # AI wants P1 HP to drop (dealing damage) and P2 HP to remain high (avoiding damage)
        damage_dealt = max(0, self.prev_p1_hp - p1_hp)
        damage_taken = max(0, self.prev_p2_hp - p2_hp)
        
        reward = (damage_dealt * 1.0) - (damage_taken * 1.2)  # slightly penalize taking damage more
        
        # Small reward for maintaining combos
        if combo_count > 0:
            reward += combo_count * 0.1
            
        return reward

    # --- Virtual Controller Action Implementations ---
    def _send_action(self, action):
        if self.gamepad is None:
            return
        
        # Reset buttons to neutral first, then press the chosen macro action
        self._release_all()
        self.ACTION_MAP[action]()
        self.gamepad.update()

    def _release_all(self):
        if self.gamepad is None:
            return
        # Reset all standard buttons
        self.gamepad.reset()
        # Reset joysticks/d-pad to neutral
        self.gamepad.left_joystick(x_value=0, y_value=0)
        self.gamepad.update()

    # Define Xbox controller mapping corresponding to the PS2 buttons:
    # Square=X, Triangle=Y, Circle=B, X=A, L1=LB, R1=RB, L2=LT, R2=RT
    def _act_idle(self):
        pass

    def _act_walk_fwd(self):
        # Press left stick full right (assuming P2 is facing left initially, this holds forward)
        self.gamepad.left_joystick(x_value=32767, y_value=0)

    def _act_walk_back(self):
        # Press left stick full left (hold back to walk back or guard)
        self.gamepad.left_joystick(x_value=-32768, y_value=0)

    def _act_jump(self):
        # Left stick up
        self.gamepad.left_joystick(x_value=0, y_value=32767)

    def _act_light(self):
        # Square is mapped to Xbox X
        self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_X)

    def _act_heavy(self):
        # Triangle is mapped to Xbox Y
        self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_Y)

    def _act_special(self):
        # X is mapped to Xbox A
        self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)

    def _act_normal_finisher(self):
        # Circle is mapped to Xbox B
        self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)

    def _act_rider_finale(self):
        # R2 is mapped to Xbox RT (Right Trigger)
        # Note: triggers in vgamepad are analog from 0 to 255
        self.gamepad.right_trigger(value=255)

    def _act_evade_left(self):
        # L1 is mapped to Xbox LB
        self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER)

    def _act_evade_right(self):
        # R1 is mapped to Xbox RB
        self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)

    def _act_charge_gauge(self):
        # Down on left stick
        self.gamepad.left_joystick(x_value=0, y_value=-32768)
