import time
import cv2
import mss
import numpy as np
import gymnasium as gym
import pygetwindow as gw
from collections import deque
import ctypes

# Make Python DPI-aware on Windows to prevent display scaling from cropping screen capture
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2) # PROCESS_PER_MONITOR_DPI_AWARE
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

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
        
        # Locate game window dynamically if not specified
        if window_region is None:
            self.window_region = self._detect_game_window()
        else:
            self.window_region = window_region
        
        # Initialize virtual gamepad
        if vg is not None:
            self.gamepad = vg.VX360Gamepad()
        else:
            self.gamepad = None
            print("Warning: vgamepad not installed or failed to load. Input injection will be simulated.")

        # Game state tracking (HP is 300 max: Green=201-300, Yellow=101-200, Red=0-100)
        self.prev_p1_hp = 300.0
        self.prev_p2_hp = 300.0
        self.prev_p1_guard = 100.0
        self.prev_p2_guard = 100.0
        self.prev_p1_rider = 0.0
        self.prev_p2_rider = 0.0
        
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

    def _detect_game_window(self):
        keywords = ["仮面ライダー", "PCSX2", "Dolphin", "Climax Heroes"]
        for kw in keywords:
            windows = gw.getWindowsWithTitle(kw)
            if not windows:
                all_titles = gw.getAllTitles()
                windows = [gw.getWindowsWithTitle(t)[0] for t in all_titles if kw.lower() in t.lower()]
            if windows:
                win = windows[0]
                print(f"[Env] Found game window: '{win.title}' ({win.width}x{win.height})")
                return {"top": win.top, "left": win.left, "width": win.width, "height": win.height}
        # Fallback: Use primary monitor dimensions to prevent screenshot scaling/boundary errors
        monitor = self.sct.monitors[1]
        print(f"[Env] Warning: Game window not found. Using primary monitor bounds ({monitor['width']}x{monitor['height']}).")
        return {"top": monitor["top"], "left": monitor["left"], "width": monitor["width"], "height": monitor["height"]}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Release all gamepad buttons to a neutral state
        self._release_all()
        
        # Capture first frame
        raw_img = np.array(self.sct.grab(self.window_region))
        gray = cv2.cvtColor(raw_img, cv2.COLOR_BGRA2GRAY)
        obs_frame = cv2.resize(gray, (84, 84))
        
        for _ in range(4):
            self.frame_stack.append(obs_frame)
            
        self.prev_p1_hp = 300.0
        self.prev_p2_hp = 300.0
        self.prev_p1_guard = 100.0
        self.prev_p2_guard = 100.0
        self.prev_p1_rider = 0.0
        self.prev_p2_rider = 0.0
        
        return self._get_stacked_obs(), {}

    def step(self, action):
        # 1. Execute action through virtual controller
        self._send_action(action)
        
        # 2. Wait for frame time (30fps -> 33ms step rate)
        time.sleep(1.0 / 30.0)
        
        # 3. Grab new high-resolution frame
        raw_img = np.array(self.sct.grab(self.window_region))
        
        # Convert to observation frame (84x84 grayscale)
        gray = cv2.cvtColor(raw_img, cv2.COLOR_BGRA2GRAY)
        obs_frame = cv2.resize(gray, (84, 84))
        self.frame_stack.append(obs_frame)
        stacked_obs = self._get_stacked_obs()
        
        # 4. Extract rewards and check round status
        p1_hp, p2_hp = self._read_hps(raw_img)
        p1_guard, p2_guard = self._read_guard_gauges(raw_img)
        p1_rider, p2_rider = self._read_rider_gauges(raw_img)
        combo_count = self._read_combo_count(raw_img)
        
        reward = self._calculate_reward(p1_hp, p2_hp, p1_guard, p2_guard, p1_rider, p2_rider, combo_count)
        
        # Update HP and Gauge memory
        self.prev_p1_hp = p1_hp
        self.prev_p2_hp = p2_hp
        self.prev_p1_guard = p1_guard
        self.prev_p2_guard = p2_guard
        self.prev_p1_rider = p1_rider
        self.prev_p2_rider = p2_rider
        
        # Check if round is over (either HP hits zero)
        terminated = (p1_hp <= 0 or p2_hp <= 0)
        truncated = False
        
        return stacked_obs, reward, terminated, truncated, {}

    def close(self):
        self.sct.close()
        if self.gamepad is not None:
            del self.gamepad

    # --- Observation Capture ---
    def _get_stacked_obs(self):
        return np.stack(self.frame_stack, axis=0)

    # --- Screen parsing / State extraction ---
    def _read_hps(self, img):
        h, w, _ = img.shape
        p1_x1, p1_x2 = int(w * 0.220), int(w * 0.445)
        p2_x1, p2_x2 = int(w * 0.555), int(w * 0.780)
        y1, y2 = int(h * 0.087), int(h * 0.113)
        
        p1_crop = img[y1:y2, p1_x1:p1_x2]
        p2_crop = img[y1:y2, p2_x1:p2_x2]
        
        return (self._estimate_layered_hp(p1_crop, is_p1=True), 
                self._estimate_layered_hp(p2_crop, is_p1=False))

    def _estimate_layered_hp(self, crop, is_p1=True):
        if crop.size == 0:
            return 0.0
            
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        # Color ranges:
        # Green (Stack 3)
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        # Yellow (Stack 2)
        lower_yellow = np.array([15, 50, 50])
        upper_yellow = np.array([35, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Red (Stack 1 - red wraps around 0 and 180 in HSV)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([15, 255, 255])
        lower_red2 = np.array([165, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        def get_fraction(mask):
            col_has_active = np.any(mask > 0, axis=0)
            active_cols = np.sum(col_has_active)
            cols = len(col_has_active)
            # Calibrated padding divisor (since green bar doesn't stretch 100% of the box)
            max_active_fraction = 0.945
            pct = (active_cols / (cols * max_active_fraction)) * 100.0
            return min(100.0, pct)

        # Check stack levels sequentially (Green -> Yellow -> Red)
        green_pct = get_fraction(mask_green)
        if green_pct > 5.0: # 5% buffer to avoid boundary noise
            return 200.0 + green_pct
            
        yellow_pct = get_fraction(mask_yellow)
        if yellow_pct > 5.0:
            return 100.0 + yellow_pct
            
        red_pct = get_fraction(mask_red)
        if red_pct > 5.0:
            return red_pct
            
        return 0.0

    def _read_guard_gauges(self, img):
        h, w, _ = img.shape
        p1_x1, p1_x2 = int(w * 0.220), int(w * 0.336)
        p2_x1, p2_x2 = int(w * 0.664), int(w * 0.780)
        y1, y2 = int(h * 0.120), int(h * 0.138)
        
        p1_crop = img[y1:y2, p1_x1:p1_x2]
        p2_crop = img[y1:y2, p2_x1:p2_x2]
        
        return (self._estimate_metric(p1_crop, is_p1=True), 
                self._estimate_metric(p2_crop, is_p1=False))

    def _read_rider_gauges(self, img):
        h, w, _ = img.shape
        p1_x1, p1_x2 = int(w * 0.220), int(w * 0.380)
        p2_x1, p2_x2 = int(w * 0.620), int(w * 0.780)
        y1, y2 = int(h * 0.884), int(h * 0.921)
        
        p1_crop = img[y1:y2, p1_x1:p1_x2]
        p2_crop = img[y1:y2, p2_x1:p2_x2]
        
        return (self._estimate_metric(p1_crop, is_p1=True), 
                self._estimate_metric(p2_crop, is_p1=False))

    def _estimate_metric(self, crop, is_p1=True):
        if crop.size == 0:
            return 0.0
            
        # Grayscale thresholding for general gauge brightness detection
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
            
        # Count total columns containing active pixels
        col_has_active = np.any(mask > 0, axis=0)
        active_cols = np.sum(col_has_active)
        cols = len(col_has_active)
        
        max_active_fraction = 0.96
        pct = (active_cols / (cols * max_active_fraction)) * 100.0
        return min(100.0, pct)

    def _read_combo_count(self, img):
        # Optional: OCR / template-matching for combo count later
        return 0

    def _calculate_reward(self, p1_hp, p2_hp, p1_guard, p2_guard, p1_rider, p2_rider, combo_count):
        # 1. HP damage dealt (to P2) vs taken (by P1)
        damage_dealt = max(0.0, self.prev_p2_hp - p2_hp)
        damage_taken = max(0.0, self.prev_p1_hp - p1_hp)
        reward = (damage_dealt * 1.0) - (damage_taken * 1.2)
        
        # 2. Guard Gauge change (shield management: P1 is AI, P2 is Opponent)
        guard_dealt = max(0.0, self.prev_p2_guard - p2_guard)
        guard_taken = max(0.0, self.prev_p1_guard - p1_guard)
        reward += (guard_dealt * 0.1) - (guard_taken * 0.15)
        
        # 3. Rider Gauge change (generating special meter is good for AI P1)
        rider_gained = max(0.0, p1_rider - self.prev_p1_rider)
        reward += rider_gained * 0.05
        
        # 4. Combo bonus
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

    # Xbox controller mappings corresponding to the PS2 buttons:
    # Square=X, Triangle=Y, Circle=B, X=A, L1=LB, R1=RB, L2=LT, R2=RT
    def _act_idle(self):
        pass

    def _act_walk_fwd(self):
        self.gamepad.left_joystick(x_value=32767, y_value=0)

    def _act_walk_back(self):
        self.gamepad.left_joystick(x_value=-32768, y_value=0)

    def _act_jump(self):
        self.gamepad.left_joystick(x_value=0, y_value=32767)

    def _act_light(self):
        self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_X)

    def _act_heavy(self):
        self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_Y)

    def _act_special(self):
        self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)

    def _act_normal_finisher(self):
        self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)

    def _act_rider_finale(self):
        self.gamepad.right_trigger(value=255)

    def _act_evade_left(self):
        self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER)

    def _act_evade_right(self):
        self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)

    def _act_charge_gauge(self):
        self.gamepad.left_joystick(x_value=0, y_value=-32768)
