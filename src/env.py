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
        
        # Action space: 15 discrete macro actions mapped to Xbox controller buttons
        self.action_space = gym.spaces.Discrete(15)
        
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
            11: self._act_charge_gauge,
            12: self._act_form_change,
            13: self._act_cancel_right,
            14: self._act_cancel_left
        }

        # Initialize pygame for manual physical controller override
        import pygame
        pygame.init()
        pygame.joystick.init()
        self.override_joystick = None
        if pygame.joystick.get_count() > 0:
            self.override_joystick = pygame.joystick.Joystick(0)
            self.override_joystick.init()
            print(f"[Env] Detected physical joystick for manual override: {self.override_joystick.get_name()}")
        self.last_user_input_time = 0.0
        self.last_action = 0
        self.round_steps = 0
        self.zero_hp_streak = 0
        self.p1_form_changed = False
        self.p1_finisher_attempted = False
        self.need_rematch = False

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
        
        # Capture first frame of the new round
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
        self.round_steps = 0
        self.zero_hp_streak = 0
        self.p1_form_changed = False
        self.p1_finisher_attempted = False
        
        return self._get_stacked_obs(), {}

    def step(self, action):
        self.prev_action = getattr(self, 'last_action', 0)
        self.last_action = action
        self.round_steps += 1
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
        p1_rounds, p2_rounds = self._read_rounds_won(raw_img)
        is_infinite = self._is_timer_infinite(raw_img)
        
        reward = self._calculate_reward(p1_hp, p2_hp, p1_guard, p2_guard, p1_rider, p2_rider, combo_count, p1_rounds, p2_rounds, is_infinite)
        
        # Update HP and Gauge memory
        self.prev_p1_hp = p1_hp
        self.prev_p2_hp = p2_hp
        self.prev_p1_guard = p1_guard
        self.prev_p2_guard = p2_guard
        self.prev_p1_rider = p1_rider
        self.prev_p2_rider = p2_rider
        
        # AI runs infinitely in a single continuous episode
        terminated = False
        truncated = False
        
        return stacked_obs, reward, terminated, truncated, {}

    def close(self):
        self.sct.close()
        # Keep the virtual gamepad connected during the Python process lifecycle
        # to prevent emulators (PCSX2/Dolphin) from losing Port 1 mapping.
        # It will automatically disconnect when the Python interpreter exits.
        if self.gamepad is not None:
            self._release_all()

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
        
        return (self._estimate_gauge_color(p1_crop), 
                self._estimate_gauge_color(p2_crop))

    def _read_rider_gauges(self, img):
        h, w, _ = img.shape
        p1_x1, p1_x2 = int(w * 0.220), int(w * 0.380)
        p2_x1, p2_x2 = int(w * 0.620), int(w * 0.780)
        y1, y2 = int(h * 0.884), int(h * 0.921)
        
        p1_crop = img[y1:y2, p1_x1:p1_x2]
        p2_crop = img[y1:y2, p2_x1:p2_x2]
        
        return (self._estimate_gauge_color(p1_crop), 
                self._estimate_gauge_color(p2_crop))

    def _estimate_gauge_color(self, crop):
        if crop.size == 0:
            return 0.0
            
        # Strip vertical margins to avoid HUD borders or screen capture marking outlines
        h_crop = crop.shape[0]
        margin = max(1, min(3, int(h_crop * 0.15)))
        crop_stripped = crop[margin:-margin, :]
        
        if crop_stripped.size == 0:
            return 0.0
            
        hsv = cv2.cvtColor(crop_stripped, cv2.COLOR_BGR2HSV)
        
        # Saturation-based threshold (captures highly saturated colors: Red, Blue, Green, Yellow, etc.)
        # and filters out metallic gray/silver casing and black/white boundaries.
        lower_color = np.array([0, 100, 100])
        upper_color = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower_color, upper_color)
            
        col_has_pixels = np.any(mask > 0, axis=0)
        active_cols = np.sum(col_has_pixels)
        cols = len(col_has_pixels)
        
        # Guard/Rider bars fill about 90% of their crop bounding boxes when full
        max_active_fraction = 0.90
        pct = (active_cols / (cols * max_active_fraction)) * 100.0
        return min(100.0, pct)

    def _read_combo_count(self, img):
        # Optional: OCR / template-matching for combo count later
        return 0

    def _read_rounds_won(self, img):
        h, w, _ = img.shape
        p1_x1, p1_x2 = int(w * 0.441), int(w * 0.472)
        p2_x1, p2_x2 = int(w * 0.528), int(w * 0.559)
        y1, y2 = int(h * 0.137), int(h * 0.212) # Extended to 0.212 to capture all the way to the bottom
        
        p1_crop = img[y1:y2, p1_x1:p1_x2]
        p2_crop = img[y1:y2, p2_x1:p2_x2]
        
        if p1_crop.size == 0 or p2_crop.size == 0:
            return 0, 0
            
        hsv_p1 = cv2.cvtColor(p1_crop, cv2.COLOR_BGR2HSV)
        hsv_p2 = cv2.cvtColor(p2_crop, cv2.COLOR_BGR2HSV)
        
        lower_yellow = np.array([15, 80, 80])
        upper_yellow = np.array([45, 255, 255])
        
        mask_p1 = cv2.inRange(hsv_p1, lower_yellow, upper_yellow)
        mask_p2 = cv2.inRange(hsv_p2, lower_yellow, upper_yellow)
        
        p1_yellow_pixels = np.sum(mask_p1 > 0)
        p2_yellow_pixels = np.sum(mask_p2 > 0)
        
        p1_rounds = 0
        if p1_yellow_pixels >= 350: # Scaled threshold for extended height
            p1_rounds = 2
        elif p1_yellow_pixels >= 25:
            p1_rounds = 1
            
        p2_rounds = 0
        if p2_yellow_pixels >= 350:
            p2_rounds = 2
        elif p2_yellow_pixels >= 25:
            p2_rounds = 1
            
        return p1_rounds, p2_rounds

    def _is_timer_infinite(self, img):
        h, w, _ = img.shape
        t_x1, t_x2 = int(w * 0.47), int(w * 0.53)
        t_y1, t_y2 = int(h * 0.06), int(h * 0.12)
        
        timer_crop = img[t_y1:t_y2, t_x1:t_x2]
        if timer_crop.size == 0:
            return False
            
        hsv = cv2.cvtColor(timer_crop, cv2.COLOR_BGR2HSV)
        # Mint green HSV range for the infinity symbol
        lower_mint = np.array([40, 30, 150])
        upper_mint = np.array([80, 180, 255])
        mask = cv2.inRange(hsv, lower_mint, upper_mint)
        
        return np.sum(mask > 0) > 100

    def _is_survival_game_over(self, img):
        h, w, _ = img.shape
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Crop the bottom banner area: y = [0.60, 0.85], x = [0.10, 0.90]
        # In 1024x575: y: 345 to 488, x: 102 to 921
        banner_area = hsv[345:488, 102:921]
        
        # Teal HSV range for the banner background
        lower_teal = np.array([85, 100, 100])
        upper_teal = np.array([115, 255, 255])
        mask_teal = cv2.inRange(banner_area, lower_teal, upper_teal)
        
        # Crop the Proceed button pill area on the right: y = [380, 460], x = [700, 880]
        proceed_hsv = hsv[380:460, 700:880]
        
        # Red HSV range for the Circle button prompt inside the pill
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        mask_red1 = cv2.inRange(proceed_hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(proceed_hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        return np.sum(mask_teal > 0) > 30000 and np.sum(mask_red > 0) > 100

    def _is_vs_splash_screen(self, img):
        h, w, _ = img.shape
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Crop the upper sky region of the screen (y = 10% to 45%) where the artificial green backdrop boxes live.
        # This completely filters out natural green trees and grass on the ground (e.g. on River Sunset stages).
        sky_hsv = hsv[int(h * 0.10):int(h * 0.45), :]
        
        # Green backdrop boxes (Hue: 35-85, Sat: 100-255, Val: 100-255)
        lower_green = np.array([35, 100, 100])
        upper_green = np.array([85, 255, 255])
        mask_green = cv2.inRange(sky_hsv, lower_green, upper_green)
        
        # Orange/red background canvas (Hue: 10-30, Sat: 100-255, Val: 100-255)
        lower_orange = np.array([10, 100, 100])
        upper_orange = np.array([30, 255, 255])
        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
        
        return np.sum(mask_green > 0) > 20000 and np.sum(mask_orange > 0) > 40000

    def _is_now_loading_screen(self, img):
        h, w, _ = img.shape
        
        # Crop around the pink card symbol on the bottom black bar:
        # x = [0.50, 0.60], y = [0.78, 0.94]
        c_x1, c_x2 = int(w * 0.50), int(w * 0.60)
        c_y1, c_y2 = int(h * 0.78), int(h * 0.94)
        
        card_crop = img[c_y1:c_y2, c_x1:c_x2]
        if card_crop.size == 0:
            return False
            
        hsv = cv2.cvtColor(card_crop, cv2.COLOR_BGR2HSV)
        lower_pink = np.array([140, 80, 80])
        upper_pink = np.array([170, 255, 255])
        mask_pink = cv2.inRange(hsv, lower_pink, upper_pink)
        
        return np.sum(mask_pink > 0) > 100

    def _calculate_reward(self, p1_hp, p2_hp, p1_guard, p2_guard, p1_rider, p2_rider, combo_count, p1_rounds, p2_rounds, is_infinite):
        # 1. HP damage dealt (to P2) vs taken (by P1)
        damage_dealt = max(0.0, self.prev_p2_hp - p2_hp)
        damage_taken = max(0.0, self.prev_p1_hp - p1_hp)
        
        # Supercharge damage dealt rewards
        if damage_dealt > 0:
            if self.last_action == 8:
                damage_dealt_reward = (damage_dealt * 3.0) + 25.0  # Massive finisher hit payoff
            elif p2_guard == 0.0:
                damage_dealt_reward = damage_dealt * 2.0  # No Mercy: double reward for hitting a guard-broken stun state!
            else:
                damage_dealt_reward = damage_dealt * 1.0
        else:
            damage_dealt_reward = 0.0
            
        reward = damage_dealt_reward - (damage_taken * 1.2)
        
        # Penalize missed/wasted finisher (action 8 executed but dealt 0 damage)
        if damage_dealt == 0 and self.last_action == 8:
            reward -= 15.0
        
        # 2. Guard Gauge change (shield management: P1 is AI, P2 is Opponent)
        # Opponent's guard gauge reduction (we want to crush their shield - increased from 0.1 to 0.3 to reward heavy pressure)
        guard_dealt = max(0.0, self.prev_p2_guard - p2_guard)
        reward += guard_dealt * 0.3
        
        # Guard Crush bonus (breaking opponent's shield completely)
        if p2_guard == 0.0 and self.prev_p2_guard > 0.0:
            reward += 15.0
        
        # AI's guard gauge reduction (we only penalize if they also took HP damage, i.e. failed block)
        # If they took no HP damage, it means they blocked successfully, so we reward it!
        guard_taken = max(0.0, self.prev_p1_guard - p1_guard)
        if guard_taken > 0:
            if damage_taken > 0:
                reward -= guard_taken * 0.15  # Failed block / hit
            else:
                reward += guard_taken * 0.20  # Successful block (absorbed hit on shield!)
                
        # 3. Rider Gauge change & Potential (generating special meter is good for AI P1)
        rider_gained = max(0.0, p1_rider - self.prev_p1_rider)
        reward += rider_gained * 0.30  # Encourage active charging
        
        # Minor dodge action cost to prevent infinite invincibility-frame spamming
        if self.last_action in [9, 10]:
            reward -= 0.08

        # Cancel penalty: Hiyori must only cancel during an attack string.
        # If she cancels (Action 13 or 14) when the previous action was not an attack, penalize it!
        if self.last_action in [13, 14]:
            prev_action = getattr(self, 'prev_action', 0)
            if prev_action not in [4, 5, 6, 7, 8]:
                reward -= 0.15  # Naked cancel penalty in neutral

        # Form Change (L2) and Finisher (R2) attempt rewards when meter is full (prev_p1_rider >= 95.0)
        # We only reward the first attempt per round to prevent spamming/exploit loops!
        if self.prev_p1_rider >= 95.0:
            if self.last_action == 12 and not self.p1_form_changed:  # Form Change (L2)
                reward += 5.0
                self.p1_form_changed = True
                if self.debug:
                    print("[Reward] First Form Change triggered! +5.0 bonus.")
            elif self.last_action == 8 and not self.p1_finisher_attempted:  # Rider Finale (R2)
                reward += 5.0
                self.p1_finisher_attempted = True
                if self.debug:
                    print("[Reward] First Rider Finale triggered! +5.0 bonus.")
        
        # 4. Combo bonus
        if combo_count > 0:
            reward += combo_count * 0.1
            
        # 5. Desperation Mode (only active when timer is finite)
        if not is_infinite:
            time_left = max(0.0, 99.0 - (self.round_steps / 30.0))
            if time_left < 20.0 and p1_hp < p2_hp:
                # We are losing by HP and time is running out!
                # Double all damage dealt rewards in desperation phase
                if damage_dealt > 0:
                    reward += damage_dealt_reward * 1.0  # extra 1.0x (total 2.0x, or 4.0x if guard is crushed!)
                # Step penalty based on HP deficit, forcing the AI to attack aggressively to close the gap
                hp_deficit = p2_hp - p1_hp
                deficit_penalty = hp_deficit * 0.05
                
                # If we are also down on rounds won (opponent has won a round, and we haven't), double the deficit penalty!
                if p2_rounds > p1_rounds:
                    deficit_penalty *= 2.0
                    
                reward -= deficit_penalty
            
        return reward

    # --- Virtual Controller Action Implementations ---
    def _send_action(self, action):
        if self.gamepad is None:
            return
         # Check for physical gamepad override input across all connected controllers
        import pygame
        pygame.event.pump()
        user_active = False
        
        for j_idx in range(pygame.joystick.get_count()):
            try:
                j = pygame.joystick.Joystick(j_idx)
                if not j.get_init():
                    j.init()
                
                # Check D-pad (hats)
                for i in range(j.get_numhats()):
                    if j.get_hat(i) != (0, 0):
                        user_active = True
                        break
                
                # Check face buttons
                if not user_active:
                    for i in range(j.get_numbuttons()):
                        if j.get_button(i):
                            user_active = True
                            break
            except Exception:
                pass
                
        if user_active:
            # User is actively pressing buttons or moving sticks
            self.last_user_input_time = time.time()
            
        # If user pressed a button within the last 4.0 seconds, silence AI inputs
        if time.time() - self.last_user_input_time < 4.0:
            self._release_all()
            return
        
        # Reset buttons to neutral first, then press the chosen macro action
        self._release_all()
        if self.gamepad is not None:
            self.ACTION_MAP[action]()
            self.gamepad.update()

    def _release_all(self):
        if self.gamepad is None:
            if vg is not None:
                self.gamepad = vg.VX360Gamepad()
                print("[Env] Gamepad re-initialized dynamically.")
            else:
                return
        # Reset all standard buttons
        self.gamepad.reset()
        # Reset joysticks/d-pad to neutral
        self.gamepad.left_joystick(x_value=0, y_value=0)
        self.gamepad.left_trigger(value=0)
        self.gamepad.right_trigger(value=0)
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
        self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN)

    def _act_form_change(self):
        self.gamepad.left_trigger(value=255)

    def _act_cancel_right(self):
        self._execute_cancel(vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT)

    def _act_cancel_left(self):
        self._execute_cancel(vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT)

    def _execute_cancel(self, dpad_btn):
        if self.gamepad is None:
            return
        
        # Rapid double-tap direction to cancel the current attack animation
        # 1. Press direction
        self.gamepad.press_button(button=dpad_btn)
        self.gamepad.update()
        time.sleep(0.02)
        
        # 2. Release direction
        self.gamepad.release_button(button=dpad_btn)
        self.gamepad.update()
        time.sleep(0.02)
        
        # 3. Press direction again
        self.gamepad.press_button(button=dpad_btn)
        self.gamepad.update()
        time.sleep(0.02)
        
        # 4. Release direction
        self.gamepad.release_button(button=dpad_btn)
        self.gamepad.update()
