import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import cv2
import mss
import pygetwindow as gw
from collections import deque
import pygame

# Optional import for virtual controller
try:
    import vgamepad as vg
except ImportError:
    vg = None

# Import modular components
from src.actions import Climax_Action, Gamepad_Executor
from src.hud import Hud_Parser
from src.rewards import Reward_Calculator

class Climax_Heroes_Env(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, window_region=None, debug=False):
        super().__init__()
        self.debug = debug
        
        # Action space: 19 discrete actions mapped via Climax_Action enum
        self.action_space = gym.spaces.Discrete(19)
        
        # Observation space: 4 stacked 84x84 grayscale frames
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(4, 84, 84), dtype=np.uint8
        )
        
        self.frame_stack = deque(maxlen=4)
        self.sct = mss.mss()
        
        # Auto-detect game window if window_region not specified
        if window_region is None:
            self.window_region = self._detect_game_window()
        else:
            self.window_region = window_region

        # Instantiate modular handlers
        self.hud_parser = Hud_Parser()
        self.reward_calculator = Reward_Calculator(debug=debug)
        
        # Initialize virtual gamepad
        self.gamepad = None
        if vg is not None:
            self.gamepad = vg.VX360Gamepad()
            print("[Env] Persistent virtual gamepad initialized.")
        else:
            print("Warning: vgamepad not installed or failed to load. Input injection will be simulated.")
            
        self.gamepad_executor = Gamepad_Executor(self.gamepad)

        # Initialize pygame for manual physical controller override
        pygame.init()
        pygame.joystick.init()
        
        detected_joysticks = []
        for i in range(pygame.joystick.get_count()):
            try:
                j = pygame.joystick.Joystick(i)
                j.init()
                detected_joysticks.append(j)
            except Exception:
                pass
                
        # Dynamically identify the virtual controller to exclude it from manual takeover overrides
        self.virtual_joystick_guid = None
        self.virtual_joystick_name = None
        if self.gamepad is not None and len(detected_joysticks) > 0:
            # Press the START button on the virtual gamepad
            self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_START)
            self.gamepad.update()
            time.sleep(0.1) # Wait for Windows XInput driver registration
            pygame.event.pump()
            
            for j in detected_joysticks:
                is_virtual = False
                try:
                    # Check button 7 (Xbox START button index)
                    if j.get_numbuttons() > 7 and j.get_button(7):
                        is_virtual = True
                except Exception:
                    pass
                    
                # Fallback: check if ANY button is pressed
                if not is_virtual:
                    try:
                        for b in range(j.get_numbuttons()):
                            if j.get_button(b):
                                is_virtual = True
                                break
                    except Exception:
                        pass
                        
                if is_virtual:
                    try:
                        self.virtual_joystick_guid = j.get_guid()
                    except AttributeError:
                        pass
                    self.virtual_joystick_name = j.get_name()
                    print(f"[Env] Identified virtual controller to exclude: {j.get_name()} (GUID: {self.virtual_joystick_guid})")
                    break
            
            # Release the START button
            self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_START)
            self.gamepad.update()
            
        self.joysticks = []
        self.physical_joy_ids = set()
        self._rebuild_joysticks()

        self.last_user_input_time = 0.0
        self.last_action = Climax_Action.IDLE
        self.prev_action = Climax_Action.IDLE
        self.round_steps = 0
        self.zero_hp_streak = 0
        self.need_rematch = False

    def _rebuild_joysticks(self):
        # Gracefully release old joysticks
        if hasattr(self, 'joysticks'):
            for j in self.joysticks:
                try:
                    j.quit()
                except Exception:
                    pass
        pygame.joystick.quit()
        pygame.joystick.init()
        
        detected_joysticks = []
        for i in range(pygame.joystick.get_count()):
            try:
                j = pygame.joystick.Joystick(i)
                j.init()
                detected_joysticks.append(j)
            except Exception:
                pass
                
        self.joysticks = []
        self.physical_joy_ids = set()
        for i, j in enumerate(detected_joysticks):
            try:
                if self.virtual_joystick_guid and j.get_guid() == self.virtual_joystick_guid:
                    continue
                if self.virtual_joystick_name and j.get_name() == self.virtual_joystick_name and not self.virtual_joystick_guid:
                    continue
            except Exception:
                pass
            self.joysticks.append(j)
            self.physical_joy_ids.add(i)
            print(f"[Env] Registered physical joystick for manual override: {j.get_name()} (ID: {i})")

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
        self.gamepad_executor.release_all()
        self.reward_calculator.reset()
        
        # Capture first frame of the new round
        raw_img = np.array(self.sct.grab(self.window_region))
        gray = cv2.cvtColor(raw_img, cv2.COLOR_BGRA2GRAY)
        obs_frame = cv2.resize(gray, (84, 84))
        
        for _ in range(4):
            self.frame_stack.append(obs_frame)
            
        self.round_steps = 0
        self.zero_hp_streak = 0
        self.last_action = Climax_Action.IDLE
        self.prev_action = Climax_Action.IDLE
        
        return self._get_stacked_obs(), {}

    def step(self, action_idx):
        # Convert raw index to Climax_Action enum
        action = Climax_Action(action_idx)
        
        # Check for manual joystick override takeover
        self._check_user_takeover()
        
        # If user pressed a button within the last 4.0 seconds, silence AI and freeze training steps
        if time.time() - self.last_user_input_time < 4.0:
            self.gamepad_executor.release_all()
            if self.debug:
                print("[Takeover] Manual override active. Silencing AI and pausing PPO steps...")
                
            # Block training loop execution until 4.0 seconds of user inactivity
            while time.time() - self.last_user_input_time < 4.0:
                time.sleep(0.1)
                self._check_user_takeover()
                
            if self.debug:
                print("[Takeover] User inactive. Rebuilding visual frame stack and resuming PPO steps...")
                
            # Grab a fresh frame after resuming to clear stale visual history
            raw_img = np.array(self.sct.grab(self.window_region))
            gray = cv2.cvtColor(raw_img, cv2.COLOR_BGRA2GRAY)
            obs_frame = cv2.resize(gray, (84, 84))
            for _ in range(4):
                self.frame_stack.append(obs_frame)

        self.prev_action = self.last_action
        self.last_action = action
        self.round_steps += 1
        
        # 1. Execute action through virtual controller
        self.gamepad_executor.execute_action(action, self.prev_action)
        
        # 2. Wait for frame time (30fps -> 33ms step rate)
        time.sleep(1.0 / 30.0)
        
        # 3. Grab new high-resolution frame
        raw_img = np.array(self.sct.grab(self.window_region))
        
        # Convert to observation frame (84x84 grayscale)
        gray = cv2.cvtColor(raw_img, cv2.COLOR_BGRA2GRAY)
        obs_frame = cv2.resize(gray, (84, 84))
        self.frame_stack.append(obs_frame)
        stacked_obs = self._get_stacked_obs()
        
        # Convert captured screen to HSV once (single-pass conversion)
        hsv_full = cv2.cvtColor(raw_img, cv2.COLOR_BGRA2HSV)
        
        # 4. Extract rewards and check round status
        p1_hp, p2_hp = self.hud_parser.read_hps(hsv_full)
        p1_guard, p2_guard = self.hud_parser.read_guard_gauges(hsv_full)
        p1_rider, p2_rider = self.hud_parser.read_rider_gauges(hsv_full)
        combo_count = self.hud_parser.read_combo_count(hsv_full)
        p1_rounds, p2_rounds = self.hud_parser.read_rounds_won(hsv_full)
        is_infinite = self.hud_parser.is_timer_infinite(hsv_full)
        
        # Check if a Rider Finale successfully connected and initiated a cinematic cutscene.
        # If Action 8 was executed and the Rider Gauge was consumed (from >= 95 to < 10),
        # we pause training dynamically until the cinematic cutscene finishes and the HUD reappears.
        if action == Climax_Action.RIDER_FINALE or self.prev_action == Climax_Action.RIDER_FINALE:
            if self.reward_calculator.prev_p1_rider >= 95.0 and p1_rider < 10.0:
                if self.debug:
                    print("[Cutscene] Rider Finale connected! Pausing training dynamically...")
                
                # 1. Sleep baseline of 4.0 seconds to allow cinematic to start and HUD to hide
                time.sleep(4.0)
                
                # 2. Poll every 200ms for health bar reappearance (resumption of combat HUD)
                max_wait = 12.0  # safety cap
                start_wait = time.time()
                while time.time() - start_wait < max_wait:
                    raw_img = np.array(self.sct.grab(self.window_region))
                    hsv_temp = cv2.cvtColor(raw_img, cv2.COLOR_BGRA2HSV)
                    h1, h2 = self.hud_parser.read_hps(hsv_temp)
                    if h1 > 0.0 or h2 > 0.0:
                        if self.debug:
                            print(f"[Cutscene] HUD detected. Resumed combat after {time.time() - start_wait + 4.0:.1f} seconds.")
                        break
                    time.sleep(0.2)
                
                # Re-read states and rebuild stack after cutscene completes
                raw_img = np.array(self.sct.grab(self.window_region))
                hsv_full = cv2.cvtColor(raw_img, cv2.COLOR_BGRA2HSV)
                p1_hp, p2_hp = self.hud_parser.read_hps(hsv_full)
                p1_guard, p2_guard = self.hud_parser.read_guard_gauges(hsv_full)
                p1_rider, p2_rider = self.hud_parser.read_rider_gauges(hsv_full)
                combo_count = self.hud_parser.read_combo_count(hsv_full)
                p1_rounds, p2_rounds = self.hud_parser.read_rounds_won(hsv_full)
                is_infinite = self.hud_parser.is_timer_infinite(hsv_full)
                
                gray = cv2.cvtColor(raw_img, cv2.COLOR_BGRA2GRAY)
                obs_frame = cv2.resize(gray, (84, 84))
                for _ in range(4):
                    self.frame_stack.append(obs_frame)
                stacked_obs = self._get_stacked_obs()

        reward = self.reward_calculator.calculate_reward(
            p1_hp, p2_hp, p1_guard, p2_guard, p1_rider, p2_rider, combo_count, 
            p1_rounds, p2_rounds, is_infinite, action, self.prev_action, self.round_steps
        )
        
        # AI runs infinitely in a single continuous episode
        terminated = False
        truncated = False
        
        return stacked_obs, reward, terminated, truncated, {}

    def _check_user_takeover(self):
        for event in pygame.event.get():
            # Handle joystick connection hot-plugging dynamically
            if event.type in [pygame.JOYDEVICEADDED, pygame.JOYDEVICEREMOVED]:
                self._rebuild_joysticks()
                continue
                
            joy_id = getattr(event, 'joy', None)
            if joy_id is not None and joy_id in self.physical_joy_ids:
                if event.type in [pygame.JOYBUTTONDOWN, pygame.JOYHATMOTION]:
                    self.last_user_input_time = time.time()
                elif event.type == pygame.JOYAXISMOTION:
                    # Ignore analog drift noise with a 0.3 threshold deadzone
                    if abs(event.value) > 0.3:
                        self.last_user_input_time = time.time()

    def close(self):
        self.sct.close()
        # Keep the virtual gamepad connected during the Python process lifecycle
        if self.gamepad is not None:
            self.gamepad_executor.release_all()

    def _get_stacked_obs(self):
        return np.stack(self.frame_stack, axis=0)
