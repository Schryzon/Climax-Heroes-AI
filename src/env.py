import ctypes
# Make Python DPI-aware on Windows to prevent display scaling from cropping screen capture
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2) # PROCESS_PER_MONITOR_DPI_AWARE
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import cv2
import mss
import pygetwindow as gw
from collections import deque
import pygame
import gc

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

    def __init__(self, window_region=None, debug=False, enable_takeover=True, enable_logging=False):
        super().__init__()
        self.debug = debug
        self.enable_takeover = enable_takeover
        self.enable_logging = enable_logging
        
        # Action space: 31 discrete actions mapped via Climax_Action enum
        self.action_space = gym.spaces.Discrete(31)
        
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

        # Action frequency logging
        self.action_counts = {action: 0 for action in Climax_Action}
        self.total_action_steps = 0

        # Round-level training statistics tracking
        self.round_index = 0
        self.round_form_changes = 0
        self.round_rider_finales = 0
        self.round_rider_kicks = 0
        self.round_support_uses = 0
        self.round_total_reward = 0.0
        self.round_damage_dealt = 0.0
        self.round_damage_taken = 0.0
        
        # Initialize CSV logging file
        if self.enable_logging:
            import os
            import csv
            self.csv_path = "checkpoints/hiyori_training_stats.csv"
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            if not os.path.exists(self.csv_path):
                with open(self.csv_path, mode='w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "round_index", "outcome", "duration_steps", "p1_final_hp", "p2_final_hp",
                        "total_damage_dealt", "total_damage_taken", "total_reward",
                        "form_changes", "rider_finales", "rider_kicks",
                        "rider_kick_hits", "rider_kick_whiffs",
                        "support_uses", "support_hits", "support_whiffs",
                        "red_shoes_steps", "attack_cancels", "timestamp"
                    ])

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
            # Press the BACK button on the virtual gamepad to identify it (prevents pausing the game)
            self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_BACK)
            self.gamepad.update()
            time.sleep(0.1) # Wait for Windows XInput driver registration
            pygame.event.pump()
            
            for j in detected_joysticks:
                is_virtual = False
                try:
                    # Check button 6 (Xbox BACK/SELECT button index)
                    if j.get_numbuttons() > 6 and j.get_button(6):
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
            
            # Release the BACK button
            self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_BACK)
            self.gamepad.update()
            
        self.joysticks = []
        self.physical_joy_ids = set()
        self._rebuild_joysticks()
        # Flush the initial startup connection events to prevent redundant rebuild logs in the first steps
        pygame.event.clear()

        self.last_user_input_time = 0.0
        self.last_action = Climax_Action.IDLE
        self.prev_action = Climax_Action.IDLE
        self.round_steps = 0
        self.zero_hp_streak = 0
        self.need_rematch = False
        self.charge_persist_steps = 0

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
            try:
                self.physical_joy_ids.add(j.get_instance_id())
            except AttributeError:
                pass
            try:
                self.physical_joy_ids.add(j.get_id())
            except AttributeError:
                pass
            self.physical_joy_ids.add(i)
            print(f"[Env] Registered physical joystick for manual override: {j.get_name()} (ID: {i})")

    def _detect_game_window(self):
        keywords = ["仮面ライダー", "PCSX2", "Dolphin", "Climax Heroes"]
        self.game_win = None
        for kw in keywords:
            windows = gw.getWindowsWithTitle(kw)
            if not windows:
                all_titles = gw.getAllTitles()
                windows = [gw.getWindowsWithTitle(t)[0] for t in all_titles if kw.lower() in t.lower()]
            if windows:
                win = windows[0]
                self.game_win = win
                print(f"[Env] Found game window: '{win.title}' ({win.width}x{win.height})")
                return {"top": win.top, "left": win.left, "width": win.width, "height": win.height}
        # Fallback: Use primary monitor dimensions to prevent screenshot scaling/boundary errors
        monitor = self.sct.monitors[1]
        print(f"[Env] Warning: Game window not found. Using primary monitor bounds ({monitor['width']}x{monitor['height']}).")
        return {"top": monitor["top"], "left": monitor["left"], "width": monitor["width"], "height": monitor["height"]}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Bring game window to focus dynamically if detected
        if hasattr(self, 'game_win') and self.game_win is not None:
            try:
                self.game_win.activate()
                time.sleep(0.2)
            except Exception:
                pass
                
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
        self.charge_persist_steps = 0
        
        return self._get_stacked_obs(), {}

    def step(self, action_idx):
        # Convert raw index to Climax_Action enum
        action = Climax_Action(action_idx)
        
        # Assisted Exploration / Action Injection & Redirection:
        # If meter is above 75.0/80.0 and we are not in form change, redirect specials/charge or inject.
        import random
        if (self.reward_calculator.prev_p1_rider > 75.0 and 
            not getattr(self.reward_calculator, 'p1_in_form', False)):
            if action == Climax_Action.CHARGE_GAUGE:
                action = Climax_Action.FORM_CHANGE
                if self.debug:
                    print("[Action Redirection] Redirected CHARGE_GAUGE to FORM_CHANGE due to high meter.")
            elif (self.reward_calculator.prev_p1_rider >= 80.0 and 
                  action == Climax_Action.SUPPORT):
                action = Climax_Action.FORM_CHANGE
                if self.debug:
                    print("[Action Redirection] Redirected SUPPORT to FORM_CHANGE due to high meter.")
            elif random.random() < 0.20:
                if self.reward_calculator.prev_p1_rider >= 95.0:
                    action = random.choices(
                        [Climax_Action.FORM_CHANGE, Climax_Action.RIDER_FINALE],
                        weights=[0.6, 0.4]
                    )[0]
                else:
                    action = Climax_Action.FORM_CHANGE
                if self.debug:
                    print(f"[Action Injection] Forced {action.name} to assist exploration.")

        # Action Persistence (Sticky Charging):
        # Force a minimum duration of 60 steps (~2.0s) for charging to guarantee meter gains.
        # But if form change is active, disable charging completely and redirect to IDLE as a fallback.
        if getattr(self.reward_calculator, 'p1_in_form', False):
            self.charge_persist_steps = 0
            if action == Climax_Action.CHARGE_GAUGE:
                action = Climax_Action.IDLE

        if self.charge_persist_steps > 0:
            action = Climax_Action.CHARGE_GAUGE
            self.charge_persist_steps -= 1
        elif action == Climax_Action.CHARGE_GAUGE:
            self.charge_persist_steps = 60
        
        # Check for manual joystick override takeover
        if self.enable_takeover:
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
        
        # Increment action frequency counters
        self.action_counts[action] += 1
        self.total_action_steps += 1
        
        if action == Climax_Action.FORM_CHANGE:
            self.round_form_changes += 1
        elif action == Climax_Action.RIDER_FINALE:
            self.round_rider_finales += 1
        elif action == Climax_Action.RIDER_KICK:
            self.round_rider_kicks += 1
        elif action == Climax_Action.SUPPORT:
            self.round_support_uses += 1
        
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
        
        # Extract BGR frame for HUD parsing (takes views, zero-copy)
        bgr_full = raw_img[:, :, :3]
        
        # 4. Extract rewards and check round status
        p1_hp, p2_hp = self.hud_parser.read_hps(bgr_full)
        
        # Accumulate round damage
        self.round_damage_dealt += max(0.0, self.reward_calculator.prev_p2_hp - p2_hp)
        self.round_damage_taken += max(0.0, self.reward_calculator.prev_p1_hp - p1_hp)
        
        # Break sticky charge lock immediately if damage is taken
        if self.reward_calculator.prev_p1_hp - p1_hp > 0.0:
            self.charge_persist_steps = 0
        
        # Detect round/match reset (both players restored to full HP from a damaged/dead state)
        # to ensure round_steps is kept accurate for Red Shoes System calculation
        if (p1_hp >= 298.0 and p2_hp >= 298.0 and 
            (self.reward_calculator.prev_p1_hp < 290.0 or self.reward_calculator.prev_p2_hp < 290.0)):
            self._log_round_stats()
            self._reset_round_stats()
            if self.debug:
                print(f"[Env] New round/match detected (HP restored to {p1_hp:.1f}/{p2_hp:.1f}). Resetting round steps.")
            self.round_steps = 0
            
        p1_guard, p2_guard = self.hud_parser.read_guard_gauges(bgr_full)
        p1_rider, p2_rider = self.hud_parser.read_rider_gauges(bgr_full)
        combo_count = self.hud_parser.read_combo_count(bgr_full)
        p1_rounds, p2_rounds = self.hud_parser.read_rounds_won(bgr_full)
        is_infinite = self.hud_parser.is_timer_infinite(bgr_full)
        
        # Check if opponent (P2) connected a Rider Finale on Hiyori (P1)
        # Triggered when opponent had full meter, health bars suddenly disappear, and Hiyori did not initiate a finisher.
        opponent_finisher_connected = False
        if (self.reward_calculator.prev_p2_rider >= 95.0 and 
            p1_hp == 0.0 and p2_hp == 0.0 and 
            self.reward_calculator.prev_p1_hp > 0.0 and self.reward_calculator.prev_p2_hp > 0.0 and
            action != Climax_Action.RIDER_FINALE and self.prev_action != Climax_Action.RIDER_FINALE):
            opponent_finisher_connected = True

        # Check if Hiyori (P1) successfully connected a Rider Finale on P2
        hiyori_finisher_connected = False
        if (action == Climax_Action.RIDER_FINALE or self.prev_action == Climax_Action.RIDER_FINALE) and \
           (self.reward_calculator.prev_p1_rider >= 95.0 and p1_rider < 10.0) and \
           (p1_hp == 0.0 and p2_hp == 0.0):
            hiyori_finisher_connected = True

        # Pause training dynamically until the cinematic cutscene finishes and the HUD reappears
        if hiyori_finisher_connected or opponent_finisher_connected:
            if self.debug:
                if hiyori_finisher_connected:
                    print("[Cutscene] Hiyori Rider Finale connected! Pausing training dynamically...")
                else:
                    print("[Cutscene] Opponent Rider Finale connected! Hiyori hit! Pausing training dynamically...")
                
            # 1. Sleep baseline of 8.0 seconds to allow cinematic to start and HUD to hide
            time.sleep(8.0)
            
            # 2. Poll every 200ms for health bar reappearance (resumption of combat HUD)
            max_wait = 12.0  # safety cap
            start_wait = time.time()
            while time.time() - start_wait < max_wait:
                raw_img = np.array(self.sct.grab(self.window_region))
                h1, h2 = self.hud_parser.read_hps(raw_img[:, :, :3])
                if h1 > 0.0 or h2 > 0.0:
                    if self.debug:
                        print(f"[Cutscene] HUD detected. Resumed combat after {time.time() - start_wait + 8.0:.1f} seconds.")
                    break
                time.sleep(0.2)
            
            # Re-read states and rebuild stack after cutscene completes
            raw_img = np.array(self.sct.grab(self.window_region))
            bgr_full = raw_img[:, :, :3]
            p1_hp, p2_hp = self.hud_parser.read_hps(bgr_full)
            p1_guard, p2_guard = self.hud_parser.read_guard_gauges(bgr_full)
            p1_rider, p2_rider = self.hud_parser.read_rider_gauges(bgr_full)
            combo_count = self.hud_parser.read_combo_count(bgr_full)
            p1_rounds, p2_rounds = self.hud_parser.read_rounds_won(bgr_full)
            is_infinite = self.hud_parser.is_timer_infinite(bgr_full)
            
            gray = cv2.cvtColor(raw_img, cv2.COLOR_BGRA2GRAY)
            obs_frame = cv2.resize(gray, (84, 84))
            for _ in range(4):
                self.frame_stack.append(obs_frame)
            stacked_obs = self._get_stacked_obs()

        reward = self.reward_calculator.calculate_reward(
            p1_hp, p2_hp, p1_guard, p2_guard, p1_rider, p2_rider, combo_count, 
            p1_rounds, p2_rounds, is_infinite, action, self.prev_action, self.round_steps,
            opponent_finisher_connected=opponent_finisher_connected
        )
        
        self.round_total_reward += reward
        
        # AI runs infinitely in a single continuous episode
        terminated = False
        truncated = False
        
        # Periodic CPU garbage collection to ensure no memory bloat/slowdown over long training runs (approx. every 30 seconds of play)
        if self.round_steps % 1000 == 0:
            gc.collect()
        # Periodic Action Distribution Logging (every 10,000 steps)
        if self.total_action_steps % 10000 == 0:
            self.print_action_distribution()
            
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

    def _log_round_stats(self):
        if not self.enable_logging:
            return
        import csv
        from datetime import datetime
        
        self.round_index += 1
        
        # Determine the winner based on final HP values of the round
        p1_final = self.reward_calculator.prev_p1_hp
        p2_final = self.reward_calculator.prev_p2_hp
        
        if p1_final > p2_final:
            outcome = "WIN"
        elif p2_final > p1_final:
            outcome = "LOSS"
        else:
            outcome = "DRAW"
            
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        row = [
            self.round_index,
            outcome,
            self.round_steps,
            p1_final,
            p2_final,
            self.round_damage_dealt,
            self.round_damage_taken,
            self.round_total_reward,
            self.round_form_changes,
            self.round_rider_finales,
            self.round_rider_kicks,
            self.reward_calculator.round_kick_hits,
            self.reward_calculator.round_kick_whiffs,
            self.round_support_uses,
            self.reward_calculator.round_support_hits,
            self.reward_calculator.round_support_whiffs,
            self.reward_calculator.round_red_shoes_steps,
            self.reward_calculator.round_attack_cancels,
            timestamp_str
        ]
        
        try:
            with open(self.csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            print(f"\n[Stats Logger] Round {self.round_index} logged: {outcome} | Steps: {self.round_steps} | DMG Dealt: {self.round_damage_dealt:.1f} | DMG Taken: {self.round_damage_taken:.1f} | Reward: {self.round_total_reward:.2f}")
        except Exception as e:
            print(f"Warning: Failed to log round stats to CSV: {e}")

    def _reset_round_stats(self):
        self.round_form_changes = 0
        self.round_rider_finales = 0
        self.round_rider_kicks = 0
        self.round_support_uses = 0
        self.round_total_reward = 0.0
        self.round_damage_dealt = 0.0
        self.round_damage_taken = 0.0
        # Reset counters in Reward_Calculator as well
        self.reward_calculator.reset()

    def print_action_distribution(self):
        if self.total_action_steps == 0:
            return
        print("\n" + "="*50)
        print("          ACTION DISTRIBUTION PERCENTAGES")
        print("="*50)
        print(f"Total Steps: {self.total_action_steps}")
        print("-"*50)
        # Sort actions by percentage descending
        sorted_actions = sorted(
            self.action_counts.items(),
            key=lambda item: item[1],
            reverse=True
        )
        for act, count in sorted_actions:
            pct = (count / self.total_action_steps) * 100
            print(f"  {act.name:<25} : {pct:>6.2f}% ({count} steps)")
        print("="*50 + "\n")

    def close(self):
        self.print_action_distribution()
        self.sct.close()
        # Keep the virtual gamepad connected during the Python process lifecycle
        if self.gamepad is not None:
            self.gamepad_executor.release_all()

    def _get_stacked_obs(self):
        return np.stack(self.frame_stack, axis=0)
