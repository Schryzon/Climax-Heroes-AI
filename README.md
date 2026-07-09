# Kamen Rider: Climax Heroes AI

<p align="center">
  <img src="assets/banner.png" alt="Climax Heroes AI Banner" width="80%" />
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" /></a>
  <a href="https://gymnasium.farama.org/"><img src="https://img.shields.io/badge/Env-Gymnasium-009688?style=flat-square" alt="Gymnasium" /></a>
  <a href="https://stable-baselines3.readthedocs.io/"><img src="https://img.shields.io/badge/RL-Stable--Baselines3-FF5722?style=flat-square" alt="Stable-Baselines3" /></a>
  <a href="https://pcsx2.net/"><img src="https://img.shields.io/badge/Emulator-PCSX2-005C97?style=flat-square" alt="PCSX2" /></a>
  <a href="https://github.com/ViGEm/ViGEmBus"><img src="https://img.shields.io/badge/Gamepad-ViGEmBus-673AB7?style=flat-square" alt="ViGEmBus" /></a>
  <img src="https://img.shields.io/badge/Platform-Windows-0078D6?style=flat-square&logo=windows&logoColor=white" alt="Platform" />
</p>

---

A high-performance Reinforcement Learning (RL) environment wrapper designed to train **Hiyori**, a custom agent, to play **Kamen Rider: Climax Heroes (PS2)** on Windows using the PCSX2 emulator. By combining real-time screen capture, OpenCV-based state parsing, and virtual controller emulation, Hiyori learns complex movement, defense, combos, and finishers.

---

## Table of Contents
1. [Project Architecture](#project-architecture)
2. [Current Project State & Key Optimizations](#current-project-state--key-optimizations)
3. [Prerequisites & Setup](#prerequisites--setup)
4. [Step-by-Step Execution Guide](#step-by-step-execution-guide)
   * [Step 1: Test Gamepad & Input Injection](#step-1-test-gamepad--input-injection)
   * [Step 2: Calibrate Bounding Boxes](#step-2-calibrate-bounding-boxes)
   * [Step 3: Run Console Verification](#step-3-run-console-verification)
   * [Step 4: Train Hiyori](#step-4-train-hiyori)
   * [Step 5: Live Progress Monitoring (PC & Mobile Phone)](#step-5-live-progress-monitoring-pc--mobile-phone)
   * [Step 6: Run the Evaluation Arena](#step-6-run-the-evaluation-arena)
5. [State Space & Action Mappings](#state-space--action-mappings)
6. [In-Game HUD Coordinate Mapping](#in-game-hud-coordinate-mapping)
7. [Detailed Reward Formulation](#detailed-reward-formulation)
8. [Disclaimer & License](#disclaimer--license)

---

## Project Architecture

The project has a modular, object-oriented structure:

```
Climax-Heroes-AI/
├── src/
│   ├── env.py                # Gymnasium environment orchestrator (Climax_Heroes_Env)
│   ├── actions.py            # Climax_Action enums and virtual gamepad driver (Gamepad_Executor)
│   ├── hud.py                # OpenCV-based visual screen parsing (Hud_Parser)
│   ├── rewards.py            # Symmetrical reward formulations (Reward_Calculator)
│   └── evaluate.py           # Multi-mode interactive arena for AI, CPU, and human matches
├── tests/
│   └── test_env.py           # Console dashboard test for real-time stats & rewards
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## Current Project State & Key Optimizations

Hiyori is backed by a highly optimized, custom-engineered framework. Key achievements include:

*   **98.7% Image Processing Optimization:** Previously, the environment converted the entire 1080p frame from BGR to HSV at each step. The pipeline now crops HUD bounding boxes in raw BGR first (a zero-copy operation in NumPy) and converts *only* those tiny crops to HSV. This drastically reduces CPU load and keeps laptops cool.
*   **Net Surgery (Warm-Starting):** When changing the action space shape (e.g. from 22 to 32 actions), standard checkpoint loaders fail. We implemented a custom state-dict transplant script in [train.py](src/train.py) that strips the shape-mismatched output layers while warm-starting the full CNN feature extractor and hidden MLP policy/value networks.
*   **Step Count & Callback Continuity:** Integrates checkpoint step counts dynamically. Re-loading progress restores the exact accumulated global timestep count (`model.num_timesteps`) in the TensorBoard logs and console progress bars. Checkpoints are automatically saved at absolute multiples (e.g., 60k, 90k, 120k) via a custom `ClimaxCheckpointCallback`.
*   **150ms Emulator Hold Time:** Analog triggers (LT/RT) mapping to Form Change (L2) and Rider Finale (R2) require sustained button presses to register in the PCSX2 emulator. We implemented a 150ms hold sleep in [actions.py](src/actions.py) to prevent input drops.
*   **Bi-directional Finisher Detection:** The environment parses screen layout transitions. It detects both when Hiyori connects a Rider Finale and when the opponent hits Hiyori with a Rider Finale (indicated by the opponent's full meter preceding a sudden HUD collapse). In both cases, the training loop is paused dynamically for the duration of the cinematic cutscene.

---

## Prerequisites & Setup

### 1. Game & Emulator Setup
*   Open the game in **PCSX2**.
*   Set screen layout to **16:9 fullscreen (1024x576 or 1920x1080 resolution)**.
*   Ensure P1 controller in PCSX2 is mapped to the virtual Xbox 360 controller, and P2 controller is mapped to your keyboard or physical gamepad.

### 2. Gamepad Bus Driver
This project emulates Xbox 360 gamepads using the **ViGEmBus** driver.
*   Download and install the driver installer: [ViGEmBus Releases](https://github.com/ViGEm/ViGEmBus/releases).

### 3. Installation
Install the required packages in Python 3.12:
```powershell
python312 -m pip install -r requirements.txt
```

---

## Step-by-Step Execution Guide

### Step 1: Test Gamepad & Input Injection
Verify that the virtual gamepad driver is correctly installed and accessible by Python:
```powershell
python312 .\tools\test_gamepad.py
```

### Step 2: Calibrate Bounding Boxes
Verify that your PCSX2 window matches the normalized coordinates in the HUD parser:
```powershell
python312 .\tools\screen_capture_helper.py
```
This saves `game_capture_annotated.png` in the root folder. Open it to check if health bars, shields, and gauges are correctly boxed.

### Step 3: Run Console Verification
Run the custom environment with random actions to inspect the real-time CLI dashboard (shows HP levels, shield values, meter changes, and step logs). Though, ensure a match is currently running and the window is focused on PCSX2:
```powershell
python312 .\tests\test_env.py
```

### Step 4: Train Hiyori
To train the PPO model against the PCSX2 CPU player:
1. In PCSX2, boot up Kamen Rider Climax Heroes and enter **Training Mode**.
2. Select **Decade** or **Dark Decade** with Blade's Form Change (recommended) for P1 (Hiyori).
3. Select **Blade** (recommended) for P2 (Opponent).
4. Go to training settings and configure:
   * Toggle CPU level to **2 stars** (Normal difficulty).
   * Toggle **off HP regeneration** (HP Regen: Off).
   * Toggle **off gauge regeneration** (Gauge Regen: Off).
5. Start the training script:
   ```powershell
   python312 .\src\train.py
   ```
*Press `Ctrl+C` at any time to save progress and stop training safely.*

### Step 5: Live Progress Monitoring (PC & Mobile Phone)
The training script **automatically spawns a background TensorBoard server** at startup binding to `0.0.0.0` (all interfaces) on port `6006`. 
*   **On your PC**: Open `http://localhost:6006` in your browser.
*   **On your phone**: Connect to the same Wi-Fi and open `http://<your-pc-ip>:6006` (the script prints your exact local IP URL on launch!).

### Step 6: Run the Evaluation Arena
Test Hiyori's skills using the interactive [evaluate.py](src/evaluate.py) script:
1. Ensure **two physical/virtual controllers are plugged in/mapped** in PCSX2 settings (the game will not allow 2-Player VS Mode if only one gamepad is detected).
2. Start the game in **VS Mode** (2-Player match).
3. **Start the battle first** (so the fighters are on screen), then run the evaluation script:
   ```powershell
   python312 .\src\evaluate.py
   ```
This opens an interactive menu supporting:
1.  **Hiyori vs CPU** (AI P1 vs CPU P2) [Default] *(No need for two controllers)*
2.  **Hiyori vs Hiyori** (AI P1 vs AI P2 - spawns a second virtual controller for Port 2)
3.  **Hiyori vs Me** (AI P1 vs Human P2 - lets you fight Hiyori directly with your own physical controller)
*Simply press `Enter` to load your latest progress and fight the CPU!*
*   **Note on Playstyle:** By default, the evaluation arena runs Hiyori in stochastic (non-deterministic) mode (`deterministic=False`). This allows her to sample from her learned action distribution, producing the same dynamic and organic playstyle observed during training. If you want her to play strictly deterministically, you can edit `src/evaluate.py`.

---

## State Space & Action Mappings

The environment represents the game state using **4 stacked $84 \times 84$ grayscale frames** (capturing the last 133ms of movement).

The policy outputs an integer corresponding to one of **32 discrete macro actions** in [Climax_Action](src/actions.py#L5):

| Index | Enum Action | Physical Mapping |
| :--- | :--- | :--- |
| `0` | `IDLE` | Guard / Stand Neutral |
| `1` | `WALK_FWD` | Left Stick Right (Walk Fwd) |
| `2` | `WALK_BACK` | Left Stick Left (Walk Back) |
| `3` | `JUMP` | Left Stick Up |
| `4` | `LIGHT` | Xbox `X` (Weak Combo) |
| `5` | `HEAVY` | Xbox `Y` (Strong Combo) |
| `6` | `SPECIAL` | Xbox `A` (Special Move / 2 bars) |
| `7` | `NORMAL_FINISHER` | Xbox `B` (Signature Strike) |
| `8` | `RIDER_FINALE` | Xbox `RT` (Ult / 5 bars) |
| `9` | `EVADE_LEFT` | Xbox `LB` (Evade Fwd) |
| `10` | `EVADE_RIGHT` | Xbox `RB` (Evade Back) |
| `11` | `CHARGE_GAUGE` | D-pad Down (Charge Meter) |
| `12` | `FORM_CHANGE` | Xbox `LT` (Form Change / 5 bars) |
| `13` | `CANCEL_RIGHT` | D-pad Double-Tap Right (Attack Cancel) |
| `14` | `CANCEL_LEFT` | D-pad Double-Tap Left (Attack Cancel) |
| `15` | `RUNNING_LIGHT_RIGHT` | Dash Right + Xbox `X` |
| `16` | `RUNNING_LIGHT_LEFT` | Dash Left + Xbox `X` |
| `17` | `RUNNING_HEAVY_RIGHT` | Dash Right + Xbox `Y` |
| `18` | `RUNNING_HEAVY_LEFT` | Dash Left + Xbox `Y` |
| `19` | `RUN_RIGHT` | Dash Right (Hold) |
| `20` | `RUN_LEFT` | Dash Left (Hold) |
| `21` | `RIDER_KICK` | D-pad Up + Xbox `B` (Simultaneous) |
| `22` | `LIGHT_DOWN` | D-pad Down + Xbox `X` (Crouching Weak) |
| `23` | `HEAVY_DOWN` | D-pad Down + Xbox `Y` (Crouching Heavy/Launcher) |
| `24` | `SPECIAL_DOWN` | D-pad Down + Xbox `A` (Crouching Special) |
| `25` | `FINISHER_DOWN` | D-pad Down + Xbox `B` (Crouching Finisher) |
| `26` | `LIGHT_RIGHT` | D-pad Right + Xbox `X` |
| `27` | `LIGHT_LEFT` | D-pad Left + Xbox `X` |
| `28` | `HEAVY_RIGHT` | D-pad Right + Xbox `Y` (Forward Grapple/Throw) |
| `29` | `HEAVY_LEFT` | D-pad Left + Xbox `Y` (Forward Grapple/Throw) |
| `30` | `SPECIAL_RIGHT` | D-pad Right + Xbox `A` |
| `31` | `SPECIAL_LEFT` | D-pad Left + Xbox `A` |

---

## In-Game HUD Coordinate Mapping

To parse stats from the emulator, the environment checks specific boundary regions of the 1024 × 576 game window:

<p align="center">
  <img src="assets/hud_annotations.png" alt="HUD Coordinates Mapping" width="90%" />
</p>

| HUD Element | Normalized X-Range | Normalized Y-Range | Absolute X-Range (1024px) | Absolute Y-Range (576px) | Bounding Box |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **P1 HP Bar** | `[0.220, 0.445]` | `[0.087, 0.113]` | `[225, 455]` | `[50, 65]` | **Green** |
| **P2 HP Bar** | `[0.555, 0.780]` | `[0.087, 0.113]` | `[568, 798]` | `[50, 65]` | **Green** |
| **P1 Guard Gauge** | `[0.220, 0.336]` | `[0.120, 0.138]` | `[225, 344]` | `[69, 79]` | **Cyan** |
| **P2 Guard Gauge** | `[0.664, 0.780]` | `[0.120, 0.138]` | `[679, 798]` | `[69, 79]` | **Cyan** |
| **P1 Rounds Won** | `[0.441, 0.472]` | `[0.137, 0.212]` | `[451, 483]` | `[78, 122]` | **Orange** |
| **P2 Rounds Won** | `[0.528, 0.559]` | `[0.137, 0.212]` | `[540, 572]` | `[78, 122]` | **Orange** |
| **Infinity Timer** | `[0.470, 0.530]` | `[0.060, 0.120]` | `[481, 542]` | `[34, 69]` | **White** |
| **P1 Rider Gauge** | `[0.220, 0.380]` | `[0.884, 0.921]` | `[225, 389]` | `[509, 530]` | **Magenta** |
| **P2 Rider Gauge** | `[0.620, 0.780]` | `[0.884, 0.921]` | `[634, 798]` | `[509, 530]` | **Magenta** |

---

## Detailed Reward Formulation

We use a dense reward framework in [rewards.py](src/rewards.py) to avoid reward-hacking and guide policy optimization:

1.  **HP Trades:**
    *   `+1.0` / `-1.2` multiplier per point of damage dealt / taken.
    *   `+1.5` damage multiplier + `+8.0` flat bonus for landing a `RIDER_FINALE`.
2.  **Shield & Guard:**
    *   `+0.1` per point of opponent shield damage.
    *   `+5.0` / `-5.0` bonus / penalty for Guard Crushes.
    *   `+0.05` / `-0.05` per point of shield damage for successful blocks / failed blocks.
3.  **Special Actions & Dynamic Penalties:**
    *   **Safely Charging:** `+0.30` per unit of meter generated.
    *   **Hit While Charging:** `-15.0` penalty if Hiyori is hit while charging (forces her to back away and create distance before building meter).
    *   **Special Move Whiff Tracking:** When Hiyori uses a Special Attack (A / Cross), a **30-step (1.0 second)** evaluation window starts. If the window closes without dealing damage (whiffed, blocked, or dodged), she receives a **`-2.5` penalty**.
    *   **Rider Kick Whiff Tracking:** When Hiyori executes a Rider Kick (D-pad Up + Xbox `B`), a **45-step (1.5 second)** evaluation window starts. If it whiffs or gets blocked, she receives a **`-3.0` penalty**.
    *   **Opponent Finisher Hit:** `-30.0` penalty if Hiyori is hit by the opponent's Rider Finale (punishes her for failing to defend against major ultimates).
    *   **Form Change / Finisher Use:** `+5.0` bonus every time she uses `FORM_CHANGE` or `RIDER_FINALE` while the meter is full.
    *   **Cancel / Quick Step Logic:** 
        *   **Rider Cancel (In Combat):** If executed during an attack string, it is treated as a Rider Cancel (spends meter) and costs **`-0.5`** to prevent meter wasting.
        *   **Quick Step (In Neutral):** If executed in neutral, it is treated as a Quick Step (free movement) and costs **`-0.08`** (same as a normal dodge) to prevent infinite backstep spamming.
4.  **Red Shoes System (Thematic Berserk Protocol):**
    *   Only active when the match timer is finite.
    *   If the time remaining drops below **20 seconds** AND Hiyori (P1) is trailing in HP (`p1_hp < p2_hp`), the forced-combat **Red Shoes System** triggers (a reference to the secret berserk protocol in Kabuto/Gatack's Zecters).
    *   While active:
        *   **Berserk Offense:** All damage dealt rewards are doubled (`damage_dealt_reward * 2.0`), transforming Hiyori into an unstoppable attacker to ensure survival.
        *   **Deficit Pressure:** Hiyori receives a step penalty based on her HP deficit (`(p2_hp - p1_hp) * 0.05`). This penalty is **doubled** if she is also trailing in round wins, forcing her to relentlessly chase and finish the target.

---

## Disclaimer & License

*   This project is licensed under the **[MIT License](LICENSE)**.
*   **Disclaimer:** This project is an independent, open-source research and educational endeavor. It is not affiliated with, authorized, sponsored, or endorsed by Bandai Namco Entertainment, Sony Interactive Entertainment, or any of their partners or subsidiaries. All trademarks, game content, character designs, and assets belong to their respective owners. No proprietary game files, ROMs, ISOs, or emulator BIOS files are distributed in this repository.
