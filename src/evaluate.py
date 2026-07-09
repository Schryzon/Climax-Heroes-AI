import os
import sys
import glob
import time

# Add parent directory to path to resolve src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.env import Climax_Heroes_Env
from src.actions import Climax_Action, Gamepad_Executor
from stable_baselines3 import PPO

def select_checkpoint(prompt_text):
    checkpoints = sorted(glob.glob("./checkpoints/climax_ppo_model_*.zip") + ["climax_ppo_interrupted.zip", "climax_ppo_final.zip"])
    checkpoints = [f for f in checkpoints if os.path.exists(f)]
    
    if not checkpoints:
        print("No checkpoints found in ./checkpoints/ or root directory!")
        return None
        
    print(f"\n{prompt_text}")
    for idx, cp in enumerate(checkpoints):
        # Format the file path for print
        print(f"  [{idx}] {os.path.basename(cp)}")
        
    # Find default checkpoint (latest modified file)
    default_cp = max(checkpoints, key=os.path.getmtime)
    default_idx = checkpoints.index(default_cp)
    
    while True:
        try:
            choice = input(f"Select checkpoint index [Default: {default_idx} ({os.path.basename(default_cp)})]: ").strip()
            if not choice:
                return default_cp
            choice_idx = int(choice)
            if 0 <= choice_idx < len(checkpoints):
                return checkpoints[choice_idx]
        except ValueError:
            pass
        print("Invalid choice, try again.")

def run_evaluation():
    print("=" * 60)
    print("         KAMEN RIDER CLIMAX HEROES - EVALUATION ARENA")
    print("=" * 60)
    print("Select Arena Mode:")
    print("  [1] Hiyori vs CPU    (AI P1 vs CPU P2) [Default]")
    print("  [2] Hiyori vs Hiyori (AI P1 vs AI P2)")
    print("  [3] Hiyori vs Me     (AI P1 vs Human P2)")
    
    mode = input("Enter choice (1, 2, or 3) [Default: 1]: ").strip()
    if not mode:
        mode = "1"
        
    if mode not in ["1", "2", "3"]:
        print("Invalid mode selection. Exiting.")
        return
        
    checkpoint_p1 = select_checkpoint("Select Checkpoint for P1 (Hiyori):")
    if not checkpoint_p1:
        return
        
    checkpoint_p2 = None
    if mode == "2":
        checkpoint_p2 = select_checkpoint("Select Checkpoint for P2 (Hiyori):")
        if not checkpoint_p2:
            return

    # 1. Initialize the environment with takeover disabled (so P2 human inputs don't freeze P1 AI)
    print("\nInitializing environment...")
    env = Climax_Heroes_Env(debug=False, enable_takeover=False)
    
    # 2. Load P1 model
    print(f"\n[P1] Loading weights from: {os.path.basename(checkpoint_p1)}")
    # Load with custom objects matching training settings
    model1 = PPO.load(checkpoint_p1, env=env, custom_objects={"learning_rate": 1.5e-4, "target_kl": 0.025, "n_steps": 2048})
    
    # 3. Setup P2 model and virtual controller if AI vs AI mode
    model2 = None
    gamepad2 = None
    executor2 = None
    if mode == "2":
        # Import vgamepad dynamically
        import vgamepad as vg
        print(f"[P2] Loading weights from: {os.path.basename(checkpoint_p2)}")
        model2 = PPO.load(checkpoint_p2, env=env, custom_objects={"learning_rate": 1.5e-4, "target_kl": 0.025, "n_steps": 2048})
        gamepad2 = vg.VX360Gamepad()
        executor2 = Gamepad_Executor(gamepad2)
        print("[P2] Port 2 Virtual Gamepad initialized for P2 AI.")

    print("\n" + "=" * 60)
    if mode == "1":
        print("Starting AI vs CPU Match!")
        print("Note: AI controls P1 (Port 1). CPU controls P2 (Port 2).")
        print("Press Ctrl+C in this terminal to stop.")
    elif mode == "2":
        print("Starting AI vs AI Arena!")
        print("Press Ctrl+C in this terminal to stop.")
    else:
        print("Starting AI vs Human Battle!")
        print("Note: AI controls P1 (Port 1). You control P2 (Port 2) via your controller.")
        print("Press Ctrl+C in this terminal to stop.")
    print("=" * 60 + "\n")
    
    # Get initial observation
    obs, info = env.reset()
    prev_action2 = Climax_Action.IDLE
    
    try:
        while True:
            # P1 predicts and steps through the environment
            action_idx1, _ = model1.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action_idx1)
            
            # P2 predicts and executes if in AI vs AI mode
            if mode == "2" and model2 is not None and executor2 is not None:
                action_idx2, _ = model2.predict(obs, deterministic=False)
                action2 = Climax_Action(action_idx2)
                executor2.execute_action(action2, prev_action2)
                prev_action2 = action2
                
    except KeyboardInterrupt:
        print("\nEvaluation Arena stopped by user.")
    finally:
        print("\nClosing environment and releasing controllers...")
        env.close()
        if gamepad2 is not None:
            gamepad2.reset()
            gamepad2.update()
        print("Done!")

if __name__ == "__main__":
    run_evaluation()
