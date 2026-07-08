import os
import sys
import gc
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback

# Add project root to path to resolve src directory imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.env import Climax_Heroes_Env

class Cuda_Cache_Callback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
    def _on_step(self) -> bool:
        return True
    def _on_rollout_end(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

class ClimaxCheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, name_prefix, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.last_save_multiple = 0

    def _on_step(self) -> bool:
        current_steps = self.model.num_timesteps
        current_multiple = current_steps // self.save_freq
        
        # Initialize the baseline multiple on first step to prevent immediate double saves
        if self.last_save_multiple == 0:
            self.last_save_multiple = current_multiple
            
        if current_multiple > self.last_save_multiple:
            self.last_save_multiple = current_multiple
            save_steps = current_multiple * self.save_freq
            os.makedirs(self.save_path, exist_ok=True)
            path = os.path.join(self.save_path, f"{self.name_prefix}_{save_steps}_steps")
            self.model.save(path)
            print(f"\n[Checkpoint] Saved model checkpoint to {path}.zip (Total steps: {current_steps})")
        return True

def train():
    tb_process = None
    print("=" * 60)
    print("         KAMEN RIDER CLIMAX HEROES AI - PPO TRAINING")
    print("=" * 60)
    
    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executing model on: {device.upper()}")
    if device == "cuda":
        print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
        # Print VRAM memory status
        free_mem, total_mem = torch.cuda.mem_get_info()
        print(f"VRAM Info: {free_mem / (1024**3):.2f} GB free / {total_mem / (1024**3):.2f} GB total")
    print("-" * 60)

    # Initialize custom environment (auto-detects emulated game window)
    print("Initializing environment...")
    env = Climax_Heroes_Env(debug=True)
    
    # Configure Checkpoint Callback to save weights at absolute step multiples
    checkpoint_callback = ClimaxCheckpointCallback(
        save_freq=30000,
        save_path="./checkpoints/",
        name_prefix="climax_ppo_model"
    )
    cuda_callback = Cuda_Cache_Callback()
    callbacks = CallbackList([checkpoint_callback, cuda_callback])
    
    # Configure PPO hyperparameters or load existing checkpoints to resume training
    import glob
    save_files = glob.glob("./checkpoints/climax_ppo_model_*.zip") + ["climax_ppo_interrupted.zip", "climax_ppo_final.zip"]
    save_files = [f for f in save_files if os.path.exists(f)]
    
    model = None
    is_resumed = False
    if save_files:
        latest_save = max(save_files, key=os.path.getmtime)
        print(f"Found existing saved model weights: {latest_save}")
        print("Resuming training from loaded weights...")
        try:
            model = PPO.load(latest_save, env=env, device=device, custom_objects={"learning_rate": 1.5e-4, "target_kl": 0.025, "n_steps": 2048})
            is_resumed = True
        except Exception as e:
            print(f"Warning: Direct checkpoint load failed ({e}).")
            print("Action space mismatch detected. Performing Net Surgery to salvage weights...")
            try:
                # 1. Initialize a fresh model with the new action space shape
                model = PPO(
                    "CnnPolicy",
                    env,
                    learning_rate=1.5e-4,
                    n_steps=2048,
                    batch_size=128,
                    n_epochs=4,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    ent_coef=0.08,
                    vf_coef=0.5,
                    max_grad_norm=0.5,
                    target_kl=0.025,
                    verbose=1,
                    tensorboard_log="./tb_logs/",
                    device=device
                )
                
                # 2. Extract policy parameters and metadata from the old zip file
                from stable_baselines3.common.save_util import load_from_zip_file
                data, params, _ = load_from_zip_file(latest_save, device=device)
                
                if "policy" in params:
                    state_dict = params["policy"]
                    # Remove mismatched action net weights
                    keys_to_remove = ["action_net.weight", "action_net.bias"]
                    for k in keys_to_remove:
                        if k in state_dict:
                            del state_dict[k]
                    # Warm-start feature extractor and value heads
                    model.policy.load_state_dict(state_dict, strict=False)
                    
                    # 3. Restore the correct accumulated timestep count from the saved metadata
                    if data and "num_timesteps" in data:
                        model.num_timesteps = data["num_timesteps"]
                        
                    print(f"[Net Surgery] Successfully warm-started: Loaded feature extractor & MLP layers!")
                    print(f"              Resuming training steps from: {model.num_timesteps}")
                    is_resumed = True
                else:
                    print("Warning: Policy weights not found in zip. Falling back to fresh training...")
                    model = None
            except Exception as surgery_err:
                print(f"Warning: Net Surgery failed ({surgery_err}). Falling back to fresh training...")
                model = None
            
    if model is None:
        print("Initializing a new model from scratch...")
        model = PPO(
            "CnnPolicy",
            env,
            learning_rate=1.5e-4,
            n_steps=2048,
            batch_size=128,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.08,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=0.025,
            verbose=1,
            tensorboard_log="./tb_logs/",
            device=device
        )
    
    # Start TensorBoard automatically in the background
    import subprocess
    import socket
    
    # Dynamically find the primary local IP address to print the correct link
    local_ip = "127.0.0.1"
    try:
        dummy_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        dummy_socket.connect(("8.8.8.8", 80))
        local_ip = dummy_socket.getsockname()[0]
        dummy_socket.close()
    except Exception:
        pass

    try:
        # Launch TensorBoard binding to 0.0.0.0 (all network interfaces) so it can be viewed on your phone
        tb_process = subprocess.Popen(
            [sys.executable, "-m", "tensorboard.main", "--logdir", "./tb_logs/", "--host", "0.0.0.0", "--port", "6006"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print("[Train] Background TensorBoard server started automatically.")
        print(f"        Access it on your phone/browser at: http://{local_ip}:6006")
    except Exception as e:
        print(f"Warning: Could not start TensorBoard server automatically ({e}).")
        
    print("\nStarting training loop...")
    print("Press Ctrl+C to stop training and save weights.")
    print("=" * 60)
    
    try:
        # Start learning (1 Million steps = ~9 hours of continuous emulated play)
        model.learn(
            total_timesteps=1000000,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=not is_resumed
        )
        print("\nTraining complete! Saving final model weights...")
        model.save("climax_ppo_final")
        
    except KeyboardInterrupt:
        print("\nTraining manually interrupted by user. Saving current model weights...")
        model.save("climax_ppo_interrupted")
        
    finally:
        print("Closing environment...")
        env.close()
        if tb_process is not None:
            print("Stopping TensorBoard server...")
            try:
                tb_process.terminate()
                tb_process.wait(timeout=2.0)
            except Exception:
                pass
        print("Done!")

if __name__ == "__main__":
    train()
