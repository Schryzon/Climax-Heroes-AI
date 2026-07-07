import os
import sys
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback

# Add project root to path to resolve src directory imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.env import ClimaxHeroesEnv

class CudaCacheCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
    def _on_step(self) -> bool:
        return True
    def _on_rollout_end(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def train():
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
    env = ClimaxHeroesEnv(debug=True)
    
    # Configure Checkpoint Callback to save weights periodically (every ~38 mins of play)
    checkpoint_callback = CheckpointCallback(
        save_freq=30000,
        save_path="./checkpoints/",
        name_prefix="climax_ppo_model"
    )
    cuda_callback = CudaCacheCallback()
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
            model = PPO.load(latest_save, env=env, device=device)
            is_resumed = True
        except Exception as e:
            print(f"Warning: Failed to load saved weights ({e}).")
            print("Action space size or model shape may have changed. Fallback to initializing from scratch...")
            model = None
            
    if model is None:
        print("Initializing a new model from scratch...")
        model = PPO(
            "CnnPolicy",
            env,
            learning_rate=2.5e-4,
            n_steps=512,
            batch_size=128,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.08,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log="./tb_logs/",
            device=device
        )
    
    print("\nStarting training loop...")
    print("To monitor training, run the following in another terminal:")
    print("  python312 -m tensorboard --logdir ./tb_logs/")
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
        print("Done!")

if __name__ == "__main__":
    train()
