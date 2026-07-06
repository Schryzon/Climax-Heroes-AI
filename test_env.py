import os
import gymnasium as gym
from stable_baselines3 import PPO

def train_and_eval():
    print("Initializing environment...")
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    
    print("Initializing PPO model on CUDA if available...")
    model = PPO("MlpPolicy", env, verbose=1, device="cuda")
    
    print("Training model for 10,000 timesteps...")
    model.learn(total_timesteps=10000)
    
    print("Saving model...")
    model.save("ppo_cartpole_test")
    
    print("Evaluating model...")
    obs, info = env.reset()
    total_reward = 0
    for _ in range(200):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            obs, info = env.reset()
            break
            
    print(f"Test evaluation completed. Final episode reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    train_and_eval()
