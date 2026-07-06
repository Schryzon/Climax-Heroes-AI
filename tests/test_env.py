import time
import sys
import os

# Add project root to path to resolve src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.env import ClimaxHeroesEnv

def test_custom_env():
    print("Initializing ClimaxHeroesEnv (this will auto-detect the game window)...")
    try:
        env = ClimaxHeroesEnv(debug=True)
    except Exception as e:
        print(f"Failed to initialize environment: {e}")
        sys.exit(1)
        
    print("Resetting environment...")
    obs, info = env.reset()
    print(f"Observation stack shape: {obs.shape}")
    
    print("\nRunning environment for 100 steps with random actions...")
    print("Move your characters or hit each other in-game to see stats update!")
    print("-" * 90)
    print(f"{'Step':<6} | {'P1 HP (Max 300)':<15} | {'P2 HP (Max 300)':<15} | {'P1 Guard':<8} | {'P2 Guard':<8} | {'P1 Rider':<8} | {'P2 Rider':<8} | {'Reward':<6}")
    print("-" * 90)
    
    for i in range(100):
        # Sample a random action (0 to 11)
        action = env.action_space.sample()
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Print state metrics
        p1_hp = env.prev_p1_hp
        p2_hp = env.prev_p2_hp
        p1_guard = env.prev_p1_guard
        p2_guard = env.prev_p2_guard
        p1_rider = env.prev_p1_rider
        p2_rider = env.prev_p2_rider
        
        print(f"{i:<6} | {p1_hp:<15.1f} | {p2_hp:<15.1f} | {p1_guard:<8.1f} | {p2_guard:<8.1f} | {p1_rider:<8.1f} | {p2_rider:<8.1f} | {reward:<6.2f}")
        
        if terminated or truncated:
            print("Round terminated! Resetting...")
            env.reset()
            
    print("-" * 90)
    print("Test completed successfully!")
    env.close()

if __name__ == "__main__":
    test_custom_env()
