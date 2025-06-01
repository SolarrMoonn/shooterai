from shooter_env import ShooterEnv
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
import pygame
import sys
import os

try:
    pygame.init()
    env = ShooterEnv()
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        action_noise=None,
        verbose=1,
        seed=42
    )
    
    os.makedirs("best_model", exist_ok=True)
    eval_env = ShooterEnv()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="best_model",
        log_path="best_model",
        eval_freq=1000,
        deterministic=True,
        render=False
    )
    
    model.learn(
        total_timesteps=10000,
        callback=eval_callback,
        progress_bar=True
    )
    
    model.save("sac_shooter")
    
except Exception as e:
    print(f"Error occurred: {str(e)}")
    import traceback
    traceback.print_exc()
    
finally:
    if 'env' in locals():
        env.close()
    if 'eval_env' in locals():
        eval_env.close()
    pygame.quit()
    sys.exit()