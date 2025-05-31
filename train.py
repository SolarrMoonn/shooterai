from shooter_env import ShooterEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import pygame
import sys
import os

try:
    # Инициализируем pygame
    pygame.init()
    
    # Создаем среду
    print("Creating environment...")
    env = ShooterEnv()
    
    # Создаем модель с улучшенными параметрами
    print("Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        seed=42
    )
    
    # Создаем директорию для сохранения лучшей модели
    os.makedirs("best_model", exist_ok=True)
    
    # Создаем среду для оценки
    eval_env = ShooterEnv()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="best_model",
        log_path="best_model",
        eval_freq=1000,
        deterministic=True,
        render=False
    )
    
    # Обучаем модель
    print("Starting training...")
    model.learn(
        total_timesteps=200000,#величиваем количество шагов обучения
        callback=eval_callback,
        progress_bar=True
    )
    
    # Сохраняем финальную модель
    print("Saving model...")
    model.save("ppo_shooter")
    
except Exception as e:
    print(f"Error occurred: {str(e)}")
    import traceback
    traceback.print_exc()
    
finally:
    # Закрываем среду и pygame
    print("Cleaning up...")
    if 'env' in locals():
        env.close()
    if 'eval_env' in locals():
        eval_env.close()
    pygame.quit()
    sys.exit()