from stable_baselines3 import PPO
from shooter_env import ShooterEnv
import pygame
import numpy as np
import os
import random
import time

# Инициализируем генераторы случайных чисел
random.seed(int(time.time()))
np.random.seed(int(time.time()))

env = ShooterEnv()
print("Загружаем модель...")

model_path = "ppo_shooter.zip"
if not os.path.exists(model_path):
    print("Модель не найдена! Запусти train.py")
    exit()

model = PPO.load(model_path, env=env, seed=int(time.time()))
print("Модель загружена успешно!")

obs, _ = env.reset()
print("=== НАЧАЛО ИГРЫ ===")
print("Фиолетовый круг - цель")
print("Красный круг - клик ИИ")
print("Счет и шаги отображаются в верхнем левом углу\n")

for step in range(100):
    action, _ = model.predict(obs, deterministic=False)
    obs, reward, done, _, info = env.step(action)
    
    click_x, click_y = env.last_click_x, env.last_click_y
    
    status = f"Шаг {step + 1:3d} | Клик: ({click_x:4.0f}, {click_y:4.0f}) | Расстояние: {info['distance']:6.1f}"
    if reward > 0:
        status += f" | ПОПАДАНИЕ! +1 (награда: {reward:4.1f})"
    else:
        status += f" | Промах (награда: {reward:4.1f})"
    print(status)
    
    env.render()
    pygame.time.delay(500)
    
    if done:
        print("\nИгра завершена!")
        print(f"Итоговый счет: {env.score}")
        break

env.close()