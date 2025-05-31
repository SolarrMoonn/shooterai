from stable_baselines3 import PPO
from shooter_env import ShooterEnv
import pygame
import numpy as np
import os

env = ShooterEnv()
print("Загружаем модель...")

model_path = "ppo_shooter.zip"
if not os.path.exists(model_path):
    print("Модель не найдена! Запусти train.py")
    exit()

model = PPO.load(model_path)
print("Модель загружена успешно!")

obs, _ = env.reset()
print("\nНачинаем игру!")
print("Фиолетовый круг - цель")
print("Красный круг - клик ИИ")
print("Счет и шаги отображаются в верхнем левом углу\n")

for step in range(100):
    action, _ = model.predict(obs, deterministic=False)
    obs, reward, done, _, info = env.step(action)
    
    click_x, click_y = env.last_click_x, env.last_click_y
    
    print(f"Шаг {step + 1}: ИИ кликает в ({click_x}, {click_y})")
    print(f"Наблюдение: {obs}, Расстояние: {info['distance']:.2f}")
    if reward > 0:
        print(f"Попадание! +1 к счету (награда: {reward:.2f})")
    else:
        print(f"Промах (награда: {reward:.2f})")
    
    env.render()
    pygame.time.delay(500)
    
    if done:
        print("\nИгра завершена!")
        print(f"Итоговый счет: {env.score}")
        break

env.close()