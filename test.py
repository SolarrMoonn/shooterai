from stable_baselines3 import SAC
from shooter_env import ShooterEnv
import pygame
import numpy as np
import os
import random
import time
import math

random.seed(int(time.time()))
np.random.seed(int(time.time()))

env = ShooterEnv()

model_path = "sac_shooter.zip"
if not os.path.exists(model_path):
    print("run the train.py first")
    exit()

model = SAC.load(model_path, env=env, seed=int(time.time()))

obs, _ = env.reset()

for step in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)
    
    angle_deg = math.degrees(action[0])
    distance = action[1]
    shoot_prob = action[2]
    shoot = shoot_prob > 0.5
    
    status = f"step {step + 1:3d} | angle: {angle_deg:5.1f}° | distance: {distance:5.1f} | shoot: {shoot} (probability: {shoot_prob:.2f})"
    if reward > 0:
        status += f" | hit! +1 (accuracy: {info['accuracy']:.2f}, reward: {reward:4.1f})"
    elif shoot:  
        status += f" | miss (reward: {reward:4.1f})"
    else:
        status += f" | aiming (reward: {reward:4.1f})"
    print(status)
    
    env.render()
    pygame.time.delay(400)
    
    if done:
        print("\n game over")
        print(f"score: {env.score}")
        break

env.close()