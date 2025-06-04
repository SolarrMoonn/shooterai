import pygame
import random
import numpy as np
import gymnasium as gym
import time
import math

class ShooterEnv(gym.Env):
    def __init__(self):
        super().__init__()
        random.seed(int(time.time()))
        np.random.seed(int(time.time()))
        
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        self.clock = pygame.time.Clock()
        self.score = 0
        self.font = pygame.font.Font(None, 36)
        self.BLACK = (0, 0, 0)
        self.PURPLE = (128, 0, 128)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        
        self.action_space = gym.spaces.Box(
            low=np.array([0, 0, 0]),
            high=np.array([2*np.pi, 400, 1]),
            dtype=np.float32
        )
        
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0]),
            high=np.array([800, 600, 100]),
            dtype=np.float32
        )

        self.target_x = random.randint(50, 750)
        self.target_y = random.randint(50, 550)
        self.radius = 40
        self.score = 0
        self.steps = 0
        self.max_steps = 100
        self.last_angle = 0
        self.last_distance = 0
        self.last_shoot = 0
        self.last_shoot_prob = 0  

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.target_x = random.randint(50, 750)
        self.target_y = random.randint(50, 550)
        self.score = 0
        self.steps = 0
        return np.array([self.target_x, self.target_y, self.max_steps - self.steps], dtype=np.float32), {}

    def step(self, action):
        angle = np.clip(action[0], 0, 2*np.pi)
        distance = np.clip(action[1], 0, 400)
        shoot_prob = np.clip(action[2], 0, 1)
        shoot = shoot_prob > 0.5
        
        self.last_angle = angle
        self.last_distance = distance
        self.last_shoot = shoot
        self.last_shoot_prob = shoot_prob
        
        if not shoot:
            reward = -0.1
            info = {
                "distance": 0,
                "hit": False,
                "score": self.score,
                "accuracy": 0.0,
                "shoot_probability": shoot_prob
            }
            self.steps += 1
            terminated = self.steps >= self.max_steps
            return np.array([self.target_x, self.target_y, self.max_steps - self.steps], dtype=np.float32), reward, terminated, False, info
        
        # Calculate shot position relative to target (target is now origin 0,0)
        relative_x = distance * np.cos(angle)
        relative_y = distance * np.sin(angle)
        
        # Convert back to screen coordinates
        click_x = self.target_x + relative_x
        click_y = self.target_y + relative_y
        
        click_x = np.clip(click_x, 0, 800)
        click_y = np.clip(click_y, 0, 600)
        
        # Distance is now just the length of the relative vector
        distance = (relative_x ** 2 + relative_y ** 2) ** 0.5
        hit = distance < self.radius
        
        if hit:
            accuracy = 1.0 - (distance / self.radius)
            reward = 20.0 * (1.0 + accuracy)
            self.score += 1
            self.target_x = random.randint(50, 750)
            self.target_y = random.randint(50, 550)
        else:
            reward = -min(distance / 50, 10.0)
        
        self.steps += 1
        terminated = self.steps >= self.max_steps
        
        info = {
            "distance": distance,
            "hit": hit,
            "score": self.score,
            "accuracy": 1.0 - (distance / self.radius) if hit else 0.0,
            "shoot_probability": shoot_prob
        }
        
        return np.array([self.target_x, self.target_y, self.max_steps - self.steps], dtype=np.float32), reward, terminated, False, info

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        
        self.screen.fill(self.BLACK)
        pygame.draw.circle(self.screen, self.PURPLE, (self.target_x, self.target_y), self.radius)
        
        end_x = self.target_x + self.last_distance * np.cos(self.last_angle)
        end_y = self.target_y + self.last_distance * np.sin(self.last_angle)
        
        if self.last_shoot:
            pygame.draw.circle(self.screen, self.RED, (int(end_x), int(end_y)), 5)
        
        text = self.font.render(f"Score: {self.score} Steps: {self.steps}", True, (255, 255, 255))
        self.screen.blit(text, (10, 10))
        
        pygame.display.flip()
        self.clock.tick(60)
    
    def close(self):
        pygame.quit()
        

    