import pygame
import random
import numpy as np
import gymnasium as gym

class ShooterEnv(gym.Env):
    def __init__(self):
        super().__init__()
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        self.clock = pygame.time.Clock()
        self.score = 0
        self.font = pygame.font.Font(None, 36)
        self.BLACK = (0, 0, 0)
        self.PURPLE = (128, 0, 128)
        self.RED = (255, 0, 0)
        # 8 направлений движения + клик на месте
        self.action_space = gym.spaces.Discrete(9)
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0, -10, 0]),
            high=np.array([800, 600, 800, 600, 10, 100]),
            dtype=np.float32
        )

        self.target_x = random.randint(50, 750)
        self.target_y = random.randint(50, 550)
        self.radius = 40
        self.score = 0
        self.steps = 0
        self.max_steps = 100
        self.last_click_x = 400
        self.last_click_y = 300
        self.last_reward = 0
        self.last_action = None
        self.max_move = 40  # Уменьшаем максимальное смещение для более точных движений
        self.last_distance = float('inf')

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.target_x = random.randint(50, 750)
        self.target_y = random.randint(50, 550)
        self.score = 0
        self.steps = 0
        self.last_click_x = 400
        self.last_click_y = 300
        self.last_reward = 0
        self.last_action = None
        self.last_distance = float('inf')  # Сбрасываем last_distance
        return np.array([self.target_x, self.target_y, self.last_click_x, self.last_click_y, self.last_reward, self.max_steps - self.steps], dtype=np.float32), {}
    
    def step(self, action):
        # Определяем смещение в зависимости от действия
        # 0-7: движения в 8 направлениях, 8: клик на месте
        if action == 0:  # вверх
            dx, dy = 0, -self.max_move
        elif action == 1:  # вверх-вправо
            dx, dy = self.max_move, -self.max_move
        elif action == 2:  # вправо
            dx, dy = self.max_move, 0
        elif action == 3:  # вниз-вправо
            dx, dy = self.max_move, self.max_move
        elif action == 4:  # вниз
            dx, dy = 0, self.max_move
        elif action == 5:  # вниз-влево
            dx, dy = -self.max_move, self.max_move
        elif action == 6:  # влево
            dx, dy = -self.max_move, 0
        elif action == 7:  # вверх-влево
            dx, dy = -self.max_move, -self.max_move
        else:  # клик на месте
            dx, dy = 0, 0

        # Вычисляем новые координаты клика
        new_x = self.last_click_x + dx
        new_y = self.last_click_y + dy
        
        # Ограничиваем координаты пределами экрана
        click_x = max(0, min(800, new_x))
        click_y = max(0, min(600, new_y))
        
        self.last_click_x = click_x
        self.last_click_y = click_y
        
        distance = ((click_x - self.target_x) ** 2 + (click_y - self.target_y) ** 2) ** 0.5
        hit = distance < self.radius
        
        # Улучшенная система наград
        if hit:
            reward = 10.0  # Увеличиваем награду за попадание
            self.score += 1
            self.target_x = random.randint(50, 750)
            self.target_y = random.randint(50, 550)
        elif distance < 90:
            reward = 1.0  # Уменьшаем награду за близкий промах
        else:
            # Штраф зависит от расстояния
            reward = -min(distance / 200, 3.0)
        
        # Дополнительная награда за приближение к цели
        if distance < self.last_distance:
            reward += 0.5
        
        self.last_distance = distance
        self.last_action = action
        self.last_reward = reward
        self.steps += 1
        terminated = self.steps >= self.max_steps
        truncated = False
        info = {
            "distance": distance,
            "hit": hit,
            "score": self.score
        }
        
        return np.array([self.target_x, self.target_y, self.last_click_x, self.last_click_y, self.last_reward, self.max_steps - self.steps], dtype=np.float32), reward, terminated, truncated, info
    
    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        
        self.screen.fill(self.BLACK)
        pygame.draw.circle(self.screen, self.PURPLE, (self.target_x, self.target_y), self.radius)
        
        if self.last_click_x is not None and self.last_click_y is not None:
            pygame.draw.circle(self.screen, self.RED, (self.last_click_x, self.last_click_y), 5)
        
        text = self.font.render(f"Score: {self.score} Steps: {self.steps}", True, (255, 255, 255))
        self.screen.blit(text, (10, 10))
        pygame.display.flip()
        self.clock.tick(60)
    
    def close(self):
        pygame.quit()
        

    