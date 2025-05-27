import pygame
import random
import stable_baselines3

# Инициализация
pygame.init()
screen = pygame.display.set_mode((800, 600))

# Переменные
target_x = random.randint(50, 750)
target_y = random.randint(50, 550)
radius = 20
score = 0
font = pygame.font.Font(None, 36)

BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)

# Игровой цикл
running = True
while running:
    # Обработка событий
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            click_x, click_y = event.pos
            distance = ((click_x - target_x) ** 2 + (click_y - target_y) ** 2) ** 0.5
            if distance < radius:
                score += 1
                target_x = random.randint(50, 750)
                target_y = random.randint(50, 550)
    
    # Рисование
    screen.fill(BLACK)
    pygame.draw.circle(screen, PURPLE, (target_x, target_y), radius) 
    text = font.render(f"Счёт: {score}", True, (255, 255, 255))
    screen.blit(text, (10, 10)) 
    pygame.display.flip()

pygame.quit()