import torch
import torch.nn as nn
from deepcow.environment import Environment
from deepcow.actions import Action
import random
import pygame
from deepcow.user_input import get_user_input

env = Environment()
clock = pygame.time.Clock()
running = True
while running:
    running, user_action = get_user_input()
    cow_rewards, wolf_rewards, done = env.step(cow_actions=[user_action], wolf_actions=[Action(random.randint(0, 6))])
    if done:
        env.reset()
        continue
    env.perceive()
    env.draw_environment(True, True)
    clock.tick(60)

pygame.quit()
