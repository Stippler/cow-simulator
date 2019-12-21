from deepcow.environment import Environment
from deepcow.entity import *
from deepcow.actions import *
from deepcow.user_input import *

import pygame
from pygame.time import Clock


ray_count = 20

environment = Environment(cow_ray_count=ray_count,
                          grass_count=1,
                          wolf_ray_count=ray_count,
                          draw=True)

running, user_action = get_user_input()

clock = Clock()
while running:
    environment.step([Action(user_action), Action(user_action)])
    running, user_action = get_user_input()
    clock.tick(60.0)
