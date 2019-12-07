from deepcow.constant import *
from deepcow.entity import Entity, Agent
from deepcow.actions import *
from pygame.math import Vector2
import os
import pygame

# Center the initial pygame windows
# TODO center windows to screen!
pos_x = 100  # screen_width / 2 - window_width / 2
pos_y = 100  # screen_height - window_height
os.environ['SDL_VIDEO_WINDOW_POS'] = '%i,%i' % (pos_x, pos_y)
os.environ['SDL_VIDEO_CENTERED'] = '0'

# initialize pygame and set set surface parameters
pygame.init()
white = (255, 255, 255)
gameDisplay = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
pygame.display.set_caption('Cow Simulator')
clock = pygame.time.Clock()

# initialize agents
cow = Agent(Vector2(200, 100), Vector2(), mass=2.0, color=(150, 75, 0))
wolf = Agent(Vector2(300, 100), Vector2(), mass=1.0, color=(25, 25, 112))
grass = Entity(Vector2(300, 300), color=(0, 255, 0))
agents = [cow, wolf]
entities = [grass]

# initialize user actions
user_action_acceleration = AccelerationAction.NOTHING
user_action_rotation = RotationAction.NOTHING

# gameloop
running = True  # True as long as the game is running
delta = 1.0 / 60.0  # "deltatime", set it to passed time between each frame to have per second movement

while running:

    # check for events to handle user actions
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # set AccelerationAction to forward if w is pressed and backward if s is pressed
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                user_action_acceleration = AccelerationAction.FORWARD
            elif event.key == pygame.K_s:
                user_action_acceleration = AccelerationAction.BACKWARD
            elif event.key == pygame.K_a:
                user_action_acceleration = AccelerationAction.LEFT
            elif event.key == pygame.K_d:
                user_action_acceleration = AccelerationAction.RIGHT

        # set AccelerationAction to forward nothing if w, s, a or d is released
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_w and user_action_acceleration == AccelerationAction.FORWARD:
                user_action_acceleration = AccelerationAction.NOTHING
            elif event.key == pygame.K_s and user_action_acceleration == AccelerationAction.BACKWARD:
                user_action_acceleration = AccelerationAction.NOTHING
            elif event.key == pygame.K_a and user_action_acceleration == AccelerationAction.LEFT:
                user_action_acceleration = AccelerationAction.NOTHING
            elif event.key == pygame.K_d and user_action_acceleration == AccelerationAction.RIGHT:
                user_action_acceleration = AccelerationAction.NOTHING

    gameDisplay.fill(white)

    cow.update_acceleration(delta, user_action_acceleration, user_action_rotation)
    for agent in agents:
        agent.update_position(delta)

    for agent in agents:
        agent.calculate_agents_collisions(agents)

    for agent in agents:
        agent.calculate_border_collisions()

    for entity in entities:
        entity.draw(gameDisplay)

    for agent in agents:
        agent.draw(gameDisplay)

    pygame.display.update()
    clock.tick(60)

pygame.quit()
quit()
