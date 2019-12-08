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
GRAY = (200, 200, 200)
gameDisplay = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
pygame.display.set_caption('Cow Simulator')
clock = pygame.time.Clock()

# initialize agents
cow = Agent(Vector2(200, 100), Vector2(), mass=2.0, color=(150, 75, 0))
wolf = Agent(Vector2(300, 100), Vector2(), mass=1.0, color=(25, 25, 112))
grass = Entity(Vector2(300, 300), color=(0, 255, 0))
agents = [cow, wolf]
static_entities = [grass]
entities = agents + static_entities

# initialize user actions
user_action = Action.NOTHING

# gameloop
running = True  # True as long as the game is running
delta = 1.0 / 60.0  # "deltatime", set it to passed time between each frame to have per second movement

while running:

    # check for events to handle user actions
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # set action to forward if w is pressed and backward if s is pressed
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                user_action = Action.MOVE_FORWARD
            elif event.key == pygame.K_s:
                user_action = Action.MOVE_BACKWARD
            elif event.key == pygame.K_a:
                user_action = Action.MOVE_LEFT
            elif event.key == pygame.K_d:
                user_action = Action.MOVE_RIGHT
            elif event.key == pygame.K_q:
                user_action = Action.ROTATE_COUNTER_CLOCK
            elif event.key == pygame.K_e:
                user_action = Action.ROTATE_CLOCKWISE

        # set action to nothing if w, s, a or d is released
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_w and user_action == Action.MOVE_FORWARD:
                user_action = Action.NOTHING
            elif event.key == pygame.K_s and user_action == Action.MOVE_BACKWARD:
                user_action = Action.NOTHING
            elif event.key == pygame.K_a and user_action == Action.MOVE_LEFT:
                user_action = Action.NOTHING
            elif event.key == pygame.K_d and user_action == Action.MOVE_RIGHT:
                user_action = Action.NOTHING
            elif event.key == pygame.K_q and user_action == Action.ROTATE_COUNTER_CLOCK:
                user_action = Action.NOTHING
            elif event.key == pygame.K_e and user_action == Action.ROTATE_CLOCKWISE:
                user_action = Action.NOTHING

    gameDisplay.fill(GRAY)

    cow.update_acceleration(delta, user_action)

    for agent in agents:
        agent.percept(entities, gameDisplay)

    for agent in agents:
        agent.update_position(delta)

    for agent in agents:
        agent.calculate_agents_collisions(agents)

    for agent in agents:
        agent.calculate_border_collisions()

    for entity in static_entities:
        entity.draw(gameDisplay)

    for agent in agents:
        agent.draw(gameDisplay)

    pygame.display.update()
    clock.tick(60)

pygame.quit()
quit()
