from pygame.time import Clock

from deepcow.environment import Environment
from deepcow.actions import Action
import pygame


def get_user_input():
    user_action = Action.NOTHING
    running = True
    for event in pygame.event.get():
        # set action to forward if w is pressed and backward if s is pressed etc.
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                user_action = Action.MOVE_FORWARD
            elif event.key == pygame.K_q:
                user_action = Action.ROTATE_COUNTER_CLOCK
            elif event.key == pygame.K_e:
                user_action = Action.ROTATE_CLOCKWISE

    return running, user_action


def user_play():
    ray_count = 20

    environment = Environment(cow_ray_count=ray_count,
                              grass_count=1,
                              wolf_ray_count=ray_count,
                              draw=True)

    running, user_action = get_user_input()

    clock = Clock()
    i = 0
    while running:
        states, rewards, done, info = environment.step([Action(user_action), Action(user_action)])
        cow_state = states[0]
        wolf_state = states[1]
        if cow_state.see_food[0]:
            print('frame counter:', i, 'cow sees grass')
        if wolf_state.see_food[0]:
            print('frame counter:', i, 'wolf sees cow')
        running, user_action = get_user_input()
        clock.tick(60.0)
        if done:
            environment.reset()
        i += 1
