import pygame
from deepcow.actions import Action


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

    return running, user_action
