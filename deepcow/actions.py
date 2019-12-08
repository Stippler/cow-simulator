from enum import Enum


class Action(Enum):
    NOTHING = 0
    MOVE_FORWARD = 1
    MOVE_BACKWARD = 2
    MOVE_LEFT = 3
    MOVE_RIGHT = 4
    ROTATE_CLOCKWISE = 5
    ROTATE_COUNTER_CLOCK = 6

