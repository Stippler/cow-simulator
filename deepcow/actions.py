from enum import Enum


class AccelerationAction(Enum):
    NOTHING = 0
    FORWARD = 1
    BACKWARD = 2
    LEFT = 3
    RIGHT = 4


class RotationAction(Enum):
    NOTHING = 0
    CLOCK = 1
    COUNTER_CLOCK = 2
