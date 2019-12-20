import unittest
from deepcow import *
from deepcow.entity import Agent, Entity, State
from pygame.math import Vector2
import numpy as np


class TestAgent(unittest.TestCase):
    def test_perception(self):
        grass = Entity(color=(0, 255, 0))
        grass.position = Vector2(20, 30)
        cow = Agent(ray_count=10, ray_length=200, direction=Vector2(-1, 0))
        cow.position = Vector2(100, 30)
        actual_cow_state = cow.perceive([grass])

        position_correct = np.allclose(actual_cow_state.position, np.array([100, 30]))
        self.assertTrue(position_correct)
        direction_correct = np.allclose(actual_cow_state.direction, np.array([-1, 0]))
        self.assertTrue(direction_correct)
        should_perception = np.zeros((10, 3))
        should_perception[1:-1, 1] = 1
        perception_correct = np.allclose(actual_cow_state.perception, should_perception)
        print(actual_cow_state.perception)
        self.assertTrue(perception_correct)


class TestEnvironment:
    pass


if __name__ == '__main__':
    unittest.main()
