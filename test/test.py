import unittest
from deepcow import *
from deepcow.agent import Agent, Entity, State
from deepcow.environment import *
from pygame.math import Vector2
import numpy as np


class TestEnvironment(unittest.TestCase):
    def test_agent_perception(self):
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

    def test_environment_perception(self):
        environment = Environment(draw=False)
        cow = environment.cows[0]
        wolf = environment.wolves[0]
        grass = environment.grass[0]
        cow.position = Vector2(3, 3)
        cow.direction = Vector2(1, 0)
        cow.radius = 1
        cow.ray_count = 5
        cow.ray_length = 5
        cow.field_of_view = 90
        wolf.position = Vector2(15, 3)
        wolf.direction = Vector2(-1, 0)
        wolf.radius = 1
        wolf.ray_count = 5
        wolf.ray_length = 6
        wolf.field_of_view = 90
        grass.position = Vector2(7, 5)
        grass.radius = 1

        states, rewards, done = environment.step([Action.NOTHING, Action.NOTHING])
        self.assertFalse(done)
        should_cow_perception = np.array([[0, 0, 0], [1, 1, 1], [1, 1, 1], [0, 1, 0], [0, 1, 0]])
        cow_perception_correct = np.allclose(states[0].perception, should_cow_perception)
        self.assertTrue(cow_perception_correct)
        should_wolf_perception = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 0, 0]])
        wolf_perception_correct = np.allclose(states[1].perception, should_wolf_perception)
        self.assertTrue(wolf_perception_correct)


if __name__ == '__main__':
    unittest.main()
