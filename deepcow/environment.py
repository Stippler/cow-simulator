from deepcow.constant import *
from deepcow.entity import Entity, Agent
from deepcow.actions import *
from pygame.math import Vector2
import random
import os
import pygame
import numpy as np

GRAY = (200, 200, 200)


class Environment(object):
    def __init__(self,
                 cow_count=1,
                 cow_ray_count=20,
                 cow_field_of_view=100,
                 cow_ray_length=300,
                 cow_mass=2.0,
                 wolf_ray_count=20,
                 wolf_field_of_view=100,
                 wolf_ray_length=300,
                 wolf_count=1,
                 wolf_mass=2.0,
                 grass_count=1,
                 delta_time=1.0 / 60.0,
                 game_width=800,
                 game_height=600):
        # Center the initial pygame windows
        pos_x = 100  # screen_width / 2 - window_width / 2
        pos_y = 100  # screen_height - window_height
        os.environ['SDL_VIDEO_WINDOW_POS'] = '%i,%i' % (pos_x, pos_y)
        os.environ['SDL_VIDEO_CENTERED'] = '0'

        # initialize pygame and set set surface parameters
        pygame.init()
        self.screen = pygame.display.set_mode((game_width, game_height))
        pygame.display.set_caption('Cow Simulator')
        self.clock = pygame.time.Clock()

        # initialize agents
        self.cows = [Agent(mass=cow_mass,
                           ray_count=cow_ray_count,
                           field_of_view=cow_field_of_view,
                           ray_length=cow_ray_length,
                           color=(150, 75, 0)) for x in range(cow_count)]
        self.wolves = [Agent(mass=wolf_mass,
                             ray_count=wolf_ray_count,
                             field_of_view=wolf_field_of_view,
                             ray_length=wolf_ray_length,
                             color=(25, 25, 112)) for x in range(wolf_count)]
        self.agents = self.cows + self.wolves

        # initialize entities
        self.grass = [Entity(color=(0, 255, 0)) for x in range(grass_count)]
        self.entities = self.agents + self.grass

        # gameloop
        self.delta_time = delta_time  # "deltatime", set it to passed time between each frame to have per second movement
        self.reset()

    def reset(self):
        for entity in self.entities:
            start_x = entity.radius
            stop_x = GAME_WIDTH - entity.radius
            start_y = entity.radius
            stop_y = GAME_HEIGHT - entity.radius
            x = random.uniform(start_x, stop_x)
            y = random.uniform(start_y, stop_y)
            entity.position = Vector2(x, y)
            entity.reward = 0
            entity.energy = 1.0
        for agent in self.agents:
            agent.direction = Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize()
        return self.perceive()

    def perceive(self):
        cow_perceptions = []
        for cow in self.cows:
            cow_perceptions.append(cow.perceive(self.entities))
        wolf_perceptions = []
        for wolf in self.wolves:
            wolf_perceptions.append(wolf.perceive(self.entities))
        return cow_perceptions, wolf_perceptions

    def step(self, cow_actions, wolf_actions):
        for i in range(0, len(cow_actions)):
            self.cows[i].perform_action(self.delta_time, cow_actions[i])
        for i in range(0, len(wolf_actions)):
            self.wolves[i].perform_action(self.delta_time, wolf_actions[i])
        for agent in self.agents:
            agent.update_position(self.delta_time)
        for agent in self.agents:
            agent.calculate_agents_collisions(agents=self.agents)
        for agent in self.agents:
            agent.calculate_border_collisions()
        cow_rewards = []
        done = False
        for cow in self.cows:
            reward, eaten = cow.eat(self.grass, self.delta_time)
            cow_rewards.append(reward + eaten)
            if eaten != 0:
                done = True
        wolf_rewards = []
        for wolf in self.wolves:
            reward, eaten = wolf.eat(self.cows, self.delta_time)
            wolf_rewards.append(reward + eaten)
            if eaten != 0:
                done = True
        for entity in self.entities:
            entity.reward = 0
        cow_perceptions, wolf_perceptions = self.perceive()
        return cow_perceptions, wolf_perceptions, cow_rewards, wolf_rewards, done

    def draw_environment(self, draw_perception=True, draw_entity_information=True):
        self.screen.fill(GRAY)
        if draw_perception:
            for grass in self.grass:
                grass.draw(screen=self.screen)
            for agent in self.agents:
                agent.draw_perception(screen=self.screen)
                agent.draw(screen=self.screen)
        else:
            for entity in self.entities:
                entity.draw(screen=self.screen)
        if draw_entity_information:
            for i, entity in enumerate(self.entities, start=0):
                entity.draw_energy_reward(self.screen, i)
        pygame.display.update()
