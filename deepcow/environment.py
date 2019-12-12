from deepcow.constant import *
from deepcow.entity import Entity, Agent
from deepcow.actions import *
from pygame.math import Vector2
import random
import os
import pygame

GRAY = (200, 200, 200)


class Environment(object):
    def __init__(self,
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
        self.cows = [Agent(Vector2(), Vector2(), mass=2.0, color=(150, 75, 0))]
        self.wolves = [Agent(Vector2(), Vector2(), mass=1.0, color=(25, 25, 112))]
        self.agents = self.cows + self.wolves

        # initialize entities
        self.grass = [Entity(Vector2(), color=(0, 255, 0))]
        self.entities = self.agents + self.grass

        # initialize user actions
        self.user_action = Action.NOTHING

        # gameloop
        self.running = True  # True as long as the game is running
        self.delta = 1.0 / 60.0  # "deltatime", set it to passed time between each frame to have per second movement
        self.cow_perceptions = []
        self.wolf_perceptions = []
        self.perceive()
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

    def perceive(self):
        self.cow_perceptions.clear()
        for cow in self.cows:
            self.cow_perceptions.append(cow.perceive(entities=self.entities))
        self.wolf_perceptions.clear()
        for wolf in self.wolves:
            self.wolf_perceptions.append(wolf.perceive(entities=self.entities))
        cow_ray_colors = []
        for cow_perception in self.cow_perceptions:
            for _, ray_color in cow_perception:
                cow_ray_colors.append(ray_color)
        wolf_ray_colors = []
        for wolf_perception in self.wolf_perceptions:
            for _, ray_color in wolf_perception:
                wolf_ray_colors.append(ray_color)
        return cow_ray_colors, wolf_ray_colors

    def step(self, cow_actions, wolf_actions):
        for i in range(0, len(cow_actions)):
            self.cows[i].perform_action(self.delta, cow_actions[i])
        for i in range(0, len(wolf_actions)):
            self.wolves[i].perform_action(self.delta, wolf_actions[i])
        for agent in self.agents:
            agent.update_position(self.delta)
        for agent in self.agents:
            agent.calculate_agents_collisions(agents=self.agents)
        for agent in self.agents:
            agent.calculate_border_collisions()
        cow_rewards = []
        done = False
        for cow in self.cows:
            reward, eaten = cow.eat(self.grass, self.delta)
            cow_rewards.append(reward + eaten)
            if eaten != 0:
                done = True
        wolf_rewards = []
        for wolf in self.wolves:
            reward, eaten = wolf.eat(self.cows, self.delta)
            wolf_rewards.append(reward + eaten)
            if eaten != 0:
                done = True
        for entity in self.entities:
            entity.reward = 0
        return cow_rewards, wolf_rewards, done

    def get_cow_perceptions(self):
        return self.cow_perceptions

    def get_wolf_perceptions(self):
        return self.wolf_perceptions

    def draw_environment(self, draw_perception=True, draw_entity_information=False):
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
        for i, entity in enumerate(self.entities, start=0):
            entity.draw_energy_reward(self.screen, i)
        pygame.display.update()
