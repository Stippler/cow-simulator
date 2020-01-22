from deepcow.constant import *
from deepcow.entity import *
from deepcow.actions import *
from pygame.math import Vector2
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
                 delta_time=1.0 / 30.0,
                 game_width=800,
                 game_height=600,
                 draw=True):
        # Center the initial pygame windows
        pos_x = 100  # screen_width / 2 - window_width / 2
        pos_y = 100  # screen_height - window_height
        os.environ['SDL_VIDEO_WINDOW_POS'] = '%i,%i' % (pos_x, pos_y)
        os.environ['SDL_VIDEO_CENTERED'] = '0'

        # initialize pygame and set set surface parameters
        pygame.init()
        self.draw = draw
        self.cow_ray_length = cow_ray_length
        self.cow_ray_count = cow_ray_count
        self.wolf_ray_length = wolf_ray_length
        self.wolf_ray_count = wolf_ray_count
        if draw:
            self.screen = pygame.display.set_mode((game_width, game_height))
            pygame.display.set_caption('Cow Simulator')

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

    def reset(self) -> [State]:
        """ randomizes and resets the entities"""
        for entity in self.entities:
            entity.reset()
        states, _, _, _ = self.step([Action.NOTHING for agent in self.agents])
        return states

    def __perform_actions(self, actions: [Action]) -> None:
        assert len(actions) == len(self.agents)
        for i in range(0, len(actions)):
            self.agents[i].perform_action(self.delta_time, actions[i])

    def __update_agents_positions(self) -> (int, int):
        cow_border_collisions = 0
        wolf_border_collisions = 0
        for agent in self.agents:
            agent.update_position(self.delta_time)
        for agent in self.agents:
            agent.calculate_agents_collisions(agents=self.agents)
        for cow in self.cows:
            if cow.calculate_border_collisions():
                cow_border_collisions += 1
        for wolf in self.wolves:
            if wolf.calculate_border_collisions():
                wolf_border_collisions += 1
        return cow_border_collisions, wolf_border_collisions

    def __calculate_rewards(self, agents: [Agent], foods: [Entity]) -> (np.ndarray, bool):
        done = False
        for index, agent in enumerate(agents):
            done = done or agent.calculate_reward(foods, 2)
        return done

    def __get_reset_rewards(self, agents: [Agent]):
        rewards = np.empty(len(agents))
        for index, agent in enumerate(agents):
            rewards[index] = agent.get_reset_reward()
        return rewards

    def __perceive(self) -> [State]:
        return [agent.perceive(self.entities) for agent in self.agents]

    def step(self, actions: [Action]) -> ([State], [float], bool, dict):
        """ performs one step of the environment by updating the position, calculating collisions,
         calculating rewards/energy loss and finally returning a tuple containing:
         (the states of the actors, the rewards of the actors, if the environment was reset, additional information)"""
        self.__perform_actions(actions)
        wolf_border_collisions, cow_border_collisions = self.__update_agents_positions()
        states = self.__perceive()
        if self.draw:
            self.__draw_environment()
        cow_done = self.__calculate_rewards(self.cows, self.grass)
        wolf_done = self.__calculate_rewards(self.wolves, self.cows)
        cow_rewards = self.__get_reset_rewards(self.cows)
        wolf_rewards = self.__get_reset_rewards(self.wolves)
        done = cow_done or wolf_done
        info = {
            'wolf_border_collisions': wolf_border_collisions,
            'cow_border_collisions': cow_border_collisions
        }
        return states, np.concatenate([cow_rewards, wolf_rewards]), done, info

    def __draw_environment(self, draw_perception=True, draw_entity_information=True):
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
                entity.draw_information(self.screen, i)
        pygame.display.update()

    def quit(self):
        """ checks if user pressed x """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return True
