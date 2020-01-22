from deepcow.agent_brain import DQNAgent, Action, State, ExtendedDQNAgent
from deepcow.environment import Environment
import numpy as np
import pandas as pd
import os.path

import seaborn as sns
import matplotlib.pyplot as plt

import random

import pygame

WOLF_PATH = 'models/wolf/extended-dqn-best-wolf.HDF5'
COW_PATH = 'models/cow/extended-dqn-best-cow.HDF5'
BOTH_WOLF_PATH = 'models/both/wolf/both-extended-dqn-best-wolf.HDF5'
BOTH_COW_PATH = 'models/both/cow/both-extended-dqn-best-cow.HDF5'
RAY_COUNT = 20


def transform_state_1d(state: State) -> np.ndarray:
    """transforms the state of an agent into a 1d numpy array for the simple deep q network"""
    return np.array([np.concatenate([state.direction, state.velocity, state.perception.ravel()])])


def transform_state_extended(state: State) -> [np.ndarray]:
    """transforms a state for the extended deep q network agent"""
    transformed_state = [[np.concatenate([state.direction, state.velocity])],
                         [np.transpose([state.perception.ravel()])]]
    return transformed_state


def evaluate_model(model, environment, game_count, game_length, index=0) -> (bool, float):
    total_reward = 0
    for games in range(game_count):
        states = environment.reset()
        state = states[index]
        for frame in range(game_length):
            action = model.select_action(state)
            states, rewards, done, info = environment.step([index for i in range(index)] + [Action(action)])
            state = states[index]
            total_reward += rewards[index]
            if frame == game_length - 1:
                done = True
            if environment.quit():
                return (True, 0)
            if done:
                break
    return False, total_reward


def evaluate_models(cow_model, wolf_model, environment, game_count, game_length) -> (bool, float, float):
    total_cow_reward = 0
    total_wolf_reward = 0
    for games in range(game_count):
        states = environment.reset()
        cow_state = states[0]
        wolf_state = states[1]
        for frame in range(game_length):
            cow_action = cow_model.select_action(cow_state)
            wolf_action = wolf_model.select_action(wolf_state)
            states, rewards, done, info = environment.step([Action(cow_action), Action(wolf_action)])
            cow_state = states[0]
            wolf_state = states[0]
            total_cow_reward += rewards[0]
            total_wolf_reward += rewards[1]
            if frame == game_length - 1:
                done = True
            if environment.quit():
                return True, 0, 0
            if done:
                break
    return False, total_cow_reward, total_wolf_reward


def train_cow():
    """ training loop for a special cow """

    ray_count = 20
    action_size = len(Action)

    cow_model = ExtendedDQNAgent(perception_size=ray_count * 3, metadata_size=3, action_size=action_size,
                                 preprocess=transform_state_extended,
                                 memory_length=100_000)

    environment = Environment(cow_ray_count=ray_count,
                              grass_count=1,
                              wolf_ray_count=0,
                              wolf_count=0,
                              draw=True)

    results = train_agent(environment,
                          cow_model,
                          0,
                          COW_PATH)
    sns.lineplot(results['epoch'], results['reward'], color='brown', )
    plt.xlabel('epoch')
    plt.ylabel('reward')
    plt.savefig('result/cow-reward.png')
    plt.show()


def train_wolf():
    ray_count = 20
    action_size = len(Action)

    wolf_model = ExtendedDQNAgent(perception_size=ray_count * 3, metadata_size=3, action_size=action_size,
                                  preprocess=transform_state_extended,
                                  memory_length=100_000)

    environment = Environment(cow_ray_count=0,
                              cow_count=1,
                              grass_count=1,
                              wolf_ray_count=ray_count,
                              wolf_count=1,
                              draw=True)
    results = train_agent(environment, wolf_model, 1, WOLF_PATH)
    sns.lineplot(results['epoch'], results['reward'], color='blue', )
    plt.xlabel('epoch')
    plt.ylabel('reward')
    plt.savefig('result/wolf-reward.png')
    plt.show()


def train_both():
    ray_count = 20
    action_size = len(Action)

    cow_model = ExtendedDQNAgent(perception_size=ray_count * 3, metadata_size=3, action_size=action_size,
                                 preprocess=transform_state_extended,
                                 memory_length=100_000)

    wolf_model = ExtendedDQNAgent(perception_size=ray_count * 3, metadata_size=3, action_size=action_size,
                                  preprocess=transform_state_extended,
                                  memory_length=100_000)

    environment = Environment(cow_ray_count=ray_count,
                              cow_count=1,
                              grass_count=1,
                              wolf_ray_count=ray_count,
                              wolf_count=1,
                              draw=True)

    result = train_agents(environment, cow_model, wolf_model, BOTH_COW_PATH, BOTH_WOLF_PATH)
    sns.lineplot(result['epoch'], result['cow_reward'], color='brown', )
    sns.lineplot(result['epoch'], result['wolf_reward'], color='blue')
    plt.xlabel('epoch')
    plt.ylabel('reward')
    plt.savefig('result/reward.png')
    plt.show()


def train_agents(environment: Environment,
                 cow_model: DQNAgent,
                 wolf_model: DQNAgent,
                 cow_path,
                 wolf_path,
                 epoch_length=1000,
                 episode_length=10,
                 game_length=1000,
                 batch_size=128):
    if os.path.isfile(cow_path + '.index'):
        print('loading cow model from ', cow_path)
        cow_model.load(cow_path)
    if os.path.isfile(wolf_path + '.index'):
        print('loading wolf model from ', wolf_path)
        wolf_model.load(wolf_path)

    summary = []
    best_cow_reward = -9999
    best_wolf_reward = -9999
    for epoch in range(epoch_length):
        print('starting epoch {}, cow exploration rate {:.2f}%, wolf exploration rate {:.2f}%'.format(
            epoch,
            cow_model.get_exploration_rate(),
            wolf_model.get_exploration_rate()))

        if epoch % 10 == 0:
            stop, total_cow_reward, total_wolf_reward = evaluate_models(cow_model, wolf_model, environment, 5,
                                                                        game_length)
            summary.append([epoch, total_cow_reward, total_wolf_reward])
            print("epoch {}, evaluated a total cow reward of {}, evaluated a total cow reward of {}".format(
                epoch,
                total_cow_reward,
                total_wolf_reward))
            if total_cow_reward > best_cow_reward:
                print("saving new best cow model to", cow_path)
                best_cow_reward = total_cow_reward
                cow_model.save(cow_path)
            if total_wolf_reward > best_wolf_reward:
                print("saving new best wolf model to", wolf_path)
                best_wolf_reward = total_wolf_reward
                wolf_model.save(cow_path)
            if stop:
                return pd.DataFrame(data=summary, columns=['epoch', 'cow_reward', 'wolf_reward'])

        for episode in range(episode_length):
            states = environment.reset()
            cow_state = states[0]
            wolf_state = states[0]
            old_cow_distance = cow_state.food_distances[0]
            old_wolf_distance = wolf_state.food_distances[0]

            for frame in range(game_length):
                cow_action = cow_model.explore_select_action(cow_state)
                wolf_action = wolf_model.explore_select_action(wolf_state)
                states, rewards, done, info = environment.step([Action(cow_action), Action(wolf_action)])
                cow_reward = rewards[0]
                wolf_reward = rewards[1]
                next_cow_state = states[0]
                next_wolf_state = states[1]
                new_cow_distance = next_cow_state.food_distances[0]
                new_wolf_distance = next_wolf_state.food_distances[0]

                if next_cow_state.see_food[0]:
                    delta_distance = old_cow_distance - new_cow_distance
                    cow_reward += delta_distance * 0.1
                if next_wolf_state.see_food[0]:
                    delta_distance = old_wolf_distance - new_wolf_distance
                    wolf_reward += delta_distance * 0.1

                old_cow_distance = new_cow_distance
                old_wolf_distance = new_wolf_distance

                if frame == game_length - 1:
                    done = True

                cow_model.remember(cow_state, cow_action, cow_reward, next_cow_state, done)
                cow_state = next_cow_state
                wolf_model.remember(wolf_state, wolf_action, wolf_reward, next_wolf_state, done)
                wolf_state = next_wolf_state
                if environment.quit():
                    return pd.DataFrame(data=summary, columns=['epoch', 'cow_reward', 'wolf_reward'])
                if done:
                    break
        cow_model.replay(batch_size)
        wolf_model.replay(batch_size)
        print('finish epoch {}'.format(epoch))
    return pd.DataFrame(data=summary, columns=['epoch', 'cow_reward', 'wolf_reward'])


def train_agent(environment: Environment,
                model: DQNAgent,
                index,
                path,
                epoch_length=1000,
                episode_length=10,
                game_length=1000,
                batch_size=128):
    if os.path.isfile(path + '.index'):
        print('loading model from ', path)
        model.load(path)

    summary = []
    best_reward = -9999
    for epoch in range(epoch_length):
        print('starting epoch {}, exploration rate {:.2f}%'.format(
            epoch,
            model.get_exploration_rate()))

        if epoch % 10 == 0:
            stop, total_reward = evaluate_model(model, environment, 5, game_length, index)
            summary.append([epoch, total_reward])
            print("epoch {}, evaluated a total reward of {}".format(epoch, total_reward))
            if total_reward > best_reward:
                print("saving new best model to", path)
                best_reward = total_reward
                model.save(path)
            if stop:
                return pd.DataFrame(data=summary, columns=['epoch', 'reward'])

        for episode in range(episode_length):
            states = environment.reset()
            state = states[index]
            old_distance = state.food_distances[0]

            for frame in range(game_length):
                action = model.explore_select_action(state)
                states, rewards, done, info = environment.step(
                    [Action.NOTHING for i in range(index)] + [Action(action)])

                reward = rewards[index]
                next_state = states[index]
                new_distance = next_state.food_distances[0]

                if next_state.see_food[0]:
                    delta_distance = old_distance - new_distance
                    reward += delta_distance * 0.1

                old_distance = new_distance

                if frame == game_length - 1:
                    done = True

                model.remember(state, action, reward, next_state, done)
                state = next_state

                if environment.quit():
                    return pd.DataFrame(data=summary, columns=['epoch', 'reward'])
                if done:
                    break
        model.replay(batch_size)
        print('finish epoch {}'.format(epoch))
    return pd.DataFrame(data=summary, columns=['epoch', 'reward'])


def play_game():
    environment = Environment(grass_count=1,
                              wolf_count=1,
                              draw=True)
    action_size = len(Action)
    cow_model = ExtendedDQNAgent(perception_size=environment.cow_ray_count * 3,
                                 metadata_size=3,
                                 action_size=action_size,
                                 preprocess=transform_state_extended,
                                 memory_length=100_000)
    wolf_model = ExtendedDQNAgent(perception_size=environment.wolf_ray_count * 3,
                                  metadata_size=3,
                                  action_size=action_size,
                                  preprocess=transform_state_extended,
                                  memory_length=100_000)

    clock = pygame.time.Clock()

    states = environment.reset()
    cow_state = states[0]
    wolf_state = states[1]
    running = True
    while running:
        cow_action = cow_model.select_action(cow_state)
        if cow_action == 0:
            cow_action = random.randrange(cow_model.action_size)
        wolf_action = wolf_model.select_action(wolf_state)
        if wolf_action == 0:
            wolf_action = random.randrange(wolf_model.action_size)
        states, rewards, done, info = environment.step([Action(cow_action), Action(wolf_action)])
        cow_state = states[0]
        wolf_state = states[1]
        if environment.quit():
            running = False
        if done:
            environment.reset()
        clock.tick(60)
