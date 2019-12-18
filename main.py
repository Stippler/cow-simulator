import math
import random
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import Sequential
from typing.io import IO

from deepcow.environment import Environment
from deepcow.constant import *
from deepcow.actions import *
from deepcow.entity import *
import pygame
import pandas as pd
from itertools import count
import matplotlib.pyplot as plt
import seaborn as sns


class DQNAgent:
    def __init__(self, state_size, action_size, memory_length=5000):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.00  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        """build simple fully connected mlp model of DQN Agent"""
        model = Sequential()
        model.add(Dense(100, input_dim=self.state_size, activation='relu'))
        model.add(Dense(14, input_dim=self.state_size, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        """add a tuple for learning"""
        self.memory.append((state, action, reward, next_state, done))

    def explore_select_action(self, state):
        """returns an action given a state"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        action_values = self.model.predict(state)
        return np.argmax(action_values[0])

    def select_action(self, state):
        action_values = self.model.predict(state)
        return np.argmax(action_values[0])

    def replay(self, batch_size):
        """replays actions for training"""
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                prediction = self.model.predict(next_state)[0]
                target = reward + self.gamma * np.amax(prediction)
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        """load a previously made model"""
        self.model.load_weights(name)

    def save(self, name):
        """save the model of this agent"""
        self.model.save_weights(name)


def transform_state_1d(state: State) -> np.ndarray:
    """transforms a state into a 1d numpy array"""
    return np.array([np.concatenate([state.direction, state.velocity, state.perception.ravel()])])


def train_dqn_agents(cow_model: DQNAgent,
                     wolf_model: DQNAgent,
                     environment: Environment,
                     cow_preprocessing=transform_state_1d,
                     wolf_preprocessing=transform_state_1d,
                     train_cow=True,
                     train_wolf=True,
                     epoch_length=1000,
                     episode_length=4,
                     game_length=1000,
                     batch_size=32,
                     plot=True) -> (DQNAgent, DQNAgent):
    # clock = pygame.time.Clock()

    all_rewards = []
    for epoch in range(epoch_length):

        print('starting epoch {}, cow exploration rate {:.2f}% and wolf exploration rate {:.2f}%'.format(
            epoch,
            cow_model.epsilon,
            wolf_model.epsilon))

        states = environment.reset()
        cow_state = cow_preprocessing(states[0])
        wolf_state = wolf_preprocessing(states[0])

        cow_reword_per_epoch = 0
        wolf_reword_per_epoch = 0

        for episode in range(episode_length):
            states = environment.reset()
            cow_state = cow_preprocessing(states[0])
            wolf_state = wolf_preprocessing(states[0])

            for frame in range(game_length):
                cow_action = cow_model.explore_select_action(cow_state)
                wolf_action = wolf_model.explore_select_action(wolf_state)

                states, rewards, done = environment.step([Action(cow_action), Action(wolf_action)])

                cow_reward = rewards[0]
                cow_reword_per_epoch += cow_reward
                cow_next_state = cow_preprocessing(states[0])

                wolf_reward = rewards[1]
                wolf_reword_per_epoch += wolf_reward
                wolf_next_state = wolf_preprocessing(states[1])

                cow_model.remember(cow_state, cow_action, cow_reword_per_epoch, cow_next_state, done)
                wolf_model.remember(wolf_state, wolf_action, wolf_reword_per_epoch, wolf_next_state, done)

                cow_state = cow_next_state
                wolf_state = wolf_next_state

                if done:
                    break
                if environment.quit():
                    return pd.DataFrame(data=all_rewards, columns=['epoch', 'cow_reward', 'wolf_reward'], )
            cow_reword_per_epoch /= episode_length
            wolf_reword_per_epoch /= episode_length

        all_rewards.append([epoch, cow_reword_per_epoch, wolf_reword_per_epoch])
        for i in range(4):
            cow_model.replay(batch_size)
            wolf_model.replay(batch_size)
        print('finish epoch {}, cow rewards {}, wolf rewards {}'.format(epoch, cow_reword_per_epoch,
                                                                        wolf_reword_per_epoch))
    return pd.DataFrame(data=all_rewards, columns=['epoch', 'cow_reward', 'wolf_rewards'], )


epoch_length = 1000
ray_count = 20
ray_length = 450
action_size = 7

cow_model = DQNAgent(4 + ray_count * 3, action_size)
wolf_model = DQNAgent(4 + ray_count * 3, action_size)

environment = Environment(cow_ray_count=ray_count,
                          cow_ray_length=ray_length,
                          grass_count=1,
                          wolf_ray_count=ray_count,
                          wolf_ray_length=ray_length,
                          draw=True)

results = train_dqn_agents(cow_model, wolf_model, environment, plot=True)
print(results)
sns.lineplot(results['epoch'], results['cow_reward'], color='brown', )
sns.lineplot(results['epoch'], results['wolf_reward'], color='blue')
plt.xlabel('epoch')
plt.ylabel('reward')
plt.savefig('result/dqn-result.png')
plt.show()
print(results)
