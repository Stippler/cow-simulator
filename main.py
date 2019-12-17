import math
import random
import matplotlib
import matplotlib.pyplot as plt
from IPython import display
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import Sequential
from deepcow.environment import Environment
from deepcow.constant import *
from deepcow.actions import *
from deepcow.entity import *


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        """build simple fully connected mlp model of DQN Agent"""
        model = Sequential()
        model.add(Dense(100, input_dim=self.state_size, activation='relu'))
        model.add(Dense(7, input_dim=self.state_size, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        """add a tuple for learning"""
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state):
        """returns an action given a state"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
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
    return np.array([np.concatenate([state.direction, state.perception.ravel()])])


def training():
    EPISODES = 1000
    ray_size = 20
    ray_length = 450
    action_size = 7
    cow_brain = DQNAgent(2+ray_size * 3, action_size)
    cow_brain.load("models/current-cow.HDF5")
    environment = Environment(cow_ray_count=ray_size, cow_ray_length=ray_length, grass_count=1,
                              wolf_ray_count=ray_size, draw=False)
    batch_size = 128
    for epoch in range(1_000_000):
        for episode in range(EPISODES):
            states = environment.reset()
            cow_state = transform_state_1d(states[0])

            for frame_number in range(2000):
                action = cow_brain.select_action(cow_state)

                states, rewards, done = environment.step([Action(action), Action.NOTHING])

                cow_reward = rewards[0]
                cow_next_state = transform_state_1d(states[0])

                cow_brain.remember(cow_state, action, cow_reward, cow_next_state, done)

                cow_state = cow_next_state

                if done:
                    break
                if environment.quit():
                    return
            cow_brain.replay(batch_size)
            print('completed epoch', epoch, 'episode', episode)
        cow_brain.save("models/current-cow.HDF5")


training()
