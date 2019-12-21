import math
import random
from collections import deque, namedtuple
from keras.layers import Dense, Conv1D, BatchNormalization, Input, Concatenate
from keras.optimizers import Adam
from keras import Sequential
from keras.models import Model
from deepcow.entity import *
from abc import ABC, abstractmethod


class DQNAgent(ABC):
    def remember(self, state, action, reward, next_state, done):
        pass

    @abstractmethod
    def explore_select_action(self, state):
        pass

    @abstractmethod
    def select_action(self, state):
        pass

    @abstractmethod
    def replay(self, batch_size):
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def get_exploration_rate(self):
        pass


class SimpleDQNAgent(DQNAgent):
    def __init__(self, state_size, action_size, preprocess, memory_length=5000):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_length)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.00  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.preprocess = preprocess
        self.model = self._build_model()

    def _build_model(self):
        """build simple fully connected mlp model of DQN Agent"""
        model = Sequential()
        model.add(Dense(100, input_dim=self.state_size, activation='relu'))
        model.add(Dense(14, input_dim=self.state_size, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def get_exploration_rate(self):
        return self.epsilon

    def remember(self, state, action, reward, next_state, done):
        """add a tuple for learning"""
        self.memory.append((state, action, reward, next_state, done))

    def explore_select_action(self, state):
        """returns an action given a state"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        action_values = self.model.predict(self.preprocess(state))
        return np.argmax(action_values[0])

    def select_action(self, state):
        action_values = self.model.predict(self.preprocess(state))
        return np.argmax(action_values[0])

    def replay(self, batch_size):
        """replays actions for training"""
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                prediction = self.model.predict(self.preprocess(next_state))[0]
                target = reward + self.gamma * np.amax(prediction)
            preprocessed_state = self.preprocess(state)
            target_f = self.model.predict(preprocessed_state)
            target_f[0][action] = target
            self.model.fit(preprocessed_state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, path: str) -> None:
        """load a previously made model"""
        self.model.load_weights(path)

    def save(self, path: str) -> None:
        """save the model of this agent"""
        self.model.save_weights(path)


class ExtendedDQNAgent(DQNAgent):
    def __init__(self,
                 perception_size,
                 metadata_size,
                 action_size,
                 preprocess,
                 batch_size=128,
                 gamma=0.999,
                 epsilon_start=0.9,
                 epsilon_end=0.05,
                 epsilon_decay=200,
                 target_update=200,
                 memory_length=10_000):
        self.perception_size = perception_size
        self.metadata_size = self.metadata_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_length)

        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.steps_done = 0

        self.preprocess = preprocess
        self.convolution_input = Input(metadata_size)
        self.model = self._build_model()

    def _build_model(self):
        """build complex model of DQN Agent"""

        metadata_input = Input(shape=self.metadata_size)
        metadata_layer = Dense(64, activation='relu')(metadata_input)
        metadata_layer = Dense(64, activation='relu')(metadata_layer)
        metadata_layer = Dense(64, activation='relu')(metadata_layer)

        perception_input = Input(shape=self.perception_size)
        perception_layer = Conv1D(self.perception_size * 2, input_shape=(1, self.perception_size), kernel_size=3,
                                  activation='relu', strides=3)(perception_input)
        perception_layer = BatchNormalization(1)(perception_layer)
        perception_layer = Conv1D(self.perception_size * 2, activation='relu')(perception_layer)
        perception_layer = BatchNormalization(1)(perception_layer)

        merge_layer = Concatenate(axis=1)([metadata_layer, perception_layer])
        merge_layer = Dense(merge_layer.output_shape * 2, activation='relu')(merge_layer)
        merge_layer = Dense(16, activation='relu')(merge_layer)
        merge_layer = Dense(self.action_size, activation='linear')(merge_layer)
        merge_layer = Dense(self.action_size, activation='softmax')(merge_layer)

        model = Model(inputs=[metadata_input, perception_input], outputs=[merge_layer])
        model.compile(loss='huber', optimizer=Adam())
        return model

    def remember(self, state, action, reward, next_state, done):
        """add a tuple for learning"""
        self.memory.append((self.preprocess(state), action, reward, next_state, done))

    def explore_select_action(self, state):
        """returns an action given a state"""
        sample = random.random()
        epsilon_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                            math.exp(-1 * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        if sample > epsilon_threshold:
            action = np.argmax(self.model.predict(self.preprocess(state))[0])
        else:
            action = random.randrange(self.action_size)
        return action

    def get_exploration_rate(self):
        epsilon_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                            math.exp(-1 * self.steps_done / self.epsilon_decay)
        return epsilon_threshold

    def select_action(self, state):
        action_values = self.model.predict(self.preprocess(state))
        return np.argmax(action_values[0])

    def replay(self, batch_size):
        """replays actions for training"""
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                prediction = self.model.predict(self.preprocess(next_state))[0]
                target = reward + self.gamma * np.amax(prediction)
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

    def load(self, path: str) -> None:
        """load a previously made model"""
        self.model.load_weights(path)

    def save(self, path: str) -> None:
        """save the model of this agent"""
        self.model.save_weights(path)
