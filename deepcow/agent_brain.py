from collections import deque, namedtuple
from tensorflow.keras.layers import Dense, Conv1D, BatchNormalization, Input, Concatenate, Flatten, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from deepcow.entity import *
from abc import ABC, abstractmethod
import tensorflow.keras as keras


class DQNAgent(ABC):
    def remember(self, state, action, reward, next_state, done):
        """saves a transition from a state to its successor state for later training"""
        pass

    @abstractmethod
    def explore_select_action(self, state):
        """returns an action that is either random or calculated by the neural network,
         depending on the exploration rate"""
        pass

    @abstractmethod
    def select_action(self, state):
        """calculates an action with the neural network"""
        pass

    @abstractmethod
    def replay(self, batch_size) -> None:
        """replays the transitions saved with remember and trains the neural network to approximate Q(s,a)"""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """loads the weights of a previously saved network"""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """saves the weights of the network"""
        pass

    @abstractmethod
    def get_exploration_rate(self):
        """returns the probability that the next action is chosen randomly"""
        pass


class SimpleDQNAgent(DQNAgent):
    """ simple deep q network agent """
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
        model.add(Dense(100, input_dim=self.state_size, activation='linear'))
        model.add(Dense(14, input_dim=self.state_size, activation='linear'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def get_exploration_rate(self):
        return self.epsilon

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def explore_select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        action_values = self.model.predict(self.preprocess(state))
        return np.argmax(action_values[0])

    def select_action(self, state):
        action_values = self.model.predict(self.preprocess(state))
        return np.argmax(action_values[0])

    def replay(self, batch_size):
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
        self.model.load_weights(path)

    def save(self, path: str) -> None:
        self.model.save_weights(path)


class ExtendedDQNAgent(DQNAgent):
    """ complex deep q network agent """

    def __init__(self,
                 perception_size,
                 metadata_size,
                 action_size,
                 preprocess,
                 batch_size=128,
                 memory_length=10_000,
                 tensorboard_callback=None):
        self.perception_size = perception_size
        self.metadata_size = metadata_size
        self.action_size = action_size
        self.preprocess = preprocess

        self.memory = deque(maxlen=memory_length)
        self.batch_size = batch_size
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.00  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.steps_done = 0
        self.tensorboard_callback = tensorboard_callback

        self.model = self._build_model()

    def _build_model(self):
        """build complex model of DQN Agent"""

        metadata_input = Input(shape=(self.metadata_size,))
        metadata_layer = Dense(self.metadata_size, activation='relu')(metadata_input)
        metadata_layer = Dense(self.metadata_size, activation='relu')(metadata_layer)
        metadata_layer = Dense(self.metadata_size, activation='relu')(metadata_layer)

        perception_input = Input(shape=(self.perception_size, 1))
        perception_layer = Conv1D(self.perception_size * 2, kernel_size=3,
                                  activation='linear', strides=3)(perception_input)
        perception_layer = BatchNormalization()(perception_layer)
        perception_layer = Conv1D(1, kernel_size=3, activation='relu')(perception_layer)
        perception_layer = BatchNormalization()(perception_layer)
        perception_layer = Conv1D(1, kernel_size=3, activation='relu')(perception_layer)
        perception_layer = BatchNormalization()(perception_layer)
        perception_layer = Flatten()(perception_layer)
        perception_layer = Dense(self.action_size, activation='linear')(perception_layer)

        merge_layer = Concatenate(axis=1)([metadata_layer, perception_layer])
        merge_layer = Dense(16, activation='relu')(merge_layer)
        merge_layer = Dense(self.action_size, activation='linear')(merge_layer)

        model = Model(inputs=[metadata_input, perception_input], outputs=[merge_layer])
        model.compile('sgd', loss=keras.losses.Huber())
        return model

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
        processed_state = self.preprocess(state)
        action_values = self.model.predict(processed_state)
        return np.argmax(action_values[0])

    def get_exploration_rate(self):
        return self.epsilon

    def replay(self, batch_size):
        """replays actions for training"""
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                prediction = self.model.predict(self.preprocess(next_state))[0]
                target = reward + self.gamma * np.amax(prediction)
            target_f = self.model.predict(self.preprocess(state))
            target_f[0][action] = target
            self.model.fit(self.preprocess(state), target_f, epochs=3, verbose=0, callbacks=self.tensorboard_callback)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, path: str) -> None:
        """load a previously made model"""
        self.model.load_weights(path)

    def save(self, path: str) -> None:
        """save the model of this agent"""
        self.model.save_weights(path)
