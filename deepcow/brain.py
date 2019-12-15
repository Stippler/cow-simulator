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
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
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
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
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


EPISODES = 1000
environment = Environment()
ray_size = 20
action_size = 7
cow_brain = DQNAgent(ray_size, action_size)
done = False
batch_size = 32

for e in range(EPISODES):
    cow_states, wolf_states = environment.reset()
    state = np.reshape(cow_states[0], [1, ray_size])
    for time in range(500):
        # env.render()
        action = cow_brain.select_action(state)
        cow_next_states, wolf_next_states, cow_rewards, wolf_rewards, cows_won, wolves_won = environment.step([action], [Action.NOTHING])

        cow_reward = cow_rewards[0]
        if cows_won:
            cow_reward = 10
        elif wolves_won:
            cow_reward = -10

        cow_next_state = np.reshape(cow_next_states[0], [1, ray_size])
        cow_brain.remember(state, action, cow_reward, cow_next_state, cows_won or wolves_won)
        state = cow_next_state
        if cows_won or wolves_won:
            print("episode: {}/{}, score: {}, e: {:.2}"
                  .format(e, EPISODES, time, cow_brain.epsilon))
            break
        if len(cow_brain.memory) > batch_size:
            cow_brain.replay(batch_size)

num_episodes = 50
for i_episode in range(num_episodes):
    # Initialize the environment and state
    environment.reset()
    cow_states, wolf_states = environment.perceive()
    for t in count():
        # Select and perform an action
        action = select_action(torch.tensor(cow_states[0]).float())
        cow_rewards, wolf_rewards, done = environment.step([action.item()], wolf_actions=[Action(random.randint(0, 6))])
        cow_rewards = torch.tensor([cow_rewards[0]], device=device)

        # Observe new state
        next_cow_states, next_wolf_states = environment.perceive()

        # Store the transition in memory
        memory.push(cow_states[0], action, next_cow_states[0], cow_rewards)

        # Move to the next state
        cow_states, wolf_states = next_cow_states, next_wolf_states

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
plt.ioff()
plt.show()
