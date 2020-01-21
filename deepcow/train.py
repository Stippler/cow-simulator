from deepcow.agent_brain import DQNAgent, Action, State, ExtendedDQNAgent
from deepcow.environment import Environment
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def transform_state_1d(state: State) -> np.ndarray:
    """transforms the state of an agent into a 1d numpy array for the simple deep q network"""
    return np.array([np.concatenate([state.direction, state.velocity, state.perception.ravel()])])


def transform_state_extended(state: State) -> [np.ndarray]:
    """transforms a state for the extended deep q network agent"""
    transformed_state = [[np.concatenate([state.direction, state.velocity])],
                         [np.transpose([state.perception.ravel()])]]
    return transformed_state


def evaluate_cow(cow_model, environment, game_count, game_length) -> (bool, float):
    total_reward = 0
    for games in range(game_count):
        states = environment.reset()
        cow_state = states[0]
        for frame in range(game_length):
            cow_action = cow_model.select_action(cow_state)
            states, rewards, done, info = environment.step([Action(cow_action)])
            cow_state = states[0]
            total_reward += rewards[0]
            if frame == game_length - 1:
                done = True
            if environment.quit():
                return (True, 0)
            if done:
                break
    return False, total_reward


def train_cow(
        epoch_length=1000,
        episode_length=10,
        game_length=1000,
        batch_size=32):
    """ training loop for a special cow """
    df = []

    ray_count = 20
    action_size = 7

    best_reward = -10000

    cow_model = ExtendedDQNAgent(perception_size=ray_count * 3, metadata_size=3, action_size=action_size,
                                 preprocess=transform_state_extended,
                                 memory_length=100_000)

    environment = Environment(cow_ray_count=ray_count,
                              grass_count=1,
                              wolf_ray_count=ray_count,
                              wolf_count=0,
                              draw=True)

    for epoch in range(epoch_length):
        print('starting epoch {}, cow exploration rate {:.2f}%'.format(
            epoch,
            cow_model.get_exploration_rate()))

        cow_reward_per_epoch = 0

        for episode in range(episode_length):
            states = environment.reset()
            cow_state = states[0]
            old_distance = cow_state.food_distances[0]

            for frame in range(game_length):
                cow_action = cow_model.explore_select_action(cow_state)
                states, rewards, done, info = environment.step([Action(cow_action)])

                cow_reward = rewards[0]
                cow_next_state = states[0]
                new_distance = cow_next_state.food_distances[0]
                delta_distance = old_distance - new_distance
                old_distance = new_distance
                cow_reward += 0.1 * delta_distance
                cow_reward_per_epoch += cow_reward

                if frame == game_length - 1:
                    done = True

                cow_model.remember(cow_state, cow_action, cow_reward, cow_next_state, done)
                cow_state = cow_next_state

                if environment.quit():
                    return pd.DataFrame(data=df, columns=['epoch', 'cow_reward'])
                if done:
                    break
        df.append([epoch, cow_reward_per_epoch])
        cow_model.replay(batch_size)

        stop, total_reward = evaluate_cow(cow_model, environment, 3, game_length)
        if total_reward > best_reward:
            cow_model.save("models/best-cow.HDF5")
        if stop:
            return pd.DataFrame(data=df, columns=['epoch', 'cow_reward'])
        print('finish epoch {}, cow rewards {}'.format(epoch, cow_reward_per_epoch))
    return pd.DataFrame(data=df, columns=['epoch', 'cow_reward'])
