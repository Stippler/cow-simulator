from deepcow.agent_brain import *

from deepcow.environment import Environment
from deepcow.entity import *

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def transform_state_1d(state: State) -> np.ndarray:
    """transforms a state into a 1d numpy array"""
    return np.array([np.concatenate([state.direction, state.velocity, state.perception.ravel()])])


def transform_state_extended(state: State) -> [np.ndarray]:
    """transforms a state for the extended dqn agent"""
    transformed_state = [[np.concatenate([state.direction, state.velocity])], [np.transpose([state.perception.ravel()])]]
    return transformed_state


def train_dqn_agents(cow_model: DQNAgent,
                     wolf_model: DQNAgent,
                     environment: Environment,
                     epoch_length=1000,
                     episode_length=10,
                     game_length=1000,
                     batch_size=32) -> (DQNAgent, DQNAgent):
    all_rewards = []
    for epoch in range(epoch_length):
        print('starting epoch {}, cow exploration rate {:.2f}% and wolf exploration rate {:.2f}%'.format(
            epoch,
            cow_model.get_exploration_rate(),
            wolf_model.get_exploration_rate()))

        cow_reward_per_epoch = 0
        wolf_reward_per_epoch = 0
        wolf_border_collision = 0
        cow_border_collision = 0

        for episode in range(episode_length):
            states = environment.reset()
            cow_state = states[0]
            wolf_state = states[0]

            for frame in range(game_length):
                cow_action = cow_model.explore_select_action(cow_state)
                wolf_action = wolf_model.explore_select_action(wolf_state)

                states, rewards, done, info = environment.step([Action(cow_action), Action(wolf_action)])

                cow_border_collision += info['cow_border_collisions']
                wolf_border_collision += info['wolf_border_collisions']

                cow_reward = rewards[0]
                cow_reward_per_epoch += cow_reward
                cow_next_state = states[0]

                wolf_reward = rewards[1]
                wolf_reward_per_epoch += wolf_reward
                wolf_next_state = states[1]

                if frame == game_length - 1:
                    done = True

                cow_model.remember(cow_state, cow_action, cow_reward_per_epoch, cow_next_state, done)
                wolf_model.remember(wolf_state, wolf_action, wolf_reward_per_epoch, wolf_next_state, done)

                cow_state = cow_next_state
                wolf_state = wolf_next_state

                if environment.quit():
                    return pd.DataFrame(data=all_rewards,
                                        columns=['epoch', 'cow_reward', 'wolf_reward',
                                                 'wolf_border_collision', 'cow_border_collision'], )
                if done:
                    break

        all_rewards.append([epoch, cow_reward_per_epoch / episode_length,
                            wolf_reward_per_epoch / episode_length,
                            cow_border_collision / episode_length,
                            wolf_border_collision / episode_length])

        for i in range(8):
            if epoch % 2 == 0:
                cow_model.replay(batch_size)
            else:
                wolf_model.replay(batch_size)
        print('finish epoch {}, cow rewards {}, wolf rewards {}'.format(epoch, cow_reward_per_epoch,
                                                                        wolf_reward_per_epoch))
    return pd.DataFrame(data=all_rewards, columns=['epoch', 'cow_reward', 'wolf_rewards'], )


epoch_length = 1000
ray_count = 20
action_size = 7

cow_model = ExtendedDQNAgent(perception_size=ray_count * 3, metadata_size=3, action_size=action_size,
                             preprocess=transform_state_extended,
                             memory_length=10_000)
wolf_model = ExtendedDQNAgent(perception_size=ray_count * 3, metadata_size=3, action_size=action_size,
                              preprocess=transform_state_extended)

environment = Environment(cow_ray_count=ray_count,
                          grass_count=1,
                          wolf_ray_count=ray_count,
                          draw=True)

results = train_dqn_agents(cow_model, wolf_model, environment, game_length=1000)
print(results)

sns.lineplot(results['epoch'], results['cow_border_collision'], color='brown', )
sns.lineplot(results['epoch'], results['wolf_border_collision'], color='blue')
plt.xlabel('epoch')
plt.ylabel('border collision count')
plt.savefig('result/dq-border-collision-result.png')
plt.show()

sns.lineplot(results['epoch'], results['cow_reward'], color='brown', )
sns.lineplot(results['epoch'], results['wolf_reward'], color='blue')
plt.xlabel('epoch')
plt.ylabel('reward')
plt.savefig('result/dq-reward.png')
plt.show()

cow_model.save('models/deepq-cow.HDF5')
wolf_model.save('models/deepq-wolf.HDF5')
