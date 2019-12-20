from deepcow.agent_brain import DQNAgent

from deepcow.environment import Environment
from deepcow.agent import *

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def transform_state_1d(state: State) -> np.ndarray:
    """transforms a state into a 1d numpy array"""
    return np.array([np.concatenate([state.direction, state.velocity, state.perception.ravel()])])


def train_dqn_agents(cow_model: DQNAgent,
                     wolf_model: DQNAgent,
                     environment: Environment,
                     epoch_length=1000,
                     episode_length=4,
                     game_length=1000,
                     batch_size=32) -> (DQNAgent, DQNAgent):
    # clock = pygame.time.Clock()

    all_rewards = []
    for epoch in range(epoch_length):

        print('starting epoch {}, cow exploration rate {:.2f}% and wolf exploration rate {:.2f}%'.format(
            epoch,
            cow_model.epsilon,
            wolf_model.epsilon))

        cow_reword_per_epoch = 0
        wolf_reword_per_epoch = 0
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
                cow_reword_per_epoch += cow_reward
                cow_next_state = states[0]

                wolf_reward = rewards[1]
                wolf_reword_per_epoch += wolf_reward
                wolf_next_state = states[1]

                cow_model.remember(cow_state, cow_action, cow_reword_per_epoch, cow_next_state, done)
                wolf_model.remember(wolf_state, wolf_action, wolf_reword_per_epoch, wolf_next_state, done)

                cow_state = cow_next_state
                wolf_state = wolf_next_state

                if done:
                    break
                if environment.quit():
                    return pd.DataFrame(data=all_rewards,
                                        columns=['epoch', 'cow_reward', 'wolf_reward',
                                                 'wolf_border_collision', 'cow_border_collision'], )

        all_rewards.append(
            [epoch, cow_reword_per_epoch / episode_length, wolf_reword_per_epoch / episode_length,
             cow_border_collision / episode_length, wolf_border_collision / episode_length])
        for i in range(4):
            cow_model.replay(batch_size)
            wolf_model.replay(batch_size)
        print('finish epoch {}, cow rewards {}, wolf rewards {}'.format(epoch, cow_reword_per_epoch,
                                                                        wolf_reword_per_epoch))
    return pd.DataFrame(data=all_rewards, columns=['epoch', 'cow_reward', 'wolf_rewards'], )


epoch_length = 1000
ray_count = 20
action_size = 7

cow_model = DQNAgent(4 + ray_count * 3, action_size, preprocess=transform_state_1d)
wolf_model = DQNAgent(4 + ray_count * 3, action_size, preprocess=transform_state_1d)

environment = Environment(cow_ray_count=ray_count,
                          grass_count=1,
                          wolf_ray_count=ray_count,
                          draw=True)

results = train_dqn_agents(cow_model, wolf_model, environment)
print(results)
sns.lineplot(results['epoch'], results['cow_reward'], color='brown', )
sns.lineplot(results['epoch'], results['wolf_reward'], color='blue')
plt.xlabel('epoch')
plt.ylabel('reward')
plt.savefig('result/dqn-result.png')
plt.show()

sns.lineplot(results['epoch'], results['cow_border_collision'], color='brown', )
sns.lineplot(results['epoch'], results['wolf_border_collision'], color='blue')
plt.xlabel('epoch')
plt.ylabel('border collision count')
plt.savefig('result/dqn-border-collision-result.png')
plt.show()
