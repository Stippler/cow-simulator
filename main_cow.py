from deepcow.agent_brain import ExtendedDQNAgent
from deepcow.environment import Environment
from deepcow.train import train_cow, transform_state_extended

import seaborn as sns
import matplotlib.pyplot as plt

results = train_cow(game_length=1000)
sns.lineplot(results['epoch'], results['cow_reward'], color='brown', )
plt.xlabel('epoch')
plt.ylabel('reward')
plt.savefig('result/reward.png')
plt.show()
