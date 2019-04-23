import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from agents.dyna_q import DynaQAgent
from agents.dyna_2 import Dyna2Agent
from agents.uct import UCTAgent
from envs.sokoban_env import MySokobanEnv


def moving_average(x, span=100): return DataFrame(
    {'x': np.asarray(x)}).x.ewm(span=span).mean().values


env = MySokobanEnv()
n_actions = env.action_space.n
env.render()
s = env.reset()

agent_dyna_q = DynaQAgent(alpha=0.6, epsilon=0.15, discount=0.9, n_steps=5000,
                          get_legal_actions=lambda s: range(n_actions), is_array=True)

agent_dyna_2 = Dyna2Agent(get_legal_actions=lambda s: range(n_actions), is_array=True)

agent_uct = UCTAgent(get_legal_actions=lambda s: range(n_actions), is_copy=False, save=lambda env: env.save_room,
                     restore=lambda env: env.restore_room)
agent_uct.set_root(s)

rewards_dyna_q = []
rewards_dyna_2 = []
rewards_uct = []

for i in range(400):
    #     rewards_dyna_q.append(agent_dyna_q.play(env))
    rewards_dyna_2.append(agent_dyna_2.play(env))
    #     rewards_uct.append(agent_uct.play(env))
    if i % 10 == 0:
        #         print('DYNA-Q mean reward =', np.mean(rewards_dyna_q[-100:]))
        print('DYNA-2 mean reward =', np.mean(rewards_dyna_2[-100:]))
        #         print('UCT mean reward =', np.mean(rewards_uct[-100:]))

        #         plt.plot(moving_average(rewards_dyna_q), label='dyna-q')
        plt.plot(moving_average(rewards_dyna_2), label='dyna-2')
        #         plt.plot(moving_average(rewards_uct), label='uct')

        plt.grid()
        plt.legend()
        plt.show()
