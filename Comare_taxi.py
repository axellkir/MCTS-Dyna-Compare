import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from dyna_q import DynaQAgent
from uct import UCTAgent
from qlearning import QLearningAgent
from taxi_env import TaxiEnv

env = TaxiEnv(1)
n_actions = env.action_space.n
env.render()
s = env.reset()

agent_uct = UCTAgent(get_legal_actions=lambda s: range(n_actions))
agent_uct.set_root(s)

agent_ql = QLearningAgent(alpha=0.25, epsilon=0.05, discount=0.9,
                          get_legal_actions=lambda s: range(n_actions))

agent_dyna_q = DynaQAgent(alpha=0.25, epsilon=0.05, discount=0.9, n_steps = 200,
                          get_legal_actions=lambda s: range(n_actions))


def play_and_train(env, agent, t_max=10**4):
    total_reward = 0.0
    state = env.reset()

    for t in range(t_max):
        a = agent.get_action(state)

        next_state, r, done, _ = env.step(a)
        agent.update(state, a, r, next_state)

        state = next_state
        total_reward += r
        if done:
            break

    return total_reward


def moving_average(x, span=100): return DataFrame(
    {'x': np.asarray(x)}).x.ewm(span=span).mean().values


rewards_dyna_q, rewards_ql, rewards_uct = [], [], []

for i in range(50):
    rewards_dyna_q.append(play_and_train(env, agent_dyna_q, 200))
    rewards_ql.append(play_and_train(env, agent_ql, 200))
    rewards_uct.append(agent_uct.play(env, 200))

print('UCT mean reward =', np.mean(rewards_uct[-100:]))
print('DYNA-Q mean reward =', np.mean(rewards_dyna_q[-100:]))
print('Q-LEARNING mean reward =', np.mean(rewards_ql[-100:]))
plt.plot(moving_average(rewards_uct), label='uct')
plt.plot(moving_average(rewards_dyna_q), label='dyna-q')
plt.plot(moving_average(rewards_ql), label='q-learning')
plt.grid()
plt.legend()
plt.show()
