from collections import defaultdict
import random
import numpy as np
from copy import deepcopy


class DynaQAgent:
    def __init__(self, alpha, epsilon, discount, n_steps, get_legal_actions, is_array=False):
        self.get_legal_actions = get_legal_actions
        self._qvalues = defaultdict(lambda: defaultdict(lambda: 0))
        self._memoryModel = []
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount
        self.n_steps = n_steps
        self.is_array = is_array
        self.states = {}

    def array_to_int(self, state):
        if self.is_array:
            for name, value in self.states.items():
                if np.array_equal(value, state):
                    return name
            self.states[str(len(self.states.keys()))] = deepcopy(state)
            return str(len(self.states.keys()) - 1)
        else:
            return state

    def get_qvalue(self, state, action):
        return self._qvalues[state][action]

    def set_qvalue(self, state, action, value):
        self._qvalues[state][action] = value

    def get_value(self, state):
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return 0.0
        if len(possible_actions) == 0:
            return 0.0

        value = max(self.get_qvalue(state, action) for action in possible_actions)

        return value

    def play(self, env, t_max=10**4):
        total_reward = 0.0
        state = env.reset()
        for t in range(t_max):
            a = self.get_action(state)
            next_state, r, done, _ = env.step(a)
            self.update(state, a, r, next_state)
            state = next_state
            total_reward += r
            if done:
                break
        return total_reward


    def update(self, state, action, reward, next_state):
        alpha = self.alpha
        gamma = self.discount
        q_update = self.get_qvalue(self.array_to_int(state), action) + alpha * \
                   (reward + gamma * self.get_value(self.array_to_int(next_state)) -
                    self.get_qvalue(self.array_to_int(state), action))
        self.set_qvalue(self.array_to_int(state), action, q_update)
        self._memoryModel.append((self.array_to_int(state), action, reward, self.array_to_int(next_state)))
        self.search()

    def search(self):
        n_steps = self.n_steps
        alpha = self.alpha
        gamma = self.discount
        for _ in range(n_steps):
            state, action, reward, next_state = random.choice(self._memoryModel)

            q_update = self.get_qvalue(state, action) + alpha * (
                    reward + gamma * self.get_value(next_state) - self.get_qvalue(state, action))
            self.set_qvalue(state, action, q_update)

    def get_best_action(self, state):

        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        array_qvalues = [self.get_qvalue(self.array_to_int(state), action) for action in possible_actions]
        return possible_actions[array_qvalues.index(max(array_qvalues))]

    def get_action(self, state):
        # Pick Action
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        # agent parameters:
        epsilon = self.epsilon

        if random.random() > epsilon:
            chosen_action = self.get_best_action(state)
        else:
            chosen_action = random.choice(possible_actions)

        return chosen_action
