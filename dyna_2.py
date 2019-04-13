import random
# from copy import deepcopy
# from math import sqrt, log
import numpy as np

class Dyna2Agent:
    def __init__(self, alpha, epsilon, discount, n_steps, get_legal_actions):
        self.get_legal_actions = get_legal_actions
        self._permanentMemory = np.zeros()
        self._transientMemory = np.zeros()
        self._permET = np.zeros()
        self._transET = np.zeros()

        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount
        self.n_steps = n_steps

    def get_qvalue(self, state, action, mem_type):
        if mem_type == 'permanent':
            return self._permQvalues[state][action]
        else:
            return self._transQvalues[state][action]

    def play(self, env):
        done = False
        state = env.reset()
        self._transientMemory = []
        self._permET = []

        self.search(state)
        action = self.get_action(state, 'transient')
        while not done:
            new_state, reward, done, _ = env.step(action)
            self.update_model(state, action, reard, new_state)
            self.search(new_state)
            new_action = self.get_action(new_state, 'transient')
            delta = reward + self.get_qvalue(new_state, new_action, 'permanent') -...
                                            self.get_qvalue(state, action, 'permanent')
            self._permanentMemory = self._permanentMemory + delta*self._permET


    def search(self):
        self._transET = []

    def get_best_action(self, state, mem_type):
        possible_actions = self.get_legal_actions(state)
        action_values = [self.get_qvalue(self.to_board(state), action, mem_type) for action in possible_actions]
        return possible_actions[action_values.index(max(action_values))]

    def get_action(self, state, mem_type):
        possible_actions = self.get_legal_actions(state)
        if len(possible_actions) == 0:
            return None
        epsilon = self.epsilon
        if random.random() > epsilon:
            chosen_action = self.get_best_action(state, mem_type)
        else:
            chosen_action = random.choice(possible_actions)
        return chosen_action