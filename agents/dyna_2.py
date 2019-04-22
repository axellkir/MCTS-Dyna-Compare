import random
from collections import defaultdict


class Dyna2Agent:
    perm_lambda = 0.8
    trans_lambda = 0.8
    epsilon = 0.1

    def __init__(self, get_legal_actions):
        self.get_legal_actions = get_legal_actions
        # Memory (theta)
        self._permanentParameters = defaultdict(lambda: defaultdict(lambda: 0))
        self._transientParameters = defaultdict(lambda: defaultdict(lambda: 0))
        # Eligibility traces
        self._permET = defaultdict(lambda: defaultdict(lambda: 0))
        self._transET = defaultdict(lambda: defaultdict(lambda: 0))

        self.rewards = defaultdict(lambda: defaultdict(lambda: 0))
        self.states = defaultdict(lambda: defaultdict(lambda: 0))
        self.dones = defaultdict(lambda: defaultdict(lambda: 0))

    def get_qvalue(self, state, action, mem_type):
        if mem_type == 'permanent':
            return self._permanentParameters[state][action]
        else:
            return self._transientParameters[state][action] + self._permanentParameters[state][action]

    def play(self, env, t_max=10**4):
        done = False
        state = env.reset()
        self._transientParameters = defaultdict(lambda: defaultdict(lambda: 0))
        self._permET = defaultdict(lambda: defaultdict(lambda: 0))
        total_reward = 0
        t = 0
        self.search(state)
        action = self.get_action(state, 'transient')
        while not done or t > t_max:
            new_state, reward, done, _ = env.step(action)
            total_reward += reward
            self.update_model(state, action, reward, new_state, done)
            self.search(new_state)
            new_action = self.get_action(new_state, 'transient')
            delta = reward + self.get_qvalue(new_state, new_action, 'permanent') \
                    - self.get_qvalue(state, action, 'permanent')
            for s in self._permanentParameters.keys():
                for a in self._permanentParameters[s].keys():
                    self._permanentParameters[s][a] = self._permanentParameters[s][a] \
                                                      + self.learning_rate(state, action, 'permanent') \
                                                      * delta * self._permET[s][a]
                    if s == state and a == action:
                        self._permET[s][a] = self.perm_lambda * self._permET[s][a] + 1
                    else:
                        self._permET[s][a] = self.perm_lambda * self._permET[s][a]

            state = new_state
            action = new_action
            t += 1
        return total_reward

    def search(self, state):
        done = False
        self._transET = defaultdict(lambda: defaultdict(lambda: 0))
        action = self.get_action(state, 'transient')
        while not done:
            new_state, reward, done = self.from_model(state, action)
            new_action = self.get_action(new_state, 'transient')
            trans_delta = reward + self.get_qvalue(new_state, new_action, 'transient') \
                          - self.get_qvalue(state, action, 'transient')
            for s in self._transientParameters.keys():
                for a in self._transientParameters[s].keys():
                    self._transientParameters[s][a] = self._transientParameters[s][a] \
                                                    + self.learning_rate(state, action, 'transient') \
                                                    * trans_delta * self._transET[s][a]
                    if s == state and a == action:
                        self._transET[s][a] = self.trans_lambda * self._transET[s][a] + 1
                    else:
                        self._transET[s][a] = self.trans_lambda * self._transET[s][a]

            state = new_state
            action = new_action

    def update_model(self, state, action, reward, new_state, done):
        self.rewards[state][action] = reward
        self.states[state][action] = new_state
        self.dones[state][action] = done

    def from_model(self, state, action):
        if state in self.rewards.keys():
            if action in self.rewards[state].keys():
                return self.states[state][action], self.rewards[state][action],self.dones[state][action]
        return state, 0, True

    def learning_rate(self, state, action, mem_type):
        if mem_type == 'permanent':
            return 0.1  # TODO : change learning rate using state and action
        else:
            return 0.1  # TODO : change learning rate using state and action

    def get_best_action(self, state, mem_type):
        possible_actions = self.get_legal_actions(state)
        action_values = [self.get_qvalue(state, action, mem_type) for action in possible_actions]
        return possible_actions[action_values.index(max(action_values))]

    def get_action(self, state, mem_type):
        possible_actions = self.get_legal_actions(state)
        if len(possible_actions) == 0:
            return None
        if random.random() > self.epsilon:
            chosen_action = self.get_best_action(state, mem_type)
        else:
            chosen_action = random.choice(possible_actions)
        return chosen_action