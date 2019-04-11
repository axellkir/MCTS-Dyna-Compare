from collections import defaultdict
from taxi_env import TaxiEnv
import random
from math import sqrt, log


class Node:
    def __init__(self, state, parent=None, action=None):
        # Предыдущее состояние
        self.parent = parent
        # Состояние
        self.state = state
        # Действие из предыдущего состояния
        self.action = action
        # Является ли эта вершина корнем поиска
        self.is_root = False
        # Словарь со значениями функции Q(s, a)
        self.qvalues = defaultdict(lambda: 0)
        # Список следующих состояний
        self.children = {}
        # Количество посещений данного состояния
        self.visits = 0


def check_state_in_children(node, state):
    for child in node.children.values():
        if child.state == state:
            return True, child
    return False, 0


def check_state_in_children_act(node, action):
    for a in node.children.keys():
        if a == action:
            assert action == node.children[a].action, 'Wrong action in NODE'
            return True, node.children[a]
    return False, 0


class UCTAgent:
    def __init__(self, get_legal_actions, rollouts=20, horizon=100, gamma=0.9, ucb_const=3):
        random.seed(42)
        self.get_legal_actions = get_legal_actions
        self.rollouts = rollouts
        self.horizon = horizon
        self.ucb = ucb_const
        self.gamma = gamma
        self.time = 0
        self.root = None
        self.temp_root = None
        print("Initialized")

    def set_root(self, state):
        root = Node(state)
        root.visits = 1
        self.root = root
        self.temp_root = root

    def new_node(self, state, parent, action):
        node = Node(state=state, parent=parent, action=action)
        assert action not in parent.children.keys(), 'Not new NODE'
        parent.children[action] = node
        return node

    def get_action(self):
        root_node = self.temp_root
        root_node.is_root = True
        for _ in range(self.rollouts):
            self.simulate(TaxiEnv(root_node.state), root_node)
        root_node.is_root = False
        action = self.ucb_select(root_node)
        return action

    def play(self, env, t_max=100):
        total_reward = 0.0
        self.time = t_max
        env.reset()
        self.temp_root = self.root
        for t in range(t_max):
            a = self.get_action()
            s, r, done, _ = env.step(a)
            is_in, child = check_state_in_children(self.temp_root, s)
            assert is_in, "Child not found!!!"
            self.temp_root = child
            total_reward += r
            if done:
                break
        return total_reward

    def simulate(self, env, root_node):
        node, total_reward, done, t = self.sim_tree(env, root_node)
        total_reward = self.sim_default(env, node.state, total_reward, done, t)
        self.back_up(node, total_reward)

    def sim_tree(self, env, root_node):
        t = 0
        done = False
        node = root_node
        total_reward = 0
        while not done and (t < self.horizon) and (t < self.time):
            action = self.ucb_select(node)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            t += 1
            is_in, child = check_state_in_children_act(node, action)
            if not is_in:
                node = self.new_node(state, node, action)
                return node, total_reward, done, t
            else:
                node = child
                assert child.state == state, 'Wrong state!'
        return node, total_reward, done, t

    def sim_default(self, env, state, total_reward, done, t):
        while not done and (t < self.time):
            action = self.default_policy(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            t += 1
        return total_reward

    def default_policy(self, state):
        return random.choice(self.get_legal_actions(state))

    def back_up(self, node, total_reward):
        node.visits += 1
        prev_visits = node.visits
        action = node.action
        while not node.is_root:
            node = node.parent
            node.visits += 1
            node.qvalues[action] += (total_reward - node.qvalues[action]) / float(prev_visits)
            prev_visits = node.visits
            action = node.action

    def ucb_select(self, node):
        possible_actions = self.get_legal_actions(node.state)
        c = self.ucb
        array_qvalues = {
                action: node.qvalues[action] + c * sqrt(log(node.visits) / float(node.children[action].visits))
                if action in node.qvalues.keys() else float('+inf') for action in possible_actions
        }
        return max(array_qvalues, key=lambda k: array_qvalues[k])
