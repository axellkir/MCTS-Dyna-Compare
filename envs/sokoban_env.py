import numpy as np
from gym_sokoban.envs import SokobanEnv
from gym.spaces.discrete import Discrete
from gym.spaces import Box
from gym_sokoban.envs.room_utils import generate_room


class MySokobanEnv(SokobanEnv):
    def __init__(self, dim_room=(10, 10),
                 max_steps=120,
                 num_boxes=4,
                 num_gen_steps=None):

        # General Configuration
        self.dim_room = dim_room
        if num_gen_steps == None:
            self.num_gen_steps = int(1.7 * (dim_room[0] + dim_room[1]))
        else:
            self.num_gen_steps = num_gen_steps

        self.num_boxes = num_boxes
        self.boxes_on_target = 0

        # Penalties and Rewards
        self.penalty_for_step = -0.1
        self.penalty_box_off_target = -0.1
        self.reward_box_on_target = 1
        self.reward_finished = 10
        self.reward_last = 0

        # Other Settings
        self.viewer = None
        self.max_steps = max_steps
        self.action_space = Discrete(len(ACTION_LOOKUP))
        screen_height, screen_width = (dim_room[0] * 16, dim_room[1] * 16)
        self.observation_space = Box(low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8)

        # Initialize Room
        self.generate()
        self.reset()

    def step(self, action):
        assert action in ACTION_LOOKUP

        self.num_env_steps += 1

        self.new_box_position = None
        self.old_box_position = None

        moved_player = False
        moved_box = False
        # All push actions are in the range of [0, 3]
        if action < 4:
            moved_player, moved_box = self._push(action)

        else:
            moved_player = self._move(action)

        self._calc_reward()

        done = self._check_if_done()

        # Convert the observation to RGB frame
        #         observation = self.render(mode='rgb_array')

        info = {
            "action.name": ACTION_LOOKUP[action],
            "action.moved_player": moved_player,
            "action.moved_box": moved_box,
        }
        if done:
            info["maxsteps_used"] = self._check_if_maxsteps()
            info["all_boxes_on_target"] = self._check_if_all_boxes_on_target()

        return self.room_state, self.reward_last, done, info

    def generate(self, second_player=False):
        try:
            self.room_fixed, self.room_state, self.box_mapping = generate_room(
                dim=self.dim_room,
                num_steps=self.num_gen_steps,
                num_boxes=self.num_boxes,
                second_player=second_player
            )
        except (RuntimeError, RuntimeWarning) as e:
            print("[SOKOBAN] Runtime Error/Warning: {}".format(e))
            print("[SOKOBAN] Retry . . .")
            return self.reset()

        self.player_position = np.argwhere(self.room_state == 5)[0]
        self.init_room = (self.room_fixed, self.room_state, self.box_mapping, self.player_position)

    def save_room(self):
        self.whole_env = (self.room_fixed, self.room_state, self.box_mapping, self.player_position)
        self.env_params = (self.num_env_steps, self.reward_last, self.boxes_on_target)

    def restore_room(self):
        (self.room_fixed, self.room_state, self.box_mapping, self.player_position) = self.whole_env
        (self.num_env_steps, self.reward_last, self.boxes_on_target) = self.env_params

    def reset(self):
        (self.room_fixed, self.room_state, self.box_mapping, self.player_position) = self.init_room
        self.num_env_steps = 0
        self.reward_last = 0
        self.boxes_on_target = 0

        return self.room_state


ACTION_LOOKUP = {
    0: 'push up',
    1: 'push down',
    2: 'push left',
    3: 'push right',
    4: 'move up',
    5: 'move down',
    6: 'move left',
    7: 'move right',
}

# Moves are mapped to coordinate changes as follows
# 0: Move up
# 1: Move down
# 2: Move left
# 3: Move right

CHANGE_COORDINATES = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}

RENDERING_MODES = ['rgb_array', 'human', 'tiny_rgb_array', 'tiny_human']
