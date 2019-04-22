from gym.spaces.discrete import Discrete
from gym.spaces import Box
from gym_sokoban.envs.render_utils import room_to_rgb, room_to_tiny_world_rgb
from gym_sokoban.envs.room_utils import generate_room
import numpy as np
from gym_sokoban.envs import SokobanEnv


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
        self.penalty_box_off_target = -1
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

    def reset(self, second_player=False):
        (self.room_fixed, self.room_state, self.box_mapping, self.player_position) = self.init_room
        self.num_env_steps = 0
        self.reward_last = 0
        self.boxes_on_target = 0

        starting_observation = room_to_rgb(self.room_state, self.room_fixed)
        return starting_observation


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
