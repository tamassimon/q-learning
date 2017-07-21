__author__ = "Tamas Simon"
__copyright__ = "Copyright 2017, Tamas Simon"
__license__ = "GPLv3"

NUMBER_OF_STEPS_WHEN_WALL_MOVES = 1000
NUMBER_OF_ACTIONS = 4
START_X = 3
START_Y = 5
GOAL_X = 8
GOAL_Y = 0
MAX_X = 8
MAX_Y = 5
ROW_LENGTH = MAX_X + 1


class GridWorldModel:
    def __init__(self):
        self.x, self.y = START_X, START_Y
        self.step_count = 0

    # convert x,y coordinates into a numeric index for the state

    @staticmethod
    def state_to_xy(state):
        return state % ROW_LENGTH, state // ROW_LENGTH

    @staticmethod
    def state_from_xy(x, y):
        return y * ROW_LENGTH + x

    # convert numeric index to action expressed as delta x, delta y
    @staticmethod
    def action_from_ndx(ndx):
        actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        return actions[ndx]

    @staticmethod
    def get_start_state():
        return GridWorldModel.state_from_xy(START_X, START_Y)

    def reset(self):
        self.x = START_X
        self.y = START_Y

    def reward(self):
        if self.x == GOAL_X and self.y == GOAL_Y:
            self.x = START_X
            self.y = START_Y
            return 1
        return 0

    def is_terminal_state(self):
        return self.x == GOAL_X and self.y == GOAL_Y

    # actions
    # when trying to move past the edge remain where it was

    def take_action(self, action):
        """ performs the action and returns the reward and the new state"""
        self.step_count += 1
        delta_x, delta_y = GridWorldModel.action_from_ndx(action)
        if not self.would_fall_off(delta_x, delta_y) and not self.would_hit_wall(delta_x, delta_y):
            self.x += delta_x
            self.y += delta_y
            if self.x == GOAL_X and self.y == GOAL_Y:
                return 1, GridWorldModel.state_from_xy(self.x, self.y)
        return 0, GridWorldModel.state_from_xy(self.x, self.y)

    @staticmethod
    def get_number_of_actions():
        return NUMBER_OF_ACTIONS

    @staticmethod
    def get_number_of_states():
        return (MAX_X + 1) * (MAX_Y + 1)

    def would_fall_off(self, delta_x, delta_y):
        future_x = self.x + delta_x
        future_y = self.y + delta_y
        if future_x < 0 or future_x > MAX_X or future_y < 0 or future_y > MAX_Y:
            return True
        return False

    def would_hit_wall(self, delta_x, delta_y):
        future_x = self.x + delta_x
        future_y = self.y + delta_y
        if self.step_count < NUMBER_OF_STEPS_WHEN_WALL_MOVES:
            if future_y == 3 and future_x < 8:
                return True
            else:
                return False
        else:
            if future_y == 3 and future_x > 0:
                return True
            else:
                return False
