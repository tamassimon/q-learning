from unittest import TestCase
import Environment

__author__ = "Tamas Simon"
__copyright__ = "Copyright 2017, Tamas Simon"
__license__ = "GPLv3"

actions = {'down': 0, 'up': 1, 'right': 2, 'left': 3}


class TestGridWorldModel(TestCase):
    def test_state_to_xy(self):
        for x in range(Environment.MAX_X + 1):
            for y in range(Environment.MAX_Y + 1):
                state = Environment.GridWorldModel.state_from_xy(x, y)
                new_x, new_y = Environment.GridWorldModel.state_to_xy(state)
                assert x == new_x
                assert y == new_y

    def test_cannot_fall_off_bottom(self):
        model = Environment.GridWorldModel()
        model.reset()
        assert model.x == Environment.START_X
        assert model.y == Environment.START_Y
        reward, state = model.take_action(actions['down'])
        assert model.x == Environment.START_X
        assert model.y == Environment.START_Y
        assert reward == 0
        assert state == Environment.GridWorldModel.state_from_xy(model.x, model.y)

    def test_cannot_fall_off_top(self):
        model = Environment.GridWorldModel()
        model.x = Environment.START_X
        model.y = 0
        reward, state = model.take_action(actions['up'])
        assert model.x == Environment.START_X
        assert model.y == 0
        assert reward == 0
        assert state == Environment.GridWorldModel.state_from_xy(model.x, model.y)

    def test_cannot_fall_off_left_side(self):
        model = Environment.GridWorldModel()
        model.x = 0
        model.y = Environment.START_Y
        reward, state = model.take_action(actions['left'])
        assert model.x == 0
        assert model.y == Environment.START_Y
        assert reward == 0
        assert state == Environment.GridWorldModel.state_from_xy(model.x, model.y)

    def test_cannot_fall_off_right_side(self):
        model = Environment.GridWorldModel()
        model.x = Environment.MAX_X
        model.y = Environment.START_Y
        reward, state = model.take_action(actions['right'])
        assert model.x == Environment.MAX_X
        assert model.y == Environment.START_Y
        assert reward == 0
        assert state == Environment.GridWorldModel.state_from_xy(model.x, model.y)

    def test_can_go_through_hole(self):
        model = Environment.GridWorldModel()
        for step in ['up', 'right', 'right', 'right', 'right', 'right', 'up', 'up', 'up', 'up']:
            model.take_action(actions[step])
        assert model.x == Environment.GOAL_X
        assert model.y == Environment.GOAL_Y
        assert model.is_terminal_state()

    def test_cannot_go_through_wall(self):
        model = Environment.GridWorldModel()
        for step in ['up', 'up', 'up', 'up']:
            reward, state = model.take_action(actions[step])
        assert model.x == Environment.START_X
        assert model.y == 4

    def test_wall_moves_after_n_steps(self):
        model = Environment.GridWorldModel()
        for s in range(Environment.NUMBER_OF_STEPS_WHEN_WALL_MOVES + 1 - 5):
            model.take_action(actions['down'])
        for step in ['up', 'left', 'left', 'left']:
            model.take_action(actions[step])
        assert model.step_count == Environment.NUMBER_OF_STEPS_WHEN_WALL_MOVES
        for step in ['up', 'up', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'up',
                     'up', 'up']:
            model.take_action(actions[step])
        assert model.x == Environment.GOAL_X
        assert model.y == Environment.GOAL_Y
        assert model.is_terminal_state()
