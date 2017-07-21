from unittest import TestCase

import itertools

import Agent
import numpy as np

__author__ = "Tamas Simon"
__copyright__ = "Copyright 2017, Tamas Simon"
__license__ = "GPLv3"


class MockEnvironment(object):
    pass


class TestAgent(TestCase):
    def test_exploit_choses_randomly_among_equals(self):
        mock_environment = MockEnvironment()
        agent = Agent.Agent(mock_environment)
        l = []
        for _ in itertools.repeat(None, 1000):
            l.append(agent.exploit())
        counts = np.bincount(np.array(l))
        assert counts[0] < 500

    def test_learn_one_step_reward(self):
        mock_environment = MockEnvironment()
        agent = Agent.Agent(mock_environment)
        agent.state = 0
        new_state = 1
        action = 0
        reward = 1
        agent.learn(new_state, action, reward)
        assert agent.Q[0, 0] > 0
        assert agent.Q[0, 0] == Agent.ALPHA * agent.dQ[0, 0]
        assert agent.dQ[0, 0] == reward
        assert agent.Q[0, 0] == Agent.ALPHA * reward

    def test_learn_two_step_reward(self):
        mock_environment = MockEnvironment()
        agent = Agent.Agent(mock_environment)
        agent.state = 0
        new_state = 1
        action = 0
        reward = 0
        agent.learn(new_state, action, reward)
        agent.state = new_state
        new_state = 2
        reward = 1
        agent.learn(new_state, action, reward)
        assert agent.Q[0, 0] == 0
        assert agent.Q[1, 0] > 0
