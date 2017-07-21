from Environment import GridWorldModel
import random
import numpy as np

__author__ = "Tamas Simon"
__copyright__ = "Copyright 2017, Tamas Simon"
__license__ = "GPL v3"

GAMMA = 0.95    # discount factor
EPSILON = 0.6  # greedyness
ALPHA = 0.5     # learning rate, 0 < alpha <= 1


class Agent:
    def __init__(self, environment):
        self.environment = environment
        self.Q = np.zeros((GridWorldModel.get_number_of_states(), GridWorldModel.get_number_of_actions()), dtype=float)
        self.dQ = np.zeros((GridWorldModel.get_number_of_states(), GridWorldModel.get_number_of_actions()), dtype=float)
        self.state = GridWorldModel.get_start_state()
        self.rewards = []
        self.number_of_steps_til_reward = []
        self.steps_til_reward = 0

    @staticmethod
    def explore():
        return random.randrange(0, GridWorldModel.get_number_of_actions())

    def exploit(self):
        # When there are multiple options with the same Q value then choose among them randomly.
        best_actions = np.argwhere(self.Q[self.state] == np.amax(self.Q[self.state]))
        return random.choice(best_actions.flatten().tolist())
        # Deterministic would be:
        # return np.argmax(self.Q[self.state])

    def act(self):
        # Implement epsilon greedy policy.
        dice = random.random()
        if dice < EPSILON:
            action = self.explore()
        else:
            action = self.exploit()

        # Take action.
        reward, new_state = self.environment.take_action(action)
        self.steps_til_reward += 1

        # Update statistics about rewards learned; this will be used for plotting the results.
        self.rewards.append(reward)
        if reward == 1:
            self.number_of_steps_til_reward.append(self.steps_til_reward)
            self.steps_til_reward = 0

        # Do the Q-Learning.
        self.learn(new_state, action, reward)
        self.state = new_state

    def learn(self, new_state, action, reward):
        """ Q-Learning implementation https://en.wikipedia.org/wiki/Q-learning

        The accumulated learning is remembered in dQ(s,a) so that it can be used for asynchronous learning.

        :param new_state: int The new state where the action lead us.
        :param action: int The last action taken.
        :param reward: int The reward (if any) that was received as a result of taking this action.
        :return:
        """
        delta_q = reward + GAMMA * np.amax(self.Q[new_state]) - self.Q[self.state, action]
        self.dQ[self.state, action] += delta_q
        self.Q[self.state, action] += ALPHA * delta_q
