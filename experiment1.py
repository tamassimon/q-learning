from Environment import GridWorldModel
from Agent import Agent
import itertools
import matplotlib.pyplot as plt
import numpy as np

__author__ = "Tamas Simon"
__copyright__ = "Copyright 2017, Tamas Simon"
__license__ = "GPLv3"

MAX_EPISODES = 200


def plot_results():
    plt.subplot(2, 1, 1)
    plt.plot(np.cumsum(np.array(agent.rewards)))
    plt.ylabel('Cumulative Rewards')
    plt.xlabel('Steps')
    plt.subplot(2, 1, 2)
    plt.plot(np.array(agent.number_of_steps_til_reward))
    plt.ylabel('# Steps to Reward')
    plt.xlabel('Episodes')
    plt.show()

# Iterate through a number of episodes (set by MAX_EPISODES) and plot the results.

if __name__ == '__main__':
    environment = GridWorldModel()
    agent = Agent(environment)

    for _ in itertools.repeat(None, MAX_EPISODES):
        environment.reset()
        agent.state = environment.get_start_state()
        while True:
            agent.act()
            if environment.is_terminal_state():
                break

    plot_results()
