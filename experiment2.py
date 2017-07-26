from multiprocessing import Manager, Pool

import matplotlib.pyplot as plt
import numpy as np
import random

from Agent import Agent
from Environment import GridWorldModel

__author__ = "Tamas Simon"
__copyright__ = "Copyright 2017, Tamas Simon"
__license__ = "GPLv3"

# This experiment implements asynchronous Q-learning.
# It's using the multiprocessing package from the standard library.
# State is shared between the processes using a Manager server process; this provides a dictionary.
# These are the keys used in the dictionary to share rewards and the Q(s,a) matrix among processes.
REWARDS_KEY = 'rewards'
Q_SHARED_KEY = 'q'

MIN_PROCESS_COUNT = 2
MAX_PROCESS_COUNT = 32
MAX_STEPS_PER_AGENT = 10000
ASYNC_UPDATE_INTERVAL = 5


def agent_loop(dictionary, lock1, lock2):
    random.seed()
    environment = GridWorldModel()
    agent = Agent(environment)
    agent.Q = dictionary[Q_SHARED_KEY]  # initialize with shared Q

    while environment.step_count < MAX_STEPS_PER_AGENT:
        environment.reset()
        agent.state = environment.get_start_state()
        while True:
            agent.act()
            if environment.step_count % ASYNC_UPDATE_INTERVAL == 0 or environment.is_terminal_state():
                lock1.acquire()
                q = dictionary[Q_SHARED_KEY]
                # Need to write it back, otherwise the proxy won't pick up the changes.
                dictionary[Q_SHARED_KEY] = np.add(q, agent.dQ)
                lock1.release()
                agent.dQ = np.zeros((GridWorldModel.get_number_of_states(), GridWorldModel.get_number_of_actions()),
                                    dtype=float)
            if environment.is_terminal_state():
                break

    lock2.acquire()
    combined_rewards = dictionary[REWARDS_KEY]
    agents_rewards = np.array(agent.rewards)
    # ...same here
    dictionary[REWARDS_KEY] = np.add(combined_rewards, agents_rewards[:MAX_STEPS_PER_AGENT])
    lock2.release()


def run_process_pool(process_count=MAX_PROCESS_COUNT):
    q_shared = np.zeros((GridWorldModel.get_number_of_states(), GridWorldModel.get_number_of_actions()), dtype=float)
    d[Q_SHARED_KEY] = q_shared
    d[REWARDS_KEY] = np.zeros(MAX_STEPS_PER_AGENT)
    pool = Pool(processes=process_count)
    for i in xrange(process_count):
        pool.apply_async(agent_loop, args=(d, l1, l2))
    pool.close()
    pool.join()
    return np.cumsum(d[REWARDS_KEY]) / process_count

# Runs asynchronous Q-learning two times.
# First with a process pool of MAX_PROCESS_COUNT processes, then with a pool of MIN_PROCESS_COUNT.
# The average reward among processes over time is collected and plotted to demonstrate that more processes
# aka more agents learning in parallel and sharing their learning leads to faster learning.

if __name__ == '__main__':
    manager = Manager()
    d = manager.dict()
    l1 = manager.Lock()
    l2 = manager.Lock()

    avg_reward_max_process_count = run_process_pool(process_count=MAX_PROCESS_COUNT)
    plt.plot(avg_reward_max_process_count, color="blue", label="Process count: {}".format(MAX_PROCESS_COUNT))
    avg_reward_min_process_count = run_process_pool(process_count=MIN_PROCESS_COUNT)
    plt.plot(avg_reward_min_process_count, color="red", label="Process count: {}".format(MIN_PROCESS_COUNT))
    plt.legend(loc='upper left')
    plt.ylabel('Average  Reward')
    plt.show()
