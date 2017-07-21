from multiprocessing import Manager, Pool

import matplotlib.pyplot as plt
import numpy as np

from Agent import Agent
from Environment import GridWorldModel


REWARDS_KEY = 'rewards'
Q_SHARED_KEY = 'q'

MIN_PROCESS_COUNT = 2
MAX_PROCESS_COUNT = 32
MAX_STEPS_PER_AGENT = 10000
ASYNC_UPDATE_INTERVAL = 5


def agent_loop(dict, lock1, lock2):
    environment = GridWorldModel()
    agent = Agent(environment)
    agent.Q = dict[Q_SHARED_KEY]  # initialize with shared Q

    while environment.step_count < MAX_STEPS_PER_AGENT:
        environment.reset()
        agent.state = environment.get_start_state()
        while True:
            agent.act()
            if environment.step_count % ASYNC_UPDATE_INTERVAL == 0 or environment.is_terminal_state():
                lock1.acquire()
                Q = dict[Q_SHARED_KEY]
                # need to write it back, otherwise the proxy won't pick up the changes
                dict[Q_SHARED_KEY] = np.add(Q, agent.dQ)
                lock1.release()
                agent.dQ = np.zeros((GridWorldModel.get_number_of_states(), GridWorldModel.get_number_of_actions()),
                                    dtype=float)
            if environment.is_terminal_state():
                break

    lock2.acquire()
    combined_rewards = dict[REWARDS_KEY]
    agents_rewards = np.array(agent.rewards)
    # same here
    dict[REWARDS_KEY] = np.add(combined_rewards, agents_rewards[:MAX_STEPS_PER_AGENT])
    lock2.release()


def run_process_pool(process_count=MAX_PROCESS_COUNT):
    Q_shared = np.zeros((GridWorldModel.get_number_of_states(), GridWorldModel.get_number_of_actions()), dtype=float)
    d[Q_SHARED_KEY] = Q_shared
    d[REWARDS_KEY] = np.zeros(MAX_STEPS_PER_AGENT)
    pool = Pool(processes=process_count)
    for i in xrange(process_count):
        pool.apply_async(agent_loop, args=(d, l1, l2))
    pool.close()
    pool.join()
    return np.cumsum(d[REWARDS_KEY]) / process_count


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
