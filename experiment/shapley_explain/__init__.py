from .config import Config
from config import set_config
set_config(Config)

from utils import eprint
import numpy as np
import gym
from shapley import data_shapley
from qlearning import OfflineQLearning
from model.linear import LinearModel
from sample import random_trajectories
from wrapper import OneHotState

def unique_trajectories(env):

    history = random_trajectories(env, 2000)
    visited = set()

    uniques = []

    for trans in history:
        state, action, *_ = trans
        state = np.argmax(state)

        if (state, action) not in visited:
            uniques.append(trans)
            visited.add((state, action))

    return uniques


def main(arg=None):
    env = OneHotState(gym.make('FrozenLake-v0', is_slippery=False))
    history = unique_trajectories(env)

    eprint(f'# history = {len(history)}')

    if arg is None:
        target_state = 0
        target_action = 1
    else:
        target_action, target_state = (int(x) for x in arg.split(','))

    def metric(data):
        agent = OfflineQLearning(env, LinearModel)
        agent.train_with(data, epoch_n=50)

        state = np.zeros(env.observation_space.shape[0])
        state[target_state] = 1.
        target = agent.Q_value(state, target_action).item()

        return target

    result = data_shapley(history, metric, n_iter=100)
    result = list(enumerate(result))
    result.sort(key=lambda p: -p[1])

    for idx, r in result:
        state, action, *_ = history[idx]
        state = np.argmax(state)
        print(state, action, r)



# if __name__ == '__main__':
