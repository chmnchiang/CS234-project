from .config import Config
from config import set_config
set_config(Config)

from utils import eprint
import numpy as np
import gym
from influence import influence_theta
from qlearning import OfflineQLearning
from model.mlp import SimpleMLP
from sample import random_trajectories
from wrapper import OneHotState
import random


def train_with(agent, history, n):

    k = 10
    test_it = [n * (i+1) // k for i in range(k)]

    now = 0
    rewards = []

    for it in test_it:
        agent.train_with(history, it - now)
        reward = agent.eval()
        rewards.append(reward)

    rewards.sort()
    top_reward = np.mean(rewards[-3:])
    return top_reward
        

def main(arg=None):

    if arg is not None:
        arg = arg.strip()

    env = gym.make('CartPole-v1')
    history = random_trajectories(env, Config.trans_n)
    K = Config.trans_picked

    agent = OfflineQLearning(env, SimpleMLP)
    agent.train_with(history, 400, evaluate=True)
    grad_thetas = influence_theta(agent, history)

    result = [(i, np.linalg.norm(d)) for i, d in enumerate(grad_thetas)]
    result.sort(key=lambda p: -p[1])
    sorted_indices = [idx for idx, _ in result]

    # states = np.array([history[i][0] for i in sorted_indices])
    # np.save('all_states', states)
    # return

    def metric(data):
        agent = OfflineQLearning(env, SimpleMLP)
        top_reward = train_with(agent, data, 100)
        return top_reward

    def eval_indices(indices, n_iter=10):
        data = [history[idx] for idx in indices]
        assert len(data) == K
        rewards = []
        for it in range(n_iter):
            rewards.append(metric(data))
        return np.mean(rewards)

    best_reward = eval_indices(sorted_indices[:K])
    print('best =', best_reward)

    worst_reward = eval_indices(sorted_indices[-K:])
    print('worst =', worst_reward)

    rewards = []
    for it in range(10):
        indices = random.sample(sorted_indices, K)
        reward = eval_indices(indices, 1)
        rewards.append(reward)

    avg_reward = np.mean(rewards)
    print('average =', avg_reward)


