from .config import Config
from config import set_config
set_config(Config)

from utils import eprint
import numpy as np
import gym
from shapley import data_shapley
from qlearning import OfflineQLearning
from model.mlp import SimpleMLP
from sample import random_episodes
from wrapper import OneHotState
import random


def train_with(agent, episodes, n):

    history = sum(episodes, start=[])
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
        

def stack(images, gamma=0.85):
    shape = images[0].shape
    canvas = np.ones_like(images[0]) * 255
    n = len(images)
    
    for i, image in enumerate(images):
        # not_whites = np.argwhere(image != 255)
        canvas[image != 255] = (
            255 - ((255 - image[image != 255]) / 255) * gamma**(n-i-1) * 255
        ).astype('uint8')

    return canvas
        

def replay(episode):

    env = gym.make('CartPole-v1')
    env.reset()

    all_img = None

    def get_img(trans):
        state, _, _, _, _ = trans
        env.env.state = state
        arr = env.render('rgb_array')
        return arr

    fin = stack([get_img(t) for t in episode[::3]])
    fin = fin[100:, 150:450, :]
    from PIL import Image
    im = Image.fromarray(fin, 'RGB')
    im.save('output.png')


def main(arg=None):
    env = gym.make('CartPole-v1')
    episodes = random_episodes(env, Config.episodes_n)

    K = Config.episodes_picked

    eprint(f'# history = {sum(len(history) for history in episodes)}')
    for i, e in enumerate(episodes):
        print('len', i, len(e))

    def metric(data):
        agent = OfflineQLearning(env, SimpleMLP)
        top_reward = train_with(agent, data, 100)
        return top_reward

    result = data_shapley(episodes, metric, n_iter=Config.shapley_iter)
    result = list(enumerate(result))
    result.sort(key=lambda p: -p[1])

    sorted_indices = [idx for idx, _ in result]

    def eval_indices(indices, n_iter=10):
        data = [episodes[idx] for idx in indices]
        assert len(data) == K
        rewards = []
        for it in range(n_iter):
            rewards.append(metric(data))
        return np.mean(rewards)

    print(result)

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


