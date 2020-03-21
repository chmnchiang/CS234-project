from .config import Config
from config import set_config
set_config(Config)

from utils import eprint
import numpy as np
import gym
import gym_grid_world
import torch
from influence import influence_theta
from qlearning import OfflineQLearning
from model.cnn import ConvModel
from sample import random_trajectories
from wrapper import ImgWrapper
import matplotlib.pyplot as plt
import torch.nn.functional as F

def convert(img):
    return np.transpose(img, (1, 2, 0))

def run(agent, env, history, delta_thetas):

    env.seed(12)
    state = env.reset()
    done = False
    total_reward = 0.0
    max_step = 200
    lamb = 10
    action_n = env.action_space.n

    for it in range(max_step):
        Q_values = F.softmax(lamb * agent.Q_values(state)).detach().cpu().numpy()
        action = np.random.choice(len(Q_values), p=Q_values)
        print(action)
        Q_grads = []

        for i in range(action_n):
            Q_grad = agent.Q_grad(state, i)
            Q_grads.append(Q_grad)

        def f(g):
            influ = np.dot(g, Q_grads[action]) - np.dot(g, np.mean(np.array(Q_grads), axis=0))
            return influ

        # influ = [(i, np.dot(g, Q_grad)) for i, g in enumerate(delta_thetas)]
        influ = [(i, f(g)) for i, g in enumerate(delta_thetas)]
        print(influ)
        influ.sort(key=lambda x: -x[1])
        print(influ[:5])
        influ = [x[0] for x in influ[:5]]

        print('state:')
        print('a =', action)
        print('')
        fig = plt.figure(figsize=(10, 4))
        fig.add_subplot(2, 5, 1)
        plt.imshow(convert(state))

        for i, idx in enumerate(influ):
            fig.add_subplot(2, 5, i+6)
            plt.imshow(convert(history[idx][0]))
            print('a =', history[idx][1])

        z = ''.join(str(x) for x in [action] + [history[idx][1] for idx in influ])
        plt.title(z)
        print(Q_grad)
        plt.show()
        # plt.clf()


        state, reward, done, _ = env.step(action)

        total_reward += reward

        if done or it >= max_step:
            break

    rewards.append(total_reward)

    return sum(rewards) / len(rewards)


def main(arg=None):

    if arg is not None:
        arg = arg.strip()

    env = ImgWrapper(gym.make('eatbullet2d-v0',
                              grid_size=(5, 5),
                              block_size=5,
                              food_n=1))
    # history = random_trajectories(env, 15000)

    # env = ImgWrapper(gym.make('eatbullet2d-v0',
                              # grid_size=(4, 4),
                              # block_size=3,
                              # food_n=1))
    history = random_trajectories(env, 4000)
    agent = OfflineQLearning(env, ConvModel)
    agent.train_with(history, 2000, evaluate=True)
    agent.save('eatbullet2d-if-2')
    grad_thetas = influence_theta(agent, history)
    run(agent, env, history, grad_thetas)

    # if arg == 'save':
        # eprint(f'# history = {len(history)}')

        # agent = OfflineQLearning(env, ConvModel)
        # agent.train_with(history, 500, evaluate=True)
        # agent.save('eatbullet2d-if-2')
    # else:
        # agent = OfflineQLearning(env, ConvModel, device='cuda')
        # agent.load('eatbullet2d-if-2')
        # grad_thetas = influence_theta(agent, history)
        # np.save('grad_thetas2', grad_thetas)

        # # grad_thetas = np.load('grad_thetas.npy')
        # run(agent, env, history, grad_thetas)



# if __name__ == '__main__':
