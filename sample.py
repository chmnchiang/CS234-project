import random
import numpy as np

from config import Config

def random_trajectories(env, n):

    seed = random.randrange(10**9)
    env.seed(seed)

    history = []

    while len(history) < n: 

        state = env.reset()

        for j in range(Config.max_step):
            action = np.random.randint(env.action_space.n)
            next_state, reward, done, _ = env.step(action)

            history.append((state, action, reward, done, next_state))

            if len(history) >= n or done:
                break

            state = next_state

    return history

def random_episodes(env, n):

    seed = random.randrange(10**9)
    env.seed(seed)

    episodes = []

    for it in range(n):

        state = env.reset()
        history = []

        for j in range(Config.max_step):
            action = np.random.randint(env.action_space.n)
            next_state, reward, done, _ = env.step(action)

            history.append((state, action, reward, done, next_state))

            if done:
                break

            state = next_state

        episodes.append(history)

    return episodes
