import numpy as np
import gym


class OneHotState(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.n = env.observation_space.n
        self.observation_space = gym.spaces.Box(low=0., high=1., shape=(self.n,))


    def one_hot(self, i):
        ret = np.zeros(self.n, dtype=np.float)
        ret[i] = 1.0
        return ret


    def reset(self, *args, **kwargs):
        state = self.env.reset(*args, **kwargs)
        return self.one_hot(state)


    def step(self, *args, **kwargs):
        state, reward, done, info = self.env.step(*args, **kwargs)
        state = self.one_hot(state)

        return state, reward, done, info


class Grayscale(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        old_shape = self.observation_space
        print(old_shape)

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs


class ImgWrapper(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = gym.spaces.Box(
            low=0.,
            high=1.,
            shape=tuple(env.observation_space.shape[i] for i in (2, 0, 1)),
        )

    def observation(self, obs):
        return np.transpose(obs, (2, 0, 1)) / 255.
