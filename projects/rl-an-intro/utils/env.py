from typing import Dict, Tuple

import numpy as np

"""
Much of this helper/utility code comes from my RL class in the MSDS at UT, Austin.
"""


class EnvSpec(object):
    def __init__(self, num_states, num_actions, gamma):
        self._num_states = num_states
        self._num_actions = num_actions
        self._gamma = gamma

    @property
    def num_states(self) -> int:
        """ # of possible states """
        return self._num_states

    @property
    def num_actions(self) -> int:
        """ # of possible actions """
        return self._num_actions

    @property
    def gamma(self) -> float:
        """ discounting factor of the environment """
        return self._gamma


class Env(object):
    def __init__(self, env_spec):
        self._env_spec = env_spec

    @property
    def spec(self) -> EnvSpec:
        return self._env_spec

    def close(self):
        pass

    def reset(self) -> Tuple[int, Dict]:
        """
        reset the environment. It should be called when you want to generate a new episode
        return:
            initial state
        """
        raise NotImplementedError()

    def step(self, action: int) -> Tuple[int, int, bool, bool, Dict]:
        """
        proceed one step.
        return:
            next state, reward, done (whether it reached to a terminal state)
        """
        raise NotImplementedError()


class EnvWithModel(Env):
    @property
    def td(self) -> np.array:
        """
        Transition Dynamics
        return: a numpy array shape of [num_states,num_actions,num_states]
            TD[s,a,s'] := the probability it will result in s' when it executes action a given state s
        """
        raise NotImplementedError()

    @property
    def r(self) -> np.array:
        """
        Reward function
        return: a numpy array shape of [num_states,num_actions,num_states]
            R[s,a,s'] := reward the agent will get it experiences (s,a,s') transition.
        """
        raise NotImplementedError()
