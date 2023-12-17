import numpy as np


class Policy(object):
    def action_prob(self, state: int, action: int) -> float:
        """
        input:
            state, action
        return:
            \pi(a|s)
        """
        raise NotImplementedError()

    def action(self, state: int) -> int:
        """
        input:
            state
        return:
            action
        """
        raise NotImplementedError()

    @property
    def p(self) -> np.array:
        raise NotImplementedError()


class RandomPolicy(Policy):
    def __init__(self, nA, p=None):
        self._p = p if p is not None else np.array([1 / nA] * nA)

    def action_prob(self, state, action=None):
        return self.p[action]

    def action(self, state):
        return np.random.choice(len(self._p), p=self._p)

    @property
    def p(self) -> np.array:
        return self._p


class DeterministicPolicy(Policy):

    def __init__(self, p: np.array):
        self._p = p

    def action_prob(self, state, action):
        return 1 if self._p[state] == action else 0

    def action(self, state):
        return self._p[state]

    @property
    def p(self) -> np.array:
        return self._p