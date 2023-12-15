from typing import Tuple

import numpy as np

from ..utils.env import Env, EnvSpec, EnvWithModel


class GridWorldConstants:
    nA = 4
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3


class GridWorldMDP(Env):

    def __init__(self, nS: int):
        env_spec = EnvSpec(nS, GridWorldConstants.nA, 1.)

        super().__init__(env_spec)

        self.nS = nS
        self._state = None
        self.final_state = 0
        self.trans_mat = self._build_trans_mat()
        self.r_mat = self._build_r_mat()

    def _build_trans_mat(self) -> np.array:
        raise NotImplementedError("_build_trans_mat")

    def reset(self):
        self._state = np.random.choice(range(1, self.nS))
        return self._state

    def step(self, action):
        assert action in list(range(self.spec.nA)), "Invalid Action"
        assert self._state != self.final_state, "Episode has ended!"

        prev_state = self._state
        self._state = np.random.choice(self.spec.nS, p=self.trans_mat[self._state, action])
        r = self.r_mat[prev_state, action, self._state]

        return self._state, r, self._state == self.final_state

    def _build_r_mat(self) -> np.array:

        r_mat = np.zeros((self.nS, GridWorldConstants.nA, self.nS))

        # initialize all rewards to -1
        r_mat[:, :, :] = -1.

        # except the final state, which is 0
        r_mat[self.final_state, :, :] = 0.

        return r_mat


class ThreeStateGridWorldMDP(GridWorldMDP):  # MDP introduced at Fig 5.4 in Sutton Book

    nS = 3

    def __init__(self):
        super().__init__(self.nS)

    def _build_trans_mat(self) -> np.array:

        trans_mat = np.zeros((self.nS, GridWorldConstants.nA, self.nS))

        # final state
        trans_mat[self.final_state, :, self.final_state] = 1

        s = 1
        trans_mat[s, GridWorldConstants.LEFT, self.final_state] = 1
        trans_mat[s, GridWorldConstants.UP, s] = 1
        trans_mat[s, GridWorldConstants.RIGHT, s] = 1
        trans_mat[s, GridWorldConstants.DOWN, self.final_state] = 1

        s = 2
        trans_mat[s, GridWorldConstants.LEFT, s] = 1
        trans_mat[s, GridWorldConstants.UP, self.final_state] = 1
        trans_mat[s, GridWorldConstants.RIGHT, self.final_state] = 1
        trans_mat[s, GridWorldConstants.DOWN, s] = 1

        return trans_mat


class ThreeStateGridWorldMDPWithModel(ThreeStateGridWorldMDP, EnvWithModel):
    @property
    def TD(self) -> np.array:
        return self.trans_mat

    @property
    def R(self) -> np.array:
        return self.r_mat


class FifteenStateGridWorldMDP(GridWorldMDP):  # MDP introduced at Fig 5.4 in Sutton Book

    nS = 15

    def __init__(self):
        super().__init__(self.nS)

    def _build_trans_mat(self) -> np.array:

        trans_mat = np.zeros((self.nS, GridWorldConstants.nA, self.nS))

        # final state
        trans_mat[self.final_state, :, self.final_state] = 1

        '''
        +---+---+---+---+
        | F | 1 | 2 | 3 |
        +---+---+---+---+
        | 4 | 5 | 6 | 7 |
        +---+---+---+---+
        | 8 | 9 | 10| 11|
        +---+---+---+---+
        | 12| 13| 14| F |
        +---+---+---+---+
        '''

        s = 1
        trans_mat[s, GridWorldConstants.LEFT, self.final_state] = 1
        trans_mat[s, GridWorldConstants.UP, s] = 1
        trans_mat[s, GridWorldConstants.RIGHT, s + 1] = 1
        trans_mat[s, GridWorldConstants.DOWN, s + 4] = 1

        s = 2
        trans_mat[s, GridWorldConstants.LEFT, s - 1] = 1
        trans_mat[s, GridWorldConstants.UP, s] = 1
        trans_mat[s, GridWorldConstants.RIGHT, s + 1] = 1
        trans_mat[s, GridWorldConstants.DOWN, s + 4] = 1

        s = 3
        trans_mat[s, GridWorldConstants.LEFT, s - 1] = 1
        trans_mat[s, GridWorldConstants.UP, s] = 1
        trans_mat[s, GridWorldConstants.RIGHT, s] = 1
        trans_mat[s, GridWorldConstants.DOWN, s + 4] = 1

        s = 4
        trans_mat[s, GridWorldConstants.LEFT, s] = 1
        trans_mat[s, GridWorldConstants.UP, self.final_state] = 1
        trans_mat[s, GridWorldConstants.RIGHT, s + 1] = 1
        trans_mat[s, GridWorldConstants.DOWN, s + 4] = 1

        s = 5
        trans_mat[s, GridWorldConstants.LEFT, s - 1] = 1
        trans_mat[s, GridWorldConstants.UP, s - 4] = 1
        trans_mat[s, GridWorldConstants.RIGHT, s + 1] = 1
        trans_mat[s, GridWorldConstants.DOWN, s + 4] = 1

        s = 6
        trans_mat[s, GridWorldConstants.LEFT, s - 1] = 1
        trans_mat[s, GridWorldConstants.UP, s - 4] = 1
        trans_mat[s, GridWorldConstants.RIGHT, s + 1] = 1
        trans_mat[s, GridWorldConstants.DOWN, s + 4] = 1

        s = 7
        trans_mat[s, GridWorldConstants.LEFT, s - 1] = 1
        trans_mat[s, GridWorldConstants.UP, s - 4] = 1
        trans_mat[s, GridWorldConstants.RIGHT, s] = 1
        trans_mat[s, GridWorldConstants.DOWN, s + 4] = 1

        s = 8
        trans_mat[s, GridWorldConstants.LEFT, s] = 1
        trans_mat[s, GridWorldConstants.UP, s - 4] = 1
        trans_mat[s, GridWorldConstants.RIGHT, s + 1] = 1
        trans_mat[s, GridWorldConstants.DOWN, s + 4] = 1

        s = 9
        trans_mat[s, GridWorldConstants.LEFT, s - 1] = 1
        trans_mat[s, GridWorldConstants.UP, s - 4] = 1
        trans_mat[s, GridWorldConstants.RIGHT, s + 1] = 1
        trans_mat[s, GridWorldConstants.DOWN, s + 4] = 1

        s = 10
        trans_mat[s, GridWorldConstants.LEFT, s - 1] = 1
        trans_mat[s, GridWorldConstants.UP, s - 4] = 1
        trans_mat[s, GridWorldConstants.RIGHT, s + 1] = 1
        trans_mat[s, GridWorldConstants.DOWN, s + 4] = 1

        s = 11
        trans_mat[s, GridWorldConstants.LEFT, s - 1] = 1
        trans_mat[s, GridWorldConstants.UP, s - 4] = 1
        trans_mat[s, GridWorldConstants.RIGHT, s] = 1
        trans_mat[s, GridWorldConstants.DOWN, self.final_state] = 1

        s = 12
        trans_mat[s, GridWorldConstants.LEFT, s] = 1
        trans_mat[s, GridWorldConstants.UP, s - 4] = 1
        trans_mat[s, GridWorldConstants.RIGHT, s + 1] = 1
        trans_mat[s, GridWorldConstants.DOWN, s] = 1

        s = 13
        trans_mat[s, GridWorldConstants.LEFT, s - 1] = 1
        trans_mat[s, GridWorldConstants.UP, s - 4] = 1
        trans_mat[s, GridWorldConstants.RIGHT, s + 1] = 1
        trans_mat[s, GridWorldConstants.DOWN, s] = 1

        s = 14
        trans_mat[s, GridWorldConstants.LEFT, s - 1] = 1
        trans_mat[s, GridWorldConstants.UP, s - 4] = 1
        trans_mat[s, GridWorldConstants.RIGHT, self.final_state] = 1
        trans_mat[s, GridWorldConstants.DOWN, s] = 1

        return trans_mat


class FifteenStateGridWorldMDPWithModel(FifteenStateGridWorldMDP, EnvWithModel):
    @property
    def TD(self) -> np.array:
        return self.trans_mat

    @property
    def R(self) -> np.array:
        return self.r_mat
