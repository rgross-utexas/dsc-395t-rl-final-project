import numpy as np

from ..utils.env import Env, EnvSpec, EnvWithModel


class ActionConstants(object):
    NUM_ACTIONS = 4
    LEFT_ACTION = 0
    UP_ACTION = 1
    RIGHT_ACTION = 2
    DOWN_ACTION = 3


class GridWorldMDP(Env):

    def __init__(self, num_states: int):

        env_spec = EnvSpec(num_states, ActionConstants.NUM_ACTIONS, 1.)

        super().__init__(env_spec)

        self.num_states = num_states
        self._state = None
        self.final_state = 0
        self.trans_mat = self._build_trans_mat()
        self.r_mat = self._build_r_mat()

    def reset(self):
        self._state = np.random.choice(range(1, self.num_states))
        return self._state

    def step(self, action):

        assert action in list(range(self.spec.nA)), "Invalid Action"
        assert self._state != self.final_state, "Episode has ended!"

        prev_state = self._state
        self._state = np.random.choice(self.spec.nS, p=self.trans_mat[self._state, action])
        r = self.r_mat[prev_state, action, self._state]

        return self._state, r, self._state == self.final_state

    def _build_trans_mat(self) -> np.array:
        raise NotImplementedError("_build_trans_mat")

    def _build_r_mat(self) -> np.array:

        r_mat = np.zeros((self.num_states, ActionConstants.NUM_ACTIONS, self.num_states))

        # initialize all rewards to -1
        r_mat[:, :, :] = -1.

        # except the final state, which is 0
        r_mat[self.final_state, :, :] = 0.

        return r_mat


class ThreeStateGridWorldMDP(GridWorldMDP):  # MDP introduced at Fig 5.4 in Sutton Book

    num_states = 3

    '''
    +---+---+
    | F | 1 |
    +---+---+
    | 2 | F |
    +---+---+
    '''

    def __init__(self):
        super().__init__(self.nS)

    def _build_trans_mat(self) -> np.array:

        trans_mat = np.zeros((self.num_states, ActionConstants.NUM_ACTIONS, self.num_states))

        # final state
        trans_mat[self.final_state, :, self.final_state] = 1

        # all the actions are deterministic

        s = 1
        trans_mat[s, ActionConstants.LEFT_ACTION, self.final_state] = 1
        trans_mat[s, ActionConstants.UP_ACTION, s] = 1
        trans_mat[s, ActionConstants.RIGHT_ACTION, s] = 1
        trans_mat[s, ActionConstants.DOWN_ACTION, self.final_state] = 1

        s = 2
        trans_mat[s, ActionConstants.LEFT_ACTION, s] = 1
        trans_mat[s, ActionConstants.UP_ACTION, self.final_state] = 1
        trans_mat[s, ActionConstants.RIGHT_ACTION, self.final_state] = 1
        trans_mat[s, ActionConstants.DOWN_ACTION, s] = 1

        return trans_mat


class ThreeStateGridWorldMDPWithModel(ThreeStateGridWorldMDP, EnvWithModel):
    @property
    def TD(self) -> np.array:
        return self.trans_mat

    @property
    def R(self) -> np.array:
        return self.r_mat


class FifteenStateGridWorldMDP(GridWorldMDP):  # MDP introduced at Fig 5.4 in Sutton Book

    num_states = 15

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

    def __init__(self):
        super().__init__(self.num_states)

    def _build_trans_mat(self) -> np.array:

        trans_mat = np.zeros((self.num_states, ActionConstants.NUM_ACTIONS, self.num_states))

        # final state
        trans_mat[self.final_state, :, self.final_state] = 1

        # all the actions are deterministic

        s = 1
        trans_mat[s, ActionConstants.LEFT_ACTION, self.final_state] = 1
        trans_mat[s, ActionConstants.UP_ACTION, s] = 1
        trans_mat[s, ActionConstants.RIGHT_ACTION, s + 1] = 1
        trans_mat[s, ActionConstants.DOWN_ACTION, s + 4] = 1

        s = 2
        trans_mat[s, ActionConstants.LEFT_ACTION, s - 1] = 1
        trans_mat[s, ActionConstants.UP_ACTION, s] = 1
        trans_mat[s, ActionConstants.RIGHT_ACTION, s + 1] = 1
        trans_mat[s, ActionConstants.DOWN_ACTION, s + 4] = 1

        s = 3
        trans_mat[s, ActionConstants.LEFT_ACTION, s - 1] = 1
        trans_mat[s, ActionConstants.UP_ACTION, s] = 1
        trans_mat[s, ActionConstants.RIGHT_ACTION, s] = 1
        trans_mat[s, ActionConstants.DOWN_ACTION, s + 4] = 1

        s = 4
        trans_mat[s, ActionConstants.LEFT_ACTION, s] = 1
        trans_mat[s, ActionConstants.UP_ACTION, self.final_state] = 1
        trans_mat[s, ActionConstants.RIGHT_ACTION, s + 1] = 1
        trans_mat[s, ActionConstants.DOWN_ACTION, s + 4] = 1

        s = 5
        trans_mat[s, ActionConstants.LEFT_ACTION, s - 1] = 1
        trans_mat[s, ActionConstants.UP_ACTION, s - 4] = 1
        trans_mat[s, ActionConstants.RIGHT_ACTION, s + 1] = 1
        trans_mat[s, ActionConstants.DOWN_ACTION, s + 4] = 1

        s = 6
        trans_mat[s, ActionConstants.LEFT_ACTION, s - 1] = 1
        trans_mat[s, ActionConstants.UP_ACTION, s - 4] = 1
        trans_mat[s, ActionConstants.RIGHT_ACTION, s + 1] = 1
        trans_mat[s, ActionConstants.DOWN_ACTION, s + 4] = 1

        s = 7
        trans_mat[s, ActionConstants.LEFT_ACTION, s - 1] = 1
        trans_mat[s, ActionConstants.UP_ACTION, s - 4] = 1
        trans_mat[s, ActionConstants.RIGHT_ACTION, s] = 1
        trans_mat[s, ActionConstants.DOWN_ACTION, s + 4] = 1

        s = 8
        trans_mat[s, ActionConstants.LEFT_ACTION, s] = 1
        trans_mat[s, ActionConstants.UP_ACTION, s - 4] = 1
        trans_mat[s, ActionConstants.RIGHT_ACTION, s + 1] = 1
        trans_mat[s, ActionConstants.DOWN_ACTION, s + 4] = 1

        s = 9
        trans_mat[s, ActionConstants.LEFT_ACTION, s - 1] = 1
        trans_mat[s, ActionConstants.UP_ACTION, s - 4] = 1
        trans_mat[s, ActionConstants.RIGHT_ACTION, s + 1] = 1
        trans_mat[s, ActionConstants.DOWN_ACTION, s + 4] = 1

        s = 10
        trans_mat[s, ActionConstants.LEFT_ACTION, s - 1] = 1
        trans_mat[s, ActionConstants.UP_ACTION, s - 4] = 1
        trans_mat[s, ActionConstants.RIGHT_ACTION, s + 1] = 1
        trans_mat[s, ActionConstants.DOWN_ACTION, s + 4] = 1

        s = 11
        trans_mat[s, ActionConstants.LEFT_ACTION, s - 1] = 1
        trans_mat[s, ActionConstants.UP_ACTION, s - 4] = 1
        trans_mat[s, ActionConstants.RIGHT_ACTION, s] = 1
        trans_mat[s, ActionConstants.DOWN_ACTION, self.final_state] = 1

        s = 12
        trans_mat[s, ActionConstants.LEFT_ACTION, s] = 1
        trans_mat[s, ActionConstants.UP_ACTION, s - 4] = 1
        trans_mat[s, ActionConstants.RIGHT_ACTION, s + 1] = 1
        trans_mat[s, ActionConstants.DOWN_ACTION, s] = 1

        s = 13
        trans_mat[s, ActionConstants.LEFT_ACTION, s - 1] = 1
        trans_mat[s, ActionConstants.UP_ACTION, s - 4] = 1
        trans_mat[s, ActionConstants.RIGHT_ACTION, s + 1] = 1
        trans_mat[s, ActionConstants.DOWN_ACTION, s] = 1

        s = 14
        trans_mat[s, ActionConstants.LEFT_ACTION, s - 1] = 1
        trans_mat[s, ActionConstants.UP_ACTION, s - 4] = 1
        trans_mat[s, ActionConstants.RIGHT_ACTION, self.final_state] = 1
        trans_mat[s, ActionConstants.DOWN_ACTION, s] = 1

        return trans_mat


class FifteenStateGridWorldMDPWithModel(FifteenStateGridWorldMDP, EnvWithModel):
    @property
    def TD(self) -> np.array:
        return self.trans_mat

    @property
    def R(self) -> np.array:
        return self.r_mat
