import random
from collections import defaultdict
from typing import Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.env import EnvSpec
from utils.mdp import CliffWalkingMDP, GeneralDeterministicGridWorldMDP
from utils.policy import DoubleEGreedyPolicy, EGreedyPolicy, RandomPolicy
from utils.utils import generate_trajectories

INFINITY = 10e10


def on_policy_n_step_td(
        env_spec: EnvSpec,
        trajs: Iterable[Iterable[Tuple[int, int, int, int]]],
        n: int,
        alpha: float,
        init_v: np.array
) -> Tuple[np.array]:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        n: how many steps?
        alpha: learning rate
        init_v: initial V values; np array shape of [nS]
    ret:
        v: $v_pi$ function; numpy array shape of [nS]
    """

    v = init_v.copy()
    gamma = env_spec.gamma

    for traj in trajs:

        t_terminal = len(traj)
        _, _, _, terminal_state = traj[-1]
        for t, (s_t, a_t, r_t1, s_t1) in enumerate(traj):

            if t < t_terminal and s_t1 == terminal_state:
                t_terminal = t + 1

            tau = t - n + 1
            if tau >= 0:

                g_upper = min(tau + n, t_terminal)
                g_lower = tau + 1

                g = 0.
                for i in range(g_lower, g_upper):
                    _, _, r_i, _ = traj[i]
                    g += (gamma ** (i - g_lower)) * r_i

                if tau + n < t_terminal:
                    s_tau_n, _, _, _ = traj[tau + n]
                    g += (gamma ** n) * v[s_tau_n]

                s_tau, _, _, _ = traj[tau]
                v[s_tau] += alpha * (g - v[s_tau])

            if tau == t_terminal - 1:
                break

    return v


if __name__ == '__main__':

    n_trajectories = 100000

    # create a model-free mdp to perform value prediction
    env = GeneralDeterministicGridWorldMDP(4, 4)
    behavior_policy = RandomPolicy(env.spec.num_actions)

    # generate trajectories from behavior policy
    trajs = generate_trajectories(env, behavior_policy, n_trajectories)

    for n_steps in [1, 2]:
        v = on_policy_n_step_td(env.spec, trajs, n_steps, .005, np.zeros(env.spec.num_states))
        print(f'{v=}')
