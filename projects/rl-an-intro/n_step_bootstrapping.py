import random
from collections import defaultdict
from typing import List, Optional, Tuple

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
        trajs: List[List[Tuple[int, int, int, int]]],
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

        # get the max t so that we can modulo by it for large number of steps
        t_max = len(traj)
        t_terminal = np.inf

        # get the terminal state from the trajectory
        _, _, _, terminal_state = traj[-1]
        t = 0
        while True:

            # get the data from the trajectory for time t
            s_t, a_t, r_t1, s_t1 = traj[t % t_max]
            if t < t_terminal:
                if s_t1 == terminal_state:
                    t_terminal = t + 1

            tau = t - n + 1
            if tau >= 0:

                g = 0.
                for i in range(tau + 1, min(tau + n, t_terminal) + 1):
                    _, _, r_i, _ = traj[i % t_max]
                    g += (gamma ** (i - (tau + 1))) * r_i

                if tau + n < t_terminal:
                    s_tau_n, _, _, _ = traj[(tau + n) % t_max]
                    g += (gamma ** n) * v[s_tau_n]

                s_tau, _, _, _ = traj[tau % t_max]
                v[s_tau] += alpha * (g - v[s_tau])

            if tau == t_terminal - 1:
                break

            t += 1

    return v


if __name__ == '__main__':

    n_trajectories = 10000

    # create a model-free mdp to perform value prediction
    env = GeneralDeterministicGridWorldMDP(4, 4)
    behavior_policy = RandomPolicy(env.spec.num_actions)

    # generate trajectories from behavior policy
    trajs = generate_trajectories(env, behavior_policy, n_trajectories)

    n_steps_list = [1, 2, 5, 10, 20, 50, 100]
    for n_steps in n_steps_list:
        v = on_policy_n_step_td(env.spec, trajs, n_steps, .005, np.zeros(env.spec.num_states))
        print(f'Number of steps: {n_steps}, number of episodes/trajectories: {n_trajectories}, {v}')
