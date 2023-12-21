from typing import Iterable, Tuple

import numpy as np

from utils.env import EnvSpec
from utils.mdp import GeneralDeterministicGridWorldMDP
from utils.policy import RandomPolicy
from utils.utils import generate_trajectories

INFINITY = 10e10


def td_0_prediction(
        env_spec: EnvSpec,
        trajs: Iterable[Iterable[Tuple[int, int, int, int]]],
        alpha: float,
        init_v: np.array
) -> Tuple[np.array]:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        alpha: learning rate
        initV: initial V values; np array shape of [nS]
    ret:
        v: $v_pi$ function; numpy array shape of [nS]
    """

    v = init_v.copy()
    gamma = env_spec.gamma

    for traj in trajs:
        for t, (s_t, a_t, r_t1, s_t1) in enumerate(traj):
            v[s_t] += alpha * (r_t1 + gamma * v[s_t1] - v[s_t])

    return v


if __name__ == '__main__':

    # create a model-free mdp to perform value prediction
    env = GeneralDeterministicGridWorldMDP(4, 4)
    behavior_policy = RandomPolicy(env.spec.num_actions)

    # generate trajectories from behavior policy
    n_trajectories = 100000
    trajs = generate_trajectories(env, behavior_policy, n_trajectories)

    alpha = .5
    v_est_td = td_0_prediction(env.spec, trajs, alpha, np.zeros(env.spec.num_states))
    print(f'td_0_prediction: {alpha=}, {n_trajectories=}, {v_est_td=}')

    alpha = .1
    v_est_td = td_0_prediction(env.spec, trajs, alpha, np.zeros(env.spec.num_states))
    print(f'td_0_prediction: {alpha=}, {n_trajectories=}, {v_est_td=}')

    alpha = .05
    v_est_td = td_0_prediction(env.spec, trajs, alpha, np.zeros(env.spec.num_states))
    print(f'td_0_prediction: {alpha=}, {n_trajectories=}, {v_est_td=}')

    alpha = .01
    v_est_td = td_0_prediction(env.spec, trajs, alpha, np.zeros(env.spec.num_states))
    print(f'td_0_prediction: {alpha=}, {n_trajectories=}, {v_est_td=}')

    alpha = .005
    v_est_td = td_0_prediction(env.spec, trajs, alpha, np.zeros(env.spec.num_states))
    print(f'td_0_prediction: {alpha=}, {n_trajectories=}, {v_est_td=}')

    alpha = .001
    v_est_td = td_0_prediction(env.spec, trajs, alpha, np.zeros(env.spec.num_states))
    print(f'td_0_prediction: {alpha=}, {n_trajectories=}, {v_est_td=}')
