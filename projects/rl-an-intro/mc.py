from typing import Iterable, Tuple

import numpy as np
from tqdm import tqdm

from utils.env import EnvSpec
from utils.mdp import GeneralDeterministicGridWorldMDP
from utils.policy import DeterministicPolicy, Policy, RandomPolicy

DELTA = 1e-10


def first_visit_prediction(
        env_spec: EnvSpec,
        trajs: Iterable[Iterable[Tuple[int, int, int, int]]],
        v_init: np.array
) -> np.array:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        v_init: initial v_copy(s); numpy array shape of [num_states,]
    ret:
        value: optimal value function; numpy array shape of [num_states]
    """

    v_final = v_init.copy()

    # maintain a count of states
    c = np.zeros_like(v_init, dtype=float)

    for traj in trajs:

        g = 0.
        v_copy = v_final.copy()

        # we iterate backwards over the trajectory, so reverse it
        enum_traj = reversed(list(enumerate(traj)))

        # grab all the states
        states = [s for s, _, _, _ in reversed(traj)]

        for t, (s_t, a_t, r_t1, _) in enum_traj:
            g = env_spec.gamma * g + r_t1

            if s_t not in states[t:]:
                c[s_t] += 1.
                v_copy[s_t] += (g - v_final[s_t]) / c[s_t]

        v_final = v_copy

    return v_final


def every_visit_prediction(
        env_spec: EnvSpec,
        trajs: Iterable[Iterable[Tuple[int, int, int, int]]],
        v_init: np.array
) -> np.array:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        v_init: initial v_copy(s); numpy array shape of [num_states,]
    ret:
        value: optimal value function; numpy array shape of [num_states]
    """

    v_final = v_init.copy()

    # maintain a count of states
    c = np.zeros_like(v_init, dtype=float)

    for traj in trajs:

        g = 0.
        v_copy = v_final.copy()

        # we iterate backwards over the trajectory, so reverse it
        for s_t, a_t, r_t1, _ in reversed(traj):
            g = env_spec.gamma * g + r_t1
            c[s_t] += 1.
            v_copy[s_t] += (g - v_final[s_t]) / c[s_t]

        v_final = v_copy

    return v_final


def ordinary_importance_sampling_prediction(
        env_spec: EnvSpec,
        trajs: Iterable[Iterable[Tuple[int, int, int, int]]],
        bpi: Policy,
        pi: Policy,
        q_init: np.array
) -> np.array:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        pi: evaluation target policy
        q_init: initial Q values; np array shape of [num_states,num_actions]
    ret:
        q_final: $q_pi$ function; numpy array shape of [num_states,num_actions]
    """

    q_final = q_init.copy()

    # maintain a count of state-action pairs
    c = np.zeros_like(q_init, dtype=float)

    for traj in trajs:

        g = 0.
        w = 1.
        q_copy = q_final.copy()

        # we iterate backwards over the trajectory, so reverse it
        for s_t, a_t, r_t1, _ in reversed(traj):
            g = env_spec.gamma * g + r_t1
            c[s_t, a_t] += 1.
            q_copy[s_t, a_t] += w * (g - q_final[s_t, a_t]) / c[s_t, a_t]
            w = w * pi.action_prob(s_t, a_t) / bpi.action_prob(s_t, a_t)

        q_final = q_copy

    return q_final


def weighted_importance_sampling_prediction(
        env_spec: EnvSpec,
        trajs: Iterable[Iterable[Tuple[int, int, int, int]]],
        bpi: Policy,
        pi: Policy,
        q_init: np.array
) -> np.array:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using behavior policy bpi
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        pi: evaluation target policy
        q_init: initial Q values; np array shape of [num_states,num_actions]
    ret:
        q: $q_pi$ function; numpy array shape of [num_states,num_actions]
    """

    q_final = q_init.copy()

    # maintain a count of state-action pairs
    c = np.zeros_like(q_init, dtype=float)

    for traj in trajs:

        g = 0.
        w = 1.
        q_copy = q_final.copy()

        # we iterate backwards over the trajectory, so reverse it
        for s_t, a_t, r_t1, _ in reversed(traj):

            if w <= DELTA:
                break

            g = env_spec.gamma * g + r_t1
            c[s_t, a_t] += w
            q_copy[s_t, a_t] += w * (g - q_final[s_t, a_t]) / c[s_t, a_t]
            w = w * pi.action_prob(s_t, a_t) / bpi.action_prob(s_t, a_t)

        q_final = q_copy

    return q_final


if __name__ == "__main__":

    # create a model-free mdp to perform value prediction
    env = GeneralDeterministicGridWorldMDP(4, 4)
    behavior_policy = RandomPolicy(env.spec.num_actions)

    # generate trajectories from behavior policy
    num_trajectories = 100000
    trajs = []
    for _ in tqdm(range(num_trajectories)):
        states, actions, rewards, done = [env.reset()], [], [], []

        while not done:
            a = behavior_policy.action(states[-1])
            s, r, done = env.step(a)

            states.append(s)
            actions.append(a)
            rewards.append(r)

        traj = list(zip(states[:-1], actions, rewards, states[1:]))
        trajs.append(traj)

    v_fv = first_visit_prediction(env.spec, trajs, np.zeros(env.spec.num_states))
    print(f'First visit state value prediction using an equiprobable random policy with {num_trajectories} '
          f'trajectories/episodes: {v_fv}')

    v_ev = every_visit_prediction(env.spec, trajs, np.zeros(env.spec.num_states))
    print(f'Every visit state value prediction using an equiprobable random policy with {num_trajectories} '
          f'trajectories/episodes: {v_ev}')

    # on-policy evaluation test
    q_ois = ordinary_importance_sampling_prediction(env.spec, trajs, behavior_policy, behavior_policy,
                                                    np.zeros((env.spec.num_states, env.spec.num_actions)))
    print(f'On-policy action-state value prediction with ordinary importance sampling from an equiprobable random '
          f'policy (for both evaluation/target and behavior) with {num_trajectories} trajectories/episodes: {q_ois}')

    q_wis = weighted_importance_sampling_prediction(env.spec, trajs, behavior_policy, behavior_policy,
                                                    np.zeros((env.spec.num_states, env.spec.num_actions)))
    print(f'On-policy action-state value prediction with weighted importance sampling from an equiprobable random '
          f'policy (for both evaluation/target and behavior) with {num_trajectories} trajectories/episodes: {q_wis}')

    # create an optimal deterministic policy for the evaluation/target policy
    p = np.array([0, 0, 0, 0, 1, 0, 0, 3, 1, 0, 2, 3, 1, 2, 2])
    eval_policy = DeterministicPolicy(p)

    # off-policy evaluation test
    q_ois = ordinary_importance_sampling_prediction(env.spec, trajs, behavior_policy, eval_policy,
                                                    np.zeros((env.spec.num_states, env.spec.num_actions)))
    print(f'Off-policy action-state value prediction with ordinary importance sampling from an optimal '
          f'evaluation/target policy and an equiprobable random behavior policy with {num_trajectories} '
          f'trajectories/episodes: {q_ois}')

    q_wis = weighted_importance_sampling_prediction(env.spec, trajs, behavior_policy, eval_policy,
                                                    np.zeros((env.spec.num_states, env.spec.num_actions)))
    print(f'Off-policy action-state value prediction with weighted importance sampling from an optimal '
          f'evaluation/target policy and an equiprobable random behavior policy with {num_trajectories} '
          f'trajectories/episodes: {q_wis}')