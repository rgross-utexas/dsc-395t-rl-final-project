from enum import Enum
from typing import Iterable, Tuple

import numpy as np

from utils.env import EnvSpec
from utils.mdp import GeneralDeterministicGridWorldMDP
from utils.policy import DeterministicPolicy, Policy, RandomPolicy
from utils.utils import generate_trajectories

DELTA = 1e-10


class ImportanceSamplingType(Enum):
    ORDINARY = 1
    WEIGHTED = 2


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


def exploring_starts_policy_iteration(
        env_spec: EnvSpec,
        trajs: Iterable[Iterable[Tuple[int, int, int, int]]],
        q_init: np.array
) -> np.array:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        tpi: evaluation target policy
        q_init: initial Q values; np array shape of [num_states,num_actions]
    ret:
        policy: optimal deterministic policy; instance of Policy class
        q_final: $q_pi$ function; numpy array shape of [num_states,num_actions]
    """

    q_final = q_init.copy()

    # maintain a count of state-action pairs
    c = np.zeros_like(q_init, dtype=float)

    # policy that we are iterating over
    p = np.zeros(env.spec.num_states)

    for traj in trajs:

        g = 0.
        q_copy = q_final.copy()

        # we iterate backwards over the trajectory, so reverse it
        for s_t, a_t, r_t1, _ in reversed(traj):
            g = env_spec.gamma * g + r_t1
            c[s_t, a_t] += 1.
            q_copy[s_t, a_t] += (g - q_final[s_t, a_t]) / c[s_t, a_t]
            p[s_t] = np.argmax(q_copy[s_t])

        q_final = q_copy

    pi = DeterministicPolicy(p)

    return pi, q_final


def importance_sampling_prediction(
        env_spec: EnvSpec,
        trajs: Iterable[Iterable[Tuple[int, int, int, int]]],
        bpi: Policy,
        tpi: Policy,
        q_init: np.array,
        importance_sampling_type: ImportanceSamplingType
) -> np.array:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using behavior policy bpi
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        tpi: evaluation target policy
        q_init: initial Q values; np array shape of [num_states,num_actions]
        importance_sampling_type: type of importance sampling to perform (i.e., ordinary or weighted)
    ret:
        q: $q_pi$ function; numpy array shape of [num_states,num_actions]
    """

    q_final = q_init.copy()

    # maintain a count of state-action pairs
    c = np.zeros_like(q_init, dtype=float)

    for i, traj in enumerate(trajs):

        g = 0.
        w = 1.
        q_copy = q_final.copy()

        # we iterate backwards over the trajectory, so reverse it
        for s_t, a_t, r_t1, _ in reversed(traj):

            g = env_spec.gamma * g + r_t1

            match importance_sampling_type:
                case ImportanceSamplingType.ORDINARY:
                    c[s_t, a_t] += 1.
                case ImportanceSamplingType.WEIGHTED:
                    c[s_t, a_t] += w
                case _:
                    raise RuntimeError(f"Unsupported importance sampling type '{importance_sampling_type}'")

            assert c[s_t, a_t] != 0
            q_copy[s_t, a_t] += (w / c[s_t, a_t]) * (g - q_final[s_t, a_t])
            w = w * tpi.action_prob(s_t, a_t) / bpi.action_prob(s_t, a_t)

            if w <= DELTA:
                break

        q_final = q_copy

    return q_final


def off_policy_control(
        env_spec: EnvSpec,
        trajs: Iterable[Iterable[Tuple[int, int, int, int]]],
        bpi: Policy,
        q_init: np.array
) -> np.array:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        tpi: evaluation target policy
        q_init: initial Q values; np array shape of [num_states,num_actions]
    ret:
        policy: optimal deterministic policy; instance of Policy class
        q_final: $q_pi$ function; numpy array shape of [num_states,num_actions]
    """

    q_final = q_init.copy()

    # maintain a count of state-action pairs
    c = np.zeros_like(q_init, dtype=float)

    # policy that we are iterating over
    p = np.zeros(env.spec.num_states)

    for traj in trajs:

        g = 0.
        w = 1.
        q_copy = q_final.copy()

        # we iterate backwards over the trajectory, so reverse it
        for s_t, a_t, r_t1, _ in reversed(traj):
            g = env_spec.gamma * g + r_t1
            c[s_t, a_t] += w
            q_copy[s_t, a_t] += (w / c[s_t, a_t]) * (g - q_final[s_t, a_t])
            p[s_t] = np.argmax(q_copy[s_t])

            if a_t != p[s_t]:
                break

            w = w / bpi.action_prob(s_t, a_t)

        q_final = q_copy

    pi = DeterministicPolicy(p)

    return pi, q_final


if __name__ == "__main__":

    # create a model-free mdp to perform value prediction
    env = GeneralDeterministicGridWorldMDP(4, 4)
    behavior_policy = RandomPolicy(env.spec.num_actions)

    # generate trajectories from behavior policy
    n_trajectories = 100000
    trajs = generate_trajectories(env, behavior_policy, n_trajectories)

    # all the runs below use trajectories from the random behavior policy

    v_fv = first_visit_prediction(env.spec, trajs, np.zeros(env.spec.num_states))
    print(f'First visit state value prediction using an equiprobable random policy with {n_trajectories} '
          f'trajectories/episodes: {v_fv}')

    v_ev = every_visit_prediction(env.spec, trajs, np.zeros(env.spec.num_states))
    print(f'Every visit state value prediction using an equiprobable random policy with {n_trajectories} '
          f'trajectories/episodes: {v_ev}')

    p_star, q = exploring_starts_policy_iteration(env.spec, trajs,
                                                  np.zeros((env.spec.num_states, env.spec.num_actions)))

    print(f'pi*: {p_star.p.astype(int)}')
    print(f'q: {q}')

    # on-policy evaluation
    q_ois = importance_sampling_prediction(env.spec, trajs, behavior_policy, behavior_policy,
                                           np.zeros((env.spec.num_states, env.spec.num_actions)),
                                           ImportanceSamplingType.ORDINARY)
    print(f'On-policy action-state value prediction with ordinary importance sampling from an equiprobable random '
          f'policy (for both evaluation/target and behavior) with {n_trajectories} trajectories/episodes: {q_ois}')

    q_wis = importance_sampling_prediction(env.spec, trajs, behavior_policy, behavior_policy,
                                           np.zeros((env.spec.num_states, env.spec.num_actions)),
                                           ImportanceSamplingType.WEIGHTED)
    print(f'On-policy action-state value prediction with weighted importance sampling from an equiprobable random '
          f'policy (for both evaluation/target and behavior) with {n_trajectories} trajectories/episodes: {q_wis}')

    # create an optimal deterministic policy for the evaluation/target policy
    target_policy = DeterministicPolicy(np.array([0, 0, 0, 0,
                                                  1, 0, 0, 3,
                                                  1, 0, 2, 3,
                                                  1, 2, 2]))

    # off-policy evaluation
    q_ois = importance_sampling_prediction(env.spec, trajs, behavior_policy, target_policy,
                                           np.zeros((env.spec.num_states, env.spec.num_actions)),
                                           ImportanceSamplingType.ORDINARY)
    print(f'Off-policy action-state value prediction with ordinary importance sampling from an optimal '
          f'evaluation/target policy and an equiprobable random behavior policy with {n_trajectories} '
          f'trajectories/episodes: {q_ois}')

    q_wis = importance_sampling_prediction(env.spec, trajs, behavior_policy, target_policy,
                                           np.zeros((env.spec.num_states, env.spec.num_actions)),
                                           ImportanceSamplingType.WEIGHTED)
    print(f'Off-policy action-state value prediction with weighted importance sampling from an optimal '
          f'evaluation/target policy and an equiprobable random behavior policy with {n_trajectories} '
          f'trajectories/episodes: {q_wis}')

    # TODO: This does not converge. Maybe there is a bug or not enough data?
    p_star, q = off_policy_control(env.spec, trajs, behavior_policy,
                                   np.zeros((env.spec.num_states, env.spec.num_actions)))

    print(f'pi*: {p_star.p.astype(int)}')
    print(f'q: {q}')
