from typing import Tuple

import numpy as np

from utils.mdp import GeneralDeterministicGridWorldMDP, GeneralDeterministicGridWorldMDPWithModel
from utils.env import EnvWithModel
from utils.policy import Policy, DeterministicPolicy, RandomPolicy


def value_prediction(e: EnvWithModel, pi: Policy,
                     v_init: np.array, theta: float) -> Tuple[np.array, np.array]:
    """
    input:
        e: environment with model information
        pi: policy
        v_init: initial v(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        predicted state values
        predicted state-action values
    """

    v_final = v_init.copy()
    q_final = np.zeros((e.spec.num_states, e.spec.num_actions))

    while True:
        delta = 0

        # TODO: try to remove the loops in favor of vectorization

        # make a copy to update, so we do a full sweep before an update
        v_copy = v_final.copy()
        q_copy = q_final.copy()
        for s in range(e.spec.num_states):

            # save the current value for later
            v = v_copy[s]
            v_cum = 0

            for a in range(e.spec.num_actions):

                q_cum = 0

                # get the s/a probability from the policy
                pi_a = pi.action_prob(s, a)

                for sp in range(e.spec.num_states):

                    # get all the values for s prime
                    p_sp = e.td[s, a, sp]
                    r_sp = e.r[s, a, sp]
                    v_sp = v_final[sp]

                    # accumulate the for each given s/a combination
                    q_cum += p_sp * (r_sp + e.spec.gamma * v_sp)

                q_copy[s, a] = q_cum
                v_cum += pi_a * q_cum

            v_copy[s] = v_cum

            delta = max(delta, abs(v - v_copy[s]))

        # copy the current to the final
        q_final = q_copy.copy()
        v_final = v_copy.copy()

        if delta < theta:
            break

    return v_final, q_final


def value_iteration(e: EnvWithModel, v_init: np.array, theta: float) -> Tuple[np.array, Policy]:
    """
    input:
        e: environment with model information
        v_init: initial v_copy(s); numpy array shape of [num_states,]
        theta: exit criteria
    return:
        value: optimal value function; numpy array shape of [num_states]
        policy: optimal deterministic policy; instance of Policy class
    """

    v_final = v_init.copy()
    p = np.zeros_like(v_final)

    while True:
        delta = 0

        # TODO: try to remove the loops in favor of vectorization

        # make a copy to update, so we do a full sweep before an update
        v_copy = v_final.copy()
        for s in range(e.spec.num_states):

            # save the current value for later
            v = v_copy[s]
            v_opt = None
            a_opt = None
            for a in range(e.spec.num_actions):

                v_cum = 0
                for sp in range(e.spec.num_states):

                    # get all the values for s prime
                    p_sp = e.td[s, a, sp]
                    r_sp = e.r[s, a, sp]
                    v_sp = v_final[sp]

                    # accumulate the for each given s/a combination
                    v_cum += p_sp * (r_sp + e.spec.gamma * v_sp)

                # keep the max
                if v_opt is None or v_cum > v_opt:
                    v_opt = v_cum
                    a_opt = a

            v_copy[s] = v_opt
            p[s] = a_opt

            delta = max(delta, abs(v - v_copy[s]))

        # copy the current v_copy to the final v_copy
        v_final = v_copy.copy()

        if delta < theta:
            break

    pi = DeterministicPolicy(p)

    return v_copy, pi


if __name__ == "__main__":

    # create a model-based mdp to perform value iteration
    env_with_model = GeneralDeterministicGridWorldMDPWithModel(4, 4)

    # test value iteration
    v_star, pi_star = value_iteration(env_with_model, np.zeros(env_with_model.spec.num_states), 1e-4)
    print(f'pi*: {pi_star.p.astype(int)}')

    # create a model-free mdp to perform value prediction
    env = GeneralDeterministicGridWorldMDP(4, 4)

    eval_policy = pi_star

    # first do value prediction using the optimal policy from the previous value iteration
    v, q = value_prediction(env_with_model, eval_policy, np.zeros(env.spec.num_states), 1e-4)
    print(f'State value prediction from optimal policy: {v}')
    print(f'Action-state value prediction from the optimal policy: {q}')

    behavior_policy = RandomPolicy(env.spec.num_actions)

    # now do value prediction using an equiprobable random policy
    v, q = value_prediction(env_with_model, behavior_policy, np.zeros(env.spec.num_states), 1e-4)
    print(f'State value prediction from an equiprobable random policy: {v}')
    print(f'Action-state value prediction from an equiprobable random policy: {q}')
