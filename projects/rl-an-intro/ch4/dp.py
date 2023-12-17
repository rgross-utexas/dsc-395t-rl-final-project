from typing import Tuple

import numpy as np

from ..utils.mdp import GeneralGridWorldMDP, GeneralGridWorldMDPWithModel
from ..utils.env import EnvWithModel
from ..utils.policy import Policy, DeterministicPolicy, RandomPolicy


def value_prediction(e: EnvWithModel, pi: Policy,
                     init_v: np.array, theta: float) -> Tuple[np.array, np.array]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        pi: policy
        init_v: initial v(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        v: $v_\pi$ function; numpy array shape of [nS]
        q: $q_\pi$ function; numpy array shape of [nS,nA]
    """

    v_final = init_v.copy()
    q_final = np.zeros((e.spec.num_states, e.spec.num_actions))

    while True:
        delta = 0

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


def value_iteration(env: EnvWithModel, init_v: np.array, theta: float) -> Tuple[np.array, Policy]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        init_v: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        value: optimal value function; numpy array shape of [nS]
        policy: optimal deterministic policy; instance of Policy class
    """

    #####################
    # TODO: Implement Value Iteration Algorithm (Hint: Sutton Book p.83)
    #####################

    V_final = init_v.copy()
    p = np.zeros((env.spec.num_states))

    while True:
        delta = 0

        # make a copy to update so we do a full sweep before an update
        V = V_final.copy()
        for s in range(env.spec.num_states):

            # save the current value for later
            v = V[s]
            v_opt = None
            a_opt = None
            for a in range(env.spec.num_actions):

                v_cum = 0
                for sp in range(env.spec.num_states):
                    # get all the values for s prime
                    p_sp = env.td[s, a, sp]
                    r_sp = env.r[s, a, sp]
                    v_sp = V_final[sp]

                    # accumulate the for each given s/a combination
                    v_cum += p_sp * (r_sp + env.spec.gamma * v_sp)

                # keep the max
                if v_opt is None or v_cum > v_opt:
                    v_opt = v_cum
                    a_opt = a

            V[s] = v_opt
            p[s] = a_opt

            delta = max(delta, abs(v - V[s]))

        # copy the current V to the final V
        V_final = V.copy()

        if delta < theta:
            break

    pi = DeterministicPolicy(p)

    return V, pi


if __name__ == "__main__":

    env = GeneralGridWorldMDP(4, 4)
    env_with_model = GeneralGridWorldMDPWithModel(4, 4)

    # Test Value Iteration
    v_star, pi_star = value_iteration(env_with_model, np.zeros(env_with_model.spec.num_states), 1e-4)
    print(f'pi_star.p: {pi_star.p}')

    eval_policy = pi_star
    behavior_policy = RandomPolicy(env.spec.num_actions)

    # Test Value Prediction
    v1, q1 = value_prediction(env_with_model,eval_policy,np.zeros(env.spec.num_states),1e-4)
    print(f'V1: {v1}')
    print(f'Q1: {q1}')
    v2, q2 = value_prediction(env_with_model, behavior_policy, np.zeros(env.spec.num_states), 1e-4)
    print(f'value_prediction: {v2}')
    print(f'q_prediction: {q2}')
