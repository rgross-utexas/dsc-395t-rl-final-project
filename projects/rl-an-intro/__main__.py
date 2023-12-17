import numpy as np

from utils.mdp import GeneralGridWorldMDP, GeneralGridWorldMDPWithModel
from utils.policy import RandomPolicy
from ch4 import dp

if __name__ == "__main__":

    env = GeneralGridWorldMDP(4, 4)
    env_with_model = GeneralGridWorldMDPWithModel(4, 4)

    # Test Value Iteration
    v_star, pi_star = dp.value_iteration(env_with_model, np.zeros(env_with_model.spec.num_states), 1e-4)
    print(f'pi_star.p: {pi_star.p}')

    eval_policy = pi_star
    behavior_policy = RandomPolicy(env.spec.num_actions)

    # Test Value Prediction
    v1, q1 = dp.value_prediction(env_with_model,eval_policy,np.zeros(env.spec.num_states),1e-4)
    print(f'V1: {v1}')
    print(f'Q1: {q1}')
    v2, q2 = dp.value_prediction(env_with_model, behavior_policy, np.zeros(env.spec.num_states), 1e-4)
    print(f'value_prediction: {v2}')
    print(f'q_prediction: {q2}')
