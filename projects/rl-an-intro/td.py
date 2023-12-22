
from collections import defaultdict
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils.env import EnvSpec
from utils.mdp import GeneralDeterministicGridWorldMDP
from utils.policy import EGreedyPolicy, GreedyPolicy, Policy, RandomPolicy
from utils.utils import generate_trajectories

INFINITY = 10e10


def td_0_prediction(
        env_spec: EnvSpec,
        trajs: Iterable[Iterable[Tuple[int, int, int, int]]],
        alpha: float,
        init_v: np.array
) -> np.array:
    """
    input:
        env_spec: environment spec
        trajs: trajectories generated using a given policy in which each element is a tuple representing
            (s_t,a_t,r_{t+1},s_{t+1})
        alpha: learning rate
        init_v: initial V values; np array shape of [num_states]
    ret:
        v: $v_pi$ function; numpy array shape of [num_states]
    """

    v = init_v.copy()
    gamma = env_spec.gamma

    for traj in trajs:
        for t, (s_t, a_t, r_t1, s_t1) in enumerate(traj):
            v[s_t] += alpha * (r_t1 + gamma * v[s_t1] - v[s_t])

    return v


def on_policy_1_step_sarsa(
        env_spec: EnvSpec,
        alpha: float,
        epsilon: float,
        num_episodes: int,
        init_q: np.array
) -> Tuple[np.array ,defaultdict, defaultdict]:
    """
    input:
        env_spec: environment spec
        alpha: learning rate
        epsilon: exploration rate
        num_episodes: number of episodes to run
        init_q: initial Q values; np array shape of [nS,nA]
    ret:
        q: $q_star$ function; numpy array shape of [nS,nA]
        policy: $pi_star$; instance of policy class
    """

    q = init_q.copy()
    pi = EGreedyPolicy(q, epsilon)
    gamma = env_spec.gamma

    # track the length and rewards for each episode
    episode_lengths = defaultdict(float)
    episode_rewards = defaultdict(float)

    for e in tqdm(range(num_episodes)):

        t = 0

        # get the initial state
        s_t = env.reset()

        # choose an action
        a_t = pi.action(s_t)

        # check if this is a terminal state
        done = env.is_final(s_t)

        while not done:

            # take a step
            s_t1, r_t1, done = env.step(a_t)

            # for sarsa we want the action for the next state
            a_t1 = pi.action(s_t1)

            # update rule
            q[s_t, a_t] += alpha * (r_t1 + gamma * q[s_t1, a_t1] - q[s_t, a_t])

            # track rewards and episode lengths
            episode_rewards[e] += r_t1
            episode_lengths[e] = t

            # follow the next state and action
            s_t = s_t1
            a_t = a_t1

            t += 1

    return q, episode_rewards, episode_lengths


def off_policy_1_step_q_learning(
        env_spec: EnvSpec,
        alpha: float,
        epsilon: float,
        num_episodes: int,
        init_q: np.array
) -> Tuple[np.array ,defaultdict, defaultdict]:
    """
    input:
        env_spec: environment spec
        alpha: learning rate
        epsilon: exploration rate
        num_episodes: number of episodes to run
        init_q: initial Q values; np array shape of [nS,nA]
    ret:
        q: $q_star$ function; numpy array shape of [nS,nA]
        policy: $pi_star$; instance of policy class
    """

    q = init_q.copy()
    pi = EGreedyPolicy(q, epsilon)
    gamma = env_spec.gamma

    # track the length and rewards for each episode
    episode_lengths = defaultdict(float)
    episode_rewards = defaultdict(float)

    for e in tqdm(range(num_episodes)):

        t = 0

        # get the initial state
        s_t = env.reset()

        # check if this is a terminal state
        done = env.is_final(s_t)

        while not done:

            # choose an action
            a_t = pi.action(s_t)

            # take a step
            s_t1, r_t1, done = env.step(a_t)

            # update rule
            q[s_t, a_t] += alpha * (r_t1 + gamma * np.max(q[s_t1]) - q[s_t, a_t])

            # track rewards and episode lengths
            episode_rewards[e] += r_t1
            episode_lengths[e] = t

            # follow the next state and action
            s_t = s_t1

            t += 1

    return q, episode_rewards, episode_lengths


def render(lengths: defaultdict, rewards: defaultdict, epsilon: float, filename_template: str):

    fig, axes = plt.subplots(2, 1)

    fig.set_figwidth(12)
    fig.set_figheight(12)

    axes[0].plot(lengths.keys(), lengths.values())
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Episode Length')
    axes[0].title.set_text(f'Episode Length with epsilon = {epsilon}')

    axes[1].plot(rewards.keys(), rewards.values())
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Episode Reward')
    axes[1].title.set_text(f'Episode Reward with epsilon = {epsilon}')

    plt.savefig(filename_template.format(epsilon=epsilon).replace('.', 'p'))


if __name__ == '__main__':

    # create a model-free mdp to perform value prediction
    env = GeneralDeterministicGridWorldMDP(4, 4)
    behavior_policy = RandomPolicy(env.spec.num_actions)

    # generate trajectories from behavior policy
    n_trajectories = 100000
    trajs = generate_trajectories(env, behavior_policy, n_trajectories)

    # generate multiple value predictions using various alphas
    for alpha in [.5, .1, .05, .01, .05, .001]:
        v = td_0_prediction(env.spec, trajs, alpha, np.zeros(env.spec.num_states))
        print(f'TD(0) value prediction with {alpha=}, {n_trajectories=}, {v=}')

    sarsa

    epsilon = .5
    q, rewards, lengths = on_policy_1_step_sarsa(env.spec, .001, epsilon, 100000, np.zeros((env.spec.num_states, env.spec.num_actions)))
    print(f'{q=}')

    render(lengths, rewards, epsilon, 'on_policy_1_step_sarsa_{epsilon}')

    epsilon = .1
    q, rewards, lengths = on_policy_1_step_sarsa(env.spec, .001, epsilon, 100000, np.zeros((env.spec.num_states, env.spec.num_actions)))
    print(f'{q=}')

    render(lengths, rewards, epsilon, 'on_policy_1_step_sarsa_{epsilon}')

    epsilon = .01
    q, rewards, lengths = on_policy_1_step_sarsa(env.spec, .001, epsilon, 100000, np.zeros((env.spec.num_states, env.spec.num_actions)))
    print(f'{q=}')

    render(lengths, rewards, epsilon, 'on_policy_1_step_sarsa_{epsilon}')

    epsilon = .001
    q, rewards, lengths = on_policy_1_step_sarsa(env.spec, .001, epsilon, 100000, np.zeros((env.spec.num_states, env.spec.num_actions)))
    print(f'{q=}')

    render(lengths, rewards, epsilon, 'on_policy_1_step_sarsa_{epsilon}')

    # q-learning

    epsilon = .5
    q, rewards, lengths = off_policy_1_step_q_learning(env.spec, .001, epsilon, 100000, np.zeros((env.spec.num_states, env.spec.num_actions)))
    print(f'{q=}')

    render(lengths, rewards, epsilon, 'off_policy_1_step_q_learning_{epsilon}')

    epsilon = .1
    q, rewards, lengths = off_policy_1_step_q_learning(env.spec, .001, epsilon, 100000, np.zeros((env.spec.num_states, env.spec.num_actions)))
    print(f'{q=}')

    render(lengths, rewards, epsilon, 'off_policy_1_step_q_learning_{epsilon}')

    epsilon = .01
    q, rewards, lengths = off_policy_1_step_q_learning(env.spec, .001, epsilon, 100000, np.zeros((env.spec.num_states, env.spec.num_actions)))
    print(f'{q=}')

    render(lengths, rewards, epsilon, 'off_policy_1_step_q_learning_{epsilon}')

    epsilon = .001
    q, rewards, lengths = off_policy_1_step_q_learning(env.spec, .001, epsilon, 100000, np.zeros((env.spec.num_states, env.spec.num_actions)))
    print(f'{q=}')

    render(lengths, rewards, epsilon, 'off_policy_1_step_q_learning_{epsilon}')
