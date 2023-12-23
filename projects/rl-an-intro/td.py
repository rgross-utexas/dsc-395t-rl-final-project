from collections import defaultdict
from typing import Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.env import EnvSpec
from utils.mdp import CliffWalkingMDP, GeneralDeterministicGridWorldMDP
from utils.policy import EGreedyPolicy, RandomPolicy
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


def sarsa(
        env_spec: EnvSpec,
        alpha: float,
        epsilon: float,
        num_episodes: int,
        init_q: np.array
) -> Tuple[np.array, defaultdict, defaultdict]:
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
        done = env.is_terminal(s_t)

        while not done:
            # take a step
            s_t1, r_t1, done = env.step(a_t)

            # for sarsa we want the action for the next state
            a_t1 = pi.action(s_t1)

            # update rule
            q[s_t, a_t] += alpha * (r_t1 + gamma * q[s_t1, a_t1] - q[s_t, a_t])

            # follow the next state and action
            s_t = s_t1
            a_t = a_t1

            # track history
            episode_rewards[e] += r_t1
            episode_lengths[e] = t

            t += 1

    return q, episode_rewards, episode_lengths


def expected_sarsa(
        env_spec: EnvSpec,
        alpha: float,
        epsilon: float,
        num_episodes: int,
        init_q: np.array
) -> Tuple[np.array, defaultdict, defaultdict]:
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
        done = env.is_terminal(s_t)

        while not done:
            # take a step
            s_t1, r_t1, done = env.step(a_t)

            # for sarsa we want the action for the next state
            a_t1 = pi.action(s_t1)

            expected_value = 0.
            for a in range(env_spec.num_actions):
                expected_value += pi.action_prob(s_t1, a) * q[s_t1, a]

            # update rule
            q[s_t, a_t] += alpha * (r_t1 + gamma * expected_value - q[s_t, a_t])

            # follow the next state and action
            s_t = s_t1
            a_t = a_t1

            # track history
            episode_rewards[e] += r_t1
            episode_lengths[e] = t

            t += 1

    return q, episode_rewards, episode_lengths


def q_learning(
        env_spec: EnvSpec,
        alpha: float,
        epsilon: float,
        num_episodes: int,
        init_q: np.array
) -> Tuple[np.array, defaultdict, defaultdict]:
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
        done = env.is_terminal(s_t)

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


def render(lengths: defaultdict, rewards: defaultdict, epsilon: float,
           filename_template: str, reward_y_min: Optional[float] = None,
           rolling_window: Optional[int] = None):

    fig, axes = plt.subplots(2, 1)

    fig.set_figwidth(12)
    fig.set_figheight(12)

    episodes = lengths.keys()
    lengths = lengths.values()
    if rolling_window is not None:
        lengths = pd.Series(lengths).rolling(rolling_window, min_periods=rolling_window).mean()

    axes[0].plot(episodes, lengths)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Episode Length')
    axes[0].title.set_text(f'Episode Length with epsilon = {epsilon}')

    episodes = rewards.keys()
    rewards = rewards.values()
    if rolling_window is not None:
        rewards = pd.Series(rewards).rolling(rolling_window, min_periods=rolling_window).mean()

    axes[1].plot(episodes, rewards)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Episode Reward')
    axes[1].title.set_text(f'Episode Reward with epsilon = {epsilon}')

    if reward_y_min:
        axes[1].set_ylim(bottom=reward_y_min, top=0)

    plt.savefig(filename_template.format(epsilon=epsilon).replace('.', 'p'))


if __name__ == '__main__':

    rolling_window = 500
    alphas = [.5, .1, .05, .01, .05, .001]
    epsilons = [.5, .1, .01, .05, .001]
    n_trajectories = 100000

    # create a model-free mdp to perform value prediction
    env = GeneralDeterministicGridWorldMDP(4, 4)
    behavior_policy = RandomPolicy(env.spec.num_actions)

    # generate trajectories from behavior policy
    trajs = generate_trajectories(env, behavior_policy, n_trajectories)

    # generate multiple value predictions using various alphas
    for alpha in alphas:
        v = td_0_prediction(env.spec, trajs, alpha, np.zeros(env.spec.num_states))
        print(f'TD(0) value prediction with {alpha=}, {n_trajectories=}, {v=}')

    # sarsa

    for epsilon in epsilons:
        q, rewards, lengths = sarsa(env.spec, .001, epsilon,
                                    n_trajectories,
                                    np.zeros((env.spec.num_states, env.spec.num_actions)))
        print(f'{q=}')
        render(lengths, rewards, epsilon, 'sarsa_{epsilon}',
               rolling_window=rolling_window)

    # q-learning

    for epsilon in epsilons:
        q, rewards, lengths = q_learning(env.spec, .001, epsilon,
                                         n_trajectories,
                                         np.zeros((env.spec.num_states, env.spec.num_actions)))
        print(f'{q=}')
        render(lengths, rewards, epsilon, 'q_learning_{epsilon}',
               rolling_window=rolling_window)

    # expected sarsa

    for epsilon in epsilons:
        q, rewards, lengths = expected_sarsa(env.spec, .001, epsilon,
                                             n_trajectories,
                                             np.zeros((env.spec.num_states, env.spec.num_actions)))
        print(f'{q=}')
        render(lengths, rewards, epsilon, 'expected_sarsa_{epsilon}',
               rolling_window=rolling_window)

    rolling_window = 25

    # create a cliff walking env

    # TODO: Refactor this to something general

    start_state = 36
    goal_state = 47
    cliff_states = list(range(37, 47))
    env = CliffWalkingMDP(12, 4, cliff_states, start_state, goal_state)

    epsilon = .1
    alpha = .5
    num_episodes = 1000
    rolling_window = 25
    reward_y_min = -100

    q, rewards, lengths = sarsa(env.spec, alpha, epsilon, num_episodes,
                                np.random.normal(size=(env.spec.num_states, env.spec.num_actions)))
    render(lengths, rewards, epsilon, 'cliff_walking_sarsa_random_init', reward_y_min, rolling_window)

    q, rewards, lengths = q_learning(env.spec, alpha, epsilon, num_episodes,
                                     np.random.normal(size=(env.spec.num_states, env.spec.num_actions)))
    render(lengths, rewards, epsilon, 'cliff_walking_q_learning_random_init', reward_y_min, rolling_window)

    q, rewards, lengths = expected_sarsa(env.spec, alpha, epsilon, num_episodes,
                                     np.random.normal(size=(env.spec.num_states, env.spec.num_actions)))
    render(lengths, rewards, epsilon, 'cliff_walking_expected_sarsa_random_init', reward_y_min, rolling_window)

    q, rewards, lengths = sarsa(env.spec, alpha, epsilon, num_episodes,
                                np.zeros((env.spec.num_states, env.spec.num_actions)))
    render(lengths, rewards, epsilon, 'cliff_walking_sarsa_zero_init', reward_y_min, rolling_window)

    q, rewards, lengths = q_learning(env.spec, alpha, epsilon, num_episodes,
                                     np.zeros((env.spec.num_states, env.spec.num_actions)))
    render(lengths, rewards, epsilon, 'cliff_walking_q_learning_zero_init', reward_y_min, rolling_window)

    q, rewards, lengths = expected_sarsa(env.spec, alpha, epsilon, num_episodes,
                                         np.zeros((env.spec.num_states, env.spec.num_actions)))
    render(lengths, rewards, epsilon, 'cliff_walking_expected_sarsa_zero_init', reward_y_min, rolling_window)
