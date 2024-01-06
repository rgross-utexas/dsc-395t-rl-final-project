import random
from collections import defaultdict
from typing import Dict, Iterable, Optional, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.mdp import GeneralDeterministicGridWorldMDP
from utils.policy import DoubleEGreedyPolicy, EGreedyPolicy, RandomPolicy
from utils.utils import generate_trajectories

INFINITY = 10e10


def td_0_prediction(
        trajs: Iterable[Iterable[Tuple[int, int, int, int]]],
        alpha: float,
        init_v: np.array,
        gamma: float
) -> np.array:
    """
    input:
        trajs: trajectories generated using a given policy in which each element is a tuple representing
            (s_t,a_t,r_{t+1},s_{t+1})
        alpha: learning rate
        init_v: initial V values; np array shape of [num_states]
        gamma: discount factor
    return:
        v: $v_pi$ function; numpy array shape of [num_states]
    """

    v = init_v.copy()

    for traj in trajs:
        for t, (s_t, a_t, r_t1, s_t1) in enumerate(traj):
            v[s_t] += alpha * (r_t1 + gamma * v[s_t1] - v[s_t])

    return v


def sarsa(
        env,
        alpha: float,
        epsilon: float,
        num_episodes: int,
        init_q: np.array,
        gamma: float
) -> Tuple[np.array, defaultdict, defaultdict]:
    """
    input:
        env: environment
        alpha: learning rate
        epsilon: exploration rate
        num_episodes: number of episodes to run
        init_q: initial Q values; np array shape of [num_states,num_actions]
        gamma: discount factor
    return:
        q: $q_star$ function; numpy array shape of [num_states,num_actions]
        episode_rewards: rewards by episode
        episode_lengths: episode lengths by episode
    """

    q = init_q.copy()

    # this is on-policy so create a policy backed by q
    pi = EGreedyPolicy(q, epsilon)

    # track the length and rewards for each episode
    episode_lengths = defaultdict(float)
    episode_rewards = defaultdict(float)

    for e in tqdm(range(num_episodes)):

        t = 0

        # get the initial state
        s_t = env.reset()
        s_t = s_t[0]

        # choose an action
        a_t = pi.action(s_t)

        done = False

        while not done:
            # take a step
            s_t1, r_t1, done, _, _ = env.step(a_t)

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
        env,
        alpha: float,
        epsilon: float,
        num_episodes: int,
        num_actions: int,
        init_q: np.array,
        gamma: float
) -> Tuple[np.array, defaultdict, defaultdict]:
    """
    input:
        env: environment
        alpha: learning rate
        epsilon: exploration rate
        num_episodes: number of episodes to run
        init_q: initial Q values; np array shape of [num_states,num_actions]
        gamma: discount factor
    return:
        q: $q_star$ function; numpy array shape of [num_states,num_actions]
        episode_rewards: rewards by episode
        episode_lengths: episode lengths by episode
    """

    q = init_q.copy()
    pi = EGreedyPolicy(q, epsilon)

    # track the length and rewards for each episode
    episode_lengths = defaultdict(float)
    episode_rewards = defaultdict(float)

    for e in tqdm(range(num_episodes)):

        t = 0

        # get the initial state
        s_t = env.reset()
        s_t = s_t[0]

        # choose an action
        a_t = pi.action(s_t)

        done = False

        while not done:
            # take a step
            s_t1, r_t1, done, _, _ = env.step(a_t)

            # for sarsa we want the action for the next state
            a_t1 = pi.action(s_t1)

            expected_value = 0.
            for a in range(num_actions):
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
        env,
        alpha: float,
        epsilon: float,
        num_episodes: int,
        init_q: np.array,
        gamma: float = 1.
) -> Tuple[np.array, defaultdict, defaultdict]:
    """
    input:
        env: environment
        alpha: learning rate
        epsilon: exploration rate
        num_episodes: number of episodes to run
        init_q: initial Q values; np array shape of [num_states,num_actions]
        gamma: discount factor
    return:
        q: $q_star$ function; numpy array shape of [num_states,num_actions]
        episode_rewards: rewards by episode
        episode_lengths: episode lengths by episode
    """

    q = init_q.copy()
    pi = EGreedyPolicy(q, epsilon)

    # track the length and rewards for each episode
    episode_lengths = defaultdict(float)
    episode_rewards = defaultdict(float)

    for e in tqdm(range(num_episodes)):

        t = 0

        # get the initial state
        s_t = env.reset()
        s_t = s_t[0]

        done = False

        while not done:
            # choose an action
            a_t = pi.action(s_t)

            # take a step
            s_t1, r_t1, done, _, _ = env.step(a_t)

            # update rule
            q[s_t, a_t] += alpha * (r_t1 + gamma * np.max(q[s_t1]) - q[s_t, a_t])

            # track rewards and episode lengths
            episode_rewards[e] += r_t1
            episode_lengths[e] = t

            # follow the next state and action
            s_t = s_t1

            t += 1

    return q, episode_rewards, episode_lengths


def double_q_learning(
        env,
        alpha: float,
        epsilon: float,
        num_episodes: int,
        init_q: np.array,
        gamma: float = 1.
) -> Tuple[np.array, defaultdict, defaultdict]:
    """
    input:
        env: environment
        alpha: learning rate
        epsilon: exploration rate
        num_episodes: number of episodes to run
        init_q: initial Q values; np array shape of [num_states,num_actions]
        gamma: discount factor
    return:
        q: $q_star$ function; numpy array shape of [num_states,num_actions]
        episode_rewards: rewards by episode
        episode_lengths: episode lengths by episode
    """

    q1 = init_q.copy()
    q2 = init_q.copy()
    pi = DoubleEGreedyPolicy(q1, q2, epsilon)

    # track the length and rewards for each episode
    episode_lengths = defaultdict(float)
    episode_rewards = defaultdict(float)

    for e in tqdm(range(num_episodes)):

        t = 0

        # get the initial state
        s_t = env.reset()
        s_t = s_t[0]

        # check if this is a terminal state
        done = False

        while not done:

            # choose an action
            a_t = pi.action(s_t)

            # take a step
            s_t1, r_t1, done, _, _ = env.step(a_t)

            # update rule
            if random.random() >= .5:
                q1[s_t, a_t] += alpha * (r_t1 + gamma * q2[s_t1, np.argmax(q1[s_t1])] - q1[s_t, a_t])
            else:
                q2[s_t, a_t] += alpha * (r_t1 + gamma * q1[s_t1, np.argmax(q2[s_t1])] - q2[s_t, a_t])

            # track rewards and episode lengths
            episode_rewards[e] += r_t1
            episode_lengths[e] = t

            # follow the next state and action
            s_t = s_t1

            t += 1

    return (q1 + q2) / 2, episode_rewards, episode_lengths


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


def render_figure(data: Dict, window: int, label: str, filename: str):

    # render the figure and write it to file
    fig, axes = plt.subplots(2, 1)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for n, data in data.items():

        rewards = pd.Series(data['r'].values()).rolling(window, min_periods=window).mean().to_numpy()
        lengths = pd.Series(data['l'].values()).rolling(window, min_periods=window).mean().to_numpy()

        axes[0].plot(rewards, label=f'{label}={n}')
        axes[1].plot(lengths, label=f'{label}={n}')

    axes[0].legend(loc='lower right')
    axes[0].title.set_text(f"Reward per Episode Over Time ({window} step rolling average)")
    axes[1].legend(loc='upper right')
    axes[1].title.set_text(f"Episode Length Over Time ({window} step rolling average)")

    plt.savefig(filename)


if __name__ == '__main__':

    window = 100
    alphas = [.5, .1, .05, .01, .05, .001]
    epsilons = [.001, .005, .01, .05, .1, .5]
    n_trajectories = 50000

    # create a model-free mdp to perform value prediction
    env = GeneralDeterministicGridWorldMDP(4, 4)
    behavior_policy = RandomPolicy(env.spec.num_actions)

    print('TD(0):')

    # generate trajectories from behavior policy
    trajs = generate_trajectories(env, behavior_policy, n_trajectories)

    # generate multiple value predictions using various alphas
    for alpha in alphas:
        v = td_0_prediction(trajs, alpha, np.zeros(env.spec.num_states), 1.)
        print(f'TD(0) value prediction with {alpha=}, {n_trajectories=}, v={v}')

    # sarsa

    print('Sarsa:')

    data = {}
    for epsilon in epsilons:
        q, rewards, lengths = sarsa(env,.001, epsilon, n_trajectories,
                                    np.zeros((env.spec.num_states, env.spec.num_actions)),
                                    1.)
        print(f'q={q}')

        data[epsilon] = {'r': rewards, 'l': lengths}

    render_figure(data, window, 'epsilon', 'sarsa_by_epsilon')

    # q-learning

    print('Q-Learning:')

    data = {}
    for epsilon in epsilons:
        q, rewards, lengths = q_learning(env, .001, epsilon,
                                         n_trajectories,
                                         np.zeros((env.spec.num_states, env.spec.num_actions)),
                                         .1)
        print(f'q={q}')

        data[epsilon] = {'r': rewards, 'l': lengths}

    render_figure(data, window, 'epsilon', 'q_learning_by_epsilon')

    # expected sarsa

    print('Expected Sarsa:')

    data = {}
    for epsilon in epsilons:
        q, rewards, lengths = expected_sarsa(env, .001, epsilon,
                                             n_trajectories, env.spec.num_actions,
                                             np.zeros((env.spec.num_states, env.spec.num_actions)),
                                             1.)
        print(f'q={q}')

        data[epsilon] = {'r': rewards, 'l': lengths}

    render_figure(data, window, 'epsilon', 'expected_sarsa_by_epsilon')

    # double q-learning

    print('Double Q-Learning:')

    data = {}
    for epsilon in epsilons:
        q, rewards, lengths = double_q_learning(env, .001, epsilon,
                                                n_trajectories,
                                                np.zeros((env.spec.num_states, env.spec.num_actions)),
                                                1.)
        print(f'q={q}')
        data[epsilon] = {'r': rewards, 'l': lengths}

    render_figure(data, window, 'epsilon', 'double_q_learning_by_epsilon')

    rolling_window = 25

    # create a cliff walking env

    env = gym.make('CliffWalking-v0')

    print('Cliff Walking:')

    epsilon = .1
    alpha = .5
    num_episodes = 1000
    window = 25
    reward_y_min = -100

    data = {}

    q, rewards, lengths = sarsa(env, alpha, epsilon, num_episodes,
                                np.random.normal(size=(env.observation_space.n, env.action_space.n)),
                                gamma=1.)

    data['sarsa'] = {'r': rewards, 'l': lengths}

    q, rewards, lengths = q_learning(env, alpha, epsilon, num_episodes,
                                     np.random.normal(size=(env.observation_space.n, env.action_space.n)),
                                     gamma=1.)

    data['q-learning'] = {'r': rewards, 'l': lengths}

    q, rewards, lengths = expected_sarsa(env, alpha, epsilon, num_episodes, env.action_space.n,
                                     np.random.normal(size=(env.observation_space.n, env.action_space.n)),
                                     gamma=1.)
    data['expected-sarsa'] = {'r': rewards, 'l': lengths}

    q, rewards, lengths = double_q_learning(env, alpha, epsilon, num_episodes,
                                            np.random.normal(size=(env.observation_space.n, env.action_space.n)),
                                            gamma=1.)
    data['double-q-learning'] = {'r': rewards, 'l': lengths}

    render_figure(data, window, 'algorithm', 'cliff_walking_random_init')

    data = {}

    q, rewards, lengths = sarsa(env, alpha, epsilon, num_episodes,
                                np.zeros((env.observation_space.n, env.action_space.n)),
                                gamma=1.)

    data['sarsa'] = {'r': rewards, 'l': lengths}

    q, rewards, lengths = q_learning(env, alpha, epsilon, num_episodes,
                                     np.zeros((env.observation_space.n, env.action_space.n)),
                                     gamma=1.)

    data['q-learning'] = {'r': rewards, 'l': lengths}

    q, rewards, lengths = expected_sarsa(env, alpha, epsilon, num_episodes, env.action_space.n,
                                         np.zeros((env.observation_space.n, env.action_space.n)),
                                         gamma=1.)

    data['expected-sarsa'] = {'r': rewards, 'l': lengths}

    q, rewards, lengths = double_q_learning(env, alpha, epsilon, num_episodes,
                                            np.zeros((env.observation_space.n, env.action_space.n)),
                                            gamma=1.)

    data['double-q-learning'] = {'r': rewards, 'l': lengths}

    render_figure(data, window, 'algorithm', 'cliff_walking_zero_init')
