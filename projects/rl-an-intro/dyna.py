from collections import defaultdict
import random
from typing import Dict, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.mdp import GeneralDeterministicGridWorldMDP
from utils.policy import EGreedyPolicy


def dyna_q(
        env,
        alpha: float,
        epsilon: float,
        num_episodes: int,
        n: int,
        init_q: np.array,
        gamma: float = 1.
) -> Tuple[np.array, Dict, defaultdict, defaultdict]:
    """
    input:
        env: environment
        alpha: learning rate
        epsilon: exploration rate
        num_episodes: number of episodes to run
        n: number of planning steps
        init_q: initial Q values; np array shape of [num_states,num_actions]
        gamma: discount factor
    return:
        q: $q_star$ function; numpy array shape of [num_states,num_actions]
        model: learned model (transition dynamics and rewards)
        episode_rewards: rewards by episode
        episode_lengths: episode lengths by episode
    """

    q = init_q.copy()
    td_model = np.zeros_like(init_q, dtype=int)
    r_model = np.zeros_like(init_q, dtype=int)
    visited_states = set()
    taken_actions = defaultdict(set)
    model = {'td': td_model, 'r': r_model}
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
            visited_states.add(s_t)
            taken_actions[s_t].add(a_t)

            # take a step
            s_t1, r_t1, done, _, _ = env.step(a_t)

            # update rule
            q[s_t, a_t] += alpha * (r_t1 + gamma * np.max(q[s_t1]) - q[s_t, a_t])

            # update the model
            model['td'][s_t, a_t] = s_t1
            model['r'][s_t, a_t] = r_t1

            # now do planning, updating q based on the learned model
            for i in range(n):
                s = random.sample(list(visited_states), 1)[0]
                a = random.sample(list(taken_actions[s]), 1)[0]
                s1 = model['td'][s, a]
                r1 = model['r'][s, a]

                q[s, a] += alpha * (r1 + gamma * np.max(q[s1]) - q[s, a])

            # track rewards and episode lengths
            episode_rewards[e] += r_t1
            episode_lengths[e] = t

            # follow the next state and action
            s_t = s_t1

            t += 1

    return q, model, episode_rewards, episode_lengths


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

    env = GeneralDeterministicGridWorldMDP(4, 4)
    num_states = env.spec.num_states
    num_actions = env.spec.num_actions

    num_episodes = 10000
    epsilon = .01
    ns = [0, 5, 50]

    data = {}
    for n in ns:
        q, model, rewards, lengths = dyna_q(env=env, alpha=.001, epsilon=epsilon,
                                            num_episodes=num_episodes, n=n,
                                            init_q=np.zeros((num_states, num_actions)),
                                            gamma=1)

        data[n] = {'r': rewards, 'l': lengths}

    render_figure(data, 100, 'n', 'gridworld_dyna_q_by_n')

    env = gym.make('CliffWalking-v0')
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    data = {}
    for n in ns:
        q, model, rewards, lengths = dyna_q(env=env, alpha=.001, epsilon=epsilon,
                                            num_episodes=num_episodes, n=n,
                                            init_q=np.zeros((num_states, num_actions)),
                                            gamma=1)

        data[n] = {'r': rewards, 'l': lengths}

    render_figure(data, 100, 'n', 'cliff_walking_dyna_q_by_n')
