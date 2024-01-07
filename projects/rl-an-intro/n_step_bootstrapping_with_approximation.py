import argparse
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.mdp import GeneralDeterministicGridWorldMDP
from utils.policy import EGreedyPolicy, Policy, RandomPolicy
from utils.function_approximation import ValueFunctionWithApproximation, ValueFunctionWithTile

INFINITY = 10e10
DEBUG = False


def on_policy_n_step_td(
        env,
        pi: Policy,
        value_function: ValueFunctionWithApproximation,
        num_episodes: int,
        n: int,
        alpha: float,
        gamma: float = 1.
) -> Tuple[np.array]:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        n: number of steps
        alpha: learning rate
        init_v: initial V values; np array shape of [num_states]
    ret:
        v: $v_pi$ function; numpy array shape of [num_states]
    """

    for e in tqdm(range(num_episodes)):

        states = []
        rewards = []

        t = 0
        t_terminal = np.inf

        # get the initial state
        s_t = env.reset()
        s_t = s_t[0]
        states.append(s_t)

        # give a reward of 0 for time 0
        rewards.append(0)

        done = False

        while True:

            # get the data from the trajectory for time t
            if not done:

                s_t1, r_t1, done, _, _ = env.step(pi.action(s_t))
                states.append(s_t1)
                rewards.append(r_t1)

                if done:
                    t_terminal = t + 1

                s_t = s_t1

            tau = t - n + 1
            if tau >= 0:

                g = 0.
                for i in range(tau + 1, min(tau + n, t_terminal) + 1):
                    r_i = rewards[i]
                    g += (gamma ** (i - (tau + 1))) * r_i

                if tau + n < t_terminal:
                    s_tau_n = states[tau + n]
                    g += (gamma ** n) * value_function(s_tau_n)

                s_tau = states[tau]
                value_function.update(alpha, g, s_tau)

            if tau == t_terminal - 1:
                break

            t += 1

    return v


def render_rmse(rmse: defaultdict, filename: str):
    fig, axis = plt.subplots(1, 1)

    fig.set_figwidth(12)
    fig.set_figheight(5)

    axis.plot(rmse.keys(), rmse.values())
    axis.set_xlabel('Episode')
    axis.set_ylabel('Episode RMSE')
    axis.title.set_text(f'Episode RMSE')

    plt.savefig(filename.replace('.', 'p'))


def render(lengths: defaultdict, rewards: defaultdict, filename: str,
           reward_y_min: Optional[float] = None, length_y_max: Optional[float] = None,
           rolling_window: Optional[int] = None, description: str = ''):
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
    axes[0].title.set_text(f'Episode Length: {description}')

    if length_y_max:
        axes[0].set_ylim(top=length_y_max, bottom=0)
    else:
        axes[0].set_ylim(bottom=0)

    episodes = rewards.keys()
    rewards = rewards.values()
    if rolling_window is not None:
        rewards = pd.Series(rewards).rolling(rolling_window, min_periods=rolling_window).mean()

    axes[1].plot(episodes, rewards)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Episode Reward')
    axes[1].title.set_text(f'Episode Reward: {description}')

    if reward_y_min:
        axes[1].set_ylim(bottom=reward_y_min, top=0)
    else:
        axes[1].set_ylim(top=0)

    plt.savefig(filename.replace('.', 'p'))


def render_figure(data: Dict, window: int, label: str, filename: str):
    # render the figure and write it to file
    fig, axes = plt.subplots(2, 1)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for value, data in data.items():
        rewards = pd.Series(data['r'].values()).rolling(window, min_periods=window).mean().to_numpy()
        lengths = pd.Series(data['l'].values()).rolling(window, min_periods=window).mean().to_numpy()

        axes[0].plot(rewards, label=f'{label}={value}')
        axes[1].plot(lengths, label=f'{label}={value}')

    axes[0].legend(loc='lower right')
    axes[0].title.set_text(f"Reward per Episode Over Time ({window} step rolling average)")
    axes[1].legend(loc='upper right')
    axes[1].title.set_text(f"Episode Length Over Time ({window} step rolling average)")

    plt.savefig(filename.replace('.', 'p'))


def save_frames_as_video(episode_frames: defaultdict, run_type: str, algorithm: str, render_cadence: int = 50,
                         filename_template: str = './{algorithm}_episode_{episode}_of_{total}.mp4'):

    for i, frames in episode_frames.items():

        if len(frames) > 0:
            np_frames = np.array(frames)
            filename = filename_template.format(run_type=run_type, algorithm=algorithm, episode=i+1,
                                                total=len(episode_frames))

            fps = 10
            height = np_frames.shape[2]
            width = np_frames.shape[1]
            out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (height, width))
            for i in range(np_frames.shape[0]):
                data = np_frames[i, :, :, :]
                # data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
                out.write(data)
            out.release()


if __name__ == '__main__':

    n_episodes = 10000
    n_steps = [1, 2, 3, 4, 5, 10]
    alpha = .01
    epsilons = [.01, .05, .1, .2, .3, .4, .5]
    gamma = 1.

    env = gym.make("MountainCar-v0")

    behavior_policy = RandomPolicy(num_actions)
    value_function = ValueFunctionWithTile()

    # n-step td

    print('\nn-step TD:')

    for n in n_steps:
        print(f'\nNumber of steps: {n}, alpha: {alpha}, number of episodes: {n_episodes}:')
        v = on_policy_n_step_td(env, behavior_policy, n_episodes, n, alpha, gamma,
                                np.zeros(num_states))
        print(f'\n{v}')
