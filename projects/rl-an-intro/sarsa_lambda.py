from collections import defaultdict
from typing import Dict, List, Tuple

import cv2
import gymnasium as gym
from gymnasium import Env
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.utils.tensorboard as tb
from tqdm import tqdm

from utils.function_approximation import StateActionFeatureVector, StateActionFeatureVectorWithTile

TB = True


def sarsa_lambda(
        env: Env,  # openai gym environment
        gamma: float,  # discount factor
        lam: float,  # decay rate
        alpha: float,  # step size
        X: StateActionFeatureVector,
        num_episodes: int,
        render: bool = False,
        render_cadence: int = 50
) -> Tuple[np.array, Dict]:

    env_id = env.spec.id
    print(f'{env_id=}')
    id = f"tb/{env_id}"
    if TB:
        tb_logger = tb.SummaryWriter(id, flush_secs=5)

    episode_data = {}

    episode_lengths = defaultdict(float)
    episode_rewards = defaultdict(float)
    episode_frames = defaultdict(List)

    episode_data['lengths'] = episode_lengths
    episode_data['rewards'] = episode_rewards
    episode_data['frames'] = episode_frames

    def epsilon_greedy_policy(state: np.array, done: bool, w: np.array, epsilon: float = .05):

        num_actions = env.action_space.n
        if np.random.rand() < epsilon:
            return np.random.randint(num_actions)
        else:
            q = [np.dot(w, X(state, action, done)) for action in range(num_actions)]
            return np.argmax(q)

    policy_fn = epsilon_greedy_policy

    # initialize weight vector
    w = np.reshape(np.zeros((X.feature_vector_len())), [1, -1])

    for e in tqdm(range(num_episodes), ascii=True, unit='episodes'):

        frames = []
        episode_frames[e] = frames

        s = env.reset()
        a = policy_fn(state=s, done=False, w=w)
        x = X(state=s, action=a, done=False)

        z = np.zeros_like(w)
        q = 0
        t = 0

        if render and (e + 1) % render_cadence == 0:
            frames.append(env.render())

        while True:

            # take a step
            s1, r1, done, _, _ = env.step(a)

            # track the rewards for this episode
            episode_rewards[e] += r1

            if render and (e + 1) % render_cadence == 0:
                frames.append(env.render())

            # calculate new X
            a1 = policy_fn(state=s1, done=done, w=w)
            x1 = X(state=s1, action=a1, done=done)

            q0 = np.reshape(np.dot(w, x), [1, -1])[0]
            q1 = np.reshape(np.dot(w, x1), [1, -1])[0]

            # perform update
            delta = r1 + gamma * q1 - q0
            z = gamma * lam * z + (1 - alpha * gamma * lam * np.dot(z, x)) * x.T
            w = w + alpha * (delta + q0 - q) * z - alpha * (q0 - q) * x.T

            # update values for next iteration
            q = q1
            x = x1
            a = a1

            # track episode length
            episode_lengths[e] = t

            if done:
                break

            t += 1

        if TB:
            tb_logger.add_scalar('length', episode_lengths[e], global_step=e)
            tb_logger.add_scalar('reward', episode_rewards[e], global_step=e)

    return w, episode_data


def save_frames_as_video(episode_frames: defaultdict, run_type: str, algorithm: str, render_cadence: int = 50,
                         filename_template: str = './{algorithm}_episode_{episode}_of_{total}.mp4'):
    for i, frames in episode_frames.items():

        if len(frames) > 0:
            np_frames = np.array(frames)
            save_episode_as_video(algorithm, len(episode_frames), filename_template, i, np_frames, run_type)


def save_episode_as_video(algorithm, num_episodes, filename_template, episode, np_frames, run_type):
    filename = filename_template.format(run_type=run_type, algorithm=algorithm, episode=(episode + 1),
                                        total=num_episodes)
    fps = 10
    height = np_frames.shape[2]
    width = np_frames.shape[1]
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (height, width))
    for i in range(np_frames.shape[0]):
        data = np_frames[i, :, :, :]
        # data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        out.write(data)
    out.release()


def render_figure(data: Dict, window: int, label: str, filename: str):
    # render the figure and write it to file
    fig, axes = plt.subplots(2, 1)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for value, d in data.items():
        rewards = pd.Series(d['r'].values()).rolling(window, min_periods=window).mean().to_numpy()
        lengths = pd.Series(d['l'].values()).rolling(window, min_periods=window).mean().to_numpy()

        axes[0].plot(rewards, label=f'{label}={value}')
        axes[1].plot(lengths, label=f'{label}={value}')

    axes[0].legend(loc='lower right')
    axes[0].title.set_text(f"Reward per Episode Over Time ({window} step rolling average)")
    axes[1].legend(loc='upper right')
    axes[1].title.set_text(f"Episode Length Over Time ({window} step rolling average)")

    plt.savefig(filename.replace('.', 'p'))


if __name__ == '__main__':

    run_type_lookup = {
        'mountain_car': {'gym_type': 'MountainCar-v0',
                         'num_episodes': 1000,
                         'num_tile_parts': 4
                         },
        'lunar_lander': {'gym_type': 'LunarLander-v2',
                         'num_episodes': 500,
                         'num_tile_parts': 4
                         },
    }

    render_mode = None  # 'rgb_array'
    render = False
    render_cadence = 100
    num_tilings = 10
    algorithm = 'sarsa_lambda'
    filename_template = '{run_type}_{algorithm}_episode_{episode}_of_{total}.mp4'

    run_type = "mountain_car"
    run_config = run_type_lookup[run_type]
    gym_type = run_config['gym_type']
    num_episodes = run_config['num_episodes']
    num_tile_parts = run_config['num_tile_parts']

    env = gym.make(gym_type, render_mode=render_mode)
    env.metadata['render_fps'] = 120

    X = StateActionFeatureVectorWithTile(
        env.observation_space.low,
        env.observation_space.high,
        env.action_space.n,
        num_tilings=num_tilings,
        num_tile_parts=num_tile_parts
    )

    gammas = [1, .999, .995, .99]
    lambdas = [.9, .8, .7, .6, .5]

    gamma = 1.
    lam = .8

    # gammas

    data = {}
    ws = []
    for gamma in gammas:

        w, d = sarsa_lambda(env, gamma=gamma, lam=lam, alpha=0.01, X=X, num_episodes=num_episodes,
                            render=render, render_cadence=render_cadence)
        ws.append(w)

        if render_mode == 'rgb_array':
            save_frames_as_video(d['frames'], run_type=run_type, algorithm=algorithm,
                                 render_cadence=render_cadence,
                                 filename_template=filename_template)

        data[gamma] = {'r': d['rewards'], 'l': d['lengths']}


    window = 100
    render_figure(data, window, 'gamma', f'{run_type}_sarsa_lambda_with_tile_function_approximation_by_gamma')

    # render some videos from the resulting weights

    env = gym.make(gym_type, render_mode="rgb_array")
    env.metadata['render_fps'] = 120

    filename_template = '{run_type}_{algorithm}_gamma_{value}.mp4'

    def greedy_policy(_w, s, done):
        Q = [np.dot(_w, X(s, a, done)) for a in range(env.action_space.n)]
        return np.argmax(Q)


    def evaluate_w(_w, render=False):

        frames = []

        s, done = env.reset(), False
        if render:
            frames.append(env.render())

        while not done:
            a = greedy_policy(_w, s, done)
            s, r, done, _, _ = env.step(a)
            if render:
                frames.append(env.render())

        return frames


    for i, w in enumerate(ws):

        frames = evaluate_w(w, render=True)

        filename = filename_template.format(run_type = run_type, algorithm = algorithm, value = gammas[i])

        save_episode_as_video(algorithm=algorithm, num_episodes=1, filename_template=filename,
                              np_frames=np.array(frames), run_type=run_type, episode=1)

    # lambdas

    data = {}
    ws = []
    for lam in lambdas:

        w, d = sarsa_lambda(env, gamma=gamma, lam=lam, alpha=0.01, X=X, num_episodes=num_episodes,
                            render=render, render_cadence=render_cadence)
        ws.append(w)

        if render_mode == 'rgb_array':
            save_frames_as_video(d['frames'], run_type=run_type, algorithm=algorithm,
                                 render_cadence=render_cadence,
                                 filename_template=filename_template)

        data[lam] = {'r': d['rewards'], 'l': d['lengths']}


    window = 100
    render_figure(data, window, 'lambda', f'{run_type}_sarsa_lambda_with_tile_function_approximation_by_lambda')

    # render some videos from the resulting weights

    env = gym.make(gym_type, render_mode="rgb_array")
    env.metadata['render_fps'] = 120

    filename_template = '{run_type}_{algorithm}_lambda_{value}.mp4'

    def greedy_policy(_w, s, done):
        Q = [np.dot(_w, X(s, a, done)) for a in range(env.action_space.n)]
        return np.argmax(Q)


    def evaluate_w(_w, render=False):

        frames = []

        s, done = env.reset(), False
        if render:
            frames.append(env.render())

        while not done:
            a = greedy_policy(_w, s, done)
            s, r, done, _, _ = env.step(a)
            if render:
                frames.append(env.render())

        return frames


    for i, w in enumerate(ws):

        frames = evaluate_w(w, render=True)

        filename = filename_template.format(run_type = run_type, algorithm = algorithm, value = lambdas[i])

        save_episode_as_video(algorithm=algorithm, num_episodes=1, filename_template=filename,
                              np_frames=np.array(frames), run_type=run_type, episode=1)
