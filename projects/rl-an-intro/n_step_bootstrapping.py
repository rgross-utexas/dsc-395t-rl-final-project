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

INFINITY = 10e10
DEBUG = False


def on_policy_n_step_td(
        env,
        pi: Policy,
        num_episodes: int,
        n: int,
        alpha: float,
        gamma: float = 1.,
        init_v: np.array = None
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

    v = init_v.copy()

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
                    g += (gamma ** n) * v[s_tau_n]

                s_tau = states[tau]
                v[s_tau] += alpha * (g - v[s_tau])

            if tau == t_terminal - 1:
                break

            t += 1

    return v


def on_policy_sarsa(
        env,
        alpha: float,
        num_episodes: int,
        n: int,
        epsilon: float = .01,
        gamma: float = 1.,
        init_q: np.array = None,
        q_star: Optional[np.array] = None,
        render: bool = False,
        render_cadence: int = 50
) -> Tuple[np.array, defaultdict, defaultdict, defaultdict, defaultdict]:
    """
    input:
        env_spec: environment spec
        alpha: learning rate
        epsilon: exploration rate
        num_episodes: number of episodes to run
        n: number of steps
        init_q: initial Q values; np array shape of [num_states,num_actions]
    ret:
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
    episode_rmse = defaultdict(float)
    episode_frames = defaultdict(List)

    for e in tqdm(range(num_episodes)):

        states = []
        actions = []
        rewards = []
        frames = []
        episode_frames[e] = frames

        t = 0
        t_terminal = np.inf

        # get the initial state
        s_t = env.reset()
        s_t = s_t[0]
        states.append(s_t)

        # choose an action
        a_t = pi.action(s_t)
        actions.append(a_t)

        # give a reward of 0 for time 0
        rewards.append(0)

        if render and (e + 1) % render_cadence == 0:
            frames.append(env.render())

        while True:

            if t < t_terminal:

                # take a step based on the action
                s_t1, r_t1, done, _, _ = env.step(a_t)
                states.append(s_t1)
                rewards.append(r_t1)
                episode_rewards[e] += r_t1

                if render and (e + 1) % render_cadence == 0:
                    frames.append(env.render())

                if done:
                    # the next time step is terminal
                    t_terminal = t + 1
                else:

                    # take an action from the next step
                    a_t = pi.action(s_t1)
                    actions.append(a_t)

            tau = t - n + 1
            if tau >= 0:

                g = 0.
                for i in range(tau + 1, min(tau + n, t_terminal) + 1):
                    g += (gamma ** (i - (tau + 1))) * rewards[i]

                if tau + n < t_terminal:
                    # get the state-action for tau + n
                    s_tau_n = states[tau + n]
                    a_tau_n = actions[tau + n]

                    g += (gamma ** n) * q[s_tau_n, a_tau_n]

                # get the state action for tau
                s_tau = states[tau]
                a_tau = actions[tau]

                # update rule
                q[s_tau, a_tau] += alpha * (g - q[s_tau, a_tau])

            # track episode length
            episode_lengths[e] = t

            if tau == t_terminal - 1:
                if DEBUG:
                    print(f'Completed episode {e+1} in {t+1} steps with a '
                          f'reward of {episode_rewards[e]}')
                break

            t += 1

        if q_star is not None:
            rmse = np.sqrt(np.power(q - q_star, 2).sum())
            episode_rmse[e] = rmse

    env.close()

    return q, episode_rewards, episode_lengths, episode_rmse, episode_frames


def off_policy_sarsa(
        env,
        alpha: float,
        num_episodes: int,
        n: int,
        tpi_epsilon: float = .0,  # default to greedy
        bpi_epsilon: float = .3,
        gamma: float = 1.,
        init_q: np.array = None,
        q_star: Optional[np.array] = None,
        render: bool = False,
        render_cadence: int = 50
) -> Tuple[np.array, defaultdict, defaultdict, defaultdict, defaultdict]:
    """
    input:
        env_spec: environment spec
        alpha: learning rate
        epsilon: exploration rate
        num_episodes: number of episodes to run
        n: number of steps
        init_q: initial Q values; np array shape of [num_states,num_actions]
    ret:
        q: $q_star$ function; numpy array shape of [num_states,num_actions]
        episode_rewards: rewards by episode
        episode_lengths: episode lengths by episode
    """

    q = init_q.copy()

    # this is the (less greedy) behavior policy
    bpi = EGreedyPolicy(q, bpi_epsilon)

    # this is the (more greedy) target policy
    tpi = EGreedyPolicy(q, tpi_epsilon)

    # track the length and rewards for each episode
    episode_lengths = defaultdict(float)
    episode_rewards = defaultdict(float)
    episode_rmse = defaultdict(float)
    episode_frames = defaultdict(List)

    for e in tqdm(range(num_episodes)):

        states = []
        actions = []
        rewards = []
        frames = []
        episode_frames[e] = frames

        t = 0
        t_terminal = np.inf

        # get the initial state
        s_t = env.reset()
        s_t = s_t[0]
        states.append(s_t)

        # choose an action
        a_t = bpi.action(s_t)
        actions.append(a_t)

        # give a reward of 0 for time 0
        rewards.append(0)

        if render and (e + 1) % render_cadence == 0:
            frames.append(env.render())

        while True:

            if t < t_terminal:

                # take a step based on the action
                s_t1, r_t1, done, _, _ = env.step(a_t)
                states.append(s_t1)
                rewards.append(r_t1)
                episode_rewards[e] += r_t1

                if render and (e + 1) % render_cadence == 0:
                    frames.append(env.render())

                if done:
                    # the next time step is terminal
                    t_terminal = t + 1
                else:

                    # take an action from the next step
                    a_t = bpi.action(s_t1)
                    actions.append(a_t)

            tau = t - n + 1
            if tau >= 0:

                rho = 1.
                for i in range(tau + 1, min(tau + n, t_terminal)):
                    s_i = states[i]
                    a_i = actions[i]
                    tpi_p = tpi.action_prob(s_i, a_i)
                    bpi_p = bpi.action_prob(s_i, a_i)
                    rho *= tpi_p / bpi_p

                g = 0.
                for i in range(tau + 1, min(tau + n, t_terminal) + 1):
                    g += (gamma ** (i - (tau + 1))) * rewards[i]

                if tau + n < t_terminal:
                    # get the state-action for tau + n
                    s_tau_n = states[tau + n]
                    a_tau_n = actions[tau + n]

                    g += (gamma ** n) * q[s_tau_n, a_tau_n]

                # get the state action for tau
                s_tau = states[tau]
                a_tau = actions[tau]

                # update rule
                q[s_tau, a_tau] += alpha * rho * (g - q[s_tau, a_tau])

            # track episode length
            episode_lengths[e] = t

            if tau == t_terminal - 1:
                if DEBUG:
                    print(f'\nCompleted episode {e+1} in {t+1} steps with a '
                          f'reward of {episode_rewards[e]}')
                break

            t += 1

        if q_star is not None:
            rmse = np.sqrt(np.power(q - q_star, 2).sum())
            episode_rmse[e] = rmse

    return q, episode_rewards, episode_lengths, episode_rmse, episode_frames


def tree_backup(
        env,
        alpha: float,
        num_episodes: int,
        num_actions: int,
        n: int,
        tpi_epsilon: float = .0,  # default to greedy
        bpi_epsilon: float = .01,
        gamma: float = 1.,
        init_q: np.array = None,
        q_star: Optional[np.array] = None
) -> Tuple[np.array, defaultdict, defaultdict, defaultdict]:
    """
    input:
        env_spec: environment spec
        alpha: learning rate
        epsilon: exploration rate
        num_episodes: number of episodes to run
        n: number of steps
        init_q: initial Q values; np array shape of [num_states,num_actions]
    ret:
        q: $q_star$ function; numpy array shape of [num_states,num_actions]
        episode_rewards: rewards by episode
        episode_lengths: episode lengths by episode
    """

    q = init_q.copy()

    # this is the (less greedy) behavior policy
    bpi = EGreedyPolicy(q, bpi_epsilon)

    # this is the (more greedy) target policy
    tpi = EGreedyPolicy(q, tpi_epsilon)

    # track the length and rewards for each episode
    episode_lengths = defaultdict(float)
    episode_rewards = defaultdict(float)
    episode_rmse = defaultdict(float)

    for e in tqdm(range(num_episodes)):

        states = []
        actions = []
        rewards = []

        t = 0
        t_terminal = np.inf

        # get the initial state
        s_t = env.reset()
        s_t = s_t[0]
        states.append(s_t)

        # take an off-policy action
        a_t = bpi.action(s_t)
        actions.append(a_t)

        # give a reward of 0 for time 0
        rewards.append(0)

        while True:

            if t < t_terminal:

                # take a step based on the action
                s_t1, r_t1, done, _, _ = env.step(a_t)
                states.append(s_t1)
                rewards.append(r_t1)
                episode_rewards[e] += r_t1

                if done:
                    # the next time step is terminal
                    t_terminal = t + 1
                else:
                    # take an off-policy action for the next state
                    a_t = bpi.action(s_t1)
                    actions.append(a_t)

            tau = t - n + 1
            if tau >= 0:

                if t + 1 >= t_terminal:
                    g = rewards[t_terminal]
                else:
                    e_sa = 0.
                    for a in range(num_actions):
                        e_sa += tpi.action_prob(s_t1, a) * q[s_t1, a]

                    g = rewards[t + 1] + gamma * e_sa

                for k in reversed(range(tau + 1, min(t, t_terminal - 1) + 1)):

                    s_k = states[k]
                    a_k = actions[k]

                    e_sa = 0.
                    for a in range(num_actions):
                        if a != a_k:
                            e_sa += tpi.action_prob(s_k, a) * q[s_k, a]
                        else:
                            e_sa += tpi.action_prob(s_k, a_k) * g

                    g += rewards[k] + gamma * e_sa

                # get the state action for time tau
                s_tau = states[tau]
                a_tau = actions[tau]

                # update rule
                q[s_tau, a_tau] += alpha * (g - q[s_tau, a_tau])

            # track episode length
            episode_lengths[e] = t

            if tau == t_terminal - 1:
                break

            t += 1

        if q_star is not None:
            rmse = np.sqrt(np.power(q - q_star, 2).sum())
            episode_rmse[e] = rmse

    return q, episode_rewards, episode_lengths, episode_rmse


# TODO: Implement n-step Q-sigma


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

    run_type_lookup = {
        'graphs': {},
        'cliff_walking': {'gym_type': 'CliffWalking-v0'},
        'frozen_lake': {'gym_type': 'FrozenLake-v1'},
    }

    parser = argparse.ArgumentParser(prog='n-step-bootstrapping')
    parser.add_argument('--run_type', choices=run_type_lookup.keys(), default='graphs')
    parser.add_argument('--algorithm',
                        choices=['on_policy_sarsa', 'off_policy_sarsa'], default='on_policy_sarsa')
    parser.add_argument('--render_mode', choices=['human', 'rgb_array'], default='rgb_array')
    parser.add_argument('--num_episodes', type=int, default=1000)
    parser.add_argument('--render_fps', type=int, default=60)
    parser.add_argument('--n', type=int, default=5)
    parser.add_argument('--epsilon', type=float, default=.2)
    parser.add_argument('--tpi_epsilon', type=float, default=.0)
    parser.add_argument('--bpi_epsilon', type=float, default=.1)
    parser.add_argument('--alpha', type=float, default=.01)
    parser.add_argument('--gamma', type=float, default=1.)
    parser.add_argument('--render_cadence', type=int, default=2500)

    args = parser.parse_args()

    if args.run_type in ['cliff_walking', 'frozen_lake']:

        run_type = args.run_type
        algorithm = args.algorithm
        gym_type = run_type_lookup[run_type]['gym_type']
        render_cadence = args.render_cadence

        print(f'Running {run_type} with {gym_type}...')

        # create a cliff walking env
        env = gym.make(gym_type, render_mode=args.render_mode)
        env.metadata['render_fps'] = args.render_fps

        epsilon = args.epsilon
        alpha = args.alpha
        num_episodes = args.num_episodes
        n = args.n
        gamma = args.gamma
        n_states = env.observation_space.n
        n_actions = env.action_space.n

        if args.algorithm == 'on_policy_sarsa':
            _, rewards, lengths, _, frames = on_policy_sarsa(env, alpha, num_episodes, n, epsilon, gamma,
                                                             init_q=np.zeros((n_states, n_actions)),
                                                             render=True,
                                                             render_cadence=render_cadence)
            data = {n: {'r': rewards, 'l': lengths}}

            filename = f'{run_type}_{algorithm}_{n}_{epsilon}_{alpha}_{gamma}'
        elif args.algorithm == 'off_policy_sarsa':
            _, rewards, lengths, _, frames = off_policy_sarsa(env=env, alpha=alpha, num_episodes=num_episodes,
                                                              n=n, gamma=gamma, tpi_epsilon=args.tpi_epsilon,
                                                              bpi_epsilon=args.bpi_epsilon,
                                                              init_q=np.zeros((n_states, n_actions)),
                                                              render=True,
                                                              render_cadence=render_cadence)
            data = {n: {'r': rewards, 'l': lengths}}
            filename = f'{run_type}_{algorithm}_{n}_{args.tpi_epsilon}_{args.bpi_epsilon}_{alpha}_{gamma}'
        else:
            raise RuntimeError('No valid algorithm selected!')

        window = 25
        render_figure(data, window, 'steps', filename)

        if args.render_mode == 'rgb_array':
            save_frames_as_video(frames, run_type=run_type, algorithm='n_step_sarsa', render_cadence=render_cadence,
                                 filename_template='{run_type}_{algorithm}_episode_{episode}_of_{total}.mp4')

    else:

        n_episodes = 10000
        n_steps = [1, 2, 3, 4, 5, 10]
        alpha = .01
        epsilons = [.01, .05, .1, .2, .3, .4, .5]
        gamma = 1.

        # create a model-free mdp to perform value prediction
        env = GeneralDeterministicGridWorldMDP(4, 4)
        num_actions = env.spec.num_actions
        num_states = env.spec.num_states

        # env = gym.make('FrozenLake-v1')
        # num_actions = env.action_space.n
        # num_states = env.observation_space.n

        init_q = np.zeros((num_states, num_actions))
        behavior_policy = RandomPolicy(num_actions)

        # n-step td

        print('\nn-step TD:')

        for n in n_steps:
            print(f'\nNumber of steps: {n}, alpha: {alpha}, number of episodes: {n_episodes}:')
            v = on_policy_n_step_td(env, behavior_policy, n_episodes, n, alpha, gamma,
                                    np.zeros(num_states))
            print(f'\n{v}')

        q_star = np.array(
            [[0., 0., 0., 0.],
             [-1., -2., -3., -3.],
             [-2., -3., -4., -4.],
             [-3., -4., -4., -3.],
             [-2., -1., -3., -3.],
             [-2., -2., -4., -4.],
             [-3., -3., -3., -3.],
             [-4., -4., -3., -2.],
             [-3., -2., -4., -4.],
             [-3., -3., -3., -3.],
             [-4., -4., -2., -2.],
             [-3., -3., -2., -1.],
             [-4., -3., -3., -4.],
             [-4., -4., -2., -3.],
             [-3., -3., -1., -2.]]
        )

        # n-step on-policy sarsa

        print('\nn-step on-policy Sarsa:')

        epsilon = .01

        data = {}
        for n in n_steps:
            print(f'\nNumber of steps: {n}, epsilon: {epsilon}, alpha: {alpha}, number of episodes: {n_episodes}:')
            q, rewards, lengths, rmse, _ = on_policy_sarsa(env, alpha, n_episodes, n, epsilon, gamma,
                                                           init_q,
                                                           q_star=q_star)
            print(f'\n{q}')

            data[n] = {'r': rewards, 'l': lengths}

        render_figure(data, 100, 'steps', 'on_policy_sarsa_by_number_of_steps')

        n = 3

        data = {}
        for epsilon in epsilons:
            print(f'\nNumber of steps: {n}, epsilon: {epsilon}, alpha: {alpha}, number of episodes: {n_episodes}:')
            q, rewards, lengths, rmse, _ = on_policy_sarsa(env, alpha, n_episodes, n, epsilon, gamma,
                                                           init_q,
                                                           q_star=q_star)
            data[epsilon] = {'r': rewards, 'l': lengths}

        render_figure(data, 100, 'epsilon', 'on_policy_sarsa_by_epsilon')

        print('\nn-step tree backup:')

        data = {}
        for n in n_steps:
            print(f'\nNumber of steps: {n}, alpha: {alpha}, '
                  f'number of episodes: {n_episodes}:')
            q, rewards, lengths, rmse = tree_backup(env, alpha, n_episodes, num_actions, n, gamma=gamma,
                                                    init_q=init_q,
                                                    q_star=q_star)
            print(f'\n{q}')

            data[n] = {'r': rewards, 'l': lengths}

        render_figure(data, 100, 'steps', 'tree_backup_by_number_of_steps')

        n = 3

        data = {}
        for bpi_epsilon in epsilons:
            print(f'\nNumber of steps: {n}, alpha: {alpha}, behavior policy epsilon: {bpi_epsilon}, '
                  f'number of episodes: {n_episodes}:')
            q, rewards, lengths, rmse = tree_backup(env, alpha, n_episodes, num_actions, n, gamma=gamma,
                                                    init_q=init_q,
                                                    bpi_epsilon=bpi_epsilon,
                                                    q_star=q_star)
            data[bpi_epsilon] = {'r': rewards, 'l': lengths}

        render_figure(data, 100, 'bpi epsilon', 'tree_backup_by_bpi_epsilon')

        # n-step off-policy sarsa

        print('\nn-step off-policy Sarsa:')

        data = {}
        for n in n_steps:
            print(f'\nNumber of steps: {n}, alpha: {alpha}, '
                  f'number of episodes: {n_episodes}:')
            q, rewards, lengths, rmse, _ = off_policy_sarsa(env, alpha, n_episodes, n, gamma=gamma,
                                                            init_q=init_q,
                                                            q_star=q_star)
            print(f'\n{q}')

            data[n] = {'r': rewards, 'l': lengths}

        render_figure(data, 100, 'steps', 'off_policy_sarsa_by_number_of_steps')

        n = 3

        # off policy sarsa has trouble with very small epsilon values for the e-greedy behavior policy,
        # so .01 was removed
        epsilons = [.05, .1, .2, .3, .4, .5]

        data = {}
        for bpi_epsilon in epsilons:
            print(f'\nNumber of steps: {n}, alpha: {alpha}, behavior policy epsilon: {bpi_epsilon}, '
                  f'number of episodes: {n_episodes}:')
            q, rewards, lengths, rmse, _ = off_policy_sarsa(env, alpha, n_episodes, n, gamma=gamma,
                                                            init_q=init_q,
                                                            bpi_epsilon=bpi_epsilon,
                                                            q_star=q_star)
            data[bpi_epsilon] = {'r': rewards, 'l': lengths}

        render_figure(data, 100, 'bpi epsilon', 'off_policy_sarsa_by_bpi_epsilon')


        epsilon = .1
        alpha = .01
        num_episodes = 1000
        window = 25

        print('Cliff Walking:')
        env = gym.make('CliffWalking-v0')

        data = {}

        _, rewards, lengths, _, _ = on_policy_sarsa(env, alpha, num_episodes, 3, epsilon, 1.,
                                                    init_q = np.zeros((env.observation_space.n,
                                                                       env.action_space.n)))

        data['on-policy-sarsa'] = {'r': rewards, 'l': lengths}

        _, rewards, lengths, _, _ = off_policy_sarsa(env, alpha, num_episodes, 3, epsilon, gamma=1.,
                                                     init_q=np.zeros((env.observation_space.n,
                                                                      env.action_space.n)))

        data['off-policy-sarsa'] = {'r': rewards, 'l': lengths}

        _, rewards, lengths, _ = tree_backup(env, alpha, num_episodes,
                                             num_actions=env.action_space.n, n=3, gamma=1.,
                                             init_q=np.zeros((env.observation_space.n,
                                                              env.action_space.n)))

        data['tree-backup'] = {'r': rewards, 'l': lengths}

        render_figure(data, window, 'algorithm', 'n_step_cliff_walking')
