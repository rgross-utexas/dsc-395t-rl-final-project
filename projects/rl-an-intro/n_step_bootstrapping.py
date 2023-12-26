from collections import defaultdict
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.env import EnvSpec
from utils.mdp import GeneralDeterministicGridWorldMDP
from utils.policy import EGreedyPolicy, Policy, RandomPolicy
from utils.utils import generate_trajectories

INFINITY = 10e10


def on_policy_n_step_td(
        env_spec: EnvSpec,
        trajs: List[List[Tuple[int, int, int, int]]],
        n: int,
        alpha: float,
        init_v: np.array
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
    gamma = env_spec.gamma

    for traj in trajs:

        # get the max t so that we can modulo by it for large number of steps
        t_max = len(traj)
        t_terminal = np.inf

        # get the terminal state from the trajectory
        _, _, _, terminal_state = traj[-1]
        t = 0
        while True:

            # get the data from the trajectory for time t
            s_t, a_t, r_t1, s_t1 = traj[t % t_max]
            if t < t_terminal:
                if s_t1 == terminal_state:
                    t_terminal = t + 1

            tau = t - n + 1
            if tau >= 0:

                g = 0.
                for i in range(tau + 1, min(tau + n, t_terminal) + 1):
                    _, _, r_i, _ = traj[i % t_max]
                    g += (gamma ** (i - (tau + 1))) * r_i

                if tau + n < t_terminal:
                    s_tau_n, _, _, _ = traj[(tau + n) % t_max]
                    g += (gamma ** n) * v[s_tau_n]

                s_tau, _, _, _ = traj[tau % t_max]
                v[s_tau] += alpha * (g - v[s_tau])

            if tau == t_terminal - 1:
                break

            t += 1

    return v


def on_policy_sarsa(
        env_spec: EnvSpec,
        alpha: float,
        epsilon: float,
        num_episodes: int,
        n: int,
        init_q: np.array
) -> Tuple[np.array, defaultdict, defaultdict]:
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
    gamma = env_spec.gamma

    # track the length and rewards for each episode
    episode_lengths = defaultdict(float)
    episode_rewards = defaultdict(float)

    for e in tqdm(range(num_episodes)):

        states = []
        actions = []
        rewards = []

        t = 0
        t_terminal = np.inf

        # get the initial state
        s_t = env.reset()
        states.append(s_t)

        # choose an action
        a_t = pi.action(s_t)
        actions.append(a_t)

        # give a reward of 0 for time 0
        rewards.append(0)

        while True:

            if t < t_terminal:

                # take a step based on the action
                s_t1, r_t1, done = env.step(a_t)
                states.append(s_t1)
                rewards.append(r_t1)

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

            if tau == t_terminal - 1:
                break

            t += 1

    return q, episode_rewards, episode_lengths


def off_policy_sarsa(
        env_spec: EnvSpec,
        alpha: float,
        epsilon: float,
        trajs: List[List[Tuple[int, int, int, int]]],
        n: int,
        bpi: Policy,
        init_q: np.array
) -> Tuple[np.array, defaultdict, defaultdict]:
    """
    input:
        env_spec: environment spec
        alpha: learning rate
        epsilon: exploration rate
        num_episodes: number of episodes to run
        n: number of steps
        bpi: behavior Policy
        init_q: initial Q values; np array shape of [num_states,num_actions]
    ret:
        q: $q_star$ function; numpy array shape of [num_states,num_actions]
        episode_rewards: rewards by episode
        episode_lengths: episode lengths by episode
    """

    q = init_q.copy()

    # this is on-policy so create a policy backed by q
    pi = EGreedyPolicy(q, epsilon)
    gamma = env_spec.gamma

    # track the length and rewards for each episode
    episode_lengths = defaultdict(float)
    episode_rewards = defaultdict(float)

    for e in tqdm(range(len(trajs))):

        traj = trajs[e]

        states = []
        actions = []
        rewards = []

        t = 0
        t_terminal = np.inf
        _, _, _, s_terminal = traj[-1]

        s_t, a_t, _, _ = traj[t]

        # get the initial state and action
        states.append(s_t)
        actions.append(a_t)

        # give a reward of 0 for time 0
        rewards.append(0)

        while True:

            # get the step data
            s_t, a_t, r_t1, s_t1 = traj[t % len(traj)]

            if t < t_terminal:

                # take a step based on the action
                states.append(s_t1)
                rewards.append(r_t1)

                if s_t1 == s_terminal:
                    # the next time step is terminal
                    t_terminal = t + 1
                else:

                    # take an action from the next step, since this is sarsa
                    _, a_t1, _, _ = traj[t + 1]
                    actions.append(a_t1)

            tau = t - n + 1
            if tau >= 0:

                rho = 1.
                for i in range(tau + 1, min(tau + n, t_terminal)):
                    s_i = states[i]
                    a_i = actions[i]
                    pi_p = pi.action_prob(s_i, a_i)
                    bpi_p = bpi.action_prob(s_i, a_i)
                    rho *= pi_p/bpi_p

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

            if tau == t_terminal - 1:
                break

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

    alpha = .01
    epsilon = .05
    n_trajectories = 10000
    n_steps = [1, 2, 3, 4, 5, 10]

    # create a model-free mdp to perform value prediction
    env = GeneralDeterministicGridWorldMDP(4, 4)
    behavior_policy = RandomPolicy(env.spec.num_actions)

    # # generate trajectories from behavior policy
    # trajs = generate_trajectories(env, behavior_policy, n_trajectories)
    #
    # # n-step td
    #
    # print('\nn-step TD:\n')
    #
    # for n in n_steps:
    #     v = on_policy_n_step_td(env.spec, trajs, n, alpha, np.zeros(env.spec.num_states))
    #     print(f'\nNumber of steps: {n}, alpha: {alpha}, '
    #           f'number of episodes/trajectories: {n_trajectories}:\n{v}')
    #
    # # n-step on-policy sarsa
    #
    # print('\nn-step on-policy Sarsa:\n')
    #
    # n_trajectories = 100000
    # for n in n_steps:
    #     q, rewards, lengths = on_policy_sarsa(env.spec, alpha, epsilon, n_trajectories, n,
    #                                           np.zeros((env.spec.num_states, env.spec.num_actions)))
    #     print(f'\nNumber of steps: {n}, epsilon: {epsilon}, alpha: {alpha}, '
    #           f'number of episodes/trajectories: {n_trajectories}:\n{q}')

    # n-step off-policy sarsa

    print('\nn-step off-policy Sarsa:\n')

    n_trajectories = 500000

    # generate trajectories from behavior policy
    trajs = generate_trajectories(env, behavior_policy, n_trajectories)

    epsilon = .1
    alpha = .005
    epsilons = [.1, .05, .01, .005, .001]
    alphas = [.1, .05, .01, .005, .001]
    for n in n_steps:
        # for alpha in alphas:
        #     for epsilon in epsilons:
        q, rewards, lengths = off_policy_sarsa(env.spec, alpha, epsilon, trajs, n, behavior_policy,
                                           np.zeros((env.spec.num_states, env.spec.num_actions)))
        print(f'\nNumber of steps: {n}, epsilon: {epsilon}, alpha: {alpha}, '
              f'number of episodes/trajectories: {n_trajectories}:\n{q}')
