from collections import defaultdict
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.env import EnvSpec
from utils.mdp import GeneralDeterministicGridWorldMDP
from utils.policy import EGreedyPolicy, GreedyPolicy, Policy, RandomPolicy
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
        init_q: np.array,
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

    # this is on-policy so create a policy backed by q
    pi = EGreedyPolicy(q, epsilon)
    gamma = env_spec.gamma

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

        if q_star is not None:
            rmse = np.sqrt(np.power(q - q_star, 2).sum())
            episode_rmse[e] = rmse

    return q, episode_rewards, episode_lengths, episode_rmse


def off_policy_sarsa(
        env_spec: EnvSpec,
        alpha: float,
        num_episodes: int,
        n: int,
        bpi: Policy,
        init_q: np.array,
        q_star: Optional[np.array] = None
) -> Tuple[np.array, defaultdict, defaultdict, defaultdict]:
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
    pi = GreedyPolicy(q)
    gamma = env_spec.gamma

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
        states.append(s_t)

        # choose an action
        a_t = bpi.action(s_t)
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
                    a_t = bpi.action(s_t1)
                    actions.append(a_t)

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

        if q_star is not None:
            rmse = np.sqrt(np.power(q - q_star, 2).sum())
            episode_rmse[e] = rmse

    return q, episode_rewards, episode_lengths, episode_rmse


def tree_backup(
        env_spec: EnvSpec,
        alpha: float,
        num_episodes: int,
        n: int,
        init_q: np.array,
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

    # this is on-policy so create a policy backed by q
    pi = GreedyPolicy(q)
    gamma = env_spec.gamma

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

                    # # take a random action for the next step
                    # a_t = np.random.choice(env_spec.num_actions)

                    # take an action from the next step
                    a_t = pi.action(s_t1)

                    actions.append(a_t)

            tau = t - n + 1
            if tau >= 0:

                if t + 1 >= t_terminal:
                    g = rewards[t_terminal]
                else:
                    e_sa = 0.
                    for a in range(env_spec.num_actions):
                        e_sa += pi.action_prob(s_t1, a) * q[s_t1, a]

                    g = rewards[t + 1] + gamma * e_sa

                for k in reversed(range(tau + 1, min(t, t_terminal - 1) + 1)):

                    s_k = states[k]
                    a_k = actions[k]

                    e_sa = 0.
                    for a in range(env_spec.num_actions):
                        if a != a_k:
                            e_sa += pi.action_prob(s_k, a) * q[s_k, a]
                        else:
                            e_sa += pi.action_prob(s_k, a_k) * g

                    g += rewards[k] + gamma * e_sa

                # get the state action for tau
                s_tau = states[tau]
                a_tau = actions[tau]

                # update rule
                q[s_tau, a_tau] += alpha * (g - q[s_tau, a_tau])

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


if __name__ == '__main__':

    alpha = .01
    epsilon = .05
    n_trajectories = 10000
    n_steps = [1, 2, 3, 4, 5, 10]

    # create a model-free mdp to perform value prediction
    env = GeneralDeterministicGridWorldMDP(4, 4)
    behavior_policy = RandomPolicy(env.spec.num_actions)

    # generate trajectories from behavior policy
    trajs = generate_trajectories(env, behavior_policy, n_trajectories)

    # n-step td

    print('\nn-step TD:\n')

    for n in n_steps:
        print(f'\nNumber of steps: {n}, alpha: {alpha}, '
              f'number of episodes/trajectories: {n_trajectories}:')
        v = on_policy_n_step_td(env.spec, trajs, n, alpha, np.zeros(env.spec.num_states))
        print(f'\n{v}')

    q_star = np.array(
        [[ 0.,  0.,  0.,  0.],
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

    epsilon = .1
    n_trajectories = 50000

    for n in n_steps:
        print(f'\nNumber of steps: {n}, epsilon: {epsilon}, alpha: {alpha}, '
              f'number of episodes/trajectories: {n_trajectories}:')
        q, rewards, lengths, rmse = on_policy_sarsa(env.spec, alpha, epsilon, n_trajectories, n,
                                                    np.zeros((env.spec.num_states, env.spec.num_actions)),
                                                    q_star)
        print(f'\n{q}')
        render_rmse(rmse, f"on_policy_sarsa_rmse_{n}")

    print('\nn-step tree backup:')

    n_trajectories = 100000
    alpha = .005
    n_steps = [1, 2, 3, 4, 5, 10]

    for n in n_steps:
        print(f'\nNumber of steps: {n}, alpha: {alpha}, '
              f'number of episodes/trajectories: {n_trajectories}:')
        q, rewards, lengths, rmse = tree_backup(env.spec, alpha, n_trajectories, n,
                                                np.zeros((env.spec.num_states, env.spec.num_actions)),
                                                q_star)
        print(f'\n{q}')
        render_rmse(rmse, f"tree_backup_rmse_{n}")

    # n-step off-policy sarsa

    print('\nn-step off-policy Sarsa:\n')

    n_trajectories = 10000000
    alpha = .001
    for n in n_steps:
        print(f'\nNumber of steps: {n}, alpha: {alpha}, '
              f'number of episodes/trajectories: {n_trajectories}:')
        q, rewards, lengths, rmse = off_policy_sarsa(env.spec, alpha, n_trajectories, n, behavior_policy,
                                                     np.random.normal(size=(env.spec.num_states, env.spec.num_actions)),
                                                     q_star)
        print(f'\n{q}')
        render_rmse(rmse, f"off_policy_sarsa_rmse_{n}")
