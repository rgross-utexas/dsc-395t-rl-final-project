"""
REINFORCE is a Monte Carlo-based policy-gradient Reinforcement Learning algorithm.
"""
import argparse
from datetime import datetime
import itertools
import logging
from tqdm import tqdm
from typing import Dict, List, Tuple

import cv2
from dynaconf import Dynaconf
import gymnasium as gym
from gymnasium import Env
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam, Optimizer
from torch.utils.tensorboard import SummaryWriter

# load the settings
settings = Dynaconf(
    settings_files=['settings.yaml', '.secrets.yaml'],
)

# set up simple logging
logging.basicConfig(filename=f'reinforce_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# for smoothing out the data that displayed data
ROLLING_AVERAGE_LENGTH = 10

# how many times to log the status during a run
NUMBER_OF_LOGS = 100

class PolicyNetwork(nn.Module):
    """
    Simple, fully connected 2-layer NeuralNetwork. The input features are dependent on the
    number of states in the space, while the output features are dependent on the
    number of actions in the space. hidden_dims is configurable and defaults to 128.
    """
    def __init__(self, input_dims: int, output_dims, hidden_dims: int = 128):
        super(PolicyNetwork, self).__init__()

        self.output_dims = output_dims

        # create the simple fully connected net
        self.net = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, output_dims)
        )

    def forward(self: nn.Module, state: np.ndarray) -> torch.Tensor:
        return F.softmax(self.net(state), dim=1)

    def get_action_log_prob(self: nn.Module, state: np.ndarray) -> Tuple[int, torch.Tensor]:

        # get the current action probability distribution for the given state
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(Variable(state))

        # sample a random action from the state-action probability distribution
        sampled_action = np.random.choice(self.output_dims, p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[sampled_action])

        return sampled_action, log_prob


# TODO: Add unit tests for this!
def discount_rewards(rewards: List[float], gamma: float, max_lookahead: int) -> torch.Tensor:

    # create the powers once outside the loop
    g_powers = np.power(gamma, np.arange(min(len(rewards) - 1, max_lookahead)))

    discounted_rewards = []
    for t in range(len(rewards)):

        g = 0

        # only lookahead max_lookahead steps
        t_lookahead = min(t + max_lookahead, len(rewards) - 1)

        # multiply the rewards by the decay and sum to get g
        g  = np.sum(rewards[t:t_lookahead] * g_powers[:t_lookahead - t])

        discounted_rewards.append(g)

    # standardize the discounted rewards, ensuring that we don't have division by 0
    discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-12)

    return torch.tensor(discounted_rewards)


def update_policy(optimizer: Optimizer, rewards: List, log_probs: List[torch.Tensor],
                  gamma: float = .99, max_lookahead: int = 100) -> None:
    """
    Updates the policy net.
    """

    # zero out the gradient
    optimizer.zero_grad()

    # discount the rewards
    discounted_rewards = discount_rewards(rewards, gamma, max_lookahead)

    # calculate the gradients
    policy_gradients = []
    for log_prob, g in zip(log_probs, discounted_rewards):
        policy_gradients.append(-log_prob * g)

    policy_gradient = torch.stack(policy_gradients).sum()
    policy_gradient.backward()

    # take a step with the optimizer
    optimizer.step()


def generate_video(env_spec_id: str, policy_net: PolicyNetwork, episode: int) -> None:

    # render episodes based on the trained policy
    env = gym.make(env_spec_id, render_mode='rgb_array')
    env.metadata['render_fps'] = 120

    # record the frames so we can create a video
    frames = []
    total_reward = 0

    # initialize/reset the environment and get it's state
    state, _ = env.reset()
    while True:

        action, _ = policy_net.get_action_log_prob(state)
        new_state, reward, terminated, truncated, _ = env.step(action)
        frames.append(env.render())
        total_reward += reward

        if terminated or truncated:
            break

        state = new_state

    np_frames = np.array(frames)
    filename = f'reinforce_{env.spec.id}_{episode + 1}_episodes_{int(total_reward)}_reward.mp4'
    logger.info(f'Generating video at {filename=}...')

    fps = 30
    height = np_frames.shape[2]
    width = np_frames.shape[1]

    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (height, width))
    for i in range(np_frames.shape[0]):
        data = np_frames[i, :, :, :]
        out.write(data)

    out.release()

# TODO: Refactor this to make it useful or remove it
def plot_data(num_steps: List[int], avg_steps: List[float],
              episode: int, env: Env, total_reward: float) -> None:

    # plot simple plot of the step data and save to file
    plt.plot(num_steps)
    plt.plot(avg_steps)
    plt.xlabel('Episode')
    plt.xlabel('Steps')
    plt.savefig(f'reinforce_{env.spec.id}_{episode + 1}_episodes_{int(total_reward)}_reward.png')
    plt.clf()


def run(**kwargs: Dict) -> None:

    env_spec_id = kwargs['env_spec_id']
    inner_dims = kwargs['inner_dims']
    num_episodes = kwargs['num_episodes']
    lr = kwargs['lr']
    gamma = kwargs['gamma']
    max_lookahead = kwargs['max_lookahead']
    num_videos = kwargs['num_videos']

    tb_id = f"tb/{env_spec_id}_{num_episodes}_{inner_dims}_{lr}_{gamma}_{max_lookahead}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tb_logger = SummaryWriter(tb_id, flush_secs=5)

    # intialize the environment that we are using
    env = gym.make(env_spec_id)

    # get number of actions from gym action space
    n_actions = env.action_space.n

    # get the number of states via the length of a state
    state, _ = env.reset()
    n_states = len(state)

    # init the pytorch pieces
    policy_net = PolicyNetwork(n_states, n_actions, inner_dims)
    optimizer = Adam(policy_net.parameters(), lr=lr)

    num_steps = []
    avg_steps = []
    rewards_sum = []

    counter = 0
    for episode in tqdm(range(num_episodes), unit='episodes'):

        # Initialize the environment and get it's state
        state, _ = env.reset()

        # init the data lists
        log_probs = []
        rewards = []

        for step in itertools.count():

            # get an action based on the current policy
            action, log_prob = policy_net.get_action_log_prob(state)
            log_probs.append(log_prob)

            # take a step with the action
            new_state, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)

            # if we have completed the run (terminated) or gone too many steps (truncated),
            # update the policy
            if terminated or truncated:

                # update the policy based on the rewards and log probabilities of the actions
                update_policy(optimizer, rewards, log_probs, gamma, max_lookahead)

                # record the step and reward data
                num_steps.append(step)
                avg_steps.append(np.mean(num_steps[-ROLLING_AVERAGE_LENGTH:]))
                rewards_sum.append(np.sum(rewards))

                # episode indexed metrics
                tb_logger.add_scalar('e_rewards_sum', rewards_sum[-1], global_step=episode)
                tb_logger.add_scalar('e_num_steps', num_steps[-1], global_step=episode)

                # log information every NUMBER_OF_LOGS episodes
                if (episode + 1) % (num_episodes//NUMBER_OF_LOGS) == 0:
                    logger.info(f"episode: {episode + 1}, reward: {np.round(np.sum(rewards), decimals=3)}, "
                                f"length: {step}, average length: {avg_steps[-1]}")

                # from time to time, generate a video
                if (episode + 1) % (num_episodes//num_videos) == 0:
                    generate_video(env.spec.id, policy_net, episode)

                break

            # counter indexed metrics
            tb_logger.add_scalar('log_prob', log_prob, global_step=counter)
            tb_logger.add_scalar('prob', torch.exp(log_prob), global_step=counter)
            tb_logger.add_scalar('reward', reward, global_step=counter)

            counter += 1

            # update the state
            state = new_state


    # plot the step data so we can see how we did
    plot_data(num_steps, avg_steps, episode, env, rewards_sum)


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(prog='reinforce')
    # currently there are no args to parse
    args = parser.parse_args()

    return args


def main():

    logger.info('Starting...')

    # run each given environment
    for env in settings.reinforce.envs:
        env_dict = dict(env)
        logger.info(f'Running {env_dict}...')
        run(**env_dict)

    logger.info('Done.')


if __name__ == '__main__':
    main()
