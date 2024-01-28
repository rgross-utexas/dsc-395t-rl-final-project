import itertools
from typing import List

import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, Optimizer

ROLLING_AVERAGE_LENGTH = 10
NUMBER_OF_LOGS = 100
NUMBER_OF_VIDEOS = 10


class PolicyNetwork(nn.Module):
    def __init__(self, input_dims: int, output_dims, hidden_dims: int = 128):
        super(PolicyNetwork, self).__init__()

        self.output_dims = output_dims

        self.net = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, output_dims)
        )

    def forward(self, state):
        return F.softmax(self.net(state), dim=1)

    def get_action(self, state):

        # get the current state-action probability distribution for the given state
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(Variable(state))

        # sample an action from the state-action probability distribution
        sampled_action = np.random.choice(self.output_dims, p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[sampled_action])

        return sampled_action, log_prob


def discount_rewards(rewards: List[float], gamma: float, max_lookahead: int) -> torch.Tensor:

    discounted_rewards = []
    for t in range(len(rewards)):
        g = 0

        # only lookahead 10 steps maximum
        t_lookahead = min(t + max_lookahead, len(rewards) - 1)
        for power, r in enumerate(rewards[t:t_lookahead]):
            g += gamma ** power * r

        discounted_rewards.append(g)

    # normalize the discounted rewards
    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
            discounted_rewards.std() + 1e-12)

    return discounted_rewards


def update_policy(optimizer: Optimizer, rewards: List, log_probs: List[torch.Tensor],
                  gamma: float = 1, max_lookahead: int = 1e6):

    # zero out the gradient
    optimizer.zero_grad()

    # discount the rewards
    discounted_rewards = discount_rewards(rewards, gamma, max_lookahead)

    # calculate the gradients
    policy_gradient = []
    for log_prob, g in zip(log_probs, discounted_rewards):
        policy_gradient.append(-log_prob * g)

    policy_gradient = torch.stack(policy_gradient).sum()
    policy_gradient.backward()
    optimizer.step()


def generate_video(id: str, policy_net: PolicyNetwork, episode: int) -> None:

    # render episodes based on the trained policy
    env = gym.make(id, render_mode='rgb_array')
    env.metadata['render_fps'] = 120

    print(f'Running episode {episode + 1}...')

    # record the frames so we can create a video
    frames = []
    total_reward = 0

    # initialize/reset the environment and get it's state
    state, _ = env.reset()
    while True:

        action, log_prob = policy_net.get_action(state)
        new_state, reward, terminated, truncated, _ = env.step(action)
        frames.append(env.render())
        total_reward += reward

        if terminated or truncated:
            break

        state = new_state

    print('Generating video...')

    np_frames = np.array(frames)
    filename = f'reinforce_{env.spec.id}_{episode + 1}_episodes_{int(total_reward)}_reward.mp4'
    fps = 30
    height = np_frames.shape[2]
    width = np_frames.shape[1]
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (height, width))
    for i in range(np_frames.shape[0]):
        data = np_frames[i, :, :, :]
        out.write(data)
    out.release()


def main():

    env_spec_id, num_episodes, inner_dims = 'LunarLander-v2', 5000, 256
    # env_spec_id = 'MountainCar-v0'
    # env_spec_id, num_episodes, inner_dims = 'Acrobot-v1', 2000, 128
    # env_spec_id, num_episodes, inner_dims = 'CartPole-v1', 2000, 128

    print(f'Running {env_spec_id}...')

    env = gym.make(env_spec_id)

    # get number of actions from gym action space
    n_actions = env.action_space.n

    # get the number of states via the length of a state
    state, _ = env.reset()
    n_states = len(state)

    policy_net = PolicyNetwork(n_states, n_actions, inner_dims)
    optimizer = Adam(policy_net.parameters(), lr=3e-4)

    num_steps = []
    avg_steps = []
    all_rewards = []

    for episode in range(num_episodes):

        # Initialize the environment and get it's state
        state, _ = env.reset()

        log_probs = []
        rewards = []

        for step in itertools.count():

            # get an action based on the current policy
            action, log_prob = policy_net.get_action(state)
            log_probs.append(log_prob)

            # take a step with the action
            new_state, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)

            # if we have complete the run (terminated) or gone too many steps (truncated),
            # update the policy
            if terminated or truncated:

                # update the policy based on the rewards and log probabilities of the actions
                update_policy(optimizer, rewards, log_probs)

                num_steps.append(step)
                avg_steps.append(np.mean(num_steps[-ROLLING_AVERAGE_LENGTH:]))
                all_rewards.append(np.sum(rewards))

                # log 50 updates
                if (episode + 1) % (num_episodes//NUMBER_OF_LOGS) == 0:
                    total_reward = np.round(np.sum(rewards), decimals=3)
                    avg_reward = np.round(np.mean(all_rewards[-ROLLING_AVERAGE_LENGTH:]), decimals=3)
                    print(f"episode: {episode + 1}, reward: {total_reward}, "
                          f"average reward: {avg_reward}, length: {step}, average length: {avg_steps[-1]}")

                if (episode + 1) % (num_episodes//NUMBER_OF_VIDEOS) == 0:
                    generate_video(env.spec.id, policy_net, episode)

                break

            # update the state
            state = new_state

    # plot the step data
    plt.plot(num_steps)
    plt.plot(avg_steps)
    plt.xlabel('Episode')
    plt.show()


if __name__ == '__main__':
    main()
