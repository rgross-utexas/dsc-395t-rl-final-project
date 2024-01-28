import argparse
from collections import namedtuple, deque, defaultdict
from itertools import count
import math
import random
from typing import Dict, List, Tuple

import cv2
import gymnasium as gym
from gymnasium import Env
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

TB = True

# This code is based on the DQN tutorial from PyTorch

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([], maxlen=self.capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.memory, batch_size)

    def sample_as_transpose(self, batch_size: int) -> Transition:
        # this is some magic that converts the batch from a collection of Transitions to
        # a Transition of collections (states, actions, next_states, and rewards). This came
        # directly from the pytorch DQN tutorial
        return Transition(*zip(*self.sample(batch_size)))

    def __len__(self):
        return len(self.memory)


class PolicyNetwork(nn.Module):

    def __init__(self, state_dims: int, action_dims: int, inner_dims: int):
        super(PolicyNetwork, self).__init__()

        self.seq = nn.Sequential(
            nn.Linear(state_dims, inner_dims),
            nn.Tanh(),
            nn.Linear(inner_dims, inner_dims),
            nn.Tanh(),
            nn.Linear(inner_dims, action_dims)
        )

    def forward(self, x):
        return self.seq(x)


def dqn(env: Env,
        lr: float,
        num_episodes: int,
        render: bool,
        num_movies: int,
        inner_dims: int,
        replay_mem_size: int,
        tau: float,
        gamma: float,
        batch_size: int,
        eps_config: Tuple[float, float, int] = (.9, .05, 5000)
        ) -> Dict:
    env_id = env.spec.id
    tb_id = f"tb/{env_id}_{num_episodes}_{inner_dims}_{replay_mem_size}_{tau}_p25_{lr}"
    tb_id = tb_id.replace('.', 'p')
    print(f'{tb_id=}')
    if TB:
        tb_logger = SummaryWriter(tb_id, flush_secs=5)

    episode_data = {'lengths': defaultdict(float),
                    'rewards': defaultdict(float),
                    'frames': defaultdict(list)}

    # if GPU is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get number of actions from gym action space
    n_actions = env.action_space.n

    # get the number of states via the length of a state
    state, _ = env.reset()
    n_states = len(state)

    # create the models
    behavior_net = PolicyNetwork(state_dims=n_states, action_dims=n_actions,
                                 inner_dims=inner_dims).to(device)
    target_net = PolicyNetwork(state_dims=n_states, action_dims=n_actions,
                               inner_dims=inner_dims).to(device)

    # update the target weights with the policy weights
    target_net.load_state_dict(behavior_net.state_dict())

    # create the optimizer
    optimizer = optim.AdamW(behavior_net.parameters(), lr=lr, amsgrad=True)

    # create the loss function
    criterion = nn.SmoothL1Loss()

    # create the replay memory
    memory = ReplayMemory(replay_mem_size)

    render_freq = num_episodes // num_movies

    steps_done = 0

    for e in tqdm(range(num_episodes), unit='episodes'):

        # Initialize the environment and get it's state
        state, _ = env.reset()

        if render and (e + 1) % render_freq == 0:
            episode_data['frames'][e].append(env.render())

        # convert state to tensor
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        for t in count():

            # election an action from the behavior network based on an epsilon greedy policy
            action = select_e_greedy_action(env=env, device=device, state=state,
                                            steps_done=steps_done, behavior_net=behavior_net,
                                            eps_config=eps_config)

            # take a step with the given action
            next_state, reward, terminated, truncated, _ = env.step(action.item())

            # track the number of steps
            steps_done += 1

            if render and (e + 1) % render_freq == 0:
                episode_data['frames'][e].append(env.render())

            # convert the reward to a tensor and track it
            reward = torch.tensor([reward], device=device)
            episode_data['rewards'][e] += reward

            if terminated:
                next_state = None
            else:
                # convert the next state to a tensor
                next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)

            # store the transition in memory
            memory.push(state, action, next_state, reward)

            # transition to the next state
            state = next_state

            # perform one step of the optimization on the policy network
            optimize_model(device=device, optimizer=optimizer, memory=memory,
                           behavior_net=behavior_net, target_net=target_net,
                           criterion=criterion, batch_size=batch_size, gamma=gamma)

            # update the target net's weights with the behavior net's weights
            update_target_weights(behavior_net, target_net, tau)

            if terminated or truncated:
                episode_data['lengths'][e] = t + 1

                break

        if TB:
            tb_logger.add_scalar('length', episode_data['lengths'][e], global_step=e)
            tb_logger.add_scalar('reward', episode_data['rewards'][e], global_step=e)

    return episode_data


def calculate_epsilon(e_start: float, e_end: float, e_decay: int, steps_done: int) -> float:
    return e_end + (e_start - e_end) * math.exp(-1. * steps_done / e_decay)


def select_e_greedy_action(env: Env, device, state: torch.Tensor, steps_done: int,
                           behavior_net: PolicyNetwork, eps_config: Tuple[float, float, int]) -> torch.Tensor:
    e = calculate_epsilon(eps_config[0], eps_config[1], eps_config[2], steps_done)
    if random.random() > e:
        with torch.no_grad():
            # get the action with the largest q value
            return behavior_net(state).max(1).indices.view(1, 1)
    else:
        # sample randomly from all actions
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


def update_target_weights(behavior_net: PolicyNetwork, target_net: PolicyNetwork, tau: float):
    """
    "soft" update of the target net's weights with the policy net's weights
    """

    target_net_state_dict = target_net.state_dict()
    behavior_net_state_dict = behavior_net.state_dict()

    for key in behavior_net_state_dict:
        target_net_state_dict[key] = behavior_net_state_dict[key] * tau + target_net_state_dict[key] * (1 - tau)

    target_net.load_state_dict(target_net_state_dict)


def optimize_model(device, optimizer: optim.Optimizer, memory: ReplayMemory,
                   behavior_net: PolicyNetwork, target_net: PolicyNetwork,
                   criterion: nn.Module,
                   batch_size: int, gamma: float) -> None:
    if len(memory) < .25 * memory.capacity:
        return

    # sample a batch from the replay memory
    batch = memory.sample_as_transpose(batch_size)

    # convert the collections to tensors
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # get the value for each state from the behavior policy (based on the action taken)
    state_action_values = behavior_net(state_batch).gather(1, action_batch)

    # create a boolean mask of non-final transitions
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                  device=device, dtype=torch.bool)

    # this next state values magic comes directly from the pytorch DQN tutorial

    # filter out all the final transitions
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    # get the max value for each next state from the target policy (or 0 if the next state is final)
    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    # compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # compute the loss with the state-action values from the behavior policy used as the gold values
    # and the expected state-action values from the target policy used as the predicted values
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # optimize the model
    optimizer.zero_grad()

    # back propagate
    loss.backward()

    # in-place gradient clipping on the behavior network to protect against large gradients,
    # helping to stabilize learning
    torch.nn.utils.clip_grad_value_(behavior_net.parameters(), 100)

    optimizer.step()


def save_frames_as_video(episode_frames: defaultdict, algorithm: str,
                         filename_template: str = './{algorithm}_episode_{episode}.mp4'):
    for i, frames in episode_frames.items():

        if len(frames) > 0:
            np_frames = np.array(frames)
            save_episode_as_video(algorithm=algorithm, filename_template=filename_template,
                                  episode=i, np_frames=np_frames)


def save_episode_as_video(algorithm, filename_template, episode, np_frames):
    filename = filename_template.format(algorithm=algorithm, episode=(episode + 1))
    fps = 30
    height = np_frames.shape[2]
    width = np_frames.shape[1]
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (height, width))
    for i in range(np_frames.shape[0]):
        data = np_frames[i, :, :, :]
        out.write(data)
    out.release()


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


def run(env_str: str, num_episodes: int, num_movies: int, inner_dims: int,
        replay_mem_size: int = 10000, tau: float = 0.005, lr: float = 1e-4,
        gamma: float = .999, batch_size: int = 128) -> None:
    print(f'Running {env_str}...')

    env = gym.make(env_str, render_mode=render_mode)

    env.metadata['render_fps'] = 120

    data = dqn(env=env, num_episodes=num_episodes, num_movies=num_movies, inner_dims=inner_dims,
               replay_mem_size=replay_mem_size, tau=tau, lr=lr, gamma=gamma, batch_size=batch_size,
               render=True)

    save_frames_as_video(episode_frames=data['frames'], algorithm=f'{env.spec.id}_dqn')

    data = {env.spec.id: {'r': data['rewards'], 'l': data['lengths']}}
    render_figure(data, 25, 'env', f'{env.spec.id}_dqn')


if __name__ == '__main__':

    render_mode = "rgb_array"

    parser = argparse.ArgumentParser(prog='dqn')
    parser.add_argument('--envs', nargs='+',
                        # skip mountain_car because it learns very slowly
                        # default=['acrobat', 'lunar_lander', 'mountain_car', 'cart_pole'])
                        default=['acrobat', 'lunar_lander', 'cart_pole'])
    args = parser.parse_args()

    envs = args.envs
    if 'acrobat' in envs:
        run(env_str="Acrobot-v1", num_episodes=500, inner_dims=128, num_movies=5)

    # Mountain car does not learn!
    if 'mountain_car' in envs:
        run(env_str="MountainCar-v0", num_episodes=5000, inner_dims=64, num_movies=10,
            replay_mem_size=10000, tau=.005, lr=.001)

    if 'cart_pole' in envs:
        run(env_str="CartPole-v1", num_episodes=1000, inner_dims=256, num_movies=10)

    if 'lunar_lander' in envs:
        run(env_str="LunarLander-v2", num_episodes=400, inner_dims=1024, num_movies=5)

    print('Done.')
