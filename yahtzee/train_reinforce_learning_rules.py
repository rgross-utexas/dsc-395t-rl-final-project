import logging
from os import path

import numpy as np
import torch
from torch.nn.utils import clip_grad
import torch.utils.tensorboard as tb
from torch.optim.lr_scheduler import ChainedScheduler

from agent import StochasticPlayer
from environment import Game
from model import Model

if __name__ == "__main__":

    train_logger = tb.SummaryWriter(path.join('tb_logs', 'train_rules_g9_noclip_lr001_pgmean_divmax'), flush_secs=1)

    NUM_EPISODES = 100000
    GAMMA = .9

    # 38 output states: 1-32 is for dice selection, 33-38 is for choosing
    model = Model(num_inputs=12, num_outputs=38)
    player = StochasticPlayer(model=model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    final_rewards = []
    final_scores = []
    global_step = 0
    for e in range(NUM_EPISODES):

        # start a new game
        game = Game()

        states = []
        actions = []
        action_log_probs = []
        rewards = []
        discounted_rewards = []

        # play a game until complete
        while not game.complete:

            # if this is the start of a new turn, roll the dice
            if game.is_new_turn():
                game.roll()

            # get the current state
            state = game.get_state()

            # act on the state
            logits, action = player.act(state=state)
            action_prob = torch.softmax(logits, dim=0)[action]
            action_log_prob = torch.log(action_prob)

            game.apply_action(action)

            reward = 0
            if game.complete:
                if game.cheat:
                    reward = -1
                else:
                    reward = game.get_reward()/100

            states.append(state)
            actions.append(action)
            action_log_probs.append(action_log_prob)
            rewards.append(reward)

            train_logger.add_scalar('logits_mean', float(logits.mean()), global_step=global_step)
            train_logger.add_scalar('logits_max', float(logits.max()), global_step=global_step)
            train_logger.add_scalar('logits_min', float(logits.min()), global_step=global_step)

            global_step +=1

        final_reward = np.sum(rewards)
        train_logger.add_scalar('final_reward', final_reward, global_step=global_step)

        final_rewards.append(final_reward)
        final_score = game.get_total_score()    
        final_scores.append(final_score)

        for i in range(len(rewards)):
            discounted_reward = 0
            gamma_p = 1
            for reward in rewards[i:]:
                discounted_reward += gamma_p * reward
                gamma_p *= GAMMA

            discounted_rewards.append(discounted_reward)
        
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
        # discounted_rewards = (discounted_rewards - torch.mean(discounted_rewards))/(torch.std(discounted_rewards))
        discounted_rewards = discounted_rewards/torch.max(torch.abs(discounted_rewards))
        discounted_rewards_mean = float(discounted_rewards.mean())
        train_logger.add_scalar('discounted_rewards_max', float(discounted_rewards.max()), global_step=global_step)
        train_logger.add_scalar('discounted_rewards_min', float(discounted_rewards.min()), global_step=global_step)
        train_logger.add_scalar('discounted_rewards_mean', discounted_rewards_mean, global_step=global_step)

        if final_score > 0:
            train_logger.add_scalar('final_score', final_score, global_step=global_step)
            print(f'For episode {e}, {final_score=}, {discounted_rewards_mean=}, for {game=}')

        action_log_probs = torch.stack(action_log_probs)
        policy_gradients = (-action_log_probs*discounted_rewards)
        policy_gradient = policy_gradients.mean()

        train_logger.add_scalar('action_log_probs_mean', float(action_log_probs.mean()), global_step=global_step)
        train_logger.add_scalar('action_log_probs_max', float(action_log_probs.max()), global_step=global_step)
        train_logger.add_scalar('action_log_probs_min', float(action_log_probs.min()), global_step=global_step)
        train_logger.add_scalar('policy_gradient', float(policy_gradient), global_step=global_step)

        model.zero_grad()
        policy_gradient.backward()
        optimizer.step()

        # clip_grad.clip_grad_value_(model.parameters(), 0.0001)
