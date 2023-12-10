import logging
from os import path

import numpy as np
import torch
from torch.nn.utils import clip_grad
import torch.utils.tensorboard as tb

from agent import StochasticPlayer
from environment import Game
from model import Model

if __name__ == "__main__":

    train_logger = tb.SummaryWriter(path.join('tb_logs', 'train_clipped_v_0_01_lr0001'), flush_secs=1)

    NUM_EPISODES = 100000
    GAMMA = .3

    # 38 output states: 1-32 is for dice selection, 33-38 is for choosing
    model = Model(num_inputs=12, num_outputs=38)
    player = StochasticPlayer(model=model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

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
                logging.debug(f'Current (after roll) {game=}')

            # get the current state
            state = game.get_state()

            # act on the state
            logits, action = player.act(state=state)
            action_prob = torch.softmax(logits, dim=0)[action]
            action_log_prob = torch.log(action_prob)

            reward = game.apply_action(action)

            train_logger.add_scalar('logits_mean', float(logits.mean()), global_step=global_step)
            train_logger.add_scalar('logits_max', float(logits.max()), global_step=global_step)
            train_logger.add_scalar('logits_min', float(logits.min()), global_step=global_step)


            global_step +=1

        final_reward = np.sum(rewards)
        train_logger.add_scalar('final_reward', final_reward, global_step=global_step)

        final_rewards.append(final_reward)
        final_score = game.get_total_score()
        
        if final_score > 0:
            train_logger.add_scalar('final_score', final_score, global_step=global_step)
            print(f'For episode {e},  Positive {final_score=}, for {game=}')
    
        final_scores.append(final_score)
        logging.debug(f'Game over: {game}')

        for i in range(len(rewards)):
            discounted_reward = 0 
            for j, reward in enumerate(rewards[i:]):
                discounted_reward += GAMMA**j * reward

            discounted_rewards.append(discounted_reward)
        
        discounted_rewards = torch.tensor(discounted_rewards,dtype=torch.float32)
        discounted_rewards = (discounted_rewards - torch.mean(discounted_rewards))/(torch.std(discounted_rewards))

        action_log_probs = torch.stack(action_log_probs)
        policy_gradient = (-action_log_probs*discounted_rewards).sum()
        train_logger.add_scalar('policy_gradient', float(policy_gradient), global_step=global_step)
        
        model.zero_grad()
        policy_gradient.backward()
        optimizer.step()

        clip_grad.clip_grad_value_(model.parameters(), 0.01)


    print(f'{np.max(final_rewards)=}, {np.mean(final_rewards)=}')
    print(f'{np.max(final_scores)=}, {np.mean(final_scores)=}')
    