import logging
from os import path

import numpy as np
import torch
from torch.nn.utils import clip_grad
import torch.utils.tensorboard as tb

from agent import Player, StochasticPlayer
from environment import Game
from model import Model

if __name__ == "__main__":

    train_logger = tb.SummaryWriter(path.join('tb_logs', 'train_new_g9_clipvalue0000001'), flush_secs=1)

    NUM_EPISODES = 100000
    GAMMA = .9
    CLIPPING = 0.0000001

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

        model.eval()

        # play a game until complete
        while not game.complete:

            # if this is the start of a new turn, roll the dice
            if game.is_new_turn():
                game.roll()

            # get the current state
            state = game.get_state()

            # act on the state
            logits, action = player.act(state=state)
            logits.detach_()

            train_logger.add_scalar('logits_mean', float(logits.mean()), global_step=global_step)
            train_logger.add_scalar('logits_max', float(logits.max()), global_step=global_step)
            train_logger.add_scalar('logits_min', float(logits.min()), global_step=global_step)

            game.apply_action(action)

            reward = 0
            if game.complete:
                reward = game.get_reward()

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            final_score = game.get_total_score()
            
            global_step +=1

        train_logger.add_scalar('num_actions', len(actions), global_step=global_step)

        if final_score > 0:
            train_logger.add_scalar('final_score', final_score, global_step=global_step)
            print(f'For episode {e},  Positive {final_score=}, for {game=}')

        # generate discounted rewards
        for i in range(len(rewards)):
            discounted_reward = 0
            gamma_p = 1
            for reward in rewards[i:]:
                discounted_reward += gamma_p * reward
                gamma_p *= GAMMA

            discounted_rewards.append(discounted_reward)
        
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
        discounted_rewards = (discounted_rewards - torch.mean(discounted_rewards))/discounted_rewards.std()

        features = []
        for state in states:
            features.append(player.get_features(state))

        features_tensor = torch.stack(features)
        rewards_tensor = discounted_rewards
        actions_tensor = torch.LongTensor(actions)

        model.train()

        logits = model(features_tensor)
        probs = torch.softmax(logits, dim=1)
        log_probs = torch.gather(torch.log(probs), dim=1, index=actions_tensor.unsqueeze(dim=1)).squeeze()

        loss = -(rewards_tensor * log_probs).mean()

        train_logger.add_scalar('loss', float(loss), global_step=global_step)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        clip_grad.clip_grad_value_(model.parameters(), CLIPPING)
