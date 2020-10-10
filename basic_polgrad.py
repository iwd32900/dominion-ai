import pickle
import time

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np

from dmstrat import *

# random.seed(123456)

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class BasicPolicyGradientStrategy(Strategy):
    '''
    Very simple policy gradient algorithm based on
    https://github.com/openai/spinningup/blob/master/spinup/examples/pytorch/pg_math/2_rtg_pg.py
    '''
    def __init__(self, logits_net=None):
        super().__init__()
        lr = 1e-2
        if logits_net is None:
            obs_dim = 20+26+4 # depends on impl. of self.state_idx()
            n_acts = len(self.buys)
            hidden_sizes = [32]
            self.logits_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])
        else:
            self.logits_net = logits_net
        self.optimizer = Adam(self.logits_net.parameters(), lr=lr)
        self.learn = True # if False, do not update any of the strategies, and do not make exploratory moves
    def reset(self):
        super().reset()
        self.state_hist = [] # [tensor(float32)]
        self.action_hist = [] # [int]
        self.logp_a_hist = [] # [tensor(float32)]
        self.rewards_to_go = [] # [float]
    # make function to compute action distribution
    # e.g. obs = torch.tensor([1, 2, 3], dtype=torch.float)
    def get_policy(self, obs, invalid_act):
        logits = self.logits_net(obs)
        logits[invalid_act] = -np.inf
        return Categorical(logits=logits)
    # make loss function whose gradient, for the right data, is policy gradient
    def compute_loss(self):
        # obs     = torch.stack(self.state_hist)
        # act     = torch.as_tensor(self.action_hist, dtype=torch.int32)
        logp    = torch.stack(self.logp_a_hist)
        weights = torch.as_tensor(self.rewards_to_go, dtype=torch.float32)
        return -(logp * weights).mean()
    def state_idx(self, game, player):
        # To start, learn a static strategy, regardless of game state:
        # return 0
        # With competent players, most games end within ~20 turns
        t = min(game.turn, 19)
        s = player.calc_victory_points() - max(p.calc_victory_points() for p in game.players if p != player)
        if s < -13: s = -13
        elif s > 13: s = 13
        p = min(game.stockpile[Province], 3) # [0,8] in the 2-player game
        # I've read that as a general rule, inputs should be on [-1,1]-ish
        # return torch.as_tensor([t/20, s/13, p/3], dtype=torch.float)
        obs = torch.zeros(20+26+4, dtype=torch.float)
        obs[0:t] = 1
        if s < 0:
            obs[20:20+(-s)] = 1
        elif s > 0:
            obs[33:33+s] = 1
        obs[46:46+p] = 1
        return obs
    def start_game(self):
        self.episode_rewards = []
    def iter_buys(self, game, player):
        s = self.state_idx(game, player)
        invalid_act = np.array([not b.can_move(game, player) for b in self.buys])
        # with torch.no_grad():
        # MUST have gradient here -- must preserve it through the logp for backprop
        pi = self.get_policy(s, invalid_act)
        buy_idx = pi.sample()
        buy = self.buys[buy_idx.item()]
        logp = pi.log_prob(buy_idx)
        self.state_hist.append(s)
        self.action_hist.append(buy_idx)
        self.logp_a_hist.append(logp)
        rew = 0
        # rew = buy.card.victory_points/100 if buy.card else 0
        self.episode_rewards.append(rew)
        return [ buy ]
    def end_game(self, reward, game, player):
        # Cumulative sum in reverse direction -- works because we're not discounting
        rtg = np.array(self.episode_rewards)[::-1].cumsum()[::-1] + reward
        # self.rewards_to_go.extend([reward] * len(self.episode_rewards))
        self.rewards_to_go.extend(rtg)
        assert len(self.rewards_to_go) == len(self.state_hist) == len(self.action_hist)
    def step(self):
        if not self.learn:
           return # do not update statistics
        # take a single policy gradient update step
        self.optimizer.zero_grad()
        batch_loss = self.compute_loss()
        batch_loss.backward()
        self.optimizer.step()
        return batch_loss

def main_basic_polygrad():
    players = 2
    popsize = players # some multiple of 2, 3, and 4
    strategies = [BasicPolicyGradientStrategy() for _ in range(popsize)]

    CYCLES = 150
    GPS = 200
    for cycle in range(CYCLES): # expect to Ctrl-C to exit early
        if cycle == CYCLES-1: # last one
            GPS = 2000
            # Play final round without random exploratory moves
            for strategy in strategies:
                strategy.learn = False
        start = time.time()
        # Might need additional games:  starting code uses batch of 5000 moves...
        run_tournament(strategies, players, games_per_strategy=GPS)

        print(f"round {cycle}    {players} players    {time.time() - start:.2f} sec " + ("="*70))
        for strategy in strategies:
            print(strategy)
        print("")

        with open("strategies.pkl", "wb") as f:
            pickle.dump(strategies, f)

        for strategy in strategies:
            strategy.step()

if __name__ == '__main__':
    main_basic_polygrad()
