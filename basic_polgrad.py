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
    def __init__(self):
        super().__init__()
        hidden_sizes = [32]
        lr = 1e-2
        obs_dim = 3 # depends on impl. of self.state_idx()
        n_acts = len(self.buys)
        self.logits_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])
        self.optimizer = Adam(self.logits_net.parameters(), lr=lr)
        self.learn = True # if False, do not update any of the strategies, and do not make exploratory moves
    def reset(self):
        super().reset()
        self.state_hist = [] # [tensor(float32)]
        self.action_hist = [] # [int]
        self.rewards_to_go = [] # [float]
    # make function to compute action distribution
    # e.g. obs = torch.tensor([1, 2, 3], dtype=torch.float)
    def get_policy(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)
    # make loss function whose gradient, for the right data, is policy gradient
    def compute_loss(self):
        #obs     = torch.as_tensor(self.state_hist, dtype=torch.float32)
        obs     = torch.stack(self.state_hist)
        act     = torch.as_tensor(self.action_hist, dtype=torch.int32)
        weights = torch.as_tensor(self.rewards_to_go, dtype=torch.float32)
        logp = self.get_policy(obs).log_prob(act)
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
        return torch.as_tensor([t/20, s/13, p/3], dtype=torch.float)
    def start_game(self):
        self.last_s = None
        self.episode_rewards = []
    def accept_buy(self, buy, game, player):
        super().accept_buy(buy, game, player)
        # Convert current game state into an integer
        # s = self.state_idx(game, player)
        s = self.last_s # state could have changed due to us buying the card
        a = self.buys.index(buy)
        self.state_hist.append(s)
        self.action_hist.append(a)
        self.episode_rewards.append(0)
    def iter_buys(self, game, player):
        self.last_s = s = self.state_idx(game, player)
        #return self.get_policy(obs).sample().item()
        probs = self.get_policy(s).probs.detach() # don't want gradient here
        # Neat trick:  https://softwareengineering.stackexchange.com/questions/233541/how-to-implement-a-weighted-shuffle
        # We've dropped the negative sign here so that high-prob items will sort to the front.
        w = np.random.random(probs.shape) ** (1 / probs).numpy()
        sorted_buys = sorted_by(self.buys, w, reverse=False)
        return sorted_buys
    def end_game(self, reward, game, player):
        if self.last_s is None:
            assert False, "Games shouldn't end with no moves taken"
        self.rewards_to_go.extend([reward] * len(self.episode_rewards))
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

    CYCLES = 500
    for cycle in range(CYCLES): # expect to Ctrl-C to exit early
        if cycle == CYCLES-1: # last one
            # Play final round without random exploratory moves
            for strategy in strategies:
                strategy.learn = False
        start = time.time()
        # Might need additional games:  starting code uses batch of 5000 moves...
        run_tournament(strategies, players, games_per_strategy=200)

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