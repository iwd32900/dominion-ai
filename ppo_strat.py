import pickle
import time

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np

from dmstrat import *
import ppo_clip

# random.seed(123456)

def onehot(val, minv, maxv, scale):
    assert val == int(val)
    val = int(val)
    val -= minv # new range [0, minv+maxv]
    maxv -= minv
    L = maxv+1 + 2 # 2 extra are for proportional above/below
    feat = torch.zeros(L, dtype=torch.float32)
    below = -val/scale if val < 0 else 0
    above = (val - maxv)/scale if val > maxv else 0
    hot = max(min(val, maxv), 0)
    feat[hot] = 1
    feat[L-2] = below
    feat[L-1] = above
    return feat

class PPOStrategy(Strategy):
    '''
    '''
    def __init__(self, ac=None):
        super().__init__() # calls reset()
        obs_dim = 58 + len(self.buys) # depends on impl. of self.state_idx()
        n_acts = len(self.buys)
        if ac is None:
            hidden_sizes = [64, 64]
            self.actor_critic = ppo_clip.MLPActorCritic(obs_dim, n_acts, hidden_sizes)
            #self.actor_critic = ppo_clip.MLPActorCritic(obs_dim, n_acts, hidden_sizes, activation=ppo_clip.nn.ReLU)
        else:
            self.actor_critic = ac
        self.ppo_buf = ppo_clip.PPOBuffer(obs_dim, n_acts, act_dim=None, size=1000 * 5 * MAX_TURNS)
        self.ppo_algo = ppo_clip.PPOAlgo(self.actor_critic, pi_lr=1e-4)
        # self.ppo_algo = ppo_clip.PPOAlgo(self.actor_critic, pi_lr=1e-4, vf_lr=3e-4)
        self.learn = True # if False, do not update any of the strategies, and do not make exploratory moves
    def reset(self):
        super().reset()
        if hasattr(self, 'ppo_buf'):
            self.ppo_buf.reset()
    def state_idx(self, game, player):
        # To start, learn a static strategy, regardless of game state:
        # return 0
        # With competent players, most games end within ~20 turns
        # t = min(game.turn, 19)
        # s = player.calc_victory_points() - max(p.calc_victory_points() for p in game.players if p != player)
        # if s < -13: s = -13
        # elif s > 13: s = 13
        # p = min(game.stockpile[Province], 3) # [0,8] in the 2-player game
        # I've read that as a general rule, inputs should be on [-1,1]-ish
        # return torch.as_tensor([t/20, s/13, p/3], dtype=torch.float)
        # obs = torch.zeros(20+27+4, dtype=torch.float)
        # # n-hot cumulative coding:
        # obs[0:t] = 1
        # if s < 0:
        #     obs[20:20+(-s)] = 1
        # elif s > 0:
        #     obs[33:33+s] = 1
        # obs[47:47+p] = 1
        # # One-hot coding:
        # # obs[t] = 1
        # # obs[20+(s+13)] = 1
        # # obs[47+p] = 1
        score = player.calc_victory_points() - max(p.calc_victory_points() for p in game.players if p != player)
        prov = game.stockpile[Province] # [0,8] in the 2-player game
        suicidal = torch.tensor([
            prov == 1 and score <= -6,
            prov == 2 and score <= 0,
        ], dtype=torch.float32)
        obs = torch.cat([
            onehot(game.turn, minv=0, maxv=19, scale=10),   # 22
            onehot(score, minv=-13, maxv=13, scale=6),      # 29
            onehot(prov, minv=1, maxv=3, scale=6),          # 5
            suicidal,                                       # 2
            self.bought / 5,                                # len(buys)
        ])
        return obs
    def start_game(self):
        self.bought = torch.zeros(len(self.buys), dtype=torch.float32)
    def iter_buys(self, game, player):
        obs = self.state_idx(game, player)
        fbn = np.array([not b.can_move(game, player) for b in self.buys]) # forbidden, or invalid, actions
        act, val, logp, pi = self.actor_critic.step(obs, fbn)
        rew = 0
        # rew = buy.card.victory_points/100 if buy.card else 0
        self.ppo_buf.store(obs, fbn, act, rew, val, logp)
        buy_idx = act.item()
        self.bought[buy_idx] += 1
        buy = self.buys[buy_idx]
        # For debugging during tournament play
        # if buy.card == Province and obs[56] != 0:
        #     import ipdb; ipdb.set_trace()
        return [ buy ]
    def end_game(self, reward, game, player):
        # reward = 2*reward - 1 # {0, 0.5, 1}  ->  {-1, 0, +1}
        self.ppo_buf.finish_path(reward)
    def step(self):
        if not self.learn:
           return # do not update statistics
        self.ppo_algo.update(self.ppo_buf)

def main_basic_polygrad():
    players = 2
    popsize = players # some multiple of 2, 3, and 4
    strategies = [PPOStrategy() for _ in range(popsize)]

    CYCLES = 100
    GPS = 250
    for cycle in range(CYCLES): # expect to Ctrl-C to exit early
        # if cycle >= 50:
        #     GPS = 2000
        if cycle == CYCLES-1: # last one
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
