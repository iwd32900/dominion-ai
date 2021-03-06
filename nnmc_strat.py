import pickle
import time

import torch
import torch.nn as nn
import numpy as np

from dmstrat import *
import ann_mc

# random.seed(123456)

def onehot(val, minv, maxv, scale):
    """
    One-hot coding for an integer between minv and maxv (inclusive),
    with positive and negative "overflow" scalars as well.
    """
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

class AnnMCStrategy(Strategy):
    '''
    '''
    def __init__(self, eps=0.1, buy_nn=None):
        super().__init__() # calls reset()
        self.eps = eps

        # obs_dim = 58 + len(self.buys) # depends on impl. of self.act_state()
        # n_acts = len(self.actions)
        # if act_ac is None:
        #     hidden_sizes = [32] #[64, 64]
        #     self.act_ac = ppo_clip.MLPActorCritic(obs_dim, n_acts, hidden_sizes)
        # else:
        #     self.act_ac = act_ac
        # self.act_buf = ppo_clip.PPOBuffer(obs_dim, n_acts, act_dim=None, size=1000 * 5 * MAX_TURNS)
        # self.act_optim = ppo_clip.PPOAlgo(self.act_ac) #, pi_lr=1e-4)

        obs_dim = 58 + len(self.buys) # depends on impl. of self.buy_state()
        n_acts = len(self.buys)
        if buy_nn is None:
            hidden_sizes = [32] #[64, 64]
            self.buy_nn = ann_mc.mlp([obs_dim] + hidden_sizes + [n_acts])
        else:
            self.buy_nn = buy_nn
        self.buy_buf = ann_mc.Buffer(obs_dim, n_acts, size=1000 * 5 * MAX_TURNS)
        self.buy_optim = ann_mc.MCAlgo(self.buy_nn)

        self.learn = True # if False, do not update any of the strategies, and do not make exploratory moves
    @property
    def learn(self):
        return self._learn
    @learn.setter
    def learn(self, learn):
        self._learn = learn
        if learn:
            # self.act_ac.train()
            self.buy_nn.train()
        else:
            # self.act_ac.eval()
            self.buy_nn.eval()
    def reset(self):
        super().reset()
        if hasattr(self, 'act_buf'):
            self.act_buf.reset()
        if hasattr(self, 'buy_buf'):
            self.buy_buf.reset()
    def __getstate__(self):
        return {
            # "act_ac": self.act_ac,
            "buy_nn": self.buy_nn,
        }
    def __setstate__(self, state):
        self.__init__(
            # act_ac=state['act_ac'],
            buy_nn=state['buy_nn']
        )
    def start_game(self):
        self.bought = torch.zeros(len(self.buys), dtype=torch.float32)
    def act_state(self, game, player):
        score = player.calc_victory_points() - max(p.calc_victory_points() for p in game.players if p != player)
        prov = game.stockpile[Province] # [0,8] in the 2-player game
        suicidal = torch.tensor([
            prov == 1 and score <= -6,
            prov == 2 and score <= 0,
        ], dtype=torch.float32)
        # I've read that as a general rule, inputs should be on [-1,1]-ish
        obs = torch.cat([
            # With competent players, most games end within ~20 turns
            onehot(game.turn, minv=0, maxv=19, scale=10),   # 22
            onehot(score, minv=-13, maxv=13, scale=6),      # 29
            onehot(prov, minv=1, maxv=3, scale=6),          # 5
            suicidal,                                       # 2
            self.bought / 5,                                # len(buys)
        ])
        return obs
    # There are relatively few hands per game with >1 action available,
    # and even fewer with >1 terminal action.
    # As such, given finite training time, a neural network is no better
    # than the base heuristic here!
    #
    # def iter_actions(self, game, player):
    #     actions_in_hand = [a for a in player.hand if a.is_action]
    #     if not actions_in_hand: return []
    #     obs = self.act_state(game, player)
    #     fbn = np.array([not a.can_play(game, player) for a in self.actions]) # forbidden, or invalid, actions
    #     act, val, logp, pi = self.act_ac.step(obs, fbn)
    #     rew = 0
    #     if self.learn:
    #         self.act_buf.store(obs, fbn, act, rew, val, logp)
    #     act_idx = act.item()
    #     act_card = self.actions[act_idx]
    #     return [ act_card ]
    def buy_state(self, game, player):
        score = player.calc_victory_points() - max(p.calc_victory_points() for p in game.players if p != player)
        prov = game.stockpile[Province] # [0,8] in the 2-player game
        suicidal = torch.tensor([
            prov == 1 and score <= -6,
            prov == 2 and score <= 0,
        ], dtype=torch.float32)
        # I've read that as a general rule, inputs should be on [-1,1]-ish
        obs = torch.cat([
            # With competent players, most games end within ~20 turns
            onehot(game.turn, minv=0, maxv=19, scale=10),   # 22
            onehot(score, minv=-13, maxv=13, scale=6),      # 29
            onehot(prov, minv=1, maxv=3, scale=6),          # 5
            suicidal,                                       # 2
            self.bought / 5,                                # len(buys)
        ])
        return obs
    def iter_buys(self, game, player):
        rew = 0
        # rew = buy.card.victory_points/100 if buy.card else 0
        obs = self.buy_state(game, player)
        q = self.buy_nn(obs)
        fbn = np.array([not b.can_buy(game, player) for b in self.buys]) # forbidden, or invalid, actions
        q[fbn] = -np.inf
        if random.random() < self.eps and self.learn:
            # Random choice among legal actions
            r = torch.rand_like(q)
            r[fbn] = -np.inf
            act = r.argmax()
        else:
            act = q.argmax()
            # Interestingly, including the random actions in the history seems to SLOW/HURT convergence.
            if self.learn:
                self.buy_buf.store(obs, fbn, act, rew)
        buy_idx = act.item()
        self.bought[buy_idx] += 1
        buy = self.buys[buy_idx]
        # For debugging during tournament play
        # if buy.card == Province and obs[56] != 0:
        #     import ipdb; ipdb.set_trace()
        return [ buy ]
    def rank_buys(self, game, player):
        # Follow the policy, sort actions by value
        obs = self.buy_state(game, player)
        q = self.buy_nn(obs)
        fbn = np.array([not b.can_buy(game, player) for b in self.buys]) # forbidden, or invalid, actions
        q[fbn] = -np.inf
        # For now, stick with a deterministic ranking, not a probabilistic one.
        buys = sorted_by(self.buys, q.tolist(), reverse=True)
        return buys + [Curse]
    def end_game(self, reward, game, player):
        super().end_game(reward, game, player)
        if self.learn:
            # self.act_buf.finish_path(reward)
            self.buy_buf.finish_path(reward)
    def step(self):
        if self.learn:
            # if self.act_buf.ptr > 0:
            #     self.act_optim.update(self.act_buf)
            self.buy_optim.update(self.buy_buf)

def main_nnmc():
    players = 2
    popsize = players # some multiple of 2, 3, and 4
    strategies = [AnnMCStrategy() for _ in range(popsize)]

    CYCLES, GPS = 500, 250
    # CYCLES, GPS = 2000, 250
    # CYCLES, GPS = 1000, 1000 # larger batches should make estimates more stable?
    for cycle in range(CYCLES): # expect to Ctrl-C to exit early
        # # Decrease random exploration over time
        # eps = np.interp(cycle, [0, 20, 0.9*CYCLES], [1.0, 0.1, 0.02])
        # for strategy in strategies:
        #     strategy.eps = eps
        if cycle == CYCLES-1: # last one
            GPS = 1000
            for strategy in strategies:
                strategy.learn = False
        start = time.time()
        run_tournament(strategies, players, games_per_strategy=GPS)

        print(f"round {cycle}    {players} players    {GPS} games    {time.time() - start:.2f} sec " + ("="*70))
        for strategy in strategies:
            print(strategy)
        print("")

        save_strategies(strategies, "save_nnmc")

        for strategy in strategies:
            strategy.step()

if __name__ == '__main__':
    main_nnmc()
