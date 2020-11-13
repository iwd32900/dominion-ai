from collections import namedtuple
import pickle
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.optim import Adam, RMSprop
import numpy as np

from dmstrat import *

# DQN tutorial from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

class DQNAlgo:
    def __init__(self, obs_dim, act_dim):
        layer_sizes = [obs_dim, 32, act_dim]
        self.policy_net = mlp(layer_sizes)
        self.target_net = mlp(layer_sizes)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)
        self.steps_done = 0
        self.episodes_done = 0
        self.act_dim = act_dim

    def select_action(self, state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                # import ipdb; ipdb.set_trace()
                return self.policy_net(state).max(0)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.act_dim)]], dtype=torch.long)

    def update(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.stack([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.stack(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        # import ipdb; ipdb.set_trace()
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def end_episode(self):
        self.episodes_done += 1
        # Update the target network, copying all weights and biases in DQN
        if self.episodes_done % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


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

class DQNStrategy(Strategy):
    def __init__(self, buy_dqn=None):
        super().__init__() # calls reset()

        obs_dim = 58 + len(self.buys) # depends on impl. of self.buy_state()
        n_acts = len(self.buys)
        if buy_dqn is None:
            self.buy_dqn = DQNAlgo(obs_dim, n_acts)
        else:
            self.buy_dqn = buy_dqn

        self.learn = True # if False, do not update any of the strategies, and do not make exploratory moves
    @property
    def learn(self):
        return self._learn
    @learn.setter
    def learn(self, learn):
        self._learn = learn
        if learn:
            self.buy_dqn.policy_net.train()
        else:
            self.buy_dqn.policy_net.eval()
    def reset(self):
        super().reset()
        self.last_obs = None
        self.last_act = None
        self.last_rew = None
    def __getstate__(self):
        return {
            "buy_dqn": self.buy_dqn,
        }
    def __setstate__(self, state):
        self.__init__(
            buy_dqn=state['buy_dqn']
        )
    def start_game(self):
        self.bought = torch.zeros(len(self.buys), dtype=torch.float32)
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
        obs = self.buy_state(game, player)
        # fbn = np.array([not b.can_buy(game, player) for b in self.buys]) # forbidden, or invalid, actions
        act = self.buy_dqn.select_action(obs)
        rew = torch.tensor([0])
        # rew = buy.card.victory_points/100 if buy.card else 0
        if self.learn and self.last_obs is not None:
            self.buy_dqn.memory.push(self.last_obs, self.last_act, obs, self.last_rew)
            self.buy_dqn.update()
        self.last_obs = obs
        self.last_act = act
        self.last_rew = rew
        buy_idx = act.item()
        self.bought[buy_idx] += 1
        buy = self.buys[buy_idx]
        # For debugging during tournament play
        # if buy.card == Province and obs[56] != 0:
        #     import ipdb; ipdb.set_trace()
        return [ buy ]
    def rank_buys(self, game, player):
        obs = self.buy_state(game, player)
        fbn = np.array([not b.can_buy(game, player) for b in self.buys]) # forbidden, or invalid, actions
        act, val, logp, pi = self.buy_ac.step(obs, fbn)
        # For now, stick with a deterministic ranking, not a probabilistic one.
        buys = sorted_by(self.buys, pi.probs.tolist(), reverse=True)
        return buys + [Curse]
    def end_game(self, reward, game, player):
        super().end_game(reward, game, player)
        if self.learn:
            rew = torch.tensor([reward])
            self.buy_dqn.memory.push(self.last_obs, self.last_act, None, rew)
            self.buy_dqn.update()
            self.buy_dqn.end_episode()

def main_basic_dqn():
    players = 2
    popsize = players # some multiple of 2, 3, and 4
    strategies = [DQNStrategy() for _ in range(popsize)]

    CYCLES = 500
    GPS = 100
    for cycle in range(CYCLES): # expect to Ctrl-C to exit early
        # if cycle == CYCLES-1: # last one
        #     GPS = 1000
        #     for strategy in strategies:
        #         strategy.learn = False
        start = time.time()
        run_tournament(strategies, players, games_per_strategy=GPS)

        print(f"round {cycle}    {players} players    {GPS} games    {time.time() - start:.2f} sec " + ("="*70))
        for strategy in strategies:
            print(strategy)
        print("")

        save_strategies(strategies, "save_dqn")

if __name__ == '__main__':
    main_basic_dqn()
