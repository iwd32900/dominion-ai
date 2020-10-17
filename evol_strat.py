from collections import Counter, defaultdict
import pickle
import random
import time

from dmstrat import *

# random.seed(123456)

# These would be lambdas, but lambdas don't pickle
def zero(): return 0
def normal_05(): return random.normalvariate(0, 0.05)

class LinearRankStrategy(Strategy):
    def __init__(self, weights=None):
        super().__init__()
        self.weight_dist = []
        # A typical game lasts 10-20 turns.  With a coef of 0.05, a card can
        # go from 0 to 1 (or 1 to 0) over the course of a game.
        # linear_coef = zero # fixed-rank strategy
        linear_coef = normal_05

        self.act_idx = ai = {} # {action Card: int position in weights}
        self.buy_idx = bi = {} # {Card: int position in weights}
        for act in self.actions:
            ai[act] = len(self.weight_dist)
            self.weight_dist.append(random.random)
            self.weight_dist.append(linear_coef)
        for buy in self.buys:
            bi[buy] = len(self.weight_dist)
            self.weight_dist.append(random.random)
            self.weight_dist.append(linear_coef)
        num_idx = len(self.weight_dist)
        if weights:
            self.weights = list(weights)
        else:
            self.weights = [self.weight_dist[ii]() for ii in range(num_idx)]
        assert len(self.weights) == num_idx
        # Raw sort order is used for things outside the normal Action phase, like Throne Rooms.
        # For now, don't include linear term for actions
        raw_act_key = lambda x: self.weights[ai[x]] #+ game.turn*self.weights[ai[x]+1]
        self.raw_sorted_actions = sorted(self.actions, key=raw_act_key)
        # Heuristic:
        # When playing a normal turn, cards that give extra actions come first.
        # Within those, cards that give extra draws come first, so there are more options.
        def act_key(x):
            if x.actions_when_played:
                return (0, -x.cards_when_played, self.weights[ai[x]])
            else:
                return (1, 0, self.weights[ai[x]])
        self.sorted_actions = sorted(self.actions, key=act_key)
        self.sorted_buys = []
        for game_turn in range(MAX_TURNS):
            buy_key = lambda x: self.weights[bi[x]] + game_turn*self.weights[bi[x]+1]
            self.sorted_buys.append(sorted(self.buys, key=buy_key))
    def __getstate__(self):
        return {
            "weights": self.weights,
        }
    def __setstate__(self, state):
        self.__init__(weights=state['weights'])
    def iter_actions(self, game, player):
        return self.sorted_actions
    def iter_actions_raw(self, game, player):
        return self.raw_sorted_actions
    def iter_buys(self, game, player):
        return self.sorted_buys[game.turn]
    # def get_action(self, game, player):
    #     for a in self.sorted_actions:
    #         # if a.can_play(game, player):
    #         if a in self.hand:
    #             return a
    #     return END
    # def get_buy(self, game, player):
    #     for b in self.sorted_buys[game.turn]:
    #         if b.can_buy(game, player):
    #             return b
    #     return END
    def fmt_actions(self):
        return '   '.join(f"{self.act_counts[m]} {m}" for m in self.sorted_actions if self.act_counts[m] > 0)
    def fmt_buys(self):
        # Refomat into more useful but probably slower form
        cbt = defaultdict(Counter)
        for (turn,card), count in self.buy_counts_by_turn.items():
            cbt[turn][card] = count

        used_buys = [m for m in self.buys if self.buy_counts[m] > 0]
        sorted_used_buys = []
        for sb in self.sorted_buys:
            sorted_used_buys.append([m for m in sb if self.buy_counts[m] > 0])

        # Show every line for turns played
        lines = ['']
        for ii, buys in enumerate(sorted_used_buys):
            line = '   '.join(f"{cbt[ii][m]} {m}" for m in buys if cbt[ii][m] > 0)
            if sum(cbt[ii].values()) > 0:
                # avoid blank lines for sequences never played
                lines.append(f'    {ii+1:2d}:   '+line)

        n = sum(self.game_lengths.values()) # number of games played
        line = '   '.join(f"{self.buy_counts[m]/n:.1f} {m}" for m in self.sorted_buys[0] if self.buy_counts[m] > 0)
        lines.append(f'    Avg   '+line)

        return '\n'.join(lines)

def cross(w1, w2):
    """
    Given two vectors, return a new vector that inherits either w1[i] or w2[i] for all i.

    Treating the whole vector as a single chromosome
    allows some crosses to be mostly one parent and a little of the other, good for fine tuning.
    But it creates linkage between adjacent elements, which is particularly problematic if the order is arbitrary.

    Treating each element as an independently-assorting chromosome
    fixes the linkage problems,
    but means offspring are always close to 50/50 mixes of both parents, which may be disruptive.

    This algorithm is non-biological, but allows various mixes of the parents without linkage.
    """
    threshold = random.random()
    w3 = [e1 if random.random() < threshold else e2 for e1, e2 in zip(w1, w2)]
    return w3

def mutate(w, weight_dist, rate):
    for ii in range(len(w)):
        if random.random() < rate:
            w[ii] = weight_dist[ii]()

def evolve(strategies):
    popsize = len(strategies)
    parents = strategies[:popsize//5]
    newstrat = list(parents)
    while len(newstrat) < popsize:
        p1 = random.choice(parents)
        p2 = random.choice(parents)
        assert p1.__class__ == p2.__class__
        w = cross(p1.weights, p2.weights)
        mutate(w, p1.weight_dist, 0.05)
        newstrat.append(p1.__class__(w))
    return newstrat

def main_evol():
    players = 2
    popsize = 12 * 32 # some multiple of 2, 3, and 4
    strategies = [LinearRankStrategy() for _ in range(popsize)]

    CYCLES = 100
    GPS = 100
    for cycle in range(CYCLES): # expect to Ctrl-C to exit early
        if cycle == CYCLES-1: # last one
            GPS = 1000 # better ranking before final save
        start = time.time()
        run_tournament(strategies, players, games_per_strategy=GPS)
        print(f"round {cycle}    {players} players    {GPS} games    {time.time() - start:.2f} sec " + ("="*70))
        for strategy in strategies[:3]:
            print(strategy)
        print("")
        save_strategies(strategies[:10], "save_evol")
        strategies = evolve(strategies)

if __name__ == '__main__':
    main_evol()

