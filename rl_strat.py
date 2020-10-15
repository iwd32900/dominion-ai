from collections import defaultdict
import pickle
import random
import time

from dmstrat import *

# random.seed(123456)

def argmax(xs):
    # Returns last one on ties, which is OK
    # return max((x,i) for i,x in enumerate(xs))[1]
    return max(zip(xs, range(len(xs))))[1]

def dot(xs, ys):
    accum = 0
    for x, y in zip(xs, ys):
        accum += x*y
    return accum

# This doesn't work right now, at least for this problem!
class TemporalDifferenceStrategy(Strategy):
    '''
    Based on the algorithms in Chapter 6 of Sutton & Barton,
    "Reinforcement Learning: An Introduction", 2020
    '''
    def __init__(self):
        super().__init__()
        self.q = defaultdict(lambda: [0.5]*len(self.buys)) # {state_idx: [action_values]}
        # Because our reward is 1 for a win and 0 for a loss,
        # our action values are estimated probabilities of winning the game
        # from the specified (state, action).
    def state_idx(self, game, player):
        # To start, learn a static strategy, regardless of game state:
        # return 0

        # With competent players, most games end within ~20 turns
        t = min(game.turn, 29)//5
        return t

        # In the basic game, 5 Gold = $15 is the max in one hand
        # actual $   0, 1, 2, 3, 4, 5, 6, 7, 8, 9 10 11 12 13 14 15
        BUY_POWER = [0, 0, 1, 2, 2, 3, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5]
        MAX_B = 5
        b = BUY_POWER[player.money] # calc_money() must have been called at start of buy phase
        return b

        s = player.calc_victory_points() - max(p.calc_victory_points() for p in game.players if p != player)
        if s < -13: s = -13
        elif s > 13: s = 13
        s += 13
        MAX_S = 26
        # delta score is [0,26], because 13 is 2 Provinces plus some
        p = game.stockpile[Province] # [0,8] in the 2-player game
        idx = (MAX_S+1)*( (MAX_B+1)*( (MAX_T+1)*p + t ) + b ) + s
        return idx
    def start_game(self):
        self.last_s = None
        self.last_a = None
    def accept_buy(self, buy, game, player):
        super().accept_buy(buy, game, player)
        # Convert current game state into an integer
        s = self.state_idx(game, player)
        # Lookup all action-values for current state
        q = self.q[s]
        a = self.buys.index(buy)
        if self.last_s is not None:
            # and all action-values for previous state
            last_q = self.q[self.last_s]
            # print("->"+str(last_q))
            last_a = self.last_a
            # Update the value of the last action taken
            # Sarsa:
            last_q[last_a] += alpha*(q[a] - last_q[last_a])
            # Q learning:
            # last_q[last_a] += alpha*(max(q) - last_q[last_a])
            # Expected Sarsa:
            # Use an epsilon-greedy policy; eps = 0.1 is common choice in the book.
            # pi = [eps/len(q)] * len(q)
            # pi[argmax(q)] += (1-eps)
            # E_q = dot(pi, q)
            # last_q[last_a] += alpha*(E_q - last_q[last_a])
            # print("<-"+str(last_q))
        # Save this move for next time
        self.last_s = s
        self.last_a = a
    def iter_buys(self, game, player):
        s = self.state_idx(game, player)
        # Lookup all action-values for current state
        q = self.q[s]
        # Use an epsilon-greedy policy; eps = 0.1 is common choice in the book.
        if random.random() < eps:
            # Sort the possible buys in random order
            b = list(self.buys)
            random.shuffle(b)
        else:
            # Follow the policy, sort actions by value
            b = sorted_by(self.buys, q, reverse=True)
        return b
    def end_game(self, reward, game, player):
        if self.last_s is None:
            assert False, "Games shouldn't end with no moves taken"
        last_q = self.q[self.last_s]
        last_a = self.last_a
        last_q[last_a] += alpha*(reward - last_q[last_a])
    def fmt_buys(self):
        lines = [ super().fmt_buys() ]
        lines.append(f'    Visited {len(self.q)} states')
        lines.append(f'    q = {dict(self.q)}')
        return '\n'.join(lines)

# Used instead of lambdas to allow pickling
class ConstArray:
    def __init__(self, value, length):
        self.value = value
        self.length = length
    def __call__(self):
        return [self.value] * self.length

class MonteCarloStrategy(Strategy):
    '''
    Based on the incremental algorithm in Chapter 5.6 of Sutton & Barton,
    "Reinforcement Learning: An Introduction", 2020
    '''
    def __init__(self, q={}, c={}):
        super().__init__()
        self.q = defaultdict(ConstArray(0.5, len(self.buys))) # {state_idx: [action_values]}
        self.q.update(q)
        # Because our reward is 1 for a win and 0 for a loss,
        # our action values are estimated probabilities of winning the game
        # from the specified (state, action).
        self.c = defaultdict(ConstArray(0, len(self.buys))) # {state_idx: [action_counts]}
        self.c.update(c)
        self.sa_hist = [] # [(state,action)]
        self.learn = True # if False, do not update any of the strategies, and do not make exploratory moves
    def __getstate__(self):
        return {
            "q": self.q,
            "c": self.c,
        }
    def __setstate__(self, state):
        self.__init__(q=state['q'], c=state['c'])
    def state_idx(self, game, player):
        # To start, learn a static strategy, regardless of game state:
        # return 0
        # With competent players, most games end within ~20 turns
        t = min(game.turn, 19)
        s = player.calc_victory_points() - max(p.calc_victory_points() for p in game.players if p != player)
        if s < -13: s = -13
        elif s > 13: s = 13
        p = min(game.stockpile[Province], 3) # [0,8] in the 2-player game
        return (t,s,p)
    def start_game(self):
        self.last_s = None
        # self.last_a = None
        self.sa_hist.clear()
    def accept_buy(self, buy, game, player):
        super().accept_buy(buy, game, player)
        # Convert current game state into an integer
        # s = self.state_idx(game, player)
        s = self.last_s # state could have changed due to us buying the card
        a = self.buys.index(buy)
        self.sa_hist.append((s, a))
    def iter_buys(self, game, player):
        self.last_s = s = self.state_idx(game, player)
        # Lookup all action-values for current state
        q = self.q[s]
        # Use an epsilon-greedy policy; eps = 0.1 is common choice in the book.
        if random.random() < eps and self.learn:
            # Sort the possible buys in random order
            b = list(self.buys)
            random.shuffle(b)
        else:
            # Follow the policy, sort actions by value
            b = sorted_by(self.buys, q, reverse=True)
        return b
    def end_game(self, reward, game, player):
        if not self.learn:
            return # do not update statistics
        if self.last_s is None:
            assert False, "Games shouldn't end with no moves taken"
        q = self.q
        c = self.c
        G = reward
        rev_hist = self.sa_hist[::-1]
        for t, (s, a) in enumerate(rev_hist):
            # First visit update rule - optional
            if (s, a) in rev_hist[t+1:]:
                continue
            c[s][a] += 1
            # c_sa = min(c[s][a], 100) # experiment: so learning doesn't get "stuck" on early experiences
            q[s][a] += (1 / c[s][a])*(G - q[s][a])
    def fmt_buys(self):
        lines = [ super().fmt_buys() ]
        lines.append(f'    Visited {len(self.q)} states')
        # lines.append(f'    q = {dict(self.q)}')
        return '\n'.join(lines)

# Not sure what the learning rate should be...
alpha = 1e-5 # needs to be smaller the deeper the states go in TD
eps = 0.1

def main_rl():
    players = 2
    popsize = 4 # some multiple of 2, 3, and 4
    # strategies = [TemporalDifferenceStrategy() for _ in range(popsize)]
    strategies = [MonteCarloStrategy() for _ in range(popsize)]

    CYCLES = 5000
    for cycle in range(CYCLES): # expect to Ctrl-C to exit early
        if cycle == CYCLES-1: # last one
            # Play final round without random exploratory moves
            for strategy in strategies:
                strategy.learn = False
        start = time.time()
        run_tournament(strategies, players, games_per_strategy=1000)
        print(f"round {cycle}    {players} players    {time.time() - start:.2f} sec " + ("="*70))
        for strategy in strategies[:1]:
            print(strategy)
        print("")
        with open("strategies.pkl", "wb") as f:
            pickle.dump(strategies, f)

        # global alpha, eps
        # alpha *= 0.950 # shrink learning rate over 100 rounds
        # eps *= 0.950 # slowly anneal toward greedy over 100 rounds
        # alpha *= 0.995 # shrink learning rate over 1000 rounds
        # eps *= 0.995 # slowly anneal toward greedy over 1000 rounds

if __name__ == '__main__':
    main_rl()

