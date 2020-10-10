from collections import Counter, defaultdict
import random
from dmgame import *

def sorted_by(target, keys, reverse=False):
    # Book specifies breaking ties at random -- maybe that's important?
    keys = [(k, random.random()) for k in keys]
    return [t for k, t in sorted(zip(keys, target), key=lambda x: x[0], reverse=reverse)]

class Strategy:
    def __init__(self):
        self.act_counts = Counter() # {Card: int}
        self.buy_counts = Counter() # {Card: int}
        self.act_counts_by_turn = Counter() # {(turn, Card): int}
        self.buy_counts_by_turn = Counter() # {(turn, Card): int}
        self.game_lengths = Counter() # {int: int}
        # Makes no sense to leave playable actions in the hand, if you bought them, so:
        self.actions = [PlayCard(c) for c in ALL_CARDS if c.is_action] #+ [EndActions()]
        # EndBuy is important:  allows us to declare some cards have negative value to us and should not be bought
        self.buys = [BuyCard(c) for c in ALL_CARDS] + [EndBuy()]
        self.reset()
    def reset(self):
        # Call once per tournament to reset statistics
        self.wins = 0
        self.suicides = 0 # we caused game to end but we lost -- converted probable loss into certain loss
        self.fitness = 0
        self.act_counts.clear()
        self.buy_counts.clear()
        self.act_counts_by_turn.clear()
        self.buy_counts_by_turn.clear()
        self.game_lengths.clear()
    def start_game(self):
        pass
    def accept_action(self, action, game, player):
        self.act_counts[action] += 1
        self.act_counts_by_turn[game.turn, action] += 1
    def accept_buy(self, buy, game, player):
        self.buy_counts[buy] += 1
        self.buy_counts_by_turn[game.turn, buy] += 1
    def end_game(self, reward, game, player):
        pass
    def iter_actions(self, game, player):
        a = list(self.actions)
        random.shuffle(a)
        return a
    def iter_buys(self, game, player):
        b = list(self.buys)
        random.shuffle(b)
        return b
    def fmt_actions(self):
        return 'not implemented'
    def fmt_buys(self):
        # Refomat into more useful but probably slower form
        cbt = defaultdict(Counter)
        for (turn,card), count in self.buy_counts_by_turn.items():
            cbt[turn][card] = count

        # Show every line for turns played
        sorted_buys = sorted(self.buys, key=lambda m: (getattr(m.card, 'cost', 0), self.buy_counts[m]), reverse=True)
        lines = ['']
        for ii in range(len(cbt)):
            # sorted_buys = sorted(self.buys, key=lambda m: cbt[ii][m], reverse=True)
            line = '   '.join(f"{cbt[ii][m]} {m}" for m in sorted_buys if cbt[ii][m] > 0)
            if sum(cbt[ii].values()) > 0:
                # avoid blank lines for sequences never played
                lines.append(f'    {ii+1:2d}:   '+line)

        sorted_buys = sorted(self.buys, key=lambda m: (getattr(m.card, 'cost', 0), self.buy_counts[m]), reverse=True)
        n = sum(self.game_lengths.values()) # number of games played
        line = '   '.join(f"{self.buy_counts[m]/n:.1f} {m}" for m in sorted_buys if self.buy_counts[m] > 0)
        lines.append(f'    Avg   '+line)

        return '\n'.join(lines)
    def __str__(self):
        lens = ','.join(str(k) for k,v in self.game_lengths.most_common(3))
        minlen = min(self.game_lengths.keys())
        maxlen = max(self.game_lengths.keys())
        return "\n".join([
            f"{self.__class__.__name__}    wins {self.wins}    suicides {self.suicides}    fitness {self.fitness}    game len {lens} ({minlen} - {maxlen})",
            f"  actions: {self.fmt_actions()}",
            f"  buys:    {self.fmt_buys()}",
        ])

