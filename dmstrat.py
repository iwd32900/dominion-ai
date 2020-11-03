from collections import Counter, defaultdict
import datetime
import pickle
import random
from dmgame import *

# Use one consistent timestamp throughout the program execution
nowstamp = datetime.datetime.now().isoformat(timespec='seconds')
def save_strategies(strategies, basename):
    filename = f"{basename}_{nowstamp}.pkl"
    with open(filename, "wb") as f:
        pickle.dump({
            "all_cards": [str(c) for c in ALL_CARDS],
            "strategies": strategies,
        }, f)
    return filename

def load_strategies(filename):
    with open(filename, "rb") as f:
        obj = pickle.load(f)
        if isinstance(obj, list):
            return obj
        return obj['strategies']

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
        self.deck_counts = Counter() # {Card: int}
        self.game_lengths = Counter() # {int: int}
        # Makes no sense to leave playable actions in the hand, if you bought them, so:
        self.actions = [c for c in ALL_CARDS if c.is_action] #+ [END]
        # EndBuy is important:  allows us to declare some cards have negative value to us and should not be bought
        self.buys = [c for c in ALL_CARDS] + [END]
        self.reset()
    def reset(self):
        # Call once per tournament to reset statistics
        self.wins = 0
        self.suicides = 0 # we caused game to end but we lost -- converted probable loss into certain loss
        self.fitness = 0
        self.multiple_actions = 0 # action moves with >1 action to choose from
        self.multiple_terminal_actions = 0 # action moves with >1 terminal action to choose from
        self.act_counts.clear()
        self.buy_counts.clear()
        self.act_counts_by_turn.clear()
        self.buy_counts_by_turn.clear()
        self.deck_counts.clear()
        self.game_lengths.clear()
    def start_game(self):
        pass
    def iter_actions(self, game, player):
        # For games with 10 action cards, most of them won't be in our hand!
        act = list(c for c in player.hand if c.is_action)
        if len(act) <= 1:
            return act
        # record keeping
        self.multiple_actions += 1
        term_act = 0
        for a in act:
            if not a.actions_when_played:
                term_act += 1
        if term_act > 1:
            self.multiple_terminal_actions += 1
        # Heuristic:
        # When playing a normal turn, cards that give extra actions come first.
        # Within those, cards that give extra draws come first, so there are more options.
        def act_key(x):
            if x.actions_when_played:
                return (0, -x.cards_when_played, random.random())
            else:
                return (1, 0, random.random())
        act.sort(key=act_key)
        return act
    def accept_action(self, action, game, player):
        self.act_counts[action] += 1
        self.act_counts_by_turn[game.turn, action] += 1
    def iter_buys(self, game, player):
        b = list(self.buys)
        random.shuffle(b)
        return b
    def accept_buy(self, buy, game, player):
        self.buy_counts[buy] += 1
        self.buy_counts_by_turn[game.turn, buy] += 1
    def end_game(self, reward, game, player):
        self.deck_counts.update(player.all_cards())
    def fmt_actions(self):
        n = sum(self.game_lengths.values()) # number of games played
        sorted_actions = sorted(self.actions, key=lambda m: self.act_counts[m], reverse=True)
        return '   '.join(f"{self.act_counts[m]/n:.1f} {m}" for m in sorted_actions if self.act_counts[m] > 0)
    def fmt_buys(self):
        # Refomat into more useful but probably slower form
        cbt = defaultdict(Counter)
        for (turn,card), count in self.buy_counts_by_turn.items():
            cbt[turn][card] = count

        # Show every line for turns played
        sorted_buys = sorted(self.buys, key=lambda m: (getattr(m, 'cost', 0), self.buy_counts[m]), reverse=True)
        lines = ['']
        for ii in range(len(cbt)):
            # sorted_buys = sorted(self.buys, key=lambda m: cbt[ii][m], reverse=True)
            line = '   '.join(f"{cbt[ii][m]} {m}" for m in sorted_buys if cbt[ii][m] > 0)
            if sum(cbt[ii].values()) > 0:
                # avoid blank lines for sequences never played
                lines.append(f'    {ii+1:2d}:   '+line)

        sorted_buys = sorted(self.buys, key=lambda m: (getattr(m, 'cost', 0), self.buy_counts[m]), reverse=True)
        n = sum(self.game_lengths.values()) # number of games played
        line = '   '.join(f"{self.buy_counts[m]/n:.1f} {m}" for m in sorted_buys if self.buy_counts[m] > 0)
        lines.append(f'    Buys   '+line)
        line = '   '.join(f"{self.deck_counts[m]/n:.1f} {m}" for m in sorted_buys if self.deck_counts[m] > 0)
        lines.append(f'    Deck   '+line)

        return '\n'.join(lines)
    def __str__(self):
        n = sum(self.game_lengths.values()) # number of games played
        lens = ','.join(str(k) for k,v in self.game_lengths.most_common(3))
        minlen = min(self.game_lengths.keys())
        maxlen = max(self.game_lengths.keys())
        return "\n".join([
            f"{self.__class__.__name__}    wins% {100*self.wins/n:.2f}    suicides% {100*self.suicides/n:.2f}    fitness {self.fitness/n:.2f}    game len {lens} ({minlen} - {maxlen})",
            f"  actions: {self.fmt_actions()}    (multiples: {self.multiple_actions/n:.2f} / {self.multiple_terminal_actions/n:.2f})",
            f"  buys:    {self.fmt_buys()}",
        ])

