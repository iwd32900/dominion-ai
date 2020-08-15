from collections import Counter
import random
import time

# random.seed(123456)

class Card:
    cost = 0
    is_action = False
    victory_points = 0
    money_in_hand = 0
    money_when_played = 0
    actions_when_played = 0
    buys_when_played = 0
    cards_when_played = 0
    def __init__(self, name, **kwargs):
        self.name = name
        self.__dict__.update(kwargs)
    def play(self, game, player):
        # money is handled separately
        player.actions += self.actions_when_played
        player.buys += self.buys_when_played
        if self.cards_when_played:
            player.draw_cards(self.cards_when_played)
    def __str__(self):
        return self.name

# Singleton objects
Copper = Card("Copper", cost=0, money_in_hand=1)
Silver = Card("Silver", cost=3, money_in_hand=2)
Gold   = Card("Gold",   cost=6, money_in_hand=3)

Estate   = Card("Estate",   cost=2, victory_points=1)
Duchy    = Card("Duchy",    cost=5, victory_points=3)
Province = Card("Province", cost=8, victory_points=6)

Gardens = Card("Gardens", cost=4) # vic pts depends on final deck size

MINIMAL_CARDS = [Copper, Silver, Gold, Estate, Duchy, Province]
ALL_CARDS = MINIMAL_CARDS + [Gardens]
STARTING_STOCKPILE = {
    Copper: 61,
    Silver: 41,
    Gold: 31,
    Estate: 25,
    Duchy: 13,
    Province: 15,
    Gardens: 13,
}
STARTING_DECK = [Copper]*7 + [Estate]*3

class Move:
    def can_move(self, game, player):
        raise NotImplementedError()
    def do_move(self, game, player):
        raise NotImplementedError()

class BuyCard(Move):
    def __init__(self, card):
        self.card = card
    def can_move(self, game, player):
        return (
            self.card.cost <= player.money
            and player.buys >= 1
            and game.stockpile[self.card] >= 1
        )
    def do_move(self, game, player):
        player.money -= self.card.cost
        player.buys -= 1
        player.discard.append(self.card)
        game.stockpile[self.card] -= 1
    def __str__(self):
        return "buy "+str(self.card)

class PlayCard(Move):
    def __init__(self, card):
        self.card = card
    def can_move(self, game, player):
        return (
            self.card in player.hand
            and player.actions >= 1
            and self.card.is_action
        )
    def __str__(self):
        return "play "+str(self.card)

class EndActions(Move):
    def can_move(self, game, player):
        return player.actions >= 1
    def do_move(self, game, player):
        player.actions = 0
    def __str__(self):
        return "actions -> buying"

class EndBuy(Move):
    def can_move(self, game, player):
        return True
    def do_move(self, game, player):
        player.actions = 0
        player.buys = 0
        player.draw_hand()
    def __str__(self):
        return "end turn"

def add_idx(*args):
    ii = 0
    for seq in args:
        for el in seq:
            el.idx = ii
            ii += 1
    return ii

class FixedRankStrategy:
    def __init__(self, weights=None):
        self.actions = [PlayCard(c) for c in ALL_CARDS if c.is_action] + [EndActions()]
        self.buys = [BuyCard(c) for c in ALL_CARDS] + [EndBuy()]
        num_idx = add_idx(self.actions, self.buys)
        if weights:
            self.weights = list(weights)
        else:
            self.weights = [random.random() for _ in range(num_idx)]
        assert len(self.weights) == num_idx
        self.sorted_actions = sorted(self.actions, key=lambda x: self.weights[x.idx])
        self.sorted_buys = sorted(self.buys, key=lambda x: self.weights[x.idx])
    def iter_actions(self, game, player):
        yield from self.sorted_actions
    def iter_buys(self, game, player):
        yield from self.sorted_buys

class Player:
    def __init__(self, name, deck, strategy):
        self.name = name
        self.deck = list(deck)
        self.strategy = strategy
        assert len(self.deck) == 10
        self.hand = []
        self.played = [] # this turn
        self.discard = []
        self.draw_hand()
    def draw_hand(self):
        self.discard.extend(self.hand)
        self.hand.clear()
        self.discard.extend(self.played)
        self.played.clear()
        self.actions = 1
        self.buys = 1
        self.money = 0
        self.draw_cards(5)
    def draw_cards(self, num):
        for ii in range(num):
            if len(self.deck) == 0:
                self.deck, self.discard = self.discard, self.deck
                random.shuffle(self.deck)
            self.hand.append(self.deck.pop())
    def calc_money(self):
        self.money = (
            sum(c.money_when_played for c in self.played)
            + sum(c.money_in_hand for c in self.hand)
        )
    def all_cards(self):
        yield from self.deck
        yield from self.hand
        yield from self.played
        yield from self.discard
    def calc_victory_points(self):
        all_cards = list(self.all_cards())
        base_pts = sum(c.victory_points for c in all_cards)
        num_gardens = sum(1 for c in all_cards if c == Gardens)
        garden_pts = num_gardens * (len(all_cards)//10)
        return base_pts + garden_pts

class Game:
    def __init__(self, players, stockpile):
        self.players = list(players)
        self.stockpile = dict(stockpile)
        for player in players:
            for card in player.all_cards():
                assert self.stockpile[card] >= 1
                self.stockpile[card] -= 1
    def is_over(self):
        exhausted_cards = [k for k, v in self.stockpile.items() if v <= 0]
        return (
            self.stockpile[Province] <= 0
            or len(exhausted_cards) >= 3
        )
    def run(self):
        game = self
        for game_round in range(100):
            # print(f"Round {game_round}")
            for player in game.players:
                if game.is_over():
                    return
                # print(f"  Player {player.name}    pts = {player.calc_victory_points()}")
                # print(f"    hand = {', '.join(str(x) for x in player.hand)}")
                while player.actions > 0:
                    for action in player.strategy.iter_actions(game, player):
                        if action.can_move(game, player):
                            # print(f"    {action}")
                            action.do_move(game, player)
                            break
                player.calc_money()
                while player.buys > 0 and player.money > 0:
                    for buy in player.strategy.iter_buys(game, player):
                        if buy.can_move(game, player):
                            # print(f"    {buy}")
                            buy.do_move(game, player)
                            break
                player.draw_hand()

def print_stockpile(stockpile):
    for card, count in stockpile.items():
        print(f"  {count} {card}")

def run_tournament(strategies, players_per_game=3, games_per_strategy=50):
    popsize = len(strategies)
    assert popsize % players_per_game == 0, "Popsize must be evenly divisible by number of players"

    for strategy in strategies:
        strategy.fitness = 0

    for _ in range(games_per_strategy):
        random.shuffle(strategies)
        for ii in range(0, popsize, players_per_game):
            players = [Player(str(jj+1), STARTING_DECK, strategies[ii+jj]) for jj in range(players_per_game)]
            game = Game(players, STARTING_STOCKPILE)
            game.run()
            for player in players:
                player.strategy.fitness += player.calc_victory_points()

    strategies.sort(key=lambda x: x.fitness, reverse=True)

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

def mutate(w, rate):
    for ii in range(len(w)):
        if random.random() < rate:
            w[ii] = random.random()

def evolve(strategies):
    popsize = len(strategies)
    parents = strategies[:popsize//5]
    newstrat = list(parents)
    while len(newstrat) < popsize:
        p1 = random.choice(parents)
        p2 = random.choice(parents)
        w = cross(p1.weights, p2.weights)
        mutate(w, 0.05)
        newstrat.append(FixedRankStrategy(w))
    return newstrat

def main():
    popsize = 12 * 32 # some multiple of 2, 3, and 4
    strategies = [FixedRankStrategy() for _ in range(popsize)]

    for cycle in range(100): # expect to Ctrl-C to exit early
        start = time.time()
        run_tournament(strategies)
        strategy = strategies[0]
        print(f"round {cycle}    fitness {strategy.fitness}    {time.time() - start:.2f} sec")
        print(f"  actions: {', '.join(str(x) for x in strategy.sorted_actions)}")
        print(f"  buys:    {', '.join(str(x) for x in strategy.sorted_buys)}")
        strategies = evolve(strategies)

main()

