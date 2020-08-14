from collections import Counter
import random
random.seed(123456)

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

ALL_CARDS = [Copper, Silver, Gold, Estate, Duchy, Province]
STARTING_STOCKPILE = {
    Copper: 61,
    Silver: 41,
    Gold: 31,
    Estate: 25,
    Duchy: 13,
    Province: 15,
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

class FixedRankStrategy:
    def __init__(self):
        self.actions = [PlayCard(c) for c in ALL_CARDS if c.is_action] + [EndActions()]
        self.buys = [BuyCard(c) for c in ALL_CARDS] + [EndBuy()]
        for action in self.actions:
            action.weight = random.random()
        for buy in self.buys:
            buy.weight = random.random()
        self.actions.sort(key=lambda x: x.weight)
        self.buys.sort(key=lambda x: x.weight)
    def iter_actions(self, game, player):
        yield from self.actions
    def iter_buys(self, game, player):
        yield from self.buys

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
        return sum(c.victory_points for c in self.all_cards())

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

def make_game_01():
    num_players = 3
    players = []
    for ii in range(num_players):
        player = Player(str(ii+1), STARTING_DECK, FixedRankStrategy())
        print(f"Player {player.name}:")
        print(f"  actions: {', '.join(str(x) for x in player.strategy.actions)}")
        print(f"  buys:    {', '.join(str(x) for x in player.strategy.buys)}")
        players.append(player)

    game = Game(players, STARTING_STOCKPILE)

    print_stockpile(game.stockpile)

    game.run()

    print(f"Deck:")
    print_stockpile(game.stockpile)
    for player in game.players:
        print(f"Player {player.name}    pts = {player.calc_victory_points()}")
        print_stockpile(Counter(player.all_cards()))

def make_game_02():
    popsize = 12 * 32 # some multiple of 2, 3, and 4
    games_per_strategy = 20
    players_per_game = 3

    strategies = [FixedRankStrategy() for _ in range(popsize)]
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
    strategy = strategies[0]
    print(f"  actions: {', '.join(str(x) for x in strategy.actions)}")
    print(f"  buys:    {', '.join(str(x) for x in strategy.buys)}")


make_game_02()

