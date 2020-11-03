import random
from dmcards import *

try:
    import numpy
    # Numpy shuffle is WAY faster than Python shuffle
    shuffle = numpy.random.shuffle
except ModuleNotFoundError:
    shuffle = random.shuffle

STARTING_STOCKPILE = {
    Copper: 60,
    Silver: 40,
    Gold: 30,
    Estate: 0,
    Duchy: 0,
    Province: 0,
    Curse: 0,
    Gardens: 0,
    # Adventurer: 10,
    # Bureaucrat: 10,
    # Cellar: 10,
    # Chancellor: 10,
    # Chapel: 10,
    # CouncilRoom: 10,
    # Festival: 10,
    # Laboratory: 10,
    # Market: 10,
    # Mine: 10,
    # Moat: 10,
    # Smithy: 10,
    # Thief: 10,
    # Village: 10,
    # Witch: 10,
    # Woodcutter: 10,
}
for card in ALL_CARDS:
    STARTING_STOCKPILE.setdefault(card, 10)
del card
STARTING_DECK = [Copper]*7 + [Estate]*3
MAX_TURNS = 50

class Player:
    def __init__(self, name, deck, strategy):
        self.name = name
        self.deck = list(deck) # note, we draw from the back of the deck!
        shuffle(self.deck)
        self.strategy = strategy
        assert len(self.deck) == 10
        self.turns_played = 0
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
        # Can't just assign, because e.g. Smithy must add to existing hand
        # self.hand = self.reveal_cards(num)
        self.hand.extend(self.reveal_cards(num))
    def reveal_cards(self, num):
        if num <= 0: return []
        # Otherwise, we return the whole contents of the deck b/c of Python slicing!
        # Significantly faster than pop/append one card at a time!
        deck = self.deck
        if num > len(deck):
            discard = self.discard
            shuffle(discard)
            # we pull from the back of the deck, so discards must go in front
            discard.extend(deck)
            deck.clear()
            # swap deck and discard
            self.discard = deck
            deck = self.deck = discard
        # Slice is faster than extend(), and is immutable (COW?)
        # into_list.extend(deck[-num:]) # this may be less than num cards
        into_list = deck[-num:]
        deck[-num:] = []
        return into_list
    def calc_money(self):
        # This is slightly faster than sum() with generators in CPython,
        # though slightly slower in PyPy.  (I'm optimizing for CPython.)
        m = 0
        for c in self.hand:
            m += c.money_in_hand
        self.money += m
    def all_cards(self):
        # This is faster than extend() in PyPy, and the same in CPython.
        yield from self.deck
        yield from self.hand
        yield from self.played
        yield from self.discard
    def calc_victory_points(self):
        base_pts = 0
        num_cards = 0
        num_gardens = 0
        for c in self.all_cards():
            num_cards += 1
            if c.victory_points:
                base_pts += c.victory_points
            elif c == Gardens:
                num_gardens += 1
        garden_pts = num_gardens * (num_cards//10)
        return base_pts + garden_pts

class Game:
    def __init__(self, players, stockpile):
        self.players = list(players)
        self.stockpile = dict(stockpile)
        self.turn = 0
        self.last_player = None
        for player in players:
            for card in player.all_cards():
                assert self.stockpile[card] >= 1
                self.stockpile[card] -= 1
    def is_over(self):
        # Cards can be removed from the stockpile by certain actions,
        # so there's really no substitute for re-checking each time:
        exhausted_cards = 0
        for v in self.stockpile.values():
            if v <= 0:
                exhausted_cards += 1
        return (
            self.stockpile[Province] <= 0
            or exhausted_cards >= 3
        )
    def run(self):
        game = self
        for player in game.players:
            player.strategy.start_game()

        self.run_loop()

        vic_pts = [(p.calc_victory_points(), -p.turns_played) for p in game.players]
        for ii, player in enumerate(game.players):
            score = vic_pts[ii]
            player.strategy.fitness += score[0] # victory points
            best = max(vp for jj, vp in enumerate(vic_pts) if ii != jj)
            if score > best: reward = 1 # win
            elif score == best:  reward = 0.5 # tie
            else: reward = 0 # loss
            player.strategy.wins += reward
            if player == game.last_player and reward == 0:
                player.strategy.suicides += 1
                # reward = -0.5 # explicitly penalize suicidal losses
            player.strategy.game_lengths[player.turns_played] += 1
            player.strategy.end_game(reward, game, player)
    def run_loop(self):
        game = self
        for turn in range(MAX_TURNS):
            game.turn = turn # starts from 0 to make LinearRankStrategy work better
            # print(f"Round {game.turn}")
            for player in game.players:
                if game.is_over():
                    return
                # print(f"  Player {player.name}    pts = {player.calc_victory_points()}")
                # print(f"    hand = {', '.join(str(x) for x in player.hand)}")
                # For reasons I *really* can't explain, the "iter" approach
                # is significantly faster than the "get" approach (at least for evol. strat.)
                while player.actions > 0:
                    for action in player.strategy.iter_actions(game, player):
                        # Inlining this check for speed (hopefully):
                        if action in player.hand: # action.can_play(game, player)
                            # print(f"    {action}")
                            action.play(game, player)
                            player.strategy.accept_action(action, game, player)
                            break
                    else:
                        break # no playable actions
                    # Variant implementation - get single legal action
                    # action = player.strategy.get_action(game, player)
                    # # assert action.can_play(game, player)
                    # # print(f"    {action}")
                    # player.strategy.accept_action(action, game, player)
                    # # if action == END: break
                    # action.play(game, player)
                # player.actions = 0 # not strictly needed
                player.calc_money()
                while player.buys > 0: #and player.money > 0: # some buys are zero cost!
                    for buy in player.strategy.iter_buys(game, player):
                        if buy.can_buy(game, player):
                            # print(f"    {buy}")
                            buy.buy(game, player)
                            player.strategy.accept_buy(buy, game, player)
                            break
                    else:
                        break # no buyable cards
                    # Variant implementation - get single legal buy
                    # buy = player.strategy.get_buy(game, player)
                    # # assert buy.can_buy(game, player)
                    # # print(f"    {buy}")
                    # player.strategy.accept_buy(buy, game, player)
                    # # if buy == END: break
                    # buy.buy(game, player)
                # player.buys = 0 # not strictly needed
                player.draw_hand()
                player.turns_played += 1
                game.last_player = player
        # exits via premature return -- this line never reached unless game runs long!

def get_starting_stockpile(num_players):
    sp = dict(STARTING_STOCKPILE)
    if num_players == 2:
        sp[Estate] = 8 + 6 # each player starts with 3
        sp[Duchy] = sp[Province] = sp[Gardens] = 8
        sp[Curse] = 10
    elif num_players == 3:
        sp[Estate] = 12 + 9 # each player starts with 3
        sp[Duchy] = sp[Province] = sp[Gardens] = 12
        sp[Curse] = 20
    elif num_players == 4:
        sp[Estate] = 12 + 12 # each player starts with 3
        sp[Duchy] = sp[Province] = sp[Gardens] = 12
        sp[Curse] = 30
    else:
        assert False
    return sp

def print_stockpile(stockpile):
    for card, count in stockpile.items():
        print(f"  {count} {card}")

def run_tournament(strategies, players_per_game=3, games_per_strategy=100):
    popsize = len(strategies)
    assert popsize % players_per_game == 0, "Popsize must be evenly divisible by number of players"

    for strategy in strategies:
        strategy.reset()

    for _ in range(games_per_strategy):
        shuffle(strategies)
        for ii in range(0, popsize, players_per_game):
            players = [Player(str(jj+1), STARTING_DECK, strategies[ii+jj]) for jj in range(players_per_game)]
            game = Game(players, get_starting_stockpile(players_per_game))
            game.run()

    strategies.sort(key=lambda x: (x.wins, x.fitness), reverse=True)
