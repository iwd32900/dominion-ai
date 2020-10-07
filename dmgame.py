import random
from dmcards import *

STARTING_STOCKPILE = {
    Copper: 60,
    Silver: 40,
    Gold: 30,
    Estate: 0,
    Duchy: 0,
    Province: 0,
    Curse: 0,
    Gardens: 0,
    Festival: 10,
    Laboratory: 10,
    Market: 10,
    Smithy: 10,
    Village: 10,
    Woodcutter: 10,
    Witch: 10,
    Adventurer: 10,
    Bureaucrat: 10,
    CouncilRoom: 10,
    Mine: 10,
    Moat: 10,
    Chancellor: 10,
    Thief: 10,
}
STARTING_DECK = [Copper]*7 + [Estate]*3
MAX_TURNS = 50

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
            #and player.buys >= 1 # already checked in outer loop
            and game.stockpile[self.card] >= 1
        )
    def do_move(self, game, player):
        player.money -= self.card.cost
        player.buys -= 1
        player.discard.append(self.card)
        game.stockpile[self.card] -= 1
    def __str__(self):
        return ""+str(self.card)

class PlayCard(Move):
    def __init__(self, card):
        self.card = card
    # def can_move(self, game, player):
    #     return (
    #         self.card in player.hand
    #         #and player.actions >= 1 # already checked in outer loop
    #         #and self.card.is_action # already checked by strategy
    #     )
    def do_move(self, game, player):
        # This order is important: while playing, a card is neither part of the hand, nor the discards
        player.hand.remove(self.card)
        player.actions -= 1
        self.card.play(game, player)
        player.played.append(self.card)
    def __str__(self):
        return ""+str(self.card)

class EndActions(Move):
    card = None
    def can_move(self, game, player):
        return player.actions >= 1
    def do_move(self, game, player):
        player.actions = 0
    def __str__(self):
        return "END"

class EndBuy(Move):
    card = None
    def can_move(self, game, player):
        return True
    def do_move(self, game, player):
        player.actions = 0
        player.buys = 0
        player.draw_hand()
    def __str__(self):
        return "END"

class Player:
    def __init__(self, name, deck, strategy):
        self.name = name
        self.deck = list(deck)
        random.shuffle(self.deck)
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
        self.reveal_cards(num, self.hand)
    def reveal_cards(self, num, into_list):
        for ii in range(num):
            if len(self.deck) == 0:
                self.deck, self.discard = self.discard, self.deck
                random.shuffle(self.deck)
                if len(self.deck) == 0:
                    # Cards played this turn are not eligible to shuffle back in.
                    # assert len(self.discard) == 0
                    break # all cards are in hand already!
            into_list.append(self.deck.pop())
        return into_list
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
        self.turn = 0
        self.last_player = None
        game = self
        for player in game.players:
            player.strategy.start_game()

        self.run_loop()

        for player in game.players:
            vp = player.calc_victory_points()
            player.strategy.fitness += vp
            score = (vp, -player.turns_played)
            best = max((p.calc_victory_points(), -p.turns_played) for p in game.players if p != player)
            if score > best: reward = 1 # win
            elif score == best:  reward = 0.5 # tie
            else: reward = 0 # loss
            player.strategy.wins += reward
            if player == game.last_player and reward == 0:
                player.strategy.suicides += 1
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
                while player.actions > 0:
                    for action in player.strategy.iter_actions(game, player):
                        # Inlining this check for speed (hopefully):
                        if action.card in player.hand: # action.can_move(game, player)
                            # print(f"    {action}")
                            action.do_move(game, player)
                            player.strategy.accept_action(action, game, player)
                            break
                    else:
                        break # no playable actions
                player.calc_money()
                while player.buys > 0 and player.money > 0:
                    for buy in player.strategy.iter_buys(game, player):
                        if buy.can_move(game, player):
                            # print(f"    {buy}")
                            buy.do_move(game, player)
                            player.strategy.accept_buy(buy, game, player)
                            break
                    else:
                        break # no buyable cards
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
        random.shuffle(strategies)
        for ii in range(0, popsize, players_per_game):
            players = [Player(str(jj+1), STARTING_DECK, strategies[ii+jj]) for jj in range(players_per_game)]
            game = Game(players, get_starting_stockpile(players_per_game))
            game.run()

    strategies.sort(key=lambda x: (x.wins, x.fitness), reverse=True)
