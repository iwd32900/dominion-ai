from collections import Counter, defaultdict
import itertools
import random
import time

# random.seed(123456)

class Card:
    cost = 0
    is_action = False
    is_victory = False
    is_treasure = False
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

class WitchCard(Card):
    def __init__(self):
        super().__init__("Witch", cost=5, is_action=True, cards_when_played=2)
    def play(self, game, attacker):
        super().play(game, attacker)
        for defender in game.players:
            if defender == attacker or game.stockpile[Curse] < 1 or (Moat in defender.hand):
                continue
            defender.discard.append(Curse)
            game.stockpile[Curse] -= 1

class AdventurerCard(Card):
    def __init__(self):
        super().__init__("Adventurer", cost=6, is_action=True)
    def play(self, game, player):
        super().play(game, player)
        treasures = 0
        # After 50 tries, we assume there are no more treasures in the deck!
        for _ in range(50):
            reveal = player.reveal_cards(1, list())
            if not reveal: break # deck is exhausted
            card = reveal[0]
            if card.money_in_hand:
                player.hand.append(card)
                treasures += 1
                if treasures >= 2: break
            else:
                player.discard.append(card)

class BureaucratCard(Card):
    def __init__(self):
        super().__init__("Bureaucrat", cost=4, is_action=True)
    def play(self, game, attacker):
        super().play(game, attacker)
        if game.stockpile[Silver] >= 1:
            attacker.deck.append(Silver)
            game.stockpile[Silver] -= 1
        for defender in game.players:
            if defender == attacker or (Moat in defender.hand):
                continue
            for ii, card in enumerate(defender.hand):
                if card.is_victory:
                    defender.deck.append(defender.hand.pop(ii))
                    break

class ChancellorCard(Card):
    def __init__(self):
        super().__init__("Chancellor", cost=3, is_action=True, money_when_played=2)
    def play(self, game, player):
        super().play(game, player)
        if len(player.deck) == 0 or len(player.discard) == 0:
            return # action has no effect, avoid divide-by-zero
        # Heuristic:  if there's a higher fraction of useless cards in the deck, then reshuffle
        deck_vp = len([c for c in player.deck if c.is_victory or c == Curse]) / len(player.deck)
        discard_vp = len([c for c in player.discard if c.is_victory or c == Curse]) / len(player.discard)
        if deck_vp > discard_vp:
            player.discard.extend(player.deck)
            player.deck.clear()

class CouncilRoomCard(Card):
    def __init__(self):
        super().__init__("Council_Room", cost=5, is_action=True, buys_when_played=1, cards_when_played=4)
    def play(self, game, attacker):
        super().play(game, attacker)
        for defender in game.players:
            if defender == attacker: # not really an attack
                continue
            defender.draw_cards(1)

class MineCard(Card):
    def __init__(self):
        super().__init__("Mine", cost=5, is_action=True)
    def play(self, game, player):
        super().play(game, player)
        if Silver in player.hand and game.stockpile[Gold] >= 1:
            player.hand.remove(Silver) # trashed - lost from game
            player.hand.append(Gold)
            game.stockpile[Gold] -= 1
        elif Copper in player.hand and game.stockpile[Silver] >= 1:
            player.hand.remove(Copper) # trashed - lost from game
            player.hand.append(Silver)
            game.stockpile[Silver] -= 1

class MoatCard(Card):
    def __init__(self):
        super().__init__("Moat", cost=2, is_action=True, cards_when_played=2)

class ThiefCard(Card):
    def __init__(self):
        super().__init__("Thief", cost=4, is_action=True)
    def play(self, game, attacker):
        super().play(game, attacker)
        for defender in game.players:
            if defender == attacker: # not really an attack
                continue
            cards = defender.reveal_cards(2, list())
            # print(f"got {cards}    deck {defender.deck}    discard {defender.discard}")
            if not cards: return
            cards.sort(key=lambda x: (x.is_treasure, x.money_in_hand), reverse=True)
            if cards[0] in [Gold, Silver]:
                attacker.discard.append(cards[0])
            elif not cards[0].is_treasure:
                defender.discard.append(cards[0])
            else:
                assert cards[0] == Copper # and we trash it
            defender.discard.extend(cards[1:])

# Singleton objects
Copper = Card("Copper", cost=0, money_in_hand=1, is_treasure=True)
Silver = Card("Silver", cost=3, money_in_hand=2, is_treasure=True)
Gold   = Card("Gold",   cost=6, money_in_hand=3, is_treasure=True)

Estate   = Card("Estate",   cost=2, victory_points=1, is_victory=True)
Duchy    = Card("Duchy",    cost=5, victory_points=3, is_victory=True)
Province = Card("Province", cost=8, victory_points=6, is_victory=True)

Curse    = Card("Curse",    cost=0, victory_points = -1)
Gardens  = Card("Gardens", cost=4, is_victory=True) # vic pts depends on final deck size

Festival = Card("Festival", cost=5, actions_when_played=2, buys_when_played=1, money_when_played=2, is_action=True)
Laboratory = Card("Laboratory", cost=5, actions_when_played=1, cards_when_played=2, is_action=True)
Market = Card("Market", cost=5, actions_when_played=1, buys_when_played=1, cards_when_played=1, money_when_played=1, is_action=True)
Smithy = Card("Smithy", cost=4, cards_when_played=3, is_action=True)
Village = Card("Village", cost=3, actions_when_played=2, cards_when_played=1, is_action=True)
Woodcutter = Card("Woodcutter", cost=5, buys_when_played=1, money_when_played=2, is_action=True)

Witch = WitchCard()
Adventurer = AdventurerCard()
Bureaucrat = BureaucratCard()
CouncilRoom = CouncilRoomCard()
Mine = MineCard()
Moat = MoatCard()

Chancellor = ChancellorCard()
Thief = ThiefCard()

MINIMAL_CARDS = [Copper, Silver, Gold, Estate, Duchy, Province]
MULTIPLIER_CARDS = [Festival, Laboratory, Market, Smithy, Village, Woodcutter]
DETERMINISTIC_CARDS = [Adventurer, Bureaucrat, CouncilRoom, Mine, Moat]
HEURISTIC_CARDS = [Chancellor, Thief]
# ALL_CARDS = MINIMAL_CARDS
# ALL_CARDS = MINIMAL_CARDS + [Gardens]
# ALL_CARDS = MINIMAL_CARDS + MULTIPLIER_CARDS
# ALL_CARDS = MINIMAL_CARDS + [Festival, Laboratory, Market, Village, Woodcutter] # no Smithy
# ALL_CARDS = MINIMAL_CARDS + [Witch]
# ALL_CARDS = MINIMAL_CARDS + [Witch, Moat]
# ALL_CARDS = MINIMAL_CARDS + MULTIPLIER_CARDS + DETERMINISTIC_CARDS + [Gardens, Witch]
ALL_CARDS = MINIMAL_CARDS + HEURISTIC_CARDS

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

class LinearRankStrategy:
    def __init__(self, weights=None):
        self.act_counts = Counter() # {Card: int}
        self.buy_counts = Counter() # {Card: int}
        self.act_counts_by_turn = Counter() # {(turn, Card): int}
        self.buy_counts_by_turn = Counter() # {(turn, Card): int}
        self.game_lengths = Counter() # {int: int}
        # Makes no sense to leave playable actions in the hand, if you bought them, so:
        self.actions = [PlayCard(c) for c in ALL_CARDS if c.is_action] #+ [EndActions()]
        # EndBuy is important:  allows us to declare some cards have negative value to us and should not be bought
        self.buys = [BuyCard(c) for c in ALL_CARDS] + [EndBuy()]
        self.weight_dist = []
        # A typical game lasts 10-20 turns.  With a coef of 0.05, a card can
        # go from 0 to 1 (or 1 to 0) over the course of a game.
        linear_coef = lambda: random.normalvariate(0, 0.05)
        for move in (self.actions + self.buys):
            move.idx = len(self.weight_dist)
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
        raw_act_key = lambda x: self.weights[x.idx] #+ game.turn*self.weights[x.idx+1]
        self.raw_sorted_actions = sorted(self.actions, key=raw_act_key)
        # Heuristic:
        # When playing a normal turn, cards that give extra actions come first.
        # Within those, cards that give extra draws come first, so there are more options.
        def act_key(x):
            if x.card.actions_when_played:
                return (0, -x.card.cards_when_played, self.weights[x.idx])
            else:
                return (1, 0, self.weights[x.idx])
        self.sorted_actions = sorted(self.actions, key=act_key)
        self.sorted_buys = []
        for game_turn in range(MAX_TURNS):
            buy_key = lambda x: self.weights[x.idx] + game_turn*self.weights[x.idx+1]
            self.sorted_buys.append(sorted(self.buys, key=buy_key))
        self.reset()
    def reset(self):
        # Call once per tournament to reset statistics
        self.wins = 0
        self.fitness = 0
        self.act_counts.clear()
        self.buy_counts.clear()
        self.act_counts_by_turn.clear()
        self.buy_counts_by_turn.clear()
        self.game_lengths.clear()
    def iter_actions(self, game, player):
        return self.sorted_actions
    def iter_actions_raw(self, game, player):
        return self.raw_sorted_actions
    def iter_buys(self, game, player):
        return self.sorted_buys[game.turn]
    def fmt_actions(self):
        #return '   '.join(f"{self.act_counts[m]} {m} ({self.weights[m.idx+1]:.3f})" for m in self.sorted_actions if self.act_counts[m] > 0)
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

        lines = ['']
        # # Group consecutive lines with same buy priority
        # for k, g in itertools.groupby(range(MAX_TURNS), key=lambda x: sorted_used_buys[x]):
        #     g = list(g)
        #     cnts = Counter()
        #     for game_turn in g:
        #         cnts += cbt[game_turn]
        #     #line = '   '.join(f"{cnts[m]} {m} ({self.weights[m.idx+1]:.3f})" for m in k if cnts[m] > 0)
        #     line = '   '.join(f"{cnts[m]} {m}" for m in k if cnts[m] > 0)
        #     if sum(cnts.values()) > 0:
        #         # avoid blank lines for sequences never played
        #         lines.append(f'    {min(g)+1:2d}:   '+line)

        # Show every line for turns played
        for ii, buys in enumerate(sorted_used_buys):
            line = '   '.join(f"{cbt[ii][m]} {m}" for m in buys if cbt[ii][m] > 0)
            if sum(cbt[ii].values()) > 0:
                # avoid blank lines for sequences never played
                lines.append(f'    {ii+1:2d}:   '+line)

        line = '   '.join(f"{self.buy_counts[m]} {m} ({self.weights[m.idx+1]:.3f})" for m in self.sorted_buys[0] if self.buy_counts[m] > 0)
        lines.append(f'    Sum   '+line)

        return '\n'.join(lines)

class Player:
    def __init__(self, name, deck, strategy):
        self.name = name
        self.deck = list(deck)
        random.shuffle(self.deck)
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
        self.turn = 0
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
                            player.strategy.act_counts[action] += 1
                            player.strategy.act_counts_by_turn[turn, action] += 1
                            break
                    else:
                        break # no playable actions
                player.calc_money()
                while player.buys > 0 and player.money > 0:
                    for buy in player.strategy.iter_buys(game, player):
                        if buy.can_move(game, player):
                            # print(f"    {buy}")
                            buy.do_move(game, player)
                            player.strategy.buy_counts[buy] += 1
                            player.strategy.buy_counts_by_turn[turn, buy] += 1
                            break
                player.draw_hand()

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

def run_tournament(strategies, players_per_game=3, games_per_strategy=50):
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
            max_vp = max(player.calc_victory_points() for player in players)
            for player in players:
                vp = player.calc_victory_points()
                player.strategy.fitness += vp
                if vp == max_vp: player.strategy.wins += 1 # get credit on tying for first
                player.strategy.game_lengths[game.turn+1] += 1


    strategies.sort(key=lambda x: (x.wins, x.fitness), reverse=True)

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

def main():
    players = 3
    popsize = 12 * 32 # some multiple of 2, 3, and 4
    strategies = [LinearRankStrategy() for _ in range(popsize)]

    for cycle in range(100): # expect to Ctrl-C to exit early
        start = time.time()
        run_tournament(strategies, players)
        for strategy in strategies[:3]:
            print(f"round {cycle}    wins {strategy.wins}    fitness {strategy.fitness}    game len {','.join(str(k) for k,v in strategy.game_lengths.most_common(3))}    {players} players    {time.time() - start:.2f} sec")
            print(f"  actions: {strategy.fmt_actions()}")
            print(f"  buys:    {strategy.fmt_buys()}")
        print("")
        strategies = evolve(strategies)

main()

