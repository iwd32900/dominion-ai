import random

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
    def can_buy(self, game, player):
        return (
            self.cost <= player.money
            # and player.buys >= 1 # already checked in outer loop
            and game.stockpile[self] >= 1
        )
    def buy(self, game, player):
        player.money -= self.cost
        player.buys -= 1
        player.discard.append(self)
        game.stockpile[self] -= 1
    def can_play(self, game, player):
        return (
            self in player.hand
            # and player.actions >= 1 # already checked in outer loop
            # and self.is_action # already checked by strategy
        )
    def play(self, game, player):
        # This order is important: while playing, a card is neither part of the hand, nor the discards
        player.hand.remove(self)
        player.actions -= 1
        # money is handled separately
        player.actions += self.actions_when_played
        player.buys += self.buys_when_played
        if self.cards_when_played:
            player.draw_cards(self.cards_when_played)
        self._play(game, player)
        player.played.append(self)
    def _play(self, game, player):
        pass
    def __repr__(self):
        return self.name

class EndCard(Card):
    def can_buy(self, game, player):
        return True
    def buy(self, game, player):
        player.buys = 0
    def can_play(self, game, player):
        return True
    def play(self, game, player):
        player.actions = 0
END = EndCard("END")

class AdventurerCard(Card):
    def __init__(self):
        super().__init__("Adventurer", cost=6, is_action=True)
    def _play(self, game, player):
        treasures = 0
        # After 50 tries, we assume there are no more treasures in the deck!
        for _ in range(50):
            reveal = player.reveal_cards(1)
            if not reveal: break # deck is exhausted
            card = reveal[0]
            if card.money_in_hand:
                player.hand.append(card)
                treasures += 1
                if treasures >= 2: break
            else:
                player.discard.append(card)
Adventurer = AdventurerCard()

class BureaucratCard(Card):
    def __init__(self):
        super().__init__("Bureaucrat", cost=4, is_action=True)
    def _play(self, game, attacker):
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
Bureaucrat = BureaucratCard()

class ChapelCard(Card):
    """
    "Trash up to 4 cards from your hand."
    Heuristic:  if we would rather end our turn than buy card X,
        then we should trash card X.  Not perfect, but reasonable.
    """
    def __init__(self):
        super().__init__("Chapel", cost=2, is_action=True)
    def _play(self, game, player):
        hand = player.hand
        buys = list(player.strategy.rank_buys(game, player))
        trash = []
        for card in [Curse] + buys[::-1]:
            if card == END or len(trash) >= 4: break
            while card in hand and len(trash) < 4:
                hand.remove(card)
                trash.append(card)
        assert len(trash) <= 4
Chapel = ChapelCard()

class ChancellorCard(Card):
    def __init__(self):
        super().__init__("Chancellor", cost=3, is_action=True, money_when_played=2)
    def _play(self, game, player):
        if len(player.deck) == 0 or len(player.discard) == 0:
            return # action has no effect, avoid divide-by-zero
        # Heuristic:  if there's a higher fraction of useless cards in the deck, then reshuffle
        deck_vp = len([c for c in player.deck if c.is_victory or c == Curse]) / len(player.deck)
        discard_vp = len([c for c in player.discard if c.is_victory or c == Curse]) / len(player.discard)
        if deck_vp > discard_vp:
            player.discard.extend(player.deck)
            player.deck.clear()
Chancellor = ChancellorCard()

class CouncilRoomCard(Card):
    def __init__(self):
        super().__init__("Council_Room", cost=5, is_action=True, buys_when_played=1, cards_when_played=4)
    def _play(self, game, attacker):
        for defender in game.players:
            if defender == attacker: # not really an attack
                continue
            defender.draw_cards(1)
CouncilRoom = CouncilRoomCard()

class MineCard(Card):
    def __init__(self):
        super().__init__("Mine", cost=5, is_action=True)
    def _play(self, game, player):
        if Silver in player.hand and game.stockpile[Gold] >= 1:
            player.hand.remove(Silver) # trashed - lost from game
            player.hand.append(Gold)
            game.stockpile[Gold] -= 1
        elif Copper in player.hand and game.stockpile[Silver] >= 1:
            player.hand.remove(Copper) # trashed - lost from game
            player.hand.append(Silver)
            game.stockpile[Silver] -= 1
Mine = MineCard()

class ThiefCard(Card):
    def __init__(self):
        super().__init__("Thief", cost=4, is_action=True)
    def _play(self, game, attacker):
        for defender in game.players:
            if defender == attacker or (Moat in defender.hand):
                continue
            cards = defender.reveal_cards(2)
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
Thief = ThiefCard()

class WitchCard(Card):
    def __init__(self):
        super().__init__("Witch", cost=5, is_action=True, cards_when_played=2)
    def _play(self, game, attacker):
        for defender in game.players:
            if defender == attacker or game.stockpile[Curse] < 1 or (Moat in defender.hand):
                continue
            defender.discard.append(Curse)
            game.stockpile[Curse] -= 1
Witch = WitchCard()

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
Moat = Card("Moat", cost=2, cards_when_played=2, is_action=True)
Smithy = Card("Smithy", cost=4, cards_when_played=3, is_action=True)
Village = Card("Village", cost=3, actions_when_played=2, cards_when_played=1, is_action=True)
Woodcutter = Card("Woodcutter", cost=5, buys_when_played=1, money_when_played=2, is_action=True)

MINIMAL_CARDS = [Copper, Silver, Gold, Estate, Duchy, Province]
MULTIPLIER_CARDS = [Festival, Laboratory, Market, Moat, Smithy, Village, Woodcutter]
DETERMINISTIC_CARDS = [Adventurer, Bureaucrat, CouncilRoom, Mine]
HEURISTIC_CARDS = [Chapel, Chancellor, Thief]
# ALL_CARDS = MINIMAL_CARDS
# ALL_CARDS = MINIMAL_CARDS + [Smithy]
# ALL_CARDS = MINIMAL_CARDS + [Smithy, Moat, Thief, Witch]
# ALL_CARDS = MINIMAL_CARDS + [Gardens]
# ALL_CARDS = MINIMAL_CARDS + MULTIPLIER_CARDS
# ALL_CARDS = MINIMAL_CARDS + [Festival, Laboratory, Market, Village, Woodcutter] # no Smithy
# ALL_CARDS = MINIMAL_CARDS + [Witch]
# ALL_CARDS = MINIMAL_CARDS + [Witch, Moat]
# ALL_CARDS = MINIMAL_CARDS + MULTIPLIER_CARDS + DETERMINISTIC_CARDS + [Gardens, Witch]
# ALL_CARDS = MINIMAL_CARDS + HEURISTIC_CARDS
ALL_CARDS = MINIMAL_CARDS + MULTIPLIER_CARDS + DETERMINISTIC_CARDS + HEURISTIC_CARDS + [Gardens, Witch]
