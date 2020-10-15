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
    def play(self, game, player):
        # money is handled separately
        player.actions += self.actions_when_played
        player.buys += self.buys_when_played
        if self.cards_when_played:
            player.draw_cards(self.cards_when_played)
    def __repr__(self):
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
            if defender == attacker or (Moat in defender.hand):
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
ALL_CARDS = MINIMAL_CARDS
# ALL_CARDS = MINIMAL_CARDS + [Gardens]
# ALL_CARDS = MINIMAL_CARDS + MULTIPLIER_CARDS
# ALL_CARDS = MINIMAL_CARDS + [Festival, Laboratory, Market, Village, Woodcutter] # no Smithy
# ALL_CARDS = MINIMAL_CARDS + [Witch]
# ALL_CARDS = MINIMAL_CARDS + [Witch, Moat]
# ALL_CARDS = MINIMAL_CARDS + MULTIPLIER_CARDS + DETERMINISTIC_CARDS + [Gardens, Witch]
# ALL_CARDS = MINIMAL_CARDS + HEURISTIC_CARDS
# ALL_CARDS = MINIMAL_CARDS + [Smithy, Moat, Thief]
# ALL_CARDS = MINIMAL_CARDS + MULTIPLIER_CARDS + DETERMINISTIC_CARDS + HEURISTIC_CARDS + [Gardens, Witch]
