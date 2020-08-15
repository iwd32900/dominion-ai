# TODO

- Track actual move count per strategy, modify printout
- Experiment with 2 and 4 player games
- Introduce the Witch and Curses

## The minimal game

In the minimal game with only coins and victory points
(Copper, Silver, Gold; Estate, Duchy, Province),
the winning strategy reliably converges to these moves, in order of preference:

- buy Province
- buy Gold
- buy Duchy
- buy Silver

Copper and Estates are never bought, even if the alternative is doing nothing on a turn.
This is because both cards dilute the deck, making it harder to accumulate the
8 coins needed to buy a Province.

## With Gardens

The base strategy rewards keeping a small, efficient deck, but
Gardens is a victory card that rewards building a large deck.

Introducing Gardens made the game *much* harder to optimize,
and required both crossover *and* mutations for my evolutionary algorithm to converge.

In fact, in most runs, two different strategies alternated winning successive generations.
First was the unmodified strategy above, ignoring the Gardens.
Second was:

- buy Province
- buy Gold
- buy Gardens
- buy Duchy
- buy Silver
- buy Copper
- buy Estate (or end turn)

Since Gardens are cheaper than Duchies, this means algorithms are fighting each other for
the fairly limited supply of Gardens.
In this regime, Estates are a toss-up as to whether they're worth buying.
Also, since it would be very difficult to exhaust the 61 Coppers in the game,
there's little evolutionary pressure to keep/drop the Estate rule.

## Runtime optimization: PyPy

With PyPy and my current settings, a typical generation is a bit over 2 seconds.
With CPython3.6, a typical generation is about 11 seconds.
This is about a 5x speedup, which is about as good a result as people ever get from PyPy -- excellent.
First generations are always slower because the strategies are inefficient and the games are long.

## The multipliers:  Festival, Laboratory, Market, Smithy, Village, Woodcutter

This set of 6 cards grants additional actions, buys, cards, and/or money,
with no additional logic beyond that, so I implemented them next.

The game is now significantly more complex, and introduces Actions for the first time.
The action order is highly variable, but the optimizer discovers early on that it should
always play action cards if it has them (not end Action phase early).
However, it doesn't matter much, because the buy order is very consistent:

- buy Province
- buy Gold
- buy Duchy
- buy Silver
- buy Market (or Festival)

The order continues but is inconsistent -- even Market/Festival is a weak trend.
If we run only with Smithy, Market, and Festival, the trend holds up:
Smithy is always the top priority, Dutchy takes precedence over the others at the same price,
making them unlikely to get bought much, if at all.

### Without Smithy

- buy Province
- buy Gold
- buy Market
- buy Duchy (or Dutchy then Market)
- buy Silver
- Woodcutter or end turn

With all 5 multipliers except the Smithy, the Market becomes the favorite to purchase
(behind Provinces and Gold), but the Woodcutter is the favorite to play!
This is strategically wrong: if both are in the same hand, the Markets should be played first
for their extra actions, so all Actions can be played.
But it suggests that given the buying priorities, Woodcutters are not often in the hand,
and even more rarely with Markets, so the optimizer can't learn this rule.

I can confirm this by using Market and Woodcutter only, and even then
Market drifts up and down the list of priorities.