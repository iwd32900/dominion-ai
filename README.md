## TODO

## The literature

[Dominion Strategy](https://dominionstrategy.com/)
[Provincial, a Dominion AI](https://graphics.stanford.edu/~mdfisher/DominionAI.html)
[Deep RL for Dominion](http://cs230.stanford.edu/projects_fall_2019/reports/26260348.pdf)
[Dominion genetic algorithm](https://github.com/octachrome/dominion)

[Sutton & Barton, "Reinforcement Learning: An Introduction", 2020](http://www.incompleteideas.net/book/the-book.html)
[Fast.AI Course](https://course.fast.ai/videos/?lesson=1)
[Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/)

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

This is true for 2, 3, or 4 players.

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
- buy Gardens (2 or 3 players)
- buy Duchy
- buy Gardens (4 players)
- buy Silver
- buy Copper

Since Gardens are cheaper than Duchies, this means algorithms are fighting each other for
the fairly limited supply of Gardens in the 2 or 3 player game.
In the 4 player game though, Gardens are somewhat less valuable.
In this regime, Estates are still not worth buying,
and it would be very difficult to exhaust the 61 Coppers in the game anyway.

## Runtime optimization: PyPy

With PyPy and my current settings, a typical generation is a bit over 2 seconds.
With CPython3.6, a typical generation is about 11 seconds.
This is about a 5x speedup, which is about as good a result as people ever get from PyPy -- excellent.
First generations are always slower because the strategies are inefficient and the games are long.

```
pypy3 -m vmprof -o prof.log optimize.py
/usr/local/share/pypy3/vmprofshow prof.log
```

## The multipliers:  Festival, Laboratory, Market, Smithy, Village, Woodcutter

This set of 6 cards grants additional actions, buys, cards, and/or money,
with no additional logic beyond that, so I implemented them next.

The game is now significantly more complex, and introduces Actions for the first time.
The action order is highly variable, but the optimizer discovers early on that it should
always play action cards if it has them (not end Action phase early).
I also track how often each action is actually taken:  in many cases the order doesn't matter
because "preferred" actions are never bought, and so never played.
The buy order is very consistent:

- buy Smithy
- buy Province
- buy Gold (not in 2 player! sometimes after Duchy in 3 player)
- buy Duchy
- buy Silver
- buy Copper

### Without Smithy

- buy Laboratory (2 or 3 players)
- buy Province
- buy Gold
- buy Laboratory (4 players)
- buy Duchy
- buy Silver
- buy Copper

With all 5 multipliers except the Smithy, the Laboratory becomes the favorite to purchase.
This makes sense, as it grants 2 extra cards instead of the Smithy's 3.
Interestingly, in the 4 player game this is somewhat less valuable than in the 2 and 3 player games.

## Burn her, she's a Witch!

The Witch is nasty combination of several mechanics explored above:
draw extra cards, like the Smithy or Laboratory;
victory points (actually negative victory points for opponents);
and deck dilution (again, for the opponents).
So, no surprise that our algorithm likes it.
Interestingly, it's apparently most valuable in the 2-player game though.

- buy Witch (2 players)
- buy Province
- buy Gold
- buy Witch (3 or 4 players)
- buy Duchy
- buy Silver
- buy Copper

## Moat

- buy Moat (3 or 4 players)
- buy Province
- buy Moat (2 players)
- buy Gold
- buy Witch (3 or 4 players only)
- buy Duchy (2 players only, in practice)
- buy Silver
- buy Estate/Copper

## Starting cards

At this point, I actually read the rules more carefully and got the number of starting cards correct...

## All deterministic cards

This is all the cards above plus ~~Adventurer~~, Bureaucrat, Council Room, and Mine.
Technically, Mine is not deterministic, but at this point I feel safe
with a heuristic that chooses to upgrade Silver to Gold in preference to Copper to Silver.

2 player: Province, Gold, Witch, Moat, Smithy/Gardens/Estate/Copper
3 player:
- Moat, Province, Gold, Witch, Smithy, Silver, Estate
- Province, Smithy, Mine, Moat (intermediate, evolves to above)

4 player:
- Province, Moat/Gold, Witch, Smith, Silver, Estate, Copper

## Deterministic

Adventurer, Bureaucrat, Council Room, Mine*, Moat

## Additional choices required

### Buy/Gain
Remodel, Thief, Workshop

### Discard
Cellar, Militia, Spy, Thief

### Trash
Chapel, Money Lender, Remodel

### Other
Chancellor (deck into discard pile)
Library (set aside Actions for more draws)
Throne Room (play an Action twice)
