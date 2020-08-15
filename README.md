
## The minimal game

In the minimal game with only coins and victory points
(Copper, Silver, Gold; Estate, Duchy, Province),
the winning strategy reliably converges to these moves, in order of preference:

- buy Province
- buy Gold
- buy Duchy
- buy Silver

Copper and Estates are never bought, even if the alternative is doing nothing on a turn.

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
