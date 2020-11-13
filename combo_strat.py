from evol_strat import *
from ppo_strat import *

def main_combo():
    players = 2
    ppo_strategies = [PPOStrategy() for _ in range(players)]
    popsize = (12 * 8) - players # some multiple of 2, 3, and 4
    ga_strategies = [LinearRankStrategy() for _ in range(popsize)]

    CYCLES = 400
    GPS = 100
    for cycle in range(CYCLES): # expect to Ctrl-C to exit early
        if cycle == CYCLES-1: # last one
            GPS = 500
            for strategy in ppo_strategies:
                strategy.learn = False
        start = time.time()

        run_tournament(ga_strategies + ppo_strategies, players, games_per_strategy=GPS)
        run_tournament(ppo_strategies, players, games_per_strategy=2*GPS, reset=False)

        print(f"round {cycle}    {players} players    {GPS} games    {time.time() - start:.2f} sec " + ("="*70))
        for strategy in ppo_strategies:
            print(strategy)
        print("")

        save_strategies(ppo_strategies, "save_ppo")

        ga_strategies.sort(key=lambda x: (x.wins, x.fitness), reverse=True)
        ga_strategies = evolve(ga_strategies)
        for strategy in ppo_strategies:
            strategy.step()

if __name__ == '__main__':
    main_combo()
