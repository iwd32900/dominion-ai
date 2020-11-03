import pickle
import random
import sys
import time

from dmstrat import *
from evol_strat import *
from rl_strat import *
try:
    import torch
except ModuleNotFoundError:
    print("Unable to find pytorch;  deep RL methods unavailable.")
else:
    from basic_polgrad import *
    from ppo_strat import *
    from nnmc_strat import *

# random.seed(123456)

def main_faceoff():
    players = 2
    strategies = []
    for fname in sys.argv[1:]:
        strategy = load_strategies(fname)[0]
        strategy.learn = False
        strategies.append(strategy)

    CYCLES = 1
    for cycle in range(CYCLES): # expect to Ctrl-C to exit early
        if cycle == CYCLES-1: # last one
            # Play final round without random exploratory moves
            for strategy in strategies:
                strategy.learn = False
        start = time.time()
        run_tournament(strategies, players, games_per_strategy=10000)
        print(f"round {cycle}    {players} players    {time.time() - start:.2f} sec " + ("="*70))
        for strategy in strategies:
            print(strategy)
        print("")

if __name__ == '__main__':
    main_faceoff()
    # import cProfile; cProfile.run("main_faceoff()", "profile.stats")

