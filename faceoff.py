import pickle
import random
import sys
import time

from dmstrat import *
from evol_strat import *
from rl_strat import *
from basic_polgrad import *

# random.seed(123456)

def main_faceoff():
    players = 2
    strategies = []
    for fname in sys.argv[1:]:
        with open(fname, "rb") as f:
            strategy = pickle.load(f)[0]
            if isinstance(strategy, LinearRankStrategy):
                strategies.append( LinearRankStrategy(weights=strategy.weights) )
            elif isinstance(strategy, MonteCarloStrategy):
                s = MonteCarloStrategy(q=strategy.q, c=strategy.c)
                strategies.append(s)
            elif isinstance(strategy, BasicPolicyGradientStrategy):
                s = BasicPolicyGradientStrategy(logits_net=strategy.logits_net)
                strategies.append(s)
            else:
                assert False, "Unsupported type of strategy"

    CYCLES = 1
    for cycle in range(CYCLES): # expect to Ctrl-C to exit early
        if cycle == CYCLES-1: # last one
            # Play final round without random exploratory moves
            for strategy in strategies:
                strategy.learn = False
        start = time.time()
        run_tournament(strategies, players, games_per_strategy=1000)
        print(f"round {cycle}    {players} players    {time.time() - start:.2f} sec " + ("="*70))
        for strategy in strategies:
            print(strategy)
        print("")

if __name__ == '__main__':
    main_faceoff()

