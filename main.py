import random, sys

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,8)

from run import run, replay, main_cli
from sim.simInterface import SimInterface

# Every run goes through run() in run.py. From the terminal, e.g.:
#   uv run python main.py --solver sac --n 500 --level L1 --random-start each --swimmers 2
#   uv run python main.py --sim brage --solver mcts --n 4000
#   uv run python main.py --replay                     (best result of the latest run)
#   uv run python main.py --help                       (all options)
# Without arguments, main.py runs whatever is called at the bottom of this file.

def main():  # MCTS on Brage's simulator
    run(simulator="brage", solver="mcts", n=4000)

def mainMarast(numberOfLoops = 500, level = "L0"):  # MCTS on MAR-AST
    run(simulator="marast", solver="mcts", n=numberOfLoops, level=level)

def mainSac(numberOfLoops = 2000):  # SAC (deep RL) solver instead of MCTS, on Brage's simulator
    run(simulator="brage", solver="sac", n=numberOfLoops)

def mainSacMarast(numberOfLoops = 500, level = "L0", random_start = "each"):  # SAC on MAR-AST, a new random start every episode
    run(simulator="marast", solver="sac", n=numberOfLoops, level=level, random_start=random_start)

def randomPath(verbose = True):  # One episode with random actions on Brage's simulator
    SBC = SimInterface()
    while not SBC.sim_in_endstate:
        SBC.step(random.random())
    if verbose:
        SBC.plot()
    return SBC.is_failure_episode()

if len(sys.argv) > 1:
    main_cli()
else:
    run()  # Defaults: MCTS, 500 iterations, MAR-AST level L0_open_water, fixed start

# main()

# mainMarast()

# mainSac()

# mainSacMarast()

# replay()

# randomPath()

# hits = 0
# for i in range(10**6):
#     if randomPath(False):
#         hits+=1
#     if (i+1)%10000 == 0:
#         print(hits, "collisions in",i+1,"random routes")
