from sim.simInterface import SimInterface
from sim.simpleBoatController import SimpleBoatController
from mcts.mctsController import MCTSController
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,8)
import random

def main():
    MCTS = MCTSController()
    MCTS.loop(4000)

def randomPath(verbose = True):
    SBC = SimInterface()
    while not SBC.sim_in_endstate:
        SBC.step(random.random())
    if verbose:
        SBC.plot()
    return SBC.collision_happened

main()

# randomPath()

# hits = 0
# for i in range(10**6):
#     if randomPath(False):
#         hits+=1
#     if (i+1)%10000 == 0:
#         print(hits, "collisions in",i,"random routes")